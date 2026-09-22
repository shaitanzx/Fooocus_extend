"""Florence-2 integration for Fooocus_extend.

This module is intentionally independent from Forge/Spaces. It does not load
models when imported and does not launch its own Gradio application.
"""

from __future__ import annotations

import sys
import importlib.util
from unittest.mock import MagicMock

# Create a fake spec object
class FakeFlashAttnSpec:
    name = 'flash_attn'
    loader = None
    origin = None
    submodule_search_locations = []
    
fake_spec = FakeFlashAttnSpec()

# Create mock modules with proper __spec__ attributes
flash_attn_mock = MagicMock()
flash_attn_mock.__spec__ = fake_spec
flash_attn_mock.__version__ = "0.0.0"  # Force version check to fail

sys.modules['flash_attn'] = flash_attn_mock
sys.modules['flash_attn.flash_attn_interface'] = MagicMock()
sys.modules['flash_attn.bert_padding'] = MagicMock()

# Patch find_spec to return our fake spec
_original_find_spec = importlib.util.find_spec

def _patched_find_spec(name, package=None):
    if name == 'flash_attn' or name.startswith('flash_attn.'):
        return fake_spec
    return _original_find_spec(name, package)

importlib.util.find_spec = _patched_find_spec

import gc
import os
import threading
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import gradio as gr
import numpy as np
import torch
from PIL import Image, ImageDraw




from transformers import AutoModelForCausalLM, AutoProcessor


# Keep only one Florence model in memory at a time. The large and base models
# are never needed simultaneously and keeping both would waste VRAM.
MODEL_IDS = (
    "microsoft/Florence-2-base",
    "microsoft/Florence-2-large",
    'microsoft/Florence-2-base-ft',
    'microsoft/Florence-2-large-ft'
)
DEFAULT_MODEL_ID = "microsoft/Florence-2-base"

IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".tif",
    ".tiff",
    ".avif",
}

SINGLE_TASKS = (
    "Caption",
    "Detailed Caption",
    "More Detailed Caption",
    "Object Detection",
    "Dense Region Caption",
    "Region Proposal",
    "Caption to Phrase Grounding",
    "Referring Expression Segmentation",
    "Region to Segmentation",
    "Open Vocabulary Detection",
    "Region to Category",
    "Region to Description",
    "OCR",
    "OCR with Region",
)

CAPTION_TASKS = (
    "Caption",
    "Detailed Caption",
    "More Detailed Caption",
)

CASCADED_TASKS = (
    "Caption + Grounding",
    "Detailed Caption + Grounding",
    "More Detailed Caption + Grounding",
)

TASK_TOKENS = {
    "Caption": "<CAPTION>",
    "Detailed Caption": "<DETAILED_CAPTION>",
    "More Detailed Caption": "<MORE_DETAILED_CAPTION>",
    "Object Detection": "<OD>",
    "Dense Region Caption": "<DENSE_REGION_CAPTION>",
    "Region Proposal": "<REGION_PROPOSAL>",
    "Caption to Phrase Grounding": "<CAPTION_TO_PHRASE_GROUNDING>",
    "Referring Expression Segmentation": "<REFERRING_EXPRESSION_SEGMENTATION>",
    "Region to Segmentation": "<REGION_TO_SEGMENTATION>",
    "Open Vocabulary Detection": "<OPEN_VOCABULARY_DETECTION>",
    "Region to Category": "<REGION_TO_CATEGORY>",
    "Region to Description": "<REGION_TO_DESCRIPTION>",
    "OCR": "<OCR>",
    "OCR with Region": "<OCR_WITH_REGION>",
}

# The processor is small enough to keep cached. The model is moved back to
# CPU after every operation and only one model is retained.
_processor_cache: Dict[str, Any] = {}
_loaded_model: Optional[torch.nn.Module] = None
_loaded_model_id: Optional[str] = None
_model_lock = threading.RLock()


def _fooocus_model_management():
    """Return Fooocus' memory manager without making import-time assumptions."""
    try:
        import ldm_patched.modules.model_management as model_management

        return model_management
    except Exception:
        return None


def _inference_device() -> torch.device:
    """Use Fooocus' selected device, with a safe fallback for standalone tests."""
    management = _fooocus_model_management()
    if management is not None:
        try:
            return management.get_torch_device()
        except Exception:
            pass
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _model_dtype(device: torch.device) -> torch.dtype:
    """Select a conservative dtype; CPU always stays float32."""
    if device.type != "cuda":
        return torch.float32

    management = _fooocus_model_management()
    if management is not None:
        try:
            if management.should_use_fp16(device=device, prioritize_performance=False):
                return torch.float16
        except Exception:
            pass

    # Float16 is normally the safest CUDA choice for Florence-2. It is not
    # selected on CPU because many CPU kernels do not support fp16 well.
    return torch.float16


def _clear_cuda_cache() -> None:
    management = _fooocus_model_management()
    if management is not None:
        try:
            management.soft_empty_cache()
            return
        except Exception:
            pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()


def _release_loaded_model() -> None:
    """Move the cached Florence model to CPU and release temporary allocations."""
    global _loaded_model, _loaded_model_id

    if _loaded_model is not None:
        try:
            _loaded_model.to(device="cpu", dtype=torch.float32)
        except TypeError:
            _loaded_model.to("cpu")
        except Exception:
            # The reference is still cleared below; this prevents a cleanup
            # failure from masking the actual inference error.
            pass

    _loaded_model = None
    _loaded_model_id = None
    gc.collect()
    _clear_cuda_cache()


def unload_florence2() -> None:
    """Public cleanup hook; safe to call before Fooocus generation."""
    with _model_lock:
        _release_loaded_model()


def _prepare_gpu_memory(device: torch.device, estimated_bytes: int = 0) -> None:
    """Ask Fooocus to unload managed models before Florence uses the GPU."""
    management = _fooocus_model_management()
    if management is not None:
        try:
            # This unloads Fooocus-managed models when the requested amount of
            # free VRAM is not available. Florence itself is not registered as
            # a ModelPatcher, so it is released explicitly by this module.
            management.free_memory(estimated_bytes, device)
        except Exception:
            pass
    _clear_cuda_cache()


def _load_model(model_id: str) -> Tuple[torch.nn.Module, Any, torch.device]:
    """Load one model lazily on CPU, then move it to Fooocus' device."""
    global _loaded_model, _loaded_model_id

    if model_id not in MODEL_IDS:
        raise ValueError(f"Unsupported Florence-2 model: {model_id}")

    with _model_lock:
        device = _inference_device()

        if _loaded_model is not None and _loaded_model_id != model_id:
            _release_loaded_model()

        if model_id not in _processor_cache:
            _processor_cache[model_id] = AutoProcessor.from_pretrained(
                model_id,
                trust_remote_code=True,
                attn_implementation="sdpa",
                cache_dir=os.path.join("models","caption")
            )

        if _loaded_model is None:
            # Загружаем модель БЕЗ принудительного dtype
            # Florence-2 сама выберет оптимальные типы для своих слоев
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                attn_implementation="sdpa",
                cache_dir=os.path.join("models","caption")
            ).eval()
            _loaded_model = model
            _loaded_model_id = model_id

        _prepare_gpu_memory(device)
        
        # Перемещаем модель на устройство БЕЗ изменения dtype
        try:
            _loaded_model.to(device=device)
        except RuntimeError as exc:
            _release_loaded_model()
            if "out of memory" in str(exc).lower():
                raise RuntimeError(
                    "Недостаточно VRAM для Florence-2. Выберите Florence-2-base, "
                    "закройте другие GPU-модели или используйте режим с большим offload."
                ) from exc
            raise

        return _loaded_model, _processor_cache[model_id], device


def _move_batch_to_device(batch: Any, device: torch.device) -> Any:
    if hasattr(batch, "to"):
        return batch.to(device)
    if isinstance(batch, dict):
        return {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in batch.items()
        }
    return batch





def _generate(
    image: Image.Image,
    task_token: str,
    text_input: Optional[str],
    model_id: str,
    max_new_tokens: int = 256,
    num_beams: int = 1,
) -> Dict[str, Any]:
    """Run one Florence task and always offload the model afterwards."""
    if image is None:
        raise ValueError("Input image is empty")

    image = image.convert("RGB")
    model, processor, device = _load_model(model_id)
    prompt = task_token if not text_input else task_token + str(text_input)

    try:
        inputs = processor(text=prompt, images=image, return_tensors="pt")
        inputs = _move_batch_to_device(inputs, device)
        
        # Используем autocast для автоматического смешивания типов
        # Это позволяет модели работать в float16 там, где это безопасно,
        # но сохраняет float32 для критичных слоев (LayerNorm)
        with torch.inference_mode():
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                generated_ids = model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=max_new_tokens,
                    early_stopping=False,
                    do_sample=False,
                    num_beams=num_beams,
                )
        
        generated_text = processor.batch_decode(
            generated_ids,
            skip_special_tokens=False,
        )[0]
        return processor.post_process_generation(
            generated_text,
            task=task_token,
            image_size=image.size,
        )
    finally:
        for name in ("inputs", "generated_ids"):
            if name in locals():
                del locals()[name]
        with _model_lock:
            _release_loaded_model()


def _plot_bboxes(image: Image.Image, data: Dict[str, Any]) -> Image.Image:
    result = image.convert("RGB").copy()
    draw = ImageDraw.Draw(result)
    for bbox, label in zip(data.get("bboxes", []), data.get("labels", [])):
        x1, y1, x2, y2 = [int(v) for v in bbox]
        draw.rectangle((x1, y1, x2, y2), outline="red", width=3)
        draw.text((x1 + 4, y1 + 4), str(label), fill="white", stroke_width=2, stroke_fill="red")
    return result


def _draw_polygons(image: Image.Image, data: Dict[str, Any]) -> Image.Image:
    result = image.convert("RGB").copy()
    draw = ImageDraw.Draw(result, "RGBA")
    colors = [(255, 80, 80, 100), (80, 160, 255, 100), (80, 220, 120, 100)]
    for index, (polygons, label) in enumerate(
        zip(data.get("polygons", []), data.get("labels", []))
    ):
        color = colors[index % len(colors)]
        for polygon in polygons:
            points = np.asarray(polygon).reshape(-1, 2).astype(int).tolist()
            if len(points) >= 3:
                draw.polygon(points, fill=color, outline=color[:3] + (255,))
                draw.text(tuple(points[0]), str(label), fill="white")
    return result


def _draw_ocr(image: Image.Image, data: Dict[str, Any]) -> Image.Image:
    result = image.convert("RGB").copy()
    draw = ImageDraw.Draw(result)
    for box, label in zip(data.get("quad_boxes", []), data.get("labels", [])):
        points = np.asarray(box).reshape(-1, 2).astype(int).tolist()
        if len(points) >= 4:
            draw.line(points + [points[0]], fill="red", width=3)
            draw.text(tuple(points[0]), str(label), fill="red")
    return result


def process_image(
    image: Image.Image,
    task_name: str,
    text_input: str = "",
    model_id: str = DEFAULT_MODEL_ID,
) -> Tuple[str, Optional[Image.Image]]:
    """Gradio callback for one image."""
    if image is None:
        return "Загрузите изображение.", None

    try:
        if task_name in CASCADED_TASKS:
            caption_name = task_name.replace(" + Grounding", "")
            caption_token = TASK_TOKENS[caption_name]
            caption = _generate(image, caption_token, None, model_id)[caption_token]
            grounding = _generate(
                image,
                TASK_TOKENS["Caption to Phrase Grounding"],
                caption,
                model_id,
            )
            result = dict(grounding)
            result[caption_token] = caption
            output = _plot_bboxes(image, grounding[TASK_TOKENS["Caption to Phrase Grounding"]])
            return str(result), output

        task_token = TASK_TOKENS[task_name]
        result = _generate(image, task_token, text_input or None, model_id)
        value = result.get(task_token, result)

        if task_name in {"Object Detection", "Dense Region Caption", "Region Proposal", "Caption to Phrase Grounding"}:
            output = _plot_bboxes(image, value)
        elif task_name in {"Referring Expression Segmentation", "Region to Segmentation"}:
            output = _draw_polygons(image, value)
        elif task_name == "Open Vocabulary Detection":
            output = _plot_bboxes(image, {
                "bboxes": value.get("bboxes", []),
                "labels": value.get("bboxes_labels", []),
            })
        elif task_name == "OCR with Region":
            output = _draw_ocr(image, value)
        else:
            output = None

        return str(result), output
    except Exception as exc:
        return f"Florence-2 error: {type(exc).__name__}: {exc}", None


def _iter_image_files(directory: str) -> Iterable[str]:
    root = Path(os.path.expanduser(directory or ""))
    if not root.is_dir():
        return []
    return (
        str(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def run_batch(
    directory: str,
    task_name: str,
    model_id: str = DEFAULT_MODEL_ID,
    save_captions: bool = False,
    prefix: str = "",
) -> str:
    """Caption all images in a directory, one image at a time."""
    if task_name not in CAPTION_TASKS:
        return "Batch mode supports Caption, Detailed Caption and More Detailed Caption."

    task_token = TASK_TOKENS[task_name]
    files = list(_iter_image_files(directory))
    if not files:
        return "В каталоге нет поддерживаемых изображений."

    output: List[str] = []
    for filename in files:
        try:
            with Image.open(filename) as source:
                image = source.convert("RGB")
            result = _generate(
                image,
                task_token,
                None,
                model_id,
                max_new_tokens=1024,
                num_beams=3,
            )
            caption = prefix + str(result[task_token])
            output.append(f"File: {filename}\nCaption: {caption}\n")
            if save_captions:
                Path(filename + ".txt").write_text(caption, encoding="utf-8")
        except Exception as exc:
            output.append(f"File: {filename}\nError: {type(exc).__name__}: {exc}\n")

    return "\n".join(output)


def update_task_dropdown(task_type: str):
    """Gradio 3-compatible update; do not return a new component instance."""
    if task_type == "Cascaded task":
        return gr.update(choices=list(CASCADED_TASKS), value=CASCADED_TASKS[0])
    return gr.update(choices=list(SINGLE_TASKS), value="More Detailed Caption")


def ui():
    """Create the Fooocus tab. Fooocus owns the Blocks and server lifecycle."""
    gr.Markdown("# Florence-2\nImage captioning, detection, segmentation and OCR.")

    with gr.Tabs():
        with gr.TabItem(label="Image analysis"):
            with gr.Row():
                with gr.Column():
                    input_image = gr.Image(label="Input image", type="pil")
                    model_selector = gr.Dropdown(
                        choices=list(MODEL_IDS),
                        value=DEFAULT_MODEL_ID,
                        label="Model",
                    )
                    task_type = gr.Radio(
                        choices=["Single task", "Cascaded task"],
                        value="Single task",
                        label="Task type",
                    )
                    task_selector = gr.Dropdown(
                        choices=list(SINGLE_TASKS),
                        value="More Detailed Caption",
                        label="Task",
                    )
                    text_input = gr.Textbox(
                        label="Text input",
                        placeholder="Required for grounding/region tasks",
                    )
                    submit = gr.Button("Run Florence-2")
                    unload = gr.Button("Unload Florence-2 from memory")
                with gr.Column():
                    output_text = gr.Textbox(label="Result", lines=12)
                    output_image = gr.Image(label="Visualization", type="pil")

            task_type.change(
                fn=update_task_dropdown,
                inputs=task_type,
                outputs=task_selector,
            )
            submit.click(
                fn=process_image,
                inputs=[input_image, task_selector, text_input, model_selector],
                outputs=[output_text, output_image],
            )
            unload.click(
                fn=lambda: (unload_florence2(), "Florence-2 unloaded.")[1],
                inputs=[],
                outputs=output_text,
            )

        with gr.TabItem(label="Batch captioning"):
            batch_directory = gr.Textbox(label="Input directory")
            batch_model = gr.Dropdown(
                choices=list(MODEL_IDS),
                value=DEFAULT_MODEL_ID,
                label="Model",
            )
            batch_task = gr.Dropdown(
                choices=list(CAPTION_TASKS),
                value="More Detailed Caption",
                label="Task",
            )
            save_captions = gr.Checkbox(
                label="Save .txt next to each image",
                value=False,
            )
            prefix = gr.Textbox(label="Caption prefix")
            batch_submit = gr.Button("Run batch captioning")
            batch_output = gr.Textbox(label="Batch output", lines=20)
            batch_submit.click(
                fn=run_batch,
                inputs=[batch_directory, batch_task, batch_model, save_captions, prefix],
                outputs=batch_output,
            )

    return output_text, output_image


# __all__ = [
#     "ui",
#     "process_image",
#     "run_batch",
#     "unload_florence2",
# ]
