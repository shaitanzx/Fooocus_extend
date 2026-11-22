"""Lukutar Retro Filters integration for Fooocus Extend."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
import re
import cv2
import gradio as gr
import numpy as np
from PIL import Image

import modules.config
import extentions.batch as batch
import modules.imageslider as imageslider
from extentions.tutorial import tutorial
import datetime
from .core.analog import DOT_SHAPES, NOISE_MODES, lcd_dot_matrix, vhs
from .core.crt import (
    ABERRATION_MODES,
    BLOOM_BLENDS,
    MASK_TYPES,
    SCANLINE_PROFILES,
    bloom,
    chromatic_aberration,
    curvature,
    scanlines,
    subpixel_mask,
    vignette,
)
from .core.dither import DITHER_METHODS, dither_frame
from .core.palettes import PALETTE_NAMES, get_palette, parse_hex_palette
from .core.pixelize import SAMPLE_METHODS, Pixelizer
from .core.quantize import PaletteQuantizer
from .chain_ui import build_chain_ui, collect_chain
temp_dir=modules.config.temp_path+os.path.sep
FILTERS = (
    "Pixelize", "Palette Quantize", "Color Depth", "Dither", "Scanlines",
    "Subpixel Mask", "Bloom", "Chromatic Aberration", "Curvature",
    "Vignette", "VHS", "LCD Dot Matrix",
)

def prepare_comparison(zip_image):
    """Show Before/After using only the final result image."""
    if zip_image:
        return imageslider.hidden_updates()

    target_dir = os.path.join(temp_dir, "batch_filters")
    result_dir = os.path.join(temp_dir, "batch_temp")

    before_files = imageslider.image_files(target_dir)
    result_files = imageslider.image_files(result_dir)

    intermediate_pattern = re.compile(r"_chain_\d+\.[^.]+$",re.IGNORECASE,)

    final_files = [
        path
        for path in result_files
        if not intermediate_pattern.search(os.path.basename(path))
    ]

    if len(before_files) != 1 or len(final_files) != 1:
        return imageslider.hidden_updates()

    return imageslider.update_pair(before_files[0],final_files[0])

def _rgb(image: np.ndarray) -> np.ndarray:
    """Return a contiguous float32 RGB frame in the engine's [0, 255] range."""
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    if arr.shape[2] == 4:
        arr = arr[:, :, :3]
    if arr.dtype.kind == "f" and float(np.nanmax(arr, initial=0)) <= 1.0:
        arr = arr * 255.0
    return np.ascontiguousarray(np.clip(arr, 0, 255).astype(np.float32))


def _apply(frame: np.ndarray, operation: str, **p: object) -> np.ndarray:
    """Dispatch one UI operation to the corresponding upstream engine."""
    if operation == "Pixelize":
        return Pixelizer(int(p["pixel_size"]), method=str(p["sampling"]), pre_blur=float(p["pre_blur"])).process(frame)
    if operation == "Palette Quantize":
        palette = parse_hex_palette(str(p["custom_palette"])) if str(p["custom_palette"]).strip() else get_palette(str(p["palette"]))
        return PaletteQuantizer(palette, dither=str(p["palette_dither"]), strength=float(p["dither_strength"]), serpentine=bool(p["serpentine"])).process(frame)
    if operation == "Color Depth":
        value = int(p["depth_value"])
        if p["depth_mode"] == "bits":
            if not 1 <= value <= 8:
                raise ValueError("Color Depth bits mode requires value 1-8")
            levels = 2 ** value
        else:
            if not 2 <= value <= 64:
                raise ValueError("Color Depth levels mode requires value 2-64")
            levels = value
        return dither_frame(frame, levels, "none")
    if operation == "Dither":
        work = frame
        if p["dither_mode"] == "mono":
            luma = 0.299 * frame[:, :, 0] + 0.587 * frame[:, :, 1] + 0.114 * frame[:, :, 2]
            work = np.stack([luma, luma, luma], axis=-1)
        return dither_frame(work, int(p["levels"]), str(p["dither_method"]), strength=float(p["dither_strength"]), serpentine=bool(p["serpentine"]))
    if operation == "Scanlines":
        return scanlines(frame, int(p["pitch"]), float(p["thickness"]), float(p["scan_strength"]), int(p["offset"]), str(p["profile"]))
    if operation == "Subpixel Mask":
        return subpixel_mask(frame, str(p["mask"]), int(p["mask_pitch"]))
    if operation == "Bloom":
        return bloom(frame, float(p["threshold"]), float(p["radius"]), float(p["intensity"]), str(p["blend"]))
    if operation == "Chromatic Aberration":
        return chromatic_aberration(frame, float(p["aberration_amount"]), str(p["aberration_mode"]))
    if operation == "Curvature":
        return curvature(frame, float(p["curvature_amount"]), float(p["edge_softness"]))[0]
    if operation == "Vignette":
        return vignette(frame, float(p["vignette_strength"]), float(p["vignette_radius"]), float(p["vignette_softness"]))
    if operation == "VHS":
        return vhs(frame, float(p["noise"]), float(p["chroma_bleed"]), float(p["jitter"]), float(p["band"]), int(p["seed"]), str(p["noise_mode"]))
    if operation == "LCD Dot Matrix":
        return lcd_dot_matrix(frame, int(p["cell_size"]), float(p["gap_ratio"]), float(p["grid_dark"]), str(p["shape"]))
    raise ValueError(f"Unknown filter: {operation}")
def apply_filters_to_images(
    images,
    operations,
    params_by_operation,
):
    """Apply the selected Filters chain to images already held in memory.

    This function is used only by the main generation pipeline.

    It does not:
    - read files;
    - write files;
    - use batch_filters;
    - use batch_temp;
    - create ZIP files;
    - return Gradio updates.
    """
    processed_images = []

    operations = [
        operation
        for operation in (operations or [])
        if operation in FILTERS
    ]

    # If the chain is empty, preserve the original images.
    if not operations:
        return list(images)

    for image_index, image in enumerate(images, start=1):
        current = _rgb(image)

        print(
            f"[Filters GENERATION] image "
            f"{image_index}/{len(images)}: start"
        )

        for chain_index, operation in enumerate(operations, start=1):
            operation_params = dict(
                params_by_operation.get(operation, {})
            )

            print(
                f"[Filters GENERATION] image "
                f"{image_index}/{len(images)}; "
                f"stage {chain_index}/{len(operations)}; "
                f"filter: {operation}; "
                f"settings: {operation_params}"
            )

            current = _apply(
                current,
                operation,
                **operation_params,
            )

        processed_images.append(
            np.clip(current, 0, 255).astype(np.uint8)
        )

    return processed_images
def run_chain_batch(*slot_values):
    """Run the selected chain while preserving the existing batch file flow."""
    chain = collect_chain(list(slot_values))
    operations = chain["operations"]
    params_by_operation = chain["params"]
    save_intermediate = chain["save_intermediate"]

    batch_path = f"{temp_dir}batch_filters"
    batch_temp = f"{temp_dir}batch_temp"
    batch_files = sorted(
        name
        for name in os.listdir(batch_path)
        if os.path.isfile(os.path.join(batch_path, name))
    )
    batch_all = len(batch_files)
    passed = 1

    for f_name in batch_files:
        print(
            f"\033[91m[Postprocess QUEUE] {passed} / "
            f"{batch_all}. Filename image: {f_name} \033[0m"
        )
        gr.Info(
            f"Postprocess Batch: start element generation "
            f"{passed}/{batch_all}. Filename image: {f_name}"
        )
        img = Image.open(
            os.path.join(batch_path, f_name)
        )

        yield gr.update(value=img, visible=True), gr.update(visible=False)

        name, _ = os.path.splitext(f_name)
        current_time = datetime.datetime.now()
        time_string = current_time.strftime("%Y-%m-%d_%H-%M-%S")
        final_filename = os.path.join(batch_temp,f"{name}_{time_string}_filters.png",)

        current = _rgb(np.array(img.convert("RGB")))
        for chain_index, operation in enumerate(operations, start=1):
            operation_params = dict(params_by_operation[operation])
            print(
                f"[Postprocess FILTER] start: {operation}; "
                f"settings: {operation_params}"
            )
            current = _apply(
                current,
                operation,
                **operation_params,
            )

            if save_intermediate and chain_index < len(operations):
                intermediate = Image.fromarray(
                    np.clip(current, 0, 255).astype(np.uint8),
                    mode="RGB",
                )
                final_stem, _ = os.path.splitext(final_filename)
                intermediate_filename = (
                    f"{final_stem}_chain_{chain_index}.png"
                )
                
                intermediate.save(intermediate_filename)

        pil = Image.fromarray(
            np.clip(current, 0, 255).astype(np.uint8),
            mode="RGB",
        )
        pil.save(final_filename)
        passed += 1

    return gr.update(value=None, visible=False), gr.update(visible=True)

def _vis(operation: str):
    """Return visibility updates for parameter rows in the same order as UI."""
    groups = {
        "pixel": operation == "Pixelize", "palette": operation == "Palette Quantize",
        "depth": operation == "Color Depth", "dither": operation == "Dither",
        "scan": operation == "Scanlines", "mask": operation == "Subpixel Mask",
        "bloom": operation == "Bloom", "aberration": operation == "Chromatic Aberration",
        "curvature": operation == "Curvature", "vignette": operation == "Vignette",
        "vhs": operation == "VHS", "lcd": operation == "LCD Dot Matrix",
    }
    return [gr.update(visible=groups[name]) for name in groups]

def gui(generator):
    """Build the Retro Filters batch UI with a sequential chain editor."""
    with gr.Row():
        filters_enabled = gr.Checkbox(label="Enabled", value=False,visible=generator)
        tutorial("filters/readme.md")
    with gr.Row(visible=not generator):
        file_in, files_single, image_single, enable_zip_image, file_out, preview, image_out = batch.ui_batch()
    with gr.Row(visible=not generator):
        compare_slider, compare_accordion = imageslider.create_ui()
    slot_inputs = build_chain_ui(generator)

    with gr.Row(visible=not generator):
        apply_btn = gr.Button("Start", variant="primary")
    with gr.Row(visible=False):
        ext_dir = gr.Textbox(value="batch_filters", visible=False)
    with gr.Row():
        gr.HTML('* \"Filters\" is powered by Mistress-Lukutar. <a href="https://github.com/Mistress-Lukutar/ComfyUI-LukutarRetro" target="_blank">\U0001F4D4 Document</a>')
    apply_btn.click(lambda: (gr.update(visible=True, interactive=False),gr.update(visible=False),gr.update(visible=False)),outputs=[apply_btn, file_out, image_out]) \
        .then(fn=imageslider.reset,outputs=[compare_slider, compare_accordion]) \
        .then(fn=batch.clear_dirs,inputs=ext_dir) \
        .then(fn=batch.unzip_file,inputs=[file_in, files_single, enable_zip_image, ext_dir]) \
        .then(fn=run_chain_batch,inputs=slot_inputs,outputs=[preview, file_out],show_progress=False) \
        .then(lambda: (gr.update(visible=True, interactive=True),gr.update(visible=False)),outputs=[file_out, preview]) \
        .then(fn=batch.output_zip_image,outputs=[image_out, file_out]) \
        .then(fn=lambda zip_image: prepare_comparison(zip_image),inputs=[enable_zip_image],outputs=[compare_slider, compare_accordion]) \
        .then(lambda: gr.update(visible=True, interactive=True),outputs=apply_btn)
    return [filters_enabled, *slot_inputs]