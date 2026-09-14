"""Isolated Diffusers runtime for local image-to-video generation.

This module is launched with the Python executable in ``path_video_runtime``.
It communicates with Fooocus by emitting one JSON object per line.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path


def emit(event: str, **payload) -> None:
    print(json.dumps({"event": event, **payload}, ensure_ascii=False), flush=True)


def rounded_size(image, max_area: int, multiple: int = 32) -> tuple[int, int]:
    width, height = image.size
    scale = math.sqrt(max_area / float(max(width * height, 1)))
    if scale > 1:
        scale = 1
    width = max(multiple, round(width * scale / multiple) * multiple)
    height = max(multiple, round(height * scale / multiple) * multiple)
    return width, height


def wan_profile(profile: str) -> dict:
    profiles = {
        "8 GB": {
            "max_area": 512 * 512,
            "num_frames": 49,
            "steps": 24,
            "fps": 16,
            "group_offload": True,
        },
        "12 GB": {
            "max_area": 832 * 480,
            "num_frames": 81,
            "steps": 30,
            "fps": 16,
            "group_offload": True,
        },
        "16 GB+": {
            "max_area": 1280 * 704,
            "num_frames": 121,
            "steps": 40,
            "fps": 24,
            "group_offload": False,
        },
    }
    return profiles.get(profile, profiles["12 GB"]).copy()


def requested_max_area(resolution: str) -> int | None:
    return {
        "Low (512px)": 512 * 512,
        "Medium (480p)": 832 * 480,
        "High (720p)": 1280 * 704,
    }.get(resolution)


def h3_num_frames(duration: float) -> int:
    requested = max(5.0, min(15.0, duration)) * 24
    return min(362, max(124, 17 * math.ceil((requested - 5) / 17) + 5))


def load_image(path: str):
    from PIL import Image

    return Image.open(path).convert("RGB")


def run_wan(request: dict) -> None:
    import torch
    from diffusers import AutoencoderKLWan, UniPCMultistepScheduler, WanImageToVideoPipeline
    from diffusers.utils import export_to_video

    model_path = request["model_path"]
    profile = wan_profile(request["profile"])
    requested_area = requested_max_area(request.get("resolution", "Auto"))
    if requested_area:
        profile["max_area"] = min(profile["max_area"], requested_area)
    if request.get("duration"):
        requested_frames = int(float(request["duration"]) * profile["fps"])
        # Wan's temporal VAE expects 4k+1 frames.
        profile["num_frames"] = max(17, min(profile["num_frames"], requested_frames // 4 * 4 + 1))

    image = load_image(request["image_path"])
    width, height = rounded_size(image, profile["max_area"])
    image = image.resize((width, height))
    emit("progress", percent=3, message="Loading Wan video VAE …")

    vae = AutoencoderKLWan.from_pretrained(
        model_path,
        subfolder="vae",
        dtype=torch.float32,
        local_files_only=True,
    )
    if hasattr(vae, "enable_tiling"):
        vae.enable_tiling()
    if hasattr(vae, "enable_slicing"):
        vae.enable_slicing()

    emit("progress", percent=8, message="Loading Wan 2.2 with CPU offload …")
    pipe = WanImageToVideoPipeline.from_pretrained(
        model_path,
        vae=vae,
        dtype=torch.bfloat16,
        local_files_only=True,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=5.0)

    if profile["group_offload"]:
        from diffusers.hooks import apply_group_offloading

        apply_group_offloading(
            pipe.text_encoder,
            onload_device=torch.device("cuda"),
            offload_device=torch.device("cpu"),
            offload_type="block_level",
            num_blocks_per_group=4,
        )
        pipe.transformer.enable_group_offload(
            onload_device=torch.device("cuda"),
            offload_device=torch.device("cpu"),
            offload_type="leaf_level",
            use_stream=True,
        )
        pipe.vae.to("cuda")
    else:
        pipe.enable_model_cpu_offload()

    last_percent = -1

    def step_callback(_pipe, step, _timestep, callback_kwargs):
        nonlocal last_percent
        percent = 12 + int(78 * (step + 1) / profile["steps"])
        if percent != last_percent:
            emit(
                "progress",
                percent=percent,
                message=f"Generating frame latents · step {step + 1}/{profile['steps']}",
            )
            last_percent = percent
        return callback_kwargs

    generator = torch.Generator(device="cpu").manual_seed(int(request["seed"]))
    emit(
        "progress",
        percent=12,
        message=f"Generating {profile['num_frames']} frames at {width}×{height} …",
    )
    result = pipe(
        image=image,
        prompt=request["prompt"],
        negative_prompt=request.get("negative_prompt") or None,
        height=height,
        width=width,
        num_frames=profile["num_frames"],
        num_inference_steps=profile["steps"],
        guidance_scale=float(request.get("guidance_scale", 5.0)),
        generator=generator,
        callback_on_step_end=step_callback,
    )

    output_path = request["output_path"]
    emit("progress", percent=92, message="Encoding MP4 …")
    export_to_video(result.frames[0], output_path, fps=profile["fps"])
    emit(
        "complete",
        path=output_path,
        fps=profile["fps"],
        frames=profile["num_frames"],
        width=width,
        height=height,
        audio=False,
    )


def _load_h3_quantized(model_path: str):
    import torch
    from diffusers import MiniMaxH3Transformer3DModel, ModularPipeline, TorchAoConfig
    from diffusers.hooks import apply_group_offloading
    from torchao.quantization import Int8WeightOnlyConfig
    from transformers import Qwen3VLForConditionalGeneration
    from transformers import TorchAoConfig as TransformersTorchAoConfig

    pipe = ModularPipeline.from_pretrained(model_path, workflow="fl2va", local_files_only=True)
    pipe.update_components(
        transformer=MiniMaxH3Transformer3DModel.from_pretrained(
            model_path,
            subfolder="transformer",
            dtype=torch.bfloat16,
            local_files_only=True,
            quantization_config=TorchAoConfig(
                Int8WeightOnlyConfig(version=2),
                modules_to_not_convert=[
                    "proj_in",
                    "audio_proj_in",
                    "context_embedder",
                    "time_embedder",
                    "time_proj",
                    "token_refiner",
                    "norm_out",
                    "proj_out",
                    "audio_proj_out",
                ],
            ),
            low_cpu_mem_usage=False,
        ),
        text_encoder=Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            subfolder="text_encoder",
            dtype=torch.bfloat16,
            local_files_only=True,
            quantization_config=TransformersTorchAoConfig(
                Int8WeightOnlyConfig(version=2),
                modules_to_not_convert=[
                    "model.visual",
                    "model.language_model.embed_tokens",
                    "model.language_model.norm",
                    "lm_head",
                ],
            ),
        ),
    )
    pipe.load_components(workflow="fl2va", dtype=torch.bfloat16)
    pipe.transformer.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    offload = {
        "onload_device": torch.device("cuda"),
        "offload_device": torch.device("cpu"),
    }
    pipe.transformer.enable_group_offload(
        offload_type="block_level",
        num_blocks_per_group=1,
        use_stream=True,
        **offload,
    )
    apply_group_offloading(
        pipe.text_encoder.model,
        offload_type="leaf_level",
        use_stream=True,
        **offload,
    )
    apply_group_offloading(
        pipe.vae,
        offload_type="leaf_level",
        use_stream=False,
        **offload,
    )
    pipe.audio_vae.to("cuda")
    return pipe


def run_h3(request: dict) -> None:
    import torch
    from diffusers.utils.export_utils import encode_video

    emit("progress", percent=3, message="Loading and quantizing MiniMax H3-Base …")
    pipe = _load_h3_quantized(request["model_path"])
    image = load_image(request["image_path"])
    max_area = 960 * 544 if request["profile"] != "16 GB+" else 1344 * 768
    requested_area = requested_max_area(request.get("resolution", "Auto"))
    if requested_area:
        max_area = min(max_area, requested_area)
    width, height = rounded_size(image, max_area)
    image = image.resize((width, height))
    frames = h3_num_frames(float(request.get("duration", 5)))
    generator = torch.Generator(device="cpu").manual_seed(int(request["seed"]))
    emit(
        "progress",
        percent=12,
        message=f"Generating H3 video and stereo audio at {width}×{height} …",
    )
    results = pipe(
        prompt=request["prompt"],
        image=image,
        height=height,
        width=width,
        num_frames=frames,
        num_inference_steps=int(request.get("steps", 30)),
        generator=generator,
        output=["videos", "audio", "sampling_rate"],
    )
    emit("progress", percent=92, message="Encoding video and stereo audio …")
    encode_video(
        results["videos"][0],
        fps=24,
        output_path=request["output_path"],
        audio=results["audio"][0],
        audio_sample_rate=results["sampling_rate"],
    )
    emit(
        "complete",
        path=request["output_path"],
        fps=24,
        frames=frames,
        width=width,
        height=height,
        audio=True,
    )


def run(request: dict) -> None:
    started_at = time.time()
    model = request["model"]
    if model == "wan":
        run_wan(request)
    elif model == "h3":
        run_h3(request)
    else:
        raise ValueError(f"Unknown video model: {model}")
    emit("timing", seconds=round(time.time() - started_at, 2))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    args = parser.parse_args()
    try:
        request = json.loads(Path(args.request).read_text(encoding="utf-8"))
        run(request)
        return 0
    except KeyboardInterrupt:
        emit("cancelled", message="Video generation cancelled.")
        return 130
    except Exception as exc:
        emit("error", message=str(exc), type=type(exc).__name__)
        return 1


if __name__ == "__main__":
    sys.exit(main())