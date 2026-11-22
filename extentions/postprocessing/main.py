"""Lukutar Retro Filters integration for Fooocus Extend.



The image engines are kept in ``core`` and are copied from the upstream
ComfyUI-LukutarRetro project. This module only supplies Fooocus/Gradio glue:
one uploaded image, one selected filter, full parameter controls, preview and
output saving. No model, checkpoint, LoRA, sampler or SD-family state is used.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
from PIL import Image

import modules.config
import extentions.batch as batch
import modules.imageslider as imageslider
from extentions.tutorial import tutorial

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
temp_dir=modules.config.temp_path+os.path.sep
FILTERS = (
    "Pixelize", "Palette Quantize", "Color Depth", "Dither", "Scanlines",
    "Subpixel Mask", "Bloom", "Chromatic Aberration", "Curvature",
    "Vignette", "VHS", "LCD Dot Matrix",
)

def prepare_comparison(zip_image):
    """Apply FaceEnhancer-specific rules and delegate to the shared slider."""
    if zip_image:
        return imageslider.hidden_updates()

    target_dir = os.path.join(temp_dir, "batch_post")
    result_dir = os.path.join(temp_dir, "batch_temp")

    return imageslider.update_from_directories(target_dir, result_dir) 
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
    raise ValueError(f"Unknown Retro filter: {operation}")
def run(op, *values):
    names = ["pixel_size", "sampling", "pre_blur", "palette", "custom_palette", "palette_dither", "dither_strength", "serpentine",
                "depth_mode", "depth_value", "levels", "dither_mode", "dither_method", "dither_strength_2", "serpentine_2",
                "pitch", "thickness", "scan_strength", "offset", "profile", "mask", "mask_pitch", "threshold", "radius", "intensity",
                "blend", "aberration_amount", "aberration_mode", "curvature_amount", "edge_softness", "vignette_strength",
                "vignette_radius", "vignette_softness", "noise", "chroma_bleed", "jitter", "band", "seed", "noise_mode",
                "cell_size", "gap_ratio", "grid_dark", "shape"]
    data = dict(zip(names, values))
    data["dither_strength"] = data["dither_strength_2"] if op == "Dither" else data["dither_strength"]
    data["serpentine"] = data["serpentine_2"] if op == "Dither" else data["serpentine"]

    batch_path=f"{temp_dir}batch_post"
    batch_temp=f"{temp_dir}batch_temp"
    batch_files=sorted([name for name in os.listdir(batch_path) if os.path.isfile(os.path.join(batch_path, name))])
    batch_all=len(batch_files)
    passed=1
    for f_name in batch_files:
        print (f"\033[91m[Postprocess QUEUE] {passed} / {batch_all}. Filename image: {f_name} \033[0m")
        gr.Info(f"Postprocess Batch: start element generation {passed}/{batch_all}. Filename image: {f_name}")
        img = Image.open(batch_path+os.path.sep+f_name)

        yield gr.update(value=img,visible=True),gr.update(visible=False)
        image=np.array(img)

        result = _apply(_rgb(image), operation, **data)

        pil = Image.fromarray(np.clip(result, 0, 255).astype(np.uint8), mode="RGB")
        name, _ = os.path.splitext(f_name)
        current_time = datetime.datetime.now()
        time_string = current_time.strftime("%Y-%m-%d_%H-%M-%S")
        suf = f'{time_string}_logo.png'
        filename =  batch_temp + os.path.sep + name + suf
        pil.save(filename)
        passed+=1
    return gr.update(value=None,visible=False),gr.update(visible=True)


def process(operation: str, **params: object):


    """Process an image and return a PIL preview plus a saved PNG path."""
    if image is None:
        raise gr.Error("Upload an image first.")
    result = _apply(_rgb(image), operation, **params)
    pil = Image.fromarray(np.clip(result, 0, 255).astype(np.uint8), mode="RGB")
    out_dir = Path(modules.config.path_outputs) / "lukutar_retro"
    out_dir.mkdir(parents=True, exist_ok=True)
    fd, path = tempfile.mkstemp(prefix="retro_", suffix=".png", dir=out_dir)
    os.close(fd)
    pil.save(path, compress_level=1)
    return pil, path


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


def gui():
    """Build the Retro Filters tab in the visual style of Fooocus Extend."""
    with gr.Row():
        tutorial("lukutar_retro.md")
    with gr.Row():
        file_in,files_single,image_single,enable_zip_image,file_out,preview, image_out = batch.ui_batch()
    with gr.Row():
        compare_slider, compare_accordion = imageslider.create_ui()
        operation = gr.Dropdown(FILTERS, value="Pixelize", label="Filter", interactive=True)
    # with gr.Row():
    #     with gr.Column(scale=1):
    #         source = gr.Image(label="Source image", type="numpy", image_mode="RGB", height=420)
    #         operation = gr.Dropdown(FILTERS, value="Pixelize", label="Filter", interactive=True)
    #         apply_btn = gr.Button("Apply Retro Filter", variant="primary")
    #     with gr.Column(scale=1):
    #         preview = gr.Image(label="Processed preview", type="pil", height=420, interactive=False)
    #         output_file = gr.File(label="Saved PNG", interactive=False)

    with gr.Accordion("Filter parameters", open=True):
        rows = {}
        with gr.Group(visible=True) as rows["pixel"]:
            pixel_size = gr.Slider(1, 128, value=8, step=1, label="Pixel size",interactive=True)
            sampling = gr.Dropdown(SAMPLE_METHODS, value="mode", label="Sampling",interactive=True)
            pre_blur = gr.Slider(0, 10, value=0, step=0.1, label="Pre-blur",interactive=True)
        with gr.Group(visible=False) as rows["palette"]:
            palette = gr.Dropdown(PALETTE_NAMES, value="pico8_16", label="Built-in palette",interactive=True)
            custom_palette = gr.Textbox(label="Custom palette (hex colors)", placeholder="#0f380f, #9bbc0f", lines=2,interactive=True)
            palette_dither = gr.Dropdown(DITHER_METHODS, value="none", label="Dithering",interactive=True)
            dither_strength = gr.Slider(0, 1, value=1, step=0.05, label="Dither strength",interactive=True)
            serpentine = gr.Checkbox(value=True, label="Serpentine diffusion",interactive=True)
        with gr.Group(visible=False) as rows["depth"]:
            depth_mode = gr.Radio(["bits", "levels"], value="bits", label="Interpret value as",interactive=True)
            depth_value = gr.Slider(1, 64, value=4, step=1, label="Bits / levels",interactive=True)
        with gr.Group(visible=False) as rows["dither"]:
            levels = gr.Slider(2, 32, value=4, step=1, label="Levels per channel",interactive=True)
            dither_mode = gr.Radio(["color", "mono"], value="color", label="Mode",interactive=True)
            dither_method = gr.Dropdown([x for x in DITHER_METHODS if x != "none"], value="bayer4", label="Method",interactive=True)
            dither_strength_2 = gr.Slider(0, 1, value=1, step=0.05, label="Strength",interactive=True)
            serpentine_2 = gr.Checkbox(value=True, label="Serpentine diffusion",interactive=True)
        with gr.Group(visible=False) as rows["scan"]:
            pitch = gr.Slider(1, 32, value=2, step=1, label="Pitch",interactive=True)
            thickness = gr.Slider(0, 1, value=0.5, step=0.05, label="Thickness",interactive=True)
            scan_strength = gr.Slider(0, 1, value=0.35, step=0.05, label="Strength",interactive=True)
            offset = gr.Slider(-64, 64, value=0, step=1, label="Offset",interactive=True)
            profile = gr.Dropdown(SCANLINE_PROFILES, value="hard", label="Profile",interactive=True)
        with gr.Group(visible=False) as rows["mask"]:
            mask = gr.Dropdown(MASK_TYPES, value="aperture_grille", label="Mask",interactive=True)
            mask_pitch = gr.Slider(1, 16, value=1, step=1, label="Pitch",interactive=True)
        with gr.Group(visible=False) as rows["bloom"]:
            threshold = gr.Slider(0, 1, value=0.7, step=0.05, label="Threshold",interactive=True)
            radius = gr.Slider(0.5, 100, value=8, step=0.5, label="Radius",interactive=True)
            intensity = gr.Slider(0, 4, value=0.5, step=0.05, label="Intensity",interactive=True)
            blend = gr.Dropdown(BLOOM_BLENDS, value="screen", label="Blend",interactive=True)
        with gr.Group(visible=False) as rows["aberration"]:
            aberration_amount = gr.Slider(0, 20, value=2, step=0.5, label="Amount",interactive=True)
            aberration_mode = gr.Dropdown(ABERRATION_MODES, value="radial", label="Mode",interactive=True)
        with gr.Group(visible=False) as rows["curvature"]:
            curvature_amount = gr.Slider(0, 1, value=0.15, step=0.05, label="Amount",interactive=True)
            edge_softness = gr.Slider(0, 32, value=0, step=0.5, label="Edge softness",interactive=True)
        with gr.Group(visible=False) as rows["vignette"]:
            vignette_strength = gr.Slider(0, 1, value=0.4, step=0.05, label="Strength",interactive=True)
            vignette_radius = gr.Slider(0, 1, value=0.8, step=0.05, label="Radius",interactive=True)
            vignette_softness = gr.Slider(0, 1, value=0.5, step=0.05, label="Softness",interactive=True)
        with gr.Group(visible=False) as rows["vhs"]:
            noise = gr.Slider(0, 1, value=0.15, step=0.05, label="Noise",interactive=True)
            chroma_bleed = gr.Slider(0, 1, value=0.5, step=0.05, label="Chroma bleed",interactive=True)
            jitter = gr.Slider(0, 1, value=0.3, step=0.05, label="Jitter",interactive=True)
            band = gr.Slider(0, 1, value=0.3, step=0.05, label="Band",interactive=True)
            seed = gr.Slider(0, 2147483647, value=0, step=1, label="Seed (0 = random)",interactive=True)
            noise_mode = gr.Dropdown(NOISE_MODES, value="color", label="Noise mode",interactive=True)
        with gr.Group(visible=False) as rows["lcd"]:
            cell_size = gr.Slider(1, 64, value=8, step=1, label="Cell size",interactive=True)
            gap_ratio = gr.Slider(0, 0.5, value=0.15, step=0.05, label="Gap ratio",interactive=True)
            grid_dark = gr.Slider(0, 1, value=0.6, step=0.05, label="Grid dark",interactive=True)
            shape = gr.Dropdown(DOT_SHAPES, value="square", label="Shape",interactive=True)
    with gr.Row():
        apply_btn = gr.Button("Start postprocessing", variant="primary")
    with gr.Row(visible=False):
        ext_dir=gr.Textbox(value='batch_post',visible=False)
    visibility_outputs = list(rows.values())
    operation.change(_vis, inputs=operation, outputs=visibility_outputs, show_progress=False)
    params = [pixel_size, sampling, pre_blur, palette, custom_palette, palette_dither, dither_strength, serpentine,
              depth_mode, depth_value, levels, dither_mode, dither_method, dither_strength_2, serpentine_2,
              pitch, thickness, scan_strength, offset, profile, mask, mask_pitch, threshold, radius, intensity,
              blend, aberration_amount, aberration_mode, curvature_amount, edge_softness, vignette_strength,
              vignette_radius, vignette_softness, noise, chroma_bleed, jitter, band, seed, noise_mode,
              cell_size, gap_ratio, grid_dark, shape]

    # apply_btn.click(run, inputs=[source, operation] + params, outputs=[preview, output_file], show_progress=True)
    # return source, preview, output_file
    apply_btn.click(lambda: 
             (gr.update(visible=True, interactive=False),gr.update(visible=False),gr.update(visible=False)),outputs=[apply_btn,file_out,image_out]) \
              .then(fn=imageslider.reset, outputs=[compare_slider, compare_accordion]) \
              .then(fn=batch.clear_dirs,inputs=ext_dir) \
              .then(fn=batch.unzip_file,inputs=[file_in,files_single,enable_zip_image,ext_dir]) \
              .then(fn=run, inputs=[operation]+params,
                        outputs=[preview,file_out],show_progress=False) \
              .then(lambda: (gr.update(visible=True, interactive=True),gr.update(visible=False)),outputs=[file_out,preview],show_progress=False) \
              .then(fn=batch.output_zip_image, outputs=[image_out,file_out]) \
              .then(fn=lambda zip_image: prepare_comparison(zip_image),inputs=[enable_zip_image],outputs=[compare_slider, compare_accordion]) \
              .then(lambda: (gr.update(visible=True, interactive=True)),outputs=apply_btn)


             