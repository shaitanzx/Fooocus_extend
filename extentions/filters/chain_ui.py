from __future__ import annotations

from typing import Any

import gradio as gr

from .core.analog import DOT_SHAPES, NOISE_MODES
from .core.crt import (
    ABERRATION_MODES,
    BLOOM_BLENDS,
    MASK_TYPES,
    SCANLINE_PROFILES,
)
from .core.dither import DITHER_METHODS
from .core.palettes import PALETTE_NAMES
from .core.pixelize import SAMPLE_METHODS

FILTERS = [
    "Pixelize",
    "Palette Quantize",
    "Color Depth",
    "Dither",
    "Scanlines",
    "Subpixel Mask",
    "Bloom",
    "Chromatic Aberration",
    "Curvature",
    "Vignette",
    "VHS",
    "LCD Dot Matrix",
]
DEFAULT_CHAIN = ["Pixelize", "Scanlines", "Bloom"]


def _summary(selected: list[str] | None) -> str:
    selected = selected or []
    if not selected:
        return "Input → Output"
    return "Input → " + " → ".join(selected) + " → Output"


def _visibility(selected: str | None):
    """Return visibility updates for the single filter-settings panel."""
    return [gr.update(visible=name == selected) for name in FILTERS]


def update_settings_ui(selected: str | None):
    """Show settings only for the filter selected in the settings dropdown."""
    return _visibility(selected)


def _params(values: list[Any]) -> dict[str, dict[str, Any]]:
    """Convert flat Gradio values into per-filter parameter dictionaries."""
    (
        pixel_size,
        sampling,
        pre_blur,
        palette,
        custom_palette,
        palette_dither,
        palette_dither_strength,
        palette_serpentine,
        depth_mode,
        depth_value,
        levels,
        dither_mode,
        dither_method,
        dither_strength,
        dither_serpentine,
        pitch,
        thickness,
        scan_strength,
        offset,
        profile,
        mask,
        mask_pitch,
        threshold,
        radius,
        intensity,
        blend,
        aberration_amount,
        aberration_mode,
        curvature_amount,
        edge_softness,
        vignette_strength,
        vignette_radius,
        vignette_softness,
        noise,
        chroma_bleed,
        jitter,
        band,
        seed,
        noise_mode,
        cell_size,
        gap_ratio,
        grid_dark,
        shape,
    ) = values

    return {
        "Pixelize": {
            "pixel_size": pixel_size,
            "sampling": sampling,
            "pre_blur": pre_blur,
        },
        "Palette Quantize": {
            "palette": palette,
            "custom_palette": custom_palette,
            "palette_dither": palette_dither,
            "dither_strength": palette_dither_strength,
            "serpentine": palette_serpentine,
        },
        "Color Depth": {
            "depth_mode": depth_mode,
            "depth_value": depth_value,
        },
        "Dither": {
            "levels": levels,
            "dither_mode": dither_mode,
            "dither_method": dither_method,
            "dither_strength": dither_strength,
            "serpentine": dither_serpentine,
        },
        "Scanlines": {
            "pitch": pitch,
            "thickness": thickness,
            "scan_strength": scan_strength,
            "offset": offset,
            "profile": profile,
        },
        "Subpixel Mask": {
            "mask": mask,
            "mask_pitch": mask_pitch,
        },
        "Bloom": {
            "threshold": threshold,
            "radius": radius,
            "intensity": intensity,
            "blend": blend,
        },
        "Chromatic Aberration": {
            "aberration_amount": aberration_amount,
            "aberration_mode": aberration_mode,
        },
        "Curvature": {
            "curvature_amount": curvature_amount,
            "edge_softness": edge_softness,
        },
        "Vignette": {
            "vignette_strength": vignette_strength,
            "vignette_radius": vignette_radius,
            "vignette_softness": vignette_softness,
        },
        "VHS": {
            "noise": noise,
            "chroma_bleed": chroma_bleed,
            "jitter": jitter,
            "band": band,
            "seed": seed,
            "noise_mode": noise_mode,
        },
        "LCD Dot Matrix": {
            "cell_size": cell_size,
            "gap_ratio": gap_ratio,
            "grid_dark": grid_dark,
            "shape": shape,
        },
    }


def collect_chain(values: list[Any]) -> dict[str, Any]:
    """Build the ordered chain at execution time, like xyz_grid.run()."""
    selected = list(values[0] or [])
    return {
        "operations": [name for name in selected if name in FILTERS],
        "save_intermediate": bool(values[1]),
        # values[2] is the settings dropdown; it selects the visible editor
        # only and is deliberately not used to choose the chain parameters.
        "params": _params(values[3:]),
    }


def build_chain_ui():
    """Create the ordered chain selector and all real filter controls."""
    chain_filters = gr.Dropdown(
        choices=FILTERS,
        value=DEFAULT_CHAIN,
        multiselect=True,
        label="Filter chain (selection order)",
        interactive=True,
    )
    save_intermediate = gr.Checkbox(
        value=False,
        label="Save result after each filter",
        info="Saves only filters before the last one; the final result is always saved.",
    )

    with gr.Group():
        settings_filter = gr.Dropdown(
            choices=FILTERS,
            value=FILTERS[0],
            label="Filter to configure",
            interactive=True,
        )

        groups = []
        controls = []

        with gr.Group(visible=True) as pixel_group:
            pixel_size = gr.Slider(1, 128, value=8, step=1, label="Pixel size")
            sampling = gr.Dropdown(SAMPLE_METHODS, value="mode", label="Sampling")
            pre_blur = gr.Slider(0, 10, value=0, step=0.1, label="Pre-blur")
        groups.append(pixel_group)
        controls += [pixel_size, sampling, pre_blur]

        with gr.Group(visible=False) as palette_group:
            palette = gr.Dropdown(PALETTE_NAMES, value="pico8_16", label="Built-in palette")
            custom_palette = gr.Textbox(label="Custom palette (hex colors)", lines=2)
            palette_dither = gr.Dropdown(DITHER_METHODS, value="none", label="Dithering")
            palette_dither_strength = gr.Slider(0, 1, value=1, step=0.05, label="Dither strength")
            palette_serpentine = gr.Checkbox(value=True, label="Serpentine diffusion")
        groups.append(palette_group)
        controls += [palette, custom_palette, palette_dither, palette_dither_strength, palette_serpentine]

        with gr.Group(visible=False) as depth_group:
            depth_mode = gr.Radio(["bits", "levels"], value="bits", label="Interpret value as")
            depth_value = gr.Slider(1, 64, value=4, step=1, label="Bits / levels")
        groups.append(depth_group)
        controls += [depth_mode, depth_value]

        with gr.Group(visible=False) as dither_group:
            levels = gr.Slider(2, 32, value=4, step=1, label="Levels per channel")
            dither_mode = gr.Radio(["color", "mono"], value="color", label="Mode")
            dither_method = gr.Dropdown([x for x in DITHER_METHODS if x != "none"], value="bayer4", label="Method")
            dither_strength = gr.Slider(0, 1, value=1, step=0.05, label="Strength")
            dither_serpentine = gr.Checkbox(value=True, label="Serpentine diffusion")
        groups.append(dither_group)
        controls += [levels, dither_mode, dither_method, dither_strength, dither_serpentine]

        with gr.Group(visible=False) as scan_group:
            pitch = gr.Slider(1, 32, value=2, step=1, label="Pitch")
            thickness = gr.Slider(0, 1, value=0.5, step=0.05, label="Thickness")
            scan_strength = gr.Slider(0, 1, value=0.35, step=0.05, label="Strength")
            offset = gr.Slider(-64, 64, value=0, step=1, label="Offset")
            profile = gr.Dropdown(SCANLINE_PROFILES, value="hard", label="Profile")
        groups.append(scan_group)
        controls += [pitch, thickness, scan_strength, offset, profile]

        with gr.Group(visible=False) as mask_group:
            mask = gr.Dropdown(MASK_TYPES, value="aperture_grille", label="Mask")
            mask_pitch = gr.Slider(1, 16, value=1, step=1, label="Pitch")
        groups.append(mask_group)
        controls += [mask, mask_pitch]

        with gr.Group(visible=False) as bloom_group:
            threshold = gr.Slider(0, 1, value=0.7, step=0.05, label="Threshold")
            radius = gr.Slider(0.5, 100, value=8, step=0.5, label="Radius")
            intensity = gr.Slider(0, 4, value=0.5, step=0.05, label="Intensity")
            blend = gr.Dropdown(BLOOM_BLENDS, value="screen", label="Blend")
        groups.append(bloom_group)
        controls += [threshold, radius, intensity, blend]

        with gr.Group(visible=False) as aberration_group:
            aberration_amount = gr.Slider(0, 20, value=2, step=0.5, label="Amount")
            aberration_mode = gr.Dropdown(ABERRATION_MODES, value="radial", label="Mode")
        groups.append(aberration_group)
        controls += [aberration_amount, aberration_mode]

        with gr.Group(visible=False) as curvature_group:
            curvature_amount = gr.Slider(0, 1, value=0.15, step=0.05, label="Amount")
            edge_softness = gr.Slider(0, 32, value=0, step=0.5, label="Edge softness")
        groups.append(curvature_group)
        controls += [curvature_amount, edge_softness]

        with gr.Group(visible=False) as vignette_group:
            vignette_strength = gr.Slider(0, 1, value=0.4, step=0.05, label="Strength")
            vignette_radius = gr.Slider(0, 1, value=0.8, step=0.05, label="Radius")
            vignette_softness = gr.Slider(0, 1, value=0.5, step=0.05, label="Softness")
        groups.append(vignette_group)
        controls += [vignette_strength, vignette_radius, vignette_softness]

        with gr.Group(visible=False) as vhs_group:
            noise = gr.Slider(0, 1, value=0.15, step=0.05, label="Noise")
            chroma_bleed = gr.Slider(0, 1, value=0.5, step=0.05, label="Chroma bleed")
            jitter = gr.Slider(0, 1, value=0.3, step=0.05, label="Jitter")
            band = gr.Slider(0, 1, value=0.3, step=0.05, label="Band")
            seed = gr.Slider(0, 2147483647, value=0, step=1, label="Seed (0 = random)")
            noise_mode = gr.Dropdown(NOISE_MODES, value="color", label="Noise mode")
        groups.append(vhs_group)
        controls += [noise, chroma_bleed, jitter, band, seed, noise_mode]

        with gr.Group(visible=False) as lcd_group:
            cell_size = gr.Slider(1, 64, value=8, step=1, label="Cell size")
            gap_ratio = gr.Slider(0, 0.5, value=0.15, step=0.05, label="Gap ratio")
            grid_dark = gr.Slider(0, 1, value=0.6, step=0.05, label="Grid dark")
            shape = gr.Dropdown(DOT_SHAPES, value="square", label="Shape")
        groups.append(lcd_group)
        controls += [cell_size, gap_ratio, grid_dark, shape]

    settings_filter.change(
        update_settings_ui,
        inputs=settings_filter,
        outputs=groups,
        show_progress=False,
    )
    return [chain_filters, save_intermediate, settings_filter, *controls]
