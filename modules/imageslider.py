"""Shared before/after comparison slider for Fooocus extensions."""

from __future__ import annotations

import os
import traceback
from pathlib import Path
from urllib.parse import quote

import gradio as gr
from PIL import Image as PILImage


def _image_url_and_size(image_path: str | Path) -> tuple[str, int, int]:
    """Build a direct Gradio file URL and read image dimensions."""
    absolute_path = Path(os.path.abspath(os.fspath(image_path)))

    with PILImage.open(absolute_path) as image:
        width, height = image.size

    encoded_path = quote(absolute_path.as_posix(), safe="/:")
    return f"/file={encoded_path}", width, height


def render_html(before: str | Path, after: str | Path) -> str:
    """Return before/after HTML referencing the original files directly."""
    before_src, image_width, image_height = _image_url_and_size(before)
    after_src, _, _ = _image_url_and_size(after)

    return f"""
<div class="fooocus-image-compare"
     style="--fooocus-compare-position: 50%;
            --fooocus-compare-aspect-ratio: {image_width} / {image_height};">
  <div class="fooocus-image-compare__canvas">
    <img class="fooocus-image-compare__image fooocus-image-compare__before"
         src="{before_src}" alt="До обработки" draggable="false">
    <img class="fooocus-image-compare__image fooocus-image-compare__after"
         src="{after_src}" alt="После обработки" draggable="false">
    <span class="fooocus-image-compare__label
                 fooocus-image-compare__label--before">До</span>
    <span class="fooocus-image-compare__label
                 fooocus-image-compare__label--after">После</span>
    <span class="fooocus-image-compare__divider" aria-hidden="true">
      <span class="fooocus-image-compare__handle">↔</span>
    </span>
    <input class="fooocus-image-compare__range"
           type="range" min="0" max="100" step="0.01" value="50"
           aria-label="Положение разделителя сравнения">
  </div>
</div>
"""


def create_ui(label: str = "Сравнение до/после"):
    """Create the collapsed accordion and its HTML component."""
    with gr.Accordion(label, open=False, visible=False) as accordion:
        html = gr.HTML(value="", show_label=False)
    return html, accordion


def hidden_updates():
    """Return updates that clear and hide the comparison view."""
    return gr.update(value=""), gr.update(visible=False, open=False)


def reset():
    """Callback used when a new processing run starts."""
    return hidden_updates()


def image_files(directory: str | Path) -> list[str]:
    """List all ordinary files directly inside a directory, without extension filtering."""
    directory = os.fspath(directory)
    if not os.path.isdir(directory):
        return []

    try:
        return [
            os.path.join(directory, name)
            for name in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, name))
        ]
    except OSError:
        return []


def has_single_image(directory: str | Path) -> bool:
    """Return whether a directory contains exactly one file."""
    return len(image_files(directory)) == 1


def update_pair(before: str | Path, after: str | Path, enabled: bool = True):
    """Populate and reveal the slider for one before/after image pair."""
    if not enabled:
        return hidden_updates()

    try:
        comparison_html = render_html(before, after)
    except Exception:
        traceback.print_exc()
        return hidden_updates()

    return gr.update(value=comparison_html), gr.update(visible=True, open=False)


def update_from_directories(
    before_directory: str | Path,
    after_directory: str | Path,
    enabled: bool = True,
):
    """Show a comparison only when each directory contains exactly one file."""
    if not enabled:
        return hidden_updates()

    before_images = image_files(before_directory)
    after_images = image_files(after_directory)

    if len(before_images) != 1 or len(after_images) != 1:
        return hidden_updates()

    return update_pair(before_images[0], after_images[0])


def temp_directory_callback(before_subdirectory: str, after_subdirectory: str):
    """Build a Gradio callback using Fooocus temp subdirectories and a ZIP flag.

    The callback accepts one boolean input indicating whether the source was
    uploaded as a ZIP. ZIP mode and directories containing multiple files keep
    the comparison hidden.
    """

    def update(zip_mode: bool = False):
        if zip_mode:
            return hidden_updates()

        import modules.config

        root = modules.config.temp_path
        before_directory = os.path.join(root, before_subdirectory)
        after_directory = os.path.join(root, after_subdirectory)

        return update_from_directories(before_directory, after_directory)

    return update
