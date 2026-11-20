"""Shared before/after comparison slider for Fooocus extensions."""

from __future__ import annotations

import os
import traceback
from pathlib import Path
from urllib.parse import quote

import gradio as gr
from PIL import Image as PILImage


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def _root_path() -> str:
    """Return Gradio's URL prefix when the app is mounted below a subpath."""
    try:
        import shared

        return getattr(getattr(shared, "gradio_root", None), "root_path", "") or ""
    except Exception:
        return ""


def _image_url_and_size(image_path: str | Path, root_path: str = "") -> tuple[str, int, int]:
    """Build a direct Gradio file URL and read image dimensions without re-encoding."""
    absolute_path = Path(os.path.abspath(os.fspath(image_path)))

    with PILImage.open(absolute_path) as image:
        width, height = image.size

    encoded_path = quote(absolute_path.as_posix(), safe="/:")
    root_path = root_path.rstrip("/")
    return f"{root_path}/file={encoded_path}", width, height


def render_html(
    before: str | Path,
    after: str | Path,
    root_path: str | None = None,
) -> str:
    """Return before/after HTML that references the original files directly."""
    if root_path is None:
        root_path = _root_path()

    before_src, image_width, image_height = _image_url_and_size(before, root_path)
    after_src, _, _ = _image_url_and_size(after, root_path)

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
    """Create the common collapsed accordion and HTML target component."""
    with gr.Accordion(label, open=False, visible=False) as accordion:
        html = gr.HTML(value="", show_label=False)
    return html, accordion


def hidden_updates():
    """Return output updates that clear and hide a comparison view."""
    return gr.update(value=""), gr.update(visible=False, open=False)


def reset():
    """Callback used when a new processing run starts."""
    return hidden_updates()


def image_files(directory: str | Path) -> list[str]:
    """List supported image files directly inside a directory, in stable order."""
    directory = os.fspath(directory)
    if not os.path.isdir(directory):
        return []

    try:
        return sorted(
            os.path.join(directory, name)
            for name in os.listdir(directory)
            if (
                os.path.isfile(os.path.join(directory, name))
                and os.path.splitext(name)[1].lower() in IMAGE_EXTENSIONS
            )
        )
    except OSError:
        return []


def has_single_image(directory: str | Path) -> bool:
    """Return whether a directory contains exactly one supported image."""
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
    """Show a comparison only when both directories contain exactly one image."""
    if not enabled:
        return hidden_updates()

    before_images = image_files(before_directory)
    after_images = image_files(after_directory)
    if len(before_images) != 1 or len(after_images) != 1:
        return hidden_updates()

    return update_pair(before_images[0], after_images[0])


def temp_directory_callback(before_subdirectory: str, after_subdirectory: str):
    """Build a Gradio callback using Fooocus temp subdirectories and a ZIP flag.

    The callback accepts one boolean input: whether the source was uploaded as a ZIP.
    ZIP/batch mode and directories containing multiple images keep the comparison hidden.
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
