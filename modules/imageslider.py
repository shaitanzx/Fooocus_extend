"""HTML renderer for the Face Enhancer before/after comparison view."""

from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import quote

from PIL import Image as PILImage


def _image_url_and_size(image_path: str | Path, root_path: str = "") -> tuple[str, int, int]:
    """Build a Gradio file URL and read image dimensions without encoding pixels."""
    absolute_path = Path(os.path.abspath(os.fspath(image_path)))

    with PILImage.open(absolute_path) as image:
        width, height = image.size

    encoded_path = quote(absolute_path.as_posix(), safe="/:")
    root_path = root_path.rstrip("/")
    return f"{root_path}/file={encoded_path}", width, height


def render_html(
    before: str | Path,
    after: str | Path,
    root_path: str = "",
) -> str:
    """Return before/after HTML that loads both images through Gradio's file route."""
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
