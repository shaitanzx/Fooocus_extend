"""HTML renderer for the Face Enhancer before/after comparison view."""

from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image as PILImage

ImageValue = Union[PILImage.Image, np.ndarray, str, Path]


def _to_pil_image(image: ImageValue) -> PILImage.Image:
    if isinstance(image, PILImage.Image):
        result = image.copy()
    elif isinstance(image, np.ndarray):
        array = np.asarray(image)
        if array.dtype != np.uint8:
            if (
                np.issubdtype(array.dtype, np.floating)
                and array.size
                and array.min() >= 0
                and array.max() <= 1
            ):
                array = array * 255.0
            array = np.clip(array, 0, 255).astype(np.uint8)
        result = PILImage.fromarray(array)
    else:
        with PILImage.open(image) as opened:
            result = opened.copy()

    if result.mode not in ("RGB", "RGBA"):
        result = result.convert(
            "RGBA" if "transparency" in result.info else "RGB"
        )
    return result


def _to_data_uri(image: ImageValue) -> str:
    buffer = BytesIO()
    pil_image = image if isinstance(image, PILImage.Image) else _to_pil_image(image)
    pil_image.save(buffer, format="PNG", optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def render_html(before: ImageValue, after: ImageValue) -> str:
    """Return self-contained before/after markup for Gradio's HTML block."""
    before_image = _to_pil_image(before)
    after_image = _to_pil_image(after)
    before_src = _to_data_uri(before_image)
    after_src = _to_data_uri(after_image)
    image_width, image_height = before_image.size

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
