'''
File:   dither.py
Brief:  Dithering engine — bit-depth reduction with ordered or error-diffusion methods.
Author: Mistress-Lukutar
Date:   2026-11-14
Version: v0.1.0

Ordered methods (bayer2/4/8) are fully vectorised via ``np.tile``.
Error-diffusion methods (floyd-steinberg, atkinson) use a sequential scanline
pass per channel plane. The scan is implemented over plain Python lists with
a sliding window of pending future rows — the same math as a naive numpy
scalar loop, but several times faster per megapixel (naive numpy scalar
arithmetic was slow enough to freeze the ComfyUI server on large frames).
'''

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

#: Dithering methods.
DITHER_METHODS: tuple[str, ...] = (
    "none", "bayer2", "bayer4", "bayer8",
    "floyd-steinberg", "atkinson",
)
DITHER_NONE: str = "none"
DITHER_BAYER2: str = "bayer2"
DITHER_BAYER4: str = "bayer4"
DITHER_BAYER8: str = "bayer8"
DITHER_FLOYDSTEINBERG: str = "floyd-steinberg"
DITHER_ATKINSON: str = "atkinson"

#: Methods applied as a tileable ordered threshold matrix.
ORDERED_METHODS: tuple[str, ...] = (DITHER_BAYER2, DITHER_BAYER4, DITHER_BAYER8)
#: Methods applied as sequential error diffusion.
DIFFUSION_METHODS: tuple[str, ...] = (DITHER_FLOYDSTEINBERG, DITHER_ATKINSON)

#: Error-diffusion kernels: ``(dx, dy, weight)`` offsets and weights.
DIFFUSION_KERNELS: dict[str, tuple[tuple[int, int, float], ...]] = {
    DITHER_FLOYDSTEINBERG: (
        (1, 0, 7 / 16),
        (-1, 1, 3 / 16),
        (0, 1, 5 / 16),
        (1, 1, 1 / 16),
    ),
    DITHER_ATKINSON: (
        (1, 0, 1 / 8),
        (2, 0, 1 / 8),
        (-1, 1, 1 / 8),
        (0, 1, 1 / 8),
        (1, 1, 1 / 8),
        (0, 2, 1 / 8),
    ),
}


def bayer_matrix(size: int) -> np.ndarray:
    """Construct a Bayer threshold matrix of the given size.

    Uses recursive construction from ``M2 = [[0, 2], [3, 1]]``:
    ``M(2n) = [[4M, 4M+2], [4M+3, 4M+1]]``.

    Args:
        size: Matrix dimension — must be 2, 4, or 8.

    Returns:
        uint8 array of shape ``(size, size)`` with values ``0 .. (size*size - 1)``,
        each appearing exactly once.

    Raises:
        ValueError: If *size* is not 2, 4, or 8.
    """
    if size not in (2, 4, 8):
        raise ValueError("bayer_matrix: size must be 2, 4, or 8")

    M = np.array([[0, 2], [3, 1]], dtype=np.uint8)
    if size == 2:
        return M

    n_steps = int(np.log2(size)) - 1
    for _ in range(n_steps):
        M = np.block([
            [4 * M, 4 * M + 2],
            [4 * M + 3, 4 * M + 1],
        ])
    return M.astype(np.uint8)


def dither_frame(
    frame: np.ndarray,
    levels: int,
    method: str,
    strength: float = 1.0,
    serpentine: bool = True,
) -> np.ndarray:
    """Dither a frame to the given number of quantisation levels.

    Args:
        frame: HWC float32 [0, 255] RGB image.
        levels: Number of quantisation levels (>= 2).
        method: One of :data:`DITHER_METHODS`.
        strength: Error diffusion / ordered-dither strength in ``[0, 1]``.
        serpentine: Alternate scanline direction for error-diffusion methods.

    Returns:
        float32 HWC array with values clipped to ``[0, 255]``.

    Raises:
        ValueError: On invalid parameters.
    """
    if levels < 2:
        raise ValueError("levels must be >= 2")
    if method not in DITHER_METHODS:
        raise ValueError(f"method must be one of {DITHER_METHODS}")
    if strength < 0 or strength > 1:
        raise ValueError("strength must be in [0, 1]")

    step = 255.0 / (levels - 1)

    def _quantize(x: np.ndarray) -> np.ndarray:
        """Vectorised quantisation to *levels* levels."""
        return np.clip(np.round(x / step) * step, 0, 255).astype(np.float32)

    if method == DITHER_NONE:
        return _quantize(frame)

    if method in ORDERED_METHODS:
        return _dither_ordered(frame, levels, method, strength)

    # Error diffusion
    return _dither_diffusion(frame, levels, method, strength, serpentine)


def _dither_ordered(
    frame: np.ndarray,
    levels: int,
    method: str,
    strength: float,
) -> np.ndarray:
    """Vectorised ordered dithering (Bayer threshold matrix)."""
    step = 255.0 / (levels - 1)

    size_map = {DITHER_BAYER2: 2, DITHER_BAYER4: 4, DITHER_BAYER8: 8}
    size = size_map[method]

    # float32 throughout — quantisation rounds to `step` granularity
    # (>= 1.0), far coarser than the float32 mantissa at these magnitudes.
    m = bayer_matrix(size).astype(np.float32)
    m_norm = (m + 0.5) / (size * size)
    bias = (m_norm - 0.5) * np.float32(step * strength)

    h, w, _ = frame.shape
    bias_3d = bias[:, :, np.newaxis]  # (size, size, 1) for correct tile broadcasting
    tiled_bias = np.tile(bias_3d, (h // size + 1, w // size + 1, 1))[:h, :w, :]

    quantized = np.clip(frame + tiled_bias, 0, 255)
    return _quantize_ordered(quantized, step)


def _quantize_ordered(x: np.ndarray, step: float) -> np.ndarray:
    """Quantise a float64 array to *levels* levels."""
    return np.clip(np.round(x / step) * step, 0, 255).astype(np.float32)


def _dither_diffusion(
    frame: np.ndarray,
    levels: int,
    method: str,
    strength: float,
    serpentine: bool,
) -> np.ndarray:
    """Sequential error-diffusion dithering (Floyd-Steinberg or Atkinson).

    Scans each channel plane row by row, propagating quantisation error to
    unprocessed neighbours according to the kernel.  For *serpentine=True*
    odd rows are scanned right-to-left and kernel x-offsets are mirrored.

    Runs over plain Python lists (``plane.tolist()``) with a sliding window
    of the current row plus the ``max(dr)`` pending future rows — error
    diffusion is inherently sequential, and numpy scalar indexing is an
    order of magnitude slower per pixel than float locals.

    A fully processed row is written back to the numpy plane immediately,
    so only ``max(dr) + 1`` rows live as Python lists at any time.
    """
    step = 255.0 / (levels - 1)
    kernel = DIFFUSION_KERNELS[method]

    h, w, _ = frame.shape
    result = frame.astype(np.float64).copy()

    # Serpentine scan mirrors the x-offsets once — identical for all odd rows.
    mirrored_kernel: tuple[tuple[int, int, float], ...] = tuple(
        (-dc, dr, wt) for dc, dr, wt in kernel
    )
    #: Largest row offset in the kernel → rows held in the pending window.
    max_dr = max(dr for _, dr, _ in kernel)

    for ch in range(3):
        plane = result[:, :, ch]

        # Sliding window: window[0] is the current row, window[dr] the row
        # `dr` steps ahead (holding accumulated pending error). Tail entries
        # are None once past the bottom of the frame.
        first_rows = min(max_dr + 1, h)
        window: list[list[float] | None] = [
            plane[r].tolist() for r in range(first_rows)
        ]
        window.extend([None] * (max_dr + 1 - first_rows))

        for row in range(h):
            if serpentine and row % 2 == 1:
                scan_kernel = mirrored_kernel
                cols: range = range(w - 1, -1, -1)
            else:
                scan_kernel = kernel
                cols = range(w)

            cur = window[0]
            # window[0] is a real row for every processed row (the window is
            # refilled ahead of the scan); the None tail only sits at dr > 0.
            assert cur is not None
            for col in cols:
                old_val = cur[col]
                # Clamp to [0, 255]: accumulated error can push the running
                # value past the outer levels; without the clamp the
                # quantized residue would leak oversized errors downstream.
                new_val = round(old_val / step) * step
                if new_val < 0.0:
                    new_val = 0.0
                elif new_val > 255.0:
                    new_val = 255.0
                cur[col] = new_val
                err = (old_val - new_val) * strength

                if err == 0.0:
                    continue

                for dc, dr, wt in scan_kernel:
                    nc = col + dc
                    if 0 <= nc < w:
                        target = window[dr] if dr <= max_dr else None
                        if target is not None:
                            target[nc] += err * wt

            # Write the completed row back; slide the window down one row.
            plane[row] = cur
            window.pop(0)
            next_row = row + max_dr + 1
            window.append(plane[next_row].tolist() if next_row < h else None)

    return np.clip(result, 0, 255).astype(np.float32)
