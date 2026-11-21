'''
File:   analog.py
Brief:  VHS degradation and LCD dot-matrix overlay engines.
Author: Mistress-Lukutar
Date:   2026-11-14
Version: v0.1.0

Frames are ``HWC float32 [0, 255] RGB``; output has the same shape/dtype.
'''

from __future__ import annotations

import logging

import cv2
import numpy as np

#: Rec. 601 luma weights from CRT engine (used by tests for verification).
from .crt import LUMA_WEIGHTS  # noqa: F401

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: VHS noise color modes.
NOISE_MODES: tuple[str, ...] = ("mono", "color")
NOISE_MONO: str = "mono"
NOISE_COLOR: str = "color"

#: LCD dot shapes.
DOT_SHAPES: tuple[str, ...] = ("square", "round")
SHAPE_SQUARE: str = "square"
SHAPE_ROUND: str = "round"


# ---------------------------------------------------------------------------
# VHS degradation
# ---------------------------------------------------------------------------


def vhs(
    frame: np.ndarray,
    noise: float = 0.15,
    chroma_bleed: float = 0.5,
    jitter: float = 0.3,
    band: float = 0.3,
    seed: int = 0,
    noise_mode: str = NOISE_COLOR,
) -> np.ndarray:
    """Apply a VHS-degradation pipeline to *frame*.

    Stages (each skipped when its strength is ``0``): chroma bleed,
    horizontal jitter, additive noise, and rolling-band brightening.

    * **Chroma bleed** converts to YUV, box-blurs the U/V planes
      horizontally, blends back to RGB — luma is preserved.
    * **Jitter** shifts each row by a random integer amount using
      edge-replication (no wrap-around).
    * **Noise** adds Gaussian noise (mono = shared plane across channels,
      color = independent per channel).
    * **Band** applies a vertical gaussian profile that only brightens.

    Args:
        frame: HWC float32 [0, 255] RGB image.
        noise: Noise amplitude in ``[0, 1]`` (sigma = noise * 40).
        chroma_bleed: U/V blur amount in ``[0, 1]``.
        jitter: Horizontal shift amplitude in ``[0, 1]`` (max amp = jitter * 6 px).
        band: Rolling-band strength in ``[0, 1]``.
        seed: Random seed (``0`` = fresh randomness each run).
        noise_mode: Noise colouring — one of :data:`NOISE_MODES`.

    Returns:
        Processed frame with same shape/dtype as input.

    Raises:
        ValueError: On invalid parameters.
    """
    # --- validation ---
    if not (0 <= noise <= 1):
        raise ValueError("noise must be in [0, 1]")
    if not (0 <= chroma_bleed <= 1):
        raise ValueError("chroma_bleed must be in [0, 1]")
    if not (0 <= jitter <= 1):
        raise ValueError("jitter must be in [0, 1]")
    if not (0 <= band <= 1):
        raise ValueError("band must be in [0, 1]")
    if seed < 0:
        raise ValueError("seed must be >= 0")
    if noise_mode not in NOISE_MODES:
        raise ValueError(f"noise_mode must be one of {NOISE_MODES}")

    h, w, _ = frame.shape
    result = frame.astype(np.float32)  # copy — stages below may be skipped

    # --- rng ---
    rng = np.random.default_rng(seed if seed > 0 else None)

    # --- stage a: chroma bleed ---
    if chroma_bleed > 0:
        k = int(round(chroma_bleed * 15))
        if k < 3:
            k = 3
        elif k % 2 == 0:
            k += 1  # force odd

        yuv = cv2.cvtColor(frame.astype(np.float32), cv2.COLOR_RGB2YUV)
        u = yuv[:, :, 1]
        v = yuv[:, :, 2]

        blurred_u = cv2.blur(u, (k, 1))
        blurred_v = cv2.blur(v, (k, 1))

        b = chroma_bleed
        yuv[:, :, 1] = u * (1 - b) + blurred_u * b
        yuv[:, :, 2] = v * (1 - b) + blurred_v * b
        result = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB).astype(np.float32, copy=False)

    # --- stage b: jitter ---
    if jitter > 0:
        amp = int(round(jitter * 6))
        if amp > 0:
            shifts = rng.integers(-amp, amp + 1, size=h)
            idx = np.clip(np.arange(w)[np.newaxis, :] + shifts[:, np.newaxis], 0, w - 1)
            idx_expanded = np.broadcast_to(idx[:, :, np.newaxis], (h, w, 3))
            result = np.take_along_axis(result, idx_expanded, axis=1)

    # --- stage c: noise (generated in float64 — Generator.normal has no
    # float32 path — and cast once on the way in) ---
    if noise > 0:
        sigma = noise * 40.0
        if noise_mode == NOISE_MONO:
            noise_plane = rng.normal(0, sigma, (h, w)).astype(np.float32)
            result += noise_plane[:, :, np.newaxis]
        else:  # NOISE_COLOR
            noise_planes = rng.normal(0, sigma, (h, w, 3)).astype(np.float32)
            result += noise_planes

    # --- stage d: band ---
    if band > 0:
        pos = rng.uniform(0, h)
        bh = max(4.0, h / 6.0)
        row_indices = np.arange(h, dtype=np.float64)
        g = np.exp(-((row_indices - pos) / max(bh, 1)) ** 2)
        factor = 1.0 + band * 0.3 * g
        result *= factor.astype(np.float32)[:, np.newaxis, np.newaxis]

    return np.clip(result, 0, 255).astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# LCD dot matrix
# ---------------------------------------------------------------------------


def lcd_dot_matrix(
    frame: np.ndarray,
    cell_size: int = 8,
    gap_ratio: float = 0.15,
    grid_dark: float = 0.6,
    shape: str = SHAPE_SQUARE,
) -> np.ndarray:
    """Apply a dot-matrix LCD overlay to *frame*.

    Each ``cell_size``-pixel block is treated as a "dot".  Pixels inside the
    dot keep full brightness (factor 1); pixels in the gap are darkened by
    ``(1 - grid_dark)``.

    The pattern is computed from local fractional coordinates within each cell:

    * **square** — rectangular active region defined by *gap_ratio*.
    * **round** — circular active region via normalised radius <= ``1 - gap_ratio``.

    Args:
        frame: HWC float32 [0, 255] RGB image.
        cell_size: Side length of each dot cell in pixels (>= 1).
        gap_ratio: Fractional gap around the active region in ``[0, 0.5]``.
        grid_dark: Darkening factor for gap pixels in ``[0, 1]``.
            ``0`` → no darkening; ``1`` → fully dark gaps.
        shape: Dot shape — one of :data:`DOT_SHAPES`.

    Returns:
        Processed frame with same shape/dtype as input.

    Raises:
        ValueError: On invalid parameters.
    """
    if cell_size < 1:
        raise ValueError("cell_size must be >= 1")
    if not (0 <= gap_ratio <= 0.5):
        raise ValueError("gap_ratio must be in [0, 0.5]")
    if not (0 <= grid_dark <= 1):
        raise ValueError("grid_dark must be in [0, 1]")
    if shape not in DOT_SHAPES:
        raise ValueError(f"shape must be one of {DOT_SHAPES}")

    h, w, _ = frame.shape
    cs = cell_size

    # The factor map is periodic in both axes with period cell_size:
    # pixel (r, c) has fractional coords ((r % cs) + 0.5) / cs and
    # ((c % cs) + 0.5) / cs. Build one (cs, cs) tile and repeat it —
    # exact and independent of frame size.
    g = (np.arange(cs, dtype=np.float64) + 0.5) / cs  # (cs,)

    if shape == SHAPE_SQUARE:
        half_gap = gap_ratio / 2.0
        col_inside = (g >= half_gap) & (g <= 1.0 - half_gap)
        row_inside = (g >= half_gap) & (g <= 1.0 - half_gap)
        inside_tile = col_inside[np.newaxis, :] & row_inside[:, np.newaxis]
    else:  # SHAPE_ROUND
        nx = 2 * (g[np.newaxis, :] - 0.5)  # (1, cs)
        ny = 2 * (g[:, np.newaxis] - 0.5)  # (cs, 1)
        threshold = 1.0 - gap_ratio
        inside_tile = (nx * nx + ny * ny) <= threshold ** 2

    dark = 1.0 - grid_dark
    tile = np.where(inside_tile, 1.0, dark).astype(np.float32)  # (cs, cs)
    reps = ((h + cs - 1) // cs, (w + cs - 1) // cs)
    factor = np.tile(tile, reps)[:h, :w]  # (H, W) float32

    result = frame * factor[:, :, np.newaxis]
    return np.clip(result, 0, 255).astype(np.float32, copy=False)
