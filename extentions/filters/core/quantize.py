'''
File:   quantize.py
Brief:  Palette-based quantization with ordered and error-diffusion dithering.
Author: Mistress-Lukutar
Date:   2026-11-14
Version: v0.1.0

Frames are ``HWC float32 [0, 255] RGB``; output has the same shape/dtype.
Error-diffusion methods use a sequential scanline pass — CPU-bound in CPython.
'''

from __future__ import annotations

import logging

import cv2
import numpy as np

from .dither import DIFFUSION_KERNELS, DITHER_METHODS, bayer_matrix

logger = logging.getLogger(__name__)

#: Palette-matching distance space (CIELAB via OpenCV).
MATCH_SPACE: str = "lab"

#: LUT bin size per RGB channel — 5 significant bits → 32 bins/channel.
LUT_BIN_SHIFT: int = 3

#: Sample-grid resolution used to self-calibrate ordered-dither amplitude.
AMPLITUDE_GRID_LEVELS: int = 16


# ---------------------------------------------------------------------------
# CIELAB helper
# ---------------------------------------------------------------------------


def _lab_values(rgb: np.ndarray) -> np.ndarray:
    """Convert ``(N, 3)`` RGB uint8/float to ``(N, 3)`` LAB float32 via OpenCV.

    Args:
        rgb: Array of shape ``(N, 3)`` with RGB values (uint8 or float).

    Returns:
        ``(N, 3)`` float32 LAB array.
    """
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    # cv2.cvtColor expects (N, 1, 3) for batch conversion
    return cv2.cvtColor(
        rgb.reshape(-1, 1, 3), cv2.COLOR_RGB2LAB
    ).reshape(-1, 3).astype(np.float32)


# ---------------------------------------------------------------------------
# Palette LUT builder
# ---------------------------------------------------------------------------


def make_palette_lut(palette: np.ndarray) -> np.ndarray:
    """Build a ``(32, 32, 32, 3)`` uint8 lookup table for palette quantisation.

    For every 5-bit RGB bin the nearest palette colour is found by CIELAB
    distance.  The query colour per bin is its centre value
    ``(idx * 8 + 4, …)``, clamped to 255 (maximum index 31 → centre 252).
    This introduces a small systematic bias of up to 4 levels per channel
    towards the palette centre.

    Args:
        palette: ``(K, 3)`` uint8 RGB array with ``K >= 1``.

    Returns:
        ``(32, 32, 32, 3)`` uint8 LUT — every entry is one of the palette colours.

    Raises:
        ValueError: If *palette* does not have shape ``(K, 3)`` with ``K >= 1``.
    """
    if palette.ndim != 2 or palette.shape[1] != 3:
        raise ValueError(
            f"palette must have shape (K, 3), got {palette.shape}"
        )
    if palette.shape[0] < 1:
        raise ValueError("palette must have K >= 1 rows")

    # Precompute LAB for the entire palette once
    palette_lab = _lab_values(palette)  # (K, 3) float32

    # Build the LUT in chunks of 4096 bins to limit memory.
    lut = np.empty((32, 32, 32, 3), dtype=np.uint8)
    chunk_size = 4096
    total_bins = 32 * 32 * 32

    for start in range(0, total_bins, chunk_size):
        end = min(start + chunk_size, total_bins)
        indices = np.arange(start, end)
        # Decode linear bin index into (i_r, i_g, i_b)
        ri = indices // (32 * 32)
        remainder = indices % (32 * 32)
        gi = remainder // 32
        bi = remainder % 32

        # Query colour = bin centre, clamped to [0, 255]
        qr = np.clip(ri.astype(np.int32) * 8 + 4, 0, 255).astype(np.uint8)
        qg = np.clip(gi.astype(np.int32) * 8 + 4, 0, 255).astype(np.uint8)
        qb = np.clip(bi.astype(np.int32) * 8 + 4, 0, 255).astype(np.uint8)

        query = np.stack([qr, qg, qb], axis=1)  # (N, 3) uint8
        query_lab = _lab_values(query)  # (N, 3) float32

        # Squared CIELAB distance: (N, K) — vectorised
        diff = query_lab[:, np.newaxis, :] - palette_lab[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=2)  # (N, K)

        # argmin over K → nearest palette index per query
        best_idx = np.argmin(dist_sq, axis=1)  # (N,)

        # Write the whole chunk into the LUT in one vectorized assignment
        lut.reshape(-1, 3)[start:end] = palette[best_idx]

    return lut


# ---------------------------------------------------------------------------
# PaletteQuantizer
# ---------------------------------------------------------------------------


class PaletteQuantizer:
    """Quantise an ``HWC float32 [0,255] RGB`` frame through a palette.

    Supports no-dither (LUT lookup), ordered Bayer dithering, and
    error-diffusion (Floyd-Steinberg / Atkinson).

    Attributes:
        amplitude: Calibrated amplitude for ordered methods; ``None`` until
            computed on first access after an ordered-method init.
    """

    def __init__(
        self,
        palette: np.ndarray,
        dither: str = "none",
        strength: float = 1.0,
        serpentine: bool = True,
    ) -> None:
        """Initialise the quantizer.

        Args:
            palette: ``(K, 3)`` uint8 RGB array with ``K >= 1``.
            dither: One of :data:`DITHER_METHODS`.
            strength: Strength in ``[0, 1]`` (for ordered and diffusion).
            serpentine: Alternate scanline direction for diffusion methods.

        Raises:
            ValueError: On invalid parameters.
        """
        if palette.ndim != 2 or palette.shape[1] != 3:
            raise ValueError(
                f"palette must have shape (K, 3), got {palette.shape}"
            )
        if palette.shape[0] < 1:
            raise ValueError("palette must have K >= 1 rows")
        if dither not in DITHER_METHODS:
            raise ValueError(f"dither must be one of {DITHER_METHODS}")
        if strength < 0 or strength > 1:
            raise ValueError("strength must be in [0, 1]")

        self._palette = palette
        self.dither = dither
        self.strength = strength
        self.serpentine = serpentine
        self._lut = make_palette_lut(palette)
        self._amplitude: float | None = None

    @property
    def amplitude(self) -> float | None:
        """Ordered-dither amplitude, computed lazily on first access."""
        if self._amplitude is None and self.dither != "none":
            # Self-calibrate: mean CIELAB distance of a uniform sample grid.
            levels = AMPLITUDE_GRID_LEVELS
            vals = np.linspace(0, 255, levels).astype(np.uint8)
            grid = np.stack(np.meshgrid(vals, vals, vals, indexing="ij"), axis=-1)
            # grid is (L, L, L, 3) → reshape to (N, 3)
            samples = grid.reshape(-1, 3)
            sample_lab = _lab_values(samples)
            palette_lab = _lab_values(self._palette)

            # Mean squared distance per sample
            diff = sample_lab[:, np.newaxis, :] - palette_lab[np.newaxis, :, :]
            dist_sq = np.sum(diff * diff, axis=2)
            distances = np.sqrt(np.min(dist_sq, axis=1))
            self._amplitude = float(np.mean(distances))
        return self._amplitude

    def process(self, frame: np.ndarray) -> np.ndarray:
        """Process a single frame through the quantizer.

        Args:
            frame: ``HWC float32 [0, 255] RGB`` image.

        Returns:
            ``float32`` array with same shape, values clipped to ``[0, 255]``.
        """
        if frame.dtype != np.float32:
            frame = frame.astype(np.float32)
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("frame must be HWC with 3 channels")

        if self.dither == "none":
            return self._quantize_none(frame)

        if self.dither in ("bayer2", "bayer4", "bayer8"):
            return self._quantize_ordered(frame)

        # Error diffusion
        return self._quantize_diffusion(frame)

    def _quantize_none(self, frame: np.ndarray) -> np.ndarray:
        """LUT-based palette quantisation (no dither)."""
        idx = frame.astype(np.uint8) >> LUT_BIN_SHIFT
        return self._lut[idx[:, :, 0], idx[:, :, 1], idx[:, :, 2]].astype(
            np.float32
        )

    def _quantize_ordered(self, frame: np.ndarray) -> np.ndarray:
        """Ordered (Bayer) dithering with LUT mapping."""
        size_map = {"bayer2": 2, "bayer4": 4, "bayer8": 8}
        size = size_map[self.dither]
        amp = self.amplitude or 1.0

        # float32 throughout — the LUT index below truncates to uint8, so
        # the 24-bit mantissa is far more precision than the path consumes.
        m = bayer_matrix(size).astype(np.float32)
        m_norm = (m + 0.5) / (size * size)
        bias = (m_norm - 0.5) * np.float32(amp * self.strength)

        h, w, _ = frame.shape
        bias_3d = bias[:, :, np.newaxis]
        tiled_bias = np.tile(
            bias_3d, (h // size + 1, w // size + 1, 1)
        )[:h, :w, :]

        biased = np.clip(frame + tiled_bias, 0, 255)
        idx = biased.astype(np.uint8) >> LUT_BIN_SHIFT
        return self._lut[idx[:, :, 0], idx[:, :, 1], idx[:, :, 2]].astype(
            np.float32
        )

    def _quantize_diffusion(self, frame: np.ndarray) -> np.ndarray:
        """Sequential error-diffusion dithering with palette LUT.

        Scans each row of the frame pixel-by-pixel. Each pixel is matched as
        one RGB colour through the LUT (not per-channel).  Quantisation error
        is propagated to unprocessed neighbours according to
        :data:`DIFFUSION_KERNELS`.

        For *serpentine=True* odd rows are scanned right-to-left and kernel
        x-offsets are mirrored.

        The running value is clamped to ``[0, 255]`` before matching and the
        diffused error is computed from the clamped value.  Unlike the
        uniform-level diffusion in :mod:`core.dither` (whose quantiser spans
        the full channel range and can always absorb error), a palette may
        have no colour that can absorb error on some channel — e.g. all
        ``dmg_4`` greens are blue≈15 — so an unclamped residual would grow
        without bound, saturate the LUT bin and collapse the output into
        giant flat blobs.

        Runs over plain Python lists with a sliding window of pending future
        rows and a flattened LUT (``list[int]`` rows) — error diffusion is
        inherently sequential and numpy scalar indexing is an order of
        magnitude slower per pixel (slow enough to freeze the ComfyUI
        server on large frames, e.g. with Atkinson).
        """
        h, w, _ = frame.shape
        result = frame.astype(np.float64).copy()
        kernel = DIFFUSION_KERNELS[self.dither]
        strength = self.strength
        serpentine = self.serpentine

        # Flattened LUT: index = (r_bin << 10) | (g_bin << 5) | b_bin.
        lut = self._lut.reshape(-1, 3).tolist()

        # Serpentine scan mirrors the x-offsets once — identical for all
        # odd rows.
        mirrored_kernel: tuple[tuple[int, int, float], ...] = tuple(
            (-dc, dr, wt) for dc, dr, wt in kernel
        )
        #: Largest row offset in the kernel → rows held in the pending window.
        max_dr = max(dr for _, dr, _ in kernel)

        # Sliding window: window[0] is the current row (list of ``[r, g, b]``
        # float lists), window[dr] the row ``dr`` steps ahead (holding
        # accumulated pending error). Tail entries are None past the bottom.
        first_rows = min(max_dr + 1, h)
        window: list[list[list[float]] | None] = [
            result[r].tolist() for r in range(first_rows)
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
                pixel = cur[col]
                r, g, b = pixel
                # Clamp the running value to [0, 255] BEFORE matching and
                # compute the diffused error from the clamped value.  A
                # palette may be unable to absorb error on a channel (all
                # dmg_4 greens are blue≈15), so the accumulated residual
                # would otherwise grow without bound — the LUT bin then
                # saturates, every match collapses onto the same palette
                # colour and the output turns into giant flat blobs.
                # Matching and error must always describe the same value:
                # clamp first, then match, then diffuse (clamp - matched).
                r = 0.0 if r < 0.0 else (255.0 if r > 255.0 else r)
                g = 0.0 if g < 0.0 else (255.0 if g > 255.0 else g)
                b = 0.0 if b < 0.0 else (255.0 if b > 255.0 else b)
                kr, kg, kb = int(r), int(g), int(b)
                matched = lut[
                    ((kr >> LUT_BIN_SHIFT) << 10)
                    | ((kg >> LUT_BIN_SHIFT) << 5)
                    | (kb >> LUT_BIN_SHIFT)
                ]
                cur[col] = matched
                er = (r - matched[0]) * strength
                eg = (g - matched[1]) * strength
                eb = (b - matched[2]) * strength

                if er == 0.0 and eg == 0.0 and eb == 0.0:
                    continue

                for dc, dr, wt in scan_kernel:
                    nc = col + dc
                    if 0 <= nc < w:
                        target = window[dr] if dr <= max_dr else None
                        if target is not None:
                            target[nc][0] += er * wt
                            target[nc][1] += eg * wt
                            target[nc][2] += eb * wt

            # Write the completed row back; slide the window down one row.
            result[row] = cur
            window.pop(0)
            next_row = row + max_dr + 1
            window.append(result[next_row].tolist() if next_row < h else None)

        return np.clip(result, 0, 255).astype(np.float32)
