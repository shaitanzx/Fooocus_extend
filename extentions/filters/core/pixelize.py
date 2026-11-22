'''
File:   pixelize.py
Brief:  Pixelization engine — reduce resolution via cell pooling and upscale.
Author: Mistress-Lukutar
Date:   2026-11-14
Version: v0.1.0
'''

from __future__ import annotations

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)

#: Sampling methods for pixel cell pooling.
SAMPLE_METHODS: tuple[str, ...] = ("center", "mean", "median", "mode")
METHOD_CENTER: str = "center"
METHOD_MEAN: str = "mean"
METHOD_MEDIAN: str = "median"
METHOD_MODE: str = "mode"

#: Per-channel bin levels used by mode pooling to merge near-duplicates.
MODE_BIN_LEVELS: int = 16


class Pixelizer:
    """Pixelize an image by downsampling cells and upscaling with nearest neighbour."""

    def __init__(
        self,
        pixel_size: int,
        method: str = METHOD_MODE,
        pre_blur: float = 0.0,
    ) -> None:
        """Initialise the pixelizer.

        Args:
            pixel_size: Cell dimension in pixels (>= 1).
            method: Pooling method from SAMPLE_METHODS.
            pre_blur: Gaussian blur sigma applied before downsampling (>= 0).

        Raises:
            ValueError: On invalid parameters.
        """
        if pixel_size < 1:
            raise ValueError("pixel_size must be >= 1")
        if method not in SAMPLE_METHODS:
            raise ValueError(f"method must be one of {SAMPLE_METHODS}")
        if pre_blur < 0:
            raise ValueError("pre_blur must be >= 0")

        self.pixel_size = pixel_size
        self.method = method
        self.pre_blur = pre_blur

    def process(self, frame: np.ndarray) -> np.ndarray:
        """Process a single frame through the pixelizer.

        Args:
            frame: HWC float32 [0, 255] RGB image.

        Returns:
            Processed frame with same shape/dtype/range as input.
        """
        if frame.dtype != np.float32:
            frame = frame.astype(np.float32)
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("frame must be HWC with 3 channels")

        h, w, _ = frame.shape
        s = self.pixel_size

        # pixel_size == 1 and pre_blur <= 0 → identity
        if s == 1 and self.pre_blur <= 0:
            return frame.copy()

        # No copy needed unless blurring: every pooling path below only
        # reads `work` (slicing / reductions all produce new arrays).
        work = frame

        # Pre-blur
        if self.pre_blur > 0:
            sigma = self.pre_blur
            ksize = max(3, 2 * int(np.ceil(3 * sigma)) + 1)
            work = cv2.GaussianBlur(work, (ksize, ksize), sigma)

        # Pad up to a multiple of pixel_size with edge replication
        pad_h = (s - h % s) % s
        pad_w = (s - w % s) % s
        if pad_h > 0 or pad_w > 0:
            work = np.pad(work, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")

        ph, pw, _ = work.shape
        cells_h = ph // s
        cells_w = pw // s

        # Downsample based on method
        if self.method == METHOD_CENTER:
            reduced = work[s // 2::s, s // 2::s]
        elif self.method == METHOD_MEAN:
            cells = work.reshape(cells_h, s, cells_w, s, 3)
            reduced = cells.mean(axis=(1, 3))
        elif self.method == METHOD_MEDIAN:
            cells = work.reshape(cells_h, s, cells_w, s, 3)
            reduced = np.median(cells, axis=(1, 3))
        else:  # METHOD_MODE
            reduced = self._mode_pool(work, cells_h, cells_w, s)

        # Upscale to original (w, h) with nearest-neighbour interpolation
        result = cv2.resize(reduced, (w, h), interpolation=cv2.INTER_NEAREST)
        result = np.clip(result.astype(np.float32), 0, 255)
        return result

    def _mode_pool(
        self,
        padded: np.ndarray,
        cells_h: int,
        cells_w: int,
        s: int,
    ) -> np.ndarray:
        """Vectorised mode pooling for pixel cells.

        For each cell the dominant colour is found by binning channel values
        into 16 levels per channel (value >> 4), forming a combined code, and
        picking the most frequent code group.  The winning centroid is the
        mean of the actual pixel values in that group.

        Args:
            padded: Frame already padded to a multiple of *s* in both dims.
            cells_h: Number of cells vertically.
            cells_w: Number of cells horizontally.
            s: Pixel size (cell dimension).

        Returns:
            Reduced grid of shape ``(cells_h, cells_w, 3)`` with dominant colours.
        """
        n_cells = cells_h * cells_w
        n_cols = s * s

        # Reshape to cell-pixel layout; swap axes so reshape groups pixels per cell
        small_grid = padded.reshape(cells_h, s, cells_w, s, 3)
        pixels = np.swapaxes(small_grid, 1, 2).reshape(n_cells, n_cols, 3)

        # Per-channel bin values (16 levels: value >> 4), combined code.
        bb = np.floor(pixels[:, :, 0]).astype(np.int32) >> 4
        gb = np.floor(pixels[:, :, 1]).astype(np.int32) >> 4
        rb = np.floor(pixels[:, :, 2]).astype(np.int32) >> 4
        codes = (rb << 8) | (gb << 4) | bb  # (n_cells, n_cols) int32, max 4095

        # Sort each cell's codes ascending (row-wise sort — no global
        # lexsort over all pixels): equal codes become contiguous spans.
        codes_sorted = np.sort(codes, axis=1)
        change = np.empty((n_cells, n_cols), dtype=bool)
        change[:, 0] = True
        change[:, 1:] = codes_sorted[:, 1:] != codes_sorted[:, :-1]

        # Run length at every element: end - start, where start is the last
        # run boundary at or before j (running max) and end the next
        # boundary after j (running min over the reversed row).
        idx = np.arange(n_cols, dtype=np.int32)
        start = np.maximum.accumulate(np.where(change, idx, -1), axis=1)
        cand = np.where(change, idx, n_cols)
        next_boundary = np.minimum.accumulate(cand[:, ::-1], axis=1)[:, ::-1]
        run_len = np.empty_like(next_boundary)
        run_len[:, :-1] = next_boundary[:, 1:]
        run_len[:, -1] = n_cols
        run_len -= start

        # Winner per cell: max count, tie → lowest code. codes_sorted is
        # ascending, so the first max-count run carries the lowest code —
        # argmax returns exactly that position.
        max_count = run_len.max(axis=1)
        winner_pos = np.argmax(run_len == max_count[:, None], axis=1)
        rows = np.arange(n_cells)
        winner_code = codes_sorted[rows, winner_pos]

        # Winning centroid = mean of the actual pixels carrying the winning
        # code in that cell. Mask-multiply keeps float32 values exact; the
        # reduction accumulates in float64.
        mask = codes == winner_code[:, None]
        sums = (pixels * mask[:, :, np.newaxis]).sum(axis=1, dtype=np.float64)
        centroids = sums / max_count[:, None]

        return centroids.reshape(cells_h, cells_w, 3)
