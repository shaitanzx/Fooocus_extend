'''
File:   crt.py
Brief:  CRT/analog-video artifact engine — scanlines, phosphor mask, bloom,
        chromatic aberration, barrel curvature, and vignette.
Author: Mistress-Lukutar
Date:   2026-11-14
Version: v0.1.0

Frames are ``HWC float32 [0, 255] RGB``; output has the same shape/dtype
(:func:`curvature` additionally returns a blackened-pixel mask).
'''

from __future__ import annotations

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Scanline brightness profiles.
SCANLINE_PROFILES: tuple[str, ...] = ("hard", "soft")
PROFILE_HARD: str = "hard"
PROFILE_SOFT: str = "soft"

#: Phosphor-mask types (fake CRT / LCD dot structure).
MASK_TYPES: tuple[str, ...] = ("aperture_grille", "shadow_mask", "slot_mask")
MASK_APERTURE_GRILLE: str = "aperture_grille"
MASK_SHADOW_MASK: str = "shadow_mask"
MASK_SLOT_MASK: str = "slot_mask"

#: Bloom blend modes.
BLOOM_BLENDS: tuple[str, ...] = ("screen", "add")
BLEND_SCREEN: str = "screen"
BLEND_ADD: str = "add"

#: Chromatic-aberration displacement modes.
ABERRATION_MODES: tuple[str, ...] = ("radial", "linear")
ABERRATION_RADIAL: str = "radial"
ABERRATION_LINEAR: str = "linear"

#: Brightness compensation for phosphor masks (each channel lights 1/3 of the area).
MASK_BRIGHTNESS_COMPENSATION: float = 3.0

#: Rec. 601 luma weights (also used by the sibling nodes).
LUMA_WEIGHTS: tuple[float, float, float] = (0.299, 0.587, 0.114)


# ---------------------------------------------------------------------------
# Scanlines
# ---------------------------------------------------------------------------


def scanlines(
    frame: np.ndarray,
    pitch: int = 2,
    thickness: float = 0.5,
    strength: float = 0.35,
    offset: int = 0,
    profile: str = PROFILE_HARD,
) -> np.ndarray:
    """Apply a CRT scanline darkening effect to *frame*.

    Every ``pitch`` rows the darkest ``thickness`` fraction is dimmed by
    *strength*.  The ``profile`` controls whether the transition is abrupt
    (hard) or uses a raised-cosine ramp (soft).

    Args:
        frame: HWC float32 [0, 255] RGB image.
        pitch: Number of rows per scanline period (>= 1).
        thickness: Fraction of the period that is darkened in ``[0, 1]``.
        strength: Maximum dimming factor in ``[0, 1]``.
        offset: Row phase shift applied before the modulo operation.
        profile: Brightness profile — one of :data:`SCANLINE_PROFILES`.

    Returns:
        Processed frame with same shape/dtype as input.

    Raises:
        ValueError: On invalid parameters.
    """
    if pitch < 1:
        raise ValueError("pitch must be >= 1")
    if thickness < 0 or thickness > 1:
        raise ValueError("thickness must be in [0, 1]")
    if strength < 0 or strength > 1:
        raise ValueError("strength must be in [0, 1]")
    if profile not in SCANLINE_PROFILES:
        raise ValueError(f"profile must be one of {SCANLINE_PROFILES}")

    h = frame.shape[0]
    row_indices = np.arange(h, dtype=np.float64)

    # Row-phase p = ((row + offset) mod pitch) / pitch  →  p in [0, 1)
    p = (row_indices + offset) % pitch / pitch

    if profile == PROFILE_HARD:
        # Hard: factor = 1 - strength when p < thickness, else 1.
        mask = np.where(p < thickness, 1.0 - strength, 1.0)
    else:  # PROFILE_SOFT
        # Soft: raised-cosine dip inside the dark band.
        # factor = 1 - strength * 0.5 * (1 + cos(pi * p / thickness))
        #          when p < thickness
        # Guard thickness == 0 → no darkening; thickness == 1 → full window.
        mask = np.ones(h, dtype=np.float64)
        dark_mask = p < thickness
        if np.any(dark_mask):
            # When thickness == 1, p < 1 so cos covers the full range.
            # Guard: avoid division by zero when thickness == 0
            # (already handled by dark_mask).
            cos_arg = np.pi * p[dark_mask] / max(thickness, 1e-12)
            soft_factor = 1.0 - strength * 0.5 * (1.0 + np.cos(cos_arg))
            mask[dark_mask] = soft_factor

    # The mask is per-row (H values); multiply in float32 — converting the
    # full frame to float64 triples the memory traffic for no visible gain.
    result = frame * mask.astype(np.float32)[:, np.newaxis, np.newaxis]
    return np.clip(result, 0, 255).astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# Subpixel mask
# ---------------------------------------------------------------------------


def subpixel_mask(
    frame: np.ndarray,
    mask: str = MASK_APERTURE_GRILLE,
    pitch: int = 1,
) -> np.ndarray:
    """Apply a fake CRT phosphor-mask (or LCD dot-grid) modulation to *frame*.

    The pattern has a period of ``3 * pitch`` horizontally.  Each channel only
    transmits light through its own stripe(s), and the result is multiplied by
    :data:`MASK_BRIGHTNESS_COMPENSATION` to restore overall brightness.

    Args:
        frame: HWC float32 [0, 255] RGB image.
        mask: Phosphor-mask type — one of :data:`MASK_TYPES`.
        pitch: Stripe width in pixels (>= 1).

    Returns:
        Processed frame with same shape/dtype as input.

    Raises:
        ValueError: On invalid parameters.
    """
    if mask not in MASK_TYPES:
        raise ValueError(f"mask must be one of {MASK_TYPES}")
    if pitch < 1:
        raise ValueError("pitch must be >= 1")

    h, w, _ = frame.shape
    row_indices = np.arange(h)

    if mask == MASK_APERTURE_GRILLE:
        pattern = _aperture_grille_pattern(w, pitch)
        result = frame * pattern[np.newaxis, :, :]

    elif mask == MASK_SHADOW_MASK:
        even_pattern, odd_pattern = _shadow_mask_patterns(w, pitch)
        result = frame * even_pattern[np.newaxis, :, :]
        odd_rows = row_indices % 2 == 1
        if odd_rows.any():
            result[odd_rows] = frame[odd_rows] * odd_pattern

    else:  # MASK_SLOT_MASK
        pattern = _aperture_grille_pattern(w, pitch)
        result = frame * pattern[np.newaxis, :, :]
        band_idx = row_indices // pitch
        band = np.where(band_idx % 2 == 1, 0.5, 1.0).astype(np.float32)
        result *= band[:, np.newaxis, np.newaxis]

    # Apply pattern and compensate brightness. All mask values are exact
    # in float32 (0, 0.5, 1), so the whole pipeline stays float32.
    result *= MASK_BRIGHTNESS_COMPENSATION
    return np.clip(result, 0, 255).astype(np.float32, copy=False)


def _aperture_grille_pattern(w: int, pitch: int) -> np.ndarray:
    """Build an aperture-grille modulation pattern (W, 3) float32.

    The pattern depends only on the column — the caller broadcasts it over
    rows, so no (H, W, 3) intermediate is ever materialised.
    """
    stripe_idx = (np.arange(w) // pitch) % 3  # (W,) transmitting channel
    pattern = np.zeros((w, 3), dtype=np.float32)
    pattern[np.arange(w), stripe_idx] = 1.0
    return pattern


def _shadow_mask_patterns(w: int, pitch: int) -> tuple[np.ndarray, np.ndarray]:
    """Build the even-row and odd-row shadow-mask patterns (W, 3) each.

    The staggered-triad pattern depends only on the column and the row
    parity, so two per-column patterns cover the whole frame.

    Returns:
        Tuple ``(even, odd)`` of (W, 3) float32 patterns.
    """
    shift = (3 * pitch) // 2  # horizontal offset for odd rows
    even = _aperture_grille_pattern(w, pitch)
    odd = np.zeros((w, 3), dtype=np.float32)
    shifted_idx = ((np.arange(w) - shift) // pitch) % 3
    odd[np.arange(w), shifted_idx] = 1.0
    return even, odd


# ---------------------------------------------------------------------------
# Bloom
# ---------------------------------------------------------------------------


def bloom(
    frame: np.ndarray,
    threshold: float = 0.7,
    radius: float = 8.0,
    intensity: float = 0.5,
    blend: str = BLEND_SCREEN,
) -> np.ndarray:
    """Add a bloom/glow effect to bright areas of *frame*.

    Bright pixels above *threshold* are blurred and blended back into the
    original using either screen or additive blending.

    Args:
        frame: HWC float32 [0, 255] RGB image.
        threshold: Luminance fraction (``[0, 1]``) above which pixels glow.
        radius: Gaussian blur sigma in pixels (>= 0.05).
        intensity: Bloom contribution multiplier in ``[0, 4]``.
        blend: Blend mode — one of :data:`BLOOM_BLENDS`.

    Returns:
        Processed frame with same shape/dtype as input.

    Raises:
        ValueError: On invalid parameters.
    """
    if threshold < 0 or threshold > 1:
        raise ValueError("threshold must be in [0, 1]")
    if radius < 0.05:
        raise ValueError("radius must be >= 0.05")
    if radius > 200:
        raise ValueError("radius must be <= 200")
    if intensity < 0 or intensity > 4:
        raise ValueError("intensity must be in [0, 4]")
    if blend not in BLOOM_BLENDS:
        raise ValueError(f"blend must be one of {BLOOM_BLENDS}")

    h, w, _ = frame.shape

    # Soft bright mask from luma. float32 weights keep the whole pipeline
    # (including the Gaussian blur, which is ~2x slower on float64) in
    # float32 — np.dot with a tuple of Python floats would upcast to f64.
    luma_weights = np.asarray(LUMA_WEIGHTS, dtype=np.float32)
    luma = np.dot(frame[:, :, :3], luma_weights)  # (H, W) float32
    thresh_px = threshold * 255.0
    denom = max(255.0 - thresh_px, 1.0)
    m = np.clip((luma - thresh_px) / denom, 0, 1)

    if threshold >= 1.0:
        # All mask values are zero → output == input
        return frame.copy()

    # Gaussian blur kernel size (odd, minimum 3)
    ksize = max(3, 2 * int(np.ceil(3 * radius)) + 1)
    glow = cv2.GaussianBlur(frame * m[..., np.newaxis], (ksize, ksize), sigmaX=radius)
    glow *= intensity

    if blend == BLEND_SCREEN:
        # out = 255 - (255 - frame) * (255 - glow) / 255
        result = 255.0 - (255.0 - frame) * (255.0 - glow) / 255.0
    else:  # BLEND_ADD
        result = frame + glow

    return np.clip(result, 0, 255).astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# Chromatic aberration
# ---------------------------------------------------------------------------


def chromatic_aberration(
    frame: np.ndarray,
    amount: float = 2.0,
    mode: str = ABERRATION_RADIAL,
) -> np.ndarray:
    """Apply a chromatic-aberration displacement to *frame*.

    The red channel is displaced rightward (linear mode) / toward the
    center (radial mode) and the blue channel the opposite way, so red
    fringes appear on the right / inner side of edges and blue on the
    left / outer side.  Green remains at its original position.

    Args:
        frame: HWC float32 [0, 255] RGB image.
        amount: Displacement magnitude in pixels (``[0, 20]``).
        mode: Displacement mode — one of :data:`ABERRATION_MODES`.

    Returns:
        Processed frame with same shape/dtype as input.

    Raises:
        ValueError: On invalid parameters.
    """
    if amount < 0 or amount > 20:
        raise ValueError("amount must be in [0, 20]")
    if mode not in ABERRATION_MODES:
        raise ValueError(f"mode must be one of {ABERRATION_MODES}")

    if amount == 0.0:
        return frame.copy()

    h, w, _ = frame.shape

    if mode == ABERRATION_LINEAR:
        (map_x_r, map_y_r), (map_x_b, map_y_b) = _build_linear_maps(h, w, amount)
    else:  # ABERRATION_RADIAL
        cx = (w - 1) / 2.0
        cy = (h - 1) / 2.0
        (map_x_r, map_y_r), (map_x_b, map_y_b) = _build_radial_maps(
            h, w, amount, cx, cy,
        )

    # Green stays at its original position; only R and B are resampled.
    # Two full-frame remaps on the contiguous HWC frame (green simply kept)
    # are several times faster than per-channel remaps on non-contiguous
    # channel views — cv2 pays once per call, and the output stays float32.
    result = frame.copy()
    result[:, :, 0] = cv2.remap(
        frame, map_x_r, map_y_r,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )[:, :, 0]
    result[:, :, 2] = cv2.remap(
        frame, map_x_b, map_y_b,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )[:, :, 2]

    return np.clip(result, 0, 255).astype(np.float32, copy=False)

def _build_linear_maps(
    h: int, w: int, amount: float,
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Build remap maps for linear chromatic aberration.

    Returns:
        Two (map_x, map_y) float32 tuples — red sampled at ``(x - amount,
        y)`` and blue at ``(x + amount, y)``.
    """
    x = np.arange(w, dtype=np.float32)
    y = np.arange(h, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)

    # Red: sampled at (x - amount, y) → red content shifts right,
    # red fringe on the right edge (blue on the left)
    return (xx - amount, yy), (xx + amount, yy)


def _build_radial_maps(
    h: int, w: int, amount: float, cx: float, cy: float,
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Build remap maps for radial chromatic aberration.

    Returns:
        Two (map_x, map_y) float32 tuples — red sampled outward
        (contracted toward the center) and blue sampled inward
        (expanded outward).
    """
    x = np.arange(w, dtype=np.float32)
    y = np.arange(h, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)

    # Unit direction vectors from center
    u_x = xx - cx
    u_y = yy - cy
    dist = np.sqrt(u_x * u_x + u_y * u_y)
    dist = np.maximum(dist, np.float32(1e-6))
    u_x /= dist
    u_y /= dist

    off = np.float32(amount)
    # Red: sampled outward → contracted toward center (red displaced inward)
    map_r = (xx + u_x * off, yy + u_y * off)
    # Blue: sampled inward → expanded outward (blue displaced outward)
    map_b = (xx - u_x * off, yy - u_y * off)
    return map_r, map_b


# ---------------------------------------------------------------------------
# Curvature (barrel screen curve)
# ---------------------------------------------------------------------------


def curvature(
    frame: np.ndarray,
    amount: float = 0.15,
    edge_softness: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a barrel-distortion (CRT screen curve) to *frame*.

    Pixels near the corners are pulled inward, simulating the convex
    curvature of an old CRT display.

    The sampling coordinates are compensated by ``1 / (1 + amount)`` so
    the midpoints of all four frame edges (r = 1) sample exactly the
    original edge pixels. Without the compensation every border pixel
    (r >= 1) sampled past the frame edge and the whole border rendered
    as straight black bands; with it only the corners can go black —
    the natural curvature artifact.

    The black edge can be feathered with *edge_softness*: instead of
    blurring the image (which would smear content into the black region
    and black into the content), the alpha ramp that darkens the border
    is widened to a smoothstep of the given width in source pixels. The
    remap then uses clamped border extension so there is real content
    under the wider ramp to fade out; at ``edge_softness == 0`` the
    result is bit-identical to the hard constant-border cut.

    Args:
        frame: HWC float32 [0, 255] RGB image.
        amount: Distortion strength in ``[0, 1]``. Zero is identity.
        edge_softness: Width of the smooth falloff into the black
            border, in source pixels (``>= 0``). Zero is a hard cut.

    Returns:
        Tuple ``(result, mask)``. ``result`` is the processed frame with
        the same shape/dtype as the input. ``mask`` is a single-channel
        (H, W) float32 map of the blackened pixels: ``1.0`` where the
        pixel is fully black, ``0.0`` where content is untouched,
        fractional on the (feathered) curved boundary.

    Raises:
        ValueError: On invalid parameters.
    """
    if amount < 0 or amount > 1:
        raise ValueError("amount must be in [0, 1]")
    if edge_softness < 0:
        raise ValueError("edge_softness must be >= 0")

    h, w, _ = frame.shape

    if amount == 0.0:
        return frame.copy(), np.zeros((h, w), dtype=np.float32)

    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0

    # Map arithmetic in float32: coordinates stay well inside the 24-bit
    # mantissa, and remap wants float32 maps anyway — float64 intermediates
    # would double the memory traffic for no visible precision gain.
    x = np.arange(w, dtype=np.float32)
    y = np.arange(h, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)

    # Normalized coordinates
    u = (xx - cx) / max(cx, 1e-6)
    v = (yy - cy) / max(cy, 1e-6)
    r2 = u * u + v * v

    # Barrel: source coords = center + (pos - center) * (1 + amount * r^2),
    # scaled by 1 / (1 + amount) so r = 1 (frame-edge midpoints) samples
    # exactly the frame edge. Corners (r^2 up to 2) still overshoot.
    scale = (1.0 + amount * r2) / (1.0 + amount)
    src_x = cx + (xx - cx) * scale
    src_y = cy + (yy - cy) * scale

    # Clamped border extension: real (edge-smeared) content under the
    # feather ramp instead of instant black. At softness = 0 the alpha
    # below reproduces the constant-border behaviour exactly, so the
    # border mode is safe to keep clamped in both cases. One remap call
    # on the 3-channel frame instead of three per-channel calls.
    result = cv2.remap(
        frame,
        src_x,
        src_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )

    # Border alpha: total valid content weight at each sample — the hard
    # 1 px bilinear ramp, or a smoothstep widened to edge_softness px.
    alpha = _edge_alpha(src_x, w, edge_softness) * _edge_alpha(src_y, h, edge_softness)
    result = result * alpha[:, :, np.newaxis]

    # Blackened-pixel mask: 1 - alpha is the blackening fraction.
    mask = (1.0 - alpha).astype(np.float32)

    return np.clip(result, 0, 255).astype(np.float32, copy=False), mask


def _edge_alpha(src: np.ndarray, n: int, softness: float) -> np.ndarray:
    """Per-axis alpha of valid content at remap samples along one axis.

    Args:
        src: Array of sample coordinates (pixel centres at ``0..n-1``).
        n: Number of pixels along the axis.
        softness: Feather width in pixels; ``0`` gives the hard 1 px
            bilinear ramp of :func:`_bilinear_coverage`.

    Returns:
        Array of per-sample alpha fractions in ``[0, 1]``.
    """
    if softness <= 0.0:
        return _bilinear_coverage(src, n)

    # Signed distance to the nearest frame edge (pixel centres at
    # 0 .. n-1): positive inside, negative outside. The smoothstep ramp
    # is symmetric about the edge — black at softness/2 outside, full
    # content at softness/2 inside. Defined in source coordinates, so
    # under the distortion the ramp compresses where the remap derivative
    # is large (near the corners) and stays ~softness px at the edge
    # midpoints — no grey mush even on 1-2 px wedges.
    dist = np.minimum(src, (n - 1) - src)
    t = np.clip((dist + softness * 0.5) / softness, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _bilinear_coverage(src: np.ndarray, n: int) -> np.ndarray:
    """Fraction of bilinear taps that land inside ``[0, n - 1]``.

    For an INTER_LINEAR remap sample at coordinate ``s`` (pixel centres
    at ``0 .. n - 1``) cv2 blends the two nearest taps with weights
    ``(1 - frac, frac)``. The valid weight sum is exactly how much real
    content survives at this pixel when the border is clamped.

    Args:
        src: Array of sample coordinates (any shape).
        n: Number of pixels along the axis.

    Returns:
        Array of per-sample coverage fractions in ``[0, 1]``.
    """
    i0 = np.floor(src)
    frac = src - i0
    valid0 = (i0 >= 0.0) & (i0 <= n - 1)
    valid1 = (i0 + 1.0 >= 0.0) & (i0 + 1.0 <= n - 1)
    return (1.0 - frac) * valid0 + frac * valid1


# ---------------------------------------------------------------------------
# Vignette
# ---------------------------------------------------------------------------


def vignette(
    frame: np.ndarray,
    strength: float = 0.4,
    radius: float = 0.8,
    softness: float = 0.5,
) -> np.ndarray:
    """Apply a vignette (corner-darkening) effect to *frame*.

    A smooth radial falloff darkens the edges of the image, with the
    transition controlled by *radius* and *softness*.

    Args:
        frame: HWC float32 [0, 255] RGB image.
        strength: Maximum darkening factor in ``[0, 1]``.
        radius: Normalised distance at which darkening begins (``[0, 1]``).
        softness: Controls the steepness of the transition in ``[0, 1]``.

    Returns:
        Processed frame with same shape/dtype as input.

    Raises:
        ValueError: On invalid parameters.
    """
    if strength < 0 or strength > 1:
        raise ValueError("strength must be in [0, 1]")
    if radius < 0 or radius > 1:
        raise ValueError("radius must be in [0, 1]")
    if softness < 0 or softness > 1:
        raise ValueError("softness must be in [0, 1]")

    h, w, _ = frame.shape
    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0
    half_diag = np.sqrt(cx * cx + cy * cy)

    # Factor map in float64: with radius close to 1 the transition divisor
    # shrinks towards 1e-6 and would amplify float32 noise in `d` into a
    # visible darkening of pixels that must stay untouched.
    x = np.arange(w, dtype=np.float64)
    y = np.arange(h, dtype=np.float64)
    xx, yy = np.meshgrid(x, y)

    # Normalised distance: 0 at center, 1 at corners
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / max(half_diag, 1e-6)

    # Transition factor t
    t = np.clip((d - radius) / max(1.0 - radius, 1e-6), 0, 1)

    # Factor: darkening exponent increases with softness
    exponent = 1.0 + softness * 4.0
    factor = (1.0 - strength * (t ** exponent)).astype(np.float32)

    result = frame * factor[:, :, np.newaxis]
    return np.clip(result, 0, 255).astype(np.float32, copy=False)
