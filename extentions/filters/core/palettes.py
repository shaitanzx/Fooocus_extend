'''
File:   palettes.py
Brief:  Retro palette definitions and hex-string parser.
Author: Mistress-Lukutar
Date:   2026-11-14
Version: v0.1.0

Palettes are ``(K, 3)`` uint8 arrays in RGB order.
'''

from __future__ import annotations

import itertools
import logging
import re

import numpy as np

logger = logging.getLogger(__name__)

#: Palette combo choices (registry order).
PALETTE_NAMES: tuple[str, ...] = (
    "dmg_4",
    "pico8_16",
    "c64_16",
    "cga_16",
    "zx_spectrum_15",
    "ega_64",
    "nes",
    "gray_4",
    "mono_1bit",
    "sweetie_16",
)

# ---------------------------------------------------------------------------
# Private hex tables
# ---------------------------------------------------------------------------

#: Raw hex strings per palette name (6-digit, "#"-prefix optional).
_HEX_PALETTES: dict[str, tuple[str, ...]] = {
    "dmg_4": (
        "0F380F",
        "306230",
        "8BAC0F",
        "9BBC0F",
    ),
    "pico8_16": (
        "000000",
        "1D2B53",
        "7E2553",
        "008751",
        "AB5236",
        "5F574F",
        "C2C3C7",
        "FFF1E8",
        "FF004D",
        "FFA300",
        "FFEC27",
        "00E436",
        "29ADFF",
        "83769C",
        "FF77A8",
        "FFCCAA",
    ),
    "c64_16": (
        "000000",
        "FFFFFF",
        "68372B",
        "70A4B2",
        "6F3D86",
        "588D43",
        "352879",
        "B8C77F",
        "6F4F25",
        "433900",
        "9A6759",
        "444444",
        "6C6C6C",
        "9AD284",
        "6C5EB5",
        "959595",
    ),
    "cga_16": (
        "000000",
        "0000AA",
        "00AA00",
        "00AAAA",
        "AA0000",
        "AA00AA",
        "AA5500",
        "AAAAAA",
        "555555",
        "5555FF",
        "55FF55",
        "55FFFF",
        "FF5555",
        "FF55FF",
        "FFFF55",
        "FFFFFF",
    ),
    "zx_spectrum_15": (
        "000000",
        "0000D7",
        "D70000",
        "D7D700",
        "00D700",
        "D700D7",
        "00D7D7",
        "D7D7D7",
        "0000FF",
        "FF0000",
        "FFFF00",
        "00FF00",
        "FF00FF",
        "00FFFF",
        "FFFFFF",
    ),
    # ega_64 is generated programmatically below
    "nes": (
        "000000",
        "fcfcfc",
        "f8f8f8",
        "bcbcbc",
        "7c7c7c",
        "a4e4fc",
        "3cbcfc",
        "0078f8",
        "0000fc",
        "b8b8f8",
        "6888fc",
        "0058f8",
        "0000bc",
        "d8b8f8",
        "9878f8",
        "6844fc",
        "4428bc",
        "f8b8f8",
        "f878f8",
        "d800cc",
        "940084",
        "f8a4c0",
        "f85898",
        "e40058",
        "a80020",
        "f0d0b0",
        "f87858",
        "f83800",
        "a81000",
        "fce0a8",
        "fca044",
        "e45c10",
        "881400",
        "f8d878",
        "f8b800",
        "ac7c00",
        "503000",
        "d8f878",
        "b8f818",
        "00b800",
        "007800",
        "b8f8b8",
        "58d854",
        "00a800",
        "006800",
        "b8f8d8",
        "58f898",
        "00a844",
        "005800",
        "00fcfc",
        "00e8d8",
        "008888",
        "004058",
        "f8d8f8",
        "787878",
    ),
    "gray_4": (
        "000000",
        "555555",
        "AAAAAA",
        "FFFFFF",
    ),
    "mono_1bit": (
        "000000",
        "FFFFFF",
    ),
    "sweetie_16": (
        "1a1c2c",
        "5d275d",
        "b13e53",
        "ef7d57",
        "ffcd75",
        "a7f070",
        "38b764",
        "257179",
        "29366f",
        "3b5dc9",
        "41a6f6",
        "73eff7",
        "f4f4f4",
        "94b0c2",
        "566c86",
        "333c57",
    ),
}

# Generate EGA 64: all combinations of (0, 85, 170, 255) per channel.
#: Four-level-per-channel EGA gamut (4 values × 4 values × 4 values = 64).
_EGA_LEVELS: tuple[int, ...] = (0, 85, 170, 255)
_ega_colors: list[str] = [
    f"{r:02x}{g:02x}{b:02x}"
    for r, g, b in itertools.product(_EGA_LEVELS, _EGA_LEVELS, _EGA_LEVELS)
]
_HEX_PALETTES["ega_64"] = tuple(_ega_colors)


# ---------------------------------------------------------------------------
# Build resolved palettes at import time
# ---------------------------------------------------------------------------


def _hex_to_rgb(hex_str: str) -> tuple[int, int, int]:
    """Parse a 6-character hex colour string to ``(r, g, b)`` ints."""
    s = hex_str.lstrip("#")
    if len(s) != 6:
        raise ValueError(f"Expected 6 hex digits, got {len(s)} in '{hex_str}'")
    return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16))


def _build_palettes() -> dict[str, np.ndarray]:
    """Build the ``PALETTES`` dict from hex strings at import time."""
    result: dict[str, np.ndarray] = {}
    for name, colors in _HEX_PALETTES.items():
        rows: list[list[int]] = []
        for c in colors:
            rows.append(list(_hex_to_rgb(c)))
        arr = np.array(rows, dtype=np.uint8)
        result[name] = arr
    return result


#: Resolved palettes (name -> ``(K, 3)`` uint8 RGB).
PALETTES: dict[str, np.ndarray] = _build_palettes()

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_palette(name: str) -> np.ndarray:
    """Return a *copy* of the named palette.

    Args:
        name: Palette identifier from :data:`PALETTE_NAMES`.

    Returns:
        A ``(K, 3)`` uint8 RGB array copy (safe to mutate).

    Raises:
        ValueError: If *name* is not recognised.
    """
    if name not in PALETTES:
        raise ValueError(f"Unknown palette: {name!r}")
    return PALETTES[name].copy()


def parse_hex_palette(text: str) -> np.ndarray:
    """Parse a custom palette string into a ``(K, 3)`` uint8 RGB array.

    Tokens are split on commas, semicolons, whitespace and newlines.
    Each token must be exactly 6 hex digits (optionally prefixed with ``#``).
    3-digit shorthand is *not* supported (raises ``ValueError``).

    Args:
        text: Palette colour string — tokens separated by comma/semicolon/whitespace.

    Returns:
        ``(K, 3)`` uint8 RGB array with ``K >= 1`` rows.

    Raises:
        ValueError: On empty input or malformed token.
    """
    # Split on any of: commas, semicolons, whitespace (including newlines).
    tokens = re.split(r"[,\s;\n]+", text.strip())
    # Filter out empty tokens that arise from leading/trailing separators.
    tokens = [t for t in tokens if t]

    if not tokens:
        raise ValueError("parse_hex_palette: no colour tokens found")

    rows: list[list[int]] = []
    hex_re = re.compile(r"^#?([0-9a-fA-F]{6})$")
    for token in tokens:
        m = hex_re.match(token)
        if not m:
            raise ValueError(
                f"parse_hex_palette: invalid colour token {token!r}"
            )
        raw = m.group(1)
        rows.append([int(raw[0:2], 16), int(raw[2:4], 16), int(raw[4:6], 16)])

    return np.array(rows, dtype=np.uint8)
