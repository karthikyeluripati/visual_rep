"""Vendored fonts (DejaVu, see assets/fonts/LICENSE_DEJAVU) so text metrics are machine-independent."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from PIL import ImageFont

FONT_DIR = Path(__file__).parent / "assets" / "fonts"
FONT_FILES = {
    "sans": "DejaVuSans.ttf",
    "serif": "DejaVuSerif.ttf",
    "mono": "DejaVuSansMono.ttf",
}


@lru_cache(maxsize=256)
def get_font(name: str, size_px: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONT_DIR / FONT_FILES[name]), max(1, int(size_px)))


def text_bbox(text: str, x: float, y: float, size: float, font: str) -> list[float]:
    """Canonical-space bbox of `text` anchored at (x, y) with PIL anchor 'lt'.

    Measured at 4x and divided to reduce integer rounding."""
    f = get_font(font, int(round(size * 4)))
    b = f.getbbox(text, anchor="lt")
    return [x + b[0] / 4.0, y + b[1] / 4.0, x + b[2] / 4.0, y + b[3] / 4.0]


def text_size(text: str, size: float, font: str) -> tuple[float, float]:
    b = text_bbox(text, 0.0, 0.0, size, font)
    return (b[2] - b[0], b[3] - b[1])
