"""Deterministic rendering: Scene + Style -> display list -> (PNG raster | SVG).

The raster produced here is the canonical model input. The SVG is an archival vector copy of the
same display list (text metrics in SVG viewers may differ slightly from the raster).
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any
from xml.sax.saxutils import escape

from PIL import Image, ImageDraw

from . import geometry as G
from .fonts import get_font
from .schema import Scene

RGB = tuple[int, int, int]


@dataclass
class Style:
    ink: RGB = (17, 17, 17)
    connector_ink: RGB | None = None  # None -> ink
    bg: RGB = (255, 255, 255)
    w_entity: float = 2.4
    w_connector: float = 1.4
    w_distractor: float = 1.2
    w_thin: float = 1.0
    leader_dash: list[float] | None = None
    terminator: str | None = None  # override every connector's terminator (style-only change)
    arrow_len: float = 11.0
    arrow_half: float = 4.2
    dot_r: float = 3.2
    distractor_dash: list[float] = field(default_factory=lambda: [10.0, 4.0, 2.0, 4.0])
    name: str = "base"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Style":
        d = dict(d)
        for k in ("ink", "bg", "connector_ink"):
            if d.get(k) is not None:
                d[k] = tuple(d[k])
        return Style(**d)


# ----------------------------------------------------------------------------- display list

def connector_primitives(c, style: Style) -> list[dict[str, Any]]:
    prims: list[dict[str, Any]] = [
        {"type": "line", "points": c.points, "width_role": "connector", "dash": style.leader_dash, "ink_role": "connector"}
    ]
    term = style.terminator or c.terminator
    tip = c.points[-1]
    prev = c.points[-2]
    dx, dy = tip[0] - prev[0], tip[1] - prev[1]
    n = math.hypot(dx, dy) or 1.0
    ux, uy = dx / n, dy / n
    if term == "arrow":
        L, H = style.arrow_len, style.arrow_half
        base = (tip[0] - ux * L, tip[1] - uy * L)
        pts = [list(tip), [base[0] - uy * H, base[1] + ux * H], [base[0] + uy * H, base[1] - ux * H]]
        prims.append({"type": "polygon", "points": pts, "fill": True, "width_role": "thin", "ink_role": "connector"})
    elif term == "dot":
        prims.append({"type": "circle", "center": list(tip), "r": style.dot_r, "fill": True, "width_role": "thin", "ink_role": "connector"})
    elif term == "tick":
        L = 6.0
        px, py = (-uy + ux) * L / math.sqrt(2), (ux + uy) * L / math.sqrt(2)
        prims.append({"type": "line", "points": [[tip[0] - px, tip[1] - py], [tip[0] + px, tip[1] + py]],
                      "width_role": "connector", "dash": None, "ink_role": "connector"})
    return prims


def display_list(scene: Scene, style: Style) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    texts: list[dict[str, Any]] = []
    for p in scene.distractors:
        (texts if p["type"] == "text" else out).append(p)
    for e in scene.entities:
        for p in e.attrs.get("draw", []):
            (texts if p["type"] == "text" else out).append(p)
        if e.label is not None:
            texts.append({"type": "text", **asdict(e.label)})
    for c in scene.connectors:
        out.extend(connector_primitives(c, style))
    for a in scene.annotations:
        texts.append({"type": "text", **asdict(a.label)})
    return out + texts


# ----------------------------------------------------------------------------- raster

def _dash_polyline(points, dash):
    """Split a polyline into dash segments (list of 2-point polylines)."""
    pattern = list(dash)
    out = []
    idx, rem, on = 0, pattern[0], True
    for p, q in G.segments(points):
        seg_len = G.dist(p, q)
        if seg_len < 1e-9:
            continue
        t = 0.0
        while t < seg_len - 1e-9:
            step = min(rem, seg_len - t)
            a = (p[0] + (q[0] - p[0]) * t / seg_len, p[1] + (q[1] - p[1]) * t / seg_len)
            b = (p[0] + (q[0] - p[0]) * (t + step) / seg_len, p[1] + (q[1] - p[1]) * (t + step) / seg_len)
            if on:
                out.append([a, b])
            t += step
            rem -= step
            if rem <= 1e-9:
                idx = (idx + 1) % len(pattern)
                rem = pattern[idx]
                on = not on
    return out


def _width(style: Style, role: str) -> float:
    return {"entity": style.w_entity, "connector": style.w_connector, "distractor": style.w_distractor,
            "thin": style.w_thin}.get(role, style.w_thin)


def _ink(style: Style, prim: dict[str, Any]) -> RGB:
    if prim.get("ink_role") == "connector" and style.connector_ink is not None:
        return style.connector_ink
    return style.ink


def render_png(scene: Scene, style: Style, scale: float = 1.0, supersample: int = 3) -> Image.Image:
    W, H = scene.canvas
    k = scale * supersample
    img = Image.new("RGB", (int(round(W * k)), int(round(H * k))), style.bg)
    d = ImageDraw.Draw(img)

    def P(pt):
        return (pt[0] * k, pt[1] * k)

    for prim in display_list(scene, style):
        t = prim["type"]
        ink = _ink(style, prim)
        if t == "line":
            w = max(1, int(round(_width(style, prim.get("width_role", "thin")) * k)))
            pieces = _dash_polyline(prim["points"], prim["dash"]) if prim.get("dash") else [prim["points"]]
            for piece in pieces:
                d.line([P(p) for p in piece], fill=ink, width=w, joint="curve")
                r = w / 2.0
                for p in (piece[0], piece[-1]):  # round caps
                    x, y = P(p)
                    d.ellipse([x - r, y - r, x + r, y + r], fill=ink)
        elif t == "polygon":
            pts = [P(p) for p in prim["points"]]
            if prim.get("fill"):
                d.polygon(pts, fill=ink)
            else:
                w = max(1, int(round(_width(style, prim.get("width_role", "thin")) * k)))
                d.line(pts + [pts[0]], fill=ink, width=w, joint="curve")
        elif t == "circle":
            x, y = P(prim["center"])
            r = prim["r"] * k
            if prim.get("fill"):
                d.ellipse([x - r, y - r, x + r, y + r], fill=ink)
            else:
                w = max(1, int(round(_width(style, prim.get("width_role", "thin")) * k)))
                d.ellipse([x - r, y - r, x + r, y + r], outline=ink, width=w)
        elif t == "rect":
            b = prim["bbox"]
            box = [b[0] * k, b[1] * k, b[2] * k, b[3] * k]
            if prim.get("fill"):
                d.rectangle(box, fill=ink)
            else:
                w = max(1, int(round(_width(style, prim.get("width_role", "thin")) * k)))
                d.rectangle(box, outline=ink, width=w)
        elif t == "text":
            f = get_font(prim["font"], int(round(prim["size"] * k)))
            d.text((prim["x"] * k, prim["y"] * k), prim["text"], font=f, anchor="lt", fill=style.ink)
        else:
            raise ValueError(f"unknown primitive {t}")
    if supersample > 1:
        img = img.resize((int(round(W * scale)), int(round(H * scale))), Image.LANCZOS)
    return img


# ----------------------------------------------------------------------------- svg

_SVG_FONTS = {"sans": "DejaVu Sans", "serif": "DejaVu Serif", "mono": "DejaVu Sans Mono"}


def render_svg(scene: Scene, style: Style) -> str:
    W, H = scene.canvas

    def col(c):
        return "rgb(%d,%d,%d)" % tuple(c)

    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
             f'<rect width="{W}" height="{H}" fill="{col(style.bg)}"/>']
    for prim in display_list(scene, style):
        t = prim["type"]
        ink = col(_ink(style, prim))
        w = _width(style, prim.get("width_role", "thin"))
        if t == "line":
            pts = " ".join(f"{p[0]:.2f},{p[1]:.2f}" for p in prim["points"])
            dash = f' stroke-dasharray="{",".join(str(x) for x in prim["dash"])}"' if prim.get("dash") else ""
            parts.append(f'<polyline points="{pts}" fill="none" stroke="{ink}" stroke-width="{w}" '
                         f'stroke-linecap="round" stroke-linejoin="round"{dash}/>')
        elif t == "polygon":
            pts = " ".join(f"{p[0]:.2f},{p[1]:.2f}" for p in prim["points"])
            fill = ink if prim.get("fill") else "none"
            parts.append(f'<polygon points="{pts}" fill="{fill}" stroke="{ink}" stroke-width="{w}"/>')
        elif t == "circle":
            fill = ink if prim.get("fill") else "none"
            c = prim["center"]
            parts.append(f'<circle cx="{c[0]:.2f}" cy="{c[1]:.2f}" r="{prim["r"]}" fill="{fill}" stroke="{ink}" stroke-width="{w}"/>')
        elif t == "rect":
            b = prim["bbox"]
            fill = ink if prim.get("fill") else "none"
            parts.append(f'<rect x="{b[0]:.2f}" y="{b[1]:.2f}" width="{b[2]-b[0]:.2f}" height="{b[3]-b[1]:.2f}" '
                         f'fill="{fill}" stroke="{ink}" stroke-width="{w}"/>')
        elif t == "text":
            b = prim["bbox"]
            parts.append(f'<text x="{b[0]:.2f}" y="{b[3]:.2f}" font-family="{_SVG_FONTS[prim["font"]]}" '
                         f'font-size="{prim["size"]}" fill="{col(style.ink)}">{escape(prim["text"])}</text>')
    parts.append("</svg>")
    return "\n".join(parts)
