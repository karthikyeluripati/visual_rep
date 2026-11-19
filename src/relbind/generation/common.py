"""Shared generation utilities: text placement, leader routing, connector assignment."""
from __future__ import annotations

import math
from typing import Callable

import numpy as np

from .. import geometry as G
from ..fonts import text_bbox, text_size
from ..schema import Annotation, Connector, Relation, Scene, TextItem
from ..verify import check_connector


class GenerationError(RuntimeError):
    pass


def make_text(text: str, x: float, y: float, size: float, font: str) -> TextItem:
    return TextItem(text=text, x=round(x, 2), y=round(y, 2), size=size, bbox=[round(v, 2) for v in text_bbox(text, x, y, size, font)], font=font)


def text_at_bbox_corner(text: str, size: float, font: str, *, left=None, right=None, top=None, bottom=None,
                        cx=None, cy=None) -> TextItem:
    """Place text so its bbox edge/centre lands at the given coordinate."""
    probe = text_bbox(text, 0.0, 0.0, size, font)
    w, h = probe[2] - probe[0], probe[3] - probe[1]
    if left is not None:
        x0 = left
    elif right is not None:
        x0 = right - w
    else:
        x0 = cx - w / 2.0
    if top is not None:
        y0 = top
    elif bottom is not None:
        y0 = bottom - h
    else:
        y0 = cy - h / 2.0
    return make_text(text, x0 - probe[0], y0 - probe[1], size, font)


def text_obstacle_ok(scene: Scene, b: list[float], pad_text=10.0, pad_line=8.0, pad_solid=10.0) -> bool:
    """Is bbox `b` free of every existing text, entity and distractor stroke?"""
    W, H = scene.canvas
    if b[0] < 4 or b[1] < 4 or b[2] > W - 4 or b[3] > H - 4:
        return False
    for a in scene.annotations:
        if G.bbox_overlap(G.inflate(b, pad_text), a.label.bbox):
            return False
    for e in scene.entities:
        if e.label is not None and G.bbox_overlap(G.inflate(b, pad_text), e.label.bbox):
            return False
        if e.crossable:
            g = e.geometry["points"]
            if any(G.seg_intersects_bbox(p, q, G.inflate(b, pad_line)) for p, q in G.segments(g)):
                return False
            for sb in e.attrs.get("solids", []):
                if G.bbox_overlap(G.inflate(b, pad_solid), sb):
                    return False
        elif G.bbox_overlap(G.inflate(b, pad_solid), e.bbox):
            return False
        for p in e.attrs.get("draw", []):
            if p["type"] == "text" and G.bbox_overlap(G.inflate(b, pad_text), p["bbox"]):
                return False
    for p in scene.distractors:
        if p["type"] == "text":
            if G.bbox_overlap(G.inflate(b, pad_text), p["bbox"]):
                return False
        elif p["type"] == "line":
            if any(G.seg_intersects_bbox(q, r, G.inflate(b, pad_line)) for q, r in G.segments(p["points"])):
                return False
        elif p["type"] == "polygon":
            pts = p["points"] + [p["points"][0]]
            if any(G.seg_intersects_bbox(q, r, G.inflate(b, pad_line)) for q, r in G.segments(pts)):
                return False
            if G.bbox_overlap(G.inflate(b, pad_solid), G.bbox_of_points(p["points"])):
                return False
        elif p["type"] == "rect":
            if G.bbox_overlap(G.inflate(b, pad_solid), p["bbox"]):
                return False
        elif p["type"] == "circle":
            c, r = p["center"], p["r"]
            if G.bbox_overlap(G.inflate(b, pad_solid), [c[0] - r, c[1] - r, c[0] + r, c[1] + r]):
                return False
    return True


def place_text_randomly(scene: Scene, rng: np.random.Generator, text: str, size: float, font: str,
                        region: list[float], tries: int = 200, **pads) -> TextItem | None:
    w, h = text_size(text, size, font)
    for _ in range(tries):
        x0 = rng.uniform(region[0], max(region[0] + 1, region[2] - w))
        y0 = rng.uniform(region[1], max(region[1] + 1, region[3] - h))
        t = text_at_bbox_corner(text, size, font, left=x0, top=y0)
        if text_obstacle_ok(scene, t.bbox, **pads):
            return t
    return None


# ----------------------------------------------------------------------------- leader routing

def leader_route(ann_bbox: list[float], end, cls: str, rng: np.random.Generator, gap: float = 4.0) -> list[list[float]]:
    """Drafting-style leader from an annotation bbox to `end`.

    straight: single segment from the bbox boundary.
    elbow:    short horizontal (or vertical) shoulder, then straight to the end.
    zigzag:   shoulder, one intermediate bend offset perpendicular to the chord, then the end."""
    cx, cy = G.bbox_center(ann_bbox)
    if cls == "straight":
        s = G.bbox_exit_point(ann_bbox, end, gap)
        return [G.round_pt(s), G.round_pt(end)]
    L = float(rng.uniform(14, 32))
    if abs(end[0] - cx) >= abs(end[1] - cy) * 0.35 and abs(end[0] - cx) > (ann_bbox[2] - ann_bbox[0]) / 2 + L + 14:
        side = "right" if end[0] > cx else "left"
        s = G.side_midpoint(ann_bbox, side, gap)
        s2 = (s[0] + (L if side == "right" else -L), s[1])
    else:
        side = "bottom" if end[1] > cy else "top"
        s = G.side_midpoint(ann_bbox, side, gap)
        s2 = (s[0], s[1] + (L if side == "bottom" else -L))
    if cls == "elbow":
        return [G.round_pt(s), G.round_pt(s2), G.round_pt(end)]
    if cls == "zigzag":
        mx, my = (s2[0] + end[0]) / 2.0, (s2[1] + end[1]) / 2.0
        dx, dy = end[0] - s2[0], end[1] - s2[1]
        n = math.hypot(dx, dy) or 1.0
        off = float(rng.uniform(28, 75)) * (1 if rng.random() < 0.5 else -1)
        w = (mx - dy / n * off, my + dx / n * off)
        return [G.round_pt(s), G.round_pt(s2), G.round_pt(w), G.round_pt(end)]
    raise ValueError(cls)


RouteFn = Callable[[Scene, Annotation, list | None, list, str, np.random.Generator, object], list]


def try_connect(scene: Scene, ann: Annotation, target, attach_candidates: list, route_fn: RouteFn,
                cls: str, rng: np.random.Generator, terminator: str, kind: str, start_mode: str,
                fixed_start=None, cid: str | None = None, tries_per_attach: int = 6,
                existing: list[Connector] | None = None) -> Connector | None:
    """Route a connector from `ann` to `target`, returning the first one that verifies."""
    existing = scene.connectors if existing is None else existing
    cid = cid or f"k{ann.id[1:]}"
    for attach in attach_candidates:
        for _ in range(tries_per_attach):
            pts = route_fn(scene, ann, fixed_start, attach, cls, rng, target)
            if pts is None:
                continue
            c = Connector(id=cid, source=ann.id, target=target.id, points=pts, terminator=terminator,
                          route_class=cls, kind=kind, start_mode=start_mode)
            if not check_connector(scene, c, others=existing):
                return c
    return None


def add_relation(scene: Scene, c: Connector) -> None:
    scene.connectors.append(c)
    scene.relations.append(Relation(annotation=c.source, entity=c.target, connector=c.id))


def unique_choice(rng: np.random.Generator, pool: list[str], k: int) -> list[str]:
    if k > len(pool):
        raise GenerationError("pool too small")
    idx = rng.permutation(len(pool))[:k]
    return [pool[i] for i in idx]


def dimension_values(rng: np.random.Generator, k: int) -> list[str]:
    vals: set[str] = set()
    while len(vals) < k:
        ft = int(rng.integers(1, 15))
        inch = int(rng.integers(0, 12))
        vals.add(f"{ft}'-{inch}\"")
    out = sorted(vals)
    rng.shuffle(out)
    return out
