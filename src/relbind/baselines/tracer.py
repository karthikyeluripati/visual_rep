"""Deterministic pixel connector tracer (B0-trace).

Oracle level: uses the GT boxes of all texts and the GT shapes of nodes (entities and flowchart
decision diamonds) as masks, exactly like `cc_tracer`. It does NOT use connector geometry or relations.

Algorithm (per annotation):
  1. seeds = unmasked ink within 8 px of the annotation box; cluster them; trace from each cluster
     starting in the direction pointing away from the annotation.
  2. step 2 px at a time; prefer the straightest continuation within ±30° (so crossings are passed
     straight through); if none, allow turns up to ±100° (elbows, dog-legs).
  3. on reaching a masked node stroke: if ink continues on the far side along the current heading
     (a crossing of an entity line), jump across; otherwise stop and predict that node.
  4. if the ink ends without touching a node, predict the nearest entity within 12 px, else nothing.
The cluster whose trace ends on an entity and starts closest to the annotation wins.
"""
from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw

from .. import geometry as G
from ..schema import Scene


def _node_masks(scene: Scene, scale: float, shape, w_entity: float):
    H, W = shape
    text = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(text)
    boxes = [a.label.bbox for a in scene.annotations] + [e.label.bbox for e in scene.entities if e.label]
    boxes += [p["bbox"] for e in scene.entities for p in e.attrs.get("draw", []) if p["type"] == "text"]
    boxes += [p["bbox"] for p in scene.distractors if p["type"] == "text"]
    for b in boxes:
        bb = G.inflate(b, 2)
        d.rectangle([v * scale for v in bb], fill=255)
    ent = {}
    for e in scene.entities:
        m = Image.new("L", (W, H), 0)
        md = ImageDraw.Draw(m)
        g = e.geometry
        if g["type"] == "segment":
            md.line([(p[0] * scale, p[1] * scale) for p in g["points"]], fill=255, width=max(3, int(w_entity * scale + 4)))
            for sb in e.attrs.get("solids", []):
                md.ellipse([v * scale for v in G.inflate(sb, 2)], fill=255)
        else:
            md.rectangle([v * scale for v in G.inflate(g["bbox"], 2)], fill=255)
        ent[e.id] = np.asarray(m) > 0
    shapes = Image.new("L", (W, H), 0)
    sd = ImageDraw.Draw(shapes)
    for p in scene.distractors:  # decision diamonds / terminals are nodes, not connectors
        if p.get("role") in ("decision", "terminal"):
            pts = p["points"] if p["type"] == "polygon" else G.bbox_corners(p["bbox"])
            sd.polygon([(x * scale, y * scale) for x, y in pts], fill=255, outline=255, width=4)
    return np.asarray(text) > 0, ent, np.asarray(shapes) > 0


def _rot(d, deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return (d[0] * c - d[1] * s, d[0] * s + d[1] * c)


class Tracer:
    def __init__(self, img: Image.Image, scene: Scene, scale: float, w_entity: float, dark: int = 170):
        g = np.asarray(img.convert("L"))
        self.H, self.W = g.shape
        text, self.ent, shapes = _node_masks(scene, scale, g.shape, w_entity)
        self.any_ent = np.zeros_like(text)
        for m in self.ent.values():
            self.any_ent |= m
        self.ink = (g < dark) & ~text & ~shapes
        self.free = self.ink & ~self.any_ent
        self.scene, self.scale = scene, scale

    def _at(self, m, x, y, r=1):
        xi, yi = int(round(x)), int(round(y))
        x0, x1 = max(0, xi - r), min(self.W, xi + r + 1)
        y0, y1 = max(0, yi - r), min(self.H, yi + r + 1)
        return x0 < x1 and y0 < y1 and bool(m[y0:y1, x0:x1].any())

    def _ent_at(self, x, y):
        xi, yi = int(round(x)), int(round(y))
        if 0 <= xi < self.W and 0 <= yi < self.H and self.any_ent[yi, xi]:
            for eid, m in self.ent.items():
                if m[yi, xi]:
                    return eid
        return None

    def trace(self, p, d, max_steps=3000, step=2.0):
        visited = np.zeros((self.H // 2 + 1, self.W // 2 + 1), dtype=bool)
        x, y = p
        length = 0.0
        prev_cell = None
        for _ in range(max_steps):
            here = (int(y) // 2, int(x) // 2)
            if prev_cell is not None and prev_cell != here:
                visited[prev_cell] = True  # mark a cell only once we have left it
            prev_cell = here
            moved = False
            for lim in ((0, 8, -8, 16, -16, 24, -24, 30, -30),
                        (45, -45, 60, -60, 75, -75, 90, -90, 100, -100)):
                for a in lim:
                    nd = _rot(d, a)
                    nx, ny = x + nd[0] * step, y + nd[1] * step
                    if not (0 <= nx < self.W and 0 <= ny < self.H):
                        continue
                    hit = self._ent_at(nx, ny)
                    if hit is not None:
                        if abs(a) > 30:
                            continue
                        # crossing an entity line? look for ink beyond the mask along the heading
                        for k in range(2, 16):
                            bx, by = nx + nd[0] * k, ny + nd[1] * k
                            if self._ent_at(bx, by) is None:
                                if self._at(self.free, bx, by, 1):
                                    x, y, d = bx, by, nd
                                    moved = True
                                break
                        if moved:
                            break
                        return hit, (nx, ny), length
                    cell = (int(ny) // 2, int(nx) // 2)
                    here = (int(y) // 2, int(x) // 2)
                    # the current cell is always allowed (small/diagonal steps stay inside it);
                    # only cells left behind earlier are blocked, which prevents loops
                    if self._at(self.free, nx, ny, 0) and (cell == here or not visited[cell]):
                        x, y = nx, ny
                        d = (0.7 * d[0] + 0.3 * nd[0], 0.7 * d[1] + 0.3 * nd[1])
                        n = math.hypot(*d) or 1.0
                        d = (d[0] / n, d[1] / n)
                        length += step
                        moved = True
                        break
                if moved:
                    break
            if not moved:
                break
        # dead end: nearest entity within 12 px
        best, bd = None, 12.0 * self.scale
        for e in self.scene.entities:
            g = e.geometry
            q = (x / self.scale, y / self.scale)
            de = (G.point_polyline_dist(q, g["points"]) if g["type"] == "segment" else G.point_bbox_dist(q, g["bbox"])) * self.scale
            if de < bd:
                best, bd = e.id, de
        return best, (x, y), length

    def assign(self, ann_bbox):
        s = self.scale
        b = [v * s for v in ann_bbox]
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        x0, y0 = max(0, int(b[0] - 8 * s)), max(0, int(b[1] - 8 * s))
        x1, y1 = min(self.W, int(b[2] + 8 * s) + 1), min(self.H, int(b[3] + 8 * s) + 1)
        ys, xs = np.nonzero(self.free[y0:y1, x0:x1])
        if len(xs) == 0:
            return None
        xs, ys = xs + x0, ys + y0
        # cluster seeds coarsely by angle around the annotation centre
        ang = np.degrees(np.arctan2(ys - cy, xs - cx))
        bins = np.round(ang / 30).astype(int)
        cands = []
        for bval in np.unique(bins):
            sel = bins == bval
            px, py = float(xs[sel].mean()), float(ys[sel].mean())
            dx, dy = px - cx, py - cy
            n = math.hypot(dx, dy) or 1.0
            # start from the seed pixel closest to the cluster mean
            k = int(np.argmin((xs[sel] - px) ** 2 + (ys[sel] - py) ** 2))
            start = (float(xs[sel][k]), float(ys[sel][k]))
            eid, end, length = self.trace(start, (dx / n, dy / n))
            dist0 = G.point_bbox_dist((start[0] / s, start[1] / s), ann_bbox)
            cands.append((eid is None, dist0, -length, eid))
        cands.sort()
        return cands[0][3] if cands and not cands[0][0] else None


def tracer_map(img: Image.Image, scene: Scene, scale: float, w_entity: float) -> dict[str, str]:
    t = Tracer(img, scene, scale, w_entity)
    out = {}
    for a in scene.annotations:
        e = t.assign(a.label.bbox)
        if e is not None:
            out[a.id] = e
    return out
