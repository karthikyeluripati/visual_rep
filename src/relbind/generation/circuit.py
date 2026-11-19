"""Family C — circuit-style diagrams.

Repeated component symbols, each with a proximity-bound designator (R1, C2, ...). Component values
(4.7kΩ, 100nF, ...) are attached by thin leader lines with a dot terminator. Wires, rails and
junction dots are distractor strokes that leaders must cross cleanly.

ladder: vertical resistors between two rails (all components same type -> no type cue).
grid:   two rows of horizontal resistors and capacitors in series (values are type-compatible, so a
        text prior can narrow candidates by type; swaps are restricted to same-type components).
"""
from __future__ import annotations

import numpy as np

from .. import geometry as G
from ..schema import Annotation, Entity, Scene
from .common import (GenerationError, add_relation, leader_route, place_text_randomly,
                     text_at_bbox_corner, try_connect, unique_choice)

FAMILY = "circuit"
TEMPLATES = ("ladder", "grid")
TERMINATOR = "dot"

R_VALUES = ["1kΩ", "2.2kΩ", "4.7kΩ", "10kΩ", "22kΩ", "47kΩ", "100Ω", "220Ω", "330Ω", "470Ω", "1MΩ", "68kΩ", "3.3kΩ", "15kΩ"]
C_VALUES = ["10nF", "100nF", "1µF", "4.7µF", "22pF", "47pF", "220nF", "10µF", "330pF"]


def route(scene, ann, fixed_start, end, cls, rng, target):
    return leader_route(ann.label.bbox, end, cls, rng)


def _zigzag(x0, y0, x1, y1, amp=8.0, n=6):
    """Resistor zigzag between (x0,y0) and (x1,y1)."""
    pts = [[x0, y0]]
    dx, dy = (x1 - x0), (y1 - y0)
    L = (dx * dx + dy * dy) ** 0.5
    ux, uy = dx / L, dy / L
    for i in range(1, 2 * n):
        t = i / (2 * n)
        s = amp if i % 2 else -amp
        pts.append([round(x0 + dx * t - uy * s, 2), round(y0 + dy * t + ux * s, 2)])
    pts.append([x1, y1])
    return pts


def _resistor(eid, cx, cy, vertical, font):
    half = 34.0
    if vertical:
        body = [cx - 10, cy - half, cx + 10, cy + half]
        path = _zigzag(cx, cy - half, cx, cy + half)
        lab = text_at_bbox_corner(eid, 13, font, left=cx + 15, cy=cy)
    else:
        body = [cx - half, cy - 10, cx + half, cy + 10]
        path = _zigzag(cx - half, cy, cx + half, cy)
        lab = text_at_bbox_corner(eid, 13, font, cx=cx, top=cy + 15)
    bb = [min(body[0], lab.bbox[0]), min(body[1], lab.bbox[1]), max(body[2], lab.bbox[2]), max(body[3], lab.bbox[3])]
    return Entity(id=eid, kind="resistor", bbox=[round(v, 2) for v in bb],
                  geometry={"type": "rect", "bbox": [round(v, 2) for v in body]}, crossable=False, label=lab,
                  attrs={"draw": [{"type": "line", "points": path, "width_role": "entity", "dash": None}],
                         "ctype": "R"})


def _capacitor(eid, cx, cy, font):
    half, gap, ph = 34.0, 5.0, 14.0
    body = [cx - half, cy - ph, cx + half, cy + ph]
    draw = [
        {"type": "line", "points": [[cx - half, cy], [cx - gap, cy]], "width_role": "entity", "dash": None},
        {"type": "line", "points": [[cx + gap, cy], [cx + half, cy]], "width_role": "entity", "dash": None},
        {"type": "line", "points": [[cx - gap, cy - ph], [cx - gap, cy + ph]], "width_role": "entity", "dash": None},
        {"type": "line", "points": [[cx + gap, cy - ph], [cx + gap, cy + ph]], "width_role": "entity", "dash": None},
    ]
    lab = text_at_bbox_corner(eid, 13, font, cx=cx, top=cy + ph + 4)
    bb = [body[0], body[1], body[2], max(body[3], lab.bbox[3])]
    return Entity(id=eid, kind="capacitor", bbox=[round(v, 2) for v in bb],
                  geometry={"type": "rect", "bbox": [round(v, 2) for v in body]}, crossable=False, label=lab,
                  attrs={"draw": draw, "ctype": "C"})


def _wire(scene, pts):
    scene.distractors.append({"type": "line", "points": [[round(p[0], 2), round(p[1], 2)] for p in pts],
                              "width_role": "distractor", "dash": None, "role": "wire"})


def _junction(scene, p):
    scene.distractors.append({"type": "circle", "center": [round(p[0], 2), round(p[1], 2)], "r": 3.5, "fill": True,
                              "width_role": "thin", "role": "junction"})


def _attach_candidates(ent, ann, rng, n=5):
    b = ent.geometry["bbox"]
    acx, acy = G.bbox_center(ann.label.bbox)
    out = []
    for _ in range(n):
        if ent.attrs.get("vertical"):
            cy = (b[1] + b[3]) / 2
            if acx < (b[0] + b[2]) / 2:
                x = b[0]
                y = float(np.clip(acy, b[1] + 6, b[3] - 6) + rng.uniform(-14, 14))
                y = float(np.clip(y, b[1] + 4, b[3] - 4))
            else:  # right side: avoid the designator label at mid-height
                x = b[2]
                y = float(rng.uniform(b[1] + 4, cy - 17) if (acy < cy) ^ (rng.random() < 0.2) else rng.uniform(cy + 17, b[3] - 4))
            out.append([x, round(y, 2)])
        else:
            y = b[1]  # label is below -> attach on the top side
            x = float(np.clip(acx, b[0] + 12, b[2] - 12) + rng.uniform(-14, 14))
            out.append([round(float(np.clip(x, b[0] + 10, b[2] - 10)), 2), y])
    return out


def _assign(scene, rng, reach):
    used: set[str] = set()
    for i in rng.permutation(len(scene.annotations)):
        ann = scene.annotations[i]
        need = ann.label.text in C_VALUES and "C" or "R"
        acx, acy = G.bbox_center(ann.label.bbox)
        cands = [e for e in scene.entities if e.id not in used and e.attrs["ctype"] == need
                 and G.dist(G.bbox_center(e.geometry["bbox"]), (acx, acy)) <= reach]
        rng.shuffle(cands)
        conn = None
        for ent in cands:
            conn = try_connect(scene, ann, ent, _attach_candidates(ent, ann, rng), route, scene.topology, rng,
                               TERMINATOR, "leader", "bbox")
            if conn is not None:
                break
        if conn is None:
            raise GenerationError(f"could not connect {ann.id}")
        used.add(conn.target)
        add_relation(scene, conn)


def _ladder(scene: Scene, rng, font: str) -> None:
    W, H = scene.canvas
    n = int(rng.integers(4, 7))
    top, bot = 100.0, H - 60.0
    x0, x1 = 190.0, W - 80.0
    sp = (x1 - x0) / (n - 1)
    _wire(scene, [[120, top], [W - 40, top]])
    _wire(scene, [[120, bot], [W - 40, bot]])
    for txt, y in (("VCC", top), ("GND", bot)):
        t = text_at_bbox_corner(txt, 13, font, right=110, cy=y)
        scene.distractors.append({"type": "text", **t.__dict__, "role": "rail_text"})
    for i in range(n):
        x = round(x0 + i * sp + rng.uniform(-12, 12), 2)
        cy = float(rng.uniform(270, H - 270))
        e = _resistor(f"R{i + 1}", x, cy, True, font)
        e.attrs["vertical"] = True
        scene.entities.append(e)
        _wire(scene, [[x, top], [x, cy - 34]])
        _wire(scene, [[x, cy + 34], [x, bot]])
        _junction(scene, (x, top))
        _junction(scene, (x, bot))
    k = int(rng.integers(2, min(4, n - 1) + 1))
    region = [130, top + 30, W - 30, bot - 30]
    for j, v in enumerate(unique_choice(rng, R_VALUES, k)):
        t = place_text_randomly(scene, rng, v, float(rng.uniform(13, 16)), font, region)
        if t is None:
            raise GenerationError("annotation placement")
        scene.annotations.append(Annotation(id=f"a{j}", value=v, label=t))
    _assign(scene, rng, reach=2.6 * sp)


def _grid(scene: Scene, rng, font: str) -> None:
    W, H = scene.canvas
    ncol = 3
    rows = [float(rng.uniform(220, 280)), float(rng.uniform(H - 260, H - 200))]
    xs = [round(260 + i * 240 + rng.uniform(-15, 15), 2) for i in range(ncol)]
    left, right = 110.0, W - 90.0
    types = ["R", "C"] * 3
    rng.shuffle(types)
    counts = {"R": 0, "C": 0}
    k_idx = 0
    for y in rows:
        y = round(y, 2)
        prev_x = left
        for x in xs:
            t = types[k_idx]
            k_idx += 1
            counts[t] += 1
            eid = f"{t}{counts[t]}"
            e = _resistor(eid, x, y, False, font) if t == "R" else _capacitor(eid, x, y, font)
            scene.entities.append(e)
            _wire(scene, [[prev_x, y], [x - 34, y]])
            prev_x = x + 34
        _wire(scene, [[prev_x, y], [right, y]])
    _wire(scene, [[left, rows[0]], [left, rows[1]]])
    _wire(scene, [[right, rows[0]], [right, rows[1]]])
    for p in ((left, rows[0]), (left, rows[1]), (right, rows[0]), (right, rows[1])):
        _junction(scene, p)
    t = text_at_bbox_corner("V+", 13, font, right=left - 8, cy=(rows[0] + rows[1]) / 2)
    scene.distractors.append({"type": "text", **t.__dict__, "role": "rail_text"})
    nR, nC = counts["R"], counts["C"]
    kR = int(rng.integers(1 if nR >= 2 else 0, max(1, min(3, nR - 1)) + 1)) if nR >= 2 else 0
    kC = int(rng.integers(1 if nC >= 2 else 0, max(1, min(3, nC - 1)) + 1)) if nC >= 2 else 0
    if kR + kC < 2 or (kR < 2 and kC < 2):
        # guarantee one type with >= 2 annotations so a same-type swap exists
        if nR >= 3:
            kR = 2
        elif nC >= 3:
            kC = 2
        else:
            raise GenerationError("no swappable type")
    vals = unique_choice(rng, R_VALUES, kR) + unique_choice(rng, C_VALUES, kC)
    region = [left + 15, 40, right - 10, H - 40]
    for j, v in enumerate(vals):
        t = place_text_randomly(scene, rng, v, float(rng.uniform(13, 16)), font, region)
        if t is None:
            raise GenerationError("annotation placement")
        scene.annotations.append(Annotation(id=f"a{j}", value=v, label=t))
    _assign(scene, rng, reach=520.0)


def generate(scene: Scene, rng: np.random.Generator) -> Scene:
    if scene.template == "ladder":
        _ladder(scene, rng, scene.font)
    elif scene.template == "grid":
        _grid(scene, rng, scene.font)
    else:
        raise ValueError(scene.template)
    return scene


def swap_compatible(scene: Scene, a1: str, a2: str) -> bool:
    e1, e2 = scene.target_of(a1), scene.target_of(a2)
    return scene.entity(e1).attrs["ctype"] == scene.entity(e2).attrs["ctype"]
