"""Family B — flowcharts.

Annotations are branch labels written next to an exit vertex of a decision diamond; the labelled
arrow leaving that vertex ends on the target process box. The relation is
  branch label --(arrow from the adjacent exit)--> target box.
A counterfactual swap keeps labels, exits and arrowhead positions fixed and re-routes the arrows.

tb: decisions on top, a row of process boxes below (arrows enter box tops).
lr: decisions on the left, a column of process boxes on the right (arrows enter box left sides).
"""
from __future__ import annotations

import numpy as np

from .. import geometry as G
from ..schema import Annotation, Entity, Scene
from .common import (GenerationError, add_relation, make_text, text_at_bbox_corner, try_connect,
                     unique_choice)

FAMILY = "flowchart"
TEMPLATES = ("tb", "lr")
TERMINATOR = "arrow"

BRANCH_LABELS = ["yes", "no", "ok", "fail", "x < 0", "x ≥ 0", "retry", "timeout", "high", "low",
                 "valid", "invalid", "hit", "miss", "> 5", "≤ 5", "new", "seen"]
DECISIONS = ["valid?", "x > t?", "done?", "found?", "n < 3?", "cached?", "auth?", "empty?"]
STEP_WORDS = ["load", "parse", "retry", "log", "send", "store", "alert", "skip", "merge", "sort", "scan", "fetch"]


def _arrowhead(tip, prev, L=10.0, Hh=4.0):
    dx, dy = tip[0] - prev[0], tip[1] - prev[1]
    n = (dx * dx + dy * dy) ** 0.5 or 1.0
    ux, uy = dx / n, dy / n
    bx, by = tip[0] - ux * L, tip[1] - uy * L
    return {"type": "polygon", "points": [list(tip), [bx - uy * Hh, by + ux * Hh], [bx + uy * Hh, by - ux * Hh]],
            "fill": True, "width_role": "thin"}


def route(scene, ann, fixed_start, end, cls, rng, target):
    """Route classes for flowcharts (every route leaves its exit vertex outward first):
    straight : one oblique segment exit -> target.
    elbow    : orthogonal routing; a single bend when the target lies on the exit's side, otherwise an
               orthogonal detour (stub, turn, cross, turn into the target).
    zigzag   : oblique dog-leg (stub, one oblique waypoint, target) — held out as unseen topology."""
    s = [float(fixed_start[0]), float(fixed_start[1])]
    e = [float(end[0]), float(end[1])]
    tb = scene.template == "tb"
    if tb:  # exits are left/right vertices; arrows enter box tops
        d = -1.0 if ann.label.bbox[2] <= s[0] else 1.0
        stub = [s[0] + d * float(rng.uniform(18, 42)), s[1]]
    else:  # exits are top/bottom vertices; arrows enter box left sides
        d = -1.0 if ann.label.bbox[1] < s[1] - 2 and ann.label.bbox[3] <= s[1] + 1 else 1.0
        stub = [s[0], s[1] + d * float(rng.uniform(18, 42))]
    R = lambda p: [round(p[0], 2), round(p[1], 2)]
    if cls == "straight":
        return [R(s), R(e)]
    if cls == "elbow":
        if tb:
            if d * (e[0] - s[0]) >= 24:
                return [R(s), R([e[0], s[1]]), R(e)]
            my = float(rng.uniform(s[1] + 45, e[1] - 35))
            return [R(s), R(stub), R([stub[0], my]), R([e[0], my]), R(e)]
        if d * (e[1] - s[1]) >= 24:
            return [R(s), R([s[0], e[1]]), R(e)]
        mx = float(rng.uniform(s[0] + 60, e[0] - 40))
        return [R(s), R(stub), R([mx, stub[1]]), R([mx, e[1]]), R(e)]
    if cls == "zigzag":
        mx, my = (stub[0] + e[0]) / 2, (stub[1] + e[1]) / 2
        dx, dy = e[0] - stub[0], e[1] - stub[1]
        n = (dx * dx + dy * dy) ** 0.5 or 1.0
        off = float(rng.uniform(30, 80)) * (1 if rng.random() < 0.5 else -1)
        w = [mx - dy / n * off, my + dx / n * off]
        return [R(s), R(stub), R(w), R(e)]
    raise ValueError(cls)


def _box(scene, rng, eid, x0, y0, w, h, font):
    b = [round(x0, 2), round(y0, 2), round(x0 + w, 2), round(y0 + h, 2)]
    cx = (b[0] + b[2]) / 2
    lab = text_at_bbox_corner(eid, 15, font, cx=cx, top=b[1] + 6)
    sub = text_at_bbox_corner(str(rng.choice(STEP_WORDS)), 11, font, cx=cx, bottom=b[3] - 6)
    return Entity(id=eid, kind="box", bbox=b, geometry={"type": "rect", "bbox": b}, crossable=False, label=lab,
                  attrs={"draw": [{"type": "rect", "bbox": b, "fill": False, "width_role": "entity"},
                                  {"type": "text", **sub.__dict__}]})


def _diamond(scene, cx, cy, hw, hh, text, font, attached):
    pts = [[cx, cy - hh], [cx + hw, cy], [cx, cy + hh], [cx - hw, cy]]
    scene.distractors.append({"type": "polygon", "points": pts, "fill": False, "width_role": "entity",
                              "role": "decision", "attached": attached})
    t = text_at_bbox_corner(text, 12, font, cx=cx, cy=cy)
    scene.distractors.append({"type": "text", **t.__dict__, "role": "decision_text"})


def _terminal(scene, cx, cy, text, font):
    t = text_at_bbox_corner(text, 13, font, cx=cx, cy=cy)
    b = G.inflate(t.bbox, 9)
    scene.distractors.append({"type": "rect", "bbox": [round(v, 2) for v in b], "fill": False, "width_role": "distractor", "role": "terminal"})
    scene.distractors.append({"type": "text", **t.__dict__, "role": "terminal_text"})
    return b


def _flow_arrow(scene, pts):
    scene.distractors.append({"type": "line", "points": pts, "width_role": "distractor", "dash": None, "role": "flow"})
    scene.distractors.append({**_arrowhead(pts[-1], pts[-2]), "role": "flow_head"})


def _connect_all(scene, rng, exits, font):
    """exits: list of (annotation, start_point). Assign each to a distinct box and route."""
    used: set[str] = set()
    order = rng.permutation(len(exits))
    for i in order:
        ann, s = exits[i]
        boxes = [e for e in scene.entities if e.id not in used]
        rng.shuffle(boxes)
        conn = None
        for ent in boxes:
            conn = try_connect(scene, ann, ent, _attach_candidates(scene, ent, rng), route, scene.topology, rng,
                               TERMINATOR, "branch_arrow", "fixed", fixed_start=s)
            if conn is not None:
                break
        if conn is None:
            raise GenerationError(f"could not connect {ann.id}")
        used.add(conn.target)
        add_relation(scene, conn)


def _attach_candidates(scene, ent, rng, n=4):
    b = ent.geometry["bbox"]
    out = []
    for _ in range(n):
        if scene.template == "tb":
            out.append([round(float((b[0] + b[2]) / 2 + rng.uniform(-30, 30)), 2), b[1]])
        else:
            out.append([b[0], round(float((b[1] + b[3]) / 2 + rng.uniform(-11, 11)), 2)])
    return out


def _tb(scene: Scene, rng, font: str) -> None:
    W, H = scene.canvas
    nd = int(rng.integers(1, 3))
    hw, hh, dy = 64.0, 38.0, 200.0
    cxs = [W / 2 + rng.uniform(-60, 60)] if nd == 1 else [W * 0.3 + rng.uniform(-40, 40), W * 0.7 + rng.uniform(-40, 40)]
    nb = int(rng.integers(2 * nd + 1, 7))
    bw, bh = 104.0, 50.0
    gap = (W - 80 - nb * bw) / (nb - 1)
    by = float(rng.uniform(440, 500))
    for i in range(nb):
        scene.entities.append(_box(scene, rng, f"S{i + 1}", 40 + i * (bw + gap), by, bw, bh, font))
    labels = unique_choice(rng, BRANCH_LABELS, 2 * nd)
    st = _terminal(scene, W / 2, 40, "Start", font)
    exits = []
    # stagger decision heights so exits of different decisions are never collinear
    dys = [dy] if nd == 1 else list(rng.permutation([dy - 35.0, dy + float(rng.uniform(25, 55))]))
    for j, cx in enumerate(cxs):
        cx = round(cx, 2)
        dy = round(float(dys[j]), 2)
        aids = [f"a{2 * j}", f"a{2 * j + 1}"]
        _diamond(scene, cx, dy, hw, hh, str(rng.choice(DECISIONS)), font, aids)
        if nd == 1:
            _flow_arrow(scene, [[W / 2, st[3]], [W / 2, dy - hh]] if abs(cx - W / 2) < 1 else
                        [[W / 2, st[3]], [W / 2, 110.0], [cx, 110.0], [cx, dy - hh]])
        else:
            _flow_arrow(scene, [[W / 2, st[3]], [W / 2, 110.0], [cx, 110.0], [cx, dy - hh]])
        for side, aid, lab in (("left", aids[0], labels[2 * j]), ("right", aids[1], labels[2 * j + 1])):
            vx = cx - hw if side == "left" else cx + hw
            sz = 14.0
            t = (text_at_bbox_corner(lab, sz, font, right=vx - 6, bottom=dy - 6) if side == "left"
                 else text_at_bbox_corner(lab, sz, font, left=vx + 6, bottom=dy - 6))
            scene.annotations.append(Annotation(id=aid, value=lab, label=t))
            exits.append((scene.annotations[-1], [round(vx, 2), dy]))
    end = _terminal(scene, W / 2, H - 40, "End", font)
    for ei in rng.permutation(len(scene.entities))[: int(rng.integers(1, 3))]:
        b = scene.entities[int(ei)].geometry["bbox"]
        x = (b[0] + b[2]) / 2
        _flow_arrow(scene, [[x, b[3]], [x, H - 90.0], [W / 2 + (x - W / 2) * 0.05, H - 90.0],
                            [W / 2 + (x - W / 2) * 0.05, end[1]]])
    _connect_all(scene, rng, exits, font)


def _lr(scene: Scene, rng, font: str) -> None:
    W, H = scene.canvas
    nd = int(rng.integers(1, 3))
    hw, hh = 62.0, 40.0
    cx = float(rng.uniform(220, 260))
    cys = [H / 2 + rng.uniform(-40, 40)] if nd == 1 else [H * 0.3 + rng.uniform(-20, 20), H * 0.7 + rng.uniform(-20, 20)]
    nb = int(rng.integers(2 * nd + 1, 7))
    bw, bh = 116.0, 46.0
    gap = (H - 60 - nb * bh) / (nb - 1)
    bx = float(rng.uniform(640, 780))
    for i in range(nb):
        scene.entities.append(_box(scene, rng, f"S{i + 1}", bx, 30 + i * (bh + gap), bw, bh, font))
    labels = unique_choice(rng, BRANCH_LABELS, 2 * nd)
    st = _terminal(scene, 60, H / 2, "Start", font)
    exits = []
    # stagger decision columns so vertical exits of different decisions are never collinear
    cx_list = [cx] if nd == 1 else list(rng.permutation([cx, cx + float(rng.uniform(130, 170))]))
    for j, cy in enumerate(cys):
        cy = round(cy, 2)
        cx = round(float(cx_list[j]), 2)
        aids = [f"a{2 * j}", f"a{2 * j + 1}"]
        _diamond(scene, cx, cy, hw, hh, str(rng.choice(DECISIONS)), font, aids)
        xk = round(st[2] + 25 + 20 * j, 2)
        _flow_arrow(scene, [[st[2], H / 2], [xk, H / 2], [xk, cy], [cx - hw, cy]]
                    if abs(cy - H / 2) > 1 else [[st[2], H / 2], [cx - hw, cy]])
        for side, aid, lab in (("top", aids[0], labels[2 * j]), ("bottom", aids[1], labels[2 * j + 1])):
            vy = cy - hh if side == "top" else cy + hh
            # labels sit LEFT of the vertical exit stroke so arrows heading right never cross them
            t = (text_at_bbox_corner(lab, 14.0, font, right=cx - 7, bottom=vy - 5) if side == "top"
                 else text_at_bbox_corner(lab, 14.0, font, right=cx - 7, top=vy + 5))
            scene.annotations.append(Annotation(id=aid, value=lab, label=t))
            exits.append((scene.annotations[-1], [round(cx, 2), round(vy, 2)]))
    _connect_all(scene, rng, exits, font)


def generate(scene: Scene, rng: np.random.Generator) -> Scene:
    if scene.template == "tb":
        _tb(scene, rng, scene.font)
    elif scene.template == "lr":
        _lr(scene, rng, scene.font)
    else:
        raise ValueError(scene.template)
    return scene


def swap_compatible(scene: Scene, a1: str, a2: str) -> bool:
    return True
