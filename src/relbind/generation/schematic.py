"""Family A — technical schematics.

elevation: repeated vertical column lines with grid bubbles; dimension annotations (e.g. 4'-0")
           tied to a column by a drafting leader; level lines + level text as distractors
           (level text looks like an annotation value but is not linked to any column).
plan:      column markers at grid intersections (dash-dot grid lines with bubbles as distractors);
           annotations tied to markers by leaders.
"""
from __future__ import annotations

import numpy as np

from .. import geometry as G
from ..schema import Annotation, Entity, Scene
from .common import (GenerationError, add_relation, dimension_values, leader_route, make_text,
                     place_text_randomly, text_at_bbox_corner, try_connect, unique_choice)

FAMILY = "schematic"
TEMPLATES = ("elevation", "plan")
TERMINATOR = "arrow"


def route(scene, ann, fixed_start, end, cls, rng, target):
    return leader_route(ann.label.bbox, end, cls, rng)


def _attach_candidates(scene: Scene, ent: Entity, ann: Annotation, rng, n: int = 6) -> list:
    acx, acy = G.bbox_center(ann.label.bbox)
    if ent.kind == "column":
        (x, y0), (_, y1) = ent.geometry["points"]
        out = []
        for _ in range(n):
            y = float(np.clip(acy + rng.uniform(-120, 120), y0 + 40, y1 - 25))
            out.append([round(x, 2), round(y, 2)])
        return out
    # marker: corners of the marker square (never the label corner), nearest-facing first
    b = ent.geometry["bbox"]
    corners = [[b[0], b[1]], [b[2], b[1]], [b[0], b[3]]]  # label sits at lower-right
    corners.sort(key=lambda p: G.dist(p, (acx, acy)))
    return [[round(p[0], 2), round(p[1], 2)] for p in corners]


def _assign(scene: Scene, rng, topology: str, reach: float) -> None:
    order = rng.permutation(len(scene.annotations))
    used: set[str] = set()
    for i in order:
        ann = scene.annotations[i]
        acx, acy = G.bbox_center(ann.label.bbox)
        cands = [e for e in scene.entities if e.id not in used
                 and G.dist(G.bbox_center(e.geometry.get("bbox", e.bbox)) if e.kind != "column" else
                            (e.geometry["points"][0][0], acy), (acx, acy)) <= reach]
        rng.shuffle(cands)
        conn = None
        for ent in cands:
            conn = try_connect(scene, ann, ent, _attach_candidates(scene, ent, ann, rng), route, topology, rng,
                               TERMINATOR, "leader", "bbox")
            if conn is not None:
                break
        if conn is None:
            raise GenerationError(f"could not connect {ann.id}")
        used.add(conn.target)
        add_relation(scene, conn)


def _id_scheme(rng, n: int) -> list[str]:
    s = int(rng.integers(0, 3))
    if s == 0:
        return [f"C{i + 1}" for i in range(n)]
    if s == 1:
        return [chr(ord("A") + i) for i in range(n)]
    return [f"G{i + 1}" for i in range(n)]


def _elevation(scene: Scene, rng, font: str, topology: str) -> None:
    W, H = scene.canvas
    n = int(rng.integers(4, 7))
    spacing = float(rng.uniform(140, 185))
    width = spacing * (n - 1)
    x0 = 240 + rng.uniform(0, max(1.0, (W - 50 - 240) - width))
    ids = _id_scheme(rng, n)
    y_top, y_bot, bub_y, bub_r = 100.0, H - 40.0, 72.0, 19.0
    for i in range(n):
        x = round(x0 + i * spacing, 2)
        lab = text_at_bbox_corner(ids[i], 15, font, cx=x, cy=bub_y)
        bub = [x - bub_r, bub_y - bub_r, x + bub_r, bub_y + bub_r]
        scene.entities.append(Entity(
            id=ids[i], kind="column", bbox=[x - bub_r, bub_y - bub_r, x + bub_r, y_bot],
            geometry={"type": "segment", "points": [[x, bub_y + bub_r], [x, y_bot]]}, crossable=True, label=lab,
            attrs={"solids": [bub], "draw": [
                {"type": "circle", "center": [x, bub_y], "r": bub_r, "fill": False, "width_role": "entity"},
                {"type": "line", "points": [[x, bub_y + bub_r], [x, y_bot]], "width_role": "entity", "dash": None}]}))
    # level lines + level text (confusable, unlinked values)
    n_lv = int(rng.integers(1, 3))
    ys = sorted(rng.choice(np.arange(200, H - 110, 10), size=n_lv, replace=False))
    for y in ys:
        y = float(y)
        scene.distractors.append({"type": "line", "points": [[150.0, y], [W - 20.0, y]], "width_role": "distractor",
                                  "dash": [10.0, 4.0, 2.0, 4.0], "role": "level_line"})
        ft, inch = int(rng.integers(0, 40)), int(rng.integers(0, 12))
        t = text_at_bbox_corner(f"EL. {100 + ft}'-{inch}\"", 12, font, left=12, bottom=y - 4)
        scene.distractors.append({"type": "text", **t.__dict__, "role": "level_text"})
    k = int(rng.integers(2, min(4, n - 1) + 1))
    vals = dimension_values(rng, k)
    region = [200, y_top + 40, W - 20, y_bot - 30]
    for j, v in enumerate(vals):
        t = place_text_randomly(scene, rng, v, float(rng.uniform(14, 18)), font, region)
        if t is None:
            raise GenerationError("annotation placement")
        scene.annotations.append(Annotation(id=f"a{j}", value=v, label=t))
    for note in unique_choice(rng, ["TYP.", "U.N.O.", "SIM.", "N.T.S."], int(rng.integers(0, 3))):
        t = place_text_randomly(scene, rng, note, 12, font, region)
        if t is not None:
            scene.distractors.append({"type": "text", **t.__dict__, "role": "note"})
    _assign(scene, rng, topology, reach=2.6 * spacing)


def _plan(scene: Scene, rng, font: str, topology: str) -> None:
    W, H = scene.canvas
    ngx, ngy = int(rng.integers(3, 5)), int(rng.integers(2, 4))
    sx = float(rng.uniform(190, min(260, (W - 260) / (ngx - 1))))
    sy = float(rng.uniform(190, min(250, (H - 260) / (ngy - 1))))
    gx0 = 170 + rng.uniform(0, max(1.0, W - 120 - 170 - sx * (ngx - 1)))
    gy0 = 170 + rng.uniform(0, max(1.0, H - 90 - 170 - sy * (ngy - 1)))
    xs = [round(gx0 + i * sx, 2) for i in range(ngx)]
    ys = [round(gy0 + j * sy, 2) for j in range(ngy)]
    letters = [chr(ord("A") + i) for i in range(ngx)]
    dash = [14.0, 4.0, 3.0, 4.0]
    for i, x in enumerate(xs):
        scene.distractors.append({"type": "circle", "center": [x, 70.0], "r": 17.0, "fill": False, "width_role": "distractor"})
        t = text_at_bbox_corner(letters[i], 14, font, cx=x, cy=70.0)
        scene.distractors.append({"type": "text", **t.__dict__, "role": "grid_bubble"})
        scene.distractors.append({"type": "line", "points": [[x, 87.0], [x, H - 30.0]], "width_role": "distractor", "dash": dash, "role": "grid"})
    for j, y in enumerate(ys):
        scene.distractors.append({"type": "circle", "center": [70.0, y], "r": 17.0, "fill": False, "width_role": "distractor"})
        t = text_at_bbox_corner(str(j + 1), 14, font, cx=70.0, cy=y)
        scene.distractors.append({"type": "text", **t.__dict__, "role": "grid_bubble"})
        scene.distractors.append({"type": "line", "points": [[87.0, y], [W - 30.0, y]], "width_role": "distractor", "dash": dash, "role": "grid"})
    hs = 9.0
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            eid = f"{letters[i]}{j + 1}"
            lab = text_at_bbox_corner(eid, 13, font, left=x + hs + 5, top=y + hs + 3)
            mb = [x - hs, y - hs, x + hs, y + hs]
            scene.entities.append(Entity(
                id=eid, kind="marker", bbox=[mb[0], mb[1], lab.bbox[2], lab.bbox[3]],
                geometry={"type": "rect", "bbox": mb}, crossable=False, label=lab,
                attrs={"draw": [{"type": "rect", "bbox": mb, "fill": True, "width_role": "entity"}]}))
    # grid lines pass through markers by design; exempt them from the connector-vs-stroke rule near
    # markers by not treating them as attached. Leaders attach at marker corners (>= hs from grid).
    n_ent = len(scene.entities)
    k = int(rng.integers(2, min(4, n_ent - 1) + 1))
    vals = dimension_values(rng, k)
    region = [95, 95, W - 35, H - 35]
    for j, v in enumerate(vals):
        t = place_text_randomly(scene, rng, v, float(rng.uniform(14, 17)), font, region)
        if t is None:
            raise GenerationError("annotation placement")
        scene.annotations.append(Annotation(id=f"a{j}", value=v, label=t))
    _assign(scene, rng, topology, reach=1.6 * max(sx, sy))


def generate(scene: Scene, rng: np.random.Generator) -> Scene:
    if scene.template == "elevation":
        _elevation(scene, rng, scene.font, scene.topology)
    elif scene.template == "plan":
        _plan(scene, rng, scene.font, scene.topology)
    else:
        raise ValueError(scene.template)
    return scene


def swap_compatible(scene: Scene, a1: str, a2: str) -> bool:
    return True
