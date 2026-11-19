"""Ground-truth verification.

A scene is accepted only if its relation graph is *geometrically unambiguous*:
each connector starts at its own annotation, ends on its own target and nowhere near any other
entity, crosses other lines only at clear angles away from endpoints, never runs parallel and close
to other strokes, and never touches any text other than its own. These rules are what a human uses
to read a leader line; they are checked on the vector scene, not on model output.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from . import geometry as G
from .schema import Connector, Scene


@dataclass(frozen=True)
class VerifyConfig:
    attach_tol: float = 2.0
    start_gap_max: float = 7.0
    other_entity_margin: float = 16.0
    min_cross_angle: float = 30.0
    min_cross_from_end: float = 16.0
    min_parallel_gap: float = 7.0
    text_pad: float = 3.0
    min_connector_len: float = 28.0
    min_segment_len: float = 8.0
    own_label_margin: float = 10.0  # fixed-start: own label must be this much closer than any other start
    canvas_pad: float = 2.0


DEFAULT = VerifyConfig()


def _entity_segments(e) -> list[tuple]:
    g = e.geometry
    if g["type"] == "segment":
        return G.segments(g["points"])
    if g["type"] == "rect":
        return G.bbox_segments(g["bbox"])
    raise ValueError(g["type"])


def _entity_point_dist(e, p) -> float:
    g = e.geometry
    if g["type"] == "segment":
        return G.point_polyline_dist(p, g["points"])
    return G.point_bbox_dist(p, g["bbox"])


def _texts(scene: Scene) -> list[tuple[str, list[float]]]:
    """(owner, bbox) for every text in the scene. owner: 'a:<id>', 'e:<id>' or 'd'."""
    out = []
    for a in scene.annotations:
        out.append((f"a:{a.id}", a.label.bbox))
    for e in scene.entities:
        if e.label is not None:
            out.append((f"e:{e.id}", e.label.bbox))
        for p in e.attrs.get("draw", []):
            if p["type"] == "text":
                out.append((f"e:{e.id}", p["bbox"]))
    for p in scene.distractors:
        if p["type"] == "text":
            out.append(("d", p["bbox"]))
    return out


def _distractor_lines(scene: Scene, with_attached: bool = False):
    """Segments of free strokes (distractor lines, polygon outlines), grouped per primitive.

    with_attached=True yields (segments, attached_annotation_ids): a connector whose source is in
    `attached` legitimately starts on that primitive (e.g. a branch arrow leaving its diamond)."""
    out = []
    for p in scene.distractors:
        if p["type"] == "line":
            segs = G.segments(p["points"])
        elif p["type"] == "polygon":
            segs = G.segments(p["points"] + [p["points"][0]])
        elif p["type"] == "rect":
            segs = G.bbox_segments(p["bbox"])
        else:
            continue
        out.append((segs, set(p.get("attached", []))) if with_attached else segs)
    return out


def _distractor_solids(scene: Scene) -> list[list[float]]:
    out = []
    for p in scene.distractors:
        if p["type"] == "circle":
            c, r = p["center"], p["r"]
            out.append([c[0] - r, c[1] - r, c[0] + r, c[1] + r])
        elif p.get("solid_bbox") is not None:
            out.append(p["solid_bbox"])
    return out


def _cross_ok(p, q, r, s, endpoints: Iterable, cfg: VerifyConfig) -> str | None:
    """Rule for two strokes: either a clean crossing or a clear gap."""
    x = G.seg_intersection(p, q, r, s)
    if x is not None:
        if G.acute_angle_deg(p, q, r, s) < cfg.min_cross_angle:
            return "shallow crossing"
        for e in endpoints:
            if G.dist(x, e) < cfg.min_cross_from_end:
                return "crossing near endpoint"
        return None
    if G.seg_seg_dist(p, q, r, s) < cfg.min_parallel_gap:
        return "strokes too close"
    return None


def trim_end(points, d: float):
    """Polyline with the last `d` px removed."""
    pts = [list(p) for p in points]
    p, q = pts[-2], pts[-1]
    L = G.dist(p, q)
    if L <= d:
        return pts[:-1]
    t = (L - d) / L
    pts[-1] = [p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t]
    return pts


def trim_start(points, d: float):
    return list(reversed(trim_end(list(reversed(points)), d)))


def check_connector(scene: Scene, c: Connector, cfg: VerifyConfig = DEFAULT,
                    others: list[Connector] | None = None) -> list[str]:
    """Issues for one connector against the rest of the scene."""
    issues: list[str] = []
    W, H = scene.canvas
    ann = scene.annotation(c.source)
    tgt = scene.entity(c.target)
    pts = c.points
    start, end = pts[0], pts[-1]

    for p in pts:
        if not (cfg.canvas_pad <= p[0] <= W - cfg.canvas_pad and cfg.canvas_pad <= p[1] <= H - cfg.canvas_pad):
            issues.append(f"{c.id}: point outside canvas")
            break
    if G.polyline_length(pts) < cfg.min_connector_len:
        issues.append(f"{c.id}: too short")
    if any(G.dist(p, q) < cfg.min_segment_len for p, q in G.segments(pts)):
        issues.append(f"{c.id}: tiny segment")

    # -- start belongs to its annotation
    d_own = G.point_bbox_dist(start, ann.label.bbox)
    if d_own > (cfg.start_gap_max if c.start_mode == "bbox" else 16.0):
        issues.append(f"{c.id}: start not at annotation")
    for a2 in scene.annotations:
        if a2.id != ann.id and G.point_bbox_dist(start, a2.label.bbox) < d_own + cfg.own_label_margin:
            issues.append(f"{c.id}: start ambiguous between annotations")

    # -- end lies on target and far from all other entities
    if _entity_point_dist(tgt, end) > cfg.attach_tol:
        issues.append(f"{c.id}: end not on target")
    for e in scene.entities:
        if e.id == tgt.id:
            continue
        de = _entity_point_dist(e, end) if e.crossable else G.point_bbox_dist(end, e.bbox)
        if de < cfg.other_entity_margin:
            issues.append(f"{c.id}: end near non-target {e.id}")

    segs = G.segments(pts)
    body = trim_end(pts, 6.0)

    # -- entities
    for e in scene.entities:
        if e.crossable:
            esegs = _entity_segments(e)
            if e.id == tgt.id:
                # the last 16 px approach the target at >= min_cross_angle, so the remaining body
                # must stay clearly off the target line (>= 16*sin(30deg) - slack)
                for p, q in G.segments(trim_end(pts, 16.0)):
                    for r, s in esegs:
                        if G.seg_seg_dist(p, q, r, s) < 4.0:
                            issues.append(f"{c.id}: touches target before end")
                last = segs[-1]
                if min(G.acute_angle_deg(last[0], last[1], r, s) for r, s in esegs) < cfg.min_cross_angle:
                    issues.append(f"{c.id}: tangential approach to target")
            else:
                for p, q in segs:
                    for r, s in esegs:
                        why = _cross_ok(p, q, r, s, [start, end] + list(e.geometry["points"]), cfg)
                        if why:
                            issues.append(f"{c.id} x {e.id}: {why}")
            if e.label is not None and G.polyline_intersects_bbox(pts, G.inflate(e.label.bbox, cfg.text_pad)):
                issues.append(f"{c.id}: crosses label of {e.id}")
            for b in e.attrs.get("solids", []):
                if G.polyline_intersects_bbox(pts, G.inflate(b, cfg.text_pad)):
                    issues.append(f"{c.id}: crosses symbol of {e.id}")
        else:
            box = e.bbox if e.id != tgt.id else e.geometry["bbox"]
            poly = pts if e.id != tgt.id else body
            if e.id != tgt.id:
                box = G.inflate(box, cfg.text_pad)
            if len(poly) >= 2 and G.polyline_intersects_bbox(poly, box):
                issues.append(f"{c.id}: passes through {e.id}")
            if e.id == tgt.id and e.label is not None and G.polyline_intersects_bbox(pts, G.inflate(e.label.bbox, 1.0)):
                issues.append(f"{c.id}: crosses own target label")

    # -- other connectors
    for c2 in (others if others is not None else scene.connectors):
        if c2.id == c.id:
            continue
        for p, q in segs:
            for r, s in G.segments(c2.points):
                why = _cross_ok(p, q, r, s, [start, end, c2.points[0], c2.points[-1]], cfg)
                if why:
                    issues.append(f"{c.id} x {c2.id}: {why}")

    # -- distractor strokes and solids
    for dsegs, attached in _distractor_lines(scene, with_attached=True):
        csegs = G.segments(trim_start(pts, 12.0)) if c.source in attached else segs
        for p, q in csegs:
            for r, s in dsegs:
                why = _cross_ok(p, q, r, s, [start, end], cfg)
                if why:
                    issues.append(f"{c.id} x distractor: {why}")
    for b in _distractor_solids(scene):
        if G.polyline_intersects_bbox(pts, G.inflate(b, cfg.text_pad)):
            issues.append(f"{c.id}: crosses distractor symbol")

    # -- text
    for owner, b in _texts(scene):
        if owner == f"a:{ann.id}":
            if G.polyline_intersects_bbox(trim_start(pts, 1.0), b):
                issues.append(f"{c.id}: crosses own annotation")
            continue
        if owner == f"e:{tgt.id}":
            continue  # handled above
        if G.polyline_intersects_bbox(pts, G.inflate(b, cfg.text_pad)):
            issues.append(f"{c.id}: crosses text {owner}")
    return issues


def check_texts(scene: Scene, cfg: VerifyConfig = DEFAULT) -> list[str]:
    issues = []
    texts = _texts(scene)
    W, H = scene.canvas
    for i, (o1, b1) in enumerate(texts):
        if b1[0] < 0 or b1[1] < 0 or b1[2] > W or b1[3] > H:
            issues.append(f"text {o1} outside canvas")
        for o2, b2 in texts[i + 1:]:
            if G.bbox_overlap(G.inflate(b1, 2.0), b2):
                issues.append(f"text overlap {o1}/{o2}")
    for owner, b in texts:
        for e in scene.entities:
            if owner == f"e:{e.id}":
                continue
            if e.crossable:
                if any(G.seg_intersects_bbox(p, q, G.inflate(b, cfg.text_pad)) for p, q in _entity_segments(e)):
                    issues.append(f"text {owner} on entity {e.id}")
            elif G.bbox_overlap(G.inflate(b, cfg.text_pad), e.bbox):
                issues.append(f"text {owner} on entity {e.id}")
        if owner.startswith("a:"):
            for dsegs in _distractor_lines(scene):
                if any(G.seg_intersects_bbox(p, q, G.inflate(b, cfg.text_pad)) for p, q in dsegs):
                    issues.append(f"text {owner} on distractor stroke")
            for sb in _distractor_solids(scene):
                if G.bbox_overlap(G.inflate(b, cfg.text_pad), sb):
                    issues.append(f"text {owner} on distractor symbol")
    return issues


def check_graph(scene: Scene) -> list[str]:
    issues = []
    ids = [e.id for e in scene.entities]
    if len(set(ids)) != len(ids):
        issues.append("duplicate entity ids")
    for e in scene.entities:
        if e.label is None or e.label.text != e.id:
            issues.append(f"entity {e.id} label does not display its id")
    vals = [a.value for a in scene.annotations]
    if len(set(vals)) != len(vals):
        issues.append("duplicate annotation values")
    for a in scene.annotations:
        if a.label.text != a.value:
            issues.append(f"annotation {a.id} text != value")
        rs = [r for r in scene.relations if r.annotation == a.id]
        if len(rs) != 1:
            issues.append(f"annotation {a.id} has {len(rs)} relations")
    for e in scene.entities:
        if len(scene.annotations_of(e.id)) > 1:
            issues.append(f"entity {e.id} has >1 annotation")
    cons = {c.id: c for c in scene.connectors}
    for r in scene.relations:
        c = cons.get(r.connector)
        if c is None or c.source != r.annotation or c.target != r.entity:
            issues.append(f"relation {r} inconsistent with connector")
    if len(cons) != len(scene.relations):
        issues.append("connector/relation count mismatch")
    return issues


def verify_scene(scene: Scene, cfg: VerifyConfig = DEFAULT) -> list[str]:
    issues = check_graph(scene) + check_texts(scene, cfg)
    for c in scene.connectors:
        issues += check_connector(scene, c, cfg)
    return issues


# ----------------------------------------------------------------------------- pairs

def verify_pair_structure(a: Scene, b: Scene, kind: str) -> list[str]:
    """Structural invariants of a counterfactual ('swap') or control ('style', 'reroute') pair."""
    issues = []
    if [e.__dict__ for e in a.entities] != [e.__dict__ for e in b.entities]:
        issues.append("entities differ")
    if [x.__dict__ for x in a.annotations] != [x.__dict__ for x in b.annotations]:
        issues.append("annotations differ")
    if a.distractors != b.distractors:
        issues.append("distractors differ")
    ea, eb = a.edge_set(), b.edge_set()
    if kind == "swap":
        changed = ea ^ eb
        if len(ea - eb) != 2 or len(eb - ea) != 2:
            issues.append(f"swap must change exactly 2 edges, changed {len(changed)}")
        ends_a = sorted(tuple(c.points[-1]) for c in a.connectors)
        ends_b = sorted(tuple(c.points[-1]) for c in b.connectors)
        if ends_a != ends_b:
            issues.append("swap moved terminator positions")
        terms_a = sorted(c.terminator for c in a.connectors)
        if terms_a != sorted(c.terminator for c in b.connectors):
            issues.append("swap changed terminator types")
    else:
        if ea != eb:
            issues.append(f"{kind} control changed relations")
    return issues


def pixel_diff_outside_mask(img_a, img_b, mask, thresh: int = 0) -> int:
    """Number of pixels that differ by > thresh outside `mask` (bool array, True = allowed to differ)."""
    A = np.asarray(img_a, dtype=np.int16)
    B = np.asarray(img_b, dtype=np.int16)
    diff = np.abs(A - B).max(axis=-1) > thresh
    return int((diff & ~mask).sum())
