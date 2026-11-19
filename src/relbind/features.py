"""Per-query difficulty features used for stratified analysis and shortcut audits."""
from __future__ import annotations

from . import geometry as G
from .schema import Scene
from .verify import _distractor_lines, _texts


def entity_distance(scene: Scene, e, p) -> float:
    g = e.geometry
    if g["type"] == "segment":
        return G.point_polyline_dist(p, g["points"])
    return G.point_bbox_dist(p, g["bbox"])


def proximity_ranking(scene: Scene, ann_id: str) -> list[str]:
    """Entities ordered by distance from the annotation's bbox centre (proximity prior)."""
    p = G.bbox_center(scene.annotation(ann_id).label.bbox)
    return [e.id for e in sorted(scene.entities, key=lambda e: (entity_distance(scene, e, p), e.id))]


def _count_crossings(poly, segs_list) -> int:
    n = 0
    for p, q in G.segments(poly):
        for segs in segs_list:
            for r, s in segs:
                if G.seg_intersection(p, q, r, s) is not None:
                    n += 1
    return n


def query_features(scene: Scene, ann_id: str) -> dict:
    c = next(c for c in scene.connectors if c.source == ann_id)
    tgt = c.target
    others = [G.segments(x.points) for x in scene.connectors if x.id != c.id]
    ents = [G.segments(e.geometry["points"]) for e in scene.entities if e.crossable and e.id != tgt]
    dl = _distractor_lines(scene)
    rank = proximity_ranking(scene, ann_id)
    p = G.bbox_center(scene.annotation(ann_id).label.bbox)
    W, H = scene.canvas
    texts = _texts(scene)
    return {
        "connector_length": round(G.polyline_length(c.points), 1),
        "n_bends": len(c.points) - 2,
        "cross_connectors": _count_crossings(c.points, others),
        "cross_entities": _count_crossings(c.points, ents),
        "cross_distractors": _count_crossings(c.points, dl),
        "target_proximity_rank": rank.index(tgt),
        "nearest_is_target": rank[0] == tgt,
        "ann_target_distance": round(entity_distance(scene, scene.entity(tgt), p), 1),
        "n_entities": len(scene.entities),
        "n_annotations": len(scene.annotations),
        "n_texts": len(texts),
        "text_density": round(sum((b[2] - b[0]) * (b[3] - b[1]) for _, b in texts) / (W * H), 4),
        "n_distractor_strokes": len(dl),
    }
