"""Planar geometry helpers.

Coordinate convention (used everywhere in RelBind):
  * canonical canvas pixels, origin at the top-left corner,
  * x grows to the right, y grows downward,
  * a bounding box is [x0, y0, x1, y1] with x0 <= x1 and y0 <= y1,
  * a polyline is a list of [x, y] points.
"""
from __future__ import annotations

import math
from typing import Iterable, Sequence

Point = Sequence[float]
BBox = Sequence[float]

EPS = 1e-9


def sub(a: Point, b: Point) -> tuple[float, float]:
    return (a[0] - b[0], a[1] - b[1])


def dot(a: Point, b: Point) -> float:
    return a[0] * b[0] + a[1] * b[1]


def cross(a: Point, b: Point) -> float:
    return a[0] * b[1] - a[1] * b[0]


def dist(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def segments(polyline: Sequence[Point]) -> list[tuple[Point, Point]]:
    return [(polyline[i], polyline[i + 1]) for i in range(len(polyline) - 1)]


def polyline_length(polyline: Sequence[Point]) -> float:
    return sum(dist(p, q) for p, q in segments(polyline))


def point_seg_dist(p: Point, a: Point, b: Point) -> float:
    ab = sub(b, a)
    L2 = dot(ab, ab)
    if L2 < EPS:
        return dist(p, a)
    t = max(0.0, min(1.0, dot(sub(p, a), ab) / L2))
    proj = (a[0] + t * ab[0], a[1] + t * ab[1])
    return dist(p, proj)


def seg_intersection(p1: Point, p2: Point, q1: Point, q2: Point) -> tuple[float, float] | None:
    """Intersection point of closed segments p1p2 and q1q2, or None.

    Collinear overlapping segments return None (callers treat them via distance checks)."""
    r = sub(p2, p1)
    s = sub(q2, q1)
    denom = cross(r, s)
    if abs(denom) < EPS:
        return None
    qp = sub(q1, p1)
    t = cross(qp, s) / denom
    u = cross(qp, r) / denom
    if -EPS <= t <= 1 + EPS and -EPS <= u <= 1 + EPS:
        return (p1[0] + t * r[0], p1[1] + t * r[1])
    return None


def seg_seg_dist(p1: Point, p2: Point, q1: Point, q2: Point) -> float:
    if seg_intersection(p1, p2, q1, q2) is not None:
        return 0.0
    return min(
        point_seg_dist(p1, q1, q2),
        point_seg_dist(p2, q1, q2),
        point_seg_dist(q1, p1, p2),
        point_seg_dist(q2, p1, p2),
    )


def acute_angle_deg(p1: Point, p2: Point, q1: Point, q2: Point) -> float:
    r = sub(p2, p1)
    s = sub(q2, q1)
    nr = math.hypot(*r)
    ns = math.hypot(*s)
    if nr < EPS or ns < EPS:
        return 0.0
    c = abs(dot(r, s)) / (nr * ns)
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def polyline_dist(a: Sequence[Point], b: Sequence[Point]) -> float:
    return min(seg_seg_dist(p, q, r, s) for p, q in segments(a) for r, s in segments(b))


def point_polyline_dist(p: Point, poly: Sequence[Point]) -> float:
    if len(poly) == 1:
        return dist(p, poly[0])
    return min(point_seg_dist(p, a, b) for a, b in segments(poly))


# ---------------------------------------------------------------- boxes

def bbox_of_points(points: Iterable[Point]) -> list[float]:
    pts = list(points)
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return [min(xs), min(ys), max(xs), max(ys)]


def inflate(b: BBox, d: float) -> list[float]:
    return [b[0] - d, b[1] - d, b[2] + d, b[3] + d]


def bbox_center(b: BBox) -> tuple[float, float]:
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


def bbox_overlap(a: BBox, b: BBox) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def bbox_iou(a: BBox, b: BBox) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def point_in_bbox(p: Point, b: BBox) -> bool:
    return b[0] <= p[0] <= b[2] and b[1] <= p[1] <= b[3]


def bbox_corners(b: BBox) -> list[tuple[float, float]]:
    return [(b[0], b[1]), (b[2], b[1]), (b[2], b[3]), (b[0], b[3])]


def bbox_segments(b: BBox) -> list[tuple[Point, Point]]:
    c = bbox_corners(b)
    return [(c[i], c[(i + 1) % 4]) for i in range(4)]


def seg_intersects_bbox(p: Point, q: Point, b: BBox) -> bool:
    if point_in_bbox(p, b) or point_in_bbox(q, b):
        return True
    return any(seg_intersection(p, q, r, s) is not None for r, s in bbox_segments(b))


def polyline_intersects_bbox(poly: Sequence[Point], b: BBox) -> bool:
    return any(seg_intersects_bbox(p, q, b) for p, q in segments(poly))


def point_bbox_dist(p: Point, b: BBox) -> float:
    dx = max(b[0] - p[0], 0.0, p[0] - b[2])
    dy = max(b[1] - p[1], 0.0, p[1] - b[3])
    return math.hypot(dx, dy)


def bbox_exit_point(b: BBox, toward: Point, gap: float = 0.0) -> tuple[float, float]:
    """Point where the ray from the bbox centre toward `toward` leaves the box, pushed out by `gap`."""
    cx, cy = bbox_center(b)
    dx, dy = toward[0] - cx, toward[1] - cy
    n = math.hypot(dx, dy)
    if n < EPS:
        return (cx, b[1] - gap)
    hw = (b[2] - b[0]) / 2.0
    hh = (b[3] - b[1]) / 2.0
    tx = hw / abs(dx) if abs(dx) > EPS else math.inf
    ty = hh / abs(dy) if abs(dy) > EPS else math.inf
    t = min(tx, ty)
    ux, uy = dx / n, dy / n
    return (cx + dx * t + ux * gap, cy + dy * t + uy * gap)


def side_midpoint(b: BBox, side: str, gap: float = 0.0) -> tuple[float, float]:
    cx, cy = bbox_center(b)
    if side == "left":
        return (b[0] - gap, cy)
    if side == "right":
        return (b[2] + gap, cy)
    if side == "top":
        return (cx, b[1] - gap)
    if side == "bottom":
        return (cx, b[3] + gap)
    raise ValueError(side)


def round_pt(p: Point, nd: int = 2) -> list[float]:
    return [round(float(p[0]), nd), round(float(p[1]), nd)]
