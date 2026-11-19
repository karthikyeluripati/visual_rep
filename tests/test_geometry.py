from relbind import geometry as G


def test_intersection_and_angles():
    assert G.seg_intersection((0, 0), (10, 10), (0, 10), (10, 0)) == (5.0, 5.0)
    assert G.seg_intersection((0, 0), (1, 0), (0, 1), (1, 1)) is None
    assert abs(G.acute_angle_deg((0, 0), (1, 0), (0, 0), (0, 1)) - 90) < 1e-6
    assert abs(G.acute_angle_deg((0, 0), (1, 0), (0, 0), (-1, 0))) < 1e-6


def test_distances():
    assert G.point_seg_dist((0, 5), (-1, 0), (1, 0)) == 5
    assert G.seg_seg_dist((0, 0), (10, 0), (0, 3), (10, 3)) == 3
    assert G.point_bbox_dist((5, 5), [0, 0, 10, 10]) == 0
    assert G.point_bbox_dist((13, 14), [0, 0, 10, 10]) == 5


def test_bbox_ops():
    b = [0, 0, 10, 10]
    assert G.seg_intersects_bbox((-5, 5), (15, 5), b)
    assert not G.seg_intersects_bbox((-5, -5), (-1, 20), b)
    assert G.bbox_iou(b, b) == 1.0
    x, y = G.bbox_exit_point(b, (100, 5), gap=2)
    assert abs(x - 12) < 1e-6 and abs(y - 5) < 1e-6
