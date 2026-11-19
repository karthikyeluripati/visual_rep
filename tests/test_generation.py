"""Generator / verifier / counterfactual invariants on every family x template x topology."""
import itertools

import numpy as np
import pytest

from relbind.build import changed_mask
from relbind.counterfactual import make_reroute, make_swap
from relbind.generation import FAMILIES, generate_scene
from relbind.generation.common import GenerationError
from relbind.render import Style, render_png
from relbind.schema import Scene
from relbind.verify import pixel_diff_outside_mask, verify_pair_structure, verify_scene

CASES = [(f, t, topo) for f, m in FAMILIES.items() for t in m.TEMPLATES for topo in ("straight", "elbow", "zigzag")]


def _group(fam, tpl, topo, seed=3):
    for k in range(40):  # some configs (flowchart/lr/straight, circuit/grid) have ~15-25% swap yield
        s, _ = generate_scene(fam, tpl, topo, "sans", seed + 1000 * k, "t")
        try:
            rng = np.random.default_rng(seed + k)
            cf, sw = make_swap(s, rng)
            return s, cf, sw, rng
        except GenerationError:
            continue
    pytest.fail(f"no swappable scene for {fam}/{tpl}/{topo}")


@pytest.mark.parametrize("fam,tpl,topo", CASES)
def test_scene_valid_and_deterministic(fam, tpl, topo):
    a, _ = generate_scene(fam, tpl, topo, "sans", 11, "x")
    b, _ = generate_scene(fam, tpl, topo, "sans", 11, "x")
    assert a.to_json() == b.to_json()
    assert verify_scene(a) == []
    assert len(a.annotations) >= 2
    assert all(c.route_class == topo for c in a.connectors)
    # round trip
    assert Scene.from_dict(a.to_dict()).to_json() == a.to_json()


@pytest.mark.parametrize("fam,tpl,topo", CASES)
def test_swap_counterfactual(fam, tpl, topo):
    s, cf, (a1, a2), rng = _group(fam, tpl, topo)
    assert verify_scene(cf) == []
    assert s.target_of(a1) == cf.target_of(a2) and s.target_of(a2) == cf.target_of(a1)
    issues = verify_pair_structure(s, cf, "swap")
    if not cf.meta["attach_fixed"]:
        issues = [i for i in issues if i != "swap moved terminator positions"]
    assert issues == []
    # all other relations untouched
    others = {a.id for a in s.annotations} - {a1, a2}
    assert all(s.target_of(a) == cf.target_of(a) for a in others)
    # pixel control: differences confined to the swapped connectors
    st = Style()
    ia, ib = render_png(s, st), render_png(cf, st)
    conns = {c.id for c in s.connectors if c.source in (a1, a2)}
    assert pixel_diff_outside_mask(ia, ib, changed_mask([s, cf], conns, st, 1.0)) == 0
    assert np.asarray(ia).tobytes() != np.asarray(ib).tobytes()


@pytest.mark.parametrize("fam,tpl", [(f, t) for f, m in FAMILIES.items() for t in m.TEMPLATES])
def test_reroute_control_preserves_relations(fam, tpl):
    s, _, _, rng = _group(fam, tpl, "elbow")
    rr = make_reroute(s, rng)
    assert rr.edge_set() == s.edge_set()
    assert verify_scene(rr) == []


def test_render_deterministic():
    s, _ = generate_scene("schematic", "elevation", "straight", "mono", 5, "x")
    a = np.asarray(render_png(s, Style()))
    b = np.asarray(render_png(s, Style()))
    assert a.tobytes() == b.tobytes()
    assert a.shape == (s.canvas[1], s.canvas[0], 3)


def test_verifier_rejects_ambiguous_endpoint():
    s, _ = generate_scene("schematic", "elevation", "straight", "sans", 21, "x")
    bad = s.copy()
    c = bad.connectors[0]
    other = next(e for e in bad.entities if e.id != c.target)
    # move the end point 5 px from a different column: must be rejected
    x = other.geometry["points"][0][0]
    c.points[-1] = [x + 5, c.points[-1][1]]
    assert verify_scene(bad) != []


def test_verifier_rejects_text_collision():
    s, _ = generate_scene("flowchart", "tb", "elbow", "sans", 4, "x")
    bad = s.copy()
    a0, a1 = bad.annotations[0], bad.annotations[1]
    a1.label.bbox = list(a0.label.bbox)
    assert any("text overlap" in i for i in verify_scene(bad))
