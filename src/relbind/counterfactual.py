"""Counterfactual and control variants.

swap     (counterfactual): pick two annotations a1->e1, a2->e2 and re-route so a1->e2, a2->e1.
         Text, entities, annotation positions, distractors, terminator *positions and types* are
         unchanged; only the two connector paths differ. The correct answer flips.
reroute  (negative control): re-sample the path shape of every connector, keeping its endpoints'
         relation. Geometry of connectors changes, relations do not. Answers must not change.
style    (negative control): identical scene geometry, different rendering style (stroke widths,
         ink, terminator glyph, dash). Answers must not change.
"""
from __future__ import annotations

import itertools

import numpy as np

from .generation import FAMILIES
from .generation.common import GenerationError
from .render import Style
from .schema import Connector, Relation, Scene
from .verify import check_connector, verify_pair_structure, verify_scene


def _reroute_one(scene: Scene, old: Connector, new_target: str, end, rng, tries: int, others) -> Connector | None:
    mod = FAMILIES[scene.family]
    ann = scene.annotation(old.source)
    tgt = scene.entity(new_target)
    fixed = old.points[0] if old.start_mode == "fixed" else None
    for _ in range(tries):
        pts = mod.route(scene, ann, fixed, list(end), old.route_class, rng, tgt)
        if pts is None:
            continue
        c = Connector(id=old.id, source=old.source, target=new_target, points=pts, terminator=old.terminator,
                      route_class=old.route_class, kind=old.kind, start_mode=old.start_mode)
        if not check_connector(scene, c, others=others):
            return c
    return None


def make_swap(scene: Scene, rng: np.random.Generator, tries: int = 30,
              strict_only: bool = False) -> tuple[Scene, tuple[str, str]]:
    """Two passes: first with terminator positions held fixed (strict control); if no pair admits a
    valid strict swap, re-sample attach points on the same two targets (meta.attach_fixed=False).
    strict_only=True (v2 builds) never falls back: only shortcut-proof swaps are produced."""
    for strict in ((True,) if strict_only else (True, False)):
        try:
            return _make_swap(scene, rng, tries, strict)
        except GenerationError:
            continue
    raise GenerationError("no valid swap")


def _make_swap(scene: Scene, rng: np.random.Generator, tries: int, strict: bool):
    mod = FAMILIES[scene.family]
    anns = [a.id for a in scene.annotations]
    pairs = list(itertools.combinations(anns, 2))
    rng.shuffle(pairs)
    for a1, a2 in pairs:
        if not mod.swap_compatible(scene, a1, a2):
            continue
        cf = scene.copy()
        c1 = next(c for c in cf.connectors if c.source == a1)
        c2 = next(c for c in cf.connectors if c.source == a2)
        e1, e2 = c1.target, c2.target
        rest = [c for c in cf.connectors if c.source not in (a1, a2)]
        for _ in range(tries if strict else tries // 2):
            if strict:
                end1, end2 = c2.points[-1], c1.points[-1]
            else:
                end1 = _sample_attach(cf, e2, a1, rng)
                end2 = _sample_attach(cf, e1, a2, rng)
            n1 = _reroute_one(cf, c1, e2, end1, rng, 4, rest)
            if n1 is None:
                continue
            n2 = _reroute_one(cf, c2, e1, end2, rng, 8, rest + [n1])
            if n2 is None:
                continue
            cf.connectors = [n1 if c.id == c1.id else n2 if c.id == c2.id else c for c in cf.connectors]
            cf.relations = [Relation(annotation=c.source, entity=c.target, connector=c.id) for c in cf.connectors]
            issues = verify_scene(cf) + verify_pair_structure(scene, cf, "swap")
            if not strict:
                issues = [i for i in issues if i != "swap moved terminator positions"]
            if issues:
                continue
            cf.meta = {**scene.meta, "variant": "cf", "swapped": [a1, a2], "attach_fixed": strict}
            return cf, (a1, a2)
    raise GenerationError("no valid swap")


def _sample_attach(scene: Scene, eid: str, aid: str, rng):
    """A fresh attach point on entity `eid` suited to annotation `aid` (family-specific sampler)."""
    mod = FAMILIES[scene.family]
    ent, ann = scene.entity(eid), scene.annotation(aid)
    if scene.family == "flowchart":
        cands = mod._attach_candidates(scene, ent, rng, n=1)
    elif scene.family == "circuit":
        cands = mod._attach_candidates(ent, ann, rng, n=1)
    else:
        cands = mod._attach_candidates(scene, ent, ann, rng, n=3)
        return cands[int(rng.integers(0, len(cands)))]
    return cands[0]


def make_reroute(scene: Scene, rng: np.random.Generator, allowed: tuple[str, ...] = ("straight", "elbow", "zigzag"),
                 tries: int = 20) -> Scene:
    """Re-route every connector with a *different shape* to the same end point.

    Each connector first tries a different route class from `allowed` (the split's topologies, so a
    held-out topology never leaks into a training split), then a re-sample of its own class.
    At least one connector must change."""
    for _ in range(tries):
        rr = scene.copy()
        new = []
        ok = True
        for c in rr.connectors:
            others = [x for x in rr.connectors if x.id != c.id and x.id not in {n.id for n in new}] + new
            alts = [k for k in allowed if k != c.route_class]
            rng.shuffle(alts)
            n = None
            for cls in alts + [c.route_class]:
                n = _reroute_one(rr, Connector(**{**c.__dict__, "route_class": cls}), c.target, c.points[-1], rng, 8, others)
                if n is not None:
                    break
            if n is None:
                ok = False
                break
            new.append(n)
        if not ok:
            continue
        rr.connectors = new
        if verify_scene(rr) or verify_pair_structure(scene, rr, "reroute"):
            continue
        if all(a.points == b.points for a, b in zip(scene.connectors, rr.connectors)):
            continue
        rr.meta = {**scene.meta, "variant": "reroute"}
        return rr
    raise GenerationError("no valid reroute")


def base_style(family: str) -> Style:
    return Style(name="base")


def control_style(base: Style, rng: np.random.Generator, family: str) -> Style:
    """Meaning-preserving style change: strokes, ink, terminator glyph, dash."""
    term_alt = {"schematic": ["dot", "tick"], "flowchart": ["arrow"], "circuit": ["arrow", "tick"]}[family]
    inks = [(17, 17, 17), (20, 40, 120), (90, 30, 30), (30, 70, 40)]
    ink = inks[int(rng.integers(1, len(inks)))]
    return Style(
        ink=ink,
        bg=(int(rng.integers(244, 256)), int(rng.integers(244, 256)), int(rng.integers(236, 256))),
        w_entity=float(base.w_entity * rng.uniform(0.7, 1.4)),
        w_connector=float(base.w_connector * rng.uniform(0.75, 1.5)),
        w_distractor=float(base.w_distractor * rng.uniform(0.75, 1.4)),
        leader_dash=[6.0, 3.0] if (family != "flowchart" and rng.random() < 0.3) else None,
        terminator=str(rng.choice(term_alt)) if family != "flowchart" else None,
        arrow_len=float(rng.uniform(9, 13)),
        name="control",
    )
