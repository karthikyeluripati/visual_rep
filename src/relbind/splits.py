"""Split definitions and leakage audits.

Splits are defined *before* generation (configs/*.json). Each split restricts which family,
template, topology (route class) and font may appear. Every counterfactual group (base scene + its
counterfactual + controls) is generated inside exactly one split from a split-specific seed stream,
so pair members can never straddle splits.
"""
from __future__ import annotations

import itertools
from collections import defaultdict

import numpy as np
from PIL import Image


def dhash(img: Image.Image, size: int = 16) -> np.ndarray:
    g = np.asarray(img.convert("L").resize((size + 1, size), Image.BILINEAR), dtype=np.int16)
    return (g[:, 1:] > g[:, :-1]).flatten()


def leakage_report(examples: list[dict], split_specs: dict, image_hashes: dict[str, str],
                   dhashes: dict[str, np.ndarray], seeds: dict[str, set], signatures: dict[str, str],
                   near_dup_bits: int = 4) -> dict:
    """examples: all example dicts; image_hashes: scene_id -> sha256; dhashes: base scene_id -> bits;
    seeds: split -> set(base seeds); signatures: base scene_id -> relation signature."""
    rep: dict = {"errors": [], "warnings": []}

    group_splits = defaultdict(set)
    scene_split = {}
    for ex in examples:
        group_splits[ex["group_id"]].add(ex["split"])
        scene_split[ex["scene_id"]] = ex["split"]
        spec = split_specs[ex["split"]]
        for key, attr in (("families", "family"), ("topologies", "topology"), ("fonts", "font")):
            if ex[attr] not in spec[key]:
                rep["errors"].append(f"{ex['example_id']}: {attr}={ex[attr]} not allowed in {ex['split']}")
        if ex["template"] not in spec["templates"][ex["family"]]:
            rep["errors"].append(f"{ex['example_id']}: template {ex['template']} not allowed in {ex['split']}")
    straddle = [g for g, s in group_splits.items() if len(s) > 1]
    if straddle:
        rep["errors"].append(f"{len(straddle)} groups straddle splits")

    by_hash = defaultdict(set)
    for sid, h in image_hashes.items():
        by_hash[h].add(scene_split.get(sid))
    dup = [h for h, s in by_hash.items() if len(s) > 1]
    if dup:
        rep["errors"].append(f"{len(dup)} identical images shared across splits")

    for (s1, a), (s2, b) in itertools.combinations(seeds.items(), 2):
        if a & b:
            rep["errors"].append(f"seed overlap between {s1} and {s2}")

    # near-duplicate base images across splits
    ids = list(dhashes)
    if ids:
        M = np.stack([dhashes[i] for i in ids])
        sp = np.array([scene_split[i] for i in ids])
        dist = (M[:, None, :] != M[None, :, :]).sum(-1)
        cross = sp[:, None] != sp[None, :]
        iu = np.triu_indices(len(ids), 1)
        dd = dist[iu][cross[iu]]
        rep["near_duplicate_pairs"] = int((dd <= near_dup_bits).sum())
        rep["min_cross_split_dhash_distance"] = int(dd.min()) if dd.size else None
        if rep["near_duplicate_pairs"]:
            rep["warnings"].append(f"{rep['near_duplicate_pairs']} cross-split base-image pairs with dHash distance <= {near_dup_bits}")

    by_sig = defaultdict(set)
    for sid, sig in signatures.items():
        by_sig[sig].add(scene_split[sid])
    rep["shared_relation_signatures"] = sum(1 for s in by_sig.values() if len(s) > 1)
    if rep["shared_relation_signatures"]:
        rep["warnings"].append(f"{rep['shared_relation_signatures']} relation signatures (value->entity maps) recur across splits")
    rep["ok"] = not rep["errors"]
    return rep
