"""Probe fitting code on synthetic features (no model). Gold links are planted in the features:
the gold entity's vector equals the annotation's vector, so a correct linear probe must recover them,
while the geometry-only control (positions identical in orig/cf) must score 0 strict-pair accuracy."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from relbind import probe  # noqa: E402


def _make_split(d, split, n_groups, rng, D=8):
    feats, images = {}, []
    for g in range(n_groups):
        ents = ["E0", "E1", "E2", "E3"]
        base = {e: rng.normal(size=D) for e in ents}
        for variant, gold in (("orig", {"a0": "E0", "a1": "E1"}), ("cf", {"a0": "E1", "a1": "E0"})):
            sid = f"{split}-{g}-{variant}"
            feats[f"{sid}|g"] = rng.normal(size=(1, D)).astype(np.float16)
            for e in ents:
                feats[f"{sid}|e|{e}"] = base[e][None].astype(np.float16)
            for a, e in gold.items():
                feats[f"{sid}|a|{a}"] = (base[e] + 0.05 * rng.normal(size=D))[None].astype(np.float16)
            images.append({"scene_id": sid, "group_id": f"{split}-{g}", "variant": variant, "split": split,
                           "crossing_cell": "leader_leader", "swapped": ["a0", "a1"], "annotations": ["a0", "a1"],
                           "entities": ents, "gold": gold, "ann_centers": {"a0": [10, 10], "a1": [10, 50]},
                           "ent_boxes": {e: [100 + 20 * i, 0, 110 + 20 * i, 90] for i, e in enumerate(ents)},
                           "canvas": [200, 100]})
    np.savez_compressed(d / f"{split}.npz", **feats)
    (d / f"{split}_meta.json").write_text(json.dumps({"layers": [0], "images": images}))


def test_probe_recovers_planted_links_and_geometry_control_is_zero(tmp_path):
    rng = np.random.default_rng(0)
    for split, n in (("train", 30), ("val", 10), ("test_id", 10)):
        _make_split(tmp_path, split, n, rng)
    out = tmp_path / "probe.json"
    probe.fit(SimpleNamespace(feats=str(tmp_path), test_splits=["test_id"], out=str(out), data=None))
    res = json.loads(out.read_text())
    t = res["test"]["test_id"]
    assert t["probe"]["pair_acc"] >= 0.9
    assert t["control_geometry_only"]["pair_acc"] == 0.0
