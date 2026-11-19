"""Phase-1 failure analysis: error taxonomy + Gate-2 decision (research/experimental_protocol.md §3-4).

    python -m relbind.analysis --data /workspace/data/diag_v1 --diag experiments/phase1_diag \
        --heuristics results/b0_diag --out results/phase1

Error taxonomy (precedence order) for wrong a2e answers on orig / cf images:
  parse       prediction is not one of the candidate entity ids
  perception  the model misread the queried annotation text or the gold target label (P_read probes)
  stuck       cf image: prediction equals the ORIGINAL gold target (did not follow the rewiring)
  proximity   prediction is the entity nearest to the annotation
  crossing    prediction is an entity whose stroke the gold connector crosses
  other       none of the above
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from . import geometry as G
from .baselines.heuristics import load_scene
from .features import proximity_ranking
from .metrics import is_correct, load_examples, load_jsonl, normalize

CATS = ("parse", "perception", "stuck", "proximity", "crossing", "other")


def probe_kind(pid: str) -> tuple[str, str]:
    """'<ex>#read_ann_crop' -> ('ann', 'crop')."""
    _, k, m = pid.split("#")[1].split("_")
    return k, m


def read_tables(probe_rows, mode: str = "crop"):
    """(group_id, 'ann'|'ent', gold text) -> read correctly? Uses the pure-recognition crop probe."""
    t = {}
    for p in probe_rows:
        k, m = probe_kind(p["example_id"])
        if m != mode:
            continue
        gid = p["example_id"].split("#")[0].split("-a2e-")[0]
        t[(gid, k, normalize(p["gold"]))] = p["correct"]
    return t


def crossed_entities(scene, aid):
    c = next(c for c in scene.connectors if c.source == aid)
    out = set()
    for e in scene.entities:
        if e.id == c.target:
            continue
        if e.crossable:
            if any(G.seg_intersection(p, q, r, s) for p, q in G.segments(c.points) for r, s in G.segments(e.geometry["points"])):
                out.add(e.id)
        elif G.polyline_intersects_bbox(c.points, G.inflate(e.bbox, 6)):
            out.add(e.id)
    return out


def taxonomy(data: Path, examples, preds, reads, split="test_id"):
    pm = {p["example_id"]: p for p in preds}
    orig_gold = {e["query_id"]: e["answer"] for e in examples if e["variant"] == "orig"}
    counts = defaultdict(Counter)
    samples = defaultdict(list)
    for ex in examples:
        if ex["split"] != split or ex["query_type"] != "a2e" or ex["variant"] not in ("orig", "cf"):
            continue
        p = pm.get(ex["example_id"])
        if p is None:
            continue
        v = ex["variant"]
        counts[v]["n"] += 1
        if is_correct(p["prediction"], ex["answer"]):
            counts[v]["correct"] += 1
            continue
        pred = p["prediction"]
        s, _, _ = load_scene(data, ex)
        gid = ex["group_id"]
        if pred not in ex["candidate_entities"]:
            cat = "parse"
        elif (reads.get((gid, "ann", normalize(ex["query_text"]))) is False
              or reads.get((gid, "ent", normalize(ex["answer"]))) is False):
            cat = "perception"
        elif v == "cf" and pred == orig_gold.get(ex["query_id"]):
            cat = "stuck"
        elif pred == proximity_ranking(s, ex["query_ref"])[0]:
            cat = "proximity"
        elif pred in crossed_entities(s, ex["query_ref"]):
            cat = "crossing"
        else:
            cat = "other"
        counts[v][cat] += 1
        if len(samples[cat]) < 6:
            samples[cat].append({"example_id": ex["example_id"], "image": ex["image"], "question": ex["question"],
                                 "gold": ex["answer"], "pred": pred, "family": ex["family"],
                                 "features": ex.get("features", {})})
    return {k: dict(v) for k, v in counts.items()}, dict(samples)


def gate2(m_a, m_c, read_acc, stuck_a, heur):
    pca_a = m_a["paired_cf_accuracy"]["value"]
    pca_c = m_c["paired_cf_accuracy"]["value"] if m_c else None
    reasons, decision = [], None
    if pca_a >= 0.90:
        decision = "STOP_H1_no_deficit"
    elif read_acc is not None and read_acc < 0.90:
        decision = "REVISE_perception_dominated"
    elif pca_c is not None and pca_c - pca_a >= 0.25 and pca_c >= 0.85:
        decision = "REVISE_resolution_or_context"
    elif any(v > 0 for v in heur.values()):
        decision = "REVISE_benchmark_shortcut"
    elif pca_a <= 0.70 and (read_acc or 0) >= 0.95 and stuck_a >= 0.15 and (pca_c is None or pca_c <= 0.85):
        decision = "GO_pending_human_audit"
    else:
        decision = "INCONCLUSIVE"
    reasons = {"PCA_A": pca_a, "PCA_C": pca_c, "read_acc": read_acc, "stuck_A": stuck_a,
               "non_connector_heuristic_PCA": heur}
    return decision, reasons


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--diag", required=True)
    ap.add_argument("--heuristics", required=True, help="dir with <method>_metrics.json for the same data")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    data, diag, out = Path(args.data), Path(args.diag), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    exs = load_examples(data)
    res = {"conditions": {}}
    for cdir in sorted(p for p in diag.iterdir() if (p / "metrics.json").exists()):
        m = json.loads((cdir / "metrics.json").read_text())
        res["conditions"][cdir.name] = {s: {k: r[k] for k in ("paired_cf_accuracy", "acc_orig", "acc_cf", "acc_style",
                                                               "acc_reroute", "cf_stuck_rate", "cf_stuck_on_old_answer",
                                                               "cf_flip_given_orig_correct", "invariance_style",
                                                               "invariance_reroute", "abstention_accuracy",
                                                               "false_none_rate")}
                                        for s, r in m["splits"].items()}
        res["conditions"][cdir.name]["_strata_test_id"] = m["splits"].get("test_id", {}).get("strata")
    read_info = json.loads((diag / "P_read" / "run_info.json").read_text()) if (diag / "P_read").exists() else {}
    probes = load_jsonl(diag / "P_read" / "predictions.jsonl") if (diag / "P_read").exists() else []
    read_by_split = defaultdict(lambda: [0, 0])
    by_kind = defaultdict(lambda: [0, 0])
    ex_split = {e["example_id"]: e["split"] for e in exs}
    for p in probes:
        k, m = probe_kind(p["example_id"])
        by_kind[f"{k}_{m}"][0] += 1
        by_kind[f"{k}_{m}"][1] += p["correct"]
        if m == "crop":  # Gate-2 recognition measure = pure recognition (crop probe)
            sp = ex_split.get(p["example_id"].split("#")[0])
            read_by_split[sp][0] += 1
            read_by_split[sp][1] += p["correct"]
    res["read_accuracy"] = {s: round(c / n, 4) for s, (n, c) in read_by_split.items()}
    res["read_accuracy_by_kind"] = {k: round(c / n, 4) for k, (n, c) in sorted(by_kind.items())}
    reads = read_tables(probes)
    tax, samples = {}, {}
    for cond in ("A_direct", "B_reasoning", "C_oracle_crop", "E_oracle_box"):
        if (diag / cond / "predictions.jsonl").exists():
            tax[cond], samples[cond] = taxonomy(data, exs, load_jsonl(diag / cond / "predictions.jsonl"), reads)
    res["taxonomy_test_id"] = tax
    res["error_samples"] = samples
    heur = {}
    for m in ("text_prior", "proximity"):
        f = Path(args.heuristics) / f"{m}_metrics.json"
        if f.exists():
            heur[m] = json.loads(f.read_text())["splits"]["test_id"]["paired_cf_accuracy"]["value"]
    a = json.loads((diag / "A_direct" / "metrics.json").read_text())["splits"]["test_id"]
    c = json.loads((diag / "C_oracle_crop" / "metrics.json").read_text())["splits"]["test_id"] if (diag / "C_oracle_crop" / "metrics.json").exists() else None
    dec, why = gate2(a, c, res["read_accuracy"].get("test_id"), a["cf_stuck_rate"]["value"], heur)
    res["gate2"] = {"decision": dec, "inputs": why, "note": "human audit not yet performed; GO is provisional"}
    (out / "phase1_summary.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(json.dumps({"gate2": res["gate2"], "read_accuracy": res["read_accuracy"],
                      "read_by_kind": res["read_accuracy_by_kind"], "taxonomy_test_id": tax}, indent=2))


if __name__ == "__main__":
    main()
