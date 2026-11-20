"""Data-efficiency / counterfactual-pair ablation / seed robustness for M1 (answer-only SFT) on strict_v3.
PRE-REGISTERED 2026-11-20, before any output exists.

    python -m relbind.datascale_analysis --data D --arms NAME=pred.jsonl ... --out results/datascale/datascale_analysis.json

Arm names: M1_n600_s{0,1,2}, M1_n150_s0, M1_n50_s0, M1_noCF_s0 (train without cf variants), M1_noStyle_s0 (train
without style variants; same example count as noCF). Metric: pair accuracy, all flipping queries; "pooled unseen" =
test_unseen_topology + test_unseen_template + test_unseen_family (cluster bootstrap over groups).
Q1 counterfactual pairs: d = noStyle - noCF on pooled unseen.
   cf_pairs_matter        if d >= +3 pp, CI lower > 0, and d > seed range of M1_n600 (max - min over seeds, pooled unseen)
   nuisance_pairs_better  if d <= -3 pp, CI upper < 0, and |d| > seed range
   no_detectable_difference otherwise
Q2 data efficiency: n* = smallest n in {50, 150, 600} whose pooled-unseen accuracy >= mean(M1_n600 seeds) - 5 pp.
Q3 seed robustness: mean, sd, min, max of M1_n600 over seeds, per split and pooled.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

from .metrics import load_examples, load_jsonl
from .pilot_analysis import pair_items, paired_diff, rate

UNSEEN = ("test_unseen_topology", "test_unseen_template", "test_unseen_family")
SPLITS = ("val",) + UNSEEN


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    P = {}
    for spec in args.arms:
        n, f = spec.split("=", 1)
        P[n] = {r["example_id"]: r["prediction"] for r in load_jsonl(f)}
    exs = {sp: load_examples(args.data, [sp]) for sp in SPLITS}
    items = {n: {sp: pair_items(exs[sp], pr) for sp in SPLITS} for n, pr in P.items()}
    pooled = {n: {k: v for sp in UNSEEN for k, v in it[sp].items()} for n, it in items.items()}
    res = {"accuracy": {}, "rules": {}}
    for n in P:
        res["accuracy"][n] = {sp: rate(list(items[n][sp].values())) for sp in SPLITS}
        res["accuracy"][n]["pooled_unseen"] = rate(list(pooled[n].values()))

    seeds = sorted(n for n in P if n.startswith("M1_n600_s"))
    q3 = {}
    for sp in SPLITS + ("pooled_unseen",):
        vals = [res["accuracy"][n][sp]["value"] for n in seeds]
        if vals:
            q3[sp] = {"seeds": seeds, "values": vals, "mean": round(st.mean(vals), 4),
                      "sd": round(st.stdev(vals), 4) if len(vals) > 1 else None, "min": min(vals), "max": max(vals)}
    res["rules"]["Q3_seed_robustness"] = q3
    seed_range_pp = round(100 * (q3["pooled_unseen"]["max"] - q3["pooled_unseen"]["min"]), 1) if seeds else None

    def diff(a, b, sp=None):
        A = pooled[a] if sp is None else items[a][sp]
        B = pooled[b] if sp is None else items[b][sp]
        common = sorted(set(A) & set(B))
        return paired_diff([(A[k][0], A[k][1], B[k][1]) for k in common])

    if {"M1_noCF_s0", "M1_noStyle_s0"} <= P.keys():
        d = diff("M1_noStyle_s0", "M1_noCF_s0")
        v = d["diff_pp"]
        if v >= 3 and d["ci95_pp"][0] > 0 and seed_range_pp is not None and v > seed_range_pp:
            verdict = "cf_pairs_matter"
        elif v <= -3 and d["ci95_pp"][1] < 0 and seed_range_pp is not None and -v > seed_range_pp:
            verdict = "nuisance_pairs_better"
        else:
            verdict = "no_detectable_difference"
        res["rules"]["Q1_counterfactual_pairs"] = {
            "noStyle_minus_noCF_pooled_unseen": d, "seed_range_pp": seed_range_pp, "verdict": verdict,
            "per_split": {sp: diff("M1_noStyle_s0", "M1_noCF_s0", sp) for sp in SPLITS}}
    if seeds:
        target = q3["pooled_unseen"]["mean"] - 0.05
        curve = {}
        for n_ in (50, 150, 600):
            names = [f"M1_n{n_}_s0"] if n_ != 600 else seeds
            vals = [res["accuracy"][x]["pooled_unseen"]["value"] for x in names if x in P]
            if vals:
                curve[n_] = round(st.mean(vals), 4)
        n_star = next((n_ for n_ in (50, 150, 600) if n_ in curve and curve[n_] >= target), None)
        res["rules"]["Q2_data_efficiency"] = {"pooled_unseen_by_n": curve, "target": round(target, 4), "n_star": n_star}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res["rules"], indent=1)[:4000])
    for n in P:
        print(n, {sp: v["value"] for sp, v in res["accuracy"][n].items()})


if __name__ == "__main__":
    main()
