"""Phase-2 pilot analysis: arms vs each other, vs base (M0 = Exp-1 A_direct) and vs the tracer.

    python -m relbind.pilot_analysis --data D --arms M1=... M2=... M3=... --base experiments/exp1/A_direct/predictions.jsonl \
        --tracer results/exp1/ray_tracer.jsonl --out results/pilot/pilot_analysis.json

Pre-registered (research/phase2_experimental_design.md §4, §6):
  primary endpoint = strict-pair accuracy (a2e+e2a flipping queries, generated answers) on the held-out split
  (pilot: test_unseen_topology; unseen template is not in strict_v2). Comparison of interest: M3 vs M2.
  Pilot rule: proceed to 3 seeds only if M3 - M2 >= +3 pp on the primary endpoint OR the M2/M3 gap to the
  tracer narrows by >= 10 pp. Otherwise stop and report.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .final_analysis import paired_diff, rate
from .metrics import is_correct, load_examples, load_jsonl


def pair_items(exs, preds, key="prediction", a2e_only=False):
    q = {}
    for e in exs:
        if e["variant"] in ("orig", "cf") and e["flips_under_cf"] and (not a2e_only or e["query_type"] == "a2e"):
            q.setdefault(e["query_id"], {})[e["variant"]] = e
    out = {}
    for qid, v in q.items():
        if {"orig", "cf"} <= v.keys() and all(x["example_id"] in preds for x in v.values()):
            out[qid] = (v["orig"]["group_id"], all(is_correct(preds[x["example_id"]], x["answer"]) for x in v.values()))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--arms", nargs="+", required=True, help="NAME=predictions.jsonl")
    ap.add_argument("--base", default=None)
    ap.add_argument("--tracer", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    P = {}
    head = {}
    for spec in args.arms:
        n, f = spec.split("=", 1)
        rows = load_jsonl(f)
        P[n] = {r["example_id"]: r["prediction"] for r in rows}
        if any("head_prediction" in r for r in rows):
            head[n + "_head"] = {r["example_id"]: r["head_prediction"] for r in rows if "head_prediction" in r}
    if args.base:
        P["M0_base"] = {r["example_id"]: r["prediction"] for r in load_jsonl(args.base)}
    if args.tracer:
        P["tracer"] = {r["example_id"]: r["prediction"] for r in load_jsonl(args.tracer)}
    res = {"splits": {}}
    for sp in ("val", "test_unseen_topology", "test_id"):
        exs = load_examples(args.data, [sp])
        if not exs:
            continue
        r = {"pair_accuracy_all_flipping": {}, "pair_accuracy_a2e": {}, "head_relationship_pair_accuracy_a2e": {}}
        items = {}
        for n, pr in P.items():
            it = pair_items(exs, pr)
            if it:
                items[n] = it
                r["pair_accuracy_all_flipping"][n] = rate(list(it.values()))
                r["pair_accuracy_a2e"][n] = rate(list(pair_items(exs, pr, a2e_only=True).values()))
        for n, pr in head.items():
            it = pair_items(exs, pr, a2e_only=True)
            if it:
                r["head_relationship_pair_accuracy_a2e"][n] = rate(list(it.values()))
        comps = {}
        for a, b in (("M3", "M2"), ("M2", "M1"), ("M3", "M1"), ("M1", "M0_base"), ("M2", "M0_base"), ("M3", "M0_base"),
                     ("M2", "tracer"), ("M3", "tracer")):
            if a in items and b in items:
                common = sorted(set(items[a]) & set(items[b]))
                comps[f"{a} - {b}"] = paired_diff([(items[a][k][0], items[a][k][1], items[b][k][1]) for k in common])
        r["paired_comparisons_all_flipping"] = comps
        res["splits"][sp] = r
    prim = res["splits"].get("test_unseen_topology", {}).get("paired_comparisons_all_flipping", {})
    d32 = prim.get("M3 - M2")
    g2, g3 = prim.get("M2 - tracer"), prim.get("M3 - tracer")
    if d32 and g2 and g3:
        narrows = g3["diff_pp"] - g2["diff_pp"]
        res["pilot_rule"] = {"M3_minus_M2_pp": d32["diff_pp"], "ci95_pp": d32["ci95_pp"], "mcnemar_p": d32["mcnemar_p"],
                             "gap_to_tracer_change_pp(M3 vs M2)": round(narrows, 1),
                             "proceed_to_3_seeds": bool(d32["diff_pp"] >= 3 or narrows >= 10)}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res.get("pilot_rule"), indent=1))
    for sp, r in res["splits"].items():
        print(sp, {n: v["value"] for n, v in r["pair_accuracy_all_flipping"].items()},
              "head:", {n: v["value"] for n, v in r["head_relationship_pair_accuracy_a2e"].items()})


if __name__ == "__main__":
    main()
