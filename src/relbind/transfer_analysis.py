"""Template/family transfer analysis (strict_v3). Decision rule PRE-REGISTERED 2026-11-19, before any output exists.

    python -m relbind.transfer_analysis --data D --arms M1=p.jsonl M2R=p.jsonl M3=p.jsonl --base b.jsonl
        --tracer t.jsonl --out results/transfer/transfer_analysis.json

Metric: pair accuracy over all flipping queries (orig and cf both correct), cluster bootstrap over groups.
Rule, per held-out split S in {test_unseen_template, test_unseen_family}:
  drop_S = M1(val) - M1(S), in pp.   transfers: drop <= 15 | collapses: drop >= 30 | partial: otherwise.
  method_helps_S: max over X in {M3, M2R} of (X - M1) on S is >= +3 pp with paired-bootstrap CI lower bound > 0.
Recommendation:
  some split partial/collapses AND method_helps there -> "method_comparison_warranted" (3 seeds on held-out splits; PI approval)
  both splits transfer AND no method_helps           -> "drop_or_reframe_method_claim"
  otherwise                                          -> "no_method_advantage"  (reframe as diagnostic / data-side study)
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from .metrics import load_examples, load_jsonl
from .pilot_analysis import pair_items, paired_diff, rate

SPLITS = ("val", "test_unseen_topology", "test_unseen_template", "test_unseen_family")
HELD_OUT = ("test_unseen_template", "test_unseen_family")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--arms", nargs="+", required=True, help="NAME=predictions.jsonl")
    ap.add_argument("--base", nargs="*", default=[])
    ap.add_argument("--tracer", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    P, head = {}, {}
    for spec in args.arms:
        n, f = spec.split("=", 1)
        rows = load_jsonl(f)
        P[n] = {r["example_id"]: r["prediction"] for r in rows}
        if any("head_prediction" in r for r in rows):
            head[n + "_head"] = {r["example_id"]: r["head_prediction"] for r in rows if "head_prediction" in r}
    if args.base:
        P["M0_base"] = {r["example_id"]: r["prediction"] for f in args.base for r in load_jsonl(f)}
    if args.tracer:
        P["tracer"] = {r["example_id"]: r["prediction"] for r in load_jsonl(args.tracer)}
    res = {"splits": {}}
    for sp in SPLITS:
        exs = load_examples(args.data, [sp])
        if not exs:
            continue
        meta = {e["query_id"]: e for e in exs if e["variant"] == "orig"}
        r = {"pair_accuracy_all_flipping": {}, "head_pair_accuracy_a2e": {}, "by_crossing_cell": defaultdict(dict),
             "by_template": defaultdict(dict)}
        items = {}
        for n, pr in P.items():
            it = pair_items(exs, pr)
            if not it:
                continue
            items[n] = it
            r["pair_accuracy_all_flipping"][n] = rate(list(it.values()))
            for fac in ("crossing_cell", "template"):
                g = defaultdict(list)
                for q, v in it.items():
                    g[meta[q][fac]].append(v)
                for k, vs in g.items():
                    r[f"by_{fac}"][k][n] = rate(vs)
        for n, pr in head.items():
            it = pair_items(exs, pr, a2e_only=True)
            if it:
                r["head_pair_accuracy_a2e"][n] = rate(list(it.values()))
        comps = {}
        names = [n for n in items if n not in ("M0_base", "tracer")]
        pairs = [(a, "M1") for a in names if a != "M1"] + [(a, b) for a in names for b in ("M0_base", "tracer")]
        for a, b in pairs:
            if a in items and b in items:
                common = sorted(set(items[a]) & set(items[b]))
                comps[f"{a} - {b}"] = paired_diff([(items[a][k][0], items[a][k][1], items[b][k][1]) for k in common])
        r["paired_comparisons"] = comps
        res["splits"][sp] = r

    rule = {}
    ref = res["splits"].get("val", {}).get("pair_accuracy_all_flipping", {}).get("M1")
    for sp in HELD_OUT:
        s = res["splits"].get(sp)
        if not s or not ref or "M1" not in s["pair_accuracy_all_flipping"]:
            continue
        drop = round(100 * (ref["value"] - s["pair_accuracy_all_flipping"]["M1"]["value"]), 1)
        verdict = "transfers" if drop <= 15 else "collapses" if drop >= 30 else "partial"
        helps = {x: c for x in ("M3", "M2R") if (c := s["paired_comparisons"].get(f"{x} - M1"))
                 and c["diff_pp"] >= 3 and c["ci95_pp"][0] > 0}
        rule[sp] = {"M1_val": ref["value"], "M1_split": s["pair_accuracy_all_flipping"]["M1"]["value"],
                    "drop_pp": drop, "verdict": verdict, "method_helps": sorted(helps)}
    if len(rule) == len(HELD_OUT):
        if any(v["verdict"] != "transfers" and v["method_helps"] for v in rule.values()):
            rec = "method_comparison_warranted"
        elif all(v["verdict"] == "transfers" and not v["method_helps"] for v in rule.values()):
            rec = "drop_or_reframe_method_claim"
        else:
            rec = "no_method_advantage"
        rule["recommendation"] = rec
    res["rule"] = rule
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(rule, indent=1))
    for sp, r in res["splits"].items():
        print(sp, {n: v["value"] for n, v in r["pair_accuracy_all_flipping"].items()},
              "head:", {n: v["value"] for n, v in r["head_pair_accuracy_a2e"].items()})


if __name__ == "__main__":
    main()
