"""Evaluate a prediction file against RelBind-Bench.

    python -m relbind.eval --data data/prototype --pred results/b0/proximity.jsonl --out results/b0/proximity_metrics.json
    python -m relbind.eval --data data/prototype --pred A.jsonl --compare B.jsonl --out cmp.json

Predictions: JSONL with {"example_id", "prediction", optional "pred_annotation_bbox", "pred_entity_bbox"}.
Metrics are reported per split with 95% cluster-bootstrap CIs (resampling groups).
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from .metrics import compare, join, load_examples, load_jsonl, localization, query_table, stratified, summarize


def evaluate(data, pred_path, splits=None, B=2000, strata=True) -> dict:
    exs = load_examples(data, splits)
    rows = join(exs, load_jsonl(pred_path))
    by_split = defaultdict(list)
    for r in rows:
        by_split[r["split"]].append(r)
    out = {"pred_file": str(pred_path), "n_examples_scored": len(rows), "n_examples_total": len(exs), "splits": {}}
    for split, rs in sorted(by_split.items()):
        Q = query_table(rs)
        res = summarize(Q, B=B)
        res["localization"] = localization(rs)
        if strata:
            res["strata"] = stratified(Q)
        out["splits"][split] = res
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--compare", default=None, help="second prediction file (paired comparison A-B)")
    ap.add_argument("--splits", nargs="*", default=None)
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--no-strata", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    res = evaluate(args.data, args.pred, args.splits, args.B, not args.no_strata)
    if args.compare:
        exs = load_examples(args.data, args.splits)
        ra, rb = join(exs, load_jsonl(args.pred)), join(exs, load_jsonl(args.compare))
        res["comparison"] = {}
        for split in sorted({r["split"] for r in ra}):
            qa = query_table([r for r in ra if r["split"] == split])
            qb = query_table([r for r in rb if r["split"] == split])
            res["comparison"][split] = {m: compare(qa, qb, m, B=args.B) for m in ("paired_cf_accuracy", "acc_orig")}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    for split, r in res["splits"].items():
        print(f"{split:24s} PCA={r['paired_cf_accuracy']['value']} {r['paired_cf_accuracy']['ci95']}  "
              f"acc_orig={r['acc_orig']['value']}  stuck={r['cf_stuck_rate']['value']}  "
              f"inv_style={r['invariance_style']['value']}  abst={r['abstention_accuracy']['value']}")


if __name__ == "__main__":
    main()
