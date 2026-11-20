"""Robustness checks before submission. PRE-REGISTERED 2026-11-20, before any output exists.

(R1) cf-pair replication: the noCF and noStyle arms at seeds 0, 1, 2.
     d_s = noStyle_s - noCF_s, pair accuracy pooled over the unseen splits.
     cf_pairs_matter_replicated iff mean(d_s) >= 3 pp AND every d_s > 0.
(R2) prompt sensitivity: B_reasoning (think step by step, then JSON) for the 3 best zero-shot models, on the first
     100 groups of each sweep split. prompt_robust iff every model's B_reasoning pooled leader-leader pair accuracy
     stays below the ray tracer's on the same pairs.

    python -m relbind.robustness_analysis --data D --cf NAME=pred ... --breason NAME=pred ... --tracer t.jsonl --out o.json
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

from .metrics import load_examples, load_jsonl
from .pilot_analysis import pair_items, rate

UNSEEN = ("test_unseen_topology", "test_unseen_template", "test_unseen_family")
SWEEP = ("test_id", "test_unseen_template", "test_unseen_family")


def _load(specs):
    out = {}
    for s in specs:
        n, f = s.split("=", 1)
        out[n] = {r["example_id"]: r["prediction"] for r in load_jsonl(f)}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--cf", nargs="*", default=[], help="M1_noCF_s{0,1,2}=..., M1_noStyle_s{0,1,2}=...")
    ap.add_argument("--breason", nargs="*", default=[])
    ap.add_argument("--tracer", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    res = {}
    cf = _load(a.cf)
    if cf:
        exs = [e for sp in UNSEEN for e in load_examples(a.data, [sp])]
        acc = {n: rate(list(pair_items(exs, p).values()))["value"] for n, p in cf.items()}
        ds = {s: round(100 * (acc[f"M1_noStyle_s{s}"] - acc[f"M1_noCF_s{s}"]), 1)
              for s in (0, 1, 2) if f"M1_noStyle_s{s}" in acc and f"M1_noCF_s{s}" in acc}
        res["R1"] = {"pooled_unseen": acc, "d_pp_by_seed": ds, "mean_d_pp": round(st.mean(ds.values()), 1) if ds else None,
                     "cf_pairs_matter_replicated": bool(len(ds) == 3 and st.mean(ds.values()) >= 3 and all(v > 0 for v in ds.values()))}
    br = _load(a.breason)
    if br:
        exs = [e for sp in SWEEP for e in load_examples(a.data, [sp])]
        meta = {e["query_id"]: e for e in exs if e["variant"] == "orig"}
        tr = {r["example_id"]: r["prediction"] for r in load_jsonl(a.tracer)}
        r2 = {}
        for n, p in br.items():
            it = pair_items(exs, p)
            ll = {q: v for q, v in it.items() if meta[q]["crossing_cell"] == "leader_leader"}
            tit = pair_items(exs, tr)
            tll = [tit[q] for q in ll if q in tit]
            r2[n] = {"pooled": rate(list(it.values())), "leader_leader": rate(list(ll.values())),
                     "tracer_same_pairs_leader_leader": rate(tll)}
        res["R2"] = {"models": r2, "prompt_robust": all(v["leader_leader"]["value"] < v["tracer_same_pairs_leader_leader"]["value"]
                                                         for v in r2.values())}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1)[:3000])


if __name__ == "__main__":
    main()
