"""Decisive Experiment 1 analysis (from artifacts only).

    python -m relbind.exp1_analysis --data <strict_v2 dir with examples/path_examples> \
        --binding experiments/exp1/A_direct/predictions.jsonl --path experiments/exp1/A_path/predictions.jsonl \
        --tracer results/exp1/ray_tracer.jsonl --probe results/exp1/probe.json --out results/exp1/exp1_analysis.json

Unit: strict swap pair of an a2e query (orig and cf both correct). Paired comparison binding vs path control
on the SAME pairs: exact McNemar + cluster bootstrap over groups. Pre-registered reading in
research/phase2_go_no_go.md §2.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from .final_analysis import paired_diff, rate
from .metrics import is_correct, load_examples, load_jsonl


def pairs(exs, preds):
    q = defaultdict(dict)
    for e in exs:
        if e["query_type"] == "a2e" and e["variant"] in ("orig", "cf"):
            q[e["query_id"].replace("-path", "")][e["variant"]] = e
    out = {}
    for qid, v in q.items():
        if {"orig", "cf"} <= v.keys() and all(x["example_id"] in preds for x in v.values()):
            out[qid] = (v["orig"]["group_id"], v["orig"]["split"], v["orig"].get("crossing_cell"),
                        all(is_correct(preds[x["example_id"]], x["answer"]) for x in v.values()))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--binding", required=True)
    ap.add_argument("--path", required=True)
    ap.add_argument("--tracer", default=None)
    ap.add_argument("--probe", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    splits = ["test_id", "test_unseen_topology"]
    ex_b = load_examples(args.data, splits)
    ex_p = load_examples(args.data, splits, "path_examples.jsonl")
    P = {n: {r["example_id"]: r["prediction"] for r in load_jsonl(f)} for n, f in
         (("binding", args.binding), ("path", args.path)) + ((("tracer", args.tracer),) if args.tracer else ())}
    pb, pp = pairs(ex_b, P["binding"]), pairs(ex_p, P["path"])
    pt = pairs(ex_b, P["tracer"]) if "tracer" in P else {}
    res = {"by_split": {}}
    for sp in splits:
        r = {}
        for cell in (None, "none", "entity_only", "leader_leader"):
            keys = [k for k in pb if k in pp and pb[k][1] == sp and (cell is None or pb[k][2] == cell)]
            name = cell or "all"
            r[name] = {
                "binding_A": rate([(pb[k][0], pb[k][3]) for k in keys]),
                "path_control": rate([(pp[k][0], pp[k][3]) for k in keys]),
                "path_minus_binding": paired_diff([(pb[k][0], pp[k][3], pb[k][3]) for k in keys]) if keys else None,
                "tracer": rate([(pt[k][0], pt[k][3]) for k in keys if k in pt]) if pt else None,
            }
        res["by_split"][sp] = r
    if args.probe and Path(args.probe).exists():
        pr = json.loads(Path(args.probe).read_text())
        res["probe"] = {"selected": pr["selected"],
                        **{sp: {k: v for k, v in pr["test"][sp]["probe"].items()} for sp in pr["test"]},
                        "controls": {sp: {"shuffled_labels": pr["test"][sp]["control_shuffled_labels"]["pair_acc"],
                                          "geometry_only": pr["test"][sp]["control_geometry_only"]["pair_acc"]}
                                     for sp in pr["test"]}}
        topo = pr["test"].get("test_unseen_topology", {}).get("probe", {}).get("pair_acc")
        res["probe_reading"] = (None if topo is None else
                                "read-out deficit (connectivity linearly available)" if topo >= 0.60 else
                                "representation deficit (not linearly available)" if topo <= 0.30 else "inconclusive")
    d = res["by_split"]["test_id"]["all"]["path_minus_binding"]
    if d:
        res["path_reading"] = ("binding-specific deficit beyond tracing" if d["diff_pp"] >= 10 and d["ci95_pp"][0] > 0 else
                               "failure is path following (binding adds nothing measurable)" if abs(d["diff_pp"]) < 5 else
                               "intermediate / see CIs")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "by_split"}, indent=1))
    for sp, r in res["by_split"].items():
        for cell, v in r.items():
            print(sp, cell, "A", v["binding_A"]["value"], "path", v["path_control"]["value"],
                  "tracer", v["tracer"]["value"] if v["tracer"] else None, "n", v["binding_A"]["den"])


if __name__ == "__main__":
    main()
