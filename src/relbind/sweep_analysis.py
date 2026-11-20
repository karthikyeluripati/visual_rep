"""Zero-shot multi-model sweep analysis (strict_v3, A_direct). PRE-REGISTERED 2026-11-20, before any sweep output exists.

    python -m relbind.sweep_analysis --data D --sweep experiments/sweep --tracer t.jsonl
        --ref M1_finetuned=a.jsonl [M1_finetuned=b.jsonl ...] --out results/sweep/sweep_analysis.json

Per model and split: pair accuracy over all flipping queries (cluster bootstrap over groups), the same for leader_leader
pairs only, a2e stuck rate (same entity predicted on orig and cf), and the unparsed-output rate.
Pooled = all sweep splits. Hypothesis H_general (descriptive claim of the paper):
  "zero-shot strict counterfactual binding failure is general" iff >= 75% of successfully evaluated models are below
  the ray tracer on pooled leader_leader pair accuracy (point estimates), AND every model is below M1_finetuned on pooled.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from .metrics import load_examples, load_jsonl
from .pilot_analysis import pair_items, paired_diff, rate

SPLITS = ("test_id", "test_unseen_template", "test_unseen_family")


def stuck_rate(exs, preds):
    q = defaultdict(dict)
    for e in exs:
        if e["variant"] in ("orig", "cf") and e["flips_under_cf"] and e["query_type"] == "a2e":
            q[e["query_id"]][e["variant"]] = e["example_id"]
    n = s = 0
    for v in q.values():
        if {"orig", "cf"} <= v.keys() and all(x in preds for x in v.values()):
            a, b = preds[v["orig"]], preds[v["cf"]]
            if a not in (None, ""):
                n += 1
                s += a == b
    return round(s / n, 4) if n else None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--sweep", required=True)
    ap.add_argument("--tracer", default=None)
    ap.add_argument("--ref", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    P, info = {}, {}
    for d in sorted(Path(args.sweep).iterdir()):
        if (d / "predictions.jsonl").exists() and (d / "run_info.json").exists():
            rows = load_jsonl(d / "predictions.jsonl")
            P[d.name] = {r["example_id"]: r.get("prediction") for r in rows}
            ri = json.loads((d / "run_info.json").read_text(encoding="utf-8"))
            info[d.name] = {"model_id": ri["backend"].get("model_id"), "resolved_commit": ri["backend"].get("resolved_commit"),
                            "seconds": ri.get("seconds"), "n": len(rows),
                            "unparsed_rate": round(sum(1 for r in rows if not r.get("parsed_json")) / max(1, len(rows)), 4)}
    models = list(P)
    for spec in args.ref:
        n, f = spec.split("=", 1)
        P.setdefault(n, {}).update({r["example_id"]: r["prediction"] for r in load_jsonl(f)})
    if args.tracer:
        P["tracer"] = {r["example_id"]: r["prediction"] for r in load_jsonl(args.tracer)}
    exs = {sp: load_examples(args.data, [sp]) for sp in SPLITS}
    meta = {e["query_id"]: e for sp in SPLITS for e in exs[sp] if e["variant"] == "orig"}
    res = {"models": info, "accuracy": {}, "comparisons": {}}
    pooled = {}
    for n, pr in P.items():
        r, allit = {}, {}
        for sp in SPLITS:
            it = pair_items(exs[sp], pr)
            if it:
                r[sp] = rate(list(it.values()))
                allit.update(it)
        if not allit:
            continue
        ll = [v for q, v in allit.items() if meta[q]["crossing_cell"] == "leader_leader"]
        r["pooled"] = rate(list(allit.values()))
        r["pooled_leader_leader"] = rate(ll)
        r["stuck_rate_a2e_pooled"] = stuck_rate([e for sp in SPLITS for e in exs[sp]], pr)
        res["accuracy"][n] = r
        pooled[n] = allit
    for n in models:
        for b in ("tracer", "M1_finetuned"):
            if n in pooled and b in pooled:
                common = sorted(set(pooled[n]) & set(pooled[b]))
                res["comparisons"][f"{n} - {b}"] = paired_diff([(pooled[n][k][0], pooled[n][k][1], pooled[b][k][1]) for k in common])
    ok = [n for n in models if n in res["accuracy"]]
    if ok and "tracer" in res["accuracy"] and "M1_finetuned" in res["accuracy"]:
        t_ll = res["accuracy"]["tracer"]["pooled_leader_leader"]["value"]
        below_t = [n for n in ok if res["accuracy"][n]["pooled_leader_leader"]["value"] < t_ll]
        m1 = res["accuracy"]["M1_finetuned"]["pooled"]["value"]
        below_m1 = [n for n in ok if res["accuracy"][n]["pooled"]["value"] < m1]
        res["H_general"] = {"n_models": len(ok), "below_tracer_leader_leader": len(below_t),
                            "below_M1_finetuned_pooled": len(below_m1),
                            "holds": len(below_t) >= 0.75 * len(ok) and len(below_m1) == len(ok)}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res.get("H_general"), indent=1))
    for n, r in res["accuracy"].items():
        print(f"{n:45s} pooled={r['pooled']['value']:.3f} LL={r['pooled_leader_leader']['value']:.3f} "
              f"stuck={r['stuck_rate_a2e_pooled']} " + " ".join(f"{sp}={r[sp]['value']:.3f}" for sp in SPLITS if sp in r))


if __name__ == "__main__":
    main()
