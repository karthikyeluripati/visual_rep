"""Cross-family learnability: does answer-only SFT also fix a non-Qwen base model? PRE-REGISTERED 2026-11-20.

transfers_across_families iff the fine-tuned model's pooled pair accuracy on test_unseen_template + test_unseen_family is
>= its own zero-shot base + 40 pp, with paired-bootstrap CI lower bound > 0 (the same pairs for both).

    python -m relbind.xfam_analysis --data D --ft pred.jsonl --base pred.jsonl --out o.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .metrics import load_examples, load_jsonl
from .pilot_analysis import pair_items, paired_diff, rate

SPLITS = ("test_unseen_template", "test_unseen_family")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ft", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    ft = {r["example_id"]: r["prediction"] for r in load_jsonl(a.ft)}
    bs = {r["example_id"]: r["prediction"] for r in load_jsonl(a.base)}
    res = {"splits": {}}
    pf, pb = {}, {}
    for sp in SPLITS + ("val", "test_unseen_topology"):
        exs = load_examples(a.data, [sp])
        f, b = pair_items(exs, ft), pair_items(exs, bs)
        res["splits"][sp] = {"ft": rate(list(f.values())) if f else None, "base": rate(list(b.values())) if b else None}
        if sp in SPLITS:
            pf.update(f)
            pb.update(b)
    common = sorted(set(pf) & set(pb))
    d = paired_diff([(pf[k][0], pf[k][1], pb[k][1]) for k in common])
    res["pooled_heldout"] = {"ft": rate([pf[k] for k in common]), "base": rate([pb[k] for k in common]), "ft_minus_base": d}
    res["transfers_across_families"] = bool(d["diff_pp"] >= 40 and d["ci95_pp"][0] > 0)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
