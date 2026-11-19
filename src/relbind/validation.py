"""Score a completed blind human audit sheet against the answer key.

    python -m relbind.validation score --audit data/validation/audit_v0 --sheet data/validation/audit_v0/audit_sheet_reviewer1.csv

Reports reviewer accuracy (unique answerability proxy) and the fraction of items flagged ambiguous,
overall and per split/variant. A benchmark item set is acceptable only if reviewer accuracy is >= 98%
and <= 2% of items are flagged ambiguous (pre-registered, research/experimental_protocol.md).
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from .metrics import is_correct


def score(audit_dir: Path, sheet: Path) -> dict:
    key = {k["item"]: k for k in json.loads((audit_dir / "answer_key.json").read_text(encoding="utf-8"))}
    rows = list(csv.DictReader(open(sheet, encoding="utf-8")))
    done = [r for r in rows if r["reviewer_answer"].strip()]
    agg = defaultdict(lambda: [0, 0, 0])
    for r in done:
        k = key[int(r["item"])]
        ok = is_correct(r["reviewer_answer"], k["answer"])
        amb = r["unambiguous(y/n)"].strip().lower().startswith("n")
        for g in ("all", f"split={k['split']}", f"variant={k['variant']}", f"family={k['family']}"):
            agg[g][0] += 1
            agg[g][1] += ok
            agg[g][2] += amb
    res = {g: {"n": n, "reviewer_accuracy": round(c / n, 4), "flagged_ambiguous": round(a / n, 4)}
           for g, (n, c, a) in sorted(agg.items())}
    res["n_unanswered"] = len(rows) - len(done)
    res["pass"] = bool(done) and res["all"]["reviewer_accuracy"] >= 0.98 and res["all"]["flagged_ambiguous"] <= 0.02
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("--audit", required=True)
    s.add_argument("--sheet", required=True)
    args = ap.parse_args(argv)
    print(json.dumps(score(Path(args.audit), Path(args.sheet)), indent=2))


if __name__ == "__main__":
    main()
