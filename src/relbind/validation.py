"""Score completed blind human audits against the answer key.

Single sheet (audit_v0 format):
    python -m relbind.validation score --audit data/validation/audit_v0 --sheet <filled>.csv
Two reviewers (audit_diag_v1 package):
    python -m relbind.validation score2 --key <pkg>_ANSWER_KEY_DO_NOT_SHARE/answer_key.json \
        --sheet-a filled_A.csv --sheet-b filled_B.csv

Pre-registered pass criterion (research/experimental_protocol.md §2): for EACH reviewer, accuracy
>= 98% and <= 2% of items flagged ambiguous; additionally (two-reviewer package) >= 95% of orig/cf
pairs answered correctly on both members by both reviewers. Items failed by either reviewer are
listed for inspection.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from .metrics import is_correct


def _load_sheet(p) -> dict[int, dict]:
    return {int(r["item"]): r for r in csv.DictReader(open(p, encoding="utf-8")) if r["reviewer_answer"].strip()}


def _amb(r) -> bool:
    return r["unambiguous(y/n)"].strip().lower().startswith("n")


def score(audit_dir: Path, sheet: Path) -> dict:
    key = {k["item"]: k for k in json.loads((audit_dir / "answer_key.json").read_text(encoding="utf-8"))}
    done = _load_sheet(sheet)
    agg = defaultdict(lambda: [0, 0, 0])
    for i, r in done.items():
        k = key[i]
        ok = is_correct(r["reviewer_answer"], k["answer"])
        for g in ("all", f"split={k['split']}", f"variant={k['variant']}", f"family={k['family']}"):
            agg[g][0] += 1
            agg[g][1] += ok
            agg[g][2] += _amb(r)
    res = {g: {"n": n, "reviewer_accuracy": round(c / n, 4), "flagged_ambiguous": round(a / n, 4)}
           for g, (n, c, a) in sorted(agg.items())}
    res["n_unanswered"] = len(key) - len(done)
    res["pass"] = bool(done) and res["all"]["reviewer_accuracy"] >= 0.98 and res["all"]["flagged_ambiguous"] <= 0.02
    return res


def cohen_kappa(a: list[bool], b: list[bool]) -> float | None:
    n = len(a)
    if n == 0:
        return None
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return None if pe == 1 else round((po - pe) / (1 - pe), 4)


def score2(key_path: Path, sheet_a: Path, sheet_b: Path) -> dict:
    key = {k["item"]: k for k in json.loads(Path(key_path).read_text(encoding="utf-8"))}
    A, B = _load_sheet(sheet_a), _load_sheet(sheet_b)
    res = {"n_items": len(key), "answered": {"A": len(A), "B": len(B)}}
    per = {}
    for name, S in (("A", A), ("B", B)):
        ok = {i: is_correct(r["reviewer_answer"], key[i]["answer"]) for i, r in S.items()}
        amb = {i: _amb(r) for i, r in S.items()}
        per[name] = (ok, amb)
        res[f"reviewer_{name}"] = {"accuracy": round(sum(ok.values()) / max(1, len(ok)), 4),
                                   "flagged_ambiguous": round(sum(amb.values()) / max(1, len(amb)), 4)}
    common = sorted(set(A) & set(B))
    res["agreement"] = {
        "n_common": len(common),
        "same_answer_rate": round(sum(is_correct(A[i]["reviewer_answer"], B[i]["reviewer_answer"]) for i in common) / max(1, len(common)), 4),
        "kappa_correctness": cohen_kappa([per["A"][0][i] for i in common], [per["B"][0][i] for i in common]),
        "kappa_ambiguity_flag": cohen_kappa([per["A"][1][i] for i in common], [per["B"][1][i] for i in common]),
    }
    pairs = defaultdict(dict)
    for i, k in key.items():
        if k["variant"] in ("orig", "cf"):
            pairs[k["query_id"]][k["variant"]] = i
    full = [p for p in pairs.values() if {"orig", "cf"} <= p.keys()]
    both_ok = [all(per[r][0].get(p[v], False) for r in ("A", "B") for v in ("orig", "cf")) for p in full]
    res["pairs"] = {"n": len(full), "both_members_correct_by_both_reviewers": round(sum(both_ok) / max(1, len(both_ok)), 4)}
    res["failed_items"] = [{"item": i, "example_id": key[i]["example_id"], "gold": key[i]["answer"],
                            "A": A.get(i, {}).get("reviewer_answer"), "B": B.get(i, {}).get("reviewer_answer"),
                            "A_ambiguous": per["A"][1].get(i), "B_ambiguous": per["B"][1].get(i)}
                           for i in sorted(key) if not (per["A"][0].get(i, False) and per["B"][0].get(i, False))
                           or per["A"][1].get(i) or per["B"][1].get(i)]
    complete = len(A) == len(B) == len(key)
    res["pass"] = (complete and all(res[f"reviewer_{r}"]["accuracy"] >= 0.98 and res[f"reviewer_{r}"]["flagged_ambiguous"] <= 0.02
                                    for r in ("A", "B")) and res["pairs"]["both_members_correct_by_both_reviewers"] >= 0.95)
    res["complete"] = complete
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("--audit", required=True)
    s.add_argument("--sheet", required=True)
    s2 = sub.add_parser("score2")
    s2.add_argument("--key", required=True)
    s2.add_argument("--sheet-a", required=True)
    s2.add_argument("--sheet-b", required=True)
    args = ap.parse_args(argv)
    if args.cmd == "score":
        print(json.dumps(score(Path(args.audit), Path(args.sheet)), indent=2))
    else:
        print(json.dumps(score2(Path(args.key), Path(args.sheet_a), Path(args.sheet_b)), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
