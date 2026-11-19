"""Metrics for RelBind-Bench.

Unit of analysis
  example  = one (image variant, query).
  query    = one question asked on all variants of a group (orig / cf / reroute / style).
  group    = one base scene and its variants; the resampling unit for confidence intervals
             (queries inside a group are correlated, so we use a cluster bootstrap over groups).

Primary metric (pre-registered): paired counterfactual accuracy (PCA) on flipping queries —
a query counts as correct only if it is answered correctly on BOTH the original and the
counterfactual image. Any strategy that ignores connector pixels scores 0 on PCA by construction,
because the two images are identical except for the swapped connector paths while the answers differ.
"""
from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Callable

import numpy as np

from .geometry import bbox_iou

NONE = "NONE"


# ----------------------------------------------------------------------------- answer matching

def normalize(s) -> str:
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", str(s)).strip()
    s = s.replace("’", "'").replace("‘", "'").replace("”", '"').replace("“", '"').replace("″", '"').replace("′", "'")
    s = s.strip().strip(".").strip()
    # strip all surrounding quote characters (applied to gold and prediction alike, so values that
    # end in an inch mark, e.g. 4'-0", compare equal whether or not the model wraps them in quotes)
    s = s.strip("\"'`").strip()
    s = s.replace("μ", "µ")
    s = re.sub(r"\s+", "", s)
    return s.lower()


def is_correct(pred, gold) -> bool:
    return normalize(pred) == normalize(gold)


# ----------------------------------------------------------------------------- loading

def load_jsonl(p) -> list[dict]:
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def load_examples(data_dir, splits=None, name: str = "examples.jsonl") -> list[dict]:
    d = Path(data_dir)
    out = []
    for f in sorted(d.glob(f"*/{name}")):
        if splits and f.parent.name not in splits:
            continue
        out += load_jsonl(f)
    return out


def join(examples: list[dict], preds: list[dict]) -> list[dict]:
    pm = {p["example_id"]: p for p in preds}
    rows = []
    for ex in examples:
        p = pm.get(ex["example_id"])
        if p is None:
            continue
        r = dict(ex)
        r["prediction"] = p.get("prediction")
        r["correct"] = is_correct(p.get("prediction"), ex["answer"])
        for k in ("pred_annotation_bbox", "pred_entity_bbox"):
            if k in p:
                r[k] = p[k]
        rows.append(r)
    return rows


# ----------------------------------------------------------------------------- per-query records

def query_table(rows: list[dict]) -> list[dict]:
    """One record per query with per-variant predictions/correctness."""
    by_q = defaultdict(dict)
    meta = {}
    for r in rows:
        by_q[r["query_id"]][r["variant"]] = r
        meta[r["query_id"]] = r
    out = []
    for qid, v in by_q.items():
        m = meta[qid]
        rec = {"query_id": qid, "group_id": m["group_id"], "split": m["split"], "family": m["family"],
               "template": m["template"], "topology": m["topology"], "query_type": m["query_type"],
               "flips": m["flips_under_cf"], "answerable": v.get("orig", m)["answerable"],
               "features": v.get("orig", m).get("features", {}), "variants": {}}
        for var, r in v.items():
            rec["variants"][var] = {"pred": normalize(r["prediction"]), "correct": r["correct"], "answer": r["answer"]}
        out.append(rec)
    return out


# ----------------------------------------------------------------------------- scalar metrics

def _mean(xs):
    xs = list(xs)
    return float(np.mean(xs)) if xs else float("nan")


def metric_fns() -> dict[str, Callable[[list[dict]], float]]:
    """Each fn maps a list of query records to a scalar (nan if undefined)."""

    def acc(var):
        return lambda Q: _mean(q["variants"][var]["correct"] for q in Q if var in q["variants"] and q["answerable"])

    def paired(Q):
        return _mean(q["variants"]["orig"]["correct"] and q["variants"]["cf"]["correct"]
                     for q in Q if q["flips"] and {"orig", "cf"} <= q["variants"].keys())

    def cond_flip(Q):
        return _mean(q["variants"]["cf"]["correct"] for q in Q
                     if q["flips"] and {"orig", "cf"} <= q["variants"].keys() and q["variants"]["orig"]["correct"])

    def stuck(Q):
        # prediction did not move although the answer did (connectivity-insensitive behaviour)
        return _mean(q["variants"]["cf"]["pred"] == q["variants"]["orig"]["pred"]
                     for q in Q if q["flips"] and {"orig", "cf"} <= q["variants"].keys())

    def stuck_on_old(Q):
        # cf prediction equals the ORIGINAL gold answer
        return _mean(q["variants"]["cf"]["pred"] == normalize(q["variants"]["orig"]["answer"])
                     for q in Q if q["flips"] and {"orig", "cf"} <= q["variants"].keys())

    def invariance(var):
        return lambda Q: _mean(q["variants"][var]["pred"] == q["variants"]["orig"]["pred"]
                               for q in Q if {"orig", var} <= q["variants"].keys())

    def inv_correct(var):
        return lambda Q: _mean(q["variants"][var]["correct"] and q["variants"]["orig"]["correct"]
                               for q in Q if {"orig", var} <= q["variants"].keys() and q["answerable"])

    def abstain_acc(Q):
        return _mean(v["correct"] for q in Q if not q["answerable"] for v in q["variants"].values())

    def false_none(Q):
        return _mean(v["pred"] == normalize(NONE) for q in Q if q["answerable"] for v in q["variants"].values())

    def all_acc(Q):
        return _mean(v["correct"] for q in Q for v in q["variants"].values())

    return {
        "paired_cf_accuracy": paired,
        "acc_orig": acc("orig"), "acc_cf": acc("cf"), "acc_reroute": acc("reroute"), "acc_style": acc("style"),
        "acc_all_examples": all_acc,
        "cf_flip_given_orig_correct": cond_flip,
        "cf_stuck_rate": stuck, "cf_stuck_on_old_answer": stuck_on_old,
        "invariance_style": invariance("style"), "invariance_reroute": invariance("reroute"),
        "invariant_correct_style": inv_correct("style"), "invariant_correct_reroute": inv_correct("reroute"),
        "abstention_accuracy": abstain_acc, "false_none_rate": false_none,
    }


def cluster_bootstrap(Q: list[dict], fn, B: int = 2000, seed: int = 0, alpha: float = 0.05):
    groups = defaultdict(list)
    for q in Q:
        groups[q["group_id"]].append(q)
    keys = list(groups)
    point = fn(Q)
    if not keys or math.isnan(point):
        return point, (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(B):
        idx = rng.integers(0, len(keys), len(keys))
        v = fn([q for i in idx for q in groups[keys[i]]])
        if not math.isnan(v):
            vals.append(v)
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2]) if vals else (float("nan"), float("nan"))
    return point, (float(lo), float(hi))


def stratified_cluster_bootstrap(Q: list[dict], fn, strata_key=lambda q: q["template"], B: int = 2000,
                                 seed: int = 0, alpha: float = 0.05):
    """Cluster bootstrap that resamples groups *within* each stratum (e.g. generation template),
    keeping every stratum's group count fixed. Groups (base scene + all its variants and queries)
    remain the resampling unit, so counterfactual pairs are never split."""
    strata = defaultdict(lambda: defaultdict(list))
    for q in Q:
        strata[strata_key(q)][q["group_id"]].append(q)
    point = fn(Q)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(B):
        sample = []
        for groups in strata.values():
            keys = list(groups)
            for i in rng.integers(0, len(keys), len(keys)):
                sample += groups[keys[i]]
        v = fn(sample)
        if not math.isnan(v):
            vals.append(v)
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return point, (float(lo), float(hi))


def design_effect(Q: list[dict], outcome) -> dict:
    """Within-group correlation of a binary per-query outcome (one-way ANOVA ICC) and the implied
    design effect relative to treating queries as independent."""
    groups = defaultdict(list)
    for q in Q:
        o = outcome(q)
        if o is not None:
            groups[q["group_id"]].append(float(o))
    gs = [v for v in groups.values() if v]
    n = sum(len(v) for v in gs)
    k = len(gs)
    if k < 2 or n <= k:
        return {"icc": None, "design_effect": None}
    grand = sum(sum(v) for v in gs) / n
    ssb = sum(len(v) * (np.mean(v) - grand) ** 2 for v in gs)
    ssw = sum(sum((x - np.mean(v)) ** 2 for x in v) for v in gs)
    msb, msw = ssb / (k - 1), ssw / (n - k)
    m0 = (n - sum(len(v) ** 2 for v in gs) / n) / (k - 1)
    icc = (msb - msw) / (msb + (m0 - 1) * msw) if (msb + (m0 - 1) * msw) > 0 else 0.0
    mbar = n / k
    return {"icc": round(float(icc), 4), "mean_cluster_size": round(mbar, 2),
            "design_effect": round(float(1 + (mbar - 1) * icc), 3), "n_groups": k, "n_obs": n}


def summarize(Q: list[dict], B: int = 2000) -> dict:
    out = {"n_queries": len(Q), "n_groups": len({q["group_id"] for q in Q}),
           "n_flipping_queries": sum(q["flips"] for q in Q)}
    for name, fn in metric_fns().items():
        p, ci = cluster_bootstrap(Q, fn, B=B)
        out[name] = {"value": None if math.isnan(p) else round(p, 4),
                     "ci95": [None if math.isnan(x) else round(x, 4) for x in ci]}
    return out


# ----------------------------------------------------------------------------- stratification

STRATA: dict[str, Callable[[dict], str | None]] = {
    "family": lambda q: q["family"],
    "template": lambda q: q["template"],
    "query_type": lambda q: q["query_type"],
    "crossings": lambda q: (None if not q["features"] else
                            str(min(2, q["features"]["cross_connectors"] + q["features"]["cross_entities"])).replace("2", "2+")),
    "nearest_is_target": lambda q: None if not q["features"] else str(q["features"]["nearest_is_target"]),
    "connector_length": lambda q: (None if not q["features"] else
                                   "short(<150)" if q["features"]["connector_length"] < 150 else
                                   "mid(150-300)" if q["features"]["connector_length"] < 300 else "long(>=300)"),
    "n_entities": lambda q: None if not q["features"] else ("<=5" if q["features"]["n_entities"] <= 5 else ">5"),
    "text_density": lambda q: (None if not q["features"] else
                               "low(<0.006)" if q["features"]["text_density"] < 0.006 else "high"),
}


def stratified(Q: list[dict], metrics=("paired_cf_accuracy", "acc_orig", "cf_stuck_rate"), B: int = 500) -> dict:
    fns = metric_fns()
    out = {}
    for sname, key in STRATA.items():
        buckets = defaultdict(list)
        for q in Q:
            k = key(q)
            if k is not None:
                buckets[k].append(q)
        out[sname] = {}
        for k, qs in sorted(buckets.items()):
            row = {"n_queries": len(qs)}
            for m in metrics:
                p, ci = cluster_bootstrap(qs, fns[m], B=B)
                row[m] = {"value": None if math.isnan(p) else round(p, 4),
                          "ci95": [None if math.isnan(x) else round(x, 4) for x in ci]}
            out[sname][k] = row
    return out


# ----------------------------------------------------------------------------- localisation

def localization(rows: list[dict], thr: float = 0.5) -> dict:
    out = {}
    for key, gold in (("pred_annotation_bbox", "gt_annotation_bbox"), ("pred_entity_bbox", "gt_entity_bbox")):
        hits = [bbox_iou(r[key], r[gold]) >= thr for r in rows if key in r and r.get(gold) and r[key]]
        out[key] = {"n": len(hits), "hit@0.5": round(_mean(hits), 4) if hits else None}
    return out


# ----------------------------------------------------------------------------- paired comparison

def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from discordant counts b (A right, B wrong) and c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def compare(QA: list[dict], QB: list[dict], metric: str = "paired_cf_accuracy", B: int = 2000, seed: int = 0) -> dict:
    """System A vs B on the same queries: difference with cluster-bootstrap CI + exact McNemar on the
    per-query binary outcome underlying `metric` (paired_cf_accuracy or acc_orig)."""
    fa = {q["query_id"]: q for q in QA}
    fb = {q["query_id"]: q for q in QB}
    common = sorted(set(fa) & set(fb))
    fn = metric_fns()[metric]

    def outcome(q):
        if metric == "paired_cf_accuracy":
            if not (q["flips"] and {"orig", "cf"} <= q["variants"].keys()):
                return None
            return q["variants"]["orig"]["correct"] and q["variants"]["cf"]["correct"]
        if metric == "acc_orig":
            return q["variants"]["orig"]["correct"] if q["answerable"] and "orig" in q["variants"] else None
        raise ValueError(metric)

    b = c = 0
    for qid in common:
        oa, ob = outcome(fa[qid]), outcome(fb[qid])
        if oa is None or ob is None:
            continue
        b += oa and not ob
        c += ob and not oa
    groups = defaultdict(list)
    for qid in common:
        groups[fa[qid]["group_id"]].append(qid)
    keys = list(groups)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(B):
        idx = rng.integers(0, len(keys), len(keys))
        sel = [qid for i in idx for qid in groups[keys[i]]]
        da, db = fn([fa[q] for q in sel]), fn([fb[q] for q in sel])
        if not (math.isnan(da) or math.isnan(db)):
            diffs.append(da - db)
    point = fn([fa[q] for q in common]) - fn([fb[q] for q in common])
    lo, hi = np.quantile(diffs, [0.025, 0.975]) if diffs else (float("nan"), float("nan"))
    return {"metric": metric, "diff_A_minus_B_pp": round(100 * point, 2), "ci95_pp": [round(100 * lo, 2), round(100 * hi, 2)],
            "mcnemar_b": b, "mcnemar_c": c, "mcnemar_p": mcnemar_exact(b, c), "n_common_queries": len(common)}
