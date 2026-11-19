"""Final Gate-2 decomposition from preserved artifacts (no scenes/images needed).

    python -m relbind.final_analysis --bundle pod_results/<stamp> --out results/phase1/final_analysis.json

Every rate is reported as {num, den, value, ci95}, with a 95% cluster bootstrap over generation groups
(a group = base scene + all its variants and queries, so counterfactual pairs are never split).
Metrics that the experiments did not measure are reported as "unavailable" with a reason.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .metrics import is_correct, normalize

B = 2000


def load_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def rate(items, B=B, seed=0):
    """items: list of (group_id, bool). Ratio estimator with cluster bootstrap over groups."""
    if not items:
        return {"num": 0, "den": 0, "value": None, "ci95": None}
    g = defaultdict(lambda: [0, 0])
    for gid, ok in items:
        g[gid][0] += int(ok)
        g[gid][1] += 1
    arr = np.array(list(g.values()), dtype=float)
    num, den = int(arr[:, 0].sum()), int(arr[:, 1].sum())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), (B, len(arr)))
    s = arr[idx]
    boots = s[:, :, 0].sum(1) / np.maximum(1, s[:, :, 1].sum(1))
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return {"num": num, "den": den, "value": round(num / den, 4), "ci95": [round(float(lo), 4), round(float(hi), 4)]}


def paired_diff(items, B=B, seed=0):
    """items: (group, a_ok, b_ok) on the same units. Difference in pp + cluster-bootstrap CI + exact McNemar."""
    from .metrics import mcnemar_exact
    g = defaultdict(lambda: [0, 0, 0])
    b = c = 0
    for gid, a, bb in items:
        g[gid][0] += int(a)
        g[gid][1] += int(bb)
        g[gid][2] += 1
        b += a and not bb
        c += bb and not a
    arr = np.array(list(g.values()), dtype=float)
    rng = np.random.default_rng(seed)
    s = arr[rng.integers(0, len(arr), (B, len(arr)))]
    d = (s[:, :, 0].sum(1) - s[:, :, 1].sum(1)) / np.maximum(1, s[:, :, 2].sum(1))
    lo, hi = np.quantile(d, [0.025, 0.975])
    pt = (arr[:, 0].sum() - arr[:, 1].sum()) / arr[:, 2].sum()
    return {"diff_pp": round(100 * pt, 1), "ci95_pp": [round(100 * lo, 1), round(100 * hi, 1)],
            "discordant_a_only": int(b), "discordant_b_only": int(c), "mcnemar_p": float(f"{mcnemar_exact(int(b), int(c)):.3g}"),
            "n": len(items)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    bd = Path(args.bundle)
    data = bd / "data" / "diag_v1"
    rep = bd / "visual_rep"
    exs = []
    for f in sorted(data.glob("*/examples.jsonl")):
        exs += load_jsonl(f)
    E = {e["example_id"]: e for e in exs}
    conds = {}
    for c in ("A_direct", "B_reasoning", "C_oracle_crop", "D_oracle_ocr", "E_oracle_box", "F_oracle_graph", "F_text_only"):
        p = rep / "experiments" / "phase1_diag" / c / "predictions.jsonl"
        if p.exists():
            conds[c] = {r["example_id"]: r["prediction"] for r in load_jsonl(p)}
    for h in ("proximity", "text_prior", "cc_tracer", "ray_tracer"):
        p = rep / "results" / "b0_diag" / f"{h}.jsonl"
        if p.exists():
            conds["H_" + h] = {r["example_id"]: r["prediction"] for r in load_jsonl(p)}
    probes = load_jsonl(rep / "experiments" / "phase1_diag" / "P_read" / "predictions.jsonl")

    # ------------------------------------------------------------ recognition tables
    # probe id: <orig a2e example id>#read_<ann|ent>_<crop|box>; gold = annotation text / target entity id
    read = defaultdict(dict)  # (group, kind, mode) -> {gold_norm: correct}
    probe_items = defaultdict(list)
    for p in probes:
        base, tag = p["example_id"].split("#")
        _, kind, mode = tag.split("_")
        gid = E[base]["group_id"]
        read[(gid, kind, mode)][normalize(p["gold"])] = p["correct"]
        probe_items[(E[base]["split"], kind, mode)].append((gid, p["correct"]))
    res = {"recognition": {}}
    for (sp, kind, mode), items in sorted(probe_items.items()):
        name = {"ann": "text_recognition(annotation)", "ent": "entity_label_recognition"}[kind]
        res["recognition"][f"{sp}/{name}/{mode}"] = rate(items)
    res["recognition"]["_note"] = ("crop = text region alone (pure recognition); box = locate a 0-1000 box on the full "
                                   "image and read it (localisation+reading). Probes were run on ORIGINAL images only. "
                                   "Entity recognition is measured only as reading the entity's label; visual recognition "
                                   "of the entity's shape/stroke is NOT measured (unavailable).")

    def rec_ok(gid, ann_text=None, ent_ids=(), mode="crop"):
        ok = True
        if ann_text is not None:
            v = read[(gid, "ann", mode)].get(normalize(ann_text))
            if v is None:
                return None
            ok &= v
        for e in ent_ids:
            v = read[(gid, "ent", mode)].get(normalize(e))
            if v is None:
                return None
            ok &= v
        return ok

    # per-query records (a2e flipping queries), all variants
    q = defaultdict(dict)
    for e in exs:
        q[e["query_id"]][e["variant"]] = e
    a2e_q = {k: v for k, v in q.items() if next(iter(v.values()))["query_type"] == "a2e" and {"orig", "cf"} <= v.keys()}

    def correct(c, e):
        return is_correct(conds[c].get(e["example_id"]), e["answer"])

    def has(c, *es):
        """A condition contributes an item only if it produced predictions for all involved examples
        (B/E/F were run on test_id only; missing predictions must not count as errors)."""
        return all(e["example_id"] in conds[c] for e in es)

    # ------------------------------------------------------------ decomposition A-E (condition A + others)
    dec = {}
    for c in [k for k in conds if not k.startswith("F_text")]:
        out = {}
        for sp in sorted({e["split"] for e in exs}):
            C_items, D_a2e, D_e2a, E_items, Eall = [], [], [], [], []
            for qid, v in a2e_q.items():
                o, f = v["orig"], v["cf"]
                if o["split"] != sp or not has(c, o, f):
                    continue
                gid = o["group_id"]
                both = rec_ok(gid, o["query_text"], (o["answer"], f["answer"]))
                for var in ("orig", "cf"):
                    ex = v[var]
                    r1 = rec_ok(gid, ex["query_text"], (ex["answer"],))
                    if r1:
                        C_items.append((gid, correct(c, ex)))
                    D_a2e.append((gid, correct(c, ex)))
                pair_ok = correct(c, o) and correct(c, f)
                Eall.append((gid, pair_ok))
                if both:
                    E_items.append((gid, pair_ok))
            for e in exs:
                if e["split"] == sp and e["query_type"] == "e2a" and e["answerable"] and e["variant"] in ("orig", "cf") and has(c, e):
                    D_e2a.append((e["group_id"], correct(c, e)))
            if not D_a2e:
                out[sp] = "unavailable: condition not run on this split"
                continue
            out[sp] = {
                "C_relational_accuracy_given_text_and_target_label_read": rate(C_items),
                "D_end_to_end_a2e_accuracy": rate(D_a2e),
                "D_end_to_end_value_extraction_e2a_accuracy": rate(D_e2a),
                "E_pair_accuracy_given_recognition_both_targets": rate(E_items),
                "pair_accuracy_a2e_all": rate(Eall),
            }
        dec[c] = out
    res["decomposition"] = dec
    res["decomposition_notes"] = {
        "C": "a2e examples (orig+cf) whose annotation text and gold-target label were read correctly in the crop probe.",
        "E": "recognition measured on the ORIGINAL image only. The text and label pixels of the cf image are identical "
             "(pixel check in build), so this is a proxy for recognition in both images, not a direct measurement.",
        "connector_interpretation": "UNAVAILABLE as a measured quantity: no probe asks the model to trace or describe the "
                                    "connector. The connector-related categories in error_taxonomy_final.md are "
                                    "post-hoc labels of the WRONG ANSWER (e.g. it is the end of another connector), "
                                    "not observations of what the model perceived.",
    }

    # ------------------------------------------------------------ strata (a2e pair accuracy), all conditions
    def strata_key(v):
        o, f = v["orig"], v["cf"]
        fo, fc = o.get("features", {}), f.get("features", {})
        xo = fo.get("cross_connectors", 0) + fo.get("cross_entities", 0)
        xc = fc.get("cross_connectors", 0) + fc.get("cross_entities", 0)
        return {
            "split": o["split"],
            "topology(route class)": o["topology"],
            "swap_type": "strict (arrowhead positions identical)" if o["attach_fixed"] else "non-strict (arrowheads moved)",
            "max_crossings_in_pair": "0" if max(xo, xc) == 0 else "1" if max(xo, xc) == 1 else "2+",
            "nearby_distractor": ("target is the nearest entity in neither image" if not (fo.get("nearest_is_target") or fc.get("nearest_is_target"))
                                  else "target is nearest in one image"),
            "clutter(entities)": "<=5 entities" if fo.get("n_entities", 0) <= 5 else ">5 entities",
            "clutter(distractor strokes)": "<=4" if fo.get("n_distractor_strokes", 0) <= 4 else ">4",
            "family": o["family"],
            "template": o["template"],
        }

    strata = defaultdict(lambda: defaultdict(dict))
    keys_by_q = {qid: strata_key(v) for qid, v in a2e_q.items()}
    for c in conds:
        buckets = defaultdict(list)
        for qid, v in a2e_q.items():
            if not has(c, v["orig"], v["cf"]):
                continue
            ok = correct(c, v["orig"]) and correct(c, v["cf"])
            for k, val in keys_by_q[qid].items():
                if k != "split" and v["orig"]["split"] != "test_id" and k not in ("topology(route class)",):
                    continue
                buckets[(k, val)].append((v["orig"]["group_id"], ok))
        for (k, val), items in buckets.items():
            strata[k][val][c] = rate(items, B=1000)
    res["strata_pair_accuracy_a2e"] = strata
    res["strata_notes"] = "All strata except 'split' and 'topology' are restricted to test_id."

    # strict-only headline (the shortcut-proof subset)
    res["strict_only_test_id"] = {}
    for c in conds:
        items = [(v["orig"]["group_id"], correct(c, v["orig"]) and correct(c, v["cf"]))
                 for v in a2e_q.values() if v["orig"]["split"] == "test_id" and v["orig"]["attach_fixed"]
                 and has(c, v["orig"], v["cf"])]
        res["strict_only_test_id"][c] = rate(items)

    # crossing type x swap type (test_id + unseen topology), a2e pair accuracy
    ct = defaultdict(lambda: defaultdict(list))
    for v in a2e_q.values():
        o, f = v["orig"], v["cf"]
        if o["split"] not in ("test_id", "test_unseen_topology"):
            continue
        fo, fc = o["features"], f["features"]
        cc = max(fo["cross_connectors"], fc["cross_connectors"]) > 0
        ce = max(fo["cross_entities"], fc["cross_entities"]) > 0
        key = (f"{o['split']} | " + ("leaders cross each other" if cc else "no leader-leader crossing") + " / "
               + ("crosses entity line" if ce else "no entity-line crossing"))
        for c in conds:
            if has(c, o, f):
                ct[key][c].append((o["group_id"], correct(c, o) and correct(c, f)))
    res["crossing_type_pair_accuracy_a2e"] = {k: {c: rate(it, B=1000) for c, it in d.items()} for k, d in sorted(ct.items())}

    # ------------------------------------------------------------ style / reroute (meaning-preserving)
    sty = {}
    for c in [k for k in conds if not k.startswith("F_text")]:
        sty[c] = {}
        for var in ("style", "reroute"):
            items_inv, d_items = [], []
            for qid, v in q.items():
                o = v.get("orig")
                if not o or o["split"] != "test_id" or var not in v or not o["answerable"] or not has(c, o, v[var]):
                    continue
                po, pv = conds[c].get(o["example_id"]), conds[c].get(v[var]["example_id"])
                items_inv.append((o["group_id"], normalize(po) == normalize(pv)))
                d_items.append((o["group_id"], correct(c, o), correct(c, v[var])))
            sty[c][var] = {"answer_unchanged": rate(items_inv), "acc_orig_minus_acc_variant": paired_diff(d_items)}
    res["meaning_preserving_controls_test_id"] = sty

    # ------------------------------------------------------------ unlinked entities (abstention queries)
    unl = {}
    for c in conds:
        items = [(e["group_id"], correct(c, e)) for e in exs
                 if e["split"] == "test_id" and not e["answerable"] and e["variant"] in ("orig", "cf") and has(c, e)]
        unl[c] = rate(items)
    res["unlinked_entity_abstention_accuracy_test_id"] = unl

    # ------------------------------------------------------------ tracer vs VLM paired comparisons
    cmp = {}
    for c in ("A_direct", "B_reasoning", "E_oracle_box", "F_oracle_graph"):
        for h in ("H_ray_tracer", "H_proximity"):
            for sub, filt in (("test_id_all", lambda v: v["orig"]["split"] == "test_id"),
                              ("test_id_strict", lambda v: v["orig"]["split"] == "test_id" and v["orig"]["attach_fixed"]),
                              ("unseen_topology", lambda v: v["orig"]["split"] == "test_unseen_topology"),
                              ("cross_domain", lambda v: v["orig"]["split"] == "test_cross_domain")):
                items = [(v["orig"]["group_id"], correct(c, v["orig"]) and correct(c, v["cf"]),
                          correct(h, v["orig"]) and correct(h, v["cf"])) for v in a2e_q.values()
                         if filt(v) and has(c, v["orig"], v["cf"]) and has(h, v["orig"], v["cf"])]
                if items:
                    cmp[f"{c} vs {h} | {sub}"] = paired_diff(items)
                else:
                    cmp[f"{c} vs {h} | {sub}"] = "unavailable: condition not run on this split"
    res["paired_comparisons_a2e_pair_accuracy"] = cmp

    # tracer failure modes (a2e): no entity reached vs wrong entity
    tf = Counter()
    for v in a2e_q.values():
        for var in ("orig", "cf"):
            ex = v[var]
            p = conds["H_ray_tracer"].get(ex["example_id"])
            tf[(ex["split"], "correct" if is_correct(p, ex["answer"]) else "no entity reached" if p in (None, "NONE") else "wrong entity")] += 1
    res["ray_tracer_outcomes_a2e"] = {f"{s}/{k}": n for (s, k), n in sorted(tf.items())}

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
