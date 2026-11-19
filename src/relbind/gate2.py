"""Gate-2 analysis (Phase 1, inference only). Produces results/phase1/gate2_analysis.json.

    python -m relbind.gate2 --data /workspace/data/diag_v1 --diag experiments/phase1_diag \
        --heur results/b0_diag --out results/phase1/gate2_analysis.json

Sections
  1 recognition      crop-probe accuracy; group-level "recognition OK" flag
  2 categories       per wrong answer: invalid / recognition / unsupported relation /
                     connector interpretation / proximity binding / other  (+ overlap tags)
  3 recog_ok         relational metrics restricted to groups whose 4 relevant texts were all read correctly
  4 pair_strata      PCA by proximity change, crossing change, crossings, attach_fixed, family;
                     zigzag split vs test_id; unlinked (abstention) behaviour
  5 ci_checks        group bootstrap vs template-stratified bootstrap, per-template estimates, ICC/design effect
  6 f_audit          condition F errors vs F_text_only (blank image) and vs A answers
  7 heuristics       paired comparison of VLM conditions with proximity / CC tracer / ray tracer
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from . import geometry as G
from .baselines.heuristics import load_scene
from .features import entity_distance, proximity_ranking
from .metrics import (cluster_bootstrap, compare, design_effect, is_correct, join, load_examples, load_jsonl,
                      metric_fns, normalize, query_table, stratified_cluster_bootstrap)
from .queries import NONE

SPLIT = "test_id"


def _ci(Q, fn, B=2000):
    p, ci = cluster_bootstrap(Q, fn, B=B)
    return {"value": None if math.isnan(p) else round(p, 4), "ci95": [round(x, 4) for x in ci] if not math.isnan(ci[0]) else None,
            "n_queries": len(Q), "n_groups": len({q["group_id"] for q in Q})}


# ----------------------------------------------------------------------------- recognition

def recognition(probes):
    """group -> {('ann', value)|('ent', id): correct} from crop probes."""
    out = defaultdict(dict)
    for p in probes:
        tag = p["example_id"].split("#")[1]  # read_ann_crop
        _, kind, mode = tag.split("_")
        if mode != "crop":
            continue
        gid = p["example_id"].split("-a2e-")[0]
        out[gid][(kind, normalize(p["gold"]))] = p["correct"]
    return out


def group_recog_ok(rec, ex_by_group):
    """True iff the two swapped annotations' texts and their two targets' labels were read correctly."""
    ok = {}
    for gid, exs in ex_by_group.items():
        r = rec.get(gid)
        if not r:
            ok[gid] = None
            continue
        need = set()
        for e in exs:
            if e["query_type"] == "a2e" and e["variant"] == "orig":
                need.add(("ann", normalize(e["query_text"])))
                need.add(("ent", normalize(e["answer"])))
        ok[gid] = all(r.get(k, False) for k in need) if need else None
    return ok


# ----------------------------------------------------------------------------- categories

def classify(ex, pred, scene, recog_ok, orig_gold):
    """Primary category + tags for one wrong answer."""
    tags = set()
    gold = ex["answer"]
    if ex["query_type"] == "a2e":
        if pred not in ex["candidate_entities"]:
            return ("invalid_output", tags)
        if recog_ok is False:
            return ("recognition", tags)
        annotated = {r.entity for r in scene.relations}
        c = next(c for c in scene.connectors if c.source == ex["query_ref"])
        other_targets = {x.target for x in scene.connectors if x.id != c.id}
        crossed = set()
        for e in scene.entities:
            if e.id == c.target:
                continue
            if e.crossable:
                if any(G.seg_intersection(p, q, r, s) for p, q in G.segments(c.points) for r, s in G.segments(e.geometry["points"])):
                    crossed.add(e.id)
            elif G.polyline_intersects_bbox(c.points, G.inflate(e.bbox, 6)):
                crossed.add(e.id)
        nearest = proximity_ranking(scene, ex["query_ref"])[0]
        if ex["variant"] == "cf" and pred == orig_gold:
            tags.add("stuck_on_old_target")
        if pred == nearest:
            tags.add("proximity_nearest")
        if pred in other_targets:
            tags.add("target_of_other_connector")
        if pred in crossed:
            tags.add("crossed_by_gold_connector")
        if pred not in annotated:
            return ("unsupported_relation", tags)
        if tags & {"target_of_other_connector", "crossed_by_gold_connector"}:
            return ("connector_interpretation", tags)
        if "proximity_nearest" in tags:
            return ("proximity_binding", tags)
        return ("other", tags)
    # e2a
    values = {normalize(a.value): a for a in scene.annotations}
    np_ = normalize(pred)
    if gold == NONE:
        if np_ in values:
            return ("unsupported_relation", {"named_other_annotation"})
        return ("unsupported_relation", {"named_non_annotation_text"} if np_ else {"empty"})
    if np_ == normalize(NONE):
        return ("missed_relation", tags)
    if recog_ok is False:
        return ("recognition", tags)
    if np_ not in values:
        return ("unsupported_relation", {"named_non_annotation_text"})
    a = values[np_]
    ent = scene.entity(ex["query_ref"])
    near_ann = min(scene.annotations, key=lambda x: entity_distance(scene, ent, G.bbox_center(x.label.bbox)))
    if ex["variant"] == "cf" and pred == orig_gold:
        tags.add("stuck_on_old_answer")
    if a.id == near_ann.id:
        tags.add("proximity_nearest")
    c = next(c for c in scene.connectors if c.source == a.id)
    if c.target != ex["query_ref"]:
        tags.add("annotation_of_other_connector")
    if "proximity_nearest" in tags and "stuck_on_old_answer" not in tags:
        return ("proximity_binding", tags)
    return ("connector_interpretation", tags)


def categories(data, rows, recog, orig_gold, variants=("orig", "cf")):
    prim = defaultdict(Counter)
    tags = defaultdict(Counter)
    n = Counter()
    cache = {}
    for r in rows:
        if r["split"] != SPLIT or r["variant"] not in variants:
            continue
        key = (r["query_type"], r["answerable"])
        n[key] += 1
        if r["correct"]:
            continue
        if r["scene_id"] not in cache:
            cache[r["scene_id"]] = load_scene(data, r)[0]
        cat, tg = classify(r, r["prediction"] or "", cache[r["scene_id"]],
                           recog.get(r["group_id"]), orig_gold.get(r["query_id"]))
        prim[key][cat] += 1
        for t in tg:
            tags[key][t] += 1
    out = {}
    for key in sorted(n):
        nm = f"{key[0]}_{'answerable' if key[1] else 'unanswerable'}"
        wrong = sum(prim[key].values())
        out[nm] = {"n_examples": n[key], "n_wrong": wrong,
                   "primary": {k: {"n": v, "frac_of_wrong": round(v / wrong, 3)} for k, v in prim[key].most_common()},
                   "tags": dict(tags[key])}
    return out


# ----------------------------------------------------------------------------- main

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--diag", required=True)
    ap.add_argument("--heur", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--B", type=int, default=2000)
    args = ap.parse_args(argv)
    data, diag = Path(args.data), Path(args.diag)
    exs = load_examples(data)
    by_group = defaultdict(list)
    for e in exs:
        by_group[e["group_id"]].append(e)
    orig_gold = {e["query_id"]: e["answer"] for e in exs if e["variant"] == "orig"}
    probes = load_jsonl(diag / "P_read" / "predictions.jsonl")
    rec = recognition(probes)
    rok = group_recog_ok(rec, by_group)
    fns = metric_fns()
    res = {"split": SPLIT, "data": str(data)}

    # 1 recognition
    pk = defaultdict(lambda: [0, 0])
    for p in probes:
        tag = p["example_id"].split("#")[1]
        sp = p["example_id"].split("-")[0]
        pk[(sp, tag)][0] += 1
        pk[(sp, tag)][1] += p["correct"]
    res["recognition"] = {
        "probe_accuracy": {f"{s}/{t}": round(c / n, 4) for (s, t), (n, c) in sorted(pk.items())},
        "groups_recog_ok": {s: f"{sum(1 for g, v in rok.items() if v and g.startswith(s + '-'))}/"
                               f"{sum(1 for g in rok if g.startswith(s + '-'))}"
                            for s in sorted({g.rsplit('-', 1)[0] for g in rok})},
    }

    conds = [c.name for c in sorted(diag.iterdir()) if (c / "predictions.jsonl").exists()
             and c.name not in ("P_read",) and "INVALID" not in c.name]
    rows = {c: join(exs, load_jsonl(diag / c / "predictions.jsonl")) for c in conds}
    Qs = {c: query_table([r for r in rows[c] if r["split"] == SPLIT]) for c in conds}

    # 2 categories
    res["categories"] = {c: categories(data, rows[c], rok, orig_gold) for c in conds
                         if c in ("A_direct", "B_reasoning", "C_oracle_crop", "E_oracle_box", "F_oracle_graph")}

    # 3 recognition-conditional relational metrics
    res["recog_ok_subset"] = {}
    for c in conds:
        Q = Qs[c]
        Qok = [q for q in Q if rok.get(q["group_id"]) is True]
        Qbad = [q for q in Q if rok.get(q["group_id"]) is False]
        res["recog_ok_subset"][c] = {
            "recog_ok": {m: _ci(Qok, fns[m], args.B) for m in ("paired_cf_accuracy", "acc_orig", "acc_cf", "cf_stuck_rate")},
            "recog_failed": {m: _ci(Qbad, fns[m], args.B) for m in ("paired_cf_accuracy", "acc_orig")},
        }

    # 4 pair strata (a2e flipping queries)
    def a2e(Q):
        return [q for q in Q if q["query_type"] == "a2e"]

    feats = defaultdict(dict)
    for e in exs:
        if e["split"] == SPLIT and e["query_type"] == "a2e":
            feats[e["query_id"]][e["variant"]] = e.get("features", {})
    attach = {e["group_id"]: e["attach_fixed"] for e in exs}

    def strata_fn(q):
        f = feats.get(q["query_id"], {})
        o, c = f.get("orig", {}), f.get("cf", {})
        if not o or not c:
            return {}
        xo = o["cross_connectors"] + o["cross_entities"]
        xc = c["cross_connectors"] + c["cross_entities"]
        return {
            "proximity_change": f"nearest_is_target orig={o['nearest_is_target']} cf={c['nearest_is_target']}",
            "crossing_change": "cf adds crossings" if xc > xo else "cf removes crossings" if xc < xo else "same crossings",
            "max_crossings_in_pair": str(min(2, max(xo, xc))).replace("2", "2+"),
            "attach_fixed": str(attach.get(q["group_id"])),
            "family": q["family"],
        }

    res["pair_strata"] = {}
    for c in ("A_direct", "B_reasoning", "C_oracle_crop", "E_oracle_box"):
        if c not in Qs:
            continue
        buckets = defaultdict(lambda: defaultdict(list))
        for q in a2e(Qs[c]):
            for k, v in strata_fn(q).items():
                buckets[k][v].append(q)
        res["pair_strata"][c] = {k: {v: _ci(qs, fns["paired_cf_accuracy"], 1000) | {"stuck": round(fns["cf_stuck_rate"](qs), 4)}
                                     for v, qs in sorted(d.items())} for k, d in buckets.items()}
    # zigzag / OOD vs test_id (independent samples: bootstrap each, report difference of point estimates + CIs)
    res["split_comparison_A"] = {}
    if "A_direct" in rows:
        for sp in ("test_id", "test_unseen_topology", "test_unseen_template", "test_cross_domain"):
            Q = query_table([r for r in rows["A_direct"] if r["split"] == sp])
            if Q:
                res["split_comparison_A"][sp] = _ci(Q, fns["paired_cf_accuracy"], args.B)
    # unlinked (abstention) behaviour
    res["unlinked_queries"] = {}
    scache = {}
    for c in conds:
        cnt = Counter()
        for r in rows[c]:
            if r["split"] != SPLIT or r["answerable"]:
                continue
            p = normalize(r["prediction"])
            if p == normalize(NONE):
                cnt["NONE (correct)"] += 1
                continue
            if r["scene_id"] not in scache:
                scache[r["scene_id"]] = load_scene(data, r)[0]
            s = scache[r["scene_id"]]
            vals = {normalize(a.value) for a in s.annotations}
            dtexts = {normalize(d["text"]): d.get("role", "text") for d in s.distractors if d["type"] == "text"}
            ents = {normalize(e.id) for e in s.entities}
            if p in vals:
                cnt["named an annotation linked elsewhere"] += 1
            elif p in dtexts:
                cnt[f"named unlinked text ({dtexts[p]})"] += 1
            elif p in ents:
                cnt["named an entity id"] += 1
            else:
                cnt["other text"] += 1
        res["unlinked_queries"][c] = dict(cnt.most_common())

    # 5 CI checks
    ci = {}
    for c in ("A_direct", "C_oracle_crop", "E_oracle_box", "F_oracle_graph", "B_reasoning"):
        if c not in Qs:
            continue
        Q = Qs[c]
        f = fns["paired_cf_accuracy"]
        g = cluster_bootstrap(Q, f, B=args.B)
        st = stratified_cluster_bootstrap(Q, f, B=args.B)
        per_t = {t: _ci([q for q in Q if q["template"] == t], f, args.B) for t in sorted({q["template"] for q in Q})}
        iid_se = None
        flips = [q for q in Q if q["flips"] and {"orig", "cf"} <= q["variants"].keys()]
        if flips:
            pp = g[0]
            iid_se = math.sqrt(pp * (1 - pp) / len(flips))
        de = design_effect(Q, lambda q: (q["variants"]["orig"]["correct"] and q["variants"]["cf"]["correct"])
                           if q["flips"] and {"orig", "cf"} <= q["variants"].keys() else None)
        ci[c] = {"group_bootstrap": [round(x, 4) for x in g[1]], "template_stratified_bootstrap": [round(x, 4) for x in st[1]],
                 "naive_iid_ci_halfwidth": round(1.96 * iid_se, 4) if iid_se else None,
                 "group_bootstrap_halfwidth": round((g[1][1] - g[1][0]) / 2, 4),
                 "per_template": per_t, "within_group_icc": de}
    seeds = [e for e in exs if e["split"] == SPLIT]
    ci["_sampling_design"] = {
        "resampling_unit": "group = one base scene + its cf/reroute/style variants and all 5 queries",
        "n_groups_test_id": len({e["group_id"] for e in seeds}),
        "templates_test_id": dict(Counter(e["template"] for e in seeds if e["variant"] == "orig" and e["query_type"] == "a2e")),
        "pairs_split_by_resampling": 0,
        "note": "Groups are generated from distinct derived seeds (manifest) and are independent draws. Only two "
                "templates exist in test_id (each tied to one family), so template-level resampling cannot support "
                "claims about new templates; that is what test_unseen_template measures.",
    }
    res["ci_checks"] = ci

    # 6 F audit
    if "F_oracle_graph" in rows and "F_text_only" in rows:
        fi = {r["example_id"]: r for r in rows["F_oracle_graph"] if r["split"] == SPLIT}
        ft = {r["example_id"]: r for r in rows["F_text_only"] if r["split"] == SPLIT}
        fa = {r["example_id"]: r for r in rows.get("A_direct", []) if r["split"] == SPLIT}
        tab = Counter()
        cat = Counter()
        examples = defaultdict(list)
        cache = {}
        for eid, r in fi.items():
            t = ft.get(eid)
            if t is None:
                continue
            tab[(r["correct"], t["correct"])] += 1
            if r["correct"]:
                continue
            if r["scene_id"] not in cache:
                cache[r["scene_id"]] = load_scene(data, r)[0]
            s = cache[r["scene_id"]]
            listed_vals = {normalize(a.value) for a in s.annotations}
            listed_ents = {rel.entity for rel in s.relations}
            p = normalize(r["prediction"])
            a_same = eid in fa and normalize(fa[eid]["prediction"]) == p
            if not t["correct"] and normalize(t["prediction"]) == p:
                k = "same error without image (task/format misunderstanding)"
            elif t["correct"] and a_same:
                k = "image-dependent: matches the model's no-oracle answer (consistent with visual-text conflict)"
            elif t["correct"]:
                k = "image-dependent: differs from no-oracle answer"
            else:
                k = "wrong both ways, different answers"
            if r["answer"] == NONE:
                sub = "gold NONE (entity not in list)"
            elif r["query_type"] == "a2e":
                sub = "a2e: pred is a listed target" if r["prediction"] in listed_ents else "a2e: pred not a listed target"
            else:
                sub = "e2a: pred is a listed value" if p in listed_vals else "e2a: pred not a listed value"
            cat[(k, sub)] += 1
            if len(examples[k]) < 4:
                examples[k].append({"example_id": eid, "question": r["question"], "gold": r["answer"],
                                    "F": r["prediction"], "F_text_only": t["prediction"], "A": fa.get(eid, {}).get("prediction")})
        res["f_audit"] = {
            "F_vs_textonly_correctness": {f"F={'ok' if a else 'wrong'}, text_only={'ok' if b else 'wrong'}": v
                                          for (a, b), v in sorted(tab.items())},
            "F_textonly_metrics": {m: _ci(Qs["F_text_only"], fns[m], args.B) for m in ("paired_cf_accuracy", "acc_orig")},
            "F_error_breakdown": {f"{k} | {s}": v for (k, s), v in cat.most_common()},
            "examples": examples,
        }

    # 7 heuristics vs VLM
    res["heuristics"] = {}
    hq = {}
    for h in ("proximity", "text_prior", "cc_tracer", "ray_tracer"):
        f = Path(args.heur) / f"{h}.jsonl"
        if f.exists():
            hq[h] = query_table([r for r in join(exs, load_jsonl(f)) if r["split"] == SPLIT])
            res["heuristics"][h] = {m: _ci(hq[h], fns[m], args.B) for m in ("paired_cf_accuracy", "acc_orig", "cf_stuck_rate")}
    res["vlm_vs_heuristics"] = {}
    for c in ("A_direct", "B_reasoning", "C_oracle_crop", "E_oracle_box"):
        if c in Qs:
            res["vlm_vs_heuristics"][c] = {h: compare(Qs[c], hq[h], "paired_cf_accuracy", B=args.B) for h in hq}
    # all conditions, headline
    res["headline"] = {c: {m: _ci(Qs[c], fns[m], args.B) for m in ("paired_cf_accuracy", "acc_orig", "acc_cf", "cf_stuck_rate",
                                                                  "invariance_style", "invariance_reroute", "abstention_accuracy")}
                       for c in conds}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
