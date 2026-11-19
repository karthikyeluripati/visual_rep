"""Build RelBind-Bench from a frozen config.

    python -m relbind.build --config configs/prototype.json --out data/prototype

Per split, per group i:
  base scene (verified) -> counterfactual swap (verified, pixel-checked) -> reroute control
  (verified, pixel-checked) -> style control (same geometry, new style).
Examples = queries about the two swapped annotations (a2e), their two original targets (e2a), and
one unannotated entity (e2a, answer NONE), asked on every variant image.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import PIL

from .counterfactual import base_style, control_style, make_reroute, make_swap
from .features import query_features
from .generation import derive_seed, generate_scene
from .generation.common import GenerationError
from .queries import NONE, answer, question
from .render import Style, render_png, render_svg
from .schema import Scene, relation_signature
from .splits import dhash, leakage_report
from .verify import pixel_diff_outside_mask


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def changed_mask(scenes: list[Scene], conn_ids: set[str], style: Style, scale: float, dilate: int = 2) -> np.ndarray:
    """Pixels that the given connectors (from any of `scenes`) may legitimately touch."""
    W, H = scenes[0].canvas
    m = np.zeros((int(round(H * scale)), int(round(W * scale))), dtype=bool)
    for s in scenes:
        only = s.copy()
        only.entities, only.annotations, only.distractors = [], [], []
        only.connectors = [c for c in s.connectors if c.id in conn_ids]
        img = np.asarray(render_png(only, style, scale), dtype=np.int16)
        m |= (np.abs(img - np.array(style.bg, dtype=np.int16)).max(-1) > 0)
    return dilate_mask(m, dilate)


def dilate_mask(m: np.ndarray, iterations: int) -> np.ndarray:
    """Binary dilation with a 3x3 square structuring element (numpy only)."""
    for _ in range(iterations):
        p = np.pad(m, 1)
        out = np.zeros_like(m)
        for dy in (0, 1, 2):
            for dx in (0, 1, 2):
                out |= p[dy:dy + m.shape[0], dx:dx + m.shape[1]]
        m = out
    return m


def build_group(split: str, idx: int, spec: dict, cfg: dict, rng_spec: np.random.Generator, max_tries: int = 80):
    # the configuration is drawn ONCE per group so that rejection sampling does not shift the
    # family/template/topology mix toward easy configurations; only the seed is retried.
    fam = str(rng_spec.choice(spec["families"]))
    tpl = str(rng_spec.choice(spec["templates"][fam]))
    topo = str(rng_spec.choice(spec["topologies"]))
    font = str(rng_spec.choice(spec["fonts"]))
    for t in range(max_tries):
        seed = derive_seed(cfg["seed"], split, idx, t)
        gid = f"{split}-{idx:05d}"
        try:
            base, rej = generate_scene(fam, tpl, topo, font, seed, f"{gid}-orig")
            vr = np.random.default_rng(derive_seed(seed, "variants"))
            cf, swapped = make_swap(base, vr)
            rr = make_reroute(base, vr, tuple(spec["topologies"]))
        except GenerationError:
            continue
        cf.scene_id, rr.scene_id = f"{gid}-cf", f"{gid}-reroute"
        st = base.copy()
        st.scene_id = f"{gid}-style"
        st.meta = {**base.meta, "variant": "style"}
        sty0 = base_style(fam)
        sty1 = control_style(sty0, vr, fam)
        base.meta.update({"variant": "orig", "group_id": gid, "split": split, "swapped": list(swapped),
                          "group_tries": t + 1})
        for s in (cf, rr, st):
            s.meta.update({"group_id": gid, "split": split, "swapped": list(swapped)})
        return gid, seed, [(base, sty0), (cf, sty0), (rr, sty0), (st, sty1)], swapped
    raise GenerationError(f"{split}/{idx}: could not build group")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None, help="cap groups per split (smoke runs)")
    ap.add_argument("--svg", action="store_true", help="also write SVG copies")
    args = ap.parse_args(argv)

    cfg = json.loads(Path(args.config).read_text())
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        sys.exit(f"refusing to overwrite non-empty {out}")
    out.mkdir(parents=True, exist_ok=True)
    scale = cfg.get("render_scale", 1.0)
    t0 = time.time()

    all_examples, image_hashes, dh, sigs = [], {}, {}, {}
    seeds = {s: set() for s in cfg["splits"]}
    stats = Counter()
    for split, spec in cfg["splits"].items():
        n = spec["n_groups"] if args.limit is None else min(args.limit, spec["n_groups"])
        sd = out / split
        (sd / "images").mkdir(parents=True)
        (sd / "scenes").mkdir()
        if args.svg:
            (sd / "svg").mkdir()
        rng_spec = np.random.default_rng(derive_seed(cfg["seed"], split, "spec"))
        examples = []
        for i in range(n):
            gid, seed, variants, swapped = build_group(split, i, spec, cfg, rng_spec)
            seeds[split].add(seed)
            base = variants[0][0]
            imgs = {}
            for s, sty in variants:
                imgs[s.scene_id] = render_png(s, sty, scale)
            # pixel-level control: cf / reroute may differ from base only where connectors changed
            (cf, _), (rr, _) = variants[1], variants[2]
            sw_conns = {c.id for c in base.connectors if c.source in swapped}
            m = changed_mask([base, cf], sw_conns, variants[0][1], scale)
            bad_cf = pixel_diff_outside_mask(imgs[base.scene_id], imgs[cf.scene_id], m)
            m2 = changed_mask([base, rr], {c.id for c in base.connectors}, variants[0][1], scale)
            bad_rr = pixel_diff_outside_mask(imgs[base.scene_id], imgs[rr.scene_id], m2)
            if bad_cf or bad_rr:
                raise RuntimeError(f"{gid}: non-semantic pixel differences (cf={bad_cf}, reroute={bad_rr})")
            stats["groups"] += 1
            stats[f"attach_fixed={cf.meta.get('attach_fixed')}"] += 1
            stats["base_attempts"] += base.meta.get("attempts", 1)

            for s, sty in variants:
                p = sd / "images" / f"{s.scene_id}.png"
                imgs[s.scene_id].save(p, optimize=False, compress_level=6)
                image_hashes[s.scene_id] = sha256_file(p)
                (sd / "scenes" / f"{s.scene_id}.json").write_text(
                    json.dumps({"scene": s.to_dict(), "style": sty.to_dict(), "render_scale": scale}, ensure_ascii=False))
                if args.svg:
                    (sd / "svg" / f"{s.scene_id}.svg").write_text(render_svg(s, sty), encoding="utf-8")
            dh[base.scene_id] = dhash(imgs[base.scene_id])
            sigs[base.scene_id] = relation_signature(base)

            a1, a2 = swapped
            e1, e2 = base.target_of(a1), base.target_of(a2)
            unannotated = sorted(e.id for e in base.entities if not base.annotations_of(e.id))
            qs = [("a2e", a1), ("a2e", a2), ("e2a", e1), ("e2a", e2)]
            if unannotated:
                qs.append(("e2a", unannotated[int(np.random.default_rng(seed).integers(0, len(unannotated)))]))
            for s, sty in variants:
                variant = s.meta["variant"]
                for qtype, ref in qs:
                    ref_text = s.annotation(ref).value if qtype == "a2e" else ref
                    ans = answer(s, qtype, ref)
                    qid = f"{gid}-{qtype}-{ref}"
                    ex = {
                        "example_id": f"{qid}-{variant}", "query_id": qid, "group_id": gid,
                        "scene_id": s.scene_id, "variant": variant, "split": split,
                        "image": f"{split}/images/{s.scene_id}.png", "image_size": list(imgs[s.scene_id].size),
                        "render_scale": scale,
                        "family": s.family, "template": s.template, "topology": s.topology, "font": s.font,
                        "style": sty.name, "attach_fixed": cf.meta.get("attach_fixed"),
                        "query_type": qtype, "query_ref": ref, "query_text": ref_text,
                        "question": question(s, qtype, ref_text), "answer": ans,
                        "answerable": ans != NONE,
                        "flips_under_cf": qtype == "a2e" or ref in (e1, e2),
                        "candidate_entities": sorted(e.id for e in s.entities),
                        "candidate_values": sorted(a.value for a in s.annotations),
                    }
                    if qtype == "a2e":
                        ex["gt_annotation_bbox"] = s.annotation(ref).label.bbox
                        ex["gt_entity_bbox"] = s.entity(ans).bbox
                        ex["features"] = query_features(s, ref)
                    else:
                        ex["gt_entity_bbox"] = s.entity(ref).bbox
                        aid = s.annotations_of(ref)
                        if aid:
                            ex["gt_annotation_bbox"] = s.annotation(aid[0]).label.bbox
                            ex["features"] = query_features(s, aid[0])
                        else:
                            ex["features"] = {}
                    examples.append(ex)
        with open(sd / "examples.jsonl", "w", encoding="utf-8") as f:
            for ex in examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        all_examples += examples
        print(f"[{split}] groups={n} examples={len(examples)} ({time.time() - t0:.0f}s)", flush=True)

    leak = leakage_report(all_examples, cfg["splits"], image_hashes, dh, seeds, sigs)
    (out / "leakage_report.json").write_text(json.dumps(leak, indent=2))
    commit = os.environ.get("RELBIND_GIT_COMMIT")
    if not commit:
        try:
            commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                    cwd=Path(__file__).parent).stdout.strip() or None
        except Exception:
            commit = None
    manifest = {
        "config": cfg, "config_sha256": hashlib.sha256(Path(args.config).read_bytes()).hexdigest(),
        "limit": args.limit, "git_commit": commit,
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pillow": PIL.__version__,
                     "platform": platform.platform()},
        "stats": dict(stats), "n_examples": len(all_examples),
        "image_sha256": image_hashes,
        "examples_sha256": {s: sha256_file(out / s / "examples.jsonl") for s in cfg["splits"]},
        "leakage_ok": leak["ok"], "build_seconds": round(time.time() - t0, 1),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: manifest[k] for k in ("stats", "n_examples", "leakage_ok", "build_seconds")}, indent=2))
    if not leak["ok"]:
        print("LEAKAGE ERRORS:", leak["errors"][:10])
        sys.exit(1)


if __name__ == "__main__":
    main()
