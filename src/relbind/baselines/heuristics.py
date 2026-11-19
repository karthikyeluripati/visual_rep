"""B0 — non-learned baselines.

random       : uniform over the candidate answers (entity ids, or values + NONE).
text_prior   : most frequent training answer given (family, query_type, query_text); falls back to the
               most frequent answer for (family, query_type). Detects text-only shortcuts.
proximity    : [oracle layout] the entity geometrically nearest to the annotation (a2e), or the
               annotation nearest to the entity (e2a). Tests the proximity shortcut.
cc_tracer    : [oracle boxes, pixels] flood-fill dark pixels from the annotation, with all text boxes
               and entity strokes masked out; predict the touched entity whose contact is farthest
               from the annotation. Represents "connected-component" reasoning; it fails where strokes
               merge (crossings, wires) and where a leader crosses an entity line.

Usage:
    python -m relbind.baselines.heuristics --data data/prototype --method proximity --out results/b0/proximity.jsonl
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict, deque
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .. import geometry as G
from ..features import entity_distance
from ..metrics import load_examples
from ..queries import NONE
from ..render import Style
from ..schema import Scene


def load_scene(data: Path, ex: dict) -> tuple[Scene, Style, float]:
    d = json.loads((data / ex["split"] / "scenes" / f"{ex['scene_id']}.json").read_text(encoding="utf-8"))
    return Scene.from_dict(d["scene"]), Style.from_dict(d["style"]), d["render_scale"]


# ----------------------------------------------------------------------------- random / text prior

def predict_random(examples, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for ex in examples:
        cands = ex["candidate_entities"] if ex["query_type"] == "a2e" else ex["candidate_values"] + [NONE]
        out.append({"example_id": ex["example_id"], "prediction": cands[int(rng.integers(0, len(cands)))]})
    return out


def predict_text_prior(examples, train_examples):
    by_text = defaultdict(Counter)
    by_type = defaultdict(Counter)
    for ex in train_examples:
        by_text[(ex["family"], ex["query_type"], ex["query_text"])][ex["answer"]] += 1
        by_type[(ex["family"], ex["query_type"])][ex["answer"]] += 1
    out = []
    for ex in examples:
        c = by_text.get((ex["family"], ex["query_type"], ex["query_text"])) or by_type.get((ex["family"], ex["query_type"]))
        pred = c.most_common(1)[0][0] if c else NONE
        out.append({"example_id": ex["example_id"], "prediction": pred})
    return out


# ----------------------------------------------------------------------------- proximity

def _proximity_map(scene: Scene) -> dict[str, str]:
    """annotation id -> nearest entity id."""
    m = {}
    for a in scene.annotations:
        p = G.bbox_center(a.label.bbox)
        m[a.id] = min(scene.entities, key=lambda e: (entity_distance(scene, e, p), e.id)).id
    return m


def _answer_from_map(scene: Scene, ex: dict, amap: dict[str, str]) -> str:
    if ex["query_type"] == "a2e":
        return amap.get(ex["query_ref"], NONE)
    hits = [aid for aid, eid in amap.items() if eid == ex["query_ref"]]
    if not hits:
        return NONE
    if len(hits) > 1:  # several annotations claim this entity: pick the closest one
        e = scene.entity(ex["query_ref"])
        hits.sort(key=lambda aid: entity_distance(scene, e, G.bbox_center(scene.annotation(aid).label.bbox)))
    return scene.annotation(hits[0]).value


def predict_proximity(data: Path, examples):
    out = []
    cache = {}
    for ex in examples:
        if ex["scene_id"] not in cache:
            s, _, _ = load_scene(data, ex)
            cache[ex["scene_id"]] = (s, _proximity_map(s))
        s, amap = cache[ex["scene_id"]]
        out.append({"example_id": ex["example_id"], "prediction": _answer_from_map(s, ex, amap)})
    return out


# ----------------------------------------------------------------------------- pixel tracer

def _masks(scene: Scene, style: Style, scale: float, shape):
    H, W = shape
    block = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(block)
    texts = [a.label.bbox for a in scene.annotations] + [e.label.bbox for e in scene.entities if e.label]
    texts += [p["bbox"] for e in scene.entities for p in e.attrs.get("draw", []) if p["type"] == "text"]
    texts += [p["bbox"] for p in scene.distractors if p["type"] == "text"]
    for b in texts:
        bb = G.inflate(b, 2)
        d.rectangle([bb[0] * scale, bb[1] * scale, bb[2] * scale, bb[3] * scale], fill=255)
    ent_masks = {}
    for e in scene.entities:
        m = Image.new("L", (W, H), 0)
        md = ImageDraw.Draw(m)
        g = e.geometry
        if g["type"] == "segment":
            md.line([(p[0] * scale, p[1] * scale) for p in g["points"]], fill=255, width=max(3, int(style.w_entity * scale + 4)))
            for sb in e.attrs.get("solids", []):
                bb = G.inflate(sb, 2)
                md.ellipse([bb[0] * scale, bb[1] * scale, bb[2] * scale, bb[3] * scale], fill=255)
        else:
            bb = G.inflate(g["bbox"], 2)
            md.rectangle([bb[0] * scale, bb[1] * scale, bb[2] * scale, bb[3] * scale], fill=255)
        ent_masks[e.id] = np.asarray(m) > 0
    return np.asarray(block) > 0, ent_masks


def _flood(passable: np.ndarray, seeds: np.ndarray) -> np.ndarray:
    H, W = passable.shape
    seen = np.zeros_like(passable)
    q = deque()
    for y, x in zip(*np.nonzero(seeds & passable)):
        seen[y, x] = True
        q.append((y, x))
    while q:
        y, x = q.popleft()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                ny, nx = y + dy, x + dx
                if 0 <= ny < H and 0 <= nx < W and passable[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    q.append((ny, nx))
    return seen


def _dilate(m: np.ndarray, it: int) -> np.ndarray:
    for _ in range(it):
        p = np.pad(m, 1)
        o = np.zeros_like(m)
        for dy in (0, 1, 2):
            for dx in (0, 1, 2):
                o |= p[dy:dy + m.shape[0], dx:dx + m.shape[1]]
        m = o
    return m


def tracer_map(img: Image.Image, scene: Scene, style: Style, scale: float, dark: int = 170) -> tuple[dict, dict]:
    """annotation id -> entity id predicted by the connected-component tracer, plus diagnostics."""
    gray = np.asarray(img.convert("L"))
    ink = gray < dark
    block, ent = _masks(scene, style, scale, gray.shape)
    all_ent = np.zeros_like(ink)
    for m in ent.values():
        all_ent |= m
    passable = ink & ~block & ~all_ent
    out, diag = {}, {}
    for a in scene.annotations:
        b = a.label.bbox
        ring_out = np.zeros_like(ink)
        # seeds: unmasked ink within 7 px of the annotation box (the leader starts <= 4 px away)
        y0, y1 = int((b[1] - 7) * scale), int((b[3] + 7) * scale)
        x0, x1 = int((b[0] - 7) * scale), int((b[2] + 7) * scale)
        ring_out[max(0, y0):y1, max(0, x0):x1] = True
        region = _flood(passable, ring_out)
        grown = _dilate(region, 2)
        touched = {}
        cx, cy = G.bbox_center(b)
        for eid, m in ent.items():
            hit = grown & m
            if hit.any():
                ys, xs = np.nonzero(hit)
                touched[eid] = float(np.max(np.hypot(xs / scale - cx, ys / scale - cy)))
        diag[a.id] = {"touched": sorted(touched)}
        if touched:
            out[a.id] = max(touched, key=touched.get)
    return out, diag


def predict_tracer(data: Path, examples):
    out = []
    cache = {}
    for ex in examples:
        if ex["scene_id"] not in cache:
            s, st, sc = load_scene(data, ex)
            img = Image.open(data / ex["image"]).convert("RGB")
            amap, diag = tracer_map(img, s, st, sc)
            cache[ex["scene_id"]] = (s, amap, diag)
        s, amap, diag = cache[ex["scene_id"]]
        out.append({"example_id": ex["example_id"], "prediction": _answer_from_map(s, ex, amap),
                    "n_touched": len(diag.get(ex["query_ref"], {}).get("touched", [])) if ex["query_type"] == "a2e" else None})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--method", required=True, choices=["random", "text_prior", "proximity", "cc_tracer"])
    ap.add_argument("--splits", nargs="*", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    data = Path(args.data)
    exs = load_examples(data, args.splits)
    if args.method == "random":
        preds = predict_random(exs)
    elif args.method == "text_prior":
        preds = predict_text_prior(exs, load_examples(data, ["train"]))
    elif args.method == "proximity":
        preds = predict_proximity(data, exs)
    else:
        preds = predict_tracer(data, exs)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for p in preds:
            f.write(json.dumps({**p, "method": args.method}, ensure_ascii=False) + "\n")
    print(f"wrote {len(preds)} predictions -> {args.out}")


if __name__ == "__main__":
    main()
