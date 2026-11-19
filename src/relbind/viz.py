"""Grounding visualisation and human-audit material.

    # side-by-side orig/cf/reroute/style panels with GT overlays (and predictions if given)
    python -m relbind.viz panels --data data/prototype --split test_id --n 8 --out results/figures/panels_test_id.png

    # blind audit sheet: images + questions WITHOUT answers, plus a CSV for the reviewer and a
    # separate answer key used only by `python -m relbind.validation score`
    python -m relbind.viz audit --data data/prototype --n 40 --out data/validation/audit_v0
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .fonts import get_font
from .metrics import load_examples, load_jsonl
from .baselines.heuristics import load_scene

GREEN, BLUE, RED, ORANGE = (0, 160, 60), (30, 90, 220), (220, 30, 30), (240, 140, 0)


def overlay(img: Image.Image, scene, ex: dict, scale: float, pred: str | None = None) -> Image.Image:
    im = img.convert("RGB").copy()
    d = ImageDraw.Draw(im)

    def box(b, col, w=3):
        d.rectangle([b[0] * scale, b[1] * scale, b[2] * scale, b[3] * scale], outline=col, width=w)

    if ex.get("gt_annotation_bbox"):
        box(ex["gt_annotation_bbox"], GREEN)
    if ex.get("gt_entity_bbox"):
        box(ex["gt_entity_bbox"], BLUE)
    aid = ex["query_ref"] if ex["query_type"] == "a2e" else (scene.annotations_of(ex["query_ref"]) or [None])[0]
    if aid:
        c = next(c for c in scene.connectors if c.source == aid)
        d.line([(p[0] * scale, p[1] * scale) for p in c.points], fill=RED, width=2)
    if pred is not None and ex["query_type"] == "a2e":
        try:
            box(scene.entity(pred).bbox, ORANGE, 2)
        except KeyError:
            pass
    f = get_font("sans", 16)
    txt = f"[{ex['variant']}] {ex['question']}\nGOLD: {ex['answer']}" + (f"   PRED: {pred}" if pred is not None else "")
    d.rectangle([0, im.height - 44, im.width, im.height], fill=(255, 255, 255))
    d.multiline_text((6, im.height - 42), txt, font=f, fill=(0, 0, 0))
    return im


def panels(data: Path, split: str, n: int, out: Path, pred_file: str | None = None, qtype: str = "a2e"):
    exs = [e for e in load_examples(data, [split]) if e["query_type"] == qtype]
    preds = {p["example_id"]: p["prediction"] for p in load_jsonl(pred_file)} if pred_file else {}
    qids = sorted({e["query_id"] for e in exs})[:n]
    rows = []
    for qid in qids:
        row = []
        for var in ("orig", "cf", "reroute", "style"):
            ex = next((e for e in exs if e["query_id"] == qid and e["variant"] == var), None)
            if ex is None:
                continue
            s, _, sc = load_scene(data, ex)
            img = Image.open(data / ex["image"])
            row.append(overlay(img, s, ex, sc, preds.get(ex["example_id"])).resize((512, 360)))
        rows.append(row)
    W = 512 * max(len(r) for r in rows)
    sheet = Image.new("RGB", (W, 360 * len(rows)), (255, 255, 255))
    for i, r in enumerate(rows):
        for j, im in enumerate(r):
            sheet.paste(im, (512 * j, 360 * i))
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    print(f"wrote {out}")


def audit(data: Path, n: int, out: Path, seed: int = 0):
    """Blind audit: one random variant per sampled query, stratified over splits."""
    exs = load_examples(data)
    rng = np.random.default_rng(seed)
    by_split = {}
    for e in exs:
        by_split.setdefault(e["split"], []).append(e)
    per = max(1, n // len(by_split))
    chosen = []
    for split, es in sorted(by_split.items()):
        idx = rng.permutation(len(es))[:per]
        chosen += [es[i] for i in idx]
    out.mkdir(parents=True, exist_ok=True)
    (out / "images").mkdir(exist_ok=True)
    with open(out / "audit_sheet.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["item", "image", "question", "reviewer_answer", "unambiguous(y/n)", "notes"])
        for i, e in enumerate(chosen):
            name = f"item_{i:03d}.png"
            Image.open(data / e["image"]).save(out / "images" / name)
            w.writerow([i, f"images/{name}", e["question"], "", "", ""])
    key = [{"item": i, "example_id": e["example_id"], "answer": e["answer"], "split": e["split"],
            "variant": e["variant"], "family": e["family"]} for i, e in enumerate(chosen)]
    (out / "answer_key.json").write_text(json.dumps(key, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {len(chosen)} audit items -> {out} (give reviewers audit_sheet.csv + images/ only)")


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("panels")
    p.add_argument("--data", required=True)
    p.add_argument("--split", required=True)
    p.add_argument("--n", type=int, default=6)
    p.add_argument("--pred", default=None)
    p.add_argument("--qtype", default="a2e")
    p.add_argument("--out", required=True)
    a = sub.add_parser("audit")
    a.add_argument("--data", required=True)
    a.add_argument("--n", type=int, default=42)
    a.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    if args.cmd == "panels":
        panels(Path(args.data), args.split, args.n, Path(args.out), args.pred, args.qtype)
    else:
        audit(Path(args.data), args.n, Path(args.out))


if __name__ == "__main__":
    main()
