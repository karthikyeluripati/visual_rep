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
from collections import defaultdict
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
    framed = Image.new("RGB", (im.width, im.height + 48), (255, 255, 255))  # caption below, never over content
    framed.paste(im, (0, 0))
    ImageDraw.Draw(framed).multiline_text((6, im.height + 4), txt, font=f, fill=(0, 0, 0))
    return framed


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
            o = overlay(img, s, ex, sc, preds.get(ex["example_id"]))
            row.append(o.resize((640, int(o.height * 640 / o.width))))
        rows.append(row)
    W = 640 * max(len(r) for r in rows)
    hs = [max(im.height for im in r) for r in rows]
    sheet = Image.new("RGB", (W, sum(hs)), (255, 255, 255))
    y = 0
    for r, h in zip(rows, hs):
        for j, im in enumerate(r):
            sheet.paste(im, (640 * j, y))
        y += h
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


REVIEWER_README = """# RelBind blind audit — reviewer instructions

You will see {n} diagram images, each with one question. For each row of YOUR sheet
(`sheet_reviewer_<you>.csv`):

1. Open the image in `images/`.
2. Answer the question using ONLY the image. Write the answer exactly as printed
   (e.g. `C3`, `S2`, `4'-0"`, `4.7kΩ`), or `NONE` if the question says so and nothing is attached.
3. In `unambiguous(y/n)`, write `n` if a careful reader could reasonably give a different answer
   (lines too close, unclear endpoint, label could belong to two lines, text unreadable). Otherwise `y`.
4. Optional: `notes` for anything odd (rendering artefacts, overlapping text, etc.).

Rules: work alone; do not discuss items with the other reviewer; do not open any other files in
this folder. Expect roughly 45–75 minutes. Return only your filled sheet.
"""


def audit_package(data: Path, out: Path, n_pairs: int = 70, n_extra: int = 30, seed: int = 20261119):
    """Two-reviewer blind package. Items: n_pairs flipping queries shown on BOTH orig and cf images
    (the primary metric depends on both members being answerable), plus n_extra items drawn from
    style/reroute/unanswerable examples. Stratified over splits in proportion to their size,
    test_id-heavy. Each reviewer gets an independently shuffled order; answer key kept separate."""
    import random

    exs = load_examples(data)
    rnd = random.Random(seed)
    by_split = defaultdict(list)
    for e in exs:
        by_split[e["split"]].append(e)
    quota = {s: max(6, round(n_pairs * len(v) / len(exs))) for s, v in by_split.items()}
    items = []
    for s, es in sorted(by_split.items()):
        qids = sorted({e["query_id"] for e in es if e["flips_under_cf"]})
        rnd.shuffle(qids)
        for qid in qids[:quota[s]]:
            for var in ("orig", "cf"):
                items.append(next(e for e in es if e["query_id"] == qid and e["variant"] == var))
    pool = [e for e in exs if e["variant"] in ("style", "reroute") or not e["answerable"]]
    rnd.shuffle(pool)
    items += pool[:n_extra]
    out.mkdir(parents=True, exist_ok=False)
    (out / "images").mkdir()
    key = []
    for i, e in enumerate(items):
        name = f"item_{i:03d}.png"
        Image.open(data / e["image"]).save(out / "images" / name)
        key.append({"item": i, "image": f"images/{name}", "question": e["question"], "example_id": e["example_id"],
                    "query_id": e["query_id"], "answer": e["answer"], "split": e["split"], "variant": e["variant"],
                    "family": e["family"], "answerable": e["answerable"]})
    for reviewer in ("A", "B"):
        order = list(range(len(items)))
        random.Random(f"{seed}-{reviewer}").shuffle(order)
        with open(out / f"sheet_reviewer_{reviewer}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["item", "image", "question", "reviewer_answer", "unambiguous(y/n)", "notes"])
            for i in order:
                w.writerow([i, key[i]["image"], key[i]["question"], "", "", ""])
    (out / "README_REVIEWERS.md").write_text(REVIEWER_README.format(n=len(items)), encoding="utf-8")
    keydir = out.parent / (out.name + "_ANSWER_KEY_DO_NOT_SHARE")
    keydir.mkdir(exist_ok=False)
    (keydir / "answer_key.json").write_text(json.dumps(key, indent=1, ensure_ascii=False), encoding="utf-8")
    n_extra_used = min(n_extra, len(pool))
    print(f"wrote {len(items)} items ({(len(items) - n_extra_used) // 2} orig/cf pairs + {n_extra_used} extra) -> {out}; key -> {keydir}")


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
    a2 = sub.add_parser("audit2", help="two-reviewer blind package with paired items")
    a2.add_argument("--data", required=True)
    a2.add_argument("--pairs", type=int, default=70)
    a2.add_argument("--extra", type=int, default=30)
    a2.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    if args.cmd == "panels":
        panels(Path(args.data), args.split, args.n, Path(args.out), args.pred, args.qtype)
    elif args.cmd == "audit2":
        audit_package(Path(args.data), Path(args.out), args.pairs, args.extra)
    else:
        audit(Path(args.data), args.n, Path(args.out))


if __name__ == "__main__":
    main()
