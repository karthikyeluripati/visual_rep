"""Real-diagram external-validity set (e.g. US patent drawings, public domain) with REAL-IMAGE strict counterfactuals.

Human labels (one JSON per image; schema in data/real/README_LABELLERS.md):
    {"image": "img/US1234567_fig2.png", "source": "<url>", "license": "public domain (US patent)",
     "callouts": [{"text": "12", "text_bbox": [x0,y0,x1,y1], "end": [x,y]}, ...]}      # pixel coordinates
`end` = the point where the callout's leader line touches the part it designates.

Item construction (deterministic, seeded; no human judgement after labelling):
  for each pair (a, b) of callouts with similar text-box sizes:
    orig image  : 4 lettered markers (A-D) at the `end` points of a, b and two other callouts (letters shuffled).
    cf image    : same markers, but the text patches of a and b are swapped (pixels only; leader lines untouched),
                  so "which marker does the leader line from label <a.text> end at?" flips from end(a) to end(b).
  Queries: a2e for a.text and b.text on both images -> a strict pair (identical lines, markers, candidates).
Scoring: pair accuracy (both images correct), same definition as the synthetic benchmark.

    python -m relbind.real build --labels data/real/labels --images data/real --out data/real/items
    python -m relbind.real eval  --items data/real/items --model Qwen/Qwen3-VL-4B-Instruct --out experiments/real/<tag>
"""
from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

LETTERS = "ABCD"
QUESTION = ('Four lettered markers (A, B, C, D) are drawn on this technical drawing. Which marker is at the end of the '
            'leader line that starts at the reference label "{t}"? Answer with the marker letter.')
FORMAT = 'Reply with a single JSON object on the last line: {"answer": "<letter>"}.'


def _font(size):
    return ImageFont.truetype(str(Path(__file__).parent / "assets" / "fonts" / "DejaVuSans.ttf"), size)  # vendored


def draw_markers(img, pts, letters, r):
    im = img.convert("RGB").copy()
    d = ImageDraw.Draw(im)
    f = _font(int(r * 1.3))
    for (x, y), L in zip(pts, letters):
        d.ellipse([x - r, y - r, x + r, y + r], outline=(220, 0, 0), width=max(2, r // 5), fill=(255, 255, 255))
        d.text((x, y), L, fill=(220, 0, 0), font=f, anchor="mm")
    return im


def swap_patches(img, b1, b2):
    """Swap the pixels of two text boxes (each patch resized to the other's box). Returns a new image."""
    im = img.convert("RGB").copy()
    b1 = tuple(int(round(v)) for v in b1)
    b2 = tuple(int(round(v)) for v in b2)
    p1, p2 = im.crop(b1), im.crop(b2)
    bg = Image.new("RGB", im.size, (255, 255, 255))
    im.paste(bg.crop(b1), b1[:2])
    im.paste(bg.crop(b2), b2[:2])
    im.paste(p2.resize((b1[2] - b1[0], b1[3] - b1[1]), Image.LANCZOS), b1[:2])
    im.paste(p1.resize((b2[2] - b2[0], b2[3] - b2[1]), Image.LANCZOS), b2[:2])
    return im


def _similar(b1, b2, tol=0.35):
    w1, h1, w2, h2 = b1[2] - b1[0], b1[3] - b1[1], b2[2] - b2[0], b2[3] - b2[1]
    return abs(w1 - w2) <= tol * max(w1, w2) and abs(h1 - h2) <= tol * max(h1, h2)


def build(labels_dir, images_dir, out_dir, seed=0, max_pairs_per_image=3):
    out = Path(out_dir)
    if out.exists():
        sys.exit(f"refusing to overwrite {out}")
    (out / "images").mkdir(parents=True)
    rng = random.Random(seed)
    exs, skipped = [], []
    for lf in sorted(Path(labels_dir).glob("*.json")):
        lab = json.loads(lf.read_text(encoding="utf-8"))
        cs = lab["callouts"]
        texts = [c["text"] for c in cs]
        if len(cs) < 4 or len(set(texts)) != len(texts):
            skipped.append({"labels": lf.name, "reason": "fewer than 4 callouts or duplicate label texts"})
            continue
        img = Image.open(Path(images_dir) / lab["image"])
        r = max(10, int(min(img.size) * 0.018))
        pairs = [(i, j) for i, j in itertools.combinations(range(len(cs)), 2) if _similar(cs[i]["text_bbox"], cs[j]["text_bbox"])]
        rng.shuffle(pairs)
        for k, (i, j) in enumerate(pairs[:max_pairs_per_image]):
            gid = f"real-{lf.stem}-{k}"
            others = [m for m in range(len(cs)) if m not in (i, j)]
            pick = [i, j] + rng.sample(others, 2)
            letters = list(LETTERS)
            rng.shuffle(letters)
            letter_of = dict(zip(pick, letters))
            pts = [cs[m]["end"] for m in pick]
            orig = draw_markers(img, pts, [letter_of[m] for m in pick], r)
            cf = draw_markers(swap_patches(img, cs[i]["text_bbox"], cs[j]["text_bbox"]), pts, [letter_of[m] for m in pick], r)
            for var, im in (("orig", orig), ("cf", cf)):
                im.save(out / "images" / f"{gid}-{var}.png")
            for a, b in ((i, j), (j, i)):
                qid = f"{gid}-a2e-{a}"
                for var in ("orig", "cf"):
                    ans = letter_of[a] if var == "orig" else letter_of[b]  # in cf, text a sits where b's text was
                    exs.append({"example_id": f"{qid}-{var}", "query_id": qid, "group_id": gid, "variant": var,
                                "split": "real", "image": f"images/{gid}-{var}.png", "query_type": "a2e",
                                "query_text": cs[a]["text"], "question": QUESTION.format(t=cs[a]["text"]),
                                "answer": ans, "candidate_entities": list(LETTERS), "candidate_values": [],
                                "flips_under_cf": True, "source_labels": lf.name, "source": lab.get("source"),
                                "license": lab.get("license"), "crossing_cell": "unknown"})
    with open(out / "examples.jsonl", "w", encoding="utf-8") as f:
        for e in exs:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    (out / "build_info.json").write_text(json.dumps({"seed": seed, "n_examples": len(exs), "n_groups": len({e["group_id"] for e in exs}),
                                                     "skipped": skipped}, indent=2), encoding="utf-8")
    return exs


def evaluate(items_dir, model, revision, out_dir, bs=8):
    from .metrics import is_correct
    from .vlm.backends import QwenVLBackend
    from .vlm.parse import raw_answer, snap
    items, out = Path(items_dir), Path(out_dir)
    if out.exists():
        sys.exit(f"refusing to overwrite {out}")
    out.mkdir(parents=True)
    exs = [json.loads(l) for l in open(items / "examples.jsonl", encoding="utf-8")]
    be = QwenVLBackend(model, revision)
    n = c = 0
    with open(out / "predictions.jsonl", "w", encoding="utf-8") as f:
        for k in range(0, len(exs), bs):
            ch = exs[k:k + bs]
            texts = be.generate([(Image.open(items / e["image"]).convert("RGB"), f"{e['question']}\n{FORMAT}") for e in ch], 64)
            for e, t in zip(ch, texts):
                ans, d = raw_answer(t)
                p = snap(ans, e["candidate_entities"])
                n += 1
                c += is_correct(p, e["answer"])
                f.write(json.dumps({"example_id": e["example_id"], "prediction": p, "raw_answer": ans,
                                    "parsed_json": bool(d), "raw_text": t}, ensure_ascii=False) + "\n")
    (out / "run_info.json").write_text(json.dumps({"backend": be.info(), "n": n, "acc_examples": round(c / max(1, n), 4)}, indent=2),
                                       encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--labels", required=True)
    b.add_argument("--images", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--seed", type=int, default=0)
    e = sub.add_parser("eval")
    e.add_argument("--items", required=True)
    e.add_argument("--model", required=True)
    e.add_argument("--revision", default=None)
    e.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "build":
        exs = build(a.labels, a.images, a.out, a.seed)
        print(f"{len(exs)} examples")
    else:
        evaluate(a.items, a.model, a.revision, a.out)


if __name__ == "__main__":
    main()
