"""Run a diagnostic condition over RelBind-Bench examples.

    # CPU smoke test of the full harness (no model):
    python -m relbind.vlm.run --data data/prototype --splits test_id --condition A_direct \
        --backend mock --mock-from results/b0/proximity.jsonl --out experiments/smoke_mock

    # Real run (Phase 1, requires gate approval):
    python -m relbind.vlm.run --data data/prototype --splits test_id --condition A_direct \
        --backend qwen --model Qwen/Qwen3-VL-4B-Instruct --revision <commit> --out experiments/diag_A_test_id

Writes predictions.jsonl (with raw model text) and run_info.json (env, model revision, timing).
Never overwrites an existing output directory.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from collections import defaultdict
from pathlib import Path

from PIL import Image

from ..baselines.heuristics import load_scene
from ..metrics import is_correct, load_examples, load_jsonl
from .backends import MockBackend, QwenVLBackend
from .parse import parse, raw_answer
from .prompts import CONDITIONS, build_prompt


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--splits", nargs="*", default=None)
    ap.add_argument("--condition", required=True, choices=CONDITIONS)
    ap.add_argument("--backend", default="qwen", choices=["qwen", "mock"])
    ap.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--max-pixels", type=int, default=None)
    ap.add_argument("--mock-from", default=None)
    ap.add_argument("--variants", nargs="*", default=None, help="restrict to these variants")
    ap.add_argument("--max-groups", type=int, default=None, help="first N groups per split (sorted)")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=None)
    ap.add_argument("--ask-boxes", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    out = Path(args.out)
    if out.exists():
        sys.exit(f"refusing to overwrite {out}")
    out.mkdir(parents=True)
    data = Path(args.data)
    exs = load_examples(data, args.splits)
    if args.variants:
        exs = [e for e in exs if e["variant"] in args.variants]
    if args.max_groups:
        keep = defaultdict(set)
        for g in sorted({(e["split"], e["group_id"]) for e in exs}):
            if len(keep[g[0]]) < args.max_groups:
                keep[g[0]].add(g[1])
        exs = [e for e in exs if e["group_id"] in keep[e["split"]]]
    mnt = args.max_new_tokens or (768 if args.condition == "B_reasoning" else 96)
    if args.condition == "P_read":
        # recognition probes on original images: the queried annotation text and the gold target's label
        probes = []
        for ex in exs:
            if ex["variant"] != "orig" or ex["query_type"] != "a2e":
                continue
            s, _, _ = load_scene(data, ex)
            ann = s.annotation(ex["query_ref"])
            ent = s.entity(ex["answer"])
            for kind, b, gold in (("ann", ann.label.bbox, ann.value), ("ent", ent.label.bbox, ent.id)):
                for mode in ("crop", "box"):
                    probes.append({**ex, "example_id": f"{ex['example_id']}#read_{kind}_{mode}", "probe_bbox": b,
                                   "probe_mode": mode, "answer": gold, "query_type": "read",
                                   "candidate_entities": [], "candidate_values": [gold]})
        exs = probes

    if args.backend == "mock":
        mock = {p["example_id"]: p["prediction"] for p in load_jsonl(args.mock_from)}
        backend = MockBackend(mock)
    else:
        backend = QwenVLBackend(args.model, args.revision, max_pixels=args.max_pixels)

    scene_cache: dict = {}

    def scene_of(ex):
        if ex["scene_id"] not in scene_cache:
            scene_cache[ex["scene_id"]] = load_scene(data, ex)
        return scene_cache[ex["scene_id"]]

    def group_scenes(ex):
        gid = ex["group_id"]
        res = []
        for var in ("orig", "cf"):
            res.append(scene_of({**ex, "scene_id": f"{gid}-{var}"})[0])
        return res

    t0 = time.time()
    n_read = [0, 0]
    with open(out / "predictions.jsonl", "w", encoding="utf-8") as f:
        for i in range(0, len(exs), args.batch_size):
            chunk = exs[i:i + args.batch_size]
            prompts = []
            for ex in chunk:
                s, _, sc = scene_of(ex)
                img = Image.open(data / ex["image"]).convert("RGB")
                prompts.append(build_prompt(args.condition, ex, s, img, sc, group_scenes(ex), args.ask_boxes))
            if isinstance(backend, MockBackend):
                backend.set_keys([ex["example_id"] for ex in chunk])
            texts = backend.generate([(p.image, p.text) for p in prompts], mnt)
            for ex, p, txt in zip(chunk, prompts, texts):
                _, _, sc = scene_of(ex)
                if ex["query_type"] == "read":
                    ans, _ = raw_answer(txt)
                    parsed = {"prediction": ans, "gold": ex["answer"], "correct": is_correct(ans, ex["answer"])}
                    n_read[0] += 1
                    n_read[1] += parsed["correct"]
                else:
                    parsed = parse(txt, ex, p.image.width, p.image.height, p.crop_box,
                                   scale=1.0 if p.crop_box else sc)
                f.write(json.dumps({"example_id": ex["example_id"], **parsed, "raw_text": txt,
                                    "condition": args.condition}, ensure_ascii=False) + "\n")
            print(f"{min(i + args.batch_size, len(exs))}/{len(exs)} ({time.time() - t0:.0f}s)", flush=True)
    info = {"argv": sys.argv, "args": vars(args), "n_examples": len(exs), "seconds": round(time.time() - t0, 1),
            "python": platform.python_version(), "backend": backend.info(), "max_new_tokens": mnt}
    if n_read[0]:
        info["read_accuracy"] = round(n_read[1] / n_read[0], 4)
    (out / "run_info.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
