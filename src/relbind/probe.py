"""Frozen-feature linkage probe (Decisive Experiment 1b). The base VLM is NEVER updated.

extract : one forward pass per image (fixed, question-free prompt), hidden states at the visual-token
          positions, pooled over each annotation box and each entity box, at 5 depths
          (0 = merger output / LM input, L/4, L/2, 3L/4, L).
fit     : linear candidate scorer s(a,e) = w·[h_a*h_e, |h_a-h_e|, g] + b, softmax over the image's
          entities, trained on the train split. Layer and weight decay are selected on val by pair accuracy.
          Reports strict-pair accuracy on the test splits, plus shuffled-label and geometry-only controls.

    python -m relbind.probe extract --data D --splits train val test_id test_unseen_topology --out F
    python -m relbind.probe fit --data D --feats F --out results/exp1/probe.json

Pre-registered reading (research/phase2_go_no_go.md §2): test pair accuracy on unseen topology >= 0.60
=> connectivity is linearly available (read-out deficit); <= 0.30 => not available (representation
deficit); otherwise inconclusive. A linear probe on pooled features is a LOWER bound on what is encoded.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from .baselines.heuristics import load_scene
from .metrics import load_examples

PROMPT = "Describe this diagram."
CELL = 32  # merged visual token = 2x2 patches of 16 px


def _token_boxes(bbox, sx, sy, H2, W2, pad=2.0):
    x0, y0, x1, y1 = (bbox[0] - pad) * sx, (bbox[1] - pad) * sy, (bbox[2] + pad) * sx, (bbox[3] + pad) * sy
    c0, c1 = max(0, int(x0 // CELL)), min(W2 - 1, int(x1 // CELL))
    r0, r1 = max(0, int(y0 // CELL)), min(H2 - 1, int(y1 // CELL))
    return r0, r1, c0, c1


def extract(args):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    data = Path(args.data)
    proc = AutoProcessor.from_pretrained(args.model, revision=args.revision)
    proc.tokenizer.padding_side = "left"
    model = AutoModelForImageTextToText.from_pretrained(args.model, revision=args.revision, dtype=torch.bfloat16,
                                                        attn_implementation="sdpa", device_map="cuda").eval()
    img_tok = getattr(model.config, "image_token_id", None) or proc.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    for split in args.splits:
        exs = [e for e in load_examples(data, [split]) if e["query_type"] == "a2e" and e["variant"] in ("orig", "cf")]
        scenes = {}
        for e in exs:
            scenes.setdefault(e["scene_id"], e)
        ids = sorted(scenes)
        feats, meta = {}, []
        layers = None
        for i in range(0, len(ids), args.batch_size):
            chunk = ids[i:i + args.batch_size]
            imgs, scs = [], []
            for sid in chunk:
                ex = scenes[sid]
                s, _, sc = load_scene(data, ex)
                imgs.append(Image.open(data / ex["image"]).convert("RGB"))
                scs.append((s, sc, ex))
            msgs = [[{"role": "user", "content": [{"type": "image", "image": im}, {"type": "text", "text": PROMPT}]}] for im in imgs]
            inp = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True,
                                           return_tensors="pt", padding=True).to(model.device)
            with torch.inference_mode():
                hs = model(**inp, output_hidden_states=True).hidden_states
            L = len(hs) - 1
            layers = layers or [0, L // 4, L // 2, (3 * L) // 4, L]
            for b, (s, sc, ex) in enumerate(scs):
                t, h, w = inp["image_grid_thw"][b].tolist()
                H2, W2 = h // 2, w // 2
                pos = (inp["input_ids"][b] == img_tok).nonzero().flatten()
                assert len(pos) == H2 * W2, (len(pos), H2, W2)
                grid = torch.stack([hs[l][b, pos] for l in layers]).float().reshape(len(layers), H2, W2, -1)
                W0, H0 = imgs[b].size
                sx, sy = (w * 16) / W0 * sc, (h * 16) / H0 * sc  # canonical -> resized pixels

                def pool(bbox):
                    r0, r1, c0, c1 = _token_boxes(bbox, sx, sy, H2, W2)
                    return grid[:, r0:r1 + 1, c0:c1 + 1].mean(dim=(1, 2)).cpu().numpy().astype(np.float16)

                sid = s.scene_id
                feats[f"{sid}|g"] = grid.mean(dim=(1, 2)).cpu().numpy().astype(np.float16)
                for a in s.annotations:
                    feats[f"{sid}|a|{a.id}"] = pool(a.label.bbox)
                for e in s.entities:
                    feats[f"{sid}|e|{e.id}"] = pool(e.bbox)
                meta.append({"scene_id": sid, "group_id": ex["group_id"], "variant": ex["variant"], "split": split,
                             "crossing_cell": ex.get("crossing_cell"), "swapped": s.meta.get("swapped"),
                             "annotations": [a.id for a in s.annotations], "entities": [e.id for e in s.entities],
                             "gold": {a.id: s.target_of(a.id) for a in s.annotations},
                             "ann_centers": {a.id: [(a.label.bbox[0] + a.label.bbox[2]) / 2, (a.label.bbox[1] + a.label.bbox[3]) / 2] for a in s.annotations},
                             "ent_boxes": {e.id: e.bbox for e in s.entities}, "canvas": s.canvas})
            print(f"[{split}] {min(i + args.batch_size, len(ids))}/{len(ids)}", flush=True)
        np.savez_compressed(out / f"{split}.npz", **feats)
        (out / f"{split}_meta.json").write_text(json.dumps({"layers": layers, "images": meta}), encoding="utf-8")


# ----------------------------------------------------------------------------- fit

def _build(split_dir, split, layer_idx, geometry=False):
    meta = json.loads((split_dir / f"{split}_meta.json").read_text(encoding="utf-8"))
    F = None if geometry else np.load(split_dir / f"{split}.npz")
    rows, queries = [], []  # queries: (start ROW offset, n_cands, gold_index, scene meta, ann id)
    offset = 0
    for m in meta["images"]:
        sid = m["scene_id"]
        if not geometry:
            g = F[f"{sid}|g"][layer_idx].astype(np.float32)
            E = np.stack([F[f"{sid}|e|{e}"][layer_idx].astype(np.float32) for e in m["entities"]])
        W, H = m["canvas"]
        for a in m["annotations"]:
            if geometry:
                ax, ay = m["ann_centers"][a]
                X = []
                for e in m["entities"]:
                    b = m["ent_boxes"][e]
                    ex_, ey_ = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
                    dx, dy = (ex_ - ax) / W, (ey_ - ay) / H
                    X.append([ax / W, ay / H, ex_ / W, ey_ / H, dx, dy, abs(dx), abs(dy), (dx * dx + dy * dy) ** 0.5])
                X = np.array(X, dtype=np.float32)
            else:
                h = F[f"{sid}|a|{a}"][layer_idx].astype(np.float32)
                X = np.concatenate([h[None] * E, np.abs(h[None] - E), np.repeat(g[None], len(E), 0)], 1)
            queries.append((offset, len(m["entities"]), m["entities"].index(m["gold"][a]), m, a))
            rows.append(X)
            offset += len(X)
    return np.concatenate(rows, 0), queries


def _train(X, Q, wd, epochs=300, lr=1e-2, seed=0, shuffle=False):
    import torch
    rng = np.random.default_rng(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Xt = torch.tensor(X, device=dev)
    w = torch.zeros(X.shape[1], device=dev, requires_grad=True)
    b0 = torch.zeros(1, device=dev, requires_grad=True)
    golds = [q[2] if not shuffle else int(rng.integers(0, q[1])) for q in Q]
    starts = torch.tensor([q[0] for q in Q], device=dev)
    ns = [q[1] for q in Q]
    seg = torch.repeat_interleave(torch.arange(len(Q), device=dev), torch.tensor(ns, device=dev))
    gold_rows = torch.tensor([q[0] + gi for q, gi in zip(Q, golds)], device=dev)
    opt = torch.optim.Adam([w, b0], lr=lr, weight_decay=0)
    for _ in range(epochs):
        opt.zero_grad()
        s = Xt @ w + b0
        mx = torch.zeros(len(Q), device=dev).scatter_reduce(0, seg, s, reduce="amax", include_self=False)
        lse = torch.log(torch.zeros(len(Q), device=dev).index_add(0, seg, torch.exp(s - mx[seg]))) + mx
        loss = (lse - s[gold_rows]).mean() + wd * (w * w).sum()
        loss.backward()
        opt.step()
    return w.detach().cpu().numpy(), float(b0.detach().cpu())


def _predict(X, Q, w, b):
    s = X @ w + b
    return [int(np.argmax(s[q[0]:q[0] + q[1]])) for q in Q]


def _pair_acc(Q, pred, only_swapped=True):
    """Pair accuracy over swapped annotations: correct on both orig and cf images of the group."""
    ok = defaultdict(dict)
    single = []
    for q, p in zip(Q, pred):
        m, a = q[3], q[4]
        c = p == q[2]
        single.append(c)
        if only_swapped and a not in (m["swapped"] or []):
            continue
        ok[(m["group_id"], a)][m["variant"]] = (c, m.get("crossing_cell"))
    pairs = [(v["orig"][0] and v["cf"][0], v["orig"][1], k[0]) for k, v in ok.items() if {"orig", "cf"} <= v.keys()]
    by_cell = defaultdict(list)
    for c, cell, _ in pairs:
        by_cell[cell].append(c)
    return {"pair_acc": round(float(np.mean([p[0] for p in pairs])), 4) if pairs else None, "n_pairs": len(pairs),
            "single_acc": round(float(np.mean(single)), 4),
            "by_cell": {k: {"pair_acc": round(float(np.mean(v)), 4), "n": len(v)} for k, v in sorted(by_cell.items(), key=lambda x: str(x[0]))},
            "pairs": [(g, bool(c)) for c, _, g in pairs]}


def _boot(pairs, B=2000, seed=0):
    g = defaultdict(list)
    for gid, c in pairs:
        g[gid].append(c)
    arr = [(sum(v), len(v)) for v in g.values()]
    a = np.array(arr, dtype=float)
    rng = np.random.default_rng(seed)
    s = a[rng.integers(0, len(a), (B, len(a)))]
    v = s[:, :, 0].sum(1) / s[:, :, 1].sum(1)
    return [round(float(x), 4) for x in np.quantile(v, [0.025, 0.975])]


def fit(args):
    fd = Path(args.feats)
    meta = json.loads((fd / "train_meta.json").read_text(encoding="utf-8"))
    layers = meta["layers"]
    tests = [s for s in args.test_splits]
    res = {"layers": layers, "selection": [], "note": __doc__.split("Pre-registered")[1].strip()}
    best = None
    for li, L in enumerate(layers):
        Xtr, Qtr = _build(fd, "train", li)
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
        Xva, Qva = _build(fd, "val", li)
        for wd in (1e-4, 1e-3, 1e-2):
            w, b = _train((Xtr - mu) / sd, Qtr, wd)
            va = _pair_acc(Qva, _predict((Xva - mu) / sd, Qva, w, b))
            res["selection"].append({"layer": L, "wd": wd, "val_pair_acc": va["pair_acc"], "val_single_acc": va["single_acc"]})
            print(res["selection"][-1], flush=True)
            if best is None or va["pair_acc"] > best[0]:
                best = (va["pair_acc"], li, wd, w, b, mu, sd)
    _, li, wd, w, b, mu, sd = best
    res["selected"] = {"layer": layers[li], "wd": wd}
    Xtr, Qtr = _build(fd, "train", li)
    ws, bs = _train((Xtr - mu) / sd, Qtr, wd, shuffle=True)
    Xg, Qg = _build(fd, "train", li, geometry=True)
    mug, sdg = Xg.mean(0), Xg.std(0) + 1e-6
    wg, bg = _train((Xg - mug) / sdg, Qg, 1e-3)
    res["test"] = {}
    for sp in tests:
        X, Q = _build(fd, sp, li)
        r = _pair_acc(Q, _predict((X - mu) / sd, Q, w, b))
        r["pair_acc_ci95"] = _boot(r.pop("pairs"))
        rs = _pair_acc(Q, _predict((X - mu) / sd, Q, ws, bs))
        rs.pop("pairs")
        Xg2, Qg2 = _build(fd, sp, li, geometry=True)
        rg = _pair_acc(Qg2, _predict((Xg2 - mug) / sdg, Qg2, wg, bg))
        rg.pop("pairs")
        res["test"][sp] = {"probe": r, "control_shuffled_labels": rs, "control_geometry_only": rg}
        print(sp, json.dumps(res["test"][sp])[:400], flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1), encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--data", required=True)
    e.add_argument("--splits", nargs="+", required=True)
    e.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    e.add_argument("--revision", default="ebb281ec70b05090aa6165b016eac8ec08e71b17")
    e.add_argument("--batch-size", type=int, default=4)
    e.add_argument("--out", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--data", required=True)
    f.add_argument("--feats", required=True)
    f.add_argument("--test-splits", nargs="+", default=["test_id", "test_unseen_topology"])
    f.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    extract(args) if args.cmd == "extract" else fit(args)


if __name__ == "__main__":
    main()
