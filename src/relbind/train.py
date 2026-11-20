"""Phase-2 pilot training (M1 / M2 / M3) on Qwen3-VL-4B with LoRA. GPU only.

    python -m relbind.train --arm M3 --data D --out experiments/pilot/M3 [--max-steps 5 --check-reload]

Identical across arms: base checkpoint, images, prompts (the A_direct prompt used in Phase 1), example order,
paired batching (each micro-batch = one group's orig+cf or reroute+style examples), optimizer, schedule,
number of steps, LoRA config (r=16, alpha=32, dropout=0.05 on LM q/k/v/o/gate/up/down), trainable
vision-language merger, frozen vision encoder.
Differences:
  M1  target = {"answer": X}
  M2  target = {"answer": X, "annotation_box": [...], "entity_box": [...], "path": [[x,y],...]}  (0-1000 coords)
  M3  M2 target + lambda * (L_rel + L_pair) from a relational scoring head on layer-18 visual-token states.
After training: generated-answer predictions on the eval splits (same parser as Phase 1). For M3, also the
head's relationship predictions (oracle-box protocol, a2e only).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from .baselines.heuristics import load_scene
from .metrics import load_examples
from .vlm.parse import parse
from .vlm.prompts import build_prompt

MODEL = "Qwen/Qwen3-VL-4B-Instruct"
REV = "ebb281ec70b05090aa6165b016eac8ec08e71b17"
HEAD_LAYER = 18  # chosen on val by the Exp-1 probe (amendment logged before any pilot output)
CELL = 32
PER_IMAGE_KEYS = ("pixel_values", "image_grid_thw", "image_sizes", "num_patches", "pixel_attention_mask")
PROJECTOR_KEYS = ("visual.merger.", "multi_modal_projector.")  # Qwen3-VL merger | InternVL/LLaVA-hf projector


def set_model(model_id, revision):
    """Select the base model (default: the pilot's Qwen3-VL-4B). Called once from main()."""
    global MODEL, REV
    MODEL, REV = model_id, revision


def is_projector(name):
    return any(k in name for k in PROJECTOR_KEYS) and "lora" not in name


# ----------------------------------------------------------------------------- targets

def _box1000(b, W, H):
    return [int(round(1000 * max(0.0, min(1.0, v / d)))) for v, d in zip(b, (W, H, W, H))]


def target_text(arm: str, ex: dict, scene) -> str:
    if arm == "M1":
        return json.dumps({"answer": ex["answer"]}, ensure_ascii=False)
    W, H = scene.canvas
    d = {"answer": ex["answer"]}
    if ex["query_type"] == "a2e":
        aid = ex["query_ref"]
    else:
        anns = scene.annotations_of(ex["query_ref"])
        aid = anns[0] if anns else None
    if aid is not None:
        c = next(c for c in scene.connectors if c.source == aid)
        d["annotation_box"] = _box1000(scene.annotation(aid).label.bbox, W, H)
        d["entity_box"] = _box1000(scene.entity(c.target).bbox, W, H)
        d["path"] = [[int(round(1000 * p[0] / W)), int(round(1000 * p[1] / H))] for p in c.points][:6]
    else:
        d["entity_box"] = _box1000(scene.entity(ex["query_ref"]).bbox, W, H)
    if arm == "M2R":  # rationale-first: boxes/path are generated BEFORE the answer, so they can condition it
        d = {k: d[k] for k in d if k != "answer"} | {"answer": d["answer"]}
    return json.dumps(d, ensure_ascii=False)


# ----------------------------------------------------------------------------- batching

class Encoder:
    def __init__(self, proc, data: Path):
        self.proc, self.data = proc, data
        self.end_ids = proc.tokenizer("<|im_end|>\n", add_special_tokens=False)["input_ids"]
        self._scene = {}

    def scene(self, ex):
        if ex["scene_id"] not in self._scene:
            self._scene[ex["scene_id"]] = load_scene(self.data, ex)
        return self._scene[ex["scene_id"]]

    def prompt_inputs(self, ex):
        s, _, sc = self.scene(ex)
        img = Image.open(self.data / ex["image"]).convert("RGB")
        p = build_prompt("A_direct", ex, s, img, sc)
        msgs = [{"role": "user", "content": [{"type": "image", "image": p.image}, {"type": "text", "text": p.text}]}]
        enc = self.proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True,
                                            return_tensors="pt")
        return enc, img

    def train_item(self, arm, ex):
        enc, img = self.prompt_inputs(ex)
        s, _, sc = self.scene(ex)
        tgt = self.proc.tokenizer(target_text(arm, ex, s), add_special_tokens=False)["input_ids"] + self.end_ids
        n = enc["input_ids"].shape[1]
        item = {"labels": torch.tensor([[-100] * n + tgt])}
        for k, v in enc.items():
            if k in PER_IMAGE_KEYS:
                item[k] = v
            elif torch.is_tensor(v) and v.dim() == 2 and v.shape[1] == n:  # per-token keys
                fill = 1 if k == "attention_mask" else 0
                ext = torch.tensor([tgt]) if k == "input_ids" else torch.full((1, len(tgt)), fill, dtype=v.dtype)
                item[k] = torch.cat([v, ext], 1)
            else:
                item[k] = v
        item["_img_size"] = img.size
        return item


def collate(items, pad_id, side="right"):
    L = max(it["input_ids"].shape[1] for it in items)
    out = {}
    for k in items[0]:
        if k.startswith("_"):
            continue
        vs = [it[k] for it in items]
        if k in PER_IMAGE_KEYS:
            out[k] = torch.cat(vs, 0)
        elif torch.is_tensor(vs[0]) and vs[0].dim() == 2:
            fill = pad_id if k == "input_ids" else (-100 if k == "labels" else 0)
            padded = []
            for v in vs:
                p = torch.full((1, L - v.shape[1]), fill, dtype=v.dtype)
                padded.append(torch.cat([v, p], 1) if side == "right" else torch.cat([p, v], 1))
            out[k] = torch.cat(padded, 0)
        else:
            out[k] = vs[0]
    return out


# ----------------------------------------------------------------------------- relational head

class RelHead(nn.Module):
    def __init__(self, d, h=512):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(3 * d, h), nn.GELU(), nn.Linear(h, 1))

    def forward(self, ha, he, g):  # ha [D], he [N,D], g [D]
        x = torch.cat([ha[None] * he, (ha[None] - he).abs(), g[None].expand_as(he)], -1)
        return self.mlp(x).squeeze(-1)


def pooled(hs_layer, input_ids, grid, img_tok, boxes, img_size, scale):
    """hs_layer [T,D] for ONE example; returns pooled vectors for each canonical box + global mean."""
    t, h, w = grid.tolist()
    H2, W2 = h // 2, w // 2
    pos = (input_ids == img_tok).nonzero().flatten()
    g = hs_layer[pos].reshape(H2, W2, -1)
    W0, H0 = img_size
    sx, sy = (w * 16) / W0 * scale, (h * 16) / H0 * scale
    outs = []
    for b in boxes:
        c0 = max(0, int(((b[0] - 2) * sx) // CELL)); c1 = min(W2 - 1, int(((b[2] + 2) * sx) // CELL))
        r0 = max(0, int(((b[1] - 2) * sy) // CELL)); r1 = min(H2 - 1, int(((b[3] + 2) * sy) // CELL))
        outs.append(g[r0:r1 + 1, c0:c1 + 1].float().mean(dim=(0, 1)))
    return torch.stack(outs), g.float().mean(dim=(0, 1))


# ----------------------------------------------------------------------------- main

def build_model(arm):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForImageTextToText, AutoProcessor
    proc = AutoProcessor.from_pretrained(MODEL, revision=REV)
    model = AutoModelForImageTextToText.from_pretrained(MODEL, revision=REV, dtype=torch.bfloat16,
                                                        attn_implementation="sdpa", device_map="cuda")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                     target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)")
    model = get_peft_model(model, cfg)
    for n, p in model.named_parameters():
        if is_projector(n):
            p.requires_grad = True
    head = None
    if arm == "M3":
        d = model.config.text_config.hidden_size if hasattr(model.config, "text_config") else model.config.hidden_size
        head = RelHead(d).cuda().float()
    return proc, model, head


def trainable_state(model, head):
    sd = {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}
    if head is not None:
        sd.update({f"__head__.{k}": v.detach().cpu() for k, v in head.state_dict().items()})
    return sd


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["M1", "M2", "M2R", "M3"])
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--train-split", default="train")
    ap.add_argument("--eval-splits", nargs="+", default=["val", "test_unseen_topology", "test_id"])  # transfer run passes its own
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lam", type=float, default=0.5)
    ap.add_argument("--margin", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-groups", type=int, default=None, help="train on the first N groups of the seed-shuffled order")
    ap.add_argument("--drop-variants", nargs="*", default=[], choices=["orig", "cf", "reroute", "style"])
    ap.add_argument("--check-reload", action="store_true")
    ap.add_argument("--no-eval", action="store_true")
    ap.add_argument("--eval-variants", nargs="+", default=["orig", "cf"])
    ap.add_argument("--model", default=MODEL, help="base model (default: pilot Qwen3-VL-4B); M1 only for non-Qwen")
    ap.add_argument("--revision", default=None)
    args = ap.parse_args(argv)
    if args.model != MODEL:
        assert args.arm == "M1", "non-default base models are supported for the answer-only arm only"
        set_model(args.model, args.revision)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    data, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    proc, model, head = build_model(args.arm)
    enc = Encoder(proc, data)
    pad_id = proc.tokenizer.pad_token_id
    img_tok = getattr(model.config, "image_token_id", None) or proc.tokenizer.convert_tokens_to_ids("<|image_pad|>")

    # paired micro-batches: per group, [orig+cf examples] and [reroute+style examples]; group order shuffled
    exs = [e for e in load_examples(data, [args.train_split]) if e["variant"] not in args.drop_variants]
    by_g = defaultdict(lambda: defaultdict(list))
    for e in exs:
        by_g[e["group_id"]][("oc" if e["variant"] in ("orig", "cf") else "rs")].append(e)
    gids = sorted(by_g)
    random.Random(args.seed).shuffle(gids)
    if args.n_groups:
        gids = gids[: args.n_groups]
    batches = [by_g[g][k] for g in gids for k in ("oc", "rs") if by_g[g][k]]
    steps_total = math.ceil(len(batches) / args.accum)
    if args.max_steps:
        steps_total = min(steps_total, args.max_steps)
        batches = batches[: steps_total * args.accum]
    params = [p for p in model.parameters() if p.requires_grad] + (list(head.parameters()) if head else [])
    n_train = sum(p.numel() for p in params)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    warm = min(30, max(1, steps_total // 10))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps_total))))
    log = open(out / "train_log.jsonl", "w", encoding="utf-8")
    model.train()
    t0, step = time.time(), 0
    recent = []
    acc = defaultdict(float)  # per-optimizer-step means over micro-batches (pilot logged only the last micro-batch)
    acc_micro = []
    for bi, batch in enumerate(batches):
        items = [enc.train_item(args.arm, e) for e in batch]
        inp = {k: v.cuda() for k, v in collate(items, pad_id).items()}
        outp = model(**inp, output_hidden_states=(head is not None))
        loss_sft = outp.loss
        loss = loss_sft
        l_rel = l_pair = torch.zeros((), device="cuda")
        if head is not None:
            hs = outp.hidden_states[HEAD_LAYER]
            scores = {}
            rel_terms = []
            for i, e in enumerate(batch):
                if e["query_type"] != "a2e":
                    continue
                s, _, sc = enc.scene(e)
                ents = [x.id for x in s.entities]
                boxes = [s.annotation(e["query_ref"]).label.bbox] + [x.bbox for x in s.entities]
                vecs, g = pooled(hs[i], inp["input_ids"][i], inp["image_grid_thw"][i], img_tok, boxes, items[i]["_img_size"], sc)
                sc_e = head(vecs[0], vecs[1:], g)
                gold = ents.index(e["answer"])
                rel_terms.append(F.cross_entropy(sc_e[None], torch.tensor([gold], device="cuda")))
                scores[(e["query_ref"], e["variant"])] = (sc_e, ents)
            if rel_terms:
                l_rel = torch.stack(rel_terms).mean()
            pair_terms = []
            for (aid, var), (sc_e, ents) in scores.items():
                if var != "orig" or (aid, "cf") not in scores:
                    continue
                sc_c, _ = scores[(aid, "cf")]
                e_o = next(x["answer"] for x in batch if x["query_ref"] == aid and x["variant"] == "orig" and x["query_type"] == "a2e")
                e_c = next(x["answer"] for x in batch if x["query_ref"] == aid and x["variant"] == "cf" and x["query_type"] == "a2e")
                io, ic = ents.index(e_o), ents.index(e_c)
                pair_terms.append(F.relu(args.margin - (sc_e[io] - sc_e[ic])) + F.relu(args.margin - (sc_c[ic] - sc_c[io])))
            if pair_terms:
                l_pair = torch.stack(pair_terms).mean()
            loss = loss_sft + args.lam * (l_rel + l_pair)
        (loss / args.accum).backward()
        for k, v in (("loss", loss), ("sft", loss_sft), ("rel", l_rel), ("pair", l_pair)):
            acc[k] += v.item() / args.accum
        acc_micro.append(round(loss_sft.item(), 4))
        if (bi + 1) % args.accum == 0:
            gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
            g_head = sum(p.grad.abs().sum().item() for p in head.parameters() if p.grad is not None) if head else None
            g_merger = sum(p.grad.abs().sum().item() for n, p in model.named_parameters()
                           if is_projector(n) and p.grad is not None)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            rec = {"step": step, "loss": round(acc["loss"], 4), "sft": round(acc["sft"], 4),
                   "rel": round(acc["rel"], 4), "pair": round(acc["pair"], 4), "grad_norm": round(float(gn), 4),
                   "grad_abs_head": g_head, "grad_abs_merger": round(g_merger, 4), "lr": sched.get_last_lr()[0],
                   "sec": round(time.time() - t0, 1), "sft_micro": list(acc_micro)}
            acc.clear()  # reset every optimizer step (missing in transfer attempts 1-2: logged values were running sums)
            acc_micro.clear()
            log.write(json.dumps(rec) + "\n")
            log.flush()
            if step % 20 == 0 or step <= 3:
                print(rec, flush=True)
            if not math.isfinite(rec["loss"]) or not math.isfinite(rec["grad_norm"]):
                raise RuntimeError(f"non-finite loss/grad at step {step}")
            recent.append(rec["sft"])
            if step >= 40 and sum(recent[-20:]) / 20 > 4.0:  # divergence guard (pilot sft never exceeded ~2.6)
                raise RuntimeError(f"diverged: mean sft over last 20 steps > 4.0 at step {step}")
        if step >= steps_total:
            break
    log.close()
    sd =trainable_state(model, head)
    torch.save(sd, out / "trainable_state.pt")
    sha = hashlib.sha256((out / "trainable_state.pt").read_bytes()).hexdigest()
    info = {"arm": args.arm, "steps": step, "micro_batches": len(batches[: step * args.accum]), "accum": args.accum,
            "n_trainable_params": n_train, "lr": args.lr, "lam": args.lam, "margin": args.margin, "seed": args.seed,
            "n_groups": len(gids), "drop_variants": args.drop_variants, "n_train_examples": sum(len(b) for b in batches),
            "head_layer": HEAD_LAYER, "train_seconds": round(time.time() - t0, 1), "state_sha256": sha,
            "state_bytes": (out / "trainable_state.pt").stat().st_size, "model": MODEL, "revision": REV}
    if args.check_reload:
        info["reload_check"] = reload_check(args.arm, out / "trainable_state.pt", enc, batch[0], model, pad_id)
    (out / "train_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps(info), flush=True)
    if not args.no_eval:
        del model, opt
        torch.cuda.empty_cache()
        proc, model, head, _, _ = load_trained(args.arm, out / "trainable_state.pt")  # evaluate the saved artifact
        evaluate(model, head, proc, enc, data, args.eval_splits, out, img_tok, pad_id, variants=args.eval_variants)


def load_trained(arm, path):
    """Fresh model + saved trainable state: what is evaluated is exactly the saved artifact."""
    proc, m, head = build_model(arm)
    sd = torch.load(path)
    res = m.load_state_dict({k: v for k, v in sd.items() if not k.startswith("__head__")}, strict=False)
    if head is not None:
        head.load_state_dict({k[len("__head__."):]: v for k, v in sd.items() if k.startswith("__head__.")})
    m.eval()
    return proc, m, head, sd, res


@torch.no_grad()
def reload_check(arm, path, enc, ex, model, pad_id):
    """(1) saved tensors == trained tensors bit-exactly; (2) two independent reloads give identical logits;
    (3) in-memory vs reloaded: same top-1 token at the last 5 positions (raw logit diff logged; it can be
    non-zero because compact in-memory tensors and loaded views can select different bf16 GEMM kernels)."""
    model.eval()
    item = enc.train_item(arm, ex)
    inp = {k: v.cuda() for k, v in collate([item], pad_id).items()}
    a = model(**inp).logits[0, -5:].float()
    live = {n: p.detach() for n, p in model.named_parameters() if p.requires_grad}
    outs, weights_equal, unexpected = [], True, 0
    for _ in range(2):
        _, m2, _, sd, res = load_trained(arm, path)
        weights_equal &= all(torch.equal(sd[n].cuda(), t) for n, t in live.items())
        unexpected = len(res.unexpected_keys)
        outs.append(m2(**inp).logits[0, -5:].float())
        del m2
        torch.cuda.empty_cache()
    model.train()
    d_live = float((a - outs[0]).abs().max())
    d_reload = float((outs[0] - outs[1]).abs().max())
    top1 = bool((a.argmax(-1) == outs[0].argmax(-1)).all())
    ok = weights_equal and d_reload == 0.0 and top1 and unexpected == 0
    return {"weights_equal": weights_equal, "reload_vs_reload_max_abs_logit_diff": d_reload,
            "max_abs_logit_diff": d_live, "top1_agree": top1, "unexpected_keys": unexpected, "ok": ok}


@torch.no_grad()
def evaluate(model, head, proc, enc, data, splits, out, img_tok, pad_id, bs=24, variants=("orig", "cf")):
    model.eval()
    proc.tokenizer.padding_side = "left"
    exs = [e for e in load_examples(data, splits) if e["variant"] in variants]
    with open(out / "predictions.jsonl", "w", encoding="utf-8") as f:
        for i in range(0, len(exs), bs):
            chunk = exs[i:i + bs]
            msgs, prompts = [], []
            for e in chunk:
                s, _, sc = enc.scene(e)
                img = Image.open(data / e["image"]).convert("RGB")
                p = build_prompt("A_direct", e, s, img, sc)
                prompts.append((p, sc, img.size))
                msgs.append([{"role": "user", "content": [{"type": "image", "image": p.image}, {"type": "text", "text": p.text}]}])
            inp = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True,
                                           return_tensors="pt", padding=True).to("cuda")
            gen = model.generate(**inp, max_new_tokens=320, do_sample=False)  # 320: M2R emits ~150 digit tokens before the answer
            texts = proc.batch_decode(gen[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)
            head_preds = {}
            if head is not None:
                hs = model(**inp, output_hidden_states=True).hidden_states[HEAD_LAYER]
                for j, e in enumerate(chunk):
                    if e["query_type"] != "a2e":
                        continue
                    s, _, sc = enc.scene(e)
                    boxes = [s.annotation(e["query_ref"]).label.bbox] + [x.bbox for x in s.entities]
                    vecs, g = pooled(hs[j], inp["input_ids"][j], inp["image_grid_thw"][j], img_tok, boxes, prompts[j][2], sc)
                    head_preds[e["example_id"]] = s.entities[int(head(vecs[0], vecs[1:], g).argmax())].id
            for e, (p, sc, _), t in zip(chunk, prompts, texts):
                r = {"example_id": e["example_id"], **parse(t, e, p.image.width, p.image.height, p.crop_box, scale=sc),
                     "raw_text": t}
                if e["example_id"] in head_preds:
                    r["head_prediction"] = head_preds[e["example_id"]]
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"eval {min(i + bs, len(exs))}/{len(exs)}", flush=True)
    proc.tokenizer.padding_side = "right"
    model.train()


if __name__ == "__main__":
    main()
