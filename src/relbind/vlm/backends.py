"""Inference backends.

mock : deterministic, CPU-only stand-in that answers from a heuristic baseline and emits text in the
       same format as a real model. Used to smoke-test the whole harness before any GPU time.
qwen : any Hugging Face transformers image-text-to-text model with a chat template (Qwen3-VL, InternVL-hf,
       LLaVA-OneVision-hf, Gemma-3, SmolVLM2, ...), greedy decoding, batched, left padding.
       Requires `pip install -e .[vlm]`; exact versions and model revision are logged per run.
"""
from __future__ import annotations

import json
from typing import Protocol

from PIL import Image


class Backend(Protocol):
    name: str

    def generate(self, batch: list[tuple[Image.Image, str]], max_new_tokens: int) -> list[str]: ...

    def info(self) -> dict: ...


class MockBackend:
    """Answers each prompt with a pre-computed answer (looked up by the caller-provided key)."""

    name = "mock"

    def __init__(self, answers: dict[str, str]):
        self.answers = answers
        self._keys: list[str] = []

    def set_keys(self, keys: list[str]):
        self._keys = keys

    def generate(self, batch, max_new_tokens):
        return [f"Looking at the diagram.\n{json.dumps({'answer': self.answers.get(k, 'NONE')}, ensure_ascii=False)}"
                for k in self._keys]

    def info(self):
        return {"backend": "mock"}


def _hub_sha(model_id, revision):
    """transformers 5 no longer sets config._commit_hash; ask the hub (sweep attempt 1-2 lost this field)."""
    try:
        from huggingface_hub import HfApi
        return HfApi().model_info(model_id, revision=revision).sha
    except Exception as e:  # offline etc. -> record why
        return f"unresolved: {type(e).__name__}"


class QwenVLBackend:
    name = "qwen"

    def __init__(self, model_id: str, revision: str | None = None, dtype: str = "bfloat16",
                 attn: str = "sdpa", max_pixels: int | None = None):
        import torch
        import transformers
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.torch = torch
        self.model_id, self.revision = model_id, revision
        kw = {}
        if max_pixels is not None:
            kw["max_pixels"] = max_pixels
        self.processor = AutoProcessor.from_pretrained(model_id, revision=revision, **kw)
        tok = getattr(self.processor, "tokenizer", None)
        if tok is not None:
            tok.padding_side = "left"
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
        if hasattr(self.processor, "padding_side"):
            self.processor.padding_side = "left"
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id, revision=revision, dtype=getattr(torch, dtype), attn_implementation=attn, device_map="cuda")
        self.model.eval()
        self._info = {"backend": "hf", "model_id": model_id, "revision": revision,
                      "resolved_commit": getattr(self.model.config, "_commit_hash", None) or _hub_sha(model_id, revision),
                      "transformers": transformers.__version__, "torch": torch.__version__,
                      "dtype": dtype, "attn": attn, "max_pixels": max_pixels,
                      "gpu": torch.cuda.get_device_name(0)}

    def generate(self, batch, max_new_tokens):
        msgs = [[{"role": "user", "content": [{"type": "image", "image": im}, {"type": "text", "text": txt}]}]
                for im, txt in batch]
        inputs = self.processor.apply_chat_template(
            msgs, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt", padding=True
        ).to(self.model.device)
        with self.torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        new = out[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(new, skip_special_tokens=True)

    def info(self):
        return self._info
