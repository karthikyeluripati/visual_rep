"""Robust output parsing. Parsing never consults the gold answer.

Order: last JSON object with an "answer" key -> 'answer: X' pattern -> whole text.
The extracted string is then snapped to the candidate set when the match is unique; otherwise the
raw string is kept (it will be scored wrong unless it normalises to the gold answer).
"""
from __future__ import annotations

import json
import re
import unicodedata

from ..metrics import normalize

NONE = "NONE"


def extract_json(text: str) -> dict | None:
    for m in reversed(list(re.finditer(r"\{[^{}]*\}", text, flags=re.S))):
        try:
            d = json.loads(m.group(0))
        except json.JSONDecodeError:
            # tolerate unescaped inner quotes such as {"answer": "4'-0""}
            mm = re.search(r'"answer"\s*:\s*"(.*?)"\s*(,|\})', m.group(0), flags=re.S)
            if mm:
                return {"answer": mm.group(1)}
            continue
        if isinstance(d, dict) and "answer" in d:
            return d
    return None


def raw_answer(text: str) -> tuple[str, dict]:
    d = extract_json(text)
    if d is not None:
        return str(d.get("answer", "")), d
    m = re.findall(r"answer\s*(?:is)?\s*[:=]\s*(.+)", text, flags=re.I)
    if m:
        return m[-1].strip(), {}
    return text.strip().splitlines()[-1] if text.strip() else "", {}


def snap(ans: str, candidates: list[str]) -> str:
    n = normalize(ans)
    for c in candidates:
        if normalize(c) == n:
            return c
    # unique candidate mentioned as a whole token inside the answer (whitespace kept for tokenisation)
    light = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", ans)).lower()
    hits = []
    for c in candidates:
        cl = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", c)).lower().strip("\"'")
        if cl and re.search(rf"(?<![a-z0-9.]){re.escape(cl)}(?![a-z0-9])", light):
            hits.append(c)
    if len(hits) == 1:
        return hits[0]
    return ans


def to_canvas_box(box, img_w, img_h, crop_box=None, scale=1.0):
    """0-1000 relative box in the model's input image -> canonical canvas coords."""
    if not (isinstance(box, (list, tuple)) and len(box) == 4):
        return None
    try:
        x0, y0, x1, y1 = (float(v) / 1000.0 for v in box)
    except (TypeError, ValueError):
        return None
    if crop_box is not None:
        cw, ch = crop_box[2] - crop_box[0], crop_box[3] - crop_box[1]
        return [crop_box[0] + x0 * cw, crop_box[1] + y0 * ch, crop_box[0] + x1 * cw, crop_box[1] + y1 * ch]
    return [x0 * img_w / scale, y0 * img_h / scale, x1 * img_w / scale, y1 * img_h / scale]


def parse(text: str, ex: dict, img_w: int, img_h: int, crop_box=None, scale: float = 1.0) -> dict:
    ans, d = raw_answer(text)
    cands = ex["candidate_entities"] if ex["query_type"] == "a2e" else ex["candidate_values"] + [NONE]
    out = {"prediction": snap(ans, cands), "raw_answer": ans, "parsed_json": bool(d)}
    for k_in, k_out in (("annotation_box", "pred_annotation_bbox"), ("entity_box", "pred_entity_bbox")):
        if k_in in d:
            b = to_canvas_box(d[k_in], img_w, img_h, crop_box, scale)
            if b is not None:
                out[k_out] = b
    return out
