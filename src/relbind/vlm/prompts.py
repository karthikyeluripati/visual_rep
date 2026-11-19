"""Diagnostic conditions (research/experimental_protocol.md §3).

A  direct      full image, answer directly
B  reasoning   full image, explicit trace-the-line reasoning, then answer
C  oracle_crop crop around the *pair region* (both swapped annotations, both targets, both
               connectors in orig and cf), identical for every variant of a group -> no answer leak
D  oracle_ocr  full image + every text string with its box (no relations)
E  oracle_box  full image + annotation box and every candidate entity box (no relations)
F  oracle_graph full image + the gold relation edges as text (reading/format ceiling)
G  heuristics  see relbind.baselines.heuristics (not a prompt condition)
"""
from __future__ import annotations

from dataclasses import dataclass

from PIL import Image

from .. import geometry as G
from ..schema import Scene

CONDITIONS = ("A_direct", "B_reasoning", "C_oracle_crop", "D_oracle_ocr", "E_oracle_box", "F_oracle_graph", "P_read")

FORMAT = ('Reply with a single JSON object on the last line: {"answer": "<answer>"}. '
          "Use exactly the text as printed in the diagram.")
FORMAT_LOC = ('Reply with a single JSON object on the last line: {"answer": "<answer>", '
              '"annotation_box": [x0, y0, x1, y1], "entity_box": [x0, y0, x1, y1]} where boxes are in '
              "0-1000 coordinates relative to the image width/height. Use exactly the text as printed in the diagram.")
REASON = ("First locate the referenced item, then follow its connecting line from end to end, noting every "
          "place where it crosses or bends near other lines, and identify where it terminates. "
          "Think step by step, then give the final answer.")


@dataclass
class Prompt:
    image: Image.Image
    text: str
    crop_box: list[float] | None = None  # canonical coords of the crop, if any (for box conversion)


def _fmt_box(b, scale, img_w, img_h):
    """Box in Qwen3-VL's native grounding convention: integers in [0, 1000] relative to width/height.
    (Amendment 2026-11-19: pixel coordinates were misread by the model; see protocol change log.)"""
    xs = (b[0] * scale / img_w, b[2] * scale / img_w)
    ys = (b[1] * scale / img_h, b[3] * scale / img_h)
    v = [xs[0], ys[0], xs[1], ys[1]]
    return "[" + ", ".join(str(int(round(max(0.0, min(1.0, t)) * 1000))) for t in v) + "]"


BOX_NOTE = "Boxes are [x0, y0, x1, y1] in 0-1000 coordinates relative to the image width and height."


def pair_region(group_scenes: list[Scene], ex: dict, pad: float = 30.0) -> list[float]:
    """Union of everything relevant to the swapped pair across orig and cf (answer-agnostic)."""
    base = group_scenes[0]
    boxes = []
    a1, a2 = base.meta["swapped"]
    for s in group_scenes:
        for aid in (a1, a2):
            boxes.append(s.annotation(aid).label.bbox)
            c = next(c for c in s.connectors if c.source == aid)
            boxes.append(G.bbox_of_points(c.points))
            boxes.append(s.entity(c.target).bbox)
    if ex["query_type"] == "e2a":
        boxes.append(base.entity(ex["query_ref"]).bbox)
    b = [min(x[0] for x in boxes), min(x[1] for x in boxes), max(x[2] for x in boxes), max(x[3] for x in boxes)]
    W, H = base.canvas
    return [max(0, b[0] - pad), max(0, b[1] - pad), min(W, b[2] + pad), min(H, b[3] + pad)]


def build_prompt(condition: str, ex: dict, scene: Scene, img: Image.Image, scale: float,
                 group_scenes: list[Scene] | None = None, ask_boxes: bool = False) -> Prompt:
    q = ex["question"]
    fmt = FORMAT_LOC if ask_boxes else FORMAT
    W, H = img.size
    if condition == "A_direct":
        return Prompt(img, f"{q}\n{fmt}")
    if condition == "B_reasoning":
        return Prompt(img, f"{q}\n{REASON}\n{fmt}")
    if condition == "C_oracle_crop":
        cb = pair_region(group_scenes or [scene], ex)
        crop = img.crop(tuple(int(round(v * scale)) for v in cb))
        up = min(2.5, 1024 / max(crop.size))
        if up > 1:
            crop = crop.resize((int(crop.width * up), int(crop.height * up)), Image.LANCZOS)
        return Prompt(crop, f"This is a zoomed-in region of a larger diagram.\n{q}\n{fmt}", crop_box=cb)
    if condition == "D_oracle_ocr":
        items = [(a.label.text, a.label.bbox) for a in scene.annotations]
        items += [(e.label.text, e.label.bbox) for e in scene.entities if e.label]
        items += [(p["text"], p["bbox"]) for e in scene.entities for p in e.attrs.get("draw", []) if p["type"] == "text"]
        items += [(p["text"], p["bbox"]) for p in scene.distractors if p["type"] == "text"]
        items.sort(key=lambda t: (t[1][1], t[1][0]))
        lines = "\n".join(f'- "{t}" at {_fmt_box(b, scale, W, H)}' for t, b in items)
        return Prompt(img, f"All text in the image, with boxes. {BOX_NOTE}\n"
                           f"{lines}\n(The list gives no information about which text is connected to which object.)\n{q}\n{fmt}")
    if condition == "E_oracle_box":
        anns = "\n".join(f'- annotation "{a.label.text}" at {_fmt_box(a.label.bbox, scale, W, H)}' for a in scene.annotations)
        ents = "\n".join(f"- {e.id} at {_fmt_box(e.bbox, scale, W, H)}" for e in scene.entities)
        return Prompt(img, f"{BOX_NOTE}\nAnnotation boxes:\n{anns}\nCandidate target boxes:\n{ents}\n"
                           f"(Boxes give no information about which annotation is connected to which target.)\n{q}\n{fmt}")
    if condition == "P_read":
        # recognition probes; ex carries "probe_bbox" (canonical coords) and "probe_mode" (added by run.py)
        b = G.inflate(ex["probe_bbox"], 3)
        if ex.get("probe_mode") == "crop":  # pure recognition: the text region alone, native scale
            crop = img.crop(tuple(int(round(v * scale)) for v in b))
            return Prompt(crop, f"Transcribe the text in this image exactly. {FORMAT}")
        return Prompt(img, f"What text is written inside the box {_fmt_box(b, scale, W, H)}? {BOX_NOTE} "
                           f"Transcribe it exactly. {FORMAT}")
    if condition == "F_oracle_graph":
        edges = "\n".join(f'- "{scene.annotation(r.annotation).value}" is connected to {r.entity}' for r in scene.relations)
        return Prompt(img, f"Verified connections in this diagram:\n{edges}\n{q}\n{fmt}")
    raise ValueError(condition)
