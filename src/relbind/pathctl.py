"""Path-following control (Decisive Experiment 1a).

For every a2e query of a built dataset, re-render its orig and cf scene with ALL annotation text removed.
Each connector start gets a hollow dot, and the queried connector's start gets a filled dot. The question asks
where the line starting at the filled dot ends. Same geometry, same gold target, no text binding involved.

    python -m relbind.pathctl --data data/strict_v2 --splits test_id test_unseen_topology

Writes <split>/path_images/*.png and <split>/path_examples.jsonl (example ids end in -path-<variant>; they
share group ids with the binding examples, so paired comparisons are possible).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .baselines.heuristics import load_scene
from .metrics import load_examples
from .queries import NOUNS
from .render import render_png

ID_WORD = {"schematic": "label", "flowchart": "ID", "circuit": "designator"}
R = 5.0


def path_scene(scene, query_ann: str):
    s = scene.copy()
    for a in s.annotations:
        a.label.text = ""  # text removed; geometry untouched
    for c in s.connectors:
        p = c.points[0]
        s.distractors.append({"type": "circle", "center": [p[0], p[1]], "r": R, "fill": c.source == query_ann,
                              "width_role": "connector", "role": "path_marker"})
    return s


def question(scene) -> str:
    noun = NOUNS[(scene.family, scene.template)]
    return (f"Which {noun} does the line starting at the filled black dot end on? "
            f"Answer with the {noun} {ID_WORD[scene.family]}.")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--splits", nargs="+", required=True)
    args = ap.parse_args(argv)
    data = Path(args.data)
    for split in args.splits:
        exs = [e for e in load_examples(data, [split]) if e["query_type"] == "a2e" and e["variant"] in ("orig", "cf")]
        (data / split / "path_images").mkdir(exist_ok=True)
        out = []
        for ex in exs:
            s, sty, sc = load_scene(data, ex)
            ps = path_scene(s, ex["query_ref"])
            name = f"{ex['scene_id']}-path-{ex['query_ref']}.png"
            render_png(ps, sty, sc).save(data / split / "path_images" / name)
            out.append({**ex, "example_id": f"{ex['query_id']}-path-{ex['variant']}", "query_id": f"{ex['query_id']}-path",
                        "image": f"{split}/path_images/{name}", "query_text": "", "question": question(s),
                        "control": "path_following", "candidate_values": []})
        with open(data / split / "path_examples.jsonl", "w", encoding="utf-8") as f:
            for e in out:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        print(f"[{split}] {len(out)} path-control examples")


if __name__ == "__main__":
    main()
