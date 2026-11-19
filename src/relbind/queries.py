"""Query construction. Two directions per relation:

a2e: annotation value -> which entity?        (answer: entity id)
e2a: entity id        -> which annotation?    (answer: annotation value, or NONE)
"""
from __future__ import annotations

from .schema import Scene

NONE = "NONE"

NOUNS = {
    ("schematic", "elevation"): "column",
    ("schematic", "plan"): "column marker",
    ("flowchart", "tb"): "step",
    ("flowchart", "lr"): "step",
    ("circuit", "ladder"): "component",
    ("circuit", "grid"): "component",
}


def question(scene: Scene, qtype: str, ref_text: str) -> str:
    noun = NOUNS[(scene.family, scene.template)]
    if scene.family == "flowchart":
        if qtype == "a2e":
            return f'Which {noun} does the branch labeled "{ref_text}" lead to? Answer with the {noun} ID.'
        return (f"Which branch label is on the arrow that leads into {noun} {ref_text}? "
                f"Answer with the label text, or {NONE} if no labeled branch leads into it.")
    if scene.family == "circuit":
        if qtype == "a2e":
            return f'Which {noun} is the value "{ref_text}" assigned to by its leader line? Answer with the {noun} designator.'
        return (f"Which value is assigned to {noun} {ref_text} by a leader line? "
                f"Answer with the value text, or {NONE} if it has no value annotation.")
    if qtype == "a2e":
        return f'Which {noun} does the annotation "{ref_text}" point to? Answer with the {noun} label.'
    return (f"Which annotation is attached to {noun} {ref_text} by a leader line? "
            f"Answer with the annotation text, or {NONE} if no annotation is attached.")


def answer(scene: Scene, qtype: str, ref_id: str) -> str:
    if qtype == "a2e":
        return scene.target_of(ref_id)
    anns = scene.annotations_of(ref_id)
    return scene.annotation(anns[0]).value if anns else NONE
