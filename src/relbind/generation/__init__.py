"""Scene generation registry + deterministic sampling with rejection."""
from __future__ import annotations

import hashlib

import numpy as np

from ..schema import Scene
from ..verify import verify_scene
from . import circuit, flowchart, schematic
from .common import GenerationError

FAMILIES = {m.FAMILY: m for m in (schematic, flowchart, circuit)}
CANVAS = {"schematic": {"elevation": [1024, 720], "plan": [1024, 768]},
          "flowchart": {"tb": [1024, 700], "lr": [1024, 720]},
          "circuit": {"ladder": [1024, 720], "grid": [1024, 720]}}


def derive_seed(*parts) -> int:
    h = hashlib.sha256(":".join(str(p) for p in parts).encode()).hexdigest()
    return int(h[:12], 16)


def generate_scene(family: str, template: str, topology: str, font: str, seed: int, scene_id: str,
                   max_attempts: int = 60) -> tuple[Scene, int]:
    """Sample a verified scene. Returns (scene, n_rejected_attempts). Deterministic in `seed`."""
    mod = FAMILIES[family]
    last = None
    for attempt in range(max_attempts):
        sub = derive_seed(seed, attempt)
        rng = np.random.default_rng(sub)
        scene = Scene(scene_id=scene_id, family=family, template=template, topology=topology, seed=sub,
                      canvas=list(CANVAS[family][template]), font=font, entities=[], annotations=[],
                      connectors=[], relations=[], distractors=[])
        try:
            mod.generate(scene, rng)
        except GenerationError as e:
            last = str(e)
            continue
        issues = verify_scene(scene)
        if issues:
            last = issues[0]
            continue
        scene.meta["attempts"] = attempt + 1
        return scene, attempt
    raise GenerationError(f"{family}/{template}/{topology}: no valid scene after {max_attempts} attempts ({last})")
