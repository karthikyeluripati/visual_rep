"""Ground-truth schema for RelBind-Bench.

The relation graph is created *before* rendering and is the single source of truth.
Rendering is a pure function of (Scene, Style, scale).

Coordinates: canonical canvas pixels, origin top-left, x right, y down; bbox = [x0, y0, x1, y1].
Rendered images at scale s map canonical (x, y) -> (s*x, s*y).
"""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any

SCHEMA_VERSION = "relbind-scene/0.1"

COORD_CONVENTION = (
    "canonical canvas pixels; origin top-left; x right; y down; bbox=[x0,y0,x1,y1]; "
    "rendered image at scale s maps (x,y)->(s*x,s*y)"
)


@dataclass
class TextItem:
    text: str
    x: float  # anchor ("lt": left / ascender-top) in canonical px
    y: float
    size: float
    bbox: list[float]
    font: str = "sans"


@dataclass
class Entity:
    id: str  # stable node id, also the answer string for entity queries
    kind: str  # column | marker | box | resistor | capacitor
    bbox: list[float]  # whole entity incl. its own label
    geometry: dict[str, Any]  # {"type": "segment", "points": [[x,y],[x,y]]} or {"type": "rect", "bbox": [...]}
    crossable: bool  # line-like entities may be crossed by connectors (at an angle); solids may not
    label: TextItem | None = None
    attrs: dict[str, Any] = field(default_factory=dict)


@dataclass
class Annotation:
    id: str  # stable node id (a0, a1, ...)
    value: str  # canonical extracted value (answer string for value queries)
    label: TextItem


@dataclass
class Connector:
    id: str
    source: str  # annotation id
    target: str  # entity id
    points: list[list[float]]  # polyline from source side to target attach point
    terminator: str  # arrow | dot | tick | none ; drawn at points[-1]
    route_class: str  # straight | elbow | zigzag
    kind: str = "leader"  # leader | branch_arrow
    start_mode: str = "bbox"  # bbox: start recomputed from annotation bbox; fixed: start is a fixed anchor


@dataclass
class Relation:
    annotation: str
    entity: str
    connector: str
    type: str = "annotates"
    directed: bool = True


@dataclass
class Scene:
    scene_id: str
    family: str  # schematic | flowchart | circuit
    template: str
    topology: str  # route class family used by the scene's connectors
    seed: int
    canvas: list[int]
    font: str
    entities: list[Entity]
    annotations: list[Annotation]
    connectors: list[Connector]
    relations: list[Relation]
    distractors: list[dict[str, Any]]  # free primitives: lines / texts / shapes (no relation semantics)
    meta: dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION
    coord_convention: str = COORD_CONVENTION

    # --------------------------------------------------------- lookups
    def entity(self, eid: str) -> Entity:
        for e in self.entities:
            if e.id == eid:
                return e
        raise KeyError(eid)

    def annotation(self, aid: str) -> Annotation:
        for a in self.annotations:
            if a.id == aid:
                return a
        raise KeyError(aid)

    def connector(self, cid: str) -> Connector:
        for c in self.connectors:
            if c.id == cid:
                return c
        raise KeyError(cid)

    def target_of(self, aid: str) -> str | None:
        for r in self.relations:
            if r.annotation == aid:
                return r.entity
        return None

    def annotations_of(self, eid: str) -> list[str]:
        return [r.annotation for r in self.relations if r.entity == eid]

    def edge_set(self) -> set[tuple[str, str]]:
        return {(r.annotation, r.entity) for r in self.relations}

    # --------------------------------------------------------- serialisation
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False)

    def content_hash(self) -> str:
        d = self.to_dict()
        d.pop("scene_id", None)
        d.pop("meta", None)
        return hashlib.sha256(json.dumps(d, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]

    def copy(self) -> "Scene":
        return copy.deepcopy(self)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Scene":
        d = dict(d)

        def ti(x):
            return TextItem(**x) if x is not None else None

        ents = [Entity(**{**e, "label": ti(e.get("label"))}) for e in d.pop("entities")]
        anns = [Annotation(**{**a, "label": ti(a["label"])}) for a in d.pop("annotations")]
        cons = [Connector(**c) for c in d.pop("connectors")]
        rels = [Relation(**r) for r in d.pop("relations")]
        return Scene(entities=ents, annotations=anns, connectors=cons, relations=rels, **d)


def relation_signature(scene: Scene) -> str:
    """Hash of the relation graph in terms of *displayed* strings (value -> entity id).

    Two scenes with the same signature pose identical binding problems as far as text goes."""
    pairs = sorted((scene.annotation(r.annotation).value, r.entity) for r in scene.relations)
    ents = sorted(e.id for e in scene.entities)
    return hashlib.sha256(json.dumps([scene.family, ents, pairs]).encode()).hexdigest()[:16]
