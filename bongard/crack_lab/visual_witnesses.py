"""Typed visual witnesses for the Bongard semantic track.

Witnesses are evidence objects, not separator scores.  They are deliberately
small dataclasses so execution traces can be serialized and replayed.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any


Json = dict[str, Any]


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {k: _jsonable(v) for k, v in asdict(value).items()}
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return value


@dataclass(frozen=True)
class Witness:
    confidence: float = 1.0
    residual: float = 0.0
    provenance: tuple[str, ...] = ()

    @property
    def witness_type(self) -> str:
        return type(self).__name__

    def to_trace(self) -> Json:
        data = _jsonable(self)
        data["witness_type"] = self.witness_type
        return data


@dataclass(frozen=True)
class PointWitness(Witness):
    x: float = 0.0
    y: float = 0.0
    source_id: str = ""


@dataclass(frozen=True)
class CurveWitness(Witness):
    source_component_id: str = ""
    points: tuple[tuple[float, float], ...] = ()
    endpoints: tuple[PointWitness, ...] = ()


@dataclass(frozen=True)
class LineSegmentWitness(CurveWitness):
    start: PointWitness = field(default_factory=PointWitness)
    end: PointWitness = field(default_factory=PointWitness)
    length: float = 0.0


@dataclass(frozen=True)
class ArcWitness(CurveWitness):
    center: PointWitness = field(default_factory=PointWitness)
    radius: float = 0.0
    angle_degrees: float = 0.0


@dataclass(frozen=True)
class CircleWitness(Witness):
    source_component_id: str = ""
    center: PointWitness = field(default_factory=PointWitness)
    radius: float = 0.0
    support_points: tuple[tuple[float, float], ...] = ()


@dataclass(frozen=True)
class ContourWitness(Witness):
    source_component_id: str = ""
    points: tuple[tuple[float, float], ...] = ()
    is_closed: bool = False


@dataclass(frozen=True)
class SkeletonGraphWitness(Witness):
    source_component_id: str = ""
    nodes: tuple[PointWitness, ...] = ()
    edges: tuple[tuple[int, int], ...] = ()
    endpoint_count: int = 0
    branch_count: int = 0
    cycle_count: int = 0


@dataclass(frozen=True)
class PolygonWitness(Witness):
    source_component_id: str = ""
    vertices: tuple[PointWitness, ...] = ()
    side_count: int = 0


@dataclass(frozen=True)
class TriangleWitness(PolygonWitness):
    side_count: int = 3


@dataclass(frozen=True)
class QuadrilateralWitness(PolygonWitness):
    side_count: int = 4


@dataclass(frozen=True)
class PartWitness(Witness):
    part_id: str = ""
    role: str = ""
    source_component_id: str = ""
    contour: ContourWitness | None = None


@dataclass(frozen=True)
class ContactWitness(Witness):
    source_a: str = ""
    source_b: str = ""
    points: tuple[PointWitness, ...] = ()
    relation: str = "contact"


@dataclass(frozen=True)
class IntersectionWitness(ContactWitness):
    relation: str = "intersection"


@dataclass(frozen=True)
class PartGraphWitness(Witness):
    parts: tuple[PartWitness, ...] = ()
    contacts: tuple[ContactWitness, ...] = ()
    adjacency: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class CirclePairWitness(Witness):
    first: CircleWitness = field(default_factory=CircleWitness)
    second: CircleWitness = field(default_factory=CircleWitness)
    center_distance: float = 0.0


@dataclass(frozen=True)
class CircleIntersectionWitness(IntersectionWitness):
    pair: CirclePairWitness = field(default_factory=CirclePairWitness)


@dataclass(frozen=True)
class RadialArrangementWitness(Witness):
    center: PointWitness = field(default_factory=PointWitness)
    parts: tuple[PartWitness, ...] = ()
    part_count: int = 0
    symmetry_order: int = 0


@dataclass(frozen=True)
class SymmetryWitness(Witness):
    source_id: str = ""
    kind: str = "reflection"
    order: int = 1
    axis_angle_degrees: float = 0.0


@dataclass(frozen=True)
class PrototypeWitness(Witness):
    prototype_name: str = ""
    roles: dict[str, str] = field(default_factory=dict)
    required_roles: tuple[str, ...] = ()


def witness_type_name(value: Any) -> str:
    if isinstance(value, Witness):
        return type(value).__name__
    return type(value).__name__
