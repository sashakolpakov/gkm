"""Typed visual witnesses for the Bongard semantic track.

Witnesses are evidence objects, not separator scores.  They are deliberately
small dataclasses so execution traces can be serialized and replayed.
"""
from __future__ import annotations

import math
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
class PairWitness(Witness):
    """Marker base for witnesses that construct exactly two peer objects.

    Pair-ness is a semantic property used by the compiler's exact-cardinality
    checks.  Making it a runtime type relation avoids encoding that property
    in a class-name suffix.
    """


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
class IncidentRayWitness(Witness):
    """One fitted, owner-labelled ray leaving a witnessed contact point."""

    ray_id: str = ""
    owner_id: str = ""
    endpoint_name: str = "start"
    segment: LineSegmentWitness = field(default_factory=LineSegmentWitness)
    direction_degrees: float = 0.0
    uncertainty_degrees: float = 0.0

    def __post_init__(self) -> None:
        values = {
            "direction_degrees": self.direction_degrees,
            "uncertainty_degrees": self.uncertainty_degrees,
            "confidence": self.confidence,
            "residual": self.residual,
        }
        if any(isinstance(value, bool) or not math.isfinite(float(value))
               for value in values.values()):
            raise ValueError("IncidentRayWitness numeric fields must be finite")
        if not self.ray_id or not self.owner_id:
            raise ValueError("IncidentRayWitness IDs must be nonempty")
        if self.endpoint_name not in {"start", "end"}:
            raise ValueError(
                "IncidentRayWitness endpoint_name must be start or end")
        if not 0.0 <= float(self.direction_degrees) < 360.0:
            raise ValueError(
                "IncidentRayWitness direction must be in [0, 360)")
        if float(self.uncertainty_degrees) < 0.0:
            raise ValueError(
                "IncidentRayWitness uncertainty must be nonnegative")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("IncidentRayWitness confidence must be in [0, 1]")
        if float(self.residual) < 0.0:
            raise ValueError("IncidentRayWitness residual must be nonnegative")
        if not self.provenance:
            raise ValueError("IncidentRayWitness provenance must be nonempty")
        if self.segment.source_component_id != self.ray_id:
            raise ValueError(
                "IncidentRayWitness ray_id must match its segment source")
        dx = float(self.segment.end.x) - float(self.segment.start.x)
        dy = float(self.segment.end.y) - float(self.segment.start.y)
        segment_length = math.hypot(dx, dy)
        if not math.isfinite(segment_length) or segment_length <= 0.0:
            raise ValueError("IncidentRayWitness segment must be nondegenerate")
        if not math.isclose(
                float(self.segment.length), segment_length,
                rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError(
                "IncidentRayWitness segment length does not match endpoints")
        fitted_direction = math.degrees(math.atan2(dy, dx)) % 360.0
        direction_error = abs(
            (float(self.direction_degrees) - fitted_direction + 180.0)
            % 360.0 - 180.0)
        if direction_error > 1e-9:
            raise ValueError(
                "IncidentRayWitness direction does not match its segment")
        if not math.isclose(
                float(self.residual), float(self.segment.residual),
                rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(
                "IncidentRayWitness residual must match its segment")
        if float(self.uncertainty_degrees) + 1e-12 < math.degrees(
                math.atan(float(self.segment.residual))):
            raise ValueError(
                "IncidentRayWitness uncertainty understates fit residual")


@dataclass(frozen=True)
class ExteriorGapWitness(Witness):
    """A certified cyclic angular gap between rays of different owners."""

    ray_a_id: str = ""
    ray_b_id: str = ""
    owner_a: str = ""
    owner_b: str = ""
    degrees: float = 0.0
    uncertainty_degrees: float = 0.0

    def __post_init__(self) -> None:
        values = {
            "degrees": self.degrees,
            "uncertainty_degrees": self.uncertainty_degrees,
            "confidence": self.confidence,
            "residual": self.residual,
        }
        if any(isinstance(value, bool) or not math.isfinite(float(value))
               for value in values.values()):
            raise ValueError("ExteriorGapWitness numeric fields must be finite")
        if not self.ray_a_id or not self.ray_b_id:
            raise ValueError("ExteriorGapWitness ray IDs must be nonempty")
        if not self.owner_a or not self.owner_b or self.owner_a == self.owner_b:
            raise ValueError(
                "ExteriorGapWitness must bind two distinct nonempty owners")
        if not 0.0 < float(self.degrees) < 360.0:
            raise ValueError("ExteriorGapWitness degrees must be in (0, 360)")
        if not 0.0 <= float(self.uncertainty_degrees) < float(self.degrees):
            raise ValueError(
                "ExteriorGapWitness uncertainty must be nonnegative and "
                "strictly below the gap")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("ExteriorGapWitness confidence must be in [0, 1]")
        if float(self.residual) < 0.0:
            raise ValueError("ExteriorGapWitness residual must be nonnegative")
        if not self.provenance:
            raise ValueError("ExteriorGapWitness provenance must be nonempty")


@dataclass(frozen=True)
class PointContactSignature(Witness):
    """Complete four-ray signature of two loops meeting at one point.

    ``rays`` are stored in increasing counter-clockwise direction.  Each loop
    owns two rays, and the owner word must have two (not four) cyclic
    transitions, i.e. ``A,A,B,B`` modulo rotation and reflection.  The two
    cross-owner cyclic gaps are retained as evidence rather than collapsing
    the configuration to an indiscriminate minimum angle.
    """

    vertex: PointWitness = field(default_factory=PointWitness)
    part_ids: tuple[str, str] = ("", "")
    contact_count: int = 1
    loop_incidence: tuple[tuple[str, bool, bool], ...] = ()
    rays: tuple[IncidentRayWitness, ...] = ()
    cyclic_owners: tuple[str, ...] = ()
    exterior_gaps: tuple[ExteriorGapWitness, ...] = ()

    def __post_init__(self) -> None:
        base_values = {
            "vertex.x": self.vertex.x,
            "vertex.y": self.vertex.y,
            "confidence": self.confidence,
            "residual": self.residual,
        }
        if any(isinstance(value, bool) or not math.isfinite(float(value))
               for value in base_values.values()):
            raise ValueError("PointContactSignature numeric fields must be finite")
        if len(self.part_ids) != 2 or any(not item for item in self.part_ids) \
                or len(set(self.part_ids)) != 2:
            raise ValueError(
                "PointContactSignature requires exactly two distinct part IDs")
        if isinstance(self.contact_count, bool) or self.contact_count != 1:
            raise ValueError("PointContactSignature requires exactly one contact")
        incidence_ids: list[str] = []
        for item in self.loop_incidence:
            if len(item) != 3:
                raise ValueError(
                    "loop incidence entries are (part_id,start_at_vertex,end_at_vertex)")
            part_id, start_at_vertex, end_at_vertex = item
            if type(start_at_vertex) is not bool or type(end_at_vertex) is not bool:
                raise ValueError("loop incidence flags must be booleans")
            if not start_at_vertex or not end_at_vertex:
                raise ValueError(
                    "each point-contact loop must start and end at the vertex")
            incidence_ids.append(part_id)
        if len(incidence_ids) != 2 or set(incidence_ids) != set(self.part_ids):
            raise ValueError(
                "loop incidence must certify both and only the two parts")
        if len(self.rays) != 4:
            raise ValueError("PointContactSignature requires exactly four rays")
        ray_ids = tuple(ray.ray_id for ray in self.rays)
        if len(set(ray_ids)) != 4:
            raise ValueError("PointContactSignature ray IDs must be unique")
        owner_counts = {
            part_id: sum(ray.owner_id == part_id for ray in self.rays)
            for part_id in self.part_ids
        }
        if owner_counts != {part_id: 2 for part_id in self.part_ids}:
            raise ValueError("each point-contact part must own exactly two rays")
        if any(ray.owner_id not in self.part_ids for ray in self.rays):
            raise ValueError("point-contact ray has an unknown owner")
        directions = tuple(float(ray.direction_degrees) for ray in self.rays)
        if any(right <= left for left, right in zip(directions, directions[1:])):
            raise ValueError(
                "PointContactSignature rays must be in strict cyclic order")
        owners = tuple(ray.owner_id for ray in self.rays)
        if self.cyclic_owners != owners:
            raise ValueError("cyclic_owners must match the ordered ray owners")
        transitions = sum(
            owners[index] != owners[(index + 1) % len(owners)]
            for index in range(len(owners))
        )
        if transitions != 2:
            raise ValueError(
                "point-contact ownership must be non-interleaving A,A,B,B")
        if len(self.exterior_gaps) != 2:
            raise ValueError(
                "PointContactSignature requires two exterior cross-owner gaps")
        expected_gaps: dict[tuple[str, str], tuple[float, float, str, str]] = {}
        for index, ray in enumerate(self.rays):
            following = self.rays[(index + 1) % len(self.rays)]
            if ray.owner_id != following.owner_id:
                expected_gaps[(ray.ray_id, following.ray_id)] = (
                    (following.direction_degrees - ray.direction_degrees)
                    % 360.0,
                    ray.uncertainty_degrees
                    + following.uncertainty_degrees,
                    ray.owner_id,
                    following.owner_id,
                )
        observed_pairs = {
            (gap.ray_a_id, gap.ray_b_id) for gap in self.exterior_gaps
        }
        if observed_pairs != set(expected_gaps) or len(observed_pairs) != 2:
            raise ValueError(
                "exterior gaps must be the two adjacent cross-owner ray pairs")
        for gap in self.exterior_gaps:
            degrees, uncertainty, owner_a, owner_b = expected_gaps[
                (gap.ray_a_id, gap.ray_b_id)]
            if gap.owner_a != owner_a or gap.owner_b != owner_b:
                raise ValueError("exterior gap owner labels do not match its rays")
            if not math.isclose(gap.degrees, degrees, rel_tol=0.0, abs_tol=1e-9):
                raise ValueError("exterior gap degrees do not match its rays")
            if not math.isclose(
                    gap.uncertainty_degrees, uncertainty,
                    rel_tol=0.0, abs_tol=1e-9):
                raise ValueError(
                    "exterior gap uncertainty does not match its rays")
            ray_by_id = {ray.ray_id: ray for ray in self.rays}
            first = ray_by_id[gap.ray_a_id]
            second = ray_by_id[gap.ray_b_id]
            if not math.isclose(
                    gap.residual, max(first.residual, second.residual),
                    rel_tol=0.0, abs_tol=1e-12):
                raise ValueError("exterior gap residual does not match its rays")
            if not math.isclose(
                    gap.confidence, min(first.confidence, second.confidence),
                    rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(
                    "exterior gap confidence does not match its rays")
        if self.exterior_gaps[0].degrees > self.exterior_gaps[1].degrees:
            raise ValueError("exterior gaps must be ordered small then large")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("PointContactSignature confidence must be in [0, 1]")
        if float(self.residual) < 0.0:
            raise ValueError("PointContactSignature residual must be nonnegative")
        if not self.provenance:
            raise ValueError("PointContactSignature provenance must be nonempty")


@dataclass(frozen=True)
class AngleWitness(Witness):
    """Unsigned angle grounded by two fitted segments at a shared vertex.

    ``uncertainty_degrees`` carries the fit-derived angular uncertainty rather
    than silently sharpening a noisy pair of lines into an exact angle.
    ``reference_frame`` distinguishes intrinsic segment-pair angles from
    future panel-axis orientation measurements.
    """

    source_a: str = ""
    source_b: str = ""
    vertex: PointWitness = field(default_factory=PointWitness)
    degrees: float = 0.0
    uncertainty_degrees: float = 0.0
    reference_frame: str = "interior"

    def __post_init__(self) -> None:
        values = {
            "degrees": self.degrees,
            "uncertainty_degrees": self.uncertainty_degrees,
            "vertex.x": self.vertex.x,
            "vertex.y": self.vertex.y,
            "confidence": self.confidence,
            "residual": self.residual,
        }
        if any(isinstance(value, bool) or not math.isfinite(float(value))
               for value in values.values()):
            raise ValueError("AngleWitness numeric fields must be finite")
        if not 0.0 <= float(self.degrees) <= 180.0:
            raise ValueError("AngleWitness degrees must be in [0, 180]")
        if float(self.uncertainty_degrees) < 0.0:
            raise ValueError("AngleWitness uncertainty must be nonnegative")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("AngleWitness confidence must be in [0, 1]")
        if float(self.residual) < 0.0:
            raise ValueError("AngleWitness residual must be nonnegative")
        if self.reference_frame not in {"interior", "panel_axes"}:
            raise ValueError(
                "AngleWitness reference_frame must be interior or panel_axes")


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
class CirclePairWitness(PairWitness):
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
