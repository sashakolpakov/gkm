"""Practical cofibration contracts for witness-preserving extensions."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class CofibrationSpec:
    name: str
    source_type: str
    target_type: str
    preserved_fields: tuple[str, ...]
    interface_fields: tuple[str, ...]
    added_fields: tuple[str, ...]
    attachment_leg: str
    projection_leg: str | None = None
    complexity_cost: int = 1

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class CofibrationCheck:
    ok: bool
    first_failed: str = ""
    details: str = ""


def _field(value: Any, name: str) -> Any:
    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name, None)


def verify_witness_preservation(source: Any, target: Any,
                                spec: CofibrationSpec) -> CofibrationCheck:
    for field in spec.preserved_fields:
        if _field(source, field) != _field(target, field):
            return CofibrationCheck(False, "witness_preservation", field)
    return CofibrationCheck(True)


def verify_interface_local_attachment(source: Any, target: Any,
                                      spec: CofibrationSpec) -> CofibrationCheck:
    for field in spec.interface_fields:
        if _field(target, field) is None:
            return CofibrationCheck(False, "interface_local_attachment", field)
    for field in spec.added_fields:
        if _field(target, field) is None:
            return CofibrationCheck(False, "missing_added_field", field)
    return CofibrationCheck(True)


def verify_projection_recovers_source(source: Any, target: Any,
                                      spec: CofibrationSpec,
                                      projection_fn=None) -> CofibrationCheck:
    if spec.projection_leg is None and projection_fn is None:
        return CofibrationCheck(True)
    projected = projection_fn(target) if projection_fn else _field(target, spec.projection_leg or "")
    if projected != source:
        return CofibrationCheck(False, "projection_recovers_source", spec.projection_leg or "")
    return CofibrationCheck(True)


def verify_trace_embedding(source_trace: dict[str, Any], target_trace: dict[str, Any],
                           spec: CofibrationSpec) -> CofibrationCheck:
    for field in spec.preserved_fields:
        if source_trace.get(field) != target_trace.get(field):
            return CofibrationCheck(False, "trace_embedding", field)
    return CofibrationCheck(True)


def verify_cofibration(source: Any, target: Any, spec: CofibrationSpec,
                       source_trace: dict[str, Any] | None = None,
                       target_trace: dict[str, Any] | None = None,
                       projection_fn=None) -> CofibrationCheck:
    checks = (
        verify_witness_preservation(source, target, spec),
        verify_interface_local_attachment(source, target, spec),
        verify_projection_recovers_source(source, target, spec, projection_fn),
    )
    for check in checks:
        if not check.ok:
            return check
    if source_trace is not None and target_trace is not None:
        return verify_trace_embedding(source_trace, target_trace, spec)
    return CofibrationCheck(True)


BIRD_APPENDAGE_COFIBRATION = CofibrationSpec(
    name="body_to_body_with_paired_appendages",
    source_type="PartWitness",
    target_type="PartGraphWitness",
    preserved_fields=("source_component_id",),
    interface_fields=("contacts",),
    added_fields=("parts",),
    attachment_leg="detect_attachment",
    complexity_cost=3,
)

PINWHEEL_COFIBRATION = CofibrationSpec(
    name="center_to_four_blade_radial_arrangement",
    source_type="PointWitness",
    target_type="RadialArrangementWitness",
    preserved_fields=(),
    interface_fields=("center",),
    added_fields=("parts", "part_count"),
    attachment_leg="detect_radial_arrangement",
    complexity_cost=3,
)

TRIANGLE_SQUARE_COFIBRATION = CofibrationSpec(
    name="quadrilateral_with_attached_triangle",
    source_type="QuadrilateralWitness",
    target_type="PartGraphWitness",
    preserved_fields=("source_component_id",),
    interface_fields=("contacts",),
    added_fields=("parts",),
    attachment_leg="detect_attachment",
    complexity_cost=3,
)

CIRCLE_INTERSECTION_COFIBRATION = CofibrationSpec(
    name="circle_pair_with_intersection",
    source_type="CircleWitness",
    target_type="CircleIntersectionWitness",
    preserved_fields=(),
    interface_fields=("points",),
    added_fields=("pair",),
    attachment_leg="circle_pair_intersection",
    projection_leg="pair",
    complexity_cost=2,
)
