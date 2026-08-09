"""Candidate-independent macro action geometry for textured Bongard panels.

Rendered ink is not the object measured here.  A visual observer supplies one
ordered, simplified *macro action trace* through any zigzags or repeated
markers, while recording those decorations in a separate micro-texture layer.
Python then derives two independent observables from the macro trace:

* convexity of the complete closed simplified trace; and
* the number (or conservative interval) of straight macro action spans.

The representation is deliberately closed and content addressed.  A missing,
failed, open, ambiguous, or insufficiently resolved macro trace evaluates as
``INDETERMINATE`` and can never be laundered into a negative predicate.  No
pixels, prose, arbitrary code, or Lean term is interpreted by this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re
from typing import Any, Mapping, Sequence

from bongard.canonical import canonical_digest
from bongard.panel_feature_observation import EngineeringFeatureDisposition
from bongard.panel_soft_ontology import (
    BoundaryPolygonError,
    BoundaryPolygonIssue,
    CanonicalBoundaryPolygon,
    ClosedCount,
    ConvexityKind,
    QuantizedPoint,
)


HIERARCHICAL_ACTION_GEOMETRY_PROTOCOL_ID = (
    "bongard.panel-hierarchical-action-geometry/ordered-macro-trace-python-v1"
)
HIERARCHICAL_ACTION_GEOMETRY_EVIDENCE_SCHEMA = (
    "gkm.bongard-hierarchical-action-geometry-evidence.v1"
)
MACRO_ACTION_TRACE_SCHEMA = "gkm.bongard-macro-action-trace.v1"
MACRO_ACTION_SPAN_SCHEMA = "gkm.bongard-macro-action-span.v1"
MICRO_TEXTURE_EVIDENCE_SCHEMA = "gkm.bongard-micro-texture-evidence.v1"
GEOMETRY_PROVENANCE_SCHEMA = "gkm.bongard-action-geometry-provenance.v1"
GEOMETRY_REPLAY_SCHEMA = "gkm.bongard-action-geometry-cold-replay.v1"

MAX_MACRO_SPANS = 12
MAX_MACRO_CONTROL_POINTS = 64
MAX_MICRO_PRIMITIVES = 64
MAX_MICRO_POINTS = 256

_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_COUNT_VALUE = {
    ClosedCount.ONE: 1,
    ClosedCount.TWO: 2,
    ClosedCount.THREE: 3,
    ClosedCount.FOUR: 4,
    ClosedCount.FIVE: 5,
    ClosedCount.SIX: 6,
    ClosedCount.SEVEN: 7,
    ClosedCount.EIGHT: 8,
    ClosedCount.NINE: 9,
    ClosedCount.TEN: 10,
    ClosedCount.ELEVEN: 11,
    ClosedCount.TWELVE: 12,
}


class HierarchicalActionGeometryError(ValueError):
    """A layer, trace, provenance edge, derivation, or replay differs."""


class TraceResolution(str, Enum):
    COMPLETE = "complete"
    INDETERMINATE = "indeterminate"
    ERROR = "error"


class GeometryTraceIssue(str, Enum):
    MISSING_ORDERED_MACRO_TRACE = "missing_ordered_macro_trace"
    AMBIGUOUS_PRIMITIVE = "ambiguous_primitive"
    AMBIGUOUS_GEOMETRY = "ambiguous_geometry"
    OPEN_MACRO_TRACE = "open_macro_trace"
    DEGENERATE_MACRO_TRACE = "degenerate_macro_trace"
    SELF_INTERSECTING_MACRO_TRACE = "self_intersecting_macro_trace"
    RESOLUTION_LIMIT = "resolution_limit"
    CAPACITY_LIMIT = "capacity_limit"
    PARSER_FAILURE = "parser_failure"
    TRANSPORT_FAILURE = "transport_failure"
    INTEGRITY_FAILURE = "integrity_failure"


class MacroActionPrimitive(str, Enum):
    LINE = "line"
    ARC = "arc"
    INDETERMINATE = "indeterminate"


class DerivedMacroSpanKind(str, Enum):
    STRAIGHT = "straight"
    CURVED = "curved"
    INDETERMINATE = "indeterminate"


class MicroTexturePrimitiveKind(str, Enum):
    ZIGZAG_STROKE = "zigzag_stroke"
    MARKER_DOT = "marker_dot"
    MARKER_CIRCLE = "marker_circle"
    MARKER_SQUARE = "marker_square"
    MARKER_TRIANGLE = "marker_triangle"


class GeometryDerivationStatus(str, Enum):
    RESOLVED = "resolved"
    INDETERMINATE = "indeterminate"
    ERROR = "error"


_ERROR_TRACE_ISSUES = frozenset(
    {
        GeometryTraceIssue.PARSER_FAILURE,
        GeometryTraceIssue.TRANSPORT_FAILURE,
        GeometryTraceIssue.INTEGRITY_FAILURE,
    }
)


def _fields(value: object, expected: set[str], label: str) -> Mapping[str, Any]:
    if (
        not isinstance(value, Mapping)
        or any(type(key) is not str for key in value)
        or set(value) != expected
    ):
        raise HierarchicalActionGeometryError(f"{label} fields differ")
    return value


def _digest(value: object, label: str) -> str:
    if type(value) is not str or _DIGEST.fullmatch(value) is None:
        raise HierarchicalActionGeometryError(
            f"{label} must be a raw lowercase SHA-256"
        )
    return value


@dataclass(frozen=True, order=True, slots=True)
class Grid16Interval:
    """One closed integer coordinate interval on the fixed Grid16 lattice."""

    minimum: int
    maximum: int

    def __post_init__(self) -> None:
        if (
            type(self.minimum) is not int
            or type(self.maximum) is not int
            or not 0 <= self.minimum <= self.maximum <= 15
        ):
            raise HierarchicalActionGeometryError(
                "Grid16 interval must be an ordered integer pair in [0, 15]"
            )

    @property
    def singleton(self) -> bool:
        return self.minimum == self.maximum

    def to_data(self) -> dict[str, int]:
        return {"minimum": self.minimum, "maximum": self.maximum}

    @classmethod
    def from_data(cls, value: object) -> "Grid16Interval":
        raw = _fields(value, {"minimum", "maximum"}, "Grid16 interval")
        result = cls(raw["minimum"], raw["maximum"])
        if result.to_data() != dict(raw):
            raise HierarchicalActionGeometryError("Grid16 interval is not canonical")
        return result


@dataclass(frozen=True, order=True, slots=True)
class UncertainGrid16Point:
    """A Grid16 point with explicit closed coordinate uncertainty."""

    x: Grid16Interval
    y: Grid16Interval

    def __post_init__(self) -> None:
        if type(self.x) is not Grid16Interval or type(self.y) is not Grid16Interval:
            raise TypeError("uncertain point needs two Grid16Interval values")

    @classmethod
    def exact(cls, x: int, y: int) -> "UncertainGrid16Point":
        return cls(Grid16Interval(x, x), Grid16Interval(y, y))

    @property
    def is_exact(self) -> bool:
        return self.x.singleton and self.y.singleton

    def exact_point(self) -> QuantizedPoint | None:
        if not self.is_exact:
            return None
        return QuantizedPoint(self.x.minimum, self.y.minimum)

    def to_data(self) -> dict[str, object]:
        return {"x": self.x.to_data(), "y": self.y.to_data()}

    @classmethod
    def from_data(cls, value: object) -> "UncertainGrid16Point":
        raw = _fields(value, {"x", "y"}, "uncertain Grid16 point")
        result = cls(
            Grid16Interval.from_data(raw["x"]),
            Grid16Interval.from_data(raw["y"]),
        )
        if result.to_data() != dict(raw):
            raise HierarchicalActionGeometryError(
                "uncertain Grid16 point is not canonical"
            )
        return result


@dataclass(frozen=True, order=True, slots=True)
class _SignedInterval:
    minimum: int
    maximum: int

    def __post_init__(self) -> None:
        if self.minimum > self.maximum:
            raise HierarchicalActionGeometryError("signed interval is reversed")

    def __sub__(self, other: "_SignedInterval") -> "_SignedInterval":
        return _SignedInterval(
            self.minimum - other.maximum, self.maximum - other.minimum
        )

    def __mul__(self, other: "_SignedInterval") -> "_SignedInterval":
        products = (
            self.minimum * other.minimum,
            self.minimum * other.maximum,
            self.maximum * other.minimum,
            self.maximum * other.maximum,
        )
        return _SignedInterval(min(products), max(products))

    @classmethod
    def coordinate(cls, value: Grid16Interval) -> "_SignedInterval":
        return cls(value.minimum, value.maximum)


def _cross_interval(
    start: UncertainGrid16Point,
    point: UncertainGrid16Point,
    end: UncertainGrid16Point,
) -> _SignedInterval:
    sx, sy = _SignedInterval.coordinate(start.x), _SignedInterval.coordinate(start.y)
    px, py = _SignedInterval.coordinate(point.x), _SignedInterval.coordinate(point.y)
    ex, ey = _SignedInterval.coordinate(end.x), _SignedInterval.coordinate(end.y)
    return ((ex - sx) * (py - sy)) - ((ey - sy) * (px - sx))


def _certifiably_distinct(
    first: UncertainGrid16Point, second: UncertainGrid16Point
) -> bool:
    return (
        first.x.maximum < second.x.minimum
        or second.x.maximum < first.x.minimum
        or first.y.maximum < second.y.minimum
        or second.y.maximum < first.y.minimum
    )


def _span_content(value: "MacroActionSpan") -> dict[str, object]:
    return {
        "schema": MACRO_ACTION_SPAN_SCHEMA,
        "resolution": value.resolution.value,
        "primitive": value.primitive.value,
        "ordered_control_evidence": [item.to_data() for item in value.points],
        "issue": None if value.issue is None else value.issue.value,
        "primitive_is_typed_observation_not_executable_prose": True,
    }


@dataclass(frozen=True, order=True, slots=True)
class MacroActionSpan:
    """One base action span; micro rendering strokes are intentionally absent."""

    resolution: TraceResolution
    primitive: MacroActionPrimitive
    points: tuple[UncertainGrid16Point, ...]
    issue: GeometryTraceIssue | None

    def __post_init__(self) -> None:
        if type(self.resolution) is not TraceResolution:
            raise TypeError("macro span resolution must be exact")
        if type(self.primitive) is not MacroActionPrimitive:
            raise TypeError("macro span primitive must be exact")
        if type(self.points) is not tuple or any(
            type(item) is not UncertainGrid16Point for item in self.points
        ):
            raise TypeError("macro span needs an uncertain-point tuple")
        if not 2 <= len(self.points) <= 8:
            raise HierarchicalActionGeometryError(
                "macro span needs 2..8 ordered endpoint/control points"
            )
        if any(a == b for a, b in zip(self.points, self.points[1:])):
            raise HierarchicalActionGeometryError(
                "macro span repeats consecutive control evidence"
            )
        if self.resolution is TraceResolution.COMPLETE:
            if self.primitive is MacroActionPrimitive.INDETERMINATE or self.issue is not None:
                raise HierarchicalActionGeometryError(
                    "complete macro span needs line/arc and no issue"
                )
            if self.primitive is MacroActionPrimitive.LINE and len(self.points) != 2:
                raise HierarchicalActionGeometryError(
                    "line action span is defined by exactly two endpoints"
                )
            if self.primitive is MacroActionPrimitive.ARC and len(self.points) < 3:
                raise HierarchicalActionGeometryError(
                    "arc action span needs endpoint/control/endpoint evidence"
                )
        elif (
            self.primitive is not MacroActionPrimitive.INDETERMINATE
            or type(self.issue) is not GeometryTraceIssue
        ):
            raise HierarchicalActionGeometryError(
                "unresolved macro span needs indeterminate primitive and typed issue"
            )
        elif (self.resolution is TraceResolution.ERROR) != (
            self.issue in _ERROR_TRACE_ISSUES
        ):
            raise HierarchicalActionGeometryError(
                "macro span protocol errors and visual indeterminacy are distinct"
            )

    @property
    def start(self) -> UncertainGrid16Point:
        return self.points[0]

    @property
    def end(self) -> UncertainGrid16Point:
        return self.points[-1]

    @property
    def span_digest(self) -> str:
        return canonical_digest(self.to_data())

    def reversed(self) -> "MacroActionSpan":
        return MacroActionSpan(
            self.resolution, self.primitive, tuple(reversed(self.points)), self.issue
        )

    def to_data(self) -> dict[str, object]:
        return _span_content(self)

    @classmethod
    def from_data(cls, value: object) -> "MacroActionSpan":
        raw = _fields(
            value,
            {
                "schema",
                "resolution",
                "primitive",
                "ordered_control_evidence",
                "issue",
                "primitive_is_typed_observation_not_executable_prose",
            },
            "macro action span",
        )
        if (
            raw["schema"] != MACRO_ACTION_SPAN_SCHEMA
            or raw["primitive_is_typed_observation_not_executable_prose"] is not True
            or type(raw["ordered_control_evidence"]) is not list
        ):
            raise HierarchicalActionGeometryError("macro span policy differs")
        try:
            result = cls(
                TraceResolution(raw["resolution"]),
                MacroActionPrimitive(raw["primitive"]),
                tuple(
                    UncertainGrid16Point.from_data(item)
                    for item in raw["ordered_control_evidence"]
                ),
                None if raw["issue"] is None else GeometryTraceIssue(raw["issue"]),
            )
        except (TypeError, ValueError) as exc:
            if isinstance(exc, HierarchicalActionGeometryError):
                raise
            raise HierarchicalActionGeometryError("macro span value differs") from exc
        if result.to_data() != dict(raw):
            raise HierarchicalActionGeometryError("macro span is not canonical")
        return result


def _span_key(value: MacroActionSpan) -> tuple[object, ...]:
    return (
        value.resolution.value,
        value.primitive.value,
        tuple(
            (p.x.minimum, p.x.maximum, p.y.minimum, p.y.maximum)
            for p in value.points
        ),
        "" if value.issue is None else value.issue.value,
    )


def _canonical_cycle(
    spans: tuple[MacroActionSpan, ...]
) -> tuple[MacroActionSpan, ...]:
    if not spans:
        return ()
    forward = spans
    reverse = tuple(item.reversed() for item in reversed(spans))
    candidates = tuple(
        sequence[index:] + sequence[:index]
        for sequence in (forward, reverse)
        for index in range(len(sequence))
    )
    return min(candidates, key=lambda rows: tuple(_span_key(item) for item in rows))


def _trace_content(value: "MacroActionTrace") -> dict[str, object]:
    return {
        "schema": MACRO_ACTION_TRACE_SCHEMA,
        "resolution": value.resolution.value,
        "ordered_spans": [item.to_data() for item in value.spans],
        "issue": None if value.issue is None else value.issue.value,
        "closure": "last_span_end_equals_first_span_start",
        "cycle_start_and_direction": "lexicographically_canonical",
        "macro_carrier": "simplified_centerline_action_trace",
        "raw_ink_envelope_or_convex_hull_used": False,
    }


@dataclass(frozen=True, slots=True)
class MacroActionTrace:
    """Complete ordered simplified macro centerline, or a typed trace gap."""

    resolution: TraceResolution
    spans: tuple[MacroActionSpan, ...]
    issue: GeometryTraceIssue | None

    def __post_init__(self) -> None:
        if type(self.resolution) is not TraceResolution:
            raise TypeError("macro trace resolution must be exact")
        if type(self.spans) is not tuple or any(
            type(item) is not MacroActionSpan for item in self.spans
        ):
            raise TypeError("macro trace spans must be an exact tuple")
        if self.resolution is TraceResolution.COMPLETE:
            if (
                not 2 <= len(self.spans) <= MAX_MACRO_SPANS
                or self.issue is not None
                or sum(len(item.points) for item in self.spans)
                > MAX_MACRO_CONTROL_POINTS
            ):
                raise HierarchicalActionGeometryError(
                    "complete macro trace shape or capacity differs"
                )
            if any(
                item.end != self.spans[(index + 1) % len(self.spans)].start
                for index, item in enumerate(self.spans)
            ):
                raise HierarchicalActionGeometryError(
                    "ordered macro trace is not explicitly closed and continuous"
                )
            if _canonical_cycle(self.spans) != self.spans:
                raise HierarchicalActionGeometryError(
                    "macro trace cycle start/direction is not canonical"
                )
        elif self.spans or type(self.issue) is not GeometryTraceIssue:
            raise HierarchicalActionGeometryError(
                "missing/failed macro trace must contain no partial spans and one issue"
            )
        elif (self.resolution is TraceResolution.ERROR) != (
            self.issue in _ERROR_TRACE_ISSUES
        ):
            raise HierarchicalActionGeometryError(
                "macro trace protocol errors and visual indeterminacy are distinct"
            )

    @classmethod
    def complete(cls, spans: Sequence[MacroActionSpan]) -> "MacroActionTrace":
        if isinstance(spans, (str, bytes, Mapping)):
            raise TypeError("macro spans must be an ordered sequence")
        rows = tuple(spans)
        return cls(TraceResolution.COMPLETE, _canonical_cycle(rows), None)

    @classmethod
    def gap(
        cls, resolution: TraceResolution, issue: GeometryTraceIssue
    ) -> "MacroActionTrace":
        if resolution is TraceResolution.COMPLETE:
            raise HierarchicalActionGeometryError("trace gap cannot be complete")
        return cls(resolution, (), issue)

    @property
    def trace_digest(self) -> str:
        return canonical_digest(self.to_data())

    def to_data(self) -> dict[str, object]:
        return _trace_content(self)

    @classmethod
    def from_data(cls, value: object) -> "MacroActionTrace":
        raw = _fields(
            value,
            {
                "schema",
                "resolution",
                "ordered_spans",
                "issue",
                "closure",
                "cycle_start_and_direction",
                "macro_carrier",
                "raw_ink_envelope_or_convex_hull_used",
            },
            "macro action trace",
        )
        if (
            raw["schema"] != MACRO_ACTION_TRACE_SCHEMA
            or raw["closure"] != "last_span_end_equals_first_span_start"
            or raw["cycle_start_and_direction"] != "lexicographically_canonical"
            or raw["macro_carrier"] != "simplified_centerline_action_trace"
            or raw["raw_ink_envelope_or_convex_hull_used"] is not False
            or type(raw["ordered_spans"]) is not list
        ):
            raise HierarchicalActionGeometryError("macro trace policy differs")
        try:
            result = cls(
                TraceResolution(raw["resolution"]),
                tuple(MacroActionSpan.from_data(item) for item in raw["ordered_spans"]),
                None if raw["issue"] is None else GeometryTraceIssue(raw["issue"]),
            )
        except (TypeError, ValueError) as exc:
            if isinstance(exc, HierarchicalActionGeometryError):
                raise
            raise HierarchicalActionGeometryError("macro trace value differs") from exc
        if result.to_data() != dict(raw):
            raise HierarchicalActionGeometryError("macro trace is not canonical")
        return result


@dataclass(frozen=True, order=True, slots=True)
class MicroTexturePrimitive:
    kind: MicroTexturePrimitiveKind
    points: tuple[UncertainGrid16Point, ...]

    def __post_init__(self) -> None:
        if type(self.kind) is not MicroTexturePrimitiveKind:
            raise TypeError("micro primitive kind must be exact")
        if type(self.points) is not tuple or any(
            type(item) is not UncertainGrid16Point for item in self.points
        ):
            raise TypeError("micro primitive points must be an exact tuple")
        expected = (
            range(2, 17)
            if self.kind is MicroTexturePrimitiveKind.ZIGZAG_STROKE
            else range(1, 2)
        )
        if len(self.points) not in expected:
            raise HierarchicalActionGeometryError(
                "micro stroke needs 2..16 points; a marker needs one location"
            )

    @property
    def primitive_digest(self) -> str:
        return canonical_digest(self.to_data())

    def to_data(self) -> dict[str, object]:
        return {
            "kind": self.kind.value,
            "points": [item.to_data() for item in self.points],
        }

    @classmethod
    def from_data(cls, value: object) -> "MicroTexturePrimitive":
        raw = _fields(value, {"kind", "points"}, "micro texture primitive")
        if type(raw["points"]) is not list:
            raise HierarchicalActionGeometryError("micro primitive points differ")
        try:
            result = cls(
                MicroTexturePrimitiveKind(raw["kind"]),
                tuple(UncertainGrid16Point.from_data(item) for item in raw["points"]),
            )
        except (TypeError, ValueError) as exc:
            if isinstance(exc, HierarchicalActionGeometryError):
                raise
            raise HierarchicalActionGeometryError("micro primitive value differs") from exc
        if result.to_data() != dict(raw):
            raise HierarchicalActionGeometryError("micro primitive is not canonical")
        return result


@dataclass(frozen=True, slots=True)
class MicroTextureEvidence:
    resolution: TraceResolution
    primitives: tuple[MicroTexturePrimitive, ...]
    issue: GeometryTraceIssue | None

    def __post_init__(self) -> None:
        if type(self.resolution) is not TraceResolution:
            raise TypeError("micro evidence resolution must be exact")
        if type(self.primitives) is not tuple or any(
            type(item) is not MicroTexturePrimitive for item in self.primitives
        ):
            raise TypeError("micro evidence primitives must be an exact tuple")
        if self.resolution is TraceResolution.COMPLETE:
            if (
                self.issue is not None
                or len(self.primitives) > MAX_MICRO_PRIMITIVES
                or sum(len(item.points) for item in self.primitives) > MAX_MICRO_POINTS
                or tuple(item.primitive_digest for item in self.primitives)
                != tuple(sorted(item.primitive_digest for item in self.primitives))
            ):
                raise HierarchicalActionGeometryError(
                    "complete micro evidence order, issue, or capacity differs"
                )
        elif self.primitives or type(self.issue) is not GeometryTraceIssue:
            raise HierarchicalActionGeometryError(
                "unresolved micro evidence must be empty with one issue"
            )
        elif (self.resolution is TraceResolution.ERROR) != (
            self.issue in _ERROR_TRACE_ISSUES
        ):
            raise HierarchicalActionGeometryError(
                "micro protocol errors and visual indeterminacy are distinct"
            )

    @classmethod
    def complete(
        cls, primitives: Sequence[MicroTexturePrimitive] = ()
    ) -> "MicroTextureEvidence":
        rows = tuple(sorted(tuple(primitives), key=lambda item: item.primitive_digest))
        return cls(TraceResolution.COMPLETE, rows, None)

    @classmethod
    def gap(
        cls, resolution: TraceResolution, issue: GeometryTraceIssue
    ) -> "MicroTextureEvidence":
        if resolution is TraceResolution.COMPLETE:
            raise HierarchicalActionGeometryError("micro gap cannot be complete")
        return cls(resolution, (), issue)

    @property
    def evidence_digest(self) -> str:
        return canonical_digest(self.to_data())

    def to_data(self) -> dict[str, object]:
        return {
            "schema": MICRO_TEXTURE_EVIDENCE_SCHEMA,
            "resolution": self.resolution.value,
            "primitives": [item.to_data() for item in self.primitives],
            "issue": None if self.issue is None else self.issue.value,
            "macro_geometry_effect": "none",
        }

    @classmethod
    def from_data(cls, value: object) -> "MicroTextureEvidence":
        raw = _fields(
            value,
            {"schema", "resolution", "primitives", "issue", "macro_geometry_effect"},
            "micro texture evidence",
        )
        if (
            raw["schema"] != MICRO_TEXTURE_EVIDENCE_SCHEMA
            or raw["macro_geometry_effect"] != "none"
            or type(raw["primitives"]) is not list
        ):
            raise HierarchicalActionGeometryError("micro evidence policy differs")
        result = cls(
            TraceResolution(raw["resolution"]),
            tuple(MicroTexturePrimitive.from_data(item) for item in raw["primitives"]),
            None if raw["issue"] is None else GeometryTraceIssue(raw["issue"]),
        )
        if result.to_data() != dict(raw):
            raise HierarchicalActionGeometryError("micro evidence is not canonical")
        return result


@dataclass(frozen=True, slots=True)
class GeometryEvidenceProvenance:
    panel_png_digest: str
    panel_png_byte_count: int
    observer_contract_digest: str
    measurement_protocol_digest: str
    observation_receipt_digest: str
    calibration_binding_digest: str | None = None

    def __post_init__(self) -> None:
        for label, value in (
            ("panel PNG", self.panel_png_digest),
            ("observer contract", self.observer_contract_digest),
            ("measurement protocol", self.measurement_protocol_digest),
            ("observation receipt", self.observation_receipt_digest),
        ):
            _digest(value, f"{label} digest")
        if type(self.panel_png_byte_count) is not int or self.panel_png_byte_count <= 8:
            raise HierarchicalActionGeometryError("panel PNG byte count differs")
        if self.calibration_binding_digest is not None:
            _digest(self.calibration_binding_digest, "calibration binding digest")

    @property
    def provenance_digest(self) -> str:
        return canonical_digest(self.to_data())

    def to_data(self) -> dict[str, object]:
        return {
            "schema": GEOMETRY_PROVENANCE_SCHEMA,
            "panel_png_digest": self.panel_png_digest,
            "panel_png_byte_count": self.panel_png_byte_count,
            "observer_contract_digest": self.observer_contract_digest,
            "measurement_protocol_digest": self.measurement_protocol_digest,
            "observation_receipt_digest": self.observation_receipt_digest,
            "calibration_binding_digest": self.calibration_binding_digest,
            "candidate_specs_model_visible": False,
            "support_or_query_role_model_visible": False,
            "side_or_class_label_model_visible": False,
            "formula_model_visible": False,
            "python_is_derivation_authority": True,
            "lean_present": False,
        }

    @classmethod
    def from_data(cls, value: object) -> "GeometryEvidenceProvenance":
        raw = _fields(
            value,
            {
                "schema",
                "panel_png_digest",
                "panel_png_byte_count",
                "observer_contract_digest",
                "measurement_protocol_digest",
                "observation_receipt_digest",
                "calibration_binding_digest",
                "candidate_specs_model_visible",
                "support_or_query_role_model_visible",
                "side_or_class_label_model_visible",
                "formula_model_visible",
                "python_is_derivation_authority",
                "lean_present",
            },
            "geometry provenance",
        )
        if (
            raw["schema"] != GEOMETRY_PROVENANCE_SCHEMA
            or any(
                raw[key] is not False
                for key in (
                    "candidate_specs_model_visible",
                    "support_or_query_role_model_visible",
                    "side_or_class_label_model_visible",
                    "formula_model_visible",
                    "lean_present",
                )
            )
            or raw["python_is_derivation_authority"] is not True
        ):
            raise HierarchicalActionGeometryError("geometry provenance policy differs")
        result = cls(
            raw["panel_png_digest"],
            raw["panel_png_byte_count"],
            raw["observer_contract_digest"],
            raw["measurement_protocol_digest"],
            raw["observation_receipt_digest"],
            raw["calibration_binding_digest"],
        )
        if result.to_data() != dict(raw):
            raise HierarchicalActionGeometryError("geometry provenance is not canonical")
        return result


def _evidence_content(value: "HierarchicalActionGeometryEvidence") -> dict[str, object]:
    return {
        "schema": HIERARCHICAL_ACTION_GEOMETRY_EVIDENCE_SCHEMA,
        "protocol_id": HIERARCHICAL_ACTION_GEOMETRY_PROTOCOL_ID,
        "provenance": value.provenance.to_data(),
        "macro_action_trace": value.macro_action_trace.to_data(),
        "micro_texture_evidence": value.micro_texture_evidence.to_data(),
        "macro_evidence_digest": value.macro_action_trace.trace_digest,
        "micro_evidence_digest": value.micro_texture_evidence.evidence_digest,
        "macro_and_micro_layers_disjoint": True,
        "raw_black_ink_convex_hull_used": False,
        "prose_executable": False,
        "python_is_canonical_authority": True,
        "lean_present": False,
        "lean_required": False,
    }


@dataclass(frozen=True, slots=True)
class HierarchicalActionGeometryEvidence:
    provenance: GeometryEvidenceProvenance
    macro_action_trace: MacroActionTrace
    micro_texture_evidence: MicroTextureEvidence
    record_digest: str

    def __post_init__(self) -> None:
        if type(self.provenance) is not GeometryEvidenceProvenance:
            raise TypeError("hierarchical evidence needs exact provenance")
        if type(self.macro_action_trace) is not MacroActionTrace:
            raise TypeError("hierarchical evidence needs exact macro trace")
        if type(self.micro_texture_evidence) is not MicroTextureEvidence:
            raise TypeError("hierarchical evidence needs exact micro evidence")
        _digest(self.record_digest, "hierarchical evidence record digest")
        if self.record_digest != canonical_digest(_evidence_content(self)):
            raise HierarchicalActionGeometryError("hierarchical evidence digest differs")

    @classmethod
    def create(
        cls,
        provenance: GeometryEvidenceProvenance,
        macro_action_trace: MacroActionTrace,
        micro_texture_evidence: MicroTextureEvidence,
    ) -> "HierarchicalActionGeometryEvidence":
        values = {
            "provenance": provenance,
            "macro_action_trace": macro_action_trace,
            "micro_texture_evidence": micro_texture_evidence,
        }
        provisional = object.__new__(cls)
        for name, item in values.items():
            object.__setattr__(provisional, name, item)
        return cls(**values, record_digest=canonical_digest(_evidence_content(provisional)))

    @property
    def evidence_address(self) -> str:
        return "sha256:" + self.record_digest

    def to_data(self) -> dict[str, object]:
        return {**_evidence_content(self), "record_digest": self.record_digest}

    @classmethod
    def from_data(cls, value: object) -> "HierarchicalActionGeometryEvidence":
        raw = _fields(
            value,
            {
                "schema",
                "protocol_id",
                "provenance",
                "macro_action_trace",
                "micro_texture_evidence",
                "macro_evidence_digest",
                "micro_evidence_digest",
                "macro_and_micro_layers_disjoint",
                "raw_black_ink_convex_hull_used",
                "prose_executable",
                "python_is_canonical_authority",
                "lean_present",
                "lean_required",
                "record_digest",
            },
            "hierarchical action geometry evidence",
        )
        if (
            raw["schema"] != HIERARCHICAL_ACTION_GEOMETRY_EVIDENCE_SCHEMA
            or raw["protocol_id"] != HIERARCHICAL_ACTION_GEOMETRY_PROTOCOL_ID
            or raw["macro_and_micro_layers_disjoint"] is not True
            or raw["raw_black_ink_convex_hull_used"] is not False
            or raw["prose_executable"] is not False
            or raw["python_is_canonical_authority"] is not True
            or raw["lean_present"] is not False
            or raw["lean_required"] is not False
        ):
            raise HierarchicalActionGeometryError("hierarchical evidence policy differs")
        result = cls(
            GeometryEvidenceProvenance.from_data(raw["provenance"]),
            MacroActionTrace.from_data(raw["macro_action_trace"]),
            MicroTextureEvidence.from_data(raw["micro_texture_evidence"]),
            raw["record_digest"],
        )
        if (
            raw["macro_evidence_digest"] != result.macro_action_trace.trace_digest
            or raw["micro_evidence_digest"]
            != result.micro_texture_evidence.evidence_digest
            or result.to_data() != dict(raw)
        ):
            raise HierarchicalActionGeometryError("hierarchical evidence is not canonical")
        return result


def _derived_span_kind(span: MacroActionSpan) -> DerivedMacroSpanKind:
    if span.resolution is not TraceResolution.COMPLETE:
        return DerivedMacroSpanKind.INDETERMINATE
    if not _certifiably_distinct(span.start, span.end):
        return DerivedMacroSpanKind.INDETERMINATE
    if span.primitive is MacroActionPrimitive.LINE:
        return DerivedMacroSpanKind.STRAIGHT
    if span.primitive is not MacroActionPrimitive.ARC:
        return DerivedMacroSpanKind.INDETERMINATE
    crosses = tuple(
        _cross_interval(span.start, point, span.end)
        for point in span.points[1:-1]
    )
    if any(item.maximum < 0 or item.minimum > 0 for item in crosses):
        return DerivedMacroSpanKind.CURVED
    return DerivedMacroSpanKind.INDETERMINATE


@dataclass(frozen=True, slots=True)
class MacroStraightSpanCountDerivation:
    macro_evidence_digest: str
    status: GeometryDerivationStatus
    lower_bound: int
    upper_bound: int
    span_kinds: tuple[DerivedMacroSpanKind, ...]
    issue: GeometryTraceIssue | None
    derivation_digest: str

    def __post_init__(self) -> None:
        _digest(self.macro_evidence_digest, "macro evidence digest")
        if type(self.status) is not GeometryDerivationStatus:
            raise TypeError("straight-count derivation status must be exact")
        if (
            type(self.lower_bound) is not int
            or type(self.upper_bound) is not int
            or not 0 <= self.lower_bound <= self.upper_bound <= MAX_MACRO_SPANS
            or type(self.span_kinds) is not tuple
            or any(type(item) is not DerivedMacroSpanKind for item in self.span_kinds)
        ):
            raise HierarchicalActionGeometryError("straight-count interval differs")
        if self.status is GeometryDerivationStatus.RESOLVED:
            if self.lower_bound != self.upper_bound or self.issue is not None:
                raise HierarchicalActionGeometryError("resolved straight count differs")
        elif type(self.issue) is not GeometryTraceIssue:
            raise HierarchicalActionGeometryError("unresolved straight count needs issue")
        elif (self.status is GeometryDerivationStatus.ERROR) != (
            self.issue in _ERROR_TRACE_ISSUES
        ):
            raise HierarchicalActionGeometryError(
                "straight-count error disposition differs from its issue"
            )
        _digest(self.derivation_digest, "straight-count derivation digest")
        if self.derivation_digest != canonical_digest(self.content_data()):
            raise HierarchicalActionGeometryError("straight-count derivation digest differs")

    def content_data(self) -> dict[str, object]:
        return {
            "schema": "gkm.bongard-macro-straight-span-count.v1",
            "macro_evidence_digest": self.macro_evidence_digest,
            "status": self.status.value,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "span_kinds": [item.value for item in self.span_kinds],
            "issue": None if self.issue is None else self.issue.value,
            "micro_texture_consulted": False,
            "derivation": "count-certified-line-action-spans-with-uncertainty-range",
        }

    def to_data(self) -> dict[str, object]:
        return {**self.content_data(), "derivation_digest": self.derivation_digest}

    @classmethod
    def from_data(cls, value: object) -> "MacroStraightSpanCountDerivation":
        raw = _fields(
            value,
            {
                "schema",
                "macro_evidence_digest",
                "status",
                "lower_bound",
                "upper_bound",
                "span_kinds",
                "issue",
                "micro_texture_consulted",
                "derivation",
                "derivation_digest",
            },
            "macro straight-span derivation",
        )
        if (
            raw["schema"] != "gkm.bongard-macro-straight-span-count.v1"
            or raw["micro_texture_consulted"] is not False
            or raw["derivation"]
            != "count-certified-line-action-spans-with-uncertainty-range"
            or type(raw["span_kinds"]) is not list
        ):
            raise HierarchicalActionGeometryError(
                "macro straight-span derivation policy differs"
            )
        try:
            result = cls(
                raw["macro_evidence_digest"],
                GeometryDerivationStatus(raw["status"]),
                raw["lower_bound"],
                raw["upper_bound"],
                tuple(DerivedMacroSpanKind(item) for item in raw["span_kinds"]),
                None if raw["issue"] is None else GeometryTraceIssue(raw["issue"]),
                raw["derivation_digest"],
            )
        except (TypeError, ValueError) as exc:
            if isinstance(exc, HierarchicalActionGeometryError):
                raise
            raise HierarchicalActionGeometryError(
                "macro straight-span derivation value differs"
            ) from exc
        if result.to_data() != dict(raw):
            raise HierarchicalActionGeometryError(
                "macro straight-span derivation is not canonical"
            )
        return result


@dataclass(frozen=True, slots=True)
class MacroConvexityDerivation:
    macro_evidence_digest: str
    status: GeometryDerivationStatus
    convexity_kind: ConvexityKind | None
    polygon: CanonicalBoundaryPolygon | None
    issue: GeometryTraceIssue | None
    derivation_digest: str

    def __post_init__(self) -> None:
        _digest(self.macro_evidence_digest, "macro evidence digest")
        if type(self.status) is not GeometryDerivationStatus:
            raise TypeError("convexity derivation status must be exact")
        if self.status is GeometryDerivationStatus.RESOLVED:
            if (
                type(self.convexity_kind) is not ConvexityKind
                or type(self.polygon) is not CanonicalBoundaryPolygon
                or self.polygon.convexity_kind is not self.convexity_kind
                or self.issue is not None
            ):
                raise HierarchicalActionGeometryError("resolved convexity differs")
        elif (
            self.convexity_kind is not None
            or self.polygon is not None
            or type(self.issue) is not GeometryTraceIssue
        ):
            raise HierarchicalActionGeometryError("unresolved convexity differs")
        elif (self.status is GeometryDerivationStatus.ERROR) != (
            self.issue in _ERROR_TRACE_ISSUES
        ):
            raise HierarchicalActionGeometryError(
                "convexity error disposition differs from its issue"
            )
        _digest(self.derivation_digest, "convexity derivation digest")
        if self.derivation_digest != canonical_digest(self.content_data()):
            raise HierarchicalActionGeometryError("convexity derivation digest differs")

    def content_data(self) -> dict[str, object]:
        return {
            "schema": "gkm.bongard-macro-trace-convexity.v1",
            "macro_evidence_digest": self.macro_evidence_digest,
            "status": self.status.value,
            "convexity_kind": (
                None if self.convexity_kind is None else self.convexity_kind.value
            ),
            "canonical_simplified_macro_polygon": (
                None if self.polygon is None else self.polygon.to_data()
            ),
            "issue": None if self.issue is None else self.issue.value,
            "raw_black_ink_envelope_or_hull_consulted": False,
            "micro_texture_consulted": False,
            "derivation": "exact-grid16-closed-simple-macro-centerline-turn-signs",
        }

    def to_data(self) -> dict[str, object]:
        return {**self.content_data(), "derivation_digest": self.derivation_digest}

    @classmethod
    def from_data(cls, value: object) -> "MacroConvexityDerivation":
        raw = _fields(
            value,
            {
                "schema",
                "macro_evidence_digest",
                "status",
                "convexity_kind",
                "canonical_simplified_macro_polygon",
                "issue",
                "raw_black_ink_envelope_or_hull_consulted",
                "micro_texture_consulted",
                "derivation",
                "derivation_digest",
            },
            "macro convexity derivation",
        )
        if (
            raw["schema"] != "gkm.bongard-macro-trace-convexity.v1"
            or raw["raw_black_ink_envelope_or_hull_consulted"] is not False
            or raw["micro_texture_consulted"] is not False
            or raw["derivation"]
            != "exact-grid16-closed-simple-macro-centerline-turn-signs"
        ):
            raise HierarchicalActionGeometryError(
                "macro convexity derivation policy differs"
            )
        try:
            result = cls(
                raw["macro_evidence_digest"],
                GeometryDerivationStatus(raw["status"]),
                (
                    None
                    if raw["convexity_kind"] is None
                    else ConvexityKind(raw["convexity_kind"])
                ),
                (
                    None
                    if raw["canonical_simplified_macro_polygon"] is None
                    else CanonicalBoundaryPolygon.from_data(
                        raw["canonical_simplified_macro_polygon"]
                    )
                ),
                None if raw["issue"] is None else GeometryTraceIssue(raw["issue"]),
                raw["derivation_digest"],
            )
        except (TypeError, ValueError) as exc:
            if isinstance(exc, HierarchicalActionGeometryError):
                raise
            raise HierarchicalActionGeometryError(
                "macro convexity derivation value differs"
            ) from exc
        if result.to_data() != dict(raw):
            raise HierarchicalActionGeometryError(
                "macro convexity derivation is not canonical"
            )
        return result


def _make_straight_derivation(**values: object) -> MacroStraightSpanCountDerivation:
    provisional = object.__new__(MacroStraightSpanCountDerivation)
    for name, item in values.items():
        object.__setattr__(provisional, name, item)
    return MacroStraightSpanCountDerivation(
        **values,  # type: ignore[arg-type]
        derivation_digest=canonical_digest(provisional.content_data()),
    )


def _make_convexity_derivation(**values: object) -> MacroConvexityDerivation:
    provisional = object.__new__(MacroConvexityDerivation)
    for name, item in values.items():
        object.__setattr__(provisional, name, item)
    return MacroConvexityDerivation(
        **values,  # type: ignore[arg-type]
        derivation_digest=canonical_digest(provisional.content_data()),
    )


def derive_macro_straight_span_count(
    evidence: HierarchicalActionGeometryEvidence,
) -> MacroStraightSpanCountDerivation:
    """Count base line-action spans; never count zigzag/stamp micro ink."""

    if type(evidence) is not HierarchicalActionGeometryEvidence:
        raise TypeError("straight-count derivation needs hierarchical evidence")
    trace = evidence.macro_action_trace
    if trace.resolution is not TraceResolution.COMPLETE:
        return _make_straight_derivation(
            macro_evidence_digest=trace.trace_digest,
            status=(
                GeometryDerivationStatus.ERROR
                if trace.resolution is TraceResolution.ERROR
                else GeometryDerivationStatus.INDETERMINATE
            ),
            lower_bound=0,
            upper_bound=MAX_MACRO_SPANS,
            span_kinds=(),
            issue=trace.issue,
        )
    span_error = next(
        (item.issue for item in trace.spans if item.resolution is TraceResolution.ERROR),
        None,
    )
    if span_error is not None:
        return _make_straight_derivation(
            macro_evidence_digest=trace.trace_digest,
            status=GeometryDerivationStatus.ERROR,
            lower_bound=0,
            upper_bound=MAX_MACRO_SPANS,
            span_kinds=tuple(_derived_span_kind(item) for item in trace.spans),
            issue=span_error,
        )
    kinds = tuple(_derived_span_kind(item) for item in trace.spans)
    lower = sum(item is DerivedMacroSpanKind.STRAIGHT for item in kinds)
    upper = lower + sum(item is DerivedMacroSpanKind.INDETERMINATE for item in kinds)
    resolved = lower == upper
    return _make_straight_derivation(
        macro_evidence_digest=trace.trace_digest,
        status=(
            GeometryDerivationStatus.RESOLVED
            if resolved
            else GeometryDerivationStatus.INDETERMINATE
        ),
        lower_bound=lower,
        upper_bound=upper,
        span_kinds=kinds,
        issue=None if resolved else GeometryTraceIssue.AMBIGUOUS_PRIMITIVE,
    )


def _macro_exact_walk(trace: MacroActionTrace) -> tuple[QuantizedPoint, ...] | None:
    points: list[QuantizedPoint] = []
    for span in trace.spans:
        if _derived_span_kind(span) is DerivedMacroSpanKind.INDETERMINATE:
            return None
        # An ARC control point is not generally a point on the rendered curve,
        # and turn signs of its control polyline do not certify convexity of the
        # curved carrier.  Arc spans remain usable by the independent straight-
        # span counter, but convexity fails closed until the IR carries explicit
        # curve/sweep semantics with an interval proof.
        if span.primitive is MacroActionPrimitive.ARC:
            return None
        for point in span.points[:-1]:
            exact = point.exact_point()
            if exact is None:
                return None
            if not points or points[-1] != exact:
                points.append(exact)
    if not points:
        return None
    points.append(points[0])
    return tuple(points)


_BOUNDARY_ISSUE = {
    BoundaryPolygonIssue.OPEN_BOUNDARY: GeometryTraceIssue.OPEN_MACRO_TRACE,
    BoundaryPolygonIssue.DEGENERATE_BOUNDARY: GeometryTraceIssue.DEGENERATE_MACRO_TRACE,
    BoundaryPolygonIssue.SELF_INTERSECTING_BOUNDARY: (
        GeometryTraceIssue.SELF_INTERSECTING_MACRO_TRACE
    ),
    BoundaryPolygonIssue.CAPACITY_LIMIT: GeometryTraceIssue.CAPACITY_LIMIT,
}


def derive_macro_convexity(
    evidence: HierarchicalActionGeometryEvidence,
) -> MacroConvexityDerivation:
    """Derive convexity from the ordered simplified macro trace, never a hull."""

    if type(evidence) is not HierarchicalActionGeometryEvidence:
        raise TypeError("convexity derivation needs hierarchical evidence")
    trace = evidence.macro_action_trace
    if trace.resolution is not TraceResolution.COMPLETE:
        return _make_convexity_derivation(
            macro_evidence_digest=trace.trace_digest,
            status=(
                GeometryDerivationStatus.ERROR
                if trace.resolution is TraceResolution.ERROR
                else GeometryDerivationStatus.INDETERMINATE
            ),
            convexity_kind=None,
            polygon=None,
            issue=trace.issue,
        )
    span_error = next(
        (item.issue for item in trace.spans if item.resolution is TraceResolution.ERROR),
        None,
    )
    if span_error is not None:
        return _make_convexity_derivation(
            macro_evidence_digest=trace.trace_digest,
            status=GeometryDerivationStatus.ERROR,
            convexity_kind=None,
            polygon=None,
            issue=span_error,
        )
    walk = _macro_exact_walk(trace)
    if walk is None:
        return _make_convexity_derivation(
            macro_evidence_digest=trace.trace_digest,
            status=GeometryDerivationStatus.INDETERMINATE,
            convexity_kind=None,
            polygon=None,
            issue=GeometryTraceIssue.RESOLUTION_LIMIT,
        )
    try:
        polygon = CanonicalBoundaryPolygon.from_closed_vertex_walk(walk)
    except BoundaryPolygonError as exc:
        return _make_convexity_derivation(
            macro_evidence_digest=trace.trace_digest,
            status=GeometryDerivationStatus.INDETERMINATE,
            convexity_kind=None,
            polygon=None,
            issue=_BOUNDARY_ISSUE[exc.issue],
        )
    return _make_convexity_derivation(
        macro_evidence_digest=trace.trace_digest,
        status=GeometryDerivationStatus.RESOLVED,
        convexity_kind=polygon.convexity_kind,
        polygon=polygon,
        issue=None,
    )


def evaluate_macro_convexity(
    evidence: HierarchicalActionGeometryEvidence, target: ConvexityKind
) -> EngineeringFeatureDisposition:
    if type(target) is not ConvexityKind:
        raise TypeError("convexity target must be exact")
    derived = derive_macro_convexity(evidence)
    if derived.status is GeometryDerivationStatus.ERROR:
        return EngineeringFeatureDisposition.ERROR
    if derived.status is not GeometryDerivationStatus.RESOLVED:
        return EngineeringFeatureDisposition.INDETERMINATE
    return (
        EngineeringFeatureDisposition.MATCH
        if derived.convexity_kind is target
        else EngineeringFeatureDisposition.NONMATCH
    )


def evaluate_macro_straight_span_count(
    evidence: HierarchicalActionGeometryEvidence, target: ClosedCount
) -> EngineeringFeatureDisposition:
    if type(target) is not ClosedCount:
        raise TypeError("straight-count target must be exact")
    derived = derive_macro_straight_span_count(evidence)
    wanted = _COUNT_VALUE[target]
    if derived.status is GeometryDerivationStatus.ERROR:
        return EngineeringFeatureDisposition.ERROR
    if derived.status is GeometryDerivationStatus.RESOLVED:
        return (
            EngineeringFeatureDisposition.MATCH
            if derived.lower_bound == wanted
            else EngineeringFeatureDisposition.NONMATCH
        )
    if evidence.macro_action_trace.resolution is not TraceResolution.COMPLETE:
        return EngineeringFeatureDisposition.INDETERMINATE
    if wanted < derived.lower_bound or wanted > derived.upper_bound:
        return EngineeringFeatureDisposition.NONMATCH
    return EngineeringFeatureDisposition.INDETERMINATE


def evaluate_positive_macro_conjunction(
    evidence: HierarchicalActionGeometryEvidence,
    *,
    convexity: ConvexityKind,
    straight_span_count: ClosedCount,
) -> EngineeringFeatureDisposition:
    """Evaluate ``convexity AND count`` without inventing a negative-side rule."""

    components = (
        evaluate_macro_convexity(evidence, convexity),
        evaluate_macro_straight_span_count(evidence, straight_span_count),
    )
    if EngineeringFeatureDisposition.ERROR in components:
        return EngineeringFeatureDisposition.ERROR
    if EngineeringFeatureDisposition.NONMATCH in components:
        return EngineeringFeatureDisposition.NONMATCH
    if all(item is EngineeringFeatureDisposition.MATCH for item in components):
        return EngineeringFeatureDisposition.MATCH
    return EngineeringFeatureDisposition.INDETERMINATE


def _replay_content(value: "HierarchicalActionGeometryReplay") -> dict[str, object]:
    return {
        "schema": GEOMETRY_REPLAY_SCHEMA,
        "evidence": value.evidence.to_data(),
        "convexity": value.convexity.to_data(),
        "straight_span_count": value.straight_span_count.to_data(),
        "model_call_count": 0,
        "python_is_canonical_authority": True,
        "lean_present": False,
    }


@dataclass(frozen=True, slots=True)
class HierarchicalActionGeometryReplay:
    evidence: HierarchicalActionGeometryEvidence
    convexity: MacroConvexityDerivation
    straight_span_count: MacroStraightSpanCountDerivation
    record_digest: str

    def __post_init__(self) -> None:
        if type(self.evidence) is not HierarchicalActionGeometryEvidence:
            raise TypeError("geometry replay needs exact evidence")
        if self.convexity != derive_macro_convexity(self.evidence):
            raise HierarchicalActionGeometryError("replayed convexity differs")
        if self.straight_span_count != derive_macro_straight_span_count(self.evidence):
            raise HierarchicalActionGeometryError("replayed straight count differs")
        _digest(self.record_digest, "geometry replay digest")
        if self.record_digest != canonical_digest(_replay_content(self)):
            raise HierarchicalActionGeometryError("geometry replay commitment differs")

    @classmethod
    def create(
        cls, evidence: HierarchicalActionGeometryEvidence
    ) -> "HierarchicalActionGeometryReplay":
        values = {
            "evidence": evidence,
            "convexity": derive_macro_convexity(evidence),
            "straight_span_count": derive_macro_straight_span_count(evidence),
        }
        provisional = object.__new__(cls)
        for name, item in values.items():
            object.__setattr__(provisional, name, item)
        return cls(**values, record_digest=canonical_digest(_replay_content(provisional)))

    @property
    def replay_address(self) -> str:
        return "sha256:" + self.record_digest

    def to_data(self) -> dict[str, object]:
        return {**_replay_content(self), "record_digest": self.record_digest}

    @classmethod
    def from_data(cls, value: object) -> "HierarchicalActionGeometryReplay":
        raw = _fields(
            value,
            {
                "schema",
                "evidence",
                "convexity",
                "straight_span_count",
                "model_call_count",
                "python_is_canonical_authority",
                "lean_present",
                "record_digest",
            },
            "hierarchical geometry replay",
        )
        if (
            raw["schema"] != GEOMETRY_REPLAY_SCHEMA
            or raw["model_call_count"] != 0
            or raw["python_is_canonical_authority"] is not True
            or raw["lean_present"] is not False
        ):
            raise HierarchicalActionGeometryError("geometry replay policy differs")
        result = cls(
            HierarchicalActionGeometryEvidence.from_data(raw["evidence"]),
            MacroConvexityDerivation.from_data(raw["convexity"]),
            MacroStraightSpanCountDerivation.from_data(raw["straight_span_count"]),
            raw["record_digest"],
        )
        if result.to_data() != dict(raw):
            raise HierarchicalActionGeometryError("geometry replay is not canonical")
        return result


def cold_replay_hierarchical_action_geometry(
    replay: HierarchicalActionGeometryReplay, *, expected_replay_address: str
) -> HierarchicalActionGeometryReplay:
    """Recompute both macro observables from archived typed evidence; call no model."""

    if type(replay) is not HierarchicalActionGeometryReplay:
        raise TypeError("cold replay needs exact HierarchicalActionGeometryReplay")
    if expected_replay_address != replay.replay_address:
        raise HierarchicalActionGeometryError("geometry replay address differs")
    restored = HierarchicalActionGeometryReplay.from_data(replay.to_data())
    if restored != replay:
        raise HierarchicalActionGeometryError("cold geometry replay differs")
    return restored


__all__ = (
    "DerivedMacroSpanKind",
    "GeometryDerivationStatus",
    "GeometryEvidenceProvenance",
    "GeometryTraceIssue",
    "Grid16Interval",
    "HIERARCHICAL_ACTION_GEOMETRY_PROTOCOL_ID",
    "HierarchicalActionGeometryError",
    "HierarchicalActionGeometryEvidence",
    "HierarchicalActionGeometryReplay",
    "MacroActionPrimitive",
    "MacroActionSpan",
    "MacroActionTrace",
    "MacroConvexityDerivation",
    "MacroStraightSpanCountDerivation",
    "MicroTextureEvidence",
    "MicroTexturePrimitive",
    "MicroTexturePrimitiveKind",
    "TraceResolution",
    "UncertainGrid16Point",
    "cold_replay_hierarchical_action_geometry",
    "derive_macro_convexity",
    "derive_macro_straight_span_count",
    "evaluate_macro_convexity",
    "evaluate_macro_straight_span_count",
    "evaluate_positive_macro_conjunction",
)
