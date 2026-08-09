"""Python-native typed visual ontology for whole-panel Bongard reasoning.

This module separates three things that the first panel-soft drill conflated:

* prose is narration for a human/model observer;
* a :class:`PanelFeatureSpec` is closed, context-free semantic identity; and
* panel evidence is a local, typed witness or an exhaustive-search artifact.

The module is intentionally not wired into the live v2 proposer or runner.
It contains no Lean import or proof field.  Python values and canonical JSON
are the only executable authority; a separately generated Lean rendering can
neither change a digest nor a decision.

The private frozen verification records below are protocol misuse guards, not
a sandbox against hostile in-process Python: code that can import private
module state can forge ordinary Python objects.  A production campaign must
exclude arbitrary proposer code from the verifier process and source every
expected pin from an externally authenticated journal/signature boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from functools import cache
from itertools import combinations, permutations
import re
from types import MappingProxyType
from typing import Any, Mapping, Sequence, TypeAlias

from bongard.canonical import canonical_digest
from bongard.evidence import Disposition
from bongard.python_predicate_authority import PYTHON_PREDICATE_AUTHORITY_ID


FEATURE_SPEC_SCHEMA = "gkm.bongard-panel-feature-spec.v1"
FEATURE_NARRATION_SCHEMA = "gkm.bongard-panel-feature-narration.v1"
NATIVE_PROPOSAL_PROVENANCE_SCHEMA = (
    "gkm.bongard-native-feature-proposal-provenance.v1"
)
NATIVE_FEATURE_PROPOSAL_SCHEMA = "gkm.bongard-native-feature-proposal.v1"
QUANTIZED_POINT_SCHEMA = "gkm.bongard-grid16-point.v1"
QUANTIZED_REGION_SCHEMA = "gkm.bongard-grid16-region.v1"
PANEL_OWNER_SCHEMA = "gkm.bongard-panel-local-owner.v1"
OWNER_INVENTORY_SCHEMA = "gkm.bongard-owner-inventory.v1"
SUBJECT_BINDING_SCHEMA = "gkm.bongard-subject-binding.v1"
SEARCH_DOMAIN_SCHEMA = "gkm.bongard-feature-search-domain.v1"
FEATURE_WITNESS_SCHEMA = "gkm.bongard-panel-feature-witness.v1"
OWNER_REJECTION_SCHEMA = "gkm.bongard-owner-rejection.v1"
ABSENCE_CERTIFICATE_SCHEMA = "gkm.bongard-feature-absence-certificate.v1"
EMPTY_DOMAIN_CERTIFICATE_SCHEMA = "gkm.bongard-empty-eligible-domain.v1"
RAW_MEASUREMENT_SCHEMA = "gkm.bongard-raw-feature-measurement.v1"
LANGUAGE_GAP_SCHEMA = "gkm.bongard-feature-language-gap.v1"
FEATURE_DOMAIN_SCHEMA = "gkm.bongard-feature-calibration-domain.v1"
PRESENCE_GRANT_SCHEMA = "gkm.bongard-feature-presence-calibration-grant.v1"
ABSENCE_GRANT_SCHEMA = "gkm.bongard-feature-absence-calibration-grant.v1"
CALIBRATION_AUTHORITY_SCHEMA = "gkm.bongard-feature-calibration-authority.v1"
CALIBRATION_ASSESSMENT_SCHEMA = "gkm.bongard-feature-calibration-assessment.v1"

FEATURE_CATALOG_SCHEMA = "gkm.bongard-panel-feature-catalog.v5"
FEATURE_CATALOG_ID = "bongard.panel-feature-catalog/typed-visual-v5"
OWNER_ENUMERATION_PROTOCOL_ID = (
    "bongard.panel-owner-enumeration/candidate-independent-grid16-v1"
)
SUBJECT_PROJECTION_RULE_ID = (
    "bongard.panel-subject-projection/static-family-scope-v1"
)
COMPONENT_MEMBERSHIP_RULE_ID = (
    "bongard.panel-component-membership/root-coherent-figure-trace-loop-v1"
)
SEGMENT_MEMBERSHIP_RULE_ID = (
    "bongard.panel-segment-membership/transitive-descendant-closure-v1"
)
STRAIGHT_SEGMENT_CLASSIFICATION_RULE_ID = (
    "bongard.panel-straight-segment-membership/"
    "explicit-exhaustive-grid16-line-evidence-v1"
)
BOUNDARY_CONVEXITY_DERIVATION_RULE_ID = (
    "bongard.panel-boundary-convexity/"
    "canonical-simple-grid16-polygon-cross-products-v1"
)
GRID16_BIN_COUNT = 16
MAX_BOUNDARY_VERTEX_COUNT = 64

_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_OWNER_ID = re.compile(r"owner_[0-9]{4}\Z")
_CODE = re.compile(r"[a-z][a-z0-9_.:-]{0,127}\Z")
_TEXT = re.compile(r"[^\x00-\x1f\x7f]{1,500}\Z")


class PanelSoftOntologyError(ValueError):
    """A typed feature, witness, certificate, or authority is invalid."""


class FeatureFamily(str, Enum):
    COMPONENT_COUNT = "component_count"
    EXACT_SEGMENT_COUNT = "exact_segment_count"
    STRAIGHT_SEGMENT_COUNT = "straight_segment_count"
    CONVEXITY = "convexity"
    MARKER_PATTERN = "marker_pattern"
    GESTALT_RESEMBLANCE = "gestalt_resemblance"
    SEGMENT_ORIENTATION = "segment_orientation"
    CORNER_ANGLE = "corner_angle"
    TURN_PROFILE = "turn_profile"
    OPEN_TRACE = "open_trace"
    CLOSED_LOOP = "closed_loop"
    POINT_CONTACT = "point_contact"
    VISIBLE_GAP = "visible_gap"
    ENCLOSURE = "enclosure"
    SYMMETRY = "symmetry"
    SHARED_BOUNDARY_ADJACENCY = "shared_boundary_adjacency"
    ASPECT_RATIO = "aspect_ratio"
    TEXTURE_COMPOSITION = "texture_composition"


class SubjectScope(str, Enum):
    WHOLE_PANEL = "whole_panel"
    ONE_COHERENT_FIGURE = "one_coherent_figure"
    ONE_TRACE = "one_trace"
    FIGURE_PAIR = "figure_pair"


class ReferenceFrame(str, Enum):
    NONE = "none"
    CANVAS_AXES = "canvas_axes"
    SUBJECT_MAJOR_AXIS = "subject_major_axis"
    FIGURE_PAIR_AXIS = "figure_pair_axis"
    LOCAL_TANGENT = "local_tangent"


class ClosedCount(str, Enum):
    ONE = "one"
    TWO = "two"
    THREE = "three"
    FOUR = "four"
    FIVE = "five"
    SIX = "six"
    SEVEN = "seven"
    EIGHT = "eight"
    NINE = "nine"
    TEN = "ten"
    ELEVEN = "eleven"
    TWELVE = "twelve"


_CLOSED_COUNT_TO_INT: Mapping[ClosedCount, int] = MappingProxyType(
    {
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
)


class ClosedAggregation(str, Enum):
    ONE_WITNESSED = "one_witnessed"
    ALL_ELIGIBLE = "all_eligible"
    AT_LEAST_TWO = "at_least_two"


class MarkerPrimitive(str, Enum):
    DOT = "dot"
    SHORT_TICK = "short_tick"
    SMALL_LOOP = "small_loop"
    CROSS = "cross"
    WEDGE = "wedge"


class MarkerArrangement(str, Enum):
    LINEAR = "linear"
    CLUSTERED = "clustered"
    AROUND_BOUNDARY = "around_boundary"
    INSIDE_REGION = "inside_region"


class GestaltKind(str, Enum):
    BIRD_LIKE = "bird_like"
    ANIMAL_LIKE = "animal_like"
    FACE_LIKE = "face_like"
    ARROW_LIKE = "arrow_like"
    LETTER_LIKE = "letter_like"
    TOOL_LIKE = "tool_like"


class ConvexityKind(str, Enum):
    CONVEX_CLOSED_BOUNDARY = "convex_closed_boundary"
    CONCAVE_CLOSED_BOUNDARY = "concave_closed_boundary"


class OrientationClass(str, Enum):
    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"
    OBLIQUE_ASCENDING = "oblique_ascending"
    OBLIQUE_DESCENDING = "oblique_descending"
    RADIAL = "radial"


class CornerAngleClass(str, Enum):
    ACUTE = "acute"
    RIGHT = "right"
    OBTUSE = "obtuse"
    REFLEX = "reflex"


class TurnProfileClass(str, Enum):
    CONSISTENT_CLOCKWISE = "consistent_clockwise"
    CONSISTENT_COUNTERCLOCKWISE = "consistent_counterclockwise"
    ALTERNATING = "alternating"
    SINGLE_REVERSAL = "single_reversal"


class OpenTraceKind(str, Enum):
    SIMPLE_UNBRANCHED = "simple_unbranched"
    BRANCHED = "branched"


class ClosedLoopKind(str, Enum):
    SIMPLE = "simple"
    SELF_INTERSECTING = "self_intersecting"
    NESTED = "nested"


class PointContactKind(str, Enum):
    TANGENTIAL = "tangential"
    CORNER_TO_EDGE = "corner_to_edge"
    ENDPOINT_TO_EDGE = "endpoint_to_edge"


class VisibleGapKind(str, Enum):
    BETWEEN_CONTOURS = "between_contours"
    BETWEEN_ENDPOINTS = "between_endpoints"
    EXTERIOR_SEPARATION = "exterior_separation"


class EnclosureKind(str, Enum):
    FULLY_INSIDE = "fully_inside"
    NESTED_LOOP = "nested_loop"


class SymmetryKind(str, Enum):
    REFLECTIONAL = "reflectional"
    HALF_TURN = "half_turn"
    RADIAL = "radial"


class SharedBoundaryKind(str, Enum):
    STRAIGHT_SEGMENT = "straight_segment"
    CURVED_ARC = "curved_arc"
    MIXED_TRACE = "mixed_trace"


class AspectRatioClass(str, Enum):
    SQUARE_LIKE = "square_like"
    WIDE = "wide"
    TALL = "tall"
    SLENDER_WIDE = "slender_wide"
    SLENDER_TALL = "slender_tall"


class TextureCompositionClass(str, Enum):
    UNIFORM_SOLID = "uniform_solid"
    UNIFORM_HATCHED = "uniform_hatched"
    UNIFORM_DOTTED = "uniform_dotted"
    MIXED_REGIONS = "mixed_regions"


class OwnerKind(str, Enum):
    FIGURE = "figure"
    TRACE = "trace"
    LOOP = "loop"
    SEGMENT = "segment"
    MARKER = "marker"


class SubjectBindingKind(str, Enum):
    PANEL = "panel"
    UNARY = "unary"
    UNORDERED_PAIR = "unordered_pair"
    ORDERED_CONTAINER_CONTAINED = "ordered_container_contained"


class EnumerationResolution(str, Enum):
    GRID16_FULL_PANEL = "grid16_full_panel"


class _OneEnumParameter:
    """Shared implementation for closed, single-enum parameter records."""

    _field_name: str
    _enum_type: type[Enum]

    def _one_enum_to_data(self) -> dict[str, str]:
        value = getattr(self, self._field_name)
        return {self._field_name: value.value}

    @classmethod
    def _one_enum_from_data(cls, value: object):
        raw = _fields(value, {cls._field_name}, cls.__name__)
        try:
            enum_value = cls._enum_type(raw[cls._field_name])
        except (TypeError, ValueError) as exc:
            raise PanelSoftOntologyError(f"{cls.__name__} value differs") from exc
        result = cls(enum_value)
        _require_canonical(result, raw, cls.__name__)
        return result


@dataclass(frozen=True, order=True, slots=True)
class ComponentCountParameters(_OneEnumParameter):
    count: ClosedCount
    _field_name = "count"
    _enum_type = ClosedCount

    def __post_init__(self) -> None:
        _enum_instance(self.count, ClosedCount, "component count")

    to_data = _OneEnumParameter._one_enum_to_data
    from_data = classmethod(_OneEnumParameter._one_enum_from_data.__func__)


@dataclass(frozen=True, order=True, slots=True)
class ExactSegmentCountParameters(_OneEnumParameter):
    count: ClosedCount
    _field_name = "count"
    _enum_type = ClosedCount

    def __post_init__(self) -> None:
        _enum_instance(self.count, ClosedCount, "segment count")

    to_data = _OneEnumParameter._one_enum_to_data
    from_data = classmethod(_OneEnumParameter._one_enum_from_data.__func__)


@dataclass(frozen=True, order=True, slots=True)
class StraightSegmentCountParameters(_OneEnumParameter):
    """Exact count of visibly straight segments, not generic segment owners."""

    count: ClosedCount
    _field_name = "count"
    _enum_type = ClosedCount

    def __post_init__(self) -> None:
        _enum_instance(self.count, ClosedCount, "straight-segment count")

    to_data = _OneEnumParameter._one_enum_to_data
    from_data = classmethod(_OneEnumParameter._one_enum_from_data.__func__)


@dataclass(frozen=True, order=True, slots=True)
class ConvexityParameters(_OneEnumParameter):
    """Python-derived class of one complete simple outer-boundary polygon."""

    kind: ConvexityKind
    _field_name = "kind"
    _enum_type = ConvexityKind

    def __post_init__(self) -> None:
        _enum_instance(self.kind, ConvexityKind, "convexity kind")

    to_data = _OneEnumParameter._one_enum_to_data
    from_data = classmethod(_OneEnumParameter._one_enum_from_data.__func__)


@dataclass(frozen=True, order=True, slots=True)
class MarkerPatternParameters:
    primitive: MarkerPrimitive
    repetition: ClosedCount
    arrangement: MarkerArrangement

    def __post_init__(self) -> None:
        _enum_instance(self.primitive, MarkerPrimitive, "marker primitive")
        _enum_instance(self.repetition, ClosedCount, "marker repetition")
        _enum_instance(self.arrangement, MarkerArrangement, "marker arrangement")

    def to_data(self) -> dict[str, str]:
        return {
            "primitive": self.primitive.value,
            "repetition": self.repetition.value,
            "arrangement": self.arrangement.value,
        }

    @classmethod
    def from_data(cls, value: object) -> "MarkerPatternParameters":
        raw = _fields(
            value, {"primitive", "repetition", "arrangement"}, "marker parameters"
        )
        try:
            result = cls(
                MarkerPrimitive(raw["primitive"]),
                ClosedCount(raw["repetition"]),
                MarkerArrangement(raw["arrangement"]),
            )
        except (TypeError, ValueError) as exc:
            raise PanelSoftOntologyError("marker parameter value differs") from exc
        _require_canonical(result, raw, "marker parameters")
        return result


@dataclass(frozen=True, order=True, slots=True)
class GestaltResemblanceParameters(_OneEnumParameter):
    kind: GestaltKind
    _field_name = "kind"
    _enum_type = GestaltKind

    def __post_init__(self) -> None:
        _enum_instance(self.kind, GestaltKind, "gestalt kind")

    to_data = _OneEnumParameter._one_enum_to_data
    from_data = classmethod(_OneEnumParameter._one_enum_from_data.__func__)


@dataclass(frozen=True, order=True, slots=True)
class SegmentOrientationParameters:
    orientation: OrientationClass
    aggregation: ClosedAggregation

    def __post_init__(self) -> None:
        _enum_instance(self.orientation, OrientationClass, "orientation")
        _enum_instance(self.aggregation, ClosedAggregation, "aggregation")

    def to_data(self) -> dict[str, str]:
        return {
            "orientation": self.orientation.value,
            "aggregation": self.aggregation.value,
        }

    @classmethod
    def from_data(cls, value: object) -> "SegmentOrientationParameters":
        raw = _fields(value, {"orientation", "aggregation"}, "orientation parameters")
        try:
            result = cls(
                OrientationClass(raw["orientation"]),
                ClosedAggregation(raw["aggregation"]),
            )
        except (TypeError, ValueError) as exc:
            raise PanelSoftOntologyError("orientation parameter value differs") from exc
        _require_canonical(result, raw, "orientation parameters")
        return result


@dataclass(frozen=True, order=True, slots=True)
class CornerAngleParameters:
    angle_class: CornerAngleClass
    aggregation: ClosedAggregation

    def __post_init__(self) -> None:
        _enum_instance(self.angle_class, CornerAngleClass, "corner angle")
        _enum_instance(self.aggregation, ClosedAggregation, "aggregation")

    def to_data(self) -> dict[str, str]:
        return {
            "angle_class": self.angle_class.value,
            "aggregation": self.aggregation.value,
        }

    @classmethod
    def from_data(cls, value: object) -> "CornerAngleParameters":
        raw = _fields(value, {"angle_class", "aggregation"}, "corner parameters")
        try:
            result = cls(
                CornerAngleClass(raw["angle_class"]),
                ClosedAggregation(raw["aggregation"]),
            )
        except (TypeError, ValueError) as exc:
            raise PanelSoftOntologyError("corner parameter value differs") from exc
        _require_canonical(result, raw, "corner parameters")
        return result


@dataclass(frozen=True, order=True, slots=True)
class TurnProfileParameters(_OneEnumParameter):
    profile: TurnProfileClass
    _field_name = "profile"
    _enum_type = TurnProfileClass

    def __post_init__(self) -> None:
        _enum_instance(self.profile, TurnProfileClass, "turn profile")

    to_data = _OneEnumParameter._one_enum_to_data
    from_data = classmethod(_OneEnumParameter._one_enum_from_data.__func__)


@dataclass(frozen=True, order=True, slots=True)
class OpenTraceParameters(_OneEnumParameter):
    kind: OpenTraceKind
    _field_name = "kind"
    _enum_type = OpenTraceKind

    def __post_init__(self) -> None:
        _enum_instance(self.kind, OpenTraceKind, "open trace kind")

    to_data = _OneEnumParameter._one_enum_to_data
    from_data = classmethod(_OneEnumParameter._one_enum_from_data.__func__)


@dataclass(frozen=True, order=True, slots=True)
class ClosedLoopParameters(_OneEnumParameter):
    kind: ClosedLoopKind
    _field_name = "kind"
    _enum_type = ClosedLoopKind

    def __post_init__(self) -> None:
        _enum_instance(self.kind, ClosedLoopKind, "closed loop kind")

    to_data = _OneEnumParameter._one_enum_to_data
    from_data = classmethod(_OneEnumParameter._one_enum_from_data.__func__)


@dataclass(frozen=True, order=True, slots=True)
class PointContactParameters(_OneEnumParameter):
    kind: PointContactKind
    _field_name = "kind"
    _enum_type = PointContactKind

    def __post_init__(self) -> None:
        _enum_instance(self.kind, PointContactKind, "point contact kind")

    to_data = _OneEnumParameter._one_enum_to_data
    from_data = classmethod(_OneEnumParameter._one_enum_from_data.__func__)


@dataclass(frozen=True, order=True, slots=True)
class VisibleGapParameters(_OneEnumParameter):
    kind: VisibleGapKind
    _field_name = "kind"
    _enum_type = VisibleGapKind

    def __post_init__(self) -> None:
        _enum_instance(self.kind, VisibleGapKind, "visible gap kind")

    to_data = _OneEnumParameter._one_enum_to_data
    from_data = classmethod(_OneEnumParameter._one_enum_from_data.__func__)


@dataclass(frozen=True, order=True, slots=True)
class EnclosureParameters(_OneEnumParameter):
    kind: EnclosureKind
    _field_name = "kind"
    _enum_type = EnclosureKind

    def __post_init__(self) -> None:
        _enum_instance(self.kind, EnclosureKind, "enclosure kind")

    to_data = _OneEnumParameter._one_enum_to_data
    from_data = classmethod(_OneEnumParameter._one_enum_from_data.__func__)


@dataclass(frozen=True, order=True, slots=True)
class SymmetryParameters(_OneEnumParameter):
    kind: SymmetryKind
    _field_name = "kind"
    _enum_type = SymmetryKind

    def __post_init__(self) -> None:
        _enum_instance(self.kind, SymmetryKind, "symmetry kind")

    to_data = _OneEnumParameter._one_enum_to_data
    from_data = classmethod(_OneEnumParameter._one_enum_from_data.__func__)


@dataclass(frozen=True, order=True, slots=True)
class SharedBoundaryAdjacencyParameters(_OneEnumParameter):
    kind: SharedBoundaryKind
    _field_name = "kind"
    _enum_type = SharedBoundaryKind

    def __post_init__(self) -> None:
        _enum_instance(self.kind, SharedBoundaryKind, "shared boundary kind")

    to_data = _OneEnumParameter._one_enum_to_data
    from_data = classmethod(_OneEnumParameter._one_enum_from_data.__func__)


@dataclass(frozen=True, order=True, slots=True)
class AspectRatioParameters(_OneEnumParameter):
    aspect_class: AspectRatioClass
    _field_name = "aspect_class"
    _enum_type = AspectRatioClass

    def __post_init__(self) -> None:
        _enum_instance(self.aspect_class, AspectRatioClass, "aspect ratio class")

    to_data = _OneEnumParameter._one_enum_to_data
    from_data = classmethod(_OneEnumParameter._one_enum_from_data.__func__)


@dataclass(frozen=True, order=True, slots=True)
class TextureCompositionParameters(_OneEnumParameter):
    composition: TextureCompositionClass
    _field_name = "composition"
    _enum_type = TextureCompositionClass

    def __post_init__(self) -> None:
        _enum_instance(self.composition, TextureCompositionClass, "texture composition")

    to_data = _OneEnumParameter._one_enum_to_data
    from_data = classmethod(_OneEnumParameter._one_enum_from_data.__func__)


FeatureParameters: TypeAlias = (
    ComponentCountParameters
    | ExactSegmentCountParameters
    | StraightSegmentCountParameters
    | ConvexityParameters
    | MarkerPatternParameters
    | GestaltResemblanceParameters
    | SegmentOrientationParameters
    | CornerAngleParameters
    | TurnProfileParameters
    | OpenTraceParameters
    | ClosedLoopParameters
    | PointContactParameters
    | VisibleGapParameters
    | EnclosureParameters
    | SymmetryParameters
    | SharedBoundaryAdjacencyParameters
    | AspectRatioParameters
    | TextureCompositionParameters
)


@dataclass(frozen=True, slots=True)
class FamilyContract:
    parameter_type: type
    allowed_scope_frames: frozenset[tuple[SubjectScope, ReferenceFrame]]
    binding_by_scope: Mapping[SubjectScope, SubjectBindingKind]
    owner_kinds_by_scope: Mapping[SubjectScope, tuple[OwnerKind, ...]]

    def __post_init__(self) -> None:
        if type(self.allowed_scope_frames) is not frozenset:
            raise TypeError("family contract pairs must be a frozenset")
        if set(self.binding_by_scope) != {
            scope for scope, _ in self.allowed_scope_frames
        }:
            raise PanelSoftOntologyError("family binding table is not total")
        if set(self.owner_kinds_by_scope) != set(self.binding_by_scope):
            raise PanelSoftOntologyError("family owner-kind table is not total")
        if any(
            len(kinds) != len(set(kinds))
            for kinds in self.owner_kinds_by_scope.values()
        ):
            raise PanelSoftOntologyError("family owner-kind rows must be unique")


def _contract(
    parameter_type: type,
    pairs: Sequence[tuple[SubjectScope, ReferenceFrame]],
    bindings: Mapping[SubjectScope, SubjectBindingKind],
    owner_kinds: Mapping[SubjectScope, tuple[OwnerKind, ...]],
) -> FamilyContract:
    return FamilyContract(
        parameter_type,
        frozenset(pairs),
        MappingProxyType(dict(bindings)),
        MappingProxyType(dict(owner_kinds)),
    )


_PANEL = SubjectBindingKind.PANEL
_UNARY = SubjectBindingKind.UNARY
_PAIR = SubjectBindingKind.UNORDERED_PAIR
_ORDERED = SubjectBindingKind.ORDERED_CONTAINER_CONTAINED
_FIGURE = (OwnerKind.FIGURE,)
_TRACE = (OwnerKind.TRACE, OwnerKind.LOOP)
_OPEN_TRACE_OWNER = (OwnerKind.TRACE,)
_CLOSED_LOOP_OWNER = (OwnerKind.LOOP,)
_RELATION_OWNERS = (OwnerKind.FIGURE,)

FAMILY_CONTRACTS: Mapping[FeatureFamily, FamilyContract] = MappingProxyType(
    {
        FeatureFamily.COMPONENT_COUNT: _contract(
            ComponentCountParameters,
            [(SubjectScope.WHOLE_PANEL, ReferenceFrame.NONE)],
            {SubjectScope.WHOLE_PANEL: _PANEL},
            {SubjectScope.WHOLE_PANEL: ()},
        ),
        FeatureFamily.EXACT_SEGMENT_COUNT: _contract(
            ExactSegmentCountParameters,
            [
                (SubjectScope.WHOLE_PANEL, ReferenceFrame.NONE),
                (SubjectScope.ONE_COHERENT_FIGURE, ReferenceFrame.NONE),
                (SubjectScope.ONE_TRACE, ReferenceFrame.NONE),
            ],
            {
                SubjectScope.WHOLE_PANEL: _PANEL,
                SubjectScope.ONE_COHERENT_FIGURE: _UNARY,
                SubjectScope.ONE_TRACE: _UNARY,
            },
            {
                SubjectScope.WHOLE_PANEL: (),
                SubjectScope.ONE_COHERENT_FIGURE: _FIGURE,
                SubjectScope.ONE_TRACE: _TRACE,
            },
        ),
        FeatureFamily.STRAIGHT_SEGMENT_COUNT: _contract(
            StraightSegmentCountParameters,
            [
                (SubjectScope.WHOLE_PANEL, ReferenceFrame.NONE),
                (SubjectScope.ONE_COHERENT_FIGURE, ReferenceFrame.NONE),
                (SubjectScope.ONE_TRACE, ReferenceFrame.NONE),
            ],
            {
                SubjectScope.WHOLE_PANEL: _PANEL,
                SubjectScope.ONE_COHERENT_FIGURE: _UNARY,
                SubjectScope.ONE_TRACE: _UNARY,
            },
            {
                SubjectScope.WHOLE_PANEL: (),
                SubjectScope.ONE_COHERENT_FIGURE: _FIGURE,
                SubjectScope.ONE_TRACE: _TRACE,
            },
        ),
        FeatureFamily.CONVEXITY: _contract(
            ConvexityParameters,
            [(SubjectScope.WHOLE_PANEL, ReferenceFrame.NONE)],
            {SubjectScope.WHOLE_PANEL: _PANEL},
            {SubjectScope.WHOLE_PANEL: ()},
        ),
        FeatureFamily.MARKER_PATTERN: _contract(
            MarkerPatternParameters,
            [(SubjectScope.ONE_COHERENT_FIGURE, ReferenceFrame.NONE)],
            {SubjectScope.ONE_COHERENT_FIGURE: _UNARY},
            {SubjectScope.ONE_COHERENT_FIGURE: _FIGURE},
        ),
        FeatureFamily.GESTALT_RESEMBLANCE: _contract(
            GestaltResemblanceParameters,
            [
                (SubjectScope.WHOLE_PANEL, ReferenceFrame.NONE),
                (SubjectScope.ONE_COHERENT_FIGURE, ReferenceFrame.NONE),
            ],
            {
                SubjectScope.WHOLE_PANEL: _PANEL,
                SubjectScope.ONE_COHERENT_FIGURE: _UNARY,
            },
            {
                SubjectScope.WHOLE_PANEL: (),
                SubjectScope.ONE_COHERENT_FIGURE: _FIGURE,
            },
        ),
        FeatureFamily.SEGMENT_ORIENTATION: _contract(
            SegmentOrientationParameters,
            [
                (SubjectScope.WHOLE_PANEL, ReferenceFrame.CANVAS_AXES),
                (SubjectScope.ONE_TRACE, ReferenceFrame.CANVAS_AXES),
                (SubjectScope.ONE_TRACE, ReferenceFrame.SUBJECT_MAJOR_AXIS),
                (SubjectScope.ONE_COHERENT_FIGURE, ReferenceFrame.CANVAS_AXES),
                (
                    SubjectScope.ONE_COHERENT_FIGURE,
                    ReferenceFrame.SUBJECT_MAJOR_AXIS,
                ),
            ],
            {
                SubjectScope.WHOLE_PANEL: _PANEL,
                SubjectScope.ONE_TRACE: _UNARY,
                SubjectScope.ONE_COHERENT_FIGURE: _UNARY,
            },
            {
                SubjectScope.WHOLE_PANEL: (),
                SubjectScope.ONE_TRACE: _TRACE,
                SubjectScope.ONE_COHERENT_FIGURE: _FIGURE,
            },
        ),
        FeatureFamily.CORNER_ANGLE: _contract(
            CornerAngleParameters,
            [
                (SubjectScope.WHOLE_PANEL, ReferenceFrame.LOCAL_TANGENT),
                (SubjectScope.ONE_TRACE, ReferenceFrame.LOCAL_TANGENT),
                (SubjectScope.ONE_COHERENT_FIGURE, ReferenceFrame.LOCAL_TANGENT),
            ],
            {
                SubjectScope.WHOLE_PANEL: _PANEL,
                SubjectScope.ONE_TRACE: _UNARY,
                SubjectScope.ONE_COHERENT_FIGURE: _UNARY,
            },
            {
                SubjectScope.WHOLE_PANEL: (),
                SubjectScope.ONE_TRACE: _TRACE,
                SubjectScope.ONE_COHERENT_FIGURE: _FIGURE,
            },
        ),
        FeatureFamily.TURN_PROFILE: _contract(
            TurnProfileParameters,
            [(SubjectScope.ONE_TRACE, ReferenceFrame.LOCAL_TANGENT)],
            {SubjectScope.ONE_TRACE: _UNARY},
            {SubjectScope.ONE_TRACE: _TRACE},
        ),
        FeatureFamily.OPEN_TRACE: _contract(
            OpenTraceParameters,
            [(SubjectScope.ONE_TRACE, ReferenceFrame.NONE)],
            {SubjectScope.ONE_TRACE: _UNARY},
            {SubjectScope.ONE_TRACE: _OPEN_TRACE_OWNER},
        ),
        FeatureFamily.CLOSED_LOOP: _contract(
            ClosedLoopParameters,
            [(SubjectScope.ONE_TRACE, ReferenceFrame.NONE)],
            {SubjectScope.ONE_TRACE: _UNARY},
            {SubjectScope.ONE_TRACE: _CLOSED_LOOP_OWNER},
        ),
        FeatureFamily.POINT_CONTACT: _contract(
            PointContactParameters,
            [(SubjectScope.FIGURE_PAIR, ReferenceFrame.NONE)],
            {SubjectScope.FIGURE_PAIR: _PAIR},
            {SubjectScope.FIGURE_PAIR: _RELATION_OWNERS},
        ),
        FeatureFamily.VISIBLE_GAP: _contract(
            VisibleGapParameters,
            [(SubjectScope.FIGURE_PAIR, ReferenceFrame.NONE)],
            {SubjectScope.FIGURE_PAIR: _PAIR},
            {SubjectScope.FIGURE_PAIR: _RELATION_OWNERS},
        ),
        FeatureFamily.ENCLOSURE: _contract(
            EnclosureParameters,
            [(SubjectScope.FIGURE_PAIR, ReferenceFrame.NONE)],
            {SubjectScope.FIGURE_PAIR: _ORDERED},
            {SubjectScope.FIGURE_PAIR: _RELATION_OWNERS},
        ),
        FeatureFamily.SYMMETRY: _contract(
            SymmetryParameters,
            [
                (SubjectScope.WHOLE_PANEL, ReferenceFrame.CANVAS_AXES),
                (
                    SubjectScope.ONE_COHERENT_FIGURE,
                    ReferenceFrame.SUBJECT_MAJOR_AXIS,
                ),
            ],
            {SubjectScope.WHOLE_PANEL: _PANEL, SubjectScope.ONE_COHERENT_FIGURE: _UNARY},
            {SubjectScope.WHOLE_PANEL: (), SubjectScope.ONE_COHERENT_FIGURE: _FIGURE},
        ),
        FeatureFamily.SHARED_BOUNDARY_ADJACENCY: _contract(
            SharedBoundaryAdjacencyParameters,
            [(SubjectScope.FIGURE_PAIR, ReferenceFrame.NONE)],
            {SubjectScope.FIGURE_PAIR: _PAIR},
            {SubjectScope.FIGURE_PAIR: _RELATION_OWNERS},
        ),
        FeatureFamily.ASPECT_RATIO: _contract(
            AspectRatioParameters,
            [
                (
                    SubjectScope.ONE_COHERENT_FIGURE,
                    ReferenceFrame.SUBJECT_MAJOR_AXIS,
                )
            ],
            {SubjectScope.ONE_COHERENT_FIGURE: _UNARY},
            {SubjectScope.ONE_COHERENT_FIGURE: _FIGURE},
        ),
        FeatureFamily.TEXTURE_COMPOSITION: _contract(
            TextureCompositionParameters,
            [
                (SubjectScope.WHOLE_PANEL, ReferenceFrame.NONE),
                (SubjectScope.ONE_COHERENT_FIGURE, ReferenceFrame.NONE),
            ],
            {
                SubjectScope.WHOLE_PANEL: _PANEL,
                SubjectScope.ONE_COHERENT_FIGURE: _UNARY,
            },
            {
                SubjectScope.WHOLE_PANEL: (),
                SubjectScope.ONE_COHERENT_FIGURE: _FIGURE,
            },
        ),
    }
)

if set(FAMILY_CONTRACTS) != set(FeatureFamily):  # pragma: no cover - import guard.
    raise RuntimeError("feature-family contract table is incomplete")


def _closed_values(enum_type: type[Enum]) -> list[object]:
    if enum_type is ClosedCount:
        return [
            {"code": item.value, "exact_count": _CLOSED_COUNT_TO_INT[item]}
            for item in sorted(ClosedCount, key=lambda value: _CLOSED_COUNT_TO_INT[value])
        ]
    return sorted(item.value for item in enum_type)


def _parameter_semantic_schema(family: FeatureFamily) -> dict[str, object]:
    """Closed semantic fields, deliberately independent of Python class names."""

    one_field: dict[FeatureFamily, tuple[str, type[Enum]]] = {
        FeatureFamily.COMPONENT_COUNT: ("count", ClosedCount),
        FeatureFamily.EXACT_SEGMENT_COUNT: ("count", ClosedCount),
        FeatureFamily.STRAIGHT_SEGMENT_COUNT: ("count", ClosedCount),
        FeatureFamily.CONVEXITY: ("kind", ConvexityKind),
        FeatureFamily.GESTALT_RESEMBLANCE: ("kind", GestaltKind),
        FeatureFamily.TURN_PROFILE: ("profile", TurnProfileClass),
        FeatureFamily.OPEN_TRACE: ("kind", OpenTraceKind),
        FeatureFamily.CLOSED_LOOP: ("kind", ClosedLoopKind),
        FeatureFamily.POINT_CONTACT: ("kind", PointContactKind),
        FeatureFamily.VISIBLE_GAP: ("kind", VisibleGapKind),
        FeatureFamily.ENCLOSURE: ("kind", EnclosureKind),
        FeatureFamily.SYMMETRY: ("kind", SymmetryKind),
        FeatureFamily.SHARED_BOUNDARY_ADJACENCY: ("kind", SharedBoundaryKind),
        FeatureFamily.ASPECT_RATIO: ("aspect_class", AspectRatioClass),
        FeatureFamily.TEXTURE_COMPOSITION: (
            "composition",
            TextureCompositionClass,
        ),
    }
    if family in one_field:
        name, enum_type = one_field[family]
        return {"fields": [{"name": name, "closed_values": _closed_values(enum_type)}]}
    if family is FeatureFamily.MARKER_PATTERN:
        return {
            "fields": [
                {"name": "primitive", "closed_values": _closed_values(MarkerPrimitive)},
                {"name": "repetition", "closed_values": _closed_values(ClosedCount)},
                {"name": "arrangement", "closed_values": _closed_values(MarkerArrangement)},
            ]
        }
    if family is FeatureFamily.SEGMENT_ORIENTATION:
        return {
            "fields": [
                {"name": "orientation", "closed_values": _closed_values(OrientationClass)},
                {"name": "aggregation", "closed_values": _closed_values(ClosedAggregation)},
            ]
        }
    return {
        "fields": [
            {"name": "angle_class", "closed_values": _closed_values(CornerAngleClass)},
            {"name": "aggregation", "closed_values": _closed_values(ClosedAggregation)},
        ]
    }


def feature_catalog_data() -> dict[str, object]:
    """Return semantic catalog identity without source or implementation state."""

    rows = []
    for family in sorted(FeatureFamily, key=lambda item: item.value):
        contract = FAMILY_CONTRACTS[family]
        rows.append(
            {
                "family": family.value,
                "parameter_schema": _parameter_semantic_schema(family),
                "allowed_scope_frames": [
                    {"scope": scope.value, "reference_frame": frame.value}
                    for scope, frame in sorted(
                        contract.allowed_scope_frames,
                        key=lambda item: (item[0].value, item[1].value),
                    )
                ],
                "binding_by_scope": [
                    {
                        "scope": scope.value,
                        "binding_kind": contract.binding_by_scope[scope].value,
                        "owner_kinds": [
                            item.value
                            for item in sorted(
                                contract.owner_kinds_by_scope[scope],
                                key=lambda value: value.value,
                            )
                        ],
                    }
                    for scope in sorted(contract.binding_by_scope, key=lambda item: item.value)
                ],
            }
        )
    return {
        "schema": FEATURE_CATALOG_SCHEMA,
        "catalog_id": FEATURE_CATALOG_ID,
        "grid16_bin_count": GRID16_BIN_COUNT,
        "subject_projection_rule_id": SUBJECT_PROJECTION_RULE_ID,
        "count_membership_rules": {
            FeatureFamily.COMPONENT_COUNT.value: COMPONENT_MEMBERSHIP_RULE_ID,
            FeatureFamily.EXACT_SEGMENT_COUNT.value: SEGMENT_MEMBERSHIP_RULE_ID,
            FeatureFamily.STRAIGHT_SEGMENT_COUNT.value: (
                STRAIGHT_SEGMENT_CLASSIFICATION_RULE_ID
            ),
        },
        "geometry_derivation_rules": {
            FeatureFamily.CONVEXITY.value: BOUNDARY_CONVEXITY_DERIVATION_RULE_ID,
        },
        "families": rows,
        "sibling_registry": sorted(
            [
            {
                "relation_id": "open-trace-vs-closed-loop-v1",
                "left_family": FeatureFamily.OPEN_TRACE.value,
                "right_family": FeatureFamily.CLOSED_LOOP.value,
                "parameter_rule_id": "any_registered_kind_v1",
                "mutually_exclusive": True,
                "exhaustive": False,
                "same_subject_only": True,
                "direct_conflict_enabled": False,
            },
            {
                "relation_id": "point-contact-and-visible-gap-v1",
                "left_family": FeatureFamily.POINT_CONTACT.value,
                "right_family": FeatureFamily.VISIBLE_GAP.value,
                "parameter_rule_id": "coobservable_distinct_loci_v1",
                "mutually_exclusive": False,
                "exhaustive": False,
                "same_subject_only": True,
                "direct_conflict_enabled": False,
            },
            {
                "relation_id": "convex-vs-concave-closed-boundary-v1",
                "left_family": FeatureFamily.CONVEXITY.value,
                "right_family": "same_as_left",
                "parameter_rule_id": "distinct_convexity_kind_v1",
                "mutually_exclusive": True,
                "exhaustive": True,
                "same_subject_only": True,
                "direct_conflict_enabled": True,
            },
            {
                "relation_id": "distinct-exact-counts-v1",
                "left_family": (
                    "component_count|exact_segment_count|marker_pattern|"
                    "straight_segment_count"
                ),
                "right_family": "same_as_left",
                "parameter_rule_id": "distinct_count_same_marker_context_v1",
                "mutually_exclusive": True,
                "exhaustive": False,
                "same_subject_only": True,
                "direct_conflict_enabled": True,
            },
            ],
            key=lambda item: item["relation_id"],
        ),
    }


@cache
def feature_catalog_digest() -> str:
    return canonical_digest(feature_catalog_data())


def _parse_parameters(family: FeatureFamily, value: object) -> FeatureParameters:
    return FAMILY_CONTRACTS[family].parameter_type.from_data(value)


@dataclass(frozen=True, order=True, slots=True)
class PanelFeatureSpec:
    """Context-free executable feature identity.

    No prose, panel, owner, task, side, orientation, proposer, source, or proof
    metadata is admitted to this value.
    """

    family: FeatureFamily
    subject_scope: SubjectScope
    reference_frame: ReferenceFrame
    parameters: FeatureParameters

    def __post_init__(self) -> None:
        _enum_instance(self.family, FeatureFamily, "feature family")
        _enum_instance(self.subject_scope, SubjectScope, "subject scope")
        _enum_instance(self.reference_frame, ReferenceFrame, "reference frame")
        contract = FAMILY_CONTRACTS[self.family]
        if type(self.parameters) is not contract.parameter_type:
            raise PanelSoftOntologyError("feature family has the wrong parameter type")
        if (self.subject_scope, self.reference_frame) not in contract.allowed_scope_frames:
            raise PanelSoftOntologyError("feature scope/reference-frame pair is not registered")

    @property
    def spec_digest(self) -> str:
        return canonical_digest(self.to_data())

    @property
    def binding_kind(self) -> SubjectBindingKind:
        return FAMILY_CONTRACTS[self.family].binding_by_scope[self.subject_scope]

    def to_data(self) -> dict[str, object]:
        return {
            "schema": FEATURE_SPEC_SCHEMA,
            "catalog_digest": feature_catalog_digest(),
            "family": self.family.value,
            "subject_scope": self.subject_scope.value,
            "reference_frame": self.reference_frame.value,
            "parameters": self.parameters.to_data(),
        }

    @classmethod
    def from_data(cls, value: object) -> "PanelFeatureSpec":
        raw = _fields(
            value,
            {
                "schema",
                "catalog_digest",
                "family",
                "subject_scope",
                "reference_frame",
                "parameters",
            },
            "feature spec",
        )
        if raw["schema"] != FEATURE_SPEC_SCHEMA:
            raise PanelSoftOntologyError("feature spec schema differs")
        if raw["catalog_digest"] != feature_catalog_digest():
            raise PanelSoftOntologyError("feature catalog digest differs")
        try:
            family = FeatureFamily(raw["family"])
            result = cls(
                family,
                SubjectScope(raw["subject_scope"]),
                ReferenceFrame(raw["reference_frame"]),
                _parse_parameters(family, raw["parameters"]),
            )
        except (TypeError, ValueError) as exc:
            if isinstance(exc, PanelSoftOntologyError):
                raise
            raise PanelSoftOntologyError("feature spec value differs") from exc
        _require_canonical(result, raw, "feature spec")
        return result


def _fields(value: object, expected: set[str], label: str) -> Mapping[str, Any]:
    if (
        not isinstance(value, Mapping)
        or any(type(key) is not str for key in value)
        or set(value) != expected
    ):
        raise PanelSoftOntologyError(f"{label} fields differ")
    return value


def _require_canonical(value: object, raw: Mapping[str, Any], label: str) -> None:
    if value.to_data() != dict(raw):
        raise PanelSoftOntologyError(f"{label} is not canonical")


def _enum_instance(value: object, enum_type: type[Enum], label: str) -> None:
    if type(value) is not enum_type:
        raise TypeError(f"{label} must be {enum_type.__name__}")


def _digest(value: object, label: str) -> str:
    if type(value) is not str or _DIGEST.fullmatch(value) is None:
        raise PanelSoftOntologyError(f"{label} must be a lowercase SHA-256")
    return value


def _code(value: object, label: str) -> str:
    if type(value) is not str or _CODE.fullmatch(value) is None:
        raise PanelSoftOntologyError(f"{label} must be a bounded lowercase code")
    return value


def _text(value: object, label: str) -> str:
    if (
        type(value) is not str
        or value != value.strip()
        or _TEXT.fullmatch(value) is None
    ):
        raise PanelSoftOntologyError(f"{label} must be bounded visible text")
    return value


@dataclass(frozen=True, slots=True)
class PanelFeatureNarration:
    """Non-executable prose attached to one exact typed feature spec."""

    spec_digest: str
    summary: str
    visible_indicators: tuple[str, ...]

    def __post_init__(self) -> None:
        _digest(self.spec_digest, "narration spec digest")
        _text(self.summary, "narration summary")
        if type(self.visible_indicators) is not tuple or not self.visible_indicators:
            raise PanelSoftOntologyError("narration indicators must be a non-empty tuple")
        if len(self.visible_indicators) > 8:
            raise PanelSoftOntologyError("narration has too many indicators")
        for item in self.visible_indicators:
            _text(item, "visible indicator")
        if len(set(self.visible_indicators)) != len(self.visible_indicators):
            raise PanelSoftOntologyError("narration indicators must be unique")

    @property
    def narration_digest(self) -> str:
        return canonical_digest(self.to_data())

    def to_data(self) -> dict[str, object]:
        return {
            "schema": FEATURE_NARRATION_SCHEMA,
            "spec_digest": self.spec_digest,
            "summary": self.summary,
            "visible_indicators": list(self.visible_indicators),
        }

    @classmethod
    def from_data(cls, value: object) -> "PanelFeatureNarration":
        raw = _fields(
            value,
            {"schema", "spec_digest", "summary", "visible_indicators"},
            "feature narration",
        )
        if raw["schema"] != FEATURE_NARRATION_SCHEMA:
            raise PanelSoftOntologyError("feature narration schema differs")
        indicators = raw["visible_indicators"]
        if type(indicators) is not list:
            raise PanelSoftOntologyError("visible indicators must be a JSON list")
        result = cls(raw["spec_digest"], raw["summary"], tuple(indicators))
        _require_canonical(result, raw, "feature narration")
        return result


class NativeOrientation(str, Enum):
    SIDE0_POSITIVE = "side0_positive"
    SIDE1_POSITIVE = "side1_positive"


@dataclass(frozen=True, slots=True)
class NativeProposalProvenance:
    """Proposal context; orientation is deliberately confined to this envelope."""

    native_orientation: NativeOrientation
    proposer_contract_digest: str
    proposer_receipt_digest: str
    support_set_digest: str
    task_context_digest: str

    def __post_init__(self) -> None:
        _enum_instance(self.native_orientation, NativeOrientation, "native orientation")
        _digest(self.proposer_contract_digest, "proposer contract digest")
        _digest(self.proposer_receipt_digest, "proposer receipt digest")
        _digest(self.support_set_digest, "support set digest")
        _digest(self.task_context_digest, "task context digest")

    @property
    def provenance_digest(self) -> str:
        return canonical_digest(self.to_data())

    def to_data(self) -> dict[str, object]:
        return {
            "schema": NATIVE_PROPOSAL_PROVENANCE_SCHEMA,
            "native_orientation": self.native_orientation.value,
            "proposer_contract_digest": self.proposer_contract_digest,
            "proposer_receipt_digest": self.proposer_receipt_digest,
            "support_set_digest": self.support_set_digest,
            "task_context_digest": self.task_context_digest,
            "predicate_authority_id": PYTHON_PREDICATE_AUTHORITY_ID,
        }

    @classmethod
    def from_data(cls, value: object) -> "NativeProposalProvenance":
        raw = _fields(
            value,
            {
                "schema",
                "native_orientation",
                "proposer_contract_digest",
                "proposer_receipt_digest",
                "support_set_digest",
                "task_context_digest",
                "predicate_authority_id",
            },
            "proposal provenance",
        )
        if raw["schema"] != NATIVE_PROPOSAL_PROVENANCE_SCHEMA:
            raise PanelSoftOntologyError("proposal provenance schema differs")
        if raw["predicate_authority_id"] != PYTHON_PREDICATE_AUTHORITY_ID:
            raise PanelSoftOntologyError("proposal predicate authority differs")
        try:
            result = cls(
                NativeOrientation(raw["native_orientation"]),
                raw["proposer_contract_digest"],
                raw["proposer_receipt_digest"],
                raw["support_set_digest"],
                raw["task_context_digest"],
            )
        except (TypeError, ValueError) as exc:
            if isinstance(exc, PanelSoftOntologyError):
                raise
            raise PanelSoftOntologyError("proposal provenance value differs") from exc
        _require_canonical(result, raw, "proposal provenance")
        return result


@dataclass(frozen=True, slots=True)
class NativeFeatureProposal:
    spec: PanelFeatureSpec
    narration: PanelFeatureNarration
    provenance: NativeProposalProvenance

    def __post_init__(self) -> None:
        if type(self.spec) is not PanelFeatureSpec:
            raise TypeError("native proposal spec must be PanelFeatureSpec")
        if type(self.narration) is not PanelFeatureNarration:
            raise TypeError("native proposal narration must be PanelFeatureNarration")
        if type(self.provenance) is not NativeProposalProvenance:
            raise TypeError("native proposal provenance has the wrong type")
        if self.narration.spec_digest != self.spec.spec_digest:
            raise PanelSoftOntologyError("narration is bound to another feature spec")

    @property
    def proposal_digest(self) -> str:
        return canonical_digest(self.to_data())

    def to_data(self) -> dict[str, object]:
        return {
            "schema": NATIVE_FEATURE_PROPOSAL_SCHEMA,
            "spec": self.spec.to_data(),
            "narration": self.narration.to_data(),
            "provenance": self.provenance.to_data(),
        }

    @classmethod
    def from_data(cls, value: object) -> "NativeFeatureProposal":
        raw = _fields(value, {"schema", "spec", "narration", "provenance"}, "proposal")
        if raw["schema"] != NATIVE_FEATURE_PROPOSAL_SCHEMA:
            raise PanelSoftOntologyError("native proposal schema differs")
        result = cls(
            PanelFeatureSpec.from_data(raw["spec"]),
            PanelFeatureNarration.from_data(raw["narration"]),
            NativeProposalProvenance.from_data(raw["provenance"]),
        )
        _require_canonical(result, raw, "native proposal")
        return result


@dataclass(frozen=True, order=True, slots=True)
class QuantizedPoint:
    """One point on the fixed 16-by-16 panel-local grid."""

    x: int
    y: int

    def __post_init__(self) -> None:
        for label, item in (("x", self.x), ("y", self.y)):
            if type(item) is not int or not 0 <= item < GRID16_BIN_COUNT:
                raise PanelSoftOntologyError(f"Grid16 {label} coordinate is outside [0, 15]")

    def to_data(self) -> dict[str, object]:
        return {"schema": QUANTIZED_POINT_SCHEMA, "x": self.x, "y": self.y}

    @classmethod
    def from_data(cls, value: object) -> "QuantizedPoint":
        raw = _fields(value, {"schema", "x", "y"}, "Grid16 point")
        if raw["schema"] != QUANTIZED_POINT_SCHEMA:
            raise PanelSoftOntologyError("Grid16 point schema differs")
        result = cls(raw["x"], raw["y"])
        _require_canonical(result, raw, "Grid16 point")
        return result


@dataclass(frozen=True, order=True, slots=True)
class QuantizedRegion:
    """Inclusive panel-local bounding box on the fixed 16-bin grid."""

    minimum: QuantizedPoint
    maximum: QuantizedPoint

    def __post_init__(self) -> None:
        if type(self.minimum) is not QuantizedPoint or type(self.maximum) is not QuantizedPoint:
            raise TypeError("Grid16 region endpoints must be QuantizedPoint")
        if self.minimum.x > self.maximum.x or self.minimum.y > self.maximum.y:
            raise PanelSoftOntologyError("Grid16 region minimum exceeds maximum")

    def to_data(self) -> dict[str, object]:
        return {
            "schema": QUANTIZED_REGION_SCHEMA,
            "minimum": self.minimum.to_data(),
            "maximum": self.maximum.to_data(),
        }

    @classmethod
    def from_data(cls, value: object) -> "QuantizedRegion":
        raw = _fields(value, {"schema", "minimum", "maximum"}, "Grid16 region")
        if raw["schema"] != QUANTIZED_REGION_SCHEMA:
            raise PanelSoftOntologyError("Grid16 region schema differs")
        result = cls(
            QuantizedPoint.from_data(raw["minimum"]),
            QuantizedPoint.from_data(raw["maximum"]),
        )
        _require_canonical(result, raw, "Grid16 region")
        return result


@dataclass(frozen=True, order=True, slots=True)
class OwnerId:
    value: str

    def __post_init__(self) -> None:
        if type(self.value) is not str or _OWNER_ID.fullmatch(self.value) is None:
            raise PanelSoftOntologyError("owner ID must have form owner_NNNN")


@dataclass(frozen=True, order=True, slots=True)
class PanelLocalOwner:
    owner_id: OwnerId
    kind: OwnerKind
    region: QuantizedRegion
    parent_owner_ids: tuple[OwnerId, ...] = ()

    def __post_init__(self) -> None:
        if type(self.owner_id) is not OwnerId:
            raise TypeError("panel owner ID must be OwnerId")
        _enum_instance(self.kind, OwnerKind, "owner kind")
        if type(self.region) is not QuantizedRegion:
            raise TypeError("panel owner region must be QuantizedRegion")
        if type(self.parent_owner_ids) is not tuple or any(
            type(item) is not OwnerId for item in self.parent_owner_ids
        ):
            raise TypeError("owner parents must be an OwnerId tuple")
        parent_values = tuple(item.value for item in self.parent_owner_ids)
        if (
            parent_values != tuple(sorted(parent_values))
            or len(parent_values) != len(set(parent_values))
            or self.owner_id in self.parent_owner_ids
        ):
            raise PanelSoftOntologyError("owner parents must be distinct, sorted, and non-self")

    def to_data(self) -> dict[str, object]:
        return {
            "schema": PANEL_OWNER_SCHEMA,
            "owner_id": self.owner_id.value,
            "kind": self.kind.value,
            "region": self.region.to_data(),
            "parent_owner_ids": [item.value for item in self.parent_owner_ids],
        }

    @classmethod
    def from_data(cls, value: object) -> "PanelLocalOwner":
        raw = _fields(
            value,
            {"schema", "owner_id", "kind", "region", "parent_owner_ids"},
            "panel owner",
        )
        if raw["schema"] != PANEL_OWNER_SCHEMA:
            raise PanelSoftOntologyError("panel owner schema differs")
        if type(raw["parent_owner_ids"]) is not list:
            raise PanelSoftOntologyError("owner parent IDs must be a JSON list")
        try:
            result = cls(
                OwnerId(raw["owner_id"]),
                OwnerKind(raw["kind"]),
                QuantizedRegion.from_data(raw["region"]),
                tuple(OwnerId(item) for item in raw["parent_owner_ids"]),
            )
        except (TypeError, ValueError) as exc:
            if isinstance(exc, PanelSoftOntologyError):
                raise
            raise PanelSoftOntologyError("panel owner value differs") from exc
        _require_canonical(result, raw, "panel owner")
        return result


@dataclass(frozen=True, slots=True)
class OwnerInventory:
    """Candidate-independent inventory produced once from exact panel pixels."""

    panel_digest: str
    enumeration_protocol_digest: str
    enumeration_resolution: EnumerationResolution
    enumeration_receipt_digest: str
    enumeration_complete: bool
    owners: tuple[PanelLocalOwner, ...]

    def __post_init__(self) -> None:
        _digest(self.panel_digest, "inventory panel digest")
        _digest(self.enumeration_protocol_digest, "enumeration protocol digest")
        _enum_instance(
            self.enumeration_resolution,
            EnumerationResolution,
            "enumeration resolution",
        )
        _digest(self.enumeration_receipt_digest, "enumeration receipt digest")
        if type(self.enumeration_complete) is not bool:
            raise TypeError("enumeration_complete must be an exact bool")
        if type(self.owners) is not tuple:
            raise TypeError("inventory owners must be a tuple")
        if any(type(item) is not PanelLocalOwner for item in self.owners):
            raise TypeError("inventory contains a non-owner")
        keys = tuple(item.owner_id.value for item in self.owners)
        if keys != tuple(sorted(keys)) or len(keys) != len(set(keys)):
            raise PanelSoftOntologyError("inventory owners must be unique and sorted")
        available = {item.owner_id for item in self.owners}
        parents = {item.owner_id: item.parent_owner_ids for item in self.owners}
        if any(not set(item) <= available for item in parents.values()):
            raise PanelSoftOntologyError("owner inventory has an unknown parent")

        def visit(owner: OwnerId, path: frozenset[OwnerId]) -> None:
            if owner in path:
                raise PanelSoftOntologyError("owner inventory ancestry contains a cycle")
            for parent in parents[owner]:
                visit(parent, path | {owner})

        for owner in parents:
            visit(owner, frozenset())

    @property
    def inventory_digest(self) -> str:
        return canonical_digest(self.to_data())

    def to_data(self) -> dict[str, object]:
        return {
            "schema": OWNER_INVENTORY_SCHEMA,
            "panel_digest": self.panel_digest,
            "enumeration_protocol_id": OWNER_ENUMERATION_PROTOCOL_ID,
            "enumeration_protocol_digest": self.enumeration_protocol_digest,
            "enumeration_resolution": self.enumeration_resolution.value,
            "enumeration_receipt_digest": self.enumeration_receipt_digest,
            "enumeration_complete": self.enumeration_complete,
            "owners": [item.to_data() for item in self.owners],
        }

    @classmethod
    def from_data(cls, value: object) -> "OwnerInventory":
        raw = _fields(
            value,
            {
                "schema",
                "panel_digest",
                "enumeration_protocol_id",
                "enumeration_protocol_digest",
                "enumeration_resolution",
                "enumeration_receipt_digest",
                "enumeration_complete",
                "owners",
            },
            "owner inventory",
        )
        if raw["schema"] != OWNER_INVENTORY_SCHEMA:
            raise PanelSoftOntologyError("owner inventory schema differs")
        if raw["enumeration_protocol_id"] != OWNER_ENUMERATION_PROTOCOL_ID:
            raise PanelSoftOntologyError("owner enumeration protocol differs")
        owners = raw["owners"]
        if type(owners) is not list:
            raise PanelSoftOntologyError("inventory owners must be a JSON list")
        try:
            result = cls(
                raw["panel_digest"],
                raw["enumeration_protocol_digest"],
                EnumerationResolution(raw["enumeration_resolution"]),
                raw["enumeration_receipt_digest"],
                raw["enumeration_complete"],
                tuple(PanelLocalOwner.from_data(item) for item in owners),
            )
        except (TypeError, ValueError) as exc:
            if isinstance(exc, PanelSoftOntologyError):
                raise
            raise PanelSoftOntologyError("owner inventory value differs") from exc
        _require_canonical(result, raw, "owner inventory")
        return result


@dataclass(frozen=True, order=True, slots=True)
class SubjectBinding:
    kind: SubjectBindingKind
    owner_ids: tuple[OwnerId, ...]

    def __post_init__(self) -> None:
        _enum_instance(self.kind, SubjectBindingKind, "subject binding kind")
        if type(self.owner_ids) is not tuple or any(
            type(item) is not OwnerId for item in self.owner_ids
        ):
            raise TypeError("subject owner IDs must be an OwnerId tuple")
        expected_arity = {
            SubjectBindingKind.PANEL: 0,
            SubjectBindingKind.UNARY: 1,
            SubjectBindingKind.UNORDERED_PAIR: 2,
            SubjectBindingKind.ORDERED_CONTAINER_CONTAINED: 2,
        }[self.kind]
        if len(self.owner_ids) != expected_arity:
            raise PanelSoftOntologyError("subject binding arity differs")
        values = tuple(item.value for item in self.owner_ids)
        if len(values) != len(set(values)):
            raise PanelSoftOntologyError("subject binding owners must be distinct")
        if self.kind is SubjectBindingKind.UNORDERED_PAIR and values != tuple(sorted(values)):
            raise PanelSoftOntologyError("unordered subject owners must be sorted")

    @property
    def binding_digest(self) -> str:
        return canonical_digest(self.to_data())

    def to_data(self) -> dict[str, object]:
        return {
            "schema": SUBJECT_BINDING_SCHEMA,
            "kind": self.kind.value,
            "owner_ids": [item.value for item in self.owner_ids],
        }

    @classmethod
    def from_data(cls, value: object) -> "SubjectBinding":
        raw = _fields(value, {"schema", "kind", "owner_ids"}, "subject binding")
        if raw["schema"] != SUBJECT_BINDING_SCHEMA:
            raise PanelSoftOntologyError("subject binding schema differs")
        owner_ids = raw["owner_ids"]
        if type(owner_ids) is not list:
            raise PanelSoftOntologyError("subject owner IDs must be a JSON list")
        try:
            result = cls(
                SubjectBindingKind(raw["kind"]),
                tuple(OwnerId(item) for item in owner_ids),
            )
        except (TypeError, ValueError) as exc:
            if isinstance(exc, PanelSoftOntologyError):
                raise
            raise PanelSoftOntologyError("subject binding value differs") from exc
        _require_canonical(result, raw, "subject binding")
        return result

    def validate_inventory(self, inventory: OwnerInventory) -> None:
        if type(inventory) is not OwnerInventory:
            raise TypeError("subject inventory must be OwnerInventory")
        available = {item.owner_id for item in inventory.owners}
        if not set(self.owner_ids) <= available:
            raise PanelSoftOntologyError("subject references an owner outside the inventory")


def subject_search_region(
    subject: SubjectBinding, inventory: OwnerInventory
) -> QuantizedRegion:
    """Derive the sole exhaustive Grid16 region for a frozen subject binding."""

    if type(subject) is not SubjectBinding or type(inventory) is not OwnerInventory:
        raise TypeError("subject search region needs typed binding and inventory")
    subject.validate_inventory(inventory)
    if subject.kind is SubjectBindingKind.PANEL:
        return QuantizedRegion(
            QuantizedPoint(0, 0),
            QuantizedPoint(GRID16_BIN_COUNT - 1, GRID16_BIN_COUNT - 1),
        )
    owner_by_id = {item.owner_id: item for item in inventory.owners}
    regions = tuple(owner_by_id[item].region for item in subject.owner_ids)
    return QuantizedRegion(
        QuantizedPoint(
            min(item.minimum.x for item in regions),
            min(item.minimum.y for item in regions),
        ),
        QuantizedPoint(
            max(item.maximum.x for item in regions),
            max(item.maximum.y for item in regions),
        ),
    )


def _point_within_region(point: QuantizedPoint, region: QuantizedRegion) -> bool:
    return (
        region.minimum.x <= point.x <= region.maximum.x
        and region.minimum.y <= point.y <= region.maximum.y
    )


def _region_within_region(inner: QuantizedRegion, outer: QuantizedRegion) -> bool:
    return _point_within_region(inner.minimum, outer) and _point_within_region(
        inner.maximum, outer
    )


def _subject_projection_rule_data(spec: PanelFeatureSpec) -> dict[str, object]:
    contract = FAMILY_CONTRACTS[spec.family]
    return {
        "rule_id": SUBJECT_PROJECTION_RULE_ID,
        "catalog_digest": feature_catalog_digest(),
        "family": spec.family.value,
        "subject_scope": spec.subject_scope.value,
        "binding_kind": contract.binding_by_scope[spec.subject_scope].value,
        "owner_kinds": [
            item.value for item in contract.owner_kinds_by_scope[spec.subject_scope]
        ],
    }


@dataclass(frozen=True, slots=True)
class SearchResolutionDomain:
    family: FeatureFamily
    subject_scope: SubjectScope
    binding_kind: SubjectBindingKind
    eligible_owner_kinds: tuple[OwnerKind, ...]
    enumeration_resolution: EnumerationResolution
    projection_rule_digest: str

    def __post_init__(self) -> None:
        _enum_instance(self.family, FeatureFamily, "search-domain family")
        _enum_instance(self.subject_scope, SubjectScope, "search-domain scope")
        _enum_instance(self.binding_kind, SubjectBindingKind, "search-domain binding")
        if type(self.eligible_owner_kinds) is not tuple or any(
            type(item) is not OwnerKind for item in self.eligible_owner_kinds
        ):
            raise TypeError("eligible owner kinds must be an OwnerKind tuple")
        values = tuple(item.value for item in self.eligible_owner_kinds)
        if values != tuple(sorted(values)) or len(values) != len(set(values)):
            raise PanelSoftOntologyError("eligible owner kinds must be unique and sorted")
        _enum_instance(
            self.enumeration_resolution,
            EnumerationResolution,
            "search-domain resolution",
        )
        _digest(self.projection_rule_digest, "subject projection rule digest")

    @classmethod
    def for_spec(cls, spec: PanelFeatureSpec) -> "SearchResolutionDomain":
        if type(spec) is not PanelFeatureSpec:
            raise TypeError("search domain requires PanelFeatureSpec")
        contract = FAMILY_CONTRACTS[spec.family]
        kinds = tuple(
            sorted(contract.owner_kinds_by_scope[spec.subject_scope], key=lambda item: item.value)
        )
        return cls(
            spec.family,
            spec.subject_scope,
            contract.binding_by_scope[spec.subject_scope],
            kinds,
            EnumerationResolution.GRID16_FULL_PANEL,
            canonical_digest(_subject_projection_rule_data(spec)),
        )

    @property
    def domain_digest(self) -> str:
        return canonical_digest(self.to_data())

    def to_data(self) -> dict[str, object]:
        return {
            "schema": SEARCH_DOMAIN_SCHEMA,
            "catalog_digest": feature_catalog_digest(),
            "family": self.family.value,
            "subject_scope": self.subject_scope.value,
            "binding_kind": self.binding_kind.value,
            "eligible_owner_kinds": [item.value for item in self.eligible_owner_kinds],
            "enumeration_resolution": self.enumeration_resolution.value,
            "projection_rule_digest": self.projection_rule_digest,
        }

    @classmethod
    def from_data(cls, value: object) -> "SearchResolutionDomain":
        raw = _fields(
            value,
            {
                "schema",
                "catalog_digest",
                "family",
                "subject_scope",
                "binding_kind",
                "eligible_owner_kinds",
                "enumeration_resolution",
                "projection_rule_digest",
            },
            "search domain",
        )
        if raw["schema"] != SEARCH_DOMAIN_SCHEMA:
            raise PanelSoftOntologyError("search domain schema differs")
        if raw["catalog_digest"] != feature_catalog_digest():
            raise PanelSoftOntologyError("search domain catalog differs")
        kinds = raw["eligible_owner_kinds"]
        if type(kinds) is not list:
            raise PanelSoftOntologyError("eligible owner kinds must be a JSON list")
        try:
            result = cls(
                FeatureFamily(raw["family"]),
                SubjectScope(raw["subject_scope"]),
                SubjectBindingKind(raw["binding_kind"]),
                tuple(OwnerKind(item) for item in kinds),
                EnumerationResolution(raw["enumeration_resolution"]),
                raw["projection_rule_digest"],
            )
        except (TypeError, ValueError) as exc:
            if isinstance(exc, PanelSoftOntologyError):
                raise
            raise PanelSoftOntologyError("search domain value differs") from exc
        _require_canonical(result, raw, "search domain")
        return result

    def validate_spec(self, spec: PanelFeatureSpec) -> None:
        if self != SearchResolutionDomain.for_spec(spec):
            raise PanelSoftOntologyError("search domain does not match feature spec")


def eligible_subject_bindings(
    spec: PanelFeatureSpec,
    inventory: OwnerInventory,
    domain: SearchResolutionDomain | None = None,
) -> tuple[SubjectBinding, ...]:
    """Deterministically project a candidate-independent owner inventory."""

    if type(spec) is not PanelFeatureSpec or type(inventory) is not OwnerInventory:
        raise TypeError("eligible binding projection requires typed spec and inventory")
    expected = SearchResolutionDomain.for_spec(spec)
    if domain is None:
        domain = expected
    if type(domain) is not SearchResolutionDomain or domain != expected:
        raise PanelSoftOntologyError("eligible binding search domain differs")
    if inventory.enumeration_resolution is not domain.enumeration_resolution:
        raise PanelSoftOntologyError("inventory and search resolution differ")
    owners = tuple(
        item.owner_id
        for item in inventory.owners
        if item.kind in set(domain.eligible_owner_kinds)
    )
    if domain.binding_kind is SubjectBindingKind.PANEL:
        return (SubjectBinding(SubjectBindingKind.PANEL, ()),)
    if domain.binding_kind is SubjectBindingKind.UNARY:
        return tuple(SubjectBinding(SubjectBindingKind.UNARY, (item,)) for item in owners)
    if domain.binding_kind is SubjectBindingKind.UNORDERED_PAIR:
        return tuple(
            SubjectBinding(SubjectBindingKind.UNORDERED_PAIR, pair)
            for pair in combinations(owners, 2)
        )
    return tuple(
        SubjectBinding(SubjectBindingKind.ORDERED_CONTAINER_CONTAINED, pair)
        for pair in permutations(owners, 2)
    )


_COHERENT_COMPONENT_KINDS = frozenset(
    {OwnerKind.FIGURE, OwnerKind.TRACE, OwnerKind.LOOP}
)


def coherent_top_level_component_owner_ids(
    inventory: OwnerInventory,
) -> tuple[OwnerId, ...]:
    """Return the explicit whole-panel component membership.

    A coherent component may be represented by a filled/compound figure, an
    open trace, or a closed loop.  Only roots count: a trace or loop already
    governed by a figure is structure inside that component, not another
    component.  Segments and markers are never promoted to panel components.
    """

    if type(inventory) is not OwnerInventory:
        raise TypeError("component membership requires OwnerInventory")
    return tuple(
        item.owner_id
        for item in inventory.owners
        if item.kind in _COHERENT_COMPONENT_KINDS and not item.parent_owner_ids
    )


def descendant_segment_owner_ids(
    subject_owner_id: OwnerId,
    inventory: OwnerInventory,
) -> tuple[OwnerId, ...]:
    """Return every segment transitively governed by one coherent subject."""

    if type(subject_owner_id) is not OwnerId:
        raise TypeError("segment membership subject must be OwnerId")
    if type(inventory) is not OwnerInventory:
        raise TypeError("segment membership requires OwnerInventory")
    parents = {
        item.owner_id: item.parent_owner_ids for item in inventory.owners
    }
    if subject_owner_id not in parents:
        raise PanelSoftOntologyError("segment membership subject is outside inventory")

    descendant_cache: dict[OwnerId, bool] = {}

    def descends_from_subject(owner_id: OwnerId) -> bool:
        retained = descendant_cache.get(owner_id)
        if retained is not None:
            return retained
        result = any(
            parent == subject_owner_id or descends_from_subject(parent)
            for parent in parents[owner_id]
        )
        descendant_cache[owner_id] = result
        return result

    return tuple(
        item.owner_id
        for item in inventory.owners
        if item.kind is OwnerKind.SEGMENT
        and descends_from_subject(item.owner_id)
    )


def segment_owner_ids_for_subject(
    subject: SubjectBinding,
    inventory: OwnerInventory,
) -> tuple[OwnerId, ...]:
    """Return the generic segment-owner universe for a panel or unary subject.

    This function deliberately says nothing about straightness.  A segment
    owner enters the straight-segment subset only through explicit geometric
    classification evidence.
    """

    if type(subject) is not SubjectBinding or type(inventory) is not OwnerInventory:
        raise TypeError("segment membership requires typed subject and inventory")
    subject.validate_inventory(inventory)
    if subject.kind is SubjectBindingKind.PANEL:
        return tuple(
            item.owner_id
            for item in inventory.owners
            if item.kind is OwnerKind.SEGMENT
        )
    if subject.kind is SubjectBindingKind.UNARY:
        return descendant_segment_owner_ids(subject.owner_ids[0], inventory)
    raise PanelSoftOntologyError(
        "segment membership is defined only for panel or unary subjects"
    )


def _closed_count_value(value: ClosedCount) -> int:
    return _CLOSED_COUNT_TO_INT[value]


@dataclass(frozen=True, order=True, slots=True)
class QuantizedSegment:
    start: QuantizedPoint
    end: QuantizedPoint

    def __post_init__(self) -> None:
        if type(self.start) is not QuantizedPoint or type(self.end) is not QuantizedPoint:
            raise TypeError("quantized segment endpoints must be QuantizedPoint")
        if self.start == self.end:
            raise PanelSoftOntologyError("quantized segment endpoints must differ")

    def to_data(self) -> dict[str, object]:
        return {"start": self.start.to_data(), "end": self.end.to_data()}

    @classmethod
    def from_data(cls, value: object) -> "QuantizedSegment":
        raw = _fields(value, {"start", "end"}, "quantized segment")
        result = cls(
            QuantizedPoint.from_data(raw["start"]),
            QuantizedPoint.from_data(raw["end"]),
        )
        _require_canonical(result, raw, "quantized segment")
        return result


class BoundaryPolygonIssue(str, Enum):
    OPEN_BOUNDARY = "open_boundary"
    DEGENERATE_BOUNDARY = "degenerate_boundary"
    SELF_INTERSECTING_BOUNDARY = "self_intersecting_boundary"
    CAPACITY_LIMIT = "capacity_limit"


class BoundaryPolygonError(PanelSoftOntologyError):
    """A typed geometric reason an ordered boundary is not a simple polygon."""

    def __init__(self, issue: BoundaryPolygonIssue, message: str) -> None:
        if type(issue) is not BoundaryPolygonIssue:
            raise TypeError("boundary polygon error needs BoundaryPolygonIssue")
        self.issue = issue
        super().__init__(message)


def _turn_cross(
    first: QuantizedPoint,
    middle: QuantizedPoint,
    last: QuantizedPoint,
) -> int:
    return (middle.x - first.x) * (last.y - middle.y) - (
        middle.y - first.y
    ) * (last.x - middle.x)


def _signed_double_area(vertices: tuple[QuantizedPoint, ...]) -> int:
    return sum(
        point.x * vertices[(index + 1) % len(vertices)].y
        - vertices[(index + 1) % len(vertices)].x * point.y
        for index, point in enumerate(vertices)
    )


def _point_on_segment(
    point: QuantizedPoint,
    start: QuantizedPoint,
    end: QuantizedPoint,
) -> bool:
    return (
        _turn_cross(start, point, end) == 0
        and min(start.x, end.x) <= point.x <= max(start.x, end.x)
        and min(start.y, end.y) <= point.y <= max(start.y, end.y)
    )


def _segments_intersect(
    first_start: QuantizedPoint,
    first_end: QuantizedPoint,
    second_start: QuantizedPoint,
    second_end: QuantizedPoint,
) -> bool:
    first_side_start = _turn_cross(first_start, first_end, second_start)
    first_side_end = _turn_cross(first_start, first_end, second_end)
    second_side_start = _turn_cross(second_start, second_end, first_start)
    second_side_end = _turn_cross(second_start, second_end, first_end)
    if (
        (first_side_start > 0) != (first_side_end > 0)
        and (second_side_start > 0) != (second_side_end > 0)
        and 0 not in {
            first_side_start,
            first_side_end,
            second_side_start,
            second_side_end,
        }
    ):
        return True
    return any(
        cross == 0 and _point_on_segment(point, start, end)
        for cross, point, start, end in (
            (first_side_start, second_start, first_start, first_end),
            (first_side_end, second_end, first_start, first_end),
            (second_side_start, first_start, second_start, second_end),
            (second_side_end, first_end, second_start, second_end),
        )
    )


def _canonical_boundary_vertices(
    vertices: tuple[QuantizedPoint, ...],
) -> tuple[QuantizedPoint, ...]:
    if type(vertices) is not tuple or any(
        type(item) is not QuantizedPoint for item in vertices
    ):
        raise TypeError("boundary vertices must be a QuantizedPoint tuple")
    if len(vertices) > MAX_BOUNDARY_VERTEX_COUNT:
        raise BoundaryPolygonError(
            BoundaryPolygonIssue.CAPACITY_LIMIT,
            "boundary exceeds the fixed vertex capacity",
        )
    if len(vertices) < 3:
        raise BoundaryPolygonError(
            BoundaryPolygonIssue.DEGENERATE_BOUNDARY,
            "closed boundary needs at least three distinct vertices",
        )
    if len(set(vertices)) != len(vertices):
        raise BoundaryPolygonError(
            BoundaryPolygonIssue.SELF_INTERSECTING_BOUNDARY,
            "closed boundary repeats a non-closure vertex",
        )

    simplified = list(vertices)
    while len(simplified) >= 3:
        redundant_index: int | None = None
        for index, middle in enumerate(simplified):
            first = simplified[index - 1]
            last = simplified[(index + 1) % len(simplified)]
            if _turn_cross(first, middle, last) != 0:
                continue
            incoming_x = middle.x - first.x
            incoming_y = middle.y - first.y
            outgoing_x = last.x - middle.x
            outgoing_y = last.y - middle.y
            if incoming_x * outgoing_x + incoming_y * outgoing_y <= 0:
                raise BoundaryPolygonError(
                    BoundaryPolygonIssue.DEGENERATE_BOUNDARY,
                    "closed boundary doubles back along an edge",
                )
            redundant_index = index
            break
        if redundant_index is None:
            break
        del simplified[redundant_index]
    if len(simplified) < 3:
        raise BoundaryPolygonError(
            BoundaryPolygonIssue.DEGENERATE_BOUNDARY,
            "closed boundary collapses after collinear normalization",
        )

    normalized = tuple(simplified)
    edge_count = len(normalized)
    for first_index in range(edge_count):
        first_end_index = (first_index + 1) % edge_count
        for second_index in range(first_index + 1, edge_count):
            second_end_index = (second_index + 1) % edge_count
            if (
                second_index == first_end_index
                or first_index == second_end_index
            ):
                continue
            if _segments_intersect(
                normalized[first_index],
                normalized[first_end_index],
                normalized[second_index],
                normalized[second_end_index],
            ):
                raise BoundaryPolygonError(
                    BoundaryPolygonIssue.SELF_INTERSECTING_BOUNDARY,
                    "closed boundary has intersecting nonadjacent edges",
                )

    area = _signed_double_area(normalized)
    if area == 0:
        raise BoundaryPolygonError(
            BoundaryPolygonIssue.DEGENERATE_BOUNDARY,
            "closed boundary has zero signed area",
        )
    oriented = normalized if area > 0 else tuple(reversed(normalized))
    start = min(range(len(oriented)), key=lambda index: oriented[index])
    return oriented[start:] + oriented[:start]


@dataclass(frozen=True, order=True, slots=True)
class CanonicalBoundaryPolygon:
    """Canonical simple nondegenerate Grid16 polygon without repeated closure."""

    vertices: tuple[QuantizedPoint, ...]

    def __post_init__(self) -> None:
        if _canonical_boundary_vertices(self.vertices) != self.vertices:
            raise PanelSoftOntologyError(
                "boundary polygon orientation, start, or collinearity is not canonical"
            )

    @classmethod
    def from_closed_vertex_walk(
        cls, vertices: tuple[QuantizedPoint, ...]
    ) -> "CanonicalBoundaryPolygon":
        if type(vertices) is not tuple or any(
            type(item) is not QuantizedPoint for item in vertices
        ):
            raise TypeError("closed boundary walk must be a QuantizedPoint tuple")
        if len(vertices) > MAX_BOUNDARY_VERTEX_COUNT + 1:
            raise BoundaryPolygonError(
                BoundaryPolygonIssue.CAPACITY_LIMIT,
                "closed boundary walk exceeds the fixed vertex capacity",
            )
        if len(vertices) < 2 or vertices[0] != vertices[-1]:
            raise BoundaryPolygonError(
                BoundaryPolygonIssue.OPEN_BOUNDARY,
                "ordered boundary walk does not explicitly return to its first vertex",
            )
        return cls(_canonical_boundary_vertices(vertices[:-1]))

    @property
    def closed_vertex_walk(self) -> tuple[QuantizedPoint, ...]:
        return self.vertices + (self.vertices[0],)

    @property
    def convexity_kind(self) -> ConvexityKind:
        turns = tuple(
            _turn_cross(
                self.vertices[index - 1],
                point,
                self.vertices[(index + 1) % len(self.vertices)],
            )
            for index, point in enumerate(self.vertices)
        )
        return (
            ConvexityKind.CONVEX_CLOSED_BOUNDARY
            if all(item > 0 for item in turns)
            else ConvexityKind.CONCAVE_CLOSED_BOUNDARY
        )

    @property
    def polygon_digest(self) -> str:
        return canonical_digest(self.to_data())

    def to_data(self) -> dict[str, object]:
        return {
            "schema": "gkm.bongard-canonical-boundary-polygon.v1",
            "vertices": [item.to_data() for item in self.vertices],
            "closure": "implicit_last_to_first",
            "orientation": "positive_signed_area",
            "start": "lexicographically_minimum_vertex",
            "collinear_redundancy": "removed",
        }

    @classmethod
    def from_data(cls, value: object) -> "CanonicalBoundaryPolygon":
        raw = _fields(
            value,
            {
                "schema",
                "vertices",
                "closure",
                "orientation",
                "start",
                "collinear_redundancy",
            },
            "canonical boundary polygon",
        )
        if (
            raw["schema"] != "gkm.bongard-canonical-boundary-polygon.v1"
            or raw["closure"] != "implicit_last_to_first"
            or raw["orientation"] != "positive_signed_area"
            or raw["start"] != "lexicographically_minimum_vertex"
            or raw["collinear_redundancy"] != "removed"
            or type(raw["vertices"]) is not list
        ):
            raise PanelSoftOntologyError("canonical boundary polygon policy differs")
        result = cls(
            tuple(QuantizedPoint.from_data(item) for item in raw["vertices"])
        )
        _require_canonical(result, raw, "canonical boundary polygon")
        return result


@dataclass(frozen=True, slots=True)
class CountWitnessPayload:
    counted_owner_ids: tuple[OwnerId, ...]
    membership_complete: bool
    membership_receipt_digest: str

    def __post_init__(self) -> None:
        if type(self.counted_owner_ids) is not tuple or any(
            type(item) is not OwnerId for item in self.counted_owner_ids
        ):
            raise TypeError("count witness owner IDs must be an OwnerId tuple")
        values = tuple(item.value for item in self.counted_owner_ids)
        if values != tuple(sorted(values)) or len(values) != len(set(values)):
            raise PanelSoftOntologyError("count witness owner IDs must be unique and sorted")
        if type(self.membership_complete) is not bool or self.membership_complete is not True:
            raise PanelSoftOntologyError("exact count witness requires complete membership")
        _digest(self.membership_receipt_digest, "count membership receipt digest")

    def to_data(self) -> dict[str, object]:
        return {
            "kind": "count",
            "counted_owner_ids": [i.value for i in self.counted_owner_ids],
            "membership_complete": self.membership_complete,
            "membership_receipt_digest": self.membership_receipt_digest,
        }

    @classmethod
    def from_data(cls, value: object) -> "CountWitnessPayload":
        raw = _fields(
            value,
            {"kind", "counted_owner_ids", "membership_complete", "membership_receipt_digest"},
            "count witness payload",
        )
        if raw["kind"] != "count" or type(raw["counted_owner_ids"]) is not list:
            raise PanelSoftOntologyError("count witness payload differs")
        result = cls(
            tuple(OwnerId(item) for item in raw["counted_owner_ids"]),
            raw["membership_complete"],
            raw["membership_receipt_digest"],
        )
        _require_canonical(result, raw, "count witness payload")
        return result


@dataclass(frozen=True, slots=True)
class StraightSegmentCountWitnessPayload:
    """Complete explicit straight/non-straight classification of segment owners.

    ``eligible_segment_owner_ids`` freezes the generic segment-owner universe.
    ``straight_segment_owner_ids`` is an explicit subset, positionally aligned
    with non-degenerate Grid16 line evidence.  Owner kind alone never places an
    item in that subset.
    """

    eligible_segment_owner_ids: tuple[OwnerId, ...]
    straight_segment_owner_ids: tuple[OwnerId, ...]
    straight_segments: tuple[QuantizedSegment, ...]
    classification_complete: bool
    classification_receipt_digest: str

    def __post_init__(self) -> None:
        for label, row in (
            ("eligible", self.eligible_segment_owner_ids),
            ("straight", self.straight_segment_owner_ids),
        ):
            if type(row) is not tuple or any(type(item) is not OwnerId for item in row):
                raise TypeError(
                    f"straight-segment {label} owner IDs must be an OwnerId tuple"
                )
            values = tuple(item.value for item in row)
            if values != tuple(sorted(values)) or len(values) != len(set(values)):
                raise PanelSoftOntologyError(
                    f"straight-segment {label} owner IDs must be unique and sorted"
                )
        if (
            not self.straight_segment_owner_ids
            or not set(self.straight_segment_owner_ids)
            <= set(self.eligible_segment_owner_ids)
        ):
            raise PanelSoftOntologyError(
                "straight-segment owners must be a nonempty eligible subset"
            )
        if (
            type(self.straight_segments) is not tuple
            or len(self.straight_segments) != len(self.straight_segment_owner_ids)
            or any(type(item) is not QuantizedSegment for item in self.straight_segments)
        ):
            raise TypeError(
                "straight-segment owners need one aligned QuantizedSegment each"
            )
        if any(item.start >= item.end for item in self.straight_segments):
            raise PanelSoftOntologyError(
                "straight-segment endpoints must use canonical ascending order"
            )
        if len(set(self.straight_segments)) != len(self.straight_segments):
            raise PanelSoftOntologyError(
                "straight-segment line evidence must be unique"
            )
        if (
            type(self.classification_complete) is not bool
            or self.classification_complete is not True
        ):
            raise PanelSoftOntologyError(
                "straight-segment count requires complete classification"
            )
        _digest(
            self.classification_receipt_digest,
            "straight-segment classification receipt digest",
        )

    def to_data(self) -> dict[str, object]:
        return {
            "kind": "straight_segment_count",
            "eligible_segment_owner_ids": [
                item.value for item in self.eligible_segment_owner_ids
            ],
            "straight_segment_owner_ids": [
                item.value for item in self.straight_segment_owner_ids
            ],
            "straight_segments": [item.to_data() for item in self.straight_segments],
            "classification_complete": self.classification_complete,
            "classification_receipt_digest": self.classification_receipt_digest,
        }

    @classmethod
    def from_data(cls, value: object) -> "StraightSegmentCountWitnessPayload":
        raw = _fields(
            value,
            {
                "kind",
                "eligible_segment_owner_ids",
                "straight_segment_owner_ids",
                "straight_segments",
                "classification_complete",
                "classification_receipt_digest",
            },
            "straight-segment count witness payload",
        )
        if (
            raw["kind"] != "straight_segment_count"
            or type(raw["eligible_segment_owner_ids"]) is not list
            or type(raw["straight_segment_owner_ids"]) is not list
            or type(raw["straight_segments"]) is not list
        ):
            raise PanelSoftOntologyError(
                "straight-segment count witness payload differs"
            )
        result = cls(
            tuple(OwnerId(item) for item in raw["eligible_segment_owner_ids"]),
            tuple(OwnerId(item) for item in raw["straight_segment_owner_ids"]),
            tuple(QuantizedSegment.from_data(item) for item in raw["straight_segments"]),
            raw["classification_complete"],
            raw["classification_receipt_digest"],
        )
        _require_canonical(result, raw, "straight-segment count witness payload")
        return result


@dataclass(frozen=True, slots=True)
class ConvexityWitnessPayload:
    """Complete outer boundary whose convexity class is derived by Python."""

    outer_boundary: CanonicalBoundaryPolygon
    boundary_complete: bool
    boundary_receipt_digest: str

    def __post_init__(self) -> None:
        if type(self.outer_boundary) is not CanonicalBoundaryPolygon:
            raise TypeError("convexity witness needs CanonicalBoundaryPolygon")
        if type(self.boundary_complete) is not bool or self.boundary_complete is not True:
            raise PanelSoftOntologyError(
                "convexity witness requires a complete outer boundary"
            )
        _digest(self.boundary_receipt_digest, "convexity boundary receipt digest")

    def to_data(self) -> dict[str, object]:
        return {
            "kind": "convexity",
            "outer_boundary": self.outer_boundary.to_data(),
            "boundary_complete": self.boundary_complete,
            "boundary_receipt_digest": self.boundary_receipt_digest,
        }

    @classmethod
    def from_data(cls, value: object) -> "ConvexityWitnessPayload":
        raw = _fields(
            value,
            {
                "kind",
                "outer_boundary",
                "boundary_complete",
                "boundary_receipt_digest",
            },
            "convexity witness payload",
        )
        if raw["kind"] != "convexity":
            raise PanelSoftOntologyError("convexity witness payload differs")
        result = cls(
            CanonicalBoundaryPolygon.from_data(raw["outer_boundary"]),
            raw["boundary_complete"],
            raw["boundary_receipt_digest"],
        )
        _require_canonical(result, raw, "convexity witness payload")
        return result


@dataclass(frozen=True, slots=True)
class MarkerWitnessPayload:
    marker_owner_ids: tuple[OwnerId, ...]
    marker_centers: tuple[QuantizedPoint, ...]
    membership_complete: bool
    membership_receipt_digest: str

    def __post_init__(self) -> None:
        if type(self.marker_centers) is not tuple or not self.marker_centers or any(
            type(item) is not QuantizedPoint for item in self.marker_centers
        ):
            raise TypeError("marker centers must be a non-empty QuantizedPoint tuple")
        if len(set(self.marker_centers)) != len(self.marker_centers):
            raise PanelSoftOntologyError("marker centers must be unique")
        if type(self.marker_owner_ids) is not tuple or any(
            type(item) is not OwnerId for item in self.marker_owner_ids
        ):
            raise TypeError("marker witness owner IDs must be an OwnerId tuple")
        owner_values = tuple(item.value for item in self.marker_owner_ids)
        if owner_values != tuple(sorted(owner_values)) or len(owner_values) != len(set(owner_values)):
            raise PanelSoftOntologyError("marker witness owner IDs must be unique and sorted")
        if len(self.marker_owner_ids) != len(self.marker_centers):
            raise PanelSoftOntologyError("marker owners and centers differ in length")
        # Centers are aligned positionally with the already sorted owner IDs;
        # sorting centers independently would silently change ownership.
        if type(self.membership_complete) is not bool or self.membership_complete is not True:
            raise PanelSoftOntologyError("marker repetition requires complete membership")
        _digest(self.membership_receipt_digest, "marker membership receipt digest")

    def to_data(self) -> dict[str, object]:
        return {
            "kind": "marker_pattern",
            "marker_owner_ids": [item.value for item in self.marker_owner_ids],
            "marker_centers": [i.to_data() for i in self.marker_centers],
            "membership_complete": self.membership_complete,
            "membership_receipt_digest": self.membership_receipt_digest,
        }

    @classmethod
    def from_data(cls, value: object) -> "MarkerWitnessPayload":
        raw = _fields(
            value,
            {"kind", "marker_owner_ids", "marker_centers", "membership_complete", "membership_receipt_digest"},
            "marker witness payload",
        )
        if (
            raw["kind"] != "marker_pattern"
            or type(raw["marker_owner_ids"]) is not list
            or type(raw["marker_centers"]) is not list
        ):
            raise PanelSoftOntologyError("marker witness payload differs")
        result = cls(
            tuple(OwnerId(item) for item in raw["marker_owner_ids"]),
            tuple(QuantizedPoint.from_data(item) for item in raw["marker_centers"]),
            raw["membership_complete"],
            raw["membership_receipt_digest"],
        )
        _require_canonical(result, raw, "marker witness payload")
        return result


_UNARY_WITNESS_FAMILIES = frozenset(
    {
        FeatureFamily.GESTALT_RESEMBLANCE,
        FeatureFamily.SEGMENT_ORIENTATION,
        FeatureFamily.CORNER_ANGLE,
        FeatureFamily.TURN_PROFILE,
        FeatureFamily.OPEN_TRACE,
        FeatureFamily.CLOSED_LOOP,
        FeatureFamily.SYMMETRY,
        FeatureFamily.ASPECT_RATIO,
        FeatureFamily.TEXTURE_COMPOSITION,
    }
)


class WitnessCoverage(str, Enum):
    LOCAL = "local"
    COMPLETE_ELIGIBLE = "complete_eligible"


@dataclass(frozen=True, slots=True)
class UnaryGeometryWitnessPayload:
    """Closed family tag plus bounded Grid16 geometry for unary visual families."""

    family: FeatureFamily
    primary_region: QuantizedRegion
    sample_points: tuple[QuantizedPoint, ...]
    coverage: WitnessCoverage
    coverage_receipt_digest: str

    def __post_init__(self) -> None:
        _enum_instance(self.family, FeatureFamily, "unary witness family")
        if self.family not in _UNARY_WITNESS_FAMILIES:
            raise PanelSoftOntologyError("family has no unary-geometry witness payload")
        if type(self.primary_region) is not QuantizedRegion:
            raise TypeError("unary witness region must be QuantizedRegion")
        if (
            type(self.sample_points) is not tuple
            or not 1 <= len(self.sample_points) <= 16
            or any(type(item) is not QuantizedPoint for item in self.sample_points)
        ):
            raise TypeError("unary witness samples must be 1..16 Grid16 points")
        if tuple(self.sample_points) != tuple(sorted(self.sample_points)):
            raise PanelSoftOntologyError("unary witness samples must be canonically sorted")
        if len(set(self.sample_points)) != len(self.sample_points):
            raise PanelSoftOntologyError("unary witness samples must be unique")
        _enum_instance(self.coverage, WitnessCoverage, "witness coverage")
        _digest(self.coverage_receipt_digest, "witness coverage receipt digest")

    def to_data(self) -> dict[str, object]:
        return {
            "kind": "unary_geometry",
            "family": self.family.value,
            "primary_region": self.primary_region.to_data(),
            "sample_points": [item.to_data() for item in self.sample_points],
            "coverage": self.coverage.value,
            "coverage_receipt_digest": self.coverage_receipt_digest,
        }

    @classmethod
    def from_data(cls, value: object) -> "UnaryGeometryWitnessPayload":
        raw = _fields(
            value,
            {"kind", "family", "primary_region", "sample_points", "coverage", "coverage_receipt_digest"},
            "unary witness payload",
        )
        if raw["kind"] != "unary_geometry" or type(raw["sample_points"]) is not list:
            raise PanelSoftOntologyError("unary witness payload differs")
        try:
            result = cls(
                FeatureFamily(raw["family"]),
                QuantizedRegion.from_data(raw["primary_region"]),
                tuple(QuantizedPoint.from_data(item) for item in raw["sample_points"]),
                WitnessCoverage(raw["coverage"]),
                raw["coverage_receipt_digest"],
            )
        except (TypeError, ValueError) as exc:
            if isinstance(exc, PanelSoftOntologyError):
                raise
            raise PanelSoftOntologyError("unary witness payload value differs") from exc
        _require_canonical(result, raw, "unary witness payload")
        return result


class RayDirection(str, Enum):
    N = "n"
    NE = "ne"
    E = "e"
    SE = "se"
    S = "s"
    SW = "sw"
    W = "w"
    NW = "nw"


@dataclass(frozen=True, order=True, slots=True)
class OwnerRay:
    owner_id: OwnerId
    direction: RayDirection

    def __post_init__(self) -> None:
        if type(self.owner_id) is not OwnerId:
            raise TypeError("owner ray owner must be OwnerId")
        _enum_instance(self.direction, RayDirection, "ray direction")

    def to_data(self) -> dict[str, str]:
        return {"owner_id": self.owner_id.value, "direction": self.direction.value}

    @classmethod
    def from_data(cls, value: object) -> "OwnerRay":
        raw = _fields(value, {"owner_id", "direction"}, "owner ray")
        try:
            result = cls(OwnerId(raw["owner_id"]), RayDirection(raw["direction"]))
        except (TypeError, ValueError) as exc:
            if isinstance(exc, PanelSoftOntologyError):
                raise
            raise PanelSoftOntologyError("owner ray value differs") from exc
        _require_canonical(result, raw, "owner ray")
        return result


@dataclass(frozen=True, slots=True)
class PointContactWitnessPayload:
    observed_kind: PointContactKind
    contact_point: QuantizedPoint
    owner_rays: tuple[OwnerRay, ...]
    exterior_gap_regions: tuple[QuantizedRegion, QuantizedRegion]

    def __post_init__(self) -> None:
        _enum_instance(self.observed_kind, PointContactKind, "observed contact kind")
        if type(self.contact_point) is not QuantizedPoint:
            raise TypeError("contact point must be QuantizedPoint")
        if type(self.owner_rays) is not tuple or len(self.owner_rays) != 4 or any(
            type(item) is not OwnerRay for item in self.owner_rays
        ):
            raise PanelSoftOntologyError("point contact requires four typed owner rays")
        ray_keys = tuple((item.owner_id.value, item.direction.value) for item in self.owner_rays)
        if ray_keys != tuple(sorted(ray_keys)) or len(ray_keys) != len(set(ray_keys)):
            raise PanelSoftOntologyError("point-contact rays must be unique and sorted")
        if type(self.exterior_gap_regions) is not tuple or len(self.exterior_gap_regions) != 2 or any(
            type(item) is not QuantizedRegion for item in self.exterior_gap_regions
        ):
            raise PanelSoftOntologyError("point contact requires both exterior gaps")
        if (
            self.exterior_gap_regions != tuple(sorted(self.exterior_gap_regions))
            or len(set(self.exterior_gap_regions)) != 2
        ):
            raise PanelSoftOntologyError("exterior gap regions must be distinct and sorted")

    def to_data(self) -> dict[str, object]:
        return {
            "kind": "point_contact",
            "observed_kind": self.observed_kind.value,
            "contact_point": self.contact_point.to_data(),
            "owner_rays": [item.to_data() for item in self.owner_rays],
            "exterior_gap_regions": [item.to_data() for item in self.exterior_gap_regions],
        }

    @classmethod
    def from_data(cls, value: object) -> "PointContactWitnessPayload":
        raw = _fields(
            value,
            {"kind", "observed_kind", "contact_point", "owner_rays", "exterior_gap_regions"},
            "point-contact witness payload",
        )
        if (
            raw["kind"] != "point_contact"
            or type(raw["owner_rays"]) is not list
            or type(raw["exterior_gap_regions"]) is not list
        ):
            raise PanelSoftOntologyError("point-contact witness payload differs")
        result = cls(
            PointContactKind(raw["observed_kind"]),
            QuantizedPoint.from_data(raw["contact_point"]),
            tuple(OwnerRay.from_data(item) for item in raw["owner_rays"]),
            tuple(QuantizedRegion.from_data(item) for item in raw["exterior_gap_regions"]),  # type: ignore[arg-type]
        )
        _require_canonical(result, raw, "point-contact witness payload")
        return result


@dataclass(frozen=True, slots=True)
class RelationRegionWitnessPayload:
    family: FeatureFamily
    relation_region: QuantizedRegion

    def __post_init__(self) -> None:
        _enum_instance(self.family, FeatureFamily, "relation witness family")
        if self.family not in {
            FeatureFamily.VISIBLE_GAP,
            FeatureFamily.SHARED_BOUNDARY_ADJACENCY,
        }:
            raise PanelSoftOntologyError("family has no relation-region payload")
        if type(self.relation_region) is not QuantizedRegion:
            raise TypeError("relation witness region must be QuantizedRegion")

    def to_data(self) -> dict[str, object]:
        return {
            "kind": "relation_region",
            "family": self.family.value,
            "relation_region": self.relation_region.to_data(),
        }

    @classmethod
    def from_data(cls, value: object) -> "RelationRegionWitnessPayload":
        raw = _fields(value, {"kind", "family", "relation_region"}, "relation payload")
        if raw["kind"] != "relation_region":
            raise PanelSoftOntologyError("relation witness payload differs")
        try:
            result = cls(
                FeatureFamily(raw["family"]),
                QuantizedRegion.from_data(raw["relation_region"]),
            )
        except (TypeError, ValueError) as exc:
            if isinstance(exc, PanelSoftOntologyError):
                raise
            raise PanelSoftOntologyError("relation witness payload value differs") from exc
        _require_canonical(result, raw, "relation witness payload")
        return result


@dataclass(frozen=True, slots=True)
class EnclosureWitnessPayload:
    container_region: QuantizedRegion
    contained_region: QuantizedRegion
    contained_interior_point: QuantizedPoint

    def __post_init__(self) -> None:
        if type(self.container_region) is not QuantizedRegion or type(self.contained_region) is not QuantizedRegion:
            raise TypeError("enclosure regions must be QuantizedRegion")
        if type(self.contained_interior_point) is not QuantizedPoint:
            raise TypeError("enclosure interior point must be QuantizedPoint")

    def to_data(self) -> dict[str, object]:
        return {
            "kind": "enclosure",
            "container_region": self.container_region.to_data(),
            "contained_region": self.contained_region.to_data(),
            "contained_interior_point": self.contained_interior_point.to_data(),
        }

    @classmethod
    def from_data(cls, value: object) -> "EnclosureWitnessPayload":
        raw = _fields(
            value,
            {"kind", "container_region", "contained_region", "contained_interior_point"},
            "enclosure payload",
        )
        if raw["kind"] != "enclosure":
            raise PanelSoftOntologyError("enclosure witness payload differs")
        result = cls(
            QuantizedRegion.from_data(raw["container_region"]),
            QuantizedRegion.from_data(raw["contained_region"]),
            QuantizedPoint.from_data(raw["contained_interior_point"]),
        )
        _require_canonical(result, raw, "enclosure payload")
        return result


FeatureWitnessPayload: TypeAlias = (
    CountWitnessPayload
    | StraightSegmentCountWitnessPayload
    | ConvexityWitnessPayload
    | MarkerWitnessPayload
    | UnaryGeometryWitnessPayload
    | PointContactWitnessPayload
    | RelationRegionWitnessPayload
    | EnclosureWitnessPayload
)


def _payload_from_data(value: object) -> FeatureWitnessPayload:
    if not isinstance(value, Mapping):
        raise PanelSoftOntologyError("witness payload must be an object")
    kind = value.get("kind")
    parser = {
        "count": CountWitnessPayload,
        "straight_segment_count": StraightSegmentCountWitnessPayload,
        "convexity": ConvexityWitnessPayload,
        "marker_pattern": MarkerWitnessPayload,
        "unary_geometry": UnaryGeometryWitnessPayload,
        "point_contact": PointContactWitnessPayload,
        "relation_region": RelationRegionWitnessPayload,
        "enclosure": EnclosureWitnessPayload,
    }.get(kind)
    if parser is None:
        raise PanelSoftOntologyError("witness payload kind differs")
    return parser.from_data(value)


def _payload_type_for_family(family: FeatureFamily) -> type:
    if family in {FeatureFamily.COMPONENT_COUNT, FeatureFamily.EXACT_SEGMENT_COUNT}:
        return CountWitnessPayload
    if family is FeatureFamily.STRAIGHT_SEGMENT_COUNT:
        return StraightSegmentCountWitnessPayload
    if family is FeatureFamily.CONVEXITY:
        return ConvexityWitnessPayload
    if family is FeatureFamily.MARKER_PATTERN:
        return MarkerWitnessPayload
    if family in _UNARY_WITNESS_FAMILIES:
        return UnaryGeometryWitnessPayload
    if family is FeatureFamily.POINT_CONTACT:
        return PointContactWitnessPayload
    if family in {FeatureFamily.VISIBLE_GAP, FeatureFamily.SHARED_BOUNDARY_ADJACENCY}:
        return RelationRegionWitnessPayload
    return EnclosureWitnessPayload


@dataclass(frozen=True, slots=True)
class PanelFeatureWitness:
    """Positive, panel-local evidence for one exact typed spec and subject."""

    spec: PanelFeatureSpec
    inventory: OwnerInventory
    observer_contract_digest: str
    subject: SubjectBinding
    payload: FeatureWitnessPayload
    witness_receipt_digest: str

    def __post_init__(self) -> None:
        if type(self.spec) is not PanelFeatureSpec or type(self.inventory) is not OwnerInventory:
            raise TypeError("feature witness needs typed spec and inventory")
        _digest(self.observer_contract_digest, "witness observer contract digest")
        if type(self.subject) is not SubjectBinding:
            raise TypeError("feature witness subject must be SubjectBinding")
        self.subject.validate_inventory(self.inventory)
        if self.subject.kind is not self.spec.binding_kind:
            raise PanelSoftOntologyError("witness subject kind differs from feature spec")
        if self.subject not in eligible_subject_bindings(self.spec, self.inventory):
            raise PanelSoftOntologyError("witness subject is not eligible for feature spec")
        expected_payload = _payload_type_for_family(self.spec.family)
        if type(self.payload) is not expected_payload:
            raise PanelSoftOntologyError("feature family has the wrong witness payload")
        if isinstance(self.payload, UnaryGeometryWitnessPayload) and self.payload.family is not self.spec.family:
            raise PanelSoftOntologyError("unary witness family differs from feature spec")
        if isinstance(self.payload, RelationRegionWitnessPayload) and self.payload.family is not self.spec.family:
            raise PanelSoftOntologyError("relation witness family differs from feature spec")
        available = {item.owner_id for item in self.inventory.owners}
        if isinstance(self.payload, CountWitnessPayload):
            if not set(self.payload.counted_owner_ids) <= available:
                raise PanelSoftOntologyError("count witness references an unknown owner")
            expected_count = _closed_count_value(self.spec.parameters.count)  # type: ignore[union-attr]
            if len(self.payload.counted_owner_ids) != expected_count:
                raise PanelSoftOntologyError("count witness does not match feature count")
            if self.inventory.enumeration_complete is not True:
                raise PanelSoftOntologyError("exact count needs a complete inventory")
            if self.spec.family is FeatureFamily.COMPONENT_COUNT:
                expected_owners = coherent_top_level_component_owner_ids(
                    self.inventory
                )
            else:
                expected_owners = segment_owner_ids_for_subject(
                    self.subject, self.inventory
                )
            if self.payload.counted_owner_ids != expected_owners:
                raise PanelSoftOntologyError("exact count does not cover registered membership")
        if isinstance(self.payload, StraightSegmentCountWitnessPayload):
            if self.inventory.enumeration_complete is not True:
                raise PanelSoftOntologyError(
                    "straight-segment count needs a complete owner inventory"
                )
            eligible = segment_owner_ids_for_subject(self.subject, self.inventory)
            if self.payload.eligible_segment_owner_ids != eligible:
                raise PanelSoftOntologyError(
                    "straight-segment classification does not cover exact membership"
                )
            expected_count = _closed_count_value(self.spec.parameters.count)  # type: ignore[union-attr]
            if len(self.payload.straight_segment_owner_ids) != expected_count:
                raise PanelSoftOntologyError(
                    "straight-segment evidence does not match feature count"
                )
            owner_by_id = {item.owner_id: item for item in self.inventory.owners}
            subject_region = subject_search_region(self.subject, self.inventory)
            for owner_id, segment in zip(
                self.payload.straight_segment_owner_ids,
                self.payload.straight_segments,
                strict=True,
            ):
                owner_region = owner_by_id[owner_id].region
                if any(
                    not _point_within_region(point, owner_region)
                    or not _point_within_region(point, subject_region)
                    for point in (segment.start, segment.end)
                ):
                    raise PanelSoftOntologyError(
                        "straight-segment evidence lies outside its owner or subject"
                    )
        if isinstance(self.payload, ConvexityWitnessPayload):
            expected_kind = self.spec.parameters.kind  # type: ignore[union-attr]
            if self.payload.outer_boundary.convexity_kind is not expected_kind:
                raise PanelSoftOntologyError(
                    "Python-derived boundary convexity differs from the feature spec"
                )
            subject_region = subject_search_region(self.subject, self.inventory)
            if any(
                not _point_within_region(point, subject_region)
                for point in self.payload.outer_boundary.vertices
            ):
                raise PanelSoftOntologyError(
                    "convexity boundary lies outside its subject search region"
                )
        if isinstance(self.payload, MarkerWitnessPayload):
            expected_count = _closed_count_value(self.spec.parameters.repetition)  # type: ignore[union-attr]
            if len(self.payload.marker_centers) != expected_count:
                raise PanelSoftOntologyError("marker witness does not match repetition")
            if self.inventory.enumeration_complete is not True:
                raise PanelSoftOntologyError("marker repetition needs a complete inventory")
            parent = self.subject.owner_ids[0]
            expected_markers = tuple(
                item.owner_id
                for item in self.inventory.owners
                if item.kind is OwnerKind.MARKER and parent in item.parent_owner_ids
            )
            if self.payload.marker_owner_ids != expected_markers:
                raise PanelSoftOntologyError("marker witness does not cover registered membership")
            owner_by_id = {item.owner_id: item for item in self.inventory.owners}
            if any(
                not _point_within_region(center, owner_by_id[owner].region)
                for owner, center in zip(
                    self.payload.marker_owner_ids, self.payload.marker_centers
                )
            ):
                raise PanelSoftOntologyError("marker center is outside its owner region")
        if isinstance(self.payload, UnaryGeometryWitnessPayload):
            parameters = self.spec.parameters
            aggregation = getattr(parameters, "aggregation", None)
            if aggregation is ClosedAggregation.ALL_ELIGIBLE:
                if self.payload.coverage is not WitnessCoverage.COMPLETE_ELIGIBLE:
                    raise PanelSoftOntologyError("all-eligible feature needs complete witness coverage")
            elif aggregation is ClosedAggregation.AT_LEAST_TWO:
                if len(self.payload.sample_points) < 2:
                    raise PanelSoftOntologyError("at-least-two feature needs two witness samples")
            subject_region = subject_search_region(self.subject, self.inventory)
            if not _region_within_region(self.payload.primary_region, subject_region):
                raise PanelSoftOntologyError("unary witness region is outside its subject")
            if any(
                not _point_within_region(point, self.payload.primary_region)
                for point in self.payload.sample_points
            ):
                raise PanelSoftOntologyError("unary witness sample is outside its region")
        if isinstance(self.payload, PointContactWitnessPayload):
            if self.payload.observed_kind is not self.spec.parameters.kind:  # type: ignore[union-attr]
                raise PanelSoftOntologyError("point-contact witness kind differs from spec")
            subjects = set(self.subject.owner_ids)
            ray_counts = {owner: 0 for owner in subjects}
            for ray in self.payload.owner_rays:
                if ray.owner_id not in ray_counts:
                    raise PanelSoftOntologyError("point-contact ray has the wrong owner")
                ray_counts[ray.owner_id] += 1
            if set(ray_counts.values()) != {2}:
                raise PanelSoftOntologyError("point contact requires two rays per owner")
            if self.payload.observed_kind is PointContactKind.TANGENTIAL:
                direction_sets = {
                    owner: frozenset(
                        ray.direction
                        for ray in self.payload.owner_rays
                        if ray.owner_id == owner
                    )
                    for owner in subjects
                }
                opposite_axes = {
                    frozenset({RayDirection.N, RayDirection.S}),
                    frozenset({RayDirection.NE, RayDirection.SW}),
                    frozenset({RayDirection.E, RayDirection.W}),
                    frozenset({RayDirection.SE, RayDirection.NW}),
                }
                if (
                    set(direction_sets.values()).issubset(opposite_axes) is False
                    or len(set(direction_sets.values())) != 1
                ):
                    raise PanelSoftOntologyError(
                        "tangential contact rays must share one opposite direction axis"
                    )
            owner_by_id = {item.owner_id: item for item in self.inventory.owners}
            point = self.payload.contact_point
            if any(
                not _point_within_region(point, owner_by_id[owner].region)
                for owner in subjects
            ):
                raise PanelSoftOntologyError(
                    "point-contact location is outside one of its owner regions"
                )
            subject_region = subject_search_region(self.subject, self.inventory)
            if any(
                not _region_within_region(gap, subject_region)
                or _point_within_region(point, gap)
                for gap in self.payload.exterior_gap_regions
            ):
                raise PanelSoftOntologyError(
                    "point-contact exterior gap has the wrong locality"
                )
        if isinstance(self.payload, RelationRegionWitnessPayload):
            if not _region_within_region(
                self.payload.relation_region,
                subject_search_region(self.subject, self.inventory),
            ):
                raise PanelSoftOntologyError("relation witness region is outside its subject")
        if isinstance(self.payload, EnclosureWitnessPayload):
            owner_by_id = {item.owner_id: item for item in self.inventory.owners}
            container_owner, contained_owner = self.subject.owner_ids
            if not _region_within_region(
                self.payload.container_region, owner_by_id[container_owner].region
            ) or not _region_within_region(
                self.payload.contained_region, owner_by_id[contained_owner].region
            ):
                raise PanelSoftOntologyError("enclosure witness regions have wrong owners")
            if not _region_within_region(
                self.payload.contained_region, self.payload.container_region
            ) or not all(
                _point_within_region(
                    self.payload.contained_interior_point,
                    region,
                )
                for region in (
                    self.payload.container_region,
                    self.payload.contained_region,
                )
            ):
                raise PanelSoftOntologyError("enclosure containment geometry differs")
        _digest(self.witness_receipt_digest, "witness receipt digest")

    @property
    def panel_digest(self) -> str:
        return self.inventory.panel_digest

    @property
    def witness_digest(self) -> str:
        return canonical_digest(self.to_data())

    def to_data(self) -> dict[str, object]:
        return {
            "schema": FEATURE_WITNESS_SCHEMA,
            "spec": self.spec.to_data(),
            "inventory": self.inventory.to_data(),
            "observer_contract_digest": self.observer_contract_digest,
            "subject": self.subject.to_data(),
            "payload": self.payload.to_data(),
            "witness_receipt_digest": self.witness_receipt_digest,
        }

    @classmethod
    def from_data(cls, value: object) -> "PanelFeatureWitness":
        raw = _fields(
            value,
            {
                "schema",
                "spec",
                "inventory",
                "observer_contract_digest",
                "subject",
                "payload",
                "witness_receipt_digest",
            },
            "feature witness",
        )
        if raw["schema"] != FEATURE_WITNESS_SCHEMA:
            raise PanelSoftOntologyError("feature witness schema differs")
        result = cls(
            PanelFeatureSpec.from_data(raw["spec"]),
            OwnerInventory.from_data(raw["inventory"]),
            raw["observer_contract_digest"],
            SubjectBinding.from_data(raw["subject"]),
            _payload_from_data(raw["payload"]),
            raw["witness_receipt_digest"],
        )
        _require_canonical(result, raw, "feature witness")
        return result


@dataclass(frozen=True, order=True, slots=True)
class SiblingRelationMetadata:
    relation_id: str
    mutually_exclusive: bool
    exhaustive: bool
    same_subject_only: bool

    def __post_init__(self) -> None:
        _code(self.relation_id, "sibling relation ID")
        if any(type(item) is not bool for item in (self.mutually_exclusive, self.exhaustive, self.same_subject_only)):
            raise TypeError("sibling relation flags must be exact bools")
        if not self.mutually_exclusive or not self.same_subject_only:
            raise PanelSoftOntologyError("registered sibling conflicts must be local and exclusive")


def registered_sibling_relation(
    left: PanelFeatureSpec, right: PanelFeatureSpec
) -> SiblingRelationMetadata | None:
    """Look up direct conflict metadata; never generate or negate a sibling."""

    if type(left) is not PanelFeatureSpec or type(right) is not PanelFeatureSpec:
        raise TypeError("sibling lookup requires PanelFeatureSpec values")
    if left.subject_scope is not right.subject_scope or left.reference_frame is not right.reference_frame:
        return None
    pair = frozenset({left.family, right.family})
    if left.family is right.family and left.family in {
        FeatureFamily.COMPONENT_COUNT,
        FeatureFamily.EXACT_SEGMENT_COUNT,
        FeatureFamily.STRAIGHT_SEGMENT_COUNT,
        FeatureFamily.MARKER_PATTERN,
    }:
        left_count = getattr(left.parameters, "count", None) or left.parameters.repetition  # type: ignore[union-attr]
        right_count = getattr(right.parameters, "count", None) or right.parameters.repetition  # type: ignore[union-attr]
        marker_context_matches = True
        if left.family is FeatureFamily.MARKER_PATTERN:
            marker_context_matches = (
                left.parameters.primitive is right.parameters.primitive  # type: ignore[union-attr]
                and left.parameters.arrangement is right.parameters.arrangement  # type: ignore[union-attr]
            )
        if left_count is not right_count and marker_context_matches:
            return SiblingRelationMetadata("distinct-exact-counts-v1", True, False, True)
    if (
        left.family is right.family is FeatureFamily.CONVEXITY
        and left.parameters.kind is not right.parameters.kind  # type: ignore[union-attr]
    ):
        return SiblingRelationMetadata(
            "convex-vs-concave-closed-boundary-v1", True, True, True
        )
    return None


class RejectionKind(str, Enum):
    EXHAUSTIVE_SEARCH_NONMATCH = "exhaustive_search_nonmatch"
    REGISTERED_SIBLING_CONFLICT = "registered_sibling_conflict"


@dataclass(frozen=True, slots=True)
class ExhaustiveSearchNonmatchEvidence:
    """Typed resolved search cells; unresolved/error cells are unrepresentable."""

    searched_regions: tuple[QuantizedRegion, ...]
    search_attempt_receipt_digest: str

    def __post_init__(self) -> None:
        if (
            type(self.searched_regions) is not tuple
            or not self.searched_regions
            or any(type(item) is not QuantizedRegion for item in self.searched_regions)
        ):
            raise PanelSoftOntologyError("search nonmatch needs resolved typed regions")
        if len(set(self.searched_regions)) != len(self.searched_regions):
            raise PanelSoftOntologyError("search nonmatch regions must be unique")
        if self.searched_regions != tuple(sorted(self.searched_regions)):
            raise PanelSoftOntologyError("search nonmatch regions must be sorted")
        _digest(self.search_attempt_receipt_digest, "search attempt receipt digest")

    def to_data(self) -> dict[str, object]:
        return {
            "kind": RejectionKind.EXHAUSTIVE_SEARCH_NONMATCH.value,
            "searched_regions": [item.to_data() for item in self.searched_regions],
            "search_attempt_receipt_digest": self.search_attempt_receipt_digest,
        }

    @classmethod
    def from_data(cls, value: object) -> "ExhaustiveSearchNonmatchEvidence":
        raw = _fields(
            value,
            {"kind", "searched_regions", "search_attempt_receipt_digest"},
            "search nonmatch evidence",
        )
        if (
            raw["kind"] != RejectionKind.EXHAUSTIVE_SEARCH_NONMATCH.value
            or type(raw["searched_regions"]) is not list
        ):
            raise PanelSoftOntologyError("search nonmatch evidence differs")
        result = cls(
            tuple(QuantizedRegion.from_data(item) for item in raw["searched_regions"]),
            raw["search_attempt_receipt_digest"],
        )
        _require_canonical(result, raw, "search nonmatch evidence")
        return result


@dataclass(frozen=True, slots=True)
class RegisteredSiblingConflictEvidence:
    """Included positive sibling witness for the exact same local subject."""

    relation_id: str
    sibling_witness: PanelFeatureWitness

    def __post_init__(self) -> None:
        _code(self.relation_id, "registered sibling relation ID")
        if type(self.sibling_witness) is not PanelFeatureWitness:
            raise TypeError("sibling conflict needs an included feature witness")

    def to_data(self) -> dict[str, object]:
        return {
            "kind": RejectionKind.REGISTERED_SIBLING_CONFLICT.value,
            "relation_id": self.relation_id,
            "sibling_witness": self.sibling_witness.to_data(),
        }

    @classmethod
    def from_data(cls, value: object) -> "RegisteredSiblingConflictEvidence":
        raw = _fields(
            value,
            {"kind", "relation_id", "sibling_witness"},
            "registered conflict evidence",
        )
        if raw["kind"] != RejectionKind.REGISTERED_SIBLING_CONFLICT.value:
            raise PanelSoftOntologyError("registered conflict evidence differs")
        result = cls(
            raw["relation_id"],
            PanelFeatureWitness.from_data(raw["sibling_witness"]),
        )
        _require_canonical(result, raw, "registered conflict evidence")
        return result


RejectionEvidence: TypeAlias = (
    ExhaustiveSearchNonmatchEvidence | RegisteredSiblingConflictEvidence
)


def _rejection_evidence_from_data(value: object) -> RejectionEvidence:
    if not isinstance(value, Mapping):
        raise PanelSoftOntologyError("rejection evidence must be an object")
    if value.get("kind") == RejectionKind.EXHAUSTIVE_SEARCH_NONMATCH.value:
        return ExhaustiveSearchNonmatchEvidence.from_data(value)
    if value.get("kind") == RejectionKind.REGISTERED_SIBLING_CONFLICT.value:
        return RegisteredSiblingConflictEvidence.from_data(value)
    raise PanelSoftOntologyError("rejection evidence kind differs")


@dataclass(frozen=True, slots=True)
class OwnerRejection:
    """Counterevidence for one exact eligible subject binding.

    The historical name is retained, but relation features bind an exact pair
    and whole-panel features bind the singleton panel subject.
    """

    target_spec: PanelFeatureSpec
    inventory: OwnerInventory
    observer_contract_digest: str
    subject: SubjectBinding
    evidence: RejectionEvidence

    def __post_init__(self) -> None:
        if type(self.target_spec) is not PanelFeatureSpec or type(self.inventory) is not OwnerInventory:
            raise TypeError("owner rejection needs typed target spec and inventory")
        _digest(self.observer_contract_digest, "rejection observer contract digest")
        if type(self.subject) is not SubjectBinding:
            raise TypeError("owner rejection subject must be SubjectBinding")
        self.subject.validate_inventory(self.inventory)
        if self.subject.kind is not self.target_spec.binding_kind:
            raise PanelSoftOntologyError("rejection subject kind differs from target spec")
        if self.subject not in eligible_subject_bindings(self.target_spec, self.inventory):
            raise PanelSoftOntologyError("rejection subject is not eligible for target spec")
        if type(self.evidence) not in {
            ExhaustiveSearchNonmatchEvidence,
            RegisteredSiblingConflictEvidence,
        }:
            raise TypeError("owner rejection has untyped evidence")
        if isinstance(self.evidence, ExhaustiveSearchNonmatchEvidence):
            required_region = subject_search_region(self.subject, self.inventory)
            if self.evidence.searched_regions != (required_region,):
                raise PanelSoftOntologyError(
                    "search nonmatch does not exactly cover the derived subject region"
                )
        if isinstance(self.evidence, RegisteredSiblingConflictEvidence):
            witness = self.evidence.sibling_witness
            relation = registered_sibling_relation(self.target_spec, witness.spec)
            if relation is None or relation.relation_id != self.evidence.relation_id:
                raise PanelSoftOntologyError("sibling conflict is not statically registered")
            if not relation.mutually_exclusive or relation.exhaustive:
                # Exhaustiveness is deliberately not used even if a future
                # registry adds it: direct presence is local counterevidence.
                raise PanelSoftOntologyError("sibling conflict metadata differs")
            if (
                witness.inventory.inventory_digest != self.inventory.inventory_digest
                or witness.panel_digest != self.inventory.panel_digest
                or witness.observer_contract_digest != self.observer_contract_digest
                or witness.subject != self.subject
            ):
                raise PanelSoftOntologyError("sibling conflict witness has different locality")

    @property
    def rejection_kind(self) -> RejectionKind:
        if isinstance(self.evidence, RegisteredSiblingConflictEvidence):
            return RejectionKind.REGISTERED_SIBLING_CONFLICT
        return RejectionKind.EXHAUSTIVE_SEARCH_NONMATCH

    @property
    def rejection_digest(self) -> str:
        return canonical_digest(self.to_data())

    def to_data(self) -> dict[str, object]:
        return {
            "schema": OWNER_REJECTION_SCHEMA,
            "target_spec": self.target_spec.to_data(),
            "inventory": self.inventory.to_data(),
            "observer_contract_digest": self.observer_contract_digest,
            "subject": self.subject.to_data(),
            "evidence": self.evidence.to_data(),
        }

    @classmethod
    def from_data(cls, value: object) -> "OwnerRejection":
        raw = _fields(
            value,
            {"schema", "target_spec", "inventory", "observer_contract_digest", "subject", "evidence"},
            "owner rejection",
        )
        if raw["schema"] != OWNER_REJECTION_SCHEMA:
            raise PanelSoftOntologyError("owner rejection schema differs")
        result = cls(
            PanelFeatureSpec.from_data(raw["target_spec"]),
            OwnerInventory.from_data(raw["inventory"]),
            raw["observer_contract_digest"],
            SubjectBinding.from_data(raw["subject"]),
            _rejection_evidence_from_data(raw["evidence"]),
        )
        _require_canonical(result, raw, "owner rejection")
        return result


@dataclass(frozen=True, slots=True)
class EmptyEligibleDomainCertificate:
    """Independent enumeration evidence required when projection is empty."""

    inventory_digest: str
    search_domain_digest: str
    enumeration_receipt_digest: str
    empty_domain_verifier_receipt_digest: str

    def __post_init__(self) -> None:
        _digest(self.inventory_digest, "empty-domain inventory digest")
        _digest(self.search_domain_digest, "empty-domain search digest")
        _digest(self.enumeration_receipt_digest, "empty-domain enumeration receipt")
        _digest(self.empty_domain_verifier_receipt_digest, "empty-domain verifier receipt")

    def to_data(self) -> dict[str, object]:
        return {
            "schema": EMPTY_DOMAIN_CERTIFICATE_SCHEMA,
            "inventory_digest": self.inventory_digest,
            "search_domain_digest": self.search_domain_digest,
            "enumeration_receipt_digest": self.enumeration_receipt_digest,
            "empty_domain_verifier_receipt_digest": self.empty_domain_verifier_receipt_digest,
        }

    @classmethod
    def from_data(cls, value: object) -> "EmptyEligibleDomainCertificate":
        raw = _fields(
            value,
            {
                "schema",
                "inventory_digest",
                "search_domain_digest",
                "enumeration_receipt_digest",
                "empty_domain_verifier_receipt_digest",
            },
            "empty-domain certificate",
        )
        if raw["schema"] != EMPTY_DOMAIN_CERTIFICATE_SCHEMA:
            raise PanelSoftOntologyError("empty-domain certificate schema differs")
        result = cls(
            raw["inventory_digest"],
            raw["search_domain_digest"],
            raw["enumeration_receipt_digest"],
            raw["empty_domain_verifier_receipt_digest"],
        )
        _require_canonical(result, raw, "empty-domain certificate")
        return result


@dataclass(frozen=True, slots=True)
class AbsenceCertificate:
    """Exact exhaustive coverage of every deterministic eligible subject."""

    target_spec: PanelFeatureSpec
    inventory: OwnerInventory
    observer_contract_digest: str
    search_domain: SearchResolutionDomain
    search_protocol_digest: str
    eligible_subjects: tuple[SubjectBinding, ...]
    rejections: tuple[OwnerRejection, ...]
    search_complete: bool
    search_receipt_digest: str
    empty_domain_certificate: EmptyEligibleDomainCertificate | None = None

    def __post_init__(self) -> None:
        if type(self.target_spec) is not PanelFeatureSpec or type(self.inventory) is not OwnerInventory:
            raise TypeError("absence certificate needs typed spec and inventory")
        _digest(self.observer_contract_digest, "absence observer contract digest")
        if type(self.search_domain) is not SearchResolutionDomain:
            raise TypeError("absence search domain has the wrong type")
        self.search_domain.validate_spec(self.target_spec)
        _digest(self.search_protocol_digest, "absence search protocol digest")
        if type(self.search_complete) is not bool or self.search_complete is not True:
            raise PanelSoftOntologyError("absence requires exact search_complete=True")
        if self.inventory.enumeration_complete is not True:
            raise PanelSoftOntologyError("absence requires a complete owner inventory")
        _digest(self.search_receipt_digest, "absence search receipt digest")
        if type(self.eligible_subjects) is not tuple or any(
            type(item) is not SubjectBinding for item in self.eligible_subjects
        ):
            raise TypeError("eligible subjects must be a SubjectBinding tuple")
        expected = eligible_subject_bindings(
            self.target_spec, self.inventory, self.search_domain
        )
        if self.eligible_subjects != expected:
            raise PanelSoftOntologyError("absence eligible subjects are not exact")
        if type(self.rejections) is not tuple or any(
            type(item) is not OwnerRejection for item in self.rejections
        ):
            raise TypeError("absence rejections must be an OwnerRejection tuple")
        if tuple(item.subject for item in self.rejections) != self.eligible_subjects:
            raise PanelSoftOntologyError("absence rejections do not exactly cover subjects")
        for item in self.rejections:
            if (
                item.target_spec != self.target_spec
                or item.inventory.inventory_digest != self.inventory.inventory_digest
                or item.observer_contract_digest != self.observer_contract_digest
            ):
                raise PanelSoftOntologyError("absence rejection has different custody")
        if expected:
            if not self.rejections or self.empty_domain_certificate is not None:
                raise PanelSoftOntologyError("nonempty absence domain has invalid empty evidence")
        else:
            if self.rejections or type(self.empty_domain_certificate) is not EmptyEligibleDomainCertificate:
                raise PanelSoftOntologyError("empty absence domain needs independent evidence")
            empty = self.empty_domain_certificate
            if (
                empty.inventory_digest != self.inventory.inventory_digest
                or empty.search_domain_digest != self.search_domain.domain_digest
                or empty.enumeration_receipt_digest != self.inventory.enumeration_receipt_digest
            ):
                raise PanelSoftOntologyError("empty-domain evidence has different custody")

    @property
    def absence_digest(self) -> str:
        return canonical_digest(self.to_data())

    @property
    def rejection_kinds(self) -> tuple[RejectionKind, ...]:
        return tuple(sorted({item.rejection_kind for item in self.rejections}, key=lambda x: x.value))

    def to_data(self) -> dict[str, object]:
        return {
            "schema": ABSENCE_CERTIFICATE_SCHEMA,
            "target_spec": self.target_spec.to_data(),
            "inventory": self.inventory.to_data(),
            "observer_contract_digest": self.observer_contract_digest,
            "search_domain": self.search_domain.to_data(),
            "search_protocol_digest": self.search_protocol_digest,
            "eligible_subjects": [item.to_data() for item in self.eligible_subjects],
            "rejections": [item.to_data() for item in self.rejections],
            "search_complete": self.search_complete,
            "search_receipt_digest": self.search_receipt_digest,
            "empty_domain_certificate": (
                None if self.empty_domain_certificate is None else self.empty_domain_certificate.to_data()
            ),
        }

    @classmethod
    def from_data(cls, value: object) -> "AbsenceCertificate":
        raw = _fields(
            value,
            {
                "schema",
                "target_spec",
                "inventory",
                "observer_contract_digest",
                "search_domain",
                "search_protocol_digest",
                "eligible_subjects",
                "rejections",
                "search_complete",
                "search_receipt_digest",
                "empty_domain_certificate",
            },
            "absence certificate",
        )
        if raw["schema"] != ABSENCE_CERTIFICATE_SCHEMA:
            raise PanelSoftOntologyError("absence certificate schema differs")
        if type(raw["eligible_subjects"]) is not list or type(raw["rejections"]) is not list:
            raise PanelSoftOntologyError("absence coverage must use JSON lists")
        raw_empty = raw["empty_domain_certificate"]
        result = cls(
            PanelFeatureSpec.from_data(raw["target_spec"]),
            OwnerInventory.from_data(raw["inventory"]),
            raw["observer_contract_digest"],
            SearchResolutionDomain.from_data(raw["search_domain"]),
            raw["search_protocol_digest"],
            tuple(SubjectBinding.from_data(item) for item in raw["eligible_subjects"]),
            tuple(OwnerRejection.from_data(item) for item in raw["rejections"]),
            raw["search_complete"],
            raw["search_receipt_digest"],
            None if raw_empty is None else EmptyEligibleDomainCertificate.from_data(raw_empty),
        )
        _require_canonical(result, raw, "absence certificate")
        return result


class RawMeasurementState(str, Enum):
    WITNESS_ASSERTED = "witness_asserted"
    EXHAUSTIVE_SEARCH_NEGATIVE = "exhaustive_search_negative"
    REGISTERED_SIBLING_CONFLICT = "registered_sibling_conflict"
    UNRESOLVED = "unresolved"
    ERROR = "error"


class MeasurementIssueCode(str, Enum):
    AMBIGUOUS_OWNER = "ambiguous_owner"
    INSUFFICIENT_RESOLUTION = "insufficient_resolution"
    MISSING_STRAIGHTNESS_EVIDENCE = "missing_straightness_evidence"
    MISSING_BOUNDARY_EVIDENCE = "missing_boundary_evidence"
    INVALID_BOUNDARY_GEOMETRY = "invalid_boundary_geometry"
    SEARCH_INCOMPLETE = "search_incomplete"
    OBSERVER_FAILURE = "observer_failure"
    PARSER_FAILURE = "parser_failure"


@dataclass(frozen=True, slots=True)
class RawFeatureMeasurement:
    spec: PanelFeatureSpec
    inventory: OwnerInventory
    observer_contract_digest: str
    measurement_protocol_digest: str
    state: RawMeasurementState
    witness: PanelFeatureWitness | None = None
    absence: AbsenceCertificate | None = None
    local_conflict: OwnerRejection | None = None
    issue_code: MeasurementIssueCode | None = None

    def __post_init__(self) -> None:
        if type(self.spec) is not PanelFeatureSpec or type(self.inventory) is not OwnerInventory:
            raise TypeError("raw measurement needs typed spec and inventory")
        _digest(self.observer_contract_digest, "measurement observer contract digest")
        _digest(self.measurement_protocol_digest, "measurement protocol digest")
        _enum_instance(self.state, RawMeasurementState, "raw measurement state")
        expected = {
            RawMeasurementState.WITNESS_ASSERTED: (PanelFeatureWitness, None, None, None),
            RawMeasurementState.EXHAUSTIVE_SEARCH_NEGATIVE: (None, AbsenceCertificate, None, None),
            RawMeasurementState.REGISTERED_SIBLING_CONFLICT: (None, None, OwnerRejection, None),
            RawMeasurementState.UNRESOLVED: (None, None, None, MeasurementIssueCode),
            RawMeasurementState.ERROR: (None, None, None, MeasurementIssueCode),
        }[self.state]
        actual = (self.witness, self.absence, self.local_conflict, self.issue_code)
        for item, wanted in zip(actual, expected):
            if wanted is None and item is not None:
                raise PanelSoftOntologyError("raw measurement carries incompatible evidence")
            if isinstance(wanted, type) and type(item) is not wanted:
                raise PanelSoftOntologyError("raw measurement is missing typed evidence")
        evidence = self.witness or self.absence or self.local_conflict
        if evidence is not None:
            target_spec = evidence.spec if isinstance(evidence, PanelFeatureWitness) else evidence.target_spec
            evidence_inventory = evidence.inventory
            evidence_observer = evidence.observer_contract_digest
            if (
                target_spec != self.spec
                or evidence_inventory.inventory_digest != self.inventory.inventory_digest
                or evidence_observer != self.observer_contract_digest
            ):
                raise PanelSoftOntologyError("raw measurement evidence has different custody")
        if self.state is RawMeasurementState.REGISTERED_SIBLING_CONFLICT:
            if self.local_conflict.rejection_kind is not RejectionKind.REGISTERED_SIBLING_CONFLICT:  # type: ignore[union-attr]
                raise PanelSoftOntologyError("local conflict measurement has wrong rejection kind")
        if self.state is RawMeasurementState.UNRESOLVED and self.issue_code not in {
            MeasurementIssueCode.AMBIGUOUS_OWNER,
            MeasurementIssueCode.INSUFFICIENT_RESOLUTION,
            MeasurementIssueCode.MISSING_STRAIGHTNESS_EVIDENCE,
            MeasurementIssueCode.MISSING_BOUNDARY_EVIDENCE,
            MeasurementIssueCode.INVALID_BOUNDARY_GEOMETRY,
            MeasurementIssueCode.SEARCH_INCOMPLETE,
        }:
            raise PanelSoftOntologyError("unresolved measurement has an error-only issue code")
        if self.state is RawMeasurementState.ERROR and self.issue_code not in {
            MeasurementIssueCode.OBSERVER_FAILURE,
            MeasurementIssueCode.PARSER_FAILURE,
        }:
            raise PanelSoftOntologyError("error measurement has an unresolved-only issue code")

    @property
    def measurement_digest(self) -> str:
        return canonical_digest(self.to_data())

    def to_data(self) -> dict[str, object]:
        return {
            "schema": RAW_MEASUREMENT_SCHEMA,
            "spec": self.spec.to_data(),
            "inventory": self.inventory.to_data(),
            "observer_contract_digest": self.observer_contract_digest,
            "measurement_protocol_digest": self.measurement_protocol_digest,
            "state": self.state.value,
            "witness": None if self.witness is None else self.witness.to_data(),
            "absence": None if self.absence is None else self.absence.to_data(),
            "local_conflict": None if self.local_conflict is None else self.local_conflict.to_data(),
            "issue_code": None if self.issue_code is None else self.issue_code.value,
        }

    @classmethod
    def from_data(cls, value: object) -> "RawFeatureMeasurement":
        raw = _fields(
            value,
            {"schema", "spec", "inventory", "observer_contract_digest", "measurement_protocol_digest", "state", "witness", "absence", "local_conflict", "issue_code"},
            "raw measurement",
        )
        if raw["schema"] != RAW_MEASUREMENT_SCHEMA:
            raise PanelSoftOntologyError("raw measurement schema differs")
        try:
            result = cls(
                PanelFeatureSpec.from_data(raw["spec"]),
                OwnerInventory.from_data(raw["inventory"]),
                raw["observer_contract_digest"],
                raw["measurement_protocol_digest"],
                RawMeasurementState(raw["state"]),
                None if raw["witness"] is None else PanelFeatureWitness.from_data(raw["witness"]),
                None if raw["absence"] is None else AbsenceCertificate.from_data(raw["absence"]),
                None if raw["local_conflict"] is None else OwnerRejection.from_data(raw["local_conflict"]),
                None if raw["issue_code"] is None else MeasurementIssueCode(raw["issue_code"]),
            )
        except (TypeError, ValueError) as exc:
            if isinstance(exc, PanelSoftOntologyError):
                raise
            raise PanelSoftOntologyError("raw measurement value differs") from exc
        _require_canonical(result, raw, "raw measurement")
        return result


class LanguageGapKind(str, Enum):
    UNREGISTERED_GESTALT = "unregistered_gestalt"
    AMBIGUOUS_FAMILY = "ambiguous_family"
    UNDECLARED_REFERENCE_FRAME = "undeclared_reference_frame"
    UNSUPPORTED_RELATION_VARIANT = "unsupported_relation_variant"
    UNRESOLVED_OWNER_ARITY = "unresolved_owner_arity"
    UNSUPPORTED_QUANTIFIER = "unsupported_quantifier"
    NONLOCAL_COMPARATIVE = "nonlocal_comparative"
    COMPLEMENT_DERIVATION_REQUESTED = "complement_derivation_requested"
    MISSING_WITNESS_SCHEMA = "missing_witness_schema"


@dataclass(frozen=True, order=True, slots=True)
class LanguageGapArtifact:
    kind: LanguageGapKind
    source_narration_digest: str
    proposal_receipt_digest: str
    context_digest: str
    partial_spec_digest: str | None = None

    def __post_init__(self) -> None:
        _enum_instance(self.kind, LanguageGapKind, "language gap kind")
        _digest(self.source_narration_digest, "gap narration digest")
        _digest(self.proposal_receipt_digest, "gap proposal receipt digest")
        _digest(self.context_digest, "gap context digest")
        if self.partial_spec_digest is not None:
            _digest(self.partial_spec_digest, "gap partial spec digest")

    @property
    def gap_digest(self) -> str:
        return canonical_digest(self.to_data())

    def to_data(self) -> dict[str, object]:
        return {
            "schema": LANGUAGE_GAP_SCHEMA,
            "catalog_digest": feature_catalog_digest(),
            "kind": self.kind.value,
            "source_narration_digest": self.source_narration_digest,
            "proposal_receipt_digest": self.proposal_receipt_digest,
            "context_digest": self.context_digest,
            "partial_spec_digest": self.partial_spec_digest,
        }

    @classmethod
    def from_data(cls, value: object) -> "LanguageGapArtifact":
        raw = _fields(
            value,
            {
                "schema",
                "catalog_digest",
                "kind",
                "source_narration_digest",
                "proposal_receipt_digest",
                "context_digest",
                "partial_spec_digest",
            },
            "language gap",
        )
        if raw["schema"] != LANGUAGE_GAP_SCHEMA or raw["catalog_digest"] != feature_catalog_digest():
            raise PanelSoftOntologyError("language gap catalog/schema differs")
        try:
            result = cls(
                LanguageGapKind(raw["kind"]),
                raw["source_narration_digest"],
                raw["proposal_receipt_digest"],
                raw["context_digest"],
                raw["partial_spec_digest"],
            )
        except (TypeError, ValueError) as exc:
            if isinstance(exc, PanelSoftOntologyError):
                raise
            raise PanelSoftOntologyError("language gap value differs") from exc
        _require_canonical(result, raw, "language gap")
        return result


class CalibrationCapability(str, Enum):
    PRESENCE = "presence"
    ABSENCE = "absence"


@dataclass(frozen=True, order=True, slots=True)
class FeatureDomain:
    """Closed calibration domain; parameters remain exact per feature spec."""

    family: FeatureFamily
    subject_scope: SubjectScope
    reference_frame: ReferenceFrame
    admitted_specs: tuple[PanelFeatureSpec, ...]

    def __post_init__(self) -> None:
        _enum_instance(self.family, FeatureFamily, "feature-domain family")
        _enum_instance(self.subject_scope, SubjectScope, "feature-domain scope")
        _enum_instance(self.reference_frame, ReferenceFrame, "feature-domain frame")
        if (
            self.subject_scope,
            self.reference_frame,
        ) not in FAMILY_CONTRACTS[self.family].allowed_scope_frames:
            raise PanelSoftOntologyError("feature calibration domain is not registered")
        if type(self.admitted_specs) is not tuple or not self.admitted_specs or any(
            type(item) is not PanelFeatureSpec for item in self.admitted_specs
        ):
            raise TypeError("feature domain needs a non-empty spec tuple")
        if any(
            item.family is not self.family
            or item.subject_scope is not self.subject_scope
            or item.reference_frame is not self.reference_frame
            for item in self.admitted_specs
        ):
            raise PanelSoftOntologyError("feature domain contains a spec outside its family/scope/frame")
        digests = tuple(item.spec_digest for item in self.admitted_specs)
        if digests != tuple(sorted(digests)) or len(digests) != len(set(digests)):
            raise PanelSoftOntologyError("feature domain specs must be unique and digest-sorted")

    @property
    def domain_digest(self) -> str:
        return canonical_digest(self.to_data())

    def contains(self, spec: PanelFeatureSpec) -> bool:
        return (
            type(spec) is PanelFeatureSpec
            and spec.family is self.family
            and spec.subject_scope is self.subject_scope
            and spec.reference_frame is self.reference_frame
            and spec.spec_digest in {item.spec_digest for item in self.admitted_specs}
        )

    def to_data(self) -> dict[str, object]:
        return {
            "schema": FEATURE_DOMAIN_SCHEMA,
            "catalog_digest": feature_catalog_digest(),
            "family": self.family.value,
            "subject_scope": self.subject_scope.value,
            "reference_frame": self.reference_frame.value,
            "admitted_specs": [item.to_data() for item in self.admitted_specs],
        }

    @classmethod
    def from_data(cls, value: object) -> "FeatureDomain":
        raw = _fields(
            value,
            {"schema", "catalog_digest", "family", "subject_scope", "reference_frame", "admitted_specs"},
            "feature domain",
        )
        if raw["schema"] != FEATURE_DOMAIN_SCHEMA or raw["catalog_digest"] != feature_catalog_digest():
            raise PanelSoftOntologyError("feature domain catalog/schema differs")
        if type(raw["admitted_specs"]) is not list:
            raise PanelSoftOntologyError("feature domain specs must be a JSON list")
        try:
            result = cls(
                FeatureFamily(raw["family"]),
                SubjectScope(raw["subject_scope"]),
                ReferenceFrame(raw["reference_frame"]),
                tuple(PanelFeatureSpec.from_data(item) for item in raw["admitted_specs"]),
            )
        except (TypeError, ValueError) as exc:
            if isinstance(exc, PanelSoftOntologyError):
                raise
            raise PanelSoftOntologyError("feature domain value differs") from exc
        _require_canonical(result, raw, "feature domain")
        return result


class CalibrationRisk(str, Enum):
    FALSE_POSITIVE_CLAIM = "false_positive_claim"
    FALSE_NEGATIVE_CLAIM = "false_negative_claim"
    OWNER_INVENTORY_OMISSION = "owner_inventory_omission"


@dataclass(frozen=True, order=True, slots=True)
class CalibrationAssessment:
    """Closed auditable risk bound; no bool/digest-only enable switch."""

    risk: CalibrationRisk
    calibration_population_digest: str
    annotation_protocol_digest: str
    sample_count: int
    accepted_error_upper_ppm: int
    assessed_error_upper_ppm: int
    confidence_ppm: int
    valid_from_unix: int
    valid_through_unix: int
    assessment_receipt_digest: str

    def __post_init__(self) -> None:
        _enum_instance(self.risk, CalibrationRisk, "calibration risk")
        _digest(self.calibration_population_digest, "calibration population digest")
        _digest(self.annotation_protocol_digest, "annotation protocol digest")
        _digest(self.assessment_receipt_digest, "assessment receipt digest")
        if type(self.sample_count) is not int or self.sample_count <= 0:
            raise PanelSoftOntologyError("calibration sample_count must be a positive exact int")
        for label, item in (
            ("accepted error", self.accepted_error_upper_ppm),
            ("assessed error", self.assessed_error_upper_ppm),
            ("confidence", self.confidence_ppm),
        ):
            if type(item) is not int or not 0 <= item <= 1_000_000:
                raise PanelSoftOntologyError(f"{label} ppm must be an exact integer in [0, 1000000]")
        if self.confidence_ppm == 0:
            raise PanelSoftOntologyError("calibration confidence must be nonzero")
        if self.assessed_error_upper_ppm > self.accepted_error_upper_ppm:
            raise PanelSoftOntologyError("assessed error exceeds the preregistered bound")
        if (
            type(self.valid_from_unix) is not int
            or type(self.valid_through_unix) is not int
            or self.valid_from_unix < 0
            or self.valid_through_unix < self.valid_from_unix
        ):
            raise PanelSoftOntologyError("calibration validity interval differs")

    @property
    def assessment_digest(self) -> str:
        return canonical_digest(self.to_data())

    def to_data(self) -> dict[str, object]:
        return {
            "schema": CALIBRATION_ASSESSMENT_SCHEMA,
            "risk": self.risk.value,
            "calibration_population_digest": self.calibration_population_digest,
            "annotation_protocol_digest": self.annotation_protocol_digest,
            "sample_count": self.sample_count,
            "accepted_error_upper_ppm": self.accepted_error_upper_ppm,
            "assessed_error_upper_ppm": self.assessed_error_upper_ppm,
            "confidence_ppm": self.confidence_ppm,
            "valid_from_unix": self.valid_from_unix,
            "valid_through_unix": self.valid_through_unix,
            "assessment_receipt_digest": self.assessment_receipt_digest,
        }

    @classmethod
    def from_data(cls, value: object) -> "CalibrationAssessment":
        raw = _fields(
            value,
            {
                "schema",
                "risk",
                "calibration_population_digest",
                "annotation_protocol_digest",
                "sample_count",
                "accepted_error_upper_ppm",
                "assessed_error_upper_ppm",
                "confidence_ppm",
                "valid_from_unix",
                "valid_through_unix",
                "assessment_receipt_digest",
            },
            "calibration assessment",
        )
        if raw["schema"] != CALIBRATION_ASSESSMENT_SCHEMA:
            raise PanelSoftOntologyError("calibration assessment schema differs")
        try:
            result = cls(
                CalibrationRisk(raw["risk"]),
                raw["calibration_population_digest"],
                raw["annotation_protocol_digest"],
                raw["sample_count"],
                raw["accepted_error_upper_ppm"],
                raw["assessed_error_upper_ppm"],
                raw["confidence_ppm"],
                raw["valid_from_unix"],
                raw["valid_through_unix"],
                raw["assessment_receipt_digest"],
            )
        except (TypeError, ValueError) as exc:
            if isinstance(exc, PanelSoftOntologyError):
                raise
            raise PanelSoftOntologyError("calibration assessment value differs") from exc
        _require_canonical(result, raw, "calibration assessment")
        return result


@dataclass(frozen=True, slots=True)
class PresenceCalibrationGrant:
    domain: FeatureDomain
    observer_contract_digest: str
    observer_measurement_protocol_digest: str
    claim_error_assessment: CalibrationAssessment
    calibration_receipt_digest: str

    def __post_init__(self) -> None:
        if type(self.domain) is not FeatureDomain:
            raise TypeError("presence grant domain must be FeatureDomain")
        _digest(self.observer_contract_digest, "presence observer contract digest")
        _digest(self.observer_measurement_protocol_digest, "presence measurement protocol digest")
        _digest(self.calibration_receipt_digest, "presence calibration receipt digest")
        if (
            type(self.claim_error_assessment) is not CalibrationAssessment
            or self.claim_error_assessment.risk is not CalibrationRisk.FALSE_POSITIVE_CLAIM
        ):
            raise PanelSoftOntologyError("presence grant needs a false-positive assessment")

    @property
    def grant_digest(self) -> str:
        return canonical_digest(self.to_data())

    def to_data(self) -> dict[str, object]:
        return {
            "schema": PRESENCE_GRANT_SCHEMA,
            "capability": CalibrationCapability.PRESENCE.value,
            "domain": self.domain.to_data(),
            "observer_contract_digest": self.observer_contract_digest,
            "observer_measurement_protocol_digest": self.observer_measurement_protocol_digest,
            "claim_error_assessment": self.claim_error_assessment.to_data(),
            "calibration_receipt_digest": self.calibration_receipt_digest,
        }

    @classmethod
    def from_data(cls, value: object) -> "PresenceCalibrationGrant":
        raw = _fields(
            value,
            {
                "schema",
                "capability",
                "domain",
                "observer_contract_digest",
                "observer_measurement_protocol_digest",
                "claim_error_assessment",
                "calibration_receipt_digest",
            },
            "presence calibration grant",
        )
        if raw["schema"] != PRESENCE_GRANT_SCHEMA or raw["capability"] != CalibrationCapability.PRESENCE.value:
            raise PanelSoftOntologyError("presence grant schema/capability differs")
        result = cls(
            FeatureDomain.from_data(raw["domain"]),
            raw["observer_contract_digest"],
            raw["observer_measurement_protocol_digest"],
            CalibrationAssessment.from_data(raw["claim_error_assessment"]),
            raw["calibration_receipt_digest"],
        )
        _require_canonical(result, raw, "presence calibration grant")
        return result


@dataclass(frozen=True, slots=True)
class AbsenceCalibrationGrant:
    domain: FeatureDomain
    observer_contract_digest: str
    observer_measurement_protocol_digest: str
    owner_enumeration_protocol_digest: str
    search_protocol_digest: str
    claim_error_assessment: CalibrationAssessment
    inventory_completeness_assessment: CalibrationAssessment
    allowed_resolution: EnumerationResolution
    allowed_rejection_kinds: tuple[RejectionKind, ...]
    calibration_receipt_digest: str

    def __post_init__(self) -> None:
        if type(self.domain) is not FeatureDomain:
            raise TypeError("absence grant domain must be FeatureDomain")
        for label, item in (
            ("absence observer contract digest", self.observer_contract_digest),
            ("absence measurement protocol digest", self.observer_measurement_protocol_digest),
            ("absence owner enumeration protocol digest", self.owner_enumeration_protocol_digest),
            ("absence search protocol digest", self.search_protocol_digest),
            ("absence calibration receipt digest", self.calibration_receipt_digest),
        ):
            _digest(item, label)
        if (
            type(self.claim_error_assessment) is not CalibrationAssessment
            or self.claim_error_assessment.risk is not CalibrationRisk.FALSE_NEGATIVE_CLAIM
        ):
            raise PanelSoftOntologyError("absence grant needs a false-negative assessment")
        if (
            type(self.inventory_completeness_assessment) is not CalibrationAssessment
            or self.inventory_completeness_assessment.risk
            is not CalibrationRisk.OWNER_INVENTORY_OMISSION
        ):
            raise PanelSoftOntologyError("absence grant needs an inventory-omission assessment")
        _enum_instance(self.allowed_resolution, EnumerationResolution, "absence resolution")
        if type(self.allowed_rejection_kinds) is not tuple or not self.allowed_rejection_kinds or any(
            type(item) is not RejectionKind for item in self.allowed_rejection_kinds
        ):
            raise TypeError("absence rejection kinds must be a non-empty tuple")
        values = tuple(item.value for item in self.allowed_rejection_kinds)
        if values != tuple(sorted(values)) or len(values) != len(set(values)):
            raise PanelSoftOntologyError("absence rejection kinds must be unique and sorted")

    @property
    def grant_digest(self) -> str:
        return canonical_digest(self.to_data())

    def to_data(self) -> dict[str, object]:
        return {
            "schema": ABSENCE_GRANT_SCHEMA,
            "capability": CalibrationCapability.ABSENCE.value,
            "domain": self.domain.to_data(),
            "observer_contract_digest": self.observer_contract_digest,
            "observer_measurement_protocol_digest": self.observer_measurement_protocol_digest,
            "owner_enumeration_protocol_digest": self.owner_enumeration_protocol_digest,
            "search_protocol_digest": self.search_protocol_digest,
            "claim_error_assessment": self.claim_error_assessment.to_data(),
            "inventory_completeness_assessment": self.inventory_completeness_assessment.to_data(),
            "allowed_resolution": self.allowed_resolution.value,
            "allowed_rejection_kinds": [item.value for item in self.allowed_rejection_kinds],
            "calibration_receipt_digest": self.calibration_receipt_digest,
        }

    @classmethod
    def from_data(cls, value: object) -> "AbsenceCalibrationGrant":
        raw = _fields(
            value,
            {
                "schema",
                "capability",
                "domain",
                "observer_contract_digest",
                "observer_measurement_protocol_digest",
                "owner_enumeration_protocol_digest",
                "search_protocol_digest",
                "claim_error_assessment",
                "inventory_completeness_assessment",
                "allowed_resolution",
                "allowed_rejection_kinds",
                "calibration_receipt_digest",
            },
            "absence calibration grant",
        )
        if raw["schema"] != ABSENCE_GRANT_SCHEMA or raw["capability"] != CalibrationCapability.ABSENCE.value:
            raise PanelSoftOntologyError("absence grant schema/capability differs")
        if type(raw["allowed_rejection_kinds"]) is not list:
            raise PanelSoftOntologyError("allowed rejection kinds must be a JSON list")
        try:
            result = cls(
                FeatureDomain.from_data(raw["domain"]),
                raw["observer_contract_digest"],
                raw["observer_measurement_protocol_digest"],
                raw["owner_enumeration_protocol_digest"],
                raw["search_protocol_digest"],
                CalibrationAssessment.from_data(raw["claim_error_assessment"]),
                CalibrationAssessment.from_data(raw["inventory_completeness_assessment"]),
                EnumerationResolution(raw["allowed_resolution"]),
                tuple(RejectionKind(item) for item in raw["allowed_rejection_kinds"]),
                raw["calibration_receipt_digest"],
            )
        except (TypeError, ValueError) as exc:
            if isinstance(exc, PanelSoftOntologyError):
                raise
            raise PanelSoftOntologyError("absence grant value differs") from exc
        _require_canonical(result, raw, "absence calibration grant")
        return result


@dataclass(frozen=True, slots=True)
class FeatureCalibrationAuthority:
    """A serializable assessment artifact; parsing it grants no trust."""

    authority_id: str
    domain: FeatureDomain
    observer_contract_digest: str
    trust_root_digest: str
    issuance_receipt_digest: str
    presence_grant: PresenceCalibrationGrant | None = None
    absence_grant: AbsenceCalibrationGrant | None = None

    def __post_init__(self) -> None:
        _code(self.authority_id, "calibration authority ID")
        if type(self.domain) is not FeatureDomain:
            raise TypeError("calibration authority domain must be FeatureDomain")
        _digest(self.observer_contract_digest, "authority observer contract digest")
        _digest(self.trust_root_digest, "authority trust-root digest")
        _digest(self.issuance_receipt_digest, "authority issuance receipt digest")
        if self.presence_grant is None and self.absence_grant is None:
            raise PanelSoftOntologyError("calibration authority has no typed grant")
        if self.presence_grant is not None:
            if type(self.presence_grant) is not PresenceCalibrationGrant:
                raise TypeError("authority presence grant has the wrong type")
            if self.presence_grant.domain != self.domain or self.presence_grant.observer_contract_digest != self.observer_contract_digest:
                raise PanelSoftOntologyError("authority presence grant has different custody")
        if self.absence_grant is not None:
            if type(self.absence_grant) is not AbsenceCalibrationGrant:
                raise TypeError("authority absence grant has the wrong type")
            if self.absence_grant.domain != self.domain or self.absence_grant.observer_contract_digest != self.observer_contract_digest:
                raise PanelSoftOntologyError("authority absence grant has different custody")

    @property
    def authority_digest(self) -> str:
        return canonical_digest(self.to_data())

    def to_data(self) -> dict[str, object]:
        return {
            "schema": CALIBRATION_AUTHORITY_SCHEMA,
            "authority_id": self.authority_id,
            "predicate_authority_id": PYTHON_PREDICATE_AUTHORITY_ID,
            "domain": self.domain.to_data(),
            "observer_contract_digest": self.observer_contract_digest,
            "trust_root_digest": self.trust_root_digest,
            "issuance_receipt_digest": self.issuance_receipt_digest,
            "presence_grant": None if self.presence_grant is None else self.presence_grant.to_data(),
            "absence_grant": None if self.absence_grant is None else self.absence_grant.to_data(),
        }

    @classmethod
    def from_data(cls, value: object) -> "FeatureCalibrationAuthority":
        raw = _fields(
            value,
            {"schema", "authority_id", "predicate_authority_id", "domain", "observer_contract_digest", "trust_root_digest", "issuance_receipt_digest", "presence_grant", "absence_grant"},
            "feature calibration authority",
        )
        if raw["schema"] != CALIBRATION_AUTHORITY_SCHEMA or raw["predicate_authority_id"] != PYTHON_PREDICATE_AUTHORITY_ID:
            raise PanelSoftOntologyError("calibration authority schema/authority differs")
        result = cls(
            raw["authority_id"],
            FeatureDomain.from_data(raw["domain"]),
            raw["observer_contract_digest"],
            raw["trust_root_digest"],
            raw["issuance_receipt_digest"],
            None if raw["presence_grant"] is None else PresenceCalibrationGrant.from_data(raw["presence_grant"]),
            None if raw["absence_grant"] is None else AbsenceCalibrationGrant.from_data(raw["absence_grant"]),
        )
        _require_canonical(result, raw, "feature calibration authority")
        return result


_VERIFICATION_SEAL = object()


@dataclass(frozen=True, slots=True)
class _VerifiedFeatureCalibrationAuthority:
    """Immutable trusted-process result, rechecked at every projection."""

    authority: FeatureCalibrationAuthority
    capability: CalibrationCapability
    authority_digest: str
    grant_digest: str
    trust_root_digest: str
    verifier_receipt_digest: str
    campaign_time_unix: int
    _verification_seal: object


def verify_feature_calibration_authority(
    authority: FeatureCalibrationAuthority,
    *,
    capability: CalibrationCapability,
    expected_authority_digest: str,
    expected_grant_digest: str,
    trusted_root_digest: str,
    verifier_receipt_digest: str,
    campaign_time_unix: int,
) -> _VerifiedFeatureCalibrationAuthority:
    """Verify externally supplied exact pins and return a non-serializable token."""

    if type(authority) is not FeatureCalibrationAuthority:
        raise TypeError("calibration verification requires a typed authority")
    _enum_instance(capability, CalibrationCapability, "calibration capability")
    for label, item in (
        ("expected authority digest", expected_authority_digest),
        ("expected grant digest", expected_grant_digest),
        ("trusted root digest", trusted_root_digest),
        ("verifier receipt digest", verifier_receipt_digest),
    ):
        _digest(item, label)
    if type(campaign_time_unix) is not int or campaign_time_unix < 0:
        raise PanelSoftOntologyError("campaign time must be a non-negative exact int")
    if authority.trust_root_digest != trusted_root_digest:
        raise PanelSoftOntologyError("externally pinned trust root differs")
    if authority.authority_digest != expected_authority_digest:
        raise PanelSoftOntologyError("externally pinned authority digest differs")
    grant = authority.presence_grant if capability is CalibrationCapability.PRESENCE else authority.absence_grant
    if grant is None or grant.grant_digest != expected_grant_digest:
        raise PanelSoftOntologyError("externally pinned calibration grant differs")
    assessments = [grant.claim_error_assessment]
    if isinstance(grant, AbsenceCalibrationGrant):
        assessments.append(grant.inventory_completeness_assessment)
    if any(
        not item.valid_from_unix <= campaign_time_unix <= item.valid_through_unix
        for item in assessments
    ):
        raise PanelSoftOntologyError("calibration grant is outside its validity window")
    return _VerifiedFeatureCalibrationAuthority(
        authority,
        capability,
        expected_authority_digest,
        expected_grant_digest,
        trusted_root_digest,
        verifier_receipt_digest,
        campaign_time_unix,
        _VERIFICATION_SEAL,
    )


@dataclass(frozen=True, slots=True)
class _VerifiedRawMeasurementCustody:
    """Immutable result of an external journal/content-address pin check."""

    measurement_digest: str
    inventory_digest: str
    enumeration_receipt_digest: str
    evidence_receipt_digest: str
    verifier_receipt_digest: str
    _verification_seal: object


def verify_raw_measurement_custody(
    measurement: RawFeatureMeasurement,
    *,
    expected_measurement_digest: str,
    expected_inventory_digest: str,
    expected_enumeration_receipt_digest: str,
    expected_evidence_receipt_digest: str,
    verifier_receipt_digest: str,
) -> _VerifiedRawMeasurementCustody:
    """Bind externally pinned inventory and terminal evidence receipts."""

    if type(measurement) is not RawFeatureMeasurement:
        raise TypeError("custody verification requires RawFeatureMeasurement")
    for label, item in (
        ("expected measurement digest", expected_measurement_digest),
        ("expected inventory digest", expected_inventory_digest),
        ("expected enumeration receipt", expected_enumeration_receipt_digest),
        ("expected evidence receipt", expected_evidence_receipt_digest),
        ("custody verifier receipt", verifier_receipt_digest),
    ):
        _digest(item, label)
    if measurement.measurement_digest != expected_measurement_digest:
        raise PanelSoftOntologyError("externally pinned measurement digest differs")
    if measurement.inventory.inventory_digest != expected_inventory_digest:
        raise PanelSoftOntologyError("externally pinned inventory digest differs")
    if (
        measurement.inventory.enumeration_receipt_digest
        != expected_enumeration_receipt_digest
    ):
        raise PanelSoftOntologyError("externally pinned enumeration receipt differs")
    if measurement.state is RawMeasurementState.WITNESS_ASSERTED:
        actual_evidence_receipt = measurement.witness.witness_receipt_digest  # type: ignore[union-attr]
    elif measurement.state is RawMeasurementState.EXHAUSTIVE_SEARCH_NEGATIVE:
        actual_evidence_receipt = measurement.absence.search_receipt_digest  # type: ignore[union-attr]
    else:
        raise PanelSoftOntologyError(
            "only positive witness or exhaustive absence has projectable custody"
        )
    if actual_evidence_receipt != expected_evidence_receipt_digest:
        raise PanelSoftOntologyError("externally pinned terminal evidence receipt differs")
    return _VerifiedRawMeasurementCustody(
        expected_measurement_digest,
        expected_inventory_digest,
        expected_enumeration_receipt_digest,
        expected_evidence_receipt_digest,
        verifier_receipt_digest,
        _VERIFICATION_SEAL,
    )


def project_raw_measurement(
    measurement: RawFeatureMeasurement,
    verified_authority: _VerifiedFeatureCalibrationAuthority,
    verified_custody: _VerifiedRawMeasurementCustody | None = None,
) -> Disposition:
    """Project a raw state only through the separately verified capability."""

    if type(measurement) is not RawFeatureMeasurement:
        raise TypeError("projection requires RawFeatureMeasurement")
    if (
        type(verified_authority) is not _VerifiedFeatureCalibrationAuthority
        or verified_authority._verification_seal is not _VERIFICATION_SEAL
    ):
        raise TypeError("projection requires an externally verified authority token")
    authority = verified_authority.authority
    grant = (
        authority.presence_grant
        if verified_authority.capability is CalibrationCapability.PRESENCE
        else authority.absence_grant
    )
    if (
        authority.authority_digest != verified_authority.authority_digest
        or authority.trust_root_digest != verified_authority.trust_root_digest
        or grant is None
        or grant.grant_digest != verified_authority.grant_digest
    ):
        raise PanelSoftOntologyError("verified calibration token custody changed")
    token_assessments = [grant.claim_error_assessment]
    if isinstance(grant, AbsenceCalibrationGrant):
        token_assessments.append(grant.inventory_completeness_assessment)
    if any(
        not item.valid_from_unix
        <= verified_authority.campaign_time_unix
        <= item.valid_through_unix
        for item in token_assessments
    ):
        raise PanelSoftOntologyError("verified calibration token expired")
    if (
        not authority.domain.contains(measurement.spec)
        or authority.observer_contract_digest != measurement.observer_contract_digest
    ):
        return Disposition.INDETERMINATE
    if measurement.state is RawMeasurementState.ERROR:
        return Disposition.ERROR
    if measurement.state in {
        RawMeasurementState.UNRESOLVED,
        RawMeasurementState.REGISTERED_SIBLING_CONFLICT,
    }:
        return Disposition.INDETERMINATE
    if verified_custody is None:
        return Disposition.INDETERMINATE
    if (
        type(verified_custody) is not _VerifiedRawMeasurementCustody
        or verified_custody._verification_seal is not _VERIFICATION_SEAL
    ):
        raise TypeError("projection custody token has the wrong type")
    if (
        measurement.measurement_digest != verified_custody.measurement_digest
        or measurement.inventory.inventory_digest != verified_custody.inventory_digest
        or measurement.inventory.enumeration_receipt_digest
        != verified_custody.enumeration_receipt_digest
    ):
        raise PanelSoftOntologyError("verified measurement custody changed")
    if measurement.state is RawMeasurementState.WITNESS_ASSERTED:
        if verified_authority.capability is not CalibrationCapability.PRESENCE:
            return Disposition.INDETERMINATE
        if (
            measurement.witness.witness_receipt_digest  # type: ignore[union-attr]
            != verified_custody.evidence_receipt_digest
        ):
            raise PanelSoftOntologyError("verified witness receipt changed")
        grant = authority.presence_grant
        if grant is None or measurement.measurement_protocol_digest != grant.observer_measurement_protocol_digest:
            return Disposition.INDETERMINATE
        # V1 deliberately keeps every positive payload diagnostic.  In
        # particular, the point-contact payload does not yet contain verified
        # background-corridor pixels in owner-labelled opposite sectors, and
        # the other families lack equally complete family-specific measures.
        return Disposition.INDETERMINATE
    if verified_authority.capability is not CalibrationCapability.ABSENCE:
        return Disposition.INDETERMINATE
    grant = authority.absence_grant
    certificate = measurement.absence
    if grant is None or certificate is None:
        return Disposition.INDETERMINATE
    if certificate.search_receipt_digest != verified_custody.evidence_receipt_digest:
        raise PanelSoftOntologyError("verified search receipt changed")
    if (
        certificate.search_domain.enumeration_resolution is not grant.allowed_resolution
        or measurement.measurement_protocol_digest
        != grant.observer_measurement_protocol_digest
        or certificate.inventory.enumeration_protocol_digest
        != grant.owner_enumeration_protocol_digest
        or certificate.search_protocol_digest != grant.search_protocol_digest
        or not set(certificate.rejection_kinds) <= set(grant.allowed_rejection_kinds)
    ):
        return Disposition.INDETERMINATE
    return Disposition.CERTIFIED_ABSENT
