"""Strict, candidate-neutral vision protocol for typed panel observations.

One call observes one raw panel and one complete feature axis.  The target
candidate parameter is never shown: the model receives every registered
variant for the axis and every eligible binding from the already frozen owner
inventory.  Missing rows, unknown aliases, incomplete rows, and evidence
outside a binding's derived Grid16 search region are rejected by Python.

The protocol produces engineering-only observations.  Its receipt can bind a
real headless Codex call, but that does not replace scientific calibration.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from itertools import product
import json
import re
from typing import Any, Mapping

from bongard.canonical import canonical_digest
from bongard.panel_feature_observation import (
    BindingFeatureObservation,
    BindingResolution,
    EligibleDomainGap,
    FeatureAxis,
    ObservationIssue,
    ObservationContext,
    PanelAxisObservation,
    PanelFeatureObservationError,
    PanelOnlyObservationContext,
    eligible_axis_bindings,
    observation_context_from_data,
    observation_context_region,
)
from bongard.panel_soft_ontology import (
    FAMILY_CONTRACTS,
    BoundaryPolygonError,
    BoundaryPolygonIssue,
    CanonicalBoundaryPolygon,
    ClosedCount,
    ConvexityParameters,
    FeatureFamily,
    OwnerInventory,
    PanelFeatureSpec,
    PanelSoftOntologyError,
    QuantizedPoint,
    QuantizedRegion,
    QuantizedSegment,
    StraightSegmentCountParameters,
    SubjectBinding,
    feature_catalog_data,
)


FEATURE_AXIS_VIEW_SCHEMA = "gkm.bongard-feature-axis-observer-view.v4"
FEATURE_AXIS_VIEW_PROTOCOL_ID = (
    "bongard.panel-feature-observer/one-panel-one-complete-axis-v5"
)
MAX_BINDINGS_PER_AXIS_CALL = 36

_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_VARIANT_ALIAS = re.compile(r"variant_[0-9]{4}\Z")
_BINDING_ALIAS = re.compile(r"binding_[0-9]{4}\Z")
_BOUNDARY_ISSUE_TO_OBSERVATION = {
    BoundaryPolygonIssue.OPEN_BOUNDARY: ObservationIssue.OPEN_BOUNDARY,
    BoundaryPolygonIssue.DEGENERATE_BOUNDARY: ObservationIssue.DEGENERATE_BOUNDARY,
    BoundaryPolygonIssue.SELF_INTERSECTING_BOUNDARY: (
        ObservationIssue.SELF_INTERSECTING_BOUNDARY
    ),
    BoundaryPolygonIssue.CAPACITY_LIMIT: ObservationIssue.CAPACITY_LIMIT,
}


class PanelFeatureObserverProtocolError(ValueError):
    """The model view, strict payload, or operational failure differs."""


class AxisCallStatus(str, Enum):
    SUCCESS = "success"
    CAPACITY_GAP = "capacity_gap"
    PARSER_ERROR = "parser_error"
    TRANSPORT_ERROR = "transport_error"
    INTEGRITY_ERROR = "integrity_error"


def _fields(value: object, expected: set[str], label: str) -> Mapping[str, Any]:
    if (
        not isinstance(value, Mapping)
        or any(type(key) is not str for key in value)
        or set(value) != expected
    ):
        raise PanelFeatureObserverProtocolError(f"{label} fields differ")
    return value


def _digest(value: object, label: str) -> str:
    if type(value) is not str or _DIGEST.fullmatch(value) is None:
        raise PanelFeatureObserverProtocolError(
            f"{label} must be a lowercase SHA-256"
        )
    return value


def _canonical_roundtrip(value: object, raw: Mapping[str, Any], label: str) -> None:
    if value.to_data() != dict(raw):  # type: ignore[attr-defined]
        raise PanelFeatureObserverProtocolError(f"{label} is not canonical")


def _closed_value(value: object) -> str:
    if type(value) is str:
        return value
    if (
        isinstance(value, Mapping)
        and set(value) == {"code", "exact_count"}
        and type(value["code"]) is str
        and type(value["exact_count"]) is int
    ):
        return value["code"]
    raise PanelFeatureObserverProtocolError("closed catalog value differs")


def all_axis_variants(axis: FeatureAxis) -> tuple[PanelFeatureSpec, ...]:
    """Enumerate the entire parameter catalog for an axis, not a target subset."""

    if type(axis) is not FeatureAxis:
        raise TypeError("axis variant enumeration requires FeatureAxis")
    rows = [
        item
        for item in feature_catalog_data()["families"]  # type: ignore[index]
        if item["family"] == axis.family.value
    ]
    if len(rows) != 1:
        raise PanelFeatureObserverProtocolError("feature family catalog row differs")
    schema_fields = rows[0]["parameter_schema"]["fields"]
    if type(schema_fields) is not list or not schema_fields:
        raise PanelFeatureObserverProtocolError("feature parameter catalog differs")
    names: list[str] = []
    domains: list[tuple[str, ...]] = []
    for field in schema_fields:
        if (
            not isinstance(field, Mapping)
            or set(field) != {"name", "closed_values"}
            or type(field["name"]) is not str
            or type(field["closed_values"]) is not list
            or not field["closed_values"]
        ):
            raise PanelFeatureObserverProtocolError(
                "feature parameter field catalog differs"
            )
        names.append(field["name"])
        domains.append(tuple(_closed_value(item) for item in field["closed_values"]))
    parameter_type = FAMILY_CONTRACTS[axis.family].parameter_type
    variants: list[PanelFeatureSpec] = []
    for values in product(*domains):
        try:
            parameters = parameter_type.from_data(dict(zip(names, values, strict=True)))
            variants.append(
                PanelFeatureSpec(
                    axis.family,
                    axis.subject_scope,
                    axis.reference_frame,
                    parameters,
                )
            )
        except (TypeError, ValueError, PanelSoftOntologyError) as exc:
            raise PanelFeatureObserverProtocolError(
                "closed axis variant construction failed"
            ) from exc
    ordered = tuple(sorted(variants, key=lambda item: item.spec_digest))
    if len({item.spec_digest for item in ordered}) != len(ordered):
        raise PanelFeatureObserverProtocolError("axis variants are not unique")
    return ordered


@dataclass(frozen=True, slots=True)
class AxisVariantAlias:
    alias: str
    spec: PanelFeatureSpec

    def __post_init__(self) -> None:
        if type(self.alias) is not str or _VARIANT_ALIAS.fullmatch(self.alias) is None:
            raise PanelFeatureObserverProtocolError("variant alias differs")
        if type(self.spec) is not PanelFeatureSpec:
            raise TypeError("variant alias needs PanelFeatureSpec")

    def to_data(self) -> dict[str, object]:
        return {"alias": self.alias, "spec": self.spec.to_data()}

    @classmethod
    def from_data(cls, value: object) -> "AxisVariantAlias":
        raw = _fields(value, {"alias", "spec"}, "axis variant alias")
        result = cls(raw["alias"], PanelFeatureSpec.from_data(raw["spec"]))
        _canonical_roundtrip(result, raw, "axis variant alias")
        return result


@dataclass(frozen=True, slots=True)
class AxisBindingAlias:
    alias: str
    binding: SubjectBinding
    search_region: QuantizedRegion

    def __post_init__(self) -> None:
        if type(self.alias) is not str or _BINDING_ALIAS.fullmatch(self.alias) is None:
            raise PanelFeatureObserverProtocolError("binding alias differs")
        if type(self.binding) is not SubjectBinding:
            raise TypeError("binding alias needs SubjectBinding")
        if type(self.search_region) is not QuantizedRegion:
            raise TypeError("binding alias needs QuantizedRegion")

    def to_data(self) -> dict[str, object]:
        return {
            "alias": self.alias,
            "binding": self.binding.to_data(),
            "search_region": self.search_region.to_data(),
        }

    @classmethod
    def from_data(cls, value: object) -> "AxisBindingAlias":
        raw = _fields(
            value, {"alias", "binding", "search_region"}, "axis binding alias"
        )
        result = cls(
            raw["alias"],
            SubjectBinding.from_data(raw["binding"]),
            QuantizedRegion.from_data(raw["search_region"]),
        )
        _canonical_roundtrip(result, raw, "axis binding alias")
        return result


@dataclass(frozen=True, slots=True)
class FeatureAxisObservationView:
    """Internal alias map whose model projection hides candidate and role data."""

    inventory: ObservationContext
    axis: FeatureAxis
    variants: tuple[AxisVariantAlias, ...]
    bindings: tuple[AxisBindingAlias, ...]

    def __post_init__(self) -> None:
        if type(self.inventory) not in {
            OwnerInventory,
            PanelOnlyObservationContext,
        } or type(self.axis) is not FeatureAxis:
            raise TypeError("axis observer view needs typed context and axis")
        expected_variants = all_axis_variants(self.axis)
        if self.variants != tuple(
            AxisVariantAlias(f"variant_{index:04d}", spec)
            for index, spec in enumerate(expected_variants)
        ):
            raise PanelFeatureObserverProtocolError(
                "axis observer view does not contain the full variant catalog"
            )
        expected_bindings = eligible_axis_bindings(self.axis, self.inventory)
        if self.bindings != tuple(
            AxisBindingAlias(
                f"binding_{index:04d}",
                binding,
                observation_context_region(binding, self.inventory),
            )
            for index, binding in enumerate(expected_bindings)
        ):
            raise PanelFeatureObserverProtocolError(
                "axis observer view does not contain the exact eligible bindings"
            )
        if len(self.bindings) > MAX_BINDINGS_PER_AXIS_CALL:
            raise PanelFeatureObserverProtocolError(
                "axis observer view exceeds the fixed per-call binding capacity"
            )

    @classmethod
    def build(
        cls, inventory: ObservationContext, axis: FeatureAxis
    ) -> "FeatureAxisObservationView":
        if type(inventory) not in {
            OwnerInventory,
            PanelOnlyObservationContext,
        } or type(axis) is not FeatureAxis:
            raise TypeError("axis observer view needs typed context and axis")
        variants = tuple(
            AxisVariantAlias(f"variant_{index:04d}", spec)
            for index, spec in enumerate(all_axis_variants(axis))
        )
        bindings = tuple(
            AxisBindingAlias(
                f"binding_{index:04d}",
                binding,
                observation_context_region(binding, inventory),
            )
            for index, binding in enumerate(eligible_axis_bindings(axis, inventory))
        )
        return cls(inventory, axis, variants, bindings)

    @property
    def view_digest(self) -> str:
        return canonical_digest(self.to_data())

    def to_data(self) -> dict[str, object]:
        return {
            "schema": FEATURE_AXIS_VIEW_SCHEMA,
            "protocol_id": FEATURE_AXIS_VIEW_PROTOCOL_ID,
            "context": self.inventory.to_data(),
            "axis": self.axis.to_data(),
            "variants": [item.to_data() for item in self.variants],
            "bindings": [item.to_data() for item in self.bindings],
            "candidate_parameter_in_view": False,
            "support_role_in_view": False,
            "engineering_only": True,
            "scientific_calibration_supplied": False,
        }

    @classmethod
    def from_data(cls, value: object) -> "FeatureAxisObservationView":
        raw = _fields(
            value,
            {
                "schema",
                "protocol_id",
                "context",
                "axis",
                "variants",
                "bindings",
                "candidate_parameter_in_view",
                "support_role_in_view",
                "engineering_only",
                "scientific_calibration_supplied",
            },
            "feature axis observer view",
        )
        if (
            raw["schema"] != FEATURE_AXIS_VIEW_SCHEMA
            or raw["protocol_id"] != FEATURE_AXIS_VIEW_PROTOCOL_ID
            or raw["candidate_parameter_in_view"] is not False
            or raw["support_role_in_view"] is not False
            or raw["engineering_only"] is not True
            or raw["scientific_calibration_supplied"] is not False
            or type(raw["variants"]) is not list
            or type(raw["bindings"]) is not list
        ):
            raise PanelFeatureObserverProtocolError("axis observer view policy differs")
        result = cls(
            observation_context_from_data(raw["context"]),
            FeatureAxis.from_data(raw["axis"]),
            tuple(AxisVariantAlias.from_data(item) for item in raw["variants"]),
            tuple(AxisBindingAlias.from_data(item) for item in raw["bindings"]),
        )
        _canonical_roundtrip(result, raw, "feature axis observer view")
        return result

    def model_data(self) -> dict[str, object]:
        """Return exactly the inert data rendered into the vision prompt."""

        owners = (
            self.inventory.owners
            if type(self.inventory) is OwnerInventory
            else ()
        )
        owner_by_id = {item.owner_id: item for item in owners}
        owner_alias = {
            item.owner_id: f"object_{index:04d}"
            for index, item in enumerate(owners)
        }
        return {
            "schema": "gkm.bongard-feature-axis-model-view.v1",
            "axis": {
                "family": self.axis.family.value,
                "subject_scope": self.axis.subject_scope.value,
                "reference_frame": self.axis.reference_frame.value,
            },
            "registered_variants": [
                {
                    "variant_alias": item.alias,
                    "parameters": item.spec.parameters.to_data(),
                }
                for item in self.variants
            ],
            "eligible_bindings": [
                {
                    "binding_alias": item.alias,
                    "binding_kind": item.binding.kind.value,
                    "search_region": item.search_region.to_data(),
                    "subjects": [
                        {
                            "object_alias": owner_alias[owner_id],
                            "kind": owner_by_id[owner_id].kind.value,
                            "region": owner_by_id[owner_id].region.to_data(),
                        }
                        for owner_id in item.binding.owner_ids
                    ],
                }
                for item in self.bindings
            ],
        }


def feature_axis_observer_output_schema(
    view: FeatureAxisObservationView,
) -> dict[str, object]:
    if type(view) is not FeatureAxisObservationView:
        raise TypeError("observer schema requires FeatureAxisObservationView")
    unclear_issues = sorted(
        item.value
        for item in ObservationIssue
        if item
        not in {
            ObservationIssue.UNVERIFIED_EMPTY_DOMAIN,
            ObservationIssue.PARSER_FAILURE,
            ObservationIssue.TRANSPORT_FAILURE,
            ObservationIssue.INTEGRITY_FAILURE,
        }
    )
    common_properties: dict[str, object] = {
        "resolution": {
            "type": "string",
            "enum": [BindingResolution.COMPLETE.value, BindingResolution.UNCLEAR.value],
        },
        "issue": {"type": "string", "enum": ["none", *unclear_issues]},
    }
    if view.axis.family is FeatureFamily.STRAIGHT_SEGMENT_COUNT:
        segment_schema = {
            "type": "object",
            "properties": {
                "start_x": {"type": "integer"},
                "start_y": {"type": "integer"},
                "end_x": {"type": "integer"},
                "end_y": {"type": "integer"},
            },
            "required": ["start_x", "start_y", "end_x", "end_y"],
            "additionalProperties": False,
        }
        properties = {
            **common_properties,
            "straight_segment_evidence": {
                "type": "array",
                "items": segment_schema,
            },
        }
        required = ["resolution", "straight_segment_evidence", "issue"]
    elif view.axis.family is FeatureFamily.CONVEXITY:
        vertex_schema = {
            "type": "object",
            "properties": {
                "x": {"type": "integer"},
                "y": {"type": "integer"},
            },
            "required": ["x", "y"],
            "additionalProperties": False,
        }
        properties = {
            **common_properties,
            "outer_boundary_vertices": {
                "type": "array",
                "items": vertex_schema,
            },
        }
        required = ["resolution", "outer_boundary_vertices", "issue"]
    else:
        properties = {
            **common_properties,
            "variant_evidence": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        # Python checks membership and uniqueness. Repeating a
                        # large enum here would exceed strict-schema budgets.
                        "variant_alias": {"type": "string"},
                        "evidence_x": {"type": "integer"},
                        "evidence_y": {"type": "integer"},
                    },
                    "required": ["variant_alias", "evidence_x", "evidence_y"],
                    "additionalProperties": False,
                },
            },
        }
        required = ["resolution", "variant_evidence", "issue"]
    row_schema = {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {item.alias: row_schema for item in view.bindings},
        "required": [item.alias for item in view.bindings],
        "additionalProperties": False,
    }


def feature_axis_observer_prompt(view: FeatureAxisObservationView) -> str:
    if type(view) is not FeatureAxisObservationView:
        raise TypeError("observer prompt requires FeatureAxisObservationView")
    rendered = json.dumps(
        view.model_data(), ensure_ascii=True, sort_keys=True, separators=(",", ":")
    )
    if view.axis.family is FeatureFamily.STRAIGHT_SEGMENT_COUNT:
        evidence_instructions = (
            "This axis counts visibly straight structural contour or boundary segments, "
            "not generic segment owners, curved arcs, turns, marker strokes, decorative "
            "ticks, hatching, or texture lines. Do not select a count alias. "
            "For each binding, use resolution complete only after exhaustively "
            "classifying the entire search region, then emit every straight segment "
            "exactly once in straight_segment_evidence using two distinct Grid16 "
            "endpoints. Python derives the registered count solely from that explicit "
            "line list. If straightness or exhaustive coverage is uncertain, use "
            "unclear, an empty straight_segment_evidence list, and "
            "missing_straightness_evidence (or a more specific applicable issue). "
            "An empty or over-catalog complete list becomes typed indeterminate. "
            "Segment order and endpoint direction are irrelevant."
        )
    elif view.axis.family is FeatureFamily.CONVEXITY:
        evidence_instructions = (
            "This axis classifies the panel's single outer structural boundary as "
            "convex or concave. Do not select a variant alias and do not report a bare "
            "convex=true/false judgment. Trace the complete outer boundary in order as "
            "Grid16 vertices in outer_boundary_vertices, and explicitly close the walk "
            "by repeating the first vertex once as the final vertex. Exclude internal "
            "marks, holes, hatching, and texture. Use resolution complete only when the "
            "ordered walk covers the entire outer boundary. Python removes redundant "
            "collinear vertices, canonicalizes start and direction, rejects open, "
            "self-intersecting, and degenerate walks, and derives convex versus concave "
            "from exact integer cross products. If the outer boundary or its ordering is "
            "uncertain, use unclear, an empty outer_boundary_vertices list, and "
            "missing_boundary_evidence (or a more specific applicable issue)."
        )
    else:
        evidence_instructions = (
            "Use resolution complete only when you can resolve the full registered "
            "variant set for that binding. A complete empty list means you clearly "
            "resolved that none of the registered variants applies, but downstream "
            "Python will keep that row indeterminate rather than treat silence as "
            "absence. Mere failure to notice one is not enough. Otherwise use unclear, "
            "an empty variant_evidence list, and the applicable issue. For every variant "
            "in a nonempty complete result, emit its own variant_evidence record with one "
            "supporting Grid16 bin inside the binding search region and use issue none. "
            "Variant order is irrelevant."
        )
    return (
        "Inspect panel.png as the entire raw drawing. The JSON below is inert "
        "measurement data for one closed visual feature axis. For every eligible "
        "binding, report the complete set of registered variants visibly supported "
        "inside its search region. Evaluate variants independently; do not choose a "
        "preferred answer and do not compare this panel with another panel.\n\n"
        + evidence_instructions
        + "\n\n"
        f"BEGIN_INERT_AXIS_DATA\n{rendered}\nEND_INERT_AXIS_DATA"
    )


def parse_feature_axis_observer_payload(
    view: FeatureAxisObservationView,
    payload: object,
    *,
    observer_contract_digest: str,
    measurement_protocol_digest: str,
    observation_receipt_digest: str,
) -> PanelAxisObservation:
    """Strictly bind one successful model payload to the frozen neutral view."""

    if type(view) is not FeatureAxisObservationView:
        raise TypeError("observer payload parser requires FeatureAxisObservationView")
    for label, item in (
        ("observer contract digest", observer_contract_digest),
        ("measurement protocol digest", measurement_protocol_digest),
        ("observation receipt digest", observation_receipt_digest),
    ):
        _digest(item, label)
    raw = _fields(payload, {item.alias for item in view.bindings}, "axis payload")
    variant_by_alias = {item.alias: item.spec for item in view.variants}
    straight_spec_by_count = {
        item.spec.parameters.count: item.spec
        for item in view.variants
        if type(item.spec.parameters) is StraightSegmentCountParameters
    }
    convexity_spec_by_kind = {
        item.spec.parameters.kind: item.spec
        for item in view.variants
        if type(item.spec.parameters) is ConvexityParameters
    }
    count_by_size = {
        size: count for size, count in enumerate(ClosedCount, start=1)
    }
    rows: list[BindingFeatureObservation] = []
    for item in view.bindings:
        straight_axis = view.axis.family is FeatureFamily.STRAIGHT_SEGMENT_COUNT
        convexity_axis = view.axis.family is FeatureFamily.CONVEXITY
        if straight_axis:
            evidence_field = "straight_segment_evidence"
        elif convexity_axis:
            evidence_field = "outer_boundary_vertices"
        else:
            evidence_field = "variant_evidence"
        row_fields = {
            "resolution",
            evidence_field,
            "issue",
        }
        row = _fields(raw[item.alias], row_fields, f"axis payload {item.alias}")
        try:
            resolution = BindingResolution(row["resolution"])
        except (TypeError, ValueError) as exc:
            raise PanelFeatureObserverProtocolError("axis payload resolution differs") from exc
        if resolution is BindingResolution.ERROR:
            raise PanelFeatureObserverProtocolError("model payload cannot self-assert error")

        if straight_axis:
            evidence = row["straight_segment_evidence"]
            expected_evidence_fields = {"start_x", "start_y", "end_x", "end_y"}
            if (
                type(evidence) is not list
                or any(
                    not isinstance(segment, Mapping)
                    or set(segment) != expected_evidence_fields
                    or any(type(segment[field]) is not int for field in expected_evidence_fields)
                    or any(not 0 <= segment[field] <= 15 for field in expected_evidence_fields)
                    for segment in evidence
                )
            ):
                raise PanelFeatureObserverProtocolError(
                    "axis payload straight-segment evidence differs"
                )
            try:
                segments = tuple(
                    sorted(
                        QuantizedSegment(*sorted((
                            QuantizedPoint(segment["start_x"], segment["start_y"]),
                            QuantizedPoint(segment["end_x"], segment["end_y"]),
                        )))
                        for segment in evidence
                    )
                )
            except (TypeError, ValueError, PanelSoftOntologyError) as exc:
                raise PanelFeatureObserverProtocolError(
                    "axis payload straight-segment evidence differs"
                ) from exc
            if len(segments) != len(set(segments)):
                raise PanelFeatureObserverProtocolError(
                    "axis payload straight-segment evidence is duplicated"
                )
            if resolution is BindingResolution.COMPLETE:
                if row["issue"] != "none":
                    raise PanelFeatureObserverProtocolError(
                        "complete axis payload row has inconsistent evidence"
                    )
                count = count_by_size.get(len(segments))
                if count is None:
                    resolution = BindingResolution.UNCLEAR
                    observed_specs = ()
                    issue = ObservationIssue.OUTSIDE_CLOSED_CATALOG
                    straight_segments = ()
                else:
                    try:
                        observed_specs = (straight_spec_by_count[count],)
                    except KeyError as exc:  # pragma: no cover - import-time catalog guard
                        raise PanelFeatureObserverProtocolError(
                            "straight-segment count catalog differs"
                        ) from exc
                    issue = None
                    straight_segments = segments
            else:
                if evidence or row["issue"] == "none":
                    raise PanelFeatureObserverProtocolError(
                        "unclear axis payload row claims resolved evidence"
                    )
                try:
                    issue = ObservationIssue(row["issue"])
                except (TypeError, ValueError) as exc:
                    raise PanelFeatureObserverProtocolError(
                        "unclear axis payload issue differs"
                    ) from exc
                observed_specs = ()
                straight_segments = ()
            points = ()
            outer_boundary = None
        elif convexity_axis:
            evidence = row["outer_boundary_vertices"]
            if (
                type(evidence) is not list
                or any(
                    not isinstance(record, Mapping)
                    or set(record) != {"x", "y"}
                    or type(record["x"]) is not int
                    or type(record["y"]) is not int
                    or not 0 <= record["x"] <= 15
                    or not 0 <= record["y"] <= 15
                    for record in evidence
                )
            ):
                raise PanelFeatureObserverProtocolError(
                    "axis payload outer-boundary evidence differs"
                )
            vertices = tuple(
                QuantizedPoint(record["x"], record["y"])
                for record in evidence
            )
            if resolution is BindingResolution.COMPLETE:
                if row["issue"] != "none":
                    raise PanelFeatureObserverProtocolError(
                        "complete axis payload row has inconsistent evidence"
                    )
                try:
                    outer_boundary = (
                        CanonicalBoundaryPolygon.from_closed_vertex_walk(vertices)
                    )
                except BoundaryPolygonError as exc:
                    resolution = BindingResolution.UNCLEAR
                    observed_specs = ()
                    issue = _BOUNDARY_ISSUE_TO_OBSERVATION[exc.issue]
                    outer_boundary = None
                except (TypeError, ValueError, PanelSoftOntologyError) as exc:
                    raise PanelFeatureObserverProtocolError(
                        "axis payload outer-boundary evidence differs"
                    ) from exc
                else:
                    try:
                        observed_specs = (
                            convexity_spec_by_kind[outer_boundary.convexity_kind],
                        )
                    except KeyError as exc:  # pragma: no cover - catalog guard
                        raise PanelFeatureObserverProtocolError(
                            "convexity variant catalog differs"
                        ) from exc
                    issue = None
            else:
                if evidence or row["issue"] == "none":
                    raise PanelFeatureObserverProtocolError(
                        "unclear axis payload row claims resolved evidence"
                    )
                try:
                    issue = ObservationIssue(row["issue"])
                except (TypeError, ValueError) as exc:
                    raise PanelFeatureObserverProtocolError(
                        "unclear axis payload issue differs"
                    ) from exc
                observed_specs = ()
                outer_boundary = None
            points = ()
            straight_segments = ()
        else:
            evidence = row["variant_evidence"]
            if (
                type(evidence) is not list
                or any(
                    not isinstance(record, Mapping)
                    or set(record)
                    != {"variant_alias", "evidence_x", "evidence_y"}
                    for record in evidence
                )
            ):
                raise PanelFeatureObserverProtocolError(
                    "axis payload variant evidence differs"
                )
            aliases = [record["variant_alias"] for record in evidence]
            if (
                any(
                    type(alias) is not str or alias not in variant_by_alias
                    for alias in aliases
                )
                or len(aliases) != len(set(aliases))
                or any(
                    type(record["evidence_x"]) is not int
                    or type(record["evidence_y"]) is not int
                    or not 0 <= record["evidence_x"] <= 15
                    or not 0 <= record["evidence_y"] <= 15
                    for record in evidence
                )
            ):
                raise PanelFeatureObserverProtocolError(
                    "axis payload variant evidence differs"
                )
            if resolution is BindingResolution.COMPLETE:
                if row["issue"] != "none":
                    raise PanelFeatureObserverProtocolError(
                        "complete axis payload row has inconsistent evidence"
                    )
                resolved = tuple(
                    sorted(
                        (
                            (
                                variant_by_alias[record["variant_alias"]],
                                QuantizedPoint(
                                    record["evidence_x"], record["evidence_y"]
                                ),
                            )
                            for record in evidence
                        ),
                        key=lambda resolved_item: resolved_item[0].spec_digest,
                    )
                )
                observed_specs = tuple(resolved_item[0] for resolved_item in resolved)
                points = tuple(resolved_item[1] for resolved_item in resolved)
                issue = None
            else:
                if evidence or row["issue"] == "none":
                    raise PanelFeatureObserverProtocolError(
                        "unclear axis payload row claims resolved evidence"
                    )
                try:
                    issue = ObservationIssue(row["issue"])
                except (TypeError, ValueError) as exc:
                    raise PanelFeatureObserverProtocolError(
                        "unclear axis payload issue differs"
                    ) from exc
                observed_specs = ()
                points = ()
            straight_segments = ()
            outer_boundary = None
        rows.append(
            BindingFeatureObservation(
                view.axis.axis_digest,
                item.binding,
                resolution,
                observed_specs,
                points,
                issue,
                observation_receipt_digest,
                straight_segments,
                outer_boundary,
            )
        )
    try:
        return PanelAxisObservation(
            view.inventory,
            view.axis,
            observer_contract_digest,
            measurement_protocol_digest,
            tuple(rows),
            (
                None
                if rows
                else EligibleDomainGap.unverified_empty(view.inventory, view.axis)
            ),
        )
    except PanelFeatureObservationError as exc:
        raise PanelFeatureObserverProtocolError(
            "axis payload failed typed observation validation"
        ) from exc


def unresolved_axis_observation(
    inventory: ObservationContext,
    axis: FeatureAxis,
    *,
    observer_contract_digest: str,
    measurement_protocol_digest: str,
    observation_receipt_digest: str,
    issue: ObservationIssue,
) -> PanelAxisObservation:
    """Create a total fail-closed row vector for capacity or call failure."""

    if type(inventory) not in {
        OwnerInventory,
        PanelOnlyObservationContext,
    } or type(axis) is not FeatureAxis:
        raise TypeError("unresolved axis observation needs typed context and axis")
    for label, item in (
        ("observer contract digest", observer_contract_digest),
        ("measurement protocol digest", measurement_protocol_digest),
        ("observation receipt digest", observation_receipt_digest),
    ):
        _digest(item, label)
    if type(issue) is not ObservationIssue:
        raise TypeError("unresolved axis issue has the wrong type")
    resolution = (
        BindingResolution.ERROR
        if issue
        in {
            ObservationIssue.PARSER_FAILURE,
            ObservationIssue.TRANSPORT_FAILURE,
            ObservationIssue.INTEGRITY_FAILURE,
        }
        else BindingResolution.UNCLEAR
    )
    rows = tuple(
        BindingFeatureObservation(
            axis.axis_digest,
            binding,
            resolution,
            (),
            (),
            issue,
            observation_receipt_digest,
        )
        for binding in eligible_axis_bindings(axis, inventory)
    )
    return PanelAxisObservation(
        inventory,
        axis,
        observer_contract_digest,
        measurement_protocol_digest,
        rows,
        (
            None
            if rows
            else EligibleDomainGap.unverified_empty(inventory, axis)
        ),
    )
