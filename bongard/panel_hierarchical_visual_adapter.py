"""One-call hierarchical visual observation of one neutral Bongard panel.

The model sees exactly one image named ``panel.png``.  Its strict response
contains (a) one ordered, simplified macro carrier trace with a disjoint
micro-texture layer and (b) the existing complete payloads for the seven
whole-panel axes that are not geometric consequences of that trace.  Python
alone derives convexity and straight macro-span count, then assembles the
canonical complete nine-axis :class:`PanelFeatureObservationSet`.

Carrier vertices are changes in the underlying drawing action.  Changes from
solid ink to zigzags, dots, circles, squares, or triangles are rendering
texture and never create carrier vertices.  A visually ambiguous carrier is
represented by one whole-trace gap; partial geometry is never interpreted.
Protocol, parser, transport, and integrity failures raise and produce no
truth-valued artifact.
"""

from __future__ import annotations

from bongard.runtime_source_snapshot import capture_loaded_source, verify_loaded_source


_LOADED_SOURCE_SHA256 = capture_loaded_source(__name__, __file__)

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping, Sequence

from bongard import prototype_scene_observer as _scene_runtime
from bongard.canonical import canonical_digest, canonical_json
from bongard.codex_no_tools_preflight import CodexNoToolsAttestation
from bongard.object_bongard_turn_journal import (
    ObjectBongardNamedImageTurnJournalTransport,
    object_bongard_turn_journal_source_digest,
)
from bongard.panel_batched_typed_codex_observer import (
    complete_whole_panel_feature_axes,
)
from bongard.panel_feature_observation import (
    BindingFeatureObservation,
    BindingResolution,
    FeatureAxis,
    ObservationIssue,
    PanelAxisObservation,
    PanelFeatureObservationSet,
    eligible_axis_bindings,
)
from bongard.panel_feature_observer_protocol import (
    FeatureAxisObservationView,
    feature_axis_observer_output_schema,
    feature_axis_observer_prompt,
    parse_feature_axis_observer_payload,
)
from bongard.panel_hierarchical_action_geometry import (
    DerivedMacroSpanKind,
    GeometryDerivationStatus,
    GeometryEvidenceProvenance,
    GeometryTraceIssue,
    Grid16Interval,
    HierarchicalActionGeometryEvidence,
    HierarchicalActionGeometryReplay,
    MacroActionPrimitive,
    MacroActionSpan,
    MacroActionTrace,
    MicroTextureEvidence,
    MicroTexturePrimitive,
    MicroTexturePrimitiveKind,
    TraceResolution,
    UncertainGrid16Point,
    panel_hierarchical_action_geometry_algorithm_digest,
    panel_hierarchical_action_geometry_source_digest,
)
from bongard.panel_owner_inventory import PANEL_OWNER_NEUTRAL_IMAGE_NAME
from bongard.panel_soft_ontology import (
    ClosedCount,
    ConvexityParameters,
    FeatureFamily,
    PanelFeatureSpec,
    QuantizedPoint,
    QuantizedSegment,
    StraightSegmentCountParameters,
)
from bongard.panel_typed_codex_observer import (
    PanelOnlyObservationContext,
    PanelTypedCodexObserverError,
    TypedCodexRuntimeBinding,
    _bind_runtime,
    _canonical_payload,
    _digest,
    _exact_png,
    _receipt_from_data,
    _validate_receipt_binding,
    panel_typed_codex_observer_source_digest,
    typed_codex_observer_contract_digest,
    typed_measurement_protocol_digest,
)
from bongard.prototype_scene_observer import PrototypeImageIdentity
from bongard.transport import (
    CloudPolicyCacheSnapshot,
    CodexModelCatalogSnapshot,
    CodexReceipt,
    run_codex_named_images_structured,
    validate_codex_named_image_receipt,
    validate_codex_strict_output_schema,
)


HIERARCHICAL_PANEL_REQUEST_SCHEMA = (
    "gkm.bongard-hierarchical-panel-observation-request.v2"
)
HIERARCHICAL_PANEL_AXIS_ALIAS_SCHEMA = (
    "gkm.bongard-hierarchical-panel-axis-alias.v1"
)
HIERARCHICAL_PANEL_MODEL_VIEW_SCHEMA = (
    "gkm.bongard-hierarchical-panel-model-view.v2"
)
HIERARCHICAL_PANEL_CONTRACT_SCHEMA = (
    "gkm.bongard-hierarchical-panel-visual-contract.v2"
)
HIERARCHICAL_PANEL_ARTIFACT_SCHEMA = (
    "gkm.bongard-hierarchical-panel-codex-artifact.v2"
)
HIERARCHICAL_PANEL_TRANSPORT_PROVENANCE_SCHEMA = (
    "gkm.bongard-hierarchical-panel-transport-provenance.v1"
)
HIERARCHICAL_PANEL_PROTOCOL_ID = (
    "bongard.panel-hierarchical-visual-adapter/one-panel-nine-axes-v2"
)

EXPECTED_WHOLE_PANEL_AXIS_COUNT = 9
EXPECTED_TYPED_AXIS_PAYLOAD_COUNT = 7
MAX_HIERARCHICAL_PROMPT_BYTES = 256 * 1024
MAX_HIERARCHICAL_OUTPUT_SCHEMA_BYTES = 256 * 1024
MAX_HIERARCHICAL_RESPONSE_BYTES = 256 * 1024

_AXIS_ALIAS = re.compile(r"axis_[0-9]{4}\Z")
_ADDRESS = re.compile(r"sha256:[0-9a-f]{64}\Z")
_TRANSPORT_KINDS = frozenset(
    {
        "production_direct",
        "production_exactly_once_journal",
        "injected_unverified",
    }
)
_DERIVED_FAMILIES = frozenset(
    {FeatureFamily.CONVEXITY, FeatureFamily.STRAIGHT_SEGMENT_COUNT}
)
_VISUAL_TRACE_ISSUES = tuple(
    item.value
    for item in GeometryTraceIssue
    if item
    not in {
        GeometryTraceIssue.PARSER_FAILURE,
        GeometryTraceIssue.TRANSPORT_FAILURE,
        GeometryTraceIssue.INTEGRITY_FAILURE,
    }
)
_GRID16_VALUES = list(range(16))


class HierarchicalPanelVisualAdapterError(PanelTypedCodexObserverError):
    """The hierarchical request, response, custody, or replay is invalid."""


def _fields(value: object, expected: set[str], label: str) -> Mapping[str, Any]:
    if (
        not isinstance(value, Mapping)
        or any(type(key) is not str for key in value)
        or set(value) != expected
    ):
        raise HierarchicalPanelVisualAdapterError(f"{label} fields differ")
    return value


def panel_hierarchical_visual_adapter_source_digest() -> str:
    """Return the exact authenticated adapter source loaded by Python."""

    return verify_loaded_source(__name__, expected_source_sha256=_LOADED_SOURCE_SHA256)


@dataclass(frozen=True, slots=True)
class _HierarchicalPanelAxisAlias:
    alias: str
    view: FeatureAxisObservationView

    def __post_init__(self) -> None:
        if type(self.alias) is not str or _AXIS_ALIAS.fullmatch(self.alias) is None:
            raise HierarchicalPanelVisualAdapterError(
                "hierarchical axis alias differs"
            )
        if type(self.view) is not FeatureAxisObservationView:
            raise TypeError("hierarchical axis alias needs FeatureAxisObservationView")
        if self.view.axis.family in _DERIVED_FAMILIES:
            raise HierarchicalPanelVisualAdapterError(
                "Python-derived geometry axis was exposed as a model payload"
            )

    def to_data(self) -> dict[str, object]:
        return {
            "schema": HIERARCHICAL_PANEL_AXIS_ALIAS_SCHEMA,
            "alias": self.alias,
            "view": self.view.to_data(),
        }

    @classmethod
    def from_data(cls, value: object) -> "_HierarchicalPanelAxisAlias":
        raw = _fields(value, {"schema", "alias", "view"}, "hierarchical axis alias")
        if raw["schema"] != HIERARCHICAL_PANEL_AXIS_ALIAS_SCHEMA:
            raise HierarchicalPanelVisualAdapterError(
                "hierarchical axis alias schema differs"
            )
        result = cls(
            raw["alias"], FeatureAxisObservationView.from_data(raw["view"])
        )
        if result.to_data() != dict(raw):
            raise HierarchicalPanelVisualAdapterError(
                "hierarchical axis alias is not canonical"
            )
        return result


def _canonical_axis_partition() -> tuple[
    tuple[FeatureAxis, ...], tuple[FeatureAxis, ...], tuple[FeatureAxis, ...]
]:
    axes = complete_whole_panel_feature_axes()
    derived = tuple(item for item in axes if item.family in _DERIVED_FAMILIES)
    typed = tuple(item for item in axes if item.family not in _DERIVED_FAMILIES)
    if (
        len(axes) != EXPECTED_WHOLE_PANEL_AXIS_COUNT
        or len(typed) != EXPECTED_TYPED_AXIS_PAYLOAD_COUNT
        or len(derived) != 2
        or {item.family for item in derived} != _DERIVED_FAMILIES
    ):
        raise HierarchicalPanelVisualAdapterError(
            "canonical whole-panel axis catalog no longer has the pinned 7+2 shape"
        )
    return axes, typed, derived


@dataclass(frozen=True, slots=True)
class HierarchicalPanelObservationRequest:
    """Candidate-blind frozen request for the one-panel hierarchical call."""

    panel_only_context: PanelOnlyObservationContext
    axes: tuple[FeatureAxis, ...]
    typed_axes: tuple[FeatureAxis, ...]
    derived_axes: tuple[FeatureAxis, ...]
    aliases: tuple[_HierarchicalPanelAxisAlias, ...]

    def __post_init__(self) -> None:
        if type(self.panel_only_context) is not PanelOnlyObservationContext:
            raise TypeError("hierarchical request needs PanelOnlyObservationContext")
        if any(
            type(value) is not tuple
            for value in (self.axes, self.typed_axes, self.derived_axes, self.aliases)
        ):
            raise TypeError("hierarchical request collections must be exact tuples")
        axes, typed, derived = _canonical_axis_partition()
        if (self.axes, self.typed_axes, self.derived_axes) != (axes, typed, derived):
            raise HierarchicalPanelVisualAdapterError(
                "hierarchical request axis partition differs"
            )
        context = self.panel_only_context.to_observation_context()
        expected_aliases = tuple(
            _HierarchicalPanelAxisAlias(
                f"axis_{index:04d}", FeatureAxisObservationView.build(context, axis)
            )
            for index, axis in enumerate(typed)
        )
        if self.aliases != expected_aliases:
            raise HierarchicalPanelVisualAdapterError(
                "hierarchical request alias/view map differs"
            )

    @classmethod
    def build(
        cls, context: PanelOnlyObservationContext
    ) -> "HierarchicalPanelObservationRequest":
        if type(context) is not PanelOnlyObservationContext:
            raise TypeError("hierarchical request needs PanelOnlyObservationContext")
        axes, typed, derived = _canonical_axis_partition()
        observation_context = context.to_observation_context()
        aliases = tuple(
            _HierarchicalPanelAxisAlias(
                f"axis_{index:04d}",
                FeatureAxisObservationView.build(observation_context, axis),
            )
            for index, axis in enumerate(typed)
        )
        return cls(context, axes, typed, derived, aliases)

    @property
    def request_digest(self) -> str:
        return canonical_digest(self.to_data())

    @property
    def axis_set_digest(self) -> str:
        return canonical_digest(
            {
                "schema": "gkm.bongard-hierarchical-panel-axis-set.v1",
                "axis_digests": [item.axis_digest for item in self.axes],
                "typed_axis_digests": [item.axis_digest for item in self.typed_axes],
                "python_derived_axis_digests": [
                    item.axis_digest for item in self.derived_axes
                ],
            }
        )

    def model_data(self) -> dict[str, object]:
        """Return the complete inert request data used to construct the prompt."""

        return {
            "schema": HIERARCHICAL_PANEL_MODEL_VIEW_SCHEMA,
            "panel_name": PANEL_OWNER_NEUTRAL_IMAGE_NAME,
            "macro_geometry_policy": {
                "coordinate_lattice": "Grid16_exact_integers_0_through_15",
                "carrier_vertices": "underlying_action_direction_changes_only",
                "micro_rendering_transitions_create_vertices": False,
                "macro_trace_is_whole_or_indeterminate": True,
                "convexity_and_straight_count_derived_by_python": True,
            },
            "typed_axes": [
                {
                    "axis_alias": item.alias,
                    "axis_measurement": item.view.model_data(),
                }
                for item in self.aliases
            ],
        }

    def to_data(self) -> dict[str, object]:
        return {
            "schema": HIERARCHICAL_PANEL_REQUEST_SCHEMA,
            "protocol_id": HIERARCHICAL_PANEL_PROTOCOL_ID,
            "panel_only_context": self.panel_only_context.to_data(),
            "axes": [item.to_data() for item in self.axes],
            "typed_axes": [item.to_data() for item in self.typed_axes],
            "derived_axes": [item.to_data() for item in self.derived_axes],
            "aliases": [item.to_data() for item in self.aliases],
            "axis_set_digest": self.axis_set_digest,
            "model_view_digest": canonical_digest(self.model_data()),
            "model_call_count": 1,
            "model_visible_image_names": [PANEL_OWNER_NEUTRAL_IMAGE_NAME],
            "candidate_identifiers_model_visible": False,
            "task_identifiers_model_visible": False,
            "phase_identifiers_model_visible": False,
            "side_or_class_identifiers_model_visible": False,
            "formula_identifiers_model_visible": False,
            "support_or_query_role_model_visible": False,
            "python_is_canonical_authority": True,
            "engineering_only": True,
            "scientific_calibration_supplied": False,
        }

    @classmethod
    def from_data(cls, value: object) -> "HierarchicalPanelObservationRequest":
        raw = _fields(
            value,
            {
                "schema",
                "protocol_id",
                "panel_only_context",
                "axes",
                "typed_axes",
                "derived_axes",
                "aliases",
                "axis_set_digest",
                "model_view_digest",
                "model_call_count",
                "model_visible_image_names",
                "candidate_identifiers_model_visible",
                "task_identifiers_model_visible",
                "phase_identifiers_model_visible",
                "side_or_class_identifiers_model_visible",
                "formula_identifiers_model_visible",
                "support_or_query_role_model_visible",
                "python_is_canonical_authority",
                "engineering_only",
                "scientific_calibration_supplied",
            },
            "hierarchical panel request",
        )
        if (
            raw["schema"] != HIERARCHICAL_PANEL_REQUEST_SCHEMA
            or raw["protocol_id"] != HIERARCHICAL_PANEL_PROTOCOL_ID
            or any(
                type(raw[key]) is not list
                for key in ("axes", "typed_axes", "derived_axes", "aliases")
            )
            or raw["model_call_count"] != 1
            or raw["model_visible_image_names"] != [PANEL_OWNER_NEUTRAL_IMAGE_NAME]
            or any(
                raw[key] is not False
                for key in (
                    "candidate_identifiers_model_visible",
                    "task_identifiers_model_visible",
                    "phase_identifiers_model_visible",
                    "side_or_class_identifiers_model_visible",
                    "formula_identifiers_model_visible",
                    "support_or_query_role_model_visible",
                    "scientific_calibration_supplied",
                )
            )
            or raw["python_is_canonical_authority"] is not True
            or raw["engineering_only"] is not True
        ):
            raise HierarchicalPanelVisualAdapterError(
                "hierarchical panel request policy differs"
            )
        result = cls(
            PanelOnlyObservationContext.from_data(raw["panel_only_context"]),
            tuple(FeatureAxis.from_data(item) for item in raw["axes"]),
            tuple(FeatureAxis.from_data(item) for item in raw["typed_axes"]),
            tuple(FeatureAxis.from_data(item) for item in raw["derived_axes"]),
            tuple(_HierarchicalPanelAxisAlias.from_data(item) for item in raw["aliases"]),
        )
        if (
            raw["axis_set_digest"] != result.axis_set_digest
            or raw["model_view_digest"] != canonical_digest(result.model_data())
            or result.to_data() != dict(raw)
        ):
            raise HierarchicalPanelVisualAdapterError(
                "hierarchical panel request digest differs"
            )
        return result


def _point_schema() -> dict[str, object]:
    return {
        "type": "object",
        "properties": {
            "x": {"type": "integer", "enum": _GRID16_VALUES},
            "y": {"type": "integer", "enum": _GRID16_VALUES},
        },
        "required": ["x", "y"],
        "additionalProperties": False,
    }


def _macro_geometry_output_schema() -> dict[str, object]:
    issue_values = ["none", *_VISUAL_TRACE_ISSUES]
    span = {
        "type": "object",
        "properties": {
            "primitive": {"type": "string", "enum": ["line", "arc"]},
            "ordered_points": {"type": "array", "items": _point_schema()},
        },
        "required": ["primitive", "ordered_points"],
        "additionalProperties": False,
    }
    micro = {
        "type": "object",
        "properties": {
            "kind": {
                "type": "string",
                "enum": [item.value for item in MicroTexturePrimitiveKind],
            },
            "ordered_points": {"type": "array", "items": _point_schema()},
        },
        "required": ["kind", "ordered_points"],
        "additionalProperties": False,
    }
    trace = {
        "type": "object",
        "properties": {
            "resolution": {
                "type": "string",
                "enum": [TraceResolution.COMPLETE.value, TraceResolution.INDETERMINATE.value],
            },
            "ordered_spans": {"type": "array", "items": span},
            "issue": {"type": "string", "enum": issue_values},
        },
        "required": ["resolution", "ordered_spans", "issue"],
        "additionalProperties": False,
    }
    texture = {
        "type": "object",
        "properties": {
            "resolution": {
                "type": "string",
                "enum": [TraceResolution.COMPLETE.value, TraceResolution.INDETERMINATE.value],
            },
            "primitives": {"type": "array", "items": micro},
            "issue": {"type": "string", "enum": issue_values},
        },
        "required": ["resolution", "primitives", "issue"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "macro_action_trace": trace,
            "micro_texture_evidence": texture,
        },
        "required": ["macro_action_trace", "micro_texture_evidence"],
        "additionalProperties": False,
    }


def hierarchical_panel_output_schema(
    request: HierarchicalPanelObservationRequest,
) -> dict[str, object]:
    """Return the strict combined macro-plus-seven-axis output schema."""

    if type(request) is not HierarchicalPanelObservationRequest:
        raise TypeError("hierarchical schema needs HierarchicalPanelObservationRequest")
    schema = {
        "type": "object",
        "properties": {
            "macro_action_geometry": _macro_geometry_output_schema(),
            "axis_payloads": {
                "type": "object",
                "properties": {
                    item.alias: feature_axis_observer_output_schema(item.view)
                    for item in request.aliases
                },
                "required": [item.alias for item in request.aliases],
                "additionalProperties": False,
            },
        },
        "required": ["macro_action_geometry", "axis_payloads"],
        "additionalProperties": False,
    }
    if len(canonical_json(schema)) > MAX_HIERARCHICAL_OUTPUT_SCHEMA_BYTES:
        raise HierarchicalPanelVisualAdapterError(
            "hierarchical output schema exceeds the fixed byte capacity"
        )
    validate_codex_strict_output_schema(schema)
    return schema


def hierarchical_panel_prompt(request: HierarchicalPanelObservationRequest) -> str:
    """Return the candidate-blind one-panel hierarchical measurement prompt."""

    if type(request) is not HierarchicalPanelObservationRequest:
        raise TypeError("hierarchical prompt needs HierarchicalPanelObservationRequest")
    protocols = "\n\n".join(
        (
            f"BEGIN_AXIS_PROTOCOL {item.alias}\n"
            f"{feature_axis_observer_prompt(item.view)}\n"
            f"END_AXIS_PROTOCOL {item.alias}"
        )
        for item in request.aliases
    )
    prompt = (
        "Inspect the one neutral image named panel.png exactly once. Return the strict "
        "macro_action_geometry record and all seven axis_payloads. Do not compare this "
        "drawing with another drawing and do not infer any preferred answer.\n\n"
        "MACRO CARRIER PROTOCOL\n"
        "Trace the complete ordered simplified centerline of the underlying drawing "
        "action, not the black-ink envelope and not a convex hull. Grid16 coordinates "
        "are exact integers 0 through 15. Every span end must exactly equal the next "
        "span start, including the last end and first start. A line has exactly two "
        "ordered points. An arc has its start, one or more ordered curve control "
        "evidence points, and its end. Carrier vertices are only genuine underlying "
        "action direction changes. A transition between solid, zigzag, dot-marker, "
        "circle-marker, square-marker, or triangle-marker rendering is micro texture "
        "and MUST NOT split a line or create a carrier vertex. Record those decorations "
        "only in micro_texture_evidence. A marker primitive may list one or more "
        "marker-center locations; Python expands them into separate marker instances. "
        "A zigzag_stroke primitive uses two through sixteen ordered points.\n\n"
        "Use macro trace resolution complete only for one unambiguous, continuous, "
        "explicitly closed whole carrier. If object segmentation, primitive type, "
        "closure, shared endpoints, or geometry is ambiguous, return resolution "
        "indeterminate with no ordered_spans and one applicable non-none issue. Never "
        "return partial spans. For a complete trace use issue none. For complete micro "
        "evidence use issue none; if only texture is unclear, return an indeterminate "
        "empty texture layer without changing an otherwise complete macro trace. "
        "Downstream Python alone derives convexity and straight macro-span count; do "
        "not report either result.\n\n"
        "SEVEN EXISTING TYPED AXIS PROTOCOLS\n"
        f"{protocols}"
    )
    if len(prompt.encode("utf-8")) > MAX_HIERARCHICAL_PROMPT_BYTES:
        raise HierarchicalPanelVisualAdapterError(
            "hierarchical prompt exceeds the fixed byte capacity"
        )
    return prompt


def _transport_source_binding(kind: str) -> str:
    transport_source = _scene_runtime.prototype_scene_transport_source_digest()
    if kind == "production_direct":
        content: dict[str, object] = {
            "schema": "gkm.bongard-hierarchical-panel-transport-source.v1",
            "kind": kind,
            "transport_source_digest": transport_source,
        }
    elif kind == "production_exactly_once_journal":
        content = {
            "schema": "gkm.bongard-hierarchical-panel-transport-source.v1",
            "kind": kind,
            "journal_source_digest": object_bongard_turn_journal_source_digest(),
            "underlying_transport_source_digest": transport_source,
        }
    elif kind == "injected_unverified":
        content = {
            "schema": "gkm.bongard-hierarchical-panel-transport-source.v1",
            "kind": kind,
            "callable_source_identity_verified": False,
        }
    else:
        raise HierarchicalPanelVisualAdapterError(
            "hierarchical transport kind differs"
        )
    return "sha256:" + canonical_digest(content)


@dataclass(frozen=True, slots=True)
class HierarchicalPanelTransportProvenance:
    """Live transport shape; external journal custody authenticates history."""

    kind: str
    source_binding: str
    production_transport_chain_verified: bool
    benchmark_sealable: bool
    live_exact_command_recheck_capable: bool

    def __post_init__(self) -> None:
        if self.kind not in _TRANSPORT_KINDS:
            raise HierarchicalPanelVisualAdapterError(
                "hierarchical transport kind differs"
            )
        if (
            type(self.source_binding) is not str
            or _ADDRESS.fullmatch(self.source_binding) is None
        ):
            raise HierarchicalPanelVisualAdapterError(
                "hierarchical transport source binding differs"
            )
        production = self.kind != "injected_unverified"
        benchmark = self.kind == "production_exactly_once_journal"
        if (
            self.source_binding != _transport_source_binding(self.kind)
            or self.production_transport_chain_verified is not production
            or self.benchmark_sealable is not benchmark
            or self.live_exact_command_recheck_capable is not production
        ):
            raise HierarchicalPanelVisualAdapterError(
                "hierarchical transport provenance differs"
            )

    @classmethod
    def create(cls, kind: str) -> "HierarchicalPanelTransportProvenance":
        production = kind in {
            "production_direct",
            "production_exactly_once_journal",
        }
        return cls(
            kind,
            _transport_source_binding(kind),
            production,
            kind == "production_exactly_once_journal",
            production,
        )

    def to_data(self) -> dict[str, object]:
        return {
            "schema": HIERARCHICAL_PANEL_TRANSPORT_PROVENANCE_SCHEMA,
            "kind": self.kind,
            "source_binding": self.source_binding,
            "production_transport_chain_verified": (
                self.production_transport_chain_verified
            ),
            "benchmark_sealable": self.benchmark_sealable,
            "live_exact_command_recheck_capable": (
                self.live_exact_command_recheck_capable
            ),
            "physical_model_call_cold_authenticated": False,
            "transport_history_authenticated_by_artifact_alone": False,
            "benchmark_requires_external_typed_journal_terminal": True,
            "injected_callable_source_identity_verified": False,
        }

    @classmethod
    def from_data(cls, value: object) -> "HierarchicalPanelTransportProvenance":
        raw = _fields(
            value,
            {
                "schema",
                "kind",
                "source_binding",
                "production_transport_chain_verified",
                "benchmark_sealable",
                "live_exact_command_recheck_capable",
                "physical_model_call_cold_authenticated",
                "transport_history_authenticated_by_artifact_alone",
                "benchmark_requires_external_typed_journal_terminal",
                "injected_callable_source_identity_verified",
            },
            "hierarchical transport provenance",
        )
        if (
            raw["schema"] != HIERARCHICAL_PANEL_TRANSPORT_PROVENANCE_SCHEMA
            or raw["physical_model_call_cold_authenticated"] is not False
            or raw["transport_history_authenticated_by_artifact_alone"] is not False
            or raw["benchmark_requires_external_typed_journal_terminal"] is not True
            or raw["injected_callable_source_identity_verified"] is not False
        ):
            raise HierarchicalPanelVisualAdapterError(
                "hierarchical transport provenance policy differs"
            )
        result = cls(
            raw["kind"],
            raw["source_binding"],
            raw["production_transport_chain_verified"],
            raw["benchmark_sealable"],
            raw["live_exact_command_recheck_capable"],
        )
        if result.to_data() != dict(raw):
            raise HierarchicalPanelVisualAdapterError(
                "hierarchical transport provenance is not canonical"
            )
        return result


def _transport_provenance(transport: object) -> HierarchicalPanelTransportProvenance:
    if transport is run_codex_named_images_structured:
        return HierarchicalPanelTransportProvenance.create("production_direct")
    if (
        type(transport) is ObjectBongardNamedImageTurnJournalTransport
        and getattr(transport, "_underlying_transport", None)
        is run_codex_named_images_structured
        and transport.runtime.transport_source_digest
        == _scene_runtime.prototype_scene_transport_source_digest()
    ):
        return HierarchicalPanelTransportProvenance.create(
            "production_exactly_once_journal"
        )
    return HierarchicalPanelTransportProvenance.create("injected_unverified")


def _hierarchical_contract_data(
    runtime: TypedCodexRuntimeBinding,
) -> dict[str, object]:
    if type(runtime) is not TypedCodexRuntimeBinding:
        raise TypeError("hierarchical contract needs TypedCodexRuntimeBinding")
    axes, typed, derived = _canonical_axis_partition()
    return {
        "schema": HIERARCHICAL_PANEL_CONTRACT_SCHEMA,
        "protocol_id": HIERARCHICAL_PANEL_PROTOCOL_ID,
        "adapter_source_digest": panel_hierarchical_visual_adapter_source_digest(),
        "typed_adapter_source_digest": panel_typed_codex_observer_source_digest(),
        "geometry_source_digest": panel_hierarchical_action_geometry_source_digest(),
        "geometry_algorithm_digest": (
            panel_hierarchical_action_geometry_algorithm_digest()
        ),
        "base_observer_contract_digest": typed_codex_observer_contract_digest(runtime),
        "measurement_protocol_digest": typed_measurement_protocol_digest(runtime),
        "runtime": runtime.to_data(),
        "neutral_panel_name": PANEL_OWNER_NEUTRAL_IMAGE_NAME,
        "axis_digests": [item.axis_digest for item in axes],
        "typed_axis_digests": [item.axis_digest for item in typed],
        "python_derived_axis_digests": [item.axis_digest for item in derived],
        "model_calls_per_panel": 1,
        "candidate_identifiers_model_visible": False,
        "task_identifiers_model_visible": False,
        "phase_identifiers_model_visible": False,
        "side_or_class_identifiers_model_visible": False,
        "formula_identifiers_model_visible": False,
        "support_or_query_role_model_visible": False,
        "raw_black_ink_convex_hull_used": False,
        "micro_texture_affects_macro_geometry": False,
        "python_is_canonical_authority": True,
        "engineering_only": True,
        "scientific_calibration_supplied": False,
    }


def _hierarchical_contract_digest(runtime: TypedCodexRuntimeBinding) -> str:
    return canonical_digest(_hierarchical_contract_data(runtime))


def _parse_grid16_point(value: object, label: str) -> UncertainGrid16Point:
    raw = _fields(value, {"x", "y"}, label)
    if (
        type(raw["x"]) is not int
        or type(raw["y"]) is not int
        or not 0 <= raw["x"] <= 15
        or not 0 <= raw["y"] <= 15
    ):
        raise HierarchicalPanelVisualAdapterError(
            f"{label} must contain exact Grid16 integers"
        )
    return UncertainGrid16Point(
        Grid16Interval(raw["x"], raw["x"]),
        Grid16Interval(raw["y"], raw["y"]),
    )


def _visual_issue(value: object, label: str) -> GeometryTraceIssue | None:
    if value == "none":
        return None
    try:
        issue = GeometryTraceIssue(value)
    except (TypeError, ValueError) as exc:
        raise HierarchicalPanelVisualAdapterError(f"{label} differs") from exc
    if issue in {
        GeometryTraceIssue.PARSER_FAILURE,
        GeometryTraceIssue.TRANSPORT_FAILURE,
        GeometryTraceIssue.INTEGRITY_FAILURE,
    }:
        raise HierarchicalPanelVisualAdapterError(
            f"{label} cannot be asserted by the visual response"
        )
    return issue


def _validate_carrier_vertices(trace: MacroActionTrace) -> None:
    """Reject a line split whose shared point is not a carrier direction change."""

    for index, first in enumerate(trace.spans):
        second = trace.spans[(index + 1) % len(trace.spans)]
        if (
            first.primitive is not MacroActionPrimitive.LINE
            or second.primitive is not MacroActionPrimitive.LINE
        ):
            continue
        start = first.start.exact_point()
        vertex = first.end.exact_point()
        end = second.end.exact_point()
        if start is None or vertex is None or end is None:
            raise HierarchicalPanelVisualAdapterError(
                "complete carrier contains a non-exact shared vertex"
            )
        first_dx, first_dy = vertex.x - start.x, vertex.y - start.y
        second_dx, second_dy = end.x - vertex.x, end.y - vertex.y
        cross = first_dx * second_dy - first_dy * second_dx
        dot = first_dx * second_dx + first_dy * second_dy
        if cross == 0 and dot > 0:
            raise HierarchicalPanelVisualAdapterError(
                "complete carrier splits one straight action at a rendering transition"
            )


def _parse_macro_action_trace(value: object) -> MacroActionTrace:
    raw = _fields(
        value, {"resolution", "ordered_spans", "issue"}, "macro action trace"
    )
    if type(raw["ordered_spans"]) is not list:
        raise HierarchicalPanelVisualAdapterError("macro ordered spans differ")
    try:
        resolution = TraceResolution(raw["resolution"])
    except (TypeError, ValueError) as exc:
        raise HierarchicalPanelVisualAdapterError(
            "macro trace resolution differs"
        ) from exc
    if resolution is TraceResolution.ERROR:
        raise HierarchicalPanelVisualAdapterError(
            "visual response cannot self-assert a protocol error"
        )
    issue = _visual_issue(raw["issue"], "macro trace issue")
    if resolution is TraceResolution.INDETERMINATE:
        if raw["ordered_spans"] or issue is None:
            raise HierarchicalPanelVisualAdapterError(
                "indeterminate macro trace must be whole, empty, and typed"
            )
        return MacroActionTrace.gap(resolution, issue)
    if issue is not None:
        raise HierarchicalPanelVisualAdapterError(
            "complete macro trace must use issue none"
        )
    # The strict transport schema cannot express array length bounds.  Project
    # schema-valid but unrepresentable topology to a whole-trace typed gap;
    # never let a hidden constructor capacity turn a valid model response into
    # an unarchived parser exception, and never retain a partial trace.
    point_counts: list[tuple[MacroActionPrimitive, int]] = []
    for index, value in enumerate(raw["ordered_spans"]):
        item = _fields(
            value,
            {"primitive", "ordered_points"},
            f"macro span {index}",
        )
        if type(item["ordered_points"]) is not list:
            raise HierarchicalPanelVisualAdapterError(
                f"macro span {index} ordered points differ"
            )
        try:
            primitive = MacroActionPrimitive(item["primitive"])
        except (TypeError, ValueError) as exc:
            raise HierarchicalPanelVisualAdapterError(
                f"macro span {index} primitive differs"
            ) from exc
        point_counts.append((primitive, len(item["ordered_points"])))
    if len(point_counts) > 12 or sum(count for _, count in point_counts) > 64:
        return MacroActionTrace.gap(
            TraceResolution.INDETERMINATE, GeometryTraceIssue.CAPACITY_LIMIT
        )
    if len(point_counts) < 2 or any(
        (primitive is MacroActionPrimitive.LINE and count != 2)
        or (primitive is MacroActionPrimitive.ARC and not 2 <= count <= 8)
        for primitive, count in point_counts
    ):
        issue = (
            GeometryTraceIssue.CAPACITY_LIMIT
            if any(
                primitive is MacroActionPrimitive.ARC and count > 8
                for primitive, count in point_counts
            )
            else GeometryTraceIssue.AMBIGUOUS_PRIMITIVE
        )
        return MacroActionTrace.gap(TraceResolution.INDETERMINATE, issue)
    spans: list[MacroActionSpan] = []
    for index, value in enumerate(raw["ordered_spans"]):
        item = _fields(
            value,
            {"primitive", "ordered_points"},
            f"macro span {index}",
        )
        if type(item["ordered_points"]) is not list:
            raise HierarchicalPanelVisualAdapterError(
                f"macro span {index} ordered points differ"
            )
        try:
            primitive = MacroActionPrimitive(item["primitive"])
        except (TypeError, ValueError) as exc:
            raise HierarchicalPanelVisualAdapterError(
                f"macro span {index} primitive differs"
            ) from exc
        if primitive is MacroActionPrimitive.INDETERMINATE:
            raise HierarchicalPanelVisualAdapterError(
                "complete macro trace cannot contain a partial span"
            )
        points = tuple(
            _parse_grid16_point(point, f"macro span {index} point {point_index}")
            for point_index, point in enumerate(item["ordered_points"])
        )
        spans.append(
            MacroActionSpan(
                TraceResolution.COMPLETE,
                primitive,
                points,
                None,
            )
        )
    try:
        trace = MacroActionTrace.complete(spans)
    except Exception as exc:
        if isinstance(exc, HierarchicalPanelVisualAdapterError):
            raise
        raise HierarchicalPanelVisualAdapterError(
            "complete macro trace is discontinuous, open, degenerate, or oversized"
        ) from exc
    _validate_carrier_vertices(trace)
    return trace


def _parse_micro_texture(value: object) -> MicroTextureEvidence:
    raw = _fields(
        value, {"resolution", "primitives", "issue"}, "micro texture evidence"
    )
    if type(raw["primitives"]) is not list:
        raise HierarchicalPanelVisualAdapterError("micro primitives differ")
    try:
        resolution = TraceResolution(raw["resolution"])
    except (TypeError, ValueError) as exc:
        raise HierarchicalPanelVisualAdapterError(
            "micro texture resolution differs"
        ) from exc
    if resolution is TraceResolution.ERROR:
        raise HierarchicalPanelVisualAdapterError(
            "visual response cannot self-assert a micro protocol error"
        )
    issue = _visual_issue(raw["issue"], "micro texture issue")
    if resolution is TraceResolution.INDETERMINATE:
        if raw["primitives"] or issue is None:
            raise HierarchicalPanelVisualAdapterError(
                "indeterminate micro texture must be empty and typed"
            )
        return MicroTextureEvidence.gap(resolution, issue)
    if issue is not None:
        raise HierarchicalPanelVisualAdapterError(
            "complete micro texture must use issue none"
        )
    primitives: list[MicroTexturePrimitive] = []
    for index, value in enumerate(raw["primitives"]):
        item = _fields(
            value,
            {"kind", "ordered_points"},
            f"micro primitive {index}",
        )
        if type(item["ordered_points"]) is not list:
            raise HierarchicalPanelVisualAdapterError(
                f"micro primitive {index} ordered points differ"
            )
        try:
            kind = MicroTexturePrimitiveKind(item["kind"])
        except (TypeError, ValueError) as exc:
            raise HierarchicalPanelVisualAdapterError(
                f"micro primitive {index} kind differs"
            ) from exc
        points = tuple(
            _parse_grid16_point(point, f"micro primitive {index} point {point_index}")
            for point_index, point in enumerate(item["ordered_points"])
        )
        if kind is MicroTexturePrimitiveKind.ZIGZAG_STROKE:
            if not 2 <= len(points) <= 16:
                return MicroTextureEvidence.gap(
                    TraceResolution.ERROR, GeometryTraceIssue.PARSER_FAILURE
                )
            primitives.append(MicroTexturePrimitive(kind, points))
        else:
            if not points:
                return MicroTextureEvidence.gap(
                    TraceResolution.ERROR, GeometryTraceIssue.PARSER_FAILURE
                )
            # The visual payload describes a repeated marker run compactly.  The
            # geometry authority represents each marker at one exact location,
            # so expand the run without changing any point or macro geometry.
            primitives.extend(
                MicroTexturePrimitive(kind, (point,)) for point in points
            )
    try:
        return MicroTextureEvidence.complete(primitives)
    except Exception as exc:
        raise HierarchicalPanelVisualAdapterError(
            "complete micro texture is malformed or oversized"
        ) from exc


_COUNT_BY_INT = {index: item for index, item in enumerate(ClosedCount, start=1)}
_GEOMETRY_TO_OBSERVATION_ISSUE = {
    GeometryTraceIssue.AMBIGUOUS_GEOMETRY: ObservationIssue.AMBIGUOUS_GEOMETRY,
    GeometryTraceIssue.OPEN_MACRO_TRACE: ObservationIssue.OPEN_BOUNDARY,
    GeometryTraceIssue.DEGENERATE_MACRO_TRACE: ObservationIssue.DEGENERATE_BOUNDARY,
    GeometryTraceIssue.SELF_INTERSECTING_MACRO_TRACE: (
        ObservationIssue.SELF_INTERSECTING_BOUNDARY
    ),
    GeometryTraceIssue.RESOLUTION_LIMIT: ObservationIssue.RESOLUTION_LIMIT,
    GeometryTraceIssue.CAPACITY_LIMIT: ObservationIssue.CAPACITY_LIMIT,
    GeometryTraceIssue.PARSER_FAILURE: ObservationIssue.PARSER_FAILURE,
    GeometryTraceIssue.TRANSPORT_FAILURE: ObservationIssue.TRANSPORT_FAILURE,
    GeometryTraceIssue.INTEGRITY_FAILURE: ObservationIssue.INTEGRITY_FAILURE,
}


def _derived_observation_issue(
    issue: GeometryTraceIssue | None, family: FeatureFamily
) -> ObservationIssue:
    if issue is GeometryTraceIssue.MISSING_ORDERED_MACRO_TRACE:
        return (
            ObservationIssue.MISSING_STRAIGHTNESS_EVIDENCE
            if family is FeatureFamily.STRAIGHT_SEGMENT_COUNT
            else ObservationIssue.MISSING_BOUNDARY_EVIDENCE
        )
    if issue is GeometryTraceIssue.AMBIGUOUS_PRIMITIVE:
        return (
            ObservationIssue.MISSING_STRAIGHTNESS_EVIDENCE
            if family is FeatureFamily.STRAIGHT_SEGMENT_COUNT
            else ObservationIssue.AMBIGUOUS_GEOMETRY
        )
    if issue is None:
        return ObservationIssue.RESOLUTION_LIMIT
    return _GEOMETRY_TO_OBSERVATION_ISSUE.get(
        issue, ObservationIssue.AMBIGUOUS_GEOMETRY
    )


def _canonical_line_evidence(
    replay: HierarchicalActionGeometryReplay,
) -> tuple[QuantizedSegment, ...] | None:
    evidence: list[QuantizedSegment] = []
    trace = replay.evidence.macro_action_trace
    for span, kind in zip(
        trace.spans, replay.straight_span_count.span_kinds, strict=True
    ):
        if kind is not DerivedMacroSpanKind.STRAIGHT:
            continue
        start = span.start.exact_point()
        end = span.end.exact_point()
        if start is None or end is None or start == end:
            return None
        low, high = sorted((start, end))
        evidence.append(QuantizedSegment(low, high))
    ordered = tuple(sorted(evidence))
    if len(ordered) != len(set(ordered)):
        return None
    return ordered


def _macro_axis_observation(
    request: HierarchicalPanelObservationRequest,
    axis: FeatureAxis,
    replay: HierarchicalActionGeometryReplay,
    receipt_digest: str,
) -> PanelAxisObservation:
    context = request.panel_only_context.to_observation_context()
    bindings = eligible_axis_bindings(axis, context)
    if len(bindings) != 1:
        raise HierarchicalPanelVisualAdapterError(
            "Python-derived whole-panel axis does not have exactly one binding"
        )
    binding = bindings[0]
    observed_specs: tuple[PanelFeatureSpec, ...] = ()
    straight_evidence: tuple[QuantizedSegment, ...] = ()
    boundary = None
    if axis.family is FeatureFamily.STRAIGHT_SEGMENT_COUNT:
        derived = replay.straight_span_count
        if derived.status is GeometryDerivationStatus.RESOLVED:
            count = _COUNT_BY_INT.get(derived.lower_bound)
            lines = _canonical_line_evidence(replay)
            if count is None:
                resolution = BindingResolution.UNCLEAR
                issue = ObservationIssue.OUTSIDE_CLOSED_CATALOG
            elif lines is None or len(lines) != derived.lower_bound:
                resolution = BindingResolution.UNCLEAR
                issue = ObservationIssue.AMBIGUOUS_GEOMETRY
            else:
                observed_specs = (
                    PanelFeatureSpec(
                        axis.family,
                        axis.subject_scope,
                        axis.reference_frame,
                        StraightSegmentCountParameters(count),
                    ),
                )
                straight_evidence = lines
                resolution = BindingResolution.COMPLETE
                issue = None
        elif derived.status is GeometryDerivationStatus.ERROR:
            resolution = BindingResolution.ERROR
            issue = _derived_observation_issue(derived.issue, axis.family)
        else:
            resolution = BindingResolution.UNCLEAR
            issue = _derived_observation_issue(derived.issue, axis.family)
    elif axis.family is FeatureFamily.CONVEXITY:
        derived = replay.convexity
        if derived.status is GeometryDerivationStatus.RESOLVED:
            if derived.convexity_kind is None or derived.polygon is None:
                raise HierarchicalPanelVisualAdapterError(
                    "resolved Python convexity lacks its canonical polygon"
                )
            observed_specs = (
                PanelFeatureSpec(
                    axis.family,
                    axis.subject_scope,
                    axis.reference_frame,
                    ConvexityParameters(derived.convexity_kind),
                ),
            )
            boundary = derived.polygon
            resolution = BindingResolution.COMPLETE
            issue = None
        elif derived.status is GeometryDerivationStatus.ERROR:
            resolution = BindingResolution.ERROR
            issue = _derived_observation_issue(derived.issue, axis.family)
        else:
            resolution = BindingResolution.UNCLEAR
            issue = _derived_observation_issue(derived.issue, axis.family)
    else:
        raise HierarchicalPanelVisualAdapterError(
            "non-geometric axis reached the Python macro derivation"
        )
    row = BindingFeatureObservation(
        axis.axis_digest,
        binding,
        resolution,
        observed_specs,
        (),
        issue,
        receipt_digest,
        straight_evidence,
        boundary,
    )
    return PanelAxisObservation(
        context,
        axis,
        request.panel_only_context.observer_contract_digest,
        request.panel_only_context.measurement_protocol_digest,
        (row,),
    )


def _parse_hierarchical_panel_payload(
    request: HierarchicalPanelObservationRequest,
    payload: object,
    *,
    observation_receipt_digest: str,
) -> tuple[HierarchicalActionGeometryReplay, PanelFeatureObservationSet]:
    _digest(observation_receipt_digest, "hierarchical observation receipt digest")
    frozen = _canonical_payload(payload, "hierarchical model payload")
    if len(canonical_json(frozen)) > MAX_HIERARCHICAL_RESPONSE_BYTES:
        raise HierarchicalPanelVisualAdapterError(
            "hierarchical response exceeds the fixed byte capacity"
        )
    raw = _fields(
        frozen,
        {"macro_action_geometry", "axis_payloads"},
        "hierarchical model payload",
    )
    geometry = _fields(
        raw["macro_action_geometry"],
        {"macro_action_trace", "micro_texture_evidence"},
        "macro action geometry",
    )
    trace = _parse_macro_action_trace(geometry["macro_action_trace"])
    texture = _parse_micro_texture(geometry["micro_texture_evidence"])
    provenance = GeometryEvidenceProvenance(
        request.panel_only_context.panel_png_digest,
        request.panel_only_context.panel_png_byte_count,
        request.panel_only_context.observer_contract_digest,
        request.panel_only_context.measurement_protocol_digest,
        observation_receipt_digest,
    )
    replay = HierarchicalActionGeometryReplay.create(
        HierarchicalActionGeometryEvidence.create(provenance, trace, texture)
    )
    axis_payloads = _fields(
        raw["axis_payloads"],
        {item.alias for item in request.aliases},
        "hierarchical typed axis payloads",
    )
    observations = [
        parse_feature_axis_observer_payload(
            item.view,
            axis_payloads[item.alias],
            observer_contract_digest=(
                request.panel_only_context.observer_contract_digest
            ),
            measurement_protocol_digest=(
                request.panel_only_context.measurement_protocol_digest
            ),
            observation_receipt_digest=observation_receipt_digest,
        )
        for item in request.aliases
    ]
    observations.extend(
        _macro_axis_observation(request, axis, replay, observation_receipt_digest)
        for axis in request.derived_axes
    )
    observation_set = PanelFeatureObservationSet(
        request.panel_only_context.panel_png_digest,
        request.panel_only_context.observer_contract_digest,
        request.panel_only_context.measurement_protocol_digest,
        tuple(sorted(observations, key=lambda item: item.axis.axis_digest)),
    )
    if tuple(item.axis for item in observation_set.axis_observations) != request.axes:
        raise HierarchicalPanelVisualAdapterError(
            "hierarchical parser did not produce the canonical complete nine axes"
        )
    return replay, observation_set


@dataclass(frozen=True, slots=True)
class HierarchicalPanelCodexArtifact:
    """One exact receipted call and its deterministic nine-axis projection."""

    panel_png_digest: str
    panel_png_byte_count: int
    runtime: TypedCodexRuntimeBinding
    request: HierarchicalPanelObservationRequest
    transport_provenance: HierarchicalPanelTransportProvenance
    request_digest: str
    adapter_source_digest: str
    typed_adapter_source_digest: str
    geometry_source_digest: str
    geometry_algorithm_digest: str
    hierarchical_contract_digest: str
    observer_contract_digest: str
    measurement_protocol_digest: str
    prompt_digest: str
    output_schema_digest: str
    payload_digest: str
    model_payload: Mapping[str, Any]
    codex_receipt: CodexReceipt
    geometry_replay: HierarchicalActionGeometryReplay
    observation_set: PanelFeatureObservationSet

    def __post_init__(self) -> None:
        _digest(self.panel_png_digest, "hierarchical artifact panel digest")
        if type(self.panel_png_byte_count) is not int or self.panel_png_byte_count <= 8:
            raise HierarchicalPanelVisualAdapterError(
                "hierarchical artifact panel byte count differs"
            )
        if type(self.runtime) is not TypedCodexRuntimeBinding:
            raise TypeError("hierarchical artifact needs TypedCodexRuntimeBinding")
        if type(self.request) is not HierarchicalPanelObservationRequest:
            raise TypeError(
                "hierarchical artifact needs HierarchicalPanelObservationRequest"
            )
        if type(self.transport_provenance) is not HierarchicalPanelTransportProvenance:
            raise TypeError(
                "hierarchical artifact needs HierarchicalPanelTransportProvenance"
            )
        for label, value in (
            ("hierarchical request digest", self.request_digest),
            ("hierarchical adapter source digest", self.adapter_source_digest),
            ("typed adapter source digest", self.typed_adapter_source_digest),
            ("geometry source digest", self.geometry_source_digest),
            ("geometry algorithm digest", self.geometry_algorithm_digest),
            ("hierarchical contract digest", self.hierarchical_contract_digest),
            ("hierarchical observer contract digest", self.observer_contract_digest),
            (
                "hierarchical measurement protocol digest",
                self.measurement_protocol_digest,
            ),
            ("hierarchical prompt digest", self.prompt_digest),
            ("hierarchical output schema digest", self.output_schema_digest),
            ("hierarchical payload digest", self.payload_digest),
        ):
            _digest(value, label)
        context = self.request.panel_only_context
        if (
            context.runtime != self.runtime
            or context.panel_png_digest != self.panel_png_digest
            or context.panel_png_byte_count != self.panel_png_byte_count
        ):
            raise HierarchicalPanelVisualAdapterError(
                "hierarchical panel/context/runtime custody differs"
            )
        payload = _canonical_payload(self.model_payload, "hierarchical model payload")
        object.__setattr__(self, "model_payload", payload)
        prompt = hierarchical_panel_prompt(self.request)
        schema = hierarchical_panel_output_schema(self.request)
        if (
            self.request_digest != self.request.request_digest
            or self.adapter_source_digest
            != panel_hierarchical_visual_adapter_source_digest()
            or self.typed_adapter_source_digest
            != panel_typed_codex_observer_source_digest()
            or self.geometry_source_digest
            != panel_hierarchical_action_geometry_source_digest()
            or self.geometry_algorithm_digest
            != panel_hierarchical_action_geometry_algorithm_digest()
            or self.hierarchical_contract_digest
            != _hierarchical_contract_digest(self.runtime)
            or self.observer_contract_digest
            != typed_codex_observer_contract_digest(self.runtime)
            or self.measurement_protocol_digest
            != typed_measurement_protocol_digest(self.runtime)
            or context.observer_contract_digest != self.observer_contract_digest
            or context.measurement_protocol_digest != self.measurement_protocol_digest
            or self.prompt_digest
            != hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            or self.output_schema_digest != canonical_digest(schema)
            or self.payload_digest != canonical_digest(payload)
        ):
            raise HierarchicalPanelVisualAdapterError(
                "hierarchical frozen invocation envelope differs"
            )
        _validate_receipt_binding(
            self.codex_receipt,
            runtime=self.runtime,
            prompt_digest=self.prompt_digest,
            output_schema_digest=self.output_schema_digest,
            payload_digest=self.payload_digest,
            presentation=(
                PrototypeImageIdentity(
                    PANEL_OWNER_NEUTRAL_IMAGE_NAME,
                    self.panel_png_byte_count,
                    self.panel_png_digest,
                ),
            ),
        )
        if type(self.geometry_replay) is not HierarchicalActionGeometryReplay:
            raise TypeError(
                "hierarchical artifact needs HierarchicalActionGeometryReplay"
            )
        if type(self.observation_set) is not PanelFeatureObservationSet:
            raise TypeError("hierarchical artifact needs PanelFeatureObservationSet")
        replayed_geometry, replayed_observations = _parse_hierarchical_panel_payload(
            self.request,
            payload,
            observation_receipt_digest=self.codex_receipt.receipt_digest,
        )
        if (
            replayed_geometry != self.geometry_replay
            or replayed_observations != self.observation_set
        ):
            raise HierarchicalPanelVisualAdapterError(
                "hierarchical parser replay differs"
            )

    @property
    def benchmark_sealable(self) -> bool:
        return self.transport_provenance.benchmark_sealable

    @property
    def artifact_digest(self) -> str:
        return canonical_digest(self.content_data())

    def content_data(self) -> dict[str, object]:
        return {
            "schema": HIERARCHICAL_PANEL_ARTIFACT_SCHEMA,
            "protocol_id": HIERARCHICAL_PANEL_PROTOCOL_ID,
            "panel_png_digest": self.panel_png_digest,
            "panel_png_byte_count": self.panel_png_byte_count,
            "runtime": self.runtime.to_data(),
            "request": self.request.to_data(),
            "transport_provenance": self.transport_provenance.to_data(),
            "benchmark_sealable": self.benchmark_sealable,
            "request_digest": self.request_digest,
            "adapter_source_digest": self.adapter_source_digest,
            "typed_adapter_source_digest": self.typed_adapter_source_digest,
            "geometry_source_digest": self.geometry_source_digest,
            "geometry_algorithm_digest": self.geometry_algorithm_digest,
            "hierarchical_contract_digest": self.hierarchical_contract_digest,
            "observer_contract_digest": self.observer_contract_digest,
            "measurement_protocol_digest": self.measurement_protocol_digest,
            "prompt_digest": self.prompt_digest,
            "output_schema_digest": self.output_schema_digest,
            "payload_digest": self.payload_digest,
            "model_payload": dict(self.model_payload),
            "codex_receipt": self.codex_receipt.to_dict(),
            "codex_receipt_digest": self.codex_receipt.receipt_digest,
            "geometry_replay": self.geometry_replay.to_data(),
            "geometry_replay_digest": self.geometry_replay.record_digest,
            "observation_set": self.observation_set.to_data(),
            "observation_set_digest": self.observation_set.observation_set_digest,
            "axis_observation_digests": [
                item.observation_digest
                for item in self.observation_set.axis_observations
            ],
            "model_call_count": 1,
            "model_visible_image_names": [PANEL_OWNER_NEUTRAL_IMAGE_NAME],
            "candidate_identifiers_model_visible": False,
            "task_identifiers_model_visible": False,
            "phase_identifiers_model_visible": False,
            "side_or_class_identifiers_model_visible": False,
            "formula_identifiers_model_visible": False,
            "support_or_query_role_model_visible": False,
            "macro_geometry_derived_by_python": True,
            "raw_black_ink_convex_hull_used": False,
            "micro_texture_affects_macro_geometry": False,
            "python_is_canonical_authority": True,
            "engineering_only": True,
            "scientific_calibration_supplied": False,
        }

    def to_data(self) -> dict[str, object]:
        return {**self.content_data(), "artifact_digest": self.artifact_digest}

    @classmethod
    def from_data(cls, value: object) -> "HierarchicalPanelCodexArtifact":
        raw = _fields(
            value,
            {
                "schema",
                "protocol_id",
                "panel_png_digest",
                "panel_png_byte_count",
                "runtime",
                "request",
                "transport_provenance",
                "benchmark_sealable",
                "request_digest",
                "adapter_source_digest",
                "typed_adapter_source_digest",
                "geometry_source_digest",
                "geometry_algorithm_digest",
                "hierarchical_contract_digest",
                "observer_contract_digest",
                "measurement_protocol_digest",
                "prompt_digest",
                "output_schema_digest",
                "payload_digest",
                "model_payload",
                "codex_receipt",
                "codex_receipt_digest",
                "geometry_replay",
                "geometry_replay_digest",
                "observation_set",
                "observation_set_digest",
                "axis_observation_digests",
                "model_call_count",
                "model_visible_image_names",
                "candidate_identifiers_model_visible",
                "task_identifiers_model_visible",
                "phase_identifiers_model_visible",
                "side_or_class_identifiers_model_visible",
                "formula_identifiers_model_visible",
                "support_or_query_role_model_visible",
                "macro_geometry_derived_by_python",
                "raw_black_ink_convex_hull_used",
                "micro_texture_affects_macro_geometry",
                "python_is_canonical_authority",
                "engineering_only",
                "scientific_calibration_supplied",
                "artifact_digest",
            },
            "hierarchical panel Codex artifact",
        )
        if (
            raw["schema"] != HIERARCHICAL_PANEL_ARTIFACT_SCHEMA
            or raw["protocol_id"] != HIERARCHICAL_PANEL_PROTOCOL_ID
            or type(raw["axis_observation_digests"]) is not list
            or raw["model_call_count"] != 1
            or raw["model_visible_image_names"] != [PANEL_OWNER_NEUTRAL_IMAGE_NAME]
            or any(
                raw[key] is not False
                for key in (
                    "candidate_identifiers_model_visible",
                    "task_identifiers_model_visible",
                    "phase_identifiers_model_visible",
                    "side_or_class_identifiers_model_visible",
                    "formula_identifiers_model_visible",
                    "support_or_query_role_model_visible",
                    "raw_black_ink_convex_hull_used",
                    "micro_texture_affects_macro_geometry",
                    "scientific_calibration_supplied",
                )
            )
            or raw["macro_geometry_derived_by_python"] is not True
            or raw["python_is_canonical_authority"] is not True
            or raw["engineering_only"] is not True
        ):
            raise HierarchicalPanelVisualAdapterError(
                "hierarchical panel artifact policy differs"
            )
        result = cls(
            panel_png_digest=raw["panel_png_digest"],
            panel_png_byte_count=raw["panel_png_byte_count"],
            runtime=TypedCodexRuntimeBinding.from_data(raw["runtime"]),
            request=HierarchicalPanelObservationRequest.from_data(raw["request"]),
            transport_provenance=HierarchicalPanelTransportProvenance.from_data(
                raw["transport_provenance"]
            ),
            request_digest=raw["request_digest"],
            adapter_source_digest=raw["adapter_source_digest"],
            typed_adapter_source_digest=raw["typed_adapter_source_digest"],
            geometry_source_digest=raw["geometry_source_digest"],
            geometry_algorithm_digest=raw["geometry_algorithm_digest"],
            hierarchical_contract_digest=raw["hierarchical_contract_digest"],
            observer_contract_digest=raw["observer_contract_digest"],
            measurement_protocol_digest=raw["measurement_protocol_digest"],
            prompt_digest=raw["prompt_digest"],
            output_schema_digest=raw["output_schema_digest"],
            payload_digest=raw["payload_digest"],
            model_payload=_canonical_payload(
                raw["model_payload"], "archived hierarchical model payload"
            ),
            codex_receipt=_receipt_from_data(raw["codex_receipt"]),
            geometry_replay=HierarchicalActionGeometryReplay.from_data(
                raw["geometry_replay"]
            ),
            observation_set=PanelFeatureObservationSet.from_data(
                raw["observation_set"]
            ),
        )
        if (
            raw["benchmark_sealable"] is not result.benchmark_sealable
            or raw["codex_receipt_digest"] != result.codex_receipt.receipt_digest
            or raw["geometry_replay_digest"] != result.geometry_replay.record_digest
            or raw["observation_set_digest"]
            != result.observation_set.observation_set_digest
            or raw["axis_observation_digests"]
            != result.content_data()["axis_observation_digests"]
            or raw["artifact_digest"] != result.artifact_digest
            or result.to_data() != dict(raw)
        ):
            raise HierarchicalPanelVisualAdapterError(
                "hierarchical panel artifact digest differs"
            )
        return result


def observe_hierarchical_panel(
    panel_png: bytes,
    *,
    request: HierarchicalPanelObservationRequest,
    model: str,
    reasoning_effort: str,
    minutes: int = 15,
    verbose: bool = False,
    executable: str = "codex",
    cloud_policy_cache_snapshot: CloudPolicyCacheSnapshot | None = None,
    expected_launcher_digest: str,
    model_catalog_snapshot: CodexModelCatalogSnapshot,
    no_tools_attestation: CodexNoToolsAttestation,
    transport=run_codex_named_images_structured,
) -> HierarchicalPanelCodexArtifact:
    """Observe one panel with exactly one neutral, receipted headless call."""

    panel = _exact_png(panel_png)
    if type(request) is not HierarchicalPanelObservationRequest:
        raise TypeError(
            "hierarchical observation needs HierarchicalPanelObservationRequest"
        )
    if not callable(transport):
        raise TypeError("hierarchical transport must be callable")
    runtime = _bind_runtime(
        model=model,
        reasoning_effort=reasoning_effort,
        expected_launcher_digest=expected_launcher_digest,
        cloud_policy_cache_snapshot=cloud_policy_cache_snapshot,
        model_catalog_snapshot=model_catalog_snapshot,
        no_tools_attestation=no_tools_attestation,
    )
    panel_digest = hashlib.sha256(panel).hexdigest()
    context = request.panel_only_context
    if (
        context.runtime != runtime
        or context.panel_png_digest != panel_digest
        or context.panel_png_byte_count != len(panel)
    ):
        raise HierarchicalPanelVisualAdapterError(
            "hierarchical request belongs to another panel or runtime"
        )
    provenance = _transport_provenance(transport)
    prompt = hierarchical_panel_prompt(request)
    schema = hierarchical_panel_output_schema(request)
    observer_contract = typed_codex_observer_contract_digest(runtime)
    measurement_protocol = typed_measurement_protocol_digest(runtime)
    try:
        payload, receipt = _scene_runtime._stage_and_call(
            ((PANEL_OWNER_NEUTRAL_IMAGE_NAME, panel),),
            prompt=prompt,
            schema=schema,
            model=model,
            reasoning_effort=reasoning_effort,
            minutes=minutes,
            verbose=verbose,
            executable=executable,
            cloud_policy_cache_snapshot=cloud_policy_cache_snapshot,
            expected_launcher_digest=expected_launcher_digest,
            model_catalog_snapshot=model_catalog_snapshot,
            no_tools_attestation=no_tools_attestation,
            transport=transport,
        )
        frozen = _canonical_payload(payload, "hierarchical model payload")
        geometry_replay, observation_set = _parse_hierarchical_panel_payload(
            request,
            frozen,
            observation_receipt_digest=receipt.receipt_digest,
        )
        return HierarchicalPanelCodexArtifact(
            panel_png_digest=panel_digest,
            panel_png_byte_count=len(panel),
            runtime=runtime,
            request=request,
            transport_provenance=provenance,
            request_digest=request.request_digest,
            adapter_source_digest=panel_hierarchical_visual_adapter_source_digest(),
            typed_adapter_source_digest=panel_typed_codex_observer_source_digest(),
            geometry_source_digest=(
                panel_hierarchical_action_geometry_source_digest()
            ),
            geometry_algorithm_digest=(
                panel_hierarchical_action_geometry_algorithm_digest()
            ),
            hierarchical_contract_digest=_hierarchical_contract_digest(runtime),
            observer_contract_digest=observer_contract,
            measurement_protocol_digest=measurement_protocol,
            prompt_digest=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            output_schema_digest=canonical_digest(schema),
            payload_digest=canonical_digest(frozen),
            model_payload=frozen,
            codex_receipt=receipt,
            geometry_replay=geometry_replay,
            observation_set=observation_set,
        )
    except HierarchicalPanelVisualAdapterError:
        raise
    except Exception as exc:
        raise HierarchicalPanelVisualAdapterError(
            "hierarchical call or parser failed with ERROR; no artifact was produced"
        ) from exc


def verify_hierarchical_panel_artifact(
    artifact: HierarchicalPanelCodexArtifact,
    panel_png: bytes,
    *,
    expected_artifact_digest: str,
) -> HierarchicalPanelCodexArtifact:
    """Cold replay exact pixels, source pins, full receipt, parser, and derivations."""

    if type(artifact) is not HierarchicalPanelCodexArtifact:
        raise TypeError(
            "hierarchical cold replay needs HierarchicalPanelCodexArtifact"
        )
    expected = _digest(
        expected_artifact_digest, "expected hierarchical artifact digest"
    )
    panel_hierarchical_visual_adapter_source_digest()
    restored = HierarchicalPanelCodexArtifact.from_data(artifact.to_data())
    if restored.artifact_digest != expected:
        raise HierarchicalPanelVisualAdapterError(
            "hierarchical artifact differs from commitment"
        )
    panel = _exact_png(panel_png)
    if (
        restored.panel_png_digest != hashlib.sha256(panel).hexdigest()
        or restored.panel_png_byte_count != len(panel)
    ):
        raise HierarchicalPanelVisualAdapterError(
            "hierarchical artifact panel bytes differ"
        )
    rebuilt = HierarchicalPanelObservationRequest.build(
        restored.request.panel_only_context
    )
    if rebuilt != restored.request:
        raise HierarchicalPanelVisualAdapterError(
            "hierarchical cold-replay request differs"
        )
    prompt = hierarchical_panel_prompt(rebuilt)
    schema = hierarchical_panel_output_schema(rebuilt)
    with tempfile.TemporaryDirectory(prefix="bongard-hierarchical-panel-replay-") as raw:
        target = Path(raw) / PANEL_OWNER_NEUTRAL_IMAGE_NAME
        target.write_bytes(panel)
        try:
            validate_codex_named_image_receipt(
                restored.codex_receipt,
                prompt,
                (str(target.resolve()),),
                (PANEL_OWNER_NEUTRAL_IMAGE_NAME,),
                schema,
                dict(restored.model_payload),
            )
        except Exception as exc:
            raise HierarchicalPanelVisualAdapterError(
                "hierarchical receipt cold replay failed with integrity ERROR"
            ) from exc
        if target.read_bytes() != panel:
            raise HierarchicalPanelVisualAdapterError(
                "hierarchical cold-replay panel changed"
            )
    replayed_geometry, replayed_observations = _parse_hierarchical_panel_payload(
        rebuilt,
        dict(restored.model_payload),
        observation_receipt_digest=restored.codex_receipt.receipt_digest,
    )
    if (
        replayed_geometry != restored.geometry_replay
        or replayed_observations != restored.observation_set
    ):
        raise HierarchicalPanelVisualAdapterError(
            "hierarchical typed cold replay differs"
        )
    return restored


__all__ = (
    "EXPECTED_TYPED_AXIS_PAYLOAD_COUNT",
    "EXPECTED_WHOLE_PANEL_AXIS_COUNT",
    "HIERARCHICAL_PANEL_PROTOCOL_ID",
    "HierarchicalPanelCodexArtifact",
    "HierarchicalPanelObservationRequest",
    "HierarchicalPanelTransportProvenance",
    "HierarchicalPanelVisualAdapterError",
    "hierarchical_panel_output_schema",
    "hierarchical_panel_prompt",
    "observe_hierarchical_panel",
    "panel_hierarchical_visual_adapter_source_digest",
    "verify_hierarchical_panel_artifact",
)
