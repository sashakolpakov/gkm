"""Focused one-call, macro/micro, derivation, and cold-replay tests."""

from __future__ import annotations

from copy import deepcopy
import inspect
import json
from pathlib import Path

import pytest

from bongard.panel_feature_observation import BindingResolution
from bongard.panel_hierarchical_action_geometry import (
    GeometryDerivationStatus,
    GeometryTraceIssue,
    TraceResolution,
)
from bongard.panel_hierarchical_visual_adapter import (
    EXPECTED_TYPED_AXIS_PAYLOAD_COUNT,
    EXPECTED_WHOLE_PANEL_AXIS_COUNT,
    HierarchicalPanelCodexArtifact,
    HierarchicalPanelObservationRequest,
    HierarchicalPanelTransportProvenance,
    HierarchicalPanelVisualAdapterError,
    hierarchical_panel_output_schema,
    hierarchical_panel_prompt,
    observe_hierarchical_panel,
    verify_hierarchical_panel_artifact,
)
from bongard.panel_soft_ontology import FeatureFamily
from bongard.panel_typed_codex_observer import (
    PanelTypedCodexObserverError,
    build_panel_only_observation_context,
)
from bongard.tests.test_prototype_scene_observer import (
    EFFORT,
    LAUNCHER_DIGEST,
    MODEL,
    NO_TOOLS_KWARGS,
    _png,
    _receipt,
)
from bongard.transport import CodexStructuredResult


def _request(panel: bytes) -> HierarchicalPanelObservationRequest:
    context = build_panel_only_observation_context(
        panel,
        model=MODEL,
        reasoning_effort=EFFORT,
        expected_launcher_digest=LAUNCHER_DIGEST,
        **NO_TOOLS_KWARGS,
    )
    return HierarchicalPanelObservationRequest.build(context)


def _axis_payloads(
    request: HierarchicalPanelObservationRequest,
) -> dict[str, object]:
    result: dict[str, object] = {}
    for item in request.aliases:
        assert len(item.view.bindings) == 1
        binding = item.view.bindings[0]
        result[item.alias] = {
            binding.alias: {
                "resolution": "complete",
                "variant_evidence": [
                    {
                        "variant_alias": item.view.variants[0].alias,
                        "evidence_x": binding.search_region.minimum.x,
                        "evidence_y": binding.search_region.minimum.y,
                    }
                ],
                "issue": "none",
            }
        }
    return result


def _payload(
    request: HierarchicalPanelObservationRequest,
    spans: list[dict[str, object]],
    *,
    trace_resolution: str = "complete",
    trace_issue: str = "none",
) -> dict[str, object]:
    return {
        "macro_action_geometry": {
            "macro_action_trace": {
                "resolution": trace_resolution,
                "ordered_spans": spans,
                "issue": trace_issue,
            },
            "micro_texture_evidence": {
                "resolution": "complete",
                "primitives": [
                    {
                        "kind": "marker_triangle",
                        "ordered_points": [{"x": 7, "y": 2}],
                    }
                ],
                "issue": "none",
            },
        },
        "axis_payloads": _axis_payloads(request),
    }


def _line(x1: int, y1: int, x2: int, y2: int) -> dict[str, object]:
    return {
        "primitive": "line",
        "ordered_points": [{"x": x1, "y": y1}, {"x": x2, "y": y2}],
    }


def _square_spans() -> list[dict[str, object]]:
    return [
        _line(2, 2, 12, 2),
        _line(12, 2, 12, 12),
        _line(12, 12, 2, 12),
        _line(2, 12, 2, 2),
    ]


def _transport(
    payload: dict[str, object], panel: bytes, calls: list[dict[str, object]]
):
    def call(prompt, paths, names, schema, **kwargs):
        assert tuple(names) == ("panel.png",)
        assert tuple(Path(path).name for path in paths) == ("panel.png",)
        assert tuple(Path(path).read_bytes() for path in paths) == (panel,)
        calls.append(
            {"prompt": prompt, "names": tuple(names), "schema": deepcopy(schema)}
        )
        return CodexStructuredResult(
            deepcopy(payload), _receipt(prompt, paths, names, schema, payload)
        )

    return call


def _observe(
    panel: bytes,
    request: HierarchicalPanelObservationRequest,
    payload: dict[str, object],
    calls: list[dict[str, object]],
) -> HierarchicalPanelCodexArtifact:
    return observe_hierarchical_panel(
        panel,
        request=request,
        model=MODEL,
        reasoning_effort=EFFORT,
        expected_launcher_digest=LAUNCHER_DIGEST,
        **NO_TOOLS_KWARGS,
        transport=_transport(payload, panel, calls),
    )


def _axis(artifact: HierarchicalPanelCodexArtifact, family: FeatureFamily):
    matches = [
        item
        for item in artifact.observation_set.axis_observations
        if item.axis.family is family
    ]
    assert len(matches) == 1
    return matches[0]


def test_request_prompt_schema_are_fixed_complete_and_identifier_blind() -> None:
    panel = _png(81)
    request = _request(panel)
    assert len(request.axes) == EXPECTED_WHOLE_PANEL_AXIS_COUNT == 9
    assert len(request.typed_axes) == EXPECTED_TYPED_AXIS_PAYLOAD_COUNT == 7
    assert {item.family for item in request.derived_axes} == {
        FeatureFamily.CONVEXITY,
        FeatureFamily.STRAIGHT_SEGMENT_COUNT,
    }
    assert HierarchicalPanelObservationRequest.from_data(request.to_data()) == request

    prompt = hierarchical_panel_prompt(request)
    schema = hierarchical_panel_output_schema(request)
    assert set(schema["properties"]) == {"macro_action_geometry", "axis_payloads"}
    axis_schema = schema["properties"]["axis_payloads"]
    assert set(axis_schema["properties"]) == {
        item.alias for item in request.aliases
    }
    assert axis_schema["additionalProperties"] is False
    assert "micro texture" in prompt
    assert "MUST NOT split a line or create a carrier vertex" in prompt
    assert "Downstream Python alone derives convexity" in prompt

    serialized_view = json.dumps(request.model_data(), sort_keys=True)
    for sentinel in (
        "candidate_id_51",
        "task_id_52",
        "phase_id_53",
        "side_id_54",
        "class_id_55",
        "formula_id_56",
        "support_panel_57",
        "query_panel_58",
    ):
        assert sentinel not in prompt
        assert sentinel not in serialized_view
    parameters = set(inspect.signature(observe_hierarchical_panel).parameters)
    assert not any(
        forbidden in parameter
        for parameter in parameters
        for forbidden in (
            "candidate",
            "task",
            "phase",
            "side",
            "class",
            "formula",
            "support",
            "query",
        )
    )


def test_one_call_square_ignores_micro_marker_for_vertices_and_cold_replays() -> None:
    panel = _png(82)
    request = _request(panel)
    payload = _payload(request, _square_spans())
    calls: list[dict[str, object]] = []
    artifact = _observe(panel, request, payload, calls)

    assert len(calls) == 1
    assert len(artifact.observation_set.axis_observations) == 9
    assert artifact.transport_provenance.kind == "injected_unverified"
    assert artifact.benchmark_sealable is False
    assert len(artifact.geometry_replay.evidence.macro_action_trace.spans) == 4
    assert len(artifact.geometry_replay.evidence.micro_texture_evidence.primitives) == 1
    assert artifact.geometry_replay.straight_span_count.status is (
        GeometryDerivationStatus.RESOLVED
    )
    assert artifact.geometry_replay.straight_span_count.lower_bound == 4
    assert artifact.geometry_replay.convexity.status is GeometryDerivationStatus.RESOLVED
    straight = _axis(artifact, FeatureFamily.STRAIGHT_SEGMENT_COUNT)
    convexity = _axis(artifact, FeatureFamily.CONVEXITY)
    assert straight.binding_observations[0].resolution is BindingResolution.COMPLETE
    assert len(straight.binding_observations[0].straight_segment_evidence) == 4
    assert convexity.binding_observations[0].resolution is BindingResolution.COMPLETE
    assert all(
        row.observation_receipt_digest == artifact.codex_receipt.receipt_digest
        for observation in artifact.observation_set.axis_observations
        for row in observation.binding_observations
    )
    assert artifact.codex_receipt.structured_output_digest == artifact.payload_digest
    assert HierarchicalPanelCodexArtifact.from_data(artifact.to_data()) == artifact
    assert verify_hierarchical_panel_artifact(
        artifact,
        panel,
        expected_artifact_digest=artifact.artifact_digest,
    ) == artifact
    assert len(calls) == 1  # Cold replay performs no transport call.

    with pytest.raises(HierarchicalPanelVisualAdapterError):
        verify_hierarchical_panel_artifact(
            artifact,
            _png(83),
            expected_artifact_digest=artifact.artifact_digest,
        )
    tampered = deepcopy(artifact.to_data())
    tampered["adapter_source_digest"] = "0" * 64
    with pytest.raises(HierarchicalPanelVisualAdapterError):
        HierarchicalPanelCodexArtifact.from_data(tampered)
    tampered = deepcopy(artifact.to_data())
    tampered["codex_receipt"]["prompt_digest"] = "0" * 64
    with pytest.raises(PanelTypedCodexObserverError):
        HierarchicalPanelCodexArtifact.from_data(tampered)


def test_repeated_marker_locations_expand_without_splitting_macro_carrier() -> None:
    panel = _png(89)
    request = _request(panel)
    payload = _payload(request, _square_spans())
    payload["macro_action_geometry"]["micro_texture_evidence"]["primitives"] = [
        {
            "kind": "marker_square",
            "ordered_points": [
                {"x": 3, "y": 2},
                {"x": 5, "y": 2},
                {"x": 7, "y": 2},
            ],
        }
    ]
    calls: list[dict[str, object]] = []

    artifact = _observe(panel, request, payload, calls)

    markers = artifact.geometry_replay.evidence.micro_texture_evidence.primitives
    assert len(markers) == 3
    assert all(len(item.points) == 1 for item in markers)
    assert len(artifact.geometry_replay.evidence.macro_action_trace.spans) == 4
    assert artifact.geometry_replay.straight_span_count.lower_bound == 4
    assert len(calls) == 1


def test_any_arc_makes_convexity_indeterminate_but_line_count_is_derived() -> None:
    panel = _png(84)
    request = _request(panel)
    spans = [
        {
            "primitive": "arc",
            "ordered_points": [
                {"x": 2, "y": 2},
                {"x": 7, "y": 0},
                {"x": 12, "y": 2},
            ],
        },
        _line(12, 2, 12, 12),
        _line(12, 12, 2, 12),
        _line(2, 12, 2, 2),
    ]
    calls: list[dict[str, object]] = []
    artifact = _observe(panel, request, _payload(request, spans), calls)
    assert len(calls) == 1
    assert artifact.geometry_replay.convexity.status is (
        GeometryDerivationStatus.INDETERMINATE
    )
    assert artifact.geometry_replay.straight_span_count.status is (
        GeometryDerivationStatus.RESOLVED
    )
    assert artifact.geometry_replay.straight_span_count.lower_bound == 3
    assert _axis(artifact, FeatureFamily.CONVEXITY).binding_observations[
        0
    ].resolution is BindingResolution.UNCLEAR
    straight = _axis(artifact, FeatureFamily.STRAIGHT_SEGMENT_COUNT)
    assert straight.binding_observations[0].resolution is BindingResolution.COMPLETE
    assert len(straight.binding_observations[0].straight_segment_evidence) == 3


def test_schema_valid_oversized_closed_arc_becomes_whole_trace_capacity_gap() -> None:
    panel = _png(90)
    request = _request(panel)
    closed_arc = {
        "primitive": "arc",
        "ordered_points": [
            {"x": 4, "y": 4},
            {"x": 6, "y": 4},
            {"x": 7, "y": 6},
            {"x": 6, "y": 8},
            {"x": 4, "y": 9},
            {"x": 2, "y": 8},
            {"x": 1, "y": 6},
            {"x": 2, "y": 5},
            {"x": 4, "y": 4},
        ],
    }
    calls: list[dict[str, object]] = []

    artifact = _observe(panel, request, _payload(request, [closed_arc]), calls)

    trace = artifact.geometry_replay.evidence.macro_action_trace
    assert trace.resolution is TraceResolution.INDETERMINATE
    assert trace.issue is GeometryTraceIssue.CAPACITY_LIMIT
    assert trace.spans == ()
    assert artifact.geometry_replay.convexity.status is (
        GeometryDerivationStatus.INDETERMINATE
    )
    assert artifact.geometry_replay.straight_span_count.status is (
        GeometryDerivationStatus.INDETERMINATE
    )
    assert len(artifact.observation_set.axis_observations) == 9
    assert len(calls) == 1


def test_ambiguous_trace_is_wholly_indeterminate_and_texture_split_is_rejected() -> None:
    panel = _png(85)
    request = _request(panel)
    calls: list[dict[str, object]] = []
    artifact = _observe(
        panel,
        request,
        _payload(
            request,
            [],
            trace_resolution="indeterminate",
            trace_issue="ambiguous_geometry",
        ),
        calls,
    )
    assert artifact.geometry_replay.evidence.macro_action_trace.spans == ()
    assert artifact.geometry_replay.convexity.status is (
        GeometryDerivationStatus.INDETERMINATE
    )
    assert artifact.geometry_replay.straight_span_count.status is (
        GeometryDerivationStatus.INDETERMINATE
    )
    assert _axis(artifact, FeatureFamily.CONVEXITY).binding_observations[
        0
    ].resolution is BindingResolution.UNCLEAR
    assert _axis(artifact, FeatureFamily.STRAIGHT_SEGMENT_COUNT).binding_observations[
        0
    ].resolution is BindingResolution.UNCLEAR

    # A marker at (7, 2) does not authorize splitting the same straight action.
    split_at_marker = [
        _line(2, 2, 7, 2),
        _line(7, 2, 12, 2),
        _line(12, 2, 12, 12),
        _line(12, 12, 2, 12),
        _line(2, 12, 2, 2),
    ]
    rejected_calls: list[dict[str, object]] = []
    with pytest.raises(
        HierarchicalPanelVisualAdapterError, match="rendering transition"
    ):
        _observe(
            panel,
            request,
            _payload(request, split_at_marker),
            rejected_calls,
        )
    assert len(rejected_calls) == 1


def test_transport_provenance_distinguishes_direct_journal_and_injection() -> None:
    direct = HierarchicalPanelTransportProvenance.create("production_direct")
    assert direct.production_transport_chain_verified is True
    assert direct.benchmark_sealable is False
    journal = HierarchicalPanelTransportProvenance.create(
        "production_exactly_once_journal"
    )
    assert journal.production_transport_chain_verified is True
    assert journal.benchmark_sealable is True
    injected = HierarchicalPanelTransportProvenance.create("injected_unverified")
    assert injected.production_transport_chain_verified is False
    assert injected.benchmark_sealable is False
