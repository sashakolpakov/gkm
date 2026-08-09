"""Offline receipt, neutrality, strictness, and cold-replay tests for batching."""

from __future__ import annotations

from copy import deepcopy
import inspect
import json
from pathlib import Path

import pytest

from bongard.panel_batched_typed_codex_observer import (
    MAX_BATCHED_AXES,
    MAX_BATCHED_OUTPUT_SCHEMA_BYTES,
    MAX_BATCHED_PROMPT_BYTES,
    BatchedFeatureAxisRequest,
    PanelBatchedTypedCodexObserverError,
    TypedBatchedAxisCodexArtifact,
    batched_feature_axis_output_schema,
    batched_feature_axis_prompt,
    complete_whole_panel_feature_axes,
    observe_typed_panel_axes_batched,
    parse_batched_feature_axis_payload,
    verify_typed_batched_axis_codex_artifact,
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


def _context(panel: bytes):
    return build_panel_only_observation_context(
        panel,
        model=MODEL,
        reasoning_effort=EFFORT,
        expected_launcher_digest=LAUNCHER_DIGEST,
        **NO_TOOLS_KWARGS,
    )


def _payload(request: BatchedFeatureAxisRequest) -> dict[str, object]:
    payload: dict[str, object] = {}
    for item in request.aliases:
        assert len(item.view.bindings) == 1
        binding = item.view.bindings[0]
        if item.view.axis.family is FeatureFamily.STRAIGHT_SEGMENT_COUNT:
            row = {
                "resolution": "unclear",
                "straight_segment_evidence": [],
                "issue": "missing_straightness_evidence",
            }
        elif item.view.axis.family is FeatureFamily.CONVEXITY:
            row = {
                "resolution": "unclear",
                "outer_boundary_vertices": [],
                "issue": "missing_boundary_evidence",
            }
        else:
            row = {
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
        payload[item.alias] = {binding.alias: row}
    return payload


def _transport(payload, panel: bytes, calls: list[dict[str, object]]):
    def call(prompt, paths, names, schema, **kwargs):
        assert tuple(names) == ("panel.png",)
        assert tuple(Path(path).name for path in paths) == ("panel.png",)
        assert tuple(Path(path).read_bytes() for path in paths) == (panel,)
        calls.append(
            {
                "prompt": prompt,
                "names": tuple(names),
                "schema": deepcopy(schema),
            }
        )
        return CodexStructuredResult(
            deepcopy(payload), _receipt(prompt, paths, names, schema, payload)
        )

    return call


def test_request_requires_the_complete_canonical_axis_tuple() -> None:
    panel = _png(71)
    context = _context(panel)
    axes = complete_whole_panel_feature_axes()
    assert 1 < len(axes) <= MAX_BATCHED_AXES
    request = BatchedFeatureAxisRequest.build(context, axes)
    assert request.axes == axes
    assert tuple(item.alias for item in request.aliases) == tuple(
        f"axis_{index:04d}" for index in range(len(axes))
    )
    assert BatchedFeatureAxisRequest.from_data(request.to_data()) == request

    for invalid in (axes[:-1], tuple(reversed(axes)), (axes[0], axes[0], *axes[1:])):
        with pytest.raises(PanelBatchedTypedCodexObserverError):
            BatchedFeatureAxisRequest.build(context, invalid)


def test_prompt_and_callable_surface_have_no_task_role_channel() -> None:
    panel = _png(72)
    request = BatchedFeatureAxisRequest.build(
        _context(panel), complete_whole_panel_feature_axes()
    )
    prompt = batched_feature_axis_prompt(request)
    model_view = json.dumps(request.model_data(), sort_keys=True)
    forbidden_sentinels = (
        "candidate_spec_9f15",
        "side0_positive",
        "side1_positive",
        "native_orientation",
        "block_a",
        "block_b",
        "query_panel_93",
        "support_panel_17",
        "frozen_formula_47",
        "task_context_61",
    )
    for sentinel in forbidden_sentinels:
        assert sentinel not in prompt
        assert sentinel not in model_view

    parameters = set(inspect.signature(observe_typed_panel_axes_batched).parameters)
    assert not any(
        forbidden in parameter
        for parameter in parameters
        for forbidden in (
            "candidate",
            "orientation",
            "query",
            "support",
            "block",
            "side",
            "formula",
            "task",
        )
    )
    assert set(request.model_data()) == {"schema", "panel_name", "axes"}
    archived = request.to_data()
    assert archived["selected_candidate_specs_model_visible"] is False
    assert archived["native_task_orientation_model_visible"] is False
    assert archived["support_or_query_role_model_visible"] is False
    assert archived["frozen_formula_model_visible"] is False


def test_schema_and_parser_require_exactly_one_result_per_opaque_alias() -> None:
    panel = _png(73)
    context = _context(panel)
    request = BatchedFeatureAxisRequest.build(
        context, complete_whole_panel_feature_axes()
    )
    schema = batched_feature_axis_output_schema(request)
    aliases = {item.alias for item in request.aliases}
    assert set(schema["properties"]) == aliases
    assert set(schema["required"]) == aliases
    assert schema["additionalProperties"] is False
    assert len(json.dumps(schema, sort_keys=True).encode("utf-8")) < (
        MAX_BATCHED_OUTPUT_SCHEMA_BYTES
    )
    assert len(batched_feature_axis_prompt(request).encode("utf-8")) < (
        MAX_BATCHED_PROMPT_BYTES
    )

    payload = _payload(request)
    observations = parse_batched_feature_axis_payload(
        request,
        payload,
        observer_contract_digest=context.observer_contract_digest,
        measurement_protocol_digest=context.measurement_protocol_digest,
        observation_receipt_digest="a" * 64,
    )
    assert len(observations.axis_observations) == len(request.axes)
    assert tuple(item.axis for item in observations.axis_observations) == request.axes

    missing = deepcopy(payload)
    missing.pop(request.aliases[0].alias)
    with pytest.raises(PanelBatchedTypedCodexObserverError):
        parse_batched_feature_axis_payload(
            request,
            missing,
            observer_contract_digest=context.observer_contract_digest,
            measurement_protocol_digest=context.measurement_protocol_digest,
            observation_receipt_digest="a" * 64,
        )
    extra = deepcopy(payload)
    extra["axis_9999"] = {}
    with pytest.raises(PanelBatchedTypedCodexObserverError):
        parse_batched_feature_axis_payload(
            request,
            extra,
            observer_contract_digest=context.observer_contract_digest,
            measurement_protocol_digest=context.measurement_protocol_digest,
            observation_receipt_digest="a" * 64,
        )


def test_one_call_artifact_round_trip_cold_replay_and_tamper_rejection() -> None:
    panel = _png(74)
    context = _context(panel)
    axes = complete_whole_panel_feature_axes()
    request = BatchedFeatureAxisRequest.build(context, axes)
    payload = _payload(request)
    calls: list[dict[str, object]] = []
    artifact = observe_typed_panel_axes_batched(
        panel,
        axes=axes,
        panel_only_context=context,
        model=MODEL,
        reasoning_effort=EFFORT,
        expected_launcher_digest=LAUNCHER_DIGEST,
        **NO_TOOLS_KWARGS,
        transport=_transport(payload, panel, calls),
    )
    assert len(calls) == 1
    assert len(artifact.observation_set.axis_observations) == len(axes)
    assert artifact.codex_receipt.structured_output_digest == artifact.payload_digest
    assert all(
        row.observation_receipt_digest == artifact.codex_receipt.receipt_digest
        for observation in artifact.observation_set.axis_observations
        for row in observation.binding_observations
    )
    assert TypedBatchedAxisCodexArtifact.from_data(artifact.to_data()) == artifact
    assert verify_typed_batched_axis_codex_artifact(
        artifact,
        panel,
        expected_artifact_digest=artifact.artifact_digest,
    ) == artifact
    assert len(calls) == 1  # Cold replay invokes no transport.

    with pytest.raises(PanelBatchedTypedCodexObserverError):
        verify_typed_batched_axis_codex_artifact(
            artifact,
            _png(75),
            expected_artifact_digest=artifact.artifact_digest,
        )

    missing_result = deepcopy(artifact.to_data())
    missing_result["model_payload"].pop(artifact.request.aliases[0].alias)
    with pytest.raises(PanelBatchedTypedCodexObserverError):
        TypedBatchedAxisCodexArtifact.from_data(missing_result)

    changed_receipt = deepcopy(artifact.to_data())
    changed_receipt["codex_receipt"]["prompt_digest"] = "0" * 64
    with pytest.raises(PanelTypedCodexObserverError):
        TypedBatchedAxisCodexArtifact.from_data(changed_receipt)

    changed_order = deepcopy(artifact.to_data())
    changed_order["request"]["axes"].reverse()
    with pytest.raises(PanelBatchedTypedCodexObserverError):
        TypedBatchedAxisCodexArtifact.from_data(changed_order)
