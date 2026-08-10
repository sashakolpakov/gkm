"""Custody and non-authority tests for the typed-axis support narrator."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from bongard.canonical import canonical_json
from bongard.object_bongard_turn_journal import (
    ObjectBongardTurnJournalSummary,
    ObjectBongardTurnRuntime,
)
from bongard import panel_typed_axis_headless_proposer as proposer_module
from bongard.panel_typed_axis_headless_proposer import (
    HEADLESS_TYPED_AXIS_PRESENTATION_NAMES,
    HeadlessTypedAxisAttemptErrorArtifact,
    HeadlessTypedAxisProposerArtifact,
    HeadlessTypedAxisProposerError,
    HeadlessTypedAxisProposerRequest,
    build_headless_typed_axis_turn_journal,
    headless_typed_axis_attempt_binding,
    headless_typed_axis_candidate_rank_prompt_material,
    headless_typed_axis_proposer_output_schema,
    headless_typed_axis_proposer_prompt,
    run_headless_typed_axis_proposer,
    verify_headless_typed_axis_proposer_artifact,
    verify_headless_typed_axis_attempt_error_artifact,
)
from bongard.panel_typed_axis_slate_v2 import (
    AXES,
    Axis,
    SupportSide,
    TypedAxisCell,
    TypedAxisInventory,
    TypedSupportMatrix,
    TypedSupportRow,
)
from bongard.prototype_scene_observer import prototype_scene_transport_source_digest
from bongard.tests.test_prototype_scene_observer import (
    EFFORT,
    LAUNCHER_DIGEST,
    MODEL,
    NO_TOOLS_KWARGS,
    _png,
    _receipt,
)
from bongard.transport import CodexStructuredResult


PROTOCOL = "sha256:" + "7" * 64
PRIMARY_VALUES = {
    Axis.TOPOLOGY: "closed",
    Axis.COMPONENT_COUNT: 1,
    Axis.STRAIGHT_ACTION_COUNT: 4,
    Axis.PRIMITIVE_MIX_OR_ARC_COUNT: "straight_only",
    Axis.CATALOG_CONVEXITY: "catalog_convex",
    Axis.SYMMETRY: "none",
    Axis.ASPECT_ORIENTATION: "elongated_oblique_positive",
    Axis.TEXTURE: "plain",
}


def _runtime() -> ObjectBongardTurnRuntime:
    return ObjectBongardTurnRuntime(
        model=MODEL,
        reasoning_effort=EFFORT,
        minutes=15,
        verbose=False,
        executable="codex",
        cloud_policy_cache_snapshot=None,
        model_catalog_snapshot=NO_TOOLS_KWARGS["model_catalog_snapshot"],
        expected_launcher_digest=LAUNCHER_DIGEST,
        no_tools_attestation=NO_TOOLS_KWARGS["no_tools_attestation"],
        transport_source_digest=prototype_scene_transport_source_digest(),
    )


def _row(index: int, side: SupportSide) -> TypedSupportRow:
    values = dict(PRIMARY_VALUES)
    if side is SupportSide.CONTRAST:
        values[Axis.TOPOLOGY] = "open"
    return TypedSupportRow(
        f"secret_{side.value}_dataset_row_{index:02d}",
        side,
        tuple(
            TypedAxisCell.python_exact(axis, values[axis], PROTOCOL)
            for axis in AXES
        ),
    )


def _matrix() -> TypedSupportMatrix:
    return TypedSupportMatrix.freeze(
        tuple(_row(index, SupportSide.PRIMARY) for index in range(6))
        + tuple(_row(index, SupportSide.CONTRAST) for index in range(6))
    )


def _images() -> tuple[tuple[bytes, ...], tuple[bytes, ...]]:
    return (
        tuple(_png(310 + index) for index in range(6)),
        tuple(_png(330 + index) for index in range(6)),
    )


def _request(
    primary: tuple[bytes, ...],
    contrast: tuple[bytes, ...],
    matrix: TypedSupportMatrix,
    runtime: ObjectBongardTurnRuntime,
) -> HeadlessTypedAxisProposerRequest:
    return HeadlessTypedAxisProposerRequest.build(
        primary, contrast, matrix=matrix, runtime=runtime
    )


def _payload() -> dict[str, object]:
    result: dict[str, object] = {
        "topology": {
            "status": "nominated",
            "value": "closed",
            "gap_reason_code": "none",
        },
        "component_count": {
            "status": "nominated",
            "value": "count_1",
            "gap_reason_code": "none",
        },
        "straight_action_count": {
            "status": "nominated",
            "value": "count_4",
            "gap_reason_code": "none",
        },
        "primitive_mix_or_arc_count": {
            "status": "nominated",
            "value": "straight_only",
            "gap_reason_code": "none",
        },
        "catalog_convexity": {
            "status": "nominated",
            "value": "catalog_convex",
            "gap_reason_code": "none",
        },
        "symmetry": {
            "status": "gap",
            "value": "gap",
            "gap_reason_code": "ambiguous_visible_evidence",
        },
        "aspect_orientation": {
            "status": "nominated",
            "value": "elongated_oblique_positive",
            "gap_reason_code": "none",
        },
        "texture": {
            "status": "nominated",
            "value": "plain",
            "gap_reason_code": "none",
        },
        "positive_description": "bird-like carrier with oblique angular wings",
    }
    return result


def _transport(payload: dict[str, object], expected: tuple[bytes, ...], calls: list):
    def call(prompt, paths, names, schema, **kwargs):
        assert tuple(names) == HEADLESS_TYPED_AXIS_PRESENTATION_NAMES
        assert tuple(Path(path).read_bytes() for path in paths) == expected
        calls.append((prompt, tuple(names), deepcopy(schema)))
        return CodexStructuredResult(
            deepcopy(payload), _receipt(prompt, paths, names, schema, payload)
        )

    return call


def _run_injected(payload: dict[str, object] | None = None):
    primary, contrast = _images()
    matrix = _matrix()
    runtime = _runtime()
    request = _request(primary, contrast, matrix, runtime)
    calls: list[object] = []
    artifact = run_headless_typed_axis_proposer(
        primary,
        contrast,
        matrix=matrix,
        request=request,
        runtime=runtime,
        transport=_transport(
            _payload() if payload is None else payload,
            (*primary, *contrast),
            calls,
        ),
    )
    if payload is None:
        assert type(artifact) is HeadlessTypedAxisProposerArtifact
    return primary, contrast, matrix, runtime, request, calls, artifact


def test_request_prompt_and_schema_expose_only_neutral_6_plus_6_supports() -> None:
    primary, contrast = _images()
    matrix = _matrix()
    runtime = _runtime()
    request = _request(primary, contrast, matrix, runtime)

    assert HeadlessTypedAxisProposerRequest.from_data(request.to_data()) == request
    assert tuple(item.name for item in request.presentation) == (
        HEADLESS_TYPED_AXIS_PRESENTATION_NAMES
    )
    prompt = headless_typed_axis_proposer_prompt(request)
    schema = headless_typed_axis_proposer_output_schema(request)
    visible = prompt + canonical_json(schema).decode("utf-8")

    assert "primary_00.png" in prompt
    assert "contrast_05.png" in prompt
    assert "count_4" in prompt
    assert "bird-like" in prompt
    assert matrix.matrix_address not in visible
    assert "secret_primary_dataset_row_00" not in visible
    assert set(schema["properties"]) == {
        *(axis.value for axis in AXES),
        "positive_description",
    }
    assert schema["properties"]["straight_action_count"]["properties"][
        "value"
    ]["enum"] == [*(f"count_{value}" for value in range(10)), "gap"]
    assert all(
        type(item) is str
        for item in schema["properties"]["component_count"]["properties"][
            "value"
        ]["enum"]
    )
    data = request.to_data()
    assert data["query_image_count"] == 0
    assert data["dataset_task_side_row_ids_model_visible"] is False
    assert data["candidate_selection_authority"] is False


def test_injected_call_decodes_counts_is_unsealable_and_cold_replay_is_zero_call() -> None:
    primary, contrast, matrix, _runtime_value, _request_value, calls, artifact = (
        _run_injected()
    )

    assert len(calls) == 1
    assert artifact.transport_provenance.kind == "injected_unverified"
    assert artifact.benchmark_sealable is False
    values = {
        nomination.axis: nomination.value
        for nomination in artifact.outcome.nomination_slate.nominations
    }
    assert values[Axis.COMPONENT_COUNT] == 1
    assert type(values[Axis.COMPONENT_COUNT]) is int
    assert values[Axis.STRAIGHT_ACTION_COUNT] == 4
    assert values[Axis.SYMMETRY] is None
    assert HeadlessTypedAxisProposerArtifact.from_data(artifact.to_data()) == artifact

    restored = verify_headless_typed_axis_proposer_artifact(
        artifact,
        primary,
        contrast,
        matrix=matrix,
        expected_artifact_digest=artifact.artifact_digest,
    )
    assert restored == artifact
    assert len(calls) == 1


def test_nominations_do_not_change_inventory_and_rank_projection_excludes_prose() -> None:
    _primary, _contrast, matrix, _runtime_value, _request_value, _calls, artifact = (
        _run_injected()
    )
    baseline = TypedAxisInventory.derive(matrix)
    hinted = TypedAxisInventory.derive(
        matrix, artifact.outcome.nomination_slate
    )

    assert canonical_json(baseline.to_data()) == canonical_json(hinted.to_data())
    assert baseline.inventory_address == hinted.inventory_address
    assert hinted.to_data()["nomination_candidate_selection_authority"] is False

    projection = headless_typed_axis_candidate_rank_prompt_material(
        artifact.outcome
    )
    assert projection == ()
    artifact_data = artifact.to_data()
    assert artifact_data["inventory_derivation_or_filtering_performed"] is False
    assert artifact_data["positive_description_embedded_in_inventory"] is False
    assert artifact_data["proposer_or_artifact_digest_embedded_in_inventory"] is False
    assert artifact_data["positive_description_enters_candidate_rank_prompt"] is False
    assert artifact_data[
        "proposer_or_artifact_digest_enters_candidate_rank_prompt"
    ] is False
    assert artifact_data["nomination_hints_enter_candidate_rank_prompt"] is False
    assert artifact_data[
        "candidate_rank_prompt_excludes_all_proposer_material"
    ] is True


@pytest.mark.parametrize(
    "mutate,match",
    [
        (
            lambda payload: payload["component_count"].update(value=4),
            "schema-safe string token",
        ),
        (
            lambda payload: payload["symmetry"].update(status="nominated"),
            "has gap fields",
        ),
        (
            lambda payload: payload["topology"].update(
                status="gap", value="gap"
            ),
            "has nomination fields",
        ),
        (
            lambda payload: payload.update(
                positive_description="bird-like carrier without rounded edges"
            ),
            "forbidden policy",
        ),
    ],
)
def test_malformed_axis_or_nonpositive_prose_is_typed_non_evidential_error(
    mutate, match: str
) -> None:
    payload = _payload()
    mutate(payload)
    *_prefix, result = _run_injected(payload)
    assert type(result) is HeadlessTypedAxisAttemptErrorArtifact
    assert result.failure_stage == "payload_contract"
    assert result.failure_code == "payload_contract_rejected"
    assert result.model_payload == payload
    assert result.codex_receipt is not None
    assert result.benchmark_sealable is False
    assert result.attempt_custody_authenticated is False
    data = result.to_data()
    assert data["runner_must_bind_attempt"] is True
    assert data["attempt_error_is_axis_gap"] is False
    assert data["attempt_error_is_negative_evidence"] is False
    assert data["attempt_error_can_nominate_or_rank"] is False
    assert HeadlessTypedAxisAttemptErrorArtifact.from_data(data) == result
    with pytest.raises(TypeError, match="headless outcome"):
        headless_typed_axis_candidate_rank_prompt_material(  # type: ignore[arg-type]
            result
        )


def test_replay_binds_exact_pixels_matrix_and_rejects_external_terminal_for_injection() -> None:
    primary, contrast, matrix, _runtime_value, _request_value, _calls, artifact = (
        _run_injected()
    )
    changed = list(contrast)
    changed[-1] = _png(900)
    with pytest.raises(HeadlessTypedAxisProposerError, match="pixels differ"):
        verify_headless_typed_axis_proposer_artifact(
            artifact,
            primary,
            tuple(changed),
            matrix=matrix,
            expected_artifact_digest=artifact.artifact_digest,
        )

    different_rows = list(matrix.rows)
    different_rows[0] = TypedSupportRow(
        "different_primary_00",
        different_rows[0].side,
        different_rows[0].cells,
    )
    different = TypedSupportMatrix.freeze(different_rows)
    with pytest.raises(HeadlessTypedAxisProposerError, match="matrix differs"):
        verify_headless_typed_axis_proposer_artifact(
            artifact,
            primary,
            contrast,
            matrix=different,
            expected_artifact_digest=artifact.artifact_digest,
        )

    address = "sha256:" + "e" * 64
    terminal = ObjectBongardTurnJournalSummary(
        address, address, "success", address, address, address, address
    )
    with pytest.raises(HeadlessTypedAxisProposerError, match="injected headless"):
        verify_headless_typed_axis_proposer_artifact(
            artifact,
            primary,
            contrast,
            matrix=matrix,
            expected_artifact_digest=artifact.artifact_digest,
            proposer_journal_terminal=terminal,
        )


def test_exactly_once_production_journal_is_only_sealable_path_and_terminal_is_external(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    primary, contrast = _images()
    matrix = _matrix()
    runtime = _runtime()
    request = _request(primary, contrast, matrix, runtime)
    payload = _payload()
    physical_calls = 0

    def physical_transport(prompt, paths, names, schema, **kwargs):
        nonlocal physical_calls
        physical_calls += 1
        return CodexStructuredResult(
            deepcopy(payload), _receipt(prompt, paths, names, schema, payload)
        )

    monkeypatch.setattr(
        proposer_module, "run_codex_named_images_structured", physical_transport
    )
    journal = build_headless_typed_axis_turn_journal(
        tmp_path / "journal",
        primary,
        contrast,
        matrix=matrix,
        request=request,
        runtime=runtime,
        authorization_digest="sha256:" + "a" * 64,
        execution_precommit_digest="sha256:" + "b" * 64,
        task_id="hd_typed_axis_fixture_0001",
        underlying_transport=physical_transport,
    )
    artifact = run_headless_typed_axis_proposer(
        primary,
        contrast,
        matrix=matrix,
        request=request,
        runtime=runtime,
        transport=journal,
    )
    assert physical_calls == 1
    assert artifact.benchmark_sealable is True
    assert artifact.transport_provenance.kind == "production_exactly_once_journal"
    terminal = journal.verify()

    with pytest.raises(HeadlessTypedAxisProposerError, match="journal terminal"):
        verify_headless_typed_axis_proposer_artifact(
            artifact,
            primary,
            contrast,
            matrix=matrix,
            expected_artifact_digest=artifact.artifact_digest,
        )
    wrong = ObjectBongardTurnJournalSummary(
        terminal.manifest_digest,
        terminal.turn_key,
        terminal.terminal_status,
        terminal.claim_digest,
        terminal.result_digest,
        terminal.outcome_digest,
        "sha256:" + "f" * 64,
    )
    with pytest.raises(HeadlessTypedAxisProposerError, match="journal terminal"):
        verify_headless_typed_axis_proposer_artifact(
            artifact,
            primary,
            contrast,
            matrix=matrix,
            expected_artifact_digest=artifact.artifact_digest,
            proposer_journal_terminal=wrong,
        )
    restored = verify_headless_typed_axis_proposer_artifact(
        artifact,
        primary,
        contrast,
        matrix=matrix,
        expected_artifact_digest=artifact.artifact_digest,
        proposer_journal_terminal=terminal,
    )
    assert restored == artifact
    assert physical_calls == 1

    second = run_headless_typed_axis_proposer(
        primary,
        contrast,
        matrix=matrix,
        request=request,
        runtime=runtime,
        transport=journal,
    )
    assert second == artifact
    assert physical_calls == 1
    assert journal.fresh_call_count == 1
    assert journal.reused_call_count == 1


def test_receipted_parser_rejection_is_mandatory_and_cannot_reroll(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    primary, contrast = _images()
    matrix = _matrix()
    runtime = _runtime()
    request = _request(primary, contrast, matrix, runtime)
    payload = _payload()
    payload["straight_action_count"]["value"] = 4
    physical_calls = 0

    def physical_transport(prompt, paths, names, schema, **kwargs):
        nonlocal physical_calls
        physical_calls += 1
        return CodexStructuredResult(
            deepcopy(payload), _receipt(prompt, paths, names, schema, payload)
        )

    monkeypatch.setattr(
        proposer_module, "run_codex_named_images_structured", physical_transport
    )
    journal = build_headless_typed_axis_turn_journal(
        tmp_path / "parser_failure_journal",
        primary,
        contrast,
        matrix=matrix,
        request=request,
        runtime=runtime,
        authorization_digest="sha256:" + "1" * 64,
        execution_precommit_digest="sha256:" + "2" * 64,
        task_id="hd_typed_axis_parser_failure_0001",
        underlying_transport=physical_transport,
    )
    first = run_headless_typed_axis_proposer(
        primary,
        contrast,
        matrix=matrix,
        request=request,
        runtime=runtime,
        transport=journal,
    )
    assert type(first) is HeadlessTypedAxisAttemptErrorArtifact
    assert first.failure_stage == "payload_contract"
    assert first.model_payload == payload
    assert first.codex_receipt is not None
    assert first.transport_provenance.journal_terminal_status == "success"
    assert first.attempt_custody_authenticated is True
    assert first.benchmark_sealable is False
    assert "benchmark_sealable" not in first.transport_provenance.to_data()
    terminal = journal.verify()
    assert terminal.terminal_status == "success"

    second = run_headless_typed_axis_proposer(
        primary,
        contrast,
        matrix=matrix,
        request=request,
        runtime=runtime,
        transport=journal,
    )
    assert type(second) is HeadlessTypedAxisAttemptErrorArtifact
    assert second == first
    assert second.attempt_digest == first.attempt_digest
    assert physical_calls == 1
    assert journal.fresh_call_count == 1
    assert journal.reused_call_count == 1

    binding = headless_typed_axis_attempt_binding(first)
    assert binding["runner_must_bind_attempt"] is True
    assert binding["omission_or_reroll_allowed"] is False
    assert binding["benchmark_sealable"] is False
    assert binding["attempt_custody_authenticated"] is True
    assert binding["error_is_axis_gap_or_negative_evidence"] is False
    with pytest.raises(HeadlessTypedAxisProposerError, match="journal terminal"):
        verify_headless_typed_axis_attempt_error_artifact(
            first,
            primary,
            contrast,
            matrix=matrix,
            expected_attempt_digest=first.attempt_digest,
        )
    assert verify_headless_typed_axis_attempt_error_artifact(
        first,
        primary,
        contrast,
        matrix=matrix,
        expected_attempt_digest=first.attempt_digest,
        proposer_journal_terminal=terminal,
    ) == first
    assert physical_calls == 1


def test_durable_physical_failure_is_mandatory_and_cannot_reroll(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    primary, contrast = _images()
    matrix = _matrix()
    runtime = _runtime()
    request = _request(primary, contrast, matrix, runtime)
    physical_calls = 0

    def physical_transport(prompt, paths, names, schema, **kwargs):
        nonlocal physical_calls
        physical_calls += 1
        raise RuntimeError("synthetic physical failure")

    monkeypatch.setattr(
        proposer_module, "run_codex_named_images_structured", physical_transport
    )
    journal = build_headless_typed_axis_turn_journal(
        tmp_path / "physical_failure_journal",
        primary,
        contrast,
        matrix=matrix,
        request=request,
        runtime=runtime,
        authorization_digest="sha256:" + "3" * 64,
        execution_precommit_digest="sha256:" + "4" * 64,
        task_id="hd_typed_axis_physical_failure_0001",
        underlying_transport=physical_transport,
    )
    first = run_headless_typed_axis_proposer(
        primary,
        contrast,
        matrix=matrix,
        request=request,
        runtime=runtime,
        transport=journal,
    )
    assert type(first) is HeadlessTypedAxisAttemptErrorArtifact
    assert first.failure_stage == "physical_turn"
    assert first.model_payload is None
    assert first.codex_receipt is None
    assert first.transport_provenance.journal_terminal_status == "failure"
    assert first.attempt_custody_authenticated is True
    assert first.benchmark_sealable is False
    terminal = journal.verify()
    assert terminal.terminal_status == "failure"

    second = run_headless_typed_axis_proposer(
        primary,
        contrast,
        matrix=matrix,
        request=request,
        runtime=runtime,
        transport=journal,
    )
    assert type(second) is HeadlessTypedAxisAttemptErrorArtifact
    assert second == first
    assert physical_calls == 1
    assert journal.fresh_call_count == 1
    assert journal.reused_call_count == 1
    assert verify_headless_typed_axis_attempt_error_artifact(
        first,
        primary,
        contrast,
        matrix=matrix,
        expected_attempt_digest=first.attempt_digest,
        proposer_journal_terminal=terminal,
    ) == first
    assert physical_calls == 1


def test_serialized_policy_cannot_be_flipped_to_authoritative() -> None:
    *_prefix, artifact = _run_injected()
    tampered = deepcopy(artifact.to_data())
    tampered["candidate_selection_authority"] = True
    with pytest.raises(HeadlessTypedAxisProposerError, match="policy differs"):
        HeadlessTypedAxisProposerArtifact.from_data(tampered)

    tampered = deepcopy(artifact.to_data())
    tampered["outcome"]["candidate_selection_authority"] = True
    with pytest.raises(HeadlessTypedAxisProposerError, match="outcome policy"):
        HeadlessTypedAxisProposerArtifact.from_data(tampered)
