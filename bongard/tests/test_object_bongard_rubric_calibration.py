"""Offline replay tests for the exact twelve-panel rubric calibration."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from threading import Lock
from unittest.mock import patch

import pytest

from bongard.canonical import canonical_digest, canonical_json
from bongard.evidence import Disposition
from bongard import object_bongard_rubric_calibration as calibration
from bongard.object_bongard_rubric_calibration import (
    CALIBRATION_SELECTED_ORDINALS,
    ObjectBongardRubricCalibrationAssessment,
    ObjectBongardRubricLiveObservation,
    ObjectBongardRubricObservationBatch,
    assess_object_bongard_rubric_calibration,
    _bind_object_bongard_rubric_calibration_nomination_content,
    cold_verify_object_bongard_rubric_calibration,
    load_object_bongard_rubric_calibration_source,
    run_object_bongard_rubric_calibration_observation,
    run_object_bongard_rubric_calibration_observations,
)
from bongard.object_bongard_semantics import describe_object_bongard_support
from bongard.object_bongard_rubric_observer import (
    object_bongard_catalog_contrast_rubric,
)
from bongard.object_bongard_turn_journal import ObjectBongardTurnRuntime
from bongard.prototype_object_hypotheses import (
    ObjectHypothesisPacket,
    object_hypothesis_extractor_artifact_digest,
)
from bongard.prototype_object_lineages import object_lineage_artifact_digest
from bongard.prototype_object_scene_observer import (
    prototype_scene_transport_source_digest,
)
from bongard.tests.test_prototype_scene_observer import (
    EFFORT,
    LAUNCHER_DIGEST,
    MODEL,
    MODEL_CATALOG,
    NO_TOOLS_ATTESTATION,
    _receipt,
)
from bongard.transport import CodexStructuredResult


SOURCE_ROOT = (
    Path(__file__).parents[2]
    / "downloads/ShapeBongard_V2_full/"
    "prototype_pair_python_campaign_20260807_object_v1/objects"
)
AUTHORIZATION_DIGEST = "sha256:" + "a" * 64
PRECOMMIT_DIGEST = "sha256:" + "c" * 64


@pytest.fixture(scope="module")
def exact_source():
    # Regression guard: current geometry must be recomputed once per pinned
    # PNG.  The historical packet mapping must not become current authority.
    original = calibration.extract_object_hypothesis_packet
    calls = 0

    def recompute(png_bytes: bytes):
        nonlocal calls
        calls += 1
        return original(png_bytes)

    with patch.object(
        calibration, "extract_object_hypothesis_packet", side_effect=recompute
    ):
        source = load_object_bongard_rubric_calibration_source(SOURCE_ROOT)
    return source, calls


@pytest.fixture(scope="module")
def runtime() -> ObjectBongardTurnRuntime:
    return ObjectBongardTurnRuntime(
        model=MODEL,
        reasoning_effort=EFFORT,
        minutes=15,
        verbose=False,
        executable="codex",
        cloud_policy_cache_snapshot=None,
        model_catalog_snapshot=MODEL_CATALOG,
        expected_launcher_digest=LAUNCHER_DIGEST,
        no_tools_attestation=NO_TOOLS_ATTESTATION,
        transport_source_digest=prototype_scene_transport_source_digest(),
    )


def _fake_transport(source, level_policy):
    lock = Lock()
    calls: list[tuple[str, str]] = []
    by_png = {item.png_sha256: item for item in source.panels}

    def transport(prompt, paths, names, schema, **kwargs):
        panel_digest = hashlib.sha256(Path(paths[0]).read_bytes()).hexdigest()
        panel = by_png[panel_digest]
        sheet = next(
            item
            for item in panel.hypothesis_packet.atlas_sheets
            if item.name == names[1]
        )
        level = level_policy(panel, prompt)
        payload = {
            "scene": {"lower": level, "upper": level},
            "slots": [
                {"slot_id": slot.slot_id, "lower": level, "upper": level}
                for slot in sheet.slots
            ],
        }
        with lock:
            calls.append((panel.panel_id, sheet.name))
        return CodexStructuredResult(
            payload, _receipt(prompt, paths, names, schema, payload)
        )

    return transport, calls


def test_exact_source_recomputes_current_geometry_and_preserves_history(
    exact_source,
) -> None:
    source, recomputation_count = exact_source
    assert recomputation_count == 12
    assert tuple(item.ordinal for item in source.panels) == CALIBRATION_SELECTED_ORDINALS
    assert len(source.group_a_panels) == len(source.group_b_panels) == 6
    assert len(source.rubric_specs) == 1
    assert tuple(item.feature_nominations for item in source.rubric_specs) == (
        (
            "paired_sector_mismatch_support_ppm",
            "bird_like_support_ppm",
        ),
    )
    assert tuple(item.rubric for item in source.rubric_specs) == tuple(
        object_bongard_catalog_contrast_rubric(*item.feature_nominations)
        for item in source.rubric_specs
    )
    for panel in source.panels:
        observer_path = (
            SOURCE_ROOT
            / "observer_artifact"
            / f"{panel.source_observer_file_sha256}.json"
        )
        historical = json.loads(observer_path.read_text(encoding="utf-8"))
        assert canonical_digest(historical["hypothesis_packet"]) == (
            panel.historical_hypothesis_packet_digest
        )
        assert panel.hypothesis_packet.extractor_artifact_digest == (
            object_hypothesis_extractor_artifact_digest()
        )
        assert panel.lineage_packet.extractor_artifact_digest == (
            object_lineage_artifact_digest()
        )
        assert ObjectHypothesisPacket.from_data(
            panel.hypothesis_packet.to_data()
        ) == panel.hypothesis_packet
        commitment = panel.commitment_data()
        assert commitment["historical_hypothesis_packet_digest"] == (
            panel.historical_hypothesis_packet_digest
        )
        assert commitment["current_hypothesis_packet_digest"] == (
            panel.hypothesis_packet.digest()
        )
    data = source.to_data()
    assert data["labels_consumed_while_observing"] is False
    assert data["fresh_broad_cohort_pixels_opened"] is False
    assert data["lean_required"] is False


def test_source_rejects_an_independently_queried_reverse_orientation(
    exact_source,
) -> None:
    source, _ = exact_source
    forward = source.rubric_specs[0]
    target, foil = forward.feature_nominations
    reverse = type(forward).create(
        forward.semantic_artifact_digest,
        object_bongard_catalog_contrast_rubric(foil, target),
        (foil, target),
    )
    with pytest.raises(
        calibration.ObjectBongardRubricCalibrationError,
        match="canonical frozen orientation",
    ):
        type(source)(
            historical_plan_file_sha256=source.historical_plan_file_sha256,
            historical_plan_record_digest=source.historical_plan_record_digest,
            historical_description_file_sha256=(
                source.historical_description_file_sha256
            ),
            historical_description_artifact_digest=(
                source.historical_description_artifact_digest
            ),
            panels=source.panels,
            rubric_specs=(forward, reverse),
            nomination_artifact=source.nomination_artifact,
            nomination_authorization_digest=(
                source.nomination_authorization_digest
            ),
            nomination_precommit_digest=source.nomination_precommit_digest,
            nomination_replay_digest=source.nomination_replay_digest,
            nomination_result_digest=source.nomination_result_digest,
            source_digest=source.source_digest,
        )


def test_internal_verified_nomination_content_replaces_historical_cues_exactly(
    exact_source, runtime
) -> None:
    source, _ = exact_source
    group_0 = tuple(sorted(item.panel_id for item in source.group_a_panels))
    group_1 = tuple(sorted(item.panel_id for item in source.group_b_panels))
    pngs = {item.panel_id: item.exact_png_bytes for item in source.panels}
    payload = {
        "profiles": [
            {
                "group_id": "group_0",
                "rubric": "A mismatched pair of sector-like subshapes recurs.",
                "feature_ids": ["paired_sector_mismatch_support_ppm"],
            },
            {
                "group_id": "group_1",
                "rubric": "A triangle accompanied by three line-like spans recurs.",
                "feature_ids": ["triangle_with_three_lines_support_ppm"],
            },
        ]
    }

    def transport(prompt, paths, names, schema, **_kwargs):
        return CodexStructuredResult(
            payload, _receipt(prompt, paths, names, schema, payload)
        )

    artifact = describe_object_bongard_support(
        task_id=source.panels[0].task_id,
        group_0_panel_ids=group_0,
        group_1_panel_ids=group_1,
        support_png_by_panel_id=pngs,
        observation_context_digest=PRECOMMIT_DIGEST,
        model=runtime.model,
        reasoning_effort=runtime.reasoning_effort,
        minutes=runtime.minutes,
        verbose=runtime.verbose,
        executable=runtime.executable,
        cloud_policy_cache_snapshot=runtime.cloud_policy_cache_snapshot,
        expected_launcher_digest=runtime.expected_launcher_digest,
        model_catalog_snapshot=runtime.model_catalog_snapshot,
        no_tools_attestation=runtime.no_tools_attestation,
        transport=transport,
    )
    nominated = _bind_object_bongard_rubric_calibration_nomination_content(
        source,
        artifact,
        nomination_authorization_digest=AUTHORIZATION_DIGEST,
        nomination_precommit_digest=PRECOMMIT_DIGEST,
        nomination_replay_digest="sha256:" + "e" * 64,
        nomination_result_digest="sha256:" + "f" * 64,
    )
    assert tuple(item.feature_nominations for item in nominated.rubric_specs) == (
        (
            "paired_sector_mismatch_support_ppm",
            "triangle_with_three_lines_support_ppm",
        ),
    )
    assert nominated.to_data()[
        "historical_description_used_for_rubric_derivation"
    ] is False
    assert nominated.to_data()["nomination_binding"]["context_task_id_policy"] == (
        "lowest-selected-ordinal-task-id-is-transport-context-only"
    )


def test_one_sheet_turn_is_exactly_once_resumable_and_serializable(
    exact_source, runtime, tmp_path: Path
) -> None:
    source, _ = exact_source
    panel = source.panels[0]
    spec = source.rubric_specs[0]
    transport, calls = _fake_transport(source, lambda _panel, _prompt: 3)
    first = run_object_bongard_rubric_calibration_observation(
        panel,
        spec,
        runtime=runtime,
        journal_root=tmp_path,
        authorization_digest=AUTHORIZATION_DIGEST,
        execution_precommit_digest=PRECOMMIT_DIGEST,
        underlying_transport=transport,
    )
    assert first.fresh_call_count == len(panel.hypothesis_packet.atlas_sheets)
    assert first.reused_call_count == 0
    assert len(calls) == len(panel.hypothesis_packet.atlas_sheets)
    assert ObjectBongardRubricLiveObservation.from_data(first.to_data()) == first

    resumed = run_object_bongard_rubric_calibration_observation(
        panel,
        spec,
        runtime=runtime,
        journal_root=tmp_path,
        authorization_digest=AUTHORIZATION_DIGEST,
        execution_precommit_digest=PRECOMMIT_DIGEST,
        underlying_transport=transport,
    )
    assert resumed.artifact == first.artifact
    assert resumed.fresh_call_count == 0
    assert resumed.reused_call_count == len(panel.hypothesis_packet.atlas_sheets)
    assert len(calls) == len(panel.hypothesis_packet.atlas_sheets)


def test_parallel_batch_persists_assesses_gaps_and_cold_replays_without_calls(
    exact_source, runtime, tmp_path: Path
) -> None:
    source, _ = exact_source
    def levels(panel, _prompt: str) -> int:
        return 4 if panel in source.group_a_panels else 0

    transport, calls = _fake_transport(source, levels)
    batch = run_object_bongard_rubric_calibration_observations(
        source,
        runtime=runtime,
        journal_root=tmp_path / "batch",
        authorization_digest=AUTHORIZATION_DIGEST,
        execution_precommit_digest=PRECOMMIT_DIGEST,
        parallel_workers=4,
        underlying_transport=transport,
    )
    assert len(batch.runs) == 12
    assert len(calls) == 15
    assert len(tuple((tmp_path / "batch").glob("**/manifest.json"))) == 15

    durable = tmp_path / "batch.json"
    durable.write_bytes(canonical_json(batch.to_data()) + b"\n")
    reloaded = ObjectBongardRubricObservationBatch.from_data(
        json.loads(durable.read_text(encoding="utf-8"))
    )
    assert reloaded == batch

    assessment = assess_object_bongard_rubric_calibration(source, reloaded)
    (perfect,) = assessment.spec_assessments
    assert len(perfect.survivor_candidate_digests) == 2
    assert perfect.gap_kind is None
    for counts in perfect.candidate_counts:
        assert counts.positive.present == 6
        assert counts.negative.certified_absent == 6
        assert counts.support_consistent is True
    assert ObjectBongardRubricCalibrationAssessment.from_data(
        assessment.to_data()
    ) == assessment

    calls_before_replay = len(calls)
    assert cold_verify_object_bongard_rubric_calibration(
        assessment, source, reloaded
    ) == assessment
    assert len(calls) == calls_before_replay


def test_calibration_driver_has_no_lean_import() -> None:
    source_path = Path(__file__).parents[1] / "object_bongard_rubric_calibration.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imports.append(node.module)
    assert not any("lean" in item.lower() for item in imports)
