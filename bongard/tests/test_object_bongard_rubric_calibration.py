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
    cold_verify_object_bongard_rubric_calibration,
    load_object_bongard_rubric_calibration_source,
    run_object_bongard_rubric_calibration_observation,
    run_object_bongard_rubric_calibration_observations,
)
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
    assert len(source.rubric_specs) == 2
    assert tuple(item.feature_nominations for item in source.rubric_specs) == (
        (
            "paired_sector_mismatch_support_ppm",
            "bird_like_support_ppm",
        ),
        (
            "bird_like_support_ppm",
            "paired_sector_mismatch_support_ppm",
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
    first_rubric = source.rubric_specs[0].rubric

    def levels(panel, prompt: str) -> int:
        # Spec zero separates A/B perfectly.  Spec one deliberately scores
        # everything high, exercising a canonical language gap.
        if first_rubric in prompt:
            return 4 if panel in source.group_a_panels else 0
        return 4

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
    assert len(batch.runs) == 24
    assert len(calls) == 30
    assert len(tuple((tmp_path / "batch").glob("**/manifest.json"))) == 30

    durable = tmp_path / "batch.json"
    durable.write_bytes(canonical_json(batch.to_data()) + b"\n")
    reloaded = ObjectBongardRubricObservationBatch.from_data(
        json.loads(durable.read_text(encoding="utf-8"))
    )
    assert reloaded == batch

    assessment = assess_object_bongard_rubric_calibration(source, reloaded)
    perfect, gap = assessment.spec_assessments
    assert len(perfect.survivor_candidate_digests) == 2
    assert perfect.gap_kind is None
    for counts in perfect.candidate_counts:
        assert counts.positive.present == 6
        assert counts.negative.certified_absent == 6
        assert counts.support_consistent is True
    assert gap.survivor_candidate_digests == ()
    assert gap.gap_kind is not None
    assert gap.gap_kind.value == "language_gap"
    assert any(
        counts.negative.present == 6 for counts in gap.candidate_counts
    )
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
