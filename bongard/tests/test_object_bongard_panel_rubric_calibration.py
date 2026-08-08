"""Offline tests for the sealed 24-call whole-panel calibration path."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from threading import Lock

import pytest

from bongard.canonical import canonical_json
from bongard.evidence import Disposition
from bongard.object_bongard_panel_rubric_calibration import (
    ObjectBongardPanelRubricCalibrationAssessment,
    ObjectBongardPanelRubricCalibrationDurableFreeze,
    ObjectBongardPanelRubricCalibrationObservationBatch,
    ObjectBongardPanelRubricFailureEvidence,
    PANEL_RUBRIC_CALIBRATION_JOB_COUNT,
    assess_object_bongard_panel_rubric_calibration,
    bind_object_bongard_panel_rubric_calibration_nomination,
    cold_verify_object_bongard_panel_rubric_calibration,
    load_object_bongard_panel_rubric_calibration_source,
    persist_and_reload_object_bongard_panel_rubric_calibration_batch,
    run_object_bongard_panel_rubric_calibration_observation,
    run_object_bongard_panel_rubric_calibration_observations,
)
from bongard.object_bongard_panel_rubric_observer import (
    object_bongard_panel_rubric_prompt,
)
from bongard.object_bongard_semantics import ObjectBongardSemanticArtifact
from bongard.object_bongard_turn_journal import ObjectBongardTurnRuntime
from bongard.prototype_scene_observer import prototype_scene_transport_source_digest
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
NOMINATION_ROOT = (
    Path(__file__).parents[2]
    / "downloads/ShapeBongard_V2_full/"
    "object_rubric_nomination_20260808_all_support_v10"
)
AUTHORIZATION_DIGEST = "sha256:" + "3" * 64
PRECOMMIT_DIGEST = "sha256:" + "4" * 64


@dataclass(frozen=True)
class _VerifiedNominationFixture:
    artifact: ObjectBongardSemanticArtifact
    authorization_digest: str
    execution_precommit_digest: str
    cold_replay_digest: str
    result_digest: str
    source_digest: str
    accepted: bool


@pytest.fixture(scope="module")
def calibration_plan():
    source = load_object_bongard_panel_rubric_calibration_source(SOURCE_ROOT)
    artifact = ObjectBongardSemanticArtifact.from_data(
        json.loads((NOMINATION_ROOT / "semantic_artifact.json").read_text("utf-8")),
        expected_artifact_digest=(
            "c765cdfaba7315ce04265e2151490a86f25d042347eac5cba8a7fc1282dc7c29"
        ),
    )
    nomination = _VerifiedNominationFixture(
        artifact,
        "sha256:65d2c58cb09bd3e7aeecde0093a50047ccb1676af105559758b589e5cdd368fe",
        "sha256:caaa7aea85d3c35838c0abfbc052743f7fe05a7e52ff817c2a3a1c2e2ba992bd",
        "sha256:b1c20a920e12f4d2e85f42a3cee06d7565e308f52378e5edfb6bc4ee7c9ed6c4",
        "sha256:2e0bcd7e0792641265806ccde66bac1af7f791746cf02051454f57ebf7fac4cf",
        "78c0228d4326dc5e9335fd506e9dce23ec08d2ce4fef6d9a53653b8ab4cbefbe",
        True,
    )
    return bind_object_bongard_panel_rubric_calibration_nomination(
        source, nomination
    )


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


def _transport_for_rank_one_fallback(plan):
    lock = Lock()
    calls: list[tuple[int, int]] = []
    by_png = {item.png_sha256: item for item in plan.source.panels}

    def transport(prompt, paths, names, schema, **_kwargs):
        assert names == ("panel.png",)
        digest = hashlib.sha256(Path(paths[0]).read_bytes()).hexdigest()
        panel = by_png[digest]
        rank = next(
            spec.candidate_rank
            for spec in plan.rubric_specs
            if object_bongard_panel_rubric_prompt(spec) == prompt
        )
        # Rank 0 reproduces the observed failure shape: both sides target-
        # preferred, with one abstention each.  Rank 1 is strict in this
        # synthetic execution, proving deterministic fallback without retries.
        if rank == 0:
            level = 2 if panel.ordinal in (7, 21) else 4
        else:
            level = 4 if panel.group_index == 0 else 0
        payload = {"lower": level, "upper": level}
        with lock:
            calls.append((rank, panel.ordinal))
        return CodexStructuredResult(
            payload, _receipt(prompt, paths, names, schema, payload)
        )

    return transport, calls


def test_exact_source_and_nomination_bind_two_panel_ranks(calibration_plan) -> None:
    plan = calibration_plan
    assert len(plan.source.panels) == 12
    assert len(plan.source.group_0_panels) == len(plan.source.group_1_panels) == 6
    assert tuple(item.candidate_rank for item in plan.rubric_specs) == (0, 1)
    assert plan.rubric_specs[0].target_cue.text == (
        "One decorated figure forms two closed loops touching at one vertex"
    )
    assert plan.rubric_specs[0].foil_cue.text == (
        "One decorated figure forms a closed loop with a dangling branch"
    )
    serialized = canonical_json(plan.to_data()).decode("utf-8")
    assert '"lean_present":false' in serialized
    assert '"lean_required":false' in serialized
    assert '"lean_removable":true' in serialized


def test_24_jobs_freeze_before_labels_rank_one_fallback_and_cold_replay(
    tmp_path: Path, calibration_plan, runtime
) -> None:
    transport, calls = _transport_for_rank_one_fallback(calibration_plan)
    batch = run_object_bongard_panel_rubric_calibration_observations(
        calibration_plan,
        runtime=runtime,
        journal_root=tmp_path / "journals",
        authorization_digest=AUTHORIZATION_DIGEST,
        execution_precommit_digest=PRECOMMIT_DIGEST,
        parallel_workers=4,
        underlying_transport=transport,
    )
    assert len(calls) == PANEL_RUBRIC_CALIBRATION_JOB_COUNT
    assert calls != []
    assert batch.fresh_call_count == 24
    assert batch.reused_call_count == 0
    assert tuple(
        (run.artifact.rubric_spec.candidate_rank, run.artifact.panel_id)
        for run in batch.runs
    ) == tuple(
        (spec.candidate_rank, panel.panel_id)
        for spec in calibration_plan.rubric_specs
        for panel in calibration_plan.source.panels
    )
    blind_bytes = canonical_json(batch.to_data())
    for forbidden in (
        b"target_side",
        b"foil_side",
        b"neutral_group_index_commitment",
        b"selected_candidate_digest",
    ):
        assert forbidden not in blind_bytes
    with pytest.raises(TypeError, match="durable freeze"):
        assess_object_bongard_panel_rubric_calibration(
            calibration_plan, batch  # type: ignore[arg-type]
        )

    frozen = persist_and_reload_object_bongard_panel_rubric_calibration_batch(
        batch, tmp_path / "observation_batch.json"
    )
    assert ObjectBongardPanelRubricCalibrationDurableFreeze.from_data(
        frozen.to_data()
    ) == frozen
    assessment = assess_object_bongard_panel_rubric_calibration(
        calibration_plan, frozen
    )
    assert assessment.version_spaces[0].survivor_candidate_digests == ()
    assert assessment.version_spaces[0].row.count(Disposition.PRESENT) == 10
    assert assessment.version_spaces[0].row.count(Disposition.INDETERMINATE) == 2
    assert assessment.version_spaces[1].strict_survivor_candidate_digests
    assert assessment.slate_selection.selected_rubric_spec is not None
    assert assessment.slate_selection.selected_rubric_spec.candidate_rank == 1
    assert ObjectBongardPanelRubricCalibrationObservationBatch.from_data(
        batch.to_data()
    ) == batch
    assert ObjectBongardPanelRubricCalibrationAssessment.from_data(
        assessment.to_data()
    ) == assessment
    assert cold_verify_object_bongard_panel_rubric_calibration(
        assessment, calibration_plan, frozen
    ) == assessment


def test_transport_failure_keeps_sanitized_actionable_evidence(
    tmp_path: Path, calibration_plan, runtime
) -> None:
    secret = "secret-token-must-not-be-persisted"

    def failed_transport(*_args, **_kwargs):
        raise RuntimeError(
            f"cloud config bundle cache expired; credential={secret}"
        )

    run = run_object_bongard_panel_rubric_calibration_observation(
        calibration_plan,
        calibration_plan.source.panels[0],
        calibration_plan.rubric_specs[0],
        runtime=runtime,
        journal_root=tmp_path / "failure_journal",
        authorization_digest=AUTHORIZATION_DIGEST,
        execution_precommit_digest=PRECOMMIT_DIGEST,
        underlying_transport=failed_transport,
    )
    assert run.artifact.observation.disposition is Disposition.ERROR
    assert run.journal_summary.terminal_status == "failure"
    assert isinstance(run.failure_evidence, ObjectBongardPanelRubricFailureEvidence)
    assert run.failure_evidence.diagnostic_code == "cloud_policy_cache_expired"
    serialized = canonical_json(run.to_data()).decode("utf-8")
    assert secret not in serialized
    assert "cloud config bundle cache expired" not in serialized
    assert run.failure_evidence.message_prefix_sha256 is not None


def test_calibration_module_has_no_atlas_geometry_ranker_or_lean_import() -> None:
    source = Path(__file__).parents[1] / "object_bongard_panel_rubric_calibration.py"
    text = source.read_text("utf-8").lower()
    for forbidden in (
        "prototype_object_hypotheses",
        "prototype_object_lineages",
        "object_bongard_rubric_observer",
        "object_bongard_rubric_ranker",
        "import lean",
    ):
        assert forbidden not in text
