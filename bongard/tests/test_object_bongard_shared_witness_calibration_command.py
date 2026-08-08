"""Offline launch/replay test for the sealed two-pass calibration command."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path

from bongard.object_bongard_shared_witness_calibration_command import (
    ASSESSMENT_FILENAME,
    AUTHORIZATION_FILENAME,
    BATCH_FILENAME,
    CALIBRATION_JOB_COUNT,
    FREEZE_FILENAME,
    PRECOMMIT_FILENAME,
    run_object_bongard_shared_witness_calibration,
    verify_object_bongard_shared_witness_calibration,
)
from bongard.object_bongard_shared_witness_nomination_command import (
    verify_object_bongard_shared_witness_nomination,
)
from bongard import object_bongard_shared_witness_observer as observer
from bongard.tests.no_tools_fixture import canonical_no_tools_runtime
from bongard.tests.test_object_bongard_panel_rubric_calibration import SOURCE_ROOT
from bongard.tests.test_prototype_scene_observer import _receipt
from bongard.transport import (
    PINNED_CODEX_CLI_VERSION,
    CloudPolicyCacheSnapshot,
    CodexStructuredResult,
)


LAUNCHER_DIGEST = "b" * 64
NOMINATION_ROOT = (
    Path(__file__).parents[2]
    / "downloads/ShapeBongard_V2_full/"
    "object_shared_witness_nomination_20260808_v2"
)


def _payload(spec, *, present: bool) -> dict[str, object]:
    cues = observer._neutral_endpoint_cues(spec)
    target_id, foil_id = observer._endpoint_mapping(spec, cues)
    judgments = {
        target_id: "clear" if present else "none",
        foil_id: "none" if present else "clear",
    }
    return {
        "inventory_status": "complete",
        "entities": [
            {
                "entity_id": "e00",
                "scope": "top_level_figure",
                "bbox_q16": {
                    "x0": 1000,
                    "y0": 1000,
                    "x1": 64000,
                    "y1": 64000,
                },
                "locator": "central outlined figure",
                "anchor_support": "clear",
                "anchor_evidence": "coherent outlined figure is visible",
                "cue_support": [
                    {
                        "cue_id": cue.cue_id,
                        "judgment": judgments[cue.cue_id],
                        "evidence": "visible junction arrangement",
                    }
                    for cue in cues
                ],
            }
        ],
    }


def test_exact_48_fresh_calls_freeze_then_model_free_cold_replay(
    tmp_path: Path,
) -> None:
    nomination = verify_object_bongard_shared_witness_nomination(
        NOMINATION_ROOT, source_root=SOURCE_ROOT
    )
    specs = tuple(
        observer.ObjectBongardSharedWitnessRubricSpec.from_contrast(
            nomination.artifact.artifact_digest, contrast
        )
        for contrast in nomination.artifact.contrast_candidates
    )
    prompt_to_spec = {
        observer.object_bongard_shared_witness_panel_prompt(spec): spec
        for spec in specs
    }
    per_rank_calls: dict[int, int] = defaultdict(int)
    physical_calls = 0
    root = tmp_path / "shared_witness_calibration"

    def transport(prompt, paths, names, schema, **_kwargs):
        nonlocal physical_calls
        spec = prompt_to_spec[prompt]
        within_rank = per_rank_calls[spec.candidate_rank]
        per_rank_calls[spec.candidate_rank] += 1
        physical_calls += 1
        assert (root / AUTHORIZATION_FILENAME).is_file()
        assert (root / PRECOMMIT_FILENAME).is_file()
        assert not (root / BATCH_FILENAME).exists()
        assert names == ["panel.png"] or tuple(names) == ("panel.png",)
        payload = _payload(spec, present=within_rank % 12 < 6)
        return CodexStructuredResult(
            payload, _receipt(prompt, paths, names, schema, payload)
        )

    catalog, attestation = canonical_no_tools_runtime(LAUNCHER_DIGEST)
    cache = CloudPolicyCacheSnapshot(None)

    def fingerprinter(executable, *, expected_launcher_digest):
        assert executable == "codex"
        assert expected_launcher_digest == LAUNCHER_DIGEST
        return {
            "version": PINNED_CODEX_CLI_VERSION,
            "launcher_digest": LAUNCHER_DIGEST,
        }

    launched = run_object_bongard_shared_witness_calibration(
        root,
        nomination_root=NOMINATION_ROOT,
        source_root=SOURCE_ROOT,
        parallel_workers=1,
        expected_launcher_sha256=LAUNCHER_DIGEST,
        transport=transport,
        cache_snapshotter=lambda: cache,
        catalog_snapshotter=lambda: catalog,
        launcher_fingerprinter=fingerprinter,
        runtime_attester=lambda **_kwargs: attestation,
    )
    assert physical_calls == CALIBRATION_JOB_COUNT == 48
    assert dict(per_rank_calls) == {0: 24, 1: 24}
    assert launched.accepted is True
    assert launched.selected_candidate_rank == 0
    assert launched.fresh_call_count == 48
    assert launched.reused_call_count == 0

    batch = json.loads((root / BATCH_FILENAME).read_text("utf-8"))
    freeze = json.loads((root / FREEZE_FILENAME).read_text("utf-8"))
    assessment = json.loads((root / ASSESSMENT_FILENAME).read_text("utf-8"))
    assert batch["support_labels_present"] is False
    assert batch["fresh_call_count"] == 48
    assert len(batch["runs"]) == 48
    assert freeze["batch_fsynced_and_reloaded"] is True
    assert freeze["support_labels_introduced"] is False
    assert assessment[
        "support_labels_first_introduced_after_durable_freeze"
    ] is True
    assert assessment["selected_candidate_rank"] == 0
    assert all(
        item["accepted_in_both_passes"]
        for item in assessment["rank_assessments"]
    )
    assert all(
        item["cross_pass_flip_count"] == 0
        for item in assessment["rank_assessments"]
    )

    replayed = verify_object_bongard_shared_witness_calibration(
        root,
        nomination_root=NOMINATION_ROOT,
        source_root=SOURCE_ROOT,
    )
    assert replayed == launched
    assert physical_calls == 48
    result = json.loads((root / "result.json").read_text("utf-8"))
    assert result["nomination_result_digest"] == nomination.result_digest
    assert result["nomination_cold_replay_digest"] == nomination.cold_replay_digest
    assert result["historical_source_digest"] == launched.source_digest
    assert result["cold_replay_digest"] == launched.replay_digest
    assert result["campaign_gate_lineage_complete"] is True
    assert result["lean_required"] is False
    assert result["lean_removable"] is True
