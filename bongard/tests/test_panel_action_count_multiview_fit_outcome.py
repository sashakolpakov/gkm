"""Static validation for the multiview FIT outcome and failure audit."""

from __future__ import annotations

import json
from pathlib import Path

from bongard.canonical import canonical_digest, canonical_json


OUTCOME = (
    Path(__file__).resolve().parents[1]
    / "data/panel_action_count_multiview_fit_outcome_20260810_v1.json"
)


def _outcome() -> dict[str, object]:
    payload = OUTCOME.read_bytes()
    value = json.loads(payload)
    assert isinstance(value, dict)
    assert payload == canonical_json(value) + b"\n"
    body = dict(value)
    digest = body.pop("record_digest")
    assert digest == "sha256:" + canonical_digest(body)
    return value


def test_multiview_fit_outcome_binds_live_result_replay_and_authority() -> None:
    outcome = _outcome()
    assert outcome["record_digest"] == (
        "sha256:395c4e3d9c52695fc3f9f5f4c8829f9f270d4d2e60d6ef6c2818f57dcc632488"
    )
    assert outcome["source_digests"]["result"] == (
        "sha256:c4da1c2ccd59b341067e846531fe4972701e6b2f368478dd368b77ac14130604"
    )
    assert outcome["source_digests"]["cold_replay"] == (
        "sha256:430ec87da326309d127dd97d2c2f7a7e0f168b46ce3cf4e8387f7bf6e61dc14b"
    )
    assert outcome["execution"] == {
        "all_280_multiviews_rebuilt_from_raw_during_cold_replay": True,
        "calibration_or_heldout_pixels_opened": False,
        "cold_replay_model_calls": 0,
        "failed_task_count": 0,
        "model_calls": 20,
        "panel_count": 280,
        "successful_task_count": 20,
        "task_count": 20,
    }
    assert outcome["authority"]["fit_protocol_tuning_only"] is True
    assert outcome["authority"]["calibration_transfer_to_query_style_calls"] is False
    assert outcome["authority"]["lean_required"] is False


def test_multiview_fit_metrics_and_count_four_failure_are_explicit() -> None:
    outcome = _outcome()
    straight = outcome["multiview_metrics"]["straight"]
    arc = outcome["multiview_metrics"]["arc"]
    assert (
        straight["top1_exact_count"],
        straight["finite_candidate_set_coverage_count"],
        straight["fallback_interval_coverage_count"],
    ) == (130, 153, 155)
    assert (
        arc["top1_exact_count"],
        arc["finite_candidate_set_coverage_count"],
        arc["fallback_interval_coverage_count"],
    ) == (231, 247, 247)
    assert straight["task_max_zero_omission_radius"] == {
        "fallback_interval": 6,
        "finite_candidate_set": 6,
        "top1": 6,
    }
    audit = outcome["straight_count_four_audit"]
    assert audit["truth_count_four_panel_count"] == 33
    assert audit["multiview"]["present_singleton_four"] == 15
    assert audit["multiview"]["indeterminate_contains_four"] == 6
    assert audit["multiview"]["false_absence_excludes_four"] == 12
    assert outcome["release_disposition"] == (
        "multiview_fit_observer_not_qualified_for_absence_calibration_or_query_release"
    )


def test_failure_attribution_and_next_fit_ablation_are_bounded() -> None:
    outcome = _outcome()
    assert len(outcome["straight_residual_six_rows"]) == 4
    assert all(row["arc_count"] == 0 for row in outcome["straight_residual_six_rows"])
    attribution = outcome["failure_attribution"]
    assert attribution["primary"] == "renderer_grammar_and_action_boundary_topology"
    assert attribution["inference_not_formal_proof"] is True
    representation = outcome["raw_representation_recommendation"]
    assert representation["canonical_raw"].startswith("finite_noncontiguous_candidate_set")
    next_fit = outcome["smallest_next_fit_frontend"]
    assert next_fit["status"] == "recommended_not_executed"
    assert next_fit["physical_calls"] == 4
    assert next_fit["panels"] == 56
    assert next_fit["calibration_or_heldout_calls"] == 0
    assert next_fit["target_calls"] == 0

