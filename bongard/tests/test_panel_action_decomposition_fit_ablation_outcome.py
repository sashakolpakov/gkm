"""Static validation for the typed-decomposition FIT ablation outcome."""

from __future__ import annotations

import json
from pathlib import Path

from bongard.canonical import canonical_digest, canonical_json


OUTCOME = (
    Path(__file__).resolve().parents[1]
    / "data/panel_action_decomposition_fit_ablation_outcome_20260810_v1.json"
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


def test_outcome_binds_live_ablation_and_exact_cold_replay() -> None:
    outcome = _outcome()
    assert outcome["record_digest"] == (
        "sha256:17f2da821f550cb407abb2ee2cc6b858c48941be76f4587cc4127672cd130840"
    )
    assert outcome["source_commit"] == "37604237e7c2f3a918a2ab5d1216529fad8e9456"
    artifacts = outcome["source_artifacts"]
    assert artifacts["result"]["record_digest"] == (
        "sha256:5dfaeefabf1d2b76a74d3eca7e6152f00b2a10cdf95fc8bc678895f53d20ec7f"
    )
    assert artifacts["cold_replay"]["record_digest"] == (
        "sha256:f90dcd9ad0f011fcd42c9517d0169d95c90f3c2af532f41d9e5e3746a04be189"
    )
    execution = outcome["execution"]
    assert execution["task_count"] == execution["physical_calls"] == 4
    assert execution["fit_panel_count"] == 56
    assert execution["typed_panel_error_count"] == 0
    assert execution["new_cohort_pixels_opened"] == 0
    assert execution["calibration_heldout_family_or_target_pixels_opened"] == 0
    assert execution["all_raw_and_threeview_bytes_rebuilt_during_cold_replay"] is True


def test_comparison_records_targeted_repair_without_generic_gain() -> None:
    outcome = _outcome()
    straight = outcome["comparison"]["straight"]
    assert (
        straight["baseline_point_exact_count"],
        straight["baseline_interval_coverage_count"],
        straight["parent_top1_exact_count"],
        straight["parent_finite_set_coverage_count"],
        straight["decomposition_top1_exact_count"],
        straight["decomposition_finite_set_coverage_count"],
    ) == (21, 28, 20, 25, 24, 25)
    assert (
        straight["baseline_task_max_zero_omission_radius"],
        straight["parent_task_max_zero_omission_radius"],
        straight["decomposition_task_max_zero_omission_radius"],
    ) == (8, 6, 5)
    paired = outcome["paired_residual_comparison_to_parent"]["straight"]
    assert paired == {
        "equal_panel_count": 26,
        "improved_panel_count": 18,
        "regressed_panel_count": 12,
    }
    assert len(outcome["targeted_witnesses"]) == 5
    assert all(
        row["new_tuple_candidates"] == [row["truth_tuple"]]
        for row in outcome["targeted_witnesses"]
    )
    acute = outcome["comparison"]["per_task"]["hd_has_acute_angle-necked_0013"]
    assert acute["straight"]["parent_finite_set_coverage_count"] == 8
    assert acute["straight"]["decomposition_finite_set_coverage_count"] == 4


def test_uncertainty_and_count_four_failures_block_release() -> None:
    outcome = _outcome()
    uncertainty = outcome["uncertainty_audit"]
    assert uncertainty["finite_tuple_set_cardinality_distribution"] == {"1": 47, "2": 9}
    assert uncertainty["component_finite_set_marginal_coverage_counts"] == {
        "decorated_arc_count": 48,
        "decorated_straight_count": 41,
        "normal_arc_count": 56,
        "normal_straight_count": 38,
    }
    audit = outcome["count_four_audit"]
    assert audit["decomposition"]["truth_four"] == {
        "absent": 3,
        "indeterminate": 0,
        "present": 3,
    }
    assert audit["decomposition"]["truth_not_four"]["present"] == 5
    assert audit["parent_multiview"]["truth_not_four"]["present"] == 2
    assert outcome["release_disposition"] == (
        "decomposition_fit_observer_not_qualified_for_absence_calibration_or_query_release"
    )


def test_posthoc_union_is_diagnostic_and_supervised_lane_is_next() -> None:
    outcome = _outcome()
    union = outcome["posthoc_union_diagnostic"]
    assert union["straight"] == {
        "finite_set_coverage_count": 33,
        "task_max_zero_omission_radius": 4,
    }
    assert union["not_deployable"] is True
    recommendation = outcome["recommendation"]
    assert recommendation["decision"] == (
        "retire_zero_shot_total_count_and_decomposition_prompt_tuning"
    )
    assert recommendation["next_lane"] == "supervised_renderer_action_program_observer"
    assert recommendation["this_outcome_authorizes_no_more_pixel_or_prompt_ablation"] is True
    assert outcome["authority"]["calibration_transfer_to_query_style_calls"] is False
    assert outcome["authority"]["lean_required"] is False
