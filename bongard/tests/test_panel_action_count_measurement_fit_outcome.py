"""Static validation for the 2026-08-09 action-count fit outcome."""

from __future__ import annotations

import json
from pathlib import Path

from bongard.artifacts import canonical_digest, canonical_json


OUTCOME = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "panel_action_count_measurement_fit_outcome_20260809_v1.json"
)


def _outcome() -> dict[str, object]:
    payload = OUTCOME.read_bytes()
    value = json.loads(payload)
    assert isinstance(value, dict)
    assert payload == canonical_json(value) + b"\n"
    return value


def test_fit_outcome_is_canonical_and_self_addressed() -> None:
    outcome = _outcome()
    assert set(outcome) == {
        "authority",
        "conclusions",
        "custody",
        "execution",
        "metrics",
        "record_digest",
        "record_digest_policy",
        "source_artifacts",
        "source_digests",
        "strata",
        "schema",
        "summary_construction",
    }
    assert outcome["schema"] == (
        "gkm.bongard-panel-action-count-measurement-fit-outcome.v1"
    )
    content = dict(outcome)
    declared = content.pop("record_digest")
    assert declared == "sha256:" + canonical_digest(content) == (
        "sha256:385a7a9bd33a54a0edfcf11b24468b22a67609f24511be28571042e4e337d40a"
    )


def test_fit_result_replay_plan_and_execution_are_exactly_bound() -> None:
    outcome = _outcome()
    assert outcome["source_artifacts"] == {
        "cold_replay": {
            "byte_count": 966,
            "raw_sha256": (
                "907a10d835bd68e78f1979def1b21f498537fb5393a58e1c064c87881133a5b9"
            ),
            "record_digest": (
                "sha256:98897b1d41718f774299b04994655b5740361f05b306960ad129a006a8847c5d"
            ),
            "relative_path": (
                "downloads/ShapeBongard_V2_full/"
                "panel_action_count_measurement_20260809_v1/fit/cold_replay.json"
            ),
            "schema": "gkm.bongard-action-count-phase-cold-replay.v1",
        },
        "fit_result": {
            "byte_count": 17_906,
            "raw_sha256": (
                "10450871bfbdfb83985cf02510d547f03f71578f20dddd85314c54ad8f744e61"
            ),
            "record_digest": (
                "sha256:2dc46a24c153e4a1d84b8e53f7a596052a864674bccde1433ba90199e9a403ff"
            ),
            "relative_path": (
                "downloads/ShapeBongard_V2_full/"
                "panel_action_count_measurement_20260809_v1/fit/result.json"
            ),
            "schema": "gkm.bongard-action-count-measurement-result.v1",
        },
    }
    assert outcome["source_digests"]["plan"] == (
        "sha256:476ff0d602d43ddc6e4c8c6a964939a01c8471936eece71d0caba8a35bda396a"
    )
    assert outcome["execution"] == {
        "failed_task_count": 0,
        "model_calls": 20,
        "panel_count": 280,
        "successful_task_count": 20,
        "task_count": 20,
    }


def test_fit_metrics_and_key_strata_are_bound() -> None:
    outcome = _outcome()
    assert outcome["metrics"] == {
        "arc": {
            "coverage_count": 220,
            "denominator": 280,
            "error_count": 0,
            "exact_count": 214,
            "interval_width_sum": 18,
            "valid_interval_count": 280,
        },
        "straight": {
            "coverage_count": 139,
            "denominator": 280,
            "error_count": 0,
            "exact_count": 125,
            "interval_width_sum": 41,
            "valid_interval_count": 280,
        },
    }
    strata = outcome["strata"]
    assert strata["straight_action_count"]["4"] == {
        "arc_coverage_count": 28,
        "arc_exact_count": 28,
        "arc_interval_width_sum": 0,
        "denominator": 33,
        "straight_coverage_count": 19,
        "straight_exact_count": 19,
        "straight_interval_width_sum": 2,
    }
    assert strata["arc_presence"] == {
        "with_arc": {
            "arc_coverage_count": 111,
            "arc_exact_count": 105,
            "arc_interval_width_sum": 17,
            "denominator": 148,
            "straight_coverage_count": 69,
            "straight_exact_count": 66,
            "straight_interval_width_sum": 18,
        },
        "without_arc": {
            "arc_coverage_count": 109,
            "arc_exact_count": 109,
            "arc_interval_width_sum": 1,
            "denominator": 132,
            "straight_coverage_count": 70,
            "straight_exact_count": 59,
            "straight_interval_width_sum": 23,
        },
    }
    decorations = strata["line_decoration_profile"]
    assert {
        name: (
            row["straight_exact_count"],
            row["straight_coverage_count"],
            row["denominator"],
            row["straight_interval_width_sum"],
        )
        for name, row in decorations.items()
    } == {
        "decorated_only": (14, 15, 34, 5),
        "mixed_normal_and_decorated": (26, 27, 76, 11),
        "no_straight_actions": (19, 19, 20, 1),
        "normal_only": (66, 78, 150, 24),
    }


def test_fit_outcome_is_oracle_tainted_and_rejects_absence_use() -> None:
    outcome = _outcome()
    assert outcome["authority"] == {
        "engineering_only": True,
        "lean_present": False,
        "lean_removable": True,
        "lean_required": False,
        "python_is_canonical_authority": True,
        "scientific_benchmark": False,
    }
    assert outcome["custody"] == {
        "action_labels_model_visible": False,
        "labels_opened_only_after_receipted_prediction_batch_fsync": True,
        "oracle_tainted_selected_task_count": 20,
        "selected_tasks_permanently_oracle_tainted": True,
        "target_family_excluded": True,
        "target_family_selected_task_count": 0,
        "target_family_task_ids_digest": (
            "sha256:810c0faaaa70934ffe7055e034f6359c46cb2180f179cbaa74387fa5db1b5f0f"
        ),
    }
    conclusions = outcome["conclusions"]
    assert conclusions["overconfident"]["asserted"] is True
    assert conclusions["unfit_for_absence"]["asserted"] is True
    assert conclusions["release_disposition"] == (
        "fit_observer_not_qualified_for_target_support_absence_or_query_release"
    )
    assert outcome["summary_construction"] == {
        "model_calls_made": 0,
        "panel_pixels_read": False,
        "source_json_files_read": 2,
    }
