"""Static validation for the failed supervised action-count FIT outcome."""

from __future__ import annotations

import json
from pathlib import Path

from bongard.canonical import canonical_digest, canonical_json


OUTCOME = (
    Path(__file__).resolve().parents[1]
    / "data/panel_action_count_cnn_fit_outcome_20260810_v3.json"
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


def test_outcome_binds_exact_fit_and_replay() -> None:
    outcome = _outcome()
    assert outcome["record_digest"] == (
        "sha256:e736e8af2dbc1c64b1c3bdd71f04413330bce6b91cd295d77f5ac28ff1fde345"
    )
    fit = outcome["fit"]
    assert fit["fit_result_record_digest"] == (
        "sha256:f8b79047228a91fd3fdd47a262299b0cd683daa727981e568450371be4e4dff2"
    )
    assert fit["replay_record_digest"] == (
        "sha256:69802bf42f429aeeca31f62863b576e8620d646db61afa1410073578cb0008dc"
    )
    assert fit["replay_predictions_exact"] is True
    assert fit["replay_metrics_exact"] is True


def test_outcome_records_failed_preregistered_gate() -> None:
    outcome = _outcome()
    metrics = outcome["metrics"]
    assert metrics["validation"]["straight_top1"] < metrics["thresholds"][
        "straight_top1_at_least"
    ]
    assert metrics["validation"]["arc_top1"] < metrics["thresholds"][
        "arc_top1_at_least"
    ]
    assert metrics["validation"]["catalog_binary_balanced_accuracy"] >= metrics[
        "thresholds"
    ]["catalog_binary_balanced_accuracy_at_least"]
    assert metrics["validation"]["straight_count_four_correct"] == 49
    assert metrics["validation"]["straight_count_four_panel_count"] == 187


def test_failure_is_measurement_gap_and_later_pixels_stay_sealed() -> None:
    outcome = _outcome()
    execution = outcome["execution"]
    assert execution["validation_occurrences_removed_as_exact_training_duplicates"] == 8
    assert execution["effective_training_panel_count"] == 11_200
    assert execution["effective_validation_panel_count"] == 1_392
    assert execution["fresh_v3_calibration_panels_opened"] == 0
    assert execution["fresh_v3_evaluation_panels_opened"] == 0
    assert execution["same_family_calibration_panels_opened"] == 0
    assert execution["target_panels_opened"] == 0
    assert outcome["release_disposition"] == "fit_validation_gap_no_calibration_no_query"
    assert outcome["diagnosis"]["synthesis_or_negation_is_not_the_failure"] is True
    assert outcome["authority"]["lean_required"] is False
