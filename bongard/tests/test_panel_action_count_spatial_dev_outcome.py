"""Static validation for the bounded spatial-observer development outcome."""

from __future__ import annotations

import json
from pathlib import Path

from bongard.canonical import canonical_digest, canonical_json


OUTCOME = (
    Path(__file__).resolve().parents[1]
    / "data/panel_action_count_spatial_dev_outcome_20260810_v1.json"
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


def test_spatial_development_outcome_binds_source_and_precommit() -> None:
    outcome = _outcome()
    assert outcome["record_digest"] == (
        "sha256:92f4905b5c002aab7cc7288f60037c538fa761bd4d3b56cb4154ffddf5bcf9d7"
    )
    assert outcome["source"] == {
        "commit": "d79583a3",
        "sha256": "3a5dcf6a707132badc2706187236135a43bb5abe57d3ab80045949d71311b838",
    }
    assert outcome["precommit"]["record_digest"] == (
        "sha256:8bfae0c6368f2faf234150f853891589be5e8f8053f68d3fe7cc8ab2bda17044"
    )
    assert outcome["precommit"]["config_digest"] == (
        "sha256:b25392c9185e1bcaba5faf8d4772bf36acc53874d8d598565bf701a95decc034"
    )


def test_spatial_run_is_a_runtime_gap_with_no_later_exposure() -> None:
    outcome = _outcome()
    execution = outcome["execution"]
    assert outcome["outcome"] == "runtime_efficiency_gap_before_metric_result"
    assert execution["exit_code"] == 130
    assert execution["operator_cutoff_minutes"] == 60
    assert execution["checkpoint_written"] is False
    assert execution["partial_model_written"] is False
    assert execution["result_written"] is False
    assert execution["fresh_calibration_panels_opened"] == 0
    assert execution["fresh_evaluation_panels_opened"] == 0
    assert execution["same_family_panels_opened"] == 0
    assert execution["target_or_query_panels_opened"] == 0
    assert outcome["authority"][
        "calibration_evaluation_family_query_or_target_authorized"
    ] is False
    assert outcome["recommendation"]["do_not_open_calibration_or_target"] is True
    assert outcome["authority"]["lean_required"] is False
