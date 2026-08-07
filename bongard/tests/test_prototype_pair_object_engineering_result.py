from __future__ import annotations

import json
from pathlib import Path

from bongard.artifacts import canonical_digest, canonical_json


RESULT = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "prototype_pair_object_engineering_20260807_result_v1.json"
)


def test_object_engineering_result_is_canonical_bound_and_fail_closed() -> None:
    payload = RESULT.read_bytes()
    result = json.loads(payload)
    assert payload == canonical_json(result) + b"\n"

    content = dict(result)
    declared_digest = content.pop("record_digest")
    assert declared_digest == "sha256:" + canonical_digest(content)

    assert result["campaign"]["status"] == "calibration_gap"
    assert result["calibration"]["all_four_bounds_accepted"] is False
    assert all(not bound["accepted"] for bound in result["calibration"]["bounds"])
    assert result["campaign"]["query_panels_released"] is False
    assert result["cold_replay"]["terminal_cli_exit_code"] == 0
    assert result["cold_replay"]["model_calls"] == 0

    authority = result["runtime_authority"]
    assert authority["python_is_canonical_authority"] is True
    assert authority["lean_required"] is False
    assert authority["lean_affects_identity_or_decision"] is False

    qualification = result["qualification"]
    assert qualification["benchmark_claim_authorized"] is False
    assert qualification["official_test_split_used"] is False
    assert qualification["query_accuracy_score_available"] is False

    confusion = result["calibration"]["raw_confusion"]
    assert sum(row["certified_absent"] for row in confusion) == 0
    assert sum(row["present"] + row["indeterminate"] + row["error"] for row in confusion) == 56
    assert result["observer"]["typed_errors_are_excluded_from_negative_evidence"] is True
    assert result["corpus_after_campaign"]["exact_unused_non_test_task_count"] == 9_922
