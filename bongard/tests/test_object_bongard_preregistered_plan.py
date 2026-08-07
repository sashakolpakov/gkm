from __future__ import annotations

import hashlib
import json
from pathlib import Path

from bongard.canonical import canonical_digest
from bongard.object_bongard_batch import ObjectBongardBatchPlan


DATA = Path(__file__).resolve().parents[1] / "data"
PLAN = DATA / "object_bongard_rubric_train_20260808.plan.json"
PREREG = DATA / "object_bongard_rubric_train_20260808.prereg.json"


def test_rubric_batch_is_preregistered_before_pixels() -> None:
    plan = ObjectBongardBatchPlan.from_data(json.loads(PLAN.read_text()))
    prereg = json.loads(PREREG.read_text())
    body = {key: value for key, value in prereg.items() if key != "record_digest"}

    assert prereg["record_digest"] == "sha256:" + canonical_digest(body)
    assert prereg["batch_plan_digest"] == plan.record_digest
    assert prereg["selection_seed_digest"] == (
        "sha256:"
        + hashlib.sha256(prereg["selection_seed"].encode("utf-8")).hexdigest()
    )
    assert prereg["selection_seed_digest"] == plan.selection_seed_digest
    assert prereg["exposure_predecessor_digest"] == plan.exposure_predecessor_digest
    assert prereg["selection_inputs_include_pixels"] is False
    assert prereg["selection_inputs_include_action_programs"] is False
    assert prereg["panel_bytes_opened_before_preregistration"] is False
    assert prereg["query_identities_sealed_before_support_pixels"] is True
    assert prereg["official_test_authorized"] is False
    assert prereg["python_is_canonical_authority"] is True
    assert prereg["lean_required"] is False
    assert prereg["lean_removable"] is True
    assert len(plan.tasks) == 12
    assert {family: sum(task.family == family for task in plan.tasks) for family in ("bd", "ff", "hd")} == {
        "bd": 4,
        "ff": 4,
        "hd": 4,
    }
    assert all(
        query not in support
        for task in plan.tasks
        for support, query in (
            (task.side_0_support_panel_ids, task.side_0_query_panel_id),
            (task.side_1_support_panel_ids, task.side_1_query_panel_id),
        )
    )
