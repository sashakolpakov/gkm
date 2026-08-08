from __future__ import annotations

import hashlib
import json
from pathlib import Path

from bongard.canonical import canonical_digest
from bongard.cohorts import classify_task
from bongard.exposure import basic_morphology_cluster_id, semantic_policy_blocked_keys
from bongard.historical_exposure import load_historical_exposure
from bongard.object_bongard_drill_batch import ObjectBongardDrillBatchPlan
from bongard.object_bongard_scene_predicate_campaign_command import (
    DEFAULT_PLAN,
    DEFAULT_PREREGISTRATION,
    _load_exact_cohort,
)


DATA = Path(__file__).resolve().parents[1] / "data"
PLAN = DATA / "object_bongard_scene_drill_train_20260808.plan.json"
PREREG = DATA / "object_bongard_scene_drill_train_20260808.prereg.json"


def test_scene_drill_batch_is_preregistered_before_pixels() -> None:
    plan = ObjectBongardDrillBatchPlan.from_data(json.loads(PLAN.read_text()))
    prereg = json.loads(PREREG.read_text())
    body = {key: value for key, value in prereg.items() if key != "record_digest"}

    assert prereg["record_digest"] == "sha256:" + canonical_digest(body)
    assert prereg["batch_plan_digest"] == plan.record_digest
    assert prereg["selection_seed_digest"] == (
        "sha256:"
        + hashlib.sha256(prereg["selection_seed"].encode("utf-8")).hexdigest()
    )
    assert prereg["selection_seed_digest"] == plan.selection_seed_digest
    assert prereg["families"] == ["bd", "hd"]
    assert prereg["semantic_cohort"] == "drill"
    assert prereg["freeform_policy"] == (
        "excluded-no-certified-unused-semantic-partition"
    )
    assert prereg["selection_inputs_include_pixels"] is False
    assert prereg["selection_inputs_include_action_programs"] is False
    assert prereg["panel_bytes_opened_before_preregistration"] is False
    assert prereg["query_identities_sealed_before_support_pixels"] is True
    assert prereg["semantic_policy_replay_required_before_output_root"] is True
    assert prereg["official_test_authorized"] is False
    assert prereg["python_is_canonical_authority"] is True
    assert prereg["lean_required"] is False
    assert prereg["lean_removable"] is True

    assert plan.prepolicy_candidate_counts == (("bd", 774), ("hd", 1148))
    assert plan.morphology_excluded_counts == (("bd", 660), ("hd", 0))
    assert plan.candidate_counts == (("bd", 114), ("hd", 1148))
    assert plan.generator_cluster_counts == (("bd", 105), ("hd", 67))
    assert plan.eligible_task_ids_digest == (
        "sha256:009fb45160d80f7d90f6d775953b287742b0d325e32d7f3a0b9513ee9ce0d4da"
    )
    assert len(plan.tasks) == 12
    assert {family: sum(task.family == family for task in plan.tasks) for family in ("bd", "hd")} == {
        "bd": 6,
        "hd": 6,
    }
    assert not any(task.family == "ff" for task in plan.tasks)

    historical = load_historical_exposure()
    blocked = {
        key.concepts[0]
        for key in semantic_policy_blocked_keys(historical)
        if key.kind == "basic_morphology_cluster"
    }
    for task in plan.tasks:
        record = classify_task(task.task_id, historical, split=task.split)
        assert task.split == "train"
        assert record.historically_clean
        assert record.semantic_cohort == "drill"
        if task.family == "bd":
            assert not any(
                basic_morphology_cluster_id(concept) in blocked
                for concept in record.parsed.concepts
            )
        for support, query in (
            (task.side_0_support_panel_ids, task.side_0_query_panel_id),
            (task.side_1_support_panel_ids, task.side_1_query_panel_id),
        ):
            assert len(support) == 6
            assert query not in support
            assert len(set(support) | {query}) == 7


def test_scene_campaign_loader_is_pinned_to_the_strict_drill_artifacts() -> None:
    preregistration, plan = _load_exact_cohort(
        DEFAULT_PREREGISTRATION,
        DEFAULT_PLAN,
    )
    assert Path(DEFAULT_PREREGISTRATION).name == PREREG.name
    assert Path(DEFAULT_PLAN).name == PLAN.name
    assert preregistration["batch_plan_digest"] == plan.record_digest
    assert plan.requested_per_family == 6
    assert {task.family for task in plan.tasks} == {"bd", "hd"}
