from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from bongard.object_bongard_scene_predicate_campaign_command import (
    DISCOVERY_CALLS_PER_TASK,
    EXPOSURE_PREDECESSOR_FILE_SHA256,
    MAX_VISUAL_CALLS,
    ObjectBongardScenePredicateCampaignBudget,
    ObjectBongardScenePredicateCampaignCommandError,
    ObjectBongardScenePredicateQueryPhase,
    ObjectBongardScenePredicateTaskFreeze,
    QUERY_DENOMINATOR,
    REGISTERED_A_CALLS_PER_TASK,
    REGISTERED_B_CALLS_PER_TASK,
    TASK_COUNT,
    _CallBudget,
    _authority_data,
    commit_and_release_object_bongard_scene_predicate_queries,
    prepare_object_bongard_scene_predicate_campaign,
    replay_object_bongard_scene_predicate_query_phase,
    verify_object_bongard_scene_predicate_exposure_transition,
)


ADDRESS_1 = "sha256:" + "1" * 64
ADDRESS_2 = "sha256:" + "2" * 64
ADDRESS_3 = "sha256:" + "3" * 64
ADDRESS_4 = "sha256:" + "4" * 64
RAW_5 = "5" * 64
RAW_6 = "6" * 64


def test_rejected_calibration_is_the_only_touched_dependency(tmp_path: Path) -> None:
    calls: list[str] = []

    def reject(_root: object, **_kwargs: object) -> object:
        calls.append("calibration")
        return SimpleNamespace(accepted=False, status="rejected")

    class PoisonPath:
        def __fspath__(self) -> str:
            calls.append("forbidden-path")
            raise AssertionError("cohort/archive path touched after rejected gate")

    output = tmp_path / "must-not-exist"
    with pytest.raises(
        ObjectBongardScenePredicateCampaignCommandError,
        match="not accepted and cold-verified",
    ):
        prepare_object_bongard_scene_predicate_campaign(
            output,
            calibration_root="synthetic-rejected",
            calibration_verifier=reject,
            preregistration_path=PoisonPath(),
            plan_path=PoisonPath(),
            descriptor_path=PoisonPath(),
            archive_path=PoisonPath(),
            split_path=PoisonPath(),
            exposure_predecessor_path=PoisonPath(),
        )
    assert calls == ["calibration"]
    assert not output.exists()


def test_exact_one_event_12_task_exposure_transition_is_required() -> None:
    old_event = SimpleNamespace(task_ids=("already",), panel_ids=())
    predecessor = SimpleNamespace(events=(old_event,))
    tasks = tuple(SimpleNamespace(task_id=f"opaque-task-{index:02d}") for index in range(12))
    plan = SimpleNamespace(tasks=tasks)
    new_event = SimpleNamespace(
        task_ids=tuple(task.task_id for task in tasks), panel_ids=()
    )
    successor = SimpleNamespace(events=(old_event, new_event), digest=ADDRESS_1)
    receipt = SimpleNamespace(
        object_kind="exposure-successor",
        object_digest=ADDRESS_1,
        record_digest=ADDRESS_2,
    )
    authorization = SimpleNamespace(
        exposure_successor_digest=ADDRESS_1,
        exposure_store_receipt_digest=ADDRESS_2,
    )
    prepared = SimpleNamespace(
        successor=successor,
        exposure_receipt=receipt,
        authorization=authorization,
    )
    verify_object_bongard_scene_predicate_exposure_transition(
        predecessor=predecessor, plan=plan, prepared=prepared
    )

    bad = SimpleNamespace(
        successor=SimpleNamespace(
            events=(old_event, replace_event(new_event, panel_ids=("leak",)),),
            digest=ADDRESS_1,
        ),
        exposure_receipt=receipt,
        authorization=authorization,
    )
    with pytest.raises(
        ObjectBongardScenePredicateCampaignCommandError,
        match="exposure transition differs",
    ):
        verify_object_bongard_scene_predicate_exposure_transition(
            predecessor=predecessor, plan=plan, prepared=bad
        )


def replace_event(event: SimpleNamespace, **changes: object) -> SimpleNamespace:
    return SimpleNamespace(**{**vars(event), **changes})


def test_formula_freeze_and_commit_are_durable_before_exactly_two_queries() -> None:
    events: list[str] = []
    task = SimpleNamespace(
        task_id="opaque-task-00",
        record_digest=ADDRESS_1,
        side_0_query_panel_id="sealed-query-a",
        side_1_query_panel_id="sealed-query-b",
    )
    prepared = SimpleNamespace(
        store=object(), precommit=SimpleNamespace(record_digest=ADDRESS_2)
    )
    freeze = ObjectBongardScenePredicateTaskFreeze.seal(
        task_id=task.task_id,
        task_plan_digest=task.record_digest,
        execution_precommit_digest=prepared.precommit.record_digest,
        version_space_digest=RAW_5,
        rank_response_digest=RAW_6,
        selected_predicate={
            "kind": "soft_tag_exists",
            "normalized_tag": "bird-like object",
        },
    )
    freeze_receipt = SimpleNamespace(
        payload_digest=ADDRESS_3,
        record_digest=ADDRESS_4,
        object_digest=freeze.record_digest,
    )

    def persist_freeze(**kwargs: object) -> object:
        assert kwargs["freeze"] == freeze
        events.append("persist-freeze")
        return freeze_receipt

    def persist_commit(**kwargs: object) -> object:
        assert events == ["persist-freeze"]
        commit = kwargs["commit"]
        events.append("persist-commit")
        return SimpleNamespace(object_digest=commit.record_digest)

    def release_query(**kwargs: object) -> tuple[object, object]:
        assert events[:2] == ["persist-freeze", "persist-commit"]
        panel_id = kwargs["panel_id"]
        events.append(f"release:{panel_id}")
        return SimpleNamespace(panel_id=panel_id), SimpleNamespace(panel_id=panel_id)

    def observe(side: str, released: object) -> object:
        events.append(f"observe:{side}")
        return {"side": side, "panel_id": released.panel_id}

    phase = commit_and_release_object_bongard_scene_predicate_queries(
        prepared=prepared,
        archive=object(),
        task=task,
        freeze=freeze,
        query_observer=observe,
        persist_freeze=persist_freeze,
        persist_commit=persist_commit,
        release_query=release_query,
    )
    assert events == [
        "persist-freeze",
        "persist-commit",
        "release:sealed-query-a",
        "observe:side_0",
        "release:sealed-query-b",
        "observe:side_1",
    ]
    assert replay_object_bongard_scene_predicate_query_phase(phase) is phase

    tampered = ObjectBongardScenePredicateQueryPhase(
        phase.freeze,
        SimpleNamespace(payload_digest=ADDRESS_1, record_digest=ADDRESS_4),
        phase.commit,
        phase.commit_receipt,
        phase.query_artifacts,
        phase.query_release_receipts,
    )
    with pytest.raises(
        ObjectBongardScenePredicateCampaignCommandError,
        match="model-free replay differs",
    ):
        replay_object_bongard_scene_predicate_query_phase(tampered)


def test_stage_budgets_denominator_and_python_authority_are_closed() -> None:
    budget = ObjectBongardScenePredicateCampaignBudget(
        discovery_calls=TASK_COUNT * DISCOVERY_CALLS_PER_TASK,
        registered_a_calls=TASK_COUNT * REGISTERED_A_CALLS_PER_TASK,
        registered_b_calls=TASK_COUNT * REGISTERED_B_CALLS_PER_TASK,
        ranker_calls=7,
        query_calls=14,
    )
    budget.validate_terminal(task_count=TASK_COUNT, completed_tasks=7)
    assert budget.visual_calls <= MAX_VISUAL_CALLS
    assert QUERY_DENOMINATOR == 24
    assert EXPOSURE_PREDECESSOR_FILE_SHA256 == "1bcde18e" + "387539f13c4006b4a147e61c75feacb86bb031f10a6e8ba3412fe48d"

    with pytest.raises(
        ObjectBongardScenePredicateCampaignCommandError,
        match="physical-call budget differs",
    ):
        replace(budget, query_calls=15).validate_terminal(
            task_count=TASK_COUNT, completed_tasks=7
        )

    calls = _CallBudget()
    calls.count("ranker", 1)
    with pytest.raises(
        ObjectBongardScenePredicateCampaignCommandError,
        match="budget exhausted",
    ):
        calls.count("ranker", 1)

    authority = _authority_data()
    assert authority["python_is_canonical_authority"] is True
    assert authority["lean_present"] is False
    assert authority["lean_required"] is False
    assert authority["lean_removable"] is True
