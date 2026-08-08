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
    _record,
    _restore_campaign_runtime,
    _runtime_record,
    _validate_task_result_record,
    commit_and_release_object_bongard_scene_predicate_queries,
    prepare_object_bongard_scene_predicate_campaign,
    replay_object_bongard_scene_predicate_query_phase,
    verify_object_bongard_scene_predicate_exposure_transition,
    verify_object_bongard_scene_predicate_campaign,
    _query_score_rows,
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

    calls.clear()
    accepted_digest_fields = {
        name: ADDRESS_1
        for name in (
            "authorization_digest",
            "execution_precommit_digest",
            "discovery_batch_digest",
            "discovery_freeze_digest",
            "registry_digest",
            "evaluation_a_batch_digest",
            "evaluation_b_batch_digest",
            "evaluation_freeze_digest",
            "role_reveal_digest",
            "assessment_digest",
            "rank_input_freeze_digest",
            "rank_result_digest",
            "formula_freeze_digest",
            "replay_digest",
            "result_digest",
        )
    }

    def accept(_root: object, **_kwargs: object) -> object:
        calls.append("calibration")
        return SimpleNamespace(
            accepted=True,
            status="accepted",
            visual_fresh_call_count=36,
            ranker_fresh_call_count=1,
            selected_survivor_digest=RAW_5,
            source_digest=RAW_6,
            **accepted_digest_fields,
        )

    with pytest.raises(
        ObjectBongardScenePredicateCampaignCommandError,
        match="runtime must be durable before campaign exposure",
    ):
        prepare_object_bongard_scene_predicate_campaign(
            output,
            calibration_root="synthetic-accepted",
            calibration_verifier=accept,
            preregistration_path=PoisonPath(),
            plan_path=PoisonPath(),
            descriptor_path=PoisonPath(),
            archive_path=PoisonPath(),
            split_path=PoisonPath(),
            exposure_predecessor_path=PoisonPath(),
        )
    assert calls == ["calibration"]
    assert not output.exists()

    calls.clear()
    with pytest.raises(
        ObjectBongardScenePredicateCampaignCommandError,
        match="not accepted and cold-verified",
    ):
        verify_object_bongard_scene_predicate_campaign(
            PoisonPath(),
            calibration_root="synthetic-rejected",
            calibration_verifier=reject,
        )
    assert calls == ["calibration"]


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
            "candidate_digest": RAW_5,
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


def test_query_scoring_uses_certified_absence_in_both_orientations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from bongard.evidence import Disposition
    import bongard.object_bongard_scene_predicate_ir as ir
    from bongard.object_bongard_scene_predicate_ir import SceneOrientation

    language = object()
    monkeypatch.setattr(
        ir.ScenePredicateLanguage,
        "from_data",
        classmethod(lambda _cls, _value: language),
    )
    monkeypatch.setattr(
        ir.ScenePredicateCandidate,
        "from_data",
        classmethod(
            lambda _cls, value, **_kwargs: SimpleNamespace(
                orientation=SceneOrientation(value["orientation"])
            )
        ),
    )
    monkeypatch.setattr(
        ir, "adapt_object_scene_registered_single", lambda _panel_id, artifact: artifact
    )
    monkeypatch.setattr(
        ir,
        "evaluate_object_scene_candidate",
        lambda _candidate, _language, panel: panel.disposition,
    )
    bundle = SimpleNamespace(version_space={"language": {}})
    group0 = {"orientation": SceneOrientation.GROUP0_POSITIVE.value}
    group1 = {"orientation": SceneOrientation.GROUP1_POSITIVE.value}
    present_absent = (
        SimpleNamespace(
            disposition=Disposition.PRESENT,
            artifact_digest=RAW_5,
            observation_digest=RAW_5,
        ),
        SimpleNamespace(
            disposition=Disposition.CERTIFIED_ABSENT,
            artifact_digest=RAW_6,
            observation_digest=RAW_6,
        ),
    )
    rows0 = _query_score_rows(
        bundle=bundle,
        selected_candidate_data=group0,
        artifacts=present_absent,
    )
    rows1 = _query_score_rows(
        bundle=bundle,
        selected_candidate_data=group1,
        artifacts=tuple(reversed(present_absent)),
    )
    assert [row["actual_disposition"] for row in rows0] == [
        "present",
        "certified_absent",
    ]
    assert [row["actual_disposition"] for row in rows1] == [
        "certified_absent",
        "present",
    ]
    assert all(row["correct"] for row in (*rows0, *rows1))


def _task_result_fixture(*, queried: bool) -> dict[str, object]:
    dependencies = {
        "discovery_batch": {},
        "registry_freeze": {},
        "registered_a_batch": {},
        "registered_b_batch": {},
        "ir_freeze": {},
        "rank_input": {},
        "rank_result": {},
        "query_batch": {} if queried else None,
    }
    selected = RAW_5 if queried else None
    score_rows = [
        {
            "side": f"side_{index}",
            "query_artifact_digest": RAW_5 if queried else None,
            "query_observation_digest": RAW_6 if queried else None,
            "expected_disposition": "present" if queried else None,
            "actual_disposition": (
                "present" if queried else "typed_gap_no_query"
            ),
            "correct": queried,
            "indeterminate_or_error_scores_incorrect": not queried,
        }
        for index in range(2)
    ]
    return _record(
        {
            "schema": "gkm.bongard-scene-predicate-task-result.v1",
            "command_id": "bongard.scene-predicate-campaign/exact-unused-train-12-v1",
            "task_ordinal": 0,
            "task_id": "opaque-task-00",
            "task_plan_digest": ADDRESS_1,
            "execution_precommit_digest": ADDRESS_2,
            "support_release_receipts": [{}] * 12,
            "dependencies": dependencies,
            "status": "evaluated" if queried else "typed_gap",
            "selected_survivor_digest": selected,
            "bundle_digest": RAW_6,
            "rank_result_digest": ADDRESS_3,
            "task_formula_freeze_digest": ADDRESS_3 if queried else None,
            "task_decision_commit_digest": ADDRESS_4 if queried else None,
            "score_rows": score_rows,
            "correct_count": 2 if queried else 0,
            "score_denominator_contribution": 2,
            "physical_call_delta": {
                "discovery_calls": 12,
                "registered_a_calls": 12,
                "registered_b_calls": 12,
                "ranker_calls": 1 if queried else 0,
                "query_calls": 2 if queried else 0,
            },
            "support_pixels_released_only_through_official_gate": True,
            "query_pixels_released_only_after_exact_formula_commit": queried,
            "typed_gap_makes_no_ranker_or_query_calls": not queried,
            "terminal_python_ir_cold_replayed": True,
            "support_journal_summary_digests": [ADDRESS_1] * 36,
            "query_journal_summary_digests": [ADDRESS_2] * (2 if queried else 0),
            "ranker_journal_cold_replayed_if_called": queried,
            "all_task_journals_cold_replayed_without_model_calls": True,
            **_authority_data(),
        },
        "task_result_digest",
    )


def test_accepted_and_gap_task_records_fail_closed_on_tamper_or_budget() -> None:
    accepted = _task_result_fixture(queried=True)
    gap = _task_result_fixture(queried=False)
    assert _validate_task_result_record(accepted)["status"] == "evaluated"
    assert _validate_task_result_record(gap)["status"] == "typed_gap"

    tampered = dict(accepted)
    tampered["correct_count"] = 0
    with pytest.raises(
        ObjectBongardScenePredicateCampaignCommandError,
        match="policy or budget differs",
    ):
        _validate_task_result_record(tampered)

    wrong_budget_body = {
        key: value for key, value in gap.items() if key != "task_result_digest"
    }
    wrong_budget_body["physical_call_delta"] = {
        **wrong_budget_body["physical_call_delta"],
        "query_calls": 2,
    }
    wrong_budget = _record(wrong_budget_body, "task_result_digest")
    with pytest.raises(
        ObjectBongardScenePredicateCampaignCommandError,
        match="policy or budget differs",
    ):
        _validate_task_result_record(wrong_budget)

    forged_gap_body = {
        key: value for key, value in gap.items() if key != "task_result_digest"
    }
    forged_gap_body["correct_count"] = False
    forged_gap = _record(forged_gap_body, "task_result_digest")
    with pytest.raises(
        ObjectBongardScenePredicateCampaignCommandError,
        match="policy or budget differs",
    ):
        _validate_task_result_record(forged_gap)

    forged_authority_body = {
        key: value for key, value in accepted.items() if key != "task_result_digest"
    }
    forged_authority_body["python_is_canonical_authority"] = False
    forged_authority = _record(forged_authority_body, "task_result_digest")
    with pytest.raises(
        ObjectBongardScenePredicateCampaignCommandError,
        match="policy or budget differs",
    ):
        _validate_task_result_record(forged_authority)


def test_runtime_record_round_trip_preserves_absent_cache_snapshot_object() -> None:
    from bongard.object_bongard_turn_journal import ObjectBongardTurnRuntime
    from bongard.prototype_scene_observer import (
        prototype_scene_transport_source_digest,
    )
    from bongard.tests.no_tools_fixture import canonical_no_tools_runtime
    from bongard.transport import (
        PINNED_CODEX_CLI_VERSION,
        CloudPolicyCacheSnapshot,
    )

    launcher = "b" * 64
    catalog, attestation = canonical_no_tools_runtime(launcher)
    runtime = ObjectBongardTurnRuntime(
        model="gpt-5.6-sol",
        reasoning_effort="medium",
        minutes=3,
        verbose=False,
        executable="codex",
        cloud_policy_cache_snapshot=CloudPolicyCacheSnapshot(None),
        model_catalog_snapshot=catalog,
        expected_launcher_digest=launcher,
        no_tools_attestation=attestation,
        transport_source_digest=prototype_scene_transport_source_digest(),
    )
    record = _runtime_record(
        runtime,
        {"version": PINNED_CODEX_CLI_VERSION, "launcher_digest": launcher},
    )
    restored = _restore_campaign_runtime(record)
    assert restored.binding == runtime.binding
    assert restored.cloud_policy_cache_snapshot is not None
    assert restored.cloud_policy_cache_snapshot.data is None
