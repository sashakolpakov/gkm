from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from bongard.object_bongard_scene_predicate_campaign_command import (
    COMMAND_ID,
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
    SEMANTIC_PROPOSER_CALLS_PER_TASK,
    TASK_COUNT,
    TASK_RESULT_SCHEMA,
    _CallBudget,
    _automatic_release_source_bindings,
    _authority_data,
    _execute_task_semantic_proposal,
    _record,
    _restore_campaign_runtime,
    _runtime_record,
    _semantic_payload_gap_code,
    _validate_task_result_record,
    commit_and_release_object_bongard_scene_predicate_queries,
    prepare_object_bongard_scene_predicate_campaign,
    replay_object_bongard_scene_predicate_query_phase,
    verify_object_bongard_scene_predicate_exposure_transition,
    verify_object_bongard_scene_predicate_campaign,
    _query_score_rows,
    _rank_task_bundle,
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
            semantic_proposer_fresh_call_count=1,
            ranker_fresh_call_count=1,
            selected_survivor_digest=RAW_5,
            semantic_proposal_digest=RAW_5,
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

    def persist_query_custody(**kwargs: object) -> object:
        panel_id = kwargs["panel_id"]
        assert events[-1] == f"release:{panel_id}"
        events.append(f"custody:{panel_id}")
        return SimpleNamespace(
            object_kind="scene-query-release-custody",
            object_digest=ADDRESS_1,
            record_digest=ADDRESS_2,
        )

    phase = commit_and_release_object_bongard_scene_predicate_queries(
        prepared=prepared,
        archive=object(),
        task=task,
        freeze=freeze,
        query_observer=observe,
        persist_freeze=persist_freeze,
        persist_commit=persist_commit,
        release_query=release_query,
        persist_query_custody=persist_query_custody,
    )
    assert events == [
        "persist-freeze",
        "persist-commit",
        "release:sealed-query-a",
        "custody:sealed-query-a",
        "observe:side_0",
        "release:sealed-query-b",
        "custody:sealed-query-b",
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
        phase.query_custody_receipts,
    )
    with pytest.raises(
        ObjectBongardScenePredicateCampaignCommandError,
        match="model-free replay differs",
    ):
        replay_object_bongard_scene_predicate_query_phase(tampered)


def test_stage_budgets_denominator_and_python_authority_are_closed() -> None:
    budget = ObjectBongardScenePredicateCampaignBudget(
        discovery_calls=TASK_COUNT * DISCOVERY_CALLS_PER_TASK,
        semantic_proposer_calls=TASK_COUNT * SEMANTIC_PROPOSER_CALLS_PER_TASK,
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
    automatic = _automatic_release_source_bindings()
    assert set(automatic) == {"batch_source", "release_gate_source"}
    assert all(
        value.startswith("sha256:") and len(value) == 71
        for value in automatic.values()
    )


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


def test_semantic_gap_code_is_determined_by_usable_discovery_evidence() -> None:
    enough = SimpleNamespace(
        alias_bindings=tuple(
            {"historical_role": role, "usable": True}
            for role in (0, 0, 1, 1)
        )
    )
    insufficient = SimpleNamespace(
        alias_bindings=tuple(
            {"historical_role": role, "usable": True}
            for role in (0, 0, 1)
        )
    )
    assert _semantic_payload_gap_code(enough) == "payload_rejected"
    assert (
        _semantic_payload_gap_code(insufficient)
        == "insufficient_discovery_evidence"
    )


def test_semantic_proposal_gap_never_calls_ranker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import bongard.object_bongard_scene_predicate_campaign_command as campaign

    persisted: list[tuple[str, dict[str, object]]] = []

    def persist(
        _store: object,
        *,
        object_kind: str,
        record: dict[str, object],
        digest_field: str,
    ) -> tuple[dict[str, object], object]:
        assert digest_field in record
        persisted.append((object_kind, record))
        return record, SimpleNamespace(object_kind=object_kind)

    monkeypatch.setattr(campaign, "_persist_record", persist)
    proposer_calls: list[object] = []
    bundle = SimpleNamespace(
        complete_survivor_digests=(RAW_5,),
        ranker_slate=({"candidate_digest": RAW_5},),
        omitted_survivors=(),
        bundle_digest=RAW_6,
    )
    prepared = SimpleNamespace(release=SimpleNamespace(store=object()))
    rank_input, _, rank_result, _, selected = _rank_task_bundle(
        tmp_path,
        prepared=prepared,
        task=SimpleNamespace(record_digest=ADDRESS_1, family="bd"),
        task_index=0,
        bundle=bundle,
        ir_record={"ir_freeze_digest": ADDRESS_2},
        semantic_proposal_record={
            "semantic_proposal_result_digest": ADDRESS_3,
            "semantic_proposal_digest": RAW_6,
            "semantic_proposal_status": "typed_proposal_gap",
            "semantic_proposal_valid": False,
        },
        runtime=object(),
        text_transport=lambda *_args, **_kwargs: proposer_calls.append(object()),
        budget=_CallBudget(),
    )
    assert selected is None
    assert proposer_calls == []
    assert rank_input["ranker_slate"] == []
    assert rank_input["omitted_survivors"] == [
        {
            "candidate_digest": RAW_5,
            "reason": "mandatory_semantic_proposal_gap",
        }
    ]
    assert rank_result["status"] == "typed_semantic_proposal_gap"
    assert rank_result["ranker_called"] is False
    assert [kind for kind, _ in persisted] == [
        "scene-task-rank-input",
        "scene-task-rank-result",
    ]


def test_semantic_payload_error_seals_exact_rejected_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import bongard.object_bongard_scene_predicate_calibration_command as calibration
    import bongard.object_bongard_scene_predicate_campaign_command as campaign
    import bongard.object_bongard_turn_journal as journal_module
    import bongard.object_scene_semantic_registry as semantic

    payload = {"side0_positive": [], "side1_positive": []}
    receipt = SimpleNamespace(
        receipt_digest=ADDRESS_1,
        to_dict=lambda: {"receipt": "semantic"},
    )

    class FakeJournal:
        def __init__(self, *_args: object, **kwargs: object) -> None:
            self.transport = kwargs["underlying_transport"]
            self.fresh_call_count = 0
            self.reused_call_count = 0

        def __call__(self, *args: object, **kwargs: object) -> object:
            self.transport(*args, **kwargs)
            self.fresh_call_count = 1
            return SimpleNamespace(payload=payload, receipt=receipt)

    proposal = SimpleNamespace(
        status="typed_proposal_gap",
        preparation_digest=RAW_5,
        registry_digest=RAW_6,
        proposal_digest=RAW_5,
        dropped_concepts=(),
        to_data=lambda: {"proposal": "typed-gap"},
    )
    registry = SimpleNamespace(
        registry_digest=RAW_6,
        tags=(),
        to_data=lambda: {"registry": "zero-tags"},
    )
    gap_calls: list[tuple[str, dict[str, object]]] = []

    def reject(_prepared: object, _payload: object) -> object:
        raise semantic.ObjectSceneSemanticRegistryPayloadError("invalid payload")

    def build_gap(
        _prepared: object, gap_code: str, rejected_payload: object
    ) -> tuple[object, object]:
        gap_calls.append((gap_code, dict(rejected_payload)))
        return proposal, registry

    monkeypatch.setattr(calibration, "_journal_runtime_kwargs", lambda _runtime: {})
    monkeypatch.setattr(
        journal_module, "ObjectBongardTextTurnJournalTransport", FakeJournal
    )
    monkeypatch.setattr(
        journal_module,
        "verify_object_bongard_turn_journal",
        lambda _journal: SimpleNamespace(record_digest=ADDRESS_2),
    )
    monkeypatch.setattr(
        semantic, "build_object_scene_semantic_registry_proposal", reject
    )
    monkeypatch.setattr(
        semantic, "build_object_scene_semantic_registry_gap", build_gap
    )
    monkeypatch.setattr(
        campaign,
        "_persist_record",
        lambda _store, **kwargs: (
            kwargs["record"],
            SimpleNamespace(object_kind=kwargs["object_kind"]),
        ),
    )
    monkeypatch.setattr(
        campaign,
        "_restore_task_semantic_proposal",
        lambda *_args, **_kwargs: (proposal, registry),
    )
    prepared_input = SimpleNamespace(
        prompt="both frozen support buckets",
        output_schema={"type": "object"},
        alias_bindings=tuple(
            {"historical_role": role, "usable": True}
            for role in (0, 0, 1, 1)
        ),
    )
    budget = _CallBudget()
    returned_proposal, returned_registry, record, _ = (
        _execute_task_semantic_proposal(
            tmp_path,
            prepared=SimpleNamespace(
                release=SimpleNamespace(
                    authorization=SimpleNamespace(record_digest=ADDRESS_1),
                    precommit=SimpleNamespace(record_digest=ADDRESS_2),
                    store=object(),
                )
            ),
            task=SimpleNamespace(record_digest=ADDRESS_3, family="bd"),
            task_index=0,
            runtime=object(),
            semantic_prepared_record={
                "preparation_digest": RAW_5,
                "semantic_prepared_digest": ADDRESS_4,
            },
            semantic_prepared=prepared_input,
            discovery_artifacts=(),
            role_rows=(),
            text_transport=lambda *_args, **_kwargs: object(),
            budget=budget,
        )
    )
    assert returned_proposal is proposal
    assert returned_registry is registry
    assert gap_calls == [("payload_rejected", payload)]
    assert record["proposer_payload"] == payload
    assert record["semantic_proposal_valid"] is False
    assert budget.snapshot().semantic_proposer_calls == 1


def test_mixed_semantic_payload_is_persisted_and_cold_replayed_without_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import bongard.object_bongard_scene_predicate_campaign_command as campaign
    from bongard.object_bongard_release_gate import ObjectBongardReleaseStore
    from bongard.object_bongard_turn_journal import ObjectBongardTurnRuntime
    from bongard.object_scene_semantic_registry import (
        prepare_object_scene_semantic_registry_proposal,
    )
    from bongard.prototype_scene_observer import (
        prototype_scene_transport_source_digest,
    )
    from bongard.tests.test_object_bongard_scene_predicate_calibration_command import (
        LAUNCHER_DIGEST,
        MODEL_CATALOG,
        NO_TOOLS_ATTESTATION,
        _text_receipt,
    )
    from bongard.tests.test_object_scene_semantic_registry import (
        _discovery_artifact,
    )
    from bongard.transport import (
        CloudPolicyCacheSnapshot,
        CodexStructuredResult,
    )

    discovery_artifacts = tuple(_discovery_artifact(index) for index in range(12))
    role_rows = tuple(
        {
            "ordinal": index,
            "neutral_panel_digest": artifact.panel_digest,
            "historical_role": index // 6,
            "blind_panel_id": artifact.scene_id,
        }
        for index, artifact in enumerate(discovery_artifacts)
    )
    semantic_prepared = prepare_object_scene_semantic_registry_proposal(
        discovery_artifacts, role_rows
    )
    aliases = {
        side: tuple(
            row["alias"]
            for row in semantic_prepared.alias_bindings
            if row["historical_role"] == side and row["usable"] is True
        )
        for side in (0, 1)
    }
    payload = {
        "side0_positive": [
            {
                "scope": "panel",
                "phrase": "paired visible forms",
                "citations": list(aliases[0][:2]),
            },
            {
                "scope": "entity",
                "phrase": "not pointed",
                "citations": list(aliases[0][1:3]),
            },
        ],
        "side1_positive": [
            {
                "scope": "entity",
                "phrase": "unequal edge lengths",
                "citations": list(aliases[1][:2]),
            }
        ],
    }

    runtime = ObjectBongardTurnRuntime(
        model=campaign.MODEL,
        reasoning_effort=campaign.REASONING_EFFORT,
        minutes=3,
        verbose=False,
        executable="codex",
        cloud_policy_cache_snapshot=CloudPolicyCacheSnapshot(None),
        model_catalog_snapshot=MODEL_CATALOG,
        expected_launcher_digest=LAUNCHER_DIGEST,
        no_tools_attestation=NO_TOOLS_ATTESTATION,
        transport_source_digest=prototype_scene_transport_source_digest(),
    )
    store = ObjectBongardReleaseStore((tmp_path / "release-store").absolute())
    prepared = SimpleNamespace(
        release=SimpleNamespace(
            authorization=SimpleNamespace(record_digest=ADDRESS_1),
            precommit=SimpleNamespace(record_digest=ADDRESS_2),
            store=store,
        )
    )
    task = SimpleNamespace(record_digest=ADDRESS_3, family="bd")
    semantic_prepared_record = campaign._semantic_prepared_record(
        task=task,
        discovery_batch={"batch_digest": ADDRESS_4},
        role_reveal={"role_reveal_digest": ADDRESS_1},
        semantic_prepared=semantic_prepared,
    )
    semantic_prepared_record, semantic_prepared_receipt = campaign._persist_record(
        store,
        object_kind="scene-task-semantic-prepared",
        record=semantic_prepared_record,
        digest_field="semantic_prepared_digest",
    )
    assert dict(
        store.verify(
            semantic_prepared_receipt,
            expected_data=semantic_prepared_record,
        )
    ) == semantic_prepared_record

    physical_calls: list[str] = []

    def offline_transport(prompt, schema, **_kwargs):
        physical_calls.append("semantic-proposer")
        return CodexStructuredResult(
            payload, _text_receipt(prompt, schema, payload)
        )

    budget = _CallBudget()
    proposal, registry, record, receipt = _execute_task_semantic_proposal(
        tmp_path,
        prepared=prepared,
        task=task,
        task_index=0,
        runtime=runtime,
        semantic_prepared_record=semantic_prepared_record,
        semantic_prepared=semantic_prepared,
        discovery_artifacts=discovery_artifacts,
        role_rows=role_rows,
        text_transport=offline_transport,
        budget=budget,
    )

    assert physical_calls == ["semantic-proposer"]
    assert budget.snapshot().semantic_proposer_calls == 1
    assert proposal.status == "proposed"
    assert len(registry.tags) == 2
    assert {item.tag for item in registry.tags} == {
        "paired visible forms",
        "unequal edge lengths",
    }
    assert len(proposal.dropped_concepts) == 1
    assert proposal.dropped_concepts[0].reason_code == "phrase_policy"
    assert record["semantic_proposal_status"] == "proposed"
    assert record["semantic_proposal_valid"] is True
    assert record["quarantined_concept_count"] == 1
    assert record["quarantined_concept_digests"] == [
        proposal.dropped_concepts[0].drop_digest
    ]
    assert record["semantic_registry"]["tags"]
    assert dict(store.verify(receipt, expected_data=record)) == record
    journal_root = tmp_path / "journals" / "semantic_registry_proposer"
    assert {path.name for path in journal_root.iterdir()} == {
        "manifest.json",
        "claim.json",
        "result.json",
        "outcome.json",
    }

    replay_calls: list[object] = []

    def forbidden_replay_transport(*args, **kwargs):
        replay_calls.append((args, kwargs))
        raise AssertionError("cold replay attempted a physical proposer call")

    monkeypatch.setattr(
        campaign, "_forbidden_text_transport", forbidden_replay_transport
    )
    replayed_proposal, replayed_registry, replay_summary_digest = (
        campaign._cold_replay_task_semantic_proposal(
            tmp_path,
            prepared=prepared,
            task=task,
            task_index=0,
            runtime=runtime,
            semantic_prepared_record=semantic_prepared_record,
            semantic_prepared=semantic_prepared,
            semantic_proposal_record=record,
            discovery_artifacts=discovery_artifacts,
            role_rows=role_rows,
        )
    )
    assert replay_calls == []
    assert physical_calls == ["semantic-proposer"]
    assert replayed_proposal == proposal
    assert replayed_registry == registry
    assert replay_summary_digest == record["proposer_journal_summary_digest"]


def _task_result_fixture(
    *, queried: bool, semantic_valid: bool = True
) -> dict[str, object]:
    dependencies = {
        "discovery_batch": {},
        "role_reveal": {},
        "semantic_prepared": {},
        "semantic_proposal": {},
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
            "schema": TASK_RESULT_SCHEMA,
            "command_id": COMMAND_ID,
            "task_ordinal": 0,
            "task_id": "opaque-task-00",
            "task_plan_digest": ADDRESS_1,
            "execution_precommit_digest": ADDRESS_2,
            "support_release_receipts": [{}] * 12,
            "dependencies": dependencies,
            "status": (
                "evaluated"
                if queried
                else (
                    "typed_version_space_gap"
                    if semantic_valid
                    else "typed_semantic_proposal_gap"
                )
            ),
            "semantic_proposal_digest": RAW_6,
            "semantic_proposal_status": (
                "proposed" if semantic_valid else "typed_proposal_gap"
            ),
            "semantic_proposal_valid": semantic_valid,
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
                "semantic_proposer_calls": 1,
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
            "semantic_proposer_journal_summary_digest": ADDRESS_3,
            "query_journal_summary_digests": [ADDRESS_2] * (2 if queried else 0),
            "semantic_proposer_journal_cold_replayed": True,
            "ranker_journal_cold_replayed_if_called": queried,
            "all_task_journals_cold_replayed_without_model_calls": True,
            **_authority_data(),
        },
        "task_result_digest",
    )


def test_accepted_and_gap_task_records_fail_closed_on_tamper_or_budget() -> None:
    accepted = _task_result_fixture(queried=True)
    gap = _task_result_fixture(queried=False)
    semantic_gap = _task_result_fixture(queried=False, semantic_valid=False)
    assert _validate_task_result_record(accepted)["status"] == "evaluated"
    assert (
        _validate_task_result_record(gap)["status"]
        == "typed_version_space_gap"
    )
    restored_semantic_gap = _validate_task_result_record(semantic_gap)
    assert restored_semantic_gap["status"] == "typed_semantic_proposal_gap"
    assert restored_semantic_gap["physical_call_delta"]["ranker_calls"] == 0
    assert restored_semantic_gap["physical_call_delta"]["query_calls"] == 0
    assert all(not row["correct"] for row in restored_semantic_gap["score_rows"])

    invalid_semantic_query = _task_result_fixture(
        queried=True, semantic_valid=False
    )
    with pytest.raises(
        ObjectBongardScenePredicateCampaignCommandError,
        match="policy or budget differs",
    ):
        _validate_task_result_record(invalid_semantic_query)

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

    forged_gap_metadata_body = {
        key: value for key, value in gap.items() if key != "task_result_digest"
    }
    forged_gap_metadata_body["score_rows"] = [
        {**row, "query_artifact_digest": RAW_5}
        if index == 0
        else dict(row)
        for index, row in enumerate(forged_gap_metadata_body["score_rows"])
    ]
    forged_gap_metadata = _record(
        forged_gap_metadata_body, "task_result_digest"
    )
    with pytest.raises(
        ObjectBongardScenePredicateCampaignCommandError,
        match="canonical no-query rows",
    ):
        _validate_task_result_record(forged_gap_metadata)


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
