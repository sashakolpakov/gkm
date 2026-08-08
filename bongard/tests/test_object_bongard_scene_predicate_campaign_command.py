from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from bongard.object_bongard_scene_predicate_campaign_command import (
    CAMPAIGN_REPLAY_SCHEMA,
    CAMPAIGN_RESULT_SCHEMA,
    CAMPAIGN_RUNTIME_CUSTODY_SCHEMA,
    CAMPAIGN_RUNTIME_SCHEMA,
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
    TASK_BATCH_SCHEMA,
    TASK_COUNT,
    TASK_IR_SCHEMA,
    TASK_RANK_INPUT_SCHEMA,
    TASK_RANK_RESULT_SCHEMA,
    TASK_REGISTRY_SCHEMA,
    TASK_RESULT_SCHEMA,
    TASK_ROLE_REVEAL_SCHEMA,
    TASK_SEMANTIC_PREPARED_SCHEMA,
    TASK_SEMANTIC_PROPOSAL_SCHEMA,
    TYPED_GROUNDING_REPEATABILITY_GAP,
    TYPED_LANGUAGE_GAP,
    TYPED_SELECTIVITY_GAP,
    _CallBudget,
    _automatic_release_source_bindings,
    _authority_data,
    _execute_task_semantic_proposal,
    _record,
    _restore_campaign_runtime,
    _runtime_record,
    _semantic_payload_gap_code,
    _digest_free_task_ranker_row,
    _validate_task_result_record,
    commit_and_release_object_bongard_scene_predicate_queries,
    prepare_object_bongard_scene_predicate_campaign,
    replay_object_bongard_scene_predicate_query_phase,
    verify_object_bongard_scene_predicate_exposure_transition,
    verify_object_bongard_scene_predicate_campaign,
    _query_score_rows,
    _ranker_prompt,
    _rank_task_bundle,
)


ADDRESS_1 = "sha256:" + "1" * 64
ADDRESS_2 = "sha256:" + "2" * 64
ADDRESS_3 = "sha256:" + "3" * 64
ADDRESS_4 = "sha256:" + "4" * 64
RAW_5 = "5" * 64
RAW_6 = "6" * 64


def _semantic_concept(
    scope: str,
    phrase: str,
    support_bindings: tuple[dict[str, str], ...] | list[dict[str, str]],
    *,
    witness_kind: str = "shape_appearance",
    witness_statement: str | None = None,
    accepted_variants: tuple[str, ...] = (),
    near_miss_boundaries: tuple[str, ...] = (),
) -> dict[str, object]:
    return {
        "scope": scope,
        "phrase": phrase,
        "required_witnesses": [
            {
                "kind": witness_kind,
                "statement": witness_statement or phrase,
            }
        ],
        "accepted_variants": list(accepted_variants),
        "near_miss_boundaries": list(near_miss_boundaries),
        "support_bindings": [dict(item) for item in support_bindings],
    }


def _semantic_support_bindings(
    prepared: object,
    orientation: str,
    scope: str,
) -> list[dict[str, str]]:
    model_view = getattr(prepared, "model_view")
    rows_key = orientation.replace("_positive", "_support_descriptions")
    rows = {
        row["panel_alias"]: row
        for row in model_view[rows_key]
    }
    return [
        {
            "panel_alias": panel_alias,
            "target_alias": (
                "whole_panel"
                if scope == "panel"
                else rows[panel_alias]["proposal_atlas_map"][0][
                    "entity_alias"
                ]
            ),
        }
        for panel_alias in model_view[
            "required_positive_binding_panels"
        ][orientation]
    ]


def test_campaign_v7_binds_tag_orientation_through_every_wrapper() -> None:
    assert COMMAND_ID.endswith("-v7")
    assert TASK_BATCH_SCHEMA.endswith(".v4")
    assert TASK_SEMANTIC_PREPARED_SCHEMA.endswith(".v7")
    assert TASK_SEMANTIC_PROPOSAL_SCHEMA.endswith(".v7")
    assert TASK_REGISTRY_SCHEMA.endswith(".v5")
    assert TASK_IR_SCHEMA.endswith(".v5")
    assert TASK_RANK_INPUT_SCHEMA.endswith(".v5")
    assert TASK_RANK_RESULT_SCHEMA.endswith(".v5")
    assert TASK_RESULT_SCHEMA.endswith(".v5")
    assert CAMPAIGN_RESULT_SCHEMA.endswith(".v5")
    assert CAMPAIGN_REPLAY_SCHEMA.endswith(".v5")
    assert TASK_ROLE_REVEAL_SCHEMA.endswith(".v2")
    assert CAMPAIGN_RUNTIME_SCHEMA.endswith(".v2")
    assert CAMPAIGN_RUNTIME_CUSTODY_SCHEMA.endswith(".v2")


def test_ranker_prompt_keeps_frozen_operational_witness_card() -> None:
    row = {
        "candidate_digest": RAW_5,
        "orientation": "group0_positive",
        "complexity": 1,
        "formula": {
            "node": "positive_atom",
            "kind": "registered_tag",
            "tag_id": "tag_0000",
            "tag_digest": RAW_6,
            "criteria_digest": RAW_5,
            "affirmative_phrase": "paired visible forms",
            "required_witnesses": [
                {
                    "witness_id": "witness_00",
                    "kind": "spatial_relation",
                    "statement": (
                        "two visible forms occupy distinct regions of the panel"
                    ),
                }
            ],
            "accepted_variants": [
                "touching forms count when both visible extents remain distinct"
            ],
            "near_miss_boundaries": [
                "one internally divided form does not qualify"
            ],
        },
        "merged_support_summary": {
            "present": 6,
            "certified_absent": 6,
            "indeterminate": 0,
            "error": 0,
        },
        "gate_summary": {
            "coverage": True,
            "selectivity": True,
            "repeatability": True,
        },
        "formula_is_frozen": True,
        "ranker_can_only_select": True,
    }
    projected = _digest_free_task_ranker_row(row)
    prompt = _ranker_prompt((projected,))
    assert '"required_witnesses"' in prompt
    assert '"accepted_variants"' in prompt
    assert '"near_miss_boundaries"' in prompt
    assert "two visible forms occupy distinct regions" in prompt
    assert '"tag_digest"' not in prompt
    assert '"criteria_digest"' not in prompt


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
    assert authority["frozen_python_predicate_is_normative"] is True
    assert authority["python_replay_is_normative"] is True
    assert authority["semantic_proposal_orientation_is_part_of_tag_identity"] is True
    assert authority["same_semantic_tag_tried_in_both_orientations"] is False
    assert authority[
        "registered_visual_observer_receives_orientation_constraint_metadata"
    ] is False
    assert authority[
        "opposite_orientation_registered_tag_candidate_copies_forbidden"
    ] is True
    assert authority["lean_present"] is False
    assert authority["lean_required"] is False
    assert authority["lean_removable"] is True
    assert authority["lean_if_present_is_optional_checker_or_export_only"] is True
    assert authority["lean_affects_acceptance_or_runtime_semantics"] is False
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


def test_multimodal_semantic_gap_is_not_caused_by_missing_discovery_prose() -> None:
    enough = SimpleNamespace(
        alias_bindings=tuple(
            {"historical_role": role, "usable": False}
            for role in ((0,) * 6 + (1,) * 6)
        )
    )
    insufficient = SimpleNamespace(
        alias_bindings=tuple(
            {"historical_role": role, "usable": True}
            for role in ((0,) * 5 + (1,) * 6)
        )
    )
    assert _semantic_payload_gap_code(enough) == "payload_rejected"
    with pytest.raises(
        ObjectBongardScenePredicateCampaignCommandError,
        match="support role inventory",
    ):
        _semantic_payload_gap_code(insufficient)


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
    assert rank_input["typed_gap_status"] == "typed_semantic_proposal_gap"
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


@pytest.mark.parametrize(
    ("coverage", "selectivity", "repeatability", "expected_status"),
    (
        (False, False, False, TYPED_LANGUAGE_GAP),
        (True, False, False, TYPED_SELECTIVITY_GAP),
        (True, True, False, TYPED_GROUNDING_REPEATABILITY_GAP),
    ),
)
def test_empty_task_slate_names_first_failed_evidence_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    coverage: bool,
    selectivity: bool,
    repeatability: bool,
    expected_status: str,
) -> None:
    import bongard.object_bongard_scene_predicate_campaign_command as campaign

    monkeypatch.setattr(
        campaign,
        "_persist_record",
        lambda _store, **kwargs: (
            kwargs["record"],
            SimpleNamespace(object_kind=kwargs["object_kind"]),
        ),
    )
    bundle = SimpleNamespace(
        complete_survivor_digests=(),
        ranker_slate=(),
        omitted_survivors=(),
        bundle_digest=RAW_6,
        coverage_gate=SimpleNamespace(passed=coverage),
        selectivity_gate=SimpleNamespace(passed=selectivity),
        repeatability_gate=SimpleNamespace(passed=repeatability),
    )
    rank_input, _, rank_result, _, selected = _rank_task_bundle(
        tmp_path,
        prepared=SimpleNamespace(release=SimpleNamespace(store=object())),
        task=SimpleNamespace(record_digest=ADDRESS_1, family="bd"),
        task_index=0,
        bundle=bundle,
        ir_record={"ir_freeze_digest": ADDRESS_2},
        semantic_proposal_record={
            "semantic_proposal_result_digest": ADDRESS_3,
            "semantic_proposal_digest": RAW_6,
            "semantic_proposal_status": "proposed",
            "semantic_proposal_valid": True,
        },
        runtime=object(),
        text_transport=lambda *_args, **_kwargs: pytest.fail(
            "typed evidence gap called ranker"
        ),
        budget=_CallBudget(),
    )
    assert selected is None
    assert rank_input["typed_gap_status"] == expected_status
    assert rank_result["status"] == expected_status
    assert rank_result["ranker_called"] is False


def test_accepted_rank_path_projects_lineage_but_keeps_witness_cards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import bongard.object_bongard_scene_predicate_calibration_command as calibration
    import bongard.object_bongard_scene_predicate_campaign_command as campaign
    import bongard.object_bongard_turn_journal as journal_module

    prompts: list[str] = []

    def ranker_transport(prompt: str, *_args: object, **_kwargs: object) -> object:
        prompts.append(prompt)
        return SimpleNamespace(payload={"selected_survivor_digest": RAW_5})

    class FakeJournal:
        def __init__(self, *_args: object, **kwargs: object) -> None:
            self.transport = kwargs["underlying_transport"]
            self.fresh_call_count = 0
            self.reused_call_count = 0

        def __call__(self, *args: object, **kwargs: object) -> object:
            result = self.transport(*args, **kwargs)
            self.fresh_call_count = 1
            return result

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

    monkeypatch.setattr(calibration, "_journal_runtime_kwargs", lambda _runtime: {})
    monkeypatch.setattr(
        journal_module, "ObjectBongardTextTurnJournalTransport", FakeJournal
    )
    monkeypatch.setattr(
        journal_module,
        "verify_object_bongard_turn_journal",
        lambda _journal: SimpleNamespace(record_digest=ADDRESS_4),
    )
    monkeypatch.setattr(campaign, "_persist_record", persist)

    ranker_row = {
        "candidate_digest": RAW_5,
        "orientation": "group0_positive",
        "complexity": 1,
        "formula_digest": RAW_6,
        "formula": {
            "node": "positive_atom",
            "kind": "registered_tag",
            "tag_id": "tag_0000",
            "tag_digest": RAW_6,
            "criteria_digest": ADDRESS_3,
            "affirmative_phrase": "bird-like object",
            "required_witnesses": [
                {
                    "witness_id": "witness_00",
                    "witness_digest": RAW_6,
                    "kind": "shape_appearance",
                    "statement": "one silhouette has a beak-like protrusion",
                }
            ],
            "accepted_variants": ["either left- or right-facing silhouette"],
            "near_miss_boundaries": ["a plain oval does not qualify"],
        },
        "merged_support_summary": {
            "present": 6,
            "certified_absent": 6,
            "indeterminate": 0,
            "error": 0,
        },
        "gate_summary": {
            "coverage": True,
            "selectivity": True,
            "repeatability": True,
        },
        "formula_is_frozen": True,
        "ranker_can_only_select": True,
    }
    bundle = SimpleNamespace(
        complete_survivor_digests=(RAW_5,),
        ranker_slate=(ranker_row,),
        omitted_survivors=(),
        bundle_digest=RAW_6,
    )
    prepared = SimpleNamespace(
        release=SimpleNamespace(
            store=object(),
            authorization=SimpleNamespace(record_digest=ADDRESS_1),
            precommit=SimpleNamespace(record_digest=ADDRESS_2),
        )
    )
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
            "semantic_proposal_status": "proposed",
            "semantic_proposal_valid": True,
        },
        runtime=object(),
        text_transport=ranker_transport,
        budget=_CallBudget(),
    )

    projected = rank_input["ranker_slate"][0]
    assert selected == RAW_5
    assert rank_input["typed_gap_status"] is None
    assert rank_result["status"] == "selected_frozen_survivor"
    assert projected["candidate_digest"] == RAW_5
    assert projected["formula"]["affirmative_phrase"] == "bird-like object"
    assert projected["formula"]["required_witnesses"] == [
        {
            "kind": "shape_appearance",
            "statement": "one silhouette has a beak-like protrusion",
            "witness_id": "witness_00",
        }
    ]
    assert projected["formula"]["accepted_variants"] == [
        "either left- or right-facing silhouette"
    ]
    assert projected["formula"]["near_miss_boundaries"] == [
        "a plain oval does not qualify"
    ]

    def forbidden_digest_keys(value: object) -> list[str]:
        if isinstance(value, dict):
            return [
                key
                for key, item in value.items()
                if key.endswith("_digest") and key != "candidate_digest"
            ] + [
                key
                for item in value.values()
                for key in forbidden_digest_keys(item)
            ]
        if isinstance(value, list):
            return [key for item in value for key in forbidden_digest_keys(item)]
        return []

    assert forbidden_digest_keys(projected) == []
    assert len(prompts) == 1
    assert "one silhouette has a beak-like protrusion" in prompts[0]
    assert RAW_6 not in prompts[0]
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
        journal_module, "ObjectBongardNamedImageTurnJournalTransport", FakeJournal
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
    monkeypatch.setattr(
        campaign,
        "_task_semantic_proposer_presentation",
        lambda *_args, **_kwargs: (("panel_000.png", b"support"),),
    )
    prepared_input = SimpleNamespace(
        prompt="both frozen support buckets",
        output_schema={"type": "object"},
        alias_bindings=tuple(
            {"historical_role": role, "usable": True}
            for role in ((0,) * 6 + (1,) * 6)
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
            panels=(),
            named_image_transport=lambda *_args, **_kwargs: object(),
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
    )
    from bongard.tests.test_object_scene_semantic_registry import (
        _discovery_artifact,
    )
    from bongard.tests.test_object_scene_visual_frontend import _scene
    from bongard.tests.test_prototype_scene_observer import _receipt
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
    payload = {
        "side0_positive": [
            _semantic_concept(
                "panel",
                "paired visible forms",
                _semantic_support_bindings(
                    semantic_prepared, "side0_positive", "panel"
                ),
                witness_kind="spatial_relation",
                witness_statement=(
                    "two visible forms occupy distinct regions of the panel"
                ),
                accepted_variants=(
                    "touching forms count when both visible extents remain distinct",
                ),
                near_miss_boundaries=(
                    "one form divided only by an internal mark does not qualify",
                ),
            ),
            _semantic_concept(
                "entity",
                "not pointed",
                _semantic_support_bindings(
                    semantic_prepared, "side0_positive", "entity"
                ),
                witness_statement="the entity has one visibly sharp terminal tip",
            ),
        ],
        "side1_positive": [
            _semantic_concept(
                "entity",
                "unequal edge lengths",
                _semantic_support_bindings(
                    semantic_prepared, "side1_positive", "entity"
                ),
                witness_kind="count_relation",
                witness_statement=(
                    "two corresponding straight edges visibly differ in length"
                ),
                near_miss_boundaries=(
                    "perspective-only apparent shortening does not qualify",
                ),
            )
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
    panels = tuple(
        campaign._TaskSupportPanel(
            index,
            artifact.scene_id,
            f"bd_scene_00_{index:02d}",
            index // 6,
            SimpleNamespace(
                exact_png_bytes=_scene(index),
                exact_png_digest=(
                    "sha256:"
                    + hashlib.sha256(_scene(index)).hexdigest()
                ),
            ),
            object(),
            artifact.inventory,
            role_rows[index]["neutral_panel_digest"],
        )
        for index, artifact in enumerate(discovery_artifacts)
    )
    presentation = campaign._task_semantic_proposer_presentation(
        panels, semantic_prepared
    )
    semantic_prepared_record = campaign._semantic_prepared_record(
        task=task,
        discovery_batch={"batch_digest": ADDRESS_4},
        role_reveal={"role_reveal_digest": ADDRESS_1},
        semantic_prepared=semantic_prepared,
        presentation=presentation,
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
    assert len(presentation) == 24
    assert semantic_prepared_record["named_image_count"] == len(presentation)
    assert semantic_prepared_record["named_image_commitments"] == [
        {
            "name": name,
            "byte_count": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }
        for name, data in presentation
    ]

    physical_calls: list[str] = []

    seen_names: list[str] = []

    def offline_transport(prompt, paths, names, schema, **_kwargs):
        physical_calls.append("semantic-proposer")
        seen_names.extend(names)
        return CodexStructuredResult(
            payload, _receipt(prompt, paths, names, schema, payload)
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
        panels=panels,
        named_image_transport=offline_transport,
        budget=budget,
    )

    assert physical_calls == ["semantic-proposer"]
    assert seen_names == [name for name, _ in presentation]
    assert all(name.startswith("panel_") for name in seen_names)
    assert all("query" not in name for name in seen_names)
    assert budget.snapshot().semantic_proposer_calls == 1
    assert proposal.status == "proposed"
    assert len(registry.tags) == 2
    assert {item.tag for item in registry.tags} == {
        "paired visible forms",
        "unequal edge lengths",
    }
    assert {
        item.tag: item.orientation_constraint for item in registry.tags
    } == {
        "paired visible forms": "group0_positive",
        "unequal edge lengths": "group1_positive",
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
    restored_registry, registry_record, registry_receipt = (
        campaign._freeze_task_registry(
            prepared=prepared,
            task=task,
            discovery_batch={"batch_digest": ADDRESS_4},
            role_reveal={"role_reveal_digest": ADDRESS_1},
            semantic_prepared_record=semantic_prepared_record,
            semantic_proposal_record=record,
            semantic_proposal=proposal,
            registry=registry,
            discovery_artifacts=discovery_artifacts,
            role_rows=role_rows,
        )
    )
    assert restored_registry == registry
    assert registry_record[
        "proposal_orientation_preserved_in_registered_tag_identity"
    ] is True
    assert registry_record[
        "registered_visual_evaluators_receive_orientation_constraint_metadata"
    ] is False
    assert registry_record["registry_orientation_manifest"] == [
        {
            "tag_id": item.tag_id,
            "tag_digest": item.tag_digest,
            "orientation_constraint": item.orientation_constraint,
        }
        for item in registry.tags
    ]
    assert dict(
        store.verify(registry_receipt, expected_data=registry_record)
    ) == registry_record
    journal_root = tmp_path / "journals" / "semantic_registry_proposer"
    assert {path.name for path in journal_root.iterdir()} == {
        "manifest.json",
        "claim.json",
        "result.json",
        "outcome.json",
    }
    manifest = json.loads((journal_root / "manifest.json").read_text("utf-8"))
    assert manifest["modality"] == "named_image_structured"
    assert manifest["named_images"] == semantic_prepared_record[
        "named_image_commitments"
    ]
    required_bindings = semantic_prepared.model_view[
        "required_positive_binding_panels"
    ]
    for orientation in ("side0_positive", "side1_positive"):
        assert all(
            [panel_alias for panel_alias, _ in item.support_bindings]
            == required_bindings[orientation]
            for item in getattr(proposal, orientation)
        )
    side1_rows = {
        row["panel_alias"]: row
        for row in semantic_prepared.model_view[
            "side1_support_descriptions"
        ]
    }
    for concept in proposal.side1_positive:
        assert all(
            target_alias
            in {
                entity["entity_alias"]
                for entity in side1_rows[panel_alias][
                    "proposal_atlas_map"
                ]
            }
            for panel_alias, target_alias in concept.support_bindings
        )

    replay_calls: list[object] = []

    def forbidden_replay_transport(*args, **kwargs):
        replay_calls.append((args, kwargs))
        raise AssertionError("cold replay attempted a physical proposer call")

    monkeypatch.setattr(
        campaign, "_forbidden_named_transport", forbidden_replay_transport
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
            panels=panels,
        )
    )
    assert replay_calls == []
    assert physical_calls == ["semantic-proposer"]
    assert replayed_proposal == proposal
    assert replayed_registry == registry
    assert replay_summary_digest == record["proposer_journal_summary_digest"]


def _task_result_fixture(
    *,
    queried: bool,
    semantic_valid: bool = True,
    gap_status: str = TYPED_LANGUAGE_GAP,
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
                    gap_status
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
    assert _validate_task_result_record(gap)["status"] == TYPED_LANGUAGE_GAP
    for status in (TYPED_SELECTIVITY_GAP, TYPED_GROUNDING_REPEATABILITY_GAP):
        assert (
            _validate_task_result_record(
                _task_result_fixture(queried=False, gap_status=status)
            )["status"]
            == status
        )
    with pytest.raises(
        ObjectBongardScenePredicateCampaignCommandError,
        match="policy or budget differs",
    ):
        _validate_task_result_record(
            _task_result_fixture(
                queried=False,
                gap_status="typed_version_space_gap",
            )
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
