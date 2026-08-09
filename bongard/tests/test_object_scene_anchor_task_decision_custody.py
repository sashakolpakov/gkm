"""Synthetic tests for pure-Python anchor task-decision custody."""

from __future__ import annotations

from copy import deepcopy
import hashlib

import pytest

from bongard.canonical import canonical_digest, canonical_json
from bongard.object_bongard_batch import ObjectBongardTaskPlan
from bongard.object_bongard_release_gate import (
    ObjectBongardExecutionPrecommit,
    ObjectBongardReleaseStore,
    ObjectBongardTaskCommitProtocol,
    ObjectBongardTaskFreezeProtocol,
    _precommit_content,
    persist_object_bongard_task_commit,
    persist_object_bongard_task_freeze,
)
from bongard.object_scene_anchor_candidate_ranker import (
    freeze_object_scene_anchor_rank_input,
)
from bongard.object_scene_anchor_python_bridge import (
    freeze_object_scene_anchor_python_bridge,
)
from bongard.object_scene_anchor_support_observation_join import (
    build_object_scene_anchor_support_observation_plan,
    finalize_object_scene_anchor_support_observations,
)
from bongard.object_scene_anchor_support_preparation import (
    ObjectSceneAnchorSupportCorpusRuntimeBundle,
    ObjectSceneAnchorSupportPanelInput,
    build_object_scene_anchor_support_panel,
    freeze_object_scene_anchor_support_corpus,
)
from bongard.object_scene_anchor_task_decision_custody import (
    ObjectSceneAnchorTaskDecisionCommit,
    ObjectSceneAnchorTaskDecisionCustodyError,
    ObjectSceneAnchorTaskDecisionFreeze,
    cold_verify_object_scene_anchor_task_decision_commit,
    cold_verify_object_scene_anchor_task_decision_freeze,
    commit_object_scene_anchor_task_decision,
    freeze_object_scene_anchor_task_decision,
    object_scene_anchor_task_decision_custody_algorithm_digest,
    object_scene_anchor_task_decision_custody_source_digest,
)
from bongard.tests.test_object_scene_anchor_candidate_ranker import (
    _Transport,
    _ranker,
)
from bongard.tests.test_object_scene_anchor_support_observation_join import (
    _artifact,
    _language,
    _panel_png,
)


def _address(value: object) -> str:
    return "sha256:" + canonical_digest(value)


def _precommit(task: ObjectBongardTaskPlan) -> ObjectBongardExecutionPrecommit:
    support = tuple(
        sorted((*task.side_0_support_panel_ids, *task.side_1_support_panel_ids))
    )
    query = tuple(sorted((task.side_0_query_panel_id, task.side_1_query_panel_id)))
    values = {
        "batch_plan_digest": _address({"batch": task.record_digest}),
        "batch_algorithm_digest": _address({"batch_algorithm": 1}),
        "batch_source_digest": _address({"batch_source": 1}),
        "release_gate_source_digest": _address({"release_gate_source": 1}),
        "release_descriptor_digest": _address({"release": 1}),
        "archive_record_digest": _address({"archive_record": 1}),
        "archive_digest": _address({"archive": 1}),
        "archive_central_directory_digest": _address({"central": 1}),
        "corpus_digest": _address({"corpus": 1}),
        "exposure_predecessor_digest": _address({"predecessor": 1}),
        "task_inventory_digest": _address({"inventory": [task.task_id]}),
        "train_task_ids_digest": _address([task.task_id]),
        "exact_used_task_ids_digest": _address([]),
        "selected_task_ids": (task.task_id,),
        "authorized_support_panel_ids": support,
        "sealed_query_panel_ids": query,
        "runtime_source_bindings": (("custody_source", _address({"source": 1})),),
        "configuration": (("headless", True),),
        "exposure_observed_at": "2026-08-09T12:00:00Z",
        "exposure_actor": "synthetic-test",
        "exposure_purpose": "anchor-custody",
        "exposure_source": "offline-synthetic",
    }
    provisional = object.__new__(ObjectBongardExecutionPrecommit)
    for name, item in values.items():
        object.__setattr__(provisional, name, item)
    return ObjectBongardExecutionPrecommit(
        **values,
        record_digest=_address(_precommit_content(provisional)),
    )


def _corpus(task: ObjectBongardTaskPlan) -> ObjectSceneAnchorSupportCorpusRuntimeBundle:
    panel_ids = (*task.side_0_support_panel_ids, *task.side_1_support_panel_ids)
    panels = []
    for index, panel_id in enumerate(panel_ids):
        payload = _panel_png(index)
        panel_input = ObjectSceneAnchorSupportPanelInput(
            panel_alias=f"panel_{index:03d}",
            support_bucket_index=0 if index < 6 else 1,
            source_digest="1" * 64,
            source_panel_binding_digest=hashlib.sha256(
                f"custody-binding-{index}".encode("ascii")
            ).hexdigest(),
            source_ordinal=index,
            task_id=task.task_id,
            panel_id=panel_id,
            original_panel_png_digest=hashlib.sha256(payload).hexdigest(),
            exact_original_png_bytes=payload,
        )
        panels.append(build_object_scene_anchor_support_panel(panel_input))
    return ObjectSceneAnchorSupportCorpusRuntimeBundle(
        freeze=freeze_object_scene_anchor_support_corpus(
            "1" * 64, tuple(item.freeze for item in panels)
        ),
        panels=tuple(panels),
    )


@pytest.fixture(scope="module")
def custody_parents():
    task = ObjectBongardTaskPlan.create(
        "bd_anchor_custody", seed_digest=_address({"seed": "custody"})
    )
    precommit = _precommit(task)
    corpus = _corpus(task)
    language = _language(corpus)
    runtime = build_object_scene_anchor_support_observation_plan(corpus, language)
    batch_artifact, calls = _artifact(runtime)
    assert calls == 2
    result = finalize_object_scene_anchor_support_observations(
        runtime.plan, batch_artifact
    )
    spaces = (
        result.bucket0_positive_version_space,
        result.bucket1_positive_version_space,
    )
    nonempty = tuple(item for item in spaces if item.survivor_candidate_digests)
    empty = tuple(item for item in spaces if not item.survivor_candidate_digests)
    assert len(nonempty) == len(empty) == 1
    rank_input = freeze_object_scene_anchor_rank_input(nonempty[0])
    transport = _Transport()
    response = _ranker(transport)(
        nonempty[0], expected_rank_input_digest=rank_input.rank_input_digest
    )
    assert transport.calls == 1
    bridge = freeze_object_scene_anchor_python_bridge(
        response,
        spaces[0],
        spaces[1],
        expected_response_digest=response.response_digest,
        expected_rank_input_digest=rank_input.rank_input_digest,
    )
    parents = {
        "task": task,
        "execution_precommit": precommit,
        "support_observation_plan": runtime.plan,
        "support_observation_result": result,
        "rank_input": rank_input,
        "rank_response": response,
        "bridge": bridge,
        "predicate": bridge.predicate,
    }
    freeze = freeze_object_scene_anchor_task_decision(**parents)
    return parents, freeze, empty[0]


def test_freeze_satisfies_release_protocol_and_cold_replays(custody_parents) -> None:
    parents, freeze, empty = custody_parents
    data = freeze.to_data()

    assert isinstance(freeze, ObjectBongardTaskFreezeProtocol)
    assert ObjectSceneAnchorTaskDecisionFreeze.from_data(data) == freeze
    assert freeze.version_space_digest == freeze.support_version_space_digest
    assert freeze.version_space_digest == freeze.rank_input.version_space_digest
    assert freeze.rank_response_digest == freeze.rank_response.response_digest
    assert freeze.selected_predicate_digest == freeze.selected_predicate.predicate_digest
    assert freeze.bridge.omitted_version_space == empty
    assert data["omitted_gap_digest"] == empty.gap.gap_digest
    assert data["selected_python_predicate_mapping"]["pure_python_evaluation"] is True
    assert data["lean_present"] is False
    assert data["lean_required"] is False
    assert data["lean_removable"] is True
    assert data["query_bytes_included"] is False
    assert data["query_labels_included"] is False
    assert freeze.algorithm_digest == (
        object_scene_anchor_task_decision_custody_algorithm_digest()
    )
    assert len(object_scene_anchor_task_decision_custody_source_digest()) == 64
    assert not freeze.version_space_digest.startswith("sha256:")
    assert freeze.record_digest == _address(
        {key: item for key, item in data.items() if key != "record_digest"}
    )
    assert cold_verify_object_scene_anchor_task_decision_freeze(
        freeze,
        **parents,
        expected_freeze_digest=freeze.record_digest,
    ) == freeze


def test_freeze_rejects_missing_gap_proof_and_foreign_task(custody_parents) -> None:
    parents, _freeze, _empty = custody_parents
    direct = freeze_object_scene_anchor_python_bridge(
        parents["rank_response"],
        parents["support_observation_result"].bucket0_positive_version_space,
        expected_response_digest=parents["rank_response"].response_digest,
        expected_rank_input_digest=parents["rank_input"].rank_input_digest,
    )
    with pytest.raises(ObjectSceneAnchorTaskDecisionCustodyError, match="bridge"):
        freeze_object_scene_anchor_task_decision(
            **{**parents, "bridge": direct, "predicate": direct.predicate}
        )

    foreign = ObjectBongardTaskPlan.create(
        "bd_foreign_custody", seed_digest=_address({"seed": "foreign"})
    )
    with pytest.raises(
        ObjectSceneAnchorTaskDecisionCustodyError, match="task|panel"
    ):
        freeze_object_scene_anchor_task_decision(
            **{**parents, "task": foreign}
        )


def test_resealed_freeze_tamper_fails_or_is_caught_by_cold_replay(
    custody_parents,
) -> None:
    parents, freeze, _empty = custody_parents
    tampered = deepcopy(freeze.to_data())
    tampered["rank_response_digest"] = "0" * 64
    tampered["record_digest"] = _address(
        {key: item for key, item in tampered.items() if key != "record_digest"}
    )
    with pytest.raises(ObjectSceneAnchorTaskDecisionCustodyError, match="binding"):
        ObjectSceneAnchorTaskDecisionFreeze.from_data(tampered)

    external = deepcopy(freeze.to_data())
    external["support_observation_result_digest"] = "1" * 64
    external["record_digest"] = _address(
        {key: item for key, item in external.items() if key != "record_digest"}
    )
    resealed = ObjectSceneAnchorTaskDecisionFreeze.from_data(external)
    with pytest.raises(ObjectSceneAnchorTaskDecisionCustodyError, match="cold replay"):
        cold_verify_object_scene_anchor_task_decision_freeze(
            resealed,
            **parents,
            expected_freeze_digest=resealed.record_digest,
        )


def test_commit_binds_real_persisted_freeze_payload_and_release_protocol(
    custody_parents,
    tmp_path,
) -> None:
    _parents, freeze, _empty = custody_parents
    payload = canonical_json(freeze.to_data()) + b"\n"
    store = ObjectBongardReleaseStore(tmp_path / "release-store")
    receipt = persist_object_bongard_task_freeze(store=store, freeze=freeze)
    commit = commit_object_scene_anchor_task_decision(
        freeze=freeze,
        exact_freeze_payload=payload,
        task_freeze_store_receipt=receipt,
    )

    assert isinstance(commit, ObjectBongardTaskCommitProtocol)
    assert ObjectSceneAnchorTaskDecisionCommit.from_data(commit.to_data()) == commit
    assert commit.task_freeze_digest == freeze.record_digest
    assert commit.exact_freeze_payload_digest == receipt.payload_digest
    assert commit.task_freeze_store_receipt_digest == receipt.record_digest
    assert persist_object_bongard_task_commit(
        store=store, commit=commit
    ).object_digest == commit.record_digest
    assert cold_verify_object_scene_anchor_task_decision_commit(
        commit,
        freeze=freeze,
        exact_freeze_payload=payload,
        expected_commit_digest=commit.record_digest,
    ) == commit
    commit.assert_matches(freeze, payload)

    with pytest.raises(ObjectSceneAnchorTaskDecisionCustodyError, match="canonical"):
        commit_object_scene_anchor_task_decision(
            freeze=freeze,
            exact_freeze_payload=payload + b" ",
            task_freeze_store_receipt=receipt,
        )
