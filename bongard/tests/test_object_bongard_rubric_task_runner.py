"""Focused tests for the durable two-rank Python rubric-task runner."""

from __future__ import annotations

import ast
from copy import deepcopy
import hashlib
import inspect
import json
from pathlib import Path

import pytest

from bongard.object_bongard_batch import ObjectBongardTaskPlan
from bongard.object_bongard_release_gate import (
    ObjectBongardTaskCommitProtocol,
    ObjectBongardTaskFreezeProtocol,
)
from bongard.object_bongard_rubric_observer import ObjectBongardRubricSpec, RubricScope
from bongard.object_bongard_rubric_task_runner import (
    ObjectBongardRubricTaskFreeze,
    ObjectBongardRubricTaskFreezeCommit,
    ObjectBongardRubricTaskRunArchive,
    ObjectBongardRubricTaskRunStatus,
    ObjectBongardRubricTaskRunnerError,
    cold_replay_object_bongard_rubric_task,
    run_object_bongard_rubric_task,
)
from bongard.tests.test_object_bongard_rubric_version_space import _observed_artifact
from bongard.tests.test_object_bongard_semantics import (
    CONTEXT_DIGEST,
    TASK_ID,
    _describe as _describe_semantic,
)


# Metadata-only seed: panel 6 is the sealed query on both fixture sides.
TASK_SEED = "bb32db7dcc68752cd7cad883937d281a26e94ece7c4388256196cce7ecee7f99"


def _parents():
    plan = ObjectBongardTaskPlan.create(TASK_ID, seed_digest=TASK_SEED)
    semantic, calls = _describe_semantic()
    assert calls == 1
    assert plan.side_0_support_panel_ids == semantic.group_panel_ids[0]
    assert plan.side_1_support_panel_ids == semantic.group_panel_ids[1]
    specs = tuple(
        ObjectBongardRubricSpec.from_semantic_artifact(
            semantic,
            expected_artifact_digest=semantic.artifact_digest,
            candidate_rank=rank,
        )
        for rank in (0, 1)
    )
    return plan, semantic, specs


def _block(spec, plan, *, exact: bool):
    positive = (3, 3) if exact else (0, 0)
    positives = tuple(
        _observed_artifact(
            panel_id,
            image_index=index,
            object_interval=positive,
            scene_interval=(0, 0),
            rubric_spec=spec,
        )
        for index, panel_id in enumerate(plan.side_0_support_panel_ids)
    )
    negatives = tuple(
        _observed_artifact(
            panel_id,
            image_index=index + 6,
            object_interval=(0, 0),
            scene_interval=(0, 0),
            rubric_spec=spec,
        )
        for index, panel_id in enumerate(plan.side_1_support_panel_ids)
    )
    return positives, negatives


def _support(specs, plan, *, rank_0_exact=True, rank_1_exact=True):
    first = _block(specs[0], plan, exact=rank_0_exact)
    second = _block(specs[1], plan, exact=rank_1_exact)
    return (first[0], second[0]), (first[1], second[1])


class _DurableFreezeStore:
    def __init__(self) -> None:
        self.payload: bytes | None = None
        self.commit: ObjectBongardRubricTaskFreezeCommit | None = None
        self.persisted = False
        self.reloaded = False

    def persist(self, payload: bytes) -> ObjectBongardRubricTaskFreezeCommit:
        assert not self.persisted
        freeze = ObjectBongardRubricTaskFreeze.from_data(json.loads(payload))
        self.payload = payload
        self.persisted = True
        receipt_digest = "sha256:" + hashlib.sha256(
            b"durable-two-rank-freeze-store-receipt:" + payload
        ).hexdigest()
        self.commit = ObjectBongardRubricTaskFreezeCommit.seal(
            freeze, payload, task_freeze_store_receipt_digest=receipt_digest
        )
        return self.commit

    def reload(self, commit_data) -> bytes:
        assert self.persisted and self.payload is not None and self.commit is not None
        assert ObjectBongardRubricTaskFreezeCommit.from_data(commit_data) == self.commit
        self.reloaded = True
        return self.payload


class _Forbidden:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, *_args, **_kwargs):
        self.calls += 1
        raise AssertionError("later execution phase was called")


def _complete_run(*, selected_rank: int = 0, uncertain_positive_query: bool = True):
    plan, semantic, specs = _parents()
    positives, negatives = _support(
        specs,
        plan,
        rank_0_exact=selected_rank == 0,
        rank_1_exact=True,
    )
    store = _DurableFreezeStore()
    selected_spec = specs[selected_rank]
    positive_interval = (2, 3) if uncertain_positive_query else (4, 4)
    query_0 = _observed_artifact(
        plan.side_0_query_panel_id,
        image_index=20,
        object_interval=positive_interval,
        scene_interval=(0, 0),
        rubric_spec=selected_spec,
    )
    query_1 = _observed_artifact(
        plan.side_1_query_panel_id,
        image_index=21,
        object_interval=(0, 0),
        scene_interval=(0, 0),
        rubric_spec=selected_spec,
    )
    query_calls = 0

    def query_source(freeze_data, commit_data):
        nonlocal query_calls
        query_calls += 1
        assert store.persisted and store.reloaded
        freeze = ObjectBongardRubricTaskFreeze.from_data(freeze_data)
        commit = ObjectBongardRubricTaskFreezeCommit.from_data(commit_data)
        assert freeze.record_digest == commit.task_freeze_digest
        assert freeze.selected_rubric_spec == selected_spec
        assert freeze.selected_formula == freeze.selected_candidate.formula
        assert freeze.sealed_query_panel_ids == (
            plan.side_0_query_panel_id,
            plan.side_1_query_panel_id,
        )
        return {"side_0": query_0, "side_1": query_1}

    archive = run_object_bongard_rubric_task(
        plan,
        semantic,
        positives,
        negatives,
        execution_precommit_digest=CONTEXT_DIGEST,
        freeze_committer=store.persist,
        freeze_reloader=store.reload,
        query_source=query_source,
    )
    assert query_calls == 1
    return archive, store


def test_fixed_slate_freezes_before_query_and_cold_replays() -> None:
    archive, store = _complete_run(selected_rank=0, uncertain_positive_query=True)

    assert archive.status is ObjectBongardRubricTaskRunStatus.COMPLETE
    assert archive.freeze is not None and archive.freeze_commit is not None
    assert isinstance(archive.freeze, ObjectBongardTaskFreezeProtocol)
    assert isinstance(archive.freeze_commit, ObjectBongardTaskCommitProtocol)
    assert store.persisted and store.reloaded
    assert archive.selection_replay_calls_made == 1
    assert archive.freeze_commit_calls_made == archive.freeze_reload_calls_made == 1
    assert archive.query_source_calls_made == 1
    assert tuple(item.scope for item in archive.slate_selection.ordered_candidates) == (
        RubricScope.OBJECT,
        RubricScope.SCENE,
        RubricScope.OBJECT,
        RubricScope.SCENE,
    )
    assert archive.selected_candidate == archive.slate_selection.ordered_candidates[0]
    assert archive.score_denominator == 2
    assert archive.correct_count == 1
    assert archive.abstention_count == 1
    assert archive.query_results[0].abstained is True
    assert archive.query_results[1].correct is True
    data = archive.to_data()
    assert data["selection_model_calls_made"] == 0
    assert "rank_response" not in data
    assert data["query_uses_selected_python_candidate_only"] is True

    replay = cold_replay_object_bongard_rubric_task(
        data, expected_archive_digest=archive.record_digest
    )
    assert replay == archive
    assert replay.to_data()["cold_replay_model_calls"] == 0
    assert ObjectBongardRubricTaskRunArchive.from_data(data) == archive


def test_rank_one_is_selected_only_after_both_rank_zero_scopes_fail() -> None:
    archive, _store = _complete_run(selected_rank=1, uncertain_positive_query=False)
    assert archive.status is ObjectBongardRubricTaskRunStatus.COMPLETE
    assert archive.selected_candidate == archive.slate_selection.ordered_candidates[2]
    assert archive.selected_rubric_spec == archive.rubric_specs[1]
    assert archive.freeze is not None
    assert archive.freeze.selected_rubric_spec == archive.rubric_specs[1]
    assert archive.correct_count == 2


def test_no_exact_survivor_never_freezes_or_opens_queries() -> None:
    plan, semantic, specs = _parents()
    positives, negatives = _support(
        specs, plan, rank_0_exact=False, rank_1_exact=False
    )
    forbidden = _Forbidden()
    archive = run_object_bongard_rubric_task(
        plan,
        semantic,
        positives,
        negatives,
        execution_precommit_digest=CONTEXT_DIGEST,
        freeze_committer=forbidden,
        freeze_reloader=forbidden,
        query_source=forbidden,
    )
    assert archive.status is ObjectBongardRubricTaskRunStatus.LANGUAGE_GAP
    assert archive.slate_selection.selected_candidate is None
    assert forbidden.calls == 0
    assert archive.score_denominator == 0
    assert archive.freeze is archive.freeze_commit is None
    assert archive.side_0_query is archive.side_1_query is None
    assert cold_replay_object_bongard_rubric_task(
        archive, expected_archive_digest=archive.record_digest
    ) == archive


def test_query_source_cannot_run_after_bad_reload_or_return_wrong_rank() -> None:
    plan, semantic, specs = _parents()
    positives, negatives = _support(specs, plan)
    store = _DurableFreezeStore()
    forbidden_query = _Forbidden()
    with pytest.raises(
        ObjectBongardRubricTaskRunnerError,
        match="durable two-rank freeze reload differs",
    ):
        run_object_bongard_rubric_task(
            plan,
            semantic,
            positives,
            negatives,
            execution_precommit_digest=CONTEXT_DIGEST,
            freeze_committer=store.persist,
            freeze_reloader=lambda _commit: b"{}\n",
            query_source=forbidden_query,
        )
    assert forbidden_query.calls == 0

    store = _DurableFreezeStore()
    wrong_0 = _observed_artifact(
        plan.side_0_query_panel_id,
        image_index=20,
        object_interval=(4, 4),
        scene_interval=(0, 0),
        rubric_spec=specs[1],
    )
    wrong_1 = _observed_artifact(
        plan.side_1_query_panel_id,
        image_index=21,
        object_interval=(0, 0),
        scene_interval=(0, 0),
        rubric_spec=specs[1],
    )
    with pytest.raises(ObjectBongardRubricTaskRunnerError, match="frozen selected spec"):
        run_object_bongard_rubric_task(
            plan,
            semantic,
            positives,
            negatives,
            execution_precommit_digest=CONTEXT_DIGEST,
            freeze_committer=store.persist,
            freeze_reloader=store.reload,
            query_source=lambda *_args: {"side_0": wrong_0, "side_1": wrong_1},
        )


def test_tamper_rejected_and_public_api_has_no_ranker_or_lean_dependency() -> None:
    archive, _store = _complete_run(selected_rank=0, uncertain_positive_query=False)
    changed = deepcopy(archive.to_data())
    changed["correct_count"] = 0
    with pytest.raises(ObjectBongardRubricTaskRunnerError):
        cold_replay_object_bongard_rubric_task(
            changed, expected_archive_digest=archive.record_digest
        )

    assert "ranker" not in inspect.signature(run_object_bongard_rubric_task).parameters
    source = (Path(__file__).parents[1] / "object_bongard_rubric_task_runner.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.append(node.module)
    assert not any("lean" in name.lower() for name in imported)
    assert archive.to_data()["lean_present"] is False
    assert archive.to_data()["lean_removable"] is True
