"""Focused tests for the durable whole-panel two-rank task runner."""

from __future__ import annotations

import ast
from copy import deepcopy
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import pytest

from bongard.evidence import Disposition
from bongard.object_bongard_batch import ObjectBongardTaskPlan
from bongard.object_bongard_panel_rubric_observer import (
    ObjectBongardPanelRubricArtifact,
    observe_object_bongard_panel_rubric,
)
from bongard.object_bongard_panel_rubric_task_runner import (
    ObjectBongardPanelRubricTaskFreeze,
    ObjectBongardPanelRubricTaskFreezeCommit,
    ObjectBongardPanelRubricTaskRunArchive,
    ObjectBongardPanelRubricTaskRunStatus,
    ObjectBongardPanelRubricTaskRunnerError,
    cold_replay_object_bongard_panel_rubric_task,
    run_object_bongard_panel_rubric_task,
)
from bongard.object_bongard_release_gate import (
    ObjectBongardTaskCommitProtocol,
    ObjectBongardTaskFreezeProtocol,
)
from bongard.object_bongard_rubric_language import ObjectBongardRubricSpec
from bongard.tests.test_object_bongard_semantics import (
    CONTEXT_DIGEST,
    TASK_ID,
    _describe as _describe_semantic,
)
from bongard.tests.test_prototype_scene_observer import (
    EFFORT,
    LAUNCHER_DIGEST,
    MODEL,
    NO_TOOLS_KWARGS,
    _png,
    _receipt,
)
from bongard.transport import CodexStructuredResult


# This seed seals panel 6 as query on both sides of the semantic fixture.
TASK_SEED = "bb32db7dcc68752cd7cad883937d281a26e94ece7c4388256196cce7ecee7f99"


@lru_cache(maxsize=1)
def _parents() -> tuple[
    ObjectBongardTaskPlan,
    object,
    tuple[ObjectBongardRubricSpec, ObjectBongardRubricSpec],
]:
    plan = ObjectBongardTaskPlan.create(TASK_ID, seed_digest=TASK_SEED)
    semantic, calls = _describe_semantic()
    assert calls == 1
    assert semantic.group_panel_ids == (
        plan.side_0_support_panel_ids,
        plan.side_1_support_panel_ids,
    )
    specs = tuple(
        ObjectBongardRubricSpec.from_semantic_artifact(
            semantic,
            expected_artifact_digest=semantic.artifact_digest,
            candidate_rank=rank,
        )
        for rank in (0, 1)
    )
    return plan, semantic, specs  # type: ignore[return-value]


@lru_cache(maxsize=None)
def _artifact(
    rank: int,
    panel_id: str,
    disposition: Disposition,
) -> ObjectBongardPanelRubricArtifact:
    spec = _parents()[2][rank]
    panel = _png(sum(panel_id.encode("utf-8")) % 31 + rank)
    payloads = {
        Disposition.PRESENT: {"lower": 3, "upper": 4},
        Disposition.CERTIFIED_ABSENT: {"lower": 0, "upper": 1},
        Disposition.INDETERMINATE: {"lower": 2, "upper": 2},
    }

    def transport(prompt, paths, names, schema, **_kwargs):
        if disposition is Disposition.ERROR:
            raise RuntimeError("synthetic whole-panel observer failure")
        payload = payloads[disposition]
        return CodexStructuredResult(
            payload,
            _receipt(prompt, paths, names, schema, payload),
        )

    return observe_object_bongard_panel_rubric(
        panel,
        panel_id=panel_id,
        rubric_spec=spec,
        expected_panel_sha256=hashlib.sha256(panel).hexdigest(),
        expected_rubric_spec_digest=spec.spec_digest,
        observation_context_digest=CONTEXT_DIGEST,
        model=MODEL,
        reasoning_effort=EFFORT,
        expected_launcher_digest=LAUNCHER_DIGEST,
        **NO_TOOLS_KWARGS,
        transport=transport,
    )


def _block(
    rank: int,
    positive_states: tuple[Disposition, ...],
    negative_states: tuple[Disposition, ...],
) -> tuple[
    tuple[ObjectBongardPanelRubricArtifact, ...],
    tuple[ObjectBongardPanelRubricArtifact, ...],
]:
    plan = _parents()[0]
    assert len(positive_states) == len(negative_states) == 6
    return (
        tuple(
            _artifact(rank, panel_id, state)
            for panel_id, state in zip(
                plan.side_0_support_panel_ids,
                positive_states,
                strict=True,
            )
        ),
        tuple(
            _artifact(rank, panel_id, state)
            for panel_id, state in zip(
                plan.side_1_support_panel_ids,
                negative_states,
                strict=True,
            )
        ),
    )


def _strict(rank: int):
    return _block(
        rank,
        (Disposition.PRESENT,) * 6,
        (Disposition.CERTIFIED_ABSENT,) * 6,
    )


def _bounded(rank: int):
    return _block(
        rank,
        (Disposition.PRESENT,) * 5 + (Disposition.INDETERMINATE,),
        (Disposition.CERTIFIED_ABSENT,) * 5 + (Disposition.INDETERMINATE,),
    )


def _rejected(rank: int):
    return _block(
        rank,
        (Disposition.PRESENT,) * 5 + (Disposition.CERTIFIED_ABSENT,),
        (Disposition.CERTIFIED_ABSENT,) * 6,
    )


def _support(*, rank_0: str, rank_1: str):
    choices = {"strict": _strict, "bounded": _bounded, "rejected": _rejected}
    first = choices[rank_0](0)
    second = choices[rank_1](1)
    return (first[0], second[0]), (first[1], second[1])


class _DurableStore:
    def __init__(self) -> None:
        self.payload: bytes | None = None
        self.commit: ObjectBongardPanelRubricTaskFreezeCommit | None = None
        self.persisted = False
        self.reloaded = False

    def persist(self, payload: bytes) -> ObjectBongardPanelRubricTaskFreezeCommit:
        assert not self.persisted
        freeze = ObjectBongardPanelRubricTaskFreeze.from_data(json.loads(payload))
        self.payload = payload
        self.persisted = True
        receipt = "sha256:" + hashlib.sha256(
            b"whole-panel-freeze-store:" + payload
        ).hexdigest()
        self.commit = ObjectBongardPanelRubricTaskFreezeCommit.seal(
            freeze,
            payload,
            task_freeze_store_receipt_digest=receipt,
        )
        return self.commit

    def reload(self, commit_data) -> bytes:
        assert self.persisted
        assert self.payload is not None and self.commit is not None
        assert ObjectBongardPanelRubricTaskFreezeCommit.from_data(
            commit_data
        ) == self.commit
        self.reloaded = True
        return self.payload


class _Forbidden:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, *_args, **_kwargs):
        self.calls += 1
        raise AssertionError("a later execution phase was called")


def _run_complete(
    *,
    rank_0: str = "bounded",
    positive_query: Disposition = Disposition.INDETERMINATE,
):
    plan, semantic, specs = _parents()
    positives, negatives = _support(rank_0=rank_0, rank_1="strict")
    selected_rank = 0 if rank_0 in ("bounded", "strict") else 1
    store = _DurableStore()
    queries = {
        "side_0": _artifact(
            selected_rank,
            plan.side_0_query_panel_id,
            positive_query,
        ),
        "side_1": _artifact(
            selected_rank,
            plan.side_1_query_panel_id,
            Disposition.CERTIFIED_ABSENT,
        ),
    }
    query_calls = 0

    def query_source(freeze_data, commit_data):
        nonlocal query_calls
        query_calls += 1
        assert store.persisted and store.reloaded
        freeze = ObjectBongardPanelRubricTaskFreeze.from_data(freeze_data)
        commit = ObjectBongardPanelRubricTaskFreezeCommit.from_data(commit_data)
        assert freeze.record_digest == commit.task_freeze_digest
        assert freeze.selected_rubric_spec == specs[selected_rank]
        assert freeze.selected_candidate == freeze.slate_selection.selected_candidate
        assert freeze.selected_formula == freeze.selected_candidate.formula
        assert freeze.sealed_query_panel_ids == (
            plan.side_0_query_panel_id,
            plan.side_1_query_panel_id,
        )
        return queries

    archive = run_object_bongard_panel_rubric_task(
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


@pytest.mark.parametrize(
    "uncertain",
    (Disposition.INDETERMINATE, Disposition.ERROR),
)
def test_bounded_rank_zero_freezes_before_query_and_reports_coverage(
    uncertain: Disposition,
) -> None:
    archive, store = _run_complete(positive_query=uncertain)

    assert archive.status is ObjectBongardPanelRubricTaskRunStatus.COMPLETE
    assert archive.freeze is not None and archive.freeze_commit is not None
    assert isinstance(archive.freeze, ObjectBongardTaskFreezeProtocol)
    assert isinstance(archive.freeze_commit, ObjectBongardTaskCommitProtocol)
    assert store.persisted and store.reloaded
    assert archive.selected_candidate == archive.slate_selection.ordered_candidates[0]
    assert archive.slate_selection.selected_has_strict_exact_support is False
    assert archive.freeze.selected_rubric_spec == archive.rubric_specs[0]
    assert archive.score_denominator == 2
    assert archive.correct_count == 1
    assert archive.incorrect_count == 1
    assert archive.abstention_count == 1
    assert archive.coverage_count == 1
    assert archive.accuracy_ppm == archive.coverage_ppm == 500_000
    assert archive.query_results[0].incorrect is True
    assert archive.query_results[0].covered is False
    assert archive.query_results[0].abstained is True
    assert archive.query_results[1].correct is True
    assert archive.to_data()["selection_model_calls_made"] == 0

    assert cold_replay_object_bongard_panel_rubric_task(
        archive.to_data(),
        expected_archive_digest=archive.record_digest,
    ) == archive
    assert ObjectBongardPanelRubricTaskRunArchive.from_data(
        archive.to_data()
    ) == archive


def test_rank_one_is_selected_only_after_rank_zero_rejection() -> None:
    archive, _store = _run_complete(
        rank_0="rejected",
        positive_query=Disposition.PRESENT,
    )
    assert archive.selected_candidate == archive.slate_selection.ordered_candidates[1]
    assert archive.selected_rubric_spec == archive.rubric_specs[1]
    assert archive.freeze is not None
    assert archive.freeze.selected_rubric_spec == archive.rubric_specs[1]
    assert archive.correct_count == archive.coverage_count == 2
    assert archive.incorrect_count == archive.abstention_count == 0
    assert archive.accuracy_ppm == archive.coverage_ppm == 1_000_000


def test_no_survivor_keeps_denominator_and_never_opens_later_phases() -> None:
    plan, semantic, _specs = _parents()
    positives, negatives = _support(rank_0="rejected", rank_1="rejected")
    forbidden = _Forbidden()
    archive = run_object_bongard_panel_rubric_task(
        plan,
        semantic,
        positives,
        negatives,
        execution_precommit_digest=CONTEXT_DIGEST,
        freeze_committer=forbidden,
        freeze_reloader=forbidden,
        query_source=forbidden,
    )
    assert archive.status is ObjectBongardPanelRubricTaskRunStatus.LANGUAGE_GAP
    assert archive.slate_selection.selected_candidate is None
    assert forbidden.calls == 0
    assert archive.score_denominator == 2
    assert archive.correct_count == archive.coverage_count == 0
    assert archive.incorrect_count == archive.abstention_count == 2
    assert archive.accuracy_ppm == archive.coverage_ppm == 0
    assert archive.freeze is archive.freeze_commit is None
    assert archive.side_0_query is archive.side_1_query is None
    assert cold_replay_object_bongard_panel_rubric_task(
        archive,
        expected_archive_digest=archive.record_digest,
    ) == archive


def test_bad_reload_stops_before_query_and_wrong_query_spec_is_rejected() -> None:
    plan, semantic, specs = _parents()
    positives, negatives = _support(rank_0="bounded", rank_1="strict")
    store = _DurableStore()
    forbidden_query = _Forbidden()

    def bad_reload(commit_data):
        payload = store.reload(commit_data)
        return payload + b" "

    with pytest.raises(
        ObjectBongardPanelRubricTaskRunnerError,
        match="reload differs",
    ):
        run_object_bongard_panel_rubric_task(
            plan,
            semantic,
            positives,
            negatives,
            execution_precommit_digest=CONTEXT_DIGEST,
            freeze_committer=store.persist,
            freeze_reloader=bad_reload,
            query_source=forbidden_query,
        )
    assert forbidden_query.calls == 0

    second_store = _DurableStore()

    def wrong_spec_query(_freeze_data, _commit_data):
        return {
            "side_0": _artifact(
                1,
                plan.side_0_query_panel_id,
                Disposition.PRESENT,
            ),
            "side_1": _artifact(
                1,
                plan.side_1_query_panel_id,
                Disposition.CERTIFIED_ABSENT,
            ),
        }

    with pytest.raises(
        ObjectBongardPanelRubricTaskRunnerError,
        match="frozen selected spec",
    ):
        run_object_bongard_panel_rubric_task(
            plan,
            semantic,
            positives,
            negatives,
            execution_precommit_digest=CONTEXT_DIGEST,
            freeze_committer=second_store.persist,
            freeze_reloader=second_store.reload,
            query_source=wrong_spec_query,
        )
    assert second_store.reloaded
    assert specs[0] != specs[1]


def test_archive_tamper_and_parent_precommit_mismatch_fail_closed() -> None:
    archive, _store = _run_complete(positive_query=Disposition.PRESENT)
    tampered = deepcopy(archive.to_data())
    tampered["coverage_count"] = 1
    with pytest.raises(ObjectBongardPanelRubricTaskRunnerError):
        ObjectBongardPanelRubricTaskRunArchive.from_data(tampered)

    plan, semantic, _specs = _parents()
    positives, negatives = _support(rank_0="bounded", rank_1="strict")
    forbidden = _Forbidden()
    with pytest.raises(
        ObjectBongardPanelRubricTaskRunnerError,
        match="semantic artifact",
    ):
        run_object_bongard_panel_rubric_task(
            plan,
            semantic,
            positives,
            negatives,
            execution_precommit_digest="sha256:" + "a" * 64,
            freeze_committer=forbidden,
            freeze_reloader=forbidden,
            query_source=forbidden,
        )
    assert forbidden.calls == 0


def test_runner_has_no_atlas_ranker_or_lean_import() -> None:
    source = Path(__file__).parents[1] / "object_bongard_panel_rubric_task_runner.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imports.append(node.module)
    lowered = tuple(item.lower() for item in imports)
    assert not any(
        "atlas" in item or "ranker" in item or "lean" in item
        for item in lowered
    )
