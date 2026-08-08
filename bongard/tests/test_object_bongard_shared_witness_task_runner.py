"""End-to-end offline tests for the structured shared-witness task runner."""

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
from bongard.object_bongard_shared_witness import (
    ObjectBongardSharedWitnessRubricSpec,
    build_shared_witness_rubric_specs,
)
from bongard.object_bongard_shared_witness_observer import (
    ObjectBongardSharedWitnessPanelArtifact,
    _endpoint_mapping,
    _neutral_endpoint_cues,
    observe_object_bongard_shared_witness_panel,
)
from bongard.object_bongard_shared_witness_semantics import (
    describe_object_bongard_shared_witness_support,
    object_bongard_shared_witness_semantics_output_schema,
)
from bongard.object_bongard_release_gate import (
    ObjectBongardTaskCommitProtocol,
    ObjectBongardTaskFreezeProtocol,
)
from bongard.object_bongard_shared_witness_task_runner import (
    ObjectBongardSharedWitnessTaskFreeze,
    ObjectBongardSharedWitnessTaskFreezeCommit,
    ObjectBongardSharedWitnessTaskRunArchive,
    ObjectBongardSharedWitnessTaskRunStatus,
    ObjectBongardSharedWitnessTaskRunnerError,
    cold_replay_object_bongard_shared_witness_task,
    run_object_bongard_shared_witness_task,
)
from bongard.tests.test_prototype_scene_observer import (
    CONTEXT_DIGEST,
    EFFORT,
    LAUNCHER_DIGEST,
    MODEL,
    NO_TOOLS_KWARGS,
    _png,
    _receipt,
)
from bongard.transport import CodexStructuredResult


TASK_ID = "ff_nact2_5_0042"
TASK_SEED = "f" * 64


def _semantic_payload() -> dict[str, object]:
    return {
        "proposal_0": {
            "shared_anchor": "patterned loop network",
            "visual_axis": "junction organization",
            "group_0_endpoint": "shared hub",
            "group_1_endpoint": "distributed junction",
        },
        "proposal_1": {
            "shared_anchor": "decorated contour network",
            "visual_axis": "contour termination",
            "group_0_endpoint": "closed circuit",
            "group_1_endpoint": "free ended",
        },
    }


@lru_cache(maxsize=1)
def _parents() -> tuple[
    ObjectBongardTaskPlan,
    object,
    tuple[ObjectBongardSharedWitnessRubricSpec, ObjectBongardSharedWitnessRubricSpec],
]:
    plan = ObjectBongardTaskPlan.create(TASK_ID, seed_digest=TASK_SEED)
    panel_ids = plan.side_0_support_panel_ids + plan.side_1_support_panel_ids
    images = {panel_id: _png(index + 1) for index, panel_id in enumerate(panel_ids)}
    calls = 0

    def transport(prompt, paths, names, schema, **_kwargs):
        nonlocal calls
        calls += 1
        assert schema == object_bongard_shared_witness_semantics_output_schema()
        payload = _semantic_payload()
        return CodexStructuredResult(
            payload, _receipt(prompt, paths, names, schema, payload)
        )

    semantic = describe_object_bongard_shared_witness_support(
        task_id=TASK_ID,
        group_0_panel_ids=plan.side_0_support_panel_ids,
        group_1_panel_ids=plan.side_1_support_panel_ids,
        support_png_by_panel_id=images,
        observation_context_digest=CONTEXT_DIGEST,
        model=MODEL,
        reasoning_effort=EFFORT,
        expected_launcher_digest=LAUNCHER_DIGEST,
        **NO_TOOLS_KWARGS,
        transport=transport,
    )
    assert calls == 1
    specs = build_shared_witness_rubric_specs(
        semantic, expected_artifact_digest=semantic.artifact_digest
    )
    return plan, semantic, specs


def _entity_payload(
    spec: ObjectBongardSharedWitnessRubricSpec,
    schema: dict[str, object],
    *,
    target: str,
    foil: str,
) -> dict[str, object]:
    cues = _neutral_endpoint_cues(spec)
    target_id, foil_id = _endpoint_mapping(spec, cues)
    judgments = {target_id: target, foil_id: foil}
    scope = schema["properties"]["entities"]["items"]["properties"]["scope"]["enum"][0]  # type: ignore[index]
    return {
        "entity_id": "e00",
        "scope": scope,
        "bbox_q16": {"x0": 1000, "y0": 2000, "x1": 12000, "y1": 15000},
        "locator": "leftmost visible individual figure",
        "anchor_support": "clear",
        "anchor_evidence": "one complete patterned figure is visible",
        "cue_support": [
            {
                "cue_id": cue.cue_id,
                "judgment": judgments[cue.cue_id],
                "evidence": "the visible endpoint organization is inspectable",
            }
            for cue in cues
        ],
    }


@lru_cache(maxsize=None)
def _artifact(
    rank: int,
    panel_id: str,
    disposition: Disposition,
) -> ObjectBongardSharedWitnessPanelArtifact:
    spec = _parents()[2][rank]
    panel = _png(sum(panel_id.encode("utf-8")) % 41 + rank + 20)

    def transport(prompt, paths, names, schema, **_kwargs):
        if disposition is Disposition.ERROR:
            raise RuntimeError("synthetic observer transport bomb")
        if disposition is Disposition.INDETERMINATE:
            payload = {"inventory_status": "uncertain", "entities": []}
        else:
            target, foil = (
                ("clear", "none")
                if disposition is Disposition.PRESENT
                else ("none", "clear")
            )
            payload = {
                "inventory_status": "complete",
                "entities": [
                    _entity_payload(spec, schema, target=target, foil=foil)
                ],
            }
        return CodexStructuredResult(
            payload, _receipt(prompt, paths, names, schema, payload)
        )

    artifact = observe_object_bongard_shared_witness_panel(
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
    assert artifact.observation.disposition is disposition
    return artifact


def _block(
    rank: int,
    target_states: tuple[Disposition, ...],
    foil_states: tuple[Disposition, ...],
) -> tuple[
    tuple[ObjectBongardSharedWitnessPanelArtifact, ...],
    tuple[ObjectBongardSharedWitnessPanelArtifact, ...],
]:
    plan = _parents()[0]
    return (
        tuple(
            _artifact(rank, panel_id, state)
            for panel_id, state in zip(
                plan.side_0_support_panel_ids, target_states, strict=True
            )
        ),
        tuple(
            _artifact(rank, panel_id, state)
            for panel_id, state in zip(
                plan.side_1_support_panel_ids, foil_states, strict=True
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


def _errored(rank: int):
    return _block(
        rank,
        (Disposition.PRESENT,) * 5 + (Disposition.ERROR,),
        (Disposition.CERTIFIED_ABSENT,) * 6,
    )


def _support(rank_0: str, rank_1: str):
    choices = {
        "strict": _strict,
        "bounded": _bounded,
        "rejected": _rejected,
        "errored": _errored,
    }
    first = choices[rank_0](0)
    second = choices[rank_1](1)
    return (first[0], second[0]), (first[1], second[1])


class _DurableStore:
    def __init__(self) -> None:
        self.payload: bytes | None = None
        self.commit: ObjectBongardSharedWitnessTaskFreezeCommit | None = None
        self.persisted = False
        self.reloaded = False

    def persist(self, payload: bytes) -> ObjectBongardSharedWitnessTaskFreezeCommit:
        assert not self.persisted
        freeze = ObjectBongardSharedWitnessTaskFreeze.from_data(json.loads(payload))
        self.payload = payload
        self.persisted = True
        receipt = "sha256:" + hashlib.sha256(b"shared-freeze-store:" + payload).hexdigest()
        self.commit = ObjectBongardSharedWitnessTaskFreezeCommit.seal(
            freeze, payload, task_freeze_store_receipt_digest=receipt
        )
        return self.commit

    def reload(self, commit_data) -> bytes:
        assert self.persisted and self.payload is not None and self.commit is not None
        assert ObjectBongardSharedWitnessTaskFreezeCommit.from_data(commit_data) == self.commit
        self.reloaded = True
        return self.payload


class _Forbidden:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, *_args, **_kwargs):
        self.calls += 1
        raise AssertionError("later execution phase was called")


def _run_complete(*, rank_0: str = "bounded", side_0_state=Disposition.PRESENT):
    plan, semantic, specs = _parents()
    targets, foils = _support(rank_0, "strict")
    selected_rank = 0 if rank_0 in {"bounded", "strict"} else 1
    store = _DurableStore()
    query_calls = 0

    def query_source(freeze_data, commit_data):
        nonlocal query_calls
        query_calls += 1
        assert store.persisted and store.reloaded
        freeze = ObjectBongardSharedWitnessTaskFreeze.from_data(freeze_data)
        commit = ObjectBongardSharedWitnessTaskFreezeCommit.from_data(commit_data)
        assert freeze.record_digest == commit.task_freeze_digest
        assert freeze.selected_rubric_spec == specs[selected_rank]
        assert freeze.selected_candidate.candidate_rank == selected_rank
        assert freeze.selected_support_version_space.support_artifacts
        assert freeze.to_data()["selected_support_artifact_count"] == 12
        assert freeze.to_data()["selected_entity_evidence_count"] >= 10
        assert freeze.to_data()["query_bytes_included"] is False
        # Query observations are created only inside this post-reload callback.
        return {
            "side_0": _artifact(
                selected_rank, plan.side_0_query_panel_id, side_0_state
            ),
            "side_1": _artifact(
                selected_rank,
                plan.side_1_query_panel_id,
                Disposition.CERTIFIED_ABSENT,
            ),
        }

    archive = run_object_bongard_shared_witness_task(
        plan,
        semantic,
        targets,
        foils,
        execution_precommit_digest=CONTEXT_DIGEST,
        freeze_committer=store.persist,
        freeze_reloader=store.reload,
        query_source=query_source,
    )
    assert query_calls == 1
    return archive, store


def test_full_ir_and_entities_freeze_before_two_queries_then_cold_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive, store = _run_complete()
    assert archive.status is ObjectBongardSharedWitnessTaskRunStatus.COMPLETE
    assert store.persisted and store.reloaded
    assert archive.selected_candidate is not None
    assert archive.selected_candidate.candidate_rank == 0
    assert archive.freeze is not None
    assert isinstance(archive.freeze, ObjectBongardTaskFreezeProtocol)
    assert isinstance(archive.freeze_commit, ObjectBongardTaskCommitProtocol)
    assert archive.freeze.selected_rubric_spec == archive.rubric_specs[0]
    assert archive.score_denominator == 2
    assert (archive.correct_count, archive.incorrect_count) == (2, 0)
    assert archive.coverage_count == 2
    assert archive.accuracy_ppm == archive.coverage_ppm == 1_000_000

    calls = 0

    def transport_bomb(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("cold replay attempted a model transport")

    import bongard.object_bongard_shared_witness_observer as observer

    monkeypatch.setattr(observer, "run_codex_named_images_structured", transport_bomb)
    assert cold_replay_object_bongard_shared_witness_task(
        archive.to_data(), expected_archive_digest=archive.record_digest
    ) == archive
    assert calls == 0


def test_rank_one_only_after_rank_zero_rejection_and_fixed_denominator() -> None:
    archive, _store = _run_complete(
        rank_0="rejected", side_0_state=Disposition.INDETERMINATE
    )
    assert archive.selected_candidate is not None
    assert archive.selected_candidate.candidate_rank == 1
    assert archive.freeze is not None
    assert archive.freeze.selected_rubric_spec == archive.rubric_specs[1]
    assert (archive.correct_count, archive.incorrect_count) == (1, 1)
    assert archive.abstention_count == 1
    assert archive.coverage_count == 1
    assert archive.score_denominator == 2


def test_error_gap_never_opens_freeze_or_query() -> None:
    plan, semantic, _specs = _parents()
    targets, foils = _support("errored", "errored")
    forbidden = _Forbidden()
    archive = run_object_bongard_shared_witness_task(
        plan,
        semantic,
        targets,
        foils,
        execution_precommit_digest=CONTEXT_DIGEST,
        freeze_committer=forbidden,
        freeze_reloader=forbidden,
        query_source=forbidden,
    )
    assert archive.status is ObjectBongardSharedWitnessTaskRunStatus.ERROR_GAP
    assert forbidden.calls == 0
    assert archive.freeze is archive.freeze_commit is None
    assert archive.side_0_query is archive.side_1_query is None
    assert (archive.correct_count, archive.incorrect_count) == (0, 2)
    assert archive.abstention_count == 2
    assert cold_replay_object_bongard_shared_witness_task(
        archive, expected_archive_digest=archive.record_digest
    ) == archive


def test_bad_reload_stops_before_query_and_archive_tamper_fails() -> None:
    plan, semantic, _specs = _parents()
    targets, foils = _support("bounded", "strict")
    store = _DurableStore()
    forbidden_query = _Forbidden()

    def bad_reload(commit_data):
        return store.reload(commit_data) + b" "

    with pytest.raises(
        ObjectBongardSharedWitnessTaskRunnerError, match="reload differs"
    ):
        run_object_bongard_shared_witness_task(
            plan,
            semantic,
            targets,
            foils,
            execution_precommit_digest=CONTEXT_DIGEST,
            freeze_committer=store.persist,
            freeze_reloader=bad_reload,
            query_source=forbidden_query,
        )
    assert forbidden_query.calls == 0

    archive, _ = _run_complete()
    tampered = deepcopy(archive.to_data())
    tampered["correct_count"] = 1
    with pytest.raises(ObjectBongardSharedWitnessTaskRunnerError):
        ObjectBongardSharedWitnessTaskRunArchive.from_data(tampered)


def test_runner_has_no_lean_atlas_or_ranker_import() -> None:
    source = Path(__file__).parents[1] / "object_bongard_shared_witness_task_runner.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imports.append(node.module)
    lowered = tuple(item.lower() for item in imports)
    assert not any(
        "lean" in item or "atlas" in item or "ranker" in item for item in lowered
    )
