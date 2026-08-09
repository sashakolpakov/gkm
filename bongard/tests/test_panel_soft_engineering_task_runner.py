"""Offline chronology and replay tests for the panel-soft engineering runner."""

from __future__ import annotations

import base64
from copy import deepcopy
from dataclasses import replace
import hashlib
import json

import pytest

from bongard.canonical import canonical_digest, canonical_json
from bongard.object_bongard_batch import ObjectBongardTaskPlan
from bongard.panel_soft_engineering_task_runner import (
    PanelSoftEngineeringProposerTerminal,
    PanelSoftEngineeringTaskFreeze,
    PanelSoftEngineeringTaskFreezeCommit,
    PanelSoftEngineeringTaskRunArchive,
    PanelSoftEngineeringTaskRunStatus,
    PanelSoftEngineeringTaskRunnerError,
    cold_replay_panel_soft_engineering_task,
    run_panel_soft_engineering_task,
)
from bongard.panel_soft_observer import (
    aggregate_panel_soft_observer_artifacts,
    observe_panel_soft_vocabulary,
    panel_soft_observer_view,
)
from bongard.panel_soft_predicate import (
    PanelSoftEngineeringPredicatePair,
    PanelSoftEngineeringVersionSpace,
)
from bongard.panel_soft_proposer import (
    PanelSoftProposerError,
    PanelSoftProposerStatus,
    propose_panel_soft_atoms,
)
from bongard.tests.test_panel_soft_proposer import _payload as proposer_payload
from bongard.tests.test_prototype_scene_observer import (
    EFFORT,
    LAUNCHER_DIGEST,
    MODEL,
    NO_TOOLS_KWARGS,
    _png,
    _receipt,
)
from bongard.transport import CodexStructuredResult


TASK_ID = "bd_panel_soft_runner_0000"
TASK_SEED = "sha256:" + "7" * 64
PRECOMMIT = "sha256:" + "8" * 64


def _unique_receipt(prompt, paths, names, schema, payload, serial):
    base = _receipt(prompt, paths, names, schema, payload)
    provisional = replace(
        base,
        thread_id=f"00000000-0000-4000-8000-{serial:012d}",
        event_stream_digest=hashlib.sha256(
            f"panel-soft-runner-event-{serial}".encode()
        ).hexdigest(),
    )
    body = provisional.to_dict()
    body.pop("receipt_digest")
    return replace(provisional, receipt_digest=canonical_digest(body))


def _fixture(*, indeterminate_support: bool = False, support_offset: int = 100):
    task = ObjectBongardTaskPlan.create(TASK_ID, seed_digest=TASK_SEED)
    support_ids = (*task.side_0_support_panel_ids, *task.side_1_support_panel_ids)
    support_pngs = tuple(_png(support_offset + index) for index in range(12))
    support_map = dict(zip(support_ids, support_pngs, strict=True))

    def proposer_transport(prompt, paths, names, schema, **_kwargs):
        payload = proposer_payload()
        return CodexStructuredResult(
            payload, _receipt(prompt, paths, names, schema, payload)
        )

    proposer = propose_panel_soft_atoms(
        support_pngs,
        support_panel_ids=support_ids,
        expected_support_sha256=tuple(
            hashlib.sha256(item).hexdigest() for item in support_pngs
        ),
        model=MODEL,
        reasoning_effort=EFFORT,
        expected_launcher_digest=LAUNCHER_DIGEST,
        **NO_TOOLS_KWARGS,
        transport=proposer_transport,
    )
    assert proposer.vocabulary is not None
    vocabulary = proposer.vocabulary
    serial = 100

    def observe(panel_id, panel, native_side):
        nonlocal serial
        view = panel_soft_observer_view(vocabulary)
        atoms = {item.atom_digest: item for item in vocabulary.atoms}
        payload = {
            item.alias: (
                "indeterminate"
                if indeterminate_support and panel_id in support_map
                else "present"
                if atoms[item.atom_digest].orientation == f"side{native_side}_positive"
                else "mismatch"
            )
            for item in view
        }

        def transport(prompt, paths, names, schema, **_kwargs):
            nonlocal serial
            serial += 1
            receipt = _unique_receipt(
                prompt, paths, names, schema, payload, serial
            )
            return CodexStructuredResult(payload, receipt)

        return observe_panel_soft_vocabulary(
            panel,
            panel_id=panel_id,
            vocabulary=vocabulary,
            expected_panel_sha256=hashlib.sha256(panel).hexdigest(),
            expected_vocabulary_digest=vocabulary.vocabulary_digest,
            model=MODEL,
            reasoning_effort=EFFORT,
            expected_launcher_digest=LAUNCHER_DIGEST,
            **NO_TOOLS_KWARGS,
            transport=transport,
        )

    support_artifacts = tuple(
        observe(panel_id, panel, 0 if index < 6 else 1)
        for index, (panel_id, panel) in enumerate(
            zip(support_ids, support_pngs, strict=True)
        )
    )
    return task, proposer, support_map, support_artifacts, observe


def _predicate_pair(task, proposer, support_artifacts):
    assert proposer.vocabulary is not None
    table = aggregate_panel_soft_observer_artifacts(
        support_artifacts,
        ordered_panel_commitments=tuple(
            (item.panel_id, item.panel_png_digest) for item in support_artifacts
        ),
        expected_vocabulary=proposer.vocabulary,
        expected_contract=support_artifacts[0].contract,
    )
    space = PanelSoftEngineeringVersionSpace.create(
        table, task.side_0_support_panel_ids, task.side_1_support_panel_ids
    )
    return PanelSoftEngineeringPredicatePair.create_deterministic_baseline(space)


def _injected_rank_artifact(space):
    from bongard.panel_soft_ranker import PanelSoftRankInput
    from bongard.tests.test_panel_soft_ranker import _run as run_fake_ranker

    rank_input = PanelSoftRankInput.freeze(space)
    formulas = rank_input.formula_by_alias
    first = tuple(
        max(
            (
                alias for alias, formula in formulas.items()
                if formula.orientation == orientation
            ),
            key=lambda alias: (len(formulas[alias].atom_digests), alias),
        )
        for orientation in ("side0_positive", "side1_positive")
    )
    order = first + tuple(
        alias for alias in rank_input.candidate_aliases if alias not in first
    )
    artifact, transport = run_fake_ranker(space, order)
    assert transport.calls == 1
    assert artifact.transport_provenance.benchmark_sealable is False
    return artifact


def test_complete_freezes_reloads_before_query_and_cold_replays() -> None:
    task, proposer, support_map, support_artifacts, observe = _fixture()
    events = []
    durable = {}

    def commit_freeze(payload):
        events.append("commit")
        freeze = PanelSoftEngineeringTaskFreeze.from_data(json.loads(payload))
        durable["payload"] = payload
        receipt = "sha256:" + canonical_digest(
            {"payload_sha256": hashlib.sha256(payload).hexdigest()}
        )
        commit = PanelSoftEngineeringTaskFreezeCommit.seal(
            freeze, payload, task_freeze_store_receipt_digest=receipt
        )
        durable["commit"] = commit.to_data()
        return commit

    def reload_freeze(commit_data):
        events.append("reload")
        assert commit_data == durable["commit"]
        return durable["payload"]

    def query_source(freeze_data, commit_data):
        events.append("query")
        assert events == ["commit", "reload", "query"]
        assert canonical_json(freeze_data) + b"\n" == durable["payload"]
        assert commit_data == durable["commit"]
        rows = {}
        for side_index, (side, panel_id) in enumerate(
            (
                ("side_0", task.side_0_query_panel_id),
                ("side_1", task.side_1_query_panel_id),
            )
        ):
            panel = _png(220 + side_index)
            rows[side] = (panel, observe(panel_id, panel, side_index))
        return rows

    archive = run_panel_soft_engineering_task(
        task,
        proposer,
        support_map,
        support_artifacts,
        execution_precommit_digest=PRECOMMIT,
        selection_mode="deterministic_baseline",
        ranker=None,
        freeze_committer=commit_freeze,
        freeze_reloader=reload_freeze,
        query_source=query_source,
    )
    assert events == ["commit", "reload", "query"]
    assert archive.status is PanelSoftEngineeringTaskRunStatus.COMPLETE
    assert (archive.correct_count, archive.determinate_count) == (2, 2)
    assert (archive.abstain_count, archive.error_count) == (0, 0)
    assert archive.accuracy_ppm == archive.coverage_ppm == 1_000_000
    assert archive.freeze is not None and archive.freeze_commit is not None
    assert archive.freeze.to_data()["query_pixels_included"] is False
    assert archive.freeze.to_data()["support_only_codex_ranker_supported"] is True
    assert archive.rank_artifact is None
    assert archive.ranker_callback_invocations == 0
    assert archive.allow_unverified_rank_artifact is False
    assert archive.rank_artifact_benchmark_sealable is None
    assert PanelSoftEngineeringTaskRunArchive.from_data(archive.to_data()) == archive
    assert cold_replay_panel_soft_engineering_task(
        archive, expected_record_digest=archive.record_digest
    ) == archive

    tampered = deepcopy(archive.to_data())
    first_id = next(iter(tampered["support_png_base64_by_panel_id"]))
    tampered["support_png_base64_by_panel_id"][first_id] = tampered[
        "query_png_base64_by_side"
    ]["side_0"]
    with pytest.raises(PanelSoftEngineeringTaskRunnerError):
        PanelSoftEngineeringTaskRunArchive.from_data(tampered)


def test_ranked_mode_calls_once_and_freezes_exact_rank_artifact() -> None:
    task, proposer, support_map, support_artifacts, observe = _fixture(
        support_offset=700
    )
    durable = {}
    callback_calls = 0
    selected = {}

    def ranker(space):
        nonlocal callback_calls
        callback_calls += 1
        artifact = _injected_rank_artifact(space)
        selected["artifact"] = artifact
        return artifact

    def commit_freeze(payload):
        freeze = PanelSoftEngineeringTaskFreeze.from_data(json.loads(payload))
        durable["payload"] = payload
        commit = PanelSoftEngineeringTaskFreezeCommit.seal(
            freeze,
            payload,
            task_freeze_store_receipt_digest="sha256:" + canonical_digest(
                {"ranked-freeze": hashlib.sha256(payload).hexdigest()}
            ),
        )
        durable["commit"] = commit.to_data()
        return commit

    def reload_freeze(commit_data):
        assert commit_data == durable["commit"]
        return durable["payload"]

    def query_source(freeze_data, commit_data):
        assert commit_data == durable["commit"]
        assert freeze_data["selection_mode"] == "support_only_codex_ranker"
        assert freeze_data["allow_unverified_rank_artifact"] is True
        assert freeze_data["rank_artifact_benchmark_sealable"] is False
        assert freeze_data["rank_artifact_digest"] == selected[
            "artifact"
        ].artifact_digest
        rows = {}
        for side_index, (side, panel_id) in enumerate(
            (
                ("side_0", task.side_0_query_panel_id),
                ("side_1", task.side_1_query_panel_id),
            )
        ):
            panel = _png(820 + side_index)
            rows[side] = (panel, observe(panel_id, panel, side_index))
        return rows

    archive = run_panel_soft_engineering_task(
        task,
        proposer,
        support_map,
        support_artifacts,
        execution_precommit_digest=PRECOMMIT,
        selection_mode="support_only_codex_ranker",
        ranker=ranker,
        allow_unverified_rank_artifact=True,
        freeze_committer=commit_freeze,
        freeze_reloader=reload_freeze,
        query_source=query_source,
    )
    artifact = selected["artifact"]
    baseline = _predicate_pair(task, proposer, support_artifacts)
    assert callback_calls == 1
    assert archive.ranker_callback_invocations == 1
    assert archive.rank_artifact == artifact
    assert archive.allow_unverified_rank_artifact is True
    assert archive.rank_artifact_benchmark_sealable is False
    assert archive.predicate_pair is not None
    assert archive.predicate_pair.selection_mode == "support_only_codex_ranker"
    assert (
        archive.predicate_pair.side0_formula_digest,
        archive.predicate_pair.side1_formula_digest,
    ) == artifact.selected_formula_digests
    assert archive.predicate_pair != baseline
    assert archive.freeze is not None
    assert archive.freeze.rank_artifact_digest == artifact.artifact_digest
    assert archive.freeze.rank_input_digest == artifact.rank_input.rank_input_digest
    assert archive.freeze.rank_receipt_digest == artifact.receipt.receipt_digest
    assert archive.freeze.allow_unverified_rank_artifact is True
    assert archive.freeze.rank_artifact_benchmark_sealable is False
    assert archive.freeze_commit is not None
    assert archive.freeze_commit.allow_unverified_rank_artifact is True
    assert archive.freeze_commit.rank_artifact_benchmark_sealable is False
    assert archive.freeze_commit.selection_mode == "support_only_codex_ranker"
    assert cold_replay_panel_soft_engineering_task(
        archive, expected_record_digest=archive.record_digest
    ) == archive
    assert callback_calls == 1


def test_ranked_mode_default_rejects_unsealable_artifact_before_freeze_query() -> None:
    task, proposer, support_map, support_artifacts, _observe = _fixture(
        support_offset=850
    )
    events = []

    def ranker(space):
        events.append("rank")
        return _injected_rank_artifact(space)

    def forbidden(*_args, **_kwargs):
        events.append("forbidden")
        raise AssertionError("freeze or query was reached")

    with pytest.raises(
        PanelSoftEngineeringTaskRunnerError,
        match="not benchmark-sealable",
    ):
        run_panel_soft_engineering_task(
            task,
            proposer,
            support_map,
            support_artifacts,
            execution_precommit_digest=PRECOMMIT,
            selection_mode="support_only_codex_ranker",
            ranker=ranker,
            freeze_committer=forbidden,
            freeze_reloader=forbidden,
            query_source=forbidden,
        )
    assert events == ["rank"]


def test_unverified_rank_override_is_rejected_outside_ranked_mode() -> None:
    task = ObjectBongardTaskPlan.create(TASK_ID, seed_digest=TASK_SEED)
    forbidden = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("artifact processing was reached")
    )
    with pytest.raises(
        PanelSoftEngineeringTaskRunnerError,
        match="only in ranked mode",
    ):
        run_panel_soft_engineering_task(
            task,
            object(),  # type: ignore[arg-type]
            {},
            (),
            execution_precommit_digest=PRECOMMIT,
            selection_mode="deterministic_baseline",
            ranker=None,
            allow_unverified_rank_artifact=True,
            freeze_committer=forbidden,
            freeze_reloader=forbidden,
            query_source=forbidden,
        )


def test_no_survivor_is_typed_and_never_calls_freeze_or_query() -> None:
    task, proposer, support_map, support_artifacts, _observe = _fixture(
        indeterminate_support=True
    )
    forbidden = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("later phase was called")
    )
    archive = run_panel_soft_engineering_task(
        task,
        proposer,
        support_map,
        support_artifacts,
        execution_precommit_digest=PRECOMMIT,
        selection_mode="support_only_codex_ranker",
        ranker=forbidden,
        freeze_committer=forbidden,
        freeze_reloader=forbidden,
        query_source=forbidden,
    )
    assert archive.status is PanelSoftEngineeringTaskRunStatus.SUPPORT_GAP
    assert archive.support_gap is not None
    assert set(archive.support_gap.missing_orientations) == {
        "side0_positive", "side1_positive"
    }
    assert archive.predicate_pair is archive.freeze is archive.freeze_commit is None
    assert archive.query_source_calls_made == 0
    assert archive.ranker_callback_invocations == 0
    assert archive.allow_unverified_rank_artifact is False
    assert archive.rank_artifact_benchmark_sealable is None
    assert (archive.correct_count, archive.determinate_count) == (0, 0)
    assert (archive.abstain_count, archive.error_count) == (2, 0)
    assert cold_replay_panel_soft_engineering_task(
        archive, expected_record_digest=archive.record_digest
    ) == archive


def test_freeze_rejects_unrelated_proposer_and_support_artifacts() -> None:
    task, proposer, _support_map, support_artifacts, _observe = _fixture()
    pair = _predicate_pair(task, proposer, support_artifacts)
    (
        unrelated_task,
        unrelated_proposer,
        _unrelated_support_map,
        unrelated_artifacts,
        _unrelated_observe,
    ) = _fixture(support_offset=500)
    assert unrelated_task == task

    with pytest.raises(PanelSoftEngineeringTaskRunnerError):
        PanelSoftEngineeringTaskFreeze.seal(
            task_plan=task,
            execution_precommit_digest=PRECOMMIT,
            proposer_artifact=unrelated_proposer,
            support_artifacts=support_artifacts,
            predicate_pair=pair,
            rank_artifact=None,
            allow_unverified_rank_artifact=False,
        )
    with pytest.raises(PanelSoftEngineeringTaskRunnerError):
        PanelSoftEngineeringTaskFreeze.seal(
            task_plan=task,
            execution_precommit_digest=PRECOMMIT,
            proposer_artifact=proposer,
            support_artifacts=unrelated_artifacts,
            predicate_pair=pair,
            rank_artifact=None,
            allow_unverified_rank_artifact=False,
        )


def test_failed_proposer_is_typed_before_support_observation() -> None:
    task = ObjectBongardTaskPlan.create(TASK_ID, seed_digest=TASK_SEED)
    support_ids = (*task.side_0_support_panel_ids, *task.side_1_support_panel_ids)
    support_pngs = tuple(_png(300 + index) for index in range(12))
    support_map = dict(zip(support_ids, support_pngs, strict=True))

    def failed_transport(*_args, **_kwargs):
        raise RuntimeError("synthetic proposer transport failure")

    proposer = propose_panel_soft_atoms(
        support_pngs,
        support_panel_ids=support_ids,
        expected_support_sha256=tuple(
            hashlib.sha256(item).hexdigest() for item in support_pngs
        ),
        model=MODEL,
        reasoning_effort=EFFORT,
        expected_launcher_digest=LAUNCHER_DIGEST,
        **NO_TOOLS_KWARGS,
        transport=failed_transport,
    )
    assert proposer.status is PanelSoftProposerStatus.TRANSPORT_ERROR
    forbidden = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("post-proposer phase was called")
    )
    result = run_panel_soft_engineering_task(
        task,
        proposer,
        support_map,
        (),
        execution_precommit_digest=PRECOMMIT,
        selection_mode="support_only_codex_ranker",
        ranker=forbidden,
        freeze_committer=forbidden,
        freeze_reloader=forbidden,
        query_source=forbidden,
    )
    assert isinstance(result, PanelSoftEngineeringProposerTerminal)
    assert result.to_data()["terminal_stage"] == "proposer"
    assert result.to_data()["support_observer_artifact_count"] == 0
    assert result.to_data()["error_count"] == 2
    assert result.to_data()["query_pixels_released"] is False
    assert result.allow_unverified_rank_artifact is False
    assert result.rank_artifact_benchmark_sealable is None
    assert PanelSoftEngineeringProposerTerminal.from_data(result.to_data()) == result
    assert cold_replay_panel_soft_engineering_task(
        result, expected_record_digest=result.record_digest
    ) == result

    tampered = deepcopy(result.to_data())
    first_id = next(iter(tampered["support_png_base64_by_panel_id"]))
    tampered["support_png_base64_by_panel_id"][first_id] = base64.b64encode(
        _png(900)
    ).decode("ascii")
    tampered.pop("record_digest")
    tampered["record_digest"] = canonical_digest(tampered)
    forged = PanelSoftEngineeringProposerTerminal.from_data(tampered)
    with pytest.raises(PanelSoftProposerError):
        cold_replay_panel_soft_engineering_task(
            forged, expected_record_digest=forged.record_digest
        )
