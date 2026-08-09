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
    return PanelSoftEngineeringPredicatePair.create(space)


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
    assert archive.freeze.to_data()["support_only_codex_ranker_present"] is False
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
        )
    with pytest.raises(PanelSoftEngineeringTaskRunnerError):
        PanelSoftEngineeringTaskFreeze.seal(
            task_plan=task,
            execution_precommit_digest=PRECOMMIT,
            proposer_artifact=proposer,
            support_artifacts=unrelated_artifacts,
            predicate_pair=pair,
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
        freeze_committer=forbidden,
        freeze_reloader=forbidden,
        query_source=forbidden,
    )
    assert isinstance(result, PanelSoftEngineeringProposerTerminal)
    assert result.to_data()["terminal_stage"] == "proposer"
    assert result.to_data()["support_observer_artifact_count"] == 0
    assert result.to_data()["error_count"] == 2
    assert result.to_data()["query_pixels_released"] is False
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
