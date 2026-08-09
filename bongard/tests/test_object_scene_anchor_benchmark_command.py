from __future__ import annotations

from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from typing import Sequence

import pytest

from bongard.canonical import canonical_digest
from bongard.object_bongard_release_gate import ObjectBongardReleaseStore
import bongard.object_scene_anchor_benchmark_command as benchmark


def _address(value: object) -> str:
    return "sha256:" + canonical_digest(value)


def _journal(tmp_path: Path) -> tuple[
    ObjectBongardReleaseStore,
    benchmark.ObjectSceneAnchorBenchmarkCallJournal,
]:
    store = ObjectBongardReleaseStore(tmp_path.resolve())
    journal = benchmark.ObjectSceneAnchorBenchmarkCallJournal(
        store=store,
        release_authorization_digest=_address({"synthetic": "authorization"}),
    )
    return store, journal


def test_call_journal_claims_before_call_and_reuses_success(tmp_path: Path) -> None:
    store, journal = _journal(tmp_path)
    task_digest = _address({"synthetic": "task"})
    context_digest = _address({"synthetic": "context"})
    artifact = {"schema": "test.synthetic-call-artifact.v1", "value": "ok"}
    calls: list[str] = []

    def invoke_and_persist():
        claim_paths = tuple(
            (store.root / "objects" / "anchor-call-claim").glob("*.json")
        )
        assert len(claim_paths) == 1
        calls.append("called")
        receipt = store.persist(
            object_kind="synthetic-call-artifact",
            object_digest=_address(artifact),
            data=artifact,
        )
        return artifact, receipt

    def load_artifact(receipt):
        return store.verify(receipt, expected_data=artifact)

    first, first_claim, first_terminal, first_reused = journal.run(
        task_plan_digest=task_digest,
        stage="proposer",
        context_digest=context_digest,
        expected_physical_call_count=1,
        object_kind="synthetic-call-artifact",
        invoke_and_persist=invoke_and_persist,
        load_artifact=load_artifact,
    )
    second, second_claim, second_terminal, second_reused = journal.run(
        task_plan_digest=task_digest,
        stage="proposer",
        context_digest=context_digest,
        expected_physical_call_count=1,
        object_kind="synthetic-call-artifact",
        invoke_and_persist=invoke_and_persist,
        load_artifact=load_artifact,
    )

    assert first == second == artifact
    assert first_claim == second_claim
    assert first_terminal == second_terminal
    assert first_terminal.status == "success"
    assert first_terminal.physical_call_slots_consumed == 1
    assert first_reused is False
    assert second_reused is True
    assert calls == ["called"]


def test_call_journal_rejects_dangling_changed_context_without_call(
    tmp_path: Path,
) -> None:
    store, journal = _journal(tmp_path)
    task_digest = _address({"synthetic": "task"})
    original_claim = benchmark.ObjectSceneAnchorBenchmarkCallClaim.create(
        release_authorization_digest=journal.release_authorization_digest,
        task_plan_digest=task_digest,
        stage="ranker",
        context_digest=_address({"synthetic": "original-context"}),
        expected_physical_call_count=1,
    )
    store.persist(
        object_kind="anchor-call-claim",
        object_digest=original_claim.record_digest,
        data=original_claim.to_data(),
    )
    calls: list[str] = []

    def forbidden_call():
        calls.append("called")
        raise AssertionError("a dangling claim must never authorize another call")

    with pytest.raises(
        benchmark.ObjectSceneAnchorBenchmarkDanglingClaim,
        match="retry is forbidden",
    ) as caught:
        journal.run(
            task_plan_digest=task_digest,
            stage="ranker",
            context_digest=_address({"synthetic": "changed-context"}),
            expected_physical_call_count=1,
            object_kind="synthetic-call-artifact",
            invoke_and_persist=forbidden_call,
            load_artifact=lambda _receipt: pytest.fail("no artifact may be loaded"),
        )

    assert caught.value.claim == original_claim
    assert calls == []


def _prepared(
    tmp_path: Path,
    task_ids: Sequence[str],
) -> tuple[SimpleNamespace, tuple[SimpleNamespace, ...]]:
    store = ObjectBongardReleaseStore(tmp_path.resolve())
    tasks = tuple(
        SimpleNamespace(
            task_id=task_id,
            record_digest=_address({"synthetic_task_index": index}),
            side_0_support_panel_ids=(),
            side_1_support_panel_ids=(),
            side_0_query_panel_id=f"synthetic-query-side0-{index}",
            side_1_query_panel_id=f"synthetic-query-side1-{index}",
        )
        for index, task_id in enumerate(task_ids)
    )
    bootstrap = {"schema": "test.synthetic-bootstrap.v1"}
    bootstrap_receipt = store.persist(
        object_kind="synthetic-bootstrap",
        object_digest=_address(bootstrap),
        data=bootstrap,
    )
    plan = SimpleNamespace(
        tasks=tasks,
        record_digest=_address({"synthetic": "batch-plan"}),
    )
    release = SimpleNamespace(
        store=store,
        authorization=SimpleNamespace(
            record_digest=_address({"synthetic": "authorization"})
        ),
        successor=SimpleNamespace(digest=_address({"synthetic": "successor"})),
    )
    prepared = SimpleNamespace(
        plan=plan,
        precommit=SimpleNamespace(
            record_digest=_address({"synthetic": "precommit"})
        ),
        release=release,
        predecessor=SimpleNamespace(digest=_address({"synthetic": "predecessor"})),
        runtime_record={"runtime_digest": _address({"synthetic": "runtime"})},
        bootstrap_receipt=bootstrap_receipt,
    )
    return prepared, tasks


def _finish_task(
    prepared: SimpleNamespace,
    task: SimpleNamespace,
    *,
    status: str,
    correct: int,
    determinate: int,
    abstain: int,
    errors: int,
):
    state = benchmark._TaskState(task)
    if status in ("success", "query_error"):
        state.query_release_count = 2
        state.formula_custody_verified = True
    return benchmark._finish_task(
        prepared,
        state,
        status=status,
        terminal_stage="score" if state.query_release_count else "version_space",
        correct_count=correct,
        determinate_count=determinate,
        abstain_count=abstain,
        error_count=errors,
        diagnostic={"synthetic_terminal_kind": status},
    )


def test_campaign_keeps_clean_success_and_typed_gap_in_denominator(
    tmp_path: Path,
) -> None:
    prepared, tasks = _prepared(
        tmp_path, ("synthetic-success", "synthetic-language-gap")
    )
    rows = (
        _finish_task(
            prepared, tasks[0], status="success", correct=2, determinate=2,
            abstain=0, errors=0,
        ),
        _finish_task(
            prepared, tasks[1], status="language_gap", correct=0,
            determinate=0, abstain=2, errors=0,
        ),
    )

    campaign = benchmark._campaign_from_tasks(prepared, rows)

    assert campaign["status"] == "completed_with_gaps"
    assert campaign["task_statuses"] == ["success", "language_gap"]
    assert campaign["query_denominator"] == 4
    assert campaign["correct_count"] == 2
    assert campaign["determinate_count"] == 2
    assert campaign["abstain_count"] == 2
    assert campaign["error_count"] == 0
    assert campaign["accuracy_ppm"] == 500_000
    assert campaign["coverage_ppm"] == 500_000
    assert campaign["all_terminal_outcomes_remain_in_denominator"] is True


def test_error_prediction_counter_prevents_campaign_success(tmp_path: Path) -> None:
    prepared, tasks = _prepared(tmp_path, ("synthetic-query-error",))
    rows = (
        _finish_task(
            prepared, tasks[0], status="query_error", correct=1,
            determinate=1, abstain=0, errors=1,
        ),
    )

    campaign = benchmark._campaign_from_tasks(prepared, rows)

    assert campaign["status"] == "completed_with_errors"
    assert campaign["status"] != "success"
    assert campaign["query_denominator"] == 2
    assert campaign["correct_count"] == 1
    assert campaign["determinate_count"] == 1
    assert campaign["error_count"] == 1
    assert campaign["accuracy_ppm"] == 500_000
    assert campaign["coverage_ppm"] == 500_000


def test_cold_replay_reloads_task_rows_from_disk_without_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared, tasks = _prepared(
        tmp_path, ("synthetic-pipeline-error-0", "synthetic-pipeline-error-1")
    )
    rows = (
        _finish_task(
            prepared, tasks[0], status="pipeline_error", correct=0,
            determinate=0, abstain=0, errors=2,
        ),
        _finish_task(
            prepared, tasks[1], status="pipeline_error", correct=0,
            determinate=0, abstain=0, errors=2,
        ),
    )
    campaign = benchmark._campaign_from_tasks(prepared, rows)

    def forbidden_execution(*_args, **_kwargs):
        raise AssertionError("cold replay must not execute a task or transport")

    monkeypatch.setattr(
        benchmark,
        "run_object_scene_anchor_benchmark_task",
        forbidden_execution,
    )
    replay = benchmark.cold_replay_object_scene_anchor_benchmark(
        prepared, campaign
    )

    assert replay["campaign_result_digest"] == campaign["campaign_result_digest"]
    assert replay["task_result_digests"] == campaign["task_result_digests"]
    assert replay["model_calls"] == 0
    assert replay["model_free"] is True
    assert replay["tamper_detecting"] is True
    assert replay["completed"] is True


def test_module_cli_help_is_side_effect_free() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "bongard.object_scene_anchor_benchmark_command",
            "--help",
        ],
        cwd=Path(__file__).resolve().parents[2],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "exact-unused TRAIN anchor predicate drill" in completed.stdout
    assert "--replay-only" in completed.stdout
    assert "--expected-campaign-digest" in completed.stdout
    assert completed.stderr == ""


def test_replay_only_requires_external_campaign_digest(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as caught:
        benchmark.main(["--output-root", str(tmp_path), "--replay-only"])

    assert caught.value.code == 2
    assert "--replay-only requires --expected-campaign-digest" in capsys.readouterr().err
