import hashlib
import json
from pathlib import Path

import pytest

import codex_campaign_status as S


def _binding(game: str, reached: int, *, action_count: int = 0):
    checkpoint = S.ZERO_SHA256 if reached == 0 else "a" * 64
    source = S.ZERO_SHA256 if reached == 0 else "b" * 64
    return S.validate_frontier_binding({
        "frontier_binding_schema": S.FRONTIER_BINDING_SCHEMA,
        "game": game,
        "reached": reached,
        "target_level": reached + 1,
        "parent_action_count": 0 if reached == 0 else action_count,
        "parent_checkpoint_sha256": checkpoint,
        "parent_source_tree_sha256": source,
        "frontier_sha256": S._sha256_json({
            "game": game,
            "reached": reached,
            "parent_checkpoint_sha256": checkpoint,
        }),
    })


def _frontier(game: str, reached: int, **extra):
    return {
        **_binding(game, reached, action_count=reached),
        "game": game,
        "current_level": reached,
        "next_level": reached + 1,
        **extra,
    }


def _turn_on(binding, **extra):
    return {
        **{
            field: binding[field]
            for field in (
                *S.FRONTIER_BINDING_FIELDS,
                "game",
                "reached",
                "target_level",
                "parent_action_count",
            )
        },
        **extra,
    }


def _binding_correction(
    binding,
    *,
    receipt_root: Path,
    thread_id="legacy",
    transcript="legacy.jsonl",
    **evidence_overrides,
):
    evidence = {
        "workspace_baseline_commit": "1" * 40,
        "baseline_checkpoint_sha256":
            binding["parent_checkpoint_sha256"],
        "baseline_source_tree_sha256":
            binding["parent_source_tree_sha256"],
        "protected_transcript_sha256": "2" * 64,
        "audit_receipt_relpath": "",
        "audit_receipt_sha256": "0" * 64,
        "baseline_checkpoint_replay_verified": True,
        "workspace_git_history_unmodified": True,
        "terminal_turn_audited": True,
        "taint_scan_passed": True,
    }
    evidence.update(evidence_overrides)
    record = {
        **{
            field: binding[field]
            for field in (
                *S.FRONTIER_BINDING_FIELDS,
                "game",
                "reached",
                "target_level",
                "parent_action_count",
            )
        },
        "event": "codex_frontier_binding_correction",
        "binding_correction_schema":
            S.FRONTIER_BINDING_CORRECTION_SCHEMA,
        "binding_authority":
            S.FRONTIER_BINDING_CORRECTION_AUTHORITY,
        "recorded_at": "2026-07-29T12:00:00+00:00",
        "thread_id": thread_id,
        "transcript": transcript,
        "evidence": evidence,
    }
    payload = S._binding_correction_receipt_payload(record)
    raw = (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        + b"\n"
    )
    digest = hashlib.sha256(raw).hexdigest()
    record["evidence"]["audit_receipt_sha256"] = digest
    record["evidence"]["audit_receipt_relpath"] = f"{digest}.json"
    receipt_root.mkdir(parents=True, exist_ok=True)
    (receipt_root / f"{digest}.json").write_bytes(raw)
    return record


def test_joined_turn_reports_solve_and_displayed_allowance_delta(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "HERE", tmp_path)
    ws = tmp_path / "runs" / "scratch" / "ws"
    ws.mkdir(parents=True)
    (ws / "turn.jsonl").write_text(
        json.dumps({
            "type": "item.completed",
            "item": {"type": "command_execution"},
        }) + "\n" +
        json.dumps({
            "type": "item.completed",
            "item": {"type": "file_change"},
        }) + "\n"
    )
    records = [
        {
            "event": "codex_exec", "thread_id": "t", "workspace": "ws",
            "transcript": "turn.jsonl", "reasoning_effort": "medium",
            "weekly_remaining_before": 86, "weekly_remaining_after": 83,
            "observed_tokens": 511_000,
        },
        {
            "event": "codex_level_outcome", "thread_id": "t",
            "game": "tr87", "target_level": 5, "solved_target": True,
            "winning_marginal_C": 249, "taint_verdict": "clean",
        },
    ]
    turn = S.joined_turns(records)[0]
    assert turn["displayed_weekly_points_used"] == 3
    assert turn["solved_target"] is True
    assert turn["winning_marginal_C"] == 249
    assert turn["command_executions"] == 1
    assert turn["file_changes"] == 1


def test_infrastructure_turn_is_visible_but_not_charged_as_solver_attempt():
    frontier = _frontier(
        "lf52", 6,
        incumbent_kind="promoted",
        frontier_scaffold_created_at=None,
        priority_score=1.0,
    )
    records = [{
        **{
            field: frontier[field]
            for field in (
                *S.FRONTIER_BINDING_FIELDS,
                "game",
                "reached",
                "target_level",
                "parent_action_count",
            )
        },
        "event": "codex_exec",
        "thread_id": "capacity",
        "run_label": "lf52:L7:propose",
        "game": "lf52",
        "target_level": 7,
        "reasoning_effort": "max",
        "failure_class": "infrastructure",
        "terminal_errors": ["Selected model is at capacity."],
        "duration_seconds": 10,
    }]
    turns = S.joined_turns(records)
    assert turns[0]["game"] == "lf52"
    assert turns[0]["target_level"] == 7
    assert turns[0]["failure_class"] == "infrastructure"
    summary = S.effort_efficiency(turns)["max"]
    assert summary["proposal_attempts"] == 0
    assert summary["infrastructure_turns_excluded"] == 1
    ranked = S.ranked_frontiers([frontier], turns)
    assert ranked[0]["paid_attempts_at_frontier"] == 0
    assert ranked[0]["infrastructure_turns_at_frontier"] == 1
    assert ranked[0]["failed_attempts_at_frontier"] == 0


def test_sandbox_isolation_receipt_is_infrastructure_noncounting():
    record = {
        "event": "codex_sandbox_isolated_generation_abandoned",
        "schema": "scheduler_sandbox_isolated_generation_abandoned_v1",
        "failure_class": "infrastructure",
        "retry_increment": 0,
        "codex_exec_appended": False,
        "process_tree_quiesced": False,
        "detached_processes_proven_absent": False,
    }
    assert S.infrastructure_noncounting_events([record]) == [record]


def test_append_only_failure_classification_correction_updates_old_turn():
    records = [
        {
            "event": "codex_exec",
            "thread_id": "old",
            "transcript": "old.jsonl",
            "run_label": "ls20:L7:propose",
            "reasoning_effort": "max",
        },
        {
            "event": "codex_exec_classification_correction",
            "thread_id": "old",
            "transcript": "old.jsonl",
            "failure_class": "infrastructure",
            "failure_detail_class": "known_transient",
            "terminal_errors": ["Selected model is at capacity."],
            "game": "ls20",
            "target_level": 7,
        },
    ]
    turn = S.joined_turns(records)[0]
    assert turn["failure_class"] == "infrastructure"
    assert turn["failure_detail_class"] == "known_transient"
    assert turn["terminal_errors"] == ["Selected model is at capacity."]
    assert turn["game"] == "ls20"
    assert turn["target_level"] == 7
    assert turn["clean_no_progress"] is False
    assert turn["retry_increment"] == 0


def test_taint_correction_remains_noncounting_after_generation_cleanup(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(S, "HERE", tmp_path)
    frontier = _frontier(
        "lf52", 8, incumbent_kind="promoted", priority_score=1.0
    )
    records = [
        {
            **{
                field: frontier[field]
                for field in (
                    *S.FRONTIER_BINDING_FIELDS,
                    "reached",
                    "parent_action_count",
                )
            },
            "event": "codex_exec",
            "thread_id": "tainted",
            "workspace": "deleted-workspace",
            "transcript": "deleted-transcript.jsonl",
            "game": "lf52",
            "target_level": 9,
            "reasoning_effort": "max",
            "failure_class": None,
            "observed_tokens": 12345,
        },
        {
            "event": "codex_exec_classification_correction",
            "thread_id": "tainted",
            "transcript": "deleted-transcript.jsonl",
            "game": "lf52",
            "target_level": 9,
            "failure_class": "taint",
            "failure_detail_class": "host_process_introspection",
            "solved_target": None,
            "taint_verdict": "tainted",
            "retry_increment": 0,
        },
    ]

    turn = S.joined_turns(records)[0]
    assert turn["transcript"] == "deleted-transcript.jsonl"
    assert turn["failure_class"] == "taint"
    assert turn["taint_verdict"] == "tainted"
    assert turn["clean_no_progress"] is False
    assert turn["retry_increment"] == 0
    ranked = S.ranked_frontiers([frontier], [turn])[0]
    assert ranked["retry_complexity_n"] == 0
    assert ranked["failed_attempts_at_frontier"] == 0
    assert ranked["non_solver_turns_at_frontier"] == 1
    assert S.Guard.local_window_totals([records[0]])["observed_tokens"] == 12345


def test_hard_timeout_correction_nulls_false_no_progress(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "HERE", tmp_path)
    ws = tmp_path / "runs" / "scratch" / "timeout-ws"
    ws.mkdir(parents=True)
    (ws / "timeout.jsonl").write_text(
        json.dumps({"type": "thread.started", "thread_id": "timeout"}) + "\n"
    )
    records = [
        {
            "event": "codex_exec",
            "thread_id": "timeout",
            "workspace": "timeout-ws",
            "transcript": "timeout.jsonl",
            "run_label": "lf52:L9:propose",
            "game": "lf52",
            "target_level": 9,
            "reasoning_effort": "max",
            "timed_out": True,
        },
        {
            "event": "codex_level_outcome",
            "thread_id": "timeout",
            "game": "lf52",
            "target_level": 9,
            "solved_target": False,
            "taint_verdict": "clean",
        },
        {
            "event": "codex_exec_classification_correction",
            "thread_id": "timeout",
            "transcript": "timeout.jsonl",
            "failure_class": "containment",
            "failure_detail_class": "hard_wall_time",
            "solved_target": None,
            "retry_increment": 0,
            "taint_verdict": "clean",
        },
    ]
    turn = S.joined_turns(records)[0]
    assert turn["failure_class"] == "containment"
    assert turn["solved_target"] is None
    assert turn["transcript_complete"] is False
    assert turn["clean_no_progress"] is False
    assert turn["retry_increment"] == 0


def test_completed_clean_failure_is_one_retry(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "HERE", tmp_path)
    ws = tmp_path / "runs" / "scratch" / "clean-ws"
    ws.mkdir(parents=True)
    (ws / "clean.jsonl").write_text(
        json.dumps({"type": "thread.started", "thread_id": "clean"}) + "\n"
        + json.dumps({"type": "turn.completed", "usage": {}}) + "\n"
    )
    records = [
        {
            "event": "codex_exec",
            "thread_id": "clean",
            "workspace": "clean-ws",
            "transcript": "clean.jsonl",
            "run_label": "bp35:L7:propose",
            "game": "bp35",
            "target_level": 7,
            "reasoning_effort": "max",
            "timed_out": False,
            "interrupted": False,
            "failure_class": None,
        },
        {
            "event": "codex_level_outcome",
            "thread_id": "clean",
            "game": "bp35",
            "target_level": 7,
            "solved_target": False,
            "taint_verdict": "clean",
        },
    ]
    turn = S.joined_turns(records)[0]
    assert turn["transcript_complete"] is True
    assert turn["clean_no_progress"] is True
    assert turn["retry_increment"] == 1


def test_completed_clean_failure_uses_protected_transcript_after_cleanup(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(S, "HERE", tmp_path)
    protected = (
        tmp_path
        / "runs"
        / "scratch"
        / ".proposer_transcripts"
        / "retired-ws"
    )
    protected.mkdir(parents=True)
    (protected / "clean.jsonl").write_text(
        json.dumps({"type": "thread.started", "thread_id": "clean"})
        + "\n"
        + json.dumps({"type": "turn.completed", "usage": {}})
        + "\n"
    )
    records = [
        {
            "event": "codex_exec",
            "thread_id": "clean",
            "workspace": "retired-ws",
            "transcript": "clean.jsonl",
            "run_label": "lf52:L9:propose",
            "game": "lf52",
            "target_level": 9,
            "reasoning_effort": "medium",
            "timed_out": False,
            "interrupted": False,
            "failure_class": None,
        },
        {
            "event": "codex_level_outcome",
            "thread_id": "clean",
            "game": "lf52",
            "target_level": 9,
            "solved_target": False,
            "taint_verdict": "clean",
        },
    ]
    turn = S.joined_turns(records)[0]
    assert turn["transcript_complete"] is True
    assert turn["clean_no_progress"] is True
    assert turn["retry_increment"] == 1


def test_audited_legacy_binding_correction_can_authorize_one_retry(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(S, "HERE", tmp_path)
    workspace = tmp_path / "runs" / "scratch" / "legacy-ws"
    workspace.mkdir(parents=True)
    (workspace / "legacy.jsonl").write_text(
        json.dumps({
            "type": "thread.started",
            "thread_id": "legacy",
        }) + "\n"
        + json.dumps({"type": "turn.completed", "usage": {}}) + "\n"
    )
    frontier = _frontier(
        "same", 4, incumbent_kind="promoted", priority_score=4.0
    )
    records = [
        {
            "event": "codex_exec",
            "thread_id": "legacy",
            "workspace": "legacy-ws",
            "transcript": "legacy.jsonl",
            "run_label": "same:L5:propose",
            "game": "same",
            "target_level": 5,
            "reasoning_effort": "max",
        },
        {
            "event": "codex_level_outcome",
            "thread_id": "legacy",
            "game": "same",
            "target_level": 5,
            "solved_target": False,
            "taint_verdict": "clean",
        },
        _binding_correction(
            frontier,
            receipt_root=tmp_path / "frontier_binding_receipts",
        ),
    ]
    turn = S.joined_turns(records)[0]
    assert turn["frontier_binding_authority"] == (
        "retrospective_receipt_backed_claim"
    )
    assert turn["clean_no_progress"] is True
    row = S.ranked_frontiers([frontier], [turn])[0]
    assert row["retry_complexity_n"] == 1
    assert row["exact_bound_turns_at_frontier"] == 1


def test_legacy_binding_correction_fails_closed_without_every_audit_gate(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(S, "HERE", tmp_path)
    workspace = tmp_path / "runs" / "scratch" / "legacy-ws"
    workspace.mkdir(parents=True)
    (workspace / "legacy.jsonl").write_text(
        json.dumps({"type": "turn.completed", "usage": {}}) + "\n"
    )
    frontier = _frontier(
        "same", 4, incumbent_kind="promoted", priority_score=4.0
    )
    records = [
        {
            "event": "codex_exec",
            "thread_id": "legacy",
            "workspace": "legacy-ws",
            "transcript": "legacy.jsonl",
            "run_label": "same:L5:propose",
            "game": "same",
            "target_level": 5,
            "reasoning_effort": "max",
        },
        {
            "event": "codex_level_outcome",
            "thread_id": "legacy",
            "game": "same",
            "target_level": 5,
            "solved_target": False,
            "taint_verdict": "clean",
        },
        _binding_correction(
            frontier,
            receipt_root=tmp_path / "frontier_binding_receipts",
            workspace_git_history_unmodified=False,
        ),
    ]
    turn = S.joined_turns(records)[0]
    assert turn["frontier_binding_authority"] == (
        "unbound_conflicting_or_malformed"
    )
    row = S.ranked_frontiers([frontier], [turn])[0]
    assert row["retry_complexity_n"] == 0
    assert row["unbound_legacy_turns_for_game_level"] == 1


def test_legacy_binding_assertions_do_not_count_without_reopened_receipt(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(S, "HERE", tmp_path)
    frontier = _frontier(
        "same", 4, incumbent_kind="promoted", priority_score=4.0
    )
    correction = _binding_correction(
        frontier,
        receipt_root=tmp_path / "frontier_binding_receipts",
    )
    receipt = (
        tmp_path
        / "frontier_binding_receipts"
        / correction["evidence"]["audit_receipt_relpath"]
    )
    receipt.unlink()
    records = [
        {
            "event": "codex_exec",
            "thread_id": "legacy",
            "transcript": "legacy.jsonl",
            "run_label": "same:L5:propose",
            "game": "same",
            "target_level": 5,
            "reasoning_effort": "max",
        },
        {
            "event": "codex_level_outcome",
            "thread_id": "legacy",
            "game": "same",
            "target_level": 5,
            "solved_target": False,
            "taint_verdict": "clean",
        },
        correction,
    ]
    turn = S.joined_turns(records)[0]
    assert turn["frontier_binding_authority"] == (
        "unbound_conflicting_or_malformed"
    )
    row = S.ranked_frontiers([frontier], [turn])[0]
    assert row["retry_complexity_n"] == 0


def test_binding_receipt_root_symlink_cannot_escape_dedicated_directory(
    tmp_path,
):
    frontier = _frontier(
        "same", 4, incumbent_kind="promoted", priority_score=4.0
    )
    outside = tmp_path / "outside"
    correction = _binding_correction(
        frontier,
        receipt_root=outside,
    )
    linked_root = tmp_path / "frontier_binding_receipts"
    linked_root.symlink_to(outside, target_is_directory=True)
    exec_record = {
        "event": "codex_exec",
        "thread_id": "legacy",
        "transcript": "legacy.jsonl",
        "game": "same",
        "target_level": 5,
    }
    try:
        S.validate_frontier_binding_correction(
            correction,
            exec_record=exec_record,
            receipt_root=linked_root,
        )
    except ValueError as exc:
        assert "reopened safely" in str(exc)
    else:
        raise AssertionError("symlinked receipt root was accepted")


def test_binding_receipt_reader_rejects_directory_rename_race(
    tmp_path, monkeypatch
):
    root = tmp_path / "receipts"
    root.mkdir()
    (root / "receipt.json").write_bytes(b"x")
    moved = tmp_path / "receipts-moved"
    real_read = S.os.read
    raced = False

    def racing_read(fd, size):
        nonlocal raced
        if not raced:
            raced = True
            root.rename(moved)
            root.mkdir()
        return real_read(fd, size)

    monkeypatch.setattr(S.os, "read", racing_read)
    try:
        S._read_receipt_regular(root, "receipt.json")
    except ValueError as exc:
        assert "root changed" in str(exc)
    else:
        raise AssertionError("receipt-root rename race was accepted")


def test_classification_correction_cannot_silently_bind_legacy_turn():
    frontier = _frontier(
        "same", 4, incumbent_kind="promoted", priority_score=4.0
    )
    records = [
        {
            "event": "codex_exec",
            "thread_id": "legacy",
            "transcript": "legacy.jsonl",
            "run_label": "same:L5:propose",
            "game": "same",
            "target_level": 5,
            "reasoning_effort": "max",
        },
        {
            **{
                field: frontier[field]
                for field in (
                    *S.FRONTIER_BINDING_FIELDS,
                    "reached",
                    "parent_action_count",
                )
            },
            "event": "codex_exec_classification_correction",
            "thread_id": "legacy",
            "transcript": "legacy.jsonl",
            "game": "same",
            "target_level": 5,
            "solved_target": False,
            "taint_verdict": "clean",
        },
    ]
    turn = S.joined_turns(records)[0]
    assert turn["frontier_binding_authority"] == "unbound_legacy"
    row = S.ranked_frontiers([frontier], [turn])[0]
    assert row["retry_complexity_n"] == 0


def test_live_prebinding_turn_stays_unbound_and_noncounting(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(S, "HERE", tmp_path)
    workspace = tmp_path / "runs" / "scratch" / "active-ws"
    workspace.mkdir(parents=True)
    (workspace / "active.jsonl").write_text(
        json.dumps({
            "type": "thread.started",
            "thread_id": "active",
        }) + "\n"
    )
    frontier = _frontier(
        "same", 4, incumbent_kind="promoted", priority_score=4.0
    )
    turn = S.joined_turns([{
        "event": "codex_exec",
        "thread_id": "active",
        "workspace": "active-ws",
        "transcript": "active.jsonl",
        "run_label": "same:L5:propose",
        "game": "same",
        "target_level": 5,
        "reasoning_effort": "max",
    }])[0]
    assert turn["frontier_binding_authority"] == "unbound_legacy"
    assert turn["transcript_complete"] is False
    assert turn["clean_no_progress"] is False
    row = S.ranked_frontiers([frontier], [turn])[0]
    assert row["retry_complexity_n"] == 0
    assert row["unbound_legacy_turns_for_game_level"] == 1


def test_cached_live_snapshot_supersedes_older_postflight():
    records = [
        {"event": "codex_exec", "weekly_remaining_after": 32,
         "weekly_resets_at": 100},
        {"event": "rate_limit_snapshot", "allowance": {
            "remaining_percent": 22, "resets_at": 101,
        }},
    ]
    allowance = S._allowance_from_records(records, S.joined_turns(records))
    assert allowance["remaining_percent"] == 22
    assert allowance["source"] == "cached_live_rate_limit_read"


def test_legacy_postflight_does_not_erase_cached_unlimited_window():
    records = [
        {"event": "rate_limit_snapshot", "allowance": {
            "remaining_percent": 100, "resets_at": None,
            "window_name": "unlimited", "limit_id": "codex",
        }},
        {"event": "codex_exec", "weekly_remaining_after": 100,
         "weekly_resets_at": None},
    ]
    allowance = S._allowance_from_records(records, S.joined_turns(records))
    assert allowance["window_name"] == "unlimited"
    assert allowance["source"] == "cached_live_rate_limit_read"


def test_explicit_postflight_supersedes_older_cached_window():
    records = [
        {"event": "rate_limit_snapshot", "allowance": {
            "remaining_percent": 100, "resets_at": None,
            "window_name": "unlimited", "limit_id": "codex",
        }},
        {"event": "codex_exec", "weekly_remaining_after": 76,
         "weekly_resets_at": 123, "weekly_window_after": "primary",
         "weekly_limit_id_after": "codex"},
    ]
    allowance = S._allowance_from_records(records, S.joined_turns(records))
    assert allowance["remaining_percent"] == 76
    assert allowance["window_name"] == "primary"
    assert allowance["source"] == "explicit_postflight"


def test_joined_window_turns_excludes_prior_window_outcomes():
    records = [
        {"event": "codex_exec", "thread_id": "old", "run_label": "x:L1:propose"},
        {"event": "codex_level_outcome", "thread_id": "old", "solved_target": True},
        {"event": "codex_exec", "thread_id": "new", "run_label": "y:L1:propose"},
        {"event": "codex_level_outcome", "thread_id": "new", "solved_target": False},
    ]
    turns = S._joined_window_turns(records, [records[2]])
    assert len(turns) == 1
    assert turns[0]["thread_id"] == "new"
    assert turns[0]["solved_target"] is False


def test_readiness_requires_headroom_and_local_budget():
    totals = {"runs": 1, "observed_tokens": 500_000}
    ready = S._readiness(86, 80, 4, 6, totals, 12, 2_000_000)
    assert ready["medium_admissible"] is True
    assert ready["high_admissible"] is True

    tight = S._readiness(83, 80, 4, 6, totals, 12, 2_000_000)
    assert tight["medium_admissible"] is False
    assert tight["high_admissible"] is False

    capped = S._readiness(90, 80, 4, 6, totals, 1, 2_000_000)
    assert capped["local_budget_ok"] is False
    assert capped["medium_admissible"] is False


def test_frontier_rows_rank_mature_completion_over_cold_start(tmp_path):
    artifacts = tmp_path / "artifacts"
    audits = tmp_path / "audits"
    environments = tmp_path / "environments"
    artifacts.mkdir(); audits.mkdir(); environments.mkdir()
    for game, reached, source in (
        ("mature", 5, "def leg(env):\n    pass\n"),
        ("cold", 1, "def leg(env):\n    pass\n"),
    ):
        art = artifacts / f"{game}_legs"
        art.mkdir()
        (art / "checkpoint.json").write_text(json.dumps({
            "game": game,
            "reached": reached,
            "final_path": [0] * reached,
            "validated": True,
        }))
        (art / "legs.py").write_text(source)
        (art / "players.py").write_text("")
        (art / "solve.py").write_text("")
        metadata = environments / game / "version" / "metadata.json"
        metadata.parent.mkdir(parents=True)
        metadata.write_text(json.dumps({"baseline_actions": [0] * 6}))
    rows = [
        {"game": "mature", "completed_levels": 6},
        {"game": "cold", "completed_levels": 6},
    ]
    for name in (
        "baseline1_gpt55_xhigh_solved_checkpoints.json",
        "retrodict-solved-checkpoint-memory.json",
    ):
        (audits / name).write_text(json.dumps({"rows": rows}))
    result = S.frontier_rows(artifacts, audits, environments)
    assert [row["game"] for row in result] == ["mature", "cold"]
    assert result[0]["next_level"] == 6


def test_frontier_rows_include_games_without_local_artifacts(tmp_path):
    artifacts = tmp_path / "artifacts"
    audits = tmp_path / "audits"
    environments = tmp_path / "environments"
    artifacts.mkdir(); audits.mkdir(); environments.mkdir()
    metadata = environments / "cold" / "version" / "metadata.json"
    metadata.parent.mkdir(parents=True)
    metadata.write_text(json.dumps({"baseline_actions": [0] * 9}))
    (audits / "baseline1_gpt55_xhigh_solved_checkpoints.json").write_text(
        json.dumps({"rows": [{"game": "cold", "completed_levels": 8}]})
    )
    (audits / "retrodict-solved-checkpoint-memory.json").write_text(
        json.dumps({"rows": [{"game": "cold", "completed_levels": 7}]})
    )
    row = S.frontier_rows(artifacts, audits, environments)[0]
    assert row["game"] == "cold"
    assert row["incumbent_kind"] == "cold_start"
    assert row["current_level"] == 0
    assert row["next_level"] == 1
    assert row["authoritative_level_count"] == 9
    assert row["external_evidence"] == {"baseline1": 8, "Retrodict": 7}


def test_frontier_rows_reject_checkpoint_beyond_authoritative_inventory(tmp_path):
    artifacts = tmp_path / "artifacts"
    audits = tmp_path / "audits"
    environments = tmp_path / "environments"
    artifacts.mkdir(); audits.mkdir(); environments.mkdir()
    art = artifacts / "x_legs"
    art.mkdir()
    (art / "checkpoint.json").write_text(
        json.dumps({"game": "x", "reached": 3})
    )
    metadata = environments / "x" / "version" / "metadata.json"
    metadata.parent.mkdir(parents=True)
    metadata.write_text(json.dumps({"baseline_actions": [0, 0]}))
    try:
        S.frontier_rows(artifacts, audits, environments)
    except ValueError as exc:
        assert "exceeds authoritative inventory" in str(exc)
    else:
        raise AssertionError("over-target checkpoint was accepted")


def _write_bound_wip(
    artifact: Path,
    *,
    game: str,
    reached: int,
    target_level: int,
    phase: str,
):
    binding = S.exact_frontier_binding(
        artifact, game=game, target_level=target_level
    )
    attempt = f"{phase}_abc123"
    attempt_dir = (
        artifact / "wip_context" / f"level_{target_level:02d}" / attempt
    )
    files_dir = attempt_dir / "files"
    files_dir.mkdir(parents=True)
    (files_dir / "probe.py").write_text("print('clean')\n")
    metadata = {
        "game": game,
        "level": target_level,
        "phase": phase,
        "reached": reached,
        "err": None,
        "attempt": attempt,
        "created_at": "2026-08-02T00:00:00+00:00",
        "taint_verdict": "clean",
        "filesystem_boundary_policy_schema": S.Boundary.POLICY_SCHEMA,
        "filesystem_boundary_policy_sha256": S.Boundary.policy_sha256(),
        "compatibility_arena_module_sha256": (
            S.Boundary.arena_module_sha256(S.HERE)
        ),
        "compatibility_boundary_authority": "behavioral_defense_in_depth",
        "frontier_binding": binding,
        "files": ["probe.py"],
    }
    (attempt_dir / "metadata.json").write_text(json.dumps(metadata))
    latest = attempt_dir.parent / "latest.json"
    latest.write_text(json.dumps({"attempt": attempt, "metadata": metadata}))
    return binding, attempt, metadata


def test_latest_wip_descriptor_recognizes_exact_infrastructure_capsule(tmp_path):
    artifact = tmp_path / "hard_legs"
    artifact.mkdir()
    (artifact / "checkpoint.json").write_text(json.dumps({
        "game": "hard",
        "reached": 2,
        "final_path": [1, 2],
        "validated": True,
    }))
    for name in ("legs.py", "players.py", "solve.py"):
        (artifact / name).write_text(f"# {name}\n")
    binding, attempt, _ = _write_bound_wip(
        artifact,
        game="hard",
        reached=2,
        target_level=3,
        phase="infrastructure_failure_transport",
    )
    result = S.latest_wip_descriptor(
        artifact,
        game="hard",
        reached=2,
        target_level=3,
        frontier_binding=binding,
    )
    assert result == {
        "warm_wip_available": True,
        "warm_wip_attempt": attempt,
        "warm_wip_phase": "infrastructure_failure_transport",
        "warm_wip_recovery_required": True,
        "warm_wip_validation": "exact_frontier_capsule",
    }


def test_latest_wip_descriptor_rejects_unsealed_or_stale_capsule(tmp_path):
    artifact = tmp_path / "hard_legs"
    artifact.mkdir()
    (artifact / "checkpoint.json").write_text(json.dumps({
        "game": "hard",
        "reached": 2,
        "final_path": [1, 2],
        "validated": True,
    }))
    for name in ("legs.py", "players.py", "solve.py"):
        (artifact / name).write_text(f"# {name}\n")
    binding, attempt, metadata = _write_bound_wip(
        artifact,
        game="hard",
        reached=2,
        target_level=3,
        phase="not_reached",
    )
    metadata["reached"] = 1
    latest = artifact / "wip_context" / "level_03" / "latest.json"
    latest.write_text(json.dumps({"attempt": attempt, "metadata": metadata}))
    result = S.latest_wip_descriptor(
        artifact,
        game="hard",
        reached=2,
        target_level=3,
        frontier_binding=binding,
    )
    assert result["warm_wip_available"] is False
    assert result["warm_wip_recovery_required"] is False
    assert result["warm_wip_validation"].startswith("rejected:")


def test_latest_wip_descriptor_rejects_legacy_unbound_capsule_forensic_only(
    tmp_path,
):
    artifact = tmp_path / "hard_legs"
    artifact.mkdir()
    (artifact / "checkpoint.json").write_text(json.dumps({
        "game": "hard",
        "reached": 2,
        "final_path": [1, 2],
        "validated": True,
    }))
    for name in ("legs.py", "players.py", "solve.py"):
        (artifact / name).write_text(f"# {name}\n")
    binding, _, metadata = _write_bound_wip(
        artifact,
        game="hard",
        reached=2,
        target_level=3,
        phase="not_reached",
    )
    metadata.pop("filesystem_boundary_policy_schema")
    metadata.pop("filesystem_boundary_policy_sha256")
    level = artifact / "wip_context" / "level_03"
    attempt = metadata["attempt"]
    (level / attempt / "metadata.json").write_text(json.dumps(metadata))
    (level / "latest.json").write_text(json.dumps({
        "attempt": attempt,
        "metadata": metadata,
    }))

    result = S.latest_wip_descriptor(
        artifact,
        game="hard",
        reached=2,
        target_level=3,
        frontier_binding=binding,
    )

    assert result["warm_wip_available"] is False
    assert result["warm_wip_recovery_required"] is False
    assert result["warm_wip_validation"] == (
        "rejected:filesystem_boundary_policy_binding"
    )


@pytest.mark.parametrize("malformed", [[], 7])
def test_latest_wip_descriptor_rejects_nonobject_metadata_without_crashing(
    tmp_path, malformed
):
    artifact = tmp_path / "hard_legs"
    artifact.mkdir()
    (artifact / "checkpoint.json").write_text(json.dumps({
        "game": "hard", "reached": 2, "final_path": [1, 2],
        "validated": True,
    }))
    for name in ("legs.py", "players.py", "solve.py"):
        (artifact / name).write_text(f"# {name}\n")
    binding, attempt, _ = _write_bound_wip(
        artifact,
        game="hard",
        reached=2,
        target_level=3,
        phase="not_reached",
    )
    level = artifact / "wip_context" / "level_03"
    (level / attempt / "metadata.json").write_text(json.dumps(malformed))
    (level / "latest.json").write_text(json.dumps({
        "attempt": attempt, "metadata": malformed,
    }))

    result = S.latest_wip_descriptor(
        artifact,
        game="hard",
        reached=2,
        target_level=3,
        frontier_binding=binding,
    )

    assert result["warm_wip_available"] is False
    assert result["warm_wip_validation"] == (
        "rejected:latest metadata is not an object"
    )


def test_effort_efficiency_charges_failures_to_cost_per_solve():
    turns = [
        {
            "run_label": "a:L1:propose", "reasoning_effort": "high",
            "solved_target": True, "displayed_weekly_points_used": 2,
            "duration_seconds": 300, "observed_tokens": 100,
            "timed_out": False,
        },
        {
            "run_label": "b:L1:propose", "reasoning_effort": "high",
            "solved_target": False, "displayed_weekly_points_used": 4,
            "duration_seconds": 600, "observed_tokens": 0,
            "timed_out": True,
        },
        {
            "run_label": "a:L1:debrief", "reasoning_effort": "medium",
            "displayed_weekly_points_used": 1, "duration_seconds": 60,
            "observed_tokens": 10, "timed_out": False,
        },
    ]
    high = S.effort_efficiency(turns)["high"]
    assert high["proposal_attempts"] == 1
    assert high["non_solver_turns_excluded"] == 1
    assert high["solved_levels"] == 1
    assert high["displayed_points_per_solved_level"] == 2.0
    assert high["success_only_points_per_solved_level"] == 2.0
    assert high["turns_with_missing_token_usage"] == 0
    assert "medium" not in S.effort_efficiency(turns)


def test_effort_efficiency_by_phase_does_not_pool_cold_and_continuation():
    turns = [
        {
            "run_label": "a:L1:propose", "reasoning_effort": "high",
            "target_level": 1, "solved_target": True,
            "displayed_weekly_points_used": 1, "duration_seconds": 60,
            "observed_tokens": 10,
        },
        {
            "run_label": "a:L2:propose", "reasoning_effort": "medium",
            "target_level": 2, "solved_target": False,
            "displayed_weekly_points_used": 2, "duration_seconds": 120,
            "observed_tokens": 20,
        },
    ]
    result = S.effort_efficiency_by_phase(turns)
    assert set(result["cold_L1"]) == {"high"}
    assert result["cold_L1"]["high"]["solved_levels"] == 1
    assert set(result["continuation_L2_plus"]) == {"medium"}
    assert result["continuation_L2_plus"]["medium"]["failed_levels"] == 1


def test_effort_solve_quality_uses_exact_gkm_checkpoints(tmp_path):
    audits = tmp_path / "audits"
    audits.mkdir()
    (audits / "marginal-literal-reuse.json").write_text(json.dumps({
        "rows": [
            {
                "system": "GKM", "game": "a", "completed_level": 2,
                "source_checkpoint_exact": True,
                "marginal_ast_zlib_bytes": 200,
                "hard_literal_reuse_witness": True,
                "sharp_marginal_drop": True,
                "sharp_drop_with_literal_reuse": True,
            },
            {
                "system": "OPINE", "game": "a", "completed_level": 2,
                "source_checkpoint_exact": True,
                "marginal_ast_zlib_bytes": 1,
                "hard_literal_reuse_witness": True,
            },
            {
                "system": "GKM", "game": "b", "completed_level": 1,
                "source_checkpoint_exact": False,
                "marginal_ast_zlib_bytes": 10,
                "hard_literal_reuse_witness": False,
            },
        ]
    }))
    turns = [
        {
            "run_label": "a:L2:propose", "reasoning_effort": "high",
            "solved_target": True, "game": "a", "target_level": 2,
            "winning_marginal_C": 40,
        },
        {
            "run_label": "b:L1:propose", "reasoning_effort": "high",
            "solved_target": True, "game": "b", "target_level": 1,
            "winning_marginal_C": 60,
        },
        {
            "run_label": "c:L1:propose", "reasoning_effort": "medium",
            "solved_target": False, "game": "c", "target_level": 1,
            "winning_marginal_C": None,
        },
    ]
    high = S.effort_solve_quality(turns, audits)["high"]
    assert high["solved_levels"] == 2
    assert high["exact_checkpoint_coverage"] == 1
    assert high["median_conditional_ast_zlib_bytes"] == 200.0
    assert high["median_pre_debrief_acquisition_charge"] == 50.0
    assert high["literal_reuse_wins"] == 1
    assert high["sharp_drop_with_literal_reuse_wins"] == 1
    assert "medium" not in S.effort_solve_quality(turns, audits)


def test_ranked_frontiers_use_medium_for_cold_screen_and_penalize_failure():
    frontiers = [
        _frontier(
            "cold", 0, incumbent_kind="cold_start",
            solver_source_bytes=0, priority_score=2.0,
        ),
        _frontier(
            "stalled", 4, incumbent_kind="promoted",
            solver_source_bytes=20_000, priority_score=2.1,
        ),
    ]
    turns = [_turn_on(frontiers[1], **{
        "game": "stalled", "target_level": 5, "solved_target": False,
        "reasoning_effort": "high",
    })]
    ranked = S.ranked_frontiers(frontiers, turns)
    assert ranked[0]["game"] == "cold"
    assert ranked[0]["recommended_effort"] == "medium"
    assert ranked[0]["recommended_minutes"] == 15
    assert ranked[1]["dispatch_mode"] == "continue_clean_wip"


def test_ranked_frontier_does_not_send_large_fresh_solver_straight_to_high():
    frontiers = [_frontier(
        "large", 4, incumbent_kind="promoted",
        solver_source_bytes=25_000, priority_score=4.0,
    )]
    row = S.ranked_frontiers(frontiers, [])
    assert row[0]["recommended_effort"] == "medium"
    assert row[0]["dispatch_mode"] == "fresh_frontier"


def test_new_scaffold_does_not_reset_exact_frontier_retry_coordinate():
    frontiers = [_frontier(
        "cold", 0, incumbent_kind="cold_start",
        solver_source_bytes=0, priority_score=2.0,
        frontier_scaffold_version="v2",
        frontier_scaffold_created_at="2026-07-24T10:00:00Z",
    )]
    binding = frontiers[0]
    turns = [
        _turn_on(binding, **{
            "game": "cold", "target_level": 1, "solved_target": False,
            "reasoning_effort": "medium",
            "started_at": "2026-07-23T10:00:00+00:00",
        }),
        _turn_on(binding, **{
            "game": "cold", "target_level": 1, "solved_target": False,
            "reasoning_effort": "high",
            "started_at": "2026-07-23T11:00:00Z",
        }),
    ]
    row = S.ranked_frontiers(frontiers, turns)[0]
    assert row["superseded_attempts_at_frontier"] == 0
    assert row["paid_attempts_at_frontier"] == 2
    assert row["retry_complexity_n"] == 2
    assert row["recommended_effort"] == "xhigh"
    assert row["quarantined_after_escalation_failure"] is False

    turns.append(_turn_on(binding, **{
        "game": "cold", "target_level": 1, "solved_target": False,
        "reasoning_effort": "medium",
        "started_at": "2026-07-24T10:01:00Z",
    }))
    row = S.ranked_frontiers(frontiers, turns)[0]
    assert row["superseded_attempts_at_frontier"] == 0
    assert row["paid_attempts_at_frontier"] == 3
    assert row["retry_complexity_n"] == 3
    assert row["recommended_effort"] == "xhigh"
    assert row["recommended_minutes"] == 40


def test_ranked_frontier_uses_same_retry_coordinate_for_sidecars():
    frontiers = [_frontier(
        "stalled", 4, incumbent_kind="promoted",
        solver_source_bytes=20_000, priority_score=4.0,
    )]
    binding = frontiers[0]
    turns = [
        _turn_on(binding, **{
            "game": "stalled", "target_level": 5, "solved_target": False,
            "reasoning_effort": "medium",
        }),
        _turn_on(binding, **{
            "game": "stalled", "target_level": 5, "solved_target": False,
            "reasoning_effort": "high",
        }),
    ]
    row = S.ranked_frontiers(frontiers, turns)[0]
    assert row["retry_complexity_n"] == 2
    assert row["quarantined_after_escalation_failure"] is False
    assert row["recommended_effort"] == "xhigh"
    assert row["recommended_auxiliary_parallelism"] == 0

    for index in range(3):
        turns.append(_turn_on(binding, **{
            "game": "stalled",
            "target_level": 5,
            "solved_target": False,
            "reasoning_effort": "max" if index else "xhigh",
        }))
    row = S.ranked_frontiers(frontiers, turns)[0]
    assert row["retry_complexity_n"] == 5
    assert row["recommended_effort"] == "max"
    assert row["recommended_wip_mode"] == "exclude"
    assert row["recommended_auxiliary_parallelism"] == 1


def test_ranked_frontier_counts_only_the_exact_bound_parent():
    frontier = _frontier(
        "same", 4, incumbent_kind="promoted", priority_score=4.0
    )
    other = dict(_binding("same", 4, action_count=99))
    other["parent_checkpoint_sha256"] = "c" * 64
    other["parent_source_tree_sha256"] = "d" * 64
    other["frontier_sha256"] = S._sha256_json({
        "game": "same",
        "reached": 4,
        "parent_checkpoint_sha256": other[
            "parent_checkpoint_sha256"
        ],
    })
    other = S.validate_frontier_binding(other)
    turns = [
        _turn_on(
            frontier,
            game="same",
            target_level=5,
            solved_target=False,
            reasoning_effort="medium",
        ),
        _turn_on(
            other,
            game="same",
            target_level=5,
            solved_target=False,
            reasoning_effort="high",
        ),
        {
            "game": "same",
            "target_level": 5,
            "solved_target": False,
            "reasoning_effort": "max",
        },
    ]
    row = S.ranked_frontiers([frontier], turns)[0]
    assert row["retry_complexity_n"] == 1
    assert row["exact_bound_turns_at_frontier"] == 1
    assert row["superseded_attempts_at_frontier"] == 1
    assert row["unbound_legacy_turns_for_game_level"] == 1
    assert row["game_level_history_turns"] == 3
    assert row["retry_history_authority"] == "exact_parent_bound_only"


def test_same_checkpoint_with_different_source_parent_cannot_pool_retries():
    frontier = _frontier(
        "same", 4, incumbent_kind="promoted", priority_score=4.0
    )
    different_source = {
        field: frontier[field]
        for field in (
            *S.FRONTIER_BINDING_FIELDS,
            "game",
            "reached",
            "target_level",
            "parent_action_count",
        )
    }
    different_source["parent_source_tree_sha256"] = "c" * 64
    different_source = S.validate_frontier_binding(different_source)
    turns = [
        _turn_on(
            frontier,
            solved_target=False,
            reasoning_effort="medium",
        ),
        _turn_on(
            different_source,
            solved_target=False,
            reasoning_effort="high",
        ),
    ]
    row = S.ranked_frontiers([frontier], turns)[0]
    assert row["retry_complexity_n"] == 1
    assert row["exact_bound_turns_at_frontier"] == 1
    assert row["superseded_attempts_at_frontier"] == 1


def test_ranked_frontier_rejects_partial_or_internally_inconsistent_binding():
    frontier = _frontier(
        "same", 4, incumbent_kind="promoted", priority_score=4.0
    )
    missing_action_count = _turn_on(
        frontier,
        solved_target=False,
        reasoning_effort="medium",
    )
    del missing_action_count["parent_action_count"]
    wrong_action_count = _turn_on(
        frontier,
        solved_target=False,
        reasoning_effort="high",
        parent_action_count=999,
    )
    wrong_reached = _turn_on(
        frontier,
        solved_target=False,
        reasoning_effort="max",
        reached=3,
    )
    row = S.ranked_frontiers(
        [frontier],
        [missing_action_count, wrong_action_count, wrong_reached],
    )[0]
    assert row["retry_complexity_n"] == 0
    assert row["exact_bound_turns_at_frontier"] == 0
    assert row["unbound_legacy_turns_for_game_level"] == 2
    assert row["superseded_attempts_at_frontier"] == 1


def test_exact_frontier_binding_changes_when_parent_source_changes(tmp_path):
    artifact = tmp_path / "g_legs"
    artifact.mkdir()
    (artifact / "checkpoint.json").write_text(json.dumps({
        "game": "g",
        "reached": 1,
        "final_path": [1, 2],
        "validated": True,
    }))
    for name in ("legs.py", "players.py", "solve.py"):
        (artifact / name).write_text(f"# {name}\n")
    first = S.exact_frontier_binding(
        artifact, game="g", target_level=2
    )
    (artifact / "legs.py").write_text("# changed\n")
    second = S.exact_frontier_binding(
        artifact, game="g", target_level=2
    )
    assert first["parent_checkpoint_sha256"] == second[
        "parent_checkpoint_sha256"
    ]
    assert first["frontier_sha256"] == second["frontier_sha256"]
    assert first["parent_source_tree_sha256"] != second[
        "parent_source_tree_sha256"
    ]


def test_retry_policy_projects_both_escalation_axes():
    expected = {
        0: ("medium", 0),
        1: ("high", 0),
        2: ("xhigh", 0),
        3: ("xhigh", 0),
        4: ("max", 0),
        5: ("max", 1),
        6: ("max", 1),
        7: ("max", 2),
        8: ("max", 2),
        9: ("max", 2),
        10: ("max", 2),
    }
    for n, (effort, sidecars) in expected.items():
        policy = S.retry_policy(n)
        assert policy["n"] == n
        assert policy["effort"] == effort
        assert policy["auxiliary_parallelism"] == sidecars
    assert S.retry_policy(9)["wip_mode"] == "exclude"
    assert S.retry_policy(10)["wip_mode"] == "restore_clean_same_frontier"
