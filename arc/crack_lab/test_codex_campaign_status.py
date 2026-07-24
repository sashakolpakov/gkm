import json
from pathlib import Path

import codex_campaign_status as S


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
    artifacts.mkdir(); audits.mkdir()
    for game, reached, source in (
        ("mature", 5, "def leg(env):\n    pass\n"),
        ("cold", 1, "def leg(env):\n    pass\n"),
    ):
        art = artifacts / f"{game}_legs"
        art.mkdir()
        (art / "checkpoint.json").write_text(json.dumps({"game": game, "reached": reached}))
        (art / "legs.py").write_text(source)
        (art / "players.py").write_text("")
        (art / "solve.py").write_text("")
    rows = [
        {"game": "mature", "completed_levels": 6},
        {"game": "cold", "completed_levels": 6},
    ]
    for name in (
        "baseline1_gpt55_xhigh_solved_checkpoints.json",
        "retrodict-solved-checkpoint-memory.json",
    ):
        (audits / name).write_text(json.dumps({"rows": rows}))
    result = S.frontier_rows(artifacts, audits)
    assert [row["game"] for row in result] == ["mature", "cold"]
    assert result[0]["next_level"] == 6


def test_frontier_rows_include_games_without_local_artifacts(tmp_path):
    artifacts = tmp_path / "artifacts"
    audits = tmp_path / "audits"
    artifacts.mkdir(); audits.mkdir()
    (audits / "baseline1_gpt55_xhigh_solved_checkpoints.json").write_text(
        json.dumps({"rows": [{"game": "cold", "completed_levels": 8}]})
    )
    (audits / "retrodict-solved-checkpoint-memory.json").write_text(
        json.dumps({"rows": [{"game": "cold", "completed_levels": 7}]})
    )
    row = S.frontier_rows(artifacts, audits)[0]
    assert row["game"] == "cold"
    assert row["incumbent_kind"] == "cold_start"
    assert row["current_level"] == 0
    assert row["next_level"] == 1
    assert row["external_artifact_ceiling"] == 8


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
    assert high["proposal_attempts"] == 2
    assert high["solved_levels"] == 1
    assert high["displayed_points_per_solved_level"] == 6.0
    assert high["success_only_points_per_solved_level"] == 2.0
    assert high["turns_with_missing_token_usage"] == 1
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
        {
            "game": "cold", "next_level": 1, "incumbent_kind": "cold_start",
            "solver_source_bytes": 0, "current_level": 0, "priority_score": 2.0,
        },
        {
            "game": "stalled", "next_level": 5, "incumbent_kind": "promoted",
            "solver_source_bytes": 20_000, "current_level": 4,
            "priority_score": 2.1,
        },
    ]
    turns = [{
        "game": "stalled", "target_level": 5, "solved_target": False,
        "reasoning_effort": "high",
    }]
    ranked = S.ranked_frontiers(frontiers, turns)
    assert ranked[0]["game"] == "cold"
    assert ranked[0]["recommended_effort"] == "medium"
    assert ranked[0]["recommended_minutes"] == 6
    assert ranked[1]["dispatch_mode"] == "continue_clean_wip"


def test_ranked_frontier_does_not_send_large_fresh_solver_straight_to_high():
    frontiers = [{
        "game": "large", "next_level": 5, "incumbent_kind": "promoted",
        "solver_source_bytes": 25_000, "current_level": 4,
        "priority_score": 4.0,
    }]
    row = S.ranked_frontiers(frontiers, [])
    assert row[0]["recommended_effort"] == "medium"
    assert row[0]["dispatch_mode"] == "medium_first"


def test_new_scaffold_supersedes_old_failures_but_not_new_attempts():
    frontiers = [{
        "game": "cold", "next_level": 1, "incumbent_kind": "cold_start",
        "solver_source_bytes": 0, "current_level": 0, "priority_score": 2.0,
        "frontier_scaffold_version": "v2",
        "frontier_scaffold_created_at": "2026-07-24T10:00:00Z",
    }]
    turns = [
        {
            "game": "cold", "target_level": 1, "solved_target": False,
            "reasoning_effort": "medium",
            "started_at": "2026-07-23T10:00:00+00:00",
        },
        {
            "game": "cold", "target_level": 1, "solved_target": False,
            "reasoning_effort": "high",
            "started_at": "2026-07-23T11:00:00Z",
        },
    ]
    row = S.ranked_frontiers(frontiers, turns)[0]
    assert row["superseded_attempts_at_frontier"] == 2
    assert row["paid_attempts_at_frontier"] == 0
    assert row["recommended_effort"] == "medium"
    assert row["quarantined_after_escalation_failure"] is False

    turns.append({
        "game": "cold", "target_level": 1, "solved_target": False,
        "reasoning_effort": "medium",
        "started_at": "2026-07-24T10:01:00Z",
    })
    row = S.ranked_frontiers(frontiers, turns)[0]
    assert row["superseded_attempts_at_frontier"] == 2
    assert row["paid_attempts_at_frontier"] == 1
    assert row["recommended_effort"] == "high"


def test_ranked_frontier_is_quarantined_after_medium_and_high_failure():
    frontiers = [{
        "game": "stalled", "next_level": 5, "incumbent_kind": "promoted",
        "solver_source_bytes": 20_000, "current_level": 4,
        "priority_score": 4.0,
    }]
    turns = [
        {
            "game": "stalled", "target_level": 5, "solved_target": False,
            "reasoning_effort": "medium",
        },
        {
            "game": "stalled", "target_level": 5, "solved_target": False,
            "reasoning_effort": "high",
        },
    ]
    row = S.ranked_frontiers(frontiers, turns)[0]
    assert row["quarantined_after_escalation_failure"] is True
    assert row["dispatch_mode"] == "quarantined_after_escalation_failure"
    assert row["recommended_effort"] is None
