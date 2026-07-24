#!/usr/bin/env python3
"""Report headless Codex allowance, solve efficiency, and next GKM frontiers.

Without ``--live`` this is entirely local and uses the last postflight snapshot
in the durable ledger.  ``--live`` performs only ``account/rateLimits/read``;
it does not start a model turn or consume a reset credit.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Optional

import codex_usage_guard as Guard


HERE = Path(__file__).resolve().parent
ARC_ROOT = HERE.parent
ARTIFACTS = HERE / "agent_solutions"
AUDITS = ARC_ROOT / "audit_results"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _external_profiles(audits: Path = AUDITS) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = defaultdict(dict)
    for system, name in (
        ("baseline1", "baseline1_gpt55_xhigh_solved_checkpoints.json"),
        ("Retrodict", "retrodict-solved-checkpoint-memory.json"),
    ):
        path = audits / name
        if not path.exists():
            continue
        for row in _read_json(path).get("rows", []):
            game = row.get("game")
            completed = row.get("completed_levels")
            if isinstance(game, str) and isinstance(completed, int):
                result[game][system] = max(
                    result[game].get(system, 0), completed
                )
    return dict(result)


def _external_ceilings(audits: Path = AUDITS) -> dict[str, int]:
    return {
        game: max(profile.values())
        for game, profile in _external_profiles(audits).items()
        if profile
    }


def _definition_count(path: Path) -> int:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError):
        return 0
    return sum(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        for node in tree.body
    )


def frontier_rows(artifacts: Path = ARTIFACTS,
                  audits: Path = AUDITS) -> list[dict[str, Any]]:
    profiles = _external_profiles(audits)
    local: dict[str, tuple[Path, dict[str, Any]]] = {}
    for artifact in sorted(artifacts.glob("*_legs")):
        checkpoint_path = artifact / "checkpoint.json"
        if not checkpoint_path.exists():
            continue
        checkpoint = _read_json(checkpoint_path)
        game = checkpoint.get("game")
        if isinstance(game, str):
            local[game] = (artifact, checkpoint)

    rows = []
    for game in sorted(set(profiles) | set(local)):
        artifact, checkpoint = local.get(game, (None, {}))
        candidate_dir = artifacts / f"{game}_legs"
        warm_wip = (candidate_dir / "wip_context").exists()
        reached = checkpoint.get("reached", 0)
        if not isinstance(reached, int):
            reached = 0
        next_level = reached + 1
        scaffold_path = (
            candidate_dir / "wip_context" / f"level_{next_level:02d}"
            / "frontier_scaffold.json"
        )
        scaffold = _read_json(scaffold_path) if scaffold_path.exists() else {}
        external = profiles.get(game, {})
        ceiling = max(external.values(), default=reached)
        if reached >= ceiling:
            continue
        sources = (
            [artifact / name for name in ("legs.py", "players.py", "solve.py")]
            if artifact is not None else []
        )
        source_bytes = sum(path.stat().st_size for path in sources if path.exists())
        definitions = (
            sum(_definition_count(artifact / name)
                for name in ("legs.py", "players.py"))
            if artifact is not None else 0
        )
        gap = ceiling - reached
        # Operational heuristic only: reward a mature incumbent and a one-level
        # completion opportunity, while penalizing context that will be replayed
        # through every agent/tool iteration.
        if artifact is None:
            values = list(external.values())
            consensus_floor = min(values, default=0)
            spread = max(values, default=0) - consensus_floor
            # Cold L1 trials have essentially no retained-source context.  Rank
            # games solved deeply by both external systems ahead of repeatedly
            # paying for a stalled mature frontier.
            priority_score = 1.15 + 0.12 * consensus_floor - 0.08 * spread
            if warm_wip:
                priority_score += 0.2
        else:
            priority_score = reached / (1.0 + source_bytes / 10_000.0)
            if gap == 1:
                priority_score += 1.0
        rows.append({
            "game": game,
            "incumbent_kind": "promoted" if artifact is not None else "cold_start",
            "warm_wip_available": warm_wip,
            "current_level": reached,
            "next_level": next_level,
            "frontier_scaffold_version": scaffold.get("version"),
            "frontier_scaffold_created_at": scaffold.get("created_at"),
            "external_artifact_ceiling": ceiling,
            "external_evidence": external,
            "levels_to_external_ceiling": gap,
            "solver_source_bytes": source_bytes,
            "top_level_definitions": definitions,
            "priority_score": round(priority_score, 3),
        })
    return sorted(rows, key=lambda row: (-row["priority_score"], row["game"]))


def effort_efficiency(turns: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Summarize proposal yield by effort without implying a randomized comparison."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for turn in turns:
        if not str(turn.get("run_label") or "").endswith(":propose"):
            continue
        effort = turn.get("reasoning_effort")
        if isinstance(effort, str):
            grouped[effort].append(turn)

    result: dict[str, dict[str, Any]] = {}
    for effort, rows in sorted(grouped.items()):
        solved = [row for row in rows if row.get("solved_target") is True]
        failed = [row for row in rows if row.get("solved_target") is False]
        points = [row.get("displayed_weekly_points_used") for row in rows]
        known_points = [value for value in points if isinstance(value, int)]
        success_points = [
            row.get("displayed_weekly_points_used") for row in solved
            if isinstance(row.get("displayed_weekly_points_used"), int)
        ]
        duration = sum(float(row.get("duration_seconds") or 0.0) for row in rows)
        missing_usage = sum(
            1 for row in rows
            if not isinstance(row.get("observed_tokens"), int)
            or row.get("observed_tokens") == 0
        )
        result[effort] = {
            "proposal_attempts": len(rows),
            "solved_levels": len(solved),
            "failed_levels": len(failed),
            "unknown_outcomes": len(rows) - len(solved) - len(failed),
            "timed_out_turns": sum(bool(row.get("timed_out")) for row in rows),
            "displayed_weekly_points": sum(known_points),
            "displayed_points_on_successes": sum(success_points),
            "displayed_points_per_solved_level": (
                round(sum(known_points) / len(solved), 3) if solved else None
            ),
            "success_only_points_per_solved_level": (
                round(sum(success_points) / len(solved), 3) if solved else None
            ),
            "displayed_points_per_wall_minute": (
                round(sum(known_points) / (duration / 60.0), 3)
                if duration else None
            ),
            "turns_with_missing_token_usage": missing_usage,
            "observed_tokens_are_complete": missing_usage == 0,
        }
    return result


def effort_efficiency_by_phase(
    turns: list[dict[str, Any]],
) -> dict[str, dict[str, dict[str, Any]]]:
    """Separate cold game entry from retained-solver continuation attempts.

    The two phases have visibly different difficulty and context sizes.  A
    pooled medium/high comparison is therefore useful for bookkeeping but not
    for deciding which arm is cheaper on a continuation frontier.
    """
    phases = {
        "cold_L1": [
            turn for turn in turns
            if turn.get("target_level") == 1
        ],
        "continuation_L2_plus": [
            turn for turn in turns
            if isinstance(turn.get("target_level"), int)
            and turn["target_level"] >= 2
        ],
    }
    return {
        phase: effort_efficiency(rows)
        for phase, rows in phases.items()
    }


def effort_solve_quality(
    turns: list[dict[str, Any]], audits: Path = AUDITS
) -> dict[str, dict[str, Any]]:
    """Join paid solves to exact GKM checkpoints and summarize solver structure.

    The conditional normalized-AST marginal is an executable-artifact proxy for
    description length. It is not inferred from transcript or episode length.
    Existing turns were not matched by frontier difficulty, so the result is
    descriptive rather than a causal high-versus-medium estimate.
    """
    path = audits / "marginal-literal-reuse.json"
    audit_rows: dict[tuple[str, int], dict[str, Any]] = {}
    if path.exists():
        for row in _read_json(path).get("rows", []):
            game, level = row.get("game"), row.get("completed_level")
            if (
                row.get("system") == "GKM"
                and row.get("source_checkpoint_exact") is True
                and isinstance(game, str)
                and isinstance(level, int)
            ):
                audit_rows[(game, level)] = row

    grouped: dict[str, list[tuple[dict[str, Any], Optional[dict[str, Any]]]]] = (
        defaultdict(list)
    )
    for turn in turns:
        if (
            not str(turn.get("run_label") or "").endswith(":propose")
            or turn.get("solved_target") is not True
        ):
            continue
        effort, game, level = (
            turn.get("reasoning_effort"),
            turn.get("game"),
            turn.get("target_level"),
        )
        if isinstance(effort, str) and isinstance(game, str) and isinstance(level, int):
            grouped[effort].append((turn, audit_rows.get((game, level))))

    result: dict[str, dict[str, Any]] = {}
    for effort, pairs in sorted(grouped.items()):
        audited = [row for _, row in pairs if row is not None]
        ast_marginals = [
            row.get("marginal_ast_zlib_bytes")
            for row in audited
            if isinstance(row.get("marginal_ast_zlib_bytes"), int)
        ]
        acquisition_charges = [
            turn.get("winning_marginal_C")
            for turn, _ in pairs
            if isinstance(turn.get("winning_marginal_C"), int)
        ]
        result[effort] = {
            "solved_levels": len(pairs),
            "exact_checkpoint_coverage": len(audited),
            "median_conditional_ast_zlib_bytes": (
                float(median(ast_marginals)) if ast_marginals else None
            ),
            "median_pre_debrief_acquisition_charge": (
                float(median(acquisition_charges))
                if acquisition_charges else None
            ),
            "literal_reuse_wins": sum(
                row.get("hard_literal_reuse_witness") is True for row in audited
            ),
            "sharp_marginal_drop_wins": sum(
                row.get("sharp_marginal_drop") is True for row in audited
            ),
            "sharp_drop_with_literal_reuse_wins": sum(
                row.get("sharp_drop_with_literal_reuse") is True for row in audited
            ),
            "checkpoint_details": [
                {
                    "game": turn["game"],
                    "completed_level": turn["target_level"],
                    "marginal_ast_zlib_bytes": (
                        row.get("marginal_ast_zlib_bytes") if row else None
                    ),
                    "pre_debrief_acquisition_charge": turn.get("winning_marginal_C"),
                    "literal_reuse": (
                        row.get("hard_literal_reuse_witness") if row else None
                    ),
                    "sharp_marginal_drop": (
                        row.get("sharp_marginal_drop") if row else None
                    ),
                }
                for turn, row in pairs
            ],
        }
    return result


def _iso_epoch(value: Any) -> Optional[float]:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.timestamp()


# Pre-adaptive static wall-time caps, kept as the conservative fallback when the
# ledger has too few replay-validated solves in an arm to size it empirically.
STATIC_WALL_MINUTES = {
    ("cold_L1", "medium"): 6,
    ("cold_L1", "high"): 6,
    ("continuation_L2+", "medium"): 8,
    ("continuation_L2+", "high"): 8,
}
# An arm needs at least this many replay-validated solves before its own solve-time
# distribution overrides the static cap.  Below it, thin/noisy evidence is not trusted.
MIN_SOLVES_TO_SIZE = 3
# Solve-preserving margin over the slowest observed solve.  Solves cluster against
# the historical cap (right-censored), so the true tail can exceed what we have seen;
# the margin decensors without wildly over-allocating.  The user's rule: never
# truncate good continuation WIP to save a few minutes.
WALL_SAFETY_FACTOR = 1.15
# Per-effort floors mirror the headroom floors; the ceiling is a secondary safety cap.
WALL_MINUTES_FLOOR = {"medium": 5, "high": 6}
WALL_MINUTES_CEILING = 15


def _phase_of_level(level: Any) -> Optional[str]:
    """Map a target level to the cold-entry vs retained-WIP continuation phase."""
    if level == 1:
        return "cold_L1"
    if isinstance(level, int) and level >= 2:
        return "continuation_L2+"
    return None


def _validated_solve_minutes(
    phase: str, effort: str, turns: list[dict[str, Any]]
) -> list[float]:
    """Wall minutes of replay-validated proposal solves in one (phase, effort) arm."""
    minutes = []
    for turn in turns:
        if not str(turn.get("run_label") or "").endswith(":propose"):
            continue
        if turn.get("solved_target") is not True:
            continue
        if turn.get("reasoning_effort") != effort:
            continue
        if _phase_of_level(turn.get("target_level")) != phase:
            continue
        duration = turn.get("duration_seconds")
        if isinstance(duration, (int, float)) and duration > 0:
            minutes.append(float(duration) / 60.0)
    return minutes


def recommend_minutes(
    phase: Optional[str], effort: str, turns: list[dict[str, Any]]
) -> dict[str, Any]:
    """Solve-preserving adaptive wall-time for a (phase, effort) arm.

    The binding constraint is that no historically replay-validated solve would
    have been truncated: the recommendation covers the slowest such solve plus a
    censoring margin.  With fewer than ``MIN_SOLVES_TO_SIZE`` solves the arm keeps
    its conservative static cap.  Returns the minutes plus provenance so the plan
    is auditable and the recommendation is never a bare unexplained number.
    """
    static = STATIC_WALL_MINUTES.get((phase, effort), 8)
    solves = (
        _validated_solve_minutes(phase, effort, turns)
        if phase is not None else []
    )
    if len(solves) < MIN_SOLVES_TO_SIZE:
        return {
            "minutes": static,
            "basis": "static_fallback",
            "solve_samples": len(solves),
            "slowest_solve_minutes": (
                round(max(solves), 2) if solves else None
            ),
        }
    slowest = max(solves)
    floor = WALL_MINUTES_FLOOR.get(effort, 5)
    needed = math.ceil(slowest * WALL_SAFETY_FACTOR)
    minutes = max(floor, min(WALL_MINUTES_CEILING, needed))
    # Hard guarantee of the solve-preserving property even if the ceiling binds.
    minutes = max(minutes, math.ceil(slowest))
    return {
        "minutes": minutes,
        "basis": "empirical_solve_preserving",
        "solve_samples": len(solves),
        "slowest_solve_minutes": round(slowest, 2),
    }


def ranked_frontiers(frontiers: list[dict[str, Any]],
                     turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Apply paid-attempt history and choose a provisional effort for each frontier."""
    attempts: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for turn in turns:
        game, level = turn.get("game"), turn.get("target_level")
        if isinstance(game, str) and isinstance(level, int):
            attempts[(game, level)].append(turn)

    ranked = []
    for row in frontiers:
        all_prior = attempts[(row["game"], row["next_level"])]
        scaffold_at = row.get("frontier_scaffold_created_at")
        scaffold_epoch = _iso_epoch(scaffold_at)
        if scaffold_epoch is not None:
            # A versioned, reviewed scaffold is a materially changed intervention.
            # Earlier failures remain charged in campaign economics but do not
            # quarantine the new policy.
            prior = [
                turn for turn in all_prior
                if (
                    _iso_epoch(turn.get("started_at")) is None
                    or _iso_epoch(turn.get("started_at")) >= scaffold_epoch
                )
            ]
        else:
            prior = all_prior
        failures = [turn for turn in prior if turn.get("solved_target") is False]
        medium_failures = sum(
            turn.get("reasoning_effort") == "medium" for turn in failures
        )
        high_failures = sum(
            turn.get("reasoning_effort") == "high" for turn in failures
        )
        quarantined = (
            high_failures >= 2
            or (medium_failures >= 1 and high_failures >= 1)
        )
        failed_efforts = sorted({
            str(turn.get("reasoning_effort")) for turn in failures
            if turn.get("reasoning_effort")
        })
        if quarantined:
            effort = None
            mode = "quarantined_after_escalation_failure"
        elif "medium" in failed_efforts or "high" in failed_efforts:
            effort = "high"
            mode = "continue_clean_wip"
        elif row["incumbent_kind"] == "cold_start":
            effort = "medium"
            mode = "cold_l1_screen"
        else:
            # Solver size is not evidence that high effort is cheaper.  In the
            # observed window, medium produced every new literal-reuse witness,
            # while high's only continuation clear was a rescue after medium
            # failed.  Therefore every genuinely fresh continuation starts on
            # medium; high is a single bounded escalation, never a size-based
            # default.
            effort = "medium"
            mode = "medium_first"
        # Wall time is sized from this arm's replay-validated solve-time
        # distribution (solve-preserving), not a static per-phase constant, so a
        # continuation whose solves historically ran to the old cap is no longer
        # truncated.  Quarantined frontiers get no turn, hence zero minutes.
        phase = _phase_of_level(row["next_level"])
        if effort is None:
            wall = {"minutes": 0, "basis": "quarantined",
                    "solve_samples": 0, "slowest_solve_minutes": None}
        else:
            wall = recommend_minutes(phase, effort, turns)
        minutes = wall["minutes"]
        adjusted = (
            -1_000_000.0 if quarantined
            else float(row["priority_score"]) - 0.8 * len(failures)
            # A cold frontier reset is worth revisiting, but should not displace a
            # fresh mature continuation merely because external agents solved it.
            - (1.0 if scaffold_epoch is not None else 0.0)
        )
        ranked.append({
            **row,
            "paid_attempts_at_frontier": len(prior),
            "superseded_attempts_at_frontier": len(all_prior) - len(prior),
            "failed_attempts_at_frontier": len(failures),
            "failed_efforts": failed_efforts,
            "quarantined_after_escalation_failure": quarantined,
            "recommended_effort": effort,
            "recommended_minutes": minutes,
            "recommended_minutes_basis": wall["basis"],
            "recommended_minutes_solve_samples": wall["solve_samples"],
            "slowest_validated_solve_minutes": wall["slowest_solve_minutes"],
            "dispatch_mode": mode,
            "adjusted_priority_score": round(adjusted, 3),
        })
    return sorted(
        ranked,
        key=lambda row: (-row["adjusted_priority_score"], row["game"]),
    )


def _transcript_counts(record: dict[str, Any]) -> dict[str, int]:
    workspace = record.get("workspace")
    transcript = record.get("transcript")
    result = {"command_executions": 0, "file_changes": 0}
    if not isinstance(workspace, str) or not isinstance(transcript, str):
        return result
    path = HERE / "runs" / "scratch" / workspace / transcript
    if not path.exists():
        return result
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        try:
            event = json.loads(raw)
        except json.JSONDecodeError:
            continue
        item = event.get("item", {})
        if event.get("type") != "item.completed" or not isinstance(item, dict):
            continue
        if item.get("type") == "command_execution":
            result["command_executions"] += 1
        elif item.get("type") == "file_change":
            result["file_changes"] += 1
    return result


def joined_turns(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    outcomes = {
        row.get("thread_id"): row
        for row in records
        if row.get("event") == "codex_level_outcome" and row.get("thread_id")
    }
    result = []
    for row in records:
        if row.get("event") != "codex_exec":
            continue
        outcome = outcomes.get(row.get("thread_id"), {})
        before, after = row.get("weekly_remaining_before"), row.get("weekly_remaining_after")
        weekly_delta = before - after if isinstance(before, int) and isinstance(after, int) else None
        result.append({
            "thread_id": row.get("thread_id"),
            "started_at": row.get("started_at"),
            "run_label": row.get("run_label"),
            "model": row.get("model"),
            "reasoning_effort": row.get("reasoning_effort"),
            "duration_seconds": row.get("duration_seconds"),
            "minutes_limit": row.get("minutes_limit"),
            "timed_out": row.get("timed_out"),
            "observed_tokens": row.get("observed_tokens"),
            "cached_input_tokens": row.get("cached_input_tokens"),
            "reasoning_output_tokens": row.get("reasoning_output_tokens"),
            "weekly_remaining_before": before,
            "weekly_remaining_after": after,
            "displayed_weekly_points_used": weekly_delta,
            "solved_target": outcome.get("solved_target"),
            "game": outcome.get("game"),
            "target_level": outcome.get("target_level"),
            "winning_marginal_C": outcome.get("winning_marginal_C"),
            "taint_verdict": outcome.get("taint_verdict"),
            **_transcript_counts(row),
        })
    return result


def _joined_window_turns(
    records: list[dict[str, Any]], exec_records: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    thread_ids = {
        row.get("thread_id") for row in exec_records if row.get("thread_id")
    }
    outcomes = [
        row for row in records
        if row.get("event") == "codex_level_outcome"
        and row.get("thread_id") in thread_ids
    ]
    return joined_turns([*exec_records, *outcomes])


def _allowance_from_records(records: list[dict[str, Any]],
                            turns: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
    for record in reversed(records):
        if record.get("event") == "rate_limit_snapshot":
            allowance = record.get("allowance")
            if isinstance(allowance, dict) and isinstance(
                allowance.get("remaining_percent"), int
            ):
                return {**allowance, "source": "cached_live_rate_limit_read"}
        if record.get("event") == "codex_exec":
            remaining = record.get("weekly_remaining_after")
            if isinstance(remaining, int):
                return {
                    "remaining_percent": remaining,
                    "resets_at": record.get("weekly_resets_at"),
                    "source": "last_postflight",
                }
    # Retain compatibility for synthetic callers that pass joined turns only.
    for turn in reversed(turns):
        remaining = turn.get("weekly_remaining_after")
        if isinstance(remaining, int):
            return {"remaining_percent": remaining, "source": "last_postflight"}
    return None


def _readiness(remaining: Optional[int], reserve: int,
               medium_headroom: int, high_headroom: int,
               totals: dict[str, int], max_runs: int,
               max_tokens: int) -> dict[str, Any]:
    local_budget_ok = (
        (max_runs < 0 or totals["runs"] < max_runs)
        and (max_tokens < 0 or totals["observed_tokens"] < max_tokens)
    )

    def ready(required: int) -> bool:
        return bool(
            remaining is not None
            and remaining > reserve
            and remaining - reserve >= required
            and local_budget_ok
        )

    return {
        "reserve_percent": reserve,
        "medium_required_headroom_percent": medium_headroom,
        "high_required_headroom_percent": high_headroom,
        "available_headroom_percent": remaining - reserve if remaining is not None else None,
        "local_budget_ok": local_budget_ok,
        "medium_admissible": ready(medium_headroom),
        "high_admissible": ready(high_headroom),
    }


def campaign_report(*, ledger: Path = Guard.DEFAULT_LEDGER,
                    artifacts: Path = ARTIFACTS, audits: Path = AUDITS,
                    live_snapshot: Optional[dict[str, Any]] = None,
                    reserve: int = 20, medium_headroom: int = 4,
                    high_headroom: int = 6, max_runs: int = 60,
                    max_tokens: int = 32_000_000) -> dict[str, Any]:
    records = Guard.read_ledger(ledger)
    turns = joined_turns(records)
    window_turns: list[dict[str, Any]] = []
    allowance = None
    local_totals = Guard.local_window_totals([])
    if live_snapshot is not None:
        live = Guard.weekly_allowance(live_snapshot)
        allowance = {**live.as_dict(), "source": "live_rate_limit_read"}
        current = Guard.current_window_records(records, live)
        local_totals = Guard.local_window_totals(current)
        window_turns = _joined_window_turns(records, current)
    else:
        allowance = _allowance_from_records(records, turns)
        if turns:
            reset = allowance.get("resets_at") if allowance else None
            if not isinstance(reset, int):
                reset = next(
                    (row.get("weekly_resets_at") for row in reversed(records)
                     if row.get("event") == "codex_exec"),
                    None,
                )
            if allowance is not None:
                allowance.setdefault("resets_at", reset)
                allowance.setdefault(
                    "resets_at_iso",
                    datetime.fromtimestamp(reset, timezone.utc).isoformat()
                    if isinstance(reset, int) else None,
                )
            current = [
                row for row in records
                if row.get("event") == "codex_exec"
                and isinstance(row.get("weekly_resets_at"), int)
                and isinstance(reset, int)
                and abs(row["weekly_resets_at"] - reset)
                <= Guard.RESET_EPOCH_TOLERANCE_SECONDS
            ]
            local_totals = Guard.local_window_totals(current)
            window_turns = _joined_window_turns(records, current)

    remaining = allowance.get("remaining_percent") if allowance else None
    frontiers = ranked_frontiers(frontier_rows(artifacts, audits), turns)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "allowance": allowance,
        "local_window": local_totals,
        "readiness": _readiness(
            remaining, reserve, medium_headroom, high_headroom,
            local_totals, max_runs, max_tokens,
        ),
        "turns": turns,
        "effort_efficiency": effort_efficiency(turns),
        "window_effort_efficiency": effort_efficiency(window_turns),
        "effort_efficiency_by_phase": effort_efficiency_by_phase(turns),
        "window_effort_efficiency_by_phase": effort_efficiency_by_phase(
            window_turns
        ),
        "solver_quality_by_effort": effort_solve_quality(turns, audits),
        "window_solver_quality_by_effort": effort_solve_quality(
            window_turns, audits
        ),
        "effort_comparison_identified": False,
        "effort_comparison_note": (
            "medium and high were not randomized or matched by frontier difficulty; "
            "cost and exact-checkpoint solver-quality summaries are descriptive, "
            "not a causal estimate"
        ),
        "frontiers": frontiers,
        "recommended_frontier": frontiers[0] if frontiers else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--ledger", type=Path, default=Guard.DEFAULT_LEDGER)
    parser.add_argument("--reserve-percent", type=int, default=20)
    parser.add_argument("--medium-headroom-percent", type=int, default=4)
    parser.add_argument("--high-headroom-percent", type=int, default=6)
    parser.add_argument("--max-campaign-runs", type=int, default=60)
    parser.add_argument("--max-campaign-tokens", type=int, default=32_000_000)
    args = parser.parse_args()
    snapshot = Guard.query_rate_limits() if args.live else None
    if snapshot is not None:
        live_allowance = Guard.weekly_allowance(snapshot)
        with Guard.campaign_lock(args.ledger):
            Guard.append_ledger({
                "event": "rate_limit_snapshot",
                "checked_at": datetime.now(timezone.utc).isoformat(),
                "allowance": live_allowance.as_dict(),
            }, args.ledger)
    report = campaign_report(
        ledger=args.ledger,
        live_snapshot=snapshot,
        reserve=args.reserve_percent,
        medium_headroom=args.medium_headroom_percent,
        high_headroom=args.high_headroom_percent,
        max_runs=args.max_campaign_runs,
        max_tokens=args.max_campaign_tokens,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
