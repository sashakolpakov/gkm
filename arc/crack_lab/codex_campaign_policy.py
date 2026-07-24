#!/usr/bin/env python3
"""Build a cost-aware post-reset Codex campaign without starting model turns.

The policy treats displayed weekly-percentage changes as the hard cost signal.
Provider token counters are retained for diagnosis, but timed-out Codex streams can
omit ``turn.completed`` and therefore report zero tokens.  Recommendations are
deliberately sequential: medium gets the first attempt at every fresh continuation;
high gets at most one bounded rescue attempt after a clean medium failure.
"""

from __future__ import annotations

import argparse
import json
import math
import shlex
from pathlib import Path
from typing import Any

import codex_campaign_status as Status
import codex_usage_guard as Guard


DEFAULT_RESERVE = 20
DEFAULT_MAX_RUNS = 60
DEFAULT_MAX_TOKENS = 32_000_000


def required_headroom(effort: str, minutes: int,
                      turns: list[dict[str, Any]]) -> int:
    """Return an empirical worst-rate allowance bound plus one displayed point."""
    if effort not in {"medium", "high"}:
        raise ValueError("effort must be medium or high")
    if minutes <= 0:
        raise ValueError("minutes must be positive")
    rates = []
    for turn in turns:
        if turn.get("reasoning_effort") != effort:
            continue
        # An operator-aborted turn is a real campaign charge, so it remains in
        # cost-per-solve accounting.  It is not a valid estimate of the burn
        # rate of a full admitted turn: displayed allowance is integer-rounded
        # and a short interruption can therefore extrapolate one or two points
        # into an arbitrarily large full-turn headroom requirement.
        if turn.get("interrupted") is True:
            continue
        points = turn.get("displayed_weekly_points_used")
        duration = turn.get("duration_seconds")
        if isinstance(points, int) and points >= 0 and isinstance(duration, (int, float)) and duration > 0:
            rates.append(points / (float(duration) / 60.0))
    # Floors cover a new effort with little evidence.  High's 12-minute observed
    # maximum was eight points, hence a nine-point default for a full turn.
    floor = 4 if effort == "medium" else 6
    if not rates:
        return floor
    return max(floor, math.ceil(max(rates) * minutes) + 1)


def _command(row: dict[str, Any], effort: str, *, reserve: int,
             turns: list[dict[str, Any]], minutes: int = 6) -> dict[str, Any]:
    headroom = required_headroom(effort, minutes, turns)
    args = [
        "python3", "-u", "arc/crack_lab/gkm_legs.py",
        f"--game={row['game']}",
        f"--max-level={row['next_level']}",
        "--proposer=codex",
        "--model=gpt-5.6-sol",
        f"--minutes={minutes}",
        f"--codex-effort={effort}",
        "--codex-debrief-effort=medium",
        "--debrief-policy=never",
        f"--codex-weekly-reserve={reserve}",
        f"--codex-weekly-headroom={headroom}",
        f"--codex-max-campaign-runs={DEFAULT_MAX_RUNS}",
        f"--codex-max-campaign-tokens={DEFAULT_MAX_TOKENS}",
        "--transient-retries=0",
        f"--tag=cheap_{effort}_screen",
    ]
    if row["incumbent_kind"] == "cold_start" and not row.get("warm_wip_available"):
        args.append("--fresh")
    return {
        "game": row["game"],
        "target_level": row["next_level"],
        "effort": effort,
        "minutes": minutes,
        "required_headroom_percent": headroom,
        "external_evidence": row.get("external_evidence", {}),
        "warm_wip_available": bool(row.get("warm_wip_available")),
        "argv": args,
        "command": shlex.join(args),
    }


def choose_exploitation_effort(
    efficiency: dict[str, dict[str, Any]],
    quality: dict[str, dict[str, Any]] | None = None,
) -> str | None:
    """Choose an arm after two attempts, using solver size only near a cost tie."""
    rows = {effort: efficiency.get(effort, {}) for effort in ("medium", "high")}
    if any(int(row.get("proposal_attempts") or 0) < 2 for row in rows.values()):
        return None

    def cost(effort: str) -> tuple[bool, float]:
        row = rows[effort]
        solves = int(row.get("solved_levels") or 0)
        points = float(row.get("displayed_weekly_points") or 0)
        attempts = int(row.get("proposal_attempts") or 0)
        if solves:
            return True, points / solves
        return False, points / max(attempts, 1)

    medium, high = cost("medium"), cost("high")
    if medium[0] != high[0]:
        return "medium" if medium[0] else "high"
    cheaper = min(("medium", "high"), key=lambda effort: (cost(effort)[1], effort))
    lower, upper = sorted((medium[1], high[1]))
    near_tie = upper == 0 or (upper - lower) / upper <= 0.10
    if near_tie and quality:
        ast_sizes = {
            effort: (quality.get(effort) or {}).get(
                "median_conditional_ast_zlib_bytes"
            )
            for effort in ("medium", "high")
        }
        if all(isinstance(value, (int, float)) for value in ast_sizes.values()):
            return min(ast_sizes, key=lambda effort: (ast_sizes[effort], effort))
    return cheaper


def high_rescue_summary(turns: list[dict[str, Any]]) -> dict[str, Any]:
    """Measure high only where it answers the operational escalation question.

    A qualifying attempt is a high turn on the same game/level after an earlier
    medium failure.  This avoids comparing high's mostly cold-L1 cohort with
    medium's harder continuation cohort.  It is still observational—the WIP may
    have improved during the failed medium turn—but it directly estimates the
    incremental cost and yield of the policy we actually intend to run.
    """
    medium_failed: set[tuple[str, int]] = set()
    attempts = 0
    rescues = 0
    points = 0
    details = []
    for turn in turns:
        game, level = turn.get("game"), turn.get("target_level")
        key = (game, level)
        if not isinstance(game, str) or not isinstance(level, int):
            continue
        effort = turn.get("reasoning_effort")
        solved = turn.get("solved_target")
        if effort == "medium" and solved is False:
            medium_failed.add(key)
            continue
        if effort != "high" or key not in medium_failed:
            continue
        attempts += 1
        rescued = solved is True
        rescues += int(rescued)
        charged = turn.get("displayed_weekly_points_used")
        if isinstance(charged, int) and charged >= 0:
            points += charged
        details.append({
            "game": game,
            "target_level": level,
            "rescued": rescued,
            "displayed_weekly_points_used": charged,
        })
    return {
        "qualifying_high_attempts": attempts,
        "replay_validated_rescues": rescues,
        "rescue_rate": round(rescues / attempts, 3) if attempts else None,
        "displayed_weekly_points": points,
        "displayed_points_per_rescue": (
            round(points / rescues, 3) if rescues else None
        ),
        "details": details,
    }


def adaptive_campaign_item(
    report: dict[str, Any], *, reserve: int
) -> dict[str, Any] | None:
    """Build the next item from fresh artifacts and charged effort outcomes."""
    frontiers = report.get("frontiers", [])
    turns = report.get("turns", [])
    if not isinstance(frontiers, list):
        return None
    untouched_cold = next(
        (
            row for row in frontiers
            if row.get("incumbent_kind") == "cold_start"
            and int(row.get("paid_attempts_at_frontier") or 0) == 0
        ),
        None,
    )
    if untouched_cold is not None:
        effort = choose_exploitation_effort(
            report.get("window_effort_efficiency")
            or report.get("effort_efficiency", {}),
            report.get("window_solver_quality_by_effort")
            or report.get("solver_quality_by_effort", {}),
        )
        if effort is None:
            return None
        item = _command(
            untouched_cold, effort, reserve=reserve, turns=turns, minutes=6
        )
        item["experiment_role"] = f"cold_L1_exploitation_{effort}"
        return item
    eligible = [
        row for row in frontiers
        if not row.get("quarantined_after_escalation_failure")
    ]
    if not eligible:
        return None
    row = eligible[0]
    effort = str(row.get("recommended_effort") or "medium")
    minutes = int(row.get("recommended_minutes") or 8)
    item = _command(
        row, effort, reserve=reserve, turns=turns, minutes=minutes
    )
    item["experiment_role"] = f"ranked_frontier_{effort}"
    return item


def initial_queue(report: dict[str, Any], *, reserve: int) -> list[dict[str, Any]]:
    """Seed one fresh, currently highest-ranked item.

    Only one item is frozen into the plan because every clear or failed attempt
    changes the frontier ranking.  The runner rebuilds all later items from fresh
    artifacts and a fresh allowance read.
    """
    eligible = [
        row for row in report.get("frontiers", [])
        if not row.get("quarantined_after_escalation_failure")
    ]
    if not eligible:
        return []
    row = eligible[0]
    effort = str(row.get("recommended_effort") or "medium")
    minutes = int(row.get("recommended_minutes") or 8)
    item = _command(
        row, effort, reserve=reserve, turns=report.get("turns", []),
        minutes=minutes,
    )
    if effort == "high" and "medium" in row.get("failed_efforts", []):
        item["experiment_role"] = "bounded_high_rescue_after_medium_failure"
    else:
        item["experiment_role"] = f"fresh_frontier_{effort}_first"
    return [item]


def cold_screen_cohort(report: dict[str, Any], *, reserve: int,
                       cohort_size: int = 4) -> list[dict[str, Any]]:
    """Pair similar cold L1 frontiers across medium and high reasoning."""
    cold = [
        row for row in report.get("frontiers", [])
        if row.get("incumbent_kind") == "cold_start"
        and row.get("paid_attempts_at_frontier", 0) == 0
    ][:cohort_size]
    cohort = []
    # Alternation keeps each arm represented among the highest-consensus games.
    for index, row in enumerate(cold):
        effort = "medium" if index % 2 == 0 else "high"
        item = _command(row, effort, reserve=reserve, turns=report.get("turns", []))
        item["experiment_role"] = f"cold_L1_{effort}"
        cohort.append(item)
    return cohort


def policy_report(report: dict[str, Any], *, reserve: int = DEFAULT_RESERVE,
                  cohort_size: int = 4) -> dict[str, Any]:
    queue = initial_queue(report, reserve=reserve)
    allowance = report.get("allowance") or {}
    remaining = allowance.get("remaining_percent")
    maximum_headroom = max(
        (row["required_headroom_percent"] for row in queue), default=0
    )
    current_runs = int((report.get("local_window") or {}).get("runs") or 0)
    if not queue:
        phase = "no_eligible_frontier"
        admit = False
    elif not isinstance(remaining, int):
        phase = "allowance_unknown"
        admit = False
    elif remaining < reserve + maximum_headroom:
        phase = "hold_for_weekly_reset"
        admit = False
    elif current_runs > 0 and remaining < 50:
        # Preserve the tail of an already productive window for ordinary work.
        phase = "hold_for_weekly_reset"
        admit = False
    else:
        phase = "run_initial_item_then_adapt"
        admit = True
    return {
        "phase": phase,
        "admit_next_turn": admit,
        "reserve_percent": reserve,
        "allowance": allowance,
        "not_before_epoch": allowance.get("resets_at") if not admit else None,
        "local_window": report.get("local_window"),
        "effort_efficiency": report.get("effort_efficiency", {}),
        "window_effort_efficiency": report.get("window_effort_efficiency", {}),
        "solver_quality_by_effort": report.get("solver_quality_by_effort", {}),
        "window_solver_quality_by_effort": report.get(
            "window_solver_quality_by_effort", {}
        ),
        "high_rescue_summary": high_rescue_summary(report.get("turns", [])),
        "causal_conclusion": "not_identified",
        "causal_reason": (
            "the existing medium and high turns differ in frontier difficulty; "
            "pooled points-per-clear and solver-size medians do not identify an "
            "intrinsic effort effect"
        ),
        "decision_rule_after_cohort": (
            "start every fresh continuation on medium; after a clean failure, "
            "allow one high rescue only if the live reserve plus empirical high "
            "headroom remains; charge failures and successes alike; quarantine "
            "after medium plus high both fail; paid debriefs remain disabled"
        ),
        "initial_queue": queue,
        # Kept empty so older readers do not mistake the completed cold-L1 screen
        # for a new calibration request.
        "cold_screen_cohort": [],
        "next_frontiers": report.get("frontiers", []),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--ledger", type=Path, default=Guard.DEFAULT_LEDGER)
    parser.add_argument("--reserve-percent", type=int, default=DEFAULT_RESERVE)
    parser.add_argument("--cohort-size", type=int, default=4)
    parser.add_argument("--write-plan", type=Path)
    args = parser.parse_args()
    snapshot = Guard.query_rate_limits() if args.live else None
    report = Status.campaign_report(
        ledger=args.ledger,
        live_snapshot=snapshot,
        reserve=args.reserve_percent,
        medium_headroom=4,
        high_headroom=6,
        max_runs=DEFAULT_MAX_RUNS,
        max_tokens=DEFAULT_MAX_TOKENS,
    )
    payload = policy_report(
        report, reserve=args.reserve_percent, cohort_size=args.cohort_size
    )
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.write_plan:
        args.write_plan.parent.mkdir(parents=True, exist_ok=True)
        args.write_plan.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
