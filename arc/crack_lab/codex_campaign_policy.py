#!/usr/bin/env python3
"""Build the exact-frontier ARC-AGI-3 Codex campaign queue.

One journal-reconstructed coordinate, ``retry_complexity_n``, selects effort,
soft allocation, WIP mode, and auxiliary eligibility through
``codex_campaign_status.retry_policy``.  Paid-turn, timeout, transcript, and
branch counts are diagnostic only and must never steer dispatch.  Explicit
provider ``unlimited`` semantics disable every cost cutoff uniformly while
retaining scheduling, taint, replay, provenance, and containment controls.
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


def lineage_input_modes(
    row: dict[str, Any], *, minutes: int
) -> tuple[str, str]:
    """Project the exact retry coordinate onto explicit seed/WIP inputs."""
    n = row.get("retry_complexity_n")
    if not isinstance(n, int) or isinstance(n, bool) or n < 0:
        raise ValueError(
            "frontier has no valid journal-derived retry_complexity_n"
        )
    policy = Status.retry_policy(n)
    if minutes != policy["minutes"]:
        raise ValueError(
            "turn allocation does not match retry policy: "
            f"n={n} requires {policy['minutes']} minutes, got {minutes}"
        )
    seed_mode = (
        "zero_seed"
        if row.get("incumbent_kind") == "cold_start"
        and int(row.get("current_level") or 0) == 0
        else "verified_parent"
    )
    wip_mode = str(policy["wip_mode"])
    if (
        wip_mode == "restore_clean_same_frontier"
        and row.get("warm_wip_available") is not True
    ):
        raise ValueError(
            "retry policy requires clean same-frontier WIP, but none is "
            "eligible"
        )
    return seed_mode, wip_mode


def required_headroom(effort: str, minutes: int,
                      turns: list[dict[str, Any]]) -> int:
    """Return an empirical worst-rate allowance bound plus one displayed point."""
    if effort not in {"medium", "high", "xhigh", "max"}:
        raise ValueError("effort must be medium, high, xhigh, or max")
    if minutes <= 0:
        raise ValueError("minutes must be positive")
    # This value is ignored when the provider explicitly reports an unlimited
    # pool, but keeping a positive placeholder preserves the plan schema.
    if effort in {"xhigh", "max"}:
        return 1
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


def _command(
    row: dict[str, Any],
    effort: str,
    *,
    reserve: int,
    turns: list[dict[str, Any]],
    minutes: int,
    unlimited: bool,
) -> dict[str, Any]:
    binding = Status.validate_frontier_binding({
        field: row.get(field)
        for field in (
            *Status.FRONTIER_BINDING_FIELDS,
            "game",
            "reached",
            "target_level",
            "parent_action_count",
        )
    })
    if (
        binding["target_level"] != row.get("next_level")
        or binding["reached"] != row.get("current_level")
    ):
        raise ValueError(
            "frontier binding disagrees with advertised campaign frontier"
        )
    n = row.get("retry_complexity_n")
    if not isinstance(n, int) or isinstance(n, bool) or n < 0:
        raise ValueError(
            "frontier has no valid journal-derived retry_complexity_n"
        )
    policy = Status.retry_policy(n)
    if effort != policy["effort"] or minutes != policy["minutes"]:
        raise ValueError(
            "requested command does not match retry policy: "
            f"n={n} requires {policy['effort']}/{policy['minutes']}"
        )
    advertised = {
        "recommended_effort": policy["effort"],
        "recommended_minutes": policy["minutes"],
        "recommended_wip_mode": policy["wip_mode"],
        "recommended_auxiliary_parallelism": policy[
            "auxiliary_parallelism"
        ],
        "dispatch_mode": policy["dispatch_mode"],
    }
    for key, value in advertised.items():
        if key in row and row[key] != value:
            raise ValueError(
                f"frontier {key} disagrees with retry policy: "
                f"{row[key]!r} != {value!r}"
            )
    headroom = 1 if unlimited else required_headroom(
        effort, minutes, turns
    )
    effective_reserve = 0 if unlimited else reserve
    max_runs = -1 if unlimited else DEFAULT_MAX_RUNS
    max_tokens = -1 if unlimited else DEFAULT_MAX_TOKENS
    seed_mode, wip_mode = lineage_input_modes(row, minutes=minutes)
    args = [
        "python3", "-u", "arc/crack_lab/gkm_legs.py",
        f"--game={row['game']}",
        f"--max-level={row['next_level']}",
        "--proposer=codex",
        "--model=gpt-5.6-sol",
        f"--minutes={minutes}",
        f"--codex-effort={effort}",
        "--codex-debrief-effort=medium",
        "--codex-allocation-policy=drain",
        "--debrief-policy=never",
        f"--codex-weekly-reserve={effective_reserve}",
        f"--codex-weekly-headroom={headroom}",
        f"--codex-max-campaign-runs={max_runs}",
        f"--codex-max-campaign-tokens={max_tokens}",
        "--transient-retries=0",
        f"--tag=arc_agi3_n{n}_{policy['dispatch_mode']}",
        f"--seed-mode={seed_mode}",
        f"--wip-mode={wip_mode}",
        f"--expected-parent-reached={binding['reached']}",
        (
            "--expected-parent-action-count="
            f"{binding['parent_action_count']}"
        ),
        (
            "--expected-parent-checkpoint-sha256="
            f"{binding['parent_checkpoint_sha256']}"
        ),
        (
            "--expected-parent-source-tree-sha256="
            f"{binding['parent_source_tree_sha256']}"
        ),
        f"--expected-frontier-sha256={binding['frontier_sha256']}",
    ]
    return {
        **binding,
        "game": row["game"],
        "target_level": row["next_level"],
        "effort": effort,
        "minutes": minutes,
        "retry_complexity_n": n,
        "dispatch_mode": policy["dispatch_mode"],
        "recommended_auxiliary_parallelism": policy[
            "auxiliary_parallelism"
        ],
        "cost_control_enabled": not unlimited,
        "max_campaign_runs": max_runs,
        "max_campaign_tokens": max_tokens,
        "required_headroom_percent": headroom,
        "external_evidence": row.get("external_evidence", {}),
        "warm_wip_available": bool(row.get("warm_wip_available")),
        "seed_mode": seed_mode,
        "wip_mode": wip_mode,
        "lineage_input_mode": f"{seed_mode}+{wip_mode}",
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


def unlimited_escalation(
    retry_complexity_n: int, recommended_minutes: int
) -> tuple[str, int, str]:
    """Return the one policy row for an unlimited exact frontier.

    ``recommended_minutes`` is retained only to reject stale callers; it may
    not enlarge or shrink the versioned retry table.
    """
    policy = Status.retry_policy(retry_complexity_n)
    if recommended_minutes not in {0, policy["minutes"]}:
        raise ValueError(
            "advertised minutes disagree with retry policy: "
            f"n={retry_complexity_n} requires {policy['minutes']}, "
            f"got {recommended_minutes}"
        )
    return (
        str(policy["effort"]),
        int(policy["minutes"]),
        f"retry_n{retry_complexity_n}_{policy['dispatch_mode']}",
    )


def adaptive_campaign_item(
    report: dict[str, Any], *, reserve: int
) -> dict[str, Any] | None:
    """Build the next item from fresh exact-frontier retry evidence."""
    frontiers = report.get("frontiers", [])
    turns = report.get("turns", [])
    if not isinstance(frontiers, list):
        return None
    unlimited = (
        (report.get("allowance") or {}).get("window_name") == "unlimited"
    )
    candidates = [
        row for row in frontiers
        if int(row.get("current_level") or 0)
        < int(row.get("authoritative_level_count") or 0)
    ]
    if not candidates:
        return None
    for row in candidates:
        n = row.get("retry_complexity_n")
        if not isinstance(n, int) or isinstance(n, bool) or n < 0:
            raise ValueError(
                "frontier is missing journal-derived retry_complexity_n"
            )
    candidates.sort(key=lambda row: (
        int(row["retry_complexity_n"]),
        -float(row.get("adjusted_priority_score")
               or row.get("priority_score") or 0.0),
        str(row.get("game") or ""),
    ))
    row = candidates[0]
    n = int(row["retry_complexity_n"])
    policy = Status.retry_policy(n)
    effort, minutes, role = unlimited_escalation(
        n, int(row.get("recommended_minutes") or 0)
    )
    item = _command(
        row,
        effort,
        reserve=reserve,
        turns=turns,
        minutes=minutes,
        unlimited=unlimited,
    )
    item["experiment_role"] = role
    item["policy_projection"] = policy
    return item


def initial_queue(report: dict[str, Any], *, reserve: int) -> list[dict[str, Any]]:
    """Seed one fresh, currently highest-ranked item.

    Only one item is frozen into the plan because every clear or failed attempt
    changes the frontier ranking.  The runner rebuilds all later items from fresh
    artifacts and a fresh allowance read.
    """
    item = adaptive_campaign_item(report, reserve=reserve)
    return [item] if item is not None else []


def cold_screen_cohort(report: dict[str, Any], *, reserve: int,
                       cohort_size: int = 4) -> list[dict[str, Any]]:
    """Historical helper: project unsolved cold roots through the n=0 row."""
    cold = [
        row for row in report.get("frontiers", [])
        if row.get("incumbent_kind") == "cold_start"
        and row.get("retry_complexity_n") == 0
    ][:cohort_size]
    cohort = []
    unlimited = (
        (report.get("allowance") or {}).get("window_name") == "unlimited"
    )
    for row in cold:
        item = _command(
            row,
            "medium",
            reserve=reserve,
            turns=report.get("turns", []),
            minutes=15,
            unlimited=unlimited,
        )
        item["experiment_role"] = "retry_n0_fresh_frontier"
        cohort.append(item)
    return cohort


def policy_report(report: dict[str, Any], *, reserve: int = DEFAULT_RESERVE,
                  cohort_size: int = 4) -> dict[str, Any]:
    queue = initial_queue(report, reserve=reserve)
    allowance = report.get("allowance") or {}
    unlimited = allowance.get("window_name") == "unlimited"
    remaining = allowance.get("remaining_percent")
    maximum_headroom = max(
        (row["required_headroom_percent"] for row in queue), default=0
    )
    current_runs = int((report.get("local_window") or {}).get("runs") or 0)
    if not queue:
        phase = "no_eligible_frontier"
        admit = False
    elif unlimited:
        phase = "run_initial_item_then_adapt"
        admit = True
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
        "reserve_percent": 0 if unlimited else reserve,
        "cost_control_enabled": not unlimited,
        "cost_control_mode": (
            "disabled_provider_unlimited"
            if unlimited
            else "finite_provider_allowance"
        ),
        "allowance": allowance,
        "not_before_epoch": allowance.get("resets_at") if not admit else None,
        "local_window": report.get("local_window"),
        "canonical_progress": report.get("canonical_progress"),
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
            "reconstruct retry_complexity_n from settled clean exact-frontier "
            "outcomes; project n through the single versioned medium-high-"
            "xhigh-max allocation/WIP/sidecar table; promotion resets n; "
            "taint, infrastructure, blocker, rate-limit, and containment "
            "outcomes do not increment it"
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
