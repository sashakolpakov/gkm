#!/usr/bin/env python3
"""Execute a guarded post-reset adaptive frontier campaign.

Dry-run is the default.  ``--execute`` never redeems a reset credit, never invokes a
shell, refreshes the live seven-day bucket before each item, and stops immediately on
taint, integrity, allowance, local-budget, or process failure.  The saved plan freezes
only one seed item; all subsequent choices are rebuilt from fresh artifacts.  Medium
gets fresh continuations and high is reserved for one bounded rescue after failure.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import codex_campaign_policy as Policy
import codex_campaign_status as Status
import codex_usage_guard as Guard


HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
DEFAULT_PLAN = HERE / "NEXT_RESET_CAMPAIGN.json"


class CampaignPlanError(RuntimeError):
    pass


def validate_item(
    item: dict[str, Any], plan: dict[str, Any] | None = None
) -> list[str]:
    argv = item.get("argv")
    if not isinstance(argv, list) or not argv or not all(isinstance(x, str) for x in argv):
        raise CampaignPlanError("plan item argv must be a nonempty string list")
    expected = ["python3", "-u", "arc/crack_lab/gkm_legs.py"]
    if argv[:3] != expected:
        raise CampaignPlanError(f"refusing non-GKM command prefix: {argv[:3]!r}")
    if "--proposer=codex" not in argv or "--model=gpt-5.6-sol" not in argv:
        raise CampaignPlanError("plan item must pin the isolated Codex proposer and model")
    if "--debrief-policy=never" not in argv:
        raise CampaignPlanError("cheap campaign items must disable paid debriefs")
    if "--transient-retries=0" not in argv:
        raise CampaignPlanError("cheap campaign items must admit at most one proposal turn")
    if not any(arg.startswith("--codex-weekly-reserve=") for arg in argv):
        raise CampaignPlanError("plan item has no weekly reserve")
    if not any(arg.startswith("--codex-weekly-headroom=") for arg in argv):
        raise CampaignPlanError("plan item has no per-turn weekly headroom")
    if plan is not None:
        reserve = plan.get("reserve_percent")
        headroom = item.get("required_headroom_percent")
        if (
            not isinstance(reserve, int)
            or f"--codex-weekly-reserve={reserve}" not in argv
        ):
            raise CampaignPlanError("command reserve does not match plan reserve")
        if (
            not isinstance(headroom, int)
            or f"--codex-weekly-headroom={headroom}" not in argv
        ):
            raise CampaignPlanError("command headroom does not match item headroom")
    return argv


def item_is_admissible(plan: dict[str, Any], item: dict[str, Any], *,
                       now: float, allowance: Guard.WeeklyAllowance) -> tuple[bool, str]:
    not_before = plan.get("not_before_epoch")
    if isinstance(not_before, int) and now < not_before:
        return False, f"plan is held until weekly reset epoch {not_before}"
    reserve = plan.get("reserve_percent")
    headroom = item.get("required_headroom_percent")
    if not isinstance(reserve, int) or not isinstance(headroom, int):
        return False, "plan has no integer reserve/headroom"
    available = allowance.remaining_percent - reserve
    if allowance.remaining_percent <= reserve or available < headroom:
        return False, (
            f"only {available}% above the {reserve}% reserve; "
            f"item requires {headroom}%"
        )
    return True, "admissible"


def _checkpoint_reached(game: str) -> int:
    path = HERE / "agent_solutions" / f"{game}_legs" / "checkpoint.json"
    if not path.exists():
        return 0
    value = json.loads(path.read_text(encoding="utf-8"))
    reached = value.get("reached")
    return reached if isinstance(reached, int) else 0


def _taint_gate() -> None:
    proc = subprocess.run(
        [sys.executable, "arc/audit_submission_taint.py",
         "arc/crack_lab/agent_solutions"],
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    try:
        result = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise CampaignPlanError("taint gate returned non-JSON output") from exc
    if proc.returncode != 0 or result.get("automated_verdict") != "PASS":
        raise CampaignPlanError("post-turn taint gate failed; campaign stopped")


def _refresh_solver_audits() -> None:
    """Refresh exact GKM checkpoints and the cross-system marginal comparator."""
    commands = [
        [
            sys.executable, "arc/audit_gkm_solved_checkpoints.py",
            "arc/crack_lab/agent_solutions",
            "--csv", "arc/audit_results/gkm-solved-checkpoints.csv",
            "--json", "arc/audit_results/gkm-solved-checkpoints.json",
        ],
        [
            sys.executable, "arc/audit_marginal_literal_reuse.py",
            "--reuse-non-gkm-from-json",
            "arc/audit_results/marginal-literal-reuse.json",
            "--json", "arc/audit_results/marginal-literal-reuse.json",
        ],
    ]
    for argv in commands:
        proc = subprocess.run(
            argv, cwd=REPO, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, check=False,
        )
        if proc.returncode != 0:
            raise CampaignPlanError(
                f"post-turn solver audit failed: {' '.join(argv)}\n{proc.stdout}"
            )


def _run_item(
    plan: dict[str, Any], item: dict[str, Any], *, allowance: Guard.WeeklyAllowance
) -> dict[str, Any]:
    argv = validate_item(item, plan)
    game = item.get("game")
    target = item.get("target_level")
    if not isinstance(game, str) or not isinstance(target, int):
        raise CampaignPlanError("plan item has invalid game or target_level")
    if _checkpoint_reached(game) >= target:
        return {"game": game, "target_level": target, "result": "already_solved"}
    admissible, reason = item_is_admissible(
        plan, item, now=time.time(), allowance=allowance
    )
    if not admissible:
        return {
            "game": game, "target_level": target,
            "result": "reserve_stop", "reason": reason,
        }
    proc = subprocess.run(argv, cwd=REPO, check=False)
    if proc.returncode != 0:
        raise CampaignPlanError(
            f"{game} L{target} process exited {proc.returncode}; campaign stopped"
        )
    _taint_gate()
    reached = _checkpoint_reached(game)
    if reached >= target:
        _refresh_solver_audits()
    return {
        "game": game,
        "target_level": target,
        "reached": reached,
        "result": "solved" if reached >= target else "not_solved",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--max-items", type=int, default=Policy.DEFAULT_MAX_RUNS)
    parser.add_argument("--calibration-only", action="store_true")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    items = plan.get("initial_queue")
    if items is None:
        # Backward compatibility for plans generated before the adaptive queue
        # replaced the completed cold-L1 screen.
        items = plan.get("cold_screen_cohort")
    if not isinstance(items, list):
        raise CampaignPlanError("plan has no initial_queue list")
    for item in items:
        argv = validate_item(item, plan)
        print("DRY" if not args.execute else "QUEUE", item.get("game"), " ".join(argv))
    if not args.execute:
        print("No model turn started; pass --execute after the recorded reset epoch.")
        return 0

    outcomes = []
    for item in items:
        if len(outcomes) >= args.max_items:
            break
        allowance = Guard.weekly_allowance(Guard.query_rate_limits())
        outcome = _run_item(plan, item, allowance=allowance)
        outcomes.append(outcome)
        if outcome["result"] == "reserve_stop":
            print(json.dumps({"outcomes": outcomes}, indent=2, sort_keys=True))
            return 0

    while not args.calibration_only and len(outcomes) < args.max_items:
        snapshot = Guard.query_rate_limits()
        allowance = Guard.weekly_allowance(snapshot)
        report = Status.campaign_report(
            live_snapshot=snapshot,
            reserve=int(plan["reserve_percent"]),
            medium_headroom=5,
            high_headroom=6,
            max_runs=Policy.DEFAULT_MAX_RUNS,
            max_tokens=Policy.DEFAULT_MAX_TOKENS,
        )
        if not report["readiness"]["local_budget_ok"]:
            outcomes.append({
                "result": "local_budget_stop",
                "local_window": report["local_window"],
            })
            break
        item = Policy.adaptive_campaign_item(
            report, reserve=int(plan["reserve_percent"])
        )
        if item is None:
            outcomes.append({
                "result": "adaptation_stop",
                "reason": "matched evidence or remaining frontier unavailable",
            })
            break
        print("ADAPT", item["game"], " ".join(validate_item(item, plan)))
        outcome = _run_item(plan, item, allowance=allowance)
        outcomes.append(outcome)
        if outcome["result"] == "reserve_stop":
            break
    print(json.dumps({"outcomes": outcomes}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
