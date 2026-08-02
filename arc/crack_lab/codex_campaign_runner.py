#!/usr/bin/env python3
"""Execute the guarded exact-frontier ARC-AGI-3 compatibility campaign.

Dry-run is the default.  Every dispatch is rederived from the live authoritative
checkpoint and journal-reconstructed retry coordinate, then checked against the
single medium→high→xhigh→max policy before process launch.  Explicit provider
``unlimited`` disables cost cutoffs but never correctness, isolation, taint,
replay, provenance, or containment controls.
"""

from __future__ import annotations

import argparse
import fcntl
import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import codex_campaign_policy as Policy
import codex_campaign_status as Status
import codex_usage_guard as Guard
import arc_agi3_contiguous_supervisor as Contiguous
import gkm_legs as Legs


HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
DEFAULT_PLAN = HERE / "ARC_AGI3_CAMPAIGN_QUEUE.json"


class CampaignPlanError(RuntimeError):
    pass


def _effective_retry_inputs(
    item: dict[str, Any], policy: dict[str, Any]
) -> tuple[str, str]:
    """Return the only admissible effective WIP/dispatch projection.

    Clean infrastructure interruption is not a solver retry.  It may override
    a policy reset only when the plan pins the sealed same-frontier capsule and
    advertises the matching infrastructure phase.  Effort, allocation, retry
    coordinate, and auxiliary policy remain those of the versioned retry row.
    """
    recovery = item.get("warm_wip_recovery_required") is True
    if not recovery:
        return str(policy["wip_mode"]), str(policy["dispatch_mode"])
    attempt = item.get("expected_wip_attempt")
    phase = item.get("warm_wip_phase")
    if (
        item.get("warm_wip_available") is not True
        or not isinstance(attempt, str)
        or not attempt
        or Path(attempt).name != attempt
        or phase not in Status.INFRASTRUCTURE_WIP_PHASES
    ):
        raise CampaignPlanError(
            "infrastructure recovery lacks one sealed exact-frontier capsule"
        )
    return (
        "restore_clean_same_frontier",
        "recover_clean_infrastructure_wip",
    )


def _authoritative_targets() -> dict[str, int]:
    targets = Status._authoritative_inventory()
    if len(targets) != 25 or sum(targets.values()) != 183:
        raise CampaignPlanError(
            "authoritative inventory gate failed: expected 25 games / "
            f"183 levels, found {len(targets)} / {sum(targets.values())}"
        )
    return targets


def validate_inventory_item(
    item: dict[str, Any], targets: dict[str, int], reached: int
) -> None:
    game = item.get("game")
    target = item.get("target_level")
    if (
        not isinstance(game, str)
        or not game
        or not isinstance(target, int)
        or isinstance(target, bool)
        or target <= 0
        or not isinstance(reached, int)
        or isinstance(reached, bool)
        or reached < 0
    ):
        raise CampaignPlanError("plan item has invalid game or target_level")
    authoritative = targets.get(game)
    if authoritative is None:
        raise CampaignPlanError(f"game is absent from authoritative inventory: {game}")
    if reached > authoritative:
        raise CampaignPlanError(
            f"checkpoint exceeds authoritative target: {game} "
            f"{reached}/{authoritative}"
        )
    if target > authoritative:
        raise CampaignPlanError(
            f"refusing nonexistent level: {game} L{target}; "
            f"authoritative target is {authoritative}"
        )
    if reached < target and target != reached + 1:
        raise CampaignPlanError(
            f"refusing nonsequential target: {game} reached={reached}, "
            f"requested L{target}"
        )
    if item.get("reached") != reached:
        raise CampaignPlanError(
            "plan item exact-parent reached value is stale"
        )
    seed_mode = item.get("seed_mode")
    expected_seed = "verified_parent" if reached > 0 else "zero_seed"
    if seed_mode != expected_seed:
        raise CampaignPlanError(
            f"lineage seed mismatch: {game} reached={reached} requires "
            f"{expected_seed}, item requested {seed_mode!r}"
        )
    wip_mode = item.get("wip_mode")
    if wip_mode not in {"exclude", "restore_clean_same_frontier"}:
        raise CampaignPlanError(f"invalid WIP mode: {wip_mode!r}")
    if (
        wip_mode == "restore_clean_same_frontier"
        and item.get("warm_wip_available") is not True
    ):
        raise CampaignPlanError(
            "WIP restore requested without a recorded clean same-frontier snapshot"
        )


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
    if "--codex-allocation-policy=drain" not in argv:
        raise CampaignPlanError(
            "plan item must use the non-interrupting drain allocation policy"
        )
    if "--debrief-policy=never" not in argv:
        raise CampaignPlanError("campaign items must disable extra debrief turns")
    if "--transient-retries=0" not in argv:
        raise CampaignPlanError("budgeted campaign items must admit at most one proposal turn")
    game = item.get("game")
    target = item.get("target_level")
    if not isinstance(game, str) or not game or f"--game={game}" not in argv:
        raise CampaignPlanError("command game does not match plan item")
    if (
        not isinstance(target, int)
        or isinstance(target, bool)
        or target <= 0
        or f"--max-level={target}" not in argv
    ):
        raise CampaignPlanError("command max-level does not match plan target")
    try:
        binding = Status.validate_frontier_binding({
            field: item.get(field)
            for field in (
                *Status.FRONTIER_BINDING_FIELDS,
                "game",
                "reached",
                "target_level",
                "parent_action_count",
            )
        })
    except ValueError as exc:
        raise CampaignPlanError(
            f"plan item has invalid exact-frontier binding: {exc}"
        ) from exc
    required_binding_args = {
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
    }
    missing_binding_args = sorted(required_binding_args - set(argv))
    if missing_binding_args:
        raise CampaignPlanError(
            "command does not consume its exact-frontier binding: "
            f"{missing_binding_args}"
        )
    seed_mode = item.get("seed_mode")
    wip_mode = item.get("wip_mode")
    if seed_mode not in {"zero_seed", "verified_parent"}:
        raise CampaignPlanError(f"invalid lineage seed mode: {seed_mode!r}")
    if wip_mode not in {"exclude", "restore_clean_same_frontier"}:
        raise CampaignPlanError(f"invalid lineage WIP mode: {wip_mode!r}")
    if f"--seed-mode={seed_mode}" not in argv:
        raise CampaignPlanError("command seed mode does not match item")
    if f"--wip-mode={wip_mode}" not in argv:
        raise CampaignPlanError("command WIP mode does not match item")
    expected_wip_attempt = item.get("expected_wip_attempt")
    expected_wip_args = [
        argument for argument in argv
        if argument.startswith("--expected-wip-attempt=")
    ]
    if wip_mode == "restore_clean_same_frontier":
        if (
            not isinstance(expected_wip_attempt, str)
            or not expected_wip_attempt
            or Path(expected_wip_attempt).name != expected_wip_attempt
            or expected_wip_args != [
                f"--expected-wip-attempt={expected_wip_attempt}"
            ]
        ):
            raise CampaignPlanError(
                "WIP restore does not pin one scheduler-selected capsule"
            )
    elif expected_wip_attempt is not None or expected_wip_args:
        raise CampaignPlanError(
            "excluded WIP item carries an unexpected capsule selector"
        )
    expected_composite = f"{seed_mode}+{wip_mode}"
    if item.get("lineage_input_mode") != expected_composite:
        raise CampaignPlanError("composite lineage input mode does not match item")
    if not any(arg.startswith("--codex-weekly-reserve=") for arg in argv):
        raise CampaignPlanError("plan item has no weekly reserve")
    if not any(arg.startswith("--codex-weekly-headroom=") for arg in argv):
        raise CampaignPlanError("plan item has no per-turn weekly headroom")
    n = item.get("retry_complexity_n")
    if not isinstance(n, int) or isinstance(n, bool) or n < 0:
        raise CampaignPlanError(
            "plan item has no valid retry_complexity_n"
        )
    policy = Status.retry_policy(n)
    effective_wip, effective_dispatch = _effective_retry_inputs(item, policy)
    expected_fields = {
        "effort": policy["effort"],
        "minutes": policy["minutes"],
        "wip_mode": effective_wip,
        "dispatch_mode": effective_dispatch,
        "recommended_auxiliary_parallelism": policy[
            "auxiliary_parallelism"
        ],
    }
    for key, expected_value in expected_fields.items():
        if item.get(key) != expected_value:
            raise CampaignPlanError(
                f"plan item {key} does not match retry policy"
            )
    if f"--codex-effort={policy['effort']}" not in argv:
        raise CampaignPlanError(
            "command effort does not match retry policy"
        )
    if f"--minutes={policy['minutes']}" not in argv:
        raise CampaignPlanError(
            "command allocation does not match retry policy"
        )
    cost_control_enabled = item.get("cost_control_enabled")
    if not isinstance(cost_control_enabled, bool):
        raise CampaignPlanError(
            "plan item has no explicit cost-control mode"
        )
    max_runs = item.get("max_campaign_runs")
    max_tokens = item.get("max_campaign_tokens")
    if (
        not isinstance(max_runs, int)
        or isinstance(max_runs, bool)
        or not isinstance(max_tokens, int)
        or isinstance(max_tokens, bool)
        or f"--codex-max-campaign-runs={max_runs}" not in argv
        or f"--codex-max-campaign-tokens={max_tokens}" not in argv
    ):
        raise CampaignPlanError(
            "command local cost caps do not match plan item"
        )
    if not cost_control_enabled and (max_runs != -1 or max_tokens != -1):
        raise CampaignPlanError(
            "unlimited item retains a local run or token cutoff"
        )
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
        plan_cost_control = plan.get("cost_control_enabled")
        if (
            not isinstance(plan_cost_control, bool)
            or plan_cost_control != cost_control_enabled
        ):
            raise CampaignPlanError(
                "item cost-control mode does not match plan"
            )
    return argv


def item_is_admissible(plan: dict[str, Any], item: dict[str, Any], *,
                       now: float, allowance: Guard.WeeklyAllowance) -> tuple[bool, str]:
    if getattr(allowance, "window_name", None) == "unlimited":
        if item.get("cost_control_enabled") is not False:
            return False, "provider is unlimited but item enables cost controls"
        return True, "admissible: provider pool is unlimited"
    if item.get("cost_control_enabled") is not True:
        return False, "finite or unknown provider limit requires cost controls"
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


def active_workspace_lock(game: str) -> Path | None:
    """Return an actively locked tagged workspace for ``game``, if any."""
    pattern = os.fspath(HERE / "runs" / "scratch" / f"gkm_legs_ws_{game}*")
    for workspace in sorted(glob.glob(pattern)):
        path = Path(workspace) / ".orchestrate.lock"
        if not path.is_file():
            continue
        try:
            lock = Legs._open_unaliased_lock(os.fspath(path), create=False)
        except RuntimeError as exc:
            raise CampaignPlanError(
                f"unsafe workspace lock path: {path}"
            ) from exc
        try:
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return path
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        finally:
            lock.close()
    return None


def _checkpoint_reached(game: str) -> int:
    path = HERE / "agent_solutions" / f"{game}_legs" / "checkpoint.json"
    if not path.exists():
        return 0
    targets = _authoritative_targets()
    target = targets.get(game)
    if target is None:
        raise CampaignPlanError(
            f"game is absent from authoritative inventory: {game}"
        )
    try:
        checkpoint = Contiguous.load_trusted_checkpoint(
            path,
            expected_game=game,
            authoritative_target=target,
        )
    except Contiguous.SupervisorContractError as exc:
        raise CampaignPlanError(
            f"refusing malformed or untrusted checkpoint for {game}: {exc}"
        ) from exc
    return checkpoint.reached


def validate_live_policy_item(item: dict[str, Any]) -> None:
    """Reject a queue item whose exact frontier or retry row has gone stale."""

    game = item.get("game")
    target = item.get("target_level")
    if not isinstance(game, str) or not isinstance(target, int):
        raise CampaignPlanError("plan item has invalid game or target_level")
    report = Status.campaign_report(
        reserve=0,
        medium_headroom=1,
        high_headroom=1,
        max_runs=-1,
        max_tokens=-1,
    )
    matches = [
        row
        for row in report.get("frontiers", [])
        if row.get("game") == game and row.get("next_level") == target
    ]
    if len(matches) != 1:
        raise CampaignPlanError(
            "plan item is not the unique live exact frontier"
        )
    row = matches[0]
    for key in (
        *Status.FRONTIER_BINDING_FIELDS,
        "reached",
        "parent_action_count",
    ):
        if item.get(key) != row.get(key):
            raise CampaignPlanError(
                f"plan item exact-frontier field {key} is stale"
            )
    n = row.get("retry_complexity_n")
    if not isinstance(n, int) or isinstance(n, bool) or n < 0:
        raise CampaignPlanError(
            "live frontier has no valid retry coordinate"
        )
    if item.get("retry_complexity_n") != n:
        raise CampaignPlanError(
            "plan item retry coordinate is stale"
        )
    policy = Status.retry_policy(n)
    effective_wip, effective_dispatch = _effective_retry_inputs(item, policy)
    comparisons = {
        "effort": policy["effort"],
        "minutes": policy["minutes"],
        "wip_mode": effective_wip,
        "dispatch_mode": effective_dispatch,
        "recommended_auxiliary_parallelism": policy[
            "auxiliary_parallelism"
        ],
    }
    for key, expected in comparisons.items():
        if item.get(key) != expected:
            raise CampaignPlanError(
                f"plan item {key} is stale at the live frontier"
            )
    if item.get("warm_wip_available") != bool(
        row.get("warm_wip_available")
    ):
        raise CampaignPlanError(
            "plan item WIP availability is stale"
        )
    # A reset lane deliberately excludes the latest same-frontier WIP capsule,
    # so its plan carries no ``expected_wip_attempt`` selector even when an
    # eligible capsule exists.  Capsule identity is live policy state only for
    # a restore lane; requiring it for an exclude lane makes every reset after
    # a clean no-progress turn impossible to launch.  Availability, phase, and
    # infrastructure-recovery status remain live-checked in both modes.
    if effective_wip == "restore_clean_same_frontier":
        if item.get("expected_wip_attempt") != row.get("warm_wip_attempt"):
            raise CampaignPlanError(
                "plan item warm_wip_attempt is stale at the live frontier"
            )
    for key in ("warm_wip_phase", "warm_wip_recovery_required"):
        if item.get(key) != row.get(key):
            raise CampaignPlanError(
                f"plan item {key} is stale at the live frontier"
            )


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
    reached_before = _checkpoint_reached(game)
    validate_inventory_item(item, _authoritative_targets(), reached_before)
    if reached_before >= target:
        return {
            "game": game, "target_level": target, "result": "already_solved",
            "seed_mode": item["seed_mode"], "wip_mode": item["wip_mode"],
            "lineage_input_mode": item["lineage_input_mode"],
        }
    validate_live_policy_item(item)
    active_lock = active_workspace_lock(game)
    if active_lock is not None:
        raise CampaignPlanError(
            f"refusing duplicate active game lineage for {game}: {active_lock}"
        )
    admissible, reason = item_is_admissible(
        plan, item, now=time.time(), allowance=allowance
    )
    if not admissible:
        return {
            "game": game, "target_level": target,
            "result": "reserve_stop", "reason": reason,
            "seed_mode": item["seed_mode"], "wip_mode": item["wip_mode"],
            "lineage_input_mode": item["lineage_input_mode"],
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
        "seed_mode": item["seed_mode"],
        "wip_mode": item["wip_mode"],
        "lineage_input_mode": item["lineage_input_mode"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--max-items", type=int, default=Policy.DEFAULT_MAX_RUNS)
    parser.add_argument("--calibration-only", action="store_true")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    targets = _authoritative_targets()
    items = plan.get("initial_queue")
    if items is None:
        # Backward compatibility for plans generated before the adaptive queue
        # replaced the completed cold-L1 screen.
        items = plan.get("cold_screen_cohort")
    if not isinstance(items, list):
        raise CampaignPlanError("plan has no initial_queue list")
    for item in items:
        argv = validate_item(item, plan)
        game = item.get("game")
        reached = _checkpoint_reached(game) if isinstance(game, str) else 0
        validate_inventory_item(item, targets, reached)
        # A dry run is the operator's review surface for the command that may
        # subsequently be executed.  It must therefore reject a queue frozen
        # at an older retry coordinate just as strictly as ``_run_item`` does;
        # printing a stale command as "DRY" is misleading even though the
        # execution path would later fail closed.
        validate_live_policy_item(item)
        print("DRY" if not args.execute else "QUEUE", item.get("game"), " ".join(argv))
    if not args.execute:
        print(
            "No model turn started; pass --execute only after reviewing the "
            "fresh policy-derived queue."
        )
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
