#!/usr/bin/env python3
"""Drive a budgeted Claude proposer mini-campaign over ranked frontiers.

Sweeps the (provider-agnostic) frontier ranking, running one guarded Claude Code
proposer turn per game to advance it as far as it can within the per-level wall-time
budget, and stops as soon as the Claude subscription window credit-outs or a local cap
is reached.  The Claude subscription exposes no readable remaining allowance, so the
real stop is reactive credit-out; the local caps (turns / wall-minutes) are a backstop.

Dry-run by default; pass --execute to spend allowance.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import codex_campaign_status as Status
import claude_usage_guard as CLG

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent


def ranked_frontiers(limit: int) -> list[dict]:
    report = Status.campaign_report()  # non-metered; reads checkpoints + audits
    rows = [
        row for row in report.get("frontiers", [])
        if not row.get("quarantined_after_escalation_failure")
    ]
    return rows[:limit]


def reached(game: str) -> int:
    path = HERE / "agent_solutions" / f"{game}_legs" / "checkpoint.json"
    if not path.exists():
        return 0
    value = json.loads(path.read_text(encoding="utf-8"))
    return value.get("reached", 0) if isinstance(value.get("reached"), int) else 0


def window_exhausted(ledger: str) -> bool:
    """True once the Claude allowance is spent for this window.

    Primary signal is the recorded ``credit_out`` flag.  Defense in depth: an
    instant, empty, non-timed-out turn is almost always a provider rejection
    (spend/usage limit) whose exact message we did not recognize, so stop rather
    than churn free no-op turns through the rest of the sweep.
    """
    claude = [r for r in CLG.read_ledger(ledger) if r.get("event") == "claude_exec"]
    if not claude:
        return False
    last = claude[-1]
    if last.get("credit_out"):
        return True
    duration = last.get("duration_seconds") or 0
    tokens = last.get("output_tokens")
    return duration < 15 and tokens in (0, None) and not last.get("timed_out")


class CampaignSafetyError(RuntimeError):
    """A promoted artifact is tainted, or a post-turn audit failed."""


def _taint_gate() -> None:
    """Uniform with codex_campaign_runner: refuse to continue if any promoted
    artifact shows forbidden source/history/runtime access -- a weaselled solve.
    Weaker proposers cheat when a task is too hard; this catches it every turn.
    """
    proc = subprocess.run(
        [sys.executable, "arc/audit_submission_taint.py",
         "arc/crack_lab/agent_solutions"],
        cwd=REPO, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        check=False,
    )
    try:
        result = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise CampaignSafetyError("taint gate returned non-JSON output") from exc
    if proc.returncode != 0 or result.get("automated_verdict") != "PASS":
        raise CampaignSafetyError(
            f"post-turn taint gate FAILED (verdict={result.get('automated_verdict')}); "
            "a run is tainted"
        )


def _refresh_solver_audits() -> None:
    """Refresh exact-checkpoint and marginal/reuse audits after a promotion."""
    for argv in (
        [sys.executable, "arc/audit_gkm_solved_checkpoints.py",
         "arc/crack_lab/agent_solutions",
         "--csv", "arc/audit_results/gkm-solved-checkpoints.csv",
         "--json", "arc/audit_results/gkm-solved-checkpoints.json"],
        [sys.executable, "arc/audit_marginal_literal_reuse.py",
         "--reuse-non-gkm-from-json", "arc/audit_results/marginal-literal-reuse.json",
         "--json", "arc/audit_results/marginal-literal-reuse.json"],
    ):
        proc = subprocess.run(argv, cwd=REPO, text=True, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, check=False)
        if proc.returncode != 0:
            raise CampaignSafetyError(
                f"post-win solver audit failed: {' '.join(argv)}\n{proc.stdout}")


def _argv(game: str, target: int, args) -> list[str]:
    return [
        sys.executable, "-u", "arc/crack_lab/gkm_legs.py",
        f"--game={game}", f"--max-level={target}",
        "--proposer=claude", f"--model={args.model}", f"--minutes={args.minutes}",
        "--debrief-policy=never", "--transient-retries=0",
        "--claude-guard", f"--claude-ledger={args.ledger}",
        f"--claude-window-hours={args.window_hours}",
        f"--claude-max-turns={args.max_turns}",
        f"--claude-max-wall-minutes={args.max_wall_minutes}",
        "--tag=claude_campaign",
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--model", default="opus")
    ap.add_argument("--minutes", type=int, default=12)
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--depth", type=int, default=2,
                    help="max levels to try to advance per game this sweep")
    ap.add_argument("--skip", default="", help="comma-separated games to skip")
    ap.add_argument("--window-hours", type=float, default=5.0)
    ap.add_argument("--max-turns", type=int, default=30)
    ap.add_argument("--max-wall-minutes", type=float, default=280.0)
    ap.add_argument("--ledger", default=str(CLG.DEFAULT_LEDGER))
    ap.add_argument("--fresh-window", action="store_true",
                    help="operator confirms the allowance window has reset; bypass a "
                         "stale trailing credit-out on the first turn")
    args = ap.parse_args()
    skip = {x for x in args.skip.split(",") if x}

    plan = []
    for row in ranked_frontiers(args.limit):
        game = row["game"]
        if game in skip:
            continue
        before = reached(game)
        plan.append((game, before, before + args.depth, row.get("recommended_effort")))

    print("=== Claude mini-campaign plan (ranked frontiers) ===", flush=True)
    for game, before, target, eff in plan:
        print(f"  {game}: L{before} -> attempt L{before+1}..L{target}", flush=True)
    if not args.execute:
        print("\nDry run. Pass --execute to spend the Claude window.", flush=True)
        return 0

    outcomes = []
    fresh = args.fresh_window
    for game, before, target, _eff in plan:
        if not fresh and window_exhausted(args.ledger):
            print("CREDIT-OUT detected before turn; window exhausted. Stopping.", flush=True)
            break
        fresh = False  # the operator's fresh-window assertion covers only the first turn
        print(f"=== {game}: L{before} -> attempt L{before+1}..L{target} ===", flush=True)
        try:
            proc = subprocess.run(_argv(game, target, args), cwd=REPO)
            rc = proc.returncode
        except Exception as exc:  # a single game crash must not kill the sweep
            rc = -1
            print(f"[warning: {game} run raised {type(exc).__name__}: {exc}]", flush=True)
        # Taint gate every turn (uniform with codex_campaign_runner): a weaselled,
        # source/history-reading "solve" stops the campaign immediately for review.
        try:
            _taint_gate()
        except CampaignSafetyError as exc:
            after = reached(game)
            outcomes.append({"game": game, "before": before, "after": after,
                             "result": "taint_stop", "detail": str(exc)})
            print(f"!!! TAINT GATE FAILED after {game}: {exc}; STOPPING for review.",
                  flush=True)
            break
        after = reached(game)
        gained = after - before
        if gained > 0:
            _refresh_solver_audits()
        outcomes.append({"game": game, "before": before, "after": after,
                         "gained": gained, "returncode": rc})
        marker = f"+{gained} level(s)" if gained > 0 else "no gain"
        print(f"=== {game}: reached L{after} ({marker}, rc={rc}) ===", flush=True)
        if window_exhausted(args.ledger):
            print("CREDIT-OUT (Claude window exhausted); stopping campaign.", flush=True)
            break

    total = sum(o["gained"] for o in outcomes)
    print(json.dumps({"levels_gained": total, "outcomes": outcomes}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
