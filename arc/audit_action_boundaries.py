#!/usr/bin/env python3
"""Replay-audit that every promoted checkpoint ends at its exact level boundary.

This is distinct from the executable-source boundary audit: a source can be the
right winning source while its recorded action path still contains post-win
actions.  Such a path is unsafe as a resumable parent because it can consume the
next level's real-move budget before the resumed solver runs.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Sequence


LAB = Path(__file__).resolve().parent / "crack_lab"
if str(LAB) not in sys.path:
    sys.path.insert(0, str(LAB))

import gkm_legs as G  # noqa: E402


@dataclass
class BoundaryResult:
    path: str
    game: str
    level: int
    recorded_actions: int
    exact_actions: int | None
    kind: str
    verdict: str


def checkpoint_files(root: Path) -> list[tuple[str, Path]]:
    current = [("current", path) for path in root.glob("*_legs/checkpoint.json")]
    historical = [
        ("promotion_evidence", path)
        for path in root.glob(
            "*_legs/promotion_evidence/level_*/files/checkpoint.json"
        )
    ]
    return sorted(current + historical, key=lambda item: str(item[1]))


def audit_checkpoint(
    root: Path,
    kind: str,
    checkpoint: Path,
    boundary_fn: Callable[[str, Sequence, int], list | None] = G.exact_level_boundary,
) -> BoundaryResult:
    data = json.loads(checkpoint.read_text())
    game = str(data["game"])
    level = int(data["reached"])
    action_path = data.get("final_path") or []
    boundary = boundary_fn(game, action_path, level)
    exact = None if boundary is None else len(boundary)
    if boundary is None:
        verdict = "UNREPRODUCED"
    elif len(boundary) != len(action_path):
        verdict = "OVERLONG"
    else:
        verdict = "PASS"
    return BoundaryResult(
        path=str(checkpoint.relative_to(root)),
        game=game,
        level=level,
        recorded_actions=len(action_path),
        exact_actions=exact,
        kind=kind,
        verdict=verdict,
    )


def _run_in_process(
    root: Path,
    games: set[str] | None = None,
    *,
    require_complete_chain: bool = False,
) -> dict:
    """Replay selected checkpoints in this process.

    The public Arena/runtime retains enough per-replay state that auditing the
    entire archive in one long-lived process can be killed by the OS.  This
    worker is therefore normally invoked once per game by :func:`run`.
    """
    selected = checkpoint_files(root)
    if games is not None:
        selected = [
            (kind, checkpoint)
            for kind, checkpoint in selected
            if checkpoint.relative_to(root).parts[0].removesuffix("_legs")
            in games
        ]
    results = [
        audit_checkpoint(root, kind, checkpoint)
        for kind, checkpoint in selected
    ]
    return _report(
        root,
        results,
        require_complete_chain=require_complete_chain,
    )


def _promotion_chain_completeness(
    results: list[BoundaryResult],
) -> dict:
    current_by_game: dict[str, list[BoundaryResult]] = {}
    promotion_by_game: dict[str, dict[int, list[BoundaryResult]]] = {}
    issues: list[dict] = []
    for row in results:
        if row.kind == "current":
            current_by_game.setdefault(row.game, []).append(row)
            continue
        if row.kind != "promotion_evidence":
            continue
        path_level = None
        for part in Path(row.path).parts:
            if part.startswith("level_"):
                try:
                    path_level = int(part.removeprefix("level_"))
                except ValueError:
                    pass
                break
        if path_level != row.level:
            issues.append({
                "game": row.game,
                "level": row.level,
                "path": row.path,
                "kind": "promotion_path_level_mismatch",
                "expected_path_level": path_level,
            })
        promotion_by_game.setdefault(row.game, {}).setdefault(
            row.level, []
        ).append(row)

    expected = 0
    present = 0
    missing: list[dict] = []
    for game in sorted(set(current_by_game) | set(promotion_by_game)):
        currents = current_by_game.get(game, [])
        if len(currents) != 1:
            issues.append({
                "game": game,
                "kind": "current_checkpoint_count",
                "found": len(currents),
                "expected": 1,
            })
            continue
        reached = currents[0].level
        expected += reached
        promotions = promotion_by_game.get(game, {})
        for level in range(1, reached + 1):
            rows = promotions.get(level, [])
            if len(rows) == 1:
                present += 1
            elif not rows:
                missing.append({
                    "game": game,
                    "level": level,
                    "expected_path": (
                        f"{game}_legs/promotion_evidence/"
                        f"level_{level:02d}/files/checkpoint.json"
                    ),
                })
            else:
                issues.append({
                    "game": game,
                    "level": level,
                    "kind": "duplicate_promotion_checkpoint",
                    "found": len(rows),
                })
        for level in sorted(set(promotions) - set(range(1, reached + 1))):
            issues.append({
                "game": game,
                "level": level,
                "kind": "promotion_beyond_current_checkpoint",
                "current_reached": reached,
            })
    return {
        "expected": expected,
        "present": present,
        "missing": missing,
        "issues": issues,
        "complete": not missing and not issues,
    }


def _report(
    root: Path,
    results: list[BoundaryResult],
    *,
    require_complete_chain: bool = False,
) -> dict:
    issues = [row for row in results if row.verdict != "PASS"]
    chain = _promotion_chain_completeness(results)
    failed = bool(issues) or (
        require_complete_chain and not chain["complete"]
    )
    return {
        "schema": 2,
        "root": str(root),
        "checkpoints": len(results),
        "exact": len(results) - len(issues),
        "issues": [asdict(row) for row in issues],
        "results": [asdict(row) for row in results],
        "complete_chain_required": require_complete_chain,
        "promotion_chain": chain,
        "verdict": "FAIL" if failed else "PASS",
    }


def _checkpoint_games(root: Path) -> list[str]:
    games = set()
    for _kind, checkpoint in checkpoint_files(root):
        try:
            game = json.loads(checkpoint.read_text()).get("game")
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(game, str) and game:
            games.add(game)
    return sorted(games)


def _isolated_game_results(root: Path, game: str) -> list[BoundaryResult]:
    proc = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            str(root),
            "--game",
            game,
            "--in-process",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    try:
        child = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"boundary worker for {game} returned no JSON "
            f"(exit={proc.returncode}): {proc.stderr[-2000:]}"
        ) from exc
    if proc.returncode not in (0, 1):
        raise RuntimeError(
            f"boundary worker for {game} failed "
            f"(exit={proc.returncode}): {proc.stderr[-2000:]}"
        )
    return [BoundaryResult(**row) for row in child["results"]]


def run(
    root: Path,
    *,
    isolate_games: bool = True,
    games: set[str] | None = None,
    require_complete_chain: bool = False,
) -> dict:
    """Replay selected checkpoints, isolating every game's runtime by default."""
    if not isolate_games:
        return _run_in_process(
            root,
            games,
            require_complete_chain=require_complete_chain,
        )
    results: list[BoundaryResult] = []
    available = set(_checkpoint_games(root))
    if games is not None:
        missing = sorted(games - available)
        if missing:
            raise ValueError(
                f"requested games have no checkpoint evidence: {missing}"
            )
        selected_games = sorted(games)
    else:
        selected_games = sorted(available)
    # Six short-lived workers bound both accumulated runtime state and total
    # wall time.  Games remain independent, and results are sorted below so the
    # report is deterministic regardless of completion order.
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(6, max(1, len(selected_games)))
    ) as pool:
        futures = {
            pool.submit(_isolated_game_results, root, game): game
            for game in selected_games
        }
        for future in concurrent.futures.as_completed(futures):
            game = futures[future]
            print(f"boundary worker complete: {game}", file=sys.stderr, flush=True)
            results.extend(future.result())
    results.sort(key=lambda row: row.path)
    return _report(
        root,
        results,
        require_complete_chain=require_complete_chain,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "root",
        nargs="?",
        type=Path,
        default=LAB / "agent_solutions",
    )
    parser.add_argument("--json", type=Path)
    parser.add_argument(
        "--game",
        action="append",
        default=[],
        help="internal/debug filter; repeat to audit more than one game",
    )
    parser.add_argument(
        "--in-process",
        action="store_true",
        help="disable per-game process isolation (used by child workers)",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="print only counts/issues while retaining full rows in --json",
    )
    parser.add_argument(
        "--require-complete-chain",
        action="store_true",
        help=(
            "fail unless every level 1..current reached has exactly one "
            "promotion-evidence checkpoint"
        ),
    )
    args = parser.parse_args()
    # Preserve the caller-supplied root spelling in the report so repository
    # reproduction does not hash machine-specific absolute checkout paths.
    selected_games = set(args.game) if args.game else None
    report = (
        _run_in_process(
            args.root,
            selected_games,
            require_complete_chain=args.require_complete_chain,
        )
        if args.in_process
        else run(
            args.root,
            games=selected_games,
            require_complete_chain=args.require_complete_chain,
        )
    )
    displayed = report
    if args.summary_only:
        displayed = {
            key: report[key]
            for key in (
                "schema",
                "root",
                "checkpoints",
                "exact",
                "issues",
                "complete_chain_required",
                "promotion_chain",
                "verdict",
            )
        }
    rendered = json.dumps(displayed, indent=2, sort_keys=True)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(rendered)
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
