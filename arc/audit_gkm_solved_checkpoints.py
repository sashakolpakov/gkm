#!/usr/bin/env python3
"""Measure GKM executable complexity only at replay-validated winning sources.

For ordinary proposer wins, ``reached_before_debrief`` is the exact source present
when replay first cleared the level.  Post-win debrief code is excluded.  Auto-solve
wins predate a dedicated pre-debrief snapshot, but their winning source is exactly
reconstructible: the harness starts from the preceding retained source and appends a
deterministic one-call ``play_level_K`` stub before verification.  The successful
called leg is preserved in the subsequent auto-solve artifact.

Failed proposals and every other interim snapshot are excluded.  Missing historical
winning artifacts are reported as gaps, never imputed from the final promoted files.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path

from audit_baseline1_artifacts import canonical_ast_bundle, canonical_bundle


SOURCE_FILES = ("legs.py", "players.py", "solve.py")
RETAINED_PHASE_PRIORITY = (
    "after_debrief",
    "after_auto_solve_debrief",
    "debrief_credit_out",
)


@dataclass
class Checkpoint:
    game: str
    checkpoint: int
    completed_levels: int
    levels_spanned: int
    phase: str
    attempt: str
    created_at: str
    files: int
    raw_bytes: int
    zlib_bytes: int
    ast_zlib_bytes: int
    delta_raw_bytes: int
    delta_zlib_bytes: int
    delta_ast_zlib: int
    positive_delta_zlib: int
    contraction_zlib: int
    historical_marginal_C: int | None
    final_winning_source_matches_promoted: bool | None


def source_files(root: Path) -> dict[str, bytes]:
    return {
        name: (root / name).read_bytes()
        for name in SOURCE_FILES
        if (root / name).is_file()
    }


def snapshot_candidates(
    game_dir: Path, level: int, phases: set[str]
) -> list[tuple[dict, Path]]:
    level_dir = game_dir / "wip_context" / f"level_{level:02d}"
    candidates: list[tuple[dict, Path]] = []
    for metadata_path in level_dir.glob("*/metadata.json"):
        metadata = json.loads(metadata_path.read_text())
        if (
            int(metadata.get("reached", 0)) >= level
            and metadata.get("phase") in phases
        ):
            candidates.append((metadata, metadata_path.parent / "files"))
    return candidates


def latest_snapshot(
    game_dir: Path, level: int, phases: set[str]
) -> tuple[dict, Path] | None:
    candidates = snapshot_candidates(game_dir, level, phases)
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda item: item[0].get("created_at", ""),
    )


def retained_snapshot(game_dir: Path, level: int) -> tuple[dict, Path] | None:
    """Return the source that entered the next level after debrief."""
    for phase in RETAINED_PHASE_PRIORITY:
        candidate = latest_snapshot(game_dir, level, {phase})
        if candidate is not None:
            return candidate
    return None


def auto_solve_leg(players_source: bytes, level: int) -> str | None:
    """Recover the public leg called by the recorded auto-solve player."""
    try:
        tree = ast.parse(players_source)
    except SyntaxError:
        return None
    target = f"play_level_{level}"
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name != target:
            continue
        for child in ast.walk(node):
            if (
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Name)
                and child.func.id != target
            ):
                return child.func.id
    return None


def exact_snapshot(
    game_dir: Path, level: int
) -> tuple[dict, dict[str, bytes]] | None:
    """Return the exact executable sources present at the winning verification."""
    direct = latest_snapshot(game_dir, level, {"reached_before_debrief"})
    if direct is not None:
        metadata, files_root = direct
        return metadata, source_files(files_root)

    auto = latest_snapshot(game_dir, level, {"after_auto_solve_debrief"})
    if auto is None or level <= 1:
        return None
    auto_metadata, auto_files_root = auto
    prior = retained_snapshot(game_dir, level - 1)
    if prior is None:
        return None
    prior_metadata, prior_files_root = prior
    files = source_files(prior_files_root)
    called_leg = auto_solve_leg(
        (auto_files_root / "players.py").read_bytes(), level
    )
    if called_leg is None or "players.py" not in files:
        return None
    stub = f"\n\ndef play_level_{level}(env):\n    {called_leg}(env)\n".encode()
    files["players.py"] += stub
    metadata = {
        **auto_metadata,
        "phase": "reconstructed_auto_solve_boundary",
        "attempt": (
            f"{auto_metadata['attempt']} from {prior_metadata['attempt']} "
            f"via {called_leg}"
        ),
    }
    return metadata, files


def canonical_records(game_dir: Path) -> tuple[dict[int, int], int]:
    data = json.loads((game_dir / "checkpoint.json").read_text())
    records = {
        int(item["level"]): int(item["marginal_C"])
        for item in data.get("records", [])
        if item.get("reached")
    }
    return records, int(data["reached"])


def same_sources(left: dict[str, bytes], right: Path) -> bool:
    return left == source_files(right)


def analyse_game(game_dir: Path) -> tuple[list[Checkpoint], list[int]]:
    game = game_dir.name.removesuffix("_legs")
    records, reached = canonical_records(game_dir)
    selected: list[tuple[int, dict, dict[str, bytes]]] = []
    gaps: list[int] = []
    # Check every claimed level, including old promotion batches for which the root
    # checkpoint no longer carries a per-level row.  Absence is a coverage gap, not
    # permission to silently reduce the denominator.
    for level in range(1, reached + 1):
        candidate = exact_snapshot(game_dir, level)
        if candidate is None:
            gaps.append(level)
        else:
            metadata, files = candidate
            selected.append((level, metadata, files))

    result: list[Checkpoint] = []
    previous_level = previous_raw = previous_zlib = previous_ast = 0
    for index, (level, metadata, files) in enumerate(selected):
        bundle = canonical_bundle(files)
        ast_bundle = canonical_ast_bundle(files)
        compressed = len(zlib.compress(bundle, 9)) if bundle else 0
        ast_compressed = len(zlib.compress(ast_bundle, 9)) if ast_bundle else 0
        result.append(
            Checkpoint(
                game=game,
                checkpoint=index,
                completed_levels=level,
                levels_spanned=level - previous_level,
                phase=str(metadata["phase"]),
                attempt=str(metadata["attempt"]),
                created_at=str(metadata.get("created_at", "")),
                files=len(files),
                raw_bytes=len(bundle),
                zlib_bytes=compressed,
                ast_zlib_bytes=ast_compressed,
                delta_raw_bytes=len(bundle) - previous_raw,
                delta_zlib_bytes=compressed - previous_zlib,
                delta_ast_zlib=ast_compressed - previous_ast,
                positive_delta_zlib=max(0, compressed - previous_zlib),
                contraction_zlib=max(0, previous_zlib - compressed),
                historical_marginal_C=records.get(level),
                final_winning_source_matches_promoted=(
                    same_sources(files, game_dir) if level == reached else None
                ),
            )
        )
        previous_level = level
        previous_raw = len(bundle)
        previous_zlib = compressed
        previous_ast = ast_compressed
    return result, gaps


def sign_reversals(values: list[int]) -> tuple[int, int]:
    differences = [b - a for a, b in zip(values, values[1:])]
    signs = [1 if value > 0 else -1 for value in differences if value]
    return (
        sum(a > 0 and b < 0 for a, b in zip(signs, signs[1:])),
        sum(a < 0 and b > 0 for a, b in zip(signs, signs[1:])),
    )


def summarize(rows: list[Checkpoint], gaps: dict[str, list[int]]) -> dict:
    games = sorted({row.game for row in rows} | set(gaps))
    all_transitions = [row for row in rows if row.checkpoint > 0]
    transitions = [row for row in all_transitions if row.levels_spanned == 1]
    peaks = troughs = 0
    historical_peaks = historical_troughs = 0
    for game in games:
        game_rows = [row for row in rows if row.game == game]
        p, t = sign_reversals([row.positive_delta_zlib for row in game_rows[1:]])
        peaks += p
        troughs += t
        hp, ht = sign_reversals(
            [row.historical_marginal_C or 0 for row in game_rows]
        )
        historical_peaks += hp
        historical_troughs += ht
    return {
        "games": len(games),
        "solved_checkpoints_with_exact_source_snapshot": len(rows),
        "captured_pre_debrief_winning_sources": sum(
            row.phase == "reached_before_debrief" for row in rows
        ),
        "deterministically_reconstructed_auto_solve_winning_sources": sum(
            row.phase == "reconstructed_auto_solve_boundary" for row in rows
        ),
        "auto_solved_checkpoints": [
            {"game": row.game, "completed_level": row.completed_levels}
            for row in rows
            if row.phase == "reconstructed_auto_solve_boundary"
        ],
        "between_available_checkpoint_transitions": len(all_transitions),
        "exact_adjacent_level_transitions": len(transitions),
        "missing_exact_checkpoint_levels": gaps,
        "measurement": "zlib-9 bytes of canonical legs.py + players.py + solve.py",
        "interpretation": "executable-solver description-length upper bound; not K itself",
        "expansions": sum(row.delta_zlib_bytes > 0 for row in transitions),
        "contractions": sum(row.delta_zlib_bytes < 0 for row in transitions),
        "source_and_ast_contractions": sum(
            row.delta_zlib_bytes < 0 and row.delta_ast_zlib < 0 for row in transitions
        ),
        "games_with_contraction": sum(
            any(row.delta_zlib_bytes < 0 for row in rows if row.game == game and row.checkpoint > 0)
            for game in games
        ),
        "local_peaks_in_positive_checkpoint_marginals": peaks,
        "local_troughs_in_positive_checkpoint_marginals": troughs,
        "historical_marginal_C_local_peaks": historical_peaks,
        "historical_marginal_C_local_troughs": historical_troughs,
        "final_winning_sources_matching_post_debrief_promoted_source": sum(
            row.final_winning_source_matches_promoted is True for row in rows
        ),
        "final_winning_sources_changed_by_subsequent_debrief": [
            row.game
            for row in rows
            if row.final_winning_source_matches_promoted is False
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifacts", type=Path)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()

    rows: list[Checkpoint] = []
    gaps: dict[str, list[int]] = {}
    for game_dir in sorted(args.artifacts.glob("*_legs")):
        if not (game_dir / "checkpoint.json").exists():
            continue
        game_rows, game_gaps = analyse_game(game_dir)
        rows.extend(game_rows)
        if game_gaps:
            gaps[game_dir.name.removesuffix("_legs")] = game_gaps
    summary = summarize(rows, gaps)
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0])))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)
    args.json.write_text(json.dumps({"summary": summary, "rows": [asdict(row) for row in rows]}, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
