#!/usr/bin/env python3
"""Measure solved-boundary source growth in published baseline1 workspaces.

The secure baseline1 release preserves an independent Git repository for every game.
Its agent commits after inspecting the current game state.  This program retains the
first commit after an observed level advance, plus a final worktree only when the
scorecard proves it solved additional levels.  Failed/interim commits are excluded.

Those Git objects are *post-solve retained snapshots*, not automatically the exact
programs present at the winning action: the prompt permits deliverable work after a
win and before the next commit.  We therefore align each retained snapshot with the
winning command in ``agent.log`` and mark it exact only when no Python file changed
between that command and the end of the Codex turn.  A level-to-level contraction is
called exact only when both adjacent endpoints pass that test.

Two predeclared source profiles are measured:

``core``
    The three executable world-model modules named by the supplied scaffold.

``authored``
    Every Python file newly created or changed from the supplied scaffold, excluding
    caches and the observation client.  This includes the core, level planners, and
    game-specific reconstruction/controller code.

For each profile, a canonical filename/content bundle is compressed with zlib.  Its
length is a computable description-length upper bound, not Kolmogorov complexity
itself.  Signed changes can establish real source contraction; log or action length
is deliberately absent from the measurement.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import io
import json
import re
import subprocess
import tokenize
import zlib
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


CORE = {
    "world_model_engine.py",
    "world_model_main_planner.py",
    "world_model_state_io.py",
}
LEVEL_RE = re.compile(r"^level_(\d+)\b")
SKIP_PARTS = {".git", "__pycache__", "client", ".pytest_cache"}


@dataclass
class Row:
    game: str
    checkpoint: int
    revision: str
    subject: str
    reached_level: int | None
    completed_levels: int | None
    levels_spanned: int
    final_worktree: bool
    win_log_line: int | None
    post_win_python_files: str
    exact_winning_source: bool
    exact_adjacent_transition: bool
    profile: str
    files: int
    retained_top_level_definitions: int
    retained_core_top_level_definitions: int
    raw_bytes: int
    zlib_bytes: int
    python_tokens: int
    ast_zlib_bytes: int
    delta_raw_bytes: int
    delta_zlib: int
    delta_python_tokens: int
    delta_ast_zlib: int
    positive_delta_zlib: int
    contraction_zlib: int
    contraction_ast_zlib: int


@dataclass(frozen=True)
class WinBoundary:
    log_line: int
    post_win_python_files: tuple[str, ...]


def git(repo: Path, *args: str, check: bool = True) -> bytes:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if check and result.returncode:
        raise RuntimeError(result.stderr.decode(errors="replace").strip())
    return result.stdout


def scaffold_files(root: Path) -> dict[str, bytes]:
    files: dict[str, bytes] = {}
    for path in root.rglob("*.py"):
        relative = path.relative_to(root).as_posix()
        if not (set(path.relative_to(root).parts) & SKIP_PARTS):
            files[relative] = path.read_bytes()
    return files


def commit_files(repo: Path, revision: str) -> dict[str, bytes]:
    names = git(repo, "ls-tree", "-r", "--name-only", revision).decode().splitlines()
    names = [
        name
        for name in names
        if name.endswith(".py") and not (set(Path(name).parts) & SKIP_PARTS)
    ]
    requests = b"".join(f"{revision}:{name}\n".encode() for name in names)
    result = subprocess.run(
        ["git", "-C", str(repo), "cat-file", "--batch"],
        input=requests,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.decode(errors="replace").strip())
    files: dict[str, bytes] = {}
    stream = io.BytesIO(result.stdout)
    for name in names:
        header = stream.readline().decode().rstrip("\n")
        fields = header.split()
        if len(fields) != 3 or fields[1] != "blob":
            raise RuntimeError(f"unexpected cat-file header for {name}: {header}")
        size = int(fields[2])
        files[name] = stream.read(size)
        if stream.read(1) != b"\n":
            raise RuntimeError(f"missing cat-file separator after {name}")
    return files


def worktree_files(repo: Path) -> dict[str, bytes]:
    files: dict[str, bytes] = {}
    tracked = git(repo, "ls-files", "-z").split(b"\0")
    untracked = git(repo, "ls-files", "--others", "--exclude-standard", "-z").split(b"\0")
    for encoded in tracked + untracked:
        if not encoded:
            continue
        name = encoded.decode()
        relative = Path(name)
        path = repo / relative
        if name.endswith(".py") and not (set(relative.parts) & SKIP_PARTS) and path.is_file():
            files[name] = path.read_bytes()
    return files


def select_profile(
    files: dict[str, bytes], scaffold: dict[str, bytes], profile: str
) -> dict[str, bytes]:
    if profile == "core":
        return {name: body for name, body in files.items() if Path(name).name in CORE}
    if profile == "authored":
        return {
            name: body
            for name, body in files.items()
            if name not in scaffold or body != scaffold[name]
        }
    raise ValueError(profile)


def canonical_bundle(files: dict[str, bytes]) -> bytes:
    chunks: list[bytes] = []
    for name in sorted(files):
        encoded = name.encode()
        body = files[name]
        chunks.extend(
            [len(encoded).to_bytes(4, "big"), encoded, len(body).to_bytes(8, "big"), body]
        )
    return b"".join(chunks)


def significant_python_tokens(files: dict[str, bytes]) -> int:
    ignored = {
        tokenize.ENCODING,
        tokenize.ENDMARKER,
        tokenize.INDENT,
        tokenize.DEDENT,
        tokenize.NEWLINE,
        tokenize.NL,
        tokenize.COMMENT,
    }
    count = 0
    for body in files.values():
        try:
            count += sum(
                token.type not in ignored
                for token in tokenize.tokenize(io.BytesIO(body).readline)
            )
        except (IndentationError, SyntaxError, tokenize.TokenError):
            # In-progress worktrees may be temporarily invalid.  Tokenize as much as
            # possible without silently replacing the primary byte metrics.
            count += sum(1 for part in re.split(rb"\s+", body) if part)
    return count


def canonical_ast_bundle(files: dict[str, bytes]) -> bytes:
    """Return a formatting/comment-insensitive executable-structure bundle."""
    normalized: dict[str, bytes] = {}
    for name, body in files.items():
        try:
            tree = ast.parse(body, filename=name)
            representation = ast.dump(tree, annotate_fields=True, include_attributes=False)
        except (SyntaxError, ValueError):
            # Preserve an explicit, deterministic representation for an in-progress
            # invalid file instead of dropping it from the measurement.
            representation = "INVALID:" + " ".join(body.decode(errors="replace").split())
        normalized[name] = representation.encode()
    return canonical_bundle(normalized)


def top_level_definition_hashes(files: dict[str, bytes]) -> dict[str, str]:
    """Hash normalized top-level functions/classes with their source filename."""
    result: dict[str, str] = {}
    for name, body in files.items():
        try:
            tree = ast.parse(body, filename=name)
        except (SyntaxError, ValueError):
            continue
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                key = f"{name}:{type(node).__name__}:{node.name}"
                normalized = ast.dump(
                    node, annotate_fields=True, include_attributes=False
                )
                result[key] = hashlib.sha256(normalized.encode()).hexdigest()
    return result


def scorecard_levels(release: Path, game: str) -> int | None:
    path = release / f"{game}_scorecard.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    value = data.get("total_levels_completed")
    return int(value) if isinstance(value, int) else None


def revisions(repo: Path) -> list[tuple[str, str, int | None, int | None]]:
    records: list[tuple[str, str, int | None, int | None]] = []
    raw = git(repo, "log", "--reverse", "--format=%H%x09%s").decode()
    for line in raw.splitlines():
        revision, subject = line.split("\t", 1)
        match = LEVEL_RE.match(subject)
        reached = int(match.group(1)) if match else None
        completed = reached - 1 if reached is not None else None
        records.append((revision, subject, reached, completed))
    return records


def winning_boundaries(log_path: Path) -> dict[int, WinBoundary]:
    """Map completed level to its real winning command and post-win Python edits.

    Direct client calls print the current ``completed`` count and session path.
    The supplied guarded executor instead reports that it stopped on level
    completion.  Requiring either signature avoids treating simulator output,
    status reads, or source-code inspection as a real win.
    """
    events = [json.loads(line) for line in log_path.read_text().splitlines()]
    current_turn = -1
    turn_ends: dict[int, int] = {}
    raw_wins: dict[int, tuple[int, int]] = {}
    completed = 0
    executor_re = re.compile(r"""python3\s+(?:\./)?plan_executor\.py(?:\s|'|")""")

    for index, event in enumerate(events):
        event_type = event.get("type")
        if event_type == "turn.started":
            current_turn += 1
        elif event_type == "turn.completed":
            turn_ends[current_turn] = index
        if event_type != "item.completed":
            continue
        item = event.get("item", {})
        if item.get("type") != "command_execution":
            continue
        command = item.get("command", "")
        output = item.get("aggregated_output", "")
        observed = []
        if (
            "path: /home/user/run/agent_run/client/session/" in output
            and "state:" in output
        ):
            observed = [
                int(value) for value in re.findall(r"completed:\s*(\d+)", output)
            ]
        guarded_win = bool(
            executor_re.search(command)
            and "plan_executor.py: stopped on level completion" in output
        )
        reached = max(observed, default=completed)
        if guarded_win and reached <= completed:
            reached = completed + 1
        if reached > completed:
            for level in range(completed + 1, reached + 1):
                raw_wins[level] = (index, current_turn)
            completed = reached

    boundaries: dict[int, WinBoundary] = {}
    for level, (win_index, turn) in raw_wins.items():
        end = turn_ends.get(turn, len(events) - 1)
        changed: set[str] = set()
        for event in events[win_index + 1 : end + 1]:
            if event.get("type") != "item.completed":
                continue
            item = event.get("item", {})
            if item.get("type") != "file_change":
                continue
            for change in item.get("changes", []):
                path = change.get("path", "")
                if path.endswith(".py"):
                    changed.add(Path(path).name)
        boundaries[level] = WinBoundary(
            log_line=win_index + 1,
            post_win_python_files=tuple(sorted(changed)),
        )
    return boundaries


def analyse_game(release: Path, repo: Path, scaffold: dict[str, bytes]) -> list[Row]:
    game = repo.parents[1].name
    checkpoints: list[tuple[str, str, int | None, int | None, bool, dict[str, bytes]]] = []
    greatest_reached = 0
    for revision, subject, reached, completed in revisions(repo):
        # A snapshot is taken at the start of an agent iteration.  The first
        # snapshot at a newly reached level L+1 is the model retained after
        # clearing level L.  It equals the winning source only if the log-fidelity
        # test below finds no post-win Python edit.  Repeated same-level snapshots
        # are failed search iterations and are deliberately excluded.
        if reached is not None and reached > greatest_reached:
            greatest_reached = reached
            if completed is not None and completed > 0:
                checkpoints.append(
                    (revision, subject, reached, completed, False, commit_files(repo, revision))
                )
    final_completed = scorecard_levels(release, game)
    checkpoint_floor = max((item[3] or 0 for item in checkpoints), default=0)
    # On a complete solve the stop condition fires before another Git snapshot.
    # The final worktree is admissible only when the scorecard proves it completed
    # more levels than the last retained snapshot.
    if final_completed is not None and final_completed > checkpoint_floor:
        checkpoints.append(
            ("WORKTREE", "scorecard-confirmed final solve", None, final_completed, True, worktree_files(repo))
        )

    boundaries = winning_boundaries(repo.parent / "agent.log")
    rows: list[Row] = []
    previous: dict[tuple[str, str], int] = defaultdict(int)
    previous_definitions: dict[str, dict[str, str]] = defaultdict(dict)
    previous_completed = 0
    previous_boundary_exact = False
    for index, (revision, subject, reached, completed, final, files) in enumerate(checkpoints):
        span = (completed or previous_completed) - previous_completed
        boundary = boundaries.get(completed or -1)
        boundary_exact = boundary is not None and not boundary.post_win_python_files
        exact_adjacent = (
            index > 0
            and span == 1
            and previous_boundary_exact
            and boundary_exact
        )
        for profile in ("core", "authored"):
            selected = select_profile(files, scaffold, profile)
            bundle = canonical_bundle(selected)
            compressed = len(zlib.compress(bundle, level=9)) if bundle else 0
            raw_bytes = len(bundle)
            python_tokens = significant_python_tokens(selected)
            ast_bundle = canonical_ast_bundle(selected)
            ast_compressed = len(zlib.compress(ast_bundle, level=9)) if ast_bundle else 0
            definitions = top_level_definition_hashes(selected)
            retained_definitions = [
                name
                for name, digest in definitions.items()
                if previous_definitions[profile].get(name) == digest
            ]
            delta = compressed - previous[(profile, "zlib")]
            delta_raw = raw_bytes - previous[(profile, "raw")]
            delta_tokens = python_tokens - previous[(profile, "tokens")]
            delta_ast = ast_compressed - previous[(profile, "ast_zlib")]
            rows.append(
                Row(
                    game=game,
                    checkpoint=index,
                    revision=revision,
                    subject=subject,
                    reached_level=reached,
                    completed_levels=completed,
                    levels_spanned=span,
                    final_worktree=final,
                    win_log_line=boundary.log_line if boundary else None,
                    post_win_python_files=(
                        ";".join(boundary.post_win_python_files) if boundary else ""
                    ),
                    exact_winning_source=boundary_exact,
                    exact_adjacent_transition=exact_adjacent,
                    profile=profile,
                    files=len(selected),
                    retained_top_level_definitions=len(retained_definitions),
                    retained_core_top_level_definitions=sum(
                        Path(name.split(":", 1)[0]).name in CORE
                        for name in retained_definitions
                    ),
                    raw_bytes=raw_bytes,
                    zlib_bytes=compressed,
                    python_tokens=python_tokens,
                    ast_zlib_bytes=ast_compressed,
                    delta_raw_bytes=delta_raw,
                    delta_zlib=delta,
                    delta_python_tokens=delta_tokens,
                    delta_ast_zlib=delta_ast,
                    positive_delta_zlib=max(0, delta),
                    contraction_zlib=max(0, -delta),
                    contraction_ast_zlib=max(0, -delta_ast),
                )
            )
            previous[(profile, "zlib")] = compressed
            previous[(profile, "raw")] = raw_bytes
            previous[(profile, "tokens")] = python_tokens
            previous[(profile, "ast_zlib")] = ast_compressed
            previous_definitions[profile] = definitions
        previous_completed = completed or previous_completed
        previous_boundary_exact = boundary_exact
    return rows


def sign_reversals(deltas: Iterable[int]) -> tuple[int, int]:
    signs = [1 if value > 0 else -1 for value in deltas if value]
    peaks = sum(a > 0 and b < 0 for a, b in zip(signs, signs[1:]))
    troughs = sum(a < 0 and b > 0 for a, b in zip(signs, signs[1:]))
    return peaks, troughs


def summarize(rows: list[Row]) -> dict[str, object]:
    by_profile: dict[str, list[Row]] = defaultdict(list)
    for row in rows:
        by_profile[row.profile].append(row)
    profiles: dict[str, object] = {}
    for profile, selected in by_profile.items():
        # The first checkpoint is acquisition from the empty measurement state, not a
        # contraction opportunity, so exclude its delta from sawtooth statistics.
        deltas_by_game: dict[str, list[int]] = defaultdict(list)
        ast_deltas_by_game: dict[str, list[int]] = defaultdict(list)
        for row in selected:
            if row.checkpoint > 0:
                deltas_by_game[row.game].append(row.delta_zlib)
                ast_deltas_by_game[row.game].append(row.delta_ast_zlib)
        deltas = [value for values in deltas_by_game.values() for value in values]
        ast_deltas = [value for values in ast_deltas_by_game.values() for value in values]
        exact_adjacent = [
            row for row in selected if row.exact_adjacent_transition
        ]
        peaks = troughs = 0
        marginal_peaks = marginal_troughs = 0
        games_with_contraction = 0
        games_with_both_signs = 0
        for values in deltas_by_game.values():
            p, t = sign_reversals(values)
            peaks += p
            troughs += t
            games_with_contraction += any(value < 0 for value in values)
            games_with_both_signs += any(value < 0 for value in values) and any(
                value > 0 for value in values
            )
            marginals = [max(0, value) for value in values]
            marginal_differences = [b - a for a, b in zip(marginals, marginals[1:])]
            marginal_signs = [1 if value > 0 else -1 for value in marginal_differences if value]
            marginal_peaks += sum(
                a > 0 and b < 0 for a, b in zip(marginal_signs, marginal_signs[1:])
            )
            marginal_troughs += sum(
                a < 0 and b > 0 for a, b in zip(marginal_signs, marginal_signs[1:])
            )
        profiles[profile] = {
            "checkpoints": len(selected),
            "exact_winning_source_checkpoints": sum(
                row.exact_winning_source for row in selected
            ),
            "transitions": len(deltas),
            "positive_transitions": sum(value > 0 for value in deltas),
            "negative_transitions": sum(value < 0 for value in deltas),
            "zero_transitions": sum(value == 0 for value in deltas),
            "total_positive_zlib_bytes": sum(max(0, value) for value in deltas),
            "total_contraction_zlib_bytes": sum(max(0, -value) for value in deltas),
            "ast_positive_transitions": sum(value > 0 for value in ast_deltas),
            "ast_negative_transitions": sum(value < 0 for value in ast_deltas),
            "ast_zero_transitions": sum(value == 0 for value in ast_deltas),
            "total_contraction_ast_zlib_bytes": sum(max(0, -value) for value in ast_deltas),
            "post_solve_retained_source_and_ast_contractions": sum(
                row.delta_zlib < 0 and row.delta_ast_zlib < 0
                for row in selected
                if row.checkpoint > 0
            ),
            "adjacent_post_solve_retained_source_and_ast_contractions": sum(
                row.delta_zlib < 0
                and row.delta_ast_zlib < 0
                and row.levels_spanned == 1
                for row in selected
                if row.checkpoint > 0
            ),
            "multi_level_post_solve_retained_source_and_ast_contractions": sum(
                row.delta_zlib < 0
                and row.delta_ast_zlib < 0
                and row.levels_spanned > 1
                for row in selected
                if row.checkpoint > 0
            ),
            "exact_adjacent_transitions": len(exact_adjacent),
            "exact_adjacent_positive_transitions": sum(
                row.delta_zlib > 0 for row in exact_adjacent
            ),
            "exact_adjacent_negative_transitions": sum(
                row.delta_zlib < 0 for row in exact_adjacent
            ),
            "exact_adjacent_zero_transitions": sum(
                row.delta_zlib == 0 for row in exact_adjacent
            ),
            "exact_adjacent_source_and_ast_contractions": sum(
                row.delta_zlib < 0 and row.delta_ast_zlib < 0
                for row in exact_adjacent
            ),
            "exact_adjacent_contractions": [
                {
                    "game": row.game,
                    "from_completed_level": (row.completed_levels or 0) - 1,
                    "to_completed_level": row.completed_levels,
                    "delta_zlib": row.delta_zlib,
                    "delta_ast_zlib": row.delta_ast_zlib,
                    "retained_top_level_definitions": (
                        row.retained_top_level_definitions
                    ),
                    "retained_core_top_level_definitions": (
                        row.retained_core_top_level_definitions
                    ),
                }
                for row in exact_adjacent
                if row.delta_zlib < 0 and row.delta_ast_zlib < 0
            ],
            "games_with_contraction": games_with_contraction,
            "games_with_both_signs": games_with_both_signs,
            "positive_to_negative_peaks": peaks,
            "negative_to_positive_troughs": troughs,
            "local_peaks_in_positive_checkpoint_marginals": marginal_peaks,
            "local_troughs_in_positive_checkpoint_marginals": marginal_troughs,
            "multi_level_checkpoint_jumps": sum(row.levels_spanned > 1 for row in selected),
        }
    return {
        "games": len({row.game for row in rows}),
        "measurement": "zlib-9 bytes of canonical post-solve retained Python-source bundles",
        "interpretation": "computable description-length upper bound; not K itself",
        "exactness_rule": (
            "a winning source is exact only when no Python file changed between "
            "the real winning command and the end of that Codex turn; a transition "
            "is exact only when both adjacent endpoints are exact"
        ),
        "profiles": profiles,
    }


def write_csv(path: Path, rows: list[Row]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0])))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("release", type=Path)
    parser.add_argument("--baseline-repo", type=Path, required=True)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    scaffold_root = (
        args.baseline_repo
        / "secure_baseline1_v1.5"
        / "src"
        / "agent"
        / "workspace_init"
    )
    scaffold = scaffold_files(scaffold_root)
    repos = sorted(args.release.glob("*/run/agent_run"))
    if not repos:
        parser.error(f"no game workspaces under {args.release}")
    rows = [row for repo in repos for row in analyse_game(args.release, repo, scaffold)]
    summary = summarize(rows)
    if args.csv:
        write_csv(args.csv, rows)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({"summary": summary, "rows": [asdict(r) for r in rows]}, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
