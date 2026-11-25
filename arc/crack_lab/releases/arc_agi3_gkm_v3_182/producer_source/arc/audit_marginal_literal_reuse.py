#!/usr/bin/env python3
"""Couple solved-level marginal complexity to literal executable reuse.

This audit implements the checkpoint rule used in the manuscript: the program
present when a level is actually cleared is the checkpoint.  Failed proposals,
post-win rewrites, and other interim states are not checkpoints.

For two adjacent winning programs P_(k-1) and P_k, the conditional AST marginal
is the zlib-9 length of the normalized top-level AST statements in P_k that do
not occur literally in P_(k-1).  It is a computable upper bound on the source
needed to describe P_k given a dictionary containing P_(k-1), not an estimate
of uncomputable Kolmogorov complexity.  Unlike a signed total-size difference,
it charges same-size rewrites.

A drop in that marginal is only a search cue.  It is classified as executable
reuse only when the winning entry point directly calls a named definition whose
normalized AST is literally unchanged from the preceding winning checkpoint.
The audit deliberately reports the two tests separately.

System-specific evidence boundaries:

* GKM: exact ``play_level_K`` plus ``legs.py``/``players.py``/``solve.py``.
* OPINE: the last synthesized ``game_engine.py`` before the positive reward;
  call witnesses are available only when the synthesized planner supplied the
  winning action batch.
* baseline1: the exact retained authored source plus the command that produced
  the real level clear.  A literal action list does not become world-model
  reuse merely because retained model code exists elsewhere in the workspace.
* Retrodict: no executable solver checkpoints are released, so executable
  marginal and literal-code reuse are both recorded as absent.
"""

from __future__ import annotations

import argparse
import ast
import collections
import hashlib
import json
import re
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from audit_baseline1_artifacts import (
    commit_files,
    scaffold_files,
    select_profile,
    top_level_definition_hashes,
    worktree_files,
)
from audit_gkm_solved_checkpoints import exact_snapshot


DefinitionKey = tuple[str, str, str]


@dataclass
class MarginalReuseRow:
    system: str
    game: str
    completed_level: int
    source_checkpoint_exact: bool
    exact_adjacent_transition: bool
    marginal_ast_zlib_bytes: int | None
    previous_level_marginal_ast_zlib_bytes: int | None
    marginal_drop_bytes: int | None
    marginal_ratio: float | None
    sharp_marginal_drop: bool
    literal_reused_top_level_nodes: int | None
    novel_top_level_nodes: int | None
    winning_policy_kind: str
    winning_entrypoint: str
    identical_winning_entrypoint: bool
    reused_world_model_literals: list[str]
    reused_world_model_literal_sha256: dict[str, str]
    hard_literal_reuse_witness: bool
    sharp_drop_with_literal_reuse: bool
    winning_command_zlib_bytes: int | None
    evidence: str


def normalized_top_level_units(files: dict[str, bytes]) -> list[tuple[str, bytes]]:
    """Return a content-addressed multiset of normalized top-level AST nodes."""
    result: list[tuple[str, bytes]] = []
    for filename, source in sorted(files.items()):
        try:
            tree = ast.parse(source, filename=filename)
        except (SyntaxError, ValueError):
            continue
        for node in tree.body:
            representation = ast.dump(
                node, annotate_fields=True, include_attributes=False
            ).encode()
            result.append(
                (hashlib.sha256(representation).hexdigest(), representation)
            )
    return result


def conditional_ast_marginal(
    previous: dict[str, bytes], current: dict[str, bytes]
) -> tuple[int, int, int]:
    """Measure literal top-level AST novelty in ``current`` given ``previous``."""
    available = collections.Counter(
        digest for digest, _ in normalized_top_level_units(previous)
    )
    novel: list[bytes] = []
    reused = 0
    for digest, representation in normalized_top_level_units(current):
        if available[digest]:
            available[digest] -= 1
            reused += 1
        else:
            novel.append(representation)
    bundle = b"".join(
        len(representation).to_bytes(8, "big") + representation
        for representation in novel
    )
    compressed = len(zlib.compress(bundle, 9)) if bundle else 0
    return compressed, reused, len(novel)


def definitions_and_calls(
    files: dict[str, bytes],
) -> tuple[dict[DefinitionKey, str], dict[str, set[str]]]:
    """Return literal definition hashes and a conservative name-call graph."""
    definitions: dict[DefinitionKey, str] = {}
    calls: dict[str, set[str]] = collections.defaultdict(set)
    for filename, source in files.items():
        try:
            tree = ast.parse(source, filename=filename)
        except (SyntaxError, ValueError):
            continue
        for node in tree.body:
            if not isinstance(
                node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            ):
                continue
            key = (filename, type(node).__name__, node.name)
            representation = ast.dump(
                node, annotate_fields=True, include_attributes=False
            )
            definitions[key] = hashlib.sha256(representation.encode()).hexdigest()
            calls[node.name].update(
                child.func.id
                for child in ast.walk(node)
                if isinstance(child, ast.Call)
                and isinstance(child.func, ast.Name)
            )
    return definitions, calls


def unchanged_called_definitions(
    previous: dict[str, bytes],
    current: dict[str, bytes],
    root: str,
    *,
    allowed_files: set[str] | None = None,
) -> list[str]:
    """Return unchanged definitions called directly by the winning entry point."""
    previous_definitions, _ = definitions_and_calls(previous)
    current_definitions, calls = definitions_and_calls(current)
    directly_called = calls.get(root, set())
    result: list[str] = []
    for key, digest in current_definitions.items():
        filename, _, name = key
        if name not in directly_called:
            continue
        if allowed_files is not None and filename not in allowed_files:
            continue
        if previous_definitions.get(key) == digest:
            result.append(f"{filename}:{name}")
    return sorted(result)


def definition_digest(
    files: dict[str, bytes], filename: str, name: str
) -> str | None:
    definitions, _ = definitions_and_calls(files)
    for (candidate_file, _, candidate_name), digest in definitions.items():
        if candidate_file == filename and candidate_name == name:
            return digest
    return None


def hashes_for_labels(
    files: dict[str, bytes], labels: Iterable[str]
) -> dict[str, str]:
    definitions, _ = definitions_and_calls(files)
    result: dict[str, str] = {}
    for label in labels:
        filename, name = label.rsplit(":", 1)
        for (candidate_file, _, candidate_name), digest in definitions.items():
            if candidate_file == filename and candidate_name == name:
                result[label] = digest
                break
    return result


def transition_fields(
    marginal: int | None, previous_marginal: int | None
) -> tuple[int | None, float | None, bool]:
    if marginal is None or previous_marginal is None:
        return None, None, False
    drop = previous_marginal - marginal
    ratio = marginal / previous_marginal if previous_marginal else None
    # "Sharp" is a descriptive, predeclared half-or-more reduction.  The
    # numerical values are retained so other thresholds can be checked.
    sharp = bool(previous_marginal and marginal * 2 <= previous_marginal)
    return drop, ratio, sharp


def gkm_rows(root: Path) -> list[MarginalReuseRow]:
    rows: list[MarginalReuseRow] = []
    for game_dir in sorted(root.glob("*_legs")):
        checkpoint_path = game_dir / "checkpoint.json"
        if not checkpoint_path.exists():
            continue
        game = game_dir.name.removesuffix("_legs")
        reached = int(json.loads(checkpoint_path.read_text())["reached"])
        snapshots: list[tuple[int, dict, dict[str, bytes]]] = []
        for level in range(1, reached + 1):
            candidate = exact_snapshot(game_dir, level)
            if candidate is not None:
                metadata, files = candidate
                snapshots.append((level, metadata, files))

        previous_level = 0
        previous_files: dict[str, bytes] = {}
        previous_marginal: int | None = None
        for level, metadata, files in snapshots:
            source_adjacent = level == previous_level + 1
            adjacent = previous_level > 0 and source_adjacent
            measurable = (level == 1 and previous_level == 0) or adjacent
            marginal = reused_nodes = novel_nodes = None
            if measurable:
                marginal, reused_nodes, novel_nodes = conditional_ast_marginal(
                    previous_files, files
                )
            drop, ratio, sharp = transition_fields(
                marginal, previous_marginal if source_adjacent else None
            )
            entrypoint = f"play_level_{level}"
            reused_literals = (
                unchanged_called_definitions(
                    previous_files,
                    files,
                    entrypoint,
                    allowed_files={"legs.py"},
                )
                if adjacent
                else []
            )
            hard_reuse = bool(reused_literals)
            entrypoint_digest = definition_digest(files, "players.py", entrypoint)
            previous_entrypoint_digest = definition_digest(
                previous_files, "players.py", entrypoint
            )
            rows.append(
                MarginalReuseRow(
                    system="GKM",
                    game=game,
                    completed_level=level,
                    source_checkpoint_exact=True,
                    exact_adjacent_transition=adjacent,
                    marginal_ast_zlib_bytes=marginal,
                    previous_level_marginal_ast_zlib_bytes=(
                        previous_marginal if source_adjacent else None
                    ),
                    marginal_drop_bytes=drop,
                    marginal_ratio=ratio,
                    sharp_marginal_drop=sharp,
                    literal_reused_top_level_nodes=reused_nodes,
                    novel_top_level_nodes=novel_nodes,
                    winning_policy_kind=str(metadata["phase"]),
                    winning_entrypoint=f"players.py:{entrypoint}",
                    identical_winning_entrypoint=bool(
                        entrypoint_digest
                        and entrypoint_digest == previous_entrypoint_digest
                    ),
                    reused_world_model_literals=reused_literals,
                    reused_world_model_literal_sha256=hashes_for_labels(
                        files, reused_literals
                    ),
                    hard_literal_reuse_witness=hard_reuse,
                    sharp_drop_with_literal_reuse=sharp and hard_reuse,
                    winning_command_zlib_bytes=None,
                    evidence=(
                        "winning player statically calls unchanged named legs"
                        if hard_reuse
                        else "no unchanged named leg is reachable from the winning player"
                    ),
                )
            )
            previous_level = level
            previous_files = files
            previous_marginal = marginal if measurable else None
    return rows


def opine_rows(root: Path, audit_json: Path) -> list[MarginalReuseRow]:
    audit_rows = json.loads(audit_json.read_text())["rows"]
    grouped: dict[str, list[dict]] = collections.defaultdict(list)
    for row in audit_rows:
        grouped[str(row["game"])].append(row)

    result: list[MarginalReuseRow] = []
    for game, game_rows in sorted(grouped.items()):
        previous_level = 0
        previous_files: dict[str, bytes] = {}
        previous_marginal: int | None = None
        for source_row in game_rows:
            level = int(source_row["completed_levels"])
            path = root / str(source_row["engine_path"])
            files = {"game_engine.py": path.read_bytes()}
            source_adjacent = level == previous_level + 1
            adjacent = previous_level > 0 and source_adjacent
            measurable = (level == 1 and previous_level == 0) or adjacent
            marginal = reused_nodes = novel_nodes = None
            if measurable:
                marginal, reused_nodes, novel_nodes = conditional_ast_marginal(
                    previous_files, files
                )
            drop, ratio, sharp = transition_fields(
                marginal, previous_marginal if source_adjacent else None
            )
            synthesized = bool(source_row["synth_planner_witness"])
            reused_literals = (
                unchanged_called_definitions(
                    previous_files,
                    files,
                    "planner",
                    allowed_files={"game_engine.py"},
                )
                if synthesized and adjacent
                else []
            )
            hard_reuse = synthesized and bool(reused_literals)
            planner_digest = definition_digest(files, "game_engine.py", "planner")
            previous_planner_digest = definition_digest(
                previous_files, "game_engine.py", "planner"
            )
            result.append(
                MarginalReuseRow(
                    system="OPINE",
                    game=game,
                    completed_level=level,
                    source_checkpoint_exact=True,
                    exact_adjacent_transition=adjacent,
                    marginal_ast_zlib_bytes=marginal,
                    previous_level_marginal_ast_zlib_bytes=(
                        previous_marginal if source_adjacent else None
                    ),
                    marginal_drop_bytes=drop,
                    marginal_ratio=ratio,
                    sharp_marginal_drop=sharp,
                    literal_reused_top_level_nodes=reused_nodes,
                    novel_top_level_nodes=novel_nodes,
                    winning_policy_kind=(
                        "synthesized_planner" if synthesized else "transient_analyzer"
                    ),
                    winning_entrypoint=(
                        "game_engine.py:planner" if synthesized else "unreleased analyzer policy"
                    ),
                    identical_winning_entrypoint=bool(
                        synthesized
                        and planner_digest
                        and planner_digest == previous_planner_digest
                    ),
                    reused_world_model_literals=reused_literals,
                    reused_world_model_literal_sha256=hashes_for_labels(
                        files, reused_literals
                    ),
                    hard_literal_reuse_witness=hard_reuse,
                    sharp_drop_with_literal_reuse=sharp and hard_reuse,
                    winning_command_zlib_bytes=None,
                    evidence=(
                        "winning synthesized planner calls unchanged engine definitions"
                        if hard_reuse
                        else (
                            "winning policy was the transient analyzer, not the retained engine planner"
                            if not synthesized
                            else "no unchanged engine definition is reachable from the winning planner"
                        )
                    ),
                )
            )
            previous_level = level
            previous_files = files
            previous_marginal = marginal if measurable else None
    return result


def winning_command(log_path: Path, line_number: int) -> str:
    line = log_path.read_text().splitlines()[line_number - 1]
    event = json.loads(line)
    return str(event.get("item", {}).get("command", ""))


def classify_baseline_command(command: str) -> str:
    if re.search(r"\bworld_model(?:_main_planner|_engine|_state_io)\b", command):
        return "inline_world_model_program"
    if re.search(r"\bplan_executor\.py\b", command):
        return "literal_action_program_via_executor"
    if re.search(
        r"\bplan\s*=\s*\[|\bactions\s*=\s*\(|for\s+\w+\s+in\s+\[",
        command,
    ):
        return "inline_literal_action_program"
    if "client/client.py" in command:
        return "direct_literal_action"
    return "other_transient_command"


def imported_world_model_names(command: str) -> list[tuple[str, str]]:
    result: list[tuple[str, str]] = []
    for module, names in re.findall(
        r"from\s+(world_model(?:_main_planner|_engine|_state_io))\s+import\s+([^\n;]+)",
        command,
    ):
        for name in re.findall(r"\b[A-Za-z_]\w*\b", names):
            if name != "as":
                result.append((f"{module}.py", name))
    return result


def baseline_rows(
    release: Path, baseline_repo: Path, audit_json: Path
) -> list[MarginalReuseRow]:
    scaffold_root = (
        baseline_repo
        / "secure_baseline1_v1.5"
        / "src"
        / "agent"
        / "workspace_init"
    )
    scaffold = scaffold_files(scaffold_root)
    audit_rows = [
        row
        for row in json.loads(audit_json.read_text())["rows"]
        if row["profile"] == "authored"
    ]
    grouped: dict[str, list[dict]] = collections.defaultdict(list)
    for row in audit_rows:
        grouped[str(row["game"])].append(row)

    result: list[MarginalReuseRow] = []
    for game, game_rows in sorted(grouped.items()):
        repo = release / game / "run" / "agent_run"
        previous_level = 0
        previous_files: dict[str, bytes] = {}
        previous_marginal: int | None = None
        previous_exact = False
        for source_row in game_rows:
            raw_files = (
                worktree_files(repo)
                if source_row["revision"] == "WORKTREE"
                else commit_files(repo, str(source_row["revision"]))
            )
            files = select_profile(raw_files, scaffold, "authored")
            level = int(source_row["completed_levels"])
            exact = bool(source_row["exact_winning_source"])
            adjacent = (
                exact
                and previous_exact
                and level == previous_level + 1
            )
            # Level 1 is an acquisition from the empty description.  Every later
            # marginal requires two exact, adjacent winning checkpoints.
            measurable = exact and (
                (level == 1 and previous_level == 0) or adjacent
            )
            marginal = reused_nodes = novel_nodes = None
            if measurable:
                marginal, reused_nodes, novel_nodes = conditional_ast_marginal(
                    previous_files, files
                )
            drop, ratio, sharp = transition_fields(
                marginal,
                previous_marginal if adjacent else None,
            )
            line_number = source_row.get("win_log_line")
            command = (
                winning_command(repo.parent / "agent.log", int(line_number))
                if exact and line_number
                else ""
            )
            command_kind = classify_baseline_command(command) if command else "missing"
            current_definition_hashes = top_level_definition_hashes(files)
            previous_definition_hashes = top_level_definition_hashes(previous_files)
            reused_imports: list[str] = []
            for filename, name in imported_world_model_names(command):
                suffix = f"{filename}:FunctionDef:{name}"
                keys = [
                    key
                    for key in current_definition_hashes
                    if key.endswith(suffix)
                ]
                if any(
                    previous_definition_hashes.get(key)
                    == current_definition_hashes[key]
                    for key in keys
                ):
                    reused_imports.append(f"{filename}:{name}")
            planner_key = "world_model_main_planner.py:FunctionDef:planner"
            identical_planner = bool(
                current_definition_hashes.get(planner_key)
                and current_definition_hashes.get(planner_key)
                == previous_definition_hashes.get(planner_key)
            )
            hard_reuse = bool(reused_imports)
            command_zlib = len(zlib.compress(command.encode(), 9)) if command else None
            result.append(
                MarginalReuseRow(
                    system="baseline1",
                    game=game,
                    completed_level=level,
                    source_checkpoint_exact=exact,
                    exact_adjacent_transition=adjacent,
                    marginal_ast_zlib_bytes=marginal,
                    previous_level_marginal_ast_zlib_bytes=(
                        previous_marginal if adjacent else None
                    ),
                    marginal_drop_bytes=drop,
                    marginal_ratio=ratio,
                    sharp_marginal_drop=sharp,
                    literal_reused_top_level_nodes=reused_nodes,
                    novel_top_level_nodes=novel_nodes,
                    winning_policy_kind=command_kind,
                    winning_entrypoint="agent.log winning command",
                    identical_winning_entrypoint=identical_planner,
                    reused_world_model_literals=sorted(set(reused_imports)),
                    reused_world_model_literal_sha256=hashes_for_labels(
                        files, reused_imports
                    ),
                    hard_literal_reuse_witness=hard_reuse,
                    sharp_drop_with_literal_reuse=sharp and hard_reuse,
                    winning_command_zlib_bytes=command_zlib,
                    evidence=(
                        "winning command imports an unchanged retained world-model definition"
                        if hard_reuse
                        else (
                            "winning command is a fresh literal action program and invokes no retained world-model definition"
                            if command_kind
                            in {
                                "literal_action_program_via_executor",
                                "direct_literal_action",
                                "inline_literal_action_program",
                            }
                            else "no unchanged retained world-model definition is invoked by the winning command"
                        )
                    ),
                )
            )
            previous_level = level
            previous_files = files
            previous_exact = exact
            previous_marginal = marginal if measurable else None
    return result


def selected_rows(
    rows: Iterable[MarginalReuseRow],
) -> list[MarginalReuseRow]:
    return [row for row in rows if row.exact_adjacent_transition]


def system_summary(rows: list[MarginalReuseRow]) -> dict[str, object]:
    exact = selected_rows(rows)
    measurable_drops = [
        row for row in exact if row.previous_level_marginal_ast_zlib_bytes is not None
    ]
    return {
        "retained_or_winning_checkpoints": len(rows),
        "exact_winning_checkpoints": sum(
            row.source_checkpoint_exact for row in rows
        ),
        "exact_adjacent_transitions": len(exact),
        "transitions_with_level_to_level_marginal_comparison": len(measurable_drops),
        "marginal_decreases": sum(
            (row.marginal_drop_bytes or 0) > 0 for row in measurable_drops
        ),
        "sharp_half_or_more_marginal_drops": sum(
            row.sharp_marginal_drop for row in measurable_drops
        ),
        "hard_literal_world_model_reuse_witnesses": sum(
            row.hard_literal_reuse_witness for row in exact
        ),
        "sharp_drops_with_literal_reuse": [
            {
                "game": row.game,
                "completed_level": row.completed_level,
                "previous_marginal_ast_zlib_bytes": (
                    row.previous_level_marginal_ast_zlib_bytes
                ),
                "marginal_ast_zlib_bytes": row.marginal_ast_zlib_bytes,
                "marginal_drop_bytes": row.marginal_drop_bytes,
                "reused_world_model_literals": row.reused_world_model_literals,
            }
            for row in measurable_drops
            if row.sharp_drop_with_literal_reuse
        ],
        "all_hard_literal_reuse_witnesses": [
            {
                "game": row.game,
                "completed_level": row.completed_level,
                "previous_marginal_ast_zlib_bytes": (
                    row.previous_level_marginal_ast_zlib_bytes
                ),
                "marginal_ast_zlib_bytes": row.marginal_ast_zlib_bytes,
                "marginal_drop_bytes": row.marginal_drop_bytes,
                "identical_winning_entrypoint": row.identical_winning_entrypoint,
                "reused_world_model_literals": row.reused_world_model_literals,
            }
            for row in exact
            if row.hard_literal_reuse_witness
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gkm-root",
        type=Path,
        default=Path("arc/crack_lab/agent_solutions"),
    )
    parser.add_argument("--opine-root", type=Path)
    parser.add_argument(
        "--opine-audit-json",
        type=Path,
        default=Path("arc/audit_results/opine-solved-checkpoints.json"),
    )
    parser.add_argument("--baseline-release", type=Path)
    parser.add_argument("--baseline-repo", type=Path)
    parser.add_argument(
        "--baseline-audit-json",
        type=Path,
        default=Path(
            "arc/audit_results/baseline1_gpt55_xhigh_solved_checkpoints.json"
        ),
    )
    parser.add_argument(
        "--retrodict-audit-json",
        type=Path,
        default=Path(
            "arc/audit_results/retrodict-solved-checkpoint-memory.json"
        ),
    )
    parser.add_argument("--json", type=Path)
    parser.add_argument(
        "--reuse-non-gkm-from-json",
        type=Path,
        help=(
            "reuse previously audited OPINE/baseline1 rows from this JSON while "
            "recomputing GKM from the current local solved checkpoints"
        ),
    )
    args = parser.parse_args()

    if args.reuse_non_gkm_from_json:
        cached = json.loads(args.reuse_non_gkm_from_json.read_text())
        cached_rows = [MarginalReuseRow(**row) for row in cached["rows"]]
        by_system = {
            "GKM": gkm_rows(args.gkm_root),
            "OPINE": [row for row in cached_rows if row.system == "OPINE"],
            "baseline1": [
                row for row in cached_rows if row.system == "baseline1"
            ],
        }
    else:
        required = {
            "--opine-root": args.opine_root,
            "--baseline-release": args.baseline_release,
            "--baseline-repo": args.baseline_repo,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            parser.error(
                "the following arguments are required without "
                f"--reuse-non-gkm-from-json: {', '.join(missing)}"
            )
        by_system = {
            "GKM": gkm_rows(args.gkm_root),
            "OPINE": opine_rows(args.opine_root, args.opine_audit_json),
            "baseline1": baseline_rows(
                args.baseline_release,
                args.baseline_repo,
                args.baseline_audit_json,
            ),
        }
    retrodict_audit = json.loads(args.retrodict_audit_json.read_text())["summary"]
    summary = {
        "measurement": (
            "zlib-9 bytes of normalized top-level AST statements in the current "
            "exact winning program that are not literal members of the preceding "
            "exact winning program"
        ),
        "sharp_drop_rule": (
            "current conditional AST marginal is at most one half of the preceding "
            "level's conditional AST marginal"
        ),
        "reuse_rule": (
            "the winning executable entry point directly calls a named definition whose "
            "normalized AST is identical at the preceding winning checkpoint"
        ),
        "systems": {
            system: system_summary(rows) for system, rows in by_system.items()
        },
    }
    summary["systems"]["Retrodict"] = {
        "retained_or_winning_checkpoints": retrodict_audit["solved_checkpoints"],
        "exact_winning_checkpoints": 0,
        "exact_adjacent_transitions": 0,
        "released_memory_transitions": retrodict_audit[
            "between_solved_checkpoint_transitions"
        ],
        "transitions_with_level_to_level_marginal_comparison": 0,
        "marginal_decreases": 0,
        "sharp_half_or_more_marginal_drops": 0,
        "hard_literal_world_model_reuse_witnesses": 0,
        "sharp_drops_with_literal_reuse": [],
        "all_hard_literal_reuse_witnesses": [],
        "reason": (
            "released checkpoints contain playbook memory and, in two games, "
            "scratch Python; they do not contain an executable winning solver "
            "or winning entry point"
        ),
    }

    payload = {
        "summary": summary,
        "rows": [
            asdict(row)
            for system in ("GKM", "OPINE", "baseline1")
            for row in by_system[system]
        ],
    }
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
