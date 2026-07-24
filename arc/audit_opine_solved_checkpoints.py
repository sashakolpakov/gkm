#!/usr/bin/env python3
"""Audit OPINE only at level-solving checkpoints recorded in the run log.

The checkpoint for a solved level is the last synthesized ``game_engine.py`` that
existed *before* the positive-reward action.  Synthesis emitted after that action is
not credited retroactively.  The script also records whether the winning action batch
came from OPINE's executable synthesized planner or from its transient LLM analyzer.

Consequently, the source measures below are description-length upper bounds for the
retained executable world-model/planner component.  They are full-solver measures only
when ``plan_source == "planner"``; analyzer-solved levels contain an unmeasured
transient policy component.  Missing traces and pre-synthesis solves are reported and
never imputed from final files or summary totals.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import re
import tokenize
import zlib
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path

from audit_baseline1_artifacts import canonical_ast_bundle


SYNTHESIS_RE = re.compile(r"^\[SYNTHESIS step=(\d+) run=(\d+)\]$")
STEP_RE = re.compile(r"^\[STEP (\d+)\]$")
LEVEL_RE = re.compile(r"^\[LEVEL\] (\d+)$")
REWARD_RE = re.compile(
    r"^\[REWARD\]\s+([-+]?(?:\d+(?:\.\d*)?|\.\d+))\s+done=(True|False)$"
)
PLAN_NOTE_RE = re.compile(r"^\[NOTE step=(\d+) source=([^\]]+)\].*\bplan=")


@dataclass
class Checkpoint:
    game: str
    checkpoint: int
    completed_levels: int
    levels_spanned: int
    solve_step: int
    level_index: int
    synthesis_run: int
    synthesis_step: int
    plan_source: str
    synth_planner_witness: bool
    engine_path: str
    source_sha256: str
    same_engine_as_previous_solve: bool
    same_executable_bundle_as_previous_solve: bool
    raw_bytes: int
    significant_tokens: int
    zlib_bytes: int
    ast_zlib_bytes: int
    runtime_data_files: int
    runtime_data_bytes: int
    executable_bundle_zlib_bytes: int
    delta_zlib_bytes: int
    delta_ast_zlib: int
    delta_executable_bundle_zlib: int
    contraction_zlib: int
    contraction_executable_bundle_zlib: int
    has_planner: bool
    planner_same_as_previous_solve: bool
    retained_top_level_definitions: int
    added_top_level_definitions: int
    removed_top_level_definitions: int


def significant_tokens(source: bytes) -> int:
    ignored = {
        tokenize.COMMENT,
        tokenize.ENCODING,
        tokenize.ENDMARKER,
        tokenize.INDENT,
        tokenize.DEDENT,
        tokenize.NEWLINE,
        tokenize.NL,
    }
    try:
        return sum(
            token.type not in ignored
            for token in tokenize.tokenize(BytesIO(source).readline)
        )
    except (IndentationError, SyntaxError, tokenize.TokenError):
        return len(re.findall(rb"\S+", source))


def definition_hashes(source: bytes) -> dict[str, str]:
    """Hash top-level functions/classes after removing source-position metadata."""
    try:
        tree = ast.parse(source.decode("utf-8", "replace"))
    except SyntaxError:
        return {}
    result: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            key = f"{type(node).__name__}:{node.name}"
            normalized = ast.dump(node, annotate_fields=True, include_attributes=False)
            result[key] = hashlib.sha256(normalized.encode()).hexdigest()
    return result


def planner_hash(definitions: dict[str, str]) -> str | None:
    return definitions.get("FunctionDef:planner") or definitions.get(
        "AsyncFunctionDef:planner"
    )


def engine_path(game_dir: Path, run: int) -> Path:
    return game_dir / "synthesis" / f"run_{run:03d}" / "game_engine.py"


def runtime_bundle(path: Path, source: bytes) -> tuple[bytes, list[Path]]:
    """Bundle source with the small level-state files available to its planner.

    OPINE's synthesized engines conventionally load ``l*_initial.pkl`` and a
    few related ``l*.pkl`` files at run time.  Replay buffers, reports, prompts,
    and training diagnostics are deliberately excluded.
    """
    dependencies = sorted(path.parent.glob("l*.pkl"))
    parts = [b"game_engine.py\0", source, b"\0"]
    for dependency in dependencies:
        parts.extend(
            [
                dependency.name.encode(),
                b"\0",
                dependency.read_bytes(),
                b"\0",
            ]
        )
    return b"".join(parts), dependencies


def reported_total_reward(game_dir: Path) -> int:
    summary_path = game_dir / "summary.json"
    if not summary_path.exists():
        return 0
    summary = json.loads(summary_path.read_text())
    # total_reward is the only field that includes the terminal level: the
    # archive's levels_completed counter counts level advances and can be one low.
    return int(round(float(summary.get("total_reward", 0) or 0)))


def analyse_game(game_dir: Path) -> tuple[list[Checkpoint], dict]:
    run_log = game_dir / "run_log.txt"
    if not run_log.exists():
        return [], {
            "game": game_dir.name,
            "reported_total_reward": reported_total_reward(game_dir),
            "positive_reward_events": 0,
            "missing_engine_checkpoint_levels": [],
        }

    current_step = -1
    current_level = -1
    current_run: int | None = None
    current_synthesis_step = -1
    current_plan_source = "unknown"
    current_plan_is_synth = False
    reward_events: list[dict] = []

    for line in run_log.read_text(errors="replace").splitlines():
        if match := STEP_RE.match(line):
            current_step = int(match.group(1))
            continue
        if match := SYNTHESIS_RE.match(line):
            current_synthesis_step = int(match.group(1))
            current_run = int(match.group(2))
            continue
        if match := PLAN_NOTE_RE.match(line):
            current_plan_source = match.group(2)
            current_plan_is_synth = (
                current_plan_source == "planner" and "source=synth_planner" in line
            )
            continue
        if match := LEVEL_RE.match(line):
            current_level = int(match.group(1))
            continue
        if match := REWARD_RE.match(line):
            reward = float(match.group(1))
            done = match.group(2) == "True"
            if reward > 0 or done:
                reward_events.append(
                    {
                        "step": current_step,
                        "level": current_level,
                        "run": current_run,
                        "synthesis_step": current_synthesis_step,
                        "plan_source": current_plan_source,
                        "synth_planner": current_plan_is_synth,
                    }
                )

    rows: list[Checkpoint] = []
    missing: list[int] = []
    previous_level = 0
    previous_zlib = previous_ast = 0
    previous_sha = ""
    previous_bundle_zlib = 0
    previous_bundle_sha = ""
    previous_definitions: dict[str, str] = {}

    for completed_levels, event in enumerate(reward_events, start=1):
        run = event["run"]
        path = engine_path(game_dir, run) if run is not None else None
        if path is None or not path.exists():
            missing.append(completed_levels)
            continue
        source = path.read_bytes()
        compressed = len(zlib.compress(source, 9))
        bundle, runtime_dependencies = runtime_bundle(path, source)
        bundle_compressed = len(zlib.compress(bundle, 9))
        bundle_sha = hashlib.sha256(bundle).hexdigest()
        ast_bundle = canonical_ast_bundle({"game_engine.py": source})
        ast_compressed = len(zlib.compress(ast_bundle, 9)) if ast_bundle else 0
        sha = hashlib.sha256(source).hexdigest()
        definitions = definition_hashes(source)
        retained = sum(
            previous_definitions.get(name) == digest
            for name, digest in definitions.items()
        )
        added = sum(
            previous_definitions.get(name) != digest
            for name, digest in definitions.items()
        )
        removed = sum(
            definitions.get(name) != digest
            for name, digest in previous_definitions.items()
        )
        current_planner_hash = planner_hash(definitions)
        previous_planner_hash = planner_hash(previous_definitions)
        rows.append(
            Checkpoint(
                game=game_dir.name,
                checkpoint=len(rows),
                completed_levels=completed_levels,
                levels_spanned=completed_levels - previous_level,
                solve_step=int(event["step"]),
                level_index=int(event["level"]),
                synthesis_run=int(run),
                synthesis_step=int(event["synthesis_step"]),
                plan_source=str(event["plan_source"]),
                synth_planner_witness=bool(event["synth_planner"]),
                engine_path=str(path.relative_to(game_dir.parent)),
                source_sha256=sha,
                same_engine_as_previous_solve=bool(previous_sha and sha == previous_sha),
                same_executable_bundle_as_previous_solve=bool(
                    previous_bundle_sha and bundle_sha == previous_bundle_sha
                ),
                raw_bytes=len(source),
                significant_tokens=significant_tokens(source),
                zlib_bytes=compressed,
                ast_zlib_bytes=ast_compressed,
                runtime_data_files=len(runtime_dependencies),
                runtime_data_bytes=sum(
                    dependency.stat().st_size for dependency in runtime_dependencies
                ),
                executable_bundle_zlib_bytes=bundle_compressed,
                delta_zlib_bytes=compressed - previous_zlib,
                delta_ast_zlib=ast_compressed - previous_ast,
                delta_executable_bundle_zlib=bundle_compressed
                - previous_bundle_zlib,
                contraction_zlib=max(0, previous_zlib - compressed),
                contraction_executable_bundle_zlib=max(
                    0, previous_bundle_zlib - bundle_compressed
                ),
                has_planner=current_planner_hash is not None,
                planner_same_as_previous_solve=bool(
                    previous_planner_hash
                    and current_planner_hash == previous_planner_hash
                ),
                retained_top_level_definitions=retained,
                added_top_level_definitions=added,
                removed_top_level_definitions=removed,
            )
        )
        previous_level = completed_levels
        previous_zlib = compressed
        previous_ast = ast_compressed
        previous_sha = sha
        previous_bundle_zlib = bundle_compressed
        previous_bundle_sha = bundle_sha
        previous_definitions = definitions

    return rows, {
        "game": game_dir.name,
        "reported_total_reward": reported_total_reward(game_dir),
        "positive_reward_events": len(reward_events),
        "missing_engine_checkpoint_levels": missing,
    }


def summarize(rows: list[Checkpoint], coverage: list[dict]) -> dict:
    transitions = [row for row in rows if row.checkpoint > 0 and row.levels_spanned == 1]
    planner_rows = [row for row in rows if row.synth_planner_witness]
    planner_transitions = [
        row
        for row in planner_rows
        if row.checkpoint > 0 and row.levels_spanned == 1
    ]
    return {
        "games": len(coverage),
        "reported_total_reward_available_summaries": sum(
            item["reported_total_reward"] for item in coverage
        ),
        "positive_reward_solve_events_in_logs": sum(
            item["positive_reward_events"] for item in coverage
        ),
        "solved_checkpoints_with_pre_solve_engine": len(rows),
        "adjacent_solved_checkpoint_transitions": len(transitions),
        "missing_engine_checkpoint_levels": {
            item["game"]: item["missing_engine_checkpoint_levels"]
            for item in coverage
            if item["missing_engine_checkpoint_levels"]
        },
        "summary_trace_solve_discrepancies": {
            item["game"]: {
                "summary_total_reward": item["reported_total_reward"],
                "trace": item["positive_reward_events"],
            }
            for item in coverage
            if item["reported_total_reward"] != item["positive_reward_events"]
        },
        "component_measured": "retained executable game_engine.py world model and planner",
        "interpretation": (
            "full-solver upper bound only for synth-planner solve witnesses; "
            "analyzer-solved levels have an unmeasured transient policy component"
        ),
        "retained_component_source_contractions": sum(
            row.delta_zlib_bytes < 0 for row in transitions
        ),
        "retained_component_source_and_ast_contractions": sum(
            row.delta_zlib_bytes < 0 and row.delta_ast_zlib < 0
            for row in transitions
        ),
        "executable_bundle_contractions": sum(
            row.delta_executable_bundle_zlib < 0 for row in transitions
        ),
        "executable_bundle_and_ast_contractions": sum(
            row.delta_executable_bundle_zlib < 0 and row.delta_ast_zlib < 0
            for row in transitions
        ),
        "identical_engine_reuse_transitions": sum(
            row.same_engine_as_previous_solve for row in transitions
        ),
        "identical_executable_bundle_reuse_transitions": sum(
            row.same_executable_bundle_as_previous_solve for row in transitions
        ),
        "unchanged_planner_transitions": sum(
            row.planner_same_as_previous_solve for row in transitions
        ),
        "synth_planner_solved_checkpoints": len(planner_rows),
        "synth_planner_adjacent_transitions": len(planner_transitions),
        "synth_planner_source_contractions": sum(
            row.delta_zlib_bytes < 0 for row in planner_transitions
        ),
        "synth_planner_executable_bundle_contractions": sum(
            row.delta_executable_bundle_zlib < 0 for row in planner_transitions
        ),
        "synth_planner_identical_engine_reuse": sum(
            row.same_engine_as_previous_solve for row in planner_transitions
        ),
        "synth_planner_identical_executable_bundle_reuse": sum(
            row.same_executable_bundle_as_previous_solve
            for row in planner_transitions
        ),
        "analyzer_or_unknown_solved_checkpoints": len(rows) - len(planner_rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifacts", type=Path)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()

    rows: list[Checkpoint] = []
    coverage: list[dict] = []
    for game_dir in sorted(path for path in args.artifacts.iterdir() if path.is_dir()):
        game_rows, game_coverage = analyse_game(game_dir)
        rows.extend(game_rows)
        coverage.append(game_coverage)

    summary = summarize(rows, coverage)
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0])))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)
    args.json.write_text(
        json.dumps(
            {
                "summary": summary,
                "coverage": coverage,
                "rows": [asdict(row) for row in rows],
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
