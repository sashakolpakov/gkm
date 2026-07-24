#!/usr/bin/env python3
"""Reconstruct Retrodict's retained workspace from its published tool traces.

The release stores the final workspace, but the OpenTelemetry JSONL traces retain every
``write`` and ``edit`` call.  Replaying those calls recovers the size of the mutable
agent memory after each invocation.  This script deliberately keeps three quantities
separate:

* retained agent memory: ``playbook.md`` and substantive ``scratch/*.py`` files;
* the append-only observation log, which grows with interaction and is not reconstructed;
* provider context tokens, which can fall at a fresh-session reset without deleting the
  retained workspace.

The resulting level profile can test for a source-like acquisition/reuse sawtooth.  A
fall in provider context alone is not evidence of a Kolmogorov structure-function drop.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import zlib
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


LEVELS_RE = re.compile(r"^\[LEVELS\] (\d+)/(\d+)")
PLAN_RE = re.compile(r"^\[PLAN\] invocation (\d+)")

# The manifest published in Retrodict's scripts/gen_level_costs.py at commit 3467229.
SELECTED_RUN_IDS = {
    "ar25": "20260714-003820",
    "bp35": "20260714-070049",
    "cd82": "20260714-132112",
    "cn04": "20260714-182321",
    "dc22": "20260714-031541",
    "ft09": "20260713-194732",
    "g50t": "20260714-132113",
    "ka59": "20260713-204514",
    "lf52": "20260715-234048",
    "lp85": "20260714-010634",
    "ls20": "20260713-193033",
    "m0r0": "20260714-010756",
    "r11l": "20260713-235741",
    "re86": "20260714-025821",
    "s5i5": "20260714-113106",
    "sb26": "20260713-194735",
    "sc25": "20260713-222807",
    "sk48": "20260714-072903",
    "sp80": "20260713-235743",
    "su15": "20260714-051305",
    "tn36": "20260713-204511",
    "tr87": "20260714-003244",
    "tu93": "20260714-013605",
    "vc33": "20260713-180730",
    "wa30": "20260714-050148",
}


@dataclass
class Invocation:
    game: str
    run_id: str
    number: int
    level: int
    retained_bytes: int
    playbook_bytes: int
    scratch_bytes: int
    retained_zlib_bytes: int
    playbook_zlib_bytes: int
    scratch_zlib_bytes: int
    delta_bytes: int
    positive_delta: int
    negative_delta: int
    delta_zlib_bytes: int
    positive_delta_zlib: int
    negative_delta_zlib: int
    context_tokens: int
    fresh_session: bool
    writes: int
    edits: int


@dataclass
class SolvedCheckpoint:
    game: str
    run_id: str
    checkpoint: int
    completed_levels: int
    invocation: int
    retained_bytes: int
    retained_zlib_bytes: int
    playbook_zlib_bytes: int
    scratch_zlib_bytes: int
    delta_scratch_zlib_bytes: int
    delta_bytes: int
    delta_zlib_bytes: int
    positive_delta_zlib: int
    contraction_zlib: int


def jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            raw = raw.strip()
            if raw:
                yield json.loads(raw)


def invocation_levels(log_path: Path) -> dict[int, int]:
    levels: dict[int, int] = {}
    current_level = 1
    with log_path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            level_match = LEVELS_RE.match(line)
            if level_match:
                current_level = int(level_match.group(1)) + 1
                continue
            plan_match = PLAN_RE.match(line)
            if plan_match:
                levels[int(plan_match.group(1))] = current_level
    return levels


def trace_events(run_dir: Path, transcript: list[dict[str, Any]]) -> tuple[list[list[dict[str, Any]]], int]:
    traces: list[tuple[float, str, str, list[dict[str, Any]]]] = []
    for path in run_dir.glob("traces/*/*.jsonl"):
        records = list(jsonl(path))
        roots = [record for record in records if record.get("name") == "invoke_agent thinharness"]
        if not roots:
            continue
        root = min(roots, key=lambda record: float(record.get("started_at", 0.0)))
        started_at = float(root.get("started_at", 0.0))
        attrs = root.get("attributes", {})
        traces.append(
            (
                started_at,
                str(attrs.get("gen_ai.prompt", "")),
                str(attrs.get("gen_ai.completion", "")),
                sorted(records, key=lambda record: float(record.get("started_at", 0.0))),
            )
        )
    traces.sort(key=lambda item: item[0])

    # Failed provider/parse attempts can leave a rooted trace without a transcript
    # record. Match on both the outer prompt and final completion so such attempts do
    # not shift every later invocation.
    unmatched = list(traces)
    aligned: list[list[dict[str, Any]]] = []
    for record in transcript:
        prompt = str(record.get("prompt", ""))
        completion = str(record.get("text", ""))
        chosen = next(
            (
                index
                for index, (_, candidate_prompt, candidate_completion, _) in enumerate(unmatched)
                if candidate_prompt == prompt and candidate_completion == completion
            ),
            None,
        )
        if chosen is None:
            chosen = next(
                (
                    index
                    for index, (_, _, candidate_completion, _) in enumerate(unmatched)
                    if candidate_completion == completion
                ),
                None,
            )
        if chosen is None:
            aligned.append([])
        else:
            aligned.append(unmatched.pop(chosen)[3])
    return aligned, len(traces)


def tool_payload(record: dict[str, Any]) -> tuple[str, dict[str, Any], dict[str, Any]] | None:
    attrs = record.get("attributes", {})
    tool = attrs.get("gen_ai.tool.name")
    if tool not in {"write", "edit"}:
        return None
    try:
        arguments = json.loads(attrs["gen_ai.tool.call.arguments"])
        result = json.loads(attrs["gen_ai.tool.call.result"])
    except (KeyError, json.JSONDecodeError, TypeError):
        return None
    return tool, arguments, result


def apply_write(files: dict[str, str], arguments: dict[str, Any], result: dict[str, Any]) -> bool:
    if not result.get("ok"):
        return False
    path = arguments.get("path")
    content = arguments.get("content")
    if not isinstance(path, str) or not isinstance(content, str):
        return False
    if arguments.get("append"):
        files[path] = files.get(path, "") + content
    else:
        files[path] = content
    return True


def apply_edit(files: dict[str, str], arguments: dict[str, Any], result: dict[str, Any]) -> int:
    if not result.get("ok"):
        return 0
    applied = 0
    for edit in arguments.get("edits", []):
        path = edit.get("path")
        old = edit.get("old_string")
        new = edit.get("new_string")
        if not all(isinstance(item, str) for item in (path, old, new)) or path not in files:
            continue
        count = files[path].count(old)
        if count == 0:
            continue
        if edit.get("all"):
            files[path] = files[path].replace(old, new)
            applied += count
        else:
            files[path] = files[path].replace(old, new, 1)
            applied += 1
    return applied


def retained_sizes(files: dict[str, str]) -> tuple[int, int, int]:
    playbook = len(files.get("playbook.md", "").encode())
    scratch = sum(
        len(content.encode())
        for path, content in files.items()
        if path.startswith("scratch/") and path.endswith(".py") and path != "scratch/__init__.py"
    )
    return playbook + scratch, playbook, scratch


def compressed_sizes(files: dict[str, str]) -> tuple[int, int, int]:
    def compressed(parts: list[tuple[str, str]]) -> int:
        if not parts:
            return 0
        bundle = b"".join(
            len(path.encode()).to_bytes(4, "big")
            + path.encode()
            + len(content.encode()).to_bytes(8, "big")
            + content.encode()
            for path, content in sorted(parts)
        )
        return len(zlib.compress(bundle, level=9))

    playbook = [("playbook.md", files["playbook.md"])] if "playbook.md" in files else []
    scratch = [
        (path, content)
        for path, content in files.items()
        if path.startswith("scratch/") and path.endswith(".py") and path != "scratch/__init__.py"
    ]
    return compressed(playbook + scratch), compressed(playbook), compressed(scratch)


def final_workspace_files(run_dir: Path) -> dict[str, str]:
    files: dict[str, str] = {}
    playbook = run_dir / "workspace" / "playbook.md"
    if playbook.exists():
        files["playbook.md"] = playbook.read_text(encoding="utf-8")
    scratch = run_dir / "workspace" / "scratch"
    if scratch.is_dir():
        for path in scratch.glob("*.py"):
            if path.name != "__init__.py":
                files[f"scratch/{path.name}"] = path.read_text(encoding="utf-8")
    return files


def reconstruct(run_dir: Path) -> tuple[list[Invocation], list[str]]:
    game = run_dir.parent.name
    run_id = run_dir.name
    transcript = list(jsonl(run_dir / "transcript.jsonl"))
    traces, raw_trace_count = trace_events(run_dir, transcript)
    levels = invocation_levels(run_dir / "workspace" / "log.txt")
    files: dict[str, str] = {}
    rows: list[Invocation] = []
    warnings: list[str] = []
    previous = 0
    previous_zlib = 0

    if raw_trace_count != len(transcript):
        warnings.append(f"{game}/{run_id}: {raw_trace_count} rooted traces for {len(transcript)} transcript records")

    for index, record in enumerate(transcript, start=1):
        events = traces[index - 1] if index <= len(traces) else []
        writes = 0
        edits = 0
        for event in events:
            payload = tool_payload(event)
            if payload is None:
                continue
            tool, arguments, result = payload
            if tool == "write" and apply_write(files, arguments, result):
                writes += 1
            elif tool == "edit":
                edits += apply_edit(files, arguments, result)

        retained, playbook, scratch = retained_sizes(files)
        retained_zlib, playbook_zlib, scratch_zlib = compressed_sizes(files)
        delta = retained - previous
        delta_zlib = retained_zlib - previous_zlib
        number = int(record.get("invocation", index))
        rows.append(
            Invocation(
                game=game,
                run_id=run_id,
                number=number,
                level=levels.get(number, rows[-1].level if rows else 1),
                retained_bytes=retained,
                playbook_bytes=playbook,
                scratch_bytes=scratch,
                retained_zlib_bytes=retained_zlib,
                playbook_zlib_bytes=playbook_zlib,
                scratch_zlib_bytes=scratch_zlib,
                delta_bytes=delta,
                positive_delta=max(0, delta),
                negative_delta=max(0, -delta),
                delta_zlib_bytes=delta_zlib,
                positive_delta_zlib=max(0, delta_zlib),
                negative_delta_zlib=max(0, -delta_zlib),
                context_tokens=int(record.get("context_tokens", 0)),
                fresh_session=not bool(record.get("resumed", False)),
                writes=writes,
                edits=edits,
            )
        )
        previous = retained
        previous_zlib = retained_zlib

    published = final_workspace_files(run_dir)
    for path in sorted(set(files) | set(published)):
        if files.get(path) != published.get(path):
            warnings.append(f"{game}/{run_id}: reconstructed {path} differs from published final file")
    return rows, warnings


def selected_run_dirs(root: Path, include_superseded: bool) -> list[Path]:
    if include_superseded:
        return sorted(path.parent for path in root.glob("*/*/transcript.jsonl"))
    dirs: list[Path] = []
    for game, run_id in SELECTED_RUN_IDS.items():
        run_dir = root / game / run_id
        if (run_dir / "transcript.jsonl").exists():
            dirs.append(run_dir)
    return dirs


def write_invocations(path: Path, rows: list[Invocation]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(Invocation.__dataclass_fields__))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def level_rows(rows: list[Invocation]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int], list[Invocation]] = defaultdict(list)
    for row in rows:
        grouped[(row.game, row.run_id, row.level)].append(row)
    result: list[dict[str, Any]] = []
    for (game, run_id, level), group in sorted(grouped.items()):
        result.append(
            {
                "game": game,
                "run_id": run_id,
                "level": level,
                "invocations": len(group),
                "start_bytes": group[0].retained_bytes - group[0].delta_bytes,
                "end_bytes": group[-1].retained_bytes,
                "min_bytes": min(row.retained_bytes for row in group),
                "max_bytes": max(row.retained_bytes for row in group),
                "positive_variation": sum(row.positive_delta for row in group),
                "negative_variation": sum(row.negative_delta for row in group),
                "start_zlib_bytes": group[0].retained_zlib_bytes - group[0].delta_zlib_bytes,
                "end_zlib_bytes": group[-1].retained_zlib_bytes,
                "min_zlib_bytes": min(row.retained_zlib_bytes for row in group),
                "max_zlib_bytes": max(row.retained_zlib_bytes for row in group),
                "positive_zlib_variation": sum(row.positive_delta_zlib for row in group),
                "negative_zlib_variation": sum(row.negative_delta_zlib for row in group),
                "fresh_sessions": sum(row.fresh_session for row in group),
                "max_context_tokens": max(row.context_tokens for row in group),
            }
        )
    return result


def write_dict_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)


def solved_checkpoints(rows: list[Invocation], runs_root: Path) -> list[SolvedCheckpoint]:
    grouped: dict[tuple[str, str], list[Invocation]] = defaultdict(list)
    for row in rows:
        grouped[(row.game, row.run_id)].append(row)
    result: list[SolvedCheckpoint] = []
    for (game, run_id), group in sorted(grouped.items()):
        ordered = sorted(group, key=lambda row: row.number)
        chosen: list[tuple[int, Invocation]] = []
        for current, following in zip(ordered, ordered[1:]):
            if following.level > current.level:
                chosen.append((following.level - 1, current))
        metrics = json.loads((runs_root / game / run_id / "metrics.json").read_text())
        completed = int(metrics["levels_completed"])
        last_completed = chosen[-1][0] if chosen else 0
        if completed > last_completed and ordered:
            chosen.append((completed, ordered[-1]))

        previous_raw = previous_zlib = previous_scratch_zlib = 0
        for index, (completed_levels, row) in enumerate(chosen):
            delta_raw = row.retained_bytes - previous_raw
            delta_zlib = row.retained_zlib_bytes - previous_zlib
            result.append(
                SolvedCheckpoint(
                    game=game,
                    run_id=run_id,
                    checkpoint=index,
                    completed_levels=completed_levels,
                    invocation=row.number,
                    retained_bytes=row.retained_bytes,
                    retained_zlib_bytes=row.retained_zlib_bytes,
                    playbook_zlib_bytes=row.playbook_zlib_bytes,
                    scratch_zlib_bytes=row.scratch_zlib_bytes,
                    delta_scratch_zlib_bytes=row.scratch_zlib_bytes
                    - previous_scratch_zlib,
                    delta_bytes=delta_raw,
                    delta_zlib_bytes=delta_zlib,
                    positive_delta_zlib=max(0, delta_zlib),
                    contraction_zlib=max(0, -delta_zlib),
                )
            )
            previous_raw = row.retained_bytes
            previous_zlib = row.retained_zlib_bytes
            previous_scratch_zlib = row.scratch_zlib_bytes
    return result


def summarize(checkpoints: list[SolvedCheckpoint]) -> dict[str, Any]:
    transitions = [row for row in checkpoints if row.checkpoint > 0]
    by_game: dict[str, list[SolvedCheckpoint]] = defaultdict(list)
    for row in checkpoints:
        by_game[row.game].append(row)
    local_peaks = local_troughs = 0
    for game_rows in by_game.values():
        marginals = [row.positive_delta_zlib for row in game_rows[1:]]
        differences = [b - a for a, b in zip(marginals, marginals[1:])]
        signs = [1 if value > 0 else -1 for value in differences if value]
        local_peaks += sum(a > 0 and b < 0 for a, b in zip(signs, signs[1:]))
        local_troughs += sum(a < 0 and b > 0 for a, b in zip(signs, signs[1:]))
    return {
        "runs": len(by_game),
        "solved_checkpoints": len(checkpoints),
        "between_solved_checkpoint_transitions": len(transitions),
        "measurement": "zlib-9 bytes of retained playbook plus substantive scratch Python",
        "interpretation": "descriptive-memory upper bound; not an executable-solver trajectory",
        "expansions": sum(row.delta_zlib_bytes > 0 for row in transitions),
        "contractions": sum(row.delta_zlib_bytes < 0 for row in transitions),
        "zero_changes": sum(row.delta_zlib_bytes == 0 for row in transitions),
        "games_with_retained_memory_contraction": sum(
            any(row.delta_zlib_bytes < 0 for row in game_rows[1:])
            for game_rows in by_game.values()
        ),
        "total_positive_zlib_bytes": sum(row.positive_delta_zlib for row in transitions),
        "total_contraction_zlib_bytes": sum(row.contraction_zlib for row in transitions),
        "median_positive_marginal_zlib_bytes": statistics.median(
            row.positive_delta_zlib for row in transitions
        ),
        "local_peaks_in_positive_marginal_sequence": local_peaks,
        "local_troughs_in_positive_marginal_sequence": local_troughs,
        "games_with_substantive_scratch_python": len(
            {row.game for row in checkpoints if row.scratch_zlib_bytes > 0}
        ),
        "scratch_python_checkpoint_transitions": sum(
            row.checkpoint > 0
            and (
                row.scratch_zlib_bytes > 0
                or row.delta_scratch_zlib_bytes != 0
            )
            for row in checkpoints
        ),
        "scratch_python_expansions": sum(
            row.checkpoint > 0 and row.delta_scratch_zlib_bytes > 0
            for row in checkpoints
        ),
        "scratch_python_contractions": sum(
            row.checkpoint > 0 and row.delta_scratch_zlib_bytes < 0
            for row in checkpoints
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs_root", type=Path, help="extracted Retrodict final-runs directory")
    parser.add_argument("--out-prefix", type=Path, default=Path("retrodict-retained"))
    parser.add_argument("--include-superseded", action="store_true")
    args = parser.parse_args()

    rows: list[Invocation] = []
    warnings: list[str] = []
    for run_dir in selected_run_dirs(args.runs_root, args.include_superseded):
        reconstructed, run_warnings = reconstruct(run_dir)
        rows.extend(reconstructed)
        warnings.extend(run_warnings)

    checkpoint_path = args.out_prefix.with_name(
        args.out_prefix.name + "-solved-checkpoint-memory.csv"
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoints = solved_checkpoints(rows, args.runs_root)
    write_dict_rows(checkpoint_path, [row.__dict__ for row in checkpoints])
    summary = summarize(checkpoints)
    summary_path = args.out_prefix.with_name(
        args.out_prefix.name + "-solved-checkpoint-memory.json"
    )
    summary_path.write_text(
        json.dumps(
            {
                "summary": summary,
                "rows": [row.__dict__ for row in checkpoints],
            },
            indent=2,
        )
        + "\n"
    )

    print(f"runs={len({(row.game, row.run_id) for row in rows})} invocations={len(rows)}")
    print(f"wrote {checkpoint_path}")
    print(f"wrote {summary_path}")
    print(json.dumps(summary, indent=2))
    for warning in warnings:
        print(f"warning: {warning}")


if __name__ == "__main__":
    main()
