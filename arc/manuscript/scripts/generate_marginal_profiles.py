#!/usr/bin/env python3
"""Publish conditional novelty at exact winning boundaries, without filling gaps."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def direction_reversals(values: list[int | None]) -> int:
    """Ignore flats within known segments, never across an unknown boundary."""
    previous = None
    direction = 0
    reversals = 0
    for value in values:
        if value is None:
            previous, direction = None, 0
            continue
        if previous is not None and value != previous:
            new_direction = 1 if value > previous else -1
            reversals += int(direction != 0 and direction != new_direction)
            direction = new_direction
        previous = value
    return reversals


def build_profiles(audit: dict, release_table: dict) -> dict:
    rows = [r for r in audit["rows"] if r["system"] == "GKM"]
    boundaries = {}
    for row in rows:
        key = (row["game"], row["completed_level"])
        if key in boundaries:
            raise ValueError(f"duplicate winning boundary: {key}")
        if row["source_checkpoint_exact"] is not True:
            raise ValueError(f"source is not exact at {key}")
        boundaries[key] = row

    games = []
    comparisons = decreases = sharp = coupled = static = 0
    release_games = release_table["games"]
    if len({g["game"] for g in release_games}) != len(release_games):
        raise ValueError("duplicate release game")
    for metadata in sorted(release_games, key=lambda r: r["game"]):
        game, reached, total = metadata["game"], metadata["reached"], metadata["total_levels"]
        levels = list(range(1, total + 1))
        values, exact, calls, coupled_levels = [], [], [], []
        for level in levels:
            row = boundaries.get((game, level))
            exact.append(row is not None)
            if row is None:
                values.append(None)
                continue
            if level > reached:
                raise ValueError(f"source exceeds claimed frontier: {game} L{level}")
            adjacent = (game, level - 1) in boundaries
            if row["exact_adjacent_transition"] != adjacent:
                raise ValueError(f"inconsistent adjacent source: {game} L{level}")
            value = row["marginal_ast_zlib_bytes"]
            if (level == 1 or adjacent) != (value is not None):
                raise ValueError(f"marginal bridges missing source: {game} L{level}")
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError(f"invalid marginal: {game} L{level}")
            values.append(value)
            previous = values[-2] if level > 1 else None
            if row["previous_level_marginal_ast_zlib_bytes"] != previous:
                raise ValueError(f"comparison bridges missing marginal: {game} L{level}")
            if value is not None and previous is not None:
                comparisons += 1
                decreases += int(value < previous)
                sharp += int(previous > 0 and value * 2 <= previous)
            if row["hard_literal_reuse_witness"]:
                calls.append(level)
                static += 1
            if row["sharp_drop_with_literal_reuse"]:
                if previous is None or previous == 0 or value is None or value * 2 > previous or level not in calls:
                    raise ValueError(f"invalid coupled source case: {game} L{level}")
                coupled_levels.append(level)
                coupled += 1
        games.append({"game": game, "reached": reached, "total_levels": total,
                      "levels": levels, "M_bytes": values,
                      "exact_winning_source": exact,
                      "static_unchanged_leg_call_levels": calls,
                      "sharp_drop_with_static_call_levels": coupled_levels,
                      "direction_reversals": direction_reversals(values)})
    if set(boundaries) - {(g["game"], l) for g in games for l in g["levels"]}:
        raise ValueError("source boundary outside the release")
    summary = {"exact_winning_checkpoints": len(rows),
               "transitions_with_level_to_level_marginal_comparison": comparisons,
               "marginal_decreases": decreases,
               "sharp_half_or_more_marginal_drops": sharp,
               "hard_literal_world_model_reuse_witnesses": static}
    expected = audit["summary"]["systems"]["GKM"]
    if any(value != expected[key] for key, value in summary.items()):
        raise ValueError("profile counts differ from the source audit")
    if coupled != len(expected["sharp_drops_with_literal_reuse"]):
        raise ValueError("coupled profile count differs from the source audit")
    summary.update({"sharp_drops_with_static_call": coupled,
                    "games_with_direction_reversals": sum(g["direction_reversals"] > 0 for g in games),
                    "direction_reversals": sum(g["direction_reversals"] for g in games)})
    return {"schema": 1, "dataset": "frozen GPT-5.6 campaign",
            "measurement": audit["summary"]["measurement"],
            "initial_boundary": "M1 describes the initial winning program against an empty AST dictionary; it is not growth from an authenticated earlier promotion",
            "missing_values": "unknown winning source, missing predecessor, or unclaimed level; never zero-filled or connected across gaps",
            "interpretation": "conditional AST novelty proxy, not ideal K, a complete decoding code, runtime reuse, discovery cost, or an admission rule",
            "summary": summary, "games": games}


def render_markdown(payload: dict) -> str:
    lines = ["<!-- Generated by scripts/generate_marginal_profiles.py; do not edit. -->", "",
             "Conditional marginal-complexity proxy at exact winning boundaries", "",
             "M is compressed normalized-AST novelty, in bytes. Dashes are unknown; gaps are not bridged. "
             "M1 uses an empty dictionary. Static calls do not establish runtime execution.", "",
             "| Game | M by level (compressed bytes) | Direction reversals | Sharp drop + static call |",
             "| --- | --- | ---: | --- |"]
    for game in payload["games"]:
        values = ", ".join("—" if v is None else str(v) for v in game["M_bytes"])
        coupled = ", ".join(f"L{l}" for l in game["sharp_drop_with_static_call_levels"]) or "none"
        lines.append(f"| {game['game']} | {values} | {game['direction_reversals']} | {coupled} |")
    lines.extend(["", "Summary: " + json.dumps(payload["summary"], sort_keys=True), ""])
    return "\n".join(lines)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, default=root.parent / "audit_results/marginal-literal-reuse.json")
    parser.add_argument("--release-table", type=Path, default=root / "generated/marginal_complexity_by_level.json")
    parser.add_argument("--output-dir", type=Path, default=root / "generated")
    args = parser.parse_args()
    audit_bytes, table_bytes = args.audit.read_bytes(), args.release_table.read_bytes()
    payload = build_profiles(json.loads(audit_bytes), json.loads(table_bytes))
    payload["input_sha256"] = {"source_audit": hashlib.sha256(audit_bytes).hexdigest(),
                               "release_table": hashlib.sha256(table_bytes).hexdigest()}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "conditional_marginal_profiles.json").write_text(json.dumps(payload, indent=2) + "\n")
    (args.output_dir / "conditional_marginal_profiles.md").write_text(render_markdown(payload))
    print(json.dumps(payload["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
