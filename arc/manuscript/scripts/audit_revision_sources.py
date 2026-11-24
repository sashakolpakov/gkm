#!/usr/bin/env python3
"""Recompute the revision's static source audit from frozen Git blobs only.

No solver is imported or executed.  Only checkpoint JSON, snapshot metadata, and
legs.py/players.py/solve.py at the pinned historical revision are materialized.
The active campaign, environment implementations, transcripts, and probes are not read.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "arc"))
import audit_gkm_solved_checkpoints as checkpoints
import audit_marginal_literal_reuse as reuse
import generate_empirical_tables as tables
import reproduce_manuscript as reproduction

REVISION = reproduction.FROZEN_SOURCE_HISTORY_REVISION
PREFIX = reproduction.FROZEN_HISTORY_TREE + "/"
RELEASE = tables.RELEASE_COMMIT
ACQUISITION = "arc/crack_lab/releases/arc_agi3_gkm_v2_181/acquisition_source/"


def git(*args: str) -> bytes:
    return subprocess.check_output(
        ["git", "-C", str(REPO), *args], env=reproduction._git_environment()
    )


class Blobs:
    def __init__(self) -> None:
        self.process = subprocess.Popen(
            ["git", "-C", str(REPO), "cat-file", "--batch"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            env=reproduction._git_environment(),
        )

    def read(self, object_id: str) -> bytes:
        assert self.process.stdin is not None and self.process.stdout is not None
        self.process.stdin.write((object_id + "\n").encode())
        self.process.stdin.flush()
        header = self.process.stdout.readline().split()
        if len(header) != 3 or header[1] != b"blob":
            raise RuntimeError(f"not a frozen blob: {object_id}")
        data = self.process.stdout.read(int(header[2]))
        if self.process.stdout.read(1) != b"\n":
            raise RuntimeError("invalid git cat-file response")
        return data

    def close(self) -> None:
        assert self.process.stdin is not None
        self.process.stdin.close()
        if self.process.wait() != 0:
            raise RuntimeError("git cat-file failed")


def materialize(output: Path) -> int:
    tree = git("rev-parse", f"{REVISION}:{PREFIX[:-1]}").decode().strip()
    if tree != reproduction.FROZEN_SOURCE_HISTORY_TREE_SHA1:
        raise RuntimeError("frozen source-history tree identity mismatch")
    entries = {}
    for line in git("ls-tree", "-r", REVISION, "--", PREFIX[:-1]).splitlines():
        info, name = line.decode().split("\t", 1)
        mode, kind, oid = info.split()
        if kind == "blob" and mode in {"100644", "100755"}:
            entries[name] = oid
    phases = (checkpoints.DIRECT_WINNING_PHASES
              | set(checkpoints.RETAINED_PHASE_PRIORITY)
              | checkpoints.AUTO_SOLVE_WINNING_PHASES)
    selected = {name for name in entries
                if len(Path(name.removeprefix(PREFIX)).parts) == 2
                and name.endswith("/checkpoint.json")}
    for name in entries:
        if not name.endswith("/metadata.json") or "/wip_context/" not in name:
            continue
        attempt = Path(name).parent.name
        if not any(attempt.startswith(phase + "_") for phase in phases):
            continue
        selected.add(name)
        parent = name.removesuffix("metadata.json")
        selected.update(parent + "files/" + filename
                        for filename in checkpoints.SOURCE_FILES
                        if parent + "files/" + filename in entries)
    reader = Blobs()
    try:
        for name in sorted(selected):
            destination = output / name.removeprefix(PREFIX)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(reader.read(entries[name]))
    finally:
        reader.close()
    return len(selected)


def previous_reconstruction(game: Path, level: int, files_root: Path) -> dict:
    parent = checkpoints.retained_snapshot(game, level - 1)
    if parent is None:
        raise RuntimeError("missing retained parent")
    files = checkpoints.source_files(parent[1])
    called = checkpoints.auto_solve_leg((files_root / "players.py").read_bytes(), level)
    files["players.py"] += (
        f"\n\ndef play_level_{level}(env):\n    {called}(env)\n".encode()
    )
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cached = json.loads((REPO / "arc/audit_results/marginal-literal-reuse.json").read_text())
    with tempfile.TemporaryDirectory(prefix="gkm-frozen-source-audit-") as directory:
        root = Path(directory)
        count = materialize(root)
        rows = reuse.gkm_rows(root)
        payload = {**cached, "rows": (
            [dataclasses.asdict(row) for row in rows]
            + [row for row in cached["rows"] if row["system"] != "GKM"]
        )}
        payload["summary"] = {**cached["summary"]}
        payload["summary"]["systems"] = {**cached["summary"]["systems"],
                                         "GKM": reuse.system_summary(rows)}
        payload["summary"]["reuse_rule"] = (
            "direct static AST call site to a definition unchanged at the preceding "
            "winning checkpoint; runtime call execution is not measured"
        )
        no_debrief = []
        paid_debrief_exclusions = []
        inventory = {}
        for game in sorted(root.glob("*_legs")):
            reached = json.loads((game / "checkpoint.json").read_text())["reached"]
            inventory[game.name.removesuffix("_legs")] = reached
            for level in range(2, reached + 1):
                paid = checkpoints.latest_snapshot(game, level, {"after_auto_solve_debrief"})
                if paid is not None and checkpoints.exact_snapshot(game, level) is None:
                    paid_debrief_exclusions.append({
                        "game": game.name.removesuffix("_legs"), "level": level,
                        "snapshot": PREFIX + paid[1].parent.relative_to(root).as_posix(),
                        "reason": "paid debrief snapshot does not authenticate the pre-debrief workspace",
                    })
                candidate = checkpoints.latest_snapshot(game, level, {"auto_solve_debrief_skipped"})
                if candidate is None:
                    continue
                metadata, files_root = candidate
                actual = checkpoints.source_files(files_root)
                reconstructed = previous_reconstruction(game, level, files_root)
                no_debrief.append({
                    "game": game.name.removesuffix("_legs"), "level": level,
                    "snapshot": PREFIX + files_root.parent.relative_to(root).as_posix(),
                    "same_normalized_top_level_units_as_parent_reconstruction": (
                        reuse.normalized_top_level_units(actual)
                        == reuse.normalized_top_level_units(reconstructed)
                    ),
                    "actual_source_sha256": {name: hashlib.sha256(value).hexdigest()
                                              for name, value in sorted(actual.items())},
                })
        game = root / "dc22_legs"
        frozen = json.loads(git("show", f"{RELEASE}:{ACQUISITION}dc22_legs/checkpoint.json"))
        measurements = []
        previous = {"legs.py": 0, "players.py": 1}
        for level, record in enumerate(frozen["records"], 1):
            metadata, files = checkpoints.exact_snapshot(game, level)
            sizes = {name: tables.description_complexity(files[name].decode())
                     for name in previous}
            growth = sum(max(0, sizes[name] - previous[name]) for name in sizes)
            measurements.append({
                "level": level, "snapshot": PREFIX + "dc22_legs/wip_context/"
                + f"level_{level:02d}/" + metadata["attempt"] + "/files",
                "file_costs": sizes, "D": sum(sizes.values()),
                "source_sha256": {name: hashlib.sha256(value).hexdigest()
                                   for name, value in sorted(files.items())},
                "recorded_C": record["marginal_C"], "adjacent_source_growth": growth,
                "adjacent_minus_recorded": growth - record["marginal_C"],
                "adjacent_source_growth_uses_authenticated_predecessor": level > 1,
            })
            previous = sizes
        endpoint_matches_release = {
            name: files[name] == git("show", f"{RELEASE}:{ACQUISITION}dc22_legs/{name}")
            for name in checkpoints.SOURCE_FILES
        }
        report = {
            "history_revision": REVISION,
            "history_tree_sha1": reproduction.FROZEN_SOURCE_HISTORY_TREE_SHA1,
            "release_commit": RELEASE,
            "materialized_source_and_metadata_files": count,
            "scope": "static audit only; no solver, game, replay, or proposer executed",
            "source_units": "nonblank noncomment lines plus AST container-literal elements, per file",
            "recorded_charge_baseline": (
                "source in the working workspace before the winning harness cycle; "
                "not necessarily the preceding promoted source"
            ),
            "runtime_call_execution": "unmeasured",
            "all_no_debrief_auto_solve_checks": no_debrief,
            "paid_debrief_exclusions": paid_debrief_exclusions,
            "source_audit_scope": reproduction._source_audit_scope(
                {"claimed_inventory": inventory}, payload
            ),
            "dc22": {"initial_scaffold_D": 1, "recorded_total": 650,
                       "initial_scaffold_qualification": (
                           "the full growth total assumes the standard initial template; "
                           "the authenticated paired comparison excludes G1"
                       ),
                       "authenticated_paired_levels": [2, 6],
                       "paired_recorded_total": 601,
                       "paired_promoted_growth_total": 801,
                       "baseline_interpretation": (
                           "recorded workspace-local charges and adjacent-promotion growth "
                           "use different baselines; their difference is not a solver or counter defect"
                       ),
                       "recomputed_positive_growth_total": sum(m["adjacent_source_growth"] for m in measurements),
                       "endpoint_D": measurements[-1]["D"],
                       "endpoint_matches_frozen_release_acquisition_source": endpoint_matches_release,
                       "measurements": measurements,
                       "corrected_conditional_AST_zlib_bytes": {
                           str(row.completed_level): row.marginal_ast_zlib_bytes
                           for row in rows if row.game == "dc22"
                       }},
            "GKM_summary": payload["summary"]["systems"]["GKM"],
        }
    (args.output_dir / "marginal-literal-reuse.json").write_text(json.dumps(payload, indent=2) + "\n")
    (args.output_dir / "revision_source_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    summary = reproduction._summary(payload)
    audits = REPO / "arc/audit_results"
    reproduction._add_system_specific_stats(
        summary, opine_json=audits / "opine-solved-checkpoints.json",
        baseline_json=audits / "baseline1_gpt55_xhigh_solved_checkpoints.json",
        retrodict_json=audits / "retrodict-solved-checkpoint-memory.json",
    )
    reproduction._add_joint_row_stats(summary, payload)
    reproduction._write_generated_stats(summary, payload, args.output_dir)
    print(json.dumps({"materialized_files": count, "GKM": summary["GKM"],
                      "dc22_recorded_total": 650, "dc22_source_growth_total": report["dc22"]["recomputed_positive_growth_total"],
                      "dc22_endpoint_D": report["dc22"]["endpoint_D"]}, indent=2))


if __name__ == "__main__":
    main()
