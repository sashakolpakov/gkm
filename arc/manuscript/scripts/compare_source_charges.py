#!/usr/bin/env python3
"""Compare old GPT5.6 cycle-local C with adjacent retained-source growth.

Read immutable Git blobs and the compact uniform sidecars only. Solver sources
are bytes/AST input to the declared metric, never imports or executable code.
The primary comparison requires two historical, manifest-bound retained states.
Frozen retained-phase candidates form a separately reported extended comparison.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import generate_empirical_tables as tables

REPO = Path(__file__).resolve().parents[3]
HISTORY_REVISION = "4d0e42f34d7b1db8305f03d725528dfdefe22511"
HISTORY_PREFIX = "arc/crack_lab/agent_solutions"
RELEASE_REVISION = tables.RELEASE_COMMIT
ACQUISITION_PREFIX = "arc/crack_lab/releases/arc_agi3_gkm_v2_181/acquisition_source"
SOURCE_FILES = ("legs.py", "players.py", "solve.py")
COST_FILES = ("legs.py", "players.py")
PROMOTED_FILES = set(SOURCE_FILES) | {
    "checkpoint.json", "legs_log.md", "auto_solve_attempts.json", "provenance.json"
}
RETAINED_PHASES = (
    "after_debrief", "after_auto_solve_debrief", "auto_solve_debrief_skipped",
    "debrief_skipped_policy", "debrief_credit_out",
)
PLOT_GAMES = ["dc22", "wa30", "ls20"]
MANIFEST_TIER = "historical_promotion_manifest"
FALLBACK_TIER = "frozen_retained_phase_candidate"


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


class FrozenBlobs:
    """Stream one game's selected blobs at a time, recording their identities."""
    def __init__(self, repo: Path = REPO):
        self.repo = repo
        self.process = subprocess.Popen(
            ["git", "-C", str(repo), "cat-file", "--batch"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        )
        self.inputs: dict[tuple[str, str], dict] = {}

    def git(self, *args: str) -> bytes:
        return subprocess.check_output(["git", "-C", str(self.repo), *args])

    def entries(self, game: str) -> dict[str, str]:
        result = {}
        for line in self.git(
            "ls-tree", "-r", HISTORY_REVISION, "--", f"{HISTORY_PREFIX}/{game}_legs"
        ).splitlines():
            info, name = line.decode().split("\t", 1)
            mode, kind, oid = info.split()
            if kind == "blob" and mode in {"100644", "100755"}:
                result[name] = oid
        return result

    def read(self, revision: str, path: str, oid: str | None = None) -> bytes:
        assert self.process.stdin is not None and self.process.stdout is not None
        self.process.stdin.write(((oid or f"{revision}:{path}") + "\n").encode())
        self.process.stdin.flush()
        header = self.process.stdout.readline().split()
        if len(header) != 3 or header[1] != b"blob":
            raise ValueError(f"missing frozen blob: {revision}:{path}")
        raw = self.process.stdout.read(int(header[2]))
        if self.process.stdout.read(1) != b"\n":
            raise RuntimeError("invalid git cat-file response")
        self.inputs[(revision, path)] = {
            "revision": revision, "path": path,
            "git_blob": header[0].decode(), "sha256": sha256(raw),
        }
        return raw

    def close(self) -> None:
        assert self.process.stdin is not None
        self.process.stdin.close()
        if self.process.wait() != 0:
            raise RuntimeError("git cat-file failed")


@dataclass
class State:
    level: int
    tier: str
    files: dict[str, bytes]
    provenance: dict

    @property
    def costs(self) -> dict[str, int]:
        return {name: tables.description_complexity(self.files[name].decode())
                for name in COST_FILES}

    def identity(self) -> dict:
        return {**self.provenance, "tier": self.tier, "level": self.level,
                "source_sha256": {name: sha256(raw) for name, raw in self.files.items()},
                "file_costs": self.costs, "D_retained": sum(self.costs.values())}


def load_manifest_state(
    read: Callable[[str], bytes], manifest_path: str, game: str, level: int,
) -> tuple[State | None, dict]:
    """Validate a promotion and reject capsules that reconstruct historical code."""
    manifest_raw = read(manifest_path)
    manifest = json.loads(manifest_raw)
    if manifest.get("game") != game or manifest.get("level") != level:
        raise ValueError("promotion manifest boundary mismatch")
    if manifest.get("schema") == 1:
        if manifest.get("validated") is not True or manifest.get("taint_verdict") != "clean":
            raise ValueError("promotion manifest is not validated and clean")
    elif manifest.get("schema") != 2:
        raise ValueError("unsupported promotion manifest schema")
    parent = Path(manifest_path).parent
    declared = manifest["promoted_files_sha256"]
    if not set(SOURCE_FILES) | {"checkpoint.json"} <= declared.keys():
        raise ValueError("promotion manifest lacks source/checkpoint hashes")
    archived = {}
    for name, digest in declared.items():
        if name not in PROMOTED_FILES:
            raise ValueError(f"unexpected promoted file: {name}")
        raw = read((parent / "files" / name).as_posix())
        if sha256(raw) != digest:
            raise ValueError(f"promotion source/file hash mismatch: {name}")
        archived[name] = raw
    checkpoint = json.loads(archived["checkpoint.json"])
    if (checkpoint.get("game") != game or checkpoint.get("reached") != level
            or checkpoint.get("validated") is not True):
        raise ValueError("promotion checkpoint is not the validated boundary")
    proof = {
        "manifest_path": manifest_path, "manifest_sha256": sha256(manifest_raw),
        "manifest_schema": manifest["schema"],
        "source_paths": {name: (parent / "files" / name).as_posix()
                         for name in SOURCE_FILES},
        "checkpoint_sha256": sha256(archived["checkpoint.json"]),
        "checkpoint_current_record": [record for record in checkpoint["records"]
                                      if record["level"] == level],
    }
    if "provenance.json" in archived:
        certification = json.loads(archived["provenance.json"])
        proof["certification_provenance"] = certification
        proof["certification_provenance_sha256"] = sha256(archived["provenance.json"])
        if (certification.get("deterministic_reconstruction") is True
                or certification.get("historical_source_boundary") is not True
                or certification.get("retained_historical_phase") is not True
                or certification.get("posthoc_acquisition_marginal_admissible") is False):
            return None, {**proof, "tier": "excluded_reconstruction_capsule",
                          "reason": "certification provenance disclaims a historical retained source"}
    elif manifest["schema"] == 2:
        return None, {**proof, "tier": "excluded_unproven_historical_capsule",
                      "reason": "schema-2 certification lacks historical-retention provenance"}
    return State(level, MANIFEST_TIER, {name: archived[name] for name in SOURCE_FILES}, proof), proof


def load_phase_candidate(
    read: Callable[[str], bytes], entries: dict[str, str], root: str, game: str, level: int,
) -> tuple[State | None, dict]:
    candidates = []
    level_prefix = f"{root}/wip_context/level_{level:02d}/"
    for path in entries:
        if not path.startswith(level_prefix) or not path.endswith("/metadata.json"):
            continue
        attempt = Path(path).parent.name
        if not any(attempt.startswith(phase + "_") for phase in RETAINED_PHASES):
            continue
        raw = read(path)
        metadata = json.loads(raw)
        if (metadata.get("game") == game and metadata.get("level") == level
                and metadata.get("reached") == level and metadata.get("err") is None
                and metadata.get("phase") in RETAINED_PHASES):
            candidates.append((path, raw, metadata))
    if not candidates:
        return None, {"tier": "missing", "reason": "no historical promotion archive or clean retained-phase snapshot"}
    if len(candidates) != 1:
        raise ValueError(f"ambiguous retained-phase candidates: {game} L{level}")
    path, raw, metadata = candidates[0]
    source_paths = {name: (Path(path).parent / "files" / name).as_posix()
                    for name in SOURCE_FILES}
    if not all(name in entries for name in source_paths.values()):
        raise ValueError("retained-phase candidate lacks source files")
    proof = {
        "metadata_path": path, "metadata_sha256": sha256(raw),
        "phase": metadata["phase"], "created_at": metadata["created_at"],
        "reached": metadata["reached"], "source_paths": source_paths,
        "qualification": "frozen clean post-debrief phase; no independent promotion hash binds actual retention",
    }
    state = State(level, FALLBACK_TIER,
                  {name: read(source_paths[name]) for name in SOURCE_FILES}, proof)
    return state, proof


def growth_series(states: list[State | None], *, primary: bool) -> list[int | None]:
    """G1 and transitions touching an unknown/disallowed state stay missing."""
    if any(state is not None and state.level != index + 1
           for index, state in enumerate(states)):
        raise ValueError("states must retain their consecutive level positions")
    result: list[int | None] = [None] * len(states)
    for index in range(1, len(states)):
        previous, current = states[index - 1], states[index]
        if previous is None or current is None:
            continue
        if primary and any(state.tier != MANIFEST_TIER for state in (previous, current)):
            continue
        before, after = previous.costs, current.costs
        result[index] = sum(max(0, after[name] - before[name]) for name in COST_FILES)
    return result


def contiguous_segments(levels: list[int]) -> list[list[int]]:
    segments: list[list[int]] = []
    for level in levels:
        if segments and level == segments[-1][-1] + 1:
            segments[-1].append(level)
        else:
            segments.append([level])
    return segments


def series_metrics(segments: list[list[int]], values: dict[int, int]) -> dict:
    reversals = peaks = troughs = 0
    peak_levels = []
    trough_levels = []
    up_down = down_up = 0
    for segment in segments:
        sequence = [values[level] for level in segment]
        reversals += tables._sign_reversals(sequence)
        signs = [1 if b > a else -1 for a, b in zip(sequence, sequence[1:]) if a != b]
        up_down += sum(a > 0 > b for a, b in zip(signs, signs[1:]))
        down_up += sum(a < 0 < b for a, b in zip(signs, signs[1:]))
        for index in range(1, len(segment) - 1):
            left, middle, right = sequence[index - 1:index + 2]
            if left < middle > right:
                peaks += 1
                peak_levels.append(segment[index])
            if left > middle < right:
                troughs += 1
                trough_levels.append(segment[index])
    return {"direction_reversals": reversals, "up_to_down": up_down,
            "down_to_up": down_up, "strict_local_peaks": peaks,
            "strict_local_troughs": troughs, "strict_local_peak_levels": peak_levels,
            "strict_local_trough_levels": trough_levels}


def paired_metrics(recorded: list[int], growth: list[int | None]) -> dict:
    if len(recorded) != len(growth):
        raise ValueError("recorded and growth series must share a level inventory")
    levels = [index + 1 for index, value in enumerate(growth) if value is not None]
    segments = contiguous_segments(levels)
    C = dict(enumerate(recorded, 1))
    G = {index + 1: value for index, value in enumerate(growth) if value is not None}
    return {
        "levels": levels, "segments": segments,
        "ranges": [[segment[0], segment[-1]] for segment in segments],
        "recorded": series_metrics(segments, C), "growth": series_metrics(segments, G),
        "unequal_value_levels": [level for level in levels if C[level] != G[level]],
        "recorded_sum": sum(C[level] for level in levels),
        "growth_sum": sum(G[level] for level in levels),
    }


def verify_sidecar(game: str, read: Callable[[str], bytes], repo: Path) -> dict:
    root = repo / "arc/manuscript/artifact_history" / game
    raw = (root / "manifest.json").read_bytes()
    manifest = json.loads(raw)
    checked = []
    for promotion in manifest["promotions"]:
        level = promotion["through_level"]
        copied = root / f"level_{level:02d}" / promotion["clean_core_digest"]
        metadata = json.loads((copied / "promotion.json").read_bytes())
        if metadata != promotion:
            raise ValueError("uniform sidecar promotion metadata mismatch")
        original_root = f"{HISTORY_PREFIX}/{game}_legs/promotion_evidence/level_{level:02d}"
        original_manifest = read(original_root + "/manifest.json")
        if sha256(original_manifest) != promotion["source"]["manifest_sha256"]:
            raise ValueError("uniform sidecar manifest hash mismatch")
        for name, digest in promotion["files"].items():
            if name not in PROMOTED_FILES:
                raise ValueError("unexpected uniform sidecar file")
            content = (copied / "files" / name).read_bytes()
            if sha256(content) != digest or content != read(original_root + "/files/" + name):
                raise ValueError("uniform sidecar source hash mismatch")
        checked.append(level)
    return {"manifest_path": str(root.relative_to(repo) / "manifest.json"),
            "manifest_sha256": sha256(raw), "verified_levels": checked,
            "ledger": [item["marginal_C"] for item in manifest["ledger"]]}


def aggregate(games: list[dict], scope: str) -> dict:
    pairs = [game["paired"][scope] for game in games]
    return {
        "paired_values": sum(len(pair["levels"]) for pair in pairs),
        "games_with_paired_values": sum(bool(pair["levels"]) for pair in pairs),
        "games_with_three_consecutive_values": sum(
            any(len(segment) >= 3 for segment in pair["segments"]) for pair in pairs),
        "recorded_direction_reversals": sum(pair["recorded"]["direction_reversals"] for pair in pairs),
        "growth_direction_reversals": sum(pair["growth"]["direction_reversals"] for pair in pairs),
        "games_with_recorded_reversal": sum(pair["recorded"]["direction_reversals"] > 0 for pair in pairs),
        "games_with_growth_reversal": sum(pair["growth"]["direction_reversals"] > 0 for pair in pairs),
        "games_with_reversals_in_both": sum(pair["recorded"]["direction_reversals"] > 0
                                             and pair["growth"]["direction_reversals"] > 0 for pair in pairs),
        "recorded_strict_peaks": sum(pair["recorded"]["strict_local_peaks"] for pair in pairs),
        "growth_strict_peaks": sum(pair["growth"]["strict_local_peaks"] for pair in pairs),
        "recorded_strict_troughs": sum(pair["recorded"]["strict_local_troughs"] for pair in pairs),
        "growth_strict_troughs": sum(pair["growth"]["strict_local_troughs"] for pair in pairs),
        "unequal_values": sum(len(pair["unequal_value_levels"]) for pair in pairs),
    }


def build_report(repo: Path = REPO) -> dict:
    blobs = FrozenBlobs(repo)
    games = []
    sidecars = []
    physical_manifests = historical_manifests = phase_candidates = 0
    excluded = []
    try:
        history_tree = blobs.git("rev-parse", f"{HISTORY_REVISION}:{HISTORY_PREFIX}").decode().strip()
        for game in tables.AUTHORITATIVE_LEVELS:
            entries = blobs.entries(game)
            root = f"{HISTORY_PREFIX}/{game}_legs"
            def read(path: str) -> bytes:
                return blobs.read(HISTORY_REVISION, path, entries[path])
            def acquisition(name: str) -> bytes:
                return blobs.read(RELEASE_REVISION, f"{ACQUISITION_PREFIX}/{game}_legs/{name}")
            checkpoint_raw = acquisition("checkpoint.json")
            checkpoint = json.loads(checkpoint_raw)
            if json.loads(read(root + "/checkpoint.json")) != checkpoint:
                raise ValueError("history and sealed acquisition checkpoints differ")
            reached = checkpoint["reached"]
            records = checkpoint["records"]
            if (checkpoint.get("game") != game or checkpoint.get("validated") is not True
                    or [record["level"] for record in records] != list(range(1, reached + 1))
                    or any(record.get("reached") is not True for record in records)):
                raise ValueError("invalid sealed acquisition ledger")
            recorded = [record["marginal_C"] for record in records]
            if sum(recorded) != checkpoint["total_marginal_C"]:
                raise ValueError("sealed acquisition ledger sum mismatch")
            states = []
            provenance = []
            for level in range(1, reached + 1):
                manifest_path = f"{root}/promotion_evidence/level_{level:02d}/manifest.json"
                if manifest_path in entries:
                    physical_manifests += 1
                    state, proof = load_manifest_state(read, manifest_path, game, level)
                    manifest = json.loads(read(manifest_path))
                    parent = manifest.get("parent_manifest")
                    parent_path = None
                    if parent:
                        parent_path = f"{root}/{parent['path'] if isinstance(parent, dict) else parent}"
                        expected = parent["sha256"] if isinstance(parent, dict) else manifest["parent_manifest_sha256"]
                        if sha256(read(parent_path)) != expected:
                            raise ValueError("broken frozen promotion manifest chain")
                    if (state is not None and states and states[-1] is not None
                            and state.tier == states[-1].tier == MANIFEST_TIER
                            and parent_path != states[-1].provenance["manifest_path"]):
                        raise ValueError("promotion parent is not the selected adjacent retained manifest")
                    if state:
                        historical_manifests += 1
                        current = state.provenance["checkpoint_current_record"]
                        if (len(current) != 1 or current[0]["marginal_C"] != recorded[level - 1]):
                            raise ValueError("promotion and acquisition current charge differ")
                    else:
                        excluded.append({"game": game, "level": level, **proof})
                else:
                    state, proof = load_phase_candidate(read, entries, root, game, level)
                    phase_candidates += int(state is not None)
                states.append(state)
                provenance.append(state.identity() if state else {"level": level, **proof})
            primary = growth_series(states, primary=True)
            extended = growth_series(states, primary=False)
            endpoint = {name: acquisition(name) for name in SOURCE_FILES}
            endpoint_identity = {name: sha256(raw) for name, raw in endpoint.items()}
            endpoint_match = states[-1].files == endpoint if states[-1] else None
            if endpoint_match is False:
                raise ValueError(f"historical retained endpoint differs from acquisition source: {game}")
            conditional_primary = primary.copy()
            conditional_extended = extended.copy()
            if states[0]:
                costs = states[0].costs
                g1 = max(0, costs["legs.py"]) + max(0, costs["players.py"] - 1)
                conditional_extended[0] = g1
                if states[0].tier == MANIFEST_TIER:
                    conditional_primary[0] = g1
            row = {
                "game": game, "reached": reached, "levels": list(range(1, reached + 1)),
                "recorded_C": recorded, "primary_G": primary, "extended_G": extended,
                "D_retained": [sum(state.costs.values()) if state else None for state in states],
                "source_provenance": provenance,
                "paired": {"primary": paired_metrics(recorded, primary),
                           "extended": paired_metrics(recorded, extended)},
                "conditional_standard_scaffold": {
                    "qualification": "assumes initial costs legs.py=0, players.py=1; actual initial workspace bytes not authenticated",
                    "primary_G": conditional_primary, "extended_G": conditional_extended,
                    "primary_paired": paired_metrics(recorded, conditional_primary),
                    "extended_paired": paired_metrics(recorded, conditional_extended),
                },
                "sealed_acquisition_endpoint": {
                    "source_sha256": endpoint_identity,
                    "file_costs": {name: tables.description_complexity(endpoint[name].decode()) for name in COST_FILES},
                    "matches_known_historical_retained_endpoint": endpoint_match,
                    "used_to_impute_missing_history": False,
                },
            }
            if game in {"wa30", "ls20"}:
                sidecar = verify_sidecar(game, read, repo)
                if sidecar["ledger"] != recorded:
                    raise ValueError("uniform sidecar and acquisition ledger differ")
                sidecars.append({"game": game, **sidecar})
            games.append(row)
        inputs = sorted(blobs.inputs.values(), key=lambda item: (item["revision"], item["path"]))
    finally:
        blobs.close()
    missing = [{"game": game["game"], "level": entry["level"], "tier": entry["tier"]}
               for game in games for entry in game["source_provenance"]
               if entry["tier"] not in {MANIFEST_TIER, FALLBACK_TIER}]
    return {
        "schema": 1,
        "dataset": {"name": "old GPT5.6 frozen 181-level campaign",
                    "history_revision": HISTORY_REVISION, "history_prefix": HISTORY_PREFIX,
                    "history_tree_sha1": history_tree, "release_revision": RELEASE_REVISION,
                    "acquisition_prefix": ACQUISITION_PREFIX},
        "scope": {
            "execution": "static source/AST and metadata only; no solver/game execution, implementation inspection, or active/new-campaign reads",
            "recorded_C": "sealed harness-cycle/workspace-local recorded charge",
            "G": "sum of positive per-file d differences between consecutive retained sources, legs.py plus players.py",
            "d": "generate_empirical_tables.description_complexity: nonblank noncomment lines plus AST container members",
            "primary": "both adjacent states must be historical source-hash-bound promotions",
            "extended": "adds unique frozen clean post-debrief phase candidates, without claiming independent promotion authentication",
            "initial_scaffold": "G1 excluded from both unconditional comparisons; scaffold assumption is separate",
            "reversals": "existing _sign_reversals; flats ignored only within contiguous common known level segments; no gap bridging",
            "extrema": "strict inequalities at three consecutive known levels on the common paired domain",
            "winning_source_audit": "independent: retained post-debrief source is used here, never substituted by pre-debrief winning source",
            "metric_source_sha256": sha256((repo / "arc/manuscript/scripts/generate_empirical_tables.py").read_bytes()),
        },
        "plot_games": PLOT_GAMES,
        "summary": {
            "games": len(games), "recorded_levels": sum(game["reached"] for game in games),
            "physical_promotion_manifests": physical_manifests,
            "historical_manifest_bound_states": historical_manifests,
            "excluded_certification_capsules": len(excluded),
            "frozen_retained_phase_candidate_states": phase_candidates,
            "missing_historical_retained_states": missing,
            "primary": aggregate(games, "primary"), "extended": aggregate(games, "extended"),
            "known_retained_endpoints_matching_acquisition": sum(
                game["sealed_acquisition_endpoint"]["matches_known_historical_retained_endpoint"] is True for game in games),
        },
        "games": games, "excluded_capsules": excluded,
        "uniform_sidecars": sidecars,
        "input_contract": {
            "history_patterns_added_to_existing_materializer": [
                "{game}_legs/promotion_evidence/level_*/manifest.json",
                "{game}_legs/promotion_evidence/level_*/files/{legs.py,players.py,solve.py,checkpoint.json,legs_log.md,auto_solve_attempts.json,provenance.json}",
            ],
            "history_fallback_patterns": [
                "{game}_legs/wip_context/level_*/*/metadata.json for declared retained phases",
                "same phase/files/{legs.py,players.py,solve.py}",
            ],
            "release_patterns": ["{game}_legs/{checkpoint.json,legs.py,players.py,solve.py}"],
            "sidecar_patterns": ["{wa30,ls20}/manifest.json", "{wa30,ls20}/level_*/*/promotion.json",
                                 "{wa30,ls20}/level_*/*/files/* declared core files"],
            "git_inputs": inputs,
        },
    }


def format_ranges(ranges: list[list[int]], *, tex: bool = False) -> str:
    dash = "--" if tex else "–"
    return ", ".join(str(a) if a == b else f"{a}{dash}{b}" for a, b in ranges) or "none"


def render_markdown(report: dict) -> str:
    summary = report["summary"]
    primary = summary["primary"]
    extended = summary["extended"]
    lines = [
        "<!-- Generated by scripts/compare_source_charges.py; do not edit. -->", "",
        "Old GPT5.6 recorded charges versus adjacent retained-source growth", "",
        "C is the sealed cycle-local charge. G is positive per-file growth in retained legs.py and players.py under the same line-plus-literal metric. Both reversal counts use the identical contiguous known level domain; G1 is excluded. Flats are ignored inside each segment, and gaps are never bridged.", "",
        f"Primary: {summary['historical_manifest_bound_states']} historical manifest-bound states, {primary['paired_values']} paired growth values across {primary['games_with_paired_values']} games; C/G direction reversals {primary['recorded_direction_reversals']}/{primary['growth_direction_reversals']}. Both series reverse direction in {primary['games_with_reversals_in_both']} games.", "",
        f"Extended: adds {summary['frozen_retained_phase_candidate_states']} frozen retained-phase candidates, which lack independent promotion authentication; {extended['paired_values']} paired values, C/G reversals {extended['recorded_direction_reversals']}/{extended['growth_direction_reversals']}.", "",
        f"Excluded: {summary['excluded_certification_capsules']} tr87 reconstruction capsules explicitly disclaim historical retained-source provenance. Missing states: "
        + ", ".join(f"{item['game']} L{item['level']}" for item in summary["missing_historical_retained_states"]) + ".", "",
        "| Game | Historical manifest / phase candidate states | Primary paired levels | C reversals | G reversals | Extended paired levels | Extended C/G reversals | Unequal primary values |",
        "| --- | ---: | --- | ---: | ---: | --- | ---: | --- |",
    ]
    for game in report["games"]:
        p, e = game["paired"]["primary"], game["paired"]["extended"]
        tiers = [entry["tier"] for entry in game["source_provenance"]]
        different = ", ".join(f"L{level}" for level in p["unequal_value_levels"]) or "none"
        lines.append(f"| {game['game']} | {tiers.count(MANIFEST_TIER)} / {tiers.count(FALLBACK_TIER)} | {format_ranges(p['ranges'])} | {p['recorded']['direction_reversals']} | {p['growth']['direction_reversals']} | {format_ranges(e['ranges'])} | {e['recorded']['direction_reversals']}/{e['growth']['direction_reversals']} | {different} |")
    lines.extend(["", "The JSON retains every source SHA-256, Git blob identity, provenance tier, paired segment and strict local extremum. Its conditional standard-scaffold arrays are secondary and do not change the primary counts. The acquisition endpoint is recorded separately and never fills a history gap.", ""])
    return "\n".join(lines)


def render_tex(report: dict) -> str:
    selected = {game["game"]: game for game in report["games"]}
    lines = [
        "% Generated by scripts/compare_source_charges.py; do not edit.",
        r"\begin{table}[H]", r"\centering", r"\small",
        r"\begin{tabular}{lcrr}", r"\toprule",
        r"Game & Paired levels & $C$ reversals & $G$ reversals \\", r"\midrule",
    ]
    for name in report["plot_games"]:
        pair = selected[name]["paired"]["primary"]
        lines.append(f"{name} & {format_ranges(pair['ranges'], tex=True)} & {pair['recorded']['direction_reversals']} & {pair['growth']['direction_reversals']} " + r"\\")
    lines.extend([
        r"\bottomrule", r"\end{tabular}",
        r"\caption{Recorded cycle-local charges $C$ and growth $G$ between historical retained sources, using the same line-plus-literal source metric. Counts use identical consecutive level ranges with both adjacent sources bound to historical promotion manifests. $G_1$ is excluded because the initial workspace source is not authenticated; flat differences are ignored within each known segment.}",
        r"\label{tab:charge-growth-comparison}", r"\end{table}", "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=REPO / "arc/manuscript/generated")
    args = parser.parse_args()
    report = build_report()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir / "source_charge_comparison"
    stem.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    stem.with_suffix(".md").write_text(render_markdown(report))
    stem.with_suffix(".tex").write_text(render_tex(report))
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
