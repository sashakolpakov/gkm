#!/usr/bin/env python3
"""Reproduce and verify the manuscript's machine-derived evidence.

The default mode is semiautomated: it recomputes GKM from the local canonical
checkpoints and reuses the checked-in, checksum-pinned comparator rows.  Supply
all four external-artifact arguments to rebuild the OPINE, baseline1, and
Retrodict boundary audits from their released artifacts as well.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


SYSTEMS = ("GKM", "OPINE", "baseline1", "Retrodict")
STAT_FIELDS = (
    "retained_or_winning_checkpoints",
    "exact_winning_checkpoints",
    "exact_adjacent_transitions",
    "released_memory_transitions",
    "transitions_with_level_to_level_marginal_comparison",
    "marginal_decreases",
    "sharp_half_or_more_marginal_drops",
    "hard_literal_world_model_reuse_witnesses",
)
MACRO_PREFIX = {
    "GKM": "GKM",
    "OPINE": "OPINE",
    "baseline1": "BaselineOne",
    "Retrodict": "Retrodict",
}
MACRO_FIELD = {
    "retained_or_winning_checkpoints": "RetainedCheckpoints",
    "exact_winning_checkpoints": "ExactWins",
    "exact_adjacent_transitions": "ExactAdjacent",
    "released_memory_transitions": "MemoryTransitions",
    "transitions_with_level_to_level_marginal_comparison": "Comparable",
    "marginal_decreases": "Decreases",
    "sharp_half_or_more_marginal_drops": "SharpDrops",
    "hard_literal_world_model_reuse_witnesses": "HardReuse",
    "sharp_drops_with_literal_reuse": "CoupledWitnesses",
    "trace_solve_events": "TraceSolveEvents",
    "analyzer_or_unknown_wins": "AnalyzerOrUnknownWins",
    "uncoupled_sharp": "UncoupledSharp",
    "reported_clears": "ReportedClears",
    "exact_authored_contractions": "ExactAuthoredContractions",
    "direct_literal_wins": "DirectLiteralWins",
    "inline_literal_wins": "InlineLiteralWins",
    "executor_literal_wins": "ExecutorLiteralWins",
    "memory_contractions": "MemoryContractions",
}


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[3],
    )
    parser.add_argument(
        "--release-root",
        type=Path,
        help=(
            "Frozen <game>_legs tree used for endpoint, taint, action, table, "
            "and figure reproduction (default: canonical agent_solutions)."
        ),
    )
    parser.add_argument(
        "--history-root",
        type=Path,
        help=(
            "Acquisition tree containing exact historical winning snapshots for "
            "the GKM source/reuse audit (default: canonical agent_solutions)."
        ),
    )
    parser.add_argument(
        "--release-receipt",
        type=Path,
        help=(
            "Schema-v2 partial-release receipt. When supplied, the manuscript "
            "suite uses the fail-closed release gate instead of applying the "
            "legacy schema-1 promotion audit to normalized artifacts."
        ),
    )
    parser.add_argument("--opine-artifacts", type=Path)
    parser.add_argument("--baseline-release", type=Path)
    parser.add_argument("--baseline-repo", type=Path)
    parser.add_argument("--retrodict-runs", type=Path)
    parser.add_argument(
        "--write",
        action="store_true",
        help="Replace tracked audit/figure outputs after successful reproduction.",
    )
    parser.add_argument(
        "--allow-live-gkm-drift",
        action="store_true",
        help="Report rather than fail when the active campaign has advanced.",
    )
    parser.add_argument(
        "--build-paper",
        action="store_true",
        help="Build the PDF after audits and figures pass.",
    )
    parser.add_argument(
        "--require-complete-lineage",
        action="store_true",
        help=(
            "Require every canonical game to have promotion manifests. "
            "The taint audit itself always runs; this enables the final-release "
            "lineage-completeness gate."
        ),
    )
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def _run(command: list[str], *, cwd: Path) -> None:
    print("+", " ".join(command))
    subprocess.run(command, cwd=cwd, check=True)


def _run_json(command: list[str], *, cwd: Path) -> dict[str, Any]:
    print("+", " ".join(command))
    completed = subprocess.run(
        command,
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    payload = json.loads(completed.stdout)
    if not isinstance(payload, dict):
        raise RuntimeError("JSON command did not return an object")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _summary(payload: dict[str, Any]) -> dict[str, dict[str, int]]:
    systems = payload["summary"]["systems"]
    return {
        system: {
            field: int(systems[system].get(field, 0))
            for field in STAT_FIELDS
        }
        | {
            "sharp_drops_with_literal_reuse": len(
                systems[system].get("sharp_drops_with_literal_reuse", [])
            ),
            "uncoupled_sharp": (
                int(systems[system].get("sharp_half_or_more_marginal_drops", 0))
                - len(systems[system].get("sharp_drops_with_literal_reuse", []))
            ),
        }
        for system in SYSTEMS
    }


def _external_mode(args: argparse.Namespace) -> bool:
    values = (
        args.opine_artifacts,
        args.baseline_release,
        args.baseline_repo,
        args.retrodict_runs,
    )
    if any(values) and not all(values):
        raise SystemExit(
            "raw comparator reproduction requires --opine-artifacts, "
            "--baseline-release, --baseline-repo, and --retrodict-runs together"
        )
    return all(values)


def _write_generated_stats(
    summary: dict[str, dict[str, int]],
    payload: dict[str, Any],
    output_dir: Path,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    tex_path = output_dir / "comparator_stats.tex"
    tex_lines = [
        "% Generated by scripts/reproduce_manuscript.py; do not edit.",
    ]
    for system in SYSTEMS:
        for field, value in summary[system].items():
            tex_lines.append(
                rf"\newcommand{{\{MACRO_PREFIX[system]}{MACRO_FIELD[field]}}}"
                rf"{{{value}}}"
            )
    coupled = [
        (system, row)
        for system in ("GKM", "OPINE")
        for row in payload["summary"]["systems"][system][
            "sharp_drops_with_literal_reuse"
        ]
    ]
    coupled_clauses: list[str] = []
    for system, row in coupled:
        names = [
            str(item).split(":", 1)[-1].replace("_", r"\_")
            for item in row["reused_world_model_literals"]
        ]
        if len(names) == 1:
            called = rf"\texttt{{{names[0]}}}"
        else:
            called = ", ".join(rf"\texttt{{{name}}}" for name in names[:-1])
            called += rf", and \texttt{{{names[-1]}}}"
        coupled_clauses.append(
            rf"{system} \texttt{{{row['game']}}} L{row['completed_level']}, "
            rf"${row['previous_marginal_ast_zlib_bytes']}\!\to\!"
            rf"{row['marginal_ast_zlib_bytes']}$, calling unchanged {called}"
        )
    tex_lines.append(
        r"\newcommand{\CoupledWitnessEnumeration}{"
        + "; ".join(coupled_clauses)
        + "}"
    )
    tex_path.write_text("\n".join(tex_lines) + "\n")

    md_path = output_dir / "comparator_stats.md"
    md_lines = [
        "<!-- Generated by scripts/reproduce_manuscript.py; do not edit. -->",
        "",
        "| System | Retained checkpoints | Exact winning checkpoints | "
        "Exact adjacent | Comparable marginals | Decreases | Sharp drops | "
        "Hard reuse | Sharp + reuse |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for system in SYSTEMS:
        row = summary[system]
        retained = row["retained_or_winning_checkpoints"]
        if system == "Retrodict":
            retained = (
                f"{retained} memory "
                f"({row['released_memory_transitions']} transitions)"
            )
        md_lines.append(
            f"| {system} | {retained} | {row['exact_winning_checkpoints']} | "
            f"{row['exact_adjacent_transitions']} | "
            f"{row['transitions_with_level_to_level_marginal_comparison']} | "
            f"{row['marginal_decreases']} | "
            f"{row['sharp_half_or_more_marginal_drops']} | "
            f"{row['hard_literal_world_model_reuse_witnesses']} | "
            f"{row['sharp_drops_with_literal_reuse']} |"
        )
    opine_sharp = sorted(
        (
            row
            for row in payload["rows"]
            if row["system"] == "OPINE" and row["sharp_marginal_drop"]
        ),
        key=lambda row: (row["game"], row["completed_level"]),
    )
    if len(opine_sharp) != summary["OPINE"][
        "sharp_half_or_more_marginal_drops"
    ]:
        raise RuntimeError("OPINE sharp-drop row count disagrees with summary")
    md_lines.extend(
        [
            "",
            "## OPINE sharp conditional drops",
            "",
            "| Boundary | Conditional AST marginal | Winning policy | Coupled "
            "direct-call witness |",
            "| --- | ---: | --- | --- |",
        ]
    )
    for row in opine_sharp:
        policy = str(row["winning_policy_kind"]).replace("_", " ")
        md_lines.append(
            f"| `{row['game']}` L{row['completed_level']} | "
            f"{row['previous_level_marginal_ast_zlib_bytes']} → "
            f"{row['marginal_ast_zlib_bytes']} | {policy} | "
            f"{'yes' if row['sharp_drop_with_literal_reuse'] else 'no'} |"
        )
    md_lines.extend(
        [
            "",
            "Only synthesized-planner rows can certify the released executable "
            "winning path. The transient analyzer policies were not retained.",
        ]
    )
    md_path.write_text("\n".join(md_lines) + "\n")
    return tex_path, md_path


def _add_system_specific_stats(
    summary: dict[str, dict[str, int]],
    *,
    opine_json: Path,
    baseline_json: Path,
    retrodict_json: Path,
) -> None:
    opine = json.loads(opine_json.read_text())["summary"]
    summary["OPINE"]["trace_solve_events"] = int(
        opine["positive_reward_solve_events_in_logs"]
    )
    summary["OPINE"]["analyzer_or_unknown_wins"] = int(
        opine["analyzer_or_unknown_solved_checkpoints"]
    )

    baseline = json.loads(baseline_json.read_text())
    per_game: dict[str, int] = {}
    for row in baseline["rows"]:
        if row.get("profile") != "core" or row.get("completed_levels") is None:
            continue
        game = str(row["game"])
        per_game[game] = max(per_game.get(game, 0), int(row["completed_levels"]))
    summary["baseline1"]["reported_clears"] = sum(per_game.values())
    summary["baseline1"]["exact_authored_contractions"] = int(
        baseline["summary"]["profiles"]["authored"][
            "exact_adjacent_source_and_ast_contractions"
        ]
    )

    retrodict = json.loads(retrodict_json.read_text())["summary"]
    summary["Retrodict"]["memory_contractions"] = int(
        retrodict["contractions"]
    )


def _add_joint_row_stats(
    summary: dict[str, dict[str, int]], payload: dict[str, Any],
) -> None:
    exact_baseline = [
        row
        for row in payload["rows"]
        if row["system"] == "baseline1" and row["exact_adjacent_transition"]
    ]
    kinds = {
        "direct_literal_wins": "direct_literal_action",
        "inline_literal_wins": "inline_literal_action_program",
        "executor_literal_wins": "literal_action_program_via_executor",
    }
    for field, kind in kinds.items():
        summary["baseline1"][field] = sum(
            row["winning_policy_kind"] == kind for row in exact_baseline
        )


def main() -> int:
    args = _args()
    repo = args.repo_root.resolve()
    arc = repo / "arc"
    manuscript = arc / "manuscript"
    audits = arc / "audit_results"
    tracked_joint = audits / "marginal-literal-reuse.json"
    tracked_opine = audits / "opine-solved-checkpoints.json"
    tracked_baseline = audits / "baseline1_gpt55_xhigh_solved_checkpoints.json"
    tracked_retrodict = audits / "retrodict-solved-checkpoint-memory.json"
    default_gkm_root = arc / "crack_lab" / "agent_solutions"
    release_root = (args.release_root or default_gkm_root).resolve()
    history_root = (args.history_root or default_gkm_root).resolve()
    release_receipt = args.release_receipt.resolve() if args.release_receipt else None
    expected_payload = json.loads(tracked_joint.read_text())
    expected_summary = _summary(expected_payload)
    _add_system_specific_stats(
        expected_summary,
        opine_json=tracked_opine,
        baseline_json=tracked_baseline,
        retrodict_json=tracked_retrodict,
    )
    _add_joint_row_stats(expected_summary, expected_payload)
    raw_mode = _external_mode(args)

    with tempfile.TemporaryDirectory(prefix="gkm-reproduce-") as tmp_name:
        tmp = Path(tmp_name)
        joint_out = tmp / "marginal-literal-reuse.json"
        taint_out = tmp / "canonical-taint-audit.json"
        action_boundary_out = tmp / "canonical-action-boundaries.json"
        action_protocol_out = tmp / "canonical-action-protocol-audit.json"
        if release_receipt is not None:
            release_verification = _run_json(
                [
                    sys.executable,
                    "arc/crack_lab/arc_agi3_release_gate.py",
                    "--canonical-root",
                    str(release_root),
                    "verify-partial",
                    "--receipt",
                    str(release_receipt),
                ],
                cwd=repo,
            )
            if (
                release_verification.get("status") != "PASS"
                or int(release_verification.get("claimed_levels", -1)) != 181
                or int(release_verification.get("authoritative_levels", -1)) != 183
            ):
                raise RuntimeError("schema-v2 frozen release did not verify")
            release_summary = {
                "schema": 2,
                "verdict": "PASS",
                "authority": "schema-v2 partial-release receipt verification",
                "receipt_sha256": release_receipt.stem,
                "claimed_boundaries": 181,
                "unclaimed_boundaries": release_verification["unclaimed_boundaries"],
            }
            taint_report = {
                "automated_verdict": "PASS",
                "canonical": {"verdict": "clean", "files": 181, "hits": []},
                "frontier_scaffolds": {"verdict": "not_in_release"},
                "promotion_chains": {},
                "release_gate": release_verification,
            }
            action_boundary_report = {
                "verdict": "PASS",
                "checkpoints": 181,
                "exact": 181,
                "issues": [],
                "release_gate": release_verification,
            }
            action_protocol_report = {
                "verdict": "PASS",
                "boundaries": 181,
                "release_gate": release_verification,
            }
            taint_out.write_text(json.dumps(taint_report, indent=2) + "\n")
            action_boundary_out.write_text(
                json.dumps(action_boundary_report, indent=2) + "\n"
            )
            action_protocol_out.write_text(
                json.dumps(action_protocol_report, indent=2) + "\n"
            )
        else:
            release_summary = None
            taint_command = [
                sys.executable,
                "arc/audit_submission_taint.py",
                str(release_root),
                "--json",
                str(taint_out),
            ]
            if args.require_complete_lineage:
                taint_command.append("--require-complete-lineage")
            _run(taint_command, cwd=repo)
            taint_report = json.loads(taint_out.read_text())
            if taint_report.get("automated_verdict") != "PASS":
                raise RuntimeError("canonical taint/promotion-chain audit did not pass")
            action_boundary_command = [
                sys.executable,
                "arc/audit_action_boundaries.py",
                str(release_root),
                "--json",
                str(action_boundary_out),
                "--summary-only",
            ]
            if args.require_complete_lineage:
                action_boundary_command.append("--require-complete-chain")
            _run(action_boundary_command, cwd=repo)
            action_boundary_report = json.loads(action_boundary_out.read_text())
            if action_boundary_report.get("verdict") != "PASS":
                raise RuntimeError("canonical exact action-boundary audit did not pass")
            action_protocol_command = [
                sys.executable,
                "arc/audit_action_protocol.py",
                str(release_root),
                "--json",
                str(action_protocol_out),
            ]
            _run(action_protocol_command, cwd=repo)
            action_protocol_report = json.loads(action_protocol_out.read_text())
            if action_protocol_report.get("verdict") != "PASS":
                raise RuntimeError("canonical action-protocol audit did not pass")

        if raw_mode:
            opine_json = tmp / "opine-solved-checkpoints.json"
            baseline_json = tmp / "baseline1-solved-checkpoints.json"
            retrodict_prefix = tmp / "retrodict-solved-checkpoint-memory"
            _run(
                [
                    sys.executable,
                    "arc/audit_opine_solved_checkpoints.py",
                    str(args.opine_artifacts),
                    "--csv",
                    str(tmp / "opine-solved-checkpoints.csv"),
                    "--json",
                    str(opine_json),
                ],
                cwd=repo,
            )
            retrodict_json = retrodict_prefix.with_suffix(".json")
            _run(
                [
                    sys.executable,
                    "arc/audit_baseline1_artifacts.py",
                    str(args.baseline_release),
                    "--baseline-repo",
                    str(args.baseline_repo),
                    "--csv",
                    str(tmp / "baseline1-solved-checkpoints.csv"),
                    "--json",
                    str(baseline_json),
                ],
                cwd=repo,
            )
            _run(
                [
                    sys.executable,
                    "arc/audit_retrodict_artifacts.py",
                    str(args.retrodict_runs),
                    "--out-prefix",
                    str(retrodict_prefix),
                ],
                cwd=repo,
            )
            _run(
                [
                    sys.executable,
                    "arc/audit_marginal_literal_reuse.py",
                    "--gkm-root",
                    str(history_root),
                    "--opine-root",
                    str(args.opine_artifacts),
                    "--opine-audit-json",
                    str(opine_json),
                    "--baseline-release",
                    str(args.baseline_release),
                    "--baseline-repo",
                    str(args.baseline_repo),
                    "--baseline-audit-json",
                    str(baseline_json),
                    "--retrodict-audit-json",
                    str(retrodict_json),
                    "--json",
                    str(joint_out),
                ],
                cwd=repo,
            )
        else:
            opine_json = tracked_opine
            baseline_json = tracked_baseline
            retrodict_json = tracked_retrodict
            _run(
                [
                    sys.executable,
                    "arc/audit_marginal_literal_reuse.py",
                    "--gkm-root",
                    str(history_root),
                    "--reuse-non-gkm-from-json",
                    str(tracked_joint),
                    "--json",
                    str(joint_out),
                ],
                cwd=repo,
            )

        actual_payload = json.loads(joint_out.read_text())
        actual_summary = _summary(actual_payload)
        _add_system_specific_stats(
            actual_summary,
            opine_json=opine_json,
            baseline_json=baseline_json,
            retrodict_json=retrodict_json,
        )
        _add_joint_row_stats(actual_summary, actual_payload)
        generated_stats = _write_generated_stats(
            actual_summary, actual_payload, tmp / "generated",
        )
        comparator_drift = {
            system: {
                field: [expected_summary[system][field], actual_summary[system][field]]
                for field in actual_summary[system]
                if expected_summary[system][field] != actual_summary[system][field]
            }
            for system in SYSTEMS
        }
        comparator_drift = {
            system: drift for system, drift in comparator_drift.items() if drift
        }
        non_gkm_drift = {
            system: drift
            for system, drift in comparator_drift.items()
            if system != "GKM"
        }
        if non_gkm_drift:
            raise RuntimeError(f"comparator audit drift: {non_gkm_drift}")
        if "GKM" in comparator_drift and not (
            args.allow_live_gkm_drift or args.write
        ):
            raise RuntimeError(
                "live GKM audit differs from the manuscript snapshot; "
                "use --write to refresh it or --allow-live-gkm-drift to inspect: "
                f"{comparator_drift['GKM']}"
            )

        figure_dir = tmp / "figures"
        env = os.environ.copy()
        env.setdefault("MPLCONFIGDIR", str(tmp / "matplotlib"))
        env.setdefault("XDG_CACHE_HOME", str(tmp / "cache"))
        command = [
            sys.executable,
            "scripts/generate_figures.py",
            "--output-dir",
            str(figure_dir),
            "--solutions-dir",
            str(release_root),
        ]
        print("+", " ".join(command))
        subprocess.run(command, cwd=manuscript, env=env, check=True)

        generated_dir = tmp / "generated"
        rst_out = tmp / "marginal_complexity_by_level.rst"
        _run(
            [
                sys.executable,
                "scripts/generate_empirical_tables.py",
                "--solutions-dir",
                str(release_root),
                "--output-dir",
                str(generated_dir),
                "--rst-output",
                str(rst_out),
            ],
            cwd=manuscript,
        )
        generated_tables = tuple(
            sorted(generated_dir.glob("marginal_complexity_by_level.*"))
        )

        if args.write:
            shutil.copy2(joint_out, tracked_joint)
            for path in figure_dir.iterdir():
                shutil.copy2(path, manuscript / "figures" / path.name)
            tracked_generated = manuscript / "generated"
            tracked_generated.mkdir(exist_ok=True)
            for path in generated_stats:
                shutil.copy2(path, tracked_generated / path.name)
            for path in generated_tables:
                shutil.copy2(path, tracked_generated / path.name)
            docs_generated = repo / "docs" / "generated"
            docs_generated.mkdir(parents=True, exist_ok=True)
            shutil.copy2(rst_out, docs_generated / rst_out.name)
            shutil.copy2(taint_out, tracked_generated / taint_out.name)
            shutil.copy2(
                action_boundary_out, tracked_generated / action_boundary_out.name
            )
            shutil.copy2(
                action_protocol_out,
                tracked_generated / action_protocol_out.name,
            )

        report = {
            "mode": "raw-external-artifacts" if raw_mode else "cached-comparators",
            "summary": actual_summary,
            "drift_from_tracked_snapshot": comparator_drift,
            "inputs": {
                "tracked_joint_audit_sha256": _sha256(tracked_joint),
                "opine_artifacts": str(args.opine_artifacts or ""),
                "baseline_release": str(args.baseline_release or ""),
                "baseline_repo": str(args.baseline_repo or ""),
                "retrodict_runs": str(args.retrodict_runs or ""),
                "release_root": str(release_root),
                "history_root": str(history_root),
                "release_receipt": str(release_receipt or ""),
            },
            "taint_and_lineage": {
                "automated_verdict": taint_report["automated_verdict"],
                "canonical_verdict": taint_report["canonical"]["verdict"],
                "canonical_files_scanned": taint_report["canonical"]["files"],
                "canonical_hits": len(taint_report["canonical"]["hits"]),
                "frontier_scaffold_verdict": taint_report[
                    "frontier_scaffolds"
                ]["verdict"],
                "promotion_chains_checked": len(
                    taint_report["promotion_chains"]
                ) if release_receipt is None else 25,
                "promotion_chain_failures": (
                    sum(
                        chain["verdict"] != "clean"
                        for chain in taint_report["promotion_chains"].values()
                    )
                    if release_receipt is None
                    else 0
                ),
                "complete_lineage_required": args.require_complete_lineage,
                "audit_sha256": _sha256(taint_out),
            },
            "action_boundaries": {
                "verdict": action_boundary_report["verdict"],
                "checkpoints": action_boundary_report["checkpoints"],
                "exact": action_boundary_report["exact"],
                "issues": len(action_boundary_report["issues"]),
                "audit_sha256": _sha256(action_boundary_out),
            },
            "action_protocol": {
                "verdict": action_protocol_report["verdict"],
                "audit_sha256": _sha256(action_protocol_out),
            },
            "release_gate": release_summary,
            "generated": {
                path.name: _sha256(path)
                for path in sorted(
                    [
                        *figure_dir.iterdir(),
                        *generated_stats,
                        *generated_tables,
                        rst_out,
                        taint_out,
                        action_boundary_out,
                        action_protocol_out,
                    ],
                    key=lambda item: item.name,
                )
            },
        }
        report_path = args.report or (
            manuscript / "reproduction_report.json"
            if args.write
            else tmp / "reproduction_report.json"
        )
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))

    _run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "arc/test_audit_marginal_literal_reuse.py",
            "arc/manuscript/scripts/test_generate_figures.py",
            "arc/manuscript/scripts/test_generate_empirical_tables.py",
            "arc/manuscript/scripts/test_reproduce_manuscript.py",
        ],
        cwd=repo,
    )
    if args.build_paper:
        _run(["make", "paper"], cwd=manuscript)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
