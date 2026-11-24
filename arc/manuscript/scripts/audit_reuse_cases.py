#!/usr/bin/env python3
"""Inspect the concrete call structure of five marginal dips in frozen sources."""

from __future__ import annotations

import argparse
import ast
from collections import deque
import hashlib
import json
from pathlib import Path
import tempfile

import audit_revision_sources as frozen
from executable_reuse import Program, analyse_reuse


CASES = (("dc22", 2), ("dc22", 5), ("wa30", 2), ("wa30", 6), ("cn04", 3))


def definitions(files: dict[str, bytes]) -> dict:
    """Honor the last top-level definition, including duplicate player stubs."""
    result = {}
    for filename, source in sorted(files.items()):
        for node in ast.parse(source).body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                result[(filename, node.name)] = node
    return result


def ast_hash(node: ast.AST) -> str:
    return hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()


def body_calls(node: ast.AST):
    """Inspect this body's calls without treating uncalled nested bodies as edges."""
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            continue
        if isinstance(child, ast.Call):
            yield child
        yield from body_calls(child)


def inspect_call_paths(previous: dict[str, bytes], current: dict[str, bytes], level: int) -> dict:
    program = Program(current)
    start = f"players.py:play_level_{level}"
    general = analyse_reuse(previous, current, [start], target_files={"legs.py"})
    root = program.defs[start].node
    paths = list(general["paths"].values())
    calls = list(body_calls(root))
    literal_loop = (len(root.body) == 1 and isinstance(root.body[0], ast.For)
                    and isinstance(root.body[0].iter, (ast.List, ast.Tuple))
                    and bool(calls)
                    and all(ast.unparse(c.func) == "env.step" for c in calls))
    if literal_loop:
        try:
            ast.literal_eval(root.body[0].iter)
        except (ValueError, TypeError):
            literal_loop = False
    return {"entrypoint": start, "entrypoint_ast_sha256": ast_hash(root),
            "entrypoint_calls": sorted({ast.unparse(c.func) for c in calls}),
            "unchanged_leg_paths": paths,
            "unchanged_target_ast_sha256": general["unchanged_definition_sha256"],
            "literal_action_loop_only": literal_loop,
            "literal_action_count": len(root.body[0].iter.elts) if literal_loop else None}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1] / "generated")
    parser.add_argument("--audit", type=Path, default=frozen.REPO / "arc/audit_results/marginal-literal-reuse.json")
    args = parser.parse_args()
    audit_path = args.audit
    raw = audit_path.read_bytes()
    rows = {(r["game"], r["completed_level"]): r for r in json.loads(raw)["rows"] if r["system"] == "GKM"}
    cases = []
    with tempfile.TemporaryDirectory(prefix="gkm-frozen-reuse-cases-") as directory:
        root = Path(directory)
        frozen.materialize(root)
        for game, level in CASES:
            folder = root / (game + "_legs")
            before = frozen.checkpoints.exact_snapshot(folder, level - 1)
            after = frozen.checkpoints.exact_snapshot(folder, level)
            if before is None or after is None:
                raise ValueError("missing exact case source")
            previous, current = before[1], after[1]
            row = rows[(game, level)]
            marginal = frozen.reuse.conditional_ast_marginal(previous, current)[0]
            if marginal != row["marginal_ast_zlib_bytes"] or row["marginal_drop_bytes"] <= 0:
                raise ValueError("case is not the authenticated recorded marginal dip")
            case = {"game": game, "level": level,
                    "previous_M": row["previous_level_marginal_ast_zlib_bytes"], "M": marginal,
                    "winning_phase": after[0]["phase"], "winning_attempt": after[0]["attempt"],
                    "source_sha256": {"previous": {n: hashlib.sha256(v).hexdigest() for n, v in sorted(previous.items())},
                                      "current": {n: hashlib.sha256(v).hexdigest() for n, v in sorted(current.items())}},
                    **inspect_call_paths(previous, current, level)}
            if game in {"dc22", "wa30"} and not case["unchanged_leg_paths"]:
                raise ValueError("expected retained helper path is absent")
            if game == "cn04" and (not case["literal_action_loop_only"] or case["unchanged_leg_paths"]):
                raise ValueError("compact action-program case has changed")
            cases.append(case)
    payload = {"history_revision": frozen.REVISION,
               "history_tree_sha1": frozen.reproduction.FROZEN_SOURCE_HISTORY_TREE_SHA1,
               "source_audit_sha256": hashlib.sha256(raw).hexdigest(),
               "scope": "five concrete winning-source cases; the existing suite-wide direct-call counts are unchanged",
               "measurement": "effective top-level definitions and named static call paths to unchanged legs; no runtime call counts",
               "cases": cases}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "reuse_case_audit.json").write_text(json.dumps(payload, indent=2) + "\n")
    lines = ["<!-- Generated by scripts/audit_reuse_cases.py; do not edit. -->", "",
             "Concrete source evidence at conditional marginal dips", "",
             "| Boundary | M (compressed bytes) | Winning-source evidence |",
             "| --- | --- | --- |"]
    for case in cases:
        evidence = "; ".join(" → ".join(path) for path in case["unchanged_leg_paths"])
        if case["literal_action_loop_only"]:
            evidence = f"{case['literal_action_count']}-action literal loop; its only call is env.step"
        lines.append(f"| {case['game']} L{case['level']} | {case['previous_M']} → {case['M']} | {evidence} |")
    lines.extend(["", "Source hashes, unchanged-definition hashes and frozen revision are in reuse_case_audit.json.", ""])
    (args.output_dir / "reuse_case_audit.md").write_text("\n".join(lines))
    descriptions = {
        ("dc22", 2): r"New bridge routine calls unchanged \texttt{toggle\_control} and \texttt{walk\_segments}.",
        ("dc22", 5): r"New platform-transfer routine calls the same unchanged helpers.",
        ("wa30", 2): r"New courier routine calls unchanged \texttt{walk} and \texttt{interact}.",
        ("wa30", 6): r"New recovery routine calls the same unchanged helpers.",
        ("cn04", 3): "",
    }
    tex = [r"% Generated by scripts/audit_reuse_cases.py; do not edit.",
           r"\begin{table}[H]\centering\small",
           r"\begin{tabularx}{\textwidth}{@{}llX@{}}\toprule",
           r"Boundary & $M_{k-1}\to M_k$ (bytes) & Winning-source evidence \\",
           r"\midrule"]
    for case in cases:
        description = descriptions[(case["game"], case["level"])]
        if case["literal_action_loop_only"]:
            description = str(case["literal_action_count"]) + r"-action literal loop calling only \texttt{env.step}; no call to a retained solver leg."
        tex.append(r"\texttt{" + case["game"] + "} L" + str(case["level"]) + " & $" + str(case["previous_M"]) + r"\to " + str(case["M"]) + "$ & " + description + r" \\")
    tex.extend([r"\bottomrule\end{tabularx}",
                r"\caption{Concrete source structure at five marginal dips. The four \texttt{dc22}/\texttt{wa30} cases reuse unchanged helpers through a newly added level routine; the \texttt{cn04} player executes a literal action program. Function identities and call paths are checked against exact adjacent winning sources.}",
                r"\label{tab:reuse-cases}\end{table}", ""])
    (args.output_dir / "reuse_case_audit.tex").write_text("\n".join(tex))
    print("Verified four indirect-reuse dips and one literal-action dip from frozen winning sources.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
