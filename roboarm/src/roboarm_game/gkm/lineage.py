"""Audit retained proposal-source growth across one RoboArm campaign.

The historical marginal charge follows the implementation documented by the
ARC manuscript: positive net description-size growth in ``legs.py`` and
``players.py``.  The stronger conditional marginal charges normalized top-level
AST units absent from the preceding admitted source state, so same-size rewrites
are not free.

Campaign generations are construction states, not separate solved levels.  A
profile can therefore demonstrate literal retained-leg reuse and a construction
curve, but it must not be presented as a solved-checkpoint sawtooth.
"""

from __future__ import annotations

import ast
import collections
import hashlib
import json
import zlib
from pathlib import Path
from typing import Any, Mapping

from .accounting import canonical_json_sha256, marginal_description
from .workspace import PROMOTED_SOURCE_FILES


LINEAGE_SCHEMA_VERSION = 1
LINEAGE_PROFILE_KIND = "campaign-construction-lineage"
HISTORICAL_GROWTH_FILES = ("legs.py", "players.py")


def _source_texts(directory: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for name in PROMOTED_SOURCE_FILES:
        path = directory / name
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"lineage source is missing or linked: {path}")
        result[name] = path.read_text(encoding="utf-8")
    return result


def _meaningful_line_count(source: str) -> int:
    return sum(
        1
        for line in source.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )


def _container_literal_cost(source: str) -> int:
    tree = ast.parse(source)
    result = 0
    for node in ast.walk(tree):
        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            result += len(node.elts)
        elif isinstance(node, ast.Dict):
            result += len(node.keys)
    return result


def historical_description_complexity(source: str) -> int:
    """Return the ARC historical ``d(f)`` proxy for one Python file."""

    return _meaningful_line_count(source) + _container_literal_cost(source)


def historical_marginal_complexity(
    previous: Mapping[str, str],
    current: Mapping[str, str],
) -> int:
    """Return positive net ``d(f)`` growth for library and player files."""

    return sum(
        max(
            0,
            historical_description_complexity(current[name])
            - historical_description_complexity(previous[name]),
        )
        for name in HISTORICAL_GROWTH_FILES
    )


def _normalized_top_level_units(
    files: Mapping[str, str],
) -> list[tuple[str, bytes]]:
    result: list[tuple[str, bytes]] = []
    for filename, source in sorted(files.items()):
        tree = ast.parse(source, filename=filename)
        for node in tree.body:
            representation = ast.dump(
                node,
                annotate_fields=True,
                include_attributes=False,
            ).encode("utf-8")
            result.append(
                (hashlib.sha256(representation).hexdigest(), representation)
            )
    return result


def conditional_ast_marginal(
    previous: Mapping[str, str],
    current: Mapping[str, str],
) -> tuple[int, int, int]:
    """Return zlib novelty bytes, reused units, and novel units."""

    available = collections.Counter(
        digest for digest, _ in _normalized_top_level_units(previous)
    )
    novel: list[bytes] = []
    reused = 0
    for digest, representation in _normalized_top_level_units(current):
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


def _definitions_and_calls(
    files: Mapping[str, str],
) -> tuple[dict[tuple[str, str, str], str], dict[str, set[str]]]:
    definitions: dict[tuple[str, str, str], str] = {}
    calls: dict[str, set[str]] = collections.defaultdict(set)
    for filename, source in files.items():
        tree = ast.parse(source, filename=filename)
        for node in tree.body:
            if not isinstance(
                node,
                (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
            ):
                continue
            key = (filename, type(node).__name__, node.name)
            representation = ast.dump(
                node,
                annotate_fields=True,
                include_attributes=False,
            )
            definitions[key] = hashlib.sha256(
                representation.encode("utf-8")
            ).hexdigest()
            calls[node.name].update(
                child.func.id
                for child in ast.walk(node)
                if isinstance(child, ast.Call)
                and isinstance(child.func, ast.Name)
            )
    return definitions, calls


def unchanged_called_legs(
    previous: Mapping[str, str],
    current: Mapping[str, str],
    entrypoint: str = "propose_level_1",
) -> tuple[list[str], list[str], dict[str, str]]:
    """Return direct and transitively invoked unchanged ``legs.py`` definitions."""

    previous_definitions, _ = _definitions_and_calls(previous)
    current_definitions, calls = _definitions_and_calls(current)
    direct_names = set(calls.get(entrypoint, set()))
    reachable: set[str] = set()
    pending = list(direct_names)
    while pending:
        name = pending.pop()
        if name in reachable:
            continue
        reachable.add(name)
        pending.extend(calls.get(name, set()) - reachable)

    unchanged: dict[str, str] = {}
    for key, digest in current_definitions.items():
        filename, _, name = key
        if filename != "legs.py" or previous_definitions.get(key) != digest:
            continue
        unchanged[f"legs.py:{name}"] = digest
    direct = sorted(
        label for label in unchanged if label.rsplit(":", 1)[1] in direct_names
    )
    transitive = sorted(
        label for label in unchanged if label.rsplit(":", 1)[1] in reachable
    )
    return direct, transitive, {label: unchanged[label] for label in transitive}


def _direction_changes(values: list[int]) -> int:
    signs: list[int] = []
    for before, after in zip(values, values[1:], strict=False):
        delta = after - before
        if delta:
            signs.append(1 if delta > 0 else -1)
    return sum(left != right for left, right in zip(signs, signs[1:], strict=False))


def _read_object(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _generation_outcome(
    attempts: list[object], generation: int
) -> dict[str, object]:
    selected = [
        attempt
        for attempt in attempts
        if isinstance(attempt, dict) and attempt.get("generation") == generation
    ]
    dispositions = sorted(
        {
            str(attempt.get("disposition"))
            for attempt in selected
            if attempt.get("disposition")
        }
    )
    failures = sorted(
        {
            str(failure)
            for attempt in selected
            for failure in attempt.get("observed_failure_evidence", [])
            if isinstance(failure, str) and failure
        }
    )
    levels_completed = 0
    for attempt in selected:
        for phase_name in ("preflight", "commit"):
            phase = attempt.get(phase_name)
            if isinstance(phase, dict):
                value = phase.get("levels_completed")
                if isinstance(value, int) and not isinstance(value, bool):
                    levels_completed = max(levels_completed, value)
    winning = any(
        attempt.get("disposition") == "committed_success" for attempt in selected
    )
    return {
        "attempt_count": len(selected),
        "dispositions": dispositions,
        "observed_failure_evidence": failures,
        "levels_completed": levels_completed,
        "winning_checkpoint": winning,
    }


def _milestone(outcome: Mapping[str, object]) -> str:
    dispositions = set(outcome.get("dispositions", []))
    failures = list(outcome.get("observed_failure_evidence", []))
    if "committed_success" in dispositions:
        return "safe revision committed, verified, and replayed"
    if "candidate_rejected_by_fsa" in dispositions:
        return "goal replay rejected by the safety FSA"
    if "probe_success_uncommitted" in dispositions:
        return "uncommitted sparse success discovered"
    if failures:
        return "frontier experiments: " + ", ".join(str(value) for value in failures)
    return "admitted retained-source revision"


def campaign_lineage_profile(campaign_root: Path) -> dict[str, object]:
    """Build an auditable construction-lineage profile for one campaign."""

    root = campaign_root.resolve(strict=True)
    result = _read_object(root / "campaign_result.json")
    campaign_id = result.get("campaign_id")
    if not isinstance(campaign_id, str) or not campaign_id:
        raise ValueError("campaign lineage requires a campaign id")
    baseline_dir = root / "parents" / "level_00"
    previous = _source_texts(baseline_dir)
    observed = _read_object(root / "evidence" / "observed_attempt_ledger.json")
    raw_attempts = observed.get("attempts", [])
    attempts = raw_attempts if isinstance(raw_attempts, list) else []

    workspace_generations: list[tuple[int, Path]] = []
    for workspace in (root / "workspaces").glob("generation_*"):
        suffix = workspace.name.removeprefix("generation_")
        if suffix.isdigit():
            workspace_generations.append((int(suffix), workspace))
    workspace_generations.sort()

    generations: list[dict[str, object]] = []
    previous_ast_marginal: int | None = None
    for expected_generation, (generation, workspace) in enumerate(
        workspace_generations,
        1,
    ):
        if generation != expected_generation:
            raise ValueError(
                "campaign lineage generations are not exact adjacent states"
            )
        current = _source_texts(workspace)
        admission = _read_object(
            root
            / "evidence"
            / f"generation_admission_{generation:03d}.json"
        )
        if admission.get("clean") is not True:
            raise ValueError(
                f"campaign lineage generation {generation} was not admitted cleanly"
            )
        conditional, reused_units, novel_units = conditional_ast_marginal(
            previous, current
        )
        direct, transitive, reused_hashes = unchanged_called_legs(
            previous, current
        )
        net_growth = historical_marginal_complexity(previous, current)
        positive_lines = sum(
            marginal_description(previous[name], current[name])
            for name in PROMOTED_SOURCE_FILES
        )
        drop = (
            previous_ast_marginal - conditional
            if previous_ast_marginal is not None
            else None
        )
        ratio = (
            conditional / previous_ast_marginal
            if previous_ast_marginal
            else None
        )
        sharp = bool(
            previous_ast_marginal
            and conditional * 2 <= previous_ast_marginal
        )
        outcome = _generation_outcome(attempts, generation)
        generations.append(
            {
                "generation": generation,
                "exact_adjacent_source_transition": True,
                "winning_checkpoint": outcome["winning_checkpoint"],
                "historical_net_growth": net_growth,
                "positive_meaningful_line_additions": positive_lines,
                "conditional_ast_zlib_bytes": conditional,
                "previous_conditional_ast_zlib_bytes": previous_ast_marginal,
                "conditional_ast_drop_bytes": drop,
                "conditional_ast_ratio": ratio,
                "sharp_marginal_drop": sharp,
                "literal_reused_top_level_nodes": reused_units,
                "novel_top_level_nodes": novel_units,
                "direct_unchanged_called_legs": direct,
                "transitive_unchanged_called_legs": transitive,
                "transitive_unchanged_called_leg_sha256": reused_hashes,
                "hard_direct_reuse_witness": bool(direct),
                "transitive_reuse_witness": bool(transitive),
                "sharp_drop_with_direct_reuse": bool(sharp and direct),
                "outcome": outcome,
                "milestone": _milestone(outcome),
            }
        )
        previous = current
        previous_ast_marginal = conditional

    if not generations:
        raise ValueError("campaign lineage contains no admitted generations")
    net_values = [int(row["historical_net_growth"]) for row in generations]
    ast_values = [int(row["conditional_ast_zlib_bytes"]) for row in generations]
    direct_generations = sum(
        bool(row["hard_direct_reuse_witness"]) for row in generations
    )
    transitive_generations = sum(
        bool(row["transitive_reuse_witness"]) for row in generations
    )
    profile: dict[str, object] = {
        "schema_version": LINEAGE_SCHEMA_VERSION,
        "profile_kind": LINEAGE_PROFILE_KIND,
        "campaign_id": campaign_id,
        "source_boundary": "clean-admitted proposer generation",
        "metric_contract": {
            "historical_net_growth": (
                "sum over legs.py and players.py of max(0, d(after)-d(before)); "
                "d is nonblank noncomment LOC plus AST container elements"
            ),
            "conditional_ast_zlib_bytes": (
                "zlib-9 bytes of normalized top-level AST units absent from the "
                "preceding admitted source state"
            ),
            "sharp_drop_threshold": "current <= previous / 2",
            "direct_reuse": (
                "propose_level_1 directly calls an unchanged legs.py definition"
            ),
            "transitive_reuse": (
                "an unchanged legs.py definition is reachable from propose_level_1"
            ),
        },
        "interpretation": {
            "solved_level_sawtooth_claim": False,
            "construction_profile_only": True,
            "reason": (
                "rb01 has one promoted round; generations 1-3 are acquisition and "
                "revision states rather than separate solved-level checkpoints"
            ),
            "historical_net_growth_direction_changes": _direction_changes(net_values),
            "conditional_ast_direction_changes": _direction_changes(ast_values),
            "direct_reuse_generations": direct_generations,
            "transitive_reuse_generations": transitive_generations,
            "sharp_direct_coupled_witnesses": sum(
                bool(row["sharp_drop_with_direct_reuse"])
                for row in generations
            ),
        },
        "generations": generations,
    }
    profile["profile_receipt_sha256"] = canonical_json_sha256(profile)
    return profile


def lineage_markdown(profile: Mapping[str, object]) -> str:
    """Render the machine-readable lineage profile as an evidence table."""

    rows = profile.get("generations", [])
    if not isinstance(rows, list):
        raise ValueError("lineage profile generations are invalid")
    table_rows = []
    for value in rows:
        if not isinstance(value, dict):
            continue
        transitive = value.get("transitive_unchanged_called_legs", [])
        reused = ", ".join(f"`{item}`" for item in transitive) or "—"
        table_rows.append(
            "| G{generation} | {net} | {ast} | {nodes} | {direct} | {transitive_count} | {milestone} |".format(
                generation=value.get("generation"),
                net=value.get("historical_net_growth"),
                ast=value.get("conditional_ast_zlib_bytes"),
                nodes=value.get("literal_reused_top_level_nodes"),
                direct="yes" if value.get("hard_direct_reuse_witness") else "no",
                transitive_count=len(transitive) if isinstance(transitive, list) else 0,
                milestone=value.get("milestone"),
            )
        )
        table_rows.append(f"|  |  |  |  |  | ↳ | {reused} |")
    interpretation = profile.get("interpretation", {})
    return """\
# RoboArm retained-leg reuse and marginal-complexity profile

This is a **campaign construction profile**, not a solved-level sawtooth. RoboArm
has one promoted round; generations 1–3 are admitted acquisition/revision source
states and generation 4 is the winning checkpoint. The profile therefore reports
literal source reuse without relabeling interim generations as separate wins.

| Boundary | Historical net-growth C | Conditional AST M (zlib bytes) | Reused AST units | Direct unchanged leg | Transitive unchanged legs | Outcome |
|---|---:|---:|---:|:---:|---:|---|
{rows}

## Interpretation

- Historical net-growth direction changes: `{net_changes}`.
- Conditional-AST direction changes: `{ast_changes}`.
- Generations with a direct unchanged-leg call: `{direct}`.
- Generations with transitively invoked unchanged legs: `{transitive}`.
- Sharp-drop/direct-call coupled witnesses: `{coupled}`.

The direct-call count is intentionally stricter than the transitive count. Each
RoboArm player calls a newly named evidence gate/composition, while those
compositions increasingly call unchanged lower-level legs. The final generation
transitively invokes eight unchanged retained definitions, but its conditional-AST
drop is about one third rather than the predeclared half-or-more threshold.

Profile receipt: `{receipt}`
""".format(
        rows="\n".join(table_rows),
        net_changes=interpretation.get("historical_net_growth_direction_changes"),
        ast_changes=interpretation.get("conditional_ast_direction_changes"),
        direct=interpretation.get("direct_reuse_generations"),
        transitive=interpretation.get("transitive_reuse_generations"),
        coupled=interpretation.get("sharp_direct_coupled_witnesses"),
        receipt=profile.get("profile_receipt_sha256"),
    )


__all__ = [
    "LINEAGE_PROFILE_KIND",
    "LINEAGE_SCHEMA_VERSION",
    "campaign_lineage_profile",
    "conditional_ast_marginal",
    "historical_description_complexity",
    "historical_marginal_complexity",
    "lineage_markdown",
    "unchanged_called_legs",
]
