"""Marginal program-description and campaign resource accounting."""

from __future__ import annotations

import ast
import hashlib
import json
from collections import Counter
from pathlib import Path

from .workspace import PROMOTED_SOURCE_FILES


def _meaningful_lines(source: str) -> Counter[str]:
    lines = (
        line.strip()
        for line in source.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    return Counter(lines)


def marginal_description(before: str, after: str) -> int:
    """Count positive net meaningful-line additions."""

    old = _meaningful_lines(before)
    new = _meaningful_lines(after)
    return sum(max(0, count - old[line]) for line, count in new.items())


def literal_action_cost(source: str) -> int:
    """Price literal action containers that can memorize a trajectory."""

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return 0
    cost = 0
    for node in ast.walk(tree):
        if not isinstance(node, (ast.List, ast.Tuple)):
            continue
        values: list[int] = []
        for element in node.elts:
            if (
                isinstance(element, ast.Constant)
                and isinstance(element.value, int)
                and not isinstance(element.value, bool)
                and 1 <= element.value <= 6
            ):
                values.append(int(element.value))
            else:
                values = []
                break
        if len(values) >= 4:
            cost += len(values)
    return cost


def source_accounting(
    before_dir: Path | None,
    after_dir: Path,
) -> dict[str, object]:
    per_file: dict[str, dict[str, int]] = {}
    total_marginal = 0
    total_literal = 0
    digest = hashlib.sha256()
    for name in PROMOTED_SOURCE_FILES:
        after = (after_dir / name).read_text(encoding="utf-8")
        before = (
            (before_dir / name).read_text(encoding="utf-8")
            if before_dir is not None and (before_dir / name).is_file()
            else ""
        )
        marginal = marginal_description(before, after)
        literal = literal_action_cost(after)
        per_file[name] = {
            "meaningful_lines": sum(_meaningful_lines(after).values()),
            "marginal_description": marginal,
            "literal_action_cost": literal,
            "bytes": len(after.encode("utf-8")),
        }
        total_marginal += marginal
        total_literal += literal
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(after.encode("utf-8"))
        digest.update(b"\0")
    return {
        "source_tree_sha256": digest.hexdigest(),
        "marginal_description": total_marginal,
        "literal_action_cost": total_literal,
        "priced_complexity": total_marginal + total_literal,
        "files": per_file,
    }


def free_energy(
    levels_completed: int,
    priced_complexity: int,
    *,
    complexity_weight: float = 0.01,
) -> float:
    return -float(levels_completed) + complexity_weight * float(priced_complexity)


def canonical_json_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


__all__ = [
    "canonical_json_sha256",
    "free_energy",
    "literal_action_cost",
    "marginal_description",
    "source_accounting",
]
