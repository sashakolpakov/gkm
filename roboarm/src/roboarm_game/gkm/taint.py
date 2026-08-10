"""Fail-closed admission checks for proposer transcripts and source."""

from __future__ import annotations

import ast
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .workspace import PROMOTED_SOURCE_FILES

PROTECTED_WORKSPACE_FILES = (
    "README.md",
    "ROUND.md",
    "evidence.json",
    "gkm_propose.py",
    "interface.py",
    "perception.py",
    "protocol.py",
    "scenario_contract.py",
    "solver_index.md",
)

FORBIDDEN_COMMAND_PATTERNS = (
    (re.compile(r"(^|[\s'\";|&])\.\.(?:/|[\s'\";|&]|$)"), "parent traversal"),
    (
        re.compile(r"/Users/|/private/|/tmp/|(?:^|\s)~/?"),
        "host path access",
    ),
    (re.compile(r"\broboarm_game\b|\bcanonical\.py\b|\bdynamics\.py\b"), "private package source"),
    (re.compile(r"\benvironment\.py\b|\bworld_state\.py\b|\boracle\.py\b"), "private environment source"),
    (
        re.compile(r"\bArena\s*\(|\.arena\.json\b|\bROBOARM_ARENA_CONFIG\b"),
        "direct actuation channel",
    ),
    (re.compile(r"\binspect\.getsource\b|\bvars\s*\(|\b__dict__\b"), "runtime introspection"),
    (
        re.compile(
            r"\b(?:curl|wget|ssh|scp|ncat)\b"
            r"|(?m:(?:^|[;&|]\s*|\n\s*|(?:-lc|-c)\s+['\"]\s*)"
            r"(?:sudo\s+|env\s+)*nc"
            r"(?=\s*(?:$|[;&|'\"])|\s+(?:-[A-Za-z]|[A-Za-z0-9_./:\[])))"
            r"|https?://"
        ),
        "external network",
    ),
    (re.compile(r"(?:^|[;&|]\s*)\s*(?:ps|pgrep|pkill|printenv|env)\b"), "host process/environment inspection"),
)

FORBIDDEN_IMPORT_ROOTS = {
    "asyncio",
    "ctypes",
    "http",
    "importlib",
    "inspect",
    "multiprocessing",
    "os",
    "pathlib",
    "requests",
    "roboarm_game",
    "shutil",
    "socket",
    "subprocess",
    "sys",
    "urllib",
}


@dataclass(frozen=True, slots=True)
class TaintReport:
    clean: bool
    reasons: tuple[str, ...]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def protected_manifest(workspace: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for name in PROTECTED_WORKSPACE_FILES:
        path = workspace / name
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"missing protected workspace file: {name}")
        result[name] = sha256_file(path)
    return result


def _command_strings(transcript: Path) -> Iterable[str]:
    if not transcript.is_file() or transcript.is_symlink():
        return ()
    commands: list[str] = []
    for line_number, line in enumerate(
        transcript.read_text(encoding="utf-8", errors="replace").splitlines(),
        1,
    ):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            commands.append(f"<malformed-json-line:{line_number}>")
            continue
        if not isinstance(event, dict):
            continue
        item = event.get("item")
        if isinstance(item, dict) and item.get("type") == "command_execution":
            command = item.get("command")
            if isinstance(command, str):
                commands.append(command)
    return tuple(commands)


def _source_reasons(path: Path) -> list[str]:
    reasons: list[str] = []
    try:
        source = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        return [f"{path.name}: unreadable source: {error}"]
    if len(source.encode("utf-8")) > 256_000:
        return [f"{path.name}: source exceeds 256 KiB"]
    try:
        tree = ast.parse(source, filename=path.name)
    except SyntaxError as error:
        return [f"{path.name}: syntax error: {error}"]
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = (
                [alias.name for alias in node.names]
                if isinstance(node, ast.Import)
                else [node.module or ""]
            )
            for name in names:
                root = name.split(".", 1)[0]
                if root in FORBIDDEN_IMPORT_ROOTS:
                    reasons.append(f"{path.name}: forbidden import {name!r}")
        if isinstance(node, ast.Attribute) and (
            node.attr.startswith("_")
            or node.attr
            in {
                "clone",
                "event_log",
                "reset",
                "snapshot",
                "state",
                "step",
            }
        ):
            reasons.append(
                f"{path.name}: private/runtime attribute access {node.attr!r}"
            )
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in {
                "Arena",
                "open",
                "eval",
                "exec",
                "compile",
                "__import__",
            }:
                reasons.append(f"{path.name}: forbidden call {node.func.id}()")
    return reasons


def _player_composition_reasons(path: Path) -> list[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=path.name)
    except (OSError, UnicodeDecodeError, SyntaxError):
        return []
    reasons: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            continue
        if isinstance(node, ast.ImportFrom) and node.module == "legs":
            continue
        if not isinstance(node, ast.FunctionDef):
            reasons.append("players.py: only imports from legs and propose_level functions are allowed")
            continue
        if not node.name.startswith("propose_level_"):
            reasons.append(f"players.py: unexpected function {node.name!r}")
        for child in ast.walk(node):
            if isinstance(
                child,
                (
                    ast.For,
                    ast.While,
                    ast.ListComp,
                    ast.SetComp,
                    ast.DictComp,
                    ast.GeneratorExp,
                ),
            ):
                reasons.append(
                    f"players.py: {node.name} contains inline iteration instead of leg composition"
                )
            if (
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Attribute)
                and child.func.attr == "step"
            ):
                reasons.append(
                    f"players.py: {node.name} calls step inline instead of a proposal leg"
                )
    return reasons


def inspect_generation(
    workspace: Path,
    transcript: Path,
    baseline: dict[str, str],
) -> TaintReport:
    """Inspect one completed proposer generation before any execution/admission."""

    reasons: list[str] = []
    resolved = workspace.resolve(strict=True)
    if transcript.is_symlink():
        reasons.append("proposer transcript is a symlink")
    elif not transcript.is_file():
        reasons.append("proposer transcript is missing")
    elif transcript.stat().st_size > 64 * 1024 * 1024:
        reasons.append("proposer transcript exceeds 64 MiB")

    for name, expected in baseline.items():
        path = resolved / name
        if not path.is_file() or path.is_symlink():
            reasons.append(f"protected workspace file missing or linked: {name}")
        elif sha256_file(path) != expected:
            reasons.append(f"protected workspace file changed: {name}")

    commands = tuple(_command_strings(transcript))
    if any(command.startswith("<malformed-json-line:") for command in commands):
        reasons.append("proposer transcript contains malformed JSON")
    for command in commands:
        for pattern, label in FORBIDDEN_COMMAND_PATTERNS:
            if pattern.search(command):
                reasons.append(f"forbidden proposer command ({label}): {command}")

    python_files = sorted(resolved.rglob("*.py"))
    if len(python_files) > 96:
        reasons.append("proposer generation contains more than 96 Python files")
    protected_python = {
        "gkm_propose.py",
        "interface.py",
        "perception.py",
        "protocol.py",
        "scenario_contract.py",
    }
    for path in python_files:
        if path.is_symlink() or not path.resolve(strict=True).is_relative_to(resolved):
            reasons.append(f"linked or escaping Python file: {path}")
            continue
        if path.name not in protected_python:
            reasons.extend(_source_reasons(path))

    for name in PROMOTED_SOURCE_FILES:
        path = resolved / name
        if not path.is_file() or path.is_symlink():
            reasons.append(f"missing candidate source file: {name}")
    players = resolved / "players.py"
    if players.is_file():
        reasons.extend(_player_composition_reasons(players))

    return TaintReport(clean=not reasons, reasons=tuple(dict.fromkeys(reasons)))


def inspect_executable_workspace(workspace: Path) -> TaintReport:
    """Check proposer-authored Python before a public probe/solver execution."""

    resolved = workspace.resolve(strict=True)
    reasons: list[str] = []
    protected_python = {
        "gkm_propose.py",
        "interface.py",
        "perception.py",
        "protocol.py",
        "scenario_contract.py",
    }
    python_files = sorted(resolved.rglob("*.py"))
    if len(python_files) > 96:
        reasons.append("proposer generation contains more than 96 Python files")
    for path in python_files:
        if path.is_symlink() or not path.resolve(strict=True).is_relative_to(resolved):
            reasons.append(f"linked or escaping Python file: {path}")
        elif path.name not in protected_python:
            reasons.extend(_source_reasons(path))
    players = resolved / "players.py"
    if players.is_file():
        reasons.extend(_player_composition_reasons(players))
    return TaintReport(clean=not reasons, reasons=tuple(dict.fromkeys(reasons)))


__all__ = [
    "PROTECTED_WORKSPACE_FILES",
    "TaintReport",
    "inspect_executable_workspace",
    "inspect_generation",
    "protected_manifest",
    "sha256_file",
]
