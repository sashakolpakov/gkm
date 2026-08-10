from __future__ import annotations

import ast
import json
import importlib.metadata
import platform
import tomllib
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src" / "roboarm_game"
FORBIDDEN_IMPORT_ROOTS = {
    "arc",
    "arc_agi",
    "arcengine",
}


def imported_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.partition(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.partition(".")[0])
    return roots


def test_runtime_has_no_arc_api_or_parent_imports() -> None:
    offenders: dict[str, list[str]] = {}
    for path in sorted(SOURCE_ROOT.glob("*.py")):
        forbidden = sorted(imported_roots(path) & FORBIDDEN_IMPORT_ROOTS)
        if forbidden:
            offenders[path.name] = forbidden
    assert offenders == {}


def test_project_declares_only_standalone_runtime_dependencies() -> None:
    project = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    runtime_dependencies = project["project"]["dependencies"]
    test_dependencies = project["project"]["optional-dependencies"]["test"]

    assert runtime_dependencies == ["numpy==2.4.4"]
    assert test_dependencies == ["pytest==9.0.3"]
    combined = " ".join(runtime_dependencies + test_dependencies).lower()
    assert "arc" not in combined
    assert "engine" not in combined


def test_lock_and_runtime_manifest_match_active_environment() -> None:
    manifest = json.loads(
        (PROJECT_ROOT / "references" / "runtime_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["python"] == platform.python_version()
    assert manifest["numpy"] == importlib.metadata.version("numpy")
    assert manifest["pytest"] == importlib.metadata.version("pytest")

    lock_lines = {
        line
        for line in (PROJECT_ROOT / "requirements.lock")
        .read_text(encoding="utf-8")
        .splitlines()
        if line and not line.startswith("#")
    }
    assert "numpy==2.4.4" in lock_lines
    assert "pytest==9.0.3" in lock_lines
    assert not any(
        dependency.lower().startswith(("arc", "arc_", "arcengine"))
        for dependency in lock_lines
    )


def test_no_dynamic_import_escape_hatch_in_runtime() -> None:
    forbidden_calls = {"__import__", "import_module"}
    offenders: list[str] = []
    for path in sorted(SOURCE_ROOT.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            function = node.func
            name = function.id if isinstance(function, ast.Name) else (
                function.attr if isinstance(function, ast.Attribute) else ""
            )
            if name in forbidden_calls:
                offenders.append(f"{path.name}:{node.lineno}:{name}")
    assert offenders == []
