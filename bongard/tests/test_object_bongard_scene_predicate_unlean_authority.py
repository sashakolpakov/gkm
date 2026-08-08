from __future__ import annotations

import ast
from collections import deque
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import textwrap
from typing import Iterable


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PACKAGE_ROOT.parent

SCENE_AUTHORITY_ROOTS = (
    "bongard.object_scene_visual_frontend",
    "bongard.object_scene_semantic_registry",
    "bongard.object_bongard_scene_predicate_ir",
    "bongard.object_bongard_scene_predicate_calibration_command",
    "bongard.object_bongard_scene_predicate_campaign_command",
)
FORBIDDEN_AUTHORITY_MODULES = frozenset(
    {
        "bongard.semantic_checker",
        "bongard.predicate_backend",
    }
)
FORBIDDEN_EXECUTABLES = frozenset({"lean", "lean4", "lake", "elan"})


def _module_path(module: str) -> Path | None:
    if module == "bongard":
        candidate = PACKAGE_ROOT / "__init__.py"
    elif module.startswith("bongard."):
        relative = module.removeprefix("bongard.").replace(".", "/")
        module_file = PACKAGE_ROOT / f"{relative}.py"
        package_file = PACKAGE_ROOT / relative / "__init__.py"
        candidate = module_file if module_file.is_file() else package_file
    else:
        return None
    return candidate if candidate.is_file() else None


def _tree(module: str) -> ast.Module:
    path = _module_path(module)
    assert path is not None, f"active local module has no source: {module}"
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _resolve_from_module(source: str, node: ast.ImportFrom) -> str | None:
    if node.level == 0:
        return node.module
    source_path = _module_path(source)
    assert source_path is not None
    package = source if source_path.name == "__init__.py" else source.rpartition(".")[0]
    relative = "." * node.level + (node.module or "")
    try:
        return importlib.util.resolve_name(relative, package)
    except (ImportError, ValueError):
        return None


def _local_imports(module: str) -> Iterable[str]:
    for node in ast.walk(_tree(module)):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if _module_path(alias.name) is not None:
                    yield alias.name
        elif isinstance(node, ast.ImportFrom):
            target = _resolve_from_module(module, node)
            if target is None:
                continue
            if _module_path(target) is not None:
                yield target
            # ``from bongard import module`` imports the named submodule even
            # though the AST records only the package as ``node.module``.
            for alias in node.names:
                child = f"{target}.{alias.name}"
                if alias.name != "*" and _module_path(child) is not None:
                    yield child


def _transitive_local_closure(roots: Iterable[str]) -> frozenset[str]:
    pending = deque(sorted(set(roots) | {"bongard"}))
    visited: set[str] = set()
    while pending:
        module = pending.popleft()
        if module in visited or _module_path(module) is None:
            continue
        visited.add(module)
        pending.extend(
            target for target in _local_imports(module) if target not in visited
        )
    return frozenset(visited)


def _exact_executable_literals(module: str) -> frozenset[str]:
    """Return only exact command-like literals, never prose or metadata keys."""

    found: set[str] = set()
    for node in ast.walk(_tree(module)):
        if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
            continue
        literal = node.value.strip()
        if not literal or any(character.isspace() for character in literal):
            continue
        executable = Path(literal).name.casefold()
        if executable in FORBIDDEN_EXECUTABLES:
            found.add(executable)
    return frozenset(found)


def test_scene_predicate_authority_has_no_transitive_checker_or_lean_reachability() -> None:
    closure = _transitive_local_closure(SCENE_AUTHORITY_ROOTS)

    assert set(SCENE_AUTHORITY_ROOTS).issubset(closure)
    assert closure.isdisjoint(FORBIDDEN_AUTHORITY_MODULES)
    assert {
        module: sorted(_exact_executable_literals(module))
        for module in sorted(closure)
        if _exact_executable_literals(module)
    } == {}


_ISOLATED_SCENE_AUTHORITY_SMOKE = r"""
from __future__ import annotations

import importlib
import importlib.abc
import json
import os
from pathlib import Path
import shlex
import shutil
import sys


package_parent = Path(sys.argv[1]).resolve(strict=True)
sys.path.insert(0, str(package_parent))

forbidden_modules = frozenset({
    "bongard.semantic_checker",
    "bongard.predicate_backend",
})
forbidden_executables = frozenset({"lean", "lean4", "lake", "elan"})


class _RejectForbiddenAuthorityImport(importlib.abc.MetaPathFinder):
    attempts = []

    def find_spec(self, fullname, path=None, target=None):
        del path, target
        if any(
            fullname == forbidden or fullname.startswith(forbidden + ".")
            for forbidden in forbidden_modules
        ):
            self.attempts.append(fullname)
            raise AssertionError(
                f"scene predicate authority imported optional checker: {fullname}"
            )
        return None


def _command_head(value):
    if isinstance(value, (tuple, list)):
        if not value:
            return None
        value = value[0]
    if isinstance(value, bytes):
        value = os.fsdecode(value)
    if not isinstance(value, str):
        return None
    try:
        words = shlex.split(value)
    except ValueError:
        words = [value]
    if not words:
        return None
    return Path(words[0]).name.casefold()


def _reject_forbidden_executable(value):
    executable = _command_head(value)
    if executable in forbidden_executables:
        raise AssertionError(
            f"scene predicate authority launched optional checker: {executable}"
        )


guard = _RejectForbiddenAuthorityImport()
sys.meta_path.insert(0, guard)

real_which = shutil.which


def guarded_which(command, *args, **kwargs):
    _reject_forbidden_executable(command)
    return real_which(command, *args, **kwargs)


shutil.which = guarded_which


def audit_process_launch(event, args):
    if event in {"subprocess.Popen", "os.exec", "os.posix_spawn", "os.system"}:
        if args:
            _reject_forbidden_executable(args[0])


sys.addaudithook(audit_process_launch)

frontend = importlib.import_module("bongard.object_scene_visual_frontend")
semantic = importlib.import_module("bongard.object_scene_semantic_registry")
ir = importlib.import_module("bongard.object_bongard_scene_predicate_ir")
calibration = importlib.import_module(
    "bongard.object_bongard_scene_predicate_calibration_command"
)
campaign = importlib.import_module(
    "bongard.object_bongard_scene_predicate_campaign_command"
)

# Exercise the dependency and protocol identity paths used by build/replay.
digests = {
    "frontend_source": frontend.object_scene_visual_frontend_source_digest(),
    "frontend_inventory_protocol": frontend.object_scene_inventory_protocol_digest(),
    "frontend_transcript_protocol": frontend.object_scene_transcript_protocol_digest(),
    "semantic_source": semantic.object_scene_semantic_registry_source_digest(),
    "semantic_protocol": semantic.object_scene_semantic_registry_protocol_digest(),
    "ir_source": ir.object_bongard_scene_predicate_ir_source_digest(),
    "calibration_source": (
        calibration.object_bongard_scene_predicate_calibration_command_source_digest()
    ),
    "campaign_source": (
        campaign.object_bongard_scene_predicate_campaign_command_source_digest()
    ),
}
if any(
    not isinstance(value, str) or len(value) != 64
    for value in digests.values()
):
    raise AssertionError("scene predicate authority emitted a malformed source identity")

for module in (frontend, semantic, ir, calibration, campaign):
    authority = module._authority_data()
    if authority.get("python_is_canonical_authority") is not True:
        raise AssertionError("scene predicate authority is not Python-canonical")
    if authority.get("lean_required") is not False:
        raise AssertionError("scene predicate authority requires an optional checker")

bindings = campaign._automatic_release_source_bindings()
if set(bindings) != {"batch_source", "release_gate_source"}:
    raise AssertionError("campaign automatic source closure differs")
if guard.attempts or forbidden_modules.intersection(sys.modules):
    raise AssertionError("optional checker entered the isolated import graph")

print(json.dumps({"status": "ok", "digests": digests}, sort_keys=True))
"""


def test_scene_predicate_authority_imports_in_isolated_unlean_process() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            textwrap.dedent(_ISOLATED_SCENE_AUTHORITY_SMOKE),
            str(WORKSPACE_ROOT),
        ],
        cwd=WORKSPACE_ROOT,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr.decode(
        "utf-8", errors="replace"
    )
    result = json.loads(completed.stdout)
    assert result["status"] == "ok"
    assert set(result["digests"]) == {
        "frontend_source",
        "frontend_inventory_protocol",
        "frontend_transcript_protocol",
        "semantic_source",
        "semantic_protocol",
        "ir_source",
        "calibration_source",
        "campaign_source",
    }
