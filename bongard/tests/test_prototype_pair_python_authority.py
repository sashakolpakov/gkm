from __future__ import annotations

import ast
from collections import deque
from dataclasses import dataclass
import importlib.util
from pathlib import Path
from typing import Iterable


PACKAGE_ROOT = Path(__file__).resolve().parents[1]

# These are superseded, untracked evaluator implementations.  None may be
# reachable after the already-hashed compatibility module is reduced to its
# intended one-symbol re-export.
SUPERSEDED_IMPLEMENTATION_MODULES = frozenset(
    {
        "bongard.calibrated_vision",
        "bongard.component_subject_policy",
        "bongard.component_vision_calibration_corpus",
        "bongard.component_vision_observer",
        "bongard.fixed_vision_calibration",
        "bongard.grounded_headless_runner",
        "bongard.grounded_support_version_space",
        "bongard.headless_security",
        "bongard.multimodal_headless_runner",
        "bongard.multimodal_predicates",
        "bongard.multimodal_release_adapter",
        "bongard.multimodal_support_version_space",
        "bongard.repeated_attribute_cohort",
        "bongard.support_version_space",
        "bongard.typed_vision_objects",
        "bongard.version_space_headless_runner",
        "bongard.vision_calibration_corpus",
        "bongard.vision_calibration_family",
        "bongard.vision_calibration_stratified_corpus",
        "bongard.vision_observer_transport",
    }
)

AUTHORITY_LEAF = "bongard.python_predicate_authority"
COMPATIBILITY_SHIM = "bongard.grounded_multimodal_predicates"
COHORT_MODULE = "bongard.prototype_pair_cohort"
FORBIDDEN_CHECKER_MODULES = frozenset(
    {
        "bongard.semantic_checker",
        "bongard.predicate_backend",
    }
)
FORBIDDEN_EXECUTABLES = frozenset({"lean", "lean4", "lake", "elan"})


@dataclass(frozen=True, slots=True)
class ImportRecord:
    source: str
    target: str
    names: tuple[str, ...]
    line: int


def _module_path(module: str) -> Path | None:
    if module == "bongard":
        path = PACKAGE_ROOT / "__init__.py"
    elif module.startswith("bongard."):
        relative = module.removeprefix("bongard.").replace(".", "/")
        module_file = PACKAGE_ROOT / f"{relative}.py"
        package_file = PACKAGE_ROOT / relative / "__init__.py"
        path = module_file if module_file.is_file() else package_file
    else:
        return None
    return path if path.is_file() else None


def _tree(module: str) -> ast.Module:
    path = _module_path(module)
    assert path is not None, f"active local module has no source: {module}"
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _resolve_relative(source: str, node: ast.ImportFrom) -> str | None:
    if node.level == 0:
        return node.module
    path = _module_path(source)
    assert path is not None
    package = source if path.name == "__init__.py" else source.rpartition(".")[0]
    relative = "." * node.level + (node.module or "")
    try:
        return importlib.util.resolve_name(relative, package)
    except (ImportError, ValueError):
        return None


def _imports(module: str) -> tuple[ImportRecord, ...]:
    records: list[ImportRecord] = []
    for node in ast.walk(_tree(module)):
        if isinstance(node, ast.Import):
            records.extend(
                ImportRecord(module, item.name, (), node.lineno)
                for item in node.names
                if item.name == "bongard" or item.name.startswith("bongard.")
            )
        elif isinstance(node, ast.ImportFrom):
            target = _resolve_relative(module, node)
            if target == "bongard" or (
                isinstance(target, str) and target.startswith("bongard.")
            ):
                records.append(
                    ImportRecord(
                        module,
                        target,
                        tuple(item.name for item in node.names),
                        node.lineno,
                    )
                )
    return tuple(records)


def _literal_dict_assignment(module: str, name: str) -> dict[str, str]:
    for node in _tree(module).body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == name
        ):
            value = ast.literal_eval(node.value)
            assert isinstance(value, dict)
            assert all(isinstance(key, str) for key in value)
            assert all(isinstance(item, str) for item in value.values())
            return value
    raise AssertionError(f"{module} has no literal {name}")


def _frozenset_literal_assignment(module: str, name: str) -> frozenset[str]:
    for node in _tree(module).body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == name
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "frozenset"
            and len(node.value.args) == 1
        ):
            value = ast.literal_eval(node.value.args[0])
            assert isinstance(value, set)
            assert all(isinstance(item, str) for item in value)
            return frozenset(value)
    raise AssertionError(f"{module} has no literal frozenset {name}")


def _projected_imports(module: str) -> Iterable[ImportRecord]:
    if module == COMPATIBILITY_SHIM:
        # The cohort and campaign source bytes remain untouched because their
        # digests are preregistered.  Only this explicitly hashed compatibility
        # module is slimmed after the campaign proof.
        yield ImportRecord(
            source=COMPATIBILITY_SHIM,
            target=AUTHORITY_LEAF,
            names=("PYTHON_PREDICATE_AUTHORITY_ID",),
            line=0,
        )
        return
    yield from _imports(module)


def _closure(roots: Iterable[str]) -> frozenset[str]:
    pending = deque(sorted(set(roots) | {"bongard"}))
    visited: set[str] = set()
    while pending:
        module = pending.popleft()
        if module in visited:
            continue
        path = _module_path(module)
        if path is None:
            continue
        visited.add(module)
        for record in _projected_imports(module):
            if record.target not in visited and _module_path(record.target) is not None:
                pending.append(record.target)
    return frozenset(visited)


def _static_executable_tokens(module: str) -> frozenset[str]:
    values: set[str] = set()
    for node in ast.walk(_tree(module)):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            token = Path(node.value).name.lower()
            if token in FORBIDDEN_EXECUTABLES:
                values.add(token)
        elif isinstance(node, (ast.Tuple, ast.List)):
            for item in node.elts:
                if isinstance(item, ast.Constant) and isinstance(item.value, str):
                    token = Path(item.value).name.lower()
                    if token in FORBIDDEN_EXECUTABLES:
                        values.add(token)
    return frozenset(values)


def test_only_current_compatibility_debt_is_behind_the_hashed_shim() -> None:
    campaign_sources = _literal_dict_assignment(
        "bongard.prototype_pair_campaign", "_RUNTIME_SOURCE_MODULES"
    )
    required_roles = _frozenset_literal_assignment(
        "bongard.prototype_pair_execution_precommit",
        "REQUIRED_RUNTIME_SOURCE_ROLES",
    )
    records = tuple(
        record
        for record in _imports(COHORT_MODULE)
        if record.target == COMPATIBILITY_SHIM
    )
    assert campaign_sources["grounded-compat"] == COMPATIBILITY_SHIM
    assert "grounded-compat" in required_roles
    assert len(records) == 1
    assert records[0].source == COHORT_MODULE
    assert records[0].target == COMPATIBILITY_SHIM
    assert records[0].names == ("PYTHON_PREDICATE_AUTHORITY_ID",)

    closure = _closure(campaign_sources.values())
    direct_retired_edges_outside_shim = {
        (record.source, record.target, record.names)
        for module in closure
        if module != COMPATIBILITY_SHIM
        if _module_path(module) is not None
        for record in _imports(module)
        if record.target in SUPERSEDED_IMPLEMENTATION_MODULES
    }
    assert not direct_retired_edges_outside_shim

    # Until the campaign is proved, the checked compatibility module retains
    # its old body.  Its remaining old imports are explicit and bounded.  Once
    # slimmed, this set becomes empty without changing the test.
    current_shim_imports = _imports(COMPATIBILITY_SHIM)
    current_shim_retired_targets = frozenset(
        record.target
        for record in current_shim_imports
        if record.target in SUPERSEDED_IMPLEMENTATION_MODULES
    )
    assert current_shim_retired_targets in {
        frozenset(),
        frozenset(
            {
                "bongard.component_subject_policy",
                "bongard.component_vision_observer",
                "bongard.typed_vision_objects",
                "bongard.vision_calibration_family",
            }
        ),
    }
    assert not {
        record.target
        for record in current_shim_imports
        if record.target in FORBIDDEN_CHECKER_MODULES
    }
    assert not _static_executable_tokens(COMPATIBILITY_SHIM)
    if not current_shim_retired_targets:
        assert tuple(
            (record.target, record.names) for record in current_shim_imports
        ) == ((AUTHORITY_LEAF, ("PYTHON_PREDICATE_AUTHORITY_ID",)),)


def test_projected_hashed_shim_closure_has_no_old_or_lean_backend() -> None:
    campaign_sources = _literal_dict_assignment(
        "bongard.prototype_pair_campaign", "_RUNTIME_SOURCE_MODULES"
    )
    roots = set(campaign_sources.values())
    closure = _closure(roots)

    assert AUTHORITY_LEAF in closure
    assert COMPATIBILITY_SHIM in closure
    assert closure.isdisjoint(SUPERSEDED_IMPLEMENTATION_MODULES)
    assert closure.isdisjoint(FORBIDDEN_CHECKER_MODULES)
    assert all("lean" not in module.lower().split(".") for module in closure)

    imported = {
        record.target for module in closure for record in _projected_imports(module)
    }
    assert imported.isdisjoint(SUPERSEDED_IMPLEMENTATION_MODULES)
    assert imported.isdisjoint(FORBIDDEN_CHECKER_MODULES)
    assert not {
        token
        for module in closure
        if module != COMPATIBILITY_SHIM
        for token in _static_executable_tokens(module)
    }


def test_package_init_keeps_superseded_exports_lazy() -> None:
    eager_targets = {record.target for record in _imports("bongard")}
    assert eager_targets.isdisjoint(SUPERSEDED_IMPLEMENTATION_MODULES)
    assert eager_targets.isdisjoint(FORBIDDEN_CHECKER_MODULES)
