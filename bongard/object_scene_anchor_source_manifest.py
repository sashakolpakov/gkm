"""Static exact-byte source closure for the anchor benchmark command.

The builder in this module never imports the root module or any module it
discovers.  It resolves local ``bongard`` imports from syntax trees, hashes the
exact source bytes, and records only repository-relative paths.  This makes the
same artifact usable before exposure and in a model-free cold replay process.
"""

from __future__ import annotations

from bongard.runtime_source_snapshot import capture_loaded_source, verify_loaded_source


_LOADED_SOURCE_SHA256 = capture_loaded_source(__name__, __file__)

import ast
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Any, Mapping


OBJECT_SCENE_ANCHOR_SOURCE_MANIFEST_SCHEMA = (
    "gkm.object-scene-anchor-source-manifest.v1"
)
OBJECT_SCENE_ANCHOR_SOURCE_MANIFEST_ENTRY_SCHEMA = (
    "gkm.object-scene-anchor-source-manifest-entry.v1"
)
OBJECT_SCENE_ANCHOR_SOURCE_MANIFEST_ID = (
    "bongard.object-scene-anchor-source-manifest/ast-local-closure-v1"
)
DEFAULT_ROOT_MODULE = "bongard.object_scene_anchor_benchmark_command"
PACKAGE_PREFIX = "bongard"

_REPOSITORY_ROOT = Path(__file__).resolve(strict=True).parent.parent
_MODULE_NAME = re.compile(r"bongard(?:\.[A-Za-z_][A-Za-z0-9_]*)*\Z")
_RAW_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_MAX_SOURCE_BYTES = 64 * 1024 * 1024


class ObjectSceneAnchorSourceManifestError(ValueError):
    """A source root, import edge, source file, or manifest differs."""


def object_scene_anchor_source_manifest_source_digest() -> str:
    """Return the import-time-sealed exact source digest of this implementation."""

    return verify_loaded_source(__name__, expected_source_sha256=_LOADED_SOURCE_SHA256)


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8", errors="strict")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ObjectSceneAnchorSourceManifestError(
            "source manifest is not canonical-JSON encodable"
        ) from exc


def _canonical_digest(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if (
        not isinstance(value, Mapping)
        or any(not isinstance(key, str) for key in value)
    ):
        raise ObjectSceneAnchorSourceManifestError(f"{label} must be a JSON object")
    return value


def _module_name(value: object, label: str = "module name") -> str:
    if not isinstance(value, str) or _MODULE_NAME.fullmatch(value) is None:
        raise ObjectSceneAnchorSourceManifestError(
            f"{label} must be an absolute local bongard module name"
        )
    return value


def _raw_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or _RAW_SHA256.fullmatch(value) is None:
        raise ObjectSceneAnchorSourceManifestError(
            f"{label} must be raw lowercase SHA-256"
        )
    return value


def _entry_content(
    value: "ObjectSceneAnchorSourceManifestEntry",
) -> dict[str, object]:
    return {
        "schema": OBJECT_SCENE_ANCHOR_SOURCE_MANIFEST_ENTRY_SCHEMA,
        "module_name": value.module_name,
        "relative_path": value.relative_path,
        "source_sha256": value.source_sha256,
        "source_byte_count": value.source_byte_count,
    }


@dataclass(frozen=True, slots=True)
class ObjectSceneAnchorSourceManifestEntry:
    module_name: str
    relative_path: str
    source_sha256: str
    source_byte_count: int

    def __post_init__(self) -> None:
        module = _module_name(self.module_name)
        if (
            not isinstance(self.relative_path, str)
            or not self.relative_path
            or "\\" in self.relative_path
        ):
            raise ObjectSceneAnchorSourceManifestError(
                "source relative path must be nonempty POSIX text"
            )
        relative = Path(self.relative_path)
        if (
            relative.is_absolute()
            or relative.as_posix() != self.relative_path
            or any(part in ("", ".", "..") for part in relative.parts)
            or relative.parts[0] != PACKAGE_PREFIX
            or relative.suffix != ".py"
        ):
            raise ObjectSceneAnchorSourceManifestError(
                "source relative path is unsafe"
            )
        parts = module.split(".")
        module_file = Path(*parts).with_suffix(".py").as_posix()
        package_file = (Path(*parts) / "__init__.py").as_posix()
        if self.relative_path not in (module_file, package_file):
            raise ObjectSceneAnchorSourceManifestError(
                "source path does not correspond to its module name"
            )
        _raw_sha256(self.source_sha256, "entry source digest")
        if (
            type(self.source_byte_count) is not int
            or not 0 <= self.source_byte_count <= _MAX_SOURCE_BYTES
        ):
            raise ObjectSceneAnchorSourceManifestError(
                "entry source byte count is invalid"
            )

    def to_data(self) -> dict[str, object]:
        return _entry_content(self)

    @classmethod
    def from_data(
        cls, value: object
    ) -> "ObjectSceneAnchorSourceManifestEntry":
        raw = _mapping(value, "source manifest entry")
        if set(raw) != {
            "schema",
            "module_name",
            "relative_path",
            "source_sha256",
            "source_byte_count",
        } or raw.get("schema") != OBJECT_SCENE_ANCHOR_SOURCE_MANIFEST_ENTRY_SCHEMA:
            raise ObjectSceneAnchorSourceManifestError(
                "source manifest entry fields differ"
            )
        result = cls(
            raw["module_name"],
            raw["relative_path"],
            raw["source_sha256"],
            raw["source_byte_count"],
        )
        if result.to_data() != dict(raw):
            raise ObjectSceneAnchorSourceManifestError(
                "source manifest entry is not canonical"
            )
        return result


def _manifest_content(value: "ObjectSceneAnchorSourceManifest") -> dict[str, object]:
    return {
        "schema": OBJECT_SCENE_ANCHOR_SOURCE_MANIFEST_SCHEMA,
        "algorithm_id": OBJECT_SCENE_ANCHOR_SOURCE_MANIFEST_ID,
        "algorithm_source_sha256": value.algorithm_source_sha256,
        "root_module": value.root_module,
        "package_prefix": PACKAGE_PREFIX,
        "entries": [item.to_data() for item in value.entries],
        "module_count": len(value.entries),
        "resolution_rule": "recursive-static-ast-local-import-closure",
        "source_hash_rule": "sha256-over-exact-file-bytes",
        "target_modules_imported_or_executed": False,
        "absolute_paths_persisted": False,
    }


@dataclass(frozen=True, slots=True)
class ObjectSceneAnchorSourceManifest:
    algorithm_source_sha256: str
    root_module: str
    entries: tuple[ObjectSceneAnchorSourceManifestEntry, ...]
    manifest_digest: str

    def __post_init__(self) -> None:
        _raw_sha256(self.algorithm_source_sha256, "manifest algorithm source digest")
        _module_name(self.root_module, "root module")
        if type(self.entries) is not tuple or any(
            type(item) is not ObjectSceneAnchorSourceManifestEntry
            for item in self.entries
        ):
            raise TypeError("manifest entries must be exact entry values")
        modules = tuple(item.module_name for item in self.entries)
        paths = tuple(item.relative_path for item in self.entries)
        if (
            not self.entries
            or modules != tuple(sorted(modules))
            or len(set(modules)) != len(modules)
            or len(set(paths)) != len(paths)
            or self.root_module not in set(modules)
        ):
            raise ObjectSceneAnchorSourceManifestError(
                "manifest entries must be nonempty, unique, sorted, and contain the root"
            )
        _raw_sha256(self.manifest_digest, "source manifest digest")
        if self.manifest_digest != _canonical_digest(_manifest_content(self)):
            raise ObjectSceneAnchorSourceManifestError(
                "source manifest digest differs"
            )

    def to_data(self) -> dict[str, object]:
        return {**_manifest_content(self), "manifest_digest": self.manifest_digest}

    @classmethod
    def from_data(cls, value: object) -> "ObjectSceneAnchorSourceManifest":
        raw = _mapping(value, "source manifest")
        if set(raw) != {
            "schema",
            "algorithm_id",
            "algorithm_source_sha256",
            "root_module",
            "package_prefix",
            "entries",
            "module_count",
            "resolution_rule",
            "source_hash_rule",
            "target_modules_imported_or_executed",
            "absolute_paths_persisted",
            "manifest_digest",
        } or (
            raw.get("schema") != OBJECT_SCENE_ANCHOR_SOURCE_MANIFEST_SCHEMA
            or raw.get("algorithm_id") != OBJECT_SCENE_ANCHOR_SOURCE_MANIFEST_ID
            or raw.get("package_prefix") != PACKAGE_PREFIX
            or raw.get("resolution_rule")
            != "recursive-static-ast-local-import-closure"
            or raw.get("source_hash_rule") != "sha256-over-exact-file-bytes"
            or raw.get("target_modules_imported_or_executed") is not False
            or raw.get("absolute_paths_persisted") is not False
            or not isinstance(raw.get("entries"), list)
            or type(raw.get("module_count")) is not int
            or raw.get("module_count") != len(raw.get("entries", ()))
        ):
            raise ObjectSceneAnchorSourceManifestError(
                "source manifest fields or policy differ"
            )
        result = cls(
            raw["algorithm_source_sha256"],
            raw["root_module"],
            tuple(
                ObjectSceneAnchorSourceManifestEntry.from_data(item)
                for item in raw["entries"]
            ),
            raw["manifest_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectSceneAnchorSourceManifestError(
                "source manifest is not canonical"
            )
        return result


@dataclass(frozen=True, slots=True)
class _ResolvedSource:
    module_name: str
    path: Path
    relative_path: str
    is_package: bool


def _safe_repository_root(value: str | os.PathLike[str] | None) -> Path:
    root = _REPOSITORY_ROOT if value is None else Path(os.path.abspath(os.fspath(value)))
    try:
        metadata = root.lstat()
        resolved = root.resolve(strict=True)
    except OSError as exc:
        raise ObjectSceneAnchorSourceManifestError(
            "source repository root is unavailable"
        ) from exc
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISDIR(metadata.st_mode)
        or resolved != root
    ):
        raise ObjectSceneAnchorSourceManifestError(
            "source repository root must be a real canonical directory"
        )
    package = root / PACKAGE_PREFIX
    try:
        package_metadata = package.lstat()
    except OSError as exc:
        raise ObjectSceneAnchorSourceManifestError(
            "local bongard package is unavailable"
        ) from exc
    if (
        stat.S_ISLNK(package_metadata.st_mode)
        or not stat.S_ISDIR(package_metadata.st_mode)
        or package.resolve(strict=True) != package
    ):
        raise ObjectSceneAnchorSourceManifestError(
            "local bongard package is unsafe"
        )
    return root


def _candidate_status(path: Path, root: Path) -> bool:
    """Return existence while rejecting symlinks and non-regular candidates."""

    if not os.path.lexists(path):
        return False
    try:
        metadata = path.lstat()
        resolved = path.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as exc:
        raise ObjectSceneAnchorSourceManifestError(
            "local source candidate escapes its repository"
        ) from exc
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISREG(metadata.st_mode)
        or resolved != path
    ):
        raise ObjectSceneAnchorSourceManifestError(
            f"local source candidate is unsafe: {path.relative_to(root).as_posix()}"
        )
    return True


def _resolve_source(module_name: str, root: Path) -> _ResolvedSource:
    module = _module_name(module_name)
    parts = module.split(".")
    module_path = root.joinpath(*parts).with_suffix(".py")
    package_path = root.joinpath(*parts, "__init__.py")
    has_module = _candidate_status(module_path, root)
    has_package = _candidate_status(package_path, root)
    if has_module and has_package:
        raise ObjectSceneAnchorSourceManifestError(
            f"local module has duplicate file/package resolutions: {module}"
        )
    if not has_module and not has_package:
        raise ObjectSceneAnchorSourceManifestError(
            f"unresolved local import: {module}"
        )
    path = package_path if has_package else module_path
    try:
        relative = path.relative_to(root).as_posix()
    except ValueError as exc:  # pragma: no cover - defended by candidate check
        raise ObjectSceneAnchorSourceManifestError(
            "resolved local source escapes its repository"
        ) from exc
    return _ResolvedSource(module, path, relative, has_package)


def _probe_source(module_name: str, root: Path) -> _ResolvedSource | None:
    """Resolve an ambiguous from-import child when it is a real submodule."""

    module = _module_name(module_name)
    parts = module.split(".")
    module_path = root.joinpath(*parts).with_suffix(".py")
    package_path = root.joinpath(*parts, "__init__.py")
    has_module = _candidate_status(module_path, root)
    has_package = _candidate_status(package_path, root)
    if has_module and has_package:
        raise ObjectSceneAnchorSourceManifestError(
            f"local module has duplicate file/package resolutions: {module}"
        )
    if not has_module and not has_package:
        return None
    path = package_path if has_package else module_path
    return _ResolvedSource(
        module,
        path,
        path.relative_to(root).as_posix(),
        has_package,
    )


def _relative_import_base(
    node: ast.ImportFrom,
    *,
    current: _ResolvedSource,
) -> str:
    package = (
        current.module_name
        if current.is_package
        else current.module_name.rpartition(".")[0]
    )
    package_parts = package.split(".") if package else []
    if node.level <= 0:
        raise AssertionError("relative import resolver requires a positive level")
    ascend = node.level - 1
    if ascend >= len(package_parts):
        raise ObjectSceneAnchorSourceManifestError(
            f"relative import escapes bongard package in {current.module_name}"
        )
    base_parts = package_parts[: len(package_parts) - ascend]
    if node.module:
        base_parts.extend(node.module.split("."))
    base = ".".join(base_parts)
    return _module_name(base, "relative local import")


def _local_imports(
    source: _ResolvedSource,
    payload: bytes,
    *,
    root: Path,
) -> tuple[str, ...]:
    try:
        tree = ast.parse(payload, filename=source.relative_path, mode="exec")
    except (SyntaxError, UnicodeError, ValueError) as exc:
        raise ObjectSceneAnchorSourceManifestError(
            f"local source cannot be parsed: {source.relative_path}"
        ) from exc
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == PACKAGE_PREFIX or alias.name.startswith(
                    PACKAGE_PREFIX + "."
                ):
                    found.add(_module_name(alias.name, "absolute local import"))
            continue
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.level:
            base = _relative_import_base(node, current=source)
        else:
            if not node.module or not (
                node.module == PACKAGE_PREFIX
                or node.module.startswith(PACKAGE_PREFIX + ".")
            ):
                continue
            base = _module_name(node.module, "absolute local from-import")
        base_source = _resolve_source(base, root)
        found.add(base)
        if base_source.is_package:
            for alias in node.names:
                if alias.name == "*":
                    continue
                child = _probe_source(f"{base}.{alias.name}", root)
                if child is not None:
                    found.add(child.module_name)
    return tuple(sorted(found))


def _parent_packages(module_name: str, root: Path) -> tuple[str, ...]:
    parts = _module_name(module_name).split(".")
    parents: list[str] = []
    for index in range(1, len(parts)):
        parent = ".".join(parts[:index])
        resolved = _resolve_source(parent, root)
        if not resolved.is_package:
            raise ObjectSceneAnchorSourceManifestError(
                f"local import parent is not a package: {parent}"
            )
        parents.append(parent)
    return tuple(parents)


def _read_source(source: _ResolvedSource, root: Path) -> bytes:
    if not _candidate_status(source.path, root):  # pragma: no cover - resolved prior
        raise ObjectSceneAnchorSourceManifestError("resolved source disappeared")
    try:
        payload = source.path.read_bytes()
    except OSError as exc:
        raise ObjectSceneAnchorSourceManifestError(
            f"local source cannot be read: {source.relative_path}"
        ) from exc
    if len(payload) > _MAX_SOURCE_BYTES:
        raise ObjectSceneAnchorSourceManifestError(
            f"local source exceeds byte bound: {source.relative_path}"
        )
    return payload


def build_object_scene_anchor_source_manifest(
    *,
    root_module: str = DEFAULT_ROOT_MODULE,
    repository_root: str | os.PathLike[str] | None = None,
) -> ObjectSceneAnchorSourceManifest:
    """Build the recursive local source closure without importing target code."""

    root = _safe_repository_root(repository_root)
    requested = _module_name(root_module, "root module")
    pending: set[str] = {requested, *_parent_packages(requested, root)}
    resolved_by_module: dict[str, _ResolvedSource] = {}
    module_by_path: dict[str, str] = {}
    payload_by_module: dict[str, bytes] = {}
    while pending:
        module = min(pending)
        pending.remove(module)
        if module in resolved_by_module:
            continue
        source = _resolve_source(module, root)
        previous = module_by_path.setdefault(source.relative_path, module)
        if previous != module:
            raise ObjectSceneAnchorSourceManifestError(
                "two local module names resolve to the same source path"
            )
        payload = _read_source(source, root)
        resolved_by_module[module] = source
        payload_by_module[module] = payload
        imported = _local_imports(source, payload, root=root)
        for imported_module in imported:
            pending.update(_parent_packages(imported_module, root))
            pending.add(imported_module)
    entries = tuple(
        ObjectSceneAnchorSourceManifestEntry(
            module,
            resolved_by_module[module].relative_path,
            hashlib.sha256(payload_by_module[module]).hexdigest(),
            len(payload_by_module[module]),
        )
        for module in sorted(resolved_by_module)
    )
    values = {
        "algorithm_source_sha256": (
            object_scene_anchor_source_manifest_source_digest()
        ),
        "root_module": requested,
        "entries": entries,
    }
    provisional = object.__new__(ObjectSceneAnchorSourceManifest)
    for name, item in values.items():
        object.__setattr__(provisional, name, item)
    return ObjectSceneAnchorSourceManifest(
        **values,
        manifest_digest=_canonical_digest(_manifest_content(provisional)),
    )


def cold_verify_object_scene_anchor_source_manifest(
    manifest: ObjectSceneAnchorSourceManifest,
    *,
    repository_root: str | os.PathLike[str] | None = None,
    expected_manifest_digest: str | None = None,
) -> ObjectSceneAnchorSourceManifest:
    """Rebuild the complete closure from current bytes and require exact equality."""

    if type(manifest) is not ObjectSceneAnchorSourceManifest:
        raise TypeError("manifest must be exact ObjectSceneAnchorSourceManifest")
    restored = ObjectSceneAnchorSourceManifest.from_data(manifest.to_data())
    if expected_manifest_digest is not None and restored.manifest_digest != _raw_sha256(
        expected_manifest_digest, "expected source manifest digest"
    ):
        raise ObjectSceneAnchorSourceManifestError(
            "source manifest differs from its external commitment"
        )
    rebuilt = build_object_scene_anchor_source_manifest(
        root_module=restored.root_module,
        repository_root=repository_root,
    )
    if rebuilt != restored:
        raise ObjectSceneAnchorSourceManifestError(
            "source manifest differs from current exact source closure"
        )
    return restored


__all__ = (
    "DEFAULT_ROOT_MODULE",
    "OBJECT_SCENE_ANCHOR_SOURCE_MANIFEST_ENTRY_SCHEMA",
    "OBJECT_SCENE_ANCHOR_SOURCE_MANIFEST_ID",
    "OBJECT_SCENE_ANCHOR_SOURCE_MANIFEST_SCHEMA",
    "ObjectSceneAnchorSourceManifest",
    "ObjectSceneAnchorSourceManifestEntry",
    "ObjectSceneAnchorSourceManifestError",
    "build_object_scene_anchor_source_manifest",
    "cold_verify_object_scene_anchor_source_manifest",
    "object_scene_anchor_source_manifest_source_digest",
)
