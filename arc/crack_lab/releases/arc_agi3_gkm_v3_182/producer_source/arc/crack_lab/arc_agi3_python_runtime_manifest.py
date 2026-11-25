#!/usr/bin/env python3
"""Exact, replayable evidence for the Python runtime used by conformance.

The production preflight must not treat one launcher-file digest as evidence
for all of the mutable code that Python subsequently imports.  This module
builds and reopens a canonical manifest covering the venv/base-runtime
identity, every interpreter-path symlink, ``pyvenv.cfg``, the standard library
and native extensions, and pytest's active dependency closure.

Manifest creation is an offline provisioning operation.  Launch admission
loads an already hash-pinned manifest, reopens every recorded byte, and runs
the two exported probes inside the supervisor's bounded process primitive.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import subprocess
from pathlib import Path
from typing import Any, Callable, Sequence


SCHEMA = 1
KIND = "arc_agi3_python_runtime_manifest"
MAX_MANIFEST_BYTES = 64 * 1024 * 1024
MAX_PROBE_BYTES = 16 * 1024 * 1024
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_IDENTITY_FIELDS = (
    "device",
    "inode",
    "mode",
    "links",
    "uid",
    "gid",
    "size",
    "mtime_ns",
    "ctime_ns",
)
_BASE_PROBE_FIELDS = frozenset({
    "implementation",
    "version",
    "version_info",
    "cache_tag",
    "abi_flags",
    "platform",
    "executable",
    "prefix",
    "base_prefix",
    "exec_prefix",
    "base_exec_prefix",
    "isolated_sys_path",
    "stdlib",
    "platstdlib",
    "purelib",
    "platlib",
    "destshared",
})
_PACKAGE_PROBE_FIELDS = frozenset({
    "pytest_version",
    "pytest_path",
    "import_suffixes",
    "distributions",
})
_DISTRIBUTION_FIELDS = frozenset({
    "name",
    "version",
    "metadata_path",
    "requires",
    "files",
})


class RuntimeManifestError(RuntimeError):
    """The selected Python runtime is incomplete, mutable, or substituted."""


def canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def manifest_sha256(value: object) -> str:
    return sha256_bytes(canonical_json(value) + b"\n")


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and SHA256_RE.fullmatch(value) is not None
    )


def _identity(metadata: os.stat_result) -> dict[str, int]:
    return {
        "device": metadata.st_dev,
        "inode": metadata.st_ino,
        "mode": stat.S_IMODE(metadata.st_mode),
        "links": metadata.st_nlink,
        "uid": metadata.st_uid,
        "gid": metadata.st_gid,
        "size": metadata.st_size,
        "mtime_ns": metadata.st_mtime_ns,
        "ctime_ns": metadata.st_ctime_ns,
    }


def _validate_identity(value: object) -> None:
    if (
        not isinstance(value, dict)
        or set(value) != set(_IDENTITY_FIELDS)
        or any(
            isinstance(value[field], bool)
            or not isinstance(value[field], int)
            or value[field] < 0
            for field in _IDENTITY_FIELDS
        )
    ):
        raise RuntimeManifestError("runtime path identity is malformed")


def _read_regular(path: Path) -> tuple[bytes, os.stat_result]:
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
        )
    except OSError as exc:
        raise RuntimeManifestError(
            f"runtime file is unavailable or aliased: {path}"
        ) from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size < 0
        ):
            raise RuntimeManifestError(
                f"runtime file is not unaliased and regular: {path}"
            )
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                raise RuntimeManifestError(
                    f"runtime file changed while reading: {path}"
                )
            chunks.append(chunk)
            remaining -= len(chunk)
        after = os.fstat(descriptor)
        if _identity(before) != _identity(after):
            raise RuntimeManifestError(
                f"runtime file changed while reading: {path}"
            )
        return b"".join(chunks), after
    finally:
        os.close(descriptor)


def _file_record(path: Path) -> dict[str, Any]:
    raw, metadata = _read_regular(path)
    return {
        "path": str(path),
        "kind": "file",
        "identity": _identity(metadata),
        "sha256": sha256_bytes(raw),
    }


def _directory_record(path: Path) -> dict[str, Any]:
    try:
        metadata = os.stat(path, follow_symlinks=False)
    except OSError as exc:
        raise RuntimeManifestError(
            f"runtime directory is unavailable: {path}"
        ) from exc
    if not stat.S_ISDIR(metadata.st_mode):
        raise RuntimeManifestError(
            f"runtime path is not a directory: {path}"
        )
    return {
        "path": str(path),
        "kind": "directory",
        "identity": _identity(metadata),
    }


def _symlink_record(path: Path) -> dict[str, Any]:
    try:
        metadata = os.stat(path, follow_symlinks=False)
        target = os.readlink(path)
    except OSError as exc:
        raise RuntimeManifestError(
            f"runtime symlink is unavailable: {path}"
        ) from exc
    if not stat.S_ISLNK(metadata.st_mode) or not target:
        raise RuntimeManifestError(
            f"runtime path is not a symlink: {path}"
        )
    return {
        "path": str(path),
        "kind": "symlink",
        "identity": _identity(metadata),
        "target": target,
    }


def _normal_absolute(path: Path, *, label: str) -> Path:
    value = Path(path)
    if (
        not value.is_absolute()
        or "\x00" in str(value)
        or Path(os.path.normpath(value)) != value
    ):
        raise RuntimeManifestError(
            f"{label} must be a normalized absolute path"
        )
    return value


def _resolution_evidence(path: Path) -> dict[str, Any]:
    requested = _normal_absolute(path, label="runtime path")
    pending = list(requested.parts[1:])
    resolved = Path(requested.anchor)
    links: list[dict[str, Any]] = []
    seen: set[tuple[str, tuple[str, ...]]] = set()
    while pending:
        component = pending.pop(0)
        candidate = resolved / component
        try:
            metadata = os.stat(candidate, follow_symlinks=False)
        except OSError as exc:
            raise RuntimeManifestError(
                f"runtime path cannot be resolved: {requested}"
            ) from exc
        if not stat.S_ISLNK(metadata.st_mode):
            resolved = candidate
            continue
        marker = (str(candidate), tuple(pending))
        if marker in seen or len(seen) >= 64:
            raise RuntimeManifestError(
                f"runtime symlink cycle is not admissible: {requested}"
            )
        seen.add(marker)
        record = _symlink_record(candidate)
        links.append(record)
        target = Path(record["target"])
        if not target.is_absolute():
            target = candidate.parent / target
        combined = Path(
            os.path.normpath(
                os.path.join(str(target), *pending)
            )
        )
        if not combined.is_absolute():
            raise RuntimeManifestError(
                f"runtime symlink escaped absolute resolution: {candidate}"
            )
        pending = list(combined.parts[1:])
        resolved = Path(combined.anchor)
    try:
        final_metadata = os.stat(resolved, follow_symlinks=False)
    except OSError as exc:
        raise RuntimeManifestError(
            f"resolved runtime target is unavailable: {resolved}"
        ) from exc
    if stat.S_ISREG(final_metadata.st_mode):
        final = _file_record(resolved)
    elif stat.S_ISDIR(final_metadata.st_mode):
        final = _directory_record(resolved)
    else:
        raise RuntimeManifestError(
            f"resolved runtime target has unsupported type: {resolved}"
        )
    return {
        "requested_path": str(requested),
        "resolved_path": str(resolved),
        "symlinks": links,
        "resolved_target": final,
    }


def _validate_resolution(value: object, *, expected: Path) -> None:
    if (
        not isinstance(value, dict)
        or set(value)
        != {
            "requested_path",
            "resolved_path",
            "symlinks",
            "resolved_target",
        }
        or value["requested_path"] != str(expected)
        or not isinstance(value["resolved_path"], str)
        or not Path(value["resolved_path"]).is_absolute()
        or not isinstance(value["symlinks"], list)
    ):
        raise RuntimeManifestError(
            "runtime path-resolution evidence is malformed"
        )
    for link in value["symlinks"]:
        if (
            not isinstance(link, dict)
            or set(link) != {"path", "kind", "identity", "target"}
            or link["kind"] != "symlink"
            or not isinstance(link["path"], str)
            or not Path(link["path"]).is_absolute()
            or not isinstance(link["target"], str)
            or not link["target"]
        ):
            raise RuntimeManifestError(
                "runtime symlink-chain evidence is malformed"
            )
        _validate_identity(link["identity"])
    final = value["resolved_target"]
    if (
        not isinstance(final, dict)
        or final.get("path") != value["resolved_path"]
        or final.get("kind") not in {"file", "directory"}
    ):
        raise RuntimeManifestError(
            "resolved runtime-target evidence is malformed"
        )
    expected_fields = (
        {"path", "kind", "identity", "sha256"}
        if final["kind"] == "file"
        else {"path", "kind", "identity"}
    )
    if set(final) != expected_fields:
        raise RuntimeManifestError(
            "resolved runtime-target schema is malformed"
        )
    _validate_identity(final["identity"])
    if (
        final["kind"] == "file"
        and not _valid_sha256(final["sha256"])
    ):
        raise RuntimeManifestError(
            "resolved runtime-target digest is malformed"
        )


def _tree_records(
    root: Path,
    *,
    exclude_top_level: frozenset[str] = frozenset(),
) -> tuple[list[dict[str, Any]], int]:
    selected = _normal_absolute(root, label="runtime tree root")
    records: list[dict[str, Any]] = []
    total_bytes = 0

    def visit(directory: Path, relative: Path) -> None:
        nonlocal total_bytes
        try:
            entries = sorted(
                os.scandir(directory), key=lambda entry: entry.name
            )
        except OSError as exc:
            raise RuntimeManifestError(
                f"runtime tree cannot be enumerated: {directory}"
            ) from exc
        for entry in entries:
            if (
                entry.name == "__pycache__"
                or entry.name.endswith((".pyc", ".pyo"))
                or (
                    not relative.parts
                    and entry.name in exclude_top_level
                )
            ):
                continue
            path = directory / entry.name
            child_relative = relative / entry.name
            try:
                metadata = entry.stat(follow_symlinks=False)
            except OSError as exc:
                raise RuntimeManifestError(
                    f"runtime tree entry is unavailable: {path}"
                ) from exc
            if stat.S_ISDIR(metadata.st_mode):
                records.append({
                    "relative_path": child_relative.as_posix(),
                    "kind": "directory",
                    "identity": _identity(metadata),
                })
                visit(path, child_relative)
            elif stat.S_ISREG(metadata.st_mode):
                file_record = _file_record(path)
                records.append({
                    "relative_path": child_relative.as_posix(),
                    "kind": "file",
                    "identity": file_record["identity"],
                    "sha256": file_record["sha256"],
                })
                total_bytes += file_record["identity"]["size"]
            elif stat.S_ISLNK(metadata.st_mode):
                link = _symlink_record(path)
                records.append({
                    "relative_path": child_relative.as_posix(),
                    "kind": "symlink",
                    "identity": link["identity"],
                    "target": link["target"],
                })
            else:
                raise RuntimeManifestError(
                    f"runtime tree contains unsupported entry: {path}"
                )

    visit(selected, Path())
    return records, total_bytes


def _tree_manifest(
    root: Path,
    *,
    exclude_top_level: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    records, total_bytes = _tree_records(
        root, exclude_top_level=exclude_top_level
    )
    return {
        "root": str(root),
        "root_resolution": _resolution_evidence(root),
        "excluded_top_level": sorted(exclude_top_level),
        "entry_count": len(records),
        "total_file_bytes": total_bytes,
        "entries_sha256": sha256_bytes(canonical_json(records)),
    }


def _paths_manifest(paths: Sequence[str]) -> dict[str, Any]:
    if (
        not paths
        or len(paths) != len(set(paths))
        or any(not Path(path).is_absolute() for path in paths)
    ):
        raise RuntimeManifestError(
            "distribution file projection is malformed"
        )
    records: list[dict[str, Any]] = []
    total_bytes = 0
    for raw_path in paths:
        path = _normal_absolute(
            Path(raw_path), label="distribution file"
        )
        try:
            metadata = os.stat(path, follow_symlinks=False)
        except OSError as exc:
            raise RuntimeManifestError(
                f"distribution file is unavailable: {path}"
            ) from exc
        if stat.S_ISREG(metadata.st_mode):
            record = _file_record(path)
            total_bytes += record["identity"]["size"]
        elif stat.S_ISLNK(metadata.st_mode):
            record = _symlink_record(path)
        else:
            raise RuntimeManifestError(
                f"distribution projection is not a file: {path}"
            )
        records.append(record)
    return {
        "file_count": len(records),
        "total_file_bytes": total_bytes,
        "files_sha256": sha256_bytes(canonical_json(records)),
    }


BASE_RUNTIME_PROBE = r"""
import json
import sys
import sysconfig

paths = sysconfig.get_paths()
value = {
    "implementation": sys.implementation.name,
    "version": sys.version,
    "version_info": list(sys.version_info),
    "cache_tag": sys.implementation.cache_tag,
    "abi_flags": getattr(sys, "abiflags", ""),
    "platform": sys.platform,
    "executable": sys.executable,
    "prefix": sys.prefix,
    "base_prefix": sys.base_prefix,
    "exec_prefix": sys.exec_prefix,
    "base_exec_prefix": sys.base_exec_prefix,
    "isolated_sys_path": list(sys.path),
    "stdlib": paths["stdlib"],
    "platstdlib": paths["platstdlib"],
    "purelib": paths["purelib"],
    "platlib": paths["platlib"],
    "destshared": sysconfig.get_config_var("DESTSHARED"),
}
print(json.dumps(value, sort_keys=True, separators=(",", ":")))
""".strip()


PACKAGE_RUNTIME_PROBE = r"""
import importlib.machinery
import importlib.metadata
import json
import os
import re
import sys

site_root = os.path.normpath(sys.argv[1])
if not os.path.isabs(site_root):
    raise SystemExit("site root is not absolute")
sys.path.append(site_root)

from packaging.requirements import Requirement
import pytest

def canonical_name(value):
    return re.sub(r"[-_.]+", "-", value).lower()

pending = ["pytest"]
seen = set()
distributions = []
while pending:
    selected = canonical_name(pending.pop(0))
    if selected in seen:
        continue
    distribution = importlib.metadata.distribution(selected)
    actual_name = canonical_name(distribution.metadata["Name"])
    if actual_name != selected:
        raise SystemExit("distribution name substitution")
    seen.add(selected)
    requirements = []
    for raw_requirement in distribution.requires or ():
        requirement = Requirement(raw_requirement)
        if (
            requirement.marker is None
            or requirement.marker.evaluate({"extra": ""})
        ):
            dependency = canonical_name(requirement.name)
            requirements.append(dependency)
            if dependency not in seen:
                pending.append(dependency)
    files = []
    metadata_path = None
    for item in distribution.files or ():
        located = os.path.normpath(
            os.path.abspath(distribution.locate_file(item))
        )
        try:
            inside = os.path.commonpath((site_root, located)) == site_root
        except ValueError:
            inside = False
        if not inside:
            continue
        if (
            "__pycache__" in located.split(os.sep)
            or located.endswith((".pyc", ".pyo"))
        ):
            continue
        files.append(located)
        if located.endswith(
            (".dist-info/METADATA", ".egg-info/PKG-INFO")
        ):
            metadata_path = located
    if metadata_path is None:
        raise SystemExit("distribution metadata path is absent")
    distributions.append({
        "name": actual_name,
        "version": distribution.version,
        "metadata_path": metadata_path,
        "requires": sorted(set(requirements)),
        "files": sorted(set(files)),
    })
distributions.sort(key=lambda item: item["name"])
value = {
    "pytest_version": pytest.__version__,
    "pytest_path": os.path.normpath(os.path.abspath(pytest.__file__)),
    "import_suffixes": sorted(set(importlib.machinery.all_suffixes())),
    "distributions": distributions,
}
print(json.dumps(value, sort_keys=True, separators=(",", ":")))
""".strip()


HERMETIC_SUITE_BOOTSTRAP = r"""
import os
import runpy
import sys

site_root, suite_path, manifest_path, manifest_sha256 = sys.argv[1:5]
if (
    not os.path.isabs(site_root)
    or not os.path.isabs(suite_path)
    or not os.path.isabs(manifest_path)
):
    raise SystemExit("hermetic suite paths must be absolute")
sys.path.extend((os.path.dirname(suite_path), site_root))
sys.argv = [
    suite_path,
    "--stdout",
    "--runtime-manifest",
    manifest_path,
    "--runtime-manifest-sha256",
    manifest_sha256,
]
runpy.run_path(suite_path, run_name="__main__")
""".strip()


def base_probe_command(python_executable: Path) -> tuple[str, ...]:
    return (
        str(python_executable),
        "-I",
        "-E",
        "-s",
        "-S",
        "-B",
        "-c",
        BASE_RUNTIME_PROBE,
    )


def package_probe_command(
    python_executable: Path, site_root: Path
) -> tuple[str, ...]:
    return (
        str(python_executable),
        "-I",
        "-E",
        "-s",
        "-S",
        "-B",
        "-c",
        PACKAGE_RUNTIME_PROBE,
        str(site_root),
    )


def suite_command(
    python_executable: Path,
    *,
    site_root: Path,
    suite_path: Path,
    runtime_manifest_path: Path,
    runtime_manifest_sha256: str,
) -> tuple[str, ...]:
    if not _valid_sha256(runtime_manifest_sha256):
        raise RuntimeManifestError(
            "runtime manifest digest is malformed"
        )
    return (
        str(python_executable),
        "-I",
        "-E",
        "-s",
        "-S",
        "-B",
        "-c",
        HERMETIC_SUITE_BOOTSTRAP,
        str(site_root),
        str(suite_path),
        str(runtime_manifest_path),
        runtime_manifest_sha256,
    )


def parse_probe(raw: bytes, *, package: bool) -> dict[str, Any]:
    if not raw or len(raw) > MAX_PROBE_BYTES:
        raise RuntimeManifestError("runtime probe output is not bounded")
    try:
        value = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeManifestError(
            "runtime probe output is not JSON"
        ) from exc
    if raw != canonical_json(value) + b"\n":
        raise RuntimeManifestError(
            "runtime probe output is not canonical JSON"
        )
    _validate_package_probe(value) if package else _validate_base_probe(value)
    return value


def _validate_base_probe(value: object) -> None:
    if (
        not isinstance(value, dict)
        or set(value) != _BASE_PROBE_FIELDS
        or value["implementation"] != "cpython"
        or not isinstance(value["version"], str)
        or not value["version"]
        or not isinstance(value["version_info"], list)
        or len(value["version_info"]) != 5
        or any(
            isinstance(item, bool) or not isinstance(item, int)
            for item in value["version_info"][:3]
        )
        or not isinstance(value["cache_tag"], str)
        or not value["cache_tag"]
        or not isinstance(value["abi_flags"], str)
        or not isinstance(value["platform"], str)
        or not value["platform"]
    ):
        raise RuntimeManifestError("base runtime probe is malformed")
    path_fields = (
        "executable",
        "prefix",
        "base_prefix",
        "exec_prefix",
        "base_exec_prefix",
        "stdlib",
        "platstdlib",
        "purelib",
        "platlib",
        "destshared",
    )
    if any(
        not isinstance(value[field], str)
        or not Path(value[field]).is_absolute()
        for field in path_fields
    ):
        raise RuntimeManifestError(
            "base runtime probe contains a nonabsolute path"
        )
    if (
        value["prefix"] == value["base_prefix"]
        or value["purelib"] != value["platlib"]
        or not isinstance(value["isolated_sys_path"], list)
        or not value["isolated_sys_path"]
        or any(
            not isinstance(path, str) or not Path(path).is_absolute()
            for path in value["isolated_sys_path"]
        )
        or value["purelib"] in value["isolated_sys_path"]
        or value["platlib"] in value["isolated_sys_path"]
    ):
        raise RuntimeManifestError(
            "runtime probe is not an isolated virtual environment"
        )


def _validate_package_probe(value: object) -> None:
    if (
        not isinstance(value, dict)
        or set(value) != _PACKAGE_PROBE_FIELDS
        or not isinstance(value["pytest_version"], str)
        or not value["pytest_version"]
        or not isinstance(value["pytest_path"], str)
        or not Path(value["pytest_path"]).is_absolute()
        or not isinstance(value["import_suffixes"], list)
        or not value["import_suffixes"]
        or value["import_suffixes"]
        != sorted(set(value["import_suffixes"]))
        or not isinstance(value["distributions"], list)
        or not value["distributions"]
    ):
        raise RuntimeManifestError("package runtime probe is malformed")
    names: list[str] = []
    for distribution in value["distributions"]:
        if (
            not isinstance(distribution, dict)
            or set(distribution) != _DISTRIBUTION_FIELDS
            or not isinstance(distribution["name"], str)
            or re.fullmatch(
                r"[a-z0-9]+(?:-[a-z0-9]+)*",
                distribution["name"],
            )
            is None
            or not isinstance(distribution["version"], str)
            or not distribution["version"]
            or not isinstance(distribution["metadata_path"], str)
            or not Path(distribution["metadata_path"]).is_absolute()
            or not isinstance(distribution["requires"], list)
            or distribution["requires"]
            != sorted(set(distribution["requires"]))
            or not isinstance(distribution["files"], list)
            or distribution["files"]
            != sorted(set(distribution["files"]))
            or not distribution["files"]
            or any(
                not isinstance(path, str)
                or not Path(path).is_absolute()
                for path in distribution["files"]
            )
        ):
            raise RuntimeManifestError(
                "pytest dependency probe is malformed"
            )
        names.append(distribution["name"])
    if names != sorted(set(names)) or "pytest" not in names:
        raise RuntimeManifestError(
            "pytest dependency closure is missing or duplicated"
        )
    known = set(names)
    if any(
        dependency not in known
        for distribution in value["distributions"]
        for dependency in distribution["requires"]
    ):
        raise RuntimeManifestError(
            "pytest dependency closure is incomplete"
        )


def _run_probe(
    command: tuple[str, ...],
    *,
    run: Callable[..., subprocess.CompletedProcess[bytes]],
) -> bytes:
    try:
        result = run(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env={"LANG": "C", "LC_ALL": "C"},
            timeout=60,
            check=False,
            shell=False,
            close_fds=True,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeManifestError(
            "Python runtime probe could not execute"
        ) from exc
    if result.returncode != 0:
        raise RuntimeManifestError(
            "Python runtime probe failed: "
            + result.stderr.decode("utf-8", errors="replace")[-2000:]
        )
    return result.stdout


def build_runtime_manifest(
    python_executable: Path,
    *,
    run: Callable[..., subprocess.CompletedProcess[bytes]] = subprocess.run,
) -> dict[str, Any]:
    """Observe one venv and build its canonical complete runtime manifest."""

    interpreter = _normal_absolute(
        python_executable, label="Python interpreter"
    )
    resolution = _resolution_evidence(interpreter)
    if resolution["resolved_target"]["kind"] != "file":
        raise RuntimeManifestError(
            "Python interpreter does not resolve to a regular file"
        )
    base = parse_probe(
        _run_probe(base_probe_command(interpreter), run=run),
        package=False,
    )
    if base["executable"] != str(interpreter):
        raise RuntimeManifestError(
            "Python runtime substituted the requested executable path"
        )
    venv_root = interpreter.parent.parent
    if base["prefix"] != str(venv_root):
        raise RuntimeManifestError(
            "Python executable is not bound to its declared venv root"
        )
    pyvenv_path = venv_root / "pyvenv.cfg"
    pyvenv = _file_record(pyvenv_path)
    site_root = Path(base["purelib"])
    site_resolution = _resolution_evidence(site_root)
    package = parse_probe(
        _run_probe(
            package_probe_command(interpreter, site_root), run=run
        ),
        package=True,
    )
    try:
        pytest_distribution = next(
            item
            for item in package["distributions"]
            if item["name"] == "pytest"
        )
    except StopIteration as exc:
        raise RuntimeManifestError(
            "pytest is absent from its dependency closure"
        ) from exc
    if (
        package["pytest_version"] != pytest_distribution["version"]
        or package["pytest_path"] not in pytest_distribution["files"]
    ):
        raise RuntimeManifestError(
            "imported pytest differs from installed distribution evidence"
        )
    package_manifests = [
        {
            "name": item["name"],
            **_paths_manifest(item["files"]),
        }
        for item in package["distributions"]
    ]
    stdlib_root = Path(base["stdlib"])
    native_root = Path(base["destshared"])
    if (
        not stdlib_root.is_dir()
        or not native_root.is_dir()
        or os.path.commonpath((
            stdlib_root.resolve(),
            native_root.resolve(),
        ))
        != os.fspath(stdlib_root.resolve())
    ):
        raise RuntimeManifestError(
            "standard-library/native-extension roots are inconsistent"
        )
    isolated_paths = []
    for raw_path in base["isolated_sys_path"]:
        path = Path(raw_path)
        try:
            metadata = os.stat(path, follow_symlinks=False)
        except FileNotFoundError:
            isolated_paths.append({
                "path": raw_path,
                "kind": "absent",
            })
            continue
        if stat.S_ISREG(metadata.st_mode):
            isolated_paths.append(_file_record(path))
        elif stat.S_ISDIR(metadata.st_mode):
            isolated_paths.append({
                "path": raw_path,
                "kind": "directory",
                "resolution": _resolution_evidence(path),
            })
        else:
            raise RuntimeManifestError(
                f"isolated import path has unsupported type: {path}"
            )
    manifest = {
        "schema": SCHEMA,
        "kind": KIND,
        "interpreter": {
            "requested_path": str(interpreter),
            "resolved_sha256":
                resolution["resolved_target"]["sha256"],
            "resolution": resolution,
        },
        "base_runtime_probe": base,
        "package_runtime_probe": package,
        "pyvenv_cfg": pyvenv,
        "site_packages_resolution": site_resolution,
        "isolated_import_paths": isolated_paths,
        "standard_library_manifest": _tree_manifest(
            stdlib_root,
            exclude_top_level=frozenset({"site-packages"}),
        ),
        "native_extension_manifest": _tree_manifest(native_root),
        "pytest_dependency_manifests": package_manifests,
    }
    validate_runtime_manifest_value(
        manifest, python_executable=interpreter
    )
    return manifest


def _validate_file_record(value: object) -> None:
    if (
        not isinstance(value, dict)
        or set(value) != {"path", "kind", "identity", "sha256"}
        or value["kind"] != "file"
        or not isinstance(value["path"], str)
        or not Path(value["path"]).is_absolute()
        or not _valid_sha256(value["sha256"])
    ):
        raise RuntimeManifestError("runtime file record is malformed")
    _validate_identity(value["identity"])


def _validate_tree_manifest(value: object) -> None:
    if (
        not isinstance(value, dict)
        or set(value)
        != {
            "root",
            "root_resolution",
            "excluded_top_level",
            "entry_count",
            "total_file_bytes",
            "entries_sha256",
        }
        or not isinstance(value["root"], str)
        or not Path(value["root"]).is_absolute()
        or not isinstance(value["excluded_top_level"], list)
        or value["excluded_top_level"]
        != sorted(set(value["excluded_top_level"]))
        or any(
            not isinstance(item, str) or not item
            for item in value["excluded_top_level"]
        )
        or isinstance(value["entry_count"], bool)
        or not isinstance(value["entry_count"], int)
        or value["entry_count"] < 0
        or isinstance(value["total_file_bytes"], bool)
        or not isinstance(value["total_file_bytes"], int)
        or value["total_file_bytes"] < 0
        or not _valid_sha256(value["entries_sha256"])
    ):
        raise RuntimeManifestError(
            "runtime tree manifest is malformed"
        )
    _validate_resolution(
        value["root_resolution"], expected=Path(value["root"])
    )


def validate_runtime_manifest_value(
    value: object,
    *,
    python_executable: Path,
    python_executable_sha256: str | None = None,
) -> dict[str, Any]:
    interpreter = _normal_absolute(
        python_executable, label="Python interpreter"
    )
    required = {
        "schema",
        "kind",
        "interpreter",
        "base_runtime_probe",
        "package_runtime_probe",
        "pyvenv_cfg",
        "site_packages_resolution",
        "isolated_import_paths",
        "standard_library_manifest",
        "native_extension_manifest",
        "pytest_dependency_manifests",
    }
    if (
        not isinstance(value, dict)
        or set(value) != required
        or value["schema"] != SCHEMA
        or isinstance(value["schema"], bool)
        or value["kind"] != KIND
        or not isinstance(value["interpreter"], dict)
        or set(value["interpreter"])
        != {"requested_path", "resolved_sha256", "resolution"}
        or value["interpreter"]["requested_path"] != str(interpreter)
        or not _valid_sha256(
            value["interpreter"]["resolved_sha256"]
        )
    ):
        raise RuntimeManifestError(
            "Python runtime manifest schema is not exact"
        )
    _validate_resolution(
        value["interpreter"]["resolution"], expected=interpreter
    )
    if (
        value["interpreter"]["resolution"]["resolved_target"]["kind"]
        != "file"
        or value["interpreter"]["resolved_sha256"]
        != value["interpreter"]["resolution"][
            "resolved_target"
        ]["sha256"]
        or (
            python_executable_sha256 is not None
            and value["interpreter"]["resolved_sha256"]
            != python_executable_sha256
        )
    ):
        raise RuntimeManifestError(
            "Python interpreter digest binding is inconsistent"
        )
    _validate_base_probe(value["base_runtime_probe"])
    _validate_package_probe(value["package_runtime_probe"])
    base = value["base_runtime_probe"]
    package = value["package_runtime_probe"]
    if (
        base["executable"] != str(interpreter)
        or base["prefix"] != str(interpreter.parent.parent)
        or base["purelib"] != base["platlib"]
        or value["pyvenv_cfg"].get("path")
        != str(interpreter.parent.parent / "pyvenv.cfg")
    ):
        raise RuntimeManifestError(
            "venv runtime identity differs from interpreter placement"
        )
    _validate_file_record(value["pyvenv_cfg"])
    _validate_resolution(
        value["site_packages_resolution"],
        expected=Path(base["purelib"]),
    )
    if (
        not isinstance(value["isolated_import_paths"], list)
        or len(value["isolated_import_paths"])
        != len(base["isolated_sys_path"])
    ):
        raise RuntimeManifestError(
            "isolated import-path evidence is incomplete"
        )
    for expected_path, evidence in zip(
        base["isolated_sys_path"],
        value["isolated_import_paths"],
        strict=True,
    ):
        if (
            not isinstance(evidence, dict)
            or evidence.get("path") != expected_path
            or evidence.get("kind")
            not in {"absent", "file", "directory"}
        ):
            raise RuntimeManifestError(
                "isolated import-path evidence is malformed"
            )
        if evidence["kind"] == "absent":
            if set(evidence) != {"path", "kind"}:
                raise RuntimeManifestError(
                    "absent import-path evidence is malformed"
                )
        elif evidence["kind"] == "file":
            _validate_file_record(evidence)
        else:
            if set(evidence) != {"path", "kind", "resolution"}:
                raise RuntimeManifestError(
                    "directory import-path evidence is malformed"
                )
            _validate_resolution(
                evidence["resolution"], expected=Path(expected_path)
            )
    _validate_tree_manifest(value["standard_library_manifest"])
    _validate_tree_manifest(value["native_extension_manifest"])
    if (
        value["standard_library_manifest"]["root"] != base["stdlib"]
        or value["standard_library_manifest"]["excluded_top_level"]
        != ["site-packages"]
        or value["native_extension_manifest"]["root"]
        != base["destshared"]
    ):
        raise RuntimeManifestError(
            "standard-library manifest roots differ from runtime probe"
        )
    distributions = package["distributions"]
    manifests = value["pytest_dependency_manifests"]
    if (
        not isinstance(manifests, list)
        or len(manifests) != len(distributions)
    ):
        raise RuntimeManifestError(
            "pytest dependency manifests are incomplete"
        )
    for distribution, file_manifest in zip(
        distributions, manifests, strict=True
    ):
        if (
            not isinstance(file_manifest, dict)
            or set(file_manifest)
            != {
                "name",
                "file_count",
                "total_file_bytes",
                "files_sha256",
            }
            or file_manifest["name"] != distribution["name"]
            or isinstance(file_manifest["file_count"], bool)
            or not isinstance(file_manifest["file_count"], int)
            or file_manifest["file_count"] <= 0
            or isinstance(file_manifest["total_file_bytes"], bool)
            or not isinstance(
                file_manifest["total_file_bytes"], int
            )
            or file_manifest["total_file_bytes"] <= 0
            or not _valid_sha256(
                file_manifest["files_sha256"]
            )
        ):
            raise RuntimeManifestError(
                "pytest dependency file manifest is malformed"
            )
    return value


def revalidate_runtime_files(value: object) -> dict[str, Any]:
    """Reopen every manifest-bound path without executing Python."""

    if not isinstance(value, dict):
        raise RuntimeManifestError(
            "Python runtime manifest is not an object"
        )
    interpreter = Path(
        value.get("interpreter", {}).get("requested_path", "")
    )
    expected = validate_runtime_manifest_value(
        value, python_executable=interpreter
    )
    if _resolution_evidence(interpreter) != (
        expected["interpreter"]["resolution"]
    ):
        raise RuntimeManifestError(
            "Python interpreter symlink chain or target changed"
        )
    if _file_record(
        Path(expected["pyvenv_cfg"]["path"])
    ) != expected["pyvenv_cfg"]:
        raise RuntimeManifestError("pyvenv.cfg changed")
    site_root = Path(expected["base_runtime_probe"]["purelib"])
    if _resolution_evidence(site_root) != (
        expected["site_packages_resolution"]
    ):
        raise RuntimeManifestError(
            "site-packages path identity changed"
        )
    for evidence in expected["isolated_import_paths"]:
        path = Path(evidence["path"])
        if evidence["kind"] == "absent":
            try:
                os.stat(path, follow_symlinks=False)
            except FileNotFoundError:
                continue
            raise RuntimeManifestError(
                f"previously absent import path appeared: {path}"
            )
        if evidence["kind"] == "file":
            observed: dict[str, Any] = _file_record(path)
        else:
            observed = {
                "path": evidence["path"],
                "kind": "directory",
                "resolution": _resolution_evidence(path),
            }
        if observed != evidence:
            raise RuntimeManifestError(
                f"isolated import path changed: {path}"
            )
    for field in (
        "standard_library_manifest",
        "native_extension_manifest",
    ):
        tree = expected[field]
        if _tree_manifest(
            Path(tree["root"]),
            exclude_top_level=frozenset(
                tree["excluded_top_level"]
            ),
        ) != tree:
            raise RuntimeManifestError(
                f"{field.replace('_', ' ')} changed"
            )
    for distribution, file_manifest in zip(
        expected["package_runtime_probe"]["distributions"],
        expected["pytest_dependency_manifests"],
        strict=True,
    ):
        observed = {
            "name": distribution["name"],
            **_paths_manifest(distribution["files"]),
        }
        if observed != file_manifest:
            raise RuntimeManifestError(
                "pytest dependency bytes or identity changed: "
                + distribution["name"]
            )
    return expected


def load_runtime_manifest(
    path: Path,
    *,
    expected_sha256: str,
    python_executable: Path,
    python_executable_sha256: str,
) -> dict[str, Any]:
    selected = _normal_absolute(
        path, label="Python runtime manifest"
    )
    if not _valid_sha256(expected_sha256):
        raise RuntimeManifestError(
            "Python runtime manifest digest is malformed"
        )
    raw, metadata = _read_regular(selected)
    if (
        metadata.st_uid != os.getuid()
        or stat.S_IMODE(metadata.st_mode) != 0o400
        or not 0 < len(raw) <= MAX_MANIFEST_BYTES
        or sha256_bytes(raw) != expected_sha256
    ):
        raise RuntimeManifestError(
            "Python runtime manifest is not exact private pinned bytes"
        )
    try:
        value = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeManifestError(
            "Python runtime manifest is not JSON"
        ) from exc
    if raw != canonical_json(value) + b"\n":
        raise RuntimeManifestError(
            "Python runtime manifest is not canonical JSON"
        )
    validate_runtime_manifest_value(
        value,
        python_executable=python_executable,
        python_executable_sha256=python_executable_sha256,
    )
    return revalidate_runtime_files(value)


def write_new_runtime_manifest(
    path: Path, value: dict[str, Any]
) -> str:
    """Durably publish one owner-private manifest without replacement."""

    payload = canonical_json(value) + b"\n"
    target = _normal_absolute(path, label="runtime manifest output")
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        target,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0),
        0o400,
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)
    directory = os.open(
        target.parent,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return sha256_bytes(payload)


__all__ = [
    "BASE_RUNTIME_PROBE",
    "HERMETIC_SUITE_BOOTSTRAP",
    "KIND",
    "PACKAGE_RUNTIME_PROBE",
    "RuntimeManifestError",
    "base_probe_command",
    "build_runtime_manifest",
    "canonical_json",
    "load_runtime_manifest",
    "manifest_sha256",
    "package_probe_command",
    "parse_probe",
    "revalidate_runtime_files",
    "suite_command",
    "validate_runtime_manifest_value",
    "write_new_runtime_manifest",
]
