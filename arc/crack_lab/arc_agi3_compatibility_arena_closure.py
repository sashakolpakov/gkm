#!/usr/bin/env python3
"""Receipt-bound preparation for future native compatibility turns.

No second Arena protocol is defined here.  The closure reuses the canonical
``arc_agi3_arena_rpc`` host, its networkless volume relay/transport, and the
same blank solver scaffold as the contiguous campaign.  Only the extracted
container-side RPC client is materialized into a proposer-visible directory.

This module prepares and verifies inputs; it never starts a proposer, engine,
container, relay, or RPC server.  Per-turn endpoint provisioning remains a
separate launch gate.  Preparation writes and fsyncs an exact private sibling
staging directory, then publishes it with one exclusive atomic rename and
fsyncs the parent.  Any staging present before an invocation is ambiguous and
is preserved fail-closed; only a staging inode still descriptor-held by the
current invocation and its recorded child inodes may be removed after an
ordinary prepublication failure.
"""

from __future__ import annotations

import argparse
import ast
import copy
import ctypes
import errno
import hashlib
import json
import os
import re
import stat
import sys
from pathlib import Path
from typing import Any, Sequence


SCHEMA = 1
KIND = "arc_agi3_compatibility_arena_closure"
CLIENT_MODULE = "arc_agi3_arena_rpc_client"
CLIENT_NAME = CLIENT_MODULE + ".py"
CONTENT_MANIFEST_NAME = "content_manifest.json"
RECEIPT_NAME = "closure_receipt.json"
STAGING_NAME_PREFIX = ".arc-agi3-compatibility-closure-"
STAGING_NAME_SUFFIX = ".partial"
STAGING_OBSERVATION_KIND = "arc_agi3_compatibility_staging_observation"
STAGING_PROVENANCE_AMBIGUITY = (
    "preexisting_staging_has_no_publisher_provenance"
)
MAX_STAGING_OBSERVATION_ENTRIES = 8
MAX_STAGING_OBSERVATION_ENTRY_BYTES = 1_000_000
MAX_STAGING_OBSERVATION_TOTAL_BYTES = 2_000_000
EXACT_INVENTORY = tuple(sorted({
    CLIENT_NAME,
    CONTENT_MANIFEST_NAME,
    RECEIPT_NAME,
}))
MAX_SOURCE_BYTES = 1_000_000
MAX_CONTROL_BYTES = 4_000_000
MAX_RECEIPT_BYTES = 1_000_000
SHA256_RE = re.compile(r"[0-9a-f]{64}")
EXACT_CLIENT_IMPORT_ROOTS = (
    "hashlib",
    "hmac",
    "json",
    "numbers",
    "numpy",
    "re",
    "socket",
    "threading",
    "typing",
)
FORBIDDEN_SOURCE_MARKERS = (
    ".env",
    "environment_files",
    "gkm_arena",
    "arcengine",
    "llm_binder",
    "from lab import",
    "import lab",
    "../",
    "..\\",
)
FORBIDDEN_IMPORT_ROOTS = frozenset({
    "arc_agi",
    "arcengine",
    "builtins",
    "glob",
    "importlib",
    "inspect",
    "lab",
    "llm_binder",
    "os",
    "pathlib",
    "pkgutil",
    "shutil",
    "subprocess",
    "sys",
})
FORBIDDEN_ATTRIBUTES = frozenset({
    "_env",
    "_game",
    "__file__",
    "__globals__",
    "__loader__",
    "f_globals",
    "f_locals",
    "meta_path",
    "modules",
    "path_hooks",
    "path_importer_cache",
})
FORBIDDEN_CALLS = frozenset({
    "__import__",
    "compile",
    "eval",
    "exec",
    "open",
})

_ROOT = Path(__file__).resolve().parent
_CLIENT_PATH = _ROOT / CLIENT_NAME
CONTROL_COMPONENTS = {
    "compatibility_closure": (
        _ROOT / "arc_agi3_compatibility_arena_closure.py"
    ),
    "arena_rpc_host": _ROOT / "arc_agi3_arena_rpc.py",
    "arena_rpc_client": _CLIENT_PATH,
    "arena_volume_relay": _ROOT / "arc_agi3_arena_volume_relay.py",
    "arena_volume_transport": _ROOT / "arc_agi3_arena_volume_transport.py",
    "container_worker": _ROOT / "arc_agi3_container_worker.py",
    "proposer_worker": _ROOT / "arc_agi3_proposer_worker.py",
    "source_schema": _ROOT / "arc_agi3_source_schema.py",
    "container_backend": _ROOT / "arc_agi3_container_backend.py",
    "contiguous_runner": _ROOT / "arc_agi3_contiguous_runner.py",
    "contiguous_orchestrator": (
        _ROOT / "arc_agi3_contiguous_orchestrator.py"
    ),
    "container_recipe": (
        _ROOT / "container" / "Containerfile.arc-agi3-contiguous"
    ),
    "solver_requirements": (
        _ROOT / "container" / "arc_agi3_solver_requirements.lock"
    ),
    "blank_legs": _ROOT / "contiguous_blank_scaffold" / "legs.py",
    "blank_players": _ROOT / "contiguous_blank_scaffold" / "players.py",
    "blank_solve": _ROOT / "contiguous_blank_scaffold" / "solve.py",
}


class CompatibilityClosureError(RuntimeError):
    """The compatibility Arena closure is not pure or exact."""


class CompatibilityStagingAmbiguityError(CompatibilityClosureError):
    """A pre-existing non-authoritative staging path requires quarantine."""


def _publication_checkpoint(_name: str) -> None:
    """Test-only crash/fault seam; production publication never overrides it."""


def _entry_name(name: str, *, label: str) -> str:
    if (
        not isinstance(name, str)
        or not name
        or "/" in name
        or "\x00" in name
        or name in {".", ".."}
    ):
        raise CompatibilityClosureError(f"{label} name is malformed")
    return name


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _identity(metadata: os.stat_result, *, full: bool) -> dict[str, int]:
    output = {
        "device": metadata.st_dev,
        "inode": metadata.st_ino,
        "mode": stat.S_IMODE(metadata.st_mode),
        "links": metadata.st_nlink,
        "uid": metadata.st_uid,
        "gid": metadata.st_gid,
    }
    if full:
        output.update({
            "size": metadata.st_size,
            "mtime_ns": metadata.st_mtime_ns,
            "ctime_ns": metadata.st_ctime_ns,
        })
    return output


def _normal_absolute(
    path: Path | str,
    *,
    label: str,
    allow_missing_leaf: bool = False,
) -> Path:
    selected = Path(path)
    if (
        not selected.is_absolute()
        or "\x00" in os.fspath(selected)
        or Path(os.path.normpath(selected)) != selected
    ):
        raise CompatibilityClosureError(
            f"{label} must be a normalized absolute path"
        )
    if Path(os.path.realpath(selected)) != selected:
        raise CompatibilityClosureError(
            f"{label} must use its physical canonical path"
        )
    current = Path(selected.anchor)
    for index, part in enumerate(selected.parts[1:], start=1):
        current /= part
        try:
            metadata = os.stat(current, follow_symlinks=False)
        except FileNotFoundError as exc:
            if allow_missing_leaf and index == len(selected.parts) - 1:
                break
            raise CompatibilityClosureError(
                f"{label} has a missing ancestor"
            ) from exc
        except OSError as exc:
            raise CompatibilityClosureError(
                f"{label} ancestry cannot be inspected"
            ) from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise CompatibilityClosureError(
                f"{label} has a symlinked ancestor"
            )
        if not stat.S_ISDIR(metadata.st_mode):
            raise CompatibilityClosureError(
                f"{label} has a non-directory ancestor"
            )
    return selected


def _read_regular(
    path: Path,
    *,
    maximum: int,
    label: str,
) -> tuple[bytes, os.stat_result]:
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
        )
    except OSError as exc:
        raise CompatibilityClosureError(
            f"{label} is unavailable or symlinked"
        ) from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size < 0
            or before.st_size > maximum
        ):
            raise CompatibilityClosureError(
                f"{label} must be an unaliased bounded regular file"
            )
        remaining = before.st_size
        chunks: list[bytes] = []
        while remaining:
            block = os.read(descriptor, min(1024 * 1024, remaining))
            if not block:
                raise CompatibilityClosureError(
                    f"{label} changed while read"
                )
            chunks.append(block)
            remaining -= len(block)
        after = os.fstat(descriptor)
        if _identity(before, full=True) != _identity(after, full=True):
            raise CompatibilityClosureError(f"{label} changed while read")
        return b"".join(chunks), after
    finally:
        os.close(descriptor)


def _read_regular_at(
    directory_fd: int,
    name: str,
    *,
    maximum: int,
    label: str,
) -> tuple[bytes, os.stat_result]:
    if not isinstance(name, str) or not name or "/" in name or name in {".", ".."}:
        raise CompatibilityClosureError(f"{label} name is malformed")
    try:
        descriptor = os.open(
            name,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=directory_fd,
        )
    except OSError as exc:
        raise CompatibilityClosureError(
            f"{label} is unavailable or symlinked"
        ) from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size < 0
            or before.st_size > maximum
        ):
            raise CompatibilityClosureError(
                f"{label} must be an unaliased bounded regular file"
            )
        remaining = before.st_size
        chunks: list[bytes] = []
        while remaining:
            block = os.read(descriptor, min(1024 * 1024, remaining))
            if not block:
                raise CompatibilityClosureError(
                    f"{label} changed while read"
                )
            chunks.append(block)
            remaining -= len(block)
        after = os.fstat(descriptor)
        if _identity(before, full=True) != _identity(after, full=True):
            raise CompatibilityClosureError(f"{label} changed while read")
        return b"".join(chunks), after
    finally:
        os.close(descriptor)


def _open_directory_chain(path: Path, *, label: str) -> int:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(path.anchor, flags)
    except OSError as exc:
        raise CompatibilityClosureError(
            f"{label} physical root cannot be opened"
        ) from exc
    try:
        for part in path.parts[1:]:
            try:
                child = os.open(part, flags, dir_fd=descriptor)
            except OSError as exc:
                raise CompatibilityClosureError(
                    f"{label} has a missing, aliased, or non-directory ancestor"
                ) from exc
            os.close(descriptor)
            descriptor = child
        metadata = os.fstat(descriptor)
        if not stat.S_ISDIR(metadata.st_mode):
            raise CompatibilityClosureError(
                f"{label} is not a physical directory"
            )
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _recheck_directory_path(
    path: Path,
    held_fd: int,
    *,
    expected_identity: dict[str, int],
    label: str,
) -> None:
    reopened = _open_directory_chain(path, label=label)
    try:
        if (
            _directory_identity_from_metadata(os.fstat(held_fd))
            != expected_identity
            or _directory_identity_from_metadata(os.fstat(reopened))
            != expected_identity
        ):
            raise CompatibilityClosureError(
                f"{label} path identity changed or was aliased"
            )
    finally:
        os.close(reopened)


def _client_imports(tree: ast.AST) -> tuple[str, ...]:
    roots: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.extend(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                raise CompatibilityClosureError(
                    "Arena RPC client uses a relative import"
                )
            if node.module != "__future__":
                if not node.module:
                    raise CompatibilityClosureError(
                        "Arena RPC client import is malformed"
                    )
                roots.append(node.module.split(".", 1)[0])
    return tuple(sorted(set(roots)))


def analyze_client_source(raw: bytes) -> dict[str, Any]:
    """Statically prove the complete proposer-visible import closure."""

    if not raw or len(raw) > MAX_SOURCE_BYTES:
        raise CompatibilityClosureError(
            "Arena RPC client source is empty or oversized"
        )
    try:
        source = raw.decode("utf-8")
        tree = ast.parse(source, filename=CLIENT_NAME)
    except (UnicodeError, SyntaxError) as exc:
        raise CompatibilityClosureError(
            "Arena RPC client source is not valid UTF-8 Python"
        ) from exc
    lowered = source.lower()
    for marker in FORBIDDEN_SOURCE_MARKERS:
        if marker.lower() in lowered:
            raise CompatibilityClosureError(
                f"Arena RPC client names forbidden host surface: {marker}"
            )
    imports = _client_imports(tree)
    if imports != EXACT_CLIENT_IMPORT_ROOTS:
        raise CompatibilityClosureError(
            "Arena RPC client import closure differs: "
            f"expected={EXACT_CLIENT_IMPORT_ROOTS!r} observed={imports!r}"
        )
    if FORBIDDEN_IMPORT_ROOTS.intersection(imports):
        raise CompatibilityClosureError(
            "Arena RPC client imports a host/filesystem capability"
        )
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_ATTRIBUTES:
            raise CompatibilityClosureError(
                "Arena RPC client accesses private/import state: "
                f"{node.attr}"
            )
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in FORBIDDEN_CALLS
        ):
            raise CompatibilityClosureError(
                "Arena RPC client has a dynamic/filesystem call: "
                f"{node.func.id}"
            )
    return {
        "source_sha256": _sha256(raw),
        "source_bytes": len(raw),
        "import_roots": list(imports),
        "local_import_closure": [CLIENT_NAME],
        "repository_imports": [],
        "engine_imports": [],
        "filesystem_calls": [],
        "private_game_state_accesses": [],
    }


def _component_snapshot() -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for name, path in sorted(CONTROL_COMPONENTS.items()):
        raw, metadata = _read_regular(
            path,
            maximum=MAX_CONTROL_BYTES,
            label=f"compatibility control {name}",
        )
        output[name] = {
            "path": os.fspath(path),
            "sha256": _sha256(raw),
            "bytes": len(raw),
            "identity": _identity(metadata, full=True),
        }
    return output


_LOADED_COMPONENTS = _component_snapshot()
_LOADED_CLIENT_RAW, _LOADED_CLIENT_METADATA = _read_regular(
    _CLIENT_PATH,
    maximum=MAX_SOURCE_BYTES,
    label="canonical Arena RPC client",
)
_LOADED_CLIENT_ANALYSIS = analyze_client_source(_LOADED_CLIENT_RAW)


def canonical_closure_snapshot() -> dict[str, Any]:
    """Reopen every canonical input and reject in-process control drift."""

    observed = _component_snapshot()
    if observed != _LOADED_COMPONENTS:
        raise CompatibilityClosureError(
            "compatibility Arena control changed after module import"
        )
    raw, metadata = _read_regular(
        _CLIENT_PATH,
        maximum=MAX_SOURCE_BYTES,
        label="canonical Arena RPC client",
    )
    if (
        raw != _LOADED_CLIENT_RAW
        or _identity(metadata, full=True)
        != _identity(_LOADED_CLIENT_METADATA, full=True)
    ):
        raise CompatibilityClosureError(
            "canonical Arena RPC client changed after module import"
        )
    return {
        "components": observed,
        "client": copy.deepcopy(_LOADED_CLIENT_ANALYSIS),
    }


def _reuse_projection() -> dict[str, str]:
    return {
        "rpc_schema": "arc-agi3-arena-rpc/v1",
        "host": "arc_agi3_arena_rpc.ArenaHostSession",
        "relay": "arc_agi3_arena_volume_relay",
        "transport": "arc_agi3_arena_volume_transport",
        "blank_scaffold": "contiguous_blank_scaffold",
    }


def _authority_projection() -> dict[str, Any]:
    return {
        "engine_shared_with_proposer": False,
        "game_specific_client": False,
        "host_filesystem_capability": False,
        "host_private_state_capability": False,
        "launch_authorized": False,
        "remaining_gate": "exact per-turn RPC host/socket/token/container receipt",
    }


def _content_manifest(snapshot: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "kind": KIND + "_content_manifest",
        "payload_inventory": [CLIENT_NAME],
        "client": snapshot["client"],
        "controls": {
            name: {
                "bytes": record["bytes"],
                "sha256": record["sha256"],
            }
            for name, record in sorted(snapshot["components"].items())
        },
        "reuse": _reuse_projection(),
        "authority": _authority_projection(),
    }


def _closure_receipt(
    *,
    root: Path,
    root_identity: dict[str, int],
    snapshot: dict[str, Any],
    client_metadata: os.stat_result,
    manifest_metadata: os.stat_result,
    content_manifest_sha256: str,
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "kind": KIND,
        "root": os.fspath(root),
        "root_identity": root_identity,
        "inventory": list(EXACT_INVENTORY),
        "client": {
            "name": CLIENT_NAME,
            "identity": _identity(client_metadata, full=True),
            "sha256": snapshot["client"]["source_sha256"],
        },
        "content_manifest": {
            "name": CONTENT_MANIFEST_NAME,
            "identity": _identity(manifest_metadata, full=True),
            "sha256": content_manifest_sha256,
        },
        "controls": snapshot["components"],
        "authority": _authority_projection(),
    }


def _write_new_at(
    directory_fd: int,
    name: str,
    payload: bytes,
    *,
    mode: int,
    created_identities: dict[str, dict[str, int]],
) -> None:
    _entry_name(name, label="closure output")
    try:
        descriptor = os.open(
            name,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0),
            mode,
            dir_fd=directory_fd,
        )
    except OSError as exc:
        raise CompatibilityClosureError(
            f"closure output is unavailable: {name}"
        ) from exc
    try:
        metadata = os.fstat(descriptor)
        stable_identity = _identity(metadata, full=False)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or stat.S_IMODE(metadata.st_mode) != mode
            or metadata.st_nlink != 1
            or name in created_identities
        ):
            raise CompatibilityClosureError(
                f"new closure output custody is ambiguous: {name}"
            )
        created_identities[name] = stable_identity
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise CompatibilityClosureError(
                    "closure write made no progress"
                )
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _directory_identity_from_metadata(
    metadata: os.stat_result,
) -> dict[str, int]:
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISDIR(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != 0o700
    ):
        raise CompatibilityClosureError(
            "closure root must be a physical owner-private directory"
        )
    # APFS changes a directory's link count as regular children are added.
    # Bind the physical directory and permissions, while exact child custody
    # is enforced independently by the exact inventory and nofollow reads.
    return {
        key: value
        for key, value in _identity(metadata, full=False).items()
        if key != "links"
    }


def _physical_directory_identity_from_metadata(
    metadata: os.stat_result,
    *,
    label: str,
) -> dict[str, int]:
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise CompatibilityClosureError(
            f"{label} must be a physical directory"
        )
    return {
        key: value
        for key, value in _identity(metadata, full=False).items()
        if key != "links"
    }


def _open_directory_at(
    parent_fd: int,
    name: str,
    *,
    label: str,
) -> int:
    _entry_name(name, label=label)
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(name, flags, dir_fd=parent_fd)
    except OSError as exc:
        raise CompatibilityClosureError(
            f"{label} is unavailable, aliased, or not a directory"
        ) from exc
    try:
        _physical_directory_identity_from_metadata(
            os.fstat(descriptor), label=label
        )
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _recheck_child_directory_at(
    parent_fd: int,
    name: str,
    held_fd: int,
    *,
    expected_identity: dict[str, int],
    label: str,
) -> None:
    reopened = _open_directory_at(parent_fd, name, label=label)
    try:
        if (
            _physical_directory_identity_from_metadata(
                os.fstat(held_fd), label=label
            )
            != expected_identity
            or _physical_directory_identity_from_metadata(
                os.fstat(reopened), label=label
            )
            != expected_identity
        ):
            raise CompatibilityClosureError(
                f"{label} path identity changed or was aliased"
            )
    finally:
        os.close(reopened)


def _recheck_parent_directory_path(
    path: Path,
    held_fd: int,
    *,
    expected_identity: dict[str, int],
) -> None:
    reopened = _open_directory_chain(path, label="closure parent")
    try:
        if (
            _physical_directory_identity_from_metadata(
                os.fstat(held_fd), label="closure parent"
            )
            != expected_identity
            or _physical_directory_identity_from_metadata(
                os.fstat(reopened), label="closure parent"
            )
            != expected_identity
        ):
            raise CompatibilityClosureError(
                "closure parent path identity changed or was aliased"
            )
    finally:
        os.close(reopened)


def _staging_name(root: Path) -> str:
    leaf = _entry_name(root.name, label="closure destination")
    digest = _sha256(os.fsencode(leaf))
    return STAGING_NAME_PREFIX + digest + STAGING_NAME_SUFFIX


def _require_absent_at(
    parent_fd: int,
    name: str,
    *,
    label: str,
    present_error: type[CompatibilityClosureError] = (
        CompatibilityClosureError
    ),
) -> None:
    _entry_name(name, label=label)
    try:
        os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return
    except OSError as exc:
        raise CompatibilityClosureError(
            f"{label} cannot be inspected"
        ) from exc
    raise present_error(f"{label} must not already exist")


def _bounded_staging_names(
    directory_fd: int,
) -> tuple[list[str], bool, int]:
    names: list[str] = []
    try:
        with os.scandir(directory_fd) as entries:
            for entry in entries:
                if len(names) >= MAX_STAGING_OBSERVATION_ENTRIES:
                    return (
                        [],
                        False,
                        MAX_STAGING_OBSERVATION_ENTRIES + 1,
                    )
                names.append(_entry_name(
                    entry.name, label="quarantined staging entry"
                ))
    except CompatibilityClosureError:
        raise
    except OSError as exc:
        raise CompatibilityStagingAmbiguityError(
            "quarantined staging inventory cannot be inspected"
        ) from exc
    return sorted(names), True, len(names)


def _staging_observation(
    *,
    destination: Path,
    staging_root: Path,
    present: bool,
    parent_identity: dict[str, int],
    root_identity: dict[str, int] | None,
    root_type: str | None,
    entries: list[dict[str, Any]],
    total_bytes: int,
    ambiguity_reasons: list[str],
    inventory_observed: bool,
    entry_count_lower_bound: int,
) -> dict[str, Any]:
    body = {
        "schema": SCHEMA,
        "kind": STAGING_OBSERVATION_KIND,
        "status": (
            "AMBIGUOUS"
            if ambiguity_reasons
            else ("OBSERVED" if present else "ABSENT")
        ),
        "destination": os.fspath(destination),
        "staging_root": os.fspath(staging_root),
        "present": present,
        "parent_identity": parent_identity,
        "root_identity": root_identity,
        "root_type": root_type,
        "max_depth": 1,
        "max_entries": MAX_STAGING_OBSERVATION_ENTRIES,
        "max_entry_bytes": MAX_STAGING_OBSERVATION_ENTRY_BYTES,
        "max_total_bytes": MAX_STAGING_OBSERVATION_TOTAL_BYTES,
        "total_bytes": total_bytes,
        "entries": entries,
        "inventory_observed": inventory_observed,
        "entry_count_lower_bound": entry_count_lower_bound,
        "ambiguity_reasons": sorted(set(ambiguity_reasons)),
        "authority": {
            "quarantine_evidence_only": True,
            "cleanup_authority": False,
            "launch_authority": False,
            "promotion_authority": False,
        },
    }
    return {
        **body,
        "observation_sha256": _sha256(_canonical_json(body)),
    }


def observe_quarantined_staging(
    destination: Path | str,
) -> dict[str, Any]:
    """Descriptor-observe one bounded sibling staging tree without deletion."""

    root = _normal_absolute(
        destination,
        label="closure destination",
        allow_missing_leaf=True,
    )
    parent = root.parent
    staging_name = _staging_name(root)
    staging_root = parent / staging_name
    parent_fd = _open_directory_chain(parent, label="closure parent")
    parent_identity = _physical_directory_identity_from_metadata(
        os.fstat(parent_fd), label="closure parent"
    )
    staging_fd: int | None = None
    try:
        _recheck_parent_directory_path(
            parent,
            parent_fd,
            expected_identity=parent_identity,
        )
        try:
            initial_metadata = os.stat(
                staging_name,
                dir_fd=parent_fd,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            _recheck_parent_directory_path(
                parent,
                parent_fd,
                expected_identity=parent_identity,
            )
            try:
                os.stat(
                    staging_name,
                    dir_fd=parent_fd,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                return _staging_observation(
                    destination=root,
                    staging_root=staging_root,
                    present=False,
                    parent_identity=parent_identity,
                    root_identity=None,
                    root_type=None,
                    entries=[],
                    total_bytes=0,
                    ambiguity_reasons=[],
                    inventory_observed=True,
                    entry_count_lower_bound=0,
                )
            except OSError as exc:
                raise CompatibilityStagingAmbiguityError(
                    "quarantined staging absence cannot be rebound"
                ) from exc
            raise CompatibilityStagingAmbiguityError(
                "quarantined staging appeared during absence observation"
            )
        except OSError as exc:
            raise CompatibilityStagingAmbiguityError(
                "quarantined staging cannot be inspected"
            ) from exc
        def metadata_type(metadata: os.stat_result) -> str:
            mode = metadata.st_mode
            if stat.S_ISREG(mode):
                return "regular"
            if stat.S_ISDIR(mode):
                return "directory"
            if stat.S_ISLNK(mode):
                return "symlink"
            if stat.S_ISFIFO(mode):
                return "fifo"
            if stat.S_ISSOCK(mode):
                return "socket"
            if stat.S_ISBLK(mode):
                return "block_device"
            if stat.S_ISCHR(mode):
                return "character_device"
            return "special"

        root_type = metadata_type(initial_metadata)
        initial_full_identity = _identity(initial_metadata, full=True)
        if root_type != "directory":
            try:
                rebound = os.stat(
                    staging_name,
                    dir_fd=parent_fd,
                    follow_symlinks=False,
                )
            except OSError as exc:
                raise CompatibilityStagingAmbiguityError(
                    "quarantined staging root changed while observed"
                ) from exc
            if _identity(rebound, full=True) != initial_full_identity:
                raise CompatibilityStagingAmbiguityError(
                    "quarantined staging root changed while observed"
                )
            _recheck_parent_directory_path(
                parent,
                parent_fd,
                expected_identity=parent_identity,
            )
            return _staging_observation(
                destination=root,
                staging_root=staging_root,
                present=True,
                parent_identity=parent_identity,
                root_identity=initial_full_identity,
                root_type=root_type,
                entries=[],
                total_bytes=0,
                ambiguity_reasons=[
                    STAGING_PROVENANCE_AMBIGUITY,
                    "staging_root_" + root_type,
                ],
                inventory_observed=False,
                entry_count_lower_bound=0,
            )
        initial_identity = _physical_directory_identity_from_metadata(
            initial_metadata, label="quarantined closure staging directory"
        )
        staging_fd = _open_directory_at(
            parent_fd,
            staging_name,
            label="quarantined closure staging directory",
        )
        root_metadata = os.fstat(staging_fd)
        root_path_identity = _physical_directory_identity_from_metadata(
            root_metadata,
            label="quarantined closure staging directory",
        )
        root_identity = _identity(root_metadata, full=True)
        if (
            root_path_identity != initial_identity
            or root_identity != initial_full_identity
        ):
            raise CompatibilityStagingAmbiguityError(
                "quarantined staging root changed before descriptor binding"
            )
        ambiguity_reasons: list[str] = [STAGING_PROVENANCE_AMBIGUITY]
        if root_path_identity["mode"] != 0o700:
            ambiguity_reasons.append("staging_root_not_owner_private")
        _recheck_child_directory_at(
            parent_fd,
            staging_name,
            staging_fd,
            expected_identity=root_path_identity,
            label="quarantined closure staging directory",
        )
        names, inventory_observed, entry_count_lower_bound = (
            _bounded_staging_names(staging_fd)
        )
        if not inventory_observed:
            rebound_names = _bounded_staging_names(staging_fd)
            if rebound_names[1] is not False:
                raise CompatibilityStagingAmbiguityError(
                    "quarantined staging inventory crossed its bound while "
                    "observed"
                )
            _recheck_child_directory_at(
                parent_fd,
                staging_name,
                staging_fd,
                expected_identity=root_path_identity,
                label="quarantined closure staging directory",
            )
            _recheck_parent_directory_path(
                parent,
                parent_fd,
                expected_identity=parent_identity,
            )
            if _identity(os.fstat(staging_fd), full=True) != root_identity:
                raise CompatibilityStagingAmbiguityError(
                    "quarantined staging root changed while observed"
                )
            return _staging_observation(
                destination=root,
                staging_root=staging_root,
                present=True,
                parent_identity=parent_identity,
                root_identity=root_identity,
                root_type=root_type,
                entries=[],
                total_bytes=0,
                ambiguity_reasons=[
                    *ambiguity_reasons,
                    "staging_inventory_exceeds_entry_bound",
                ],
                inventory_observed=False,
                entry_count_lower_bound=entry_count_lower_bound,
            )
        entries: list[dict[str, Any]] = []
        retained: dict[str, tuple[bytes | None, dict[str, int]]] = {}
        total_bytes = 0
        for name in names:
            try:
                metadata = os.stat(
                    name,
                    dir_fd=staging_fd,
                    follow_symlinks=False,
                )
            except OSError as exc:
                raise CompatibilityStagingAmbiguityError(
                    f"quarantined staging entry is unavailable: {name}"
                ) from exc
            entry_type = metadata_type(metadata)
            raw: bytes | None = None
            reason: str | None = None
            if entry_type != "regular":
                reason = "entry_" + entry_type
            elif metadata.st_nlink != 1:
                reason = "entry_regular_is_aliased"
            elif metadata.st_size > MAX_STAGING_OBSERVATION_ENTRY_BYTES:
                reason = "entry_regular_exceeds_byte_bound"
            elif (
                total_bytes + metadata.st_size
                > MAX_STAGING_OBSERVATION_TOTAL_BYTES
            ):
                reason = "staging_regular_bytes_exceed_total_bound"
            else:
                try:
                    raw, metadata = _read_regular_at(
                        staging_fd,
                        name,
                        maximum=MAX_STAGING_OBSERVATION_ENTRY_BYTES,
                        label=f"quarantined staging entry {name}",
                    )
                except CompatibilityClosureError as exc:
                    raise CompatibilityStagingAmbiguityError(
                        f"quarantined staging entry drifted while read: {name}"
                    ) from exc
                total_bytes += len(raw)
            identity = _identity(metadata, full=True)
            retained[name] = (raw, identity)
            if reason is not None:
                ambiguity_reasons.append(f"{name}:{reason}")
            entries.append({
                "name": name,
                "type": entry_type,
                "identity": identity,
                "size": metadata.st_size,
                "content_observed": raw is not None,
                "observed_bytes": 0 if raw is None else len(raw),
                "sha256": None if raw is None else _sha256(raw),
                "ambiguity_reason": reason,
            })
        rebound_names = _bounded_staging_names(staging_fd)
        if rebound_names != (names, True, len(names)):
            raise CompatibilityStagingAmbiguityError(
                "quarantined staging inventory changed while observed"
            )
        for name in names:
            try:
                metadata = os.stat(
                    name,
                    dir_fd=staging_fd,
                    follow_symlinks=False,
                )
            except OSError as exc:
                raise CompatibilityStagingAmbiguityError(
                    f"quarantined staging entry changed: {name}"
                ) from exc
            expected_raw, expected_identity = retained[name]
            if _identity(metadata, full=True) != expected_identity:
                raise CompatibilityStagingAmbiguityError(
                    f"quarantined staging entry changed: {name}"
                )
            if expected_raw is not None:
                try:
                    raw, metadata = _read_regular_at(
                        staging_fd,
                        name,
                        maximum=MAX_STAGING_OBSERVATION_ENTRY_BYTES,
                        label=f"quarantined staging entry {name}",
                    )
                except CompatibilityClosureError as exc:
                    raise CompatibilityStagingAmbiguityError(
                        f"quarantined staging entry changed: {name}"
                    ) from exc
                if (
                    raw != expected_raw
                    or _identity(metadata, full=True) != expected_identity
                ):
                    raise CompatibilityStagingAmbiguityError(
                        f"quarantined staging entry changed: {name}"
                    )
        _recheck_child_directory_at(
            parent_fd,
            staging_name,
            staging_fd,
            expected_identity=root_path_identity,
            label="quarantined closure staging directory",
        )
        _recheck_parent_directory_path(
            parent,
            parent_fd,
            expected_identity=parent_identity,
        )
        if _identity(os.fstat(staging_fd), full=True) != root_identity:
            raise CompatibilityStagingAmbiguityError(
                "quarantined staging root changed while observed"
            )
        return _staging_observation(
            destination=root,
            staging_root=staging_root,
            present=True,
            parent_identity=parent_identity,
            root_identity=root_identity,
            root_type=root_type,
            entries=entries,
            total_bytes=total_bytes,
            ambiguity_reasons=ambiguity_reasons,
            inventory_observed=True,
            entry_count_lower_bound=len(names),
        )
    finally:
        if staging_fd is not None:
            os.close(staging_fd)
        os.close(parent_fd)


def _cleanup_current_staging_at(
    parent_fd: int,
    staging_name: str,
    *,
    staging_fd: int,
    expected_identity: dict[str, int],
    created_file_identities: dict[str, dict[str, int]],
) -> None:
    """Clean only this invocation's still-held, unambiguous staging inode."""

    _entry_name(staging_name, label="current closure staging directory")
    observed_identity = _directory_identity_from_metadata(
        os.fstat(staging_fd)
    )
    if observed_identity != expected_identity:
        raise CompatibilityClosureError(
            "current closure staging descriptor identity drifted"
        )
    _recheck_child_directory_at(
        parent_fd,
        staging_name,
        staging_fd,
        expected_identity=expected_identity,
        label="current closure staging directory",
    )
    try:
        names = sorted(entry.name for entry in os.scandir(staging_fd))
    except OSError as exc:
        raise CompatibilityClosureError(
            "current closure staging inventory cannot be inspected"
        ) from exc
    if (
        not set(names).issubset(EXACT_INVENTORY)
        or names != sorted(created_file_identities)
    ):
        raise CompatibilityClosureError(
            "current closure staging contains ambiguous state"
        )
    retained: dict[str, tuple[bytes, dict[str, int], int]] = {}
    for name in names:
        maximum = (
            MAX_SOURCE_BYTES if name == CLIENT_NAME else MAX_RECEIPT_BYTES
        )
        raw, metadata = _read_regular_at(
            staging_fd,
            name,
            maximum=maximum,
            label=f"current closure staging file {name}",
        )
        _validate_file_custody(
            metadata,
            root_identity=expected_identity,
            label=f"current closure staging file {name}",
        )
        if (
            _identity(metadata, full=False)
            != created_file_identities[name]
        ):
            raise CompatibilityClosureError(
                f"current closure staging file was not created here: {name}"
            )
        retained[name] = (raw, _identity(metadata, full=True), maximum)
    _recheck_child_directory_at(
        parent_fd,
        staging_name,
        staging_fd,
        expected_identity=expected_identity,
        label="current closure staging directory",
    )
    for name in names:
        raw, metadata = _read_regular_at(
            staging_fd,
            name,
            maximum=retained[name][2],
            label=f"current closure staging file {name}",
        )
        if (
            raw != retained[name][0]
            or _identity(metadata, full=True) != retained[name][1]
        ):
            raise CompatibilityClosureError(
                f"current closure staging file changed: {name}"
            )
        try:
            os.unlink(name, dir_fd=staging_fd)
        except OSError as exc:
            raise CompatibilityClosureError(
                f"current closure staging file cannot be removed: {name}"
            ) from exc
    os.fsync(staging_fd)
    try:
        _recheck_child_directory_at(
            parent_fd,
            staging_name,
            staging_fd,
            expected_identity=expected_identity,
            label="current closure staging directory",
        )
        os.rmdir(staging_name, dir_fd=parent_fd)
        os.fsync(parent_fd)
    except OSError as exc:
        raise CompatibilityClosureError(
            "current closure staging directory cannot be removed durably"
        ) from exc
    _require_absent_at(
        parent_fd,
        staging_name,
        label="removed current closure staging directory",
    )


def _atomic_rename_noreplace_at(
    parent_fd: int,
    source_name: str,
    destination_name: str,
) -> None:
    """Atomically publish one sibling directory without replacement."""

    _entry_name(source_name, label="closure staging directory")
    _entry_name(destination_name, label="closure destination")
    library = ctypes.CDLL(None, use_errno=True)
    source_raw = os.fsencode(source_name)
    destination_raw = os.fsencode(destination_name)
    if sys.platform == "darwin":
        function = getattr(library, "renameatx_np", None)
        flags = 0x00000004  # RENAME_EXCL from <sys/stdio.h>.
    elif sys.platform.startswith("linux"):
        function = getattr(library, "renameat2", None)
        flags = 0x00000001  # RENAME_NOREPLACE from <linux/fs.h>.
    else:
        function = None
        flags = 0
    if function is None:
        raise CompatibilityClosureError(
            "atomic no-replace directory publication is unavailable"
        )
    function.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    function.restype = ctypes.c_int
    ctypes.set_errno(0)
    result = function(
        parent_fd,
        source_raw,
        parent_fd,
        destination_raw,
        flags,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise CompatibilityClosureError(
            "closure destination appeared during atomic publication"
        )
    detail = OSError(error_number, os.strerror(error_number))
    raise CompatibilityClosureError(
        "atomic no-replace closure publication failed"
    ) from detail


def _directory_identity(path: Path) -> dict[str, int]:
    try:
        metadata = os.stat(path, follow_symlinks=False)
    except OSError as exc:
        raise CompatibilityClosureError(
            "closure root is unavailable"
        ) from exc
    return _directory_identity_from_metadata(metadata)


def _validate_file_custody(
    metadata: os.stat_result,
    *,
    root_identity: dict[str, int],
    label: str,
) -> None:
    if (
        not stat.S_ISREG(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != 0o400
        or metadata.st_nlink != 1
        or metadata.st_uid != root_identity["uid"]
        or metadata.st_gid != root_identity["gid"]
    ):
        raise CompatibilityClosureError(
            f"{label} custody, owner, links, or mode differs"
        )


def _recheck_materialized_files(
    directory_fd: int,
    *,
    root_identity: dict[str, int],
    expected: dict[str, tuple[bytes, dict[str, int], int]],
) -> None:
    try:
        names = sorted(entry.name for entry in os.scandir(directory_fd))
    except OSError as exc:
        raise CompatibilityClosureError(
            "closure inventory cannot be rechecked"
        ) from exc
    if names != list(EXACT_INVENTORY) or set(expected) != set(EXACT_INVENTORY):
        raise CompatibilityClosureError(
            "closure inventory changed after validation read"
        )
    for name in EXACT_INVENTORY:
        expected_raw, expected_identity, maximum = expected[name]
        raw, metadata = _read_regular_at(
            directory_fd,
            name,
            maximum=maximum,
            label=f"final closure file {name}",
        )
        _validate_file_custody(
            metadata,
            root_identity=root_identity,
            label=f"final closure file {name}",
        )
        if (
            raw != expected_raw
            or _identity(metadata, full=True) != expected_identity
        ):
            raise CompatibilityClosureError(
                f"closure file changed after validation read: {name}"
            )


def prepare_closure(destination: Path | str) -> dict[str, Any]:
    """Crash-atomically publish exact bytes; never provision or launch."""

    root = _normal_absolute(
        destination,
        label="closure destination",
        allow_missing_leaf=True,
    )
    destination_name = _entry_name(
        root.name, label="closure destination"
    )
    parent = root.parent
    staging_name = _staging_name(root)
    parent_fd = _open_directory_chain(parent, label="closure parent")
    parent_identity = _physical_directory_identity_from_metadata(
        os.fstat(parent_fd), label="closure parent"
    )
    staging_fd: int | None = None
    staging_identity: dict[str, int] | None = None
    staging_file_identities: dict[str, dict[str, int]] = {}
    staging_exists = False
    published = False
    try:
        _recheck_parent_directory_path(
            parent,
            parent_fd,
            expected_identity=parent_identity,
        )
        _require_absent_at(
            parent_fd,
            destination_name,
            label="closure destination",
        )
        _require_absent_at(
            parent_fd,
            staging_name,
            label="closure staging directory",
            present_error=CompatibilityStagingAmbiguityError,
        )
        snapshot = canonical_closure_snapshot()
        try:
            os.mkdir(staging_name, 0o700, dir_fd=parent_fd)
        except OSError as exc:
            raise CompatibilityClosureError(
                "closure staging directory cannot be created exclusively"
            ) from exc
        staging_exists = True
        staging_fd = _open_directory_at(
            parent_fd,
            staging_name,
            label="materialized closure staging directory",
        )
        staging_identity = _directory_identity_from_metadata(
            os.fstat(staging_fd)
        )
        root_identity = staging_identity
        _recheck_child_directory_at(
            parent_fd,
            staging_name,
            staging_fd,
            expected_identity=root_identity,
            label="materialized closure staging directory",
        )
        _publication_checkpoint("staging_created")
        _write_new_at(
            staging_fd,
            CLIENT_NAME,
            _LOADED_CLIENT_RAW,
            mode=0o400,
            created_identities=staging_file_identities,
        )
        _publication_checkpoint("client_fsynced")
        client_raw, client_metadata = _read_regular_at(
            staging_fd,
            CLIENT_NAME,
            maximum=MAX_SOURCE_BYTES,
            label="materialized Arena RPC client",
        )
        _validate_file_custody(
            client_metadata,
            root_identity=root_identity,
            label="materialized Arena RPC client",
        )
        if client_raw != _LOADED_CLIENT_RAW:
            raise CompatibilityClosureError(
                "materialized Arena RPC client differs"
            )
        content_manifest = _content_manifest(snapshot)
        content_manifest_raw = _canonical_json(content_manifest) + b"\n"
        _write_new_at(
            staging_fd,
            CONTENT_MANIFEST_NAME,
            content_manifest_raw,
            mode=0o400,
            created_identities=staging_file_identities,
        )
        _publication_checkpoint("content_manifest_fsynced")
        observed_manifest, manifest_metadata = _read_regular_at(
            staging_fd,
            CONTENT_MANIFEST_NAME,
            maximum=MAX_RECEIPT_BYTES,
            label="materialized content manifest",
        )
        _validate_file_custody(
            manifest_metadata,
            root_identity=root_identity,
            label="materialized content manifest",
        )
        if observed_manifest != content_manifest_raw:
            raise CompatibilityClosureError(
                "materialized content manifest differs"
            )
        content_manifest_sha256 = _sha256(content_manifest_raw)
        receipt = _closure_receipt(
            root=root,
            root_identity=root_identity,
            snapshot=snapshot,
            client_metadata=client_metadata,
            manifest_metadata=manifest_metadata,
            content_manifest_sha256=content_manifest_sha256,
        )
        receipt_raw = _canonical_json(receipt) + b"\n"
        _write_new_at(
            staging_fd,
            RECEIPT_NAME,
            receipt_raw,
            mode=0o400,
            created_identities=staging_file_identities,
        )
        _publication_checkpoint("receipt_fsynced")
        observed_receipt, receipt_metadata = _read_regular_at(
            staging_fd,
            RECEIPT_NAME,
            maximum=MAX_RECEIPT_BYTES,
            label="materialized closure receipt",
        )
        _validate_file_custody(
            receipt_metadata,
            root_identity=root_identity,
            label="materialized closure receipt",
        )
        if observed_receipt != receipt_raw:
            raise CompatibilityClosureError(
                "materialized closure receipt differs"
            )
        if canonical_closure_snapshot() != snapshot:
            raise CompatibilityClosureError(
                "compatibility controls drifted during closure preparation"
            )
        _recheck_materialized_files(
            staging_fd,
            root_identity=root_identity,
            expected={
                CLIENT_NAME: (
                    client_raw,
                    _identity(client_metadata, full=True),
                    MAX_SOURCE_BYTES,
                ),
                CONTENT_MANIFEST_NAME: (
                    observed_manifest,
                    _identity(manifest_metadata, full=True),
                    MAX_RECEIPT_BYTES,
                ),
                RECEIPT_NAME: (
                    observed_receipt,
                    _identity(receipt_metadata, full=True),
                    MAX_RECEIPT_BYTES,
                ),
            },
        )
        os.fsync(staging_fd)
        _publication_checkpoint("staging_directory_fsynced")
        _recheck_child_directory_at(
            parent_fd,
            staging_name,
            staging_fd,
            expected_identity=root_identity,
            label="materialized closure staging directory",
        )
        _recheck_parent_directory_path(
            parent,
            parent_fd,
            expected_identity=parent_identity,
        )
        _require_absent_at(
            parent_fd,
            destination_name,
            label="closure destination",
        )
        _publication_checkpoint("before_atomic_publication")
        _atomic_rename_noreplace_at(
            parent_fd, staging_name, destination_name
        )
        staging_exists = False
        published = True
        _publication_checkpoint("published_before_parent_fsync")
        os.fsync(parent_fd)
        _publication_checkpoint("parent_directory_fsynced")
        _recheck_parent_directory_path(
            parent,
            parent_fd,
            expected_identity=parent_identity,
        )
        _require_absent_at(
            parent_fd,
            staging_name,
            label="closure staging directory",
        )
        _recheck_child_directory_at(
            parent_fd,
            destination_name,
            staging_fd,
            expected_identity=root_identity,
            label="materialized closure root",
        )
        _recheck_materialized_files(
            staging_fd,
            root_identity=root_identity,
            expected={
                CLIENT_NAME: (
                    client_raw,
                    _identity(client_metadata, full=True),
                    MAX_SOURCE_BYTES,
                ),
                CONTENT_MANIFEST_NAME: (
                    observed_manifest,
                    _identity(manifest_metadata, full=True),
                    MAX_RECEIPT_BYTES,
                ),
                RECEIPT_NAME: (
                    observed_receipt,
                    _identity(receipt_metadata, full=True),
                    MAX_RECEIPT_BYTES,
                ),
            },
        )
        _recheck_directory_path(
            root,
            staging_fd,
            expected_identity=root_identity,
            label="materialized closure root",
        )
        if canonical_closure_snapshot() != snapshot:
            raise CompatibilityClosureError(
                "compatibility controls drifted during closure publication"
            )
        return {
            "root": os.fspath(root),
            "receipt_sha256": _sha256(receipt_raw),
            "content_manifest_sha256": content_manifest_sha256,
            "client_sha256": snapshot["client"]["source_sha256"],
            "launch_authorized": False,
        }
    except Exception as exc:
        if (
            staging_exists
            and not published
            and staging_fd is not None
            and staging_identity is not None
        ):
            _cleanup_current_staging_at(
                parent_fd,
                staging_name,
                staging_fd=staging_fd,
                expected_identity=staging_identity,
                created_file_identities=staging_file_identities,
            )
            staging_exists = False
        if staging_fd is not None:
            os.close(staging_fd)
            staging_fd = None
        if isinstance(exc, CompatibilityClosureError):
            raise
        phase = "after publication" if published else "before publication"
        raise CompatibilityClosureError(
            f"closure preparation failed {phase}"
        ) from exc
    finally:
        if staging_fd is not None:
            os.close(staging_fd)
        os.close(parent_fd)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise CompatibilityClosureError(
                "closure receipt has duplicate JSON keys"
            )
        output[key] = value
    return output


def _parse_canonical_json(raw: bytes, *, label: str) -> object:
    try:
        value = json.loads(
            raw,
            object_pairs_hook=_unique_object,
            parse_constant=lambda _value: (_ for _ in ()).throw(
                CompatibilityClosureError(
                    f"{label} has a non-finite number"
                )
            ),
        )
    except CompatibilityClosureError:
        raise
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise CompatibilityClosureError(
            f"{label} is not valid JSON"
        ) from exc
    if _canonical_json(value) + b"\n" != raw:
        raise CompatibilityClosureError(
            f"{label} is not canonical JSON"
        )
    return value


def validate_closure(
    destination: Path | str,
    expected_receipt_sha256: str,
) -> dict[str, Any]:
    """Reopen every path and byte; reject alias, shadow, path, or hash drift."""

    root = _normal_absolute(destination, label="closure root")
    if (
        not isinstance(expected_receipt_sha256, str)
        or SHA256_RE.fullmatch(expected_receipt_sha256) is None
    ):
        raise CompatibilityClosureError(
            "expected closure receipt digest is malformed"
        )
    root_fd = _open_directory_chain(root, label="closure root")
    try:
        root_identity = _directory_identity_from_metadata(
            os.fstat(root_fd)
        )
        try:
            names = sorted(entry.name for entry in os.scandir(root_fd))
        except OSError as exc:
            raise CompatibilityClosureError(
                "closure inventory cannot be read"
            ) from exc
        if names != list(EXACT_INVENTORY):
            raise CompatibilityClosureError(
                "closure inventory has missing or shadow files"
            )
        receipt_raw, receipt_metadata = _read_regular_at(
            root_fd,
            RECEIPT_NAME,
            maximum=MAX_RECEIPT_BYTES,
            label="closure receipt",
        )
        _validate_file_custody(
            receipt_metadata,
            root_identity=root_identity,
            label="closure receipt",
        )
        if _sha256(receipt_raw) != expected_receipt_sha256:
            raise CompatibilityClosureError("closure receipt hash drifted")
        receipt = _parse_canonical_json(
            receipt_raw, label="closure receipt"
        )
        if (
            not isinstance(receipt, dict)
            or set(receipt) != {
                "schema", "kind", "root", "root_identity", "inventory",
                "client", "content_manifest", "controls", "authority",
            }
            or receipt["schema"] != SCHEMA
            or receipt["kind"] != KIND
            or receipt["root"] != os.fspath(root)
            or receipt["root_identity"] != root_identity
            or receipt["inventory"] != list(EXACT_INVENTORY)
        ):
            raise CompatibilityClosureError(
                "closure receipt schema, path, or root identity differs"
            )
        client_raw, client_metadata = _read_regular_at(
            root_fd,
            CLIENT_NAME,
            maximum=MAX_SOURCE_BYTES,
            label="materialized Arena RPC client",
        )
        _validate_file_custody(
            client_metadata,
            root_identity=root_identity,
            label="materialized Arena RPC client",
        )
        analysis = analyze_client_source(client_raw)
        expected_client = {
            "name": CLIENT_NAME,
            "identity": _identity(client_metadata, full=True),
            "sha256": analysis["source_sha256"],
        }
        if receipt["client"] != expected_client:
            raise CompatibilityClosureError(
                "closure client identity or hash drifted"
            )
        manifest_raw, manifest_metadata = _read_regular_at(
            root_fd,
            CONTENT_MANIFEST_NAME,
            maximum=MAX_RECEIPT_BYTES,
            label="closure content manifest",
        )
        _validate_file_custody(
            manifest_metadata,
            root_identity=root_identity,
            label="closure content manifest",
        )
        manifest_sha256 = _sha256(manifest_raw)
        expected_manifest_custody = {
            "name": CONTENT_MANIFEST_NAME,
            "identity": _identity(manifest_metadata, full=True),
            "sha256": manifest_sha256,
        }
        if receipt["content_manifest"] != expected_manifest_custody:
            raise CompatibilityClosureError(
                "closure content-manifest custody or hash drifted"
            )
        manifest = _parse_canonical_json(
            manifest_raw, label="closure content manifest"
        )
        snapshot = canonical_closure_snapshot()
        expected_manifest = _content_manifest(snapshot)
        if (
            manifest != expected_manifest
            or manifest_raw != _canonical_json(expected_manifest) + b"\n"
            or receipt["controls"] != snapshot["components"]
            or receipt["authority"] != _authority_projection()
            or analysis != snapshot["client"]
            or client_raw != _LOADED_CLIENT_RAW
        ):
            raise CompatibilityClosureError(
                "closure differs from current reviewed controls"
            )
        if canonical_closure_snapshot() != snapshot:
            raise CompatibilityClosureError(
                "compatibility controls drifted during closure validation"
            )
        _recheck_materialized_files(
            root_fd,
            root_identity=root_identity,
            expected={
                CLIENT_NAME: (
                    client_raw,
                    _identity(client_metadata, full=True),
                    MAX_SOURCE_BYTES,
                ),
                CONTENT_MANIFEST_NAME: (
                    manifest_raw,
                    _identity(manifest_metadata, full=True),
                    MAX_RECEIPT_BYTES,
                ),
                RECEIPT_NAME: (
                    receipt_raw,
                    _identity(receipt_metadata, full=True),
                    MAX_RECEIPT_BYTES,
                ),
            },
        )
        _recheck_directory_path(
            root,
            root_fd,
            expected_identity=root_identity,
            label="closure root",
        )
        authority = receipt["authority"]
        return {
            "schema": SCHEMA,
            "kind": KIND,
            "status": "PASS",
            "root": os.fspath(root),
            "receipt_sha256": expected_receipt_sha256,
            "content_manifest_sha256": manifest_sha256,
            "client_sha256": analysis["source_sha256"],
            "launch_authorized": False,
            "remaining_gate": authority["remaining_gate"],
        }
    finally:
        os.close(root_fd)


def remove_closure(
    destination: Path | str,
    expected_receipt_sha256: str,
) -> None:
    """Remove one exact closure after a failed backend preparation.

    This is intentionally not a recursive cleanup primitive.  The complete
    descriptor-safe closure validation runs first, and only the three known,
    unaliased files are unlinked through the still-open directory.  Ambiguous
    or partial state is preserved and reported instead of being erased.
    """

    root = _normal_absolute(destination, label="closure root")
    destination_name = _entry_name(root.name, label="closure root")
    parent = root.parent
    parent_fd = _open_directory_chain(parent, label="closure parent")
    parent_identity = _physical_directory_identity_from_metadata(
        os.fstat(parent_fd), label="closure parent"
    )
    root_fd: int | None = None
    try:
        validate_closure(root, expected_receipt_sha256)
        root_fd = _open_directory_at(
            parent_fd, destination_name, label="closure cleanup root"
        )
        root_identity = _directory_identity_from_metadata(os.fstat(root_fd))
        _recheck_parent_directory_path(
            parent,
            parent_fd,
            expected_identity=parent_identity,
        )
        _recheck_child_directory_at(
            parent_fd,
            destination_name,
            root_fd,
            expected_identity=root_identity,
            label="closure cleanup root",
        )
        retained: dict[str, tuple[bytes, dict[str, int], int]] = {}
        for name in EXACT_INVENTORY:
            maximum = (
                MAX_SOURCE_BYTES
                if name == CLIENT_NAME
                else MAX_RECEIPT_BYTES
            )
            raw, metadata = _read_regular_at(
                root_fd,
                name,
                maximum=maximum,
                label=f"closure cleanup file {name}",
            )
            _validate_file_custody(
                metadata,
                root_identity=root_identity,
                label=f"closure cleanup file {name}",
            )
            retained[name] = (
                raw,
                _identity(metadata, full=True),
                maximum,
            )
        if _sha256(retained[RECEIPT_NAME][0]) != expected_receipt_sha256:
            raise CompatibilityClosureError(
                "closure cleanup receipt hash drifted"
            )
        reopened_receipt = _parse_canonical_json(
            retained[RECEIPT_NAME][0],
            label="closure cleanup receipt",
        )
        if (
            not isinstance(reopened_receipt, dict)
            or reopened_receipt.get("root") != os.fspath(root)
            or reopened_receipt.get("root_identity") != root_identity
            or reopened_receipt.get("inventory") != list(EXACT_INVENTORY)
        ):
            raise CompatibilityClosureError(
                "closure cleanup reopened a substituted root"
            )
        _recheck_materialized_files(
            root_fd,
            root_identity=root_identity,
            expected=retained,
        )
        for name in reversed(EXACT_INVENTORY):
            raw, metadata = _read_regular_at(
                root_fd,
                name,
                maximum=retained[name][2],
                label=f"closure cleanup file {name}",
            )
            if (
                raw != retained[name][0]
                or _identity(metadata, full=True) != retained[name][1]
            ):
                raise CompatibilityClosureError(
                    f"closure cleanup file changed: {name}"
                )
            os.unlink(name, dir_fd=root_fd)
        os.fsync(root_fd)
        if any(os.scandir(root_fd)):
            raise CompatibilityClosureError(
                "closure cleanup left an unexpected entry"
            )
        _recheck_directory_path(
            root,
            root_fd,
            expected_identity=root_identity,
            label="closure cleanup root",
        )
        os.close(root_fd)
        root_fd = None
        os.rmdir(destination_name, dir_fd=parent_fd)
        os.fsync(parent_fd)
        _require_absent_at(
            parent_fd,
            destination_name,
            label="removed closure root",
        )
        _recheck_parent_directory_path(
            parent,
            parent_fd,
            expected_identity=parent_identity,
        )
    except OSError as exc:
        raise CompatibilityClosureError(
            "validated closure could not be removed"
        ) from exc
    finally:
        if root_fd is not None:
            os.close(root_fd)
        os.close(parent_fd)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", allow_abbrev=False)
    prepare.add_argument("destination", type=Path)
    verify = commands.add_parser("verify", allow_abbrev=False)
    verify.add_argument("destination", type=Path)
    verify.add_argument("receipt_sha256")
    arguments = parser.parse_args(argv)
    destination = arguments.destination.absolute()
    if arguments.command == "prepare":
        result = prepare_closure(destination)
    else:
        result = validate_closure(destination, arguments.receipt_sha256)
    print(_canonical_json(result).decode("ascii"))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except CompatibilityClosureError as error:
        print(f"compatibility closure failed: {error}")
        raise SystemExit(70)


__all__ = (
    "CLIENT_NAME",
    "CompatibilityClosureError",
    "CompatibilityStagingAmbiguityError",
    "STAGING_OBSERVATION_KIND",
    "STAGING_PROVENANCE_AMBIGUITY",
    "analyze_client_source",
    "canonical_closure_snapshot",
    "observe_quarantined_staging",
    "prepare_closure",
    "remove_closure",
    "validate_closure",
)
