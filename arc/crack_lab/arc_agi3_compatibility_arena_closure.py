#!/usr/bin/env python3
"""Receipt-bound preparation for future native compatibility turns.

No second Arena protocol is defined here.  The closure reuses the canonical
``arc_agi3_arena_rpc`` host, its networkless volume relay/transport, and the
same blank solver scaffold as the contiguous campaign.  Only the extracted
container-side RPC client is materialized into a proposer-visible directory.

This module prepares and verifies inputs; it never starts a proposer, engine,
container, relay, or RPC server.  Per-turn endpoint provisioning remains a
separate launch gate.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import stat
from pathlib import Path
from typing import Any, Sequence


SCHEMA = 1
KIND = "arc_agi3_compatibility_arena_closure"
CLIENT_MODULE = "arc_agi3_arena_rpc_client"
CLIENT_NAME = CLIENT_MODULE + ".py"
CONTENT_MANIFEST_NAME = "content_manifest.json"
RECEIPT_NAME = "closure_receipt.json"
EXACT_INVENTORY = tuple(sorted({
    CLIENT_NAME,
    CONTENT_MANIFEST_NAME,
    RECEIPT_NAME,
}))
MAX_SOURCE_BYTES = 1_000_000
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
            maximum=MAX_SOURCE_BYTES,
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
    analysis = analyze_client_source(raw)
    if (
        raw != _LOADED_CLIENT_RAW
        or analysis != _LOADED_CLIENT_ANALYSIS
        or _identity(metadata, full=True)
        != _identity(_LOADED_CLIENT_METADATA, full=True)
    ):
        raise CompatibilityClosureError(
            "canonical Arena RPC client changed after module import"
        )
    return {
        "components": observed,
        "client": analysis,
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


def _write_new_at(
    directory_fd: int,
    name: str,
    payload: bytes,
    *,
    mode: int,
) -> None:
    if not isinstance(name, str) or not name or "/" in name or name in {".", ".."}:
        raise CompatibilityClosureError("closure output name is malformed")
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
    # is enforced independently by the two-name inventory and nofollow reads.
    return {
        key: value
        for key, value in _identity(metadata, full=False).items()
        if key != "links"
    }


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
    """Materialize exact client bytes; do not provision or launch a turn."""

    root = _normal_absolute(
        destination,
        label="closure destination",
        allow_missing_leaf=True,
    )
    snapshot = canonical_closure_snapshot()
    try:
        os.mkdir(root, 0o700)
    except OSError as exc:
        raise CompatibilityClosureError(
            "closure destination must not already exist"
        ) from exc
    root_fd: int | None = None
    try:
        root_fd = _open_directory_chain(
            root, label="materialized closure root"
        )
        root_identity = _directory_identity_from_metadata(
            os.fstat(root_fd)
        )
        _write_new_at(
            root_fd, CLIENT_NAME, _LOADED_CLIENT_RAW, mode=0o400
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
        if client_raw != _LOADED_CLIENT_RAW:
            raise CompatibilityClosureError(
                "materialized Arena RPC client differs"
            )
        content_manifest = _content_manifest(snapshot)
        content_manifest_raw = _canonical_json(content_manifest) + b"\n"
        _write_new_at(
            root_fd,
            CONTENT_MANIFEST_NAME,
            content_manifest_raw,
            mode=0o400,
        )
        observed_manifest, manifest_metadata = _read_regular_at(
            root_fd,
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
        receipt = {
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
        receipt_raw = _canonical_json(receipt) + b"\n"
        _write_new_at(root_fd, RECEIPT_NAME, receipt_raw, mode=0o400)
        observed_receipt, receipt_metadata = _read_regular_at(
            root_fd,
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
            root_fd,
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
        os.fsync(root_fd)
        _recheck_directory_path(
            root,
            root_fd,
            expected_identity=root_identity,
            label="materialized closure root",
        )
        return {
            "root": os.fspath(root),
            "receipt_sha256": _sha256(receipt_raw),
            "content_manifest_sha256": content_manifest_sha256,
            "client_sha256": snapshot["client"]["source_sha256"],
            "launch_authorized": False,
        }
    except BaseException:
        if root_fd is not None:
            for name in reversed(EXACT_INVENTORY):
                try:
                    os.unlink(name, dir_fd=root_fd)
                except FileNotFoundError:
                    pass
            try:
                root_identity = _directory_identity_from_metadata(
                    os.fstat(root_fd)
                )
                _recheck_directory_path(
                    root,
                    root_fd,
                    expected_identity=root_identity,
                    label="failed closure root",
                )
                os.rmdir(root)
            except (OSError, CompatibilityClosureError):
                pass
        raise
    finally:
        if root_fd is not None:
            os.close(root_fd)


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
    "analyze_client_source",
    "canonical_closure_snapshot",
    "prepare_closure",
    "validate_closure",
)
