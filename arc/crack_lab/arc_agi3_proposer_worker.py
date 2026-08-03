#!/usr/bin/env python3
"""Descriptor-confined proposer worker for one contiguous-campaign attempt.

This process runs inside the network-disabled proposer container.  It owns the
writable workspace and declared export root, accepts exactly one authenticated
host bridge client, and executes only the closed operation policy embedded in
the immutable input bundle.  It has no model/provider credential and no
general host transport.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import json
import os
import re
import selectors
import signal
import socket
import stat
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

import arc_agi3_source_schema as SourceSchema


SCHEMA = 1
PROTOCOL_VERSION = 1
MAX_PATH_DEPTH = 12
MAX_PATH_BYTES = 1024
MAX_CACHED_RESPONSES = 4096
MAX_BOUNDARY_WORKSPACE_FILES = 4096
MAX_BOUNDARY_WORKSPACE_TOTAL_BYTES = 512 * 1024 * 1024
SOCKET_BACKLOG = 1
MAX_SEQUENTIAL_CONNECTIONS = 4
RECONNECT_GRACE_SECONDS = 2.0
WORKER_OUTCOME_NAME = "worker_outcome.json"
CANDIDATE_NAME = "candidate_path.json"
WIP_MANIFEST_NAME = "wip_manifest.json"
FORBIDDEN_EXPORT_NAMES = frozenset(
    {
        "checkpoint.json",
        "current.json",
        "promotion_receipt.json",
        "promotion_manifest.json",
        "usage.jsonl",
    }
)
FORBIDDEN_ENV_NAME = re.compile(
    r"(?:API[_-]?KEY|ACCESS[_-]?TOKEN|AUTH(?:ORIZATION)?|CHATGPT|"
    r"CODEX_(?:TOKEN|AUTH)|OPENAI_API_KEY)",
    re.IGNORECASE,
)


class ProposerBridgeError(RuntimeError):
    """A worker policy, filesystem, protocol, or child failed closed."""


@dataclass(frozen=True)
class BridgePolicy:
    campaign_id: str
    generation_id: str
    attempt_id: str
    game: str
    target_level: int
    frontier_sha256: str
    parent_checkpoint_sha256: str
    operation_allowlist: tuple[str, ...]
    exec_allowlist: tuple[str, ...]
    max_request_bytes: int
    max_response_bytes: int
    max_file_bytes: int
    max_total_export_bytes: int
    max_processes: int
    max_exec_seconds: int


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _payload_tree_sha256(payloads: Mapping[str, bytes]) -> str:
    """Match the trusted host's path/hash tree digest over staged payloads."""

    digest = hashlib.sha256()
    for relative, raw in sorted(payloads.items()):
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256(raw).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _reject_duplicate_pairs(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ProposerBridgeError(
                f"duplicate JSON object key: {key}"
            )
        result[key] = value
    return result


def _strict_json_loads(raw: bytes | str) -> Any:
    try:
        return json.loads(
            raw,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ProposerBridgeError(
                    f"non-finite JSON number: {value}"
                )
            ),
        )
    except ProposerBridgeError:
        raise
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ProposerBridgeError("malformed JSON") from exc


def _regular_bytes(path: Path, *, max_bytes: int) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or metadata.st_size > max_bytes
        ):
            raise ProposerBridgeError(
                f"expected one bounded unaliased regular file: {path}"
            )
        data = bytearray()
        while len(data) < metadata.st_size:
            block = os.read(
                descriptor,
                min(1024 * 1024, metadata.st_size - len(data)),
            )
            if not block:
                raise ProposerBridgeError(
                    f"file changed while reading: {path}"
                )
            data.extend(block)
        return bytes(data)
    finally:
        os.close(descriptor)


def _load_policy(path: Path) -> tuple[BridgePolicy, str]:
    raw = _regular_bytes(path, max_bytes=1024 * 1024)
    try:
        value = _strict_json_loads(raw)
    except ProposerBridgeError as exc:
        raise ProposerBridgeError("invalid bridge policy JSON") from exc
    required = {
        "schema",
        "kind",
        "campaign_id",
        "generation_id",
        "attempt_id",
        "game",
        "target_level",
        "frontier_sha256",
        "parent_checkpoint_sha256",
        "protocol_version",
        "operation_allowlist",
        "exec_allowlist",
        "workspace_root",
        "export_root",
        "bounds",
    }
    bounds_required = {
        "max_request_bytes",
        "max_response_bytes",
        "max_file_bytes",
        "max_total_export_bytes",
        "max_processes",
        "max_exec_seconds",
    }
    if (
        not isinstance(value, dict)
        or set(value) != required
        or value["schema"] != SCHEMA
        or value["kind"] != "arc_agi3_contiguous_bridge_policy"
        or value["protocol_version"] != PROTOCOL_VERSION
        or value["workspace_root"] != "/arc/workspace"
        or value["export_root"] != "/arc/export"
        or not isinstance(value["bounds"], dict)
        or set(value["bounds"]) != bounds_required
        or not isinstance(value["operation_allowlist"], list)
        or not isinstance(value["exec_allowlist"], list)
    ):
        raise ProposerBridgeError("bridge policy schema mismatch")
    numeric = tuple(value["bounds"][name] for name in bounds_required)
    if (
        any(
            not isinstance(item, int)
            or isinstance(item, bool)
            or item <= 0
            for item in numeric
        )
        or len(value["operation_allowlist"])
        != len(set(value["operation_allowlist"]))
        or len(value["exec_allowlist"]) != len(set(value["exec_allowlist"]))
        or value["exec_allowlist"] != []
        or "workspace_run_python" in value["operation_allowlist"]
        or not all(
            isinstance(item, str) and item
            for item in (
                *value["operation_allowlist"],
                *value["exec_allowlist"],
            )
        )
        or not all(
            isinstance(value[name], str)
            and 0 < len(value[name]) <= 200
            and "\x00" not in value[name]
            for name in (
                "campaign_id",
                "generation_id",
                "attempt_id",
                "game",
            )
        )
        or not isinstance(value["target_level"], int)
        or isinstance(value["target_level"], bool)
        or value["target_level"] <= 0
        or not all(
            isinstance(value[name], str)
            and re.fullmatch(r"[0-9a-f]{64}", value[name])
            for name in (
                "frontier_sha256",
                "parent_checkpoint_sha256",
            )
        )
    ):
        raise ProposerBridgeError("bridge policy bounds/allowlists invalid")
    policy = BridgePolicy(
        campaign_id=value["campaign_id"],
        generation_id=value["generation_id"],
        attempt_id=value["attempt_id"],
        game=value["game"],
        target_level=value["target_level"],
        frontier_sha256=value["frontier_sha256"],
        parent_checkpoint_sha256=value["parent_checkpoint_sha256"],
        operation_allowlist=tuple(value["operation_allowlist"]),
        exec_allowlist=tuple(value["exec_allowlist"]),
        max_request_bytes=value["bounds"]["max_request_bytes"],
        max_response_bytes=value["bounds"]["max_response_bytes"],
        max_file_bytes=value["bounds"]["max_file_bytes"],
        max_total_export_bytes=value["bounds"][
            "max_total_export_bytes"
        ],
        max_processes=value["bounds"]["max_processes"],
        max_exec_seconds=value["bounds"]["max_exec_seconds"],
    )
    return policy, _sha256(raw)


def _relative_parts(value: object) -> tuple[str, ...]:
    if (
        not isinstance(value, str)
        or not value
        or len(value.encode("utf-8")) > MAX_PATH_BYTES
        or "\x00" in value
    ):
        raise ProposerBridgeError("relative path is invalid")
    path = Path(value)
    parts = path.parts
    if (
        path.is_absolute()
        or len(parts) > MAX_PATH_DEPTH
        or any(
            part in {"", ".", ".."}
            or part.startswith(".")
            or "/" in part
            or "\x00" in part
            for part in parts
        )
    ):
        raise ProposerBridgeError("relative path escapes or is hidden")
    return tuple(parts)


def _open_root(path: Path, *, label: str) -> int:
    if path.is_symlink() or not path.is_dir():
        raise ProposerBridgeError(f"{label} is not a regular directory")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    metadata = os.fstat(descriptor)
    if not stat.S_ISDIR(metadata.st_mode):
        os.close(descriptor)
        raise ProposerBridgeError(f"{label} descriptor is not a directory")
    return descriptor


def _open_directory_at(
    root_fd: int,
    parts: Sequence[str],
    *,
    create: bool = False,
) -> int:
    current = os.dup(root_fd)
    try:
        for part in parts:
            if create:
                try:
                    os.mkdir(part, mode=0o700, dir_fd=current)
                except FileExistsError:
                    pass
            flags = (
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0)
            )
            child = os.open(part, flags, dir_fd=current)
            metadata = os.fstat(child)
            if not stat.S_ISDIR(metadata.st_mode):
                os.close(child)
                raise ProposerBridgeError("path component is not a directory")
            os.close(current)
            current = child
        return current
    except BaseException:
        os.close(current)
        raise


def _read_at(root_fd: int, relative: str, *, max_bytes: int) -> bytes:
    parts = _relative_parts(relative)
    parent = _open_directory_at(root_fd, parts[:-1])
    try:
        descriptor = os.open(
            parts[-1],
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent,
        )
        try:
            metadata = os.fstat(descriptor)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_nlink != 1
                or metadata.st_size > max_bytes
            ):
                raise ProposerBridgeError(
                    "workspace file is aliased, nonregular, or oversized"
                )
            chunks: list[bytes] = []
            remaining = metadata.st_size
            while remaining:
                block = os.read(
                    descriptor, min(1024 * 1024, remaining)
                )
                if not block:
                    raise ProposerBridgeError(
                        "workspace file changed while reading"
                    )
                chunks.append(block)
                remaining -= len(block)
            return b"".join(chunks)
        finally:
            os.close(descriptor)
    finally:
        os.close(parent)


def _write_at(
    root_fd: int,
    relative: str,
    payload: bytes,
    *,
    max_bytes: int,
    exclusive: bool,
) -> str:
    if len(payload) > max_bytes:
        raise ProposerBridgeError("write exceeds file byte bound")
    parts = _relative_parts(relative)
    parent = _open_directory_at(root_fd, parts[:-1], create=True)
    temporary = f".contiguous-{uuid.uuid4().hex}"
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=parent,
        )
        try:
            view = memoryview(payload)
            while view:
                written = os.write(descriptor, view)
                view = view[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        try:
            existing = os.stat(
                parts[-1], dir_fd=parent, follow_symlinks=False
            )
        except FileNotFoundError:
            existing = None
        if existing is not None and (
            exclusive
            or not stat.S_ISREG(existing.st_mode)
            or existing.st_nlink != 1
        ):
            raise ProposerBridgeError(
                "write destination exists, is aliased, or is nonregular"
            )
        os.replace(
            temporary,
            parts[-1],
            src_dir_fd=parent,
            dst_dir_fd=parent,
        )
        os.fsync(parent)
        final = os.stat(
            parts[-1], dir_fd=parent, follow_symlinks=False
        )
        if not stat.S_ISREG(final.st_mode) or final.st_nlink != 1:
            raise ProposerBridgeError(
                "write destination changed after replacement"
            )
        return _sha256(payload)
    finally:
        try:
            os.unlink(temporary, dir_fd=parent)
        except FileNotFoundError:
            pass
        os.close(parent)


def _list_at(root_fd: int, relative: str | None) -> list[dict[str, Any]]:
    parts = () if relative in {None, ""} else _relative_parts(relative)
    directory = _open_directory_at(root_fd, parts)
    try:
        records: list[dict[str, Any]] = []
        for name in sorted(os.listdir(directory)):
            if name.startswith("."):
                continue
            metadata = os.stat(
                name, dir_fd=directory, follow_symlinks=False
            )
            if stat.S_ISDIR(metadata.st_mode):
                kind = "directory"
            elif stat.S_ISREG(metadata.st_mode) and metadata.st_nlink == 1:
                kind = "file"
            else:
                raise ProposerBridgeError(
                    "workspace contains an aliased or special entry"
                )
            records.append(
                {
                    "name": name,
                    "kind": kind,
                    "size": metadata.st_size if kind == "file" else None,
                }
            )
        return records
    finally:
        os.close(directory)


def _complete_workspace_inventory(
    root_fd: int,
    *,
    max_file_bytes: int,
) -> tuple[tuple[tuple[str, str, int], ...], str, str, int]:
    """Inventory every workspace byte at the exact target-crossing barrier."""

    rows: list[tuple[str, str, int]] = []
    total_bytes = 0

    def visit(
        directory_fd: int,
        relative_parent: tuple[str, ...],
        depth: int,
    ) -> None:
        nonlocal total_bytes
        if depth > MAX_PATH_DEPTH:
            raise ProposerBridgeError(
                "boundary workspace exceeds its path-depth bound"
            )
        try:
            names = sorted(os.listdir(directory_fd))
        except OSError as exc:
            raise ProposerBridgeError(
                "boundary workspace cannot be enumerated"
            ) from exc
        for name in names:
            if (
                not isinstance(name, str)
                or not name
                or name in {".", ".."}
                or name.startswith(".")
                or "/" in name
                or "\x00" in name
            ):
                raise ProposerBridgeError(
                    "boundary workspace contains an unsafe entry"
                )
            parts = (*relative_parent, name)
            relative = "/".join(parts)
            if len(relative.encode("utf-8")) > MAX_PATH_BYTES:
                raise ProposerBridgeError(
                    "boundary workspace path exceeds its byte bound"
                )
            before = os.stat(
                name,
                dir_fd=directory_fd,
                follow_symlinks=False,
            )
            if stat.S_ISDIR(before.st_mode):
                child = os.open(
                    name,
                    os.O_RDONLY
                    | getattr(os, "O_DIRECTORY", 0)
                    | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=directory_fd,
                )
                try:
                    opened = os.fstat(child)
                    if (
                        not stat.S_ISDIR(opened.st_mode)
                        or (opened.st_dev, opened.st_ino)
                        != (before.st_dev, before.st_ino)
                    ):
                        raise ProposerBridgeError(
                            "boundary workspace directory changed"
                        )
                    visit(child, parts, depth + 1)
                finally:
                    os.close(child)
                continue
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_nlink != 1
                or before.st_size < 0
                or before.st_size > max_file_bytes
                or len(rows) >= MAX_BOUNDARY_WORKSPACE_FILES
            ):
                raise ProposerBridgeError(
                    "boundary workspace contains an aliased, special, "
                    "oversized, or excess file"
                )
            descriptor = os.open(
                name,
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=directory_fd,
            )
            try:
                opened = os.fstat(descriptor)
                stable = (
                    "st_dev",
                    "st_ino",
                    "st_mode",
                    "st_nlink",
                    "st_size",
                    "st_mtime_ns",
                    "st_ctime_ns",
                )
                if any(
                    getattr(before, field) != getattr(opened, field)
                    for field in stable
                ):
                    raise ProposerBridgeError(
                        "boundary workspace file identity changed"
                    )
                digest = hashlib.sha256()
                observed = 0
                while True:
                    block = os.read(descriptor, 1024 * 1024)
                    if not block:
                        break
                    digest.update(block)
                    observed += len(block)
                    if observed > max_file_bytes:
                        raise ProposerBridgeError(
                            "boundary workspace file grew beyond its bound"
                        )
                after = os.fstat(descriptor)
                if (
                    observed != before.st_size
                    or any(
                        getattr(opened, field) != getattr(after, field)
                        for field in stable
                    )
                ):
                    raise ProposerBridgeError(
                        "boundary workspace changed while freezing"
                    )
            finally:
                os.close(descriptor)
            total_bytes += observed
            if total_bytes > MAX_BOUNDARY_WORKSPACE_TOTAL_BYTES:
                raise ProposerBridgeError(
                    "boundary workspace exceeds its aggregate byte bound"
                )
            rows.append((relative, digest.hexdigest(), observed))

    root_copy = os.dup(root_fd)
    try:
        visit(root_copy, (), 0)
    finally:
        os.close(root_copy)
    normalized = tuple(sorted(rows))
    tree_digest = hashlib.sha256()
    for path, digest, _byte_count in normalized:
        tree_digest.update(path.encode("utf-8"))
        tree_digest.update(b"\0")
        tree_digest.update(digest.encode("ascii"))
        tree_digest.update(b"\n")
    inventory_payload = _canonical_json(
        {
            "files": [
                {"path": path, "sha256": digest, "bytes": byte_count}
                for path, digest, byte_count in normalized
            ],
            "file_count": len(normalized),
            "total_bytes": total_bytes,
        }
    )
    return (
        normalized,
        tree_digest.hexdigest(),
        _sha256(inventory_payload),
        total_bytes,
    )


def _valid_action(action: object) -> bool:
    if isinstance(action, int) and not isinstance(action, bool):
        return 1 <= action <= 7 and action != 6
    return (
        isinstance(action, list)
        and len(action) == 3
        and action[0] == 6
        and all(
            isinstance(value, int) and not isinstance(value, bool)
            for value in action
        )
        and all(0 <= value < 64 for value in action[1:])
    )


class ProposerBridgeServer:
    """Single-client authenticated worker bridge."""

    def __init__(
        self,
        *,
        socket_path: Path,
        token_file: Path,
        policy_path: Path,
        arena_socket: Path,
        arena_token_file: Path,
        workspace: Path,
        export: Path,
    ) -> None:
        self.socket_path = Path(socket_path)
        self.token_file = Path(token_file)
        self.policy, self.policy_sha256 = _load_policy(policy_path)
        self.token = _regular_bytes(
            self.token_file, max_bytes=4096
        ).decode("ascii").strip()
        if (
            len(self.token) < 32
            or not re.fullmatch(r"[0-9a-f]+", self.token)
        ):
            raise ProposerBridgeError("bridge token is malformed")
        arena_token = _regular_bytes(
            Path(arena_token_file), max_bytes=4096
        ).decode("ascii").strip()
        secret_names = sorted(
            name for name in os.environ if FORBIDDEN_ENV_NAME.search(name)
        )
        if secret_names:
            raise ProposerBridgeError(
                "provider/auth environment entered proposer container"
            )
        self.workspace = Path(workspace)
        self.export = Path(export)
        self.workspace_fd = _open_root(
            self.workspace, label="proposer workspace"
        )
        self.export_fd = _open_root(
            self.export, label="proposer export"
        )
        if any(self.export.iterdir()):
            raise ProposerBridgeError("proposer export root is not empty")
        self._write_outcome()
        try:
            from arc_agi3_arena_rpc_client import ArenaRpcClient

            self.arena_client = ArenaRpcClient(arena_socket, arena_token)
        except Exception as exc:
            raise ProposerBridgeError(
                "Arena RPC client could not initialize"
            ) from exc
        self.sequence = 0
        self.connection_challenge = uuid.uuid4().hex
        self.session_nonce: str | None = None
        self.cached: dict[str, tuple[str, dict[str, Any]]] = {}
        self.mutation_sequence = 0
        self.published = False
        self.progress_sequence = 0
        self.exploration_suffix: list[Any] = []
        self.target_boundary: dict[str, Any] | None = None
        self.target_boundary_sha256: str | None = None
        self.boundary_workspace_files: dict[
            str, tuple[str, int]
        ] | None = None
        self._active_request: Mapping[str, Any] | None = None

    def _write_outcome(self) -> None:
        payload = _canonical_json(
            {
                "schema": SCHEMA,
                "kind": "arc_agi3_contiguous_proposer_worker",
                "attempt_id": self.policy.attempt_id,
                "authoritative": False,
            }
        ) + b"\n"
        _write_at(
            self.export_fd,
            WORKER_OUTCOME_NAME,
            payload,
            max_bytes=self.policy.max_file_bytes,
            exclusive=True,
        )

    def close(self) -> None:
        try:
            self.arena_client.close()
        except Exception:
            pass
        os.close(self.workspace_fd)
        os.close(self.export_fd)

    def _response(
        self,
        request: Mapping[str, Any],
        *,
        success: bool,
        result: object = None,
        error: str | None = None,
    ) -> dict[str, Any]:
        value = {
            "schema": SCHEMA,
            "kind": "arc_agi3_contiguous_bridge_response",
            "attempt_id": self.policy.attempt_id,
            "request_id": request.get("request_id"),
            "sequence": request.get("sequence"),
            "success": success,
            "result": result if success else None,
            "error": error if not success else None,
        }
        if len(_canonical_json(value)) > self.policy.max_response_bytes:
            raise ProposerBridgeError("bridge response exceeds byte bound")
        return value

    def _validate_request(
        self, request: object
    ) -> tuple[dict[str, Any], str]:
        required = {
            "schema",
            "kind",
            "protocol_version",
            "attempt_id",
            "request_id",
            "sequence",
            "session_nonce",
            "operation",
            "mutation_id",
            "challenge_nonce",
            "auth_hmac",
            "arguments",
        }
        if (
            not isinstance(request, dict)
            or set(request) != required
            or request["schema"] != SCHEMA
            or request["kind"]
            != "arc_agi3_contiguous_bridge_request"
            or request["protocol_version"] != PROTOCOL_VERSION
            or request["attempt_id"] != self.policy.attempt_id
            or not isinstance(request["request_id"], str)
            or not isinstance(request["sequence"], int)
            or isinstance(request["sequence"], bool)
            or not isinstance(request["operation"], str)
            or request["operation"] not in self.policy.operation_allowlist
            or not isinstance(request["arguments"], dict)
            or request["challenge_nonce"] != self.connection_challenge
            or not isinstance(request["auth_hmac"], str)
            or not re.fullmatch(r"[0-9a-f]{64}", request["auth_hmac"])
        ):
            raise ProposerBridgeError("bridge request schema/auth mismatch")
        authenticated = {
            key: value
            for key, value in request.items()
            if key != "auth_hmac"
        }
        expected_hmac = hmac.new(
            self.token.encode("ascii"),
            _canonical_json(authenticated),
            hashlib.sha256,
        ).hexdigest()
        if not hmac.compare_digest(request["auth_hmac"], expected_hmac):
            raise ProposerBridgeError("bridge request authentication failed")
        try:
            parsed = uuid.UUID(request["request_id"])
        except (ValueError, TypeError) as exc:
            raise ProposerBridgeError("request_id is not a UUID") from exc
        if str(parsed) != request["request_id"]:
            raise ProposerBridgeError("request_id is not canonical")
        digest = _sha256(_canonical_json(request))
        prior = self.cached.get(request["request_id"])
        if prior is not None:
            if prior[0] != digest:
                raise ProposerBridgeError(
                    "request_id was replayed with different bytes"
                )
            return request, digest
        if request["sequence"] != self.sequence + 1:
            raise ProposerBridgeError("bridge sequence is not contiguous")
        if request["operation"] == "handshake":
            if (
                self.sequence != 0
                or self.session_nonce is not None
                or request["session_nonce"] is not None
                or request["mutation_id"] is not None
            ):
                raise ProposerBridgeError("handshake order is invalid")
        elif (
            not isinstance(request["session_nonce"], str)
            or request["session_nonce"] != self.session_nonce
        ):
            raise ProposerBridgeError("bridge session nonce mismatch")
        return request, digest

    def dispatch(self, request_value: object) -> dict[str, Any]:
        request, digest = self._validate_request(request_value)
        cached = self.cached.get(request["request_id"])
        if cached is not None:
            return cached[1]
        # Capacity is reserved before _execute: no mutation may commit unless
        # its exact response can be retained for byte-identical replay.
        if len(self.cached) >= MAX_CACHED_RESPONSES:
            raise ProposerBridgeError(
                "bridge idempotency cache exhausted"
            )
        operation = request["operation"]
        mutating = operation in {
            "arena_reset",
            "arena_step",
            "candidate_publish",
            "progress",
            "wip_publish",
            "workspace_mkdir",
            "workspace_remove",
            "workspace_write",
        }
        if mutating:
            expected_mutation = self.mutation_sequence + 1
            if request["mutation_id"] != (
                f"{self.policy.attempt_id}:{expected_mutation:08d}"
            ):
                raise ProposerBridgeError(
                    "mutating call lacks the next idempotency identity"
                )
        elif request["mutation_id"] is not None:
            raise ProposerBridgeError(
                "read-only bridge call carries mutation identity"
            )
        try:
            self._active_request = request
            result = self._execute(operation, request["arguments"])
            response = self._response(
                request, success=True, result=result
            )
        except ProposerBridgeError as exc:
            response = self._response(
                request,
                success=False,
                error=type(exc).__name__ + ": " + str(exc),
            )
        finally:
            self._active_request = None
        # Publish the response to the replay cache before advancing either
        # logical counter.  The single-client dispatcher makes this one
        # indivisible protocol transition from the next request's perspective.
        self.cached[request["request_id"]] = (digest, response)
        self.sequence += 1
        if mutating:
            self.mutation_sequence += 1
        return response

    def _execute(
        self, operation: str, arguments: Mapping[str, Any]
    ) -> object:
        if self.target_boundary is not None and operation in {
            "arena_observe",
            "arena_reset",
            "arena_step",
            "workspace_mkdir",
            "workspace_remove",
            "workspace_write",
            "wip_publish",
        }:
            raise ProposerBridgeError(
                "target boundary is frozen; post-target mutation/"
                "observation is forbidden"
            )
        if operation == "handshake":
            if arguments:
                raise ProposerBridgeError("handshake arguments must be empty")
            self.session_nonce = uuid.uuid4().hex
            return {
                "protocol_version": PROTOCOL_VERSION,
                "campaign_id": self.policy.campaign_id,
                "generation_id": self.policy.generation_id,
                "attempt_id": self.policy.attempt_id,
                "game": self.policy.game,
                "target_level": self.policy.target_level,
                "frontier_sha256": self.policy.frontier_sha256,
                "policy_sha256": self.policy_sha256,
                "session_nonce": self.session_nonce,
                "operation_allowlist": list(
                    self.policy.operation_allowlist
                ),
                "exec_allowlist": list(self.policy.exec_allowlist),
                "environment_names": sorted(os.environ),
                "provider_credential_names": [],
            }
        if operation == "workspace_list":
            if set(arguments) != {"path"}:
                raise ProposerBridgeError("workspace_list schema mismatch")
            return {"entries": _list_at(self.workspace_fd, arguments["path"])}
        if operation == "arena_observe":
            if arguments:
                raise ProposerBridgeError(
                    "arena_observe arguments must be empty"
                )
            return self._arena_observation()
        if operation == "arena_reset":
            if arguments:
                raise ProposerBridgeError(
                    "arena_reset arguments must be empty"
                )
            self.arena_client.root.reset()
            self.exploration_suffix = []
            return self._arena_observation()
        if operation == "arena_step":
            if set(arguments) != {"action"} or not _valid_action(
                arguments["action"]
            ):
                raise ProposerBridgeError("arena_step action is invalid")
            action = arguments["action"]
            levels_before = self.arena_client.root.levels_completed
            if levels_before >= self.policy.target_level:
                raise ProposerBridgeError(
                    "Arena was already at or beyond the target without a "
                    "frozen crossing"
                )
            if isinstance(action, list):
                self.arena_client.root.step(*action)
            else:
                self.arena_client.root.step(action)
            self.exploration_suffix.append(action)
            levels_after = self.arena_client.root.levels_completed
            if levels_after > self.policy.target_level:
                raise ProposerBridgeError(
                    "Arena crossed beyond the exact target level"
                )
            if levels_after == self.policy.target_level:
                return self._freeze_target_boundary(
                    action=action,
                    levels_before=levels_before,
                    levels_after=levels_after,
                )
            return self._arena_observation()
        if operation == "workspace_read":
            if set(arguments) != {"path"}:
                raise ProposerBridgeError("workspace_read schema mismatch")
            raw = _read_at(
                self.workspace_fd,
                arguments["path"],
                max_bytes=self.policy.max_file_bytes,
            )
            try:
                text = raw.decode("utf-8")
            except UnicodeError as exc:
                raise ProposerBridgeError(
                    "workspace_read requires UTF-8 text"
                ) from exc
            return {"text": text, "sha256": _sha256(raw), "bytes": len(raw)}
        if operation == "workspace_write":
            if set(arguments) != {"path", "text"} or not isinstance(
                arguments["text"], str
            ):
                raise ProposerBridgeError("workspace_write schema mismatch")
            raw = arguments["text"].encode("utf-8")
            digest = _write_at(
                self.workspace_fd,
                arguments["path"],
                raw,
                max_bytes=self.policy.max_file_bytes,
                exclusive=False,
            )
            return {"sha256": digest, "bytes": len(raw)}
        if operation == "workspace_mkdir":
            if set(arguments) != {"path"}:
                raise ProposerBridgeError("workspace_mkdir schema mismatch")
            directory = _open_directory_at(
                self.workspace_fd,
                _relative_parts(arguments["path"]),
                create=True,
            )
            os.fsync(directory)
            os.close(directory)
            return {"created": True}
        if operation == "workspace_remove":
            if set(arguments) != {"path"}:
                raise ProposerBridgeError("workspace_remove schema mismatch")
            parts = _relative_parts(arguments["path"])
            parent = _open_directory_at(self.workspace_fd, parts[:-1])
            try:
                metadata = os.stat(
                    parts[-1], dir_fd=parent, follow_symlinks=False
                )
                if (
                    not stat.S_ISREG(metadata.st_mode)
                    or metadata.st_nlink != 1
                ):
                    raise ProposerBridgeError(
                        "workspace_remove accepts regular files only"
                    )
                os.unlink(parts[-1], dir_fd=parent)
                os.fsync(parent)
            finally:
                os.close(parent)
            return {"removed": True}
        if operation == "probe_snapshot":
            return self._probe_snapshot(arguments)
        if operation == "candidate_publish":
            return self._publish(arguments, candidate=True)
        if operation == "wip_publish":
            return self._publish(arguments, candidate=False)
        if operation == "progress":
            if set(arguments) != {"message"} or not isinstance(
                arguments["message"], str
            ):
                raise ProposerBridgeError("progress schema mismatch")
            message = arguments["message"]
            if not message or len(message.encode("utf-8")) > 4096:
                raise ProposerBridgeError("progress message is invalid")
            self.progress_sequence += 1
            return {
                "progress_sequence": self.progress_sequence,
                "message_sha256": _sha256(message.encode("utf-8")),
            }
        raise ProposerBridgeError("unimplemented bridge operation")

    def _probe_snapshot(
        self, arguments: Mapping[str, Any]
    ) -> dict[str, Any]:
        if set(arguments) != {"paths", "binding"}:
            raise ProposerBridgeError("probe_snapshot schema mismatch")
        paths = arguments["paths"]
        binding = arguments["binding"]
        if (
            not isinstance(paths, list)
            or not 1 <= len(paths) <= 128
            or len(paths) != len(set(paths))
            or not isinstance(binding, dict)
            or set(binding)
            != {
                "dynamic_request_id",
                "dynamic_call_id",
                "thread_id",
                "turn_id",
            }
            or not isinstance(binding["dynamic_request_id"], (str, int))
            or isinstance(binding["dynamic_request_id"], bool)
            or not all(
                isinstance(binding[name], str)
                and 0 < len(binding[name]) <= 200
                and "\x00" not in binding[name]
                for name in (
                    "dynamic_call_id",
                    "thread_id",
                    "turn_id",
                )
            )
        ):
            raise ProposerBridgeError("probe snapshot binding is invalid")
        entries: list[dict[str, Any]] = []
        total = 0
        for path in sorted(paths):
            if not isinstance(path, str):
                raise ProposerBridgeError(
                    "probe snapshot path is not text"
                )
            raw = _read_at(
                self.workspace_fd,
                path,
                max_bytes=self.policy.max_file_bytes,
            )
            total += len(raw)
            if total > min(
                self.policy.max_total_export_bytes,
                self.policy.max_response_bytes // 2,
            ):
                raise ProposerBridgeError(
                    "probe snapshot aggregate exceeds response bound"
                )
            entries.append(
                {
                    "path": path,
                    "sha256": _sha256(raw),
                    "bytes": len(raw),
                    "base64": base64.b64encode(raw).decode("ascii"),
                }
            )
        inventory = [
            {
                "path": entry["path"],
                "sha256": entry["sha256"],
                "bytes": entry["bytes"],
            }
            for entry in entries
        ]
        return {
            "binding": dict(binding),
            "entries": entries,
            "inventory_sha256": _sha256(_canonical_json(inventory)),
            "total_bytes": total,
            "quiescent": True,
            "no_writeback": True,
        }

    def _freeze_target_boundary(
        self,
        *,
        action: Any,
        levels_before: int,
        levels_after: int,
    ) -> dict[str, Any]:
        """Freeze source/action identity before the response can be emitted."""

        request = self._active_request
        if (
            self.target_boundary is not None
            or not isinstance(request, Mapping)
            or request.get("operation") != "arena_step"
            or request.get("sequence") != self.sequence + 1
            or request.get("mutation_id")
            != (
                f"{self.policy.attempt_id}:"
                f"{self.mutation_sequence + 1:08d}"
            )
            or not isinstance(levels_before, int)
            or isinstance(levels_before, bool)
            or levels_before != self.policy.target_level - 1
            or levels_after != self.policy.target_level
        ):
            raise ProposerBridgeError(
                "target crossing lacks an exact active bridge identity"
            )
        (
            rows,
            workspace_tree_sha256,
            workspace_inventory_sha256,
            total_bytes,
        ) = _complete_workspace_inventory(
            self.workspace_fd,
            max_file_bytes=self.policy.max_file_bytes,
        )
        suffix_payload = _canonical_json(self.exploration_suffix)
        boundary = {
            "schema": SCHEMA,
            "kind": "arc_agi3_contiguous_target_boundary",
            "attempt_id": self.policy.attempt_id,
            "game": self.policy.game,
            "target_level": self.policy.target_level,
            "levels_before": levels_before,
            "levels_completed": levels_after,
            "arena_binding_sha256":
                self.arena_client.binding_sha256,
            "bridge_request_id": request["request_id"],
            "bridge_sequence": request["sequence"],
            "bridge_mutation_id": request["mutation_id"],
            "crossing_action_sha256": _sha256(
                _canonical_json(action)
            ),
            "exploration_suffix_sha256": _sha256(suffix_payload),
            "exploration_suffix_length": len(
                self.exploration_suffix
            ),
            "workspace_tree_sha256": workspace_tree_sha256,
            "workspace_inventory_sha256":
                workspace_inventory_sha256,
            "workspace_file_count": len(rows),
            "workspace_total_bytes": total_bytes,
        }
        boundary_sha256 = _sha256(_canonical_json(boundary))
        result = {
            "target_reached": True,
            "boundary": boundary,
            "boundary_sha256": boundary_sha256,
        }
        # Reserve response capacity before committing the in-memory freeze.
        # If this fails, no target-crossing bytes are deliverable to the model.
        self._response(request, success=True, result=result)
        self.boundary_workspace_files = {
            path: (digest, byte_count)
            for path, digest, byte_count in rows
        }
        self.target_boundary = boundary
        self.target_boundary_sha256 = boundary_sha256
        return result

    def _arena_observation(self) -> dict[str, Any]:
        # Refresh through the authenticated RPC on every observation.  The
        # client proxy is a cache, not evidence that the host seed/clone state
        # remained unchanged.
        frame = self.arena_client.root.observe()
        if hasattr(frame, "tolist"):
            frame = frame.tolist()
        if (
            not isinstance(frame, list)
            or not all(isinstance(row, list) for row in frame)
        ):
            raise ProposerBridgeError("Arena frame is not public JSON data")
        return {
            "frame": frame,
            "actions": list(self.arena_client.root.actions),
            "levels_completed": self.arena_client.root.levels_completed,
            "terminal": self.arena_client.root.terminal(),
        }

    def _publish(
        self, arguments: Mapping[str, Any], *, candidate: bool
    ) -> dict[str, Any]:
        expected = (
            {"candidate_path", "exports"} if candidate else {"exports"}
        )
        if set(arguments) != expected or self.published:
            raise ProposerBridgeError(
                "publish schema mismatch or output already finalized"
            )
        if candidate and (
            self.target_boundary is None
            or self.target_boundary_sha256 is None
            or self.boundary_workspace_files is None
        ):
            raise ProposerBridgeError(
                "candidate publication requires the frozen exact target "
                "boundary"
            )
        exports = arguments["exports"]
        if (
            not isinstance(exports, dict)
            or not exports
            or len(exports) > 512
        ):
            raise ProposerBridgeError("declared exports are invalid")
        payloads: dict[str, bytes] = {}
        total = 0
        for source, destination in exports.items():
            _relative_parts(source)
            destination_parts = _relative_parts(destination)
            if destination_parts[-1] in FORBIDDEN_EXPORT_NAMES:
                raise ProposerBridgeError("forbidden export destination")
            raw = _read_at(
                self.workspace_fd,
                source,
                max_bytes=self.policy.max_file_bytes,
            )
            if candidate:
                assert self.boundary_workspace_files is not None
                frozen = self.boundary_workspace_files.get(source)
                if frozen != (_sha256(raw), len(raw)):
                    raise ProposerBridgeError(
                        "candidate source differs from the pre-debrief "
                        "workspace boundary"
                    )
            try:
                raw.decode("utf-8")
            except UnicodeError as exc:
                raise ProposerBridgeError(
                    "only UTF-8 candidate/WIP exports are admitted"
                ) from exc
            if (
                not candidate
                and (
                    len(destination_parts) < 2
                    or destination_parts[0] != "wip"
                )
            ):
                raise ProposerBridgeError(
                    "WIP exports must use the wip/ destination prefix"
                )
            total += len(raw)
            if total > self.policy.max_total_export_bytes:
                raise ProposerBridgeError("aggregate export bound exceeded")
            if destination in payloads:
                raise ProposerBridgeError("duplicate export destination")
            payloads[destination] = raw
        actions: list[Any] | None = None
        wip_source_payloads: dict[str, bytes] | None = None
        wip_tree_payloads: dict[str, bytes] | None = None
        if candidate:
            actions_value = arguments["candidate_path"]
            if (
                not isinstance(actions_value, list)
                or not actions_value
                or len(actions_value) > 600
                or not all(
                    _valid_action(action)
                    for action in actions_value
                )
            ):
                raise ProposerBridgeError(
                    "candidate path grammar invalid"
                )
            source_payloads: dict[str, bytes] = {}
            for destination, raw in payloads.items():
                parts = _relative_parts(destination)
                if len(parts) != 1:
                    raise ProposerBridgeError(
                        "candidate source destinations must be flat"
                    )
                source_payloads[destination] = raw
            try:
                SourceSchema.validate_source_payloads(
                    source_payloads
                )
            except SourceSchema.SourceSchemaError as exc:
                raise ProposerBridgeError(
                    "candidate exports violate the winning-source schema"
                ) from exc
            actions = actions_value
        else:
            wip_source_payloads = {}
            wip_tree_payloads = {}
            for destination, raw in payloads.items():
                parts = _relative_parts(destination)
                if (
                    len(parts) == 3
                    and parts[:2] == ("wip", "solver_source")
                ):
                    wip_source_payloads[parts[2]] = raw
                elif (
                    len(parts) >= 3
                    and parts[:2] == ("wip", "context")
                ):
                    pass
                else:
                    raise ProposerBridgeError(
                        "WIP exports must be either flat "
                        "wip/solver_source files or wip/context files"
                    )
                relative = PurePosixPath(*parts[1:]).as_posix()
                wip_tree_payloads[relative] = raw
            try:
                SourceSchema.validate_source_payloads(
                    wip_source_payloads
                )
            except SourceSchema.SourceSchemaError as exc:
                raise ProposerBridgeError(
                    "WIP solver source violates the closed source schema"
                ) from exc
        existing = {
            entry.name for entry in self.export.iterdir()
        }
        if existing != {WORKER_OUTCOME_NAME}:
            raise ProposerBridgeError(
                "export root contains unexpected prepublication bytes"
            )
        exported_hashes = {
            WORKER_OUTCOME_NAME: _sha256(
                _read_at(
                    self.export_fd,
                    WORKER_OUTCOME_NAME,
                    max_bytes=self.policy.max_file_bytes,
                )
            )
        }
        if candidate:
            assert actions is not None
            final_name = "source"
            exported_hashes.update(
                {
                    f"{final_name}/{destination}": _sha256(raw)
                    for destination, raw in sorted(payloads.items())
                }
            )
            manifest_name = CANDIDATE_NAME
            manifest = {
                "schema": SCHEMA,
                "game": self.policy.game,
                "target_level": self.policy.target_level,
                # The parent digest is deliberately taken from immutable policy
                # context by the trusted input builder, never from model text.
                "parent_checkpoint_sha256":
                    self.policy.parent_checkpoint_sha256,
                "target_boundary_sha256":
                    self.target_boundary_sha256,
                "target_boundary_sequence":
                    self.target_boundary["bridge_sequence"],
                "target_boundary_mutation_id":
                    self.target_boundary["bridge_mutation_id"],
                "boundary_workspace_tree_sha256":
                    self.target_boundary["workspace_tree_sha256"],
                "candidate_path": actions,
                "exported_files_sha256": exported_hashes,
            }
            if not isinstance(
                manifest["parent_checkpoint_sha256"], str
            ) or not re.fullmatch(
                r"[0-9a-f]{64}",
                manifest["parent_checkpoint_sha256"],
            ):
                raise ProposerBridgeError(
                    "host bridge omitted bound parent checkpoint"
                )
        else:
            assert (
                wip_source_payloads is not None
                and wip_tree_payloads is not None
            )
            exported_hashes.update(
                {
                    destination: _sha256(raw)
                    for destination, raw in sorted(payloads.items())
                }
            )
            manifest_name = WIP_MANIFEST_NAME
            manifest = {
                "schema": SCHEMA,
                "kind": "arc_agi3_contiguous_wip",
                "game": self.policy.game,
                "target_level": self.policy.target_level,
                "frontier_sha256": self.policy.frontier_sha256,
                "parent_checkpoint_sha256":
                    self.policy.parent_checkpoint_sha256,
                "wip_root_relative_path": "wip",
                "wip_tree_sha256":
                    _payload_tree_sha256(wip_tree_payloads),
                "solver_source_relative_path":
                    "wip/solver_source",
                "solver_source_tree_sha256":
                    _payload_tree_sha256(wip_source_payloads),
                "exported_files_sha256": exported_hashes,
            }
        manifest_raw = _canonical_json(manifest) + b"\n"
        projected_result = {
            "outcome": "candidate" if candidate else "wip",
            "manifest": manifest_name,
            "manifest_sha256": "0" * 64,
            "exported_files_sha256": exported_hashes,
            "total_export_bytes": total,
        }
        if (
            len(manifest_raw) > self.policy.max_file_bytes
            or len(_canonical_json(projected_result)) + 512
            > self.policy.max_response_bytes
        ):
            raise ProposerBridgeError(
                "publication manifest/response would exceed its bound"
            )

        if candidate:
            stage_name = (
                "candidate_stage_"
                + uuid.uuid4().hex
            )
            os.mkdir(
                stage_name,
                mode=0o700,
                dir_fd=self.export_fd,
            )
            stage_fd = os.open(
                stage_name,
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=self.export_fd,
            )
            try:
                for destination, raw in sorted(payloads.items()):
                    digest = _write_at(
                        stage_fd,
                        destination,
                        raw,
                        max_bytes=self.policy.max_file_bytes,
                        exclusive=True,
                    )
                    manifest_path = f"{final_name}/{destination}"
                    if digest != exported_hashes[manifest_path]:
                        raise ProposerBridgeError(
                            "candidate source changed during staging"
                        )
                os.fsync(stage_fd)
            finally:
                os.close(stage_fd)
            os.rename(
                stage_name,
                final_name,
                src_dir_fd=self.export_fd,
                dst_dir_fd=self.export_fd,
            )
            os.fsync(self.export_fd)
        else:
            for destination, raw in sorted(payloads.items()):
                digest = _write_at(
                    self.export_fd,
                    destination,
                    raw,
                    max_bytes=self.policy.max_file_bytes,
                    exclusive=True,
                )
                if digest != exported_hashes[destination]:
                    raise ProposerBridgeError(
                        "published WIP hash changed during staging"
                    )
        manifest_sha256 = _write_at(
            self.export_fd,
            manifest_name,
            manifest_raw,
            max_bytes=self.policy.max_file_bytes,
            exclusive=True,
        )
        self.published = True
        return {
            "outcome": "candidate" if candidate else "wip",
            "manifest": manifest_name,
            "manifest_sha256": manifest_sha256,
            "exported_files_sha256": exported_hashes,
            "total_export_bytes": total,
        }

    def serve(self) -> int:
        if self.socket_path.exists() or self.socket_path.is_symlink():
            raise ProposerBridgeError("bridge socket path already exists")
        if self.socket_path.parent.is_symlink() or not (
            self.socket_path.parent.is_dir()
        ):
            raise ProposerBridgeError("bridge socket root is invalid")
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            listener.bind(str(self.socket_path))
            os.chmod(self.socket_path, 0o600, follow_symlinks=False)
            listener.listen(SOCKET_BACKLOG)
            connections = 0
            while connections < MAX_SEQUENTIAL_CONNECTIONS:
                listener.settimeout(
                    None
                    if connections == 0
                    else RECONNECT_GRACE_SECONDS
                )
                try:
                    connection, _address = listener.accept()
                except TimeoutError:
                    return 0
                connections += 1
                challenge = _canonical_json(
                    {
                        "schema": SCHEMA,
                        "kind":
                            "arc_agi3_contiguous_bridge_challenge",
                        "protocol_version": PROTOCOL_VERSION,
                        "attempt_id": self.policy.attempt_id,
                        "challenge_nonce": self.connection_challenge,
                    }
                ) + b"\n"
                try:
                    connection.sendall(challenge)
                    # A queued simultaneous client remains a boundary
                    # violation.  A later sequential reconnect is admitted
                    # only during the short lost-response recovery grace.
                    listener.setblocking(False)
                    selector = selectors.DefaultSelector()
                    selector.register(
                        listener, selectors.EVENT_READ, "listener"
                    )
                    selector.register(
                        connection,
                        selectors.EVENT_READ,
                        "connection",
                    )
                    try:
                        buffer = bytearray()
                        connection_lost = False
                        while True:
                            events = selector.select()
                            listener_ready = any(
                                key.data == "listener"
                                for key, _mask in events
                            )
                            connection_ready = any(
                                key.data == "connection"
                                for key, _mask in events
                            )
                            if listener_ready:
                                try:
                                    peek = connection.recv(
                                        1,
                                        socket.MSG_PEEK
                                        | getattr(
                                            socket,
                                            "MSG_DONTWAIT",
                                            0,
                                        ),
                                    )
                                except BlockingIOError:
                                    peek = None
                                if peek == b"":
                                    # The old descriptor reached EOF before
                                    # the queued connection is admitted.
                                    break
                                raise ProposerBridgeError(
                                    "second bridge client attempted to connect"
                                )
                            if not connection_ready:
                                continue
                            block = connection.recv(65536)
                            if not block:
                                break
                            buffer.extend(block)
                            if (
                                len(buffer)
                                > self.policy.max_request_bytes
                            ):
                                raise ProposerBridgeError(
                                    "bridge request exceeds byte bound"
                                )
                            while b"\n" in buffer:
                                line, _, remainder = buffer.partition(
                                    b"\n"
                                )
                                buffer[:] = remainder
                                if not line:
                                    raise ProposerBridgeError(
                                        "empty bridge request"
                                    )
                                try:
                                    request = _strict_json_loads(line)
                                except ProposerBridgeError as exc:
                                    raise ProposerBridgeError(
                                        "invalid bridge request JSON"
                                    ) from exc
                                response = self.dispatch(request)
                                payload = (
                                    _canonical_json(response) + b"\n"
                                )
                                try:
                                    connection.sendall(payload)
                                except (
                                    BrokenPipeError,
                                    ConnectionError,
                                    OSError,
                                ):
                                    connection_lost = True
                                    break
                            if connection_lost:
                                break
                        # A partial unauthenticated record has no effect and
                        # is discarded with this connection.  The reconnecting
                        # client must resend one complete, byte-identical HMAC
                        # request before dispatch can occur.
                    finally:
                        selector.close()
                finally:
                    connection.close()
            raise ProposerBridgeError(
                "bridge sequential reconnect bound exhausted"
            )
        finally:
            listener.close()
            try:
                self.socket_path.unlink()
            except FileNotFoundError:
                pass
            self.close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bridge-socket", type=Path, required=True)
    parser.add_argument("--bridge-token-file", type=Path, required=True)
    parser.add_argument("--bridge-policy", type=Path, required=True)
    parser.add_argument("--arena-socket", type=Path, required=True)
    parser.add_argument("--arena-token-file", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--export", type=Path, required=True)
    return parser


def _terminate(_signum: int, _frame: object) -> None:
    raise SystemExit(143)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    signal.signal(signal.SIGTERM, _terminate)
    error_line = b""
    exit_code = 0
    try:
        server = ProposerBridgeServer(
            socket_path=arguments.bridge_socket,
            token_file=arguments.bridge_token_file,
            policy_path=arguments.bridge_policy,
            arena_socket=arguments.arena_socket,
            arena_token_file=arguments.arena_token_file,
            workspace=arguments.workspace,
            export=arguments.export,
        )
        exit_code = server.serve()
    except BaseException as exc:
        exit_code = (
            int(exc.code)
            if isinstance(exc, SystemExit)
            and isinstance(exc.code, int)
            else 1
        )
        error_line = _canonical_json(
            {
                "schema": SCHEMA,
                "kind": "arc_agi3_contiguous_proposer_worker_error",
                "error_type": type(exc).__name__,
            }
        ) + b"\n"
        sys.stderr.buffer.write(error_line)
        sys.stderr.buffer.flush()
    terminal = _canonical_json(
        {
            "schema": SCHEMA,
            "kind": "arc_agi3_contiguous_proposer_worker_terminal",
            "exit_code": exit_code,
            "child_stdio_captured": True,
            "stderr_bytes": len(error_line),
            "stderr_sha256": _sha256(error_line),
            "stdout_prefix_bytes": 0,
            "stdout_prefix_sha256": _sha256(b""),
        }
    ) + b"\n"
    sys.stdout.buffer.write(terminal)
    sys.stdout.buffer.flush()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
