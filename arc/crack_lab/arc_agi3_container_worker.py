#!/usr/bin/env python3
"""Pinned container-side entrypoint for one ARC-AGI-3 solver execution.

This process has no engine import and receives no game/runtime source.  It
loads one exported ``solve(env)`` module, connects to the host-owned Arena RPC
socket, and passes the public proxy to the solver.  The worker outcome is
diagnostic only: the host RPC session owns the authoritative action path and
reward, and promotion requires independent trusted replay.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import sys
import time
import types
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from arc_agi3_arena_rpc_client import ArenaRpcClient


WORKER_SCHEMA = "arc-agi3-container-worker/v1"
MAX_SOLVER_BYTES = 16 * 1024 * 1024
_SHA256_RE = __import__("re").compile(r"^[0-9a-f]{64}$")


class WorkerContractError(RuntimeError):
    """Invalid entrypoint arguments, source, or output path."""


@dataclass(frozen=True)
class WorkerConfig:
    socket_path: Path
    token_file: Path
    solve_path: Path
    outcome_path: Path


def _reject_alias(path: Path, *, label: str, must_exist: bool) -> os.stat_result | None:
    absolute = path.absolute()
    current = Path(absolute.anchor)
    for part in absolute.parts[1:-1]:
        current /= part
        try:
            metadata = os.lstat(current)
        except FileNotFoundError:
            if must_exist:
                raise WorkerContractError(f"{label} ancestor is missing")
            continue
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise WorkerContractError(
                f"{label} has a symlinked or non-directory ancestor"
            )
    try:
        metadata = os.lstat(absolute)
    except FileNotFoundError:
        if must_exist:
            raise WorkerContractError(f"{label} is missing")
        return None
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
    ):
        raise WorkerContractError(f"{label} must be an unaliased regular file")
    return metadata


def parse_args(argv: Sequence[str]) -> WorkerConfig:
    """Parse four required ``--name=value`` fields with no last-value-wins."""

    if not isinstance(argv, (list, tuple)):
        raise WorkerContractError("worker arguments must be a sequence")
    if any(arg in ("-h", "--help") for arg in argv):
        if len(argv) != 1:
            raise WorkerContractError("--help cannot be combined with other arguments")
        raise SystemExit(
            "usage: arc_agi3_container_worker.py "
            "--socket=PATH --token-file=PATH --solve=PATH --outcome=PATH"
        )
    allowed = {"socket", "token-file", "solve", "outcome"}
    parsed: dict[str, str] = {}
    for raw in argv:
        if not isinstance(raw, str) or not raw.startswith("--") or "=" not in raw:
            raise WorkerContractError(
                "worker arguments must use exact --name=value syntax"
            )
        name, value = raw[2:].split("=", 1)
        if name not in allowed:
            raise WorkerContractError(f"unknown worker argument: --{name}")
        if name in parsed:
            raise WorkerContractError(f"duplicate worker argument: --{name}")
        if not value:
            raise WorkerContractError(f"worker argument --{name} cannot be empty")
        parsed[name] = value
    missing = sorted(allowed - set(parsed))
    if missing:
        raise WorkerContractError(f"missing worker arguments: {missing}")
    return WorkerConfig(
        socket_path=Path(parsed["socket"]),
        token_file=Path(parsed["token-file"]),
        solve_path=Path(parsed["solve"]),
        outcome_path=Path(parsed["outcome"]),
    )


def _read_token(path: Path) -> str:
    token = _read_regular_bytes(
        path, label="RPC token file", max_bytes=512
    )
    try:
        text = token.decode("ascii")
    except UnicodeDecodeError as exc:
        raise WorkerContractError("RPC token must be ASCII") from exc
    if (
        not 32 <= len(text) <= 256
        or any(ord(character) < 33 or ord(character) > 126
               for character in text)
    ):
        raise WorkerContractError("RPC token format is invalid")
    return text


def _read_regular_bytes(
    path: Path,
    *,
    label: str,
    max_bytes: int,
) -> bytes:
    metadata = _reject_alias(path, label=label, must_exist=True)
    assert metadata is not None
    if metadata.st_size > max_bytes:
        raise WorkerContractError(f"{label} exceeds byte limit")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise WorkerContractError(f"{label} cannot be opened safely") from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or (before.st_dev, before.st_ino)
            != (metadata.st_dev, metadata.st_ino)
        ):
            raise WorkerContractError(f"{label} changed during admission")
        data = bytearray()
        while True:
            block = os.read(descriptor, min(1024 * 1024, max_bytes + 1))
            if not block:
                break
            data.extend(block)
            if len(data) > max_bytes:
                raise WorkerContractError(f"{label} exceeds byte limit")
        after = os.fstat(descriptor)
        for field in (
            "st_dev",
            "st_ino",
            "st_mode",
            "st_nlink",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
        ):
            if getattr(before, field) != getattr(after, field):
                raise WorkerContractError(f"{label} changed while being read")
        if len(data) != after.st_size:
            raise WorkerContractError(f"{label} produced a short read")
        return bytes(data)
    finally:
        os.close(descriptor)


def _load_solver(path: Path) -> tuple[Callable[[Any], Any], str]:
    source = _read_regular_bytes(
        path, label="solver source", max_bytes=MAX_SOLVER_BYTES
    )
    source_hash = hashlib.sha256(source).hexdigest()
    return _load_solver_bytes(path, source), source_hash


def _load_solver_bytes(
    path: Path,
    source: bytes,
) -> Callable[[Any], Any]:
    for name in ("solve", "players", "legs"):
        sys.modules.pop(name, None)
    parent = str(path.absolute().parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    # The worker executes exactly one solver and then exits, so keeping this
    # one attempt-local directory on sys.path is both bounded and necessary for
    # imports performed lazily inside solve(env).
    try:
        code = compile(
            source,
            "<attempt-solver>",
            "exec",
            dont_inherit=True,
        )
    except (SyntaxError, ValueError, TypeError) as exc:
        raise WorkerContractError("solver source could not be compiled") from exc
    module = types.ModuleType("solve")
    module.__file__ = "<attempt-solver>"
    module.__package__ = ""
    sys.modules["solve"] = module
    exec(code, module.__dict__)
    solve = getattr(module, "solve", None)
    if not callable(solve):
        raise WorkerContractError(
            "solver module must define callable solve(env)"
        )
    return solve


def _write_outcome(path: Path, outcome: dict[str, Any]) -> None:
    _reject_alias(path, label="worker outcome", must_exist=False)
    if set(outcome) != {
        "schema",
        "status",
        "solver_sha256",
        "elapsed_ns",
        "error",
        "authoritative",
    }:
        raise WorkerContractError("worker outcome schema mismatch")
    if (
        outcome["schema"] != WORKER_SCHEMA
        or outcome["status"] not in {"completed", "solver_error"}
        or not isinstance(outcome["solver_sha256"], str)
        or _SHA256_RE.fullmatch(outcome["solver_sha256"]) is None
        or not isinstance(outcome["elapsed_ns"], int)
        or isinstance(outcome["elapsed_ns"], bool)
        or outcome["elapsed_ns"] < 0
        or outcome["authoritative"] is not False
    ):
        raise WorkerContractError("worker outcome contains invalid values")
    error = outcome["error"]
    if outcome["status"] == "completed":
        if error is not None:
            raise WorkerContractError(
                "completed worker outcome cannot contain an error"
            )
    elif error != {
        "type": "SolverError",
        "message": "solver execution failed",
    }:
        raise WorkerContractError("solver error outcome is malformed")
    path.parent.mkdir(parents=True, exist_ok=True)
    # Revalidate every ancestor now that missing directories have been created.
    _reject_alias(path, label="worker outcome", must_exist=False)
    try:
        parent_metadata = os.lstat(path.parent)
    except OSError as exc:
        raise WorkerContractError(
            "worker outcome parent is unavailable"
        ) from exc
    if (
        stat.S_ISLNK(parent_metadata.st_mode)
        or not stat.S_ISDIR(parent_metadata.st_mode)
    ):
        raise WorkerContractError(
            "worker outcome parent must be a regular directory"
        )
    encoded = (
        json.dumps(
            outcome,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")
    parent_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    parent_flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        parent_descriptor = os.open(path.parent, parent_flags)
    except OSError as exc:
        raise WorkerContractError(
            "worker outcome parent cannot be opened safely"
        ) from exc
    opened_parent = os.fstat(parent_descriptor)
    if (
        not stat.S_ISDIR(opened_parent.st_mode)
        or (opened_parent.st_dev, opened_parent.st_ino)
        != (parent_metadata.st_dev, parent_metadata.st_ino)
    ):
        os.close(parent_descriptor)
        raise WorkerContractError(
            "worker outcome parent changed during admission"
        )
    temporary_name = f".{path.name}.{uuid.uuid4().hex}.tmp"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(
            temporary_name,
            flags,
            0o600,
            dir_fd=parent_descriptor,
        )
    except OSError as exc:
        os.close(parent_descriptor)
        raise WorkerContractError(
            "temporary worker outcome cannot be created exclusively"
        ) from exc
    temporary_identity: tuple[int, int] | None = None
    temporary_removed = False
    final_created = False
    publication_complete = False
    descriptor_open = True
    try:
        metadata = os.fstat(descriptor)
        temporary_identity = (metadata.st_dev, metadata.st_ino)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise WorkerContractError(
                "temporary worker outcome must be an unaliased regular file"
            )
        view = memoryview(encoded)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise WorkerContractError("short write for worker outcome")
            view = view[written:]
        os.fsync(descriptor)
        descriptor_open = False
        os.close(descriptor)
        try:
            # Mark the cleanup obligation before the syscall so an asynchronous
            # exception cannot strand a successfully linked final name.
            final_created = True
            os.link(
                temporary_name,
                path.name,
                src_dir_fd=parent_descriptor,
                dst_dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        except FileExistsError as exc:
            raise WorkerContractError(
                "worker outcome path already exists"
            ) from exc
        except OSError as exc:
            raise WorkerContractError(
                "worker outcome could not be published atomically"
            ) from exc
        final_metadata = os.stat(
            path.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if (
            temporary_identity is None
            or not stat.S_ISREG(final_metadata.st_mode)
            or final_metadata.st_nlink != 2
            or (final_metadata.st_dev, final_metadata.st_ino)
            != temporary_identity
        ):
            raise WorkerContractError(
                "published worker outcome does not match trusted temporary bytes"
            )
        temporary_metadata = os.stat(
            temporary_name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if (
            temporary_metadata.st_dev,
            temporary_metadata.st_ino,
        ) != temporary_identity:
            raise WorkerContractError(
                "temporary worker outcome was replaced before publication"
            )
        os.unlink(temporary_name, dir_fd=parent_descriptor)
        temporary_removed = True
        final_metadata = os.stat(
            path.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if not stat.S_ISREG(
            final_metadata.st_mode
        ) or final_metadata.st_nlink != 1:
            raise WorkerContractError(
                "published worker outcome is aliased or non-regular"
            )
        parent_after = os.lstat(path.parent)
        if (
            parent_after.st_dev,
            parent_after.st_ino,
        ) != (opened_parent.st_dev, opened_parent.st_ino):
            raise WorkerContractError(
                "worker outcome parent changed during publication"
            )
        os.fsync(parent_descriptor)
        publication_complete = True
    finally:
        cleanup_error: BaseException | None = None
        if descriptor_open:
            try:
                os.close(descriptor)
            except BaseException as exc:
                cleanup_error = exc
        if (
            not publication_complete
            and final_created
            and temporary_identity is not None
        ):
            try:
                try:
                    failed_final = os.stat(
                        path.name,
                        dir_fd=parent_descriptor,
                        follow_symlinks=False,
                    )
                except FileNotFoundError:
                    failed_final = None
                if (
                    failed_final is not None
                    and (failed_final.st_dev, failed_final.st_ino)
                    == temporary_identity
                ):
                    os.unlink(path.name, dir_fd=parent_descriptor)
            except BaseException as exc:
                if cleanup_error is None:
                    cleanup_error = exc
        if not temporary_removed and temporary_identity is not None:
            try:
                try:
                    remaining = os.stat(
                        temporary_name,
                        dir_fd=parent_descriptor,
                        follow_symlinks=False,
                    )
                except FileNotFoundError:
                    remaining = None
                if (
                    remaining is not None
                    and (remaining.st_dev, remaining.st_ino)
                    == temporary_identity
                ):
                    os.unlink(temporary_name, dir_fd=parent_descriptor)
            except BaseException as exc:
                if cleanup_error is None:
                    cleanup_error = exc
        try:
            os.close(parent_descriptor)
        except BaseException as exc:
            if cleanup_error is None:
                cleanup_error = exc
        if cleanup_error is not None:
            raise WorkerContractError(
                "worker outcome cleanup or synchronization failed"
            ) from cleanup_error


def _sanitized_error(_exc: BaseException) -> dict[str, str]:
    # Exception messages are controlled by solver code and can contain the RPC
    # token, container paths, source excerpts, or encoded private data.  The
    # host transcript carries transport diagnostics; the exported outcome gets
    # only this fixed non-authoritative marker.
    return {
        "type": "SolverError",
        "message": "solver execution failed",
    }


def run_worker(
    config: WorkerConfig,
    *,
    client_factory: Callable[[str | os.PathLike[str], str], ArenaRpcClient]
    = ArenaRpcClient,
) -> dict[str, Any]:
    """Run one solver and write a non-authoritative diagnostic outcome."""

    token = _read_token(config.token_file)
    source = _read_regular_bytes(
        config.solve_path,
        label="solver source",
        max_bytes=MAX_SOLVER_BYTES,
    )
    source_hash = hashlib.sha256(source).hexdigest()
    monotonic_ns = time.monotonic_ns
    outcome_writer = _write_outcome
    sanitize_error = _sanitized_error
    worker_schema = WORKER_SCHEMA
    started = monotonic_ns()
    status = "completed"
    error: dict[str, str] | None = None
    client: ArenaRpcClient | None = None
    try:
        solve = _load_solver_bytes(config.solve_path, source)
        client = client_factory(config.socket_path, token)
        solve(client.root)
    except BaseException as exc:
        # The trusted host transcript retains transport details.  The exported
        # outcome contains no Python traceback, host path, or engine object.
        status = "solver_error"
        error = sanitize_error(exc)
    finally:
        if client is not None:
            try:
                client.close()
            except BaseException as exc:
                status = "solver_error"
                error = sanitize_error(exc)
        token = ""
    outcome: dict[str, Any] = {
        "schema": worker_schema,
        "status": status,
        "solver_sha256": source_hash,
        "elapsed_ns": monotonic_ns() - started,
        "error": error,
        "authoritative": False,
    }
    outcome_writer(config.outcome_path, outcome)
    return outcome


def main(argv: Sequence[str] | None = None) -> int:
    try:
        config = parse_args(list(sys.argv[1:] if argv is None else argv))
        outcome = run_worker(config)
    except SystemExit:
        raise
    except BaseException as exc:
        # CLI stderr is host-captured, but even exception types can be defined
        # by solver code and exception messages routinely contain secrets or
        # paths.  Emit only a fixed marker; full transport evidence is host-side.
        del exc
        print("WORKER_CONTRACT_ERROR", file=sys.stderr)
        return 2
    return 0 if outcome["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
