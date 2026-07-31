#!/usr/local/bin/python3
"""PID-1 guardian for the isolated ARC-AGI-3 Codex controller.

The guardian is intentionally small.  It validates the immutable in-image
supply-chain manifest, starts exactly one Codex app-server child in a private
process group, forwards termination signals only to that group, enforces the
frozen hard wall-clock ceiling, and writes diagnostic start/exit receipts to
the controller-state volume.  Container/cgroup absence observed by the trusted
host remains authoritative; these receipts are never sufficient on their own.
"""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import os
import re
import selectors
import signal
import stat
import subprocess
import sys
import time
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


HARD_SAFETY_SECONDS = 21_600
TERMINATION_GRACE_SECONDS = 10.0
CONTROL_EOF_GRACE_SECONDS = 30.0
CONTROL_WRITE_STALL_SECONDS = 30.0
MAX_PENDING_CONTROL_BYTES = 1024 * 1024
CONTROL_READ_BYTES = 64 * 1024
CONTROL_POLL_SECONDS = 0.1
STATE_ROOT = Path("/controller-state")
NEUTRAL_CWD = Path("/controller-neutral")
PIN_MANIFEST = Path(
    "/usr/local/share/arc-agi3/controller-supply-chain.json"
)
RECEIPT_DIRECTORY = Path("/run/arc-agi3-controller")
STATE_WRITE_PROBE_NAME = ".arc-agi3-controller-write-probe"
STATE_WRITE_PROBE_PAYLOAD = b"arc-agi3-controller-state-write-probe-v1\n"
EXPECTED_CHILD = (
    "/usr/local/bin/codex",
    "app-server",
    "--strict-config",
    "--listen",
    "stdio://",
)
SHA256_RE = re.compile(r"[0-9a-f]{64}")
SAFE_LABEL_RE = re.compile(r"[A-Za-z0-9_.:-]{1,200}")
MAX_MANIFEST_BYTES = 1024 * 1024
MAX_PINNED_FILE_BYTES = 512 * 1024 * 1024
MAX_PINNED_FILES = 16
NATIVE_WORKSPACE_POLICY = (
    "isolated-local-git-root-no-parent-discovery-v1"
)
NATIVE_WORKSPACE_HEAD_REF = "refs/heads/contiguous"
NATIVE_WORKSPACE_GIT_ENVIRONMENT = {
    "GIT_CEILING_DIRECTORIES": str(NEUTRAL_CWD),
    "GIT_CONFIG_GLOBAL": "/dev/null",
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_DISCOVERY_ACROSS_FILESYSTEM": "0",
    "GIT_OPTIONAL_LOCKS": "0",
}
NATIVE_WORKSPACE_FORBIDDEN_CLASSES = (
    "campaign-plan",
    "sidecar-or-quarantine-output",
    "manuscript",
    "comparator",
    "benchmark",
    "parent-repository-git-metadata",
)


class GuardianError(RuntimeError):
    """The controller cannot be started without weakening containment."""


@dataclass(frozen=True)
class PumpOutcome:
    return_code: int
    hard_safety_expired: bool
    control_eof_observed: bool
    control_fault: str | None
    pending_input_peak_bytes: int


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _sha256_file(path: Path, *, maximum: int) -> tuple[str, int]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or metadata.st_size < 1
            or metadata.st_size > maximum
        ):
            raise GuardianError("pinned file is not one bounded regular inode")
        digest = hashlib.sha256()
        observed = 0
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            observed += len(block)
            if observed > maximum:
                raise GuardianError("pinned file exceeds its byte ceiling")
            digest.update(block)
        if observed != metadata.st_size:
            raise GuardianError("pinned file changed during hashing")
        after = os.fstat(descriptor)
        if (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ) != (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_size,
            metadata.st_mtime_ns,
        ):
            raise GuardianError("pinned file identity changed during hashing")
        return digest.hexdigest(), observed
    finally:
        os.close(descriptor)


def _load_pin_manifest(path: Path = PIN_MANIFEST) -> dict[str, Any]:
    raw_digest, raw_bytes = _sha256_file(path, maximum=MAX_MANIFEST_BYTES)
    descriptor = os.open(
        path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        raw = b""
        while len(raw) < raw_bytes:
            block = os.read(descriptor, raw_bytes - len(raw))
            if not block:
                break
            raw += block
    finally:
        os.close(descriptor)
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GuardianError("supply-chain manifest is not strict JSON") from exc
    if (
        not isinstance(value, dict)
        or set(value)
        != {"schema", "kind", "codex_cli_version", "files"}
        or value["schema"] != 1
        or value["kind"] != "arc_agi3_controller_supply_chain"
        or not isinstance(value["codex_cli_version"], str)
        or re.fullmatch(
            r"codex-cli [0-9]+(?:\.[0-9]+){2}",
            value["codex_cli_version"],
        )
        is None
        or not isinstance(value["files"], list)
        or not 1 <= len(value["files"]) <= MAX_PINNED_FILES
        or _canonical_json(value) + b"\n" != raw
    ):
        raise GuardianError("supply-chain manifest schema is not exact")
    value["_manifest_sha256"] = raw_digest
    return value


def _validate_supply_chain(
    path: Path = PIN_MANIFEST,
) -> tuple[str, tuple[dict[str, Any], ...]]:
    manifest = _load_pin_manifest(path)
    observations: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in manifest["files"]:
        if (
            not isinstance(item, dict)
            or set(item) != {"path", "sha256", "bytes", "executable"}
            or not isinstance(item["path"], str)
            or not Path(item["path"]).is_absolute()
            or "\x00" in item["path"]
            or not isinstance(item["sha256"], str)
            or SHA256_RE.fullmatch(item["sha256"]) is None
            or isinstance(item["bytes"], bool)
            or not isinstance(item["bytes"], int)
            or not 1 <= item["bytes"] <= MAX_PINNED_FILE_BYTES
            or not isinstance(item["executable"], bool)
            or item["path"] in seen
        ):
            raise GuardianError("supply-chain file entry is malformed")
        seen.add(item["path"])
        file_path = Path(item["path"])
        digest, byte_count = _sha256_file(
            file_path, maximum=MAX_PINNED_FILE_BYTES
        )
        mode = stat.S_IMODE(file_path.stat(follow_symlinks=False).st_mode)
        if (
            digest != item["sha256"]
            or byte_count != item["bytes"]
            or bool(mode & 0o111) is not item["executable"]
            or mode & 0o022
        ):
            raise GuardianError("in-image supply-chain pin differs")
        observations.append(
            {
                "path": item["path"],
                "sha256": digest,
                "bytes": byte_count,
                "executable": item["executable"],
            }
        )
    if EXPECTED_CHILD[0] not in seen:
        raise GuardianError("manifest omits the exact Codex child")
    return str(manifest["_manifest_sha256"]), tuple(observations)


def _write_new_receipt(path: Path, value: Mapping[str, Any]) -> str:
    payload = _canonical_json(value) + b"\n"
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor = os.open(
        path,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    directory = os.open(
        path.parent,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return hashlib.sha256(payload).hexdigest()


def _validate_runtime_roots(
    state_root: Path = STATE_ROOT,
    neutral_cwd: Path = NEUTRAL_CWD,
) -> None:
    for label, path in (
        ("controller state", state_root),
        ("neutral cwd", neutral_cwd),
    ):
        metadata = path.lstat()
        if (
            not path.is_absolute()
            or not stat.S_ISDIR(metadata.st_mode)
            or stat.S_ISLNK(metadata.st_mode)
            or metadata.st_uid != os.getuid()
            or stat.S_IMODE(metadata.st_mode) & 0o077
        ):
            raise GuardianError(f"{label} root is not private")
    if any(neutral_cwd.iterdir()):
        raise GuardianError("neutral cwd is not empty")
    if os.getuid() == 0 or os.getgid() == 0:
        raise GuardianError("controller guardian must be nonroot")


def _probe_state_root_write(state_root: Path) -> dict[str, Any]:
    """Exercise the bind mount under the exact controller UID before Codex."""

    path = state_root / STATE_WRITE_PROBE_NAME
    descriptor = -1
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        view = memoryview(STATE_WRITE_PROBE_PAYLOAD)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise GuardianError(
                    "controller state write probe made no progress"
                )
            view = view[written:]
        os.fsync(descriptor)
    except OSError as exc:
        raise GuardianError(
            "controller state root is not writable by its runtime identity"
        ) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    try:
        metadata = path.stat(follow_symlinks=False)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.getuid()
            or metadata.st_gid != os.getgid()
            or metadata.st_nlink != 1
            or metadata.st_size != len(STATE_WRITE_PROBE_PAYLOAD)
            or path.read_bytes() != STATE_WRITE_PROBE_PAYLOAD
        ):
            raise GuardianError(
                "controller state write probe identity changed"
            )
        path.unlink()
        directory = os.open(
            state_root,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except OSError as exc:
        raise GuardianError(
            "controller state write probe cannot be removed durably"
        ) from exc
    if path.exists() or path.is_symlink():
        raise GuardianError("controller state write probe remained present")
    return {
        "schema": 1,
        "kind": "controller_state_root_write_probe",
        "runtime_uid": os.getuid(),
        "runtime_gid": os.getgid(),
        "relative_path": STATE_WRITE_PROBE_NAME,
        "payload_sha256": hashlib.sha256(
            STATE_WRITE_PROBE_PAYLOAD
        ).hexdigest(),
        "payload_bytes": len(STATE_WRITE_PROBE_PAYLOAD),
        "file_fsync": True,
        "directory_fsync_after_unlink": True,
        "probe_absent_after_fsync": True,
        "status": "PASS",
    }


def _git_object(kind: str, payload: bytes) -> tuple[str, bytes]:
    raw = kind.encode("ascii") + b" " + str(len(payload)).encode("ascii")
    raw += b"\0" + payload
    return (
        hashlib.sha1(raw, usedforsecurity=False).hexdigest(),
        zlib.compress(raw, level=9),
    )


def _native_workspace_files() -> dict[str, bytes]:
    empty_tree_sha1, empty_tree = _git_object("tree", b"")
    commit_payload = (
        f"tree {empty_tree_sha1}\n"
        "author ARC-AGI-3 Contiguous "
        "<contiguous@invalid> 0 +0000\n"
        "committer ARC-AGI-3 Contiguous "
        "<contiguous@invalid> 0 +0000\n"
        "\n"
        "isolated zero-source proposer root\n"
    ).encode("ascii")
    commit_sha1, commit = _git_object("commit", commit_payload)
    config = (
        "[core]\n"
        "\trepositoryformatversion = 0\n"
        "\tfilemode = true\n"
        "\tbare = false\n"
        "\tsymlinks = false\n"
        "\tlogallrefupdates = false\n"
    ).encode("ascii")
    return {
        ".git/HEAD": (
            f"ref: {NATIVE_WORKSPACE_HEAD_REF}\n"
        ).encode("ascii"),
        ".git/config": config,
        ".git/description": b"ARC-AGI-3 isolated proposer workspace\n",
        f".git/objects/{empty_tree_sha1[:2]}/{empty_tree_sha1[2:]}":
            empty_tree,
        f".git/objects/{commit_sha1[:2]}/{commit_sha1[2:]}": commit,
        f".git/{NATIVE_WORKSPACE_HEAD_REF}":
            (commit_sha1 + "\n").encode("ascii"),
    }


def _native_workspace_directories(
    files: Mapping[str, bytes],
) -> tuple[str, ...]:
    directories = {".git", ".git/objects", ".git/refs", ".git/refs/heads"}
    for relative in files:
        parent = Path(relative).parent
        while str(parent) not in {".", ""}:
            directories.add(parent.as_posix())
            parent = parent.parent
    return tuple(sorted(directories, key=lambda value: (value.count("/"), value)))


def _write_workspace_file(path: Path, payload: bytes) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0),
        0o400,
    )
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise GuardianError(
                    "native proposer workspace write made no progress"
                )
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _validate_native_workspace(
    workspace: Path,
) -> dict[str, Any]:
    """Prove the Codex cwd is one exact local Git root and nothing else."""

    expected_files = _native_workspace_files()
    expected_directories = _native_workspace_directories(expected_files)
    root_metadata = workspace.lstat()
    if (
        not workspace.is_absolute()
        or not stat.S_ISDIR(root_metadata.st_mode)
        or stat.S_ISLNK(root_metadata.st_mode)
        or root_metadata.st_uid != os.getuid()
        or stat.S_IMODE(root_metadata.st_mode) & 0o077
    ):
        raise GuardianError(
            "native proposer workspace root is not private"
        )
    observed_files: dict[str, dict[str, Any]] = {}
    observed_directories: set[str] = set()
    for current, directory_names, file_names in os.walk(
        workspace, topdown=True, followlinks=False
    ):
        current_path = Path(current)
        current_relative = current_path.relative_to(workspace)
        for name in sorted(directory_names):
            selected = current_path / name
            metadata = selected.lstat()
            relative = (
                current_relative / name
            ).as_posix().removeprefix("./")
            if (
                stat.S_ISLNK(metadata.st_mode)
                or not stat.S_ISDIR(metadata.st_mode)
                or metadata.st_dev != root_metadata.st_dev
                or stat.S_IMODE(metadata.st_mode) & 0o077
            ):
                raise GuardianError(
                    "native proposer workspace directory escapes or aliases"
                )
            observed_directories.add(relative)
        for name in sorted(file_names):
            selected = current_path / name
            metadata = selected.lstat()
            relative = (
                current_relative / name
            ).as_posix().removeprefix("./")
            if (
                stat.S_ISLNK(metadata.st_mode)
                or not stat.S_ISREG(metadata.st_mode)
                or metadata.st_nlink != 1
                or metadata.st_dev != root_metadata.st_dev
                or relative not in expected_files
            ):
                raise GuardianError(
                    "native proposer workspace file escapes or aliases"
                )
            digest, byte_count = _sha256_file(
                selected, maximum=MAX_MANIFEST_BYTES
            )
            if selected.read_bytes() != expected_files[relative]:
                raise GuardianError(
                    "native proposer workspace bytes changed"
                )
            observed_files[relative] = {
                "path": relative,
                "sha256": digest,
                "bytes": byte_count,
                "links": metadata.st_nlink,
            }
    if (
        observed_directories != set(expected_directories)
        or set(observed_files) != set(expected_files)
    ):
        raise GuardianError(
            "native proposer workspace allowlist differs"
        )
    head = expected_files[
        f".git/{NATIVE_WORKSPACE_HEAD_REF}"
    ].decode("ascii").strip()
    inventory = [
        observed_files[path] for path in sorted(observed_files)
    ]
    inventory_sha256 = hashlib.sha256(
        _canonical_json({"files": inventory})
    ).hexdigest()
    return {
        "policy": NATIVE_WORKSPACE_POLICY,
        "workspace_root": str(workspace),
        "git_dir": str(workspace / ".git"),
        "git_root_equals_workspace": True,
        "head_ref": NATIVE_WORKSPACE_HEAD_REF,
        "head_commit": head,
        "file_count": len(inventory),
        "inventory_sha256": inventory_sha256,
        "symlink_count": 0,
        "hardlink_count": 0,
        "path_escape_count": 0,
        "forbidden_classes_absent":
            list(NATIVE_WORKSPACE_FORBIDDEN_CLASSES),
        "git_ceiling_directories": str(workspace),
        "git_discovery_across_filesystem": False,
        "git_global_config_disabled": True,
        "git_system_config_disabled": True,
    }


def _initialize_native_workspace(
    workspace: Path,
) -> dict[str, Any]:
    if any(workspace.iterdir()):
        raise GuardianError(
            "native proposer workspace was not initially empty"
        )
    files = _native_workspace_files()
    for relative in _native_workspace_directories(files):
        selected = workspace.joinpath(*Path(relative).parts)
        selected.mkdir(mode=0o700)
    for relative, payload in sorted(files.items()):
        selected = workspace.joinpath(*Path(relative).parts)
        _write_workspace_file(selected, payload)
    descriptor = os.open(
        workspace,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return _validate_native_workspace(workspace)


def _enable_child_subreaper() -> None:
    if sys.platform != "linux":
        raise GuardianError("controller guardian requires Linux")
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
        error = ctypes.get_errno()
        raise GuardianError(
            f"cannot enable child subreaper: errno={error}"
        )


def _signal_group(process_group_id: int, number: int) -> None:
    try:
        os.killpg(process_group_id, number)
    except ProcessLookupError:
        pass


def _reap_adopted_children(deadline: float) -> None:
    while time.monotonic() < deadline:
        try:
            pid, _ = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            return
        if pid == 0:
            time.sleep(0.01)


def _terminate_child_group(
    child: subprocess.Popen[bytes],
    child_group: int,
) -> None:
    _signal_group(child_group, signal.SIGTERM)
    try:
        child.wait(timeout=TERMINATION_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        _signal_group(child_group, signal.SIGKILL)


def _pump_control_input(
    child: subprocess.Popen[bytes],
    *,
    child_group: int,
    input_descriptor: int,
    deadline: float,
) -> PumpOutcome:
    """Bounded, nonblocking parent-input → app-server-stdin pump."""

    if child.stdin is None:
        raise GuardianError("Codex child stdin pipe is unavailable")
    child_input_descriptor = child.stdin.fileno()
    os.set_blocking(input_descriptor, False)
    os.set_blocking(child_input_descriptor, False)
    selector = selectors.DefaultSelector()
    selector.register(
        input_descriptor, selectors.EVENT_READ, "controller_input"
    )
    child_registered = False
    pending = bytearray()
    pending_peak = 0
    last_write_progress = time.monotonic()
    hard_expired = False
    control_eof = False
    control_eof_deadline: float | None = None
    control_fault: str | None = None

    def register_child_writer() -> None:
        nonlocal child_registered
        if not child_registered:
            selector.register(
                child_input_descriptor,
                selectors.EVENT_WRITE,
                "child_input",
            )
            child_registered = True

    def unregister_child_writer() -> None:
        nonlocal child_registered
        if child_registered:
            try:
                selector.unregister(child_input_descriptor)
            except KeyError:
                pass
            child_registered = False

    def close_child_input_after_drain() -> None:
        if (
            control_eof
            and not pending
            and child.stdin is not None
            and not child.stdin.closed
        ):
            unregister_child_writer()
            child.stdin.close()

    try:
        while child.poll() is None:
            now = time.monotonic()
            remaining = deadline - now
            if remaining <= 0:
                hard_expired = True
                _terminate_child_group(child, child_group)
                break
            if (
                control_eof_deadline is not None
                and now >= control_eof_deadline
            ):
                control_fault = "control_eof_timeout"
                _terminate_child_group(child, child_group)
                break
            if (
                pending
                and now - last_write_progress
                >= CONTROL_WRITE_STALL_SECONDS
            ):
                control_fault = "child_stdin_stall"
                _terminate_child_group(child, child_group)
                break

            timeout = min(CONTROL_POLL_SECONDS, remaining)
            if control_eof_deadline is not None:
                timeout = min(
                    timeout, max(0.0, control_eof_deadline - now)
                )
            if pending:
                timeout = min(
                    timeout,
                    max(
                        0.0,
                        CONTROL_WRITE_STALL_SECONDS
                        - (now - last_write_progress),
                    ),
                )
            if selector.get_map():
                ready = selector.select(timeout)
            else:
                time.sleep(timeout)
                ready = []

            for key, mask in ready:
                if (
                    key.data == "controller_input"
                    and mask & selectors.EVENT_READ
                ):
                    try:
                        incoming = os.read(
                            input_descriptor, CONTROL_READ_BYTES
                        )
                    except BlockingIOError:
                        incoming = None
                    except OSError:
                        control_fault = "controller_input_read_error"
                        break
                    if incoming == b"":
                        control_eof = True
                        try:
                            selector.unregister(input_descriptor)
                        except KeyError:
                            pass
                        control_eof_deadline = (
                            time.monotonic()
                            + CONTROL_EOF_GRACE_SECONDS
                        )
                        close_child_input_after_drain()
                    elif incoming:
                        if (
                            len(pending) + len(incoming)
                            > MAX_PENDING_CONTROL_BYTES
                        ):
                            control_fault = "child_stdin_buffer_overflow"
                            break
                        was_empty = not pending
                        pending.extend(incoming)
                        pending_peak = max(
                            pending_peak, len(pending)
                        )
                        if was_empty:
                            last_write_progress = time.monotonic()
                        register_child_writer()
                elif (
                    key.data == "child_input"
                    and mask & selectors.EVENT_WRITE
                    and pending
                ):
                    try:
                        written = os.write(
                            child_input_descriptor, pending
                        )
                    except BlockingIOError:
                        written = 0
                    except BrokenPipeError:
                        control_fault = "child_stdin_epipe"
                        break
                    except OSError as exc:
                        if exc.errno == errno.EPIPE:
                            control_fault = "child_stdin_epipe"
                        else:
                            control_fault = "child_stdin_write_error"
                        break
                    if written < 0 or written > len(pending):
                        control_fault = "child_stdin_invalid_write"
                        break
                    if written:
                        del pending[:written]
                        last_write_progress = time.monotonic()
                    if not pending:
                        unregister_child_writer()
                        close_child_input_after_drain()
            if control_fault is not None:
                _terminate_child_group(child, child_group)
                break
        return_code = int(
            child.wait(timeout=TERMINATION_GRACE_SECONDS)
        )
    finally:
        selector.close()
    return PumpOutcome(
        return_code=return_code,
        hard_safety_expired=hard_expired,
        control_eof_observed=control_eof,
        control_fault=control_fault,
        pending_input_peak_bytes=pending_peak,
    )


def run_guarded(
    command: Sequence[str],
    *,
    state_root: Path = STATE_ROOT,
    neutral_cwd: Path = NEUTRAL_CWD,
    pin_manifest: Path = PIN_MANIFEST,
    hard_safety_seconds: int = HARD_SAFETY_SECONDS,
) -> int:
    if (
        tuple(command) != EXPECTED_CHILD
        or isinstance(hard_safety_seconds, bool)
        or hard_safety_seconds != HARD_SAFETY_SECONDS
    ):
        raise GuardianError("guardian command or hard bound differs")
    _validate_runtime_roots(state_root, neutral_cwd)
    _enable_child_subreaper()
    manifest_sha256, file_observations = _validate_supply_chain(
        pin_manifest
    )
    state_root_write_probe = _probe_state_root_write(state_root)
    native_workspace = _initialize_native_workspace(neutral_cwd)
    minimal_environment = {
        "CODEX_HOME": str(state_root),
        "HOME": str(state_root),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "TMPDIR": "/tmp",
        **{
            name: (
                str(neutral_cwd)
                if name == "GIT_CEILING_DIRECTORIES"
                else value
            )
            for name, value in NATIVE_WORKSPACE_GIT_ENVIRONMENT.items()
        },
    }
    child = subprocess.Popen(
        tuple(command),
        cwd=neutral_cwd,
        env=minimal_environment,
        stdin=subprocess.PIPE,
        stdout=None,
        stderr=None,
        start_new_session=True,
        close_fds=True,
    )
    if child.stdin is None:
        raise GuardianError("Codex child stdin pipe is unavailable")
    child_group = os.getpgid(child.pid)
    if child_group != child.pid:
        _signal_group(child.pid, signal.SIGKILL)
        child.wait(timeout=5)
        raise GuardianError("Codex child lacks a private process group")

    forwarded: list[int] = []

    def forward(number: int, _frame: object) -> None:
        forwarded.append(number)
        _signal_group(child_group, number)

    prior_handlers: dict[int, Any] = {}
    for number in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        prior_handlers[number] = signal.signal(number, forward)
    started_monotonic_ns = time.monotonic_ns()
    start_receipt = {
        "schema": 1,
        "kind": "arc_agi3_controller_guardian_start",
        "guardian_pid": os.getpid(),
        "child_pid": child.pid,
        "child_process_group_id": child_group,
        "hard_safety_seconds": hard_safety_seconds,
        "supply_chain_manifest_sha256": manifest_sha256,
        "supply_chain_files": list(file_observations),
        "state_root_write_probe": state_root_write_probe,
        "native_workspace": native_workspace,
        "started_monotonic_ns": started_monotonic_ns,
    }
    _write_new_receipt(
        RECEIPT_DIRECTORY / "process_start.json",
        start_receipt,
    )
    deadline = time.monotonic() + hard_safety_seconds
    input_descriptor = sys.stdin.fileno()
    try:
        pump = _pump_control_input(
            child,
            child_group=child_group,
            input_descriptor=input_descriptor,
            deadline=deadline,
        )
    finally:
        if child.poll() is None:
            _signal_group(child_group, signal.SIGKILL)
            child.wait(timeout=TERMINATION_GRACE_SECONDS)
        _signal_group(child_group, signal.SIGKILL)
        _reap_adopted_children(
            time.monotonic() + TERMINATION_GRACE_SECONDS
        )
        for number, handler in prior_handlers.items():
            signal.signal(number, handler)
    exit_receipt = {
        "schema": 1,
        "kind": "arc_agi3_controller_guardian_exit",
        "guardian_pid": os.getpid(),
        "child_pid": child.pid,
        "child_process_group_id": child_group,
        "return_code": pump.return_code,
        "hard_safety_expired": pump.hard_safety_expired,
        "control_eof_observed": pump.control_eof_observed,
        "control_fault": pump.control_fault,
        "pending_input_peak_bytes": pump.pending_input_peak_bytes,
        "forwarded_signals": forwarded,
        "started_monotonic_ns": started_monotonic_ns,
        "finished_monotonic_ns": time.monotonic_ns(),
        "supply_chain_manifest_sha256": manifest_sha256,
    }
    _write_new_receipt(
        RECEIPT_DIRECTORY / "process_exit.json",
        exit_receipt,
    )
    if pump.hard_safety_expired:
        return 124
    if pump.control_fault is not None:
        return 70
    return pump.return_code


def main(argv: Sequence[str] | None = None) -> int:
    arguments = tuple(sys.argv[1:] if argv is None else argv)
    try:
        return run_guarded(arguments)
    except (GuardianError, OSError, subprocess.SubprocessError) as exc:
        # This is a pre-app-server containment diagnostic.  The host treats
        # any stderr as a failed preflight and never exposes it to a proposer.
        print(f"arc-agi3 controller guardian: {exc}", file=sys.stderr)
        return 70


if __name__ == "__main__":
    raise SystemExit(main())
