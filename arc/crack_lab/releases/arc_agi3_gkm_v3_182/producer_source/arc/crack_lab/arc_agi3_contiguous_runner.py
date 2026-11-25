#!/usr/bin/env python3
"""Durable scheduler core for an isolated contiguous ARC-AGI-3 campaign.

This module is deliberately backend-agnostic.  It owns scheduling, lineage
admission, effort escalation, WIP eligibility, exact-edge promotion admission,
and crash recovery.  An OCI adapter must implement :class:`AttemptBackend`;
trusted host promotion must implement :class:`PromotionGate`.

The scheduler never interprets a soft allocation as permission to signal a
running proposer.  Once an allocation expires that game lane enters DRAINING:
the turn may finish naturally and the same game may not overlap, while unrelated
games remain dispatchable.  Hard containment and teardown belong to the
backend's container/cgroup boundary.
"""

from __future__ import annotations

import copy
import errno
import fcntl
import hashlib
import hmac
import json
import math
import os
import re
import shutil
import stat
import time
import uuid
from dataclasses import asdict, dataclass, is_dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Literal, Mapping, Protocol

import arc_agi3_contiguous_supervisor as Contract
import arc_agi3_codex_app_server_transport as Transport
import arc_agi3_source_schema as SourceSchema
import arc_agi3_contiguous_taint as Taint
import arc_agi3_contiguous_scheduler as Scheduler


JOURNAL_SCHEMA = 1
RUNNER_SCHEMA = 1
MAX_LANES = 6
JOURNAL_SEGMENT_EVENT_LIMIT = 256
MIN_JOURNAL_FILESYSTEM_FREE_BYTES = 64 * 1024 * 1024
MIN_JOURNAL_FILESYSTEM_FREE_INODES = 256
JOURNAL_EMERGENCY_RESERVE_BYTES = 64 * 1024
JOURNAL_QUIESCENCE_RESERVE_BYTES = 1024 * 1024
# The runtime is deliberately not a release authorization.  The independent
# image-level release receipt must flip its own gate after the backend adapter,
# taint suite, and exact replay checks all pass.
CONTIGUOUS_RUNNER_LAUNCH_READY = False
# No production auxiliary adapter is released by this module.  Tests and future
# adapters can exercise the abstract contract only when the durable campaign
# manifest independently attests every isolation/input/admission boundary.
CONTIGUOUS_AUXILIARY_LAUNCH_READY = False
POLL_TIMEOUT_SECONDS = 5.0
MAX_HOST_TRANSCRIPT_BYTES = 64 * 1024 * 1024
MAX_AUXILIARY_RECEIPT_BYTES = 32 * 1024 * 1024
MAX_PARENT_SOURCE_FILES = SourceSchema.MAX_FILES
MAX_PARENT_SOURCE_FILE_BYTES = SourceSchema.MAX_FILE_BYTES
MAX_PARENT_SOURCE_TOTAL_BYTES = SourceSchema.MAX_TOTAL_BYTES
MAX_APP_SERVER_STATE_FILES = 8192
MAX_APP_SERVER_STATE_FILE_BYTES = 320 * 1024 * 1024
MAX_APP_SERVER_STATE_TOTAL_BYTES = 2 * 1024 * 1024 * 1024
PARENT_SOURCE_REQUIRED_FILES = SourceSchema.REQUIRED_FILES
WORKER_OUTCOME_NAME = "worker_outcome.json"
ARENA_VOLUME_TRANSPORT = "docker-attach-stdio+named-volume-unix"
EXPECTED_WORKER_COMMAND = (
    "-I",
    "-m",
    "arc_agi3_proposer_worker",
    "--bridge-socket=/run/arc-agi3/proposer.sock",
    "--bridge-token-file=/run/arc-agi3/proposer-token",
    "--bridge-policy=/arc/input/bridge_policy.json",
    "--arena-socket=/arena/arena.sock",
    "--arena-token-file=/run/arc-agi3/token",
    "--workspace=/arc/workspace",
    "--export=/arc/export",
)
EXPECTED_CONTROLLER_PREFLIGHT_REQUEST_ALLOWLIST = (
    Transport.PREFLIGHT_REQUEST_SEQUENCE
)
EXPECTED_CONTROLLER_PREFLIGHT_NOTIFICATION_ALLOWLIST = tuple(
    Transport.PREFLIGHT_NOTIFICATION_CARDINALITY
)
EXPECTED_CONTROLLER_TURN_REQUEST_ALLOWLIST = (
    Transport.TURN_REQUEST_SEQUENCE
)
EXPECTED_DYNAMIC_TOOL_NAMESPACE = "contiguous_lane"
EXPECTED_CONTROLLER_ENTRYPOINT = (
    "/usr/local/bin/arc-agi3-contiguous-controller-guardian",
)
EXPECTED_CONTROLLER_IMAGE_USER = "65532:65532"
# The runtime identity deliberately matches the private state-tree owner.
# This keeps the bind-mounted state writable on native Linux without making
# it group/world-accessible or requiring a privileged chown helper.
EXPECTED_CONTROLLER_USER = f"{os.getuid()}:{os.getgid()}"
EXPECTED_CONTROLLER_EGRESS_POLICY = "openai_https_only"
EXPECTED_REASONING_EFFORT_ALLOWLIST = (
    "medium",
    "high",
    "xhigh",
    "max",
)
EXPECTED_DYNAMIC_TOOL_NAMES = Transport.DYNAMIC_TOOL_NAMES
EXPECTED_BRIDGE_OPERATION_ALLOWLIST = (
    Transport.BRIDGE_OPERATION_ALLOWLIST
)
EXPECTED_BRIDGE_EXEC_ALLOWLIST = Transport.BRIDGE_EXEC_ALLOWLIST
SHA256_RE = re.compile(r"[0-9a-f]{64}")
EVENT_ID_RE = re.compile(r"[A-Za-z0-9_.:-]{1,200}")
RESULT_KINDS = set(Scheduler.TERMINAL_RESULT_KINDS)
EXACT_FRONTIER_COORDINATE_OUTCOMES = frozenset(
    (
        *Scheduler.TERMINAL_RESULT_KINDS,
        *Scheduler.NONCOUNTING_RUNTIME_OUTCOMES,
    )
)
CANONICAL_BLANK_SCAFFOLD_TREE_SHA256 = (
    "17642e5756772eb723a2a152d67f9a939401f1306c6a4ecf628b6833e7f7f0df"
)
SCHEDULER_POLICY_SHA256 = Scheduler.SCHEDULER_POLICY_SHA256
HOST_BLOCKER_CODES = Scheduler.HOST_BLOCKER_CODES
HOST_BLOCKER_RECEIPT_KIND = Scheduler.HOST_BLOCKER_RECEIPT_KIND
HOST_BLOCKER_RECEIPT_NAME = Scheduler.HOST_BLOCKER_RECEIPT_NAME
HOST_BLOCKER_REASON_PREFIX = Scheduler.HOST_BLOCKER_REASON_PREFIX
HOST_BLOCKER_AUTHORITY = Scheduler.HOST_BLOCKER_AUTHORITY
TERMINAL_RETENTION_INTENT_NAME = "terminal_retention_intent.json"
TERMINAL_RETENTION_RECEIPT_NAME = "terminal_retention_receipt.json"
TERMINAL_RETENTION_EVIDENCE_NAME = "terminal_attempt_evidence"
TERMINAL_RETENTION_SCHEMA = 1
SUBSTRATE_REPROBE_AUTHORIZATION_ROOT = (
    "substrate_reprobe_authorizations"
)
META_SUBSTRATE_RECOVERY_AUTHORIZATION_ROOT = (
    "meta_substrate_recovery_authorizations"
)
META_SUBSTRATE_RECOVERY_RECOMMENDATION = (
    "REMATERIALIZE_AND_REPROBE_CONTROLLER_SUBSTRATE"
)
MAX_TERMINAL_COMPACT_EVIDENCE_BYTES = 64 * 1024 * 1024
MAX_TERMINAL_COMPACT_EVIDENCE_TOTAL_BYTES = 4 * 1024 * 1024 * 1024
OPERATION_RETRY_BACKOFF_SECONDS = (
    Scheduler.OPERATION_RETRY_BACKOFF_SECONDS
)
SUBSTRATE_HEALTH_REPROBE_BACKOFF_SECONDS = (
    Scheduler.SUBSTRATE_HEALTH_REPROBE_BACKOFF_SECONDS
)
FAILURE_CIRCUIT_THRESHOLD = Scheduler.FAILURE_CIRCUIT_THRESHOLD
FAILURE_FAULT_DOMAINS = frozenset(
    Scheduler.FAILURE_FAULT_DOMAINS
)


class ContiguousRunnerError(RuntimeError):
    """A fail-closed scheduler, journal, or backend-contract error."""


class JournalStorageExhausted(ContiguousRunnerError):
    """Typed pre-commit signal backed by a filesystem admission snapshot."""

    def __init__(
        self,
        *,
        failed_event_id: str,
        failed_event_kind: str,
        failure_stage: str,
        error_code: str,
        storage_snapshot: Mapping[str, Any],
    ) -> None:
        super().__init__("journal_or_storage_exhausted")
        self.failed_event_id = failed_event_id
        self.failed_event_kind = failed_event_kind
        self.failure_stage = failure_stage
        self.error_code = error_code
        self.storage_snapshot = dict(storage_snapshot)


def _storage_error_code(error: BaseException) -> str | None:
    current: BaseException | None = error
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, OSError):
            if current.errno == errno.ENOSPC:
                return "ENOSPC"
            if current.errno == getattr(errno, "EDQUOT", errno.ENOSPC):
                return "EDQUOT"
        current = (
            current.__cause__
            if current.__cause__ is not None
            else current.__context__
        )
    return None


class SimulatedCrash(BaseException):
    """Fault-injection exception intentionally not swallowed by the runner."""


class PromotionRejected(ContiguousRunnerError):
    """Trusted gate rejection proven to precede any publication side effect."""


class AuxiliaryBackendFatalError(RuntimeError):
    """Backend-declared terminal sidecar failure requiring durable abort."""


class BackendPublicActionProtocolInvalidError(RuntimeError):
    """Typed terminal signal for one protocol-invalid public action."""

    def __init__(
        self,
        *,
        receipt_path: str,
        receipt_sha256: str,
        controller_state_scan_receipt_path: str,
        controller_state_scan_receipt_sha256: str,
        retained_canary_scan_receipt_path: str,
        retained_canary_scan_receipt_sha256: str,
        partial_taint_scan_receipt_path: str,
        partial_taint_scan_receipt_sha256: str,
        partial_usage_receipt_path: str,
        partial_usage_receipt_sha256: str,
        cost_used: float,
    ) -> None:
        super().__init__("public_action_protocol_invalid")
        self.receipt_path = receipt_path
        self.receipt_sha256 = receipt_sha256
        self.controller_state_scan_receipt_path = (
            controller_state_scan_receipt_path
        )
        self.controller_state_scan_receipt_sha256 = (
            controller_state_scan_receipt_sha256
        )
        self.retained_canary_scan_receipt_path = (
            retained_canary_scan_receipt_path
        )
        self.retained_canary_scan_receipt_sha256 = (
            retained_canary_scan_receipt_sha256
        )
        self.partial_taint_scan_receipt_path = (
            partial_taint_scan_receipt_path
        )
        self.partial_taint_scan_receipt_sha256 = (
            partial_taint_scan_receipt_sha256
        )
        self.partial_usage_receipt_path = partial_usage_receipt_path
        self.partial_usage_receipt_sha256 = partial_usage_receipt_sha256
        self.cost_used = cost_used


class BackendSubstratePreflightError(RuntimeError):
    """One reconciled pre-turn controller-substrate infrastructure failure."""

    def __init__(
        self,
        *,
        substrate_identity_sha256: str,
        failure_receipt_path: str,
        failure_receipt_sha256: str,
    ) -> None:
        if (
            not _is_sha256(substrate_identity_sha256)
            or not _safe_path_string(failure_receipt_path)
            or not Path(failure_receipt_path).is_absolute()
            or not _is_sha256(failure_receipt_sha256)
        ):
            raise ContiguousRunnerError(
                "substrate preflight failure evidence is malformed"
            )
        super().__init__(
            "controller substrate preflight failed before proposer launch"
        )
        self.substrate_identity_sha256 = substrate_identity_sha256
        self.failure_receipt_path = failure_receipt_path
        self.failure_receipt_sha256 = failure_receipt_sha256


# Internal compatibility alias for adapters built against the pre-conformance
# name.  No public journal event, receipt, or terminal result uses "poison".
BackendProtocolPoisonError = BackendPublicActionProtocolInvalidError


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def meta_substrate_recovery_authentication_sha256(
    receipt_body: Mapping[str, Any],
    *,
    operator_configuration_sha256: str,
) -> str:
    """Bind one meta recommendation to the sealed controller configuration."""

    if (
        not isinstance(receipt_body, Mapping)
        or "authorization_authentication_sha256" in receipt_body
        or not _is_sha256(operator_configuration_sha256)
        or receipt_body.get("operator_configuration_sha256")
        != operator_configuration_sha256
    ):
        raise ContiguousRunnerError(
            "meta substrate authorization body is malformed"
        )
    return hashlib.sha256(
        b"arc-agi3-meta-substrate-recovery-auth-v1\0"
        + operator_configuration_sha256.encode("ascii")
        + b"\0"
        + _canonical_json(dict(receipt_body))
    ).hexdigest()


def substrate_incident_identity_sha256(
    *,
    campaign_id: str,
    attempt_id: str,
    game: str,
    frontier_sha256: str,
    substrate_identity_sha256: str,
    failure_receipt_sha256: str,
    failure_class: str,
    failure_code: str,
) -> str:
    """Stable identity for one incident, independent of later journal heads."""

    body = {
        "schema": 1,
        "kind": "contiguous_controller_substrate_incident",
        "campaign_id": campaign_id,
        "attempt_id": attempt_id,
        "game": game,
        "frontier_sha256": frontier_sha256,
        "substrate_identity_sha256": substrate_identity_sha256,
        "failure_receipt_sha256": failure_receipt_sha256,
        "failure_class": failure_class,
        "failure_code": failure_code,
    }
    if (
        not _safe_identifier(campaign_id)
        or not _is_canonical_uuid(attempt_id)
        or not _safe_identifier(game)
        or any(
            not _is_sha256(body[name])
            for name in (
                "frontier_sha256",
                "substrate_identity_sha256",
                "failure_receipt_sha256",
            )
        )
        or failure_class not in {
            "DETERMINISTIC_CONFIGURATION",
            "TRANSIENT_INFRASTRUCTURE",
        }
        or not _safe_identifier(failure_code)
    ):
        raise ContiguousRunnerError(
            "substrate incident identity input is malformed"
        )
    return hashlib.sha256(
        b"arc-agi3-controller-substrate-incident-v1\0"
        + _canonical_json(body)
    ).hexdigest()


def meta_substrate_resume_authentication_sha256(
    receipt_body: Mapping[str, Any],
    *,
    operator_configuration_sha256: str,
) -> str:
    if (
        not isinstance(receipt_body, Mapping)
        or "resume_authentication_sha256" in receipt_body
        or not _is_sha256(operator_configuration_sha256)
        or receipt_body.get("operator_configuration_sha256")
        != operator_configuration_sha256
    ):
        raise ContiguousRunnerError(
            "meta substrate resume body is malformed"
        )
    return hashlib.sha256(
        b"arc-agi3-meta-substrate-resume-auth-v1\0"
        + operator_configuration_sha256.encode("ascii")
        + b"\0"
        + _canonical_json(dict(receipt_body))
    ).hexdigest()


def host_blocker_authentication_sha256(
    receipt_body: Mapping[str, Any],
    canaries: tuple[Taint.LiveCanary, ...],
) -> str:
    """Authenticate blocker evidence with live-only host containment values."""

    if (
        not isinstance(receipt_body, Mapping)
        or "host_authentication_sha256" in receipt_body
    ):
        raise ContiguousRunnerError(
            "host blocker authentication body is malformed"
        )
    try:
        normalized = Taint.validate_live_canaries(
            canaries, require_complete=True
        )
    except Exception as exc:
        raise ContiguousRunnerError(
            "host blocker authentication lacks complete live canaries"
        ) from exc
    key_material = b"arc-agi3-host-blocker-key-v1\0" + b"\0".join(
        b"\x1f".join((
            item.category.encode("utf-8"),
            item.location_name.encode("utf-8"),
            item.provenance.encode("ascii"),
            item.value.encode("ascii"),
        ))
        for item in normalized
    )
    key = hashlib.sha256(key_material).digest()
    return hmac.new(
        key,
        _canonical_json(dict(receipt_body)),
        hashlib.sha256,
    ).hexdigest()


def substrate_reprobe_authentication_sha256(
    receipt_body: Mapping[str, Any],
    canaries: tuple[Taint.LiveCanary, ...],
) -> str:
    """Authenticate one explicit, single-use substrate health reprobe."""

    if (
        not isinstance(receipt_body, Mapping)
        or "host_authentication_sha256" in receipt_body
    ):
        raise ContiguousRunnerError(
            "substrate reprobe authentication body is malformed"
        )
    try:
        normalized = Taint.validate_live_canaries(
            canaries, require_complete=True
        )
    except Exception as exc:
        raise ContiguousRunnerError(
            "substrate reprobe authorization lacks complete live "
            "canaries"
        ) from exc
    key_material = (
        b"arc-agi3-substrate-reprobe-key-v1\0"
        + b"\0".join(
            b"\x1f".join((
                item.category.encode("utf-8"),
                item.location_name.encode("utf-8"),
                item.provenance.encode("ascii"),
                item.value.encode("ascii"),
            ))
            for item in normalized
        )
    )
    return hmac.new(
        hashlib.sha256(key_material).digest(),
        _canonical_json(dict(receipt_body)),
        hashlib.sha256,
    ).hexdigest()


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and SHA256_RE.fullmatch(value) is not None


def _is_finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _safe_identifier(value: object) -> bool:
    return isinstance(value, str) and EVENT_ID_RE.fullmatch(value) is not None


def _safe_path_string(value: object) -> bool:
    return (
        isinstance(value, str)
        and 0 < len(value) <= 4096
        and "\x00" not in value
    )


def _is_uuid4(value: object) -> bool:
    if not isinstance(value, str):
        return False
    try:
        parsed = uuid.UUID(value)
    except (ValueError, AttributeError):
        return False
    return (
        parsed.version == 4
        and parsed.variant == uuid.RFC_4122
        and str(parsed) == value
    )


def _is_canonical_uuid(value: object) -> bool:
    if not isinstance(value, str):
        return False
    try:
        parsed = uuid.UUID(value)
    except (ValueError, AttributeError):
        return False
    return parsed.variant == uuid.RFC_4122 and str(parsed) == value


def _open_unaliased(path: Path, flags: int, mode: int = 0o600) -> int:
    try:
        descriptor = os.open(
            path, flags | getattr(os, "O_NOFOLLOW", 0), mode
        )
    except OSError as exc:
        raise ContiguousRunnerError(
            f"expected unaliased regular host file: {path}"
        ) from exc
    metadata = os.fstat(descriptor)
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        os.close(descriptor)
        raise ContiguousRunnerError(
            f"expected unaliased regular host file: {path}"
        )
    return descriptor


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_new_file(path: Path, value: object) -> None:
    payload = _canonical_json(value) + b"\n"
    descriptor = _open_unaliased(
        path, os.O_WRONLY | os.O_CREAT | os.O_EXCL
    )
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise ContiguousRunnerError(f"short write: {path}")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_new_immutable_file_atomic(
    path: Path, value: object
) -> None:
    """Publish one immutable JSON file with no writable-final crash window."""

    pending = path.parent / f".pending-{uuid.uuid4().hex}"
    try:
        _write_new_file(pending, value)
        os.chmod(pending, 0o400, follow_symlinks=False)
        descriptor = _open_unaliased(pending, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.replace(pending, path)
        _fsync_directory(path.parent)
    except BaseException:
        try:
            if pending.exists() and not pending.is_symlink():
                pending.unlink()
        except OSError:
            pass
        raise


def _read_json_file(path: Path) -> dict[str, Any]:
    descriptor = _open_unaliased(path, os.O_RDONLY)
    try:
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            raw = handle.read()
    finally:
        os.close(descriptor)
    try:
        value = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ContiguousRunnerError(f"invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ContiguousRunnerError(f"expected JSON object: {path}")
    return value


def _sha256_file(path: Path) -> str:
    descriptor = _open_unaliased(path, os.O_RDONLY)
    digest = hashlib.sha256()
    try:
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    finally:
        os.close(descriptor)
    return digest.hexdigest()


def _sha256_file_identity(
    path: Path,
) -> tuple[str, os.stat_result]:
    descriptor = _open_unaliased(path, os.O_RDONLY)
    digest = hashlib.sha256()
    try:
        before = os.fstat(descriptor)
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            digest.update(block)
        after = os.fstat(descriptor)
        stable = (
            "st_dev",
            "st_ino",
            "st_mode",
            "st_nlink",
            "st_uid",
            "st_gid",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
        )
        if any(
            getattr(before, name) != getattr(after, name)
            for name in stable
        ):
            raise ContiguousRunnerError(
                f"host file changed during anchored read: {path}"
            )
        return digest.hexdigest(), after
    finally:
        os.close(descriptor)


def _bounded_regular_bytes(path: Path, *, maximum: int) -> bytes:
    descriptor = _open_unaliased(path, os.O_RDONLY)
    try:
        metadata = os.fstat(descriptor)
        if metadata.st_size > maximum:
            raise ContiguousRunnerError(
                f"trusted input file exceeds byte bound: {path}"
            )
        chunks: list[bytes] = []
        remaining = metadata.st_size
        while remaining:
            block = os.read(
                descriptor, min(1024 * 1024, remaining)
            )
            if not block:
                raise ContiguousRunnerError(
                    f"trusted input file changed while reading: {path}"
                )
            chunks.append(block)
            remaining -= len(block)
        after = os.fstat(descriptor)
        for name in (
            "st_dev",
            "st_ino",
            "st_mode",
            "st_nlink",
            "st_uid",
            "st_gid",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
        ):
            if getattr(metadata, name) != getattr(after, name):
                raise ContiguousRunnerError(
                    f"trusted input file changed while reading: {path}"
                )
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _install_regular_bytes(
    path: Path,
    payload: bytes,
    *,
    overwrite: bool = False,
) -> None:
    """Idempotently install exact regular bytes and durably bind the parent."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if target.exists() or target.is_symlink():
        if (
            not target.is_symlink()
            and target.is_file()
            and _bounded_regular_bytes(
                target,
                maximum=max(len(payload), 1),
            )
            == payload
        ):
            return
        if not overwrite:
            raise ContiguousRunnerError(
                f"attempt input conflicts with immutable bytes: {target}"
            )
        # A WIP overlay is the sole admitted overwrite.  Replace one exact
        # unaliased regular file atomically; never follow or mutate it in place.
        _open_descriptor = _open_unaliased(target, os.O_RDONLY)
        os.close(_open_descriptor)
        pending = target.parent / (
            f".{target.name}.pending-{uuid.uuid4().hex}"
        )
        descriptor = _open_unaliased(
            pending, os.O_WRONLY | os.O_CREAT | os.O_EXCL
        )
        try:
            view = memoryview(payload)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise ContiguousRunnerError(
                        f"short write: {pending}"
                    )
                view = view[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.replace(pending, target)
        os.chmod(target, 0o600, follow_symlinks=False)
        _fsync_directory(target.parent)
        return
    descriptor = _open_unaliased(
        target, os.O_WRONLY | os.O_CREAT | os.O_EXCL
    )
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise ContiguousRunnerError(
                    f"short write: {target}"
                )
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.chmod(target, 0o600, follow_symlinks=False)
    _fsync_directory(target.parent)


def _copy_regular_tree(
    source: Path,
    destination: Path,
    *,
    overwrite: bool,
    maximum_files: int,
    maximum_file_bytes: int,
    maximum_total_bytes: int,
    allow_hidden_state_paths: bool = False,
) -> None:
    try:
        Contract._validate_regular_tree(
            source, label="trusted source tree"
        )
    except Exception as exc:
        raise ContiguousRunnerError(
            "trusted source tree contains an alias or special file"
        ) from exc
    files = [
        path
        for path in sorted(source.rglob("*"))
        if path.is_file()
    ]
    if len(files) > maximum_files:
        raise ContiguousRunnerError(
            "trusted source tree exceeds file-count bound"
        )
    total = 0
    for source_file in files:
        relative = source_file.relative_to(source).as_posix()
        state_path = PurePosixPath(relative)
        safe_state_path = bool(
            not state_path.is_absolute()
            and str(state_path) == relative
            and 0 < len(state_path.parts) <= 32
            and all(
                part not in {"", ".", ".."}
                and part
                not in Transport.PROJECT_DISCOVERY_MARKERS
                for part in state_path.parts
            )
        )
        if not (
            Transport.is_safe_relative_path(relative)
            or (allow_hidden_state_paths and safe_state_path)
        ):
            raise ContiguousRunnerError(
                "trusted source tree contains an unsafe relative path"
            )
        raw = _bounded_regular_bytes(
            source_file, maximum=maximum_file_bytes
        )
        total += len(raw)
        if total > maximum_total_bytes:
            raise ContiguousRunnerError(
                "trusted source tree exceeds aggregate byte bound"
            )
        _install_regular_bytes(
            destination / relative,
            raw,
            overwrite=overwrite,
        )


def _seal_regular_tree(root: Path) -> None:
    try:
        Contract._validate_regular_tree(root, label="sealed tree")
    except Exception as exc:
        raise ContiguousRunnerError(
            "tree cannot be sealed because it is not regular"
        ) from exc
    directories = [root]
    for path in root.rglob("*"):
        if path.is_file():
            os.chmod(path, 0o400, follow_symlinks=False)
        elif path.is_dir():
            directories.append(path)
    for directory in sorted(
        directories,
        key=lambda value: len(value.parts),
        reverse=True,
    ):
        os.chmod(directory, 0o500, follow_symlinks=False)
        _fsync_directory(directory)


def _path_pointer_prefix(
    path: Path,
) -> tuple[tuple[object, ...], ...]:
    """Bind an absolute path to every non-symlinked ancestor inode."""

    selected = Path(path)
    if not selected.is_absolute():
        raise ContiguousRunnerError(
            f"cached evidence path is not absolute: {selected}"
        )
    current = Path(selected.anchor)
    result: list[tuple[object, ...]] = []
    for part in selected.parts[1:]:
        current = current / part
        metadata = current.stat(follow_symlinks=False)
        if stat.S_ISLNK(metadata.st_mode):
            raise ContiguousRunnerError(
                f"cached evidence path contains a symlink: {current}"
            )
        # Ancestor directory mtimes intentionally are not included: adding an
        # unrelated sibling cannot change this pointer.  Inode/type/mode bind
        # every path component and catch replacement or aliasing.
        result.append((
            str(current),
            metadata.st_dev,
            metadata.st_ino,
            stat.S_IFMT(metadata.st_mode),
            stat.S_IMODE(metadata.st_mode),
        ))
    return tuple(result)


def _regular_file_pointer(
    path: Path,
) -> tuple[tuple[object, ...], ...]:
    prefix = _path_pointer_prefix(path)
    metadata = Path(path).stat(follow_symlinks=False)
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
    ):
        raise ContiguousRunnerError(
            f"cached evidence is not one unaliased regular file: {path}"
        )
    return (
        *prefix[:-1],
        (
            str(path),
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_mode,
            metadata.st_nlink,
            metadata.st_size,
            metadata.st_mtime_ns,
            metadata.st_ctime_ns,
        ),
    )


def _regular_tree_pointer(
    root: Path,
    *,
    maximum_entries: int = MAX_PARENT_SOURCE_FILES,
) -> tuple[tuple[object, ...], ...]:
    selected = Path(root)
    prefix = _path_pointer_prefix(selected)
    root_metadata = selected.stat(follow_symlinks=False)
    if not stat.S_ISDIR(root_metadata.st_mode):
        raise ContiguousRunnerError(
            f"cached evidence is not a regular tree: {selected}"
        )
    entries = sorted(selected.rglob("*"), key=lambda item: str(item))
    if len(entries) > maximum_entries:
        raise ContiguousRunnerError(
            "cached regular tree exceeds its entry bound"
        )
    result: list[tuple[object, ...]] = [
        *prefix[:-1],
        (
            str(selected),
            root_metadata.st_dev,
            root_metadata.st_ino,
            root_metadata.st_mode,
            root_metadata.st_nlink,
            root_metadata.st_size,
            root_metadata.st_mtime_ns,
            root_metadata.st_ctime_ns,
        ),
    ]
    for entry in entries:
        metadata = entry.stat(follow_symlinks=False)
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not (
                stat.S_ISDIR(metadata.st_mode)
                or (
                    stat.S_ISREG(metadata.st_mode)
                    and metadata.st_nlink == 1
                )
            )
        ):
            raise ContiguousRunnerError(
                f"cached source tree contains an unsafe entry: {entry}"
            )
        result.append((
            str(entry.relative_to(selected)),
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_mode,
            metadata.st_nlink,
            metadata.st_size,
            metadata.st_mtime_ns,
            metadata.st_ctime_ns,
        ))
    return tuple(result)


class DurableAttemptJournal:
    """Append immutable, hash-chained events as atomic files.

    One-file-per-event avoids an unrecoverable torn JSONL tail.  Temporary files
    are never history: a committed event appears only after a same-directory
    atomic rename and directory fsync.  Re-appending an identical ``event_id`` is
    idempotent; conflicting reuse fails closed.
    """

    def __init__(self, root: Path):
        self.root = Path(root)
        if self.root.is_symlink() or (
            self.root.exists() and not self.root.is_dir()
        ):
            raise ContiguousRunnerError(
                "journal root must be a regular host directory"
            )
        self.root.mkdir(parents=True, exist_ok=True)
        if self.root.is_symlink() or not self.root.is_dir():
            raise ContiguousRunnerError(
                "journal root must be a regular host directory"
            )
        os.chmod(self.root, 0o700, follow_symlinks=False)
        self.lock_path = self.root / ".journal.lock"
        self.emergency_reserve_path = (
            self.root / ".storage-emergency-reserve"
        )
        incident_name = (
            "-campaign:journal-or-storage-exhausted.json"
        )
        retained_incident_path = next(
            (
                path
                for path in self.root.rglob("*.json")
                if path.name.endswith(incident_name)
            ),
            None,
        )
        if (
            retained_incident_path is None
            and not (
            self.emergency_reserve_path.exists()
            or self.emergency_reserve_path.is_symlink()
            )
        ):
            descriptor = _open_unaliased(
                self.emergency_reserve_path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            )
            try:
                remaining = JOURNAL_EMERGENCY_RESERVE_BYTES
                block = b"\0" * min(4096, remaining)
                while remaining:
                    selected = block[: min(len(block), remaining)]
                    written = os.write(descriptor, selected)
                    if written <= 0:
                        raise ContiguousRunnerError(
                            "journal emergency reserve write stalled"
                        )
                    remaining -= written
                os.fsync(descriptor)
                os.fchmod(descriptor, 0o400)
            finally:
                os.close(descriptor)
            _fsync_directory(self.root)
        elif (
            self.emergency_reserve_path.exists()
            or self.emergency_reserve_path.is_symlink()
        ):
            reserve = self.emergency_reserve_path.stat(
                follow_symlinks=False
            )
            if (
                self.emergency_reserve_path.is_symlink()
                or not stat.S_ISREG(reserve.st_mode)
                or reserve.st_nlink != 1
                or reserve.st_uid != os.getuid()
                or stat.S_IMODE(reserve.st_mode) != 0o400
                or reserve.st_size
                != JOURNAL_EMERGENCY_RESERVE_BYTES
            ):
                raise ContiguousRunnerError(
                    "journal emergency reserve is unsafe"
                )
            if retained_incident_path is not None:
                self.emergency_reserve_path.unlink()
                _fsync_directory(self.root)
        self.quiescence_reserve_path = (
            self.root / ".storage-quiescence-reserve"
        )
        quiescence_name = (
            "-campaign:storage-emergency-quiesced.json"
        )
        retained_quiescence_path = next(
            (
                path
                for path in self.root.rglob("*.json")
                if path.name.endswith(quiescence_name)
            ),
            None,
        )
        if (
            retained_quiescence_path is None
            and not (
                self.quiescence_reserve_path.exists()
                or self.quiescence_reserve_path.is_symlink()
            )
        ):
            descriptor = _open_unaliased(
                self.quiescence_reserve_path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            )
            try:
                remaining = JOURNAL_QUIESCENCE_RESERVE_BYTES
                block = b"\0" * min(4096, remaining)
                while remaining:
                    selected = block[: min(len(block), remaining)]
                    written = os.write(descriptor, selected)
                    if written <= 0:
                        raise ContiguousRunnerError(
                            "journal quiescence reserve write stalled"
                        )
                    remaining -= written
                os.fsync(descriptor)
                os.fchmod(descriptor, 0o400)
            finally:
                os.close(descriptor)
            _fsync_directory(self.root)
        elif (
            self.quiescence_reserve_path.exists()
            or self.quiescence_reserve_path.is_symlink()
        ):
            reserve = self.quiescence_reserve_path.stat(
                follow_symlinks=False
            )
            if (
                self.quiescence_reserve_path.is_symlink()
                or not stat.S_ISREG(reserve.st_mode)
                or reserve.st_nlink != 1
                or reserve.st_uid != os.getuid()
                or stat.S_IMODE(reserve.st_mode) != 0o400
                or reserve.st_size
                != JOURNAL_QUIESCENCE_RESERVE_BYTES
            ):
                raise ContiguousRunnerError(
                    "journal quiescence reserve is unsafe"
                )
            if retained_quiescence_path is not None:
                self.quiescence_reserve_path.unlink()
                _fsync_directory(self.root)
        self._cache: list[dict[str, Any]] | None = None
        self._cache_names: tuple[str, ...] = ()
        self._cache_file_signatures: tuple[
            tuple[int, int, int, int, int, int, int], ...
        ] = ()
        self._cache_directory_signature: (
            tuple[tuple[str, int, int], ...] | None
        ) = None

    def filesystem_admission_snapshot(
        self,
        *,
        required_event_bytes: int,
    ) -> dict[str, Any]:
        try:
            observed = os.statvfs(self.root)
            metadata = self.root.stat(follow_symlinks=False)
        except OSError as exc:
            raise JournalStorageExhausted(
                failed_event_id="unknown",
                failed_event_kind="unknown",
                failure_stage="statvfs",
                error_code=type(exc).__name__,
                storage_snapshot={},
            ) from exc
        available_bytes = int(
            observed.f_bavail * observed.f_frsize
        )
        available_inodes = int(observed.f_favail)
        return {
            "schema": 1,
            "kind": "contiguous_journal_filesystem_admission",
            "filesystem_device": metadata.st_dev,
            "available_bytes": available_bytes,
            "available_inodes": available_inodes,
            "required_event_bytes": required_event_bytes,
            "minimum_free_bytes":
                MIN_JOURNAL_FILESYSTEM_FREE_BYTES,
            "minimum_free_inodes":
                MIN_JOURNAL_FILESYSTEM_FREE_INODES,
            "byte_admitted":
                available_bytes
                >= (
                    MIN_JOURNAL_FILESYSTEM_FREE_BYTES
                    + required_event_bytes
                ),
            "inode_admitted":
                available_inodes
                >= MIN_JOURNAL_FILESYSTEM_FREE_INODES,
        }

    @staticmethod
    def _has_storage_incident(
        events: Sequence[Mapping[str, Any]],
    ) -> bool:
        return any(
            event.get("kind")
            == "JOURNAL_OR_STORAGE_EXHAUSTED"
            for event in events
        )

    def _directory_signature(
        self,
    ) -> tuple[tuple[str, int, int], ...]:
        result: list[tuple[str, int, int]] = []
        metadata = self.root.stat(follow_symlinks=False)
        result.append((
            ".",
            metadata.st_mtime_ns,
            metadata.st_ctime_ns,
        ))
        for child in sorted(self.root.iterdir()):
            if (
                child.is_symlink()
                or not child.is_dir()
                or re.fullmatch(
                    r"segment-\d{8}", child.name
                )
                is None
            ):
                continue
            selected = child.stat(follow_symlinks=False)
            result.append((
                child.name,
                selected.st_mtime_ns,
                selected.st_ctime_ns,
            ))
        return tuple(result)

    @staticmethod
    def _file_signature(
        path: Path,
    ) -> tuple[int, int, int, int, int, int, int]:
        """Bind one pathname sample to an open regular-file descriptor.

        A separate ``lstat``/``is_symlink`` pair can combine event A's
        metadata with event B's later pathname state.  Keep both the parent
        directory and event descriptors open, require the directory entry to
        resolve to the exact file descriptor, and reject any parent or file
        metadata transition observed during the transaction.
        """

        try:
            parent_descriptor = os.open(
                path.parent,
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0),
            )
        except OSError as exc:
            raise ContiguousRunnerError(
                f"journal event parent is unavailable: {path.parent}"
            ) from exc
        descriptor: int | None = None
        try:
            parent_before = os.fstat(parent_descriptor)
            if not stat.S_ISDIR(parent_before.st_mode):
                raise ContiguousRunnerError(
                    f"journal event parent is not a directory: {path.parent}"
                )
            descriptor = _open_unaliased(path, os.O_RDONLY)
            before = os.fstat(descriptor)
            linked = path.stat(follow_symlinks=False)
            after = os.fstat(descriptor)
            parent_after = os.fstat(parent_descriptor)
            file_fields = (
                "st_dev",
                "st_ino",
                "st_mode",
                "st_nlink",
                "st_uid",
                "st_gid",
                "st_size",
                "st_mtime_ns",
                "st_ctime_ns",
            )
            parent_fields = (
                "st_dev",
                "st_ino",
                "st_mode",
                "st_nlink",
                "st_uid",
                "st_gid",
                "st_size",
                "st_mtime_ns",
                "st_ctime_ns",
            )
            if (
                not stat.S_ISREG(linked.st_mode)
                or linked.st_nlink != 1
                or any(
                    getattr(before, name) != getattr(after, name)
                    or getattr(linked, name) != getattr(after, name)
                    for name in file_fields
                )
                or any(
                    getattr(parent_before, name)
                    != getattr(parent_after, name)
                    for name in parent_fields
                )
            ):
                raise ContiguousRunnerError(
                    f"journal event pointer changed during signature: {path}"
                )
            return (
                after.st_dev,
                after.st_ino,
                after.st_mode,
                after.st_nlink,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            )
        finally:
            if descriptor is not None:
                os.close(descriptor)
            os.close(parent_descriptor)

    def _read_event_anchored(
        self,
        path: Path,
    ) -> tuple[
        dict[str, Any],
        tuple[int, int, int, int, int, int, int],
    ]:
        """Read one event and bind its parsed bytes to one exact inode state.

        A path read followed by an independent metadata sample can cache bytes
        from event A under event B's signature when an in-place rewrite lands
        between those operations.  Keep the descriptor open, compare its
        identity before and after the bounded read, and only then require the
        path to resolve to that same signature.  The returned signature is
        therefore the signature of the bytes that were actually parsed.
        """

        descriptor = _open_unaliased(path, os.O_RDONLY)
        try:
            before = os.fstat(descriptor)
            if before.st_size > Scheduler.MAX_JOURNAL_EVENT_BYTES:
                raise ContiguousRunnerError(
                    "journal event exceeds the scheduler evidence bound"
                )
            chunks: list[bytes] = []
            remaining = before.st_size
            while remaining:
                block = os.read(
                    descriptor, min(1024 * 1024, remaining)
                )
                if not block:
                    raise ContiguousRunnerError(
                        f"journal event changed during anchored read: {path}"
                    )
                chunks.append(block)
                remaining -= len(block)
            after = os.fstat(descriptor)
            stable = (
                "st_dev",
                "st_ino",
                "st_mode",
                "st_nlink",
                "st_uid",
                "st_gid",
                "st_size",
                "st_mtime_ns",
                "st_ctime_ns",
            )
            if any(
                getattr(before, name) != getattr(after, name)
                for name in stable
            ):
                raise ContiguousRunnerError(
                    f"journal event changed during anchored read: {path}"
                )
            signature = (
                after.st_dev,
                after.st_ino,
                after.st_mode,
                after.st_nlink,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            )
            try:
                raw = b"".join(chunks)
                value = json.loads(raw)
            except (UnicodeError, json.JSONDecodeError) as exc:
                raise ContiguousRunnerError(
                    f"invalid JSON: {path}"
                ) from exc
            if not isinstance(value, dict):
                raise ContiguousRunnerError(
                    f"expected JSON object: {path}"
                )
        finally:
            os.close(descriptor)
        if self._file_signature(path) != signature:
            raise ContiguousRunnerError(
                f"journal event changed during anchored read: {path}"
            )
        return value, signature

    def _lock(self):
        descriptor = _open_unaliased(
            self.lock_path, os.O_RDWR | os.O_CREAT
        )
        handle = os.fdopen(descriptor, "r+", encoding="utf-8")
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        return handle

    @staticmethod
    def _event_digest(event: dict[str, Any]) -> str:
        return hashlib.sha256(_canonical_json(event)).hexdigest()

    @staticmethod
    def _segment_number(sequence: int) -> int:
        return (
            (sequence - 1) // JOURNAL_SEGMENT_EVENT_LIMIT
        ) + 1

    def _segment_directory(self, segment_number: int) -> Path:
        return (
            self.root
            if segment_number == 1
            else self.root / f"segment-{segment_number:08d}"
        )

    def _segment_closure_path(self, segment_number: int) -> Path:
        directory = self._segment_directory(segment_number)
        return (
            self.root
            / f".segment-{segment_number:08d}-closure.json"
            if segment_number == 1
            else directory / ".closure.json"
        )

    def _expected_segment_closure(
        self,
        *,
        segment_number: int,
        events: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        start = (
            (segment_number - 1)
            * JOURNAL_SEGMENT_EVENT_LIMIT
            + 1
        )
        end = segment_number * JOURNAL_SEGMENT_EVENT_LIMIT
        selected = [
            event
            for event in events
            if start <= int(event["sequence"]) <= end
        ]
        if len(selected) != JOURNAL_SEGMENT_EVENT_LIMIT:
            raise ContiguousRunnerError(
                "journal segment closure lacks its full event range"
            )
        inventory = [
            {
                "sequence": event["sequence"],
                "event_id": event["event_id"],
                "digest": event["digest"],
            }
            for event in selected
        ]
        return {
            "schema": 1,
            "kind": "contiguous_journal_checkpoint_segment",
            "segment_number": segment_number,
            "start_sequence": start,
            "end_sequence": end,
            "event_count": len(selected),
            "first_event_digest": selected[0]["digest"],
            "last_event_digest": selected[-1]["digest"],
            "event_inventory_sha256": hashlib.sha256(
                _canonical_json(inventory)
            ).hexdigest(),
            "status": "CLOSED",
        }

    def _close_segment(
        self,
        *,
        segment_number: int,
        events: Sequence[Mapping[str, Any]],
    ) -> tuple[str, str]:
        closure_path = self._segment_closure_path(segment_number)
        document = self._expected_segment_closure(
            segment_number=segment_number,
            events=events,
        )
        if closure_path.exists() or closure_path.is_symlink():
            if _read_json_file(closure_path) != document:
                raise ContiguousRunnerError(
                    "journal segment closure changed"
                )
        else:
            _write_new_immutable_file_atomic(
                closure_path, document
            )
        return str(closure_path), _sha256_file(closure_path)

    def _ensure_segment_for_append(
        self,
        *,
        sequence: int,
        events: Sequence[Mapping[str, Any]],
    ) -> Path:
        segment_number = self._segment_number(sequence)
        directory = self._segment_directory(segment_number)
        if segment_number == 1:
            return directory
        previous_path, previous_sha256 = self._close_segment(
            segment_number=segment_number - 1,
            events=events,
        )
        if directory.exists() or directory.is_symlink():
            if directory.is_symlink() or not directory.is_dir():
                raise ContiguousRunnerError(
                    "journal segment directory is unsafe"
                )
        else:
            directory.mkdir(mode=0o700)
            _fsync_directory(self.root)
        checkpoint_path = directory / ".checkpoint.json"
        checkpoint = {
            "schema": 1,
            "kind": "contiguous_journal_segment_genesis",
            "segment_number": segment_number,
            "start_sequence": sequence,
            "previous_segment_closure_path": previous_path,
            "previous_segment_closure_sha256": previous_sha256,
            "previous_event_digest": events[-1]["digest"],
            "status": "OPEN",
        }
        if checkpoint_path.exists() or checkpoint_path.is_symlink():
            if _read_json_file(checkpoint_path) != checkpoint:
                raise ContiguousRunnerError(
                    "journal segment checkpoint changed"
                )
        else:
            _write_new_immutable_file_atomic(
                checkpoint_path, checkpoint
            )
        return directory

    def _read_segment_control(
        self, path: Path, *, label: str
    ) -> dict[str, Any]:
        value, signature = self._read_event_anchored(path)
        if (
            not stat.S_ISREG(signature[2])
            or stat.S_IMODE(signature[2]) != 0o400
            or signature[3] != 1
        ):
            raise ContiguousRunnerError(
                f"{label} is not one immutable regular file"
            )
        return value

    def _validate_segment_chain(
        self, events: Sequence[Mapping[str, Any]]
    ) -> None:
        """Reopen the complete closure/genesis chain around event history."""

        segment_directories: dict[int, Path] = {}
        root_closures: dict[int, Path] = {}
        for entry in self.root.iterdir():
            match = re.fullmatch(r"segment-(\d{8})", entry.name)
            if match is not None and entry.is_dir() and not entry.is_symlink():
                segment_directories[int(match.group(1))] = entry
                continue
            match = re.fullmatch(
                r"\.segment-(\d{8})-closure\.json",
                entry.name,
            )
            if match is not None:
                root_closures[int(match.group(1))] = entry
        if 1 in segment_directories or any(
            number < 2 for number in segment_directories
        ):
            raise ContiguousRunnerError(
                "journal segment directory numbering is invalid"
            )
        if segment_directories:
            highest = max(segment_directories)
            if set(segment_directories) != set(range(2, highest + 1)):
                raise ContiguousRunnerError(
                    "journal segment directory chain has a gap"
                )
        if set(root_closures) - {1}:
            raise ContiguousRunnerError(
                "journal root contains a misplaced segment closure"
            )
        event_count = len(events)
        full_segment_count = (
            event_count // JOURNAL_SEGMENT_EVENT_LIMIT
        )
        if (
            event_count % JOURNAL_SEGMENT_EVENT_LIMIT == 0
            and event_count > 0
        ):
            last_event_segment = full_segment_count
        else:
            last_event_segment = full_segment_count + (
                1 if event_count else 0
            )
        allowed_highest_directory = (
            last_event_segment + 1
            if event_count > 0
            and event_count % JOURNAL_SEGMENT_EVENT_LIMIT == 0
            else last_event_segment
        )
        if segment_directories and (
            max(segment_directories) > allowed_highest_directory
        ):
            raise ContiguousRunnerError(
                "journal segment directory is ahead of history"
            )

        closures: dict[int, Path] = dict(root_closures)
        for number, directory in segment_directories.items():
            closure_path = directory / ".closure.json"
            if closure_path.exists() or closure_path.is_symlink():
                closures[number] = closure_path
        for number, path in sorted(closures.items()):
            if number > full_segment_count:
                raise ContiguousRunnerError(
                    "journal closes an incomplete segment"
                )
            expected = self._expected_segment_closure(
                segment_number=number,
                events=events,
            )
            observed = self._read_segment_control(
                path,
                label="journal segment closure",
            )
            if set(observed) != set(expected) or observed != expected:
                raise ContiguousRunnerError(
                    "journal segment closure changed"
                )

        for number, directory in sorted(
            segment_directories.items()
        ):
            prior_number = number - 1
            prior_closure_path = self._segment_closure_path(
                prior_number
            )
            if (
                prior_number not in closures
                or closures[prior_number] != prior_closure_path
            ):
                raise ContiguousRunnerError(
                    "journal segment lacks its prior closure"
                )
            checkpoint_path = directory / ".checkpoint.json"
            segment_events = [
                event
                for event in events
                if self._segment_number(
                    int(event["sequence"])
                )
                == number
            ]
            if not (
                checkpoint_path.exists()
                or checkpoint_path.is_symlink()
            ):
                # A directory-only cut can occur after mkdir+parent fsync and
                # before genesis creation.  It has no authority and the next
                # append deterministically completes the same checkpoint.
                if segment_events or (
                    directory / ".closure.json"
                ).exists():
                    raise ContiguousRunnerError(
                        "journal segment lacks its checkpoint"
                    )
                continue
            expected_checkpoint = {
                "schema": 1,
                "kind": "contiguous_journal_segment_genesis",
                "segment_number": number,
                "start_sequence": (
                    (number - 1)
                    * JOURNAL_SEGMENT_EVENT_LIMIT
                    + 1
                ),
                "previous_segment_closure_path":
                    str(prior_closure_path),
                "previous_segment_closure_sha256":
                    _sha256_file(prior_closure_path),
                "previous_event_digest": events[
                    (number - 1)
                    * JOURNAL_SEGMENT_EVENT_LIMIT
                    - 1
                ]["digest"],
                "status": "OPEN",
            }
            observed_checkpoint = self._read_segment_control(
                checkpoint_path,
                label="journal segment checkpoint",
            )
            if (
                set(observed_checkpoint)
                != set(expected_checkpoint)
                or observed_checkpoint != expected_checkpoint
            ):
                raise ContiguousRunnerError(
                    "journal segment checkpoint changed"
                )

        for number in range(1, full_segment_count + 1):
            next_directory_exists = (
                number + 1 in segment_directories
            )
            if next_directory_exists and number not in closures:
                raise ContiguousRunnerError(
                    "journal segment transition lacks its closure"
                )
            if (
                number in closures
                and not next_directory_exists
                and number != full_segment_count
            ):
                raise ContiguousRunnerError(
                    "journal segment closure chain is disconnected"
                )

    def _paths(self) -> list[Path]:
        paths: list[Path] = []
        directories = [self.root]
        for entry in self.root.iterdir():
            if entry.name.startswith("."):
                if (
                    entry.name
                    in {
                        ".journal.lock",
                        ".storage-emergency-reserve",
                        ".storage-quiescence-reserve",
                    }
                    or re.fullmatch(
                        r"\.pending-[A-Za-z0-9_.:-]+",
                        entry.name,
                    )
                    or re.fullmatch(
                        r"\.segment-\d{8}-closure\.json",
                        entry.name,
                    )
                ):
                    continue
                raise ContiguousRunnerError(
                    "journal contains an unexpected hidden entry: "
                    f"{entry.name}"
                )
            if entry.is_symlink():
                raise ContiguousRunnerError(
                    f"journal contains symlink entry: {entry}"
                )
            if entry.is_dir():
                if re.fullmatch(
                    r"segment-\d{8}", entry.name
                ) is None:
                    raise ContiguousRunnerError(
                        "journal contains an unexpected directory: "
                        f"{entry.name}"
                    )
                directories.append(entry)
                continue
            if not entry.is_file():
                raise ContiguousRunnerError(
                    f"journal contains nonregular entry: {entry}"
                )
            if re.fullmatch(
                r"\d{20}-[A-Za-z0-9_.:-]+\.json",
                entry.name,
            ):
                paths.append(entry)
            else:
                raise ContiguousRunnerError(
                    f"journal contains unexpected entry: {entry.name}"
                )
        for directory in directories[1:]:
            segment_number = int(directory.name.split("-")[1])
            checkpoint_path = directory / ".checkpoint.json"
            for entry in directory.iterdir():
                if entry.name.startswith("."):
                    if (
                        entry.name
                        in {
                            ".checkpoint.json",
                            ".closure.json",
                        }
                        or re.fullmatch(
                            r"\.pending-[A-Za-z0-9_.:-]+",
                            entry.name,
                        )
                    ):
                        continue
                    raise ContiguousRunnerError(
                        "journal segment contains an unexpected "
                        f"hidden entry: {entry.name}"
                    )
                if (
                    entry.is_symlink()
                    or not entry.is_file()
                    or re.fullmatch(
                        r"\d{20}-[A-Za-z0-9_.:-]+\.json",
                        entry.name,
                    )
                    is None
                ):
                    raise ContiguousRunnerError(
                        "journal segment contains an unexpected entry"
                    )
                sequence = int(entry.name[:20])
                if self._segment_number(sequence) != segment_number:
                    raise ContiguousRunnerError(
                        "journal event is stored in the wrong segment"
                    )
                paths.append(entry)
        return sorted(paths, key=lambda path: path.name)

    def _read_authenticated(self) -> list[dict[str, Any]]:
        """Return the private authenticated view used by the reducer.

        The event dictionaries alias the cache and therefore never cross the
        public journal API.  Keeping this private view avoids an O(history)
        deep copy on every supervision-cycle state reduction.
        """

        signature = self._directory_signature()
        if (
            self._cache is not None
            and signature == self._cache_directory_signature
        ):
            cached_paths = tuple(
                self.root / name for name in self._cache_names
            )
            try:
                cached_signatures = tuple(
                    self._file_signature(path)
                    for path in cached_paths
                )
            except (FileNotFoundError, OSError):
                cached_signatures = ()
            if (
                cached_signatures == self._cache_file_signatures
                and self._directory_signature() == signature
            ):
                self._validate_segment_chain(self._cache)
                return list(self._cache)
        paths = self._paths()
        names = tuple(
            str(path.relative_to(self.root))
            for path in paths
        )
        if self._cache is not None:
            prefix_length = len(self._cache_names)
            if (
                len(names) < prefix_length
                or names[:prefix_length] != self._cache_names
            ):
                raise ContiguousRunnerError(
                    "journal immutable prefix was truncated or replaced"
                )
            try:
                prefix_signatures = tuple(
                    self._file_signature(path)
                    for path in paths[:prefix_length]
                )
            except (FileNotFoundError, OSError) as exc:
                raise ContiguousRunnerError(
                    "journal immutable prefix became unreadable"
                ) from exc
            if prefix_signatures != self._cache_file_signatures:
                raise ContiguousRunnerError(
                    "journal immutable prefix pointer or metadata changed"
                )
        prefix_length = (
            len(self._cache_names) if self._cache is not None else 0
        )
        events: list[dict[str, Any]] = (
            list(self._cache) if self._cache is not None else []
        )
        prior: str | None = (
            str(events[-1]["digest"]) if events else None
        )
        seen_ids: set[str] = {
            str(event["event_id"]) for event in events
        }
        observed_signatures = (
            list(self._cache_file_signatures)
            if self._cache is not None
            else []
        )
        for expected_sequence, path in enumerate(
            paths[prefix_length:],
            prefix_length + 1,
        ):
            event, event_signature = self._read_event_anchored(path)
            required = {
                "schema",
                "sequence",
                "event_id",
                "kind",
                "recorded_at",
                "previous_digest",
                "payload",
                "digest",
            }
            body = {key: event[key] for key in required - {"digest"}} \
                if set(event) == required else None
            if (
                body is None
                or event["schema"] != JOURNAL_SCHEMA
                or isinstance(event["schema"], bool)
                or event["sequence"] != expected_sequence
                or isinstance(event["sequence"], bool)
                or not _safe_identifier(event["event_id"])
                or not _safe_identifier(event["kind"])
                or not _is_finite_number(event["recorded_at"])
                or event["previous_digest"] != prior
                or not isinstance(event["payload"], dict)
                or event["event_id"] in seen_ids
                or not _is_sha256(event["digest"])
                or self._event_digest(body) != event["digest"]
                or not path.name.startswith(f"{expected_sequence:020d}-")
            ):
                raise ContiguousRunnerError(
                    f"invalid journal event at sequence {expected_sequence}: {path}"
                )
            events.append(event)
            observed_signatures.append(event_signature)
            seen_ids.add(event["event_id"])
            prior = event["digest"]
        self._validate_segment_chain(events)
        if self._directory_signature() != signature:
            raise ContiguousRunnerError(
                "journal directory changed during authentication"
            )
        self._cache = list(events)
        self._cache_names = names
        self._cache_file_signatures = tuple(observed_signatures)
        self._cache_directory_signature = signature
        return list(events)

    def read(self) -> list[dict[str, Any]]:
        """Return a fully independent public copy of authenticated history."""

        return copy.deepcopy(self._read_authenticated())

    def append(
        self,
        *,
        event_id: str,
        kind: str,
        payload: dict[str, Any],
        recorded_at: float,
    ) -> dict[str, Any]:
        if not _safe_identifier(event_id) or not _safe_identifier(kind):
            raise ContiguousRunnerError("invalid journal event identifier")
        if (
            not _is_finite_number(recorded_at)
        ):
            raise ContiguousRunnerError("recorded_at must be a number")
        try:
            normalized_payload = json.loads(
                _canonical_json(payload)
            )
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ContiguousRunnerError(
                "journal payload is not canonical JSON"
            ) from exc
        if not isinstance(normalized_payload, dict):
            raise ContiguousRunnerError(
                "journal payload must be a JSON object"
            )
        payload = normalized_payload
        handle = self._lock()
        pending: Path | None = None
        try:
            events = self._read_authenticated()
            for existing in events:
                if existing["event_id"] != event_id:
                    continue
                if (
                    existing["kind"] != kind
                    or existing["payload"] != payload
                ):
                    raise ContiguousRunnerError(
                        f"conflicting reuse of journal event_id {event_id}"
                    )
                return copy.deepcopy(existing)
            sequence = len(events) + 1
            body = {
                "schema": JOURNAL_SCHEMA,
                "sequence": sequence,
                "event_id": event_id,
                "kind": kind,
                "recorded_at": float(recorded_at),
                "previous_digest": (
                    events[-1]["digest"] if events else None
                ),
                "payload": payload,
            }
            event = {**body, "digest": self._event_digest(body)}
            if (
                len(_canonical_json(event)) + 1
                > Scheduler.MAX_JOURNAL_EVENT_BYTES
            ):
                raise ContiguousRunnerError(
                    "journal event exceeds the scheduler evidence bound"
                )
            encoded_event_bytes = len(_canonical_json(event)) + 1
            storage: dict[str, Any] = {}
            if (
                kind != "JOURNAL_OR_STORAGE_EXHAUSTED"
                and not self._has_storage_incident(events)
            ):
                storage = self.filesystem_admission_snapshot(
                    required_event_bytes=encoded_event_bytes
                )
                if (
                    storage["byte_admitted"] is not True
                    or storage["inode_admitted"] is not True
                ):
                    raise JournalStorageExhausted(
                        failed_event_id=event_id,
                        failed_event_kind=kind,
                        failure_stage="pre_append_admission",
                        error_code=(
                            "insufficient_bytes"
                            if not storage["byte_admitted"]
                            else "insufficient_inodes"
                        ),
                        storage_snapshot=storage,
                    )
            try:
                event_directory = self._ensure_segment_for_append(
                    sequence=sequence,
                    events=events,
                )
            except (OSError, ContiguousRunnerError) as exc:
                storage_code = _storage_error_code(exc)
                if storage_code is not None:
                    raise JournalStorageExhausted(
                        failed_event_id=event_id,
                        failed_event_kind=kind,
                        failure_stage="segment_rollover",
                        error_code=storage_code,
                        storage_snapshot=storage,
                    ) from exc
                raise
            final = (
                event_directory
                / f"{sequence:020d}-{event_id}.json"
            )
            pending = (
                event_directory / f".pending-{uuid.uuid4().hex}"
            )
            try:
                _write_new_file(pending, event)
                os.chmod(pending, 0o400, follow_symlinks=False)
                pending_descriptor = _open_unaliased(
                    pending, os.O_RDONLY
                )
                try:
                    os.fsync(pending_descriptor)
                finally:
                    os.close(pending_descriptor)
                os.replace(pending, final)
                pending = None
                _fsync_directory(event_directory)
            except (OSError, ContiguousRunnerError) as exc:
                if pending is not None:
                    try:
                        pending.unlink()
                    except OSError:
                        pass
                storage_code = _storage_error_code(exc)
                if storage_code is not None:
                    try:
                        storage = (
                            self.filesystem_admission_snapshot(
                                required_event_bytes=(
                                    encoded_event_bytes
                                )
                            )
                        )
                    except JournalStorageExhausted:
                        storage = {}
                    raise JournalStorageExhausted(
                        failed_event_id=event_id,
                        failed_event_kind=kind,
                        failure_stage="event_commit",
                        error_code=storage_code,
                        storage_snapshot=storage,
                    ) from exc
                raise
            self._cache = [*events, event]
            self._cache_names = (
                *self._cache_names,
                str(final.relative_to(self.root)),
            )
            self._cache_file_signatures = (
                *self._cache_file_signatures,
                self._file_signature(final),
            )
            self._cache_directory_signature = self._directory_signature()
            return copy.deepcopy(event)
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            handle.close()

    def commit_storage_incident(
        self, failure: JournalStorageExhausted, *, recorded_at: float
    ) -> dict[str, Any]:
        """Consume the preallocated reserve and commit one terminal latch."""

        events = self._read_authenticated()
        existing = next(
            (
                event
                for event in events
                if event["kind"]
                == "JOURNAL_OR_STORAGE_EXHAUSTED"
            ),
            None,
        )
        if existing is not None:
            return copy.deepcopy(existing)
        if self.emergency_reserve_path.exists():
            self.emergency_reserve_path.unlink()
            _fsync_directory(self.root)
        return self.append(
            event_id="campaign:journal-or-storage-exhausted",
            kind="JOURNAL_OR_STORAGE_EXHAUSTED",
            payload={
                "reason_code": "journal_or_storage_exhausted",
                "failed_event_id": failure.failed_event_id,
                "failed_event_kind": failure.failed_event_kind,
                "failure_stage": failure.failure_stage,
                "error_code": failure.error_code,
                "storage_snapshot": failure.storage_snapshot,
                "solver_authority": False,
                "wip_authority": False,
                "cost_authority": False,
                "promotion_authority": False,
                "status": "OPERATOR_INCIDENT",
            },
            recorded_at=recorded_at,
        )

    def release_quiescence_reserve(self) -> None:
        """Release the second reserve exactly once for containment evidence.

        The storage incident itself consumes only the small incident reserve.
        Live descendants may still need compact absence receipts plus one final
        journal event.  This independently preallocated reserve is released
        only after that incident is authenticated and before any emergency
        containment side effect is attempted.
        """

        events = self._read_authenticated()
        if not self._has_storage_incident(events):
            raise ContiguousRunnerError(
                "quiescence reserve cannot be released before a storage "
                "incident"
            )
        if any(
            event.get("kind") == "STORAGE_EMERGENCY_QUIESCED"
            for event in events
        ):
            if (
                self.quiescence_reserve_path.exists()
                or self.quiescence_reserve_path.is_symlink()
            ):
                raise ContiguousRunnerError(
                    "quiescence reserve reappeared after its terminal event"
                )
            return
        if (
            self.quiescence_reserve_path.exists()
            or self.quiescence_reserve_path.is_symlink()
        ):
            metadata = self.quiescence_reserve_path.stat(
                follow_symlinks=False
            )
            if (
                self.quiescence_reserve_path.is_symlink()
                or not stat.S_ISREG(metadata.st_mode)
                or metadata.st_nlink != 1
                or metadata.st_uid != os.getuid()
                or stat.S_IMODE(metadata.st_mode) != 0o400
                or metadata.st_size
                != JOURNAL_QUIESCENCE_RESERVE_BYTES
            ):
                raise ContiguousRunnerError(
                    "journal quiescence reserve is unsafe"
                )
            self.quiescence_reserve_path.unlink()
            _fsync_directory(self.root)


class ReadOnlyAttemptJournal(DurableAttemptJournal):
    """Open an existing journal without creating, locking, or chmodding it."""

    def __init__(self, root: Path):
        selected = Path(root)
        if (
            selected.is_symlink()
            or not selected.exists()
            or not selected.is_dir()
        ):
            raise ContiguousRunnerError(
                "read-only journal root must be an existing regular directory"
            )
        self.root = selected
        # ``read`` uses only these cache fields.  Deliberately do not create a
        # lock path or call the mutating DurableAttemptJournal constructor.
        self.lock_path = selected / ".journal.lock"
        self._cache = None
        self._cache_names = ()
        self._cache_file_signatures = ()
        self._cache_directory_signature = None

    def _lock(self):
        raise ContiguousRunnerError(
            "read-only journal cannot acquire a mutation lock"
        )

    def append(
        self,
        *,
        event_id: str,
        kind: str,
        payload: dict[str, Any],
        recorded_at: float,
    ) -> dict[str, Any]:
        del event_id, kind, payload, recorded_at
        raise ContiguousRunnerError(
            "read-only journal cannot append"
        )


# There is one WIP schema and one strict parser.  The runner alias preserves
# its public API while making scheduler selection and terminal-WIP replay use
# the identical field set and validity rules.
WipSnapshot = Scheduler.WipBinding


@dataclass(frozen=True)
class ProposerTransportConfiguration:
    """Frozen controller-container, host-mediator, and bridge projection.

    The runner journals this complete value before any external process is
    created.  Codex launcher/package/native-binary/schema paths are absolute
    paths *inside the digest-pinned controller image*, never host executables.
    Backend image-label and in-container descriptor receipts must prove those
    bytes, the guardian, effective capability projection, enforcing egress
    proxy, and host-mediated bridge policy all match this projection exactly.
    """

    model: str
    model_provider: str
    allow_provider_model_fallback: Literal[False]
    reasoning_effort_allowlist: tuple[str, ...]
    controller_image_reference: str
    controller_image_digest: str
    controller_entrypoint: tuple[str, ...]
    controller_guardian_path: str
    controller_guardian_sha256: str
    controller_user: str
    controller_egress_policy: str
    controller_egress_proxy_image_reference: str
    controller_egress_proxy_image_digest: str
    controller_egress_policy_sha256: str
    controller_cpus: float
    controller_memory_bytes: int
    controller_pids: int
    controller_tmpfs_bytes: int
    arena_transport: Literal[
        "docker-attach-stdio+named-volume-unix"
    ]
    arena_relay_image_reference: str
    arena_relay_image_digest: str
    arena_relay_source_sha256: str
    codex_launcher_path: str
    codex_launcher_sha256: str
    codex_package_manifest_path: str
    codex_package_manifest_sha256: str
    codex_binary_path: str
    codex_binary_sha256: str
    codex_binary_bytes: int
    codex_cli_version: str
    app_server_protocol_schema_path: str
    app_server_protocol_schema_sha256: str
    app_server_protocol_schema_bundle_path: str
    app_server_protocol_schema_bundle_sha256: str
    controller_preflight_request_allowlist: tuple[str, ...]
    controller_preflight_notification_allowlist: tuple[str, ...]
    controller_turn_request_allowlist: tuple[str, ...]
    dynamic_tool_namespace: str
    dynamic_tool_names: tuple[str, ...]
    bridge_protocol_version: int
    bridge_operation_allowlist: tuple[str, ...]
    bridge_exec_allowlist: tuple[str, ...]
    bridge_max_request_bytes: int
    bridge_max_response_bytes: int
    bridge_max_file_bytes: int
    bridge_max_total_export_bytes: int
    bridge_max_processes: int
    bridge_max_exec_seconds: int


@dataclass(frozen=True)
class AttemptSpec:
    schema: int
    campaign_id: str
    generation_id: str
    attempt_id: str
    game: str
    target_level: int
    authoritative_target: int
    parent_checkpoint_path: str
    parent_checkpoint_sha256: str
    frontier_sha256: str
    generation_dir: str
    input_dir: str
    scratch_dir: str
    workspace_dir: str
    output_dir: str
    arena_socket_path: str
    arena_token_file_path: str
    bridge_dir: str
    bridge_socket_path: str
    bridge_token_file_path: str
    bridge_policy_receipt_path: str
    host_transcript_path: str
    app_server_transcript_path: str
    neutral_host_cwd_path: str
    app_server_state_dir: str
    app_server_control_dir: str
    image_reference: str
    image_digest: str
    worker_command: tuple[str, ...]
    resource_limits: "ResourceLimitsProjection"
    proposer_transport: ProposerTransportConfiguration
    input_tree_sha256: str
    parent_source_path: str
    parent_source_tree_sha256: str
    initial_workspace_tree_sha256: str
    initial_app_server_state_tree_sha256: str
    hard_safety_seconds: int
    max_auth_refreshes: int
    input_bundle_receipt_path: str
    input_bundle_receipt_sha256: str
    frontier_brief_path: str
    frontier_brief_sha256: str
    supervisory_handoff_path: str | None
    supervisory_handoff_sha256: str | None
    supervisory_handoff_binding_receipt_path: str | None
    supervisory_handoff_binding_receipt_sha256: str | None
    bridge_policy_path: str
    bridge_policy_sha256: str
    parent_action_count: int
    remaining_action_budget: int
    fresh_prefix_required: bool
    effort: Literal["medium", "high", "xhigh", "max"]
    soft_allocation_seconds: int
    wip_mode: Literal["exclude", "restore_clean_same_frontier"]
    thread_mode: Literal["new", "resume"]
    resume_thread_id: str | None
    resume_thread_binding_sha256: str | None
    wip: WipSnapshot | None
    supervisory_handoff: Scheduler.SupervisoryHandoffBinding | None
    cost_limit_remaining: float | None


@dataclass(frozen=True)
class PromotionCandidate:
    game: str
    from_level: int
    to_level: int
    parent_checkpoint_sha256: str
    candidate_manifest_path: str
    candidate_manifest_sha256: str
    probe_isolation_mode: Literal[
        "verified_isolated_clone",
        "fresh_process_per_candidate",
    ]
    probe_isolation_evidence_sha256: str
    supervisory_handoff_sha256: str | None
    supervisory_native_reproduction_receipt_sha256: str | None


@dataclass(frozen=True)
class PromotionCommit:
    game: str
    from_level: int
    to_level: int
    parent_checkpoint_sha256: str
    checkpoint_path: str
    checkpoint_sha256: str
    exact_path: tuple[Any, ...]
    promotion_receipt_sha256: str
    source_version_id: str
    source_tree_sha256: str
    supervisory_handoff_sha256: str | None
    supervisory_native_reproduction_receipt_sha256: str | None


@dataclass(frozen=True)
class HostBlockerEvidence:
    """Host-only blocker authority bound to one exact attempt/frontier."""

    code: str
    receipt_path: str
    receipt_sha256: str


@dataclass(frozen=True)
class AttemptResult:
    kind: Literal[
        "clean_no_progress",
        "tainted",
        "protocol_invalid",
        "infrastructure",
        "candidate",
        "blocker",
    ]
    cost_used: float = 0.0
    reason: str = ""
    candidate: PromotionCandidate | None = None
    wip: WipSnapshot | None = None
    blocker: HostBlockerEvidence | None = None
    native_sidecar_request_draft: (
        Scheduler.NativeSidecarRequestDraft | None
    ) = None


@dataclass(frozen=True)
class ResourceLimitsProjection:
    cpus: float
    memory_bytes: int
    pids: int
    tmpfs_bytes: int


@dataclass(frozen=True)
class BackendConfiguration:
    image_reference: str
    image_digest: str
    worker_command: tuple[str, ...]
    resource_limits: ResourceLimitsProjection
    proposer_transport: ProposerTransportConfiguration


@dataclass(frozen=True)
class AttemptLayout:
    campaign_id: str
    generation_id: str
    attempt_id: str
    game: str
    target_level: int
    authoritative_target: int
    parent_checkpoint_path: str
    parent_checkpoint_sha256: str
    frontier_sha256: str
    generation_dir: str
    input_dir: str
    scratch_dir: str
    workspace_dir: str
    output_dir: str
    arena_socket_path: str
    arena_token_file_path: str
    bridge_dir: str
    bridge_socket_path: str
    bridge_token_file_path: str
    bridge_policy_receipt_path: str
    host_transcript_path: str
    app_server_transcript_path: str
    neutral_host_cwd_path: str
    app_server_state_dir: str
    app_server_control_dir: str
    parent_source_path: str
    parent_source_tree_sha256: str
    effort: Literal["medium", "high", "xhigh", "max"]
    soft_allocation_seconds: int
    wip_mode: Literal["exclude", "restore_clean_same_frontier"]
    thread_mode: Literal["new", "resume"]
    resume_thread_id: str | None
    resume_thread_binding_sha256: str | None
    proposer_transport: ProposerTransportConfiguration
    wip: WipSnapshot | None
    supervisory_handoff: Scheduler.SupervisoryHandoffBinding | None


@dataclass(frozen=True)
class AttemptReservation:
    schema: int
    campaign_id: str
    generation_id: str
    attempt_id: str
    game: str
    target_level: int
    authoritative_target: int
    parent_checkpoint_path: str
    parent_checkpoint_sha256: str
    frontier_sha256: str
    generation_dir: str
    input_dir: str
    scratch_dir: str
    workspace_dir: str
    output_dir: str
    arena_socket_path: str
    arena_token_file_path: str
    bridge_dir: str
    bridge_socket_path: str
    bridge_token_file_path: str
    bridge_policy_receipt_path: str
    host_transcript_path: str
    app_server_transcript_path: str
    neutral_host_cwd_path: str
    app_server_state_dir: str
    app_server_control_dir: str
    image_reference: str
    image_digest: str
    worker_command: tuple[str, ...]
    resource_limits: ResourceLimitsProjection
    proposer_transport: ProposerTransportConfiguration
    parent_source_path: str
    parent_source_tree_sha256: str
    effort: Literal["medium", "high", "xhigh", "max"]
    soft_allocation_seconds: int
    wip_mode: Literal["exclude", "restore_clean_same_frontier"]
    thread_mode: Literal["new", "resume"]
    resume_thread_id: str | None
    resume_thread_binding_sha256: str | None
    wip: WipSnapshot | None
    supervisory_handoff: Scheduler.SupervisoryHandoffBinding | None
    cost_limit_remaining: float | None


@dataclass(frozen=True)
class InputBundleReceipt:
    receipt_path: str
    receipt_sha256: str
    input_tree_sha256: str
    parent_source_tree_sha256: str
    initial_workspace_tree_sha256: str
    parent_checkpoint_sha256: str
    wip_tree_sha256: str | None
    wip_solver_source_tree_sha256: str | None
    frontier_brief_path: str
    frontier_brief_sha256: str
    bridge_policy_path: str
    bridge_policy_sha256: str
    parent_action_count: int
    remaining_action_budget: int
    fresh_prefix_required: bool
    supervisory_handoff_path: str | None
    supervisory_handoff_sha256: str | None
    supervisory_handoff_binding_receipt_path: str | None
    supervisory_handoff_binding_receipt_sha256: str | None


@dataclass
class _ReducerCheckpoint:
    """Validated journal-derived state at one exact hash-chain head.

    The checkpoint contains no authority beyond the immutable journal prefix:
    every file that can still affect execution or promotion is revalidated
    after suffix reduction on every ``state()`` call.  Closed-generation bytes
    remain the full audit's concern, exactly as before.  Independent mutable
    shells prevent a caller's returned view from poisoning this reducer cache.
    """

    head_sequence: int
    head_digest: str
    genesis_digest: str
    lanes: dict[str, dict[str, Any]]
    attempts: dict[str, dict[str, Any]]
    budget: Scheduler.BudgetState
    pending_decision: Scheduler.SchedulerDecision | None
    pending_auxiliary_decision: Scheduler.AuxiliaryDecision | None
    auxiliary_assignments: dict[str, dict[str, Any]]
    sidecar_requests: dict[str, dict[str, Any]]
    complexity_rounds: list[Scheduler.ComplexityRoundState]
    used_decision_ids: set[str]
    used_attempt_ids: set[str]
    used_generation_ids: set[str]
    used_reservation_ids: set[str]
    used_expert_ids: set[str]
    used_thread_ids: set[str]
    failure_operation_circuits: dict[str, dict[str, Any]]
    failure_domain_circuits: dict[str, dict[str, Any]]
    operator_incident: dict[str, Any] | None
    substrate_incident: dict[str, Any] | None
    storage_incident: dict[str, Any] | None
    storage_quiescence: dict[str, Any] | None


def _clone_reducer_lanes(
    lanes: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for game, source in lanes.items():
        lane = dict(source)
        lane["clean_proposer_settlements"] = list(
            source["clean_proposer_settlements"]
        )
        lane["public_observation_receipt_sha256s"] = list(
            source["public_observation_receipt_sha256s"]
        )
        result[game] = lane
    return result


def _clone_reducer_records(
    records: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Clone mutable reducer shells; all contained records are frozen."""

    return {
        identity: dict(record)
        for identity, record in records.items()
    }


class InputBundleBuilder(Protocol):
    """Trusted, idempotent constructor for one immutable container input."""

    def initialize_lane_source(
        self, game: str, destination: Path
    ) -> tuple[str, str]:
        """Install and return the immutable generic L0 source snapshot."""
        ...

    def prepare(self, layout: AttemptLayout) -> InputBundleReceipt:
        ...


class ProductionInputBundleBuilder:
    """Build one immutable input bundle and one exact initial workspace.

    L0 is always seeded from one audited, game-agnostic blank scaffold.  Every
    later input is copied from the immutable source version bound into durable
    lane state by the preceding promotion.  A clean same-frontier WIP may
    overlay workspace files, while the exact parent source remains separately
    immutable and hash-addressable under ``input/parent_source``.
    """

    def __init__(
        self,
    ) -> None:
        root = (
            Path(__file__).resolve().parent
            / "contiguous_blank_scaffold"
        )
        if (
            root.is_symlink()
            or not root.is_dir()
        ):
            raise ContiguousRunnerError(
                "canonical blank scaffold is missing or unsafe"
            )
        self.blank_scaffold_root = root
        self.blank_scaffold_tree_sha256 = (
            CANONICAL_BLANK_SCAFFOLD_TREE_SHA256
        )
        self._validate_parent_source(
            root,
            expected_tree_sha256=self.blank_scaffold_tree_sha256,
            label="blank scaffold",
        )

    @staticmethod
    def _validate_parent_source(
        root: Path,
        *,
        expected_tree_sha256: str,
        label: str,
    ) -> tuple[str, ...]:
        try:
            Contract._validate_regular_tree(root, label=label)
        except Exception as exc:
            raise ContiguousRunnerError(
                f"{label} is not a regular immutable source tree"
            ) from exc
        if any(not path.is_file() for path in root.iterdir()):
            raise ContiguousRunnerError(
                f"{label} must be one flat source-file view"
            )
        payloads = {
            path.name: _bounded_regular_bytes(
                path, maximum=MAX_PARENT_SOURCE_FILE_BYTES
            )
            for path in root.iterdir()
        }
        try:
            files = SourceSchema.validate_source_payloads(payloads)
        except SourceSchema.SourceSchemaError as exc:
            raise ContiguousRunnerError(
                f"{label} violates the winning-source schema"
            ) from exc
        if Contract._tree_hash(root) != expected_tree_sha256:
            raise ContiguousRunnerError(
                f"{label} differs from its frozen tree hash"
            )
        return files

    def initialize_lane_source(
        self, game: str, destination: Path
    ) -> tuple[str, str]:
        if not _safe_identifier(game):
            raise ContiguousRunnerError(
                "zero-source game identity is malformed"
            )
        self._validate_parent_source(
            self.blank_scaffold_root,
            expected_tree_sha256=self.blank_scaffold_tree_sha256,
            label="blank scaffold",
        )
        destination = Path(destination)
        if destination.is_symlink() or (
            destination.exists() and not destination.is_dir()
        ):
            raise ContiguousRunnerError(
                "zero-source destination is unsafe"
            )
        destination.mkdir(parents=True, exist_ok=True, mode=0o700)
        _copy_regular_tree(
            self.blank_scaffold_root,
            destination,
            overwrite=False,
            maximum_files=MAX_PARENT_SOURCE_FILES,
            maximum_file_bytes=MAX_PARENT_SOURCE_FILE_BYTES,
            maximum_total_bytes=MAX_PARENT_SOURCE_TOTAL_BYTES,
        )
        self._validate_parent_source(
            destination,
            expected_tree_sha256=self.blank_scaffold_tree_sha256,
            label="zero source",
        )
        _seal_regular_tree(destination)
        return (
            str(destination),
            self.blank_scaffold_tree_sha256,
        )

    @staticmethod
    def _frontier_brief(
        layout: AttemptLayout,
        *,
        parent_action_count: int,
        remaining_action_budget: int,
        fresh_prefix_required: bool,
    ) -> dict[str, Any]:
        handoff = layout.supervisory_handoff
        handoff_prompt = (
            None
            if handoff is None
            else Scheduler.supervisory_prompt_projection(handoff)
        )
        return {
            "schema": 1,
            "kind": "arc_agi3_contiguous_frontier_brief",
            "campaign_id": layout.campaign_id,
            "generation_id": layout.generation_id,
            "attempt_id": layout.attempt_id,
            "game": layout.game,
            "target_level": layout.target_level,
            "authoritative_target": layout.authoritative_target,
            "parent_checkpoint_sha256":
                layout.parent_checkpoint_sha256,
            "frontier_sha256": layout.frontier_sha256,
            "parent_action_count": parent_action_count,
            "remaining_action_budget": remaining_action_budget,
            "fresh_prefix_required": fresh_prefix_required,
            "effort": layout.effort,
            "soft_allocation_seconds":
                layout.soft_allocation_seconds,
            "wip_mode": layout.wip_mode,
            "thread_mode": layout.thread_mode,
            "supervisory_handoff": handoff_prompt,
        }

    @staticmethod
    def _bridge_policy(layout: AttemptLayout) -> dict[str, Any]:
        transport = layout.proposer_transport
        return {
            "schema": 1,
            "kind": "arc_agi3_contiguous_bridge_policy",
            "campaign_id": layout.campaign_id,
            "generation_id": layout.generation_id,
            "attempt_id": layout.attempt_id,
            "game": layout.game,
            "target_level": layout.target_level,
            "frontier_sha256": layout.frontier_sha256,
            "parent_checkpoint_sha256":
                layout.parent_checkpoint_sha256,
            "protocol_version": transport.bridge_protocol_version,
            "operation_allowlist":
                list(transport.bridge_operation_allowlist),
            "exec_allowlist": list(
                transport.bridge_exec_allowlist
            ),
            "workspace_root": "/arc/workspace",
            "export_root": "/arc/export",
            "bounds": {
                "max_request_bytes":
                    transport.bridge_max_request_bytes,
                "max_response_bytes":
                    transport.bridge_max_response_bytes,
                "max_file_bytes":
                    transport.bridge_max_file_bytes,
                "max_total_export_bytes":
                    transport.bridge_max_total_export_bytes,
                "max_processes": transport.bridge_max_processes,
                "max_exec_seconds":
                    transport.bridge_max_exec_seconds,
            },
        }

    def prepare(self, layout: AttemptLayout) -> InputBundleReceipt:
        source_root = Path(layout.parent_source_path)
        frozen_source_sha = layout.parent_source_tree_sha256
        if not source_root.is_absolute():
            raise ContiguousRunnerError(
                "parent source path is not absolute"
            )
        self._validate_parent_source(
            source_root,
            expected_tree_sha256=frozen_source_sha,
            label="lane-bound parent source",
        )
        input_root = Path(layout.input_dir)
        workspace_root = Path(layout.workspace_dir)
        if (
            input_root.is_symlink()
            or workspace_root.is_symlink()
            or not input_root.is_dir()
            or not workspace_root.is_dir()
        ):
            raise ContiguousRunnerError(
                "attempt input/workspace roots are unsafe"
            )
        checkpoint = Contract.load_trusted_checkpoint(
            Path(layout.parent_checkpoint_path),
            expected_game=layout.game,
            authoritative_target=layout.authoritative_target,
        )
        checkpoint_raw = _bounded_regular_bytes(
            Path(layout.parent_checkpoint_path),
            maximum=MAX_PARENT_SOURCE_FILE_BYTES,
        )
        if (
            hashlib.sha256(checkpoint_raw).hexdigest()
            != layout.parent_checkpoint_sha256
        ):
            raise ContiguousRunnerError(
                "parent checkpoint changed before input construction"
            )
        parent_action_count = len(checkpoint.final_path)
        remaining_action_budget = 600 - parent_action_count
        fresh_prefix_required = remaining_action_budget == 0
        brief = self._frontier_brief(
            layout,
            parent_action_count=parent_action_count,
            remaining_action_budget=remaining_action_budget,
            fresh_prefix_required=fresh_prefix_required,
        )
        brief_raw = _canonical_json(brief) + b"\n"
        policy_raw = (
            _canonical_json(self._bridge_policy(layout)) + b"\n"
        )
        handoff_raw: bytes | None = None
        handoff_binding_receipt_raw: bytes | None = None
        handoff_file_sha256: str | None = None
        handoff_binding_receipt_sha256: str | None = None
        if layout.supervisory_handoff is not None:
            binding = layout.supervisory_handoff
            handoff_raw = (
                _canonical_json(
                    Scheduler.supervisory_prompt_projection(binding)
                )
                + b"\n"
            )
            handoff_file_sha256 = hashlib.sha256(
                handoff_raw
            ).hexdigest()
            handoff_binding_receipt_raw = (
                _canonical_json({
                    "schema": 1,
                    "kind":
                        "contiguous_supervisory_handoff_prompt_binding",
                    "campaign_id": layout.campaign_id,
                    "generation_id": layout.generation_id,
                    "attempt_id": layout.attempt_id,
                    "game": layout.game,
                    "frontier_sha256": layout.frontier_sha256,
                    "parent_checkpoint_sha256":
                        layout.parent_checkpoint_sha256,
                    "assignment_id": binding.assignment_id,
                    "output_manifest_sha256":
                        binding.output_manifest_sha256,
                    "supervisory_handoff_sha256":
                        binding.supervisory_handoff_sha256,
                    "handoff_file_sha256": handoff_file_sha256,
                    "admission_receipt_sha256":
                        binding.admission_receipt_sha256,
                    "prompt_authority":
                        "unverified_hypothesis_only",
                    "native_reproduction_required_before_wip_candidate_or_promotion":
                        True,
                    "scheduler_authority": False,
                    "mutation_authority": False,
                    "promotion_authority": False,
                })
                + b"\n"
            )
            handoff_binding_receipt_sha256 = hashlib.sha256(
                handoff_binding_receipt_raw
            ).hexdigest()

        parent_source = input_root / "parent_source"
        parent_source.mkdir(mode=0o700, exist_ok=True)
        _copy_regular_tree(
            source_root,
            parent_source,
            overwrite=False,
            maximum_files=MAX_PARENT_SOURCE_FILES,
            maximum_file_bytes=MAX_PARENT_SOURCE_FILE_BYTES,
            maximum_total_bytes=MAX_PARENT_SOURCE_TOTAL_BYTES,
        )
        _copy_regular_tree(
            source_root,
            workspace_root,
            overwrite=False,
            maximum_files=MAX_PARENT_SOURCE_FILES,
            maximum_file_bytes=MAX_PARENT_SOURCE_FILE_BYTES,
            maximum_total_bytes=MAX_PARENT_SOURCE_TOTAL_BYTES,
        )
        if (
            Contract._tree_hash(parent_source) != frozen_source_sha
        ):
            raise ContiguousRunnerError(
                "parent source copy is incomplete or substituted"
            )

        _install_regular_bytes(
            input_root / "checkpoint.json", checkpoint_raw
        )
        _install_regular_bytes(
            input_root / "frontier_brief.json", brief_raw
        )
        _install_regular_bytes(
            input_root / "bridge_policy.json", policy_raw
        )
        _install_regular_bytes(
            workspace_root / "checkpoint.json", checkpoint_raw
        )
        _install_regular_bytes(
            workspace_root / "frontier_brief.json", brief_raw
        )
        if handoff_raw is not None:
            assert handoff_binding_receipt_raw is not None
            _install_regular_bytes(
                input_root / "supervisory_handoff.json",
                handoff_raw,
            )
            _install_regular_bytes(
                input_root
                / "supervisory_handoff_binding_receipt.json",
                handoff_binding_receipt_raw,
            )

        if layout.wip is not None:
            wip_root = Path(layout.wip.wip_root_path)
            solver_source_root = Path(
                layout.wip.solver_source_path
            )
            try:
                Contract._validate_regular_tree(
                    wip_root, label="selected clean WIP"
                )
                self._validate_parent_source(
                    solver_source_root,
                    expected_tree_sha256=(
                        layout.wip.solver_source_tree_sha256
                    ),
                    label="selected clean WIP solver source",
                )
            except Exception as exc:
                raise ContiguousRunnerError(
                    "selected WIP tree is unsafe"
                ) from exc
            if (
                Contract._tree_hash(wip_root)
                != layout.wip.wip_tree_sha256
                or solver_source_root.parent != wip_root
                or solver_source_root.name != "solver_source"
                or not {
                    entry.name for entry in wip_root.iterdir()
                }.issubset({"solver_source", "context"})
                or not solver_source_root.is_dir()
            ):
                raise ContiguousRunnerError(
                    "selected WIP changed before workspace construction"
                )
            for path in wip_root.rglob("*"):
                if not path.is_file():
                    continue
                relative = path.relative_to(wip_root).as_posix()
                if relative in {
                    "checkpoint.json",
                    "frontier_brief.json",
                }:
                    raise ContiguousRunnerError(
                        "WIP attempted to replace immutable frontier input"
                    )
            _copy_regular_tree(
                wip_root,
                input_root / "wip",
                overwrite=False,
                maximum_files=512,
                maximum_file_bytes=
                    layout.proposer_transport.bridge_max_file_bytes,
                maximum_total_bytes=layout.proposer_transport
                .bridge_max_total_export_bytes,
            )
            _copy_regular_tree(
                solver_source_root,
                workspace_root,
                overwrite=True,
                maximum_files=MAX_PARENT_SOURCE_FILES,
                maximum_file_bytes=MAX_PARENT_SOURCE_FILE_BYTES,
                maximum_total_bytes=MAX_PARENT_SOURCE_TOTAL_BYTES,
            )
            context_root = wip_root / "context"
            if context_root.exists():
                Contract._validate_regular_tree(
                    context_root, label="selected clean WIP context"
                )
                _copy_regular_tree(
                    context_root,
                    workspace_root / "wip_context",
                    overwrite=True,
                    maximum_files=512,
                    maximum_file_bytes=layout.proposer_transport
                    .bridge_max_file_bytes,
                    maximum_total_bytes=layout.proposer_transport
                    .bridge_max_total_export_bytes,
                )
            if (
                Contract._tree_hash(input_root / "wip")
                != layout.wip.wip_tree_sha256
            ):
                raise ContiguousRunnerError(
                    "bundled WIP copy is incomplete"
                )
        elif (input_root / "wip").exists():
            raise ContiguousRunnerError(
                "no-WIP input contains a WIP subtree"
            )

        expected_input_entries = {
            "bridge_policy.json",
            "checkpoint.json",
            "frontier_brief.json",
            "parent_source",
        }
        if layout.wip is not None:
            expected_input_entries.add("wip")
        if layout.supervisory_handoff is not None:
            expected_input_entries.update({
                "supervisory_handoff.json",
                "supervisory_handoff_binding_receipt.json",
            })
        if {
            entry.name for entry in input_root.iterdir()
        } != expected_input_entries:
            raise ContiguousRunnerError(
                "input bundle contains an undeclared top-level entry"
            )
        input_tree_sha = Contract._tree_hash(input_root)
        initial_workspace_sha = Contract._tree_hash(workspace_root)
        receipt_body = {
            "schema": RUNNER_SCHEMA,
            "campaign_id": layout.campaign_id,
            "generation_id": layout.generation_id,
            "attempt_id": layout.attempt_id,
            "game": layout.game,
            "target_level": layout.target_level,
            "frontier_sha256": layout.frontier_sha256,
            "input_tree_sha256": input_tree_sha,
            "parent_source_tree_sha256": frozen_source_sha,
            "initial_workspace_tree_sha256":
                initial_workspace_sha,
            "parent_checkpoint_sha256":
                layout.parent_checkpoint_sha256,
            "wip_tree_sha256": (
                layout.wip.wip_tree_sha256
                if layout.wip is not None
                else None
            ),
            "wip_solver_source_tree_sha256": (
                layout.wip.solver_source_tree_sha256
                if layout.wip is not None
                else None
            ),
            "frontier_brief_sha256":
                hashlib.sha256(brief_raw).hexdigest(),
            "bridge_policy_sha256":
                hashlib.sha256(policy_raw).hexdigest(),
            "parent_action_count": parent_action_count,
            "remaining_action_budget": remaining_action_budget,
            "fresh_prefix_required": fresh_prefix_required,
            "supervisory_handoff_sha256": handoff_file_sha256,
            "supervisory_handoff_binding_receipt_sha256":
                handoff_binding_receipt_sha256,
        }
        receipt_path = (
            Path(layout.generation_dir)
            / "input_bundle_receipt.json"
        )
        receipt_raw = _canonical_json(receipt_body) + b"\n"
        _install_regular_bytes(receipt_path, receipt_raw)
        return InputBundleReceipt(
            receipt_path=str(receipt_path),
            receipt_sha256=hashlib.sha256(
                receipt_raw
            ).hexdigest(),
            input_tree_sha256=input_tree_sha,
            parent_source_tree_sha256=frozen_source_sha,
            initial_workspace_tree_sha256=initial_workspace_sha,
            parent_checkpoint_sha256=
                layout.parent_checkpoint_sha256,
            wip_tree_sha256=(
                layout.wip.wip_tree_sha256
                if layout.wip is not None
                else None
            ),
            wip_solver_source_tree_sha256=(
                layout.wip.solver_source_tree_sha256
                if layout.wip is not None
                else None
            ),
            frontier_brief_path=str(
                input_root / "frontier_brief.json"
            ),
            frontier_brief_sha256=
                receipt_body["frontier_brief_sha256"],
            bridge_policy_path=str(
                input_root / "bridge_policy.json"
            ),
            bridge_policy_sha256=
                receipt_body["bridge_policy_sha256"],
            parent_action_count=parent_action_count,
            remaining_action_budget=remaining_action_budget,
            fresh_prefix_required=fresh_prefix_required,
            supervisory_handoff_path=(
                str(input_root / "supervisory_handoff.json")
                if handoff_raw is not None
                else None
            ),
            supervisory_handoff_sha256=handoff_file_sha256,
            supervisory_handoff_binding_receipt_path=(
                str(
                    input_root
                    / "supervisory_handoff_binding_receipt.json"
                )
                if handoff_binding_receipt_raw is not None
                else None
            ),
            supervisory_handoff_binding_receipt_sha256=(
                handoff_binding_receipt_sha256
            ),
        )


@dataclass(frozen=True)
class BackendPreparation:
    preparation_id: str
    launch_attestation_path: str
    launch_attestation_sha256: str
    observed_image_digest: str
    image_observation_sha256: str
    container_observation_sha256: str
    bridge_policy_receipt_path: str
    bridge_policy_receipt_sha256: str
    arena_session_binding_receipt_path: str
    arena_session_binding_receipt_sha256: str
    arena_transport: Literal[
        "docker-attach-stdio+named-volume-unix"
    ]
    arena_volume_name: str
    arena_volume_observation_sha256: str
    arena_relay_container_id: str
    arena_relay_image_digest: str
    arena_relay_image_observation_sha256: str
    arena_relay_container_observation_sha256: str
    arena_relay_readiness_receipt_path: str
    arena_relay_readiness_receipt_sha256: str
    arena_relay_attach_argv_sha256: str
    arena_relay_socket_identity_sha256: str
    arena_relay_preparation_receipt_path: str
    arena_relay_preparation_receipt_sha256: str
    probe_isolation_mode: Literal[
        "verified_isolated_clone",
        "fresh_process_per_candidate",
    ]
    probe_isolation_evidence_sha256: str
    neutral_cwd_attestation_path: str
    neutral_cwd_attestation_sha256: str
    app_server_config_receipt_path: str
    app_server_config_receipt_sha256: str
    codex_binary_receipt_path: str
    codex_binary_receipt_sha256: str
    protocol_schema_receipt_path: str
    protocol_schema_receipt_sha256: str
    controller_image_digest: str
    controller_egress_proxy_image_digest: str
    controller_egress_policy_sha256: str
    controller_canary_escrow_path: str
    controller_canary_escrow_sha256: str
    controller_canary_escrow_identity_sha256: str
    controller_canary_commitments_json: str
    controller_canary_commitments_sha256: str
    controller_canary_placement_descriptors_json: str
    controller_canary_placement_descriptors_sha256: str
    controller_supply_chain_unobserved_until_launch: Literal[True]


@dataclass(frozen=True)
class BackendLaunch:
    backend_id: str
    container_id: str
    running_observation_sha256: str
    substrate_identity_sha256: str
    substrate_preflight_receipt_path: str
    substrate_preflight_receipt_sha256: str
    bridge_runtime_attestation_path: str
    bridge_runtime_attestation_sha256: str
    app_server_runtime_receipt_path: str
    app_server_runtime_receipt_sha256: str
    app_server_pid: int
    app_server_process_start: str
    app_server_process_group_id: int
    app_server_pid_is_diagnostic: Literal[True]
    process_identity_authority: Literal[
        "controller_container_cgroup"
    ]
    controller_container_id: str
    controller_image_digest: str
    egress_proxy_container_id: str
    egress_proxy_image_digest: str
    egress_policy_sha256: str
    controller_launch_intent_sha256: str
    controller_launch_receipt_path: str
    controller_launch_receipt_sha256: str
    controller_guardian_start_receipt_path: str
    controller_guardian_start_receipt_sha256: str
    controller_supply_chain_manifest_sha256: str
    codex_thread_id: str
    codex_turn_id: str
    thread_binding_path: str
    thread_binding_sha256: str
    transcript_chain_receipt_path: str
    transcript_chain_receipt_sha256: str
    transcript_chain_sha256: str
    thread_rebinding_receipt_path: str | None
    thread_rebinding_receipt_sha256: str | None


@dataclass(frozen=True)
class BackendSubstrateHealthProbe:
    authorization_id: str
    probe_index: int
    remediation_epoch_sha256: str
    failed_substrate_identity_sha256: str
    healthy_substrate_identity_sha256: str | None
    incident_failure_receipt_sha256: str
    failure_class: Literal[
        "DETERMINISTIC_CONFIGURATION",
        "TRANSIENT_INFRASTRUCTURE",
    ] | None
    failure_code: str | None
    status: Literal["PASS", "FAILED"]
    receipt_path: str
    receipt_sha256: str


@dataclass(frozen=True)
class BackendPoll:
    status: Literal["running", "exited", "containment_fault"]
    observation_sha256: str
    exit_code: int | None = None


@dataclass(frozen=True)
class BackendCollection:
    result: AttemptResult
    worker_outcome_sha256: str
    output_tree_sha256: str
    host_transcript_path: str
    host_transcript_sha256: str
    native_public_observation_receipt_sha256s: tuple[str, ...]
    container_stdout_path: str
    container_stdout_sha256: str
    container_stderr_path: str
    container_stderr_sha256: str
    app_server_transcript_path: str
    app_server_transcript_sha256: str
    codex_thread_id: str
    codex_turn_id: str
    structured_turn_status: Literal[
        "completed", "interrupted", "failed"
    ]
    structured_provider_outcome: Literal[
        "completed",
        "capacity",
        "rate_limit",
        "provider_failure",
        "containment_fault",
    ]
    token_usage_receipt_path: str
    token_usage_receipt_sha256: str
    provider_usage_receipt_path: str
    provider_usage_receipt_sha256: str
    final_transcript_chain_receipt_path: str
    final_transcript_chain_receipt_sha256: str
    final_transcript_chain_sha256: str
    final_thread_binding_path: str
    final_thread_binding_sha256: str
    bridge_export_receipt_path: str
    bridge_export_receipt_sha256: str
    secret_scan_receipt_path: str
    secret_scan_receipt_sha256: str
    controller_state_scan_receipt_path: str
    controller_state_scan_receipt_sha256: str
    controller_state_inventory_sha256: str
    retained_canary_scan_receipt_path: str
    retained_canary_scan_receipt_sha256: str
    supervisory_native_reproduction_receipt_path: str | None
    supervisory_native_reproduction_receipt_sha256: str | None
    target_boundary_receipt_path: str | None
    target_boundary_receipt_sha256: str | None
    target_boundary_sha256: str | None
    target_boundary_workspace_tree_sha256: str | None
    taint_scan_receipt_path: str
    taint_scan_receipt_sha256: str
    app_server_state_tree_sha256: str
    model_final_text_sha256: str


@dataclass(frozen=True)
class BackendTeardownProof:
    container_id: str
    cause: Literal["normal_exit", "containment_fault"]
    proof_sha256: str
    container_inspect_absent: bool
    container_top_absent: bool
    identity_query_empty: bool
    no_descendants: bool
    app_server_process_absent: bool
    app_server_process_group_absent: bool
    bridge_socket_absent: bool
    bridge_token_absent: bool
    app_server_control_absent: bool
    arena_relay_container_id: str
    arena_volume_name: str
    arena_relay_inspect_absent: Literal[True]
    arena_relay_top_absent: Literal[True]
    arena_relay_identity_query_empty: Literal[True]
    arena_volume_inspect_absent: Literal[True]
    arena_volume_identity_query_empty: Literal[True]
    arena_relay_attachment_status: Literal[
        "CLEAN_EOF", "ABORTED_CONTAINMENT"
    ]
    arena_relay_teardown_receipt_path: str
    arena_relay_teardown_receipt_sha256: str
    process_identity_authority: Literal[
        "controller_container_cgroup"
    ]
    controller_container_id: str
    egress_proxy_container_id: str
    controller_inspect_absent: Literal[True]
    controller_identity_query_empty: Literal[True]
    controller_top_absent: Literal[True]
    controller_no_descendants: Literal[True]
    egress_proxy_inspect_absent: Literal[True]
    egress_proxy_identity_query_empty: Literal[True]
    egress_proxy_top_absent: Literal[True]
    egress_proxy_no_descendants: Literal[True]
    controller_absence_receipt_sha256: str
    canary_reveal_path: str
    canary_reveal_sha256: str
    canary_cleanup_receipt_path: str | None = None
    canary_cleanup_receipt_sha256: str | None = None


@dataclass(frozen=True)
class BackendEmergencyContainment:
    """Zero-authority absence proof used only after journal exhaustion."""

    containment_receipt_path: str
    containment_receipt_sha256: str
    launched_container_id: str | None
    attempt_container_absent: Literal[True]
    controller_roles_absent: Literal[True]
    arena_resources_absent: Literal[True]
    rpc_endpoints_absent: Literal[True]
    workspace_probe_containers_absent: Literal[True]
    host_process_groups_absent: Literal[True]
    containment_canaries_absent: Literal[True]
    no_descendants: Literal[True]


@dataclass(frozen=True)
class AuxiliaryPreparedInput:
    input_manifest_path: str
    input_manifest_sha256: str
    input_bundle_receipt_path: str
    input_bundle_receipt_sha256: str


@dataclass(frozen=True)
class AuxiliaryLaunch:
    launch_receipt_path: str
    launch_receipt_sha256: str


@dataclass(frozen=True)
class AuxiliaryPoll:
    status: Literal["running", "exited", "containment_fault"]
    observation_sha256: str
    reason: str = ""


@dataclass(frozen=True)
class AuxiliaryCollection:
    output: Scheduler.AuxiliaryOutputEvidence | None
    cost_used: float
    abort_reason: str | None = None


@dataclass(frozen=True)
class AuxiliaryTeardown:
    teardown_receipt_path: str
    teardown_receipt_sha256: str


@dataclass(frozen=True)
class AuxiliaryAdmission:
    verdict: Literal["ADMITTED", "REJECTED"]
    profile: Scheduler.ComplexityProfile | None
    reason: str | None
    fresh_replay_receipt_path: str | None
    fresh_replay_receipt_sha256: str | None
    taint_receipt_path: str | None
    taint_receipt_sha256: str | None
    provenance_receipt_path: str | None
    provenance_receipt_sha256: str | None
    admission_receipt_path: str
    admission_receipt_sha256: str


@dataclass(frozen=True)
class AuxiliaryAbort:
    cost_used: float
    teardown: AuxiliaryTeardown | None


class AuxiliaryBackend(Protocol):
    """Idempotent private sidecar and host-only admission control plane.

    Every method is keyed by the decision's durable assignment identity.
    Recovery may repeat a method after its external effect but before journal
    acknowledgement.  ``prepare`` must not run before ``AUXILIARY_RESERVED``;
    it materializes only the committed immutable manifest.  The sidecar has no
    live-lineage write or promotion authority.  ``admit`` executes outside the
    sidecar and returns host-only fresh-replay/taint/provenance receipts.
    """

    backend_contract_sha256: str
    input_bundle_contract_sha256: str
    admission_contract_sha256: str
    production_isolation_attested: bool
    immutable_private_input_attested: bool
    host_admission_attested: bool
    descriptor_confined_receipts_attested: bool

    def read_confined_receipt(
        self,
        decision: Scheduler.AuxiliaryDecision,
        path_value: str,
        *,
        maximum: int,
    ) -> bytes:
        """Read once below the immutable assignment-root descriptor."""
        ...

    def prepare(
        self, decision: Scheduler.AuxiliaryDecision
    ) -> AuxiliaryPreparedInput:
        ...

    def launch(
        self,
        decision: Scheduler.AuxiliaryDecision,
        prepared: AuxiliaryPreparedInput,
    ) -> AuxiliaryLaunch:
        ...

    def poll(
        self,
        decision: Scheduler.AuxiliaryDecision,
        prepared: AuxiliaryPreparedInput,
        launched: AuxiliaryLaunch,
        *,
        timeout_seconds: float,
    ) -> AuxiliaryPoll:
        ...

    def collect(
        self,
        decision: Scheduler.AuxiliaryDecision,
        prepared: AuxiliaryPreparedInput,
        launched: AuxiliaryLaunch,
        terminal: AuxiliaryPoll,
    ) -> AuxiliaryCollection:
        ...

    def teardown(
        self,
        decision: Scheduler.AuxiliaryDecision,
        prepared: AuxiliaryPreparedInput,
        launched: AuxiliaryLaunch,
        collection: AuxiliaryCollection,
    ) -> AuxiliaryTeardown:
        ...

    def admit(
        self,
        decision: Scheduler.AuxiliaryDecision,
        output: Scheduler.AuxiliaryOutputEvidence,
    ) -> AuxiliaryAdmission:
        ...

    def abort(
        self,
        decision: Scheduler.AuxiliaryDecision,
        prepared: AuxiliaryPreparedInput | None,
        launched: AuxiliaryLaunch | None,
        *,
        prior_phase: Literal["RESERVED", "INPUT_PREPARED", "RUNNING"],
        reason: str,
    ) -> AuxiliaryAbort:
        ...


class AttemptBackend(Protocol):
    """Bounded adapter for a fresh isolated attempt container.

    ``launch`` must be idempotent by ``attempt_id``: recovery may call it after a
    crash between external launch and journal acknowledgement. ``poll`` must
    return within ``timeout_seconds``.  The scheduler never calls a kill method
    at a soft deadline; an implementation's hard safety bound and cgroup teardown
    remain explicit backend responsibilities.
    """

    def prepare(self, spec: AttemptSpec) -> BackendPreparation:
        ...

    def launch(
        self, spec: AttemptSpec, prepared: BackendPreparation
    ) -> BackendLaunch:
        ...

    def probe_substrate_health(
        self,
        *,
        spec: AttemptSpec,
        prepared: BackendPreparation,
        authorization_id: str,
        authorization_receipt_sha256: str,
        probe_index: int,
        failed_substrate_identity_sha256: str,
        incident_failure_receipt_sha256: str,
    ) -> BackendSubstrateHealthProbe:
        """Execute one circuit-authorized real controller-only health probe."""
        ...

    def poll(
        self,
        *,
        spec: AttemptSpec,
        prepared: BackendPreparation,
        launched: BackendLaunch,
        timeout_seconds: float,
    ) -> BackendPoll:
        ...

    def collect(
        self,
        *,
        spec: AttemptSpec,
        prepared: BackendPreparation,
        launched: BackendLaunch,
        terminal: BackendPoll,
    ) -> BackendCollection:
        ...

    def teardown(
        self,
        *,
        spec: AttemptSpec,
        prepared: BackendPreparation,
        launched: BackendLaunch,
        cause: Literal["normal_exit", "containment_fault"],
    ) -> BackendTeardownProof:
        ...

    def emergency_contain(
        self,
        *,
        spec: AttemptSpec,
        prepared: BackendPreparation | None,
        launched: BackendLaunch | None,
        prior_phase: str,
        reason: Literal["journal_or_storage_exhausted"],
    ) -> BackendEmergencyContainment:
        """Prove exact absence without collecting or granting authority."""
        ...


class PromotionGate(Protocol):
    """Trusted idempotent host verifier and immutable artifact publisher."""

    def commit(
        self, *, spec: AttemptSpec, candidate: PromotionCandidate
    ) -> PromotionCommit:
        ...

    def recover(
        self, *, spec: AttemptSpec, candidate: PromotionCandidate
    ) -> PromotionCommit | None:
        """Reconcile an ambiguous acknowledgement without republishing."""
        ...


def frontier_sha256(
    game: str, reached: int, parent_checkpoint_sha256: str
) -> str:
    if not _is_sha256(parent_checkpoint_sha256):
        raise ContiguousRunnerError("invalid parent checkpoint hash")
    return hashlib.sha256(
        _canonical_json(
            {
                "game": game,
                "reached": reached,
                "parent_checkpoint_sha256": parent_checkpoint_sha256,
            }
        )
    ).hexdigest()


@dataclass(frozen=True)
class FrontierRetryPolicy:
    """One versioned row of the canonical same-frontier retry schedule."""

    schema: Literal[1]
    no_progress: int
    effort: Literal["medium", "high", "xhigh", "max"]
    soft_allocation_seconds: int
    wip_mode: Literal["exclude", "restore_clean_same_frontier"]
    coherence_reset: bool


def frontier_retry_policy(no_progress: int) -> FrontierRetryPolicy:
    """Compatibility projection delegated to the canonical scheduler."""

    try:
        policy = Scheduler.retry_policy(no_progress)
    except Scheduler.SchedulerError as exc:
        raise ContiguousRunnerError(str(exc)) from exc
    return FrontierRetryPolicy(
        schema=policy.schema,
        no_progress=policy.no_progress,
        effort=policy.effort,
        soft_allocation_seconds=policy.soft_allocation_seconds,
        wip_mode=policy.requested_wip_mode,
        coherence_reset=policy.coherence_reset,
    )


def advance_exact_frontier_clean_no_progress(
    no_progress: int, outcome: str
) -> int:
    """Compatibility projection of the canonical scheduler transition."""

    try:
        return Scheduler.advance_retry_coordinate(no_progress, outcome)
    except Scheduler.SchedulerError as exc:
        raise ContiguousRunnerError(
            "exact-frontier retry coordinate transition is malformed"
        ) from exc


def apply_terminal_result_precedence(
    terminal_status: object,
    result: AttemptResult,
) -> AttemptResult:
    """Prevent containment infrastructure from becoming solver authority.

    A containment fault is authenticated by the host-side backend lifecycle
    and outranks proposer result labels. Taint remains a separate quarantine
    verdict; every other solver-labelled result is settled as infrastructure,
    with authenticated usage preserved and candidate/WIP authority removed.
    """

    if terminal_status not in {
        "running", "exited", "containment_fault"
    }:
        raise ContiguousRunnerError(
            "terminal result precedence received an invalid status"
        )
    if not isinstance(result, AttemptResult):
        raise ContiguousRunnerError(
            "terminal result precedence received an invalid result"
        )
    if terminal_status != "containment_fault":
        return result
    if result.kind in {
        "tainted", "protocol_invalid", "infrastructure"
    }:
        return AttemptResult(
            kind=result.kind,
            cost_used=result.cost_used,
            reason=result.reason,
        )
    return AttemptResult(
        kind="infrastructure",
        cost_used=result.cost_used,
        reason=(
            "containment fault superseded "
            f"{result.kind}"
            + (f": {result.reason}" if result.reason else "")
        ),
    )


def escalation(no_progress: int) -> tuple[str, int]:
    """Compatibility projection derived from ``frontier_retry_policy``."""

    policy = frontier_retry_policy(no_progress)
    return policy.effort, policy.soft_allocation_seconds


def should_restore_wip(no_progress: int) -> bool:
    """Compatibility projection derived from ``frontier_retry_policy``."""

    return (
        frontier_retry_policy(no_progress).wip_mode
        == "restore_clean_same_frontier"
    )


def _wip_to_dict(wip: WipSnapshot | None) -> dict[str, Any] | None:
    return (
        Scheduler.wip_binding_to_dict(wip)
        if wip is not None
        else None
    )


def _wip_publication_fields(
    wip: WipSnapshot,
) -> dict[str, Any]:
    """Return the terminal publication payload without its self-reference."""

    fields = asdict(wip)
    fields.pop("wip_publication_receipt_path")
    fields.pop("wip_publication_receipt_sha256")
    return fields


def _wip_from_dict(value: object) -> WipSnapshot | None:
    if value is None:
        return None
    try:
        return Scheduler.wip_binding_from_dict(value)
    except Scheduler.SchedulerError as exc:
        raise ContiguousRunnerError("invalid WIP descriptor") from exc


def _transport_from_dict(
    value: object,
) -> ProposerTransportConfiguration:
    if not isinstance(value, dict):
        raise ContiguousRunnerError(
            "proposer_transport must be an object"
        )
    fields = dict(value)
    for name in (
        "reasoning_effort_allowlist",
        "controller_entrypoint",
        "controller_preflight_request_allowlist",
        "controller_preflight_notification_allowlist",
        "controller_turn_request_allowlist",
        "dynamic_tool_names",
        "bridge_operation_allowlist",
        "bridge_exec_allowlist",
    ):
        item = fields.get(name)
        if not isinstance(item, list):
            raise ContiguousRunnerError(
                f"proposer_transport {name} must be a JSON list"
            )
        fields[name] = tuple(item)
    try:
        transport = ProposerTransportConfiguration(**fields)
    except TypeError as exc:
        raise ContiguousRunnerError(
            "proposer_transport schema mismatch"
        ) from exc
    positive_bounds = (
        transport.bridge_max_request_bytes,
        transport.bridge_max_response_bytes,
        transport.bridge_max_file_bytes,
        transport.bridge_max_total_export_bytes,
        transport.bridge_max_processes,
        transport.bridge_max_exec_seconds,
    )
    controller_image = transport.controller_image_reference
    controller_digest = transport.controller_image_digest
    relay_image = transport.arena_relay_image_reference
    relay_digest = transport.arena_relay_image_digest
    if (
        transport.model != "gpt-5.6-sol"
        or transport.model_provider != "openai"
        or transport.allow_provider_model_fallback is not False
        or transport.reasoning_effort_allowlist
        != EXPECTED_REASONING_EFFORT_ALLOWLIST
        or not isinstance(controller_image, str)
        or not isinstance(controller_digest, str)
        or controller_image.count("@") != 1
        or not controller_image.endswith("@" + controller_digest)
        or re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9./:_-]*@sha256:[0-9a-f]{64}",
            controller_image,
        )
        is None
        or re.fullmatch(
            r"sha256:[0-9a-f]{64}", controller_digest
        )
        is None
        or transport.controller_entrypoint
        != EXPECTED_CONTROLLER_ENTRYPOINT
        or transport.controller_guardian_path
        != EXPECTED_CONTROLLER_ENTRYPOINT[0]
        or not _is_sha256(
            transport.controller_guardian_sha256
        )
        or transport.controller_user != EXPECTED_CONTROLLER_USER
        or transport.controller_egress_policy
        != EXPECTED_CONTROLLER_EGRESS_POLICY
        or transport.controller_egress_proxy_image_reference.count(
            "@"
        )
        != 1
        or not transport.controller_egress_proxy_image_reference.endswith(
            "@" + transport.controller_egress_proxy_image_digest
        )
        or re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9./:_-]*@sha256:[0-9a-f]{64}",
            transport.controller_egress_proxy_image_reference,
        )
        is None
        or re.fullmatch(
            r"sha256:[0-9a-f]{64}",
            transport.controller_egress_proxy_image_digest,
        )
        is None
        or not _is_sha256(
            transport.controller_egress_policy_sha256
        )
        or not _is_finite_number(transport.controller_cpus)
        or not 0 < float(transport.controller_cpus) <= 64
        or not isinstance(
            transport.controller_memory_bytes, int
        )
        or isinstance(transport.controller_memory_bytes, bool)
        or not 64 * 1024 * 1024
        <= transport.controller_memory_bytes
        <= 1024 * 1024 * 1024 * 1024
        or not isinstance(transport.controller_pids, int)
        or isinstance(transport.controller_pids, bool)
        or not 16 <= transport.controller_pids <= 4096
        or not isinstance(transport.controller_tmpfs_bytes, int)
        or isinstance(transport.controller_tmpfs_bytes, bool)
        or not 16 * 1024 * 1024
        <= transport.controller_tmpfs_bytes
        <= 64 * 1024 * 1024 * 1024
        or transport.arena_transport
        != "docker-attach-stdio+named-volume-unix"
        or not isinstance(relay_image, str)
        or not isinstance(relay_digest, str)
        or relay_image.count("@") != 1
        or not relay_image.endswith("@" + relay_digest)
        or re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9./:_-]*@sha256:[0-9a-f]{64}",
            relay_image,
        )
        is None
        or re.fullmatch(r"sha256:[0-9a-f]{64}", relay_digest)
        is None
        or not _is_sha256(transport.arena_relay_source_sha256)
        or not _safe_path_string(transport.codex_launcher_path)
        or not Path(transport.codex_launcher_path).is_absolute()
        or not _is_sha256(transport.codex_launcher_sha256)
        or not _safe_path_string(
            transport.codex_package_manifest_path
        )
        or not Path(
            transport.codex_package_manifest_path
        ).is_absolute()
        or not _is_sha256(
            transport.codex_package_manifest_sha256
        )
        or not _safe_path_string(transport.codex_binary_path)
        or not Path(transport.codex_binary_path).is_absolute()
        or not _is_sha256(transport.codex_binary_sha256)
        or not isinstance(transport.codex_binary_bytes, int)
        or isinstance(transport.codex_binary_bytes, bool)
        or not 1
        <= transport.codex_binary_bytes
        <= Transport.MAX_PINNED_CODEX_BINARY_BYTES
        or not isinstance(transport.codex_cli_version, str)
        or not re.fullmatch(
            r"codex-cli [0-9]+(?:\.[0-9]+){2}",
            transport.codex_cli_version,
        )
        or not _safe_path_string(
            transport.app_server_protocol_schema_path
        )
        or not Path(
            transport.app_server_protocol_schema_path
        ).is_absolute()
        or not _is_sha256(
            transport.app_server_protocol_schema_sha256
        )
        or not _safe_path_string(
            transport.app_server_protocol_schema_bundle_path
        )
        or not Path(
            transport.app_server_protocol_schema_bundle_path
        ).is_absolute()
        or not _is_sha256(
            transport.app_server_protocol_schema_bundle_sha256
        )
        or transport.controller_preflight_request_allowlist
        != EXPECTED_CONTROLLER_PREFLIGHT_REQUEST_ALLOWLIST
        or transport.controller_preflight_notification_allowlist
        != EXPECTED_CONTROLLER_PREFLIGHT_NOTIFICATION_ALLOWLIST
        or transport.controller_turn_request_allowlist
        != EXPECTED_CONTROLLER_TURN_REQUEST_ALLOWLIST
        or transport.dynamic_tool_namespace
        != EXPECTED_DYNAMIC_TOOL_NAMESPACE
        or transport.dynamic_tool_names != EXPECTED_DYNAMIC_TOOL_NAMES
        or transport.bridge_protocol_version != 1
        or isinstance(transport.bridge_protocol_version, bool)
        or transport.bridge_operation_allowlist
        != EXPECTED_BRIDGE_OPERATION_ALLOWLIST
        or transport.bridge_exec_allowlist
        != EXPECTED_BRIDGE_EXEC_ALLOWLIST
        or any(
            not isinstance(item, int)
            or isinstance(item, bool)
            or item <= 0
            for item in positive_bounds
        )
        or transport.bridge_max_request_bytes > 4 * 1024 * 1024
        or transport.bridge_max_response_bytes > 4 * 1024 * 1024
        or transport.bridge_max_file_bytes > 32 * 1024 * 1024
        or transport.bridge_max_total_export_bytes
        > 128 * 1024 * 1024
        or transport.bridge_max_processes > 64
        or transport.bridge_max_exec_seconds > 60 * 60
    ):
        raise ContiguousRunnerError(
            "invalid proposer transport configuration"
        )
    return transport


def _transport_to_dict(
    transport: ProposerTransportConfiguration,
) -> dict[str, Any]:
    value = asdict(transport)
    for name in (
        "reasoning_effort_allowlist",
        "controller_entrypoint",
        "controller_preflight_request_allowlist",
        "controller_preflight_notification_allowlist",
        "controller_turn_request_allowlist",
        "dynamic_tool_names",
        "bridge_operation_allowlist",
        "bridge_exec_allowlist",
    ):
        value[name] = list(getattr(transport, name))
    return value


def _candidate_to_dict(
    candidate: PromotionCandidate | None,
) -> dict[str, Any] | None:
    return asdict(candidate) if candidate is not None else None


def _candidate_from_dict(value: object) -> PromotionCandidate | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ContiguousRunnerError("candidate descriptor must be an object")
    try:
        candidate = PromotionCandidate(**value)
    except TypeError as exc:
        raise ContiguousRunnerError(
            "candidate descriptor schema mismatch"
        ) from exc
    return candidate


def _blocker_to_dict(
    blocker: HostBlockerEvidence | None,
) -> dict[str, Any] | None:
    return asdict(blocker) if blocker is not None else None


def _blocker_from_dict(value: object) -> HostBlockerEvidence | None:
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != {
        "code",
        "receipt_path",
        "receipt_sha256",
    }:
        raise ContiguousRunnerError(
            "host blocker descriptor schema mismatch"
        )
    try:
        blocker = HostBlockerEvidence(**value)
    except TypeError as exc:
        raise ContiguousRunnerError(
            "host blocker descriptor schema mismatch"
        ) from exc
    if (
        not isinstance(blocker.code, str)
        or not _safe_path_string(blocker.receipt_path)
        or not _is_sha256(blocker.receipt_sha256)
    ):
        raise ContiguousRunnerError(
            "host blocker descriptor is malformed"
        )
    return blocker


def _native_sidecar_request_draft_to_dict(
    draft: Scheduler.NativeSidecarRequestDraft | None,
) -> dict[str, Any] | None:
    if draft is None:
        return None
    value = asdict(draft)
    value["cited_public_observation_receipt_sha256s"] = list(
        draft.cited_public_observation_receipt_sha256s
    )
    return value


def _native_sidecar_request_draft_from_dict(
    value: object,
) -> Scheduler.NativeSidecarRequestDraft | None:
    if value is None:
        return None
    try:
        return Scheduler.native_sidecar_request_draft_from_dict(
            value
        )
    except Scheduler.SchedulerError as exc:
        raise ContiguousRunnerError(
            "native sidecar request draft is malformed"
        ) from exc


def _spec_from_dict(value: object) -> AttemptSpec:
    if not isinstance(value, dict):
        raise ContiguousRunnerError("attempt spec must be an object")
    fields = dict(value)
    fields["wip"] = _wip_from_dict(fields.get("wip"))
    try:
        fields["supervisory_handoff"] = (
            None
            if fields.get("supervisory_handoff") is None
            else Scheduler.supervisory_handoff_binding_from_dict(
                fields["supervisory_handoff"]
            )
        )
    except Scheduler.SchedulerError as exc:
        raise ContiguousRunnerError(
            "attempt supervisory handoff is malformed"
        ) from exc
    fields["proposer_transport"] = _transport_from_dict(
        fields.get("proposer_transport")
    )
    resource_limits = fields.get("resource_limits")
    if not isinstance(resource_limits, dict):
        raise ContiguousRunnerError("resource_limits must be an object")
    try:
        fields["resource_limits"] = ResourceLimitsProjection(
            **resource_limits
        )
    except TypeError as exc:
        raise ContiguousRunnerError(
            "resource_limits schema mismatch"
        ) from exc
    worker_command = fields.get("worker_command")
    if not isinstance(worker_command, list):
        raise ContiguousRunnerError("worker_command must be a JSON list")
    fields["worker_command"] = tuple(worker_command)
    try:
        spec = AttemptSpec(**fields)
    except TypeError as exc:
        raise ContiguousRunnerError("attempt spec schema mismatch") from exc
    if (
        spec.schema != RUNNER_SCHEMA
        or not _is_uuid4(spec.campaign_id)
        or not _is_uuid4(spec.generation_id)
        or not _is_uuid4(spec.attempt_id)
        or spec.generation_id == spec.attempt_id
        or not isinstance(spec.target_level, int)
        or isinstance(spec.target_level, bool)
        or not isinstance(spec.authoritative_target, int)
        or isinstance(spec.authoritative_target, bool)
        or not 1 <= spec.target_level <= spec.authoritative_target
        or not _is_sha256(spec.parent_checkpoint_sha256)
        or not _is_sha256(spec.frontier_sha256)
        or not _is_sha256(spec.input_tree_sha256)
        or not _safe_path_string(spec.parent_source_path)
        or not Path(spec.parent_source_path).is_absolute()
        or not _is_sha256(spec.parent_source_tree_sha256)
        or not _is_sha256(spec.initial_workspace_tree_sha256)
        or not _is_sha256(
            spec.initial_app_server_state_tree_sha256
        )
        or spec.hard_safety_seconds
        != Taint.APP_SERVER_HARD_SAFETY_SECONDS
        or spec.max_auth_refreshes != Taint.MAX_AUTH_REFRESHES
        or not _is_sha256(spec.input_bundle_receipt_sha256)
        or not _is_sha256(spec.frontier_brief_sha256)
        or (
            spec.supervisory_handoff is None
            and any(
                item is not None
                for item in (
                    spec.supervisory_handoff_path,
                    spec.supervisory_handoff_sha256,
                    spec.supervisory_handoff_binding_receipt_path,
                    spec.supervisory_handoff_binding_receipt_sha256,
                )
            )
        )
        or (
            spec.supervisory_handoff is not None
            and (
                not _safe_path_string(
                    spec.supervisory_handoff_path
                )
                or not _is_sha256(
                    spec.supervisory_handoff_sha256
                )
                or not _safe_path_string(
                    spec.supervisory_handoff_binding_receipt_path
                )
                or not _is_sha256(
                    spec
                    .supervisory_handoff_binding_receipt_sha256
                )
            )
        )
        or not _is_sha256(spec.bridge_policy_sha256)
        or not isinstance(spec.parent_action_count, int)
        or isinstance(spec.parent_action_count, bool)
        or not 0 <= spec.parent_action_count <= 600
        or not isinstance(spec.remaining_action_budget, int)
        or isinstance(spec.remaining_action_budget, bool)
        or spec.remaining_action_budget
        != 600 - spec.parent_action_count
        or not isinstance(spec.fresh_prefix_required, bool)
        or spec.fresh_prefix_required
        != (spec.remaining_action_budget == 0)
        or not _valid_backend_configuration(
            BackendConfiguration(
                image_reference=spec.image_reference,
                image_digest=spec.image_digest,
                worker_command=spec.worker_command,
                resource_limits=spec.resource_limits,
                proposer_transport=spec.proposer_transport,
            )
        )
        or spec.effort not in {"medium", "high", "xhigh", "max"}
        or not isinstance(spec.soft_allocation_seconds, int)
        or isinstance(spec.soft_allocation_seconds, bool)
        or spec.soft_allocation_seconds <= 0
        or spec.wip_mode
        not in {"exclude", "restore_clean_same_frontier"}
        or (spec.wip_mode == "exclude") != (spec.wip is None)
        or spec.thread_mode not in {"new", "resume"}
        or (spec.thread_mode == "new")
        != (spec.resume_thread_id is None)
        or (spec.thread_mode == "new")
        != (spec.resume_thread_binding_sha256 is None)
        or (
            spec.thread_mode == "resume"
            and (
                spec.wip_mode != "restore_clean_same_frontier"
                or spec.wip is None
                or not _is_canonical_uuid(spec.resume_thread_id)
                or not _is_sha256(
                    spec.resume_thread_binding_sha256
                )
                or spec.resume_thread_id
                != spec.wip.codex_thread_id
                or spec.resume_thread_binding_sha256
                != spec.wip.final_thread_binding_sha256
            )
        )
        or (
            spec.thread_mode == "new"
            and spec.wip_mode != "exclude"
        )
        or (
            spec.cost_limit_remaining is not None
            and (
                not _is_finite_number(spec.cost_limit_remaining)
                or spec.cost_limit_remaining < 0
            )
        )
        or (
            spec.supervisory_handoff is not None
            and (
                spec.supervisory_handoff.frontier_sha256
                != spec.frontier_sha256
                or spec.supervisory_handoff
                .parent_checkpoint_sha256
                != spec.parent_checkpoint_sha256
                or spec.supervisory_handoff.prompt_authority
                != "unverified_hypothesis_only"
                or spec.supervisory_handoff.scheduler_authority
                is not False
                or spec.supervisory_handoff.mutation_authority
                is not False
                or spec.supervisory_handoff.promotion_authority
                is not False
            )
        )
    ):
        raise ContiguousRunnerError("invalid attempt spec")
    generation = Path(spec.generation_dir)
    expected_children = {
        "input": Path(spec.input_dir),
        "scratch": Path(spec.scratch_dir),
        "output": Path(spec.output_dir),
        "rpc": Path(spec.arena_socket_path).parent,
        "bridge": Path(spec.bridge_dir),
        "host": Path(spec.host_transcript_path).parent,
        "state": Path(spec.app_server_state_dir).parent,
    }
    if (
        generation.name != spec.generation_id
        or not all(
            Path(path).is_absolute()
            for path in (
                spec.parent_checkpoint_path,
                spec.generation_dir,
                spec.input_dir,
                spec.scratch_dir,
                spec.workspace_dir,
                spec.output_dir,
                spec.arena_socket_path,
                spec.arena_token_file_path,
                spec.bridge_dir,
                spec.bridge_socket_path,
                spec.bridge_token_file_path,
                spec.bridge_policy_receipt_path,
                spec.host_transcript_path,
                spec.app_server_transcript_path,
                spec.neutral_host_cwd_path,
                spec.app_server_state_dir,
                spec.app_server_control_dir,
                spec.input_bundle_receipt_path,
                spec.frontier_brief_path,
                *(
                    (
                        spec.supervisory_handoff_path,
                        spec
                        .supervisory_handoff_binding_receipt_path,
                    )
                    if spec.supervisory_handoff is not None
                    else ()
                ),
                spec.bridge_policy_path,
            )
        )
        or any(
            path.parent != generation or path.name != name
            for name, path in expected_children.items()
        )
        or Path(spec.arena_token_file_path).parent
        != Path(spec.arena_socket_path).parent
        or Path(spec.arena_token_file_path).name != "token"
        or Path(spec.arena_socket_path).name != "arena.sock"
        or Path(spec.workspace_dir) != Path(spec.scratch_dir)
        or Path(spec.bridge_socket_path).parent
        != Path(spec.bridge_dir)
        or Path(spec.bridge_token_file_path).parent
        != Path(spec.bridge_dir)
        or Path(spec.bridge_socket_path).name != "proposer.sock"
        or Path(spec.bridge_token_file_path).name
        != "proposer-token"
        or Path(spec.bridge_policy_receipt_path).parent
        != Path(spec.host_transcript_path).parent
        or Path(spec.bridge_policy_receipt_path).name
        != "bridge_policy_receipt.json"
        or Path(spec.host_transcript_path).name != "backend.jsonl"
        or Path(spec.frontier_brief_path)
        != Path(spec.input_dir) / "frontier_brief.json"
        or (
            spec.supervisory_handoff is not None
            and (
                Path(str(spec.supervisory_handoff_path))
                != Path(spec.input_dir)
                / "supervisory_handoff.json"
                or Path(
                    str(
                        spec
                        .supervisory_handoff_binding_receipt_path
                    )
                )
                != Path(spec.input_dir)
                / "supervisory_handoff_binding_receipt.json"
            )
        )
        or Path(spec.bridge_policy_path)
        != Path(spec.input_dir) / "bridge_policy.json"
        or Path(spec.app_server_transcript_path).parent
        != Path(spec.host_transcript_path).parent
        or Path(spec.app_server_transcript_path).name
        != "app_server.jsonl"
        or Path(spec.neutral_host_cwd_path).parent
        != Path(spec.host_transcript_path).parent
        or Path(spec.neutral_host_cwd_path).name != "neutral"
        or Path(spec.app_server_state_dir).name != "codex_home"
        or Path(spec.app_server_state_dir).parent.name != "state"
        or Path(spec.app_server_state_dir).parent.parent
        != generation
        or Path(spec.app_server_control_dir).parent
        != Path(spec.host_transcript_path).parent
        or Path(spec.app_server_control_dir).name
        != "app_server_control"
        or not _safe_path_string(spec.input_bundle_receipt_path)
        or not all(
            _safe_path_string(path)
            for path in (
                spec.parent_checkpoint_path,
                spec.generation_dir,
                spec.input_dir,
                spec.scratch_dir,
                spec.workspace_dir,
                spec.output_dir,
                spec.arena_socket_path,
                spec.arena_token_file_path,
                spec.bridge_dir,
                spec.bridge_socket_path,
                spec.bridge_token_file_path,
                spec.bridge_policy_receipt_path,
                spec.host_transcript_path,
                spec.app_server_transcript_path,
                spec.neutral_host_cwd_path,
                spec.app_server_state_dir,
                spec.app_server_control_dir,
                spec.frontier_brief_path,
                *(
                    (
                        str(spec.supervisory_handoff_path),
                        str(
                            spec
                            .supervisory_handoff_binding_receipt_path
                        ),
                    )
                    if spec.supervisory_handoff is not None
                    else ()
                ),
                spec.bridge_policy_path,
            )
        )
    ):
        raise ContiguousRunnerError("attempt paths do not match generation layout")
    return spec


def _spec_to_dict(spec: AttemptSpec) -> dict[str, Any]:
    value = asdict(spec)
    value["wip"] = _wip_to_dict(spec.wip)
    value["supervisory_handoff"] = (
        None
        if spec.supervisory_handoff is None
        else Scheduler.supervisory_handoff_binding_to_dict(
            spec.supervisory_handoff
        )
    )
    value["proposer_transport"] = _transport_to_dict(
        spec.proposer_transport
    )
    return value


def proposer_attempt_binding_sha256(spec: AttemptSpec) -> str:
    """Bind every immutable proposer, frontier, transport, and path input."""
    return hashlib.sha256(
        _canonical_json(_spec_to_dict(spec))
    ).hexdigest()


def _arena_volume_name(spec: AttemptSpec) -> str:
    """Derive the one Docker volume identity bound to this attempt."""

    return (
        "arc-agi3-arena-"
        f"{spec.generation_id.replace('-', '')[:12]}-"
        f"{spec.attempt_id.replace('-', '')}"
    )


def _validate_bound_receipt(
    path_value: str,
    digest: str,
    *,
    expected_path: Path,
    expected_kind: str,
    spec: AttemptSpec,
) -> dict[str, Any]:
    path = Path(path_value)
    if (
        path != expected_path
        or not path.is_absolute()
        or path.is_symlink()
        or _sha256_file(path) != digest
    ):
        raise ContiguousRunnerError(
            f"{expected_kind} receipt path/hash mismatch"
        )
    value = _read_json_file(path)
    required_binding = {
        "schema": 1,
        "kind": expected_kind,
        "campaign_id": spec.campaign_id,
        "generation_id": spec.generation_id,
        "attempt_id": spec.attempt_id,
        "attempt_spec_sha256": proposer_attempt_binding_sha256(spec),
    }
    if (
        not isinstance(value, dict)
        or any(value.get(key) != expected
               for key, expected in required_binding.items())
    ):
        raise ContiguousRunnerError(
            f"{expected_kind} receipt does not bind the attempt spec"
        )
    return value


def _reservation_from_dict(value: object) -> AttemptReservation:
    if not isinstance(value, dict):
        raise ContiguousRunnerError("attempt reservation must be an object")
    fields = dict(value)
    fields["wip"] = _wip_from_dict(fields.get("wip"))
    try:
        fields["supervisory_handoff"] = (
            None
            if fields.get("supervisory_handoff") is None
            else Scheduler.supervisory_handoff_binding_from_dict(
                fields["supervisory_handoff"]
            )
        )
    except Scheduler.SchedulerError as exc:
        raise ContiguousRunnerError(
            "reservation supervisory handoff is malformed"
        ) from exc
    fields["proposer_transport"] = _transport_from_dict(
        fields.get("proposer_transport")
    )
    limits = fields.get("resource_limits")
    if not isinstance(limits, dict):
        raise ContiguousRunnerError(
            "reservation resource_limits must be an object"
        )
    try:
        fields["resource_limits"] = ResourceLimitsProjection(**limits)
    except TypeError as exc:
        raise ContiguousRunnerError(
            "reservation resource_limits schema mismatch"
        ) from exc
    command = fields.get("worker_command")
    if not isinstance(command, list):
        raise ContiguousRunnerError(
            "reservation worker_command must be a JSON list"
        )
    fields["worker_command"] = tuple(command)
    try:
        reservation = AttemptReservation(**fields)
    except TypeError as exc:
        raise ContiguousRunnerError(
            "attempt reservation schema mismatch"
        ) from exc
    common = {
        name: getattr(reservation, name)
        for name in AttemptReservation.__dataclass_fields__
    }
    provisional = AttemptSpec(
        **common,
        input_tree_sha256="0" * 64,
        initial_workspace_tree_sha256="0" * 64,
        initial_app_server_state_tree_sha256="0" * 64,
        hard_safety_seconds=Taint.APP_SERVER_HARD_SAFETY_SECONDS,
        max_auth_refreshes=Taint.MAX_AUTH_REFRESHES,
        input_bundle_receipt_path=str(
            Path(reservation.generation_dir)
            / "input_bundle_receipt.json"
        ),
        input_bundle_receipt_sha256="0" * 64,
        frontier_brief_path=str(
            Path(reservation.input_dir) / "frontier_brief.json"
        ),
        frontier_brief_sha256="0" * 64,
        supervisory_handoff_path=(
            str(
                Path(reservation.input_dir)
                / "supervisory_handoff.json"
            )
            if reservation.supervisory_handoff is not None
            else None
        ),
        supervisory_handoff_sha256=(
            "0" * 64
            if reservation.supervisory_handoff is not None
            else None
        ),
        supervisory_handoff_binding_receipt_path=(
            str(
                Path(reservation.input_dir)
                / "supervisory_handoff_binding_receipt.json"
            )
            if reservation.supervisory_handoff is not None
            else None
        ),
        supervisory_handoff_binding_receipt_sha256=(
            "0" * 64
            if reservation.supervisory_handoff is not None
            else None
        ),
        bridge_policy_path=str(
            Path(reservation.input_dir) / "bridge_policy.json"
        ),
        bridge_policy_sha256="0" * 64,
        parent_action_count=0,
        remaining_action_budget=600,
        fresh_prefix_required=False,
    )
    encoded = asdict(provisional)
    encoded["worker_command"] = list(provisional.worker_command)
    encoded["wip"] = _wip_to_dict(provisional.wip)
    encoded["proposer_transport"] = _transport_to_dict(
        provisional.proposer_transport
    )
    _spec_from_dict(encoded)
    return reservation


def _reservation_to_dict(
    reservation: AttemptReservation,
) -> dict[str, Any]:
    value = asdict(reservation)
    value["wip"] = _wip_to_dict(reservation.wip)
    value["supervisory_handoff"] = (
        None
        if reservation.supervisory_handoff is None
        else Scheduler.supervisory_handoff_binding_to_dict(
            reservation.supervisory_handoff
        )
    )
    value["proposer_transport"] = _transport_to_dict(
        reservation.proposer_transport
    )
    return value


def _valid_backend_configuration(config: BackendConfiguration) -> bool:
    if not isinstance(config, BackendConfiguration) or not isinstance(
        config.resource_limits, ResourceLimitsProjection
    ):
        return False
    try:
        transport = _transport_from_dict(
            _transport_to_dict(config.proposer_transport)
        )
    except (ContiguousRunnerError, TypeError):
        return False
    limits = config.resource_limits
    return bool(
        isinstance(config.image_reference, str)
        and isinstance(config.image_digest, str)
        and config.image_reference.endswith("@" + config.image_digest)
        and config.image_reference.count("@") == 1
        and re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9./:_-]*@sha256:[0-9a-f]{64}",
            config.image_reference,
        )
        and re.fullmatch(r"sha256:[0-9a-f]{64}", config.image_digest)
        and isinstance(config.worker_command, tuple)
        and config.worker_command == EXPECTED_WORKER_COMMAND
        and transport == config.proposer_transport
        and all(
            isinstance(part, str)
            and 0 < len(part) <= 1024
            and "\x00" not in part
            for part in config.worker_command
        )
        and _is_finite_number(limits.cpus)
        and 0 < float(limits.cpus) <= 64
        and isinstance(limits.memory_bytes, int)
        and not isinstance(limits.memory_bytes, bool)
        and 64 * 1024 * 1024 <= limits.memory_bytes
        <= 1024 * 1024 * 1024 * 1024
        and isinstance(limits.pids, int)
        and not isinstance(limits.pids, bool)
        and 16 <= limits.pids <= 4096
        and isinstance(limits.tmpfs_bytes, int)
        and not isinstance(limits.tmpfs_bytes, bool)
        and 16 * 1024 * 1024 <= limits.tmpfs_bytes
        <= 64 * 1024 * 1024 * 1024
    )


def _backend_configuration_to_dict(
    config: BackendConfiguration,
) -> dict[str, Any]:
    return {
        "image_reference": config.image_reference,
        "image_digest": config.image_digest,
        "worker_command": list(config.worker_command),
        "resource_limits": asdict(config.resource_limits),
        "proposer_transport": _transport_to_dict(
            config.proposer_transport
        ),
    }


def _backend_preparation_from_dict(value: object) -> BackendPreparation:
    if not isinstance(value, dict):
        raise ContiguousRunnerError("backend preparation must be an object")
    try:
        prepared = BackendPreparation(**value)
    except TypeError as exc:
        raise ContiguousRunnerError(
            "backend preparation schema mismatch"
        ) from exc
    try:
        canary_commitments = json.loads(
            prepared.controller_canary_commitments_json
        )
        canary_placements = json.loads(
            prepared.controller_canary_placement_descriptors_json
        )
    except (TypeError, UnicodeError, json.JSONDecodeError) as exc:
        raise ContiguousRunnerError(
            "backend canary anchor rows are not JSON"
        ) from exc
    expected_categories = sorted(Taint.CONTROLLER_CANARY_CATEGORIES)
    simple_canary_placement_fields = {
        "category",
        "location_name",
        "provenance",
    }
    full_canary_placement_fields = {
        "category",
        "placement_kind",
        "location_name",
        "device",
        "inode",
        "mode",
        "owner_uid",
        "owner_gid",
        "size",
        "environment_owner_pid",
        "provenance",
        "commitment_sha256",
    }
    canary_rows_valid = (
        isinstance(canary_commitments, list)
        and isinstance(canary_placements, list)
        and len(canary_commitments) == len(expected_categories)
        and len(canary_placements) == len(expected_categories)
        and _canonical_json(canary_commitments).decode("ascii")
        == prepared.controller_canary_commitments_json
        and _canonical_json(canary_placements).decode("ascii")
        == prepared.controller_canary_placement_descriptors_json
        and all(
            isinstance(commitment, dict)
            and set(commitment)
            == {
                "category",
                "location_name",
                "provenance",
                "commitment_sha256",
            }
            and isinstance(placement, dict)
            and frozenset(placement)
            in {
                frozenset(simple_canary_placement_fields),
                frozenset(full_canary_placement_fields),
            }
            and commitment["category"] == category
            and all(
                placement.get(key) == commitment[key]
                for key in (
                    "category",
                    "location_name",
                    "provenance",
                )
            )
            and (
                frozenset(placement)
                == frozenset(simple_canary_placement_fields)
                or (
                    placement.get("commitment_sha256")
                    == commitment["commitment_sha256"]
                    and placement.get("placement_kind")
                    in {
                        "host_file",
                        "credential_decoy_file",
                        "host_environment",
                    }
                    and all(
                        isinstance(placement.get(name), int)
                        and not isinstance(
                            placement.get(name), bool
                        )
                        and placement[name] >= 0
                        for name in (
                            "device",
                            "inode",
                            "mode",
                            "owner_uid",
                            "owner_gid",
                            "size",
                        )
                    )
                )
            )
            and commitment["provenance"] == "secrets.token_hex_32"
            and isinstance(commitment["location_name"], str)
            and 0 < len(commitment["location_name"]) <= 1024
            and "\x00" not in commitment["location_name"]
            and "\r" not in commitment["location_name"]
            and "\n" not in commitment["location_name"]
            and _is_sha256(commitment["commitment_sha256"])
            for commitment, placement, category in zip(
                canary_commitments,
                canary_placements,
                expected_categories,
                strict=True,
            )
        )
    )
    if (
        not _safe_identifier(prepared.preparation_id)
        or not _safe_path_string(prepared.launch_attestation_path)
        or not _is_sha256(prepared.launch_attestation_sha256)
        or not re.fullmatch(
            r"sha256:[0-9a-f]{64}", prepared.observed_image_digest
        )
        or not _is_sha256(prepared.image_observation_sha256)
        or not _is_sha256(prepared.container_observation_sha256)
        or not _safe_path_string(
            prepared.bridge_policy_receipt_path
        )
        or not _is_sha256(
            prepared.bridge_policy_receipt_sha256
        )
        or not _safe_path_string(
            prepared.arena_session_binding_receipt_path
        )
        or not _is_sha256(
            prepared.arena_session_binding_receipt_sha256
        )
        or prepared.arena_transport != ARENA_VOLUME_TRANSPORT
        or not isinstance(prepared.arena_volume_name, str)
        or re.fullmatch(
            r"arc-agi3-arena-[0-9a-f]{12}-[0-9a-f]{32}",
            prepared.arena_volume_name,
        )
        is None
        or not _is_sha256(
            prepared.arena_volume_observation_sha256
        )
        or not isinstance(prepared.arena_relay_container_id, str)
        or re.fullmatch(
            r"[0-9a-f]{64}", prepared.arena_relay_container_id
        )
        is None
        or not re.fullmatch(
            r"sha256:[0-9a-f]{64}",
            prepared.arena_relay_image_digest,
        )
        or not _is_sha256(
            prepared.arena_relay_image_observation_sha256
        )
        or not _is_sha256(
            prepared.arena_relay_container_observation_sha256
        )
        or not _safe_path_string(
            prepared.arena_relay_readiness_receipt_path
        )
        or not _is_sha256(
            prepared.arena_relay_readiness_receipt_sha256
        )
        or not _is_sha256(
            prepared.arena_relay_attach_argv_sha256
        )
        or not _is_sha256(
            prepared.arena_relay_socket_identity_sha256
        )
        or not _safe_path_string(
            prepared.arena_relay_preparation_receipt_path
        )
        or not _is_sha256(
            prepared.arena_relay_preparation_receipt_sha256
        )
        or prepared.probe_isolation_mode
        not in Contract.PROBE_ISOLATION_MODES
        or not _is_sha256(
            prepared.probe_isolation_evidence_sha256
        )
        or not _safe_path_string(
            prepared.neutral_cwd_attestation_path
        )
        or not _is_sha256(
            prepared.neutral_cwd_attestation_sha256
        )
        or not _safe_path_string(
            prepared.app_server_config_receipt_path
        )
        or not _is_sha256(
            prepared.app_server_config_receipt_sha256
        )
        or not _safe_path_string(
            prepared.codex_binary_receipt_path
        )
        or not _is_sha256(
            prepared.codex_binary_receipt_sha256
        )
        or not _safe_path_string(
            prepared.protocol_schema_receipt_path
        )
        or not _is_sha256(
            prepared.protocol_schema_receipt_sha256
        )
        or not re.fullmatch(
            r"sha256:[0-9a-f]{64}",
            prepared.controller_image_digest,
        )
        or not re.fullmatch(
            r"sha256:[0-9a-f]{64}",
            prepared.controller_egress_proxy_image_digest,
        )
        or not _is_sha256(
            prepared.controller_egress_policy_sha256
        )
        or not _safe_path_string(
            prepared.controller_canary_escrow_path
        )
        or not Path(
            prepared.controller_canary_escrow_path
        ).is_absolute()
        or not _is_sha256(
            prepared.controller_canary_escrow_sha256
        )
        or not _is_sha256(
            prepared.controller_canary_escrow_identity_sha256
        )
        or not _is_sha256(
            prepared.controller_canary_commitments_sha256
        )
        or hashlib.sha256(
            prepared.controller_canary_commitments_json.encode(
                "ascii"
            )
        ).hexdigest()
        != prepared.controller_canary_commitments_sha256
        or not _is_sha256(
            prepared
            .controller_canary_placement_descriptors_sha256
        )
        or hashlib.sha256(
            prepared
            .controller_canary_placement_descriptors_json
            .encode("ascii")
        ).hexdigest()
        != prepared.controller_canary_placement_descriptors_sha256
        or not canary_rows_valid
        or prepared.controller_supply_chain_unobserved_until_launch
        is not True
        or not all(
            Path(path).is_absolute()
            for path in (
                prepared.launch_attestation_path,
                prepared.bridge_policy_receipt_path,
                prepared.arena_session_binding_receipt_path,
                prepared.arena_relay_readiness_receipt_path,
                prepared.arena_relay_preparation_receipt_path,
                prepared.neutral_cwd_attestation_path,
                prepared.app_server_config_receipt_path,
                prepared.codex_binary_receipt_path,
                prepared.protocol_schema_receipt_path,
            )
        )
    ):
        raise ContiguousRunnerError("invalid backend preparation")
    return prepared


def _backend_launch_from_dict(value: object) -> BackendLaunch:
    if not isinstance(value, dict):
        raise ContiguousRunnerError("backend launch must be an object")
    try:
        launched = BackendLaunch(**value)
    except TypeError as exc:
        raise ContiguousRunnerError("backend launch schema mismatch") from exc
    if (
        not _safe_identifier(launched.backend_id)
        or not isinstance(launched.container_id, str)
        or not re.fullmatch(r"[0-9a-f]{64}", launched.container_id)
        or not _is_sha256(launched.running_observation_sha256)
        or not _is_sha256(launched.substrate_identity_sha256)
        or not _safe_path_string(
            launched.substrate_preflight_receipt_path
        )
        or not Path(
            launched.substrate_preflight_receipt_path
        ).is_absolute()
        or not _is_sha256(
            launched.substrate_preflight_receipt_sha256
        )
        or not _safe_path_string(
            launched.bridge_runtime_attestation_path
        )
        or not _is_sha256(
            launched.bridge_runtime_attestation_sha256
        )
        or not _safe_path_string(
            launched.app_server_runtime_receipt_path
        )
        or not _is_sha256(
            launched.app_server_runtime_receipt_sha256
        )
        or not isinstance(launched.app_server_pid, int)
        or isinstance(launched.app_server_pid, bool)
        or launched.app_server_pid <= 1
        or not _safe_identifier(
            launched.app_server_process_start
        )
        or not isinstance(
            launched.app_server_process_group_id, int
        )
        or isinstance(
            launched.app_server_process_group_id, bool
        )
        or launched.app_server_process_group_id <= 1
        or launched.app_server_pid_is_diagnostic is not True
        or launched.process_identity_authority
        != "controller_container_cgroup"
        or not isinstance(launched.controller_container_id, str)
        or not re.fullmatch(
            r"[0-9a-f]{64}", launched.controller_container_id
        )
        or not re.fullmatch(
            r"sha256:[0-9a-f]{64}",
            launched.controller_image_digest,
        )
        or not isinstance(
            launched.egress_proxy_container_id, str
        )
        or not re.fullmatch(
            r"[0-9a-f]{64}",
            launched.egress_proxy_container_id,
        )
        or not re.fullmatch(
            r"sha256:[0-9a-f]{64}",
            launched.egress_proxy_image_digest,
        )
        or not _is_sha256(launched.egress_policy_sha256)
        or not _is_sha256(
            launched.controller_launch_intent_sha256
        )
        or not _safe_path_string(
            launched.controller_launch_receipt_path
        )
        or not _is_sha256(
            launched.controller_launch_receipt_sha256
        )
        or not _safe_path_string(
            launched.controller_guardian_start_receipt_path
        )
        or not _is_sha256(
            launched.controller_guardian_start_receipt_sha256
        )
        or not _is_sha256(
            launched.controller_supply_chain_manifest_sha256
        )
        or not _is_canonical_uuid(launched.codex_thread_id)
        or not _is_canonical_uuid(launched.codex_turn_id)
        or not _safe_path_string(launched.thread_binding_path)
        or not _is_sha256(launched.thread_binding_sha256)
        or not _safe_path_string(
            launched.transcript_chain_receipt_path
        )
        or not _is_sha256(
            launched.transcript_chain_receipt_sha256
        )
        or not _is_sha256(launched.transcript_chain_sha256)
        or (
            (launched.thread_rebinding_receipt_path is None)
            != (
                launched.thread_rebinding_receipt_sha256
                is None
            )
        )
        or (
            launched.thread_rebinding_receipt_path is not None
            and (
                not _safe_path_string(
                    launched.thread_rebinding_receipt_path
                )
                or not _is_sha256(
                    launched.thread_rebinding_receipt_sha256
                )
                or not Path(
                    launched.thread_rebinding_receipt_path
                ).is_absolute()
            )
        )
        or not all(
            Path(path).is_absolute()
            for path in (
                launched.bridge_runtime_attestation_path,
                launched.app_server_runtime_receipt_path,
                launched.controller_launch_receipt_path,
                launched.controller_guardian_start_receipt_path,
                launched.thread_binding_path,
                launched.transcript_chain_receipt_path,
            )
        )
    ):
        raise ContiguousRunnerError("invalid backend launch")
    return launched


def _validate_preparation_receipts(
    spec: AttemptSpec, prepared: BackendPreparation
) -> None:
    host_root = Path(spec.host_transcript_path).parent
    attestation = Path(prepared.launch_attestation_path)
    escrow = Path(prepared.controller_canary_escrow_path)
    campaign_root = Path(spec.generation_dir).parent.parent
    expected_escrow = (
        campaign_root
        / "containment_canary_escrow"
        / f"{spec.generation_id}.json"
    )
    escrow_sha256, escrow_metadata = _sha256_file_identity(escrow)
    escrow_identity_sha256 = hashlib.sha256(
        _canonical_json(
            {
                "path": str(escrow),
                "device": escrow_metadata.st_dev,
                "inode": escrow_metadata.st_ino,
                "mode": stat.S_IMODE(escrow_metadata.st_mode),
                "owner_uid": escrow_metadata.st_uid,
                "owner_gid": escrow_metadata.st_gid,
                "size": escrow_metadata.st_size,
                "sha256": escrow_sha256,
            }
        )
    ).hexdigest()
    attestation_value = _read_json_file(attestation)
    expected_canary_anchor = {
        "escrow_path": prepared.controller_canary_escrow_path,
        "escrow_sha256": prepared.controller_canary_escrow_sha256,
        "escrow_identity_sha256":
            prepared.controller_canary_escrow_identity_sha256,
        "commitments_json":
            prepared.controller_canary_commitments_json,
        "commitments_sha256":
            prepared.controller_canary_commitments_sha256,
        "placement_descriptors_json":
            prepared.controller_canary_placement_descriptors_json,
        "placement_descriptors_sha256":
            prepared
            .controller_canary_placement_descriptors_sha256,
    }
    if (
        prepared.observed_image_digest != spec.image_digest
        or attestation != host_root / "launch_attestation.json"
        or _sha256_file(attestation)
        != prepared.launch_attestation_sha256
        or escrow != expected_escrow
        or escrow_metadata.st_uid != os.getuid()
        or escrow_metadata.st_nlink != 1
        or stat.S_IMODE(escrow_metadata.st_mode) != 0o400
        or escrow_sha256
        != prepared.controller_canary_escrow_sha256
        or escrow_identity_sha256
        != prepared.controller_canary_escrow_identity_sha256
        or attestation_value.get("containment_canary_anchor")
        != expected_canary_anchor
    ):
        raise ContiguousRunnerError(
            "backend launch attestation does not bind the attempt"
        )
    bridge_policy = _validate_bound_receipt(
        prepared.bridge_policy_receipt_path,
        prepared.bridge_policy_receipt_sha256,
        expected_path=Path(spec.bridge_policy_receipt_path),
        expected_kind="contiguous_bridge_policy",
        spec=spec,
    )
    arena_receipt = _validate_bound_receipt(
        prepared.arena_session_binding_receipt_path,
        prepared.arena_session_binding_receipt_sha256,
        expected_path=host_root
        / "arena_session_binding_receipt.json",
        expected_kind="contiguous_arena_session_binding",
        spec=spec,
    )
    relay_readiness_path = Path(
        prepared.arena_relay_readiness_receipt_path
    )
    relay_preparation_path = Path(
        prepared.arena_relay_preparation_receipt_path
    )
    relay_readiness_sha256, relay_readiness_metadata = (
        _sha256_file_identity(relay_readiness_path)
    )
    relay_preparation_sha256, relay_preparation_metadata = (
        _sha256_file_identity(relay_preparation_path)
    )
    relay_readiness = _read_json_file(relay_readiness_path)
    relay_preparation = _read_json_file(relay_preparation_path)
    expected_volume_name = _arena_volume_name(spec)
    relay_common = {
        "campaign_id": spec.campaign_id,
        "generation_id": spec.generation_id,
        "attempt_id": spec.attempt_id,
        "transport": ARENA_VOLUME_TRANSPORT,
    }
    if (
        prepared.arena_transport != ARENA_VOLUME_TRANSPORT
        or prepared.arena_volume_name != expected_volume_name
        or prepared.arena_relay_image_digest
        != spec.proposer_transport.arena_relay_image_digest
        or relay_readiness_path
        != host_root / "arena_volume_readiness.json"
        or relay_preparation_path
        != host_root / "arena_volume_preparation.json"
        or relay_readiness_sha256
        != prepared.arena_relay_readiness_receipt_sha256
        or relay_preparation_sha256
        != prepared.arena_relay_preparation_receipt_sha256
        or any(
            metadata.st_uid != os.getuid()
            or metadata.st_nlink != 1
            or not stat.S_ISREG(metadata.st_mode)
            or stat.S_IMODE(metadata.st_mode) != 0o600
            for metadata in (
                relay_readiness_metadata,
                relay_preparation_metadata,
            )
        )
        or set(relay_readiness)
        != {
            "schema",
            "kind",
            "status",
            "campaign_id",
            "generation_id",
            "attempt_id",
            "readiness_nonce",
            "relay_pid",
            "socket_path",
            "socket_mode",
            "network_mode_required",
            "transport",
        }
        or relay_readiness.get("schema") != 1
        or relay_readiness.get("kind")
        != "arc_agi3_arena_volume_relay_readiness"
        or relay_readiness.get("status") != "READY"
        or any(
            relay_readiness.get(name) != value
            for name, value in relay_common.items()
        )
        or not isinstance(relay_readiness.get("readiness_nonce"), str)
        or re.fullmatch(
            r"[0-9a-f]{64}",
            relay_readiness.get("readiness_nonce", ""),
        )
        is None
        or not isinstance(relay_readiness.get("relay_pid"), int)
        or isinstance(relay_readiness.get("relay_pid"), bool)
        or relay_readiness["relay_pid"] <= 0
        or relay_readiness.get("socket_path") != "/arena/arena.sock"
        or relay_readiness.get("socket_mode") != 0o666
        or relay_readiness.get("network_mode_required") != "none"
        or set(relay_preparation)
        != {
            "schema",
            "kind",
            "campaign_id",
            "generation_id",
            "attempt_id",
            "game",
            "target_level",
            "transport",
            "volume_name",
            "volume_observation_sha256",
            "relay_container_id",
            "relay_image_reference",
            "relay_image_digest",
            "relay_image_observation_sha256",
            "relay_container_observation_sha256",
            "readiness_nonce",
            "readiness_receipt_path",
            "readiness_receipt_sha256",
            "attach_argv_sha256",
            "arena_socket_identity_sha256",
        }
        or relay_preparation.get("schema") != 1
        or relay_preparation.get("kind")
        != "arc_agi3_arena_volume_preparation"
        or any(
            relay_preparation.get(name) != value
            for name, value in relay_common.items()
        )
        or relay_preparation.get("game") != spec.game
        or relay_preparation.get("target_level") != spec.target_level
        or relay_preparation.get("volume_name") != expected_volume_name
        or relay_preparation.get("volume_observation_sha256")
        != prepared.arena_volume_observation_sha256
        or relay_preparation.get("relay_container_id")
        != prepared.arena_relay_container_id
        or relay_preparation.get("relay_image_reference")
        != spec.proposer_transport.arena_relay_image_reference
        or relay_preparation.get("relay_image_digest")
        != prepared.arena_relay_image_digest
        or relay_preparation.get("relay_image_observation_sha256")
        != prepared.arena_relay_image_observation_sha256
        or relay_preparation.get(
            "relay_container_observation_sha256"
        )
        != prepared.arena_relay_container_observation_sha256
        or relay_preparation.get("readiness_nonce")
        != relay_readiness.get("readiness_nonce")
        or relay_preparation.get("readiness_receipt_path")
        != str(relay_readiness_path)
        or relay_preparation.get("readiness_receipt_sha256")
        != relay_readiness_sha256
        or relay_preparation.get("attach_argv_sha256")
        != prepared.arena_relay_attach_argv_sha256
        or relay_preparation.get("arena_socket_identity_sha256")
        != prepared.arena_relay_socket_identity_sha256
    ):
        raise ContiguousRunnerError(
            "Arena relay preparation receipt is stale or substituted"
        )
    binding_event = arena_receipt.get("binding_event")
    if (
        set(arena_receipt)
        != {
            "schema",
            "kind",
            "campaign_id",
            "generation_id",
            "attempt_id",
            "attempt_spec_sha256",
            "binding_event",
        }
        or not isinstance(binding_event, dict)
        or binding_event.get("game") != spec.game
        or binding_event.get("campaign_id") != spec.campaign_id
        or binding_event.get("generation_id") != spec.generation_id
        or binding_event.get("attempt_id") != spec.attempt_id
        or binding_event.get("target_level") != spec.target_level
        or binding_event.get("parent_level")
        != spec.target_level - 1
        or binding_event.get("parent_checkpoint_sha256")
        != spec.parent_checkpoint_sha256
        or binding_event.get("frontier_sha256")
        != spec.frontier_sha256
        or not _is_sha256(
            binding_event.get("exploration_seed_snapshot_sha256")
        )
        or not _is_sha256(
            binding_event.get("exploration_seed_path_sha256")
        )
        or not isinstance(
            binding_event.get("probe_isolation_evidence"), dict
        )
    ):
        raise ContiguousRunnerError(
            "Arena probe-isolation receipt is stale or malformed"
        )
    try:
        probe_mode, probe_digest = (
            Contract.validate_probe_isolation_evidence(
                binding_event["probe_isolation_evidence"],
                expected_seed_snapshot_sha256=binding_event[
                    "exploration_seed_snapshot_sha256"
                ],
                expected_seed_path_sha256=binding_event[
                    "exploration_seed_path_sha256"
                ],
            )
        )
    except Contract.SupervisorContractError as exc:
        raise ContiguousRunnerError(
            "Arena probe-isolation evidence failed closed"
        ) from exc
    if (
        probe_mode != prepared.probe_isolation_mode
        or probe_digest
        != prepared.probe_isolation_evidence_sha256
        or binding_event.get("probe_isolation_mode")
        != probe_mode
        or binding_event.get("probe_isolation_evidence_sha256")
        != probe_digest
        or bridge_policy.get(
            "arena_session_binding_receipt_path"
        )
        != prepared.arena_session_binding_receipt_path
        or bridge_policy.get(
            "arena_session_binding_receipt_sha256"
        )
        != prepared.arena_session_binding_receipt_sha256
        or bridge_policy.get("probe_isolation_mode") != probe_mode
        or bridge_policy.get(
            "probe_isolation_evidence_sha256"
        )
        != probe_digest
    ):
        raise ContiguousRunnerError(
            "backend preparation did not bind one controller-selected "
            "probe-isolation mode"
        )
    neutral = _validate_bound_receipt(
        prepared.neutral_cwd_attestation_path,
        prepared.neutral_cwd_attestation_sha256,
        expected_path=host_root / "neutral_cwd_attestation.json",
        expected_kind="contiguous_neutral_cwd_attestation",
        spec=spec,
    )
    neutral_path = Path(spec.neutral_host_cwd_path)
    try:
        neutral_metadata = neutral_path.stat(follow_symlinks=False)
    except OSError as exc:
        raise ContiguousRunnerError(
            "neutral host cwd is unavailable"
        ) from exc
    if (
        neutral_path.is_symlink()
        or not neutral_path.is_dir()
        or any(neutral_path.iterdir())
        or neutral.get("path") != spec.neutral_host_cwd_path
        or neutral.get("owner_uid") != neutral_metadata.st_uid
        or neutral.get("owner_gid") != neutral_metadata.st_gid
        or neutral.get("mode")
        != stat.S_IMODE(neutral_metadata.st_mode)
        or neutral.get("tree_sha256")
        != Contract._tree_hash(neutral_path)
        or neutral.get("write_probe_status") != "DENIED"
    ):
        raise ContiguousRunnerError(
            "neutral host cwd is not observed empty and read-only"
        )
    config = _validate_bound_receipt(
        prepared.app_server_config_receipt_path,
        prepared.app_server_config_receipt_sha256,
        expected_path=host_root / "app_server_config_receipt.json",
        expected_kind="contiguous_app_server_config",
        spec=spec,
    )
    expected_config = {
        "model": spec.proposer_transport.model,
        "model_provider": spec.proposer_transport.model_provider,
        "allow_provider_model_fallback": False,
        "reasoning_effort": spec.effort,
        "environments": [],
        "selected_capability_roots": [],
        "runtime_workspace_roots": ["/controller-neutral"],
        "native_proposer_workspace": {
            "root": "/controller-neutral",
            "storage": "private-tmpfs",
            "git_root_equals_workspace": True,
            "git_ceiling_directories": "/controller-neutral",
            "git_discovery_across_filesystem": False,
            "parent_repo_mounts": 0,
            "campaign_plan_mounts": 0,
            "sidecar_or_quarantine_mounts": 0,
            "manuscript_comparator_benchmark_mounts": 0,
            "symlinks_allowed": False,
            "hardlinks_allowed": False,
        },
        "dynamic_tool_namespace":
            spec.proposer_transport.dynamic_tool_namespace,
        "dynamic_tool_names": list(
            spec.proposer_transport.dynamic_tool_names
        ),
        "controller_method_policy": {
            "preflight_requests": list(
                spec.proposer_transport
                .controller_preflight_request_allowlist
            ),
            "preflight_notifications": list(
                spec.proposer_transport
                .controller_preflight_notification_allowlist
            ),
            "turn_requests": list(
                spec.proposer_transport
                .controller_turn_request_allowlist
            ),
        },
        "builtin_tool_names": [],
        "approval_policy": "never",
        "sandbox_policy": {
            "type": "readOnly",
            "networkAccess": False,
        },
        "state_root": "/controller-state",
        "state_host_staging_root": spec.app_server_state_dir,
        "state_mode": (
            "resume_staged_copy"
            if spec.thread_mode == "resume"
            else "new_reset"
        ),
        "prior_state_root": (
            spec.wip.app_server_state_dir
            if spec.wip is not None
            else None
        ),
        "prior_state_tree_sha256": (
            spec.wip.app_server_state_tree_sha256
            if spec.wip is not None
            else None
        ),
        "staged_state_root": spec.app_server_state_dir,
        "staged_initial_state_tree_sha256":
            spec.initial_app_server_state_tree_sha256,
        "ambient_state_access_status": "DENIED",
        "state_root_write_probe_status":
            "PENDING_REAL_CONTROLLER_PREFLIGHT",
        "ambient_environment_names_stripped": [
            "CODEX_HOME",
            "HOME",
            "XDG_CONFIG_HOME",
            "XDG_DATA_HOME",
            "XDG_STATE_HOME",
        ],
    }
    if any(config.get(key) != value
           for key, value in expected_config.items()):
        raise ContiguousRunnerError(
            "app-server configuration receipt is unsafe or stale"
        )
    binary = _validate_bound_receipt(
        prepared.codex_binary_receipt_path,
        prepared.codex_binary_receipt_sha256,
        expected_path=host_root / "codex_binary_receipt.json",
        expected_kind="contiguous_codex_binary",
        spec=spec,
    )
    if (
        binary.get("launcher_path")
        != spec.proposer_transport.codex_launcher_path
        or binary.get("launcher_sha256")
        != spec.proposer_transport.codex_launcher_sha256
        or binary.get("package_manifest_path")
        != spec.proposer_transport.codex_package_manifest_path
        or binary.get("package_manifest_sha256")
        != spec.proposer_transport.codex_package_manifest_sha256
        or binary.get("native_binary_path")
        != spec.proposer_transport.codex_binary_path
        or binary.get("native_binary_sha256")
        != spec.proposer_transport.codex_binary_sha256
        or binary.get("native_binary_bytes")
        != spec.proposer_transport.codex_binary_bytes
        or binary.get("version")
        != spec.proposer_transport.codex_cli_version
        or binary.get("observation_stage")
        != "pending_controller_guardian"
        or binary.get("controller_image_digest")
        != spec.proposer_transport.controller_image_digest
        or binary.get("host_file_observation") is not False
        or prepared.controller_image_digest
        != spec.proposer_transport.controller_image_digest
        or prepared.controller_egress_proxy_image_digest
        != spec.proposer_transport
        .controller_egress_proxy_image_digest
        or prepared.controller_egress_policy_sha256
        != spec.proposer_transport.controller_egress_policy_sha256
        or prepared.controller_supply_chain_unobserved_until_launch
        is not True
    ):
        raise ContiguousRunnerError("Codex binary receipt mismatch")
    protocol = _validate_bound_receipt(
        prepared.protocol_schema_receipt_path,
        prepared.protocol_schema_receipt_sha256,
        expected_path=host_root
        / "app_server_protocol_schema_receipt.json",
        expected_kind="contiguous_app_server_protocol_schema",
        spec=spec,
    )
    if (
        protocol.get("path")
        != spec.proposer_transport.app_server_protocol_schema_path
        or protocol.get("sha256")
        != spec.proposer_transport.app_server_protocol_schema_sha256
        or protocol.get("bundle_path")
        != spec.proposer_transport
        .app_server_protocol_schema_bundle_path
        or protocol.get("bundle_sha256")
        != spec.proposer_transport
        .app_server_protocol_schema_bundle_sha256
        or protocol.get("observation_stage")
        != "pending_controller_guardian"
        or protocol.get("controller_image_digest")
        != spec.proposer_transport.controller_image_digest
        or protocol.get("host_file_observation") is not False
    ):
        raise ContiguousRunnerError(
            "app-server protocol schema receipt mismatch"
        )


def _validate_launch_receipts(
    spec: AttemptSpec,
    prepared: BackendPreparation,
    launched: BackendLaunch,
) -> None:
    host_root = Path(spec.host_transcript_path).parent
    substrate = _validate_bound_receipt(
        launched.substrate_preflight_receipt_path,
        launched.substrate_preflight_receipt_sha256,
        expected_path=host_root / "substrate_preflight_receipt.json",
        expected_kind="contiguous_substrate_preflight",
        spec=spec,
    )
    if (
        substrate.get("substrate_identity_sha256")
        != launched.substrate_identity_sha256
        or substrate.get("state_root_write_probe_status") != "PASS"
        or substrate.get("state_database_initialized") is not True
        or substrate.get("path_alias_setup_status") != "PASS"
        or substrate.get("status") != "PASS"
        or any(
            substrate.get(name) is not False
            for name in (
                "proposer_container_started",
                "bridge_connected",
                "thread_started",
                "turn_started",
            )
        )
    ):
        raise ContiguousRunnerError(
            "substrate preflight receipt is incomplete"
        )
    bridge = _validate_bound_receipt(
        launched.bridge_runtime_attestation_path,
        launched.bridge_runtime_attestation_sha256,
        expected_path=host_root / "bridge_runtime_attestation.json",
        expected_kind="contiguous_bridge_runtime",
        spec=spec,
    )
    if (
        bridge.get("container_id") != launched.container_id
        or bridge.get("socket_path") != spec.bridge_socket_path
        or bridge.get("token_file_path")
        != spec.bridge_token_file_path
        or not isinstance(bridge.get("socket_inode"), int)
        or isinstance(bridge.get("socket_inode"), bool)
        or bridge["socket_inode"] <= 0
        or not isinstance(bridge.get("token_inode"), int)
        or isinstance(bridge.get("token_inode"), bool)
        or bridge["token_inode"] <= 0
        or not _is_sha256(bridge.get("token_sha256"))
        or not _is_sha256(bridge.get("handshake_nonce_sha256"))
        or bridge.get("policy_receipt_sha256")
        != prepared.bridge_policy_receipt_sha256
    ):
        raise ContiguousRunnerError("bridge runtime receipt mismatch")
    controller_launch = _validate_bound_receipt(
        launched.controller_launch_receipt_path,
        launched.controller_launch_receipt_sha256,
        expected_path=host_root / "controller_launch_receipt.json",
        expected_kind="arc_agi3_controller_launch",
        spec=spec,
    )
    expected_canary_anchor = {
        "escrow_path": prepared.controller_canary_escrow_path,
        "escrow_sha256": prepared.controller_canary_escrow_sha256,
        "escrow_identity_sha256":
            prepared.controller_canary_escrow_identity_sha256,
        "commitments_json":
            prepared.controller_canary_commitments_json,
        "commitments_sha256":
            prepared.controller_canary_commitments_sha256,
        "placement_descriptors_json":
            prepared.controller_canary_placement_descriptors_json,
        "placement_descriptors_sha256":
            prepared
            .controller_canary_placement_descriptors_sha256,
    }
    if (
        controller_launch.get("controller_container_id")
        != launched.controller_container_id
        or controller_launch.get("controller_image_digest")
        != launched.controller_image_digest
        or controller_launch.get("egress_proxy_container_id")
        != launched.egress_proxy_container_id
        or controller_launch.get("egress_proxy_image_digest")
        != launched.egress_proxy_image_digest
        or controller_launch.get("egress_policy_sha256")
        != launched.egress_policy_sha256
        or controller_launch.get("launch_intent_sha256")
        != launched.controller_launch_intent_sha256
        or controller_launch.get("credentials_in_argv_or_env")
        is not False
        or controller_launch.get("bridge_or_arena_mounts") != 0
        or controller_launch.get("authoritative_identity")
        != "controller_container_cgroup"
        or controller_launch.get("containment_canary_anchor")
        != expected_canary_anchor
    ):
        raise ContiguousRunnerError(
            "controller launch receipt is incomplete or substituted"
        )
    guardian_path = Path(
        launched.controller_guardian_start_receipt_path
    )
    if (
        guardian_path
        != host_root / "controller_guardian_start.json"
        or _sha256_file(guardian_path)
        != launched.controller_guardian_start_receipt_sha256
    ):
        raise ContiguousRunnerError(
            "controller guardian start receipt is substituted"
        )
    guardian = _read_json_file(guardian_path)
    if (
        guardian.get("schema") != 1
        or guardian.get("kind")
        != "arc_agi3_controller_guardian_start"
        or guardian.get("supply_chain_manifest_sha256")
        != launched.controller_supply_chain_manifest_sha256
        or guardian.get("hard_safety_seconds")
        != spec.hard_safety_seconds
    ):
        raise ContiguousRunnerError(
            "controller guardian supply-chain receipt mismatch"
        )
    runtime = _validate_bound_receipt(
        launched.app_server_runtime_receipt_path,
        launched.app_server_runtime_receipt_sha256,
        expected_path=host_root / "app_server_runtime_receipt.json",
        expected_kind="contiguous_app_server_runtime",
        spec=spec,
    )
    expected_runtime = {
        "pid": launched.app_server_pid,
        "process_start": launched.app_server_process_start,
        "process_group_id": launched.app_server_process_group_id,
        "state_root": spec.app_server_state_dir,
        "neutral_cwd": "/controller-neutral",
        "neutral_host_staging_cwd": spec.neutral_host_cwd_path,
        "thread_id": launched.codex_thread_id,
        "turn_id": launched.codex_turn_id,
        "thread_mode": spec.thread_mode,
        "model": spec.proposer_transport.model,
        "model_provider": spec.proposer_transport.model_provider,
        "reasoning_effort": spec.effort,
        "allow_provider_model_fallback": False,
        "builtin_tool_names": [],
        "dynamic_tool_namespace":
            spec.proposer_transport.dynamic_tool_namespace,
        "dynamic_tool_names": list(
            spec.proposer_transport.dynamic_tool_names
        ),
        "controller_method_policy": {
            "preflight_requests": list(
                spec.proposer_transport
                .controller_preflight_request_allowlist
            ),
            "preflight_notifications": list(
                spec.proposer_transport
                .controller_preflight_notification_allowlist
            ),
            "turn_requests": list(
                spec.proposer_transport
                .controller_turn_request_allowlist
            ),
        },
        "startup_probe_status": "PASS",
        "auth_probe_status": "PASS",
        "model_probe_status": "PASS",
        "bridge_probe_status": "PASS",
        "substrate_identity_sha256":
            launched.substrate_identity_sha256,
        "substrate_preflight_receipt_path":
            launched.substrate_preflight_receipt_path,
        "substrate_preflight_receipt_sha256":
            launched.substrate_preflight_receipt_sha256,
        "state_root_write_probe_status": "PASS",
        "state_database_initialized": True,
        "path_alias_setup_status": "PASS",
        "ambient_state_loaded": False,
    }
    if any(runtime.get(key) != value
           for key, value in expected_runtime.items()):
        raise ContiguousRunnerError(
            "app-server runtime capability receipt mismatch"
        )
    binding = _validate_bound_receipt(
        launched.thread_binding_path,
        launched.thread_binding_sha256,
        expected_path=host_root / "turn_start_binding.json",
        expected_kind="contiguous_turn_start_binding",
        spec=spec,
    )
    expected_binding = {
        "thread_id": launched.codex_thread_id,
        "turn_id": launched.codex_turn_id,
        "thread_mode": spec.thread_mode,
        "bridge_runtime_attestation_sha256":
            launched.bridge_runtime_attestation_sha256,
        "app_server_runtime_receipt_sha256":
            launched.app_server_runtime_receipt_sha256,
        "reasoning_effort": spec.effort,
        "model": spec.proposer_transport.model,
        "transcript_chain_sha256":
            launched.transcript_chain_sha256,
    }
    if any(binding.get(key) != value
           for key, value in expected_binding.items()):
        raise ContiguousRunnerError("turn-start binding mismatch")
    transcript = _validate_bound_receipt(
        launched.transcript_chain_receipt_path,
        launched.transcript_chain_receipt_sha256,
        expected_path=host_root
        / "turn_start_transcript_chain_receipt.json",
        expected_kind="contiguous_turn_start_transcript_chain",
        spec=spec,
    )
    if (
        transcript.get("thread_id") != launched.codex_thread_id
        or transcript.get("turn_id") != launched.codex_turn_id
        or transcript.get("chain_head_sha256")
        != launched.transcript_chain_sha256
    ):
        raise ContiguousRunnerError(
            "turn-start transcript chain receipt mismatch"
        )
    if spec.thread_mode == "new":
        if (
            launched.thread_rebinding_receipt_path is not None
            or launched.thread_rebinding_receipt_sha256 is not None
        ):
            raise ContiguousRunnerError(
                "new thread carries a rebinding receipt"
            )
        return
    if (
        spec.wip is None
        or launched.codex_thread_id != spec.resume_thread_id
        or launched.thread_rebinding_receipt_path is None
        or launched.thread_rebinding_receipt_sha256 is None
    ):
        raise ContiguousRunnerError(
            "resumed thread lacks its prior binding"
        )
    rebinding = _validate_bound_receipt(
        launched.thread_rebinding_receipt_path,
        launched.thread_rebinding_receipt_sha256,
        expected_path=host_root / "thread_rebinding_receipt.json",
        expected_kind="contiguous_thread_rebinding",
        spec=spec,
    )
    expected_rebinding = {
        "thread_id": spec.resume_thread_id,
        "prior_thread_binding_sha256":
            spec.resume_thread_binding_sha256,
        "prior_transcript_chain_sha256":
            spec.wip.transcript_chain_sha256,
        "prior_app_server_state_tree_sha256":
            spec.wip.app_server_state_tree_sha256,
        "prior_app_server_state_dir":
            spec.wip.app_server_state_dir,
        "staged_app_server_state_dir":
            spec.app_server_state_dir,
        "staged_initial_state_tree_sha256":
            spec.wip.app_server_state_tree_sha256,
        "new_container_id": launched.container_id,
        "new_bridge_runtime_attestation_sha256":
            launched.bridge_runtime_attestation_sha256,
        "old_bridge_revoked": True,
        "no_binding_overlap": True,
    }
    if any(rebinding.get(key) != value
           for key, value in expected_rebinding.items()):
        raise ContiguousRunnerError(
            "thread rebinding is stale, replayed, or overlapping"
        )


def _backend_poll_from_dict(value: object) -> BackendPoll:
    if not isinstance(value, dict):
        raise ContiguousRunnerError("backend poll must be an object")
    try:
        observed = BackendPoll(**value)
    except TypeError as exc:
        raise ContiguousRunnerError("backend poll schema mismatch") from exc
    if (
        observed.status not in {"running", "exited", "containment_fault"}
        or not _is_sha256(observed.observation_sha256)
        or (
            observed.status == "running"
            and observed.exit_code is not None
        )
        or (
            observed.exit_code is not None
            and (
                not isinstance(observed.exit_code, int)
                or isinstance(observed.exit_code, bool)
            )
        )
    ):
        raise ContiguousRunnerError("invalid backend poll")
    return observed


def _backend_collection_to_dict(
    collection: BackendCollection,
) -> dict[str, Any]:
    value = {
        "result": {
            "kind": collection.result.kind,
            "cost_used": float(collection.result.cost_used),
            "reason": collection.result.reason,
            "candidate": _candidate_to_dict(collection.result.candidate),
            "wip": _wip_to_dict(collection.result.wip),
            "blocker": _blocker_to_dict(collection.result.blocker),
            "native_sidecar_request_draft":
                _native_sidecar_request_draft_to_dict(
                    collection.result.native_sidecar_request_draft
                ),
        },
        "worker_outcome_sha256": collection.worker_outcome_sha256,
        "output_tree_sha256": collection.output_tree_sha256,
        "host_transcript_path": collection.host_transcript_path,
        "host_transcript_sha256": collection.host_transcript_sha256,
    }
    for name in BackendCollection.__dataclass_fields__:
        if name not in value and name != "result":
            value[name] = getattr(collection, name)
    value["native_public_observation_receipt_sha256s"] = list(
        collection.native_public_observation_receipt_sha256s
    )
    return value


def _backend_collection_from_dict(value: object) -> BackendCollection:
    if not isinstance(value, dict):
        raise ContiguousRunnerError("backend collection must be an object")
    required = set(BackendCollection.__dataclass_fields__)
    result_value = value.get("result")
    if set(value) != required or not isinstance(result_value, dict):
        raise ContiguousRunnerError("backend collection schema mismatch")
    result = ContiguousCampaignRunner._result_from_payload(
        {"attempt_id": "collection", **result_value}
    )
    collection_fields = {
        name: value[name]
        for name in required
        if name != "result"
    }
    native_receipts = collection_fields.get(
        "native_public_observation_receipt_sha256s"
    )
    if not isinstance(native_receipts, list):
        raise ContiguousRunnerError(
            "backend semantic observation receipts must be a list"
        )
    collection_fields[
        "native_public_observation_receipt_sha256s"
    ] = tuple(native_receipts)
    collection = BackendCollection(
        result=result,
        **collection_fields,
    )
    if (
        not _is_sha256(collection.worker_outcome_sha256)
        or not _is_sha256(collection.output_tree_sha256)
        or not _safe_path_string(collection.host_transcript_path)
        or not _is_sha256(collection.host_transcript_sha256)
        or tuple(
            sorted(
                set(
                    collection
                    .native_public_observation_receipt_sha256s
                )
            )
        )
        != collection.native_public_observation_receipt_sha256s
        or any(
            not _is_sha256(item)
            for item in (
                collection
                .native_public_observation_receipt_sha256s
            )
        )
        or not _safe_path_string(
            collection.app_server_transcript_path
        )
        or not _is_sha256(
            collection.app_server_transcript_sha256
        )
        or not _safe_path_string(
            collection.container_stdout_path
        )
        or not _is_sha256(collection.container_stdout_sha256)
        or not _safe_path_string(
            collection.container_stderr_path
        )
        or not _is_sha256(collection.container_stderr_sha256)
        or not _is_canonical_uuid(collection.codex_thread_id)
        or not _is_canonical_uuid(collection.codex_turn_id)
        or collection.structured_turn_status
        not in {"completed", "interrupted", "failed"}
        or collection.structured_provider_outcome
        not in {
            "completed",
            "capacity",
            "rate_limit",
            "provider_failure",
            "containment_fault",
        }
        or not _safe_path_string(
            collection.token_usage_receipt_path
        )
        or not _is_sha256(
            collection.token_usage_receipt_sha256
        )
        or not _safe_path_string(
            collection.provider_usage_receipt_path
        )
        or not _is_sha256(
            collection.provider_usage_receipt_sha256
        )
        or not _safe_path_string(
            collection.final_transcript_chain_receipt_path
        )
        or not _is_sha256(
            collection.final_transcript_chain_receipt_sha256
        )
        or not _is_sha256(
            collection.final_transcript_chain_sha256
        )
        or not _safe_path_string(
            collection.final_thread_binding_path
        )
        or not _is_sha256(
            collection.final_thread_binding_sha256
        )
        or not _safe_path_string(
            collection.bridge_export_receipt_path
        )
        or not _is_sha256(
            collection.bridge_export_receipt_sha256
        )
        or not _safe_path_string(
            collection.secret_scan_receipt_path
        )
        or not _is_sha256(
            collection.secret_scan_receipt_sha256
        )
        or not _safe_path_string(
            collection.controller_state_scan_receipt_path
        )
        or not _is_sha256(
            collection.controller_state_scan_receipt_sha256
        )
        or not _is_sha256(
            collection.controller_state_inventory_sha256
        )
        or not _safe_path_string(
            collection.retained_canary_scan_receipt_path
        )
        or not _is_sha256(
            collection.retained_canary_scan_receipt_sha256
        )
        or (
            (
                collection
                .supervisory_native_reproduction_receipt_path
                is None
            )
            != (
                collection
                .supervisory_native_reproduction_receipt_sha256
                is None
            )
        )
        or (
            collection
            .supervisory_native_reproduction_receipt_path
            is not None
            and (
                not _safe_path_string(
                    collection
                    .supervisory_native_reproduction_receipt_path
                )
                or not Path(
                    collection
                    .supervisory_native_reproduction_receipt_path
                ).is_absolute()
                or not _is_sha256(
                    collection
                    .supervisory_native_reproduction_receipt_sha256
                )
            )
        )
        or (
            any(
                value is not None
                for value in (
                    collection.target_boundary_receipt_path,
                    collection.target_boundary_receipt_sha256,
                    collection.target_boundary_sha256,
                    collection.target_boundary_workspace_tree_sha256,
                )
            )
            and (
                not _safe_path_string(
                    collection.target_boundary_receipt_path
                )
                or not _is_sha256(
                    collection.target_boundary_receipt_sha256
                )
                or not _is_sha256(
                    collection.target_boundary_sha256
                )
                or not _is_sha256(
                    collection
                    .target_boundary_workspace_tree_sha256
                )
                or not Path(
                    collection.target_boundary_receipt_path
                ).is_absolute()
            )
        )
        or (
            collection.result.kind == "candidate"
            and any(
                value is None
                for value in (
                    collection.target_boundary_receipt_path,
                    collection.target_boundary_receipt_sha256,
                    collection.target_boundary_sha256,
                    collection.target_boundary_workspace_tree_sha256,
                )
            )
        )
        or not _safe_path_string(
            collection.taint_scan_receipt_path
        )
        or not _is_sha256(
            collection.taint_scan_receipt_sha256
        )
        or not _is_sha256(
            collection.app_server_state_tree_sha256
        )
        or not _is_sha256(collection.model_final_text_sha256)
        or not all(
            Path(path).is_absolute()
            for path in (
                collection.host_transcript_path,
                collection.app_server_transcript_path,
                collection.container_stdout_path,
                collection.container_stderr_path,
                collection.token_usage_receipt_path,
                collection.provider_usage_receipt_path,
                collection.final_transcript_chain_receipt_path,
                collection.final_thread_binding_path,
                collection.bridge_export_receipt_path,
                collection.secret_scan_receipt_path,
                collection.controller_state_scan_receipt_path,
                collection.retained_canary_scan_receipt_path,
                collection.taint_scan_receipt_path,
            )
        )
    ):
        raise ContiguousRunnerError("invalid backend collection")
    return collection


def _backend_teardown_from_dict(value: object) -> BackendTeardownProof:
    if not isinstance(value, dict):
        raise ContiguousRunnerError("backend teardown proof must be an object")
    try:
        proof = BackendTeardownProof(**value)
    except TypeError as exc:
        raise ContiguousRunnerError(
            "backend teardown proof schema mismatch"
        ) from exc
    if (
        not isinstance(proof.container_id, str)
        or not re.fullmatch(r"[0-9a-f]{64}", proof.container_id)
        or proof.cause not in {"normal_exit", "containment_fault"}
        or not _is_sha256(proof.proof_sha256)
        or not all(
            value is True
            for value in (
                proof.container_inspect_absent,
                proof.container_top_absent,
                proof.identity_query_empty,
                proof.no_descendants,
                proof.app_server_process_absent,
                proof.app_server_process_group_absent,
                proof.bridge_socket_absent,
                proof.bridge_token_absent,
                proof.app_server_control_absent,
                proof.arena_relay_inspect_absent,
                proof.arena_relay_top_absent,
                proof.arena_relay_identity_query_empty,
                proof.arena_volume_inspect_absent,
                proof.arena_volume_identity_query_empty,
                proof.controller_inspect_absent,
                proof.controller_identity_query_empty,
                proof.controller_top_absent,
                proof.controller_no_descendants,
                proof.egress_proxy_inspect_absent,
                proof.egress_proxy_identity_query_empty,
                proof.egress_proxy_top_absent,
                proof.egress_proxy_no_descendants,
            )
        )
        or not isinstance(proof.arena_relay_container_id, str)
        or re.fullmatch(
            r"[0-9a-f]{64}", proof.arena_relay_container_id
        )
        is None
        or not isinstance(proof.arena_volume_name, str)
        or re.fullmatch(
            r"arc-agi3-arena-[0-9a-f]{12}-[0-9a-f]{32}",
            proof.arena_volume_name,
        )
        is None
        or proof.arena_relay_attachment_status
        not in {"CLEAN_EOF", "ABORTED_CONTAINMENT"}
        or not _safe_path_string(
            proof.arena_relay_teardown_receipt_path
        )
        or not Path(
            proof.arena_relay_teardown_receipt_path
        ).is_absolute()
        or not _is_sha256(
            proof.arena_relay_teardown_receipt_sha256
        )
        or proof.process_identity_authority
        != "controller_container_cgroup"
        or not isinstance(proof.controller_container_id, str)
        or not re.fullmatch(
            r"[0-9a-f]{64}", proof.controller_container_id
        )
        or not isinstance(
            proof.egress_proxy_container_id, str
        )
        or not re.fullmatch(
            r"[0-9a-f]{64}",
            proof.egress_proxy_container_id,
        )
        or not _is_sha256(
            proof.controller_absence_receipt_sha256
        )
        or not _safe_path_string(proof.canary_reveal_path)
        or not Path(proof.canary_reveal_path).is_absolute()
        or not _is_sha256(proof.canary_reveal_sha256)
        or (
            (proof.canary_cleanup_receipt_path is None)
            != (proof.canary_cleanup_receipt_sha256 is None)
        )
        or (
            proof.canary_cleanup_receipt_path is not None
            and (
                not _safe_path_string(
                    proof.canary_cleanup_receipt_path
                )
                or not Path(
                    proof.canary_cleanup_receipt_path
                ).is_absolute()
                or not _is_sha256(
                    proof.canary_cleanup_receipt_sha256
                )
            )
        )
    ):
        raise ContiguousRunnerError("invalid backend teardown proof")
    return proof


def _validate_arena_volume_teardown_receipt(
    *,
    spec: AttemptSpec,
    prepared: BackendPreparation,
    receipt_path: str | Path,
    expected_sha256: str | None = None,
    expected_status: str | None = None,
) -> tuple[str, str]:
    """Independently reopen a bound relay/volume absence receipt.

    The receipt can be durably created before the scheduler journals the
    backend teardown proof.  Recovery therefore authenticates the exact
    controller-owned terminal artifact without relying on an as-yet
    unjournaled proof.  Only that authenticated path may be projected out of
    the pre-teardown retained-evidence scan.
    """

    path = Path(receipt_path)
    digest, metadata = _sha256_file_identity(path)
    value = _read_json_file(path)
    attachment = value.get("attachment_receipt")
    attachment_sha256 = (
        hashlib.sha256(_canonical_json(attachment)).hexdigest()
        if isinstance(attachment, dict)
        else None
    )
    observed_status = value.get("attachment_status")
    if (
        path
        != Path(spec.host_transcript_path).parent
        / "arena_volume_teardown.json"
        or (
            expected_sha256 is not None
            and digest != expected_sha256
        )
        or metadata.st_uid != os.getuid()
        or metadata.st_nlink != 1
        or not stat.S_ISREG(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != 0o600
        or set(value)
        != {
            "schema",
            "kind",
            "campaign_id",
            "generation_id",
            "attempt_id",
            "transport",
            "preparation_receipt_sha256",
            "relay_container_id",
            "volume_name",
            "attachment_status",
            "attachment_receipt",
            "attachment_receipt_sha256",
            "relay_inspect_absent",
            "relay_top_absent",
            "relay_identity_query_empty",
            "volume_inspect_absent",
            "volume_identity_query_empty",
        }
        or value.get("schema") != 1
        or value.get("kind") != "arc_agi3_arena_volume_teardown"
        or value.get("campaign_id") != spec.campaign_id
        or value.get("generation_id") != spec.generation_id
        or value.get("attempt_id") != spec.attempt_id
        or value.get("transport") != ARENA_VOLUME_TRANSPORT
        or value.get("preparation_receipt_sha256")
        != prepared.arena_relay_preparation_receipt_sha256
        or value.get("relay_container_id")
        != prepared.arena_relay_container_id
        or value.get("volume_name") != prepared.arena_volume_name
        or prepared.arena_volume_name != _arena_volume_name(spec)
        or observed_status
        not in {"CLEAN_EOF", "ABORTED_CONTAINMENT"}
        or (
            expected_status is not None
            and observed_status != expected_status
        )
        or value.get("attachment_receipt_sha256")
        != attachment_sha256
        or (
            observed_status == "CLEAN_EOF"
            and not isinstance(attachment, dict)
        )
        or (
            observed_status == "ABORTED_CONTAINMENT"
            and attachment is not None
        )
        or any(
            value.get(name) is not True
            for name in (
                "relay_inspect_absent",
                "relay_top_absent",
                "relay_identity_query_empty",
                "volume_inspect_absent",
                "volume_identity_query_empty",
            )
        )
    ):
        raise ContiguousRunnerError(
            "Arena relay/volume teardown receipt is stale or substituted"
        )
    return digest, observed_status


def _validate_arena_volume_teardown(
    *,
    spec: AttemptSpec,
    prepared: BackendPreparation,
    proof: BackendTeardownProof,
) -> None:
    """Reopen the exact relay/volume absence receipt after teardown."""

    expected_status = (
        "CLEAN_EOF"
        if proof.cause == "normal_exit"
        else "ABORTED_CONTAINMENT"
    )
    _digest, observed_status = _validate_arena_volume_teardown_receipt(
        spec=spec,
        prepared=prepared,
        receipt_path=proof.arena_relay_teardown_receipt_path,
        expected_sha256=proof.arena_relay_teardown_receipt_sha256,
        expected_status=expected_status,
    )
    if (
        proof.arena_relay_container_id
        != prepared.arena_relay_container_id
        or proof.arena_volume_name != prepared.arena_volume_name
        or proof.arena_relay_attachment_status != observed_status
    ):
        raise ContiguousRunnerError(
            "Arena relay/volume teardown proof differs from its receipt"
        )


def _validate_protocol_invalid_terminal_evidence(
    *,
    spec: AttemptSpec,
    receipt: Mapping[str, Any],
    evidence: Mapping[str, Any],
) -> dict[str, str]:
    """Reopen every terminal scan/accounting receipt transitively bound here."""

    names = {
        "controller_state_scan": (
            "controller_state_scan_receipt_path",
            "controller_state_scan_receipt_sha256",
            Path(spec.host_transcript_path).parent
            / "controller_state_scan_receipt.json",
            "contiguous_controller_state_scan",
        ),
        "retained_canary_scan": (
            "retained_canary_scan_receipt_path",
            "retained_canary_scan_receipt_sha256",
            Path(spec.generation_dir)
            / "retained_canary_scan_receipt.json",
            "contiguous_retained_canary_scan",
        ),
        "partial_taint_scan": (
            "partial_taint_scan_receipt_path",
            "partial_taint_scan_receipt_sha256",
            Path(spec.host_transcript_path).parent
            / "protocol_invalid_partial_taint_scan_receipt.json",
            "contiguous_protocol_invalid_partial_taint_scan",
        ),
        "partial_usage": (
            "partial_usage_receipt_path",
            "partial_usage_receipt_sha256",
            Path(spec.host_transcript_path).parent
            / "protocol_invalid_partial_usage_receipt.json",
            "contiguous_protocol_invalid_partial_usage",
        ),
    }
    expected_evidence_keys = {
        field
        for path_field, digest_field, _path, _kind in names.values()
        for field in (path_field, digest_field)
    }
    if set(evidence) != expected_evidence_keys:
        raise ContiguousRunnerError(
            "protocol-invalid terminal evidence schema mismatch"
        )
    reopened: dict[str, Mapping[str, Any]] = {}
    normalized: dict[str, str] = {}
    for label, (
        path_field,
        digest_field,
        expected_path,
        expected_kind,
    ) in names.items():
        path_value = evidence.get(path_field)
        digest_value = evidence.get(digest_field)
        if (
            receipt.get(path_field) != path_value
            or receipt.get(digest_field) != digest_value
        ):
            raise ContiguousRunnerError(
                "protocol-invalid receipt omits terminal evidence binding"
            )
        reopened[label] = _validate_bound_receipt(
            path_value,
            digest_value,
            expected_path=expected_path,
            expected_kind=expected_kind,
            spec=spec,
        )
        normalized[path_field] = str(path_value)
        normalized[digest_field] = str(digest_value)
    state_scan = reopened["controller_state_scan"]
    retained_scan = reopened["retained_canary_scan"]
    taint_scan = reopened["partial_taint_scan"]
    usage = reopened["partial_usage"]
    taint_status = taint_scan.get("status")
    accounting_complete = usage.get("accounting_complete")
    if (
        state_scan.get("scanner_source_sha256")
        != Taint.source_sha256()
        or retained_scan.get("scanner_source_sha256")
        != Taint.source_sha256()
        or retained_scan.get(
            "controller_state_scan_receipt_sha256"
        )
        != evidence["controller_state_scan_receipt_sha256"]
        or taint_scan.get("scanner_source_sha256")
        != Taint.source_sha256()
        or taint_scan.get("classification_authority")
        != "source_environment_taint_only"
        or taint_status not in {"CLEAN", "TAINT"}
        or not isinstance(taint_scan.get("hits"), list)
        or bool(taint_scan["hits"]) != (taint_status == "TAINT")
        or receipt.get("partial_taint_status") != taint_status
        or not isinstance(accounting_complete, bool)
        or usage.get("unknown_token_usage")
        is not (not accounting_complete)
        or usage.get("cost_settlement_authority") is not False
        or receipt.get("usage_accounting_complete")
        is not accounting_complete
        or not isinstance(
            usage.get("token_usage_observations"), list
        )
        or (
            not accounting_complete
            and any(
                usage.get(name) is not None
                for name in (
                    "post_provider_usage_window",
                    "provider_usage_settlement",
                )
            )
        )
        or (
            not usage["token_usage_observations"]
            and usage.get("observed_total_tokens") is not None
        )
    ):
        raise ContiguousRunnerError(
            "protocol-invalid terminal scan/accounting evidence is malformed"
        )
    return normalized


def _validate_terminal_canary_reveal(
    *,
    spec: AttemptSpec,
    prepared: BackendPreparation,
    launched: BackendLaunch,
    proof: BackendTeardownProof,
    canaries: tuple[Taint.LiveCanary, ...] = (),
) -> None:
    """Independently reopen the post-absence reveal and prelaunch anchor."""

    campaign_root = Path(spec.generation_dir).parent.parent
    host_root = Path(spec.host_transcript_path).parent
    expected_path = (
        campaign_root
        / "containment_canary_reveals"
        / f"{spec.generation_id}.json"
    )
    path = Path(proof.canary_reveal_path)
    raw_sha256, metadata = _sha256_file_identity(path)
    if (
        path != expected_path
        or raw_sha256 != proof.canary_reveal_sha256
        or metadata.st_uid != os.getuid()
        or metadata.st_nlink != 1
        or stat.S_IMODE(metadata.st_mode) != 0o400
    ):
        raise ContiguousRunnerError(
            "terminal canary reveal path/identity is substituted"
        )
    value = _read_json_file(path)
    fields = {
        "schema",
        "kind",
        "campaign_id",
        "generation_id",
        "attempt_id",
        "attempt_spec_sha256",
        "canary_escrow_sha256",
        "canary_escrow_identity_sha256",
        "canary_commitments_sha256",
        "canary_placement_descriptors_sha256",
        "controller_container_id",
        "egress_proxy_container_id",
        "controller_absence_receipt_sha256",
        "controller_state_scan_receipt_sha256",
        "retained_canary_scan_receipt_sha256",
        "canary_commitments",
        "reveal",
        "teardown_observation_sha256",
    }
    try:
        commitments = json.loads(
            prepared.controller_canary_commitments_json
        )
        placement_descriptors = json.loads(
            prepared.controller_canary_placement_descriptors_json
        )
    except (TypeError, UnicodeError, json.JSONDecodeError) as exc:
        raise ContiguousRunnerError(
            "journaled canary anchor is not canonical JSON"
        ) from exc
    if (
        set(value) != fields
        or value.get("schema") != 1
        or value.get("kind")
        != "contiguous_controller_canary_reveal"
        or value.get("campaign_id") != spec.campaign_id
        or value.get("generation_id") != spec.generation_id
        or value.get("attempt_id") != spec.attempt_id
        or value.get("attempt_spec_sha256")
        != proposer_attempt_binding_sha256(spec)
        or value.get("canary_escrow_sha256")
        != prepared.controller_canary_escrow_sha256
        or value.get("canary_escrow_identity_sha256")
        != prepared.controller_canary_escrow_identity_sha256
        or value.get("canary_commitments_sha256")
        != prepared.controller_canary_commitments_sha256
        or value.get("canary_placement_descriptors_sha256")
        != prepared.controller_canary_placement_descriptors_sha256
        or value.get("canary_commitments") != commitments
        or not isinstance(placement_descriptors, list)
        or value.get("controller_container_id")
        != launched.controller_container_id
        or value.get("egress_proxy_container_id")
        != launched.egress_proxy_container_id
        or value.get("controller_absence_receipt_sha256")
        != proof.controller_absence_receipt_sha256
    ):
        raise ContiguousRunnerError(
            "terminal canary reveal differs from its prelaunch anchor"
        )
    expected_rows = tuple(
        (
            row.get("category"),
            row.get("location_name"),
            row.get("provenance"),
            row.get("commitment_sha256"),
        )
        for row in commitments
        if isinstance(row, dict)
    )
    try:
        revealed = Taint.validate_live_canary_reveal(
            value.get("reveal"),
            expected_commitments=expected_rows,
        )
    except Exception as exc:
        raise ContiguousRunnerError(
            "terminal canary reveal values do not open the commitments"
        ) from exc
    if canaries:
        try:
            expected_canaries = Taint.validate_live_canaries(
                canaries, require_complete=True
            )
        except Exception as exc:
            raise ContiguousRunnerError(
                "live canary audit set is malformed"
            ) from exc
        if revealed != expected_canaries:
            raise ContiguousRunnerError(
                "terminal reveal differs from the live canary set"
            )
    state_scan_path = host_root / "controller_state_scan_receipt.json"
    retained_scan_path = (
        Path(spec.generation_dir)
        / "retained_canary_scan_receipt.json"
    )
    if (
        _sha256_file(state_scan_path)
        != value.get("controller_state_scan_receipt_sha256")
        or _sha256_file(retained_scan_path)
        != value.get("retained_canary_scan_receipt_sha256")
    ):
        raise ContiguousRunnerError(
            "terminal canary reveal scan receipts changed"
        )
    state_scan = _read_json_file(state_scan_path).get(
        "controller_state_scan"
    )
    retained_scan = _read_json_file(retained_scan_path).get(
        "retained_canary_scan"
    )
    if (
        not isinstance(state_scan, dict)
        or not isinstance(retained_scan, dict)
        or state_scan.get("canary_commitments") != commitments
        or retained_scan.get("canary_commitments") != commitments
        or state_scan.get("canary_occurrences") != 0
        or retained_scan.get("canary_occurrences") != 0
        or state_scan.get("status") != "CLEAN"
        or retained_scan.get("status") != "CLEAN"
    ):
        raise ContiguousRunnerError(
            "terminal canary reveal is not backed by two clean scans"
        )
    absence_path = host_root / "controller_absence_receipt.json"
    reconciliation_path = (
        host_root / "probe_reconciliation_teardown.json"
    )
    if (
        _sha256_file(absence_path)
        != proof.controller_absence_receipt_sha256
    ):
        raise ContiguousRunnerError(
            "terminal controller absence receipt changed"
        )
    teardown_observation = hashlib.sha256(
        _canonical_json({
            "container_proof_sha256": proof.proof_sha256,
            "controller_absence_receipt_sha256":
                proof.controller_absence_receipt_sha256,
            "probe_reconciliation_receipt_sha256":
                _sha256_file(reconciliation_path),
            "controller_container_id":
                launched.controller_container_id,
            "egress_proxy_container_id":
                launched.egress_proxy_container_id,
            "all_exact_roles_absent": True,
        })
    ).hexdigest()
    if value.get("teardown_observation_sha256") != teardown_observation:
        raise ContiguousRunnerError(
            "terminal canary reveal teardown observation differs"
        )


def _validate_terminal_canary_cleanup(
    *,
    spec: AttemptSpec,
    prepared: BackendPreparation,
    proof: BackendTeardownProof,
) -> None:
    """Reopen the post-reveal marker-removal intent and completion receipt."""

    campaign_root = Path(spec.generation_dir).parent.parent
    expected_receipt_path = (
        campaign_root
        / "containment_canary_cleanups"
        / f"{spec.generation_id}.json"
    )
    expected_intent_path = (
        campaign_root
        / "containment_canary_cleanups"
        / f"{spec.generation_id}.intent.json"
    )
    try:
        placements = json.loads(
            prepared.controller_canary_placement_descriptors_json
        )
        commitments = json.loads(
            prepared.controller_canary_commitments_json
        )
    except (TypeError, UnicodeError, json.JSONDecodeError) as exc:
        raise ContiguousRunnerError(
            "terminal canary cleanup anchor is malformed"
        ) from exc
    formal_placements = (
        isinstance(placements, list)
        and len(placements) == len(commitments)
        and all(
            isinstance(row, dict)
            and row.get("placement_kind")
            in {
                "host_file",
                "credential_decoy_file",
                "host_environment",
            }
            for row in placements
        )
    )
    if not formal_placements:
        if (
            proof.canary_cleanup_receipt_path is not None
            or proof.canary_cleanup_receipt_sha256 is not None
        ):
            raise ContiguousRunnerError(
                "static canary proof cannot claim operator cleanup"
            )
        return
    if (
        proof.canary_cleanup_receipt_path is None
        or proof.canary_cleanup_receipt_sha256 is None
    ):
        raise ContiguousRunnerError(
            "formal canary planting lacks terminal cleanup proof"
        )
    receipt_path = Path(proof.canary_cleanup_receipt_path)
    receipt_sha256, receipt_metadata = _sha256_file_identity(
        receipt_path
    )
    if (
        receipt_path != expected_receipt_path
        or receipt_sha256
        != proof.canary_cleanup_receipt_sha256
        or receipt_metadata.st_uid != os.getuid()
        or receipt_metadata.st_nlink != 1
        or stat.S_IMODE(receipt_metadata.st_mode) != 0o400
    ):
        raise ContiguousRunnerError(
            "terminal canary cleanup receipt identity is substituted"
        )
    receipt = _read_json_file(receipt_path)
    expected_absence = [
        {
            "category": row["category"],
            "commitment_sha256": row["commitment_sha256"],
            "placement_absent": True,
        }
        for row in commitments
    ]
    expected_receipt_fields = {
        "schema",
        "kind",
        "campaign_id",
        "generation_id",
        "attempt_id",
        "cleanup_intent_path",
        "cleanup_intent_sha256",
        "planting_receipt_sha256",
        "placement_descriptors_sha256",
        "reveal_path",
        "reveal_sha256",
        "placement_absence",
        "all_six_absent_after_terminal_reveal",
    }
    if (
        set(receipt) != expected_receipt_fields
        or receipt.get("schema") != 1
        or receipt.get("kind")
        != "arc_agi3_containment_canary_cleanup"
        or receipt.get("campaign_id") != spec.campaign_id
        or receipt.get("generation_id") != spec.generation_id
        or receipt.get("attempt_id") != spec.attempt_id
        or receipt.get("cleanup_intent_path")
        != str(expected_intent_path)
        or receipt.get("placement_descriptors_sha256")
        != prepared.controller_canary_placement_descriptors_sha256
        or receipt.get("reveal_path") != proof.canary_reveal_path
        or receipt.get("reveal_sha256")
        != proof.canary_reveal_sha256
        or receipt.get("placement_absence") != expected_absence
        or receipt.get("all_six_absent_after_terminal_reveal")
        is not True
        or not _is_sha256(
            receipt.get("planting_receipt_sha256")
        )
        or not _is_sha256(
            receipt.get("cleanup_intent_sha256")
        )
    ):
        raise ContiguousRunnerError(
            "terminal canary cleanup receipt lineage differs"
        )
    intent_sha256, intent_metadata = _sha256_file_identity(
        expected_intent_path
    )
    intent = _read_json_file(expected_intent_path)
    expected_intent_fields = {
        "schema",
        "kind",
        "campaign_id",
        "generation_id",
        "attempt_id",
        "planting_receipt_path",
        "planting_receipt_sha256",
        "placement_descriptors_sha256",
        "reveal_path",
        "reveal_sha256",
        "teardown_absence_bound_by_reveal",
        "cleanup_policy",
    }
    if (
        intent_sha256 != receipt["cleanup_intent_sha256"]
        or intent_metadata.st_uid != os.getuid()
        or intent_metadata.st_nlink != 1
        or stat.S_IMODE(intent_metadata.st_mode) != 0o400
        or set(intent) != expected_intent_fields
        or intent.get("schema") != 1
        or intent.get("kind")
        != "arc_agi3_containment_canary_cleanup_intent"
        or intent.get("campaign_id") != spec.campaign_id
        or intent.get("generation_id") != spec.generation_id
        or intent.get("attempt_id") != spec.attempt_id
        or intent.get("planting_receipt_sha256")
        != receipt["planting_receipt_sha256"]
        or intent.get("placement_descriptors_sha256")
        != prepared.controller_canary_placement_descriptors_sha256
        or intent.get("reveal_path") != proof.canary_reveal_path
        or intent.get("reveal_sha256")
        != proof.canary_reveal_sha256
        or intent.get("teardown_absence_bound_by_reveal")
        is not True
        or intent.get("cleanup_policy")
        != "descriptor_relative_exact_identity_unlinkat"
    ):
        raise ContiguousRunnerError(
            "terminal canary cleanup intent lineage differs"
        )
    for row in placements:
        location = row["location_name"]
        if row["placement_kind"] == "host_environment":
            if location in os.environ:
                raise ContiguousRunnerError(
                    "terminal host-environment canary remains"
                )
            continue
        path = Path(location)
        if (
            not path.is_absolute()
            or path.name != "marker"
            or path.parent.name
            != f".arc-agi3-containment-{spec.generation_id}"
            or path.exists()
            or path.is_symlink()
            or path.parent.exists()
            or path.parent.is_symlink()
        ):
            raise ContiguousRunnerError(
                "terminal host-file canary cleanup is incomplete"
            )


def _retained_canary_scan_receipt_excluding(
    roots: Mapping[str, Path],
    scan: Taint.RetainedCanaryScan,
    *,
    excluded: Mapping[str, set[str]],
) -> dict[str, Any]:
    """Project a sealed scan across one explicitly typed terminal artifact.

    Collection seals retained evidence before the Arena relay and named volume
    can be destroyed.  Teardown subsequently creates its absence receipt.
    Recovery compares the original scan against the live trees with only that
    exact, independently checked path removed; this is deliberately not a
    general-purpose ignore list.
    """

    if (
        not excluded
        or not set(excluded) <= set(roots)
        or any(
            not paths
            or any(
                not isinstance(path, str)
                or not path
                or path.startswith("/")
                or ".." in PurePosixPath(path).parts
                for path in paths
            )
            for paths in excluded.values()
        )
    ):
        raise ContiguousRunnerError(
            "retained evidence exclusion policy is malformed"
        )
    root_inventories: list[dict[str, Any]] = []
    excluded_records = {
        f"{label}/{relative}"
        for label, paths in excluded.items()
        for relative in paths
    }
    for label, root in sorted(roots.items()):
        inventory = Transport.inventory_controller_state(root)
        excluded_paths = excluded.get(label, set())
        observed_paths = {
            path for path, _digest, _size in inventory.files
        }
        if not excluded_paths <= observed_paths:
            raise ContiguousRunnerError(
                "terminal retained-evidence exclusion is absent"
            )
        rows = tuple(
            row
            for row in inventory.files
            if row[0] not in excluded_paths
        )
        tree_digest = hashlib.sha256()
        for path, digest, _byte_count in rows:
            tree_digest.update(path.encode("utf-8"))
            tree_digest.update(b"\0")
            tree_digest.update(digest.encode("ascii"))
            tree_digest.update(b"\n")
        total_bytes = sum(
            byte_count for _path, _digest, byte_count in rows
        )
        inventory_payload = _canonical_json(
            {
                "files": [
                    {
                        "path": path,
                        "sha256": digest,
                        "bytes": byte_count,
                    }
                    for path, digest, byte_count in rows
                ],
                "file_count": len(rows),
                "total_bytes": total_bytes,
            }
        )
        root_inventories.append(
            {
                "label": label,
                "tree_sha256": tree_digest.hexdigest(),
                "inventory_sha256": hashlib.sha256(
                    inventory_payload
                ).hexdigest(),
                "file_count": len(rows),
                "total_bytes": total_bytes,
            }
        )
    value = scan.as_receipt()
    value["root_inventories"] = root_inventories
    value["records"] = [
        row
        for row in value["records"]
        if row["path"] not in excluded_records
    ]
    return value


class ContiguousCampaignRunner:
    """Replay-derived, crash-resumable scheduler for the 25-game inventory."""

    def __init__(
        self,
        root: Path,
        *,
        backend: AttemptBackend,
        promotion_gate: PromotionGate,
        input_builder: InputBundleBuilder,
        backend_configuration: BackendConfiguration,
        cost_window_id: str,
        max_lanes: int = MAX_LANES,
        limit: float | None = None,
        operator_configuration_sha256: str | None = None,
        secret_sentinels: tuple[str, ...] = (),
        controller_state_canaries: tuple[Taint.LiveCanary, ...] = (),
        auxiliary_backend: AuxiliaryBackend | None = None,
        auxiliary_launch_configuration:
            Scheduler.AuxiliaryLaunchConfiguration | None = None,
        clock: Callable[[], float] = time.time,
        id_factory: Callable[[], str] = lambda: str(uuid.uuid4()),
    ):
        if (
            not isinstance(max_lanes, int)
            or isinstance(max_lanes, bool)
            or not 1 <= max_lanes <= MAX_LANES
        ):
            raise ContiguousRunnerError(
                f"max_lanes must be an integer in 1..{MAX_LANES}"
            )
        if limit is not None and (
            not _is_finite_number(limit)
            or limit < 0
        ):
            raise ContiguousRunnerError("finite limit must be nonnegative")
        if (
            operator_configuration_sha256 is not None
            and not _is_sha256(operator_configuration_sha256)
        ):
            raise ContiguousRunnerError(
                "operator configuration digest is malformed"
            )
        if not _valid_backend_configuration(backend_configuration):
            raise ContiguousRunnerError("invalid backend configuration")
        if not _safe_identifier(cost_window_id):
            raise ContiguousRunnerError(
                "cost_window_id must be an explicit safe identifier"
            )
        auxiliary_configuration = (
            Scheduler.disabled_auxiliary_launch_configuration()
            if auxiliary_launch_configuration is None
            else auxiliary_launch_configuration
        )
        try:
            auxiliary_configuration = (
                Scheduler.validate_auxiliary_launch_configuration(
                    auxiliary_configuration
                )
            )
        except Scheduler.SchedulerError as exc:
            raise ContiguousRunnerError(
                "invalid auxiliary launch configuration"
            ) from exc
        if auxiliary_configuration.automatic_dispatch_enabled:
            if auxiliary_backend is None:
                raise ContiguousRunnerError(
                    "automatic auxiliary dispatch has no backend"
                )
            expected_contracts = (
                auxiliary_configuration.backend_contract_sha256,
                auxiliary_configuration.input_bundle_contract_sha256,
                auxiliary_configuration.admission_contract_sha256,
            )
            observed_contracts = (
                getattr(
                    auxiliary_backend,
                    "backend_contract_sha256",
                    None,
                ),
                getattr(
                    auxiliary_backend,
                    "input_bundle_contract_sha256",
                    None,
                ),
                getattr(
                    auxiliary_backend,
                    "admission_contract_sha256",
                    None,
                ),
            )
            attestations = (
                getattr(
                    auxiliary_backend,
                    "production_isolation_attested",
                    False,
                ),
                getattr(
                    auxiliary_backend,
                    "immutable_private_input_attested",
                    False,
                ),
                getattr(
                    auxiliary_backend,
                    "host_admission_attested",
                    False,
                ),
                getattr(
                    auxiliary_backend,
                    "descriptor_confined_receipts_attested",
                    False,
                ),
            )
            if (
                observed_contracts != expected_contracts
                or attestations != (True, True, True, True)
                or not callable(
                    getattr(
                        auxiliary_backend,
                        "read_confined_receipt",
                        None,
                    )
                )
            ):
                raise ContiguousRunnerError(
                    "auxiliary backend lacks the exact attested isolation, "
                    "private-input, descriptor-confinement, or "
                    "host-admission contract"
                )
        elif auxiliary_backend is not None:
            raise ContiguousRunnerError(
                "disabled auxiliary dispatch must not retain a backend"
            )
        try:
            Scheduler.verify_runner_policy(
                frontier_retry_policy,
                declared_policy_sha256=SCHEDULER_POLICY_SHA256,
            )
        except Scheduler.SchedulerError as exc:
            raise ContiguousRunnerError(
                "runner policy differs from the canonical scheduler"
            ) from exc
        if (
            not isinstance(secret_sentinels, tuple)
            or any(
                not isinstance(value, str)
                or not value
                or value == "REDACTED"
                for value in secret_sentinels
            )
            or len(set(secret_sentinels)) != len(secret_sentinels)
        ):
            raise ContiguousRunnerError(
                "live credential sentinels are malformed"
            )
        try:
            normalized_controller_state_canaries = (
                Taint.validate_live_canaries(
                    controller_state_canaries,
                    require_complete=bool(
                        getattr(
                            backend,
                            "requires_controller_state_canaries",
                            False,
                        )
                    ),
                )
                if controller_state_canaries
                else ()
            )
        except Exception as exc:
            raise ContiguousRunnerError(
                "live controller-state canaries are malformed"
            ) from exc
        if (
            getattr(
                backend, "requires_controller_state_canaries", False
            )
            and not normalized_controller_state_canaries
        ):
            raise ContiguousRunnerError(
                "production backend requires the complete six-category "
                "controller containment canary set"
            )
        requested_root = Path(root)
        if requested_root.is_symlink() or (
            requested_root.exists() and not requested_root.is_dir()
        ):
            raise ContiguousRunnerError(
                "runner root must be a regular host directory"
            )
        self.root = requested_root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        os.chmod(self.root, 0o700, follow_symlinks=False)
        self.backend = backend
        self.auxiliary_backend = auxiliary_backend
        self.auxiliary_launch_configuration = auxiliary_configuration
        self._trusted_auxiliary_event_digests: set[str] = set()
        self.promotion_gate = promotion_gate
        self.input_builder = input_builder
        self.backend_configuration = backend_configuration
        # Live-only containment inputs.  They are never serialized into the
        # journal, attempt spec, or retained scan receipt.
        self._secret_sentinels = secret_sentinels
        self._controller_state_canaries = (
            normalized_controller_state_canaries
        )
        self.clock = clock
        self.id_factory = id_factory
        self.journal = DurableAttemptJournal(self.root / "attempt_journal")
        self._reducer_checkpoint: _ReducerCheckpoint | None = None
        self._verified_lane_checkpoints: dict[
            tuple[object, ...], None
        ] = {}
        self._verified_lane_sources: dict[
            tuple[object, ...], None
        ] = {}
        self._verified_attempt_inputs: dict[
            tuple[object, ...], None
        ] = {}
        self.generations = self.root / "generations"
        self.auxiliary = self.root / "auxiliary"
        self.zero_checkpoints = self.root / "zero_checkpoints"
        self.zero_sources = self.root / "zero_sources"
        self.public_observation_registry = (
            self.root / "public_observation_registry"
        )
        for path in (
            self.generations,
            self.auxiliary,
            self.zero_checkpoints,
            self.zero_sources,
            self.public_observation_registry,
        ):
            if path.is_symlink() or (
                path.exists() and not path.is_dir()
            ):
                raise ContiguousRunnerError(
                    f"runner path must be a regular directory: {path}"
                )
            path.mkdir(parents=True, exist_ok=True)
            os.chmod(path, 0o700, follow_symlinks=False)
        inventory = Contract.authoritative_inventory()
        Contract.validate_inventory(inventory)
        self._initialize_campaign(
            inventory=inventory,
            max_lanes=max_lanes,
            limit=limit,
            cost_window_id=cost_window_id,
            backend_configuration=backend_configuration,
            operator_configuration_sha256=(
                operator_configuration_sha256
            ),
            auxiliary_launch_configuration=auxiliary_configuration,
        )
        # A complete replay is the admission check.  No external action occurs in
        # construction, so callers explicitly choose when to call cycle().
        self.state()

    def _initialize_campaign(
        self,
        *,
        inventory: dict[str, int],
        max_lanes: int,
        limit: float | None,
        cost_window_id: str,
        backend_configuration: BackendConfiguration,
        operator_configuration_sha256: str | None,
        auxiliary_launch_configuration:
            Scheduler.AuxiliaryLaunchConfiguration,
    ) -> None:
        descriptor = _open_unaliased(
            self.root / ".init.lock", os.O_RDWR | os.O_CREAT
        )
        handle = os.fdopen(descriptor, "r+", encoding="utf-8")
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            self._initialize_campaign_locked(
                inventory=inventory,
                max_lanes=max_lanes,
                limit=limit,
                cost_window_id=cost_window_id,
                backend_configuration=backend_configuration,
                operator_configuration_sha256=(
                    operator_configuration_sha256
                ),
                auxiliary_launch_configuration=(
                    auxiliary_launch_configuration
                ),
            )
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            handle.close()

    def _initialize_campaign_locked(
        self,
        *,
        inventory: dict[str, int],
        max_lanes: int,
        limit: float | None,
        cost_window_id: str,
        backend_configuration: BackendConfiguration,
        operator_configuration_sha256: str | None,
        auxiliary_launch_configuration:
            Scheduler.AuxiliaryLaunchConfiguration,
    ) -> None:
        events = self.journal._read_authenticated()
        if not events:
            campaign_id = self._new_identifier("campaign")
            zero = {
                game: self._create_zero_checkpoint(game)
                for game in sorted(inventory)
            }
            zero_sources = {
                game: self._create_zero_source(game)
                for game in sorted(inventory)
            }
            zero_source_hashes = {
                value["sha256"] for value in zero_sources.values()
            }
            if len(zero_source_hashes) != 1:
                raise ContiguousRunnerError(
                    "L0 source scaffold differs between game lanes"
                )
            self.journal.append(
                event_id="campaign:genesis",
                kind="GENESIS",
                payload={
                    "schema": RUNNER_SCHEMA,
                    "campaign_id": campaign_id,
                    "inventory": inventory,
                    "inventory_sha256":
                        Contract.authoritative_inventory_sha256(inventory),
                    "max_lanes": max_lanes,
                    "limit": float(limit) if limit is not None else None,
                    "limit_units": Scheduler.limit_to_units(limit),
                    "cost_window_id": cost_window_id,
                    "scheduler_policy_sha256":
                        SCHEDULER_POLICY_SHA256,
                    "backend_configuration":
                        _backend_configuration_to_dict(
                            backend_configuration
                        ),
                    "operator_configuration_sha256":
                        operator_configuration_sha256,
                    "auxiliary_launch_configuration":
                        Scheduler.auxiliary_launch_configuration_to_dict(
                            auxiliary_launch_configuration
                        ),
                    "zero_checkpoints": zero,
                    "zero_sources": zero_sources,
                    "l0_source_tree_sha256":
                        next(iter(zero_source_hashes)),
                },
                recorded_at=self.clock(),
            )
        else:
            genesis = events[0]
            if genesis["kind"] != "GENESIS":
                raise ContiguousRunnerError("journal does not begin with GENESIS")
            payload = genesis["payload"]
            if (
                payload.get("inventory") != inventory
                or payload.get("inventory_sha256")
                != Contract.authoritative_inventory_sha256(inventory)
                or payload.get("max_lanes") != max_lanes
                or payload.get("limit")
                != (float(limit) if limit is not None else None)
                or payload.get("limit_units")
                != Scheduler.limit_to_units(limit)
                or payload.get("cost_window_id") != cost_window_id
                or payload.get("scheduler_policy_sha256")
                != SCHEDULER_POLICY_SHA256
                or payload.get("backend_configuration")
                != _backend_configuration_to_dict(
                    backend_configuration
                )
                or payload.get("operator_configuration_sha256")
                != operator_configuration_sha256
                or payload.get("auxiliary_launch_configuration")
                != Scheduler.auxiliary_launch_configuration_to_dict(
                    auxiliary_launch_configuration
                )
                or not isinstance(
                    payload.get("zero_sources"), dict
                )
                or set(payload["zero_sources"]) != set(inventory)
                or not _is_sha256(
                    payload.get("l0_source_tree_sha256")
                )
                or {
                    value.get("sha256")
                    for value in payload["zero_sources"].values()
                    if isinstance(value, dict)
                }
                != {payload["l0_source_tree_sha256"]}
            ):
                raise ContiguousRunnerError(
                    "runner configuration disagrees with durable genesis"
                )

    def _new_identifier(self, prefix: str) -> str:
        value = self.id_factory()
        if not _is_uuid4(value):
            raise ContiguousRunnerError(
                f"id_factory returned a non-UUIDv4 {prefix}: {value!r}"
            )
        return value

    def _fresh_decision_identifiers(
        self,
        state: Mapping[str, Any],
        labels: tuple[str, ...],
    ) -> tuple[str, ...]:
        """Skip a finite durable ID prefix and fail closed on exhaustion."""

        durable = state.get("used_scheduler_identifiers")
        auxiliary_threads = state.get("used_auxiliary_thread_ids")
        campaign_id = state.get("campaign_id")
        attempts = state.get("attempts")
        if (
            not isinstance(durable, list)
            or any(not isinstance(value, str) for value in durable)
            or not isinstance(auxiliary_threads, list)
            or any(
                not isinstance(value, str)
                for value in auxiliary_threads
            )
            or not isinstance(campaign_id, str)
            or not isinstance(attempts, dict)
            or not labels
        ):
            raise ContiguousRunnerError(
                "campaign identity registry is malformed"
            )
        blocked = set(durable) | set(auxiliary_threads) | {campaign_id}
        for attempt in attempts.values():
            if not isinstance(attempt, dict):
                raise ContiguousRunnerError(
                    "campaign attempt registry is malformed"
                )
            launched = attempt.get("launched")
            thread_id = (
                launched.codex_thread_id
                if isinstance(launched, BackendLaunch)
                else None
            )
            if isinstance(thread_id, str):
                blocked.add(thread_id)

        # A restarted deterministic source may replay every durable UUID once.
        # The extra slack bounds a broken source that repeats one collision
        # forever while allowing a finite durable prefix to be skipped.
        draws_remaining = len(blocked) + len(labels) + 64
        chosen: list[str] = []
        for label in labels:
            while draws_remaining > 0:
                draws_remaining -= 1
                value = self._new_identifier(label)
                if value in blocked:
                    continue
                blocked.add(value)
                chosen.append(value)
                break
            else:
                raise ContiguousRunnerError(
                    "id_factory could not produce a fresh durable identity"
                )
        return tuple(chosen)

    def _create_zero_checkpoint(self, game: str) -> dict[str, str]:
        path = self.zero_checkpoints / f"{game}.json"
        value = {
            "game": game,
            "reached": 0,
            "total_marginal_C": 0,
            "records": [],
            "final_path": [],
            "validated": False,
        }
        if path.exists():
            if _read_json_file(path) != value:
                raise ContiguousRunnerError(
                    f"zero checkpoint changed for {game}"
                )
        else:
            _write_new_file(path, value)
            _fsync_directory(self.zero_checkpoints)
        Contract.load_trusted_checkpoint(
            path,
            expected_game=game,
            authoritative_target=Contract.authoritative_inventory()[game],
        )
        return {"path": str(path), "sha256": _sha256_file(path)}

    def _create_zero_source(self, game: str) -> dict[str, str]:
        destination = self.zero_sources / game
        try:
            path_value, tree_sha256 = (
                self.input_builder.initialize_lane_source(
                    game, destination
                )
            )
        except AttributeError as exc:
            raise ContiguousRunnerError(
                "input builder cannot initialize the audited L0 source"
            ) from exc
        path = Path(path_value)
        if (
            path != destination
            or not path.is_absolute()
            or not _is_sha256(tree_sha256)
        ):
            raise ContiguousRunnerError(
                "input builder returned an invalid L0 source binding"
            )
        try:
            Contract._validate_regular_tree(
                path, label="zero source"
            )
        except Exception as exc:
            raise ContiguousRunnerError(
                "zero source tree is unsafe"
            ) from exc
        if Contract._tree_hash(path) != tree_sha256:
            raise ContiguousRunnerError(
                "zero source tree differs from its admitted hash"
            )
        return {"path": str(path), "sha256": tree_sha256}

    @staticmethod
    def _blank_lane(
        target: int,
        zero: dict[str, str],
        zero_source: dict[str, str],
    ) -> dict[str, Any]:
        return {
            "target": target,
            "reached": 0,
            "checkpoint_path": zero["path"],
            "checkpoint_sha256": zero["sha256"],
            "source_path": zero_source["path"],
            "source_tree_sha256": zero_source["sha256"],
            "no_progress": 0,
            "last_dispatch_sequence": 0,
            "wip": None,
            "active": None,
            "blocked": None,
            "clean_proposer_settlements": [],
            "public_observation_receipt_sha256s": [],
        }

    @staticmethod
    def _scheduler_wip(
        wip: WipSnapshot | None,
    ) -> Scheduler.WipBinding | None:
        if wip is None:
            return None
        try:
            return Scheduler.wip_binding_from_dict(
                Scheduler.wip_binding_to_dict(wip)
            )
        except Scheduler.SchedulerError as exc:
            raise ContiguousRunnerError(
                "scheduler-eligible WIP is invalid"
            ) from exc

    @classmethod
    def _scheduler_snapshot(
        cls,
        *,
        genesis: Mapping[str, Any],
        lanes: Mapping[str, Mapping[str, Any]],
        attempts: Mapping[str, Mapping[str, Any]],
        budget: Scheduler.BudgetState,
        journal_head_sequence: int,
        journal_head_digest: str,
        clean_proposer_settlements: tuple[
            Scheduler.CleanProposerSettlement, ...
        ] = (),
        complexity_rounds: tuple[
            Scheduler.ComplexityRoundState, ...
        ] = (),
        auxiliary_assignments: tuple[
            Scheduler.AuxiliaryAssignmentState, ...
        ] = (),
        sidecar_requests: tuple[
            Scheduler.SidecarRequestEvidence, ...
        ] = (),
    ) -> Scheduler.CampaignSnapshot:
        try:
            frontiers: list[Scheduler.Frontier] = []
            for game in sorted(lanes):
                lane = lanes[game]
                active_id = lane["active"]
                active = (
                    attempts.get(active_id)
                    if isinstance(active_id, str)
                    else None
                )
                wip = cls._scheduler_wip(lane["wip"])
                evidence = Scheduler.selection_evidence(
                    parent_source_path=lane["source_path"],
                    parent_source_tree_sha256=(
                        lane["source_tree_sha256"]
                    ),
                    candidate_source_path=(
                        wip.solver_source_path
                        if wip is not None else None
                    ),
                    candidate_source_tree_sha256=(
                        wip.solver_source_tree_sha256
                        if wip is not None else None
                    ),
                )
                public_receipts = tuple(
                    sorted(
                        set(
                            lane[
                                "public_observation_receipt_sha256s"
                            ]
                        )
                    )
                )
                current_frontier_sha256 = frontier_sha256(
                    game,
                    lane["reached"],
                    lane["checkpoint_sha256"],
                )
                frontiers.append(Scheduler.Frontier(
                    game=game,
                    target=lane["target"],
                    reached=lane["reached"],
                    no_progress=lane["no_progress"],
                    last_dispatch_sequence=(
                        lane["last_dispatch_sequence"]
                    ),
                    parent_checkpoint_sha256=(
                        lane["checkpoint_sha256"]
                    ),
                    parent_source_path=lane["source_path"],
                    parent_source_tree_sha256=(
                        lane["source_tree_sha256"]
                    ),
                    frontier_sha256=current_frontier_sha256,
                    active_attempt_id=active_id,
                    draining=(
                        active is not None
                        and active.get("phase") == "DRAINING"
                    ),
                    blocked_reason=lane["blocked"],
                    wip=wip,
                    evidence=evidence,
                    public_observation_receipt_sha256s=(
                        public_receipts
                    ),
                    observation_ledger_sha256=(
                        Scheduler.public_observation_ledger_sha256(
                            game=game,
                            frontier_sha256=(
                                current_frontier_sha256
                            ),
                            parent_checkpoint_sha256=(
                                lane["checkpoint_sha256"]
                            ),
                            receipt_sha256s=public_receipts,
                        )
                    ),
                ))
            snapshot = Scheduler.CampaignSnapshot(
                campaign_id=genesis["campaign_id"],
                journal_head_sequence=journal_head_sequence,
                journal_head_digest=journal_head_digest,
                inventory=tuple(
                    sorted(genesis["inventory"].items())
                ),
                max_lanes=genesis["max_lanes"],
                frontiers=tuple(frontiers),
                budget=budget,
                clean_proposer_settlements=(
                    clean_proposer_settlements
                ),
                complexity_rounds=complexity_rounds,
                auxiliary_assignments=auxiliary_assignments,
                sidecar_requests=sidecar_requests,
            )
            return Scheduler.validate_snapshot(snapshot)
        except Scheduler.SchedulerError as exc:
            raise ContiguousRunnerError(
                "campaign state cannot form a scheduler snapshot"
            ) from exc

    @staticmethod
    def _remember_verified_pointer(
        cache: dict[tuple[object, ...], None],
        key: tuple[object, ...],
    ) -> None:
        if len(cache) >= 512:
            cache.pop(next(iter(cache)))
        cache[key] = None

    def _validate_lane_checkpoint_cached(
        self,
        *,
        game: str,
        lane: Mapping[str, Any],
    ) -> None:
        path = Path(str(lane["checkpoint_path"]))
        pointer = _regular_file_pointer(path)
        key = (
            str(path),
            game,
            int(lane["target"]),
            int(lane["reached"]),
            str(lane["checkpoint_sha256"]),
            pointer,
        )
        cache = getattr(self, "_verified_lane_checkpoints", None)
        if cache is None:
            cache = {}
            self._verified_lane_checkpoints = cache
        if key in cache:
            if _regular_file_pointer(path) != pointer:
                raise ContiguousRunnerError(
                    f"lane checkpoint changed while inspected: {game}"
                )
            return
        checkpoint = Contract.load_trusted_checkpoint(
            path,
            expected_game=game,
            authoritative_target=int(lane["target"]),
        )
        if (
            checkpoint.reached != lane["reached"]
            or _sha256_file(path) != lane["checkpoint_sha256"]
        ):
            raise ContiguousRunnerError(
                f"lane checkpoint changed or disagrees: {game}"
            )
        if _regular_file_pointer(path) != pointer:
            raise ContiguousRunnerError(
                f"lane checkpoint changed while inspected: {game}"
            )
        self._remember_verified_pointer(cache, key)

    def _validate_lane_source_cached(
        self,
        *,
        game: str,
        lane: Mapping[str, Any],
    ) -> None:
        source_root = Path(str(lane["source_path"]))
        pointer = _regular_tree_pointer(source_root)
        key = (
            str(source_root),
            str(lane["source_tree_sha256"]),
            pointer,
        )
        cache = getattr(self, "_verified_lane_sources", None)
        if cache is None:
            cache = {}
            self._verified_lane_sources = cache
        if key in cache:
            if _regular_tree_pointer(source_root) != pointer:
                raise ContiguousRunnerError(
                    f"lane source changed while inspected: {game}"
                )
            return
        try:
            Contract._validate_regular_tree(
                source_root, label=f"{game} lane source"
            )
        except Exception as exc:
            raise ContiguousRunnerError(
                f"lane source is unsafe: {game}"
            ) from exc
        if (
            not source_root.is_absolute()
            or Contract._tree_hash(source_root)
            != lane["source_tree_sha256"]
        ):
            raise ContiguousRunnerError(
                f"lane source changed or disagrees: {game}"
            )
        if _regular_tree_pointer(source_root) != pointer:
            raise ContiguousRunnerError(
                f"lane source changed while inspected: {game}"
            )
        self._remember_verified_pointer(cache, key)

    def state(self) -> dict[str, Any]:
        events = self.journal._read_authenticated()
        if not events or events[0]["kind"] != "GENESIS":
            raise ContiguousRunnerError("missing durable genesis")
        auxiliary_events = tuple(
            event
            for event in events
            if (
                str(event.get("kind", "")).startswith("AUXILIARY_")
                or event.get("kind")
                in {
                    "NATIVE_SIDECAR_REQUEST_ADMITTED",
                    "SUPERVISORY_SIDECAR_REQUEST_ADMITTED",
                }
            )
        )
        trusted_auxiliary_digests = getattr(
            self, "_trusted_auxiliary_event_digests", set()
        )
        if any(
            event["digest"] not in trusted_auxiliary_digests
            for event in auxiliary_events
        ):
            try:
                Scheduler.validate_journal_event_sequence(events)
            except Scheduler.SchedulerError as exc:
                raise ContiguousRunnerError(
                    "auxiliary journal lifecycle is invalid"
                ) from exc
            trusted_auxiliary_digests.update(
                event["digest"] for event in auxiliary_events
            )
            self._trusted_auxiliary_event_digests = (
                trusted_auxiliary_digests
            )
        genesis = events[0]["payload"]
        inventory = genesis.get("inventory")
        Contract.validate_inventory(inventory)
        zero = genesis.get("zero_checkpoints")
        if not isinstance(zero, dict) or set(zero) != set(inventory):
            raise ContiguousRunnerError("zero-checkpoint map mismatch")
        zero_sources = genesis.get("zero_sources")
        if (
            not isinstance(zero_sources, dict)
            or set(zero_sources) != set(inventory)
        ):
            raise ContiguousRunnerError("zero-source map mismatch")
        lanes = {
            game: self._blank_lane(
                inventory[game], zero[game], zero_sources[game]
            )
            for game in sorted(inventory)
        }
        attempts: dict[str, dict[str, Any]] = {}
        if (
            not _safe_identifier(genesis.get("cost_window_id"))
            or (
                genesis.get("operator_configuration_sha256")
                is not None
                and not _is_sha256(
                    genesis.get("operator_configuration_sha256")
                )
            )
            or genesis.get("limit_units")
            != Scheduler.limit_to_units(genesis.get("limit"))
            or genesis.get("scheduler_policy_sha256")
            != SCHEDULER_POLICY_SHA256
        ):
            raise ContiguousRunnerError(
                "genesis lacks the canonical scheduler/budget binding"
            )
        try:
            auxiliary_configuration = (
                Scheduler.auxiliary_launch_configuration_from_dict(
                    genesis.get("auxiliary_launch_configuration")
                )
            )
        except Scheduler.SchedulerError as exc:
            raise ContiguousRunnerError(
                "genesis auxiliary launch configuration is invalid"
            ) from exc
        if auxiliary_configuration != self.auxiliary_launch_configuration:
            raise ContiguousRunnerError(
                "live auxiliary configuration disagrees with genesis"
            )
        budget = Scheduler.BudgetState(
            cost_window_id=genesis["cost_window_id"],
            limit_units=genesis["limit_units"],
            settled_units=0,
            live_reservations=(),
        )
        pending_decision: Scheduler.SchedulerDecision | None = None
        pending_auxiliary_decision: (
            Scheduler.AuxiliaryDecision | None
        ) = None
        auxiliary_assignments: dict[str, dict[str, Any]] = {}
        sidecar_requests: dict[str, dict[str, Any]] = {}
        complexity_rounds: list[Scheduler.ComplexityRoundState] = []
        used_decision_ids: set[str] = set()
        used_attempt_ids: set[str] = set()
        used_generation_ids: set[str] = set()
        used_reservation_ids: set[str] = set()
        used_expert_ids: set[str] = set()
        used_thread_ids: set[str] = set()
        failure_operation_circuits: dict[
            str, dict[str, Any]
        ] = {}
        failure_domain_circuits: dict[
            str, dict[str, Any]
        ] = {}
        operator_incident: dict[str, Any] | None = None
        substrate_incident: dict[str, Any] | None = None
        storage_incident: dict[str, Any] | None = None
        storage_quiescence: dict[str, Any] | None = None

        reducer_start = 1
        cached_reducer = getattr(self, "_reducer_checkpoint", None)
        if (
            isinstance(cached_reducer, _ReducerCheckpoint)
            and cached_reducer.genesis_digest == events[0]["digest"]
            and 1 <= cached_reducer.head_sequence <= len(events)
            and events[cached_reducer.head_sequence - 1]["digest"]
            == cached_reducer.head_digest
        ):
            # The journal reader has independently authenticated the complete
            # current hash chain.  Reuse only the already-validated prefix at
            # its exact sequence/digest coordinate, then validate the suffix
            # normally.  Current executable/promotion inputs are intentionally
            # *not* cached and are checked again below.
            lanes = _clone_reducer_lanes(cached_reducer.lanes)
            attempts = _clone_reducer_records(
                cached_reducer.attempts
            )
            budget = cached_reducer.budget
            pending_decision = cached_reducer.pending_decision
            pending_auxiliary_decision = (
                cached_reducer.pending_auxiliary_decision
            )
            auxiliary_assignments = _clone_reducer_records(
                cached_reducer.auxiliary_assignments
            )
            sidecar_requests = _clone_reducer_records(
                cached_reducer.sidecar_requests
            )
            complexity_rounds = list(
                cached_reducer.complexity_rounds
            )
            used_decision_ids = set(
                cached_reducer.used_decision_ids
            )
            used_attempt_ids = set(
                cached_reducer.used_attempt_ids
            )
            used_generation_ids = set(
                cached_reducer.used_generation_ids
            )
            used_reservation_ids = set(
                cached_reducer.used_reservation_ids
            )
            used_expert_ids = set(
                cached_reducer.used_expert_ids
            )
            used_thread_ids = set(
                cached_reducer.used_thread_ids
            )
            failure_operation_circuits = copy.deepcopy(
                cached_reducer.failure_operation_circuits
            )
            failure_domain_circuits = copy.deepcopy(
                cached_reducer.failure_domain_circuits
            )
            operator_incident = copy.deepcopy(
                cached_reducer.operator_incident
            )
            substrate_incident = copy.deepcopy(
                cached_reducer.substrate_incident
            )
            storage_incident = copy.deepcopy(
                cached_reducer.storage_incident
            )
            storage_quiescence = copy.deepcopy(
                cached_reducer.storage_quiescence
            )
            reducer_start = cached_reducer.head_sequence

        for event in events[reducer_start:]:
            kind = event["kind"]
            payload = event["payload"]
            if (
                pending_decision is not None
                and kind not in {
                    "ATTEMPT_RESERVED",
                    "JOURNAL_OR_STORAGE_EXHAUSTED",
                    "STORAGE_EMERGENCY_QUIESCED",
                }
            ):
                raise ContiguousRunnerError(
                    "scheduler decision is not immediately followed by "
                    "its reservation"
                )
            if (
                pending_auxiliary_decision is not None
                and kind not in {
                    "AUXILIARY_RESERVED",
                    "JOURNAL_OR_STORAGE_EXHAUSTED",
                    "STORAGE_EMERGENCY_QUIESCED",
                }
            ):
                raise ContiguousRunnerError(
                    "auxiliary decision is not immediately followed by "
                    "its reservation"
                )
            if kind == "JOURNAL_OR_STORAGE_EXHAUSTED":
                required = {
                    "reason_code",
                    "failed_event_id",
                    "failed_event_kind",
                    "failure_stage",
                    "error_code",
                    "storage_snapshot",
                    "solver_authority",
                    "wip_authority",
                    "cost_authority",
                    "promotion_authority",
                    "status",
                }
                snapshot = payload.get("storage_snapshot")
                if (
                    set(payload) != required
                    or storage_incident is not None
                    or payload["reason_code"]
                    != "journal_or_storage_exhausted"
                    or not _safe_identifier(
                        payload["failed_event_id"]
                    )
                    or not _safe_identifier(
                        payload["failed_event_kind"]
                    )
                    or not _safe_identifier(
                        payload["failure_stage"]
                    )
                    or not _safe_identifier(payload["error_code"])
                    or not isinstance(snapshot, dict)
                    or any(
                        payload[name] is not False
                        for name in (
                            "solver_authority",
                            "wip_authority",
                            "cost_authority",
                            "promotion_authority",
                        )
                    )
                    or payload["status"] != "OPERATOR_INCIDENT"
                ):
                    raise ContiguousRunnerError(
                        "journal/storage incident schema mismatch"
                    )
                if snapshot and (
                    set(snapshot)
                    != {
                        "schema", "kind", "filesystem_device",
                        "available_bytes", "available_inodes",
                        "required_event_bytes",
                        "minimum_free_bytes",
                        "minimum_free_inodes",
                        "byte_admitted", "inode_admitted",
                    }
                    or snapshot.get("schema") != 1
                    or snapshot.get("kind")
                    != "contiguous_journal_filesystem_admission"
                    or not all(
                        isinstance(snapshot.get(name), int)
                        and not isinstance(
                            snapshot.get(name), bool
                        )
                        and snapshot[name] >= 0
                        for name in (
                            "filesystem_device",
                            "available_bytes",
                            "available_inodes",
                            "required_event_bytes",
                            "minimum_free_bytes",
                            "minimum_free_inodes",
                        )
                    )
                    or not isinstance(
                        snapshot.get("byte_admitted"), bool
                    )
                    or not isinstance(
                        snapshot.get("inode_admitted"), bool
                    )
                ):
                    raise ContiguousRunnerError(
                        "journal/storage filesystem snapshot is "
                        "malformed"
                    )
                storage_incident = {
                    **dict(payload),
                    "incident_event_sequence": event["sequence"],
                    "incident_event_digest": event["digest"],
                }
                continue
            if kind == "STORAGE_EMERGENCY_QUIESCED":
                required = {
                    "storage_incident_event_sequence",
                    "storage_incident_event_digest",
                    "primary_containments",
                    "auxiliary_aborts",
                    "promotion_quarantines",
                    "all_primary_children_absent",
                    "all_auxiliary_children_absent",
                    "all_promotions_non_authoritative",
                    "solver_authority",
                    "wip_authority",
                    "cost_authority",
                    "promotion_authority",
                    "status",
                }
                primary = payload.get("primary_containments")
                auxiliary = payload.get("auxiliary_aborts")
                promotions = payload.get("promotion_quarantines")
                if (
                    set(payload) != required
                    or storage_incident is None
                    or storage_quiescence is not None
                    or payload[
                        "storage_incident_event_sequence"
                    ]
                    != storage_incident[
                        "incident_event_sequence"
                    ]
                    or payload["storage_incident_event_digest"]
                    != storage_incident["incident_event_digest"]
                    or not isinstance(primary, list)
                    or not isinstance(auxiliary, list)
                    or not isinstance(promotions, list)
                    or any(
                        payload[name] is not True
                        for name in (
                            "all_primary_children_absent",
                            "all_auxiliary_children_absent",
                            "all_promotions_non_authoritative",
                        )
                    )
                    or any(
                        payload[name] is not False
                        for name in (
                            "solver_authority",
                            "wip_authority",
                            "cost_authority",
                            "promotion_authority",
                        )
                    )
                    or payload["status"] != "QUIESCED"
                ):
                    raise ContiguousRunnerError(
                        "storage emergency quiescence schema mismatch"
                    )
                expected_primary = {
                    attempt_id
                    for attempt_id, attempt in attempts.items()
                    if attempt["phase"]
                    in {
                        "PREPARED",
                        "BACKEND_PREPARED",
                        "RUNNING",
                        "DRAINING",
                        "EXITED",
                        "COLLECTED",
                        "COLLECTION_REJECTED",
                    }
                }
                observed_primary: set[str] = set()
                for item in primary:
                    if (
                        not isinstance(item, dict)
                        or set(item)
                        != {
                            "attempt_id",
                            "prior_phase",
                            "containment",
                        }
                    ):
                        raise ContiguousRunnerError(
                            "storage primary containment is malformed"
                        )
                    selected_attempt_id = item["attempt_id"]
                    attempt = attempts.get(selected_attempt_id)
                    try:
                        containment = BackendEmergencyContainment(
                            **item["containment"]
                        )
                    except (TypeError, AttributeError) as exc:
                        raise ContiguousRunnerError(
                            "storage primary containment proof "
                            "changed schema"
                        ) from exc
                    if (
                        selected_attempt_id in observed_primary
                        or selected_attempt_id not in expected_primary
                        or attempt is None
                        or item["prior_phase"] != attempt["phase"]
                        or any(
                            getattr(containment, name) is not True
                            for name in (
                                "attempt_container_absent",
                                "controller_roles_absent",
                                "arena_resources_absent",
                                "rpc_endpoints_absent",
                                "workspace_probe_containers_absent",
                                "host_process_groups_absent",
                                "containment_canaries_absent",
                                "no_descendants",
                            )
                        )
                    ):
                        raise ContiguousRunnerError(
                            "storage primary containment coverage "
                            "changed"
                        )
                    receipt = _validate_bound_receipt(
                        containment.containment_receipt_path,
                        containment.containment_receipt_sha256,
                        expected_path=(
                            Path(
                                attempt["spec"]
                                .host_transcript_path
                            ).parent
                            / "storage_emergency_containment.json"
                        ),
                        expected_kind=(
                            "contiguous_storage_emergency_"
                            "containment"
                        ),
                        spec=attempt["spec"],
                    )
                    receipt_expected = {
                        "schema",
                        "kind",
                        "campaign_id",
                        "generation_id",
                        "attempt_id",
                        "attempt_spec_sha256",
                        "prior_phase",
                        "reason",
                        "launched_container_id",
                        "attempt_container_absent",
                        "controller_roles_absent",
                        "arena_resources_absent",
                        "rpc_endpoints_absent",
                        "workspace_probe_containers_absent",
                        "host_process_groups_absent",
                        "containment_canaries_absent",
                        "no_descendants",
                        "solver_authority",
                        "wip_authority",
                        "cost_authority",
                        "promotion_authority",
                        "status",
                    }
                    if (
                        set(receipt) != receipt_expected
                        or receipt["prior_phase"]
                        != item["prior_phase"]
                        or receipt["reason"]
                        != "journal_or_storage_exhausted"
                        or any(
                            receipt[name]
                            != getattr(containment, name)
                            for name in (
                                "launched_container_id",
                                "attempt_container_absent",
                                "controller_roles_absent",
                                "arena_resources_absent",
                                "rpc_endpoints_absent",
                                "workspace_probe_containers_absent",
                                "host_process_groups_absent",
                                "containment_canaries_absent",
                                "no_descendants",
                            )
                        )
                        or any(
                            receipt[name] is not False
                            for name in (
                                "solver_authority",
                                "wip_authority",
                                "cost_authority",
                                "promotion_authority",
                            )
                        )
                        or receipt["status"] != "QUIESCED"
                    ):
                        raise ContiguousRunnerError(
                            "storage primary containment receipt "
                            "grants authority"
                        )
                    observed_primary.add(selected_attempt_id)
                if observed_primary != expected_primary:
                    raise ContiguousRunnerError(
                        "storage primary containment coverage is "
                        "incomplete"
                    )
                expected_auxiliary = {
                    assignment_id
                    for assignment_id, assignment
                    in auxiliary_assignments.items()
                    if assignment["state"].phase
                    in Scheduler.AUXILIARY_ACTIVE_PHASES
                }
                observed_auxiliary: set[str] = set()
                for item in auxiliary:
                    if (
                        not isinstance(item, dict)
                        or set(item)
                        != {
                            "assignment_id",
                            "prior_phase",
                            "teardown_receipt_path",
                            "teardown_receipt_sha256",
                            "no_descendants",
                            "cost_authority",
                        }
                        or item["assignment_id"]
                        in observed_auxiliary
                        or item["assignment_id"]
                        not in expected_auxiliary
                        or item["no_descendants"] is not True
                        or item["cost_authority"] is not False
                    ):
                        raise ContiguousRunnerError(
                            "storage auxiliary abort is malformed"
                        )
                    assignment = auxiliary_assignments[
                        item["assignment_id"]
                    ]
                    if (
                        item["prior_phase"]
                        != assignment["state"].phase
                    ):
                        raise ContiguousRunnerError(
                            "storage auxiliary phase changed"
                        )
                    if item["prior_phase"] == "RUNNING":
                        expected_teardown = {
                            "schema": 1,
                            "kind":
                                "auxiliary_backend_abort_teardown",
                            "assignment_id":
                                item["assignment_id"],
                            "backend_contract_sha256":
                                assignment["decision"]
                                .backend_contract_sha256,
                            "prior_phase": "RUNNING",
                            "descendants_absent": True,
                            "live_lineage_mutated": False,
                        }
                        self._verify_auxiliary_receipt(
                            assignment["decision"],
                            item["teardown_receipt_path"],
                            item["teardown_receipt_sha256"],
                            expected=expected_teardown,
                            label=(
                                "storage emergency auxiliary "
                                "teardown"
                            ),
                        )
                    elif (
                        item["teardown_receipt_path"] is not None
                        or item["teardown_receipt_sha256"] is not None
                    ):
                        raise ContiguousRunnerError(
                            "unlaunched storage auxiliary gained "
                            "teardown evidence"
                        )
                    observed_auxiliary.add(item["assignment_id"])
                if observed_auxiliary != expected_auxiliary:
                    raise ContiguousRunnerError(
                        "storage auxiliary abort coverage is "
                        "incomplete"
                    )
                expected_promotions = {
                    attempt_id
                    for attempt_id, attempt in attempts.items()
                    if attempt["phase"] == "PROMOTING"
                }
                observed_promotions: set[str] = set()
                for item in promotions:
                    if (
                        not isinstance(item, dict)
                        or set(item)
                        != {
                            "attempt_id",
                            "external_commit_observed",
                            "external_commit_sha256",
                            "promotion_authority",
                        }
                        or item["attempt_id"]
                        in observed_promotions
                        or item["attempt_id"]
                        not in expected_promotions
                        or not isinstance(
                            item["external_commit_observed"], bool
                        )
                        or (
                            item["external_commit_observed"]
                            and not _is_sha256(
                                item["external_commit_sha256"]
                            )
                        )
                        or (
                            not item["external_commit_observed"]
                            and item["external_commit_sha256"]
                            is not None
                        )
                        or item["promotion_authority"] is not False
                    ):
                        raise ContiguousRunnerError(
                            "storage promotion quarantine is "
                            "malformed"
                        )
                    observed_promotions.add(item["attempt_id"])
                if observed_promotions != expected_promotions:
                    raise ContiguousRunnerError(
                        "storage promotion quarantine coverage is "
                        "incomplete"
                    )
                storage_quiescence = copy.deepcopy(payload)
                continue
            if kind == "FAILURE_CIRCUIT_FAILURE":
                if set(payload) != {
                    "attempt_id",
                    "operation",
                    "fault_domain",
                    "operation_consecutive",
                    "operation_failure_index",
                    "domain_consecutive",
                    "domain_failure_index",
                    "backoff_seconds",
                    "retry_not_before",
                }:
                    raise ContiguousRunnerError(
                        "failure circuit event schema mismatch"
                    )
                operation = payload["operation"]
                domain = payload["fault_domain"]
                circuit_attempt_id = payload["attempt_id"]
                if (
                    not _safe_identifier(operation)
                    or domain not in FAILURE_FAULT_DOMAINS
                    or (
                        circuit_attempt_id is not None
                        and not _safe_identifier(circuit_attempt_id)
                    )
                ):
                    raise ContiguousRunnerError(
                        "failure circuit identity is malformed"
                    )
                if operation == "substrate_health_reprobe" and (
                    substrate_incident is None
                    or substrate_incident[
                        "circuit_failure_recorded"
                    ]
                    is not False
                    or circuit_attempt_id
                    != substrate_incident["attempt_id"]
                    or domain != "controller_substrate"
                ):
                    raise ContiguousRunnerError(
                        "substrate circuit failure is duplicated or "
                        "unbound"
                    )
                if operation == "backend_terminal":
                    terminal_attempt = attempts.get(
                        circuit_attempt_id
                    )
                    if (
                        terminal_attempt is None
                        or terminal_attempt["phase"] != "TORN_DOWN"
                        or terminal_attempt[
                            "terminal_failure_circuit_recorded"
                        ]
                    ):
                        raise ContiguousRunnerError(
                            "terminal failure circuit is duplicated or "
                            "out of lifecycle order"
                        )
                    terminal_attempt[
                        "terminal_failure_circuit_recorded"
                    ] = True
                operation_key = f"{operation}:{domain}"
                operation_state = (
                    failure_operation_circuits.get(
                        operation_key,
                        {
                            "consecutive": 0,
                            "failure_index": 0,
                            "retry_not_before": None,
                        },
                    )
                )
                domain_state = failure_domain_circuits.get(
                    domain,
                    {
                        "consecutive": 0,
                        "failure_index": 0,
                        "retry_not_before": None,
                        "last_operation": None,
                    },
                )
                operation_consecutive = (
                    operation_state["consecutive"] + 1
                )
                domain_consecutive = (
                    domain_state["consecutive"] + 1
                )
                backoff_schedule = (
                    SUBSTRATE_HEALTH_REPROBE_BACKOFF_SECONDS
                    if operation == "substrate_health_reprobe"
                    else OPERATION_RETRY_BACKOFF_SECONDS
                )
                expected_backoff = (
                    backoff_schedule[
                        min(
                            max(
                                operation_consecutive,
                                domain_consecutive,
                            ),
                            len(backoff_schedule),
                        )
                        - 1
                    ]
                )
                if (
                    payload["operation_consecutive"]
                    != operation_consecutive
                    or payload["operation_failure_index"]
                    != operation_state["failure_index"] + 1
                    or payload["domain_consecutive"]
                    != domain_consecutive
                    or payload["domain_failure_index"]
                    != domain_state["failure_index"] + 1
                    or payload["backoff_seconds"]
                    != expected_backoff
                    or payload["retry_not_before"]
                    != float(event["recorded_at"]) + expected_backoff
                ):
                    raise ContiguousRunnerError(
                        "failure circuit count/backoff is noncanonical"
                    )
                failure_operation_circuits[operation_key] = {
                    "consecutive": operation_consecutive,
                    "failure_index":
                        payload["operation_failure_index"],
                    "retry_not_before":
                        payload["retry_not_before"],
                }
                failure_domain_circuits[domain] = {
                    "consecutive": domain_consecutive,
                    "failure_index":
                        payload["domain_failure_index"],
                    "retry_not_before":
                        payload["retry_not_before"],
                    "last_operation": operation,
                }
                if operation == "substrate_health_reprobe":
                    substrate_incident[
                        "circuit_failure_recorded"
                    ] = True
                continue
            if kind == "FAILURE_CIRCUIT_RESET":
                if set(payload) != {
                    "attempt_id",
                    "operation",
                    "fault_domain",
                    "operation_consecutive",
                    "domain_consecutive",
                    "evidence_kind",
                    "reset_operation",
                    "reset_domain",
                }:
                    raise ContiguousRunnerError(
                        "failure circuit reset schema mismatch"
                    )
                operation = payload["operation"]
                domain = payload["fault_domain"]
                circuit_attempt_id = payload["attempt_id"]
                operation_key = f"{operation}:{domain}"
                operation_state = (
                    failure_operation_circuits.get(operation_key)
                )
                domain_state = failure_domain_circuits.get(domain)
                evidence_index = int(event["sequence"]) - 2
                while (
                    evidence_index >= 0
                    and events[evidence_index]["kind"]
                    == "FAILURE_CIRCUIT_RESET"
                ):
                    evidence_index -= 1
                evidence_event = (
                    events[evidence_index]
                    if evidence_index >= 0
                    else None
                )
                evidence_contract = {
                    "attempt_prepared":
                        (
                            "ATTEMPT_PREPARED",
                            "attempt_id",
                            {"input_materialize"},
                        ),
                    "backend_prepared":
                        (
                            "BACKEND_PREPARED",
                            "attempt_id",
                            {"backend_prepare"},
                        ),
                    "attempt_launched":
                        (
                            "ATTEMPT_LAUNCHED",
                            "attempt_id",
                            {"backend_launch"},
                        ),
                    "backend_poll_observation":
                        (
                            {
                                "ATTEMPT_OBSERVED",
                                "ATTEMPT_EXITED",
                            },
                            "attempt_id",
                            {"backend_poll"},
                        ),
                    "attempt_collected":
                        (
                            "ATTEMPT_COLLECTED",
                            "attempt_id",
                            {"backend_collect"},
                        ),
                    "attempt_torn_down":
                        (
                            "ATTEMPT_TORN_DOWN",
                            "attempt_id",
                            {"backend_teardown"},
                        ),
                    "promotion_committed":
                        (
                            "PROMOTION_COMMITTED",
                            "attempt_id",
                            {
                                "promotion_commit",
                                "promotion_recover",
                            },
                        ),
                    "substrate_health_restored":
                        (
                            "SUBSTRATE_HEALTH_RESTORED",
                            "attempt_id",
                            {"substrate_health_reprobe"},
                        ),
                    "auxiliary_input_prepared":
                        (
                            "AUXILIARY_INPUT_PREPARED",
                            "assignment_id",
                            {"auxiliary_prepare"},
                        ),
                    "auxiliary_launched":
                        (
                            "AUXILIARY_LAUNCHED",
                            "assignment_id",
                            {"auxiliary_launch"},
                        ),
                    "auxiliary_result_quarantined":
                        (
                            "AUXILIARY_RESULT_QUARANTINED",
                            "assignment_id",
                            {
                                "auxiliary_collect",
                                "auxiliary_teardown",
                            },
                        ),
                    "auxiliary_output_rejected":
                        (
                            "AUXILIARY_OUTPUT_REJECTED",
                            "assignment_id",
                            {"auxiliary_admit"},
                        ),
                    "auxiliary_output_admitted":
                        (
                            {
                                "AUXILIARY_OUTPUT_ADMITTED",
                                "AUXILIARY_PROFILE_ADMITTED",
                            },
                            "assignment_id",
                            {"auxiliary_admit"},
                        ),
                }
                evidence_rule = evidence_contract.get(
                    payload["evidence_kind"]
                )
                evidence_kinds = (
                    set()
                    if evidence_rule is None
                    else (
                        evidence_rule[0]
                        if isinstance(evidence_rule[0], set)
                        else {evidence_rule[0]}
                    )
                )
                if (
                    not _safe_identifier(operation)
                    or domain not in FAILURE_FAULT_DOMAINS
                    or (
                        circuit_attempt_id is not None
                        and not _safe_identifier(circuit_attempt_id)
                    )
                    or not _safe_identifier(
                        payload["evidence_kind"]
                    )
                    or evidence_event is None
                    or evidence_rule is None
                    or evidence_event["kind"]
                    not in evidence_kinds
                    or evidence_event["payload"].get(
                        evidence_rule[1]
                    )
                    != circuit_attempt_id
                    or operation not in evidence_rule[2]
                    or operation_state is None
                    or domain_state is None
                    or not isinstance(
                        payload["reset_operation"], bool
                    )
                    or not isinstance(payload["reset_domain"], bool)
                    or not (
                        payload["reset_operation"]
                        or payload["reset_domain"]
                    )
                    or (
                        payload["reset_operation"]
                        and operation_state["consecutive"] == 0
                    )
                    or (
                        payload["reset_domain"]
                        and domain_state["consecutive"] == 0
                    )
                    or (
                        payload["reset_domain"]
                        and domain_state.get("last_operation")
                        != operation
                    )
                    or payload["operation_consecutive"]
                    != operation_state["consecutive"]
                    or payload["domain_consecutive"]
                    != domain_state["consecutive"]
                ):
                    raise ContiguousRunnerError(
                        "failure circuit reset lacks matching success"
                    )
                failure_operation_circuits[operation_key] = {
                    **operation_state,
                    "consecutive": (
                        0
                        if payload["reset_operation"]
                        else operation_state["consecutive"]
                    ),
                    "retry_not_before": (
                        None
                        if payload["reset_operation"]
                        else operation_state["retry_not_before"]
                    ),
                }
                failure_domain_circuits[domain] = {
                    **domain_state,
                    "consecutive": (
                        0
                        if payload["reset_domain"]
                        else domain_state["consecutive"]
                    ),
                    "retry_not_before": (
                        None
                        if payload["reset_domain"]
                        else domain_state["retry_not_before"]
                    ),
                    "last_operation": (
                        None
                        if payload["reset_domain"]
                        else domain_state["last_operation"]
                    ),
                }
                continue
            if kind == "OPERATOR_INCIDENT":
                deterministic_substrate_incident = (
                    set(payload)
                    == {
                        "attempt_id",
                        "operation",
                        "fault_domain",
                        "operation_consecutive",
                        "domain_consecutive",
                        "threshold",
                        "reason_code",
                    }
                    and payload.get("operation")
                    == "substrate_health_reprobe"
                    and payload.get("fault_domain")
                    == "controller_substrate"
                    and payload.get("threshold") == 2
                    and payload.get("reason_code")
                    == (
                        "deterministic_substrate_configuration_"
                        "repeated"
                    )
                    and substrate_incident is not None
                    and payload.get("attempt_id")
                    == substrate_incident["attempt_id"]
                    and substrate_incident["failure_class"]
                    == "DETERMINISTIC_CONFIGURATION"
                    and substrate_incident["health_probe_count"] == 1
                    and isinstance(
                        substrate_incident["last_health_probe"],
                        dict,
                    )
                    and substrate_incident[
                        "last_health_probe"
                    ].get("failure_class")
                    == "DETERMINISTIC_CONFIGURATION"
                    and substrate_incident[
                        "last_health_probe"
                    ].get("failure_code")
                    == substrate_incident["failure_code"]
                    and payload.get("operation_consecutive") == 2
                    and payload.get("domain_consecutive") == 2
                )
                if (
                    set(payload)
                    != {
                        "attempt_id",
                        "operation",
                        "fault_domain",
                        "operation_consecutive",
                        "domain_consecutive",
                        "threshold",
                        "reason_code",
                    }
                    or operator_incident is not None
                ):
                    raise ContiguousRunnerError(
                        "operator incident schema/order mismatch"
                    )
                operation = payload["operation"]
                domain = payload["fault_domain"]
                operation_state = (
                    failure_operation_circuits.get(
                        f"{operation}:{domain}"
                    )
                )
                domain_state = failure_domain_circuits.get(domain)
                if deterministic_substrate_incident:
                    operator_incident = dict(payload)
                    continue
                if (
                    not _safe_identifier(operation)
                    or domain not in FAILURE_FAULT_DOMAINS
                    or (
                        payload["attempt_id"] is not None
                        and not _safe_identifier(
                            payload["attempt_id"]
                        )
                    )
                    or payload["threshold"]
                    != FAILURE_CIRCUIT_THRESHOLD
                    or payload["reason_code"]
                    != "failure_circuit_exhausted"
                    or operation_state is None
                    or domain_state is None
                    or payload["operation_consecutive"]
                    != operation_state["consecutive"]
                    or payload["domain_consecutive"]
                    != domain_state["consecutive"]
                    or max(
                        operation_state["consecutive"],
                        domain_state["consecutive"],
                    ) < FAILURE_CIRCUIT_THRESHOLD
                ):
                    raise ContiguousRunnerError(
                        "operator incident lacks exhausted circuit"
                    )
                operator_incident = dict(payload)
                continue
            if kind == "SCHEDULER_DECISION":
                if (
                    set(payload) != {"decision"}
                    or pending_decision is not None
                ):
                    raise ContiguousRunnerError(
                        "scheduler decision event schema/order mismatch"
                    )
                try:
                    decision = Scheduler.decision_from_dict(
                        payload["decision"]
                    )
                    if (
                        decision.decision_id in used_decision_ids
                        or decision.attempt_id in used_attempt_ids
                        or decision.generation_id in used_generation_ids
                        or decision.reservation_id
                        in used_reservation_ids
                    ):
                        raise Scheduler.SchedulerError(
                            "scheduler identity was reused"
                        )
                    snapshot = self._scheduler_snapshot(
                        genesis=genesis,
                        lanes=lanes,
                        attempts=attempts,
                        budget=budget,
                        journal_head_sequence=event["sequence"] - 1,
                        journal_head_digest=event["previous_digest"],
                        clean_proposer_settlements=tuple(
                            settlement
                            for lane in lanes.values()
                            for settlement in lane[
                                "clean_proposer_settlements"
                            ]
                        ),
                        complexity_rounds=tuple(complexity_rounds),
                        auxiliary_assignments=tuple(
                            item["state"]
                            for item in auxiliary_assignments.values()
                        ),
                        sidecar_requests=tuple(
                            item["request"]
                            for item in sidecar_requests.values()
                            if not item["invalidated"]
                        ),
                    )
                    Scheduler.verify_decision(snapshot, decision)
                except Scheduler.SchedulerError as exc:
                    raise ContiguousRunnerError(
                        "scheduler decision is stale or forged"
                    ) from exc
                used_decision_ids.add(decision.decision_id)
                used_attempt_ids.add(decision.attempt_id)
                used_generation_ids.add(decision.generation_id)
                used_reservation_ids.add(decision.reservation_id)
                pending_decision = decision
                continue
            if kind == "AUXILIARY_DECISION":
                if (
                    set(payload) != {"decision"}
                    or pending_auxiliary_decision is not None
                    or pending_decision is not None
                ):
                    raise ContiguousRunnerError(
                        "auxiliary decision event schema/order mismatch"
                    )
                try:
                    decision = Scheduler.auxiliary_decision_from_dict(
                        payload["decision"]
                    )
                    identities = {
                        decision.decision_id,
                        decision.assignment_id,
                        decision.reservation_id,
                        decision.expert_id,
                    }
                    if (
                        identities
                        & (
                            used_decision_ids
                            | used_attempt_ids
                            | used_generation_ids
                            | used_reservation_ids
                            | used_expert_ids
                        )
                        or decision.thread_id in used_thread_ids
                    ):
                        raise Scheduler.SchedulerError(
                            "auxiliary identity was reused"
                        )
                    snapshot = self._scheduler_snapshot(
                        genesis=genesis,
                        lanes=lanes,
                        attempts=attempts,
                        budget=budget,
                        journal_head_sequence=event["sequence"] - 1,
                        journal_head_digest=event["previous_digest"],
                        clean_proposer_settlements=tuple(
                            settlement
                            for lane in lanes.values()
                            for settlement in lane[
                                "clean_proposer_settlements"
                            ]
                        ),
                        complexity_rounds=tuple(complexity_rounds),
                        auxiliary_assignments=tuple(
                            item["state"]
                            for item in auxiliary_assignments.values()
                        ),
                        sidecar_requests=tuple(
                            item["request"]
                            for item in sidecar_requests.values()
                            if not item["invalidated"]
                        ),
                    )
                    Scheduler.verify_auxiliary_decision(
                        snapshot,
                        decision,
                        launch_configuration=auxiliary_configuration,
                    )
                except Scheduler.SchedulerError as exc:
                    raise ContiguousRunnerError(
                        "auxiliary decision is stale or forged"
                    ) from exc
                used_decision_ids.add(decision.decision_id)
                used_attempt_ids.add(decision.assignment_id)
                used_reservation_ids.add(decision.reservation_id)
                used_expert_ids.add(decision.expert_id)
                used_thread_ids.add(decision.thread_id)
                pending_auxiliary_decision = decision
                continue
            if kind == "AUXILIARY_RESERVED":
                decision = pending_auxiliary_decision
                if (
                    decision is None
                    or set(payload) != {
                        "assignment_id",
                        "reservation",
                    }
                    or payload["assignment_id"]
                    != decision.assignment_id
                    or payload["reservation"]
                    != Scheduler.auxiliary_reservation_projection(decision)
                ):
                    raise ContiguousRunnerError(
                        "auxiliary reservation event schema mismatch"
                    )
                try:
                    budget = Scheduler.reserve_budget(
                        budget,
                        reservation_id=decision.reservation_id,
                        attempt_id=decision.assignment_id,
                        units=decision.reservation_units,
                    )
                    assignment_state = Scheduler.AuxiliaryAssignmentState(
                        schema=1,
                        assignment_id=decision.assignment_id,
                        decision_id=decision.decision_id,
                        reservation_id=decision.reservation_id,
                        game=decision.game,
                        frontier_sha256=decision.frontier_sha256,
                        parent_checkpoint_sha256=(
                            decision.parent_checkpoint_sha256
                        ),
                        trigger_no_progress=decision.no_progress,
                        trigger_history_sha256=(
                            decision.trigger_history_sha256
                        ),
                        profile_id=decision.profile_id,
                        round_index=decision.round_index,
                        specialization=decision.specialization,
                        expert_id=decision.expert_id,
                        thread_id=decision.thread_id,
                        active_proposer_attempt_id=(
                            decision.active_proposer_attempt_id
                        ),
                        input_manifest=decision.input_manifest,
                        input_manifest_sha256=(
                            decision.input_manifest_sha256
                        ),
                        observation_ledger_sha256=(
                            decision.observation_ledger_sha256
                        ),
                        model=decision.model,
                        reasoning_effort=decision.reasoning_effort,
                        role=decision.role,
                        context_limit_tokens=(
                            decision.context_limit_tokens
                        ),
                        role_max_concurrency=(
                            decision.role_max_concurrency
                        ),
                        supervisory_launch_configuration_sha256=(
                            decision
                            .supervisory_launch_configuration_sha256
                        ),
                        sidecar_request=decision.sidecar_request,
                        sidecar_request_sha256=(
                            decision.sidecar_request_sha256
                        ),
                        phase="RESERVED",
                    )
                    Scheduler.validate_auxiliary_assignment(
                        assignment_state
                    )
                except Scheduler.SchedulerError as exc:
                    raise ContiguousRunnerError(
                        "auxiliary reservation violates scheduler state"
                    ) from exc
                auxiliary_assignments[decision.assignment_id] = {
                    "state": assignment_state,
                    "decision": decision,
                    "prepared": None,
                    "launched": None,
                    "terminal": None,
                    "collection": None,
                    "teardown": None,
                    "admission": None,
                    "abort_reason": None,
                }
                if sum(
                    item["active"] is not None
                    for item in lanes.values()
                ) + sum(
                    item["state"].phase
                    in Scheduler.AUXILIARY_ACTIVE_PHASES
                    for item in auxiliary_assignments.values()
                ) > genesis["max_lanes"]:
                    raise ContiguousRunnerError(
                        "auxiliary reservation exceeds lane capacity"
                    )
                pending_auxiliary_decision = None
                continue
            if kind == "AUXILIARY_INPUT_PREPARED":
                assignment = auxiliary_assignments.get(
                    str(payload.get("assignment_id"))
                )
                if (
                    assignment is None
                    or assignment["state"].phase != "RESERVED"
                    or set(payload) != {
                        "assignment_id",
                        "input_manifest_path",
                        "input_manifest_sha256",
                        "input_bundle_receipt_path",
                        "input_bundle_receipt_sha256",
                    }
                ):
                    raise ContiguousRunnerError(
                        "auxiliary input preparation transition is invalid"
                    )
                prepared = AuxiliaryPreparedInput(
                    input_manifest_path=str(
                        payload["input_manifest_path"]
                    ),
                    input_manifest_sha256=str(
                        payload["input_manifest_sha256"]
                    ),
                    input_bundle_receipt_path=str(
                        payload["input_bundle_receipt_path"]
                    ),
                    input_bundle_receipt_sha256=str(
                        payload["input_bundle_receipt_sha256"]
                    ),
                )
                assignment["prepared"] = prepared
                assignment["state"] = replace(
                    assignment["state"], phase="INPUT_PREPARED"
                )
                continue
            if kind == "AUXILIARY_LAUNCHED":
                assignment = auxiliary_assignments.get(
                    str(payload.get("assignment_id"))
                )
                if (
                    assignment is None
                    or assignment["state"].phase != "INPUT_PREPARED"
                    or set(payload) != {
                        "assignment_id",
                        "launch_receipt_path",
                        "launch_receipt_sha256",
                    }
                ):
                    raise ContiguousRunnerError(
                        "auxiliary launch transition is invalid"
                    )
                launched = AuxiliaryLaunch(
                    launch_receipt_path=str(
                        payload["launch_receipt_path"]
                    ),
                    launch_receipt_sha256=str(
                        payload["launch_receipt_sha256"]
                    ),
                )
                assignment["launched"] = launched
                assignment["state"] = replace(
                    assignment["state"], phase="RUNNING"
                )
                continue
            if kind == "AUXILIARY_ABORTED":
                assignment = auxiliary_assignments.get(
                    str(payload.get("assignment_id"))
                )
                if (
                    assignment is None
                    or assignment["state"].phase
                    not in Scheduler.AUXILIARY_ACTIVE_PHASES
                ):
                    raise ContiguousRunnerError(
                        "auxiliary abort transition is invalid"
                    )
                try:
                    charged_units = Scheduler.charge_to_units(
                        payload.get("cost_used")
                    )
                    if (
                        payload.get("authenticated_cost_units")
                        != charged_units
                        or payload.get("budget_reservation_id")
                        != assignment["state"].reservation_id
                        or payload.get("auxiliary_decision_id")
                        != assignment["state"].decision_id
                    ):
                        raise Scheduler.SchedulerError(
                            "auxiliary abort settlement mismatch"
                        )
                    budget = Scheduler.settle_budget(
                        budget,
                        reservation_id=(
                            assignment["state"].reservation_id
                        ),
                        attempt_id=(
                            assignment["state"].assignment_id
                        ),
                        charged_units=charged_units,
                    )
                except Scheduler.SchedulerError as exc:
                    raise ContiguousRunnerError(
                        "auxiliary abort settlement is invalid"
                    ) from exc
                assignment["abort_reason"] = payload.get("reason")
                assignment["teardown"] = (
                    None
                    if payload.get("teardown_receipt_path") is None
                    else AuxiliaryTeardown(
                        teardown_receipt_path=str(
                            payload["teardown_receipt_path"]
                        ),
                        teardown_receipt_sha256=str(
                            payload["teardown_receipt_sha256"]
                        ),
                    )
                )
                assignment["state"] = replace(
                    assignment["state"], phase="ABORTED"
                )
                continue
            if kind == "AUXILIARY_RESULT_QUARANTINED":
                assignment = auxiliary_assignments.get(
                    str(payload.get("assignment_id"))
                )
                if (
                    assignment is None
                    or assignment["state"].phase != "RUNNING"
                ):
                    raise ContiguousRunnerError(
                        "auxiliary quarantine transition is invalid"
                    )
                try:
                    output = Scheduler.auxiliary_output_from_dict(
                        payload.get("output"),
                        assignment=assignment["state"],
                    )
                    charged_units = Scheduler.charge_to_units(
                        payload.get("cost_used")
                    )
                    if (
                        payload.get("authenticated_cost_units")
                        != charged_units
                        or payload.get("budget_reservation_id")
                        != assignment["state"].reservation_id
                        or payload.get("auxiliary_decision_id")
                        != assignment["state"].decision_id
                    ):
                        raise Scheduler.SchedulerError(
                            "auxiliary result settlement mismatch"
                        )
                    budget = Scheduler.settle_budget(
                        budget,
                        reservation_id=(
                            assignment["state"].reservation_id
                        ),
                        attempt_id=(
                            assignment["state"].assignment_id
                        ),
                        charged_units=charged_units,
                    )
                except Scheduler.SchedulerError as exc:
                    raise ContiguousRunnerError(
                        "auxiliary quarantined result is invalid"
                    ) from exc
                assignment["collection"] = AuxiliaryCollection(
                    output=output,
                    cost_used=float(payload["cost_used"]),
                )
                assignment["teardown"] = AuxiliaryTeardown(
                    teardown_receipt_path=str(
                        payload["teardown_receipt_path"]
                    ),
                    teardown_receipt_sha256=str(
                        payload["teardown_receipt_sha256"]
                    ),
                )
                assignment["state"] = replace(
                    assignment["state"],
                    phase="QUARANTINED",
                    output=output,
                )
                continue
            if kind == "AUXILIARY_PROFILE_ADMITTED":
                assignment = auxiliary_assignments.get(
                    str(payload.get("assignment_id"))
                )
                if (
                    assignment is None
                    or assignment["state"].phase != "QUARANTINED"
                ):
                    raise ContiguousRunnerError(
                        "auxiliary profile admission transition is invalid"
                    )
                try:
                    profile = Scheduler.complexity_profile_from_dict(
                        payload.get("profile")
                    )
                    lane = lanes[assignment["state"].game]
                    round_state = Scheduler.ComplexityRoundState(
                        schema=1,
                        game=assignment["state"].game,
                        frontier_sha256=(
                            assignment["state"].frontier_sha256
                        ),
                        parent_checkpoint_sha256=(
                            assignment["state"]
                            .parent_checkpoint_sha256
                        ),
                        parent_source_tree_sha256=(
                            lane["source_tree_sha256"]
                        ),
                        round_index=assignment["state"].round_index,
                        profile=profile,
                        diagnosis_assignment_id=(
                            assignment["state"].assignment_id
                        ),
                        trigger_no_progress=(
                            assignment["state"].trigger_no_progress
                        ),
                        trigger_history_sha256=(
                            assignment["state"].trigger_history_sha256
                        ),
                        input_manifest_sha256=(
                            assignment["state"].input_manifest_sha256
                        ),
                        observation_ledger_sha256=(
                            assignment["state"]
                            .observation_ledger_sha256
                        ),
                        admission_receipt_path=str(
                            payload["admission_receipt_path"]
                        ),
                        admission_receipt_sha256=str(
                            payload["admission_receipt_sha256"]
                        ),
                        admitted_sequence=int(event["sequence"]),
                        admitted_event_digest=str(event["digest"]),
                    )
                    Scheduler.validate_complexity_round(round_state)
                except Scheduler.SchedulerError as exc:
                    raise ContiguousRunnerError(
                        "admitted complexity profile is invalid"
                    ) from exc
                complexity_rounds.append(round_state)
                assignment["admission"] = dict(payload)
                assignment["state"] = replace(
                    assignment["state"], phase="ADMITTED"
                )
                continue
            if kind in {
                "AUXILIARY_OUTPUT_ADMITTED",
                "AUXILIARY_OUTPUT_REJECTED",
            }:
                assignment = auxiliary_assignments.get(
                    str(payload.get("assignment_id"))
                )
                if (
                    assignment is None
                    or assignment["state"].phase != "QUARANTINED"
                ):
                    raise ContiguousRunnerError(
                        "auxiliary output disposition is invalid"
                    )
                assignment["admission"] = dict(payload)
                assignment["state"] = replace(
                    assignment["state"],
                    phase=(
                        "ADMITTED"
                        if kind == "AUXILIARY_OUTPUT_ADMITTED"
                        else "REJECTED"
                    ),
                    admission_receipt_path=(
                        str(payload["admission_receipt_path"])
                        if kind == "AUXILIARY_OUTPUT_ADMITTED"
                        else None
                    ),
                    admission_receipt_sha256=(
                        str(payload["admission_receipt_sha256"])
                        if kind == "AUXILIARY_OUTPUT_ADMITTED"
                        else None
                    ),
                    admitted_sequence=(
                        int(event["sequence"])
                        if kind == "AUXILIARY_OUTPUT_ADMITTED"
                        else None
                    ),
                    admitted_event_digest=(
                        str(event["digest"])
                        if kind == "AUXILIARY_OUTPUT_ADMITTED"
                        else None
                    ),
                )
                continue
            if kind == "NATIVE_SIDECAR_REQUEST_ADMITTED":
                if set(payload) != {
                    "attempt_id", "draft", "request"
                }:
                    raise ContiguousRunnerError(
                        "native sidecar request event schema mismatch"
                    )
                native_attempt_id = str(payload["attempt_id"])
                attempt = attempts.get(native_attempt_id)
                try:
                    draft = (
                        Scheduler
                        .native_sidecar_request_draft_from_dict(
                            payload["draft"]
                        )
                    )
                    request = Scheduler.sidecar_request_from_dict(
                        payload["request"]
                    )
                except Scheduler.SchedulerError as exc:
                    raise ContiguousRunnerError(
                        "native sidecar request event is malformed"
                    ) from exc
                if (
                    attempt is None
                    or attempt["phase"] != "CLOSED"
                    or not isinstance(
                        attempt["settled_result"], AttemptResult
                    )
                    or attempt["settled_result"]
                    .native_sidecar_request_draft
                    != draft
                ):
                    raise ContiguousRunnerError(
                        "native sidecar request lacks its closed result"
                    )
                lane = lanes[draft.game]
                settlement = next(
                    (
                        item
                        for item in lane[
                            "clean_proposer_settlements"
                        ]
                        if item.attempt_id == native_attempt_id
                    ),
                    None,
                )
                try:
                    expected_request = (
                        None
                        if settlement is None
                        else Scheduler
                        .native_sidecar_request_from_draft(
                            draft, settlement=settlement
                        )
                    )
                except Scheduler.SchedulerError as exc:
                    raise ContiguousRunnerError(
                        "native sidecar request origin is invalid"
                    ) from exc
                if (
                    expected_request is None
                    or request != expected_request
                    or lane["checkpoint_sha256"]
                    != request.parent_checkpoint_sha256
                    or frontier_sha256(
                        request.game,
                        lane["reached"],
                        lane["checkpoint_sha256"],
                    )
                    != request.frontier_sha256
                    or not set(
                        request
                        .cited_public_observation_receipt_sha256s
                    ).issubset(
                        lane[
                            "public_observation_receipt_sha256s"
                        ]
                    )
                    or any(
                        row["request"].request_id
                        == request.request_id
                        or row["request"].request_sha256
                        == request.request_sha256
                        or (
                            row["request"].authority
                            == "native_proposer"
                            and row["request"].native_attempt_id
                            == request.native_attempt_id
                        )
                        for row in sidecar_requests.values()
                    )
                ):
                    raise ContiguousRunnerError(
                        "native sidecar request is stale, forged, or reused"
                    )
                sidecar_requests[request.request_sha256] = {
                    "request": request,
                    "origin_kind": kind,
                    "origin_id": native_attempt_id,
                    "admitted_sequence": int(event["sequence"]),
                    "admitted_event_digest": str(event["digest"]),
                    "invalidated": False,
                }
                continue
            if kind == "SUPERVISORY_SIDECAR_REQUEST_ADMITTED":
                if set(payload) != {"assignment_id", "request"}:
                    raise ContiguousRunnerError(
                        "supervisory sidecar request event schema mismatch"
                    )
                assignment_id = str(payload["assignment_id"])
                assignment = auxiliary_assignments.get(assignment_id)
                try:
                    request = Scheduler.sidecar_request_from_dict(
                        payload["request"]
                    )
                    expected_request = (
                        None
                        if assignment is None
                        else Scheduler
                        .supervisory_sidecar_request_from_assignment(
                            assignment["state"]
                        )
                    )
                except Scheduler.SchedulerError as exc:
                    raise ContiguousRunnerError(
                        "supervisory sidecar request origin is invalid"
                    ) from exc
                if (
                    expected_request is None
                    or request != expected_request
                    or any(
                        row["request"].request_id
                        == request.request_id
                        or row["request"].request_sha256
                        == request.request_sha256
                        or (
                            row["request"].authority
                            == "admitted_supervisory_proposer"
                            and row["request"]
                            .supervisory_assignment_id
                            == request.supervisory_assignment_id
                        )
                        for row in sidecar_requests.values()
                    )
                ):
                    raise ContiguousRunnerError(
                        "supervisory sidecar request is forged or reused"
                    )
                sidecar_requests[request.request_sha256] = {
                    "request": request,
                    "origin_kind": kind,
                    "origin_id": assignment_id,
                    "admitted_sequence": int(event["sequence"]),
                    "admitted_event_digest": str(event["digest"]),
                    "invalidated": False,
                }
                continue
            attempt_id = payload.get("attempt_id")
            if not _safe_identifier(attempt_id):
                raise ContiguousRunnerError(
                    f"{kind} has invalid attempt_id"
                )
            if kind == "ATTEMPT_RESERVED":
                if attempt_id in attempts:
                    raise ContiguousRunnerError(
                        f"duplicate attempt reservation: {attempt_id}"
                    )
                if (
                    set(payload)
                    != {"attempt_id", "reservation", "scheduler"}
                    or pending_decision is None
                    or payload["scheduler"]
                    != Scheduler.reservation_binding(pending_decision)
                    or attempt_id != pending_decision.attempt_id
                ):
                    raise ContiguousRunnerError(
                        "attempt reservation event schema mismatch"
                    )
                reservation = _reservation_from_dict(
                    payload.get("reservation")
                )
                game = reservation.game
                if game not in lanes:
                    raise ContiguousRunnerError(
                        f"attempt targets unknown game: {game}"
                    )
                lane = lanes[game]
                if lane["active"] is not None or lane["blocked"] is not None:
                    raise ContiguousRunnerError(
                        f"attempt overlaps active/blocked game: {game}"
                    )
                if (
                    reservation.campaign_id != genesis["campaign_id"]
                    or reservation.campaign_id
                    != pending_decision.campaign_id
                    or reservation.attempt_id != attempt_id
                    or reservation.attempt_id
                    != pending_decision.attempt_id
                    or reservation.generation_id
                    != pending_decision.generation_id
                    or reservation.game
                    != pending_decision.choice.game
                    or reservation.target_level != lane["reached"] + 1
                    or reservation.authoritative_target != lane["target"]
                    or reservation.parent_checkpoint_path
                    != lane["checkpoint_path"]
                    or reservation.parent_checkpoint_sha256
                    != lane["checkpoint_sha256"]
                    or reservation.parent_source_path
                    != lane["source_path"]
                    or reservation.parent_source_tree_sha256
                    != lane["source_tree_sha256"]
                    or reservation.frontier_sha256
                    != frontier_sha256(
                        game, lane["reached"], lane["checkpoint_sha256"]
                    )
                    or _backend_configuration_to_dict(
                        BackendConfiguration(
                            image_reference=(
                                reservation.image_reference
                            ),
                            image_digest=reservation.image_digest,
                            worker_command=reservation.worker_command,
                            resource_limits=(
                                reservation.resource_limits
                            ),
                            proposer_transport=(
                                reservation.proposer_transport
                            ),
                        )
                    )
                    != genesis["backend_configuration"]
                ):
                    raise ContiguousRunnerError(
                        f"attempt does not match current frontier: {attempt_id}"
                    )
                expected_remaining = (
                    None
                    if pending_decision.choice.reservation_units is None
                    else (
                        pending_decision.choice.reservation_units
                        / Scheduler.COST_SCALE
                    )
                )
                if reservation.cost_limit_remaining != expected_remaining:
                    raise ContiguousRunnerError(
                        "attempt has stale or wrong cost-limit admission"
                    )
                self._validate_selected_wip(
                    reservation, lane, attempts=attempts
                )
                selected_wip = pending_decision.choice.selected_wip
                if (
                    reservation.game != pending_decision.choice.game
                    or reservation.target_level
                    != pending_decision.choice.target_level
                    or reservation.authoritative_target
                    != pending_decision.choice.authoritative_target
                    or reservation.effort
                    != pending_decision.choice.effort
                    or reservation.soft_allocation_seconds
                    != pending_decision.choice.soft_allocation_seconds
                    or reservation.wip_mode
                    != pending_decision.choice.effective_wip_mode
                    or reservation.thread_mode
                    != pending_decision.choice.thread_mode
                    or self._scheduler_wip(reservation.wip)
                    != selected_wip
                    or reservation.supervisory_handoff
                    != pending_decision.choice
                    .selected_supervisory_handoff
                    or reservation.resume_thread_id
                    != (
                        selected_wip.codex_thread_id
                        if selected_wip is not None
                        else None
                    )
                    or reservation.resume_thread_binding_sha256
                    != (
                        selected_wip.final_thread_binding_sha256
                        if selected_wip is not None
                        else None
                    )
                ):
                    raise ContiguousRunnerError(
                        "attempt differs from its scheduler decision: "
                        f"{attempt_id}"
                    )
                try:
                    budget = Scheduler.reserve_budget(
                        budget,
                        reservation_id=pending_decision.reservation_id,
                        attempt_id=attempt_id,
                        units=pending_decision.choice.reservation_units,
                    )
                except Scheduler.SchedulerError as exc:
                    raise ContiguousRunnerError(
                        "attempt budget reservation is invalid"
                    ) from exc
                attempts[attempt_id] = {
                    "reservation": reservation,
                    "spec": None,
                    "phase": "RESERVED",
                    "prepared": None,
                    "launched": None,
                    "launched_at": None,
                    "terminal": None,
                    "collection": None,
                    "public_observation_transition": None,
                    "protocol_invalid": None,
                    "substrate_failure": None,
                    "outcome": None,
                    "settled_result": None,
                    "teardown": None,
                    "observation_count": 0,
                    "last_observation_at": None,
                    "last_observation_sha256": None,
                    "retry_count": 0,
                    "operation_retry_counts": {},
                    "operation_retry_not_before": {},
                    "terminal_failure_circuit_recorded": False,
                    "candidate": None,
                    "scheduler_decision_id":
                        pending_decision.decision_id,
                    "scheduler_decision_sha256":
                        pending_decision.decision_sha256,
                    "scheduler_decision": pending_decision,
                    "budget_reservation_id":
                        pending_decision.reservation_id,
                    "budget_reservation_units":
                        pending_decision.choice.reservation_units,
                }
                lane["active"] = attempt_id
                lane["last_dispatch_sequence"] = event["sequence"]
                pending_decision = None
                if sum(
                    item["active"] is not None for item in lanes.values()
                ) + sum(
                    item["state"].phase
                    in Scheduler.AUXILIARY_ACTIVE_PHASES
                    for item in auxiliary_assignments.values()
                ) > genesis["max_lanes"]:
                    raise ContiguousRunnerError(
                        "journal exceeds configured lane capacity"
                    )
            elif kind == "ATTEMPT_PREPARED":
                attempt = self._attempt(
                    attempts, attempt_id, "RESERVED"
                )
                if set(payload) != {"attempt_id", "spec"}:
                    raise ContiguousRunnerError(
                        "attempt preparation event schema mismatch"
                    )
                spec = _spec_from_dict(payload.get("spec"))
                comparable = _spec_to_dict(spec)
                for key in (
                    "input_tree_sha256",
                    "initial_workspace_tree_sha256",
                    "initial_app_server_state_tree_sha256",
                    "hard_safety_seconds",
                    "max_auth_refreshes",
                    "input_bundle_receipt_path",
                    "input_bundle_receipt_sha256",
                    "frontier_brief_path",
                    "frontier_brief_sha256",
                    # These are materialization products, not scheduler
                    # reservation fields.  The authoritative handoff binding
                    # itself remains in ``comparable`` and must still equal
                    # the durable reservation; _spec_from_dict plus the
                    # input-bundle audit bind these paths and hashes to that
                    # exact reserved handoff.
                    "supervisory_handoff_path",
                    "supervisory_handoff_sha256",
                    "supervisory_handoff_binding_receipt_path",
                    "supervisory_handoff_binding_receipt_sha256",
                    "bridge_policy_path",
                    "bridge_policy_sha256",
                    "parent_action_count",
                    "remaining_action_budget",
                    "fresh_prefix_required",
                ):
                    comparable.pop(key)
                if comparable != _reservation_to_dict(
                    attempt["reservation"]
                ):
                    raise ContiguousRunnerError(
                        "prepared spec differs from durable reservation"
                    )
                attempt.update(spec=spec, phase="PREPARED")
            elif kind == "ATTEMPT_RETRY":
                attempt = attempts.get(attempt_id)
                if (
                    attempt is None
                    or set(payload)
                    != {
                        "attempt_id",
                        "retry_index",
                        "operation",
                        "operation_retry_index",
                        "error_type",
                        "backoff_seconds",
                        "retry_not_before",
                    }
                    or payload["retry_index"]
                    != attempt["retry_count"] + 1
                    or not _safe_identifier(payload["operation"])
                    or not _safe_identifier(payload["error_type"])
                    or payload["operation_retry_index"]
                    != attempt["operation_retry_counts"].get(
                        payload["operation"], 0
                    ) + 1
                    or not _is_finite_number(
                        payload["backoff_seconds"]
                    )
                    or payload["backoff_seconds"] < 0
                    or not _is_finite_number(
                        payload["retry_not_before"]
                    )
                ):
                    raise ContiguousRunnerError(
                        "attempt retry event schema/order mismatch"
                    )
                allowed_phases = {
                    "input_materialize": {"RESERVED"},
                    "backend_prepare": {"PREPARED"},
                    "backend_launch": {"BACKEND_PREPARED"},
                    "backend_poll": {"RUNNING", "DRAINING"},
                    "backend_collect": {"EXITED"},
                    "backend_teardown": {
                        "COLLECTED", "COLLECTION_REJECTED"
                    },
                    "promotion_commit": {"PROMOTING"},
                    "promotion_recover": {"PROMOTING"},
                }
                if attempt["phase"] not in allowed_phases.get(
                    payload["operation"], set()
                ):
                    raise ContiguousRunnerError(
                        "attempt retry operation does not match phase"
                    )
                attempt["retry_count"] += 1
                attempt["operation_retry_counts"][
                    payload["operation"]
                ] = payload["operation_retry_index"]
                attempt["operation_retry_not_before"][
                    payload["operation"]
                ] = payload["retry_not_before"]
            elif kind == "BACKEND_PREPARED":
                attempt = self._attempt(attempts, attempt_id, "PREPARED")
                if set(payload) != {"attempt_id", "prepared"}:
                    raise ContiguousRunnerError(
                        "backend preparation event schema mismatch"
                    )
                prepared = _backend_preparation_from_dict(
                    payload["prepared"]
                )
                if (
                    prepared.observed_image_digest
                    != attempt["spec"].image_digest
                ):
                    raise ContiguousRunnerError(
                        "backend observed the wrong image digest"
                    )
                _validate_preparation_receipts(
                    attempt["spec"], prepared
                )
                attempt.update(
                    phase="BACKEND_PREPARED", prepared=prepared
                )
            elif kind == "ATTEMPT_SUBSTRATE_INFRASTRUCTURE":
                attempt = self._attempt(
                    attempts, attempt_id, "BACKEND_PREPARED"
                )
                required = {
                    "attempt_id",
                    "substrate_identity_sha256",
                    "failure_receipt_path",
                    "failure_receipt_sha256",
                    "result",
                    "authenticated_cost_units",
                    "budget_reservation_id",
                    "scheduler_decision_id",
                }
                if (
                    set(payload) != required
                    or substrate_incident is not None
                    or not _is_sha256(
                        payload["substrate_identity_sha256"]
                    )
                    or payload["authenticated_cost_units"] != 0
                    or payload["budget_reservation_id"]
                    != attempt["budget_reservation_id"]
                    or payload["scheduler_decision_id"]
                    != attempt["scheduler_decision_id"]
                    or not isinstance(payload["result"], dict)
                ):
                    raise ContiguousRunnerError(
                        "substrate infrastructure event schema/order "
                        "mismatch"
                    )
                result = self._result_from_payload({
                    "attempt_id": attempt_id,
                    **payload["result"],
                })
                if (
                    result
                    != AttemptResult(
                        kind="infrastructure",
                        cost_used=0.0,
                        reason="codex_substrate_preflight_failed",
                    )
                ):
                    raise ContiguousRunnerError(
                        "substrate failure gained solver/result authority"
                    )
                host_root = Path(
                    attempt["spec"].host_transcript_path
                ).parent
                failure = _validate_bound_receipt(
                    payload["failure_receipt_path"],
                    payload["failure_receipt_sha256"],
                    expected_path=host_root
                    / "substrate_preflight_failure_receipt.json",
                    expected_kind=(
                        "contiguous_substrate_preflight_failure"
                    ),
                    spec=attempt["spec"],
                )
                expected_failure_keys = {
                    "schema",
                    "kind",
                    "campaign_id",
                    "generation_id",
                    "attempt_id",
                    "attempt_spec_sha256",
                    "substrate_identity_sha256",
                    "substrate_preflight_intent_path",
                    "substrate_preflight_intent_sha256",
                    "preflight_root",
                    "state_root",
                    "failure_stage",
                    "error_type",
                    "failure_class",
                    "failure_code",
                    "partial_scan_receipt_path",
                    "partial_scan_receipt_sha256",
                    "purge_receipt_path",
                    "purge_receipt_sha256",
                    "post_failure_state_tree_sha256",
                    "state_root_empty",
                    "preflight_root_absent",
                    "prior_clean_wip_tree_sha256",
                    "post_purge_clean_wip_tree_sha256",
                    "backend_launch_failure_tombstone_path",
                    "backend_launch_failure_tombstone_sha256",
                    "proposer_container_started",
                    "bridge_connected",
                    "thread_started",
                    "turn_started",
                    "candidate_authority",
                    "wip_authority",
                    "promotion_authority",
                    "cost_used",
                    "status",
                }
                intent = _validate_bound_receipt(
                    failure.get(
                        "substrate_preflight_intent_path"
                    ),
                    failure.get(
                        "substrate_preflight_intent_sha256"
                    ),
                    expected_path=host_root
                    / "substrate_preflight_intent.json",
                    expected_kind=(
                        "contiguous_substrate_preflight_intent"
                    ),
                    spec=attempt["spec"],
                )
                partial_scan = _validate_bound_receipt(
                    failure.get("partial_scan_receipt_path"),
                    failure.get("partial_scan_receipt_sha256"),
                    expected_path=host_root
                    / "substrate_preflight_partial_scan_receipt.json",
                    expected_kind=(
                        "contiguous_substrate_preflight_partial_scan"
                    ),
                    spec=attempt["spec"],
                )
                purge = _validate_bound_receipt(
                    failure.get("purge_receipt_path"),
                    failure.get("purge_receipt_sha256"),
                    expected_path=host_root
                    / "substrate_preflight_purge_receipt.json",
                    expected_kind=(
                        "contiguous_substrate_preflight_purge"
                    ),
                    spec=attempt["spec"],
                )
                state_inventory = (
                    Transport.inventory_controller_state(
                        Path(attempt["spec"].app_server_state_dir),
                        sentinels=self._secret_sentinels,
                    )
                )
                selected_wip = attempt["spec"].wip
                current_wip_tree_sha256 = (
                    None
                    if selected_wip is None
                    else Contract._tree_hash(
                        Path(selected_wip.wip_root_path)
                    )
                )
                if (
                    set(failure) != expected_failure_keys
                    or failure["substrate_identity_sha256"]
                    != payload["substrate_identity_sha256"]
                    or failure["preflight_root"]
                    != str(host_root / "substrate_preflight")
                    or failure["state_root"]
                    != attempt["spec"].app_server_state_dir
                    or failure[
                        "substrate_preflight_intent_sha256"
                    ]
                    != _sha256_file(
                        host_root
                        / "substrate_preflight_intent.json"
                    )
                    or intent.get("substrate_identity_sha256")
                    != payload["substrate_identity_sha256"]
                    or intent.get("preflight_root")
                    != str(host_root / "substrate_preflight")
                    or intent.get("state_root")
                    != attempt["spec"].app_server_state_dir
                    or intent.get("proposer_container_started")
                    is not False
                    or intent.get("bridge_connected") is not False
                    or intent.get("thread_started") is not False
                    or intent.get("turn_started") is not False
                    or intent.get("status") != "PENDING"
                    or not _safe_identifier(
                        failure["failure_stage"]
                    )
                    or not _safe_identifier(failure["error_type"])
                    or failure["failure_class"] not in {
                        "DETERMINISTIC_CONFIGURATION",
                        "TRANSIENT_INFRASTRUCTURE",
                    }
                    or not _safe_identifier(failure["failure_code"])
                    or partial_scan.get(
                        "substrate_preflight_intent_sha256"
                    )
                    != failure[
                        "substrate_preflight_intent_sha256"
                    ]
                    or partial_scan.get(
                        "substrate_identity_sha256"
                    )
                    != payload["substrate_identity_sha256"]
                    or partial_scan.get("failure_stage")
                    != failure["failure_stage"]
                    or partial_scan.get("error_type")
                    != failure["error_type"]
                    or partial_scan.get("failure_class")
                    != failure["failure_class"]
                    or partial_scan.get("failure_code")
                    != failure["failure_code"]
                    or partial_scan.get(
                        "scan_completed_before_purge"
                    )
                    is not True
                    or partial_scan.get("status") != "COMPLETE"
                    or purge.get(
                        "substrate_preflight_intent_sha256"
                    )
                    != failure[
                        "substrate_preflight_intent_sha256"
                    ]
                    or purge.get("substrate_identity_sha256")
                    != payload["substrate_identity_sha256"]
                    or purge.get("partial_scan_receipt_sha256")
                    != failure["partial_scan_receipt_sha256"]
                    or purge.get("state_root_empty") is not True
                    or purge.get("preflight_root_absent")
                    is not True
                    or purge.get("candidate_authority") is not False
                    or purge.get("wip_authority") is not False
                    or purge.get("promotion_authority") is not False
                    or purge.get("status") != "PASS"
                    or not _is_sha256(
                        failure[
                            "post_failure_state_tree_sha256"
                        ]
                    )
                    or failure["post_failure_state_tree_sha256"]
                    != purge.get(
                        "post_purge_state_tree_sha256"
                    )
                    or failure["post_failure_state_tree_sha256"]
                    != state_inventory.tree_sha256
                    or state_inventory.files
                    or state_inventory.secret_occurrences
                    or failure["state_root_empty"] is not True
                    or failure["preflight_root_absent"]
                    is not True
                    or (
                        host_root / "substrate_preflight"
                    ).exists()
                    or (
                        host_root / "substrate_preflight"
                    ).is_symlink()
                    or failure[
                        "prior_clean_wip_tree_sha256"
                    ]
                    != failure[
                        "post_purge_clean_wip_tree_sha256"
                    ]
                    or failure[
                        "prior_clean_wip_tree_sha256"
                    ]
                    != current_wip_tree_sha256
                    or purge.get(
                        "prior_clean_wip_tree_sha256"
                    )
                    != current_wip_tree_sha256
                    or purge.get(
                        "post_purge_clean_wip_tree_sha256"
                    )
                    != current_wip_tree_sha256
                    or failure[
                        "backend_launch_failure_tombstone_path"
                    ]
                    != str(host_root / "backend_launch_failure.json")
                    or _sha256_file(
                        Path(
                            failure[
                                "backend_launch_failure_tombstone_path"
                            ]
                        )
                    )
                    != failure[
                        "backend_launch_failure_tombstone_sha256"
                    ]
                    or any(
                        failure[name] is not False
                        for name in (
                            "proposer_container_started",
                            "bridge_connected",
                            "thread_started",
                            "turn_started",
                            "candidate_authority",
                            "wip_authority",
                            "promotion_authority",
                        )
                    )
                    or failure["cost_used"] != 0.0
                    or failure["status"] != "INFRASTRUCTURE"
                ):
                    raise ContiguousRunnerError(
                        "substrate failure receipt is not the exact "
                        "pre-turn non-authority boundary"
                    )
                try:
                    budget = Scheduler.settle_budget(
                        budget,
                        reservation_id=attempt[
                            "budget_reservation_id"
                        ],
                        attempt_id=attempt_id,
                        charged_units=0,
                    )
                except Scheduler.SchedulerError as exc:
                    raise ContiguousRunnerError(
                        "substrate infrastructure settlement is invalid"
                    ) from exc
                lane = lanes[attempt["spec"].game]
                if (
                    lane["active"] != attempt_id
                    or lane["no_progress"]
                    != attempt["scheduler_decision"].choice.no_progress
                ):
                    raise ContiguousRunnerError(
                        "substrate failure does not close its exact "
                        "reserved frontier"
                    )
                lane["active"] = None
                attempt.update(
                    phase="CLOSED",
                    settled_result=result,
                    substrate_failure={
                        "substrate_identity_sha256":
                            payload["substrate_identity_sha256"],
                        "failure_receipt_path":
                            payload["failure_receipt_path"],
                        "failure_receipt_sha256":
                            payload["failure_receipt_sha256"],
                    },
                )
                substrate_incident = {
                    "attempt_id": attempt_id,
                    "game": attempt["spec"].game,
                    "frontier_sha256":
                        attempt["spec"].frontier_sha256,
                    "substrate_identity_sha256":
                        payload["substrate_identity_sha256"],
                    "failure_receipt_path":
                        payload["failure_receipt_path"],
                    "failure_receipt_sha256":
                        payload["failure_receipt_sha256"],
                    "reason_code":
                        "codex_substrate_preflight_failed",
                    "failure_class": failure["failure_class"],
                    "failure_code": failure["failure_code"],
                    "incident_event_sequence":
                        event["sequence"],
                    "incident_event_digest": event["digest"],
                    "incident_identity_sha256":
                        substrate_incident_identity_sha256(
                            campaign_id=genesis["campaign_id"],
                            attempt_id=attempt_id,
                            game=attempt["spec"].game,
                            frontier_sha256=(
                                attempt["spec"].frontier_sha256
                            ),
                            substrate_identity_sha256=(
                                payload[
                                    "substrate_identity_sha256"
                                ]
                            ),
                            failure_receipt_sha256=(
                                payload[
                                    "failure_receipt_sha256"
                                ]
                            ),
                            failure_class=(
                                failure["failure_class"]
                            ),
                            failure_code=(
                                failure["failure_code"]
                            ),
                        ),
                    "health_probe_count": 0,
                    "pending_reprobe": None,
                    "attempted_remediation_epochs": [],
                    "last_health_probe": None,
                    "circuit_failure_recorded": False,
                    "meta_recovery_invocation_count": 0,
                    "meta_recovery": None,
                }
            elif kind == "META_SUBSTRATE_RECOVERY_AUTHORIZED":
                required = {
                    "authorization_id",
                    "attempt_id",
                    "substrate_identity_sha256",
                    "incident_failure_receipt_sha256",
                    "incident_event_sequence",
                    "incident_event_digest",
                    "incident_identity_sha256",
                    "meta_request_sha256",
                    "meta_response_sha256",
                    "meta_terminal_sha256",
                    "recommendation",
                    "operator_configuration_sha256",
                    "authorization_receipt_path",
                    "authorization_receipt_sha256",
                    "authorization_authentication_sha256",
                    "invocation_index",
                }
                if (
                    set(payload) != required
                    or substrate_incident is None
                    or operator_incident is None
                    or substrate_incident["pending_reprobe"]
                    is not None
                    or substrate_incident["meta_recovery"] is not None
                    or substrate_incident[
                        "meta_recovery_invocation_count"
                    ]
                    != 0
                    or payload["attempt_id"]
                    != substrate_incident["attempt_id"]
                    or payload["substrate_identity_sha256"]
                    != substrate_incident[
                        "substrate_identity_sha256"
                    ]
                    or payload["incident_failure_receipt_sha256"]
                    != substrate_incident[
                        "failure_receipt_sha256"
                    ]
                    or payload["incident_event_sequence"]
                    != substrate_incident[
                        "incident_event_sequence"
                    ]
                    or payload["incident_event_digest"]
                    != substrate_incident[
                        "incident_event_digest"
                    ]
                    or payload["incident_identity_sha256"]
                    != substrate_incident[
                        "incident_identity_sha256"
                    ]
                    or payload["recommendation"]
                    != META_SUBSTRATE_RECOVERY_RECOMMENDATION
                    or payload["operator_configuration_sha256"]
                    != genesis["operator_configuration_sha256"]
                    or payload["invocation_index"] != 1
                    or not _is_canonical_uuid(
                        payload["authorization_id"]
                    )
                    or any(
                        not _is_sha256(payload[name])
                        for name in (
                            "meta_request_sha256",
                            "meta_response_sha256",
                            "meta_terminal_sha256",
                            "incident_event_digest",
                            "incident_identity_sha256",
                            "authorization_receipt_sha256",
                            "authorization_authentication_sha256",
                        )
                    )
                ):
                    raise ContiguousRunnerError(
                        "meta substrate recovery authorization "
                        "schema/order mismatch"
                    )
                authorization_path = (
                    self.root
                    / META_SUBSTRATE_RECOVERY_AUTHORIZATION_ROOT
                    / f"{payload['authorization_id']}.json"
                )
                if (
                    payload["authorization_receipt_path"]
                    != str(authorization_path)
                    or _sha256_file(authorization_path)
                    != payload["authorization_receipt_sha256"]
                ):
                    raise ContiguousRunnerError(
                        "meta substrate authorization receipt is "
                        "substituted"
                    )
                authorization = _read_json_file(
                    authorization_path
                )
                authorization_keys = {
                    "schema", "kind", "campaign_id",
                    "authorization_id", "attempt_id",
                    "substrate_identity_sha256",
                    "incident_failure_receipt_sha256",
                    "incident_event_sequence",
                    "incident_event_digest",
                    "incident_identity_sha256",
                    "meta_request_sha256", "meta_response_sha256",
                    "meta_terminal_sha256", "recommendation",
                    "operator_configuration_sha256",
                    "invocation_index", "single_use",
                    "solver_authority", "wip_authority",
                    "cost_authority", "promotion_authority",
                    "authorization_authentication_sha256",
                }
                unsigned_authorization = dict(authorization)
                observed_authentication = (
                    unsigned_authorization.pop(
                        "authorization_authentication_sha256",
                        None,
                    )
                )
                if (
                    set(authorization) != authorization_keys
                    or authorization.get("schema") != 1
                    or authorization.get("kind")
                    != (
                        "contiguous_meta_substrate_"
                        "recovery_authorization"
                    )
                    or authorization.get("campaign_id")
                    != genesis["campaign_id"]
                    or any(
                        authorization.get(name) != payload[name]
                        for name in (
                            "authorization_id",
                            "attempt_id",
                            "substrate_identity_sha256",
                            "incident_failure_receipt_sha256",
                            "incident_event_sequence",
                            "incident_event_digest",
                            "incident_identity_sha256",
                            "meta_request_sha256",
                            "meta_response_sha256",
                            "meta_terminal_sha256",
                            "recommendation",
                            "operator_configuration_sha256",
                            "invocation_index",
                        )
                    )
                    or authorization.get("single_use") is not True
                    or any(
                        authorization.get(name) is not False
                        for name in (
                            "solver_authority",
                            "wip_authority",
                            "cost_authority",
                            "promotion_authority",
                        )
                    )
                    or observed_authentication
                    != payload[
                        "authorization_authentication_sha256"
                    ]
                    or not hmac.compare_digest(
                        str(observed_authentication),
                        meta_substrate_recovery_authentication_sha256(
                            unsigned_authorization,
                            operator_configuration_sha256=(
                                genesis[
                                    "operator_configuration_sha256"
                                ]
                            ),
                        ),
                    )
                ):
                    raise ContiguousRunnerError(
                        "meta substrate authorization receipt is "
                        "malformed"
                    )
                substrate_incident[
                    "meta_recovery_invocation_count"
                ] = 1
                substrate_incident["meta_recovery"] = {
                    **payload,
                    "probe_index":
                        substrate_incident["health_probe_count"] + 1,
                    "phase": "AUTHORIZED",
                    "result": None,
                }
                substrate_incident["pending_reprobe"] = {
                    "authorization_id":
                        payload["authorization_id"],
                    "attempt_id": payload["attempt_id"],
                    "substrate_identity_sha256":
                        payload["substrate_identity_sha256"],
                    "incident_failure_receipt_sha256":
                        payload[
                            "incident_failure_receipt_sha256"
                        ],
                    "probe_index":
                        substrate_incident["health_probe_count"] + 1,
                    "authorization_receipt_sha256":
                        payload["authorization_receipt_sha256"],
                    "meta_recovery": True,
                }
            elif kind == "SUBSTRATE_HEALTH_REPROBE_AUTHORIZED":
                required = {
                    "authorization_id",
                    "attempt_id",
                    "substrate_identity_sha256",
                    "incident_failure_receipt_sha256",
                    "probe_index",
                    "reason_code",
                    "authorization_mode",
                    "retry_not_before",
                    "authorization_receipt_path",
                    "authorization_receipt_sha256",
                }
                if (
                    set(payload) != required
                    or substrate_incident is None
                    or substrate_incident["pending_reprobe"]
                    is not None
                    or payload["attempt_id"]
                    != substrate_incident["attempt_id"]
                    or payload["substrate_identity_sha256"]
                    != substrate_incident[
                        "substrate_identity_sha256"
                    ]
                    or payload["incident_failure_receipt_sha256"]
                    != substrate_incident[
                        "failure_receipt_sha256"
                    ]
                    or not _is_canonical_uuid(
                        payload["authorization_id"]
                    )
                    or payload["probe_index"]
                    != substrate_incident["health_probe_count"] + 1
                    or not _safe_identifier(
                        payload["reason_code"]
                    )
                    or payload["authorization_mode"] not in {
                        "sealed_autonomous_circuit",
                        "trusted_operator_early_override",
                    }
                    or not _is_finite_number(
                        payload["retry_not_before"]
                    )
                    or not isinstance(
                        failure_operation_circuits.get(
                            "substrate_health_reprobe:"
                            "controller_substrate"
                        ),
                        dict,
                    )
                    or payload["retry_not_before"]
                    != failure_operation_circuits[
                        "substrate_health_reprobe:"
                        "controller_substrate"
                    ]["retry_not_before"]
                ):
                    raise ContiguousRunnerError(
                        "substrate reprobe authorization schema/order "
                        "mismatch"
                    )
                authorization_path = (
                    self.root
                    / SUBSTRATE_REPROBE_AUTHORIZATION_ROOT
                    / (
                        payload["authorization_id"]
                        + ".json"
                    )
                )
                if (
                    payload["authorization_receipt_path"]
                    != str(authorization_path)
                    or _sha256_file(authorization_path)
                    != payload["authorization_receipt_sha256"]
                ):
                    raise ContiguousRunnerError(
                        "substrate reprobe authorization receipt is "
                        "substituted"
                    )
                authorization = _read_json_file(
                    authorization_path
                )
                expected_authorization_keys = {
                    "schema",
                    "kind",
                    "campaign_id",
                    "authorization_id",
                    "attempt_id",
                    "substrate_identity_sha256",
                    "incident_failure_receipt_sha256",
                    "probe_index",
                    "reason_code",
                    "authorization_mode",
                    "operator_configuration_sha256",
                    "retry_not_before",
                    "issued_at",
                    "single_use",
                    "sealed_supervisor_authority",
                    "trusted_operator_authority",
                    "game_scheduler_authority",
                    "meta_scheduler_authority",
                    "authorization_binding_sha256",
                }
                if (
                    set(authorization)
                    != expected_authorization_keys
                    or authorization["schema"] != 2
                    or authorization["kind"]
                    != "contiguous_substrate_health_reprobe_authorization"
                    or authorization["campaign_id"]
                    != genesis["campaign_id"]
                    or any(
                        authorization[name] != payload[name]
                        for name in (
                            "authorization_id",
                            "attempt_id",
                            "substrate_identity_sha256",
                            "incident_failure_receipt_sha256",
                            "probe_index",
                            "reason_code",
                            "authorization_mode",
                            "retry_not_before",
                        )
                    )
                    or authorization[
                        "operator_configuration_sha256"
                    ]
                    != genesis["operator_configuration_sha256"]
                    or not _is_finite_number(
                        authorization["issued_at"]
                    )
                    or authorization["single_use"] is not True
                    or authorization["issued_at"]
                    != float(event["recorded_at"])
                    or authorization["game_scheduler_authority"]
                    is not False
                    or authorization["meta_scheduler_authority"]
                    is not False
                    or authorization["sealed_supervisor_authority"]
                    is not (
                        authorization["authorization_mode"]
                        == "sealed_autonomous_circuit"
                    )
                    or authorization["trusted_operator_authority"]
                    is not (
                        authorization["authorization_mode"]
                        == "trusted_operator_early_override"
                    )
                    or (
                        authorization["authorization_mode"]
                        == "sealed_autonomous_circuit"
                        and authorization["issued_at"]
                        < authorization["retry_not_before"]
                    )
                ):
                    raise ContiguousRunnerError(
                        "substrate reprobe authorization receipt is "
                        "malformed"
                    )
                unsigned_authorization = dict(authorization)
                observed_binding = unsigned_authorization.pop(
                    "authorization_binding_sha256"
                )
                expected_binding = hashlib.sha256(
                    _canonical_json(unsigned_authorization)
                ).hexdigest()
                if not hmac.compare_digest(
                    observed_binding, expected_binding
                ):
                    raise ContiguousRunnerError(
                        "substrate reprobe authorization binding changed"
                    )
                substrate_incident["pending_reprobe"] = {
                    **payload,
                    "issued_at": authorization["issued_at"],
                }
            elif kind in {
                "SUBSTRATE_HEALTH_REPROBE_FAILED",
                "SUBSTRATE_HEALTH_RESTORED",
                "META_SUBSTRATE_RECOVERY_FAILED",
                "META_SUBSTRATE_HEALTH_RESTORED",
            }:
                health_result_required = {
                    "authorization_id",
                    "attempt_id",
                    "substrate_identity_sha256",
                    "incident_failure_receipt_sha256",
                    "probe_index",
                    "remediation_epoch_sha256",
                    "healthy_substrate_identity_sha256",
                    "failure_class",
                    "failure_code",
                    "health_receipt_path",
                    "health_receipt_sha256",
                    "status",
                }
                meta_result_required = {
                    "incident_event_sequence",
                    "incident_event_digest",
                    "incident_identity_sha256",
                    "meta_request_sha256",
                    "meta_response_sha256",
                    "meta_terminal_sha256",
                    "recommendation",
                    "authorization_receipt_sha256",
                    "authorization_authentication_sha256",
                    "rematerialization_evidence_path",
                    "rematerialization_evidence_sha256",
                    "invocation_index",
                }
                is_meta_result = kind.startswith("META_")
                required = (
                    health_result_required | meta_result_required
                    if is_meta_result
                    else health_result_required
                )
                pending = (
                    None
                    if substrate_incident is None
                    else substrate_incident["pending_reprobe"]
                )
                expected_status = (
                    "FAILED"
                    if kind
                    in {
                        "SUBSTRATE_HEALTH_REPROBE_FAILED",
                        "META_SUBSTRATE_RECOVERY_FAILED",
                    }
                    else "PASS"
                )
                meta_recovery = (
                    None
                    if substrate_incident is None
                    else substrate_incident["meta_recovery"]
                )
                if (
                    set(payload) != required
                    or substrate_incident is None
                    or pending is None
                    or payload["status"] != expected_status
                    or any(
                        payload[name] != pending[name]
                        for name in (
                            "authorization_id",
                            "attempt_id",
                            "substrate_identity_sha256",
                            "incident_failure_receipt_sha256",
                            "probe_index",
                        )
                    )
                    or not _is_sha256(
                        payload["remediation_epoch_sha256"]
                    )
                    or payload["remediation_epoch_sha256"]
                    in substrate_incident[
                        "attempted_remediation_epochs"
                    ]
                    or (
                        expected_status == "PASS"
                        and (
                            not _is_sha256(
                                payload[
                                    "healthy_substrate_identity_sha256"
                                ]
                            )
                            or payload["failure_class"] is not None
                            or payload["failure_code"] is not None
                        )
                    )
                    or (
                        expected_status == "FAILED"
                        and (
                            payload[
                                "healthy_substrate_identity_sha256"
                            ]
                            is not None
                            or payload["failure_class"] not in {
                                "DETERMINISTIC_CONFIGURATION",
                                "TRANSIENT_INFRASTRUCTURE",
                            }
                            or not _safe_identifier(
                                payload["failure_code"]
                            )
                        )
                    )
                    or (
                        is_meta_result
                        and (
                            not isinstance(meta_recovery, dict)
                            or meta_recovery.get("phase")
                            != "AUTHORIZED"
                            or pending.get("meta_recovery") is not True
                            or payload["invocation_index"] != 1
                            or any(
                                payload[name]
                                != meta_recovery[name]
                                for name in (
                                    "meta_request_sha256",
                                    "meta_response_sha256",
                                    "meta_terminal_sha256",
                                    "incident_event_sequence",
                                    "incident_event_digest",
                                    "incident_identity_sha256",
                                    "recommendation",
                                    "authorization_receipt_sha256",
                                    "authorization_authentication_sha256",
                                    "invocation_index",
                                )
                            )
                        )
                    )
                    or (
                        not is_meta_result
                        and pending.get("meta_recovery") is not None
                    )
                ):
                    raise ContiguousRunnerError(
                        "substrate health result schema/order mismatch"
                    )
                health_root = (
                    Path(
                        attempts[payload["attempt_id"]][
                            "spec"
                        ].host_transcript_path
                    ).parent
                    / "substrate_health_reprobes"
                    / payload["authorization_id"]
                )
                health = _validate_bound_receipt(
                    payload["health_receipt_path"],
                    payload["health_receipt_sha256"],
                    expected_path=health_root / "receipt.json",
                    expected_kind=(
                        "contiguous_substrate_health_reprobe"
                    ),
                    spec=attempts[
                        payload["attempt_id"]
                    ]["spec"],
                )
                expected_health_keys = {
                    "schema", "kind", "campaign_id", "generation_id",
                    "attempt_id", "attempt_spec_sha256",
                    "authorization_id",
                    "authorization_receipt_sha256", "probe_index",
                    "failed_substrate_identity_sha256",
                    "healthy_substrate_identity_sha256",
                    "incident_failure_receipt_sha256",
                    "remediation_epoch_sha256",
                    "rematerialization_evidence_path",
                    "rematerialization_evidence_sha256",
                    "fresh_state_root_created", "health_state_root",
                    "health_runtime_root", "preflight_receipt_path",
                    "preflight_receipt_sha256",
                    "guardian_state_root_write_probe_status",
                    "scan_receipt_path", "scan_receipt_sha256",
                    "purge_receipt_path", "purge_receipt_sha256",
                    "failure_class", "failure_code",
                    "health_state_root_absent",
                    "health_runtime_root_absent",
                    "proposer_container_started", "bridge_connected",
                    "thread_started", "turn_started",
                    "candidate_authority", "wip_authority",
                    "promotion_authority", "cost_used", "status",
                }
                remediation = _validate_bound_receipt(
                    health.get("rematerialization_evidence_path"),
                    health.get("rematerialization_evidence_sha256"),
                    expected_path=health_root
                    / "rematerialization.json",
                    expected_kind=(
                        "contiguous_substrate_health_"
                        "rematerialization_evidence"
                    ),
                    spec=attempts[
                        payload["attempt_id"]
                    ]["spec"],
                )
                scan = _validate_bound_receipt(
                    health.get("scan_receipt_path"),
                    health.get("scan_receipt_sha256"),
                    expected_path=health_root / "scan.json",
                    expected_kind=(
                        "contiguous_substrate_health_reprobe_scan"
                    ),
                    spec=attempts[
                        payload["attempt_id"]
                    ]["spec"],
                )
                purge = _validate_bound_receipt(
                    health.get("purge_receipt_path"),
                    health.get("purge_receipt_sha256"),
                    expected_path=health_root / "purge.json",
                    expected_kind=(
                        "contiguous_substrate_health_reprobe_purge"
                    ),
                    spec=attempts[
                        payload["attempt_id"]
                    ]["spec"],
                )
                observed_remediation_epoch = remediation.pop(
                    "remediation_epoch_sha256", None
                )
                computed_remediation_epoch = hashlib.sha256(
                    _canonical_json(remediation)
                ).hexdigest()
                expected_healthy_identity = (
                    hashlib.sha256(
                        _canonical_json({
                            "schema": 1,
                            "kind":
                                "healthy_controller_substrate_identity",
                            "failed_substrate_identity_sha256":
                                payload["substrate_identity_sha256"],
                            "remediation_epoch_sha256":
                                payload["remediation_epoch_sha256"],
                            "preflight_receipt_sha256":
                                health.get(
                                    "preflight_receipt_sha256"
                                ),
                            "guardian_state_root_write_probe_status":
                                health.get(
                                    "guardian_state_root_write_probe_status"
                                ),
                            "status": "PASS",
                        })
                    ).hexdigest()
                    if expected_status == "PASS"
                    else None
                )
                if (
                    set(health) != expected_health_keys
                    or health.get("authorization_id")
                    != payload["authorization_id"]
                    or health.get("probe_index")
                    != payload["probe_index"]
                    or health.get("authorization_receipt_sha256")
                    != pending["authorization_receipt_sha256"]
                    or health.get("remediation_epoch_sha256")
                    != payload["remediation_epoch_sha256"]
                    or health.get("failed_substrate_identity_sha256")
                    != payload["substrate_identity_sha256"]
                    or health.get(
                        "incident_failure_receipt_sha256"
                    )
                    != payload["incident_failure_receipt_sha256"]
                    or health.get(
                        "healthy_substrate_identity_sha256"
                    )
                    != payload[
                        "healthy_substrate_identity_sha256"
                    ]
                    or health.get("failure_class")
                    != payload["failure_class"]
                    or health.get("failure_code")
                    != payload["failure_code"]
                    or health.get("status") != expected_status
                    or health.get("fresh_state_root_created")
                    is not True
                    or health.get("health_state_root")
                    == attempts[payload["attempt_id"]][
                        "spec"
                    ].app_server_state_dir
                    or not _is_sha256(
                        health.get(
                            "rematerialization_evidence_sha256"
                        )
                    )
                    or observed_remediation_epoch
                    != payload["remediation_epoch_sha256"]
                    or computed_remediation_epoch
                    != payload["remediation_epoch_sha256"]
                    or health.get(
                        "healthy_substrate_identity_sha256"
                    )
                    != expected_healthy_identity
                    or (
                        expected_status == "PASS"
                        and (
                            health.get(
                                "guardian_state_root_write_probe_status"
                            )
                            != "PASS"
                            or not _is_sha256(
                                health.get(
                                    "preflight_receipt_sha256"
                                )
                            )
                            or health.get("preflight_receipt_path")
                            != str(
                                health_root
                                / "substrate_preflight_receipt.json"
                            )
                            or _sha256_file(
                                Path(
                                    health[
                                        "preflight_receipt_path"
                                    ]
                                )
                            )
                            != health["preflight_receipt_sha256"]
                        )
                    )
                    or health.get("scan_receipt_path")
                    != str(health_root / "scan.json")
                    or set(scan)
                    != {
                        "schema", "kind", "campaign_id",
                        "generation_id", "attempt_id",
                        "attempt_spec_sha256", "authorization_id",
                        "probe_index", "source_scan_receipt_path",
                        "source_scan_receipt_sha256",
                        "state_inventory_before_purge", "status",
                    }
                    or scan.get("authorization_id")
                    != payload["authorization_id"]
                    or scan.get("probe_index")
                    != payload["probe_index"]
                    or scan.get("status") != "COMPLETE"
                    or health.get("purge_receipt_path")
                    != str(health_root / "purge.json")
                    or set(purge)
                    != {
                        "schema", "kind", "campaign_id",
                        "generation_id", "attempt_id",
                        "attempt_spec_sha256", "authorization_id",
                        "probe_index", "scan_receipt_sha256",
                        "health_state_root_absent",
                        "health_runtime_root_absent",
                        "prior_clean_wip_tree_sha256",
                        "post_clean_wip_tree_sha256", "status",
                    }
                    or purge.get("authorization_id")
                    != payload["authorization_id"]
                    or purge.get("probe_index")
                    != payload["probe_index"]
                    or purge.get("scan_receipt_sha256")
                    != health.get("scan_receipt_sha256")
                    or purge.get("health_state_root_absent")
                    is not True
                    or purge.get("health_runtime_root_absent")
                    is not True
                    or purge.get("prior_clean_wip_tree_sha256")
                    != purge.get("post_clean_wip_tree_sha256")
                    or purge.get("status") != "PASS"
                    or health.get("proposer_container_started")
                    is not False
                    or health.get("bridge_connected") is not False
                    or health.get("thread_started") is not False
                    or health.get("turn_started") is not False
                    or health.get("candidate_authority") is not False
                    or health.get("wip_authority") is not False
                    or health.get("promotion_authority") is not False
                    or health.get("cost_used") != 0.0
                    or health.get("health_state_root_absent")
                    is not True
                    or health.get("health_runtime_root_absent")
                    is not True
                ):
                    raise ContiguousRunnerError(
                        "substrate health receipt gained authority or "
                        "retained mutable probe state"
                    )
                if (
                    is_meta_result
                    and (
                        payload[
                            "rematerialization_evidence_path"
                        ]
                        != health.get(
                            "rematerialization_evidence_path"
                        )
                        or payload[
                            "rematerialization_evidence_sha256"
                        ]
                        != health.get(
                            "rematerialization_evidence_sha256"
                        )
                    )
                ):
                    raise ContiguousRunnerError(
                        "meta substrate result substituted "
                        "rematerialization evidence"
                    )
                substrate_incident[
                    "attempted_remediation_epochs"
                ].append(payload["remediation_epoch_sha256"])
                substrate_incident["health_probe_count"] += 1
                substrate_incident["last_health_probe"] = {
                    "authorization_id":
                        payload["authorization_id"],
                    "probe_index": payload["probe_index"],
                    "remediation_epoch_sha256":
                        payload["remediation_epoch_sha256"],
                    "health_receipt_path":
                        payload["health_receipt_path"],
                    "health_receipt_sha256":
                        payload["health_receipt_sha256"],
                    "status": expected_status,
                    "healthy_substrate_identity_sha256":
                        payload[
                            "healthy_substrate_identity_sha256"
                        ],
                    "failure_class": payload["failure_class"],
                    "failure_code": payload["failure_code"],
                }
                substrate_incident["pending_reprobe"] = None
                if kind == "SUBSTRATE_HEALTH_RESTORED":
                    substrate_incident = None
                elif kind == "META_SUBSTRATE_HEALTH_RESTORED":
                    substrate_incident["meta_recovery"] = {
                        **meta_recovery,
                        "phase": "HEALTH_RESTORED",
                        "result": {
                            "recovery_result_event_sequence":
                                event["sequence"],
                            "recovery_result_event_digest":
                                event["digest"],
                            "health_receipt_path":
                                payload["health_receipt_path"],
                            "health_receipt_sha256":
                                payload["health_receipt_sha256"],
                            "rematerialization_evidence_path":
                                payload[
                                    "rematerialization_evidence_path"
                                ],
                            "rematerialization_evidence_sha256":
                                payload[
                                    "rematerialization_evidence_sha256"
                                ],
                            "remediation_epoch_sha256":
                                payload[
                                    "remediation_epoch_sha256"
                                ],
                            "healthy_substrate_identity_sha256":
                                payload[
                                    "healthy_substrate_identity_sha256"
                                ],
                        },
                    }
                elif kind == "META_SUBSTRATE_RECOVERY_FAILED":
                    substrate_incident["meta_recovery"] = {
                        **meta_recovery,
                        "phase": "FAILED",
                        "result": {
                            "event_sequence": event["sequence"],
                            "event_digest": event["digest"],
                            "health_receipt_path":
                                payload["health_receipt_path"],
                            "health_receipt_sha256":
                                payload["health_receipt_sha256"],
                            "failure_class":
                                payload["failure_class"],
                            "failure_code":
                                payload["failure_code"],
                        },
                    }
                else:
                    substrate_incident[
                        "circuit_failure_recorded"
                    ] = False
            elif kind == "META_SUBSTRATE_RESUME_AUTHORIZED":
                required = {
                    "authorization_id",
                    "attempt_id",
                    "incident_event_sequence",
                    "incident_event_digest",
                    "incident_identity_sha256",
                    "recovery_result_event_sequence",
                    "recovery_result_event_digest",
                    "health_receipt_sha256",
                    "rematerialization_evidence_sha256",
                    "healthy_substrate_identity_sha256",
                    "resume_receipt_path",
                    "resume_receipt_sha256",
                    "resume_authentication_sha256",
                    "invocation_index",
                }
                meta_recovery = (
                    None
                    if substrate_incident is None
                    else substrate_incident["meta_recovery"]
                )
                result = (
                    None
                    if not isinstance(meta_recovery, dict)
                    else meta_recovery.get("result")
                )
                if (
                    set(payload) != required
                    or substrate_incident is None
                    or operator_incident is None
                    or not isinstance(meta_recovery, dict)
                    or meta_recovery.get("phase")
                    != "HEALTH_RESTORED"
                    or not isinstance(result, dict)
                    or payload["authorization_id"]
                    != meta_recovery["authorization_id"]
                    or payload["attempt_id"]
                    != substrate_incident["attempt_id"]
                    or payload["incident_event_sequence"]
                    != substrate_incident[
                        "incident_event_sequence"
                    ]
                    or payload["incident_event_digest"]
                    != substrate_incident[
                        "incident_event_digest"
                    ]
                    or payload["incident_identity_sha256"]
                    != substrate_incident[
                        "incident_identity_sha256"
                    ]
                    or payload["invocation_index"] != 1
                    or payload["recovery_result_event_sequence"]
                    != result["recovery_result_event_sequence"]
                    or payload["recovery_result_event_digest"]
                    != result["recovery_result_event_digest"]
                    or payload["health_receipt_sha256"]
                    != result["health_receipt_sha256"]
                    or payload[
                        "rematerialization_evidence_sha256"
                    ]
                    != result[
                        "rematerialization_evidence_sha256"
                    ]
                    or payload[
                        "healthy_substrate_identity_sha256"
                    ]
                    != result[
                        "healthy_substrate_identity_sha256"
                    ]
                    or any(
                        not _is_sha256(payload[name])
                        for name in (
                            "recovery_result_event_digest",
                            "incident_event_digest",
                            "incident_identity_sha256",
                            "health_receipt_sha256",
                            "rematerialization_evidence_sha256",
                            "healthy_substrate_identity_sha256",
                            "resume_receipt_sha256",
                            "resume_authentication_sha256",
                        )
                    )
                ):
                    raise ContiguousRunnerError(
                        "meta substrate resume schema/order mismatch"
                    )
                resume_path = (
                    self.root
                    / META_SUBSTRATE_RECOVERY_AUTHORIZATION_ROOT
                    / (
                        payload["authorization_id"]
                        + "-resume.json"
                    )
                )
                if (
                    payload["resume_receipt_path"]
                    != str(resume_path)
                    or _sha256_file(resume_path)
                    != payload["resume_receipt_sha256"]
                ):
                    raise ContiguousRunnerError(
                        "meta substrate resume receipt is substituted"
                    )
                resume = _read_json_file(resume_path)
                resume_keys = {
                    "schema", "kind", "campaign_id",
                    "authorization_id", "attempt_id",
                    "substrate_identity_sha256",
                    "incident_failure_receipt_sha256",
                    "incident_event_sequence",
                    "incident_event_digest",
                    "incident_identity_sha256",
                    "meta_request_sha256", "meta_response_sha256",
                    "meta_terminal_sha256", "recommendation",
                    "operator_configuration_sha256",
                    "recovery_result_event_sequence",
                    "recovery_result_event_digest",
                    "health_receipt_sha256",
                    "rematerialization_evidence_sha256",
                    "remediation_epoch_sha256",
                    "healthy_substrate_identity_sha256",
                    "invocation_index", "single_use",
                    "solver_authority", "wip_authority",
                    "cost_authority", "promotion_authority",
                    "resume_authentication_sha256",
                }
                unsigned_resume = dict(resume)
                observed_authentication = unsigned_resume.pop(
                    "resume_authentication_sha256", None
                )
                if (
                    set(resume) != resume_keys
                    or resume.get("schema") != 1
                    or resume.get("kind")
                    != (
                        "contiguous_meta_substrate_"
                        "resume_authorization"
                    )
                    or resume.get("campaign_id")
                    != genesis["campaign_id"]
                    or any(
                        resume.get(name) != meta_recovery[name]
                        for name in (
                            "authorization_id",
                            "attempt_id",
                            "substrate_identity_sha256",
                            "incident_failure_receipt_sha256",
                            "incident_event_sequence",
                            "incident_event_digest",
                            "incident_identity_sha256",
                            "meta_request_sha256",
                            "meta_response_sha256",
                            "meta_terminal_sha256",
                            "recommendation",
                            "operator_configuration_sha256",
                            "invocation_index",
                        )
                    )
                    or any(
                        resume.get(name) != result[name]
                        for name in (
                            "recovery_result_event_sequence",
                            "recovery_result_event_digest",
                            "health_receipt_sha256",
                            "rematerialization_evidence_sha256",
                            "remediation_epoch_sha256",
                            "healthy_substrate_identity_sha256",
                        )
                    )
                    or resume.get("single_use") is not True
                    or any(
                        resume.get(name) is not False
                        for name in (
                            "solver_authority",
                            "wip_authority",
                            "cost_authority",
                            "promotion_authority",
                        )
                    )
                    or observed_authentication
                    != payload["resume_authentication_sha256"]
                    or not hmac.compare_digest(
                        str(observed_authentication),
                        meta_substrate_resume_authentication_sha256(
                            unsigned_resume,
                            operator_configuration_sha256=(
                                genesis[
                                    "operator_configuration_sha256"
                                ]
                            ),
                        ),
                    )
                ):
                    raise ContiguousRunnerError(
                        "meta substrate resume receipt is malformed"
                    )
                operation_key = (
                    "substrate_health_reprobe:controller_substrate"
                )
                operation_state = failure_operation_circuits.get(
                    operation_key
                )
                domain_state = failure_domain_circuits.get(
                    "controller_substrate"
                )
                if (
                    not isinstance(operation_state, dict)
                    or not isinstance(domain_state, dict)
                    or operator_incident.get("attempt_id")
                    != substrate_incident["attempt_id"]
                    or operator_incident.get("operation")
                    != "substrate_health_reprobe"
                    or operator_incident.get("fault_domain")
                    != "controller_substrate"
                ):
                    raise ContiguousRunnerError(
                        "meta substrate resume lacks its exact incident"
                    )
                failure_operation_circuits[operation_key] = {
                    **operation_state,
                    "consecutive": 0,
                    "retry_not_before": None,
                }
                failure_domain_circuits["controller_substrate"] = {
                    **domain_state,
                    "consecutive": 0,
                    "retry_not_before": None,
                    "last_operation": None,
                }
                operator_incident = None
                substrate_incident = None
            elif kind == "SUBSTRATE_HEALTH_REPROBE_ABORTED":
                required = {
                    "authorization_id",
                    "attempt_id",
                    "substrate_identity_sha256",
                    "incident_failure_receipt_sha256",
                    "probe_index",
                    "error_type",
                    "status",
                }
                pending = (
                    None
                    if substrate_incident is None
                    else substrate_incident["pending_reprobe"]
                )
                if (
                    set(payload) != required
                    or substrate_incident is None
                    or pending is None
                    or payload["status"] != "ABORTED"
                    or not _safe_identifier(payload["error_type"])
                    or any(
                        payload[name] != pending[name]
                        for name in (
                            "authorization_id",
                            "attempt_id",
                            "substrate_identity_sha256",
                            "incident_failure_receipt_sha256",
                            "probe_index",
                        )
                    )
                ):
                    raise ContiguousRunnerError(
                        "substrate health abort schema/order mismatch"
                    )
                substrate_incident["health_probe_count"] += 1
                substrate_incident["last_health_probe"] = {
                    "authorization_id":
                        payload["authorization_id"],
                    "probe_index": payload["probe_index"],
                    "remediation_epoch_sha256": None,
                    "health_receipt_path": None,
                    "health_receipt_sha256": None,
                    "status": "ABORTED",
                    "error_type": payload["error_type"],
                }
                substrate_incident["pending_reprobe"] = None
                substrate_incident["circuit_failure_recorded"] = False
            elif kind == "ATTEMPT_LAUNCHED":
                attempt = self._attempt(
                    attempts, attempt_id, "BACKEND_PREPARED"
                )
                launched_at = payload.get("launched_at")
                if (
                    set(payload)
                    != {"attempt_id", "launched_at", "launched"}
                    or not _is_finite_number(launched_at)
                ):
                    raise ContiguousRunnerError(
                        "invalid ATTEMPT_LAUNCHED payload"
                    )
                launched = _backend_launch_from_dict(payload["launched"])
                _validate_launch_receipts(
                    attempt["spec"],
                    attempt["prepared"],
                    launched,
                )
                attempt.update(
                    phase="RUNNING",
                    launched=launched,
                    launched_at=float(launched_at),
                )
            elif kind == "ATTEMPT_OBSERVED":
                attempt = self._attempt(
                    attempts, attempt_id, {"RUNNING", "DRAINING"}
                )
                observation = _backend_poll_from_dict(
                    payload.get("observation")
                )
                if (
                    set(payload)
                    != {"attempt_id", "observation_index", "observation"}
                    or payload["observation_index"]
                    != attempt["observation_count"] + 1
                    or payload["observation_index"]
                    > Scheduler.MAX_JOURNALED_OBSERVATIONS_PER_ATTEMPT
                    or (
                        attempt["last_observation_at"] is not None
                        and event["recorded_at"]
                        - attempt["last_observation_at"]
                        < Scheduler
                        .MIN_JOURNALED_OBSERVATION_INTERVAL_SECONDS
                    )
                    or observation.observation_sha256
                    == attempt["last_observation_sha256"]
                ):
                    raise ContiguousRunnerError(
                        "backend observation event schema/order mismatch"
                    )
                if observation.status != "running":
                    raise ContiguousRunnerError(
                        "ATTEMPT_OBSERVED must be a running observation"
                    )
                attempt["observation_count"] += 1
                attempt["last_observation_at"] = event["recorded_at"]
                attempt["last_observation_sha256"] = (
                    observation.observation_sha256
                )
            elif kind == "ATTEMPT_DRAINING":
                attempt = self._attempt(attempts, attempt_id, "RUNNING")
                if (
                    set(payload) != {"attempt_id", "soft_deadline"}
                    or payload.get("soft_deadline") != (
                    attempt["launched_at"]
                    + attempt["spec"].soft_allocation_seconds
                    )
                ):
                    raise ContiguousRunnerError(
                        "draining event has wrong soft deadline"
                    )
                attempt["phase"] = "DRAINING"
            elif kind == "ATTEMPT_EXITED":
                attempt = self._attempt(
                    attempts, attempt_id, {"RUNNING", "DRAINING"}
                )
                if set(payload) != {"attempt_id", "terminal"}:
                    raise ContiguousRunnerError(
                        "terminal event schema mismatch"
                    )
                terminal = _backend_poll_from_dict(payload["terminal"])
                if terminal.status not in {
                    "exited", "containment_fault"
                }:
                    raise ContiguousRunnerError(
                        "terminal event is not terminal"
                    )
                attempt.update(phase="EXITED", terminal=terminal)
            elif kind == "ATTEMPT_PUBLIC_OBSERVATIONS_STAGING":
                attempt = self._attempt(attempts, attempt_id, "EXITED")
                if (
                    set(payload) != {"attempt_id", "transition"}
                    or attempt["public_observation_transition"]
                    is not None
                ):
                    raise ContiguousRunnerError(
                        "public observation staging event schema mismatch"
                    )
                try:
                    Scheduler.validate_public_observation_transition(
                        payload["transition"],
                        attempt_id=attempt_id,
                        generation_id=(
                            attempt["spec"].generation_id
                        ),
                        game=attempt["spec"].game,
                        frontier_sha256=(
                            attempt["spec"].frontier_sha256
                        ),
                        parent_checkpoint_sha256=(
                            attempt["spec"]
                            .parent_checkpoint_sha256
                        ),
                        host_transcript_path=(
                            attempt["spec"].host_transcript_path
                        ),
                        reopen_receipts=False,
                    )
                except Scheduler.SchedulerError as exc:
                    raise ContiguousRunnerError(
                        "public observation staging identity is invalid"
                    ) from exc
                attempt["public_observation_transition"] = copy.deepcopy(
                    payload["transition"]
                )
            elif kind == "ATTEMPT_COLLECTED":
                attempt = self._attempt(attempts, attempt_id, "EXITED")
                transition = attempt["public_observation_transition"]
                if (
                    set(payload)
                    != {
                        "attempt_id",
                        "collection",
                        "public_observation_transition_sha256",
                    }
                    or not isinstance(transition, dict)
                    or payload[
                        "public_observation_transition_sha256"
                    ] != Scheduler.sha256_json(transition)
                ):
                    raise ContiguousRunnerError(
                        "collection event schema mismatch"
                    )
                collection = _backend_collection_from_dict(
                    payload["collection"]
                )
                try:
                    Scheduler.validate_public_observation_transition(
                        transition,
                        attempt_id=attempt_id,
                        generation_id=(
                            attempt["spec"].generation_id
                        ),
                        game=attempt["spec"].game,
                        frontier_sha256=(
                            attempt["spec"].frontier_sha256
                        ),
                        parent_checkpoint_sha256=(
                            attempt["spec"]
                            .parent_checkpoint_sha256
                        ),
                        host_transcript_path=(
                            attempt["spec"].host_transcript_path
                        ),
                        result_kind=collection.result.kind,
                        receipt_sha256s=(
                            collection
                            .native_public_observation_receipt_sha256s
                        ),
                        reopen_receipts=True,
                    )
                except Scheduler.SchedulerError as exc:
                    raise ContiguousRunnerError(
                        "collection crosses its public observation "
                        "transition"
                    ) from exc
                attempt.update(
                    phase="COLLECTED",
                    collection=collection,
                    outcome=collection.result,
                )
            elif kind == "ATTEMPT_COLLECTION_REJECTED":
                attempt = self._attempt(attempts, attempt_id, "EXITED")
                if (
                    set(payload) != {"attempt_id", "reason", "result"}
                    or not isinstance(payload["reason"], str)
                ):
                    raise ContiguousRunnerError(
                        "collection rejection event schema mismatch"
                    )
                result_payload = payload["result"]
                if not isinstance(result_payload, dict):
                    raise ContiguousRunnerError(
                        "collection rejection result must be an object"
                    )
                result = self._result_from_payload(
                    {"attempt_id": attempt_id, **result_payload}
                )
                if result.kind != "infrastructure":
                    raise ContiguousRunnerError(
                        "collection rejection must be infrastructure"
                    )
                attempt.update(
                    phase="COLLECTION_REJECTED", outcome=result
                )
            elif kind == "ATTEMPT_PUBLIC_ACTION_PROTOCOL_INVALID":
                attempt = self._attempt(attempts, attempt_id, "EXITED")
                if (
                    set(payload)
                    != {
                        "attempt_id",
                        "protocol_invalid_receipt_path",
                        "protocol_invalid_receipt_sha256",
                        "terminal_evidence",
                        "result",
                    }
                    or attempt["terminal"].status
                    != "containment_fault"
                ):
                    raise ContiguousRunnerError(
                        "protocol-invalid event schema/terminal mismatch"
                    )
                result_payload = payload["result"]
                if not isinstance(result_payload, dict):
                    raise ContiguousRunnerError(
                        "protocol-invalid result must be an object"
                    )
                result = self._result_from_payload(
                    {"attempt_id": attempt_id, **result_payload}
                )
                if (
                    result.kind != "protocol_invalid"
                    or result.candidate is not None
                    or result.wip is not None
                    or result.blocker is not None
                    or result.native_sidecar_request_draft is not None
                    or result.reason != "public_action_protocol_invalid"
                ):
                    raise ContiguousRunnerError(
                        "protocol-invalid outcome retained lineage authority"
                    )
                receipt = _validate_bound_receipt(
                    payload["protocol_invalid_receipt_path"],
                    payload["protocol_invalid_receipt_sha256"],
                    expected_path=Path(
                        attempt["spec"].host_transcript_path
                    ).parent
                    / "arena_public_action_protocol_invalid_receipt.json",
                    expected_kind=(
                        "contiguous_arena_public_action_protocol_invalid"
                    ),
                    spec=attempt["spec"],
                )
                raw_terminal_evidence = payload[
                    "terminal_evidence"
                ]
                if not isinstance(raw_terminal_evidence, dict):
                    raise ContiguousRunnerError(
                        "protocol-invalid terminal evidence is not an object"
                    )
                terminal_evidence = (
                    _validate_protocol_invalid_terminal_evidence(
                        spec=attempt["spec"],
                        receipt=receipt,
                        evidence=raw_terminal_evidence,
                    )
                )
                if (
                    receipt.get("cost_used") != result.cost_used
                    or receipt.get("status") != "PROTOCOL_INVALID"
                    or any(
                        receipt.get(name) is not False
                        for name in (
                            "candidate_admissible",
                            "wip_admissible",
                            "public_observation_admissible",
                            "sidecar_request_admissible",
                            "supervisory_handoff_admissible",
                            "promotion_admissible",
                            "restart_restoration_admissible",
                        )
                    )
                ):
                    raise ContiguousRunnerError(
                        "protocol-invalid receipt regained authority"
                    )
                attempt.update(
                    phase="COLLECTION_REJECTED",
                    outcome=result,
                    protocol_invalid={
                        "path": payload[
                            "protocol_invalid_receipt_path"
                        ],
                        "sha256": payload[
                            "protocol_invalid_receipt_sha256"
                        ],
                        "terminal_evidence": terminal_evidence,
                    },
                )
            elif kind == "ATTEMPT_TORN_DOWN":
                attempt = self._attempt(
                    attempts,
                    attempt_id,
                    {"COLLECTED", "COLLECTION_REJECTED"},
                )
                if set(payload) != {"attempt_id", "teardown"}:
                    raise ContiguousRunnerError(
                        "teardown event schema mismatch"
                    )
                proof = _backend_teardown_from_dict(payload["teardown"])
                expected_cause = (
                    "containment_fault"
                    if attempt["terminal"].status == "containment_fault"
                    else "normal_exit"
                )
                if (
                    proof.container_id
                    != attempt["launched"].container_id
                    or proof.cause != expected_cause
                    or Path(
                        attempt["spec"].arena_socket_path
                    ).exists()
                    or Path(
                        attempt["spec"].arena_socket_path
                    ).is_symlink()
                    or Path(
                        attempt["spec"].arena_token_file_path
                    ).exists()
                    or Path(
                        attempt["spec"].arena_token_file_path
                    ).is_symlink()
                    or Path(
                        attempt["spec"].bridge_socket_path
                    ).exists()
                    or Path(
                        attempt["spec"].bridge_socket_path
                    ).is_symlink()
                    or Path(
                        attempt["spec"].bridge_token_file_path
                    ).exists()
                    or Path(
                        attempt["spec"].bridge_token_file_path
                    ).is_symlink()
                    or Path(
                        attempt["spec"].app_server_control_dir
                    ).exists()
                    or Path(
                        attempt["spec"].app_server_control_dir
                    ).is_symlink()
                ):
                    raise ContiguousRunnerError(
                        "teardown proof does not match launched container"
                    )
                _validate_arena_volume_teardown(
                    spec=attempt["spec"],
                    prepared=attempt["prepared"],
                    proof=proof,
                )
                _validate_terminal_canary_reveal(
                    spec=attempt["spec"],
                    prepared=attempt["prepared"],
                    launched=attempt["launched"],
                    proof=proof,
                    canaries=self._controller_state_canaries,
                )
                _validate_terminal_canary_cleanup(
                    spec=attempt["spec"],
                    prepared=attempt["prepared"],
                    proof=proof,
                )
                attempt.update(phase="TORN_DOWN", teardown=proof)
            elif kind == "ATTEMPT_RESULT":
                attempt = self._attempt(
                    attempts, attempt_id, "TORN_DOWN"
                )
                result = self._result_from_payload(payload)
                expected_result = apply_terminal_result_precedence(
                    attempt["terminal"].status,
                    attempt["outcome"],
                )
                if result != expected_result:
                    raise ContiguousRunnerError(
                        "attempt result differs from authenticated terminal "
                        "precedence"
                    )
                if (
                    result.kind == "blocker"
                    and self._sanitize_result(
                        attempt["spec"], result
                    ) != result
                ):
                    raise ContiguousRunnerError(
                        "attempt blocker lacks current host authentication"
                    )
                attempt["settled_result"] = result
                try:
                    authenticated_cost_units = (
                        Scheduler.charge_to_units(result.cost_used)
                    )
                    if (
                        payload.get("authenticated_cost_units")
                        != authenticated_cost_units
                        or payload.get("budget_reservation_id")
                        != attempt["budget_reservation_id"]
                        or payload.get("scheduler_decision_id")
                        != attempt["scheduler_decision_id"]
                    ):
                        raise Scheduler.SchedulerError(
                            "settlement identity/units mismatch"
                        )
                    budget = Scheduler.settle_budget(
                        budget,
                        reservation_id=attempt[
                            "budget_reservation_id"
                        ],
                        attempt_id=attempt_id,
                        charged_units=authenticated_cost_units,
                    )
                except Scheduler.SchedulerError as exc:
                    raise ContiguousRunnerError(
                        "attempt result violates budget settlement"
                    ) from exc
                lane = lanes[attempt["spec"].game]
                if (
                    attempt["collection"] is not None
                    and result.kind
                    in {"clean_no_progress", "candidate"}
                ):
                    semantic_receipts = {
                        *lane[
                            "public_observation_receipt_sha256s"
                        ],
                        *attempt["collection"]
                        .native_public_observation_receipt_sha256s,
                    }
                    # WIP source novelty is a separately typed evidence
                    # epoch carried by lane["wip"].  It is executable source,
                    # never a public Arena observation receipt, and therefore
                    # must not enter the content-addressed observation ledger.
                    lane[
                        "public_observation_receipt_sha256s"
                    ] = sorted(semantic_receipts)
                coordinate_outcome = result.kind
                if (
                    attempt["collection"] is not None
                    and attempt["collection"]
                    .structured_provider_outcome != "completed"
                ):
                    coordinate_outcome = (
                        attempt["collection"]
                        .structured_provider_outcome
                    )
                elif (
                    attempt["terminal"] is not None
                    and attempt["terminal"].status
                    == "containment_fault"
                ):
                    coordinate_outcome = "containment_fault"
                next_no_progress = (
                    advance_exact_frontier_clean_no_progress(
                        lane["no_progress"], coordinate_outcome
                    )
                )
                try:
                    transition = Scheduler.terminal_policy_transition(
                        result.kind
                    )
                except Scheduler.SchedulerError as exc:
                    raise ContiguousRunnerError(
                        "attempt result has no canonical lane transition"
                    ) from exc
                exposure_detected = (
                    attempt["terminal"].status
                    == "containment_fault"
                    or (
                        attempt["collection"] is not None
                        and attempt["collection"]
                        .structured_provider_outcome
                        == "containment_fault"
                    )
                )
                if result.wip is not None:
                    self._validate_wip_for_spec(
                        result.wip, attempt["spec"]
                    )
                try:
                    lane["wip"] = Scheduler.reduce_terminal_wip(
                        transition=transition,
                        prior_wip=lane["wip"],
                        current_attempt_wip=result.wip,
                        exposure_detected=exposure_detected,
                    )
                except Scheduler.SchedulerError as exc:
                    raise ContiguousRunnerError(
                        "terminal WIP reduction violates policy"
                    ) from exc
                if transition.next_lane_phase == "PROMOTING":
                    if next_no_progress != lane["no_progress"]:
                        raise ContiguousRunnerError(
                            "candidate illegally advances retry coordinate"
                        )
                    self._validate_candidate(attempt["spec"], result.candidate)
                    attempt["phase"] = "PROMOTING"
                    attempt["candidate"] = result.candidate
                else:
                    if transition.retry_coordinate_delta == 1:
                        decision = attempt["scheduler_decision"]
                        settlement = Scheduler.CleanProposerSettlement(
                            schema=1,
                            game=attempt["spec"].game,
                            frontier_sha256=attempt["spec"].frontier_sha256,
                            parent_checkpoint_sha256=(
                                attempt["spec"]
                                .parent_checkpoint_sha256
                            ),
                            attempt_id=attempt_id,
                            scheduler_decision_id=(
                                decision.decision_id
                            ),
                            no_progress_before=lane["no_progress"],
                            effort=decision.choice.effort,
                            soft_allocation_seconds=(
                                decision.choice
                                .soft_allocation_seconds
                            ),
                            requested_wip_mode=(
                                decision.choice.requested_wip_mode
                            ),
                            supervisory_handoff_sha256=(
                                decision.choice
                                .selected_supervisory_handoff
                                .supervisory_handoff_sha256
                                if decision.choice
                                .selected_supervisory_handoff
                                is not None
                                else None
                            ),
                            result_sequence=event["sequence"],
                            result_digest=event["digest"],
                        )
                        try:
                            Scheduler.validate_clean_proposer_settlement(
                                settlement
                            )
                        except Scheduler.SchedulerError as exc:
                            raise ContiguousRunnerError(
                                "clean proposer settlement is invalid"
                            ) from exc
                        lane[
                            "clean_proposer_settlements"
                        ].append(settlement)
                        lane["no_progress"] = next_no_progress
                        if (
                            transition
                            .current_attempt_wip_disposition
                            != "admit_clean_same_frontier_replacement"
                        ):
                            raise ContiguousRunnerError(
                                "clean transition cannot admit replacement WIP"
                            )
                    elif transition.next_lane_phase == "BLOCKED":
                        lane["blocked"] = result.reason or "unspecified blocker"
                    if (
                        transition.retry_coordinate_delta == 0
                        and lane["no_progress"] != next_no_progress
                    ):
                        raise ContiguousRunnerError(
                            "non-clean outcome changed retry coordinate"
                        )
                    lane["active"] = None
                    attempt["phase"] = "CLOSED"
            elif kind == "PROMOTION_COMMITTED":
                attempt = self._attempt(attempts, attempt_id, "PROMOTING")
                commit = self._commit_from_payload(payload)
                if (
                    attempt["candidate"] is None
                    or payload["candidate_manifest_sha256"]
                    != attempt["candidate"].candidate_manifest_sha256
                ):
                    raise ContiguousRunnerError(
                        "promotion does not consume the exact candidate"
                    )
                self._validate_commit(
                    attempt["spec"],
                    commit,
                    lanes,
                    attempt["candidate"],
                )
                lane = lanes[commit.game]
                promoted_source = self._commit_source_path(commit)
                lane.update(
                    reached=commit.to_level,
                    checkpoint_path=commit.checkpoint_path,
                    checkpoint_sha256=commit.checkpoint_sha256,
                    source_path=str(promoted_source),
                    source_tree_sha256=commit.source_tree_sha256,
                    no_progress=0,
                    wip=None,
                    active=None,
                    blocked=None,
                    clean_proposer_settlements=[],
                    public_observation_receipt_sha256s=[],
                )
                old_frontier = attempt["spec"].frontier_sha256
                complexity_rounds[:] = [
                    replace(item, invalidated=True)
                    if (
                        item.game == commit.game
                        and item.frontier_sha256 == old_frontier
                    )
                    else item
                    for item in complexity_rounds
                ]
                for auxiliary in auxiliary_assignments.values():
                    if (
                        auxiliary["state"].game == commit.game
                        and auxiliary["state"].frontier_sha256
                        == old_frontier
                    ):
                        auxiliary["state"] = replace(
                            auxiliary["state"], invalidated=True
                        )
                for request_record in sidecar_requests.values():
                    request = request_record["request"]
                    if (
                        request.game == commit.game
                        and request.frontier_sha256 == old_frontier
                    ):
                        request_record["invalidated"] = True
                attempt["phase"] = "CLOSED"
            elif kind == "PROMOTION_FAILED":
                attempt = self._attempt(attempts, attempt_id, "PROMOTING")
                transition = (
                    Scheduler.promotion_failure_policy_transition()
                )
                if (
                    set(payload) != {"attempt_id", "code"}
                    or payload["code"]
                    not in Scheduler.PROMOTION_FAILURE_CODES
                    or transition.next_lane_phase != "READY"
                    or transition.retry_coordinate_delta != 0
                    or transition.blocker_authority is not False
                ):
                    raise ContiguousRunnerError(
                        "promotion failure schema mismatch"
                    )
                lane = lanes[attempt["spec"].game]
                lane["active"] = None
                attempt["phase"] = "CLOSED"
            else:
                raise ContiguousRunnerError(f"unknown journal event: {kind}")

        expected_generations = {
            attempt["reservation"].generation_id
            for attempt in attempts.values()
        }
        actual_generations: set[str] = set()
        for path in self.generations.iterdir():
            if (
                path.is_symlink()
                or not path.is_dir()
                or not _is_uuid4(path.name)
            ):
                raise ContiguousRunnerError(
                    f"unexpected generation entry: {path}"
                )
            actual_generations.add(path.name)
        if not actual_generations <= expected_generations:
            raise ContiguousRunnerError(
                "generation tree contains an unjournaled identity"
            )

        # Only files that can still influence execution/promotion are rehashed
        # on every scheduler pass.  A closed blocker is the sole exception:
        # its external receipt remains live lane-stopping authority and must be
        # reopened and host-authenticated on every pass, including cached
        # reducer recovery.
        for attempt in attempts.values():
            if attempt["phase"] == "CLOSED":
                settled = attempt.get("settled_result")
                if (
                    isinstance(settled, AttemptResult)
                    and settled.kind == "blocker"
                    and self._sanitize_result(
                        attempt["spec"], settled
                    ) != settled
                ):
                    raise ContiguousRunnerError(
                        "closed blocker authority changed after settlement"
                    )
                if attempt.get("protocol_invalid") is not None:
                    spec = attempt["spec"]
                    protocol_invalid = attempt["protocol_invalid"]
                    receipt = _validate_bound_receipt(
                        protocol_invalid["path"],
                        protocol_invalid["sha256"],
                        expected_path=Path(
                            spec.host_transcript_path
                        ).parent
                        / (
                            "arena_public_action_protocol_invalid_"
                            "receipt.json"
                        ),
                        expected_kind=(
                            "contiguous_arena_public_action_protocol_"
                            "invalid"
                        ),
                        spec=spec,
                    )
                    if (
                        receipt.get("status")
                        != "PROTOCOL_INVALID"
                        or receipt.get("cost_used")
                        != settled.cost_used
                    ):
                        raise ContiguousRunnerError(
                            "closed protocol-invalid evidence changed"
                        )
                    _validate_protocol_invalid_terminal_evidence(
                        spec=spec,
                        receipt=receipt,
                        evidence=protocol_invalid[
                            "terminal_evidence"
                        ],
                    )
                continue
            if attempt["phase"] == "RESERVED":
                continue
            spec = attempt["spec"]
            self._validate_prepared_input(
                spec,
                require_initial_workspace=attempt["phase"]
                in {"PREPARED", "BACKEND_PREPARED"},
            )
            if attempt["prepared"] is not None:
                prepared = attempt["prepared"]
                _validate_preparation_receipts(spec, prepared)
            if attempt["launched"] is not None:
                _validate_launch_receipts(
                    spec, attempt["prepared"], attempt["launched"]
                )
            if attempt["collection"] is not None:
                self._validate_collection(
                    spec,
                    attempt["prepared"],
                    attempt["launched"],
                    attempt["collection"],
                    allow_arena_teardown_receipt=(
                        attempt["teardown"] is not None
                    ),
                )
            if attempt.get("protocol_invalid") is not None:
                protocol_invalid = attempt["protocol_invalid"]
                receipt = _validate_bound_receipt(
                    protocol_invalid["path"],
                    protocol_invalid["sha256"],
                    expected_path=Path(
                        spec.host_transcript_path
                    ).parent
                    / "arena_public_action_protocol_invalid_receipt.json",
                    expected_kind=(
                        "contiguous_arena_public_action_protocol_invalid"
                    ),
                    spec=spec,
                )
                if (
                    receipt.get("status") != "PROTOCOL_INVALID"
                    or receipt.get("cost_used")
                    != attempt["outcome"].cost_used
                ):
                    raise ContiguousRunnerError(
                        "live protocol-invalid authority changed"
                    )
                _validate_protocol_invalid_terminal_evidence(
                    spec=spec,
                    receipt=receipt,
                    evidence=protocol_invalid[
                        "terminal_evidence"
                    ],
                )

        for game, lane in lanes.items():
            if not 0 <= lane["reached"] <= lane["target"]:
                raise ContiguousRunnerError(
                    f"lane exceeds authoritative target: {game}"
                )
            self._validate_lane_checkpoint_cached(
                game=game, lane=lane
            )
            self._validate_lane_source_cached(game=game, lane=lane)
        public_observation_registry_sha256 = (
            self._validate_public_observation_registry(attempts)
        )
        solved = sum(lane["reached"] for lane in lanes.values())
        if solved > Contract.EXPECTED_LEVELS:
            raise ContiguousRunnerError("campaign solved count exceeds inventory")
        try:
            journal_prefix = Scheduler.journal_prefix_status(self.root)
        except Scheduler.SchedulerError as exc:
            raise ContiguousRunnerError(
                "journal prefix exceeds scheduler evidence bounds"
            ) from exc
        # Publish the reducer checkpoint only after every current executable
        # artifact and journal-prefix bound has passed.  It is an optimization
        # of deterministic replay, never an alternate admission path.
        self._reducer_checkpoint = _ReducerCheckpoint(
            head_sequence=int(events[-1]["sequence"]),
            head_digest=str(events[-1]["digest"]),
            genesis_digest=str(events[0]["digest"]),
            # The public state returned below must share no mutable object,
            # including a dataclass that an adversary could alter through
            # ``object.__setattr__``, with the next reducer checkpoint.
            lanes=copy.deepcopy(lanes),
            attempts=copy.deepcopy(attempts),
            budget=copy.deepcopy(budget),
            pending_decision=copy.deepcopy(pending_decision),
            pending_auxiliary_decision=copy.deepcopy(
                pending_auxiliary_decision
            ),
            auxiliary_assignments=copy.deepcopy(
                auxiliary_assignments
            ),
            sidecar_requests=copy.deepcopy(sidecar_requests),
            complexity_rounds=copy.deepcopy(complexity_rounds),
            used_decision_ids=set(used_decision_ids),
            used_attempt_ids=set(used_attempt_ids),
            used_generation_ids=set(used_generation_ids),
            used_reservation_ids=set(used_reservation_ids),
            used_expert_ids=set(used_expert_ids),
            used_thread_ids=set(used_thread_ids),
            failure_operation_circuits=copy.deepcopy(
                failure_operation_circuits
            ),
            failure_domain_circuits=copy.deepcopy(
                failure_domain_circuits
            ),
            operator_incident=copy.deepcopy(operator_incident),
            substrate_incident=copy.deepcopy(substrate_incident),
            storage_incident=copy.deepcopy(storage_incident),
            storage_quiescence=copy.deepcopy(storage_quiescence),
        )
        return {
            "schema": RUNNER_SCHEMA,
            "campaign_id": genesis["campaign_id"],
            "inventory": inventory,
            "max_lanes": genesis["max_lanes"],
            "limit": genesis["limit"],
            "cost_window_id": budget.cost_window_id,
            "operator_configuration_sha256":
                genesis.get("operator_configuration_sha256"),
            "limit_units": budget.limit_units,
            "settled_cost_units": budget.settled_units,
            "live_budget_reservations": [
                asdict(item) for item in budget.live_reservations
            ],
            "cost_used": budget.settled_units / Scheduler.COST_SCALE,
            "scheduler_policy_sha256": SCHEDULER_POLICY_SHA256,
            "auxiliary_launch_configuration":
                Scheduler.auxiliary_launch_configuration_to_dict(
                    auxiliary_configuration
                ),
            "auxiliary_launch_ready":
                CONTIGUOUS_AUXILIARY_LAUNCH_READY,
            "pending_scheduler_decision": (
                json.loads(_canonical_json(
                    Scheduler.decision_to_dict(pending_decision)
                ))
                if pending_decision is not None
                else None
            ),
            "pending_auxiliary_decision": (
                json.loads(_canonical_json(
                    Scheduler.auxiliary_decision_to_dict(
                        pending_auxiliary_decision
                    )
                ))
                if pending_auxiliary_decision is not None
                else None
            ),
            "used_scheduler_identifiers": sorted(
                used_decision_ids
                | used_attempt_ids
                | used_generation_ids
                | used_reservation_ids
                | used_expert_ids
            ),
            "used_auxiliary_thread_ids": sorted(used_thread_ids),
            "failure_operation_circuits":
                failure_operation_circuits,
            "failure_domain_circuits": failure_domain_circuits,
            "operator_incident": operator_incident,
            "substrate_incident": substrate_incident,
            "storage_incident": storage_incident,
            "storage_quiescence": storage_quiescence,
            "journal_prefix": journal_prefix,
            "public_observation_registry_sha256":
                public_observation_registry_sha256,
            "lanes": lanes,
            "attempts": attempts,
            "auxiliary_assignments": auxiliary_assignments,
            "sidecar_requests": sidecar_requests,
            "complexity_rounds": complexity_rounds,
            "solved_levels": solved,
            "total_levels": sum(inventory.values()),
            "complete": solved == sum(inventory.values()),
            "draining": any(
                attempt["phase"] == "DRAINING"
                for attempt in attempts.values()
            ),
        }

    @staticmethod
    def _attempt(
        attempts: dict[str, dict[str, Any]],
        attempt_id: str,
        expected: str | set[str],
    ) -> dict[str, Any]:
        attempt = attempts.get(attempt_id)
        allowed = {expected} if isinstance(expected, str) else expected
        if attempt is None or attempt["phase"] not in allowed:
            raise ContiguousRunnerError(
                f"invalid transition for attempt {attempt_id}: "
                f"expected={sorted(allowed)}, "
                f"found={attempt and attempt['phase']}"
            )
        return attempt

    def _validate_wip_for_spec(
        self,
        wip: WipSnapshot,
        spec: AttemptSpec,
        *,
        require_current_state: bool = True,
        receipt_spec: AttemptSpec | None = None,
    ) -> None:
        _wip_from_dict(asdict(wip))
        authority_spec = (
            spec if require_current_state else receipt_spec
        )
        if authority_spec is None:
            raise ContiguousRunnerError(
                "retained WIP lacks its authority attempt spec"
            )
        authority_handoff = authority_spec.supervisory_handoff
        if (
            wip.game != spec.game
            or wip.target_level != spec.target_level
            or wip.parent_checkpoint_sha256
            != spec.parent_checkpoint_sha256
            or wip.frontier_sha256 != spec.frontier_sha256
            or wip.taint_verdict != "clean"
            or (
                require_current_state
                and wip.app_server_state_dir
                != spec.app_server_state_dir
            )
            or (
                authority_handoff is None
                and any(
                    item is not None
                    for item in (
                        wip.supervisory_handoff_sha256,
                        wip
                        .supervisory_native_reproduction_receipt_path,
                        wip
                        .supervisory_native_reproduction_receipt_sha256,
                    )
                )
            )
            or (
                authority_handoff is not None
                and (
                    wip.supervisory_handoff_sha256
                    != authority_handoff
                    .supervisory_handoff_sha256
                    or not _safe_path_string(
                        wip
                        .supervisory_native_reproduction_receipt_path
                    )
                    or not _is_sha256(
                        wip
                        .supervisory_native_reproduction_receipt_sha256
                    )
                )
            )
        ):
            raise ContiguousRunnerError(
                "WIP snapshot does not match exact clean frontier"
            )
        publication_path = Path(wip.wip_publication_receipt_path)
        origin = publication_path.parent
        host_root = origin / "host"
        output_root = origin / "output"
        expected_paths = {
            "wip_root": output_root / "wip",
            "solver_source": output_root / "wip" / "solver_source",
            "state": origin / "state" / "codex_home",
            "final_binding": origin / "final_thread_binding.json",
            "final_chain": host_root
            / "final_transcript_chain_receipt.json",
            "controller_scan": host_root
            / "controller_state_scan_receipt.json",
            "retained_scan": origin
            / "retained_canary_scan_receipt.json",
            "taint_scan": host_root / "taint_scan_receipt.json",
            "token_usage": host_root / "token_usage_receipt.json",
            "provider_usage": host_root
            / "provider_usage_receipt.json",
            "wip_export": host_root / "wip_export_receipt.json",
            "wip_publication": origin
            / "wip_publication_receipt.json",
        }
        if authority_handoff is not None:
            expected_paths["supervisory_reproduction"] = (
                host_root
                / "supervisory_native_reproduction_receipt.json"
            )
        observed_paths = {
            "wip_root": Path(wip.wip_root_path),
            "solver_source": Path(wip.solver_source_path),
            "state": Path(wip.app_server_state_dir),
            "final_binding": Path(wip.final_thread_binding_path),
            "final_chain": Path(
                wip.final_transcript_chain_receipt_path
            ),
            "controller_scan": Path(
                wip.controller_state_scan_receipt_path
            ),
            "retained_scan": Path(
                wip.retained_canary_scan_receipt_path
            ),
            "taint_scan": Path(wip.taint_scan_receipt_path),
            "token_usage": Path(wip.token_usage_receipt_path),
            "provider_usage": Path(
                wip.provider_usage_receipt_path
            ),
            "wip_export": Path(wip.wip_export_receipt_path),
            "wip_publication": publication_path,
        }
        if authority_handoff is not None:
            observed_paths["supervisory_reproduction"] = Path(
                str(
                    wip
                    .supervisory_native_reproduction_receipt_path
                )
            )
        if (
            not origin.is_absolute()
            or origin.name == ""
            or not _is_uuid4(origin.name)
            or any(
                observed_paths[name] != expected_path
                for name, expected_path in expected_paths.items()
            )
        ):
            raise ContiguousRunnerError(
                "WIP evidence paths do not form one canonical generation"
            )

        publication = _read_json_file(publication_path)
        publication_base = {
            "schema": 1,
            "kind": "contiguous_wip_publication",
            "campaign_id": spec.campaign_id,
            "generation_id": origin.name,
        }
        receipt_identity = {
            "campaign_id": publication.get("campaign_id"),
            "generation_id": publication.get("generation_id"),
            "attempt_id": publication.get("attempt_id"),
            "attempt_spec_sha256": publication.get(
                "attempt_spec_sha256"
            ),
        }
        publication_fields = _wip_publication_fields(wip)
        bound_spec = authority_spec
        expected_receipt_identity = {
            "campaign_id": bound_spec.campaign_id,
            "generation_id": bound_spec.generation_id,
            "attempt_id": bound_spec.attempt_id,
            "attempt_spec_sha256":
                proposer_attempt_binding_sha256(bound_spec),
        }
        if (
            _sha256_file(publication_path)
            != wip.wip_publication_receipt_sha256
            or any(
                publication.get(key) != value
                for key, value in publication_base.items()
            )
            or not _is_uuid4(receipt_identity["attempt_id"])
            or not _is_sha256(
                receipt_identity["attempt_spec_sha256"]
            )
            or set(publication)
            != {
                *publication_base,
                "attempt_id",
                "attempt_spec_sha256",
                *publication_fields,
            }
            or any(
                publication.get(key) != value
                for key, value in publication_fields.items()
            )
            or receipt_identity != expected_receipt_identity
        ):
            raise ContiguousRunnerError(
                "WIP publication receipt is incomplete or substituted"
            )

        def reopen_receipt(
            *,
            path: Path,
            digest: str,
            kind: str,
        ) -> dict[str, Any]:
            value = _read_json_file(path)
            if (
                _sha256_file(path) != digest
                or value.get("schema") != 1
                or value.get("kind") != kind
                or any(
                    value.get(key) != expected
                    for key, expected in receipt_identity.items()
                )
            ):
                raise ContiguousRunnerError(
                    f"retained WIP {kind} receipt is not independently "
                    "reopenable"
                )
            return value

        final_chain = reopen_receipt(
            path=observed_paths["final_chain"],
            digest=wip.final_transcript_chain_receipt_sha256,
            kind="contiguous_final_transcript_chain",
        )
        controller_scan_receipt = reopen_receipt(
            path=observed_paths["controller_scan"],
            digest=wip.controller_state_scan_receipt_sha256,
            kind="contiguous_controller_state_scan",
        )
        retained_receipt = reopen_receipt(
            path=observed_paths["retained_scan"],
            digest=wip.retained_canary_scan_receipt_sha256,
            kind="contiguous_retained_canary_scan",
        )
        taint_receipt = reopen_receipt(
            path=observed_paths["taint_scan"],
            digest=wip.taint_scan_receipt_sha256,
            kind="contiguous_taint_scan",
        )
        token_receipt = reopen_receipt(
            path=observed_paths["token_usage"],
            digest=wip.token_usage_receipt_sha256,
            kind="contiguous_token_usage",
        )
        provider_receipt = reopen_receipt(
            path=observed_paths["provider_usage"],
            digest=wip.provider_usage_receipt_sha256,
            kind="contiguous_provider_usage",
        )
        if set(provider_receipt) != {
            "schema",
            "kind",
            "campaign_id",
            "generation_id",
            "attempt_id",
            "attempt_spec_sha256",
            "thread_id",
            "turn_id",
            "token_usage_observations",
            "pre_provider_usage_window",
            "post_provider_usage_window",
            "provider_usage_settlement",
        }:
            raise ContiguousRunnerError(
                "retained WIP provider usage receipt has extra/missing fields"
            )
        export_receipt = reopen_receipt(
            path=observed_paths["wip_export"],
            digest=wip.wip_export_receipt_sha256,
            kind="contiguous_wip_export",
        )
        final_binding = reopen_receipt(
            path=observed_paths["final_binding"],
            digest=wip.final_thread_binding_sha256,
            kind="contiguous_final_thread_binding",
        )
        if authority_handoff is not None:
            reproduction_envelope = reopen_receipt(
                path=observed_paths["supervisory_reproduction"],
                digest=str(
                    wip
                    .supervisory_native_reproduction_receipt_sha256
                ),
                kind="SUPERVISORY_NATIVE_REPRODUCTION",
            )
            if set(reproduction_envelope) != {
                "schema",
                "kind",
                "campaign_id",
                "generation_id",
                "attempt_id",
                "attempt_spec_sha256",
                "receipt",
            }:
                raise ContiguousRunnerError(
                    "retained supervisory reproduction envelope changed"
                )
            handoff = authority_handoff.output.supervisory_handoff
            assert handoff is not None
            try:
                reproduction = (
                    Scheduler
                    .supervisory_native_reproduction_from_dict(
                        reproduction_envelope["receipt"],
                        handoff=handoff,
                    )
                )
            except Scheduler.SchedulerError as exc:
                raise ContiguousRunnerError(
                    "retained supervisory reproduction is invalid"
                ) from exc
            if (
                reproduction.native_attempt_id
                != authority_spec.attempt_id
                or reproduction.native_attempt_spec_sha256
                != proposer_attempt_binding_sha256(authority_spec)
            ):
                raise ContiguousRunnerError(
                    "retained supervisory reproduction changed attempt"
                )
        try:
            retained_usage_observations = provider_receipt[
                "token_usage_observations"
            ]
            retained_pre_window = (
                Transport.provider_usage_window_from_dict(
                    provider_receipt[
                        "pre_provider_usage_window"
                    ]
                )
            )
            retained_post_window = (
                Transport.provider_usage_window_from_dict(
                    provider_receipt[
                        "post_provider_usage_window"
                    ]
                )
            )
            Transport.provider_usage_settlement_from_dict(
                provider_receipt["provider_usage_settlement"],
                pre=retained_pre_window,
                post=retained_post_window,
                token_usage_observations=retained_usage_observations,
            )
        except Exception as exc:
            raise ContiguousRunnerError(
                "retained WIP provider usage cannot be independently settled"
            ) from exc

        try:
            Contract._validate_regular_tree(
                observed_paths["wip_root"], label="retained WIP"
            )
            ProductionInputBundleBuilder._validate_parent_source(
                observed_paths["solver_source"],
                expected_tree_sha256=wip.solver_source_tree_sha256,
                label="retained WIP solver source",
            )
            state_inventory = Transport.inventory_controller_state(
                observed_paths["state"],
                sentinels=self._secret_sentinels,
            )
            controller_scan = Taint.scan_controller_state(
                observed_paths["state"],
                inventory=state_inventory,
                canaries=self._controller_state_canaries,
            )
            retained_scan = Taint.scan_retained_canary_roots(
                {
                    "host_evidence": host_root,
                    "proposer_output": output_root,
                },
                canaries=self._controller_state_canaries,
            )
        except Exception as exc:
            raise ContiguousRunnerError(
                "retained WIP trees or scans failed independent replay"
            ) from exc
        if (
            Contract._tree_hash(observed_paths["wip_root"])
            != wip.wip_tree_sha256
            or not {
                entry.name
                for entry in observed_paths["wip_root"].iterdir()
            }.issubset({"solver_source", "context"})
            or "solver_source"
            not in {
                entry.name
                for entry in observed_paths["wip_root"].iterdir()
            }
            or state_inventory.tree_sha256
            != wip.app_server_state_tree_sha256
            or controller_scan.status != "CLEAN"
            or controller_scan.hits
            or controller_scan.canary_occurrences
            or retained_scan.status != "CLEAN"
            or retained_scan.hits
            or retained_scan.canary_occurrences
            or controller_scan_receipt.get("scanner_source_sha256")
            != Taint.source_sha256()
            or controller_scan_receipt.get("controller_state_scan")
            != controller_scan.as_receipt()
            or retained_receipt.get("scanner_source_sha256")
            != Taint.source_sha256()
            or retained_receipt.get("retained_canary_scan")
            != retained_scan.as_receipt()
            or retained_receipt.get(
                "controller_state_scan_receipt_sha256"
            )
            != wip.controller_state_scan_receipt_sha256
            or taint_receipt.get("status") != "CLEAN"
            or taint_receipt.get("hits") != []
            or final_chain.get("thread_id") != wip.codex_thread_id
            or final_chain.get("chain_head_sha256")
            != wip.transcript_chain_sha256
            or token_receipt.get("thread_id") != wip.codex_thread_id
            or provider_receipt.get("thread_id")
            != wip.codex_thread_id
            or provider_receipt.get("token_usage_observations")
            != token_receipt.get("observations")
            or export_receipt.get("outcome") != "wip"
            or export_receipt.get("wip_root_path")
            != wip.wip_root_path
            or export_receipt.get("wip_tree_sha256")
            != wip.wip_tree_sha256
            or export_receipt.get("solver_source_path")
            != wip.solver_source_path
            or export_receipt.get("solver_source_tree_sha256")
            != wip.solver_source_tree_sha256
        ):
            raise ContiguousRunnerError(
                "retained WIP evidence differs from its scans/source"
            )
        final_binding_expectations = {
            "thread_id": wip.codex_thread_id,
            "transcript_chain_sha256": wip.transcript_chain_sha256,
            "final_transcript_chain_receipt_sha256":
                wip.final_transcript_chain_receipt_sha256,
            "token_usage_receipt_sha256":
                wip.token_usage_receipt_sha256,
            "provider_usage_receipt_sha256":
                wip.provider_usage_receipt_sha256,
            "app_server_state_tree_sha256":
                wip.app_server_state_tree_sha256,
            "controller_state_scan_receipt_sha256":
                wip.controller_state_scan_receipt_sha256,
            "retained_canary_scan_receipt_sha256":
                wip.retained_canary_scan_receipt_sha256,
            "taint_scan_receipt_sha256":
                wip.taint_scan_receipt_sha256,
            "wip_export_receipt_sha256":
                wip.wip_export_receipt_sha256,
        }
        if any(
            final_binding.get(key) != value
            for key, value in final_binding_expectations.items()
        ):
            raise ContiguousRunnerError(
                "retained WIP final binding omits terminal evidence"
            )
        for path, evidence_kind in (
            (
                observed_paths["final_binding"],
                "final_thread_binding",
            ),
            (
                observed_paths["wip_publication"],
                "wip_publication_receipt",
            ),
            (
                observed_paths["retained_scan"],
                "retained_canary_receipt",
            ),
        ):
            scan = Taint.scan_canaries_in_file(
                path,
                canaries=self._controller_state_canaries,
                evidence_kind=evidence_kind,
            )
            if scan.hits:
                raise ContiguousRunnerError(
                    "retained WIP terminal receipt leaks a live canary"
                )

    def _validate_selected_wip(
        self,
        spec: AttemptSpec,
        lane: dict[str, Any],
        *,
        attempts: Mapping[str, Mapping[str, Any]],
    ) -> None:
        expected_restore = should_restore_wip(lane["no_progress"])
        eligible = lane["wip"]
        if spec.wip_mode == "restore_clean_same_frontier":
            if not expected_restore or eligible is None or spec.wip != eligible:
                raise ContiguousRunnerError(
                    "attempt restores ineligible or wrong WIP"
                )
            publication = _read_json_file(
                Path(spec.wip.wip_publication_receipt_path)
            )
            origin = attempts.get(str(publication.get("attempt_id")))
            origin_spec = (
                origin.get("spec")
                if isinstance(origin, Mapping)
                else None
            )
            if (
                not isinstance(origin_spec, AttemptSpec)
                or origin.get("phase") != "CLOSED"
                or origin.get("settled_result") is None
                or origin["settled_result"].kind
                != "clean_no_progress"
                or origin["settled_result"].wip != spec.wip
                or origin_spec.generation_id
                != publication.get("generation_id")
            ):
                raise ContiguousRunnerError(
                    "restored WIP lacks its exact journaled origin"
                )
            self._validate_wip_for_spec(
                spec.wip,
                spec,
                require_current_state=False,
                receipt_spec=origin_spec,
            )
            if (
                spec.thread_mode != "resume"
                or spec.resume_thread_id
                != spec.wip.codex_thread_id
                or spec.resume_thread_binding_sha256
                != spec.wip.final_thread_binding_sha256
            ):
                raise ContiguousRunnerError(
                    "restored WIP and app-server thread mode disagree"
                )
        elif (
            spec.wip is not None
            or (expected_restore and eligible is not None)
            or spec.thread_mode != "new"
            or spec.resume_thread_id is not None
            or spec.resume_thread_binding_sha256 is not None
        ):
            raise ContiguousRunnerError(
                "exclude attempt violates eligible-WIP selection policy"
            )

    def _validate_prepared_input(
        self,
        spec: AttemptSpec,
        *,
        require_initial_workspace: bool = True,
    ) -> None:
        generation = Path(spec.generation_dir)
        input_root = Path(spec.input_dir)
        receipt_path = Path(spec.input_bundle_receipt_path)

        def pointer() -> tuple[object, ...]:
            selected: list[object] = [
                (
                    "input",
                    _regular_tree_pointer(
                        input_root,
                        maximum_entries=MAX_APP_SERVER_STATE_FILES,
                    ),
                ),
                (
                    "parent_source",
                    _regular_tree_pointer(
                        Path(spec.parent_source_path)
                    ),
                ),
                (
                    "input_receipt",
                    _regular_file_pointer(receipt_path),
                ),
                (
                    "parent_checkpoint",
                    _regular_file_pointer(
                        Path(spec.parent_checkpoint_path)
                    ),
                ),
            ]
            if require_initial_workspace:
                selected.append((
                    "workspace",
                    _regular_tree_pointer(
                        Path(spec.workspace_dir),
                        maximum_entries=MAX_APP_SERVER_STATE_FILES,
                    ),
                ))
            if spec.wip is not None:
                selected.append((
                    "selected_wip",
                    _regular_tree_pointer(
                        Path(spec.wip.wip_root_path),
                        maximum_entries=MAX_APP_SERVER_STATE_FILES,
                    ),
                ))
            return tuple(selected)

        input_pointer = pointer()
        cache_key = (
            proposer_attempt_binding_sha256(spec),
            require_initial_workspace,
            input_pointer,
        )
        cache = getattr(self, "_verified_attempt_inputs", None)
        if cache is None:
            cache = {}
            self._verified_attempt_inputs = cache
        if cache_key in cache:
            if pointer() != input_pointer:
                raise ContiguousRunnerError(
                    "attempt input changed while it was inspected"
                )
            return
        if (
            receipt_path.parent != generation
            or receipt_path.name != "input_bundle_receipt.json"
        ):
            raise ContiguousRunnerError(
                "input-bundle receipt is outside its generation"
            )
        try:
            Contract._validate_regular_tree(
                input_root, label="attempt input bundle"
            )
            Contract._validate_regular_tree(
                Path(spec.parent_source_path),
                label="lane-bound parent source",
            )
        except Exception as exc:
            raise ContiguousRunnerError(
                "attempt input bundle is not a regular tree"
            ) from exc
        if (
            Contract._tree_hash(input_root) != spec.input_tree_sha256
            or Contract._tree_hash(Path(spec.parent_source_path))
            != spec.parent_source_tree_sha256
            or Contract._tree_hash(input_root / "parent_source")
            != spec.parent_source_tree_sha256
            or _sha256_file(receipt_path)
            != spec.input_bundle_receipt_sha256
            or _sha256_file(Path(spec.parent_checkpoint_path))
            != spec.parent_checkpoint_sha256
            or _sha256_file(input_root / "checkpoint.json")
            != spec.parent_checkpoint_sha256
            or _sha256_file(Path(spec.frontier_brief_path))
            != spec.frontier_brief_sha256
            or (
                spec.supervisory_handoff is not None
                and (
                    _sha256_file(
                        Path(str(spec.supervisory_handoff_path))
                    )
                    != spec.supervisory_handoff_sha256
                    or _sha256_file(
                        Path(
                            str(
                                spec
                                .supervisory_handoff_binding_receipt_path
                            )
                        )
                    )
                    != spec
                    .supervisory_handoff_binding_receipt_sha256
                )
            )
            or _sha256_file(Path(spec.bridge_policy_path))
            != spec.bridge_policy_sha256
        ):
            raise ContiguousRunnerError(
                "attempt input bundle or receipt changed"
            )
        receipt = _read_json_file(receipt_path)
        required = {
            "schema",
            "campaign_id",
            "generation_id",
            "attempt_id",
            "game",
            "target_level",
            "frontier_sha256",
            "input_tree_sha256",
            "parent_source_tree_sha256",
            "initial_workspace_tree_sha256",
            "parent_checkpoint_sha256",
            "wip_tree_sha256",
            "wip_solver_source_tree_sha256",
            "frontier_brief_sha256",
            "bridge_policy_sha256",
            "parent_action_count",
            "remaining_action_budget",
            "fresh_prefix_required",
            "supervisory_handoff_sha256",
            "supervisory_handoff_binding_receipt_sha256",
        }
        if (
            set(receipt) != required
            or receipt["schema"] != RUNNER_SCHEMA
            or receipt["campaign_id"] != spec.campaign_id
            or receipt["generation_id"] != spec.generation_id
            or receipt["attempt_id"] != spec.attempt_id
            or receipt["game"] != spec.game
            or receipt["target_level"] != spec.target_level
            or receipt["frontier_sha256"] != spec.frontier_sha256
            or receipt["input_tree_sha256"] != spec.input_tree_sha256
            or receipt["parent_source_tree_sha256"]
            != spec.parent_source_tree_sha256
            or receipt["initial_workspace_tree_sha256"]
            != spec.initial_workspace_tree_sha256
            or receipt["parent_checkpoint_sha256"]
            != spec.parent_checkpoint_sha256
            or receipt["wip_tree_sha256"]
            != (spec.wip.wip_tree_sha256 if spec.wip else None)
            or receipt["wip_solver_source_tree_sha256"]
            != (
                spec.wip.solver_source_tree_sha256
                if spec.wip else None
            )
            or receipt["frontier_brief_sha256"]
            != spec.frontier_brief_sha256
            or receipt["bridge_policy_sha256"]
            != spec.bridge_policy_sha256
            or receipt["parent_action_count"]
            != spec.parent_action_count
            or receipt["remaining_action_budget"]
            != spec.remaining_action_budget
            or receipt["fresh_prefix_required"]
            is not spec.fresh_prefix_required
            or receipt["supervisory_handoff_sha256"]
            != spec.supervisory_handoff_sha256
            or receipt[
                "supervisory_handoff_binding_receipt_sha256"
            ]
            != spec.supervisory_handoff_binding_receipt_sha256
        ):
            raise ContiguousRunnerError(
                "input-bundle receipt does not bind the admitted lineage"
            )
        checkpoint = Contract.load_trusted_checkpoint(
            input_root / "checkpoint.json",
            expected_game=spec.game,
            authoritative_target=spec.authoritative_target,
        )
        if (
            len(checkpoint.final_path) != spec.parent_action_count
            or 600 - len(checkpoint.final_path)
            != spec.remaining_action_budget
        ):
            raise ContiguousRunnerError(
                "frontier action budget differs from the trusted parent"
            )
        parent_source = input_root / "parent_source"
        workspace_root = Path(spec.workspace_dir)
        try:
            Contract._validate_regular_tree(
                parent_source, label="parent source"
            )
            if require_initial_workspace:
                Contract._validate_regular_tree(
                    workspace_root, label="initial workspace"
                )
        except Exception as exc:
            raise ContiguousRunnerError(
                "source/workspace seed is not a regular tree"
            ) from exc
        if (
            Contract._tree_hash(parent_source)
            != spec.parent_source_tree_sha256
            or (
                require_initial_workspace
                and Contract._tree_hash(workspace_root)
                != spec.initial_workspace_tree_sha256
            )
            or not PARENT_SOURCE_REQUIRED_FILES.issubset(
                {
                    entry.name
                    for entry in parent_source.iterdir()
                    if entry.is_file()
                }
            )
        ):
            raise ContiguousRunnerError(
                "source/workspace seed changed after construction"
            )
        brief = _read_json_file(Path(spec.frontier_brief_path))
        expected_brief = {
            "schema": 1,
            "kind": "arc_agi3_contiguous_frontier_brief",
            "campaign_id": spec.campaign_id,
            "generation_id": spec.generation_id,
            "attempt_id": spec.attempt_id,
            "game": spec.game,
            "target_level": spec.target_level,
            "authoritative_target": spec.authoritative_target,
            "parent_checkpoint_sha256":
                spec.parent_checkpoint_sha256,
            "frontier_sha256": spec.frontier_sha256,
            "parent_action_count": spec.parent_action_count,
            "remaining_action_budget":
                spec.remaining_action_budget,
            "fresh_prefix_required":
                spec.fresh_prefix_required,
            "effort": spec.effort,
            "soft_allocation_seconds":
                spec.soft_allocation_seconds,
            "wip_mode": spec.wip_mode,
            "thread_mode": spec.thread_mode,
            "supervisory_handoff": (
                None
                if spec.supervisory_handoff is None
                else Scheduler.supervisory_prompt_projection(
                    spec.supervisory_handoff
                )
            ),
        }
        if brief != expected_brief:
            raise ContiguousRunnerError(
                "frontier brief is nonminimal, stale, or malformed"
            )
        policy = _read_json_file(Path(spec.bridge_policy_path))
        expected_policy = {
            "schema": 1,
            "kind": "arc_agi3_contiguous_bridge_policy",
            "campaign_id": spec.campaign_id,
            "generation_id": spec.generation_id,
            "attempt_id": spec.attempt_id,
            "game": spec.game,
            "target_level": spec.target_level,
            "frontier_sha256": spec.frontier_sha256,
            "parent_checkpoint_sha256":
                spec.parent_checkpoint_sha256,
            "protocol_version":
                spec.proposer_transport.bridge_protocol_version,
            "operation_allowlist": list(
                spec.proposer_transport
                .bridge_operation_allowlist
            ),
            "exec_allowlist": list(
                spec.proposer_transport.bridge_exec_allowlist
            ),
            "workspace_root": "/arc/workspace",
            "export_root": "/arc/export",
            "bounds": {
                "max_request_bytes":
                    spec.proposer_transport
                    .bridge_max_request_bytes,
                "max_response_bytes":
                    spec.proposer_transport
                    .bridge_max_response_bytes,
                "max_file_bytes":
                    spec.proposer_transport.bridge_max_file_bytes,
                "max_total_export_bytes":
                    spec.proposer_transport
                    .bridge_max_total_export_bytes,
                "max_processes":
                    spec.proposer_transport.bridge_max_processes,
                "max_exec_seconds":
                    spec.proposer_transport
                    .bridge_max_exec_seconds,
            },
        }
        if policy != expected_policy:
            raise ContiguousRunnerError(
                "lane bridge policy differs from the frozen projection"
            )
        if spec.supervisory_handoff is not None:
            binding = spec.supervisory_handoff
            handoff_document = _read_json_file(
                Path(str(spec.supervisory_handoff_path))
            )
            binding_receipt = _read_json_file(
                Path(
                    str(
                        spec
                        .supervisory_handoff_binding_receipt_path
                    )
                )
            )
            expected_handoff = (
                Scheduler.supervisory_prompt_projection(binding)
            )
            expected_binding_receipt = {
                "schema": 1,
                "kind":
                    "contiguous_supervisory_handoff_prompt_binding",
                "campaign_id": spec.campaign_id,
                "generation_id": spec.generation_id,
                "attempt_id": spec.attempt_id,
                "game": spec.game,
                "frontier_sha256": spec.frontier_sha256,
                "parent_checkpoint_sha256":
                    spec.parent_checkpoint_sha256,
                "assignment_id": binding.assignment_id,
                "output_manifest_sha256":
                    binding.output_manifest_sha256,
                "supervisory_handoff_sha256":
                    binding.supervisory_handoff_sha256,
                "handoff_file_sha256":
                    spec.supervisory_handoff_sha256,
                "admission_receipt_sha256":
                    binding.admission_receipt_sha256,
                "prompt_authority":
                    "unverified_hypothesis_only",
                "native_reproduction_required_before_wip_candidate_or_promotion":
                    True,
                "scheduler_authority": False,
                "mutation_authority": False,
                "promotion_authority": False,
            }
            if (
                handoff_document != expected_handoff
                or binding_receipt != expected_binding_receipt
            ):
                raise ContiguousRunnerError(
                    "supervisory handoff prompt binding is substituted"
                )
        elif (
            (input_root / "supervisory_handoff.json").exists()
            or (
                input_root
                / "supervisory_handoff_binding_receipt.json"
            ).exists()
        ):
            raise ContiguousRunnerError(
                "attempt without a handoff contains supervisory input"
            )
        if spec.wip is not None:
            wip_root = Path(spec.wip.wip_root_path)
            solver_source_root = Path(spec.wip.solver_source_path)
            bundled_wip = input_root / "wip"
            try:
                Contract._validate_regular_tree(
                    wip_root, label="selected WIP"
                )
                Contract._validate_regular_tree(
                    bundled_wip, label="bundled WIP"
                )
                ProductionInputBundleBuilder._validate_parent_source(
                    solver_source_root,
                    expected_tree_sha256=(
                        spec.wip.solver_source_tree_sha256
                    ),
                    label="selected WIP solver source",
                )
                ProductionInputBundleBuilder._validate_parent_source(
                    bundled_wip / "solver_source",
                    expected_tree_sha256=(
                        spec.wip.solver_source_tree_sha256
                    ),
                    label="bundled WIP solver source",
                )
            except Exception as exc:
                raise ContiguousRunnerError(
                    "selected WIP is not a regular tree"
                ) from exc
            if (
                Contract._tree_hash(wip_root)
                != spec.wip.wip_tree_sha256
                or Contract._tree_hash(bundled_wip)
                != spec.wip.wip_tree_sha256
            ):
                raise ContiguousRunnerError(
                    "selected WIP changed before/recovery of launch"
                )
        elif (input_root / "wip").exists():
            raise ContiguousRunnerError(
                "no-WIP attempt contains an undeclared WIP tree"
            )
        if pointer() != input_pointer:
            raise ContiguousRunnerError(
                "attempt input changed while it was inspected"
            )
        self._remember_verified_pointer(cache, cache_key)

    @staticmethod
    def _validate_supervisory_reproduction_gate(
        spec: AttemptSpec,
        collection: BackendCollection,
    ) -> Scheduler.SupervisoryNativeReproductionReceipt | None:
        """Reopen the host receipt before any exposed turn gains authority."""

        handoff_binding = spec.supervisory_handoff
        path_value = (
            collection
            .supervisory_native_reproduction_receipt_path
        )
        digest_value = (
            collection
            .supervisory_native_reproduction_receipt_sha256
        )
        has_derived_artifact = (
            collection.result.wip is not None
            or collection.result.candidate is not None
        )
        if handoff_binding is None:
            if path_value is not None or digest_value is not None:
                raise ContiguousRunnerError(
                    "unexposed attempt carries a supervisory reproduction"
                )
            return None
        handoff = handoff_binding.output.supervisory_handoff
        if handoff is None:
            raise ContiguousRunnerError(
                "supervisory prompt binding lost its typed handoff"
            )
        if path_value is None or digest_value is None:
            if has_derived_artifact:
                raise ContiguousRunnerError(
                    "handoff-exposed turn produced WIP/candidate without "
                    "native public reproduction"
                )
            return None
        host_root = Path(spec.host_transcript_path).parent
        envelope = _validate_bound_receipt(
            path_value,
            digest_value,
            expected_path=(
                host_root
                / "supervisory_native_reproduction_receipt.json"
            ),
            expected_kind="SUPERVISORY_NATIVE_REPRODUCTION",
            spec=spec,
        )
        expected_envelope_keys = {
            "schema",
            "kind",
            "campaign_id",
            "generation_id",
            "attempt_id",
            "attempt_spec_sha256",
            "receipt",
        }
        if set(envelope) != expected_envelope_keys:
            raise ContiguousRunnerError(
                "supervisory reproduction envelope has extra/missing fields"
            )
        try:
            receipt = (
                Scheduler.supervisory_native_reproduction_from_dict(
                    envelope["receipt"],
                    handoff=handoff,
                )
            )
        except Scheduler.SchedulerError as exc:
            raise ContiguousRunnerError(
                "supervisory native reproduction receipt is invalid"
            ) from exc
        try:
            import arc_agi3_arena_rpc as arena_contract
        except ImportError as exc:
            raise ContiguousRunnerError(
                "public observation schema is unavailable"
            ) from exc
        campaign_root = Path(spec.generation_dir).parent.parent
        if Path(spec.generation_dir).parent.name != "generations":
            raise ContiguousRunnerError(
                "supervisory reproduction has no campaign registry"
            )
        source_root = campaign_root / "public_observation_registry"
        native_root = host_root / "public_observations"
        arena_binding_path = (
            host_root / "arena_session_binding_receipt.json"
        )
        if (
            receipt.native_attempt_id != spec.attempt_id
            or receipt.native_attempt_spec_sha256
            != proposer_attempt_binding_sha256(spec)
            or receipt.frontier_sha256 != spec.frontier_sha256
            or receipt.parent_checkpoint_sha256
            != spec.parent_checkpoint_sha256
            or receipt.supervisory_handoff_sha256
            != handoff_binding.supervisory_handoff_sha256
            or receipt.native_host_transcript_sha256
            != collection.host_transcript_sha256
            or _sha256_file(arena_binding_path)
            != receipt
            .native_arena_session_binding_receipt_sha256
        ):
            raise ContiguousRunnerError(
                "supervisory reproduction is not exact native evidence"
            )
        native_semantics = set(
            collection.native_public_observation_receipt_sha256s
        )
        for row in receipt.reproductions:
            source_path = source_root / (
                f"{row.source_observation_receipt_sha256}.json"
            )
            native_path = native_root / (
                f"{row.native_observation_receipt_sha256}.json"
            )
            try:
                if (
                    _sha256_file(source_path)
                    != row.source_observation_receipt_sha256
                    or _sha256_file(native_path)
                    != row.native_observation_receipt_sha256
                    or row.native_observation_receipt_sha256
                    not in native_semantics
                ):
                    raise ContiguousRunnerError(
                        "supervisory reproduction receipt file changed"
                    )
                source_observation = _read_json_file(source_path)
                native_observation = _read_json_file(native_path)
                source_semantic = (
                    arena_contract
                    .validate_public_observation_receipt(
                        source_observation,
                        game=spec.game,
                        frontier_sha256=spec.frontier_sha256,
                        parent_checkpoint_sha256=(
                            spec.parent_checkpoint_sha256
                        ),
                    )
                )
                native_semantic = (
                    arena_contract
                    .validate_public_observation_receipt(
                        native_observation,
                        game=spec.game,
                        frontier_sha256=spec.frontier_sha256,
                        parent_checkpoint_sha256=(
                            spec.parent_checkpoint_sha256
                        ),
                    )
                )
            except ContiguousRunnerError:
                raise
            except Exception as exc:
                raise ContiguousRunnerError(
                    "supervisory reproduction content is malformed"
                ) from exc
            if (
                source_semantic
                != row.source_observation_receipt_sha256
                or native_semantic
                != row.native_observation_receipt_sha256
                or source_observation[
                    "public_action_basis_sha256"
                ]
                != row.public_action_basis_sha256
                or native_observation[
                    "public_action_basis_sha256"
                ]
                != row.public_action_basis_sha256
                or source_observation[
                    "public_response_signature_sha256"
                ]
                != row.public_response_signature_sha256
                or native_observation[
                    "public_response_signature_sha256"
                ]
                != row.public_response_signature_sha256
                or source_observation["public_action_basis"]
                != native_observation["public_action_basis"]
                or source_observation["public_response_signature"]
                != native_observation["public_response_signature"]
            ):
                raise ContiguousRunnerError(
                    "supervisory reproduction commitments differ"
                )
        if collection.result.candidate is not None:
            candidate = collection.result.candidate
            if (
                candidate.supervisory_handoff_sha256
                != handoff_binding.supervisory_handoff_sha256
                or candidate
                .supervisory_native_reproduction_receipt_sha256
                != digest_value
            ):
                raise ContiguousRunnerError(
                    "candidate substitutes supervisory reproduction"
                )
        if collection.result.wip is not None:
            wip = collection.result.wip
            if (
                wip.supervisory_handoff_sha256
                != handoff_binding.supervisory_handoff_sha256
                or wip
                .supervisory_native_reproduction_receipt_path
                != path_value
                or wip
                .supervisory_native_reproduction_receipt_sha256
                != digest_value
            ):
                raise ContiguousRunnerError(
                    "WIP substitutes supervisory reproduction"
                )
        return receipt

    @staticmethod
    def _validate_native_public_observation_receipts(
        spec: AttemptSpec,
        collection: BackendCollection,
    ) -> None:
        """Reopen content receipts and match them to applied host RPC events."""

        try:
            import arc_agi3_arena_rpc as arena_contract
        except ImportError as exc:
            raise ContiguousRunnerError(
                "public observation schema is unavailable"
            ) from exc
        host_root = Path(spec.host_transcript_path).parent
        receipt_root = host_root / "public_observations"
        try:
            metadata = receipt_root.lstat()
        except OSError as exc:
            raise ContiguousRunnerError(
                "public observation receipt directory is missing"
            ) from exc
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or stat.S_ISLNK(metadata.st_mode)
            or metadata.st_nlink < 1
        ):
            raise ContiguousRunnerError(
                "public observation receipt root is aliased"
            )
        expected_names = {
            f"{digest}.json"
            for digest in (
                collection.native_public_observation_receipt_sha256s
            )
        }
        observed_names = {
            path.name for path in receipt_root.iterdir()
        }
        if observed_names != expected_names:
            raise ContiguousRunnerError(
                "public observation receipt inventory differs"
            )
        receipts_by_index: dict[
            int, tuple[str, Mapping[str, Any]]
        ] = {}
        for digest in (
            collection.native_public_observation_receipt_sha256s
        ):
            path = receipt_root / f"{digest}.json"
            if _sha256_file(path) != digest:
                raise ContiguousRunnerError(
                    "public observation receipt is not content-addressed"
                )
            value = _read_json_file(path)
            try:
                observed_digest = (
                    arena_contract
                    .validate_public_observation_receipt(
                        value,
                        game=spec.game,
                        frontier_sha256=spec.frontier_sha256,
                        parent_checkpoint_sha256=(
                            spec.parent_checkpoint_sha256
                        ),
                    )
                )
            except Exception as exc:
                raise ContiguousRunnerError(
                    "public observation receipt is malformed"
                ) from exc
            if observed_digest != digest:
                raise ContiguousRunnerError(
                    "public observation semantic identity changed"
                )
            index = value["public_action_basis"]["operation_index"]
            if index in receipts_by_index:
                raise ContiguousRunnerError(
                    "public observation operation index was duplicated"
                )
            receipts_by_index[index] = (digest, value)
        ordered = tuple(
            receipts_by_index[index]
            for index in sorted(receipts_by_index)
        )
        prior_basis = (
            arena_contract.PUBLIC_ACTION_BASIS_GENESIS_SHA256
        )
        for expected_index, (_digest, value) in enumerate(ordered):
            basis = value["public_action_basis"]
            if (
                basis["operation_index"] != expected_index
                or basis["previous_public_action_basis_sha256"]
                != prior_basis
            ):
                raise ContiguousRunnerError(
                    "public observation action basis is discontinuous"
                )
            prior_basis = value["public_action_basis_sha256"]
        if (
            collection.result.kind
            in {"clean_no_progress", "candidate"}
            and not ordered
        ):
            raise ContiguousRunnerError(
                "admissible native result has no public observations"
            )
        try:
            transcript_raw = _bounded_regular_bytes(
                Path(collection.host_transcript_path),
                maximum=MAX_HOST_TRANSCRIPT_BYTES,
            )
            transcript_events = tuple(
                json.loads(line)
                for line in transcript_raw.decode("utf-8").splitlines()
            )
        except Exception as exc:
            raise ContiguousRunnerError(
                "host RPC transcript cannot be reopened"
            ) from exc
        logged_commitments: set[tuple[str, str, str]] = set()
        for event in transcript_events:
            if (
                isinstance(event, dict)
                and event.get("kind") == "rpc"
                and event.get("phase") == "applied"
                and "public_observation_receipt_sha256" in event
            ):
                commitment = (
                    event.get("public_observation_receipt_sha256"),
                    event.get("public_action_basis_sha256"),
                    event.get("public_response_signature_sha256"),
                )
                if any(not _is_sha256(item) for item in commitment):
                    raise ContiguousRunnerError(
                        "host RPC public commitment is malformed"
                    )
                logged_commitments.add(commitment)
        expected_commitments = {
            (
                digest,
                value["public_action_basis_sha256"],
                value["public_response_signature_sha256"],
            )
            for digest, value in ordered
        }
        if logged_commitments != expected_commitments:
            raise ContiguousRunnerError(
                "public receipts do not match logged native observations"
            )

    def _register_native_public_observation_receipts(
        self,
        spec: AttemptSpec,
        collection: BackendCollection,
    ) -> None:
        """Install validated semantic receipts in the host-only campaign map."""

        expected_campaign_root = (
            Path(spec.generation_dir).parent.parent
        )
        if (
            expected_campaign_root != self.root
            or Path(spec.generation_dir).parent != self.generations
        ):
            raise ContiguousRunnerError(
                "attempt generation cannot locate its host receipt registry"
            )
        source_root = (
            Path(spec.host_transcript_path).parent
            / "public_observations"
        )
        for digest in (
            collection.native_public_observation_receipt_sha256s
        ):
            source = source_root / f"{digest}.json"
            raw = _bounded_regular_bytes(
                source, maximum=MAX_HOST_TRANSCRIPT_BYTES
            )
            if hashlib.sha256(raw).hexdigest() != digest:
                raise ContiguousRunnerError(
                    "validated public receipt changed before registration"
                )
            target = self.public_observation_registry / (
                f"{digest}.json"
            )
            _install_regular_bytes(target, raw)
            os.chmod(target, 0o400, follow_symlinks=False)
            if _sha256_file(target) != digest:
                raise ContiguousRunnerError(
                    "host public-observation registry changed content"
                )
        _fsync_directory(self.public_observation_registry)

    def _validate_public_observation_registry(
        self,
        attempts: Mapping[str, Mapping[str, Any]],
    ) -> str:
        """Reconstruct the host registry exactly from journaled collections."""

        try:
            import arc_agi3_arena_rpc as arena_contract
        except ImportError as exc:
            raise ContiguousRunnerError(
                "public observation schema is unavailable"
            ) from exc
        root = self.public_observation_registry
        try:
            root_metadata = root.stat(follow_symlinks=False)
        except OSError as exc:
            raise ContiguousRunnerError(
                "public observation registry is unavailable"
            ) from exc
        if (
            root.is_symlink()
            or not stat.S_ISDIR(root_metadata.st_mode)
            or root_metadata.st_uid != os.getuid()
            or stat.S_IMODE(root_metadata.st_mode) != 0o700
        ):
            raise ContiguousRunnerError(
                "public observation registry root is unsafe"
            )
        expected: dict[str, AttemptSpec] = {}
        staged: dict[str, AttemptSpec] = {}
        for attempt in attempts.values():
            collection = attempt.get("collection")
            spec = attempt.get("spec")
            transition = attempt.get(
                "public_observation_transition"
            )
            if not isinstance(spec, AttemptSpec):
                continue
            if (
                isinstance(collection, BackendCollection)
                and collection.result.kind
                in {"clean_no_progress", "candidate"}
            ):
                selected_receipts = (
                    collection
                    .native_public_observation_receipt_sha256s
                )
                selected_registry = expected
            elif (
                isinstance(transition, dict)
                and transition.get("authority")
                == "same_frontier_lineage"
                and isinstance(
                    transition.get("receipt_sha256s"), list
                )
            ):
                selected_receipts = tuple(
                    transition["receipt_sha256s"]
                )
                selected_registry = staged
            else:
                continue
            for digest in selected_receipts:
                prior = (
                    expected.get(digest)
                    or staged.get(digest)
                    or spec
                )
                if (
                    prior.game != spec.game
                    or prior.frontier_sha256 != spec.frontier_sha256
                    or prior.parent_checkpoint_sha256
                    != spec.parent_checkpoint_sha256
                ):
                    raise ContiguousRunnerError(
                        "one public receipt is claimed across frontiers"
                    )
                selected_registry.setdefault(digest, spec)
        observed_names = set(os.listdir(root))
        expected_names = {
            f"{digest}.json" for digest in expected
        }
        staged_names = {
            f"{digest}.json" for digest in staged
        }
        if (
            not expected_names <= observed_names
            or not observed_names <= expected_names | staged_names
        ):
            raise ContiguousRunnerError(
                "public observation registry differs from journal history"
            )
        inventory: list[dict[str, Any]] = []
        admissible = {**staged, **expected}
        observed_digests = {
            name.removesuffix(".json")
            for name in observed_names
        }
        for digest, spec in sorted(admissible.items()):
            if digest not in observed_digests:
                continue
            path = root / f"{digest}.json"
            try:
                observed_digest, metadata = _sha256_file_identity(path)
                value = _read_json_file(path)
                semantic_digest = (
                    arena_contract.validate_public_observation_receipt(
                        value,
                        game=spec.game,
                        frontier_sha256=spec.frontier_sha256,
                        parent_checkpoint_sha256=(
                            spec.parent_checkpoint_sha256
                        ),
                    )
                )
            except Exception as exc:
                raise ContiguousRunnerError(
                    "public observation registry entry is malformed"
                ) from exc
            if (
                observed_digest != digest
                or semantic_digest != digest
                or metadata.st_uid != os.getuid()
                or metadata.st_nlink != 1
                or not stat.S_ISREG(metadata.st_mode)
                or (
                    digest in expected
                    and stat.S_IMODE(metadata.st_mode) != 0o400
                )
                or (
                    digest not in expected
                    and stat.S_IMODE(metadata.st_mode)
                    not in {0o400, 0o600}
                )
            ):
                raise ContiguousRunnerError(
                    "public observation registry entry is aliased or "
                    "substituted"
                )
            inventory.append(
                {
                    "sha256": digest,
                    "game": spec.game,
                    "frontier_sha256": spec.frontier_sha256,
                    "parent_checkpoint_sha256":
                        spec.parent_checkpoint_sha256,
                }
            )
        return Scheduler.sha256_json(
            {
                "schema": 1,
                "kind":
                    "arc_agi3_public_observation_registry_inventory",
                "entries": inventory,
            }
        )

    def _validate_collection(
        self,
        spec: AttemptSpec,
        prepared: BackendPreparation,
        launched: BackendLaunch,
        collection: BackendCollection,
        *,
        allow_arena_teardown_receipt: bool = False,
    ) -> None:
        if collection.host_transcript_path != spec.host_transcript_path:
            raise ContiguousRunnerError(
                "backend collection references the wrong host transcript"
            )
        self._validate_native_public_observation_receipts(
            spec, collection
        )
        self._validate_supervisory_reproduction_gate(
            spec, collection
        )
        if (
            collection.app_server_transcript_path
            != spec.app_server_transcript_path
            or collection.codex_thread_id
            != launched.codex_thread_id
            or collection.codex_turn_id != launched.codex_turn_id
        ):
            raise ContiguousRunnerError(
                "collection references another app-server turn"
            )
        transcript = Path(collection.host_transcript_path)
        app_transcript = Path(collection.app_server_transcript_path)
        host_root = transcript.parent
        stdout_path = Path(collection.container_stdout_path)
        stderr_path = Path(collection.container_stderr_path)
        worker_outcome = Path(spec.output_dir) / WORKER_OUTCOME_NAME
        try:
            Contract._validate_regular_tree(
                Path(spec.output_dir), label="attempt output"
            )
            Contract._validate_candidate_output_quota(
                Path(spec.output_dir)
            )
        except Exception as exc:
            raise ContiguousRunnerError(
                "attempt output is not a regular tree"
            ) from exc
        try:
            digest_mismatch = (
                transcript.stat(follow_symlinks=False).st_size
                > MAX_HOST_TRANSCRIPT_BYTES
                or app_transcript.stat(
                    follow_symlinks=False
                ).st_size > MAX_HOST_TRANSCRIPT_BYTES
                or stdout_path.stat(
                    follow_symlinks=False
                ).st_size > MAX_HOST_TRANSCRIPT_BYTES
                or stderr_path.stat(
                    follow_symlinks=False
                ).st_size > MAX_HOST_TRANSCRIPT_BYTES
                or _sha256_file(transcript)
                != collection.host_transcript_sha256
                or _sha256_file(app_transcript)
                != collection.app_server_transcript_sha256
                or _sha256_file(stdout_path)
                != collection.container_stdout_sha256
                or _sha256_file(stderr_path)
                != collection.container_stderr_sha256
                or _sha256_file(worker_outcome)
                != collection.worker_outcome_sha256
                or Contract._tree_hash(Path(spec.output_dir))
                != collection.output_tree_sha256
            )
        except Exception as exc:
            raise ContiguousRunnerError(
                "collected output/transcript is missing or aliased"
            ) from exc
        if digest_mismatch:
            raise ContiguousRunnerError(
                "collected output/transcript digest mismatch"
            )
        token_usage = _validate_bound_receipt(
            collection.token_usage_receipt_path,
            collection.token_usage_receipt_sha256,
            expected_path=host_root / "token_usage_receipt.json",
            expected_kind="contiguous_token_usage",
            spec=spec,
        )
        usage_observations = token_usage.get("observations")
        required_usage_fields = {
            "inputTokens",
            "cachedInputTokens",
            "outputTokens",
            "reasoningOutputTokens",
            "totalTokens",
        }
        prior_total = -1
        if (
            token_usage.get("thread_id") != launched.codex_thread_id
            or token_usage.get("turn_id") != launched.codex_turn_id
            or token_usage.get("final_event_observed") is not True
            or token_usage.get("wrong_identity_events") != 0
            or token_usage.get("duplicate_events") != 0
            or not isinstance(usage_observations, list)
            or not usage_observations
            or token_usage.get("hard_safety_seconds")
            != spec.hard_safety_seconds
            or token_usage.get("max_auth_refreshes")
            != spec.max_auth_refreshes
            or not isinstance(
                token_usage.get("auth_refresh_count"), int
            )
            or isinstance(
                token_usage.get("auth_refresh_count"), bool
            )
            or not 0
            <= token_usage["auth_refresh_count"]
            <= spec.max_auth_refreshes
            or token_usage.get(
                "credential_sentinel_scan_passed"
            )
            is not True
            or not isinstance(
                token_usage.get(
                    "redacted_auth_refresh_response_sha256"
                ),
                list,
            )
            or len(
                token_usage[
                    "redacted_auth_refresh_response_sha256"
                ]
            )
            != token_usage["auth_refresh_count"]
            or any(
                not _is_sha256(value)
                for value in token_usage[
                    "redacted_auth_refresh_response_sha256"
                ]
            )
            or token_usage.get("pipes_drained_to_eof") is not True
            or not isinstance(
                token_usage.get("post_turn_event_count"), int
            )
            or isinstance(
                token_usage.get("post_turn_event_count"), bool
            )
            or token_usage["post_turn_event_count"] < 0
        ):
            raise ContiguousRunnerError(
                "structured token-usage receipt is incomplete"
            )
        for observation in usage_observations:
            totals = (
                observation.get("total")
                if isinstance(observation, dict)
                else None
            )
            if (
                not isinstance(observation, dict)
                or observation.get("threadId")
                != launched.codex_thread_id
                or observation.get("turnId")
                != launched.codex_turn_id
                or not isinstance(totals, dict)
                or set(totals) != required_usage_fields
                or any(
                    not isinstance(value, int)
                    or isinstance(value, bool)
                    or value < 0
                    for value in totals.values()
                )
                or totals["totalTokens"] < prior_total
            ):
                raise ContiguousRunnerError(
                    "structured token usage is mismatched or nonmonotone"
                )
            prior_total = totals["totalTokens"]
        provider_usage = _validate_bound_receipt(
            collection.provider_usage_receipt_path,
            collection.provider_usage_receipt_sha256,
            expected_path=host_root / "provider_usage_receipt.json",
            expected_kind="contiguous_provider_usage",
            spec=spec,
        )
        if set(provider_usage) != {
            "schema",
            "kind",
            "campaign_id",
            "generation_id",
            "attempt_id",
            "attempt_spec_sha256",
            "thread_id",
            "turn_id",
            "token_usage_observations",
            "pre_provider_usage_window",
            "post_provider_usage_window",
            "provider_usage_settlement",
        }:
            raise ContiguousRunnerError(
                "typed provider usage receipt has extra/missing fields"
            )
        try:
            pre_usage_window = (
                Transport.provider_usage_window_from_dict(
                    provider_usage.get(
                        "pre_provider_usage_window"
                    )
                )
            )
            post_usage_window = (
                Transport.provider_usage_window_from_dict(
                    provider_usage.get(
                        "post_provider_usage_window"
                    )
                )
            )
            usage_settlement = (
                Transport.provider_usage_settlement_from_dict(
                    provider_usage.get(
                        "provider_usage_settlement"
                    ),
                    pre=pre_usage_window,
                    post=post_usage_window,
                    token_usage_observations=usage_observations,
                )
            )
        except Exception as exc:
            raise ContiguousRunnerError(
                "typed provider usage receipt failed independent settlement"
            ) from exc
        if (
            provider_usage.get("thread_id")
            != launched.codex_thread_id
            or provider_usage.get("turn_id") != launched.codex_turn_id
            or provider_usage.get("token_usage_observations")
            != usage_observations
            or usage_settlement.charge
            != collection.result.cost_used
            or (
                not usage_settlement.cost_control_enabled
                and (
                    usage_settlement.limit is not None
                    or spec.cost_limit_remaining is not None
                )
            )
            or (
                usage_settlement.cost_control_enabled
                and spec.cost_limit_remaining is None
            )
        ):
            raise ContiguousRunnerError(
                "provider settlement and admitted/result cost disagree"
            )
        final_chain = _validate_bound_receipt(
            collection.final_transcript_chain_receipt_path,
            collection.final_transcript_chain_receipt_sha256,
            expected_path=host_root
            / "final_transcript_chain_receipt.json",
            expected_kind="contiguous_final_transcript_chain",
            spec=spec,
        )
        if (
            final_chain.get("thread_id") != launched.codex_thread_id
            or final_chain.get("turn_id") != launched.codex_turn_id
            or final_chain.get("chain_head_sha256")
            != collection.final_transcript_chain_sha256
            or final_chain.get("raw_transcript_sha256")
            != collection.app_server_transcript_sha256
            or not isinstance(final_chain.get("event_count"), int)
            or isinstance(final_chain.get("event_count"), bool)
            or final_chain["event_count"] <= 0
        ):
            raise ContiguousRunnerError(
                "final app-server transcript chain is incomplete"
            )
        export_receipt = _validate_bound_receipt(
            collection.bridge_export_receipt_path,
            collection.bridge_export_receipt_sha256,
            expected_path=host_root / "bridge_export_receipt.json",
            expected_kind="contiguous_bridge_export",
            spec=spec,
        )
        if (
            export_receipt.get("container_id")
            != launched.container_id
            or export_receipt.get(
                "bridge_runtime_attestation_sha256"
            )
            != launched.bridge_runtime_attestation_sha256
            or export_receipt.get("output_tree_sha256")
            != collection.output_tree_sha256
            or export_receipt.get("model_final_text_eligible")
            is not False
            or export_receipt.get("outcome")
            != collection.result.kind
            or export_receipt.get("host_blocker_code")
            != (
                collection.result.blocker.code
                if collection.result.blocker is not None
                else None
            )
            or export_receipt.get("host_blocker_receipt_sha256")
            != (
                collection.result.blocker.receipt_sha256
                if collection.result.blocker is not None
                else None
            )
        ):
            raise ContiguousRunnerError(
                "bridge export receipt is missing or substituted"
            )
        target_boundary_receipt: dict[str, Any] | None = None
        if collection.target_boundary_receipt_path is not None:
            assert (
                collection.target_boundary_receipt_sha256 is not None
                and collection.target_boundary_sha256 is not None
                and collection.target_boundary_workspace_tree_sha256
                is not None
            )
            target_boundary_receipt = _validate_bound_receipt(
                collection.target_boundary_receipt_path,
                collection.target_boundary_receipt_sha256,
                expected_path=host_root
                / "target_boundary_receipt.json",
                expected_kind="contiguous_target_boundary",
                spec=spec,
            )
            boundary_request = target_boundary_receipt.get(
                "bridge_request"
            )
            boundary_response = target_boundary_receipt.get(
                "bridge_response"
            )
            boundary = target_boundary_receipt.get("boundary")
            request_sha256 = hashlib.sha256(
                _canonical_json(boundary_request)
            ).hexdigest()
            response_sha256 = hashlib.sha256(
                _canonical_json(boundary_response)
            ).hexdigest()
            try:
                validated_boundary_sha256 = (
                    Transport._validate_target_boundary_result(
                        (
                            boundary_response.get("result")
                            if isinstance(
                                boundary_response, dict
                            )
                            else None
                        ),
                        attempt_id=spec.attempt_id,
                        request=(
                            boundary_request
                            if isinstance(boundary_request, dict)
                            else {}
                        ),
                        target_level=spec.target_level,
                    )
                )
                boundary_inventory = (
                    Transport.inventory_controller_state(
                        host_root / "target_boundary_workspace"
                    )
                )
            except Exception as exc:
                raise ContiguousRunnerError(
                    "target boundary failed independent replay"
                ) from exc
            if (
                boundary != boundary_response.get("result", {}).get(
                    "boundary"
                )
                or validated_boundary_sha256
                != collection.target_boundary_sha256
                or target_boundary_receipt.get("boundary_sha256")
                != collection.target_boundary_sha256
                or target_boundary_receipt.get(
                    "bridge_request_sha256"
                )
                != request_sha256
                or target_boundary_receipt.get(
                    "bridge_response_sha256"
                )
                != response_sha256
                or target_boundary_receipt.get("snapshot_root")
                != str(host_root / "target_boundary_workspace")
                or target_boundary_receipt.get(
                    "workspace_inventory"
                )
                != boundary_inventory.as_receipt()
                or boundary_inventory.tree_sha256
                != collection
                .target_boundary_workspace_tree_sha256
                or target_boundary_receipt.get(
                    "pre_response_delivery"
                )
                is not True
                or target_boundary_receipt.get(
                    "next_level_observation_withheld"
                )
                is not True
                or target_boundary_receipt.get("workspace_frozen")
                is not True
                or export_receipt.get(
                    "target_boundary_receipt_sha256"
                )
                != collection.target_boundary_receipt_sha256
                or export_receipt.get("target_boundary_sha256")
                != collection.target_boundary_sha256
                or export_receipt.get(
                    "target_boundary_workspace_tree_sha256"
                )
                != collection
                .target_boundary_workspace_tree_sha256
            ):
                raise ContiguousRunnerError(
                    "target boundary receipt or snapshot differs"
                )
        elif any(
            export_receipt.get(name) is not None
            for name in (
                "target_boundary_receipt_sha256",
                "target_boundary_sha256",
                "target_boundary_workspace_tree_sha256",
            )
        ):
            raise ContiguousRunnerError(
                "bridge export references an undeclared target boundary"
            )
        secret_scan = _validate_bound_receipt(
            collection.secret_scan_receipt_path,
            collection.secret_scan_receipt_sha256,
            expected_path=host_root / "secret_scan_receipt.json",
            expected_kind="contiguous_secret_scan",
            spec=spec,
        )
        state_root = Path(spec.app_server_state_dir)
        try:
            controller_state_inventory = (
                Transport.inventory_controller_state(
                    state_root,
                    sentinels=self._secret_sentinels,
                )
            )
        except Exception as exc:
            raise ContiguousRunnerError(
                "controller state inventory failed bounded descriptor replay"
            ) from exc
        if (
            secret_scan.get("scanned_sha256")
            != {
                "app_server_transcript":
                    collection.app_server_transcript_sha256,
                "backend_transcript":
                    collection.host_transcript_sha256,
                "container_stderr":
                    collection.container_stderr_sha256,
                "container_stdout":
                    collection.container_stdout_sha256,
                "output_tree": collection.output_tree_sha256,
                "app_server_state_tree":
                    collection.app_server_state_tree_sha256,
            }
            or secret_scan.get("controller_state_inventory")
            != controller_state_inventory.as_receipt()
            or secret_scan.get("secret_occurrences") != 0
            or controller_state_inventory.secret_occurrences != 0
            or secret_scan.get("credential_generations_scanned")
            != token_usage["auth_refresh_count"] + 1
            or secret_scan.get("controller_terminal_scan_passed")
            is not True
            or secret_scan.get("status") != "PASS"
        ):
            raise ContiguousRunnerError(
                "secret scan did not cover every immutable byte stream"
            )
        try:
            controller_state_scan = Taint.scan_controller_state(
                state_root,
                inventory=controller_state_inventory,
                canaries=self._controller_state_canaries,
            )
        except Exception as exc:
            raise ContiguousRunnerError(
                "controller state failed independent taint/canary replay"
            ) from exc
        state_scan_receipt = _validate_bound_receipt(
            collection.controller_state_scan_receipt_path,
            collection.controller_state_scan_receipt_sha256,
            expected_path=
                host_root / "controller_state_scan_receipt.json",
            expected_kind="contiguous_controller_state_scan",
            spec=spec,
        )
        if (
            collection.controller_state_inventory_sha256
            != controller_state_inventory.inventory_sha256
            or state_scan_receipt.get("scanner_source_sha256")
            != Taint.source_sha256()
            or state_scan_receipt.get("controller_state_scan")
            != controller_state_scan.as_receipt()
        ):
            raise ContiguousRunnerError(
                "controller-state scan receipt omits or alters coverage"
            )
        retained_roots = {
            "host_evidence": host_root,
            "proposer_output": Path(spec.output_dir),
        }
        try:
            retained_canary_scan = Taint.scan_retained_canary_roots(
                retained_roots,
                canaries=self._controller_state_canaries,
            )
        except Exception as exc:
            raise ContiguousRunnerError(
                "retained evidence failed independent canary replay"
            ) from exc
        retained_canary_receipt = _validate_bound_receipt(
            collection.retained_canary_scan_receipt_path,
            collection.retained_canary_scan_receipt_sha256,
            expected_path=(
                Path(spec.generation_dir)
                / "retained_canary_scan_receipt.json"
            ),
            expected_kind="contiguous_retained_canary_scan",
            spec=spec,
        )
        retained_canary_receipt_scan = Taint.scan_canaries_in_file(
            Path(collection.retained_canary_scan_receipt_path),
            canaries=self._controller_state_canaries,
            evidence_kind="retained_canary_receipt",
        )
        retained_scan_value = retained_canary_scan.as_receipt()
        teardown_path = host_root / "arena_volume_teardown.json"
        teardown_receipt_exists = (
            teardown_path.exists() or teardown_path.is_symlink()
        )
        if allow_arena_teardown_receipt and not teardown_receipt_exists:
            raise ContiguousRunnerError(
                "journaled Arena teardown receipt is absent"
            )
        if teardown_receipt_exists:
            try:
                _validate_arena_volume_teardown_receipt(
                    spec=spec,
                    prepared=prepared,
                    receipt_path=teardown_path,
                )
                teardown_canary_scan = Taint.scan_canaries_in_file(
                    teardown_path,
                    canaries=self._controller_state_canaries,
                    evidence_kind="arena_volume_teardown",
                )
                retained_scan_value = (
                    _retained_canary_scan_receipt_excluding(
                        retained_roots,
                        retained_canary_scan,
                        excluded={
                            "host_evidence": {
                                "arena_volume_teardown.json"
                            },
                        },
                    )
                )
            except Exception as exc:
                raise ContiguousRunnerError(
                    "terminal Arena teardown evidence failed retained "
                    "canary replay"
                ) from exc
            if teardown_canary_scan.hits:
                raise ContiguousRunnerError(
                    "terminal Arena teardown receipt contains a canary"
                )
        if (
            retained_canary_receipt.get("scanner_source_sha256")
            != Taint.source_sha256()
            or retained_canary_receipt.get("retained_canary_scan")
            != retained_scan_value
            or retained_canary_receipt.get(
                "controller_state_scan_receipt_sha256"
            )
            != collection.controller_state_scan_receipt_sha256
            or retained_canary_receipt_scan.hits
            or retained_canary_receipt_scan.sha256
            != collection.retained_canary_scan_receipt_sha256
        ):
            raise ContiguousRunnerError(
                "retained canary receipt omits, alters, or leaks coverage"
            )
        if (
            controller_state_scan.status == "TAINT"
            or controller_state_scan.hits
            or controller_state_scan.canary_occurrences
            or retained_canary_scan.status == "TAINT"
            or retained_canary_scan.hits
            or retained_canary_scan.canary_occurrences
        ) and (
            collection.result.kind != "tainted"
            or collection.result.candidate is not None
            or collection.result.wip is not None
        ):
            raise ContiguousRunnerError(
                "tainted controller state remained promotion/WIP eligible"
            )
        final_binding = _validate_bound_receipt(
            collection.final_thread_binding_path,
            collection.final_thread_binding_sha256,
            expected_path=Path(spec.generation_dir)
            / "final_thread_binding.json",
            expected_kind="contiguous_final_thread_binding",
            spec=spec,
        )
        expected_final_binding = {
            "thread_id": launched.codex_thread_id,
            "turn_id": launched.codex_turn_id,
            "thread_mode": spec.thread_mode,
            "turn_status": collection.structured_turn_status,
            "provider_outcome":
                collection.structured_provider_outcome,
            "transcript_chain_sha256":
                collection.final_transcript_chain_sha256,
            "final_transcript_chain_receipt_sha256":
                collection.final_transcript_chain_receipt_sha256,
            "token_usage_receipt_sha256":
                collection.token_usage_receipt_sha256,
            "provider_usage_receipt_sha256":
                collection.provider_usage_receipt_sha256,
            "bridge_export_receipt_sha256":
                collection.bridge_export_receipt_sha256,
            "host_blocker_code": (
                collection.result.blocker.code
                if collection.result.blocker is not None
                else None
            ),
            "host_blocker_receipt_sha256": (
                collection.result.blocker.receipt_sha256
                if collection.result.blocker is not None
                else None
            ),
            "secret_scan_receipt_sha256":
                collection.secret_scan_receipt_sha256,
            "app_server_state_tree_sha256":
                collection.app_server_state_tree_sha256,
            "controller_state_inventory_sha256":
                collection.controller_state_inventory_sha256,
            "controller_state_scan_receipt_sha256":
                collection.controller_state_scan_receipt_sha256,
            "retained_canary_scan_receipt_sha256":
                collection.retained_canary_scan_receipt_sha256,
            "taint_scan_receipt_sha256":
                collection.taint_scan_receipt_sha256,
            "wip_export_receipt_sha256": (
                collection.result.wip.wip_export_receipt_sha256
                if collection.result.wip is not None
                else None
            ),
            "target_boundary_receipt_sha256":
                collection.target_boundary_receipt_sha256,
            "target_boundary_sha256":
                collection.target_boundary_sha256,
            "target_boundary_workspace_tree_sha256":
                collection.target_boundary_workspace_tree_sha256,
            "model_final_text_sha256":
                collection.model_final_text_sha256,
            "model_final_text_eligible": False,
        }
        if any(final_binding.get(key) != value
               for key, value in expected_final_binding.items()):
            raise ContiguousRunnerError(
                "final thread binding omits terminal evidence"
            )
        final_binding_canary_scan = Taint.scan_canaries_in_file(
            Path(collection.final_thread_binding_path),
            canaries=self._controller_state_canaries,
            evidence_kind="final_thread_binding",
        )
        if (
            final_binding_canary_scan.hits
            or final_binding_canary_scan.sha256
            != collection.final_thread_binding_sha256
        ):
            raise ContiguousRunnerError(
                "final thread binding leaks a live containment canary"
            )
        try:
            Contract._validate_regular_tree(
                state_root, label="per-lane Codex state"
            )
        except Exception as exc:
            raise ContiguousRunnerError(
                "per-lane Codex state is not a regular tree"
            ) from exc
        if (
            Contract._tree_hash(state_root)
            != collection.app_server_state_tree_sha256
        ):
            raise ContiguousRunnerError(
                "per-lane Codex state changed after final binding"
            )
        if collection.structured_provider_outcome != "completed":
            if collection.result.kind != "infrastructure":
                raise ContiguousRunnerError(
                    "structured provider failure is not infrastructure"
                )
        elif collection.structured_turn_status != "completed":
            raise ContiguousRunnerError(
                "completed provider outcome has noncompleted turn"
            )
        if collection.result.kind == "candidate":
            candidate = collection.result.candidate
            assert candidate is not None
            if (
                export_receipt.get("outcome") != "candidate"
                or export_receipt.get("candidate_manifest_sha256")
                != candidate.candidate_manifest_sha256
                or candidate.probe_isolation_mode
                != prepared.probe_isolation_mode
                or candidate.probe_isolation_evidence_sha256
                != prepared.probe_isolation_evidence_sha256
            ):
                raise ContiguousRunnerError(
                    "candidate was not created through the lane bridge "
                    "under its controller-selected probe substrate"
                )
        if collection.result.wip is not None and (
            collection.result.wip.codex_thread_id
            != launched.codex_thread_id
            or collection.result.wip.final_thread_binding_sha256
            != collection.final_thread_binding_sha256
            or collection.result.wip.transcript_chain_sha256
            != collection.final_transcript_chain_sha256
            or collection.result.wip.app_server_state_dir
            != spec.app_server_state_dir
            or collection.result.wip.app_server_state_tree_sha256
            != collection.app_server_state_tree_sha256
            or collection.result.wip.provider_usage_receipt_sha256
            != collection.provider_usage_receipt_sha256
            or collection.result.wip.token_usage_receipt_sha256
            != collection.token_usage_receipt_sha256
            or collection.result.wip.taint_scan_receipt_sha256
            != collection.taint_scan_receipt_sha256
            or collection.result.wip.retained_canary_scan_receipt_sha256
            != collection.retained_canary_scan_receipt_sha256
        ):
            raise ContiguousRunnerError(
                "WIP does not include finalized thread/state evidence"
            )
        try:
            frontier_brief = _read_json_file(
                Path(spec.frontier_brief_path)
            )
            prompt = (
                "Solve exactly this receipt-bound ARC-AGI-3 frontier using "
                "only the contiguous_lane namespace. Immutable frontier:\n"
                + Transport.canonical_json(frontier_brief).decode("ascii")
            )
            scan_policy = Taint.AppServerScanPolicy(
                state_root=spec.app_server_state_dir,
                neutral_cwd=spec.neutral_host_cwd_path,
                model=spec.proposer_transport.model,
                model_provider=spec.proposer_transport.model_provider,
                reasoning_effort=spec.effort,
                thread_mode=spec.thread_mode,
                resume_thread_id=spec.resume_thread_id,
                hard_safety_seconds=spec.hard_safety_seconds,
                max_auth_refreshes=spec.max_auth_refreshes,
                prompt_sha256=hashlib.sha256(
                    prompt.encode("utf-8")
                ).hexdigest(),
                secret_sentinels=self._secret_sentinels,
            )
            scan_records = [
                Taint.scan_evidence(
                    app_transcript,
                    evidence_kind="app_server_jsonl",
                    app_server_policy=scan_policy,
                ),
                Taint.scan_evidence(
                    transcript,
                    evidence_kind="backend_jsonl",
                ),
                Taint.scan_evidence(
                    stdout_path,
                    evidence_kind="container_stdout",
                ),
                Taint.scan_evidence(
                    stderr_path,
                    evidence_kind="container_stderr",
                ),
                *[
                    Taint.scan_evidence(
                        entry,
                        evidence_kind="candidate_output",
                    )
                    for entry in sorted(
                        entry
                        for entry in Path(spec.output_dir).rglob("*")
                        if entry.is_file()
                    )
                ],
                *controller_state_scan.records,
            ]
        except Exception as exc:
            raise ContiguousRunnerError(
                "trusted taint scan could not inspect collected evidence"
            ) from exc
        taint_receipt = _validate_bound_receipt(
            collection.taint_scan_receipt_path,
            collection.taint_scan_receipt_sha256,
            expected_path=host_root / "taint_scan_receipt.json",
            expected_kind="contiguous_taint_scan",
            spec=spec,
        )
        expected_scan_records = [
            {
                **asdict(record),
                "hits": list(record.hits),
            }
            for record in scan_records
        ]
        actionable_taint = sorted(
            {
                hit
                for record in scan_records
                for hit in record.hits
            }
        )
        if (
            taint_receipt.get("scanner_source_sha256")
            != Taint.source_sha256()
            or taint_receipt.get("records") != expected_scan_records
            or taint_receipt.get("hits") != actionable_taint
            or taint_receipt.get("status")
            != ("TAINT" if actionable_taint else "CLEAN")
        ):
            raise ContiguousRunnerError(
                "taint receipt omits or alters strict scan coverage"
            )
        if actionable_taint and collection.result.kind != "tainted":
            raise ContiguousRunnerError(
                "collected transcript/output contains unreported taint: "
                + ",".join(actionable_taint)
            )
        sanitized = self._sanitize_result(spec, collection.result)
        if sanitized != collection.result:
            raise ContiguousRunnerError(
                "collection retained an inadmissible WIP descriptor"
            )

    @staticmethod
    def _validate_candidate(
        spec: AttemptSpec, candidate: PromotionCandidate | None
    ) -> None:
        if (
            candidate is None
            or candidate.game != spec.game
            or candidate.from_level != spec.target_level - 1
            or candidate.to_level != spec.target_level
            or candidate.parent_checkpoint_sha256
            != spec.parent_checkpoint_sha256
            or not _safe_path_string(candidate.candidate_manifest_path)
            or not _is_sha256(candidate.candidate_manifest_sha256)
            or candidate.probe_isolation_mode
            not in Contract.PROBE_ISOLATION_MODES
            or not _is_sha256(
                candidate.probe_isolation_evidence_sha256
            )
            or (
                spec.supervisory_handoff is None
                and (
                    candidate.supervisory_handoff_sha256
                    is not None
                    or candidate
                    .supervisory_native_reproduction_receipt_sha256
                    is not None
                )
            )
            or (
                spec.supervisory_handoff is not None
                and (
                    candidate.supervisory_handoff_sha256
                    != spec.supervisory_handoff
                    .supervisory_handoff_sha256
                    or not _is_sha256(
                        candidate
                        .supervisory_native_reproduction_receipt_sha256
                    )
                )
            )
        ):
            raise ContiguousRunnerError(
                "candidate is not the exact admitted K→K+1 edge"
            )

    @staticmethod
    def _result_from_payload(payload: dict[str, Any]) -> AttemptResult:
        base = {
            "attempt_id",
            "kind",
            "cost_used",
            "reason",
            "candidate",
            "wip",
            "blocker",
            "native_sidecar_request_draft",
        }
        settlement = {
            "authenticated_cost_units",
            "budget_reservation_id",
            "scheduler_decision_id",
        }
        keys = set(payload)
        if (
            frozenset(keys)
            not in {frozenset(base), frozenset(base | settlement)}
            or payload.get("kind") not in RESULT_KINDS
            or (
                keys == base | settlement
                and (
                    not isinstance(
                        payload["authenticated_cost_units"], int
                    )
                    or isinstance(
                        payload["authenticated_cost_units"], bool
                    )
                    or payload["authenticated_cost_units"] < 0
                    or not _safe_identifier(
                        payload["budget_reservation_id"]
                    )
                    or not _safe_identifier(
                        payload["scheduler_decision_id"]
                    )
                )
            )
        ):
            raise ContiguousRunnerError("attempt result schema mismatch")
        cost = payload["cost_used"]
        reason = payload["reason"]
        if (
            not _is_finite_number(cost)
            or cost < 0
            or not isinstance(reason, str)
            or len(reason) > 4096
            or "\x00" in reason
        ):
            raise ContiguousRunnerError("invalid attempt result")
        candidate = _candidate_from_dict(payload["candidate"])
        wip = _wip_from_dict(payload["wip"])
        blocker = _blocker_from_dict(payload["blocker"])
        native_request_draft = (
            _native_sidecar_request_draft_from_dict(
                payload["native_sidecar_request_draft"]
            )
        )
        if (payload["kind"] == "candidate") != (candidate is not None):
            raise ContiguousRunnerError(
                "candidate result/candidate descriptor mismatch"
            )
        if payload["kind"] != "clean_no_progress" and wip is not None:
            raise ContiguousRunnerError(
                "only clean no-progress outcomes may retain WIP"
            )
        if (payload["kind"] == "blocker") != (blocker is not None):
            raise ContiguousRunnerError(
                "blocker result/evidence descriptor mismatch"
            )
        if (
            native_request_draft is not None
            and payload["kind"] != "clean_no_progress"
        ):
            raise ContiguousRunnerError(
                "only clean no-progress may carry a native sidecar request"
            )
        return AttemptResult(
            kind=payload["kind"],
            cost_used=float(cost),
            reason=reason,
            candidate=candidate,
            wip=wip,
            blocker=blocker,
            native_sidecar_request_draft=native_request_draft,
        )

    @staticmethod
    def _commit_source_path(commit: PromotionCommit) -> Path:
        return (
            Path(commit.checkpoint_path).parent
            / Contract.WINNING_SOURCE_NAME
        )

    @staticmethod
    def _commit_from_payload(payload: dict[str, Any]) -> PromotionCommit:
        required = {
            "attempt_id",
            "game",
            "from_level",
            "to_level",
            "parent_checkpoint_sha256",
            "checkpoint_path",
            "checkpoint_sha256",
            "exact_path",
            "promotion_receipt_sha256",
            "source_version_id",
            "source_tree_sha256",
            "supervisory_handoff_sha256",
            "supervisory_native_reproduction_receipt_sha256",
            "source_path",
            "candidate_manifest_sha256",
        }
        if set(payload) != required:
            raise ContiguousRunnerError("promotion commit schema mismatch")
        value = {
            key: payload[key]
            for key in required
            - {
                "attempt_id",
                "source_path",
                "candidate_manifest_sha256",
            }
        }
        if not isinstance(value["exact_path"], list):
            raise ContiguousRunnerError("promotion exact_path must be a list")
        value["exact_path"] = tuple(value["exact_path"])
        try:
            commit = PromotionCommit(**value)
        except TypeError as exc:
            raise ContiguousRunnerError(
                "promotion commit schema mismatch"
            ) from exc
        if payload["source_path"] != str(
            ContiguousCampaignRunner._commit_source_path(commit)
        ) or not _is_sha256(payload["candidate_manifest_sha256"]):
            raise ContiguousRunnerError(
                "promotion commit artifact/candidate binding is invalid"
            )
        return commit

    @staticmethod
    def _validate_commit(
        spec: AttemptSpec,
        commit: PromotionCommit,
        lanes: dict[str, dict[str, Any]],
        candidate: PromotionCandidate,
    ) -> None:
        lane = lanes[spec.game]
        checkpoint_path = Path(commit.checkpoint_path)
        subject_root = checkpoint_path.parent
        version_root = subject_root.parent
        source_root = ContiguousCampaignRunner._commit_source_path(
            commit
        )
        if (
            commit.game != spec.game
            or commit.from_level != lane["reached"]
            or commit.to_level != lane["reached"] + 1
            or commit.to_level != spec.target_level
            or commit.parent_checkpoint_sha256
            != lane["checkpoint_sha256"]
            or not _is_sha256(commit.checkpoint_sha256)
            or not _is_sha256(commit.promotion_receipt_sha256)
            or not _is_sha256(commit.source_tree_sha256)
            or commit.supervisory_handoff_sha256
            != candidate.supervisory_handoff_sha256
            or commit
            .supervisory_native_reproduction_receipt_sha256
            != candidate
            .supervisory_native_reproduction_receipt_sha256
            or not isinstance(commit.source_version_id, str)
            or re.fullmatch(
                r"[0-9a-f]{32}", commit.source_version_id
            )
            is None
            or not _safe_path_string(commit.checkpoint_path)
            or not checkpoint_path.is_absolute()
            or checkpoint_path.name != Contract.CHECKPOINT_NAME
            or subject_root.name != f"{commit.game}_legs"
            or version_root.name != commit.source_version_id
            or version_root.parent.name != "versions"
            or source_root.parent != subject_root
            or not commit.exact_path
        ):
            raise ContiguousRunnerError(
                "promotion commit is not the exact admitted K→K+1 edge"
            )
        try:
            Contract._validate_regular_tree(
                source_root, label="promoted winning source"
            )
        except Exception as exc:
            raise ContiguousRunnerError(
                "promoted winning source is unsafe"
            ) from exc
        try:
            source_payloads = {
                entry.name: _bounded_regular_bytes(
                    entry,
                    maximum=SourceSchema.MAX_FILE_BYTES,
                )
                for entry in source_root.iterdir()
                if entry.is_file()
            }
            SourceSchema.validate_source_payloads(source_payloads)
        except (
            ContiguousRunnerError,
            SourceSchema.SourceSchemaError,
        ) as exc:
            raise ContiguousRunnerError(
                "promoted winning source violates its shared schema"
            ) from exc
        if (
            any(not entry.is_file() for entry in source_root.iterdir())
            or Contract._tree_hash(source_root)
            != commit.source_tree_sha256
            or not (
                subject_root / Contract.HOST_RECEIPT_NAME
            ).is_file()
            or _sha256_file(
                subject_root / Contract.HOST_RECEIPT_NAME
            )
            != commit.promotion_receipt_sha256
        ):
            raise ContiguousRunnerError(
                "promoted winning source hash/path is invalid"
            )
        promotion_receipt = _read_json_file(
            subject_root / Contract.HOST_RECEIPT_NAME
        )
        if (
            promotion_receipt.get("supervisory_handoff_sha256")
            != commit.supervisory_handoff_sha256
            or promotion_receipt.get(
                "supervisory_native_reproduction_receipt_sha256"
            )
            != commit
            .supervisory_native_reproduction_receipt_sha256
        ):
            raise ContiguousRunnerError(
                "promotion receipt drops or substitutes supervisory "
                "provenance"
            )
        checkpoint = Contract.load_trusted_checkpoint(
            checkpoint_path,
            expected_game=commit.game,
            authoritative_target=lane["target"],
        )
        parent = Contract.load_trusted_checkpoint(
            Path(lane["checkpoint_path"]),
            expected_game=commit.game,
            authoritative_target=lane["target"],
        )
        if (
            checkpoint.reached != commit.to_level
            or tuple(checkpoint.final_path) != commit.exact_path
            or _sha256_file(checkpoint_path) != commit.checkpoint_sha256
            or tuple(checkpoint.records[:-1]) != tuple(parent.records)
            or checkpoint.total_marginal_C
            != parent.total_marginal_C
            + checkpoint.records[-1]["marginal_C"]
        ):
            raise ContiguousRunnerError(
                "promotion checkpoint does not preserve the exact parent lineage"
            )

    def _validate_host_blocker(
        self,
        spec: AttemptSpec,
        blocker: HostBlockerEvidence | None,
    ) -> HostBlockerEvidence:
        if (
            blocker is None
            or blocker.code not in HOST_BLOCKER_CODES
            or not _safe_path_string(blocker.receipt_path)
            or not _is_sha256(blocker.receipt_sha256)
        ):
            raise ContiguousRunnerError(
                "blocker lacks a recognized host evidence descriptor"
            )
        host_root = Path(spec.host_transcript_path).parent
        receipt = _validate_bound_receipt(
            blocker.receipt_path,
            blocker.receipt_sha256,
            expected_path=host_root / HOST_BLOCKER_RECEIPT_NAME,
            expected_kind=HOST_BLOCKER_RECEIPT_KIND,
            spec=spec,
        )
        arena_result = receipt.get("arena_host_result")
        parent_level = spec.target_level - 1
        if (
            set(receipt) != Scheduler.HOST_BLOCKER_RECEIPT_FIELDS
            or receipt.get("authority") != HOST_BLOCKER_AUTHORITY
            or receipt.get("code") != blocker.code
            or receipt.get("game") != spec.game
            or receipt.get("frontier_sha256") != spec.frontier_sha256
            or receipt.get("parent_checkpoint_sha256")
            != spec.parent_checkpoint_sha256
            or receipt.get("parent_level") != parent_level
            or receipt.get("target_level") != spec.target_level
            or receipt.get("parent_terminal") is not True
            or not _is_sha256(receipt.get("arena_binding_sha256"))
            or not _is_sha256(receipt.get("parent_path_sha256"))
            or not _is_sha256(receipt.get("parent_snapshot_sha256"))
            or not _is_sha256(receipt.get("arena_host_result_sha256"))
            or not _is_sha256(receipt.get("host_authentication_sha256"))
            or not isinstance(arena_result, dict)
            or set(arena_result)
            != Scheduler.HOST_BLOCKER_ARENA_RESULT_FIELDS
            or hashlib.sha256(
                _canonical_json(arena_result)
            ).hexdigest()
            != receipt.get("arena_host_result_sha256")
            or arena_result.get("binding_sha256")
            != receipt.get("arena_binding_sha256")
            or arena_result.get("game") != spec.game
            or arena_result.get("parent_level") != parent_level
            or arena_result.get("levels_completed") != parent_level
            or arena_result.get("parent_terminal") is not True
            or arena_result.get("parent_snapshot_sha256")
            != receipt.get("parent_snapshot_sha256")
            or not isinstance(arena_result.get("parent_path"), list)
            or hashlib.sha256(
                _canonical_json(arena_result["parent_path"])
            ).hexdigest()
            != receipt.get("parent_path_sha256")
        ):
            raise ContiguousRunnerError(
                "host blocker receipt is malformed or bound to another "
                "frontier"
            )
        arena_receipt = _validate_bound_receipt(
            receipt["arena_session_binding_receipt_path"],
            receipt["arena_session_binding_receipt_sha256"],
            expected_path=host_root
            / "arena_session_binding_receipt.json",
            expected_kind="contiguous_arena_session_binding",
            spec=spec,
        )
        binding_event = arena_receipt.get("binding_event")
        if (
            set(arena_receipt)
            != {
                "schema",
                "kind",
                "campaign_id",
                "generation_id",
                "attempt_id",
                "attempt_spec_sha256",
                "binding_event",
            }
            or not isinstance(binding_event, dict)
            or binding_event.get("binding_sha256")
            != receipt.get("arena_binding_sha256")
            or binding_event.get("seed_snapshot_sha256")
            != receipt.get("parent_snapshot_sha256")
            or binding_event.get("parent_path_sha256")
            != receipt.get("parent_path_sha256")
            or binding_event.get("game") != spec.game
            or binding_event.get("frontier_sha256")
            != spec.frontier_sha256
            or binding_event.get("parent_checkpoint_sha256")
            != spec.parent_checkpoint_sha256
            or binding_event.get("parent_level") != parent_level
            or binding_event.get("target_level") != spec.target_level
        ):
            raise ContiguousRunnerError(
                "host blocker Arena binding is stale or substituted"
            )
        authentication = receipt["host_authentication_sha256"]
        unsigned = dict(receipt)
        del unsigned["host_authentication_sha256"]
        expected_authentication = host_blocker_authentication_sha256(
            unsigned, self._controller_state_canaries
        )
        if not hmac.compare_digest(
            authentication, expected_authentication
        ):
            raise ContiguousRunnerError(
                "host blocker receipt authentication failed"
            )
        return blocker

    def _sanitize_result(
        self, spec: AttemptSpec, result: AttemptResult
    ) -> AttemptResult:
        if result.kind not in RESULT_KINDS:
            raise ContiguousRunnerError("backend returned unknown result kind")
        if result.kind == "blocker":
            cost_is_valid = (
                _is_finite_number(result.cost_used)
                and result.cost_used >= 0
                and (
                    spec.cost_limit_remaining is None
                    or result.cost_used <= spec.cost_limit_remaining
                )
            )
            try:
                if (
                    not cost_is_valid
                    or result.candidate is not None
                    or result.wip is not None
                    or result.native_sidecar_request_draft is not None
                ):
                    raise ContiguousRunnerError(
                        "blocker carries invalid cost/candidate/WIP data"
                    )
                blocker = self._validate_host_blocker(
                    spec, result.blocker
                )
            except ContiguousRunnerError:
                # A proposer, collector, stale journal suffix, or copied
                # receipt can claim "blocked", but cannot create BLOCKED
                # authority.  Keep the attempt noncounting and retryable.
                return AttemptResult(
                    kind="infrastructure",
                    cost_used=(
                        float(result.cost_used)
                        if cost_is_valid
                        else 0.0
                    ),
                    reason="rejected unauthenticated blocker claim",
                )
            return AttemptResult(
                kind="blocker",
                cost_used=float(result.cost_used),
                reason=HOST_BLOCKER_REASON_PREFIX + blocker.code,
                blocker=blocker,
            )
        if (
            not _is_finite_number(result.cost_used)
            or result.cost_used < 0
            or not isinstance(result.reason, str)
            or len(result.reason) > 4096
            or "\x00" in result.reason
            or (
                spec.cost_limit_remaining is not None
                and result.cost_used > spec.cost_limit_remaining
            )
        ):
            raise ContiguousRunnerError("backend returned invalid result")
        if result.kind in {
            "tainted", "protocol_invalid", "infrastructure"
        }:
            # Terminal quarantine/protocol/infrastructure precedence strips
            # every lower-level candidate, WIP, blocker, or sidecar claim
            # while retaining accounting.
            return AttemptResult(
                kind=result.kind,
                cost_used=float(result.cost_used),
                reason=result.reason,
            )
        if result.blocker is not None:
            raise ContiguousRunnerError(
                "nonblocker outcome carries blocker authority"
            )
        if result.kind == "candidate":
            self._validate_candidate(spec, result.candidate)
            if result.wip is not None:
                raise ContiguousRunnerError(
                    "candidate outcome may not retain unpromoted WIP"
                )
            if result.native_sidecar_request_draft is not None:
                raise ContiguousRunnerError(
                    "candidate outcome may not retain a sidecar request"
                )
            manifest = Path(result.candidate.candidate_manifest_path)
            output = Path(spec.output_dir).resolve(strict=True)
            try:
                resolved_manifest = manifest.resolve(strict=True)
            except OSError as exc:
                raise ContiguousRunnerError(
                    "candidate manifest is missing"
                ) from exc
            if (
                resolved_manifest.parent != output
                or resolved_manifest.name != Contract.CANDIDATE_NAME
                or manifest.is_symlink()
                or _sha256_file(resolved_manifest)
                != result.candidate.candidate_manifest_sha256
            ):
                raise ContiguousRunnerError(
                    "candidate manifest is outside/broken for this generation"
                )
            return result
        if result.candidate is not None:
            raise ContiguousRunnerError(
                "noncandidate outcome carries candidate data"
            )
        if result.kind == "clean_no_progress" and result.wip is not None:
            try:
                self._validate_wip_for_spec(result.wip, spec)
                wip_root = Path(
                    result.wip.wip_root_path
                ).resolve(strict=True)
                output_root = Path(spec.output_dir).resolve(strict=True)
                if (
                    wip_root.parent != output_root
                    or wip_root.name != "wip"
                ):
                    raise ContiguousRunnerError(
                        "returned WIP is outside this attempt output"
                    )
                Contract._validate_regular_tree(
                    wip_root, label="returned WIP"
                )
                if (
                    Contract._tree_hash(wip_root)
                    != result.wip.wip_tree_sha256
                ):
                    raise ContiguousRunnerError(
                        "returned WIP hash mismatch"
                    )
            except Exception:
                # The outcome remains an auditable clean failure, but mismatched
                # WIP is never made restore-eligible.
                return AttemptResult(
                    kind=result.kind,
                    cost_used=result.cost_used,
                    reason=(
                        result.reason
                        + ("; " if result.reason else "")
                        + "WIP rejected: parent/frontier mismatch"
                    ),
                )
        elif result.wip is not None:
            raise ContiguousRunnerError(
                "non-clean terminal outcome carries WIP"
            )
        draft = result.native_sidecar_request_draft
        if draft is not None:
            try:
                Scheduler.validate_native_sidecar_request_draft(draft)
            except Scheduler.SchedulerError as exc:
                raise ContiguousRunnerError(
                    "native sidecar request draft is malformed"
                ) from exc
            if (
                result.kind != "clean_no_progress"
                or draft.game != spec.game
                or draft.frontier_sha256 != spec.frontier_sha256
                or draft.parent_checkpoint_sha256
                != spec.parent_checkpoint_sha256
                or draft.native_attempt_id != spec.attempt_id
            ):
                raise ContiguousRunnerError(
                    "native sidecar request draft crosses attempt/frontier"
                )
        return result

    def _result_payload(
        self,
        attempt_id: str,
        result: AttemptResult,
        *,
        attempt: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        value = {
            "attempt_id": attempt_id,
            "kind": result.kind,
            "cost_used": float(result.cost_used),
            "reason": result.reason,
            "candidate": _candidate_to_dict(result.candidate),
            "wip": _wip_to_dict(result.wip),
            "blocker": _blocker_to_dict(result.blocker),
            "native_sidecar_request_draft":
                _native_sidecar_request_draft_to_dict(
                    result.native_sidecar_request_draft
                ),
        }
        if attempt is not None:
            try:
                authenticated = Scheduler.charge_to_units(
                    result.cost_used
                )
            except Scheduler.SchedulerError as exc:
                raise ContiguousRunnerError(
                    "attempt result cost cannot be normalized"
                ) from exc
            value.update({
                "authenticated_cost_units": authenticated,
                "budget_reservation_id":
                    attempt["budget_reservation_id"],
                "scheduler_decision_id":
                    attempt["scheduler_decision_id"],
            })
        return value

    @staticmethod
    def _budget_from_state(
        state: Mapping[str, Any],
    ) -> Scheduler.BudgetState:
        try:
            return Scheduler.validate_budget_state(
                Scheduler.BudgetState(
                    cost_window_id=state["cost_window_id"],
                    limit_units=state["limit_units"],
                    settled_units=state["settled_cost_units"],
                    live_reservations=tuple(
                        Scheduler.BudgetReservation(**item)
                        for item in state[
                            "live_budget_reservations"
                        ]
                    ),
                )
            )
        except (
            KeyError,
            TypeError,
            Scheduler.SchedulerError,
        ) as exc:
            raise ContiguousRunnerError(
                "live state has an invalid scheduler budget projection"
            ) from exc

    def _scheduler_snapshot_from_state(
        self, state: Mapping[str, Any]
    ) -> Scheduler.CampaignSnapshot:
        events = self.journal._read_authenticated()
        if not events:
            raise ContiguousRunnerError(
                "cannot schedule without durable genesis"
            )
        return self._scheduler_snapshot(
            genesis={
                "campaign_id": state["campaign_id"],
                "inventory": state["inventory"],
                "max_lanes": state["max_lanes"],
            },
            lanes=state["lanes"],
            attempts=state["attempts"],
            budget=self._budget_from_state(state),
            journal_head_sequence=events[-1]["sequence"],
            journal_head_digest=events[-1]["digest"],
            clean_proposer_settlements=tuple(
                settlement
                for lane in state["lanes"].values()
                for settlement in lane["clean_proposer_settlements"]
            ),
            complexity_rounds=tuple(state["complexity_rounds"]),
            auxiliary_assignments=tuple(
                item["state"]
                for item in state["auxiliary_assignments"].values()
            ),
            sidecar_requests=tuple(
                item["request"]
                for item in state["sidecar_requests"].values()
                if not item["invalidated"]
            ),
        )

    def _append_scheduler_decision(
        self, state: dict[str, Any]
    ) -> Scheduler.SchedulerDecision | None:
        try:
            Scheduler.require_dispatch_headroom(self.root)
        except Scheduler.SchedulerError as exc:
            raise ContiguousRunnerError(
                "journal lacks terminal-evidence dispatch headroom"
            ) from exc
        identities = self._fresh_decision_identifiers(
            state,
            (
                "decision",
                "attempt",
                "generation",
                "budget reservation",
            ),
        )
        try:
            decision = Scheduler.build_decision(
                self._scheduler_snapshot_from_state(state),
                decision_id=identities[0],
                attempt_id=identities[1],
                generation_id=identities[2],
                reservation_id=identities[3],
            )
        except Scheduler.SchedulerError as exc:
            raise ContiguousRunnerError(
                "canonical scheduler could not build a decision"
            ) from exc
        if decision is None:
            return None
        self.journal.append(
            event_id=f"{decision.decision_id}:decision",
            kind="SCHEDULER_DECISION",
            payload={
                "decision": Scheduler.decision_to_dict(decision)
            },
            recorded_at=self.clock(),
        )
        return decision

    def _consume_scheduler_decision(
        self,
        decision: Scheduler.SchedulerDecision,
        state: dict[str, Any],
    ) -> AttemptReservation:
        pending_raw = state.get("pending_scheduler_decision")
        try:
            pending = Scheduler.decision_from_dict(pending_raw)
        except Scheduler.SchedulerError as exc:
            raise ContiguousRunnerError(
                "pending scheduler decision cannot be recovered"
            ) from exc
        if pending != decision:
            raise ContiguousRunnerError(
                "scheduler attempted to consume a different pending decision"
            )
        game = decision.choice.game
        lane = state["lanes"].get(game)
        if not isinstance(lane, dict):
            raise ContiguousRunnerError(
                "scheduler selected an unknown game"
            )
        attempt_id = decision.attempt_id
        generation_id = decision.generation_id
        generation = self.generations / generation_id
        if generation.exists() or generation.is_symlink():
            raise ContiguousRunnerError(
                f"generation directory already exists: {generation}"
            )
        input_dir = generation / "input"
        scratch = generation / "scratch"
        output = generation / "output"
        rpc = generation / "rpc"
        bridge = generation / "bridge"
        host = generation / "host"
        arena_socket = rpc / "arena.sock"
        arena_token = rpc / "token"
        bridge_socket = bridge / "proposer.sock"
        bridge_token = bridge / "proposer-token"
        bridge_policy_receipt = host / "bridge_policy_receipt.json"
        host_transcript = host / "backend.jsonl"
        app_server_transcript = host / "app_server.jsonl"
        neutral_host_cwd = host / "neutral"
        app_server_state = generation / "state" / "codex_home"
        app_server_control = host / "app_server_control"
        wip = (
            lane["wip"]
            if decision.choice.selected_wip is not None
            else None
        )
        if (
            (wip is None) != (decision.choice.selected_wip is None)
            or (
                wip is not None
                and asdict(self._scheduler_wip(wip))
                != asdict(decision.choice.selected_wip)
            )
        ):
            raise ContiguousRunnerError(
                "scheduler decision selected stale or substituted WIP"
            )
        if wip is not None and (
            wip.codex_thread_id is None
            or wip.final_thread_binding_sha256 is None
            or wip.final_thread_binding_path is None
            or wip.transcript_chain_sha256 is None
            or wip.app_server_state_tree_sha256 is None
        ):
            raise ContiguousRunnerError(
                "restored WIP lacks its exact app-server thread binding"
            )
        thread_mode: Literal["new", "resume"] = (
            "resume" if wip is not None else "new"
        )
        remaining = (
            None
            if decision.choice.reservation_units is None
            else (
                decision.choice.reservation_units
                / Scheduler.COST_SCALE
            )
        )
        reservation = AttemptReservation(
            schema=RUNNER_SCHEMA,
            campaign_id=state["campaign_id"],
            generation_id=generation_id,
            attempt_id=attempt_id,
            game=game,
            target_level=lane["reached"] + 1,
            authoritative_target=lane["target"],
            parent_checkpoint_path=lane["checkpoint_path"],
            parent_checkpoint_sha256=lane["checkpoint_sha256"],
            frontier_sha256=frontier_sha256(
                game, lane["reached"], lane["checkpoint_sha256"]
            ),
            generation_dir=str(generation),
            input_dir=str(input_dir),
            scratch_dir=str(scratch),
            workspace_dir=str(scratch),
            output_dir=str(output),
            arena_socket_path=str(arena_socket),
            arena_token_file_path=str(arena_token),
            bridge_dir=str(bridge),
            bridge_socket_path=str(bridge_socket),
            bridge_token_file_path=str(bridge_token),
            bridge_policy_receipt_path=str(bridge_policy_receipt),
            host_transcript_path=str(host_transcript),
            app_server_transcript_path=str(app_server_transcript),
            neutral_host_cwd_path=str(neutral_host_cwd),
            app_server_state_dir=str(app_server_state),
            app_server_control_dir=str(app_server_control),
            image_reference=self.backend_configuration.image_reference,
            image_digest=self.backend_configuration.image_digest,
            worker_command=self.backend_configuration.worker_command,
            resource_limits=self.backend_configuration.resource_limits,
            proposer_transport=self.backend_configuration.proposer_transport,
            parent_source_path=lane["source_path"],
            parent_source_tree_sha256=lane["source_tree_sha256"],
            effort=decision.choice.effort,
            soft_allocation_seconds=(
                decision.choice.soft_allocation_seconds
            ),
            wip_mode=decision.choice.effective_wip_mode,
            thread_mode=thread_mode,
            resume_thread_id=(
                wip.codex_thread_id if wip is not None else None
            ),
            resume_thread_binding_sha256=(
                wip.final_thread_binding_sha256
                if wip is not None
                else None
            ),
            wip=wip,
            supervisory_handoff=(
                decision.choice.selected_supervisory_handoff
            ),
            cost_limit_remaining=remaining,
        )
        # This append immediately consumes the decision.  Neither operation
        # creates a generation directory, process, container, or network request.
        self.journal.append(
            event_id=f"{attempt_id}:reserved",
            kind="ATTEMPT_RESERVED",
            payload={
                "attempt_id": attempt_id,
                "reservation": _reservation_to_dict(reservation),
                "scheduler": Scheduler.reservation_binding(decision),
            },
            recorded_at=self.clock(),
        )
        return reservation

    def _reserve_attempt(
        self, state: dict[str, Any]
    ) -> AttemptReservation | None:
        if (
            state.get("operator_incident") is not None
            or state.get("substrate_incident") is not None
            or state.get("storage_incident") is not None
        ):
            return None
        pending_raw = state.get("pending_scheduler_decision")
        if pending_raw is not None:
            try:
                decision = Scheduler.decision_from_dict(pending_raw)
            except Scheduler.SchedulerError as exc:
                raise ContiguousRunnerError(
                    "pending scheduler decision is malformed"
                ) from exc
            return self._consume_scheduler_decision(decision, state)
        decision = self._append_scheduler_decision(state)
        if decision is None:
            return None
        pending_state = self.state()
        return self._consume_scheduler_decision(
            decision, pending_state
        )

    def _auxiliary_observation_ledger_sha256(
        self, state: Mapping[str, Any], game: str
    ) -> str:
        lane = state["lanes"][game]
        receipts = tuple(sorted(
            set(lane["public_observation_receipt_sha256s"])
        ))
        if (
            not receipts
            or any(not _is_sha256(item) for item in receipts)
        ):
            raise ContiguousRunnerError(
                "public observation ledger contains a malformed receipt"
            )
        try:
            import arc_agi3_arena_rpc as arena_contract
        except ImportError as exc:
            raise ContiguousRunnerError(
                "public observation schema is unavailable"
            ) from exc
        selected_frontier_sha256 = frontier_sha256(
            game,
            lane["reached"],
            lane["checkpoint_sha256"],
        )
        by_basis_sha256: dict[str, Mapping[str, Any]] = {}
        for digest in receipts:
            path = self.public_observation_registry / f"{digest}.json"
            try:
                observed_digest, metadata = _sha256_file_identity(path)
                value = _read_json_file(path)
                semantic_digest = (
                    arena_contract.validate_public_observation_receipt(
                        value,
                        game=game,
                        frontier_sha256=selected_frontier_sha256,
                        parent_checkpoint_sha256=(
                            lane["checkpoint_sha256"]
                        ),
                    )
                )
            except Exception as exc:
                raise ContiguousRunnerError(
                    "public observation ledger cannot reopen its exact "
                    "registry receipt"
                ) from exc
            if (
                observed_digest != digest
                or semantic_digest != digest
                or metadata.st_uid != os.getuid()
                or metadata.st_nlink != 1
                or not stat.S_ISREG(metadata.st_mode)
                or stat.S_IMODE(metadata.st_mode) != 0o400
            ):
                raise ContiguousRunnerError(
                    "public observation registry receipt is aliased or "
                    "substituted"
                )
            basis_sha256 = value["public_action_basis_sha256"]
            if basis_sha256 in by_basis_sha256:
                raise ContiguousRunnerError(
                    "public observation ledger duplicates an action basis"
                )
            by_basis_sha256[basis_sha256] = value
        for value in by_basis_sha256.values():
            basis = value["public_action_basis"]
            index = basis["operation_index"]
            prior_basis_sha256 = (
                basis["previous_public_action_basis_sha256"]
            )
            if (
                index == 0
                and prior_basis_sha256
                != arena_contract
                .PUBLIC_ACTION_BASIS_GENESIS_SHA256
            ):
                raise ContiguousRunnerError(
                    "public observation ledger has a non-genesis root"
                )
            if index > 0 and (
                prior_basis_sha256 not in by_basis_sha256
                or by_basis_sha256[prior_basis_sha256][
                    "public_action_basis"
                ]["operation_index"]
                != index - 1
            ):
                raise ContiguousRunnerError(
                    "public observation ledger action basis is discontinuous"
                )
        return Scheduler.public_observation_ledger_sha256(
            game=game,
            frontier_sha256=selected_frontier_sha256,
            parent_checkpoint_sha256=lane["checkpoint_sha256"],
            receipt_sha256s=receipts,
        )

    def _append_auxiliary_event(
        self,
        *,
        event_id: str,
        kind: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        if not kind.startswith("AUXILIARY_"):
            raise ContiguousRunnerError(
                "auxiliary append helper received a non-auxiliary event"
            )
        event = self.journal.append(
            event_id=event_id,
            kind=kind,
            payload=payload,
            recorded_at=self.clock(),
        )
        self._trusted_auxiliary_event_digests.add(event["digest"])
        return event

    def _append_sidecar_request_event(
        self,
        *,
        event_id: str,
        kind: Literal[
            "NATIVE_SIDECAR_REQUEST_ADMITTED",
            "SUPERVISORY_SIDECAR_REQUEST_ADMITTED",
        ],
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        event = self.journal.append(
            event_id=event_id,
            kind=kind,
            payload=payload,
            recorded_at=self.clock(),
        )
        self._trusted_auxiliary_event_digests.add(event["digest"])
        return event

    def _admit_pending_native_sidecar_requests(
        self, state: Mapping[str, Any]
    ) -> int:
        admitted = 0
        origins = {
            row["origin_id"]
            for row in state["sidecar_requests"].values()
            if row["origin_kind"]
            == "NATIVE_SIDECAR_REQUEST_ADMITTED"
        }
        for attempt_id, attempt in sorted(state["attempts"].items()):
            result = attempt.get("settled_result")
            draft = (
                result.native_sidecar_request_draft
                if isinstance(result, AttemptResult)
                else None
            )
            if draft is None or attempt_id in origins:
                continue
            if attempt["phase"] != "CLOSED":
                raise ContiguousRunnerError(
                    "native sidecar request draft precedes clean close"
                )
            lane = state["lanes"][draft.game]
            settlement = next(
                (
                    item
                    for item in lane["clean_proposer_settlements"]
                    if item.attempt_id == attempt_id
                ),
                None,
            )
            try:
                request = (
                    Scheduler.native_sidecar_request_from_draft(
                        draft, settlement=settlement
                    )
                    if settlement is not None
                    else None
                )
            except Scheduler.SchedulerError as exc:
                raise ContiguousRunnerError(
                    "native sidecar request cannot be admitted"
                ) from exc
            if (
                request is None
                or not set(
                    request
                    .cited_public_observation_receipt_sha256s
                ).issubset(
                    lane["public_observation_receipt_sha256s"]
                )
            ):
                raise ContiguousRunnerError(
                    "native sidecar request cites unavailable observations"
                )
            self._append_sidecar_request_event(
                event_id=f"{attempt_id}:native-sidecar-request",
                kind="NATIVE_SIDECAR_REQUEST_ADMITTED",
                payload={
                    "attempt_id": attempt_id,
                    "draft":
                        _native_sidecar_request_draft_to_dict(draft),
                    "request": asdict(request),
                },
            )
            admitted += 1
        return admitted

    def _admit_pending_supervisory_sidecar_requests(
        self, state: Mapping[str, Any]
    ) -> int:
        admitted = 0
        origins = {
            row["origin_id"]
            for row in state["sidecar_requests"].values()
            if row["origin_kind"]
            == "SUPERVISORY_SIDECAR_REQUEST_ADMITTED"
        }
        for assignment_id, row in sorted(
            state["auxiliary_assignments"].items()
        ):
            assignment = row["state"]
            if (
                assignment_id in origins
                or assignment.phase != "ADMITTED"
                or assignment.role != Scheduler.SUPERVISORY_PROPOSER_ROLE
                or assignment.output is None
                or assignment.output.supervisory_handoff is None
                or assignment.invalidated
            ):
                continue
            try:
                request = (
                    Scheduler
                    .supervisory_sidecar_request_from_assignment(
                        assignment
                    )
                )
            except Scheduler.SchedulerError as exc:
                raise ContiguousRunnerError(
                    "supervisory sidecar request cannot be admitted"
                ) from exc
            self._append_sidecar_request_event(
                event_id=(
                    f"{assignment_id}:supervisory-sidecar-request"
                ),
                kind="SUPERVISORY_SIDECAR_REQUEST_ADMITTED",
                payload={
                    "assignment_id": assignment_id,
                    "request": asdict(request),
                },
            )
            admitted += 1
        return admitted

    def _append_auxiliary_decision(
        self, state: dict[str, Any]
    ) -> Scheduler.AuxiliaryDecision | None:
        configuration = self.auxiliary_launch_configuration
        if not configuration.automatic_dispatch_enabled:
            return None
        if self.auxiliary_backend is None:
            raise ContiguousRunnerError(
                "enabled auxiliary scheduler lost its backend"
            )
        try:
            Scheduler.require_dispatch_headroom(self.root)
        except Scheduler.SchedulerError as exc:
            raise ContiguousRunnerError(
                "journal lacks auxiliary terminal-evidence headroom"
            ) from exc
        identities = self._fresh_decision_identifiers(
            state,
            (
                "auxiliary decision",
                "auxiliary assignment",
                "auxiliary reservation",
                "auxiliary expert",
                "auxiliary thread",
            ),
        )
        snapshot = self._scheduler_snapshot_from_state(state)
        auxiliary_frontier = Scheduler.choose_auxiliary_frontier(
            snapshot,
            supervisory_enabled=(
                configuration.supervisory_proposer
                .automatic_dispatch_enabled
            ),
            supervisory_max_concurrency=(
                configuration.supervisory_proposer.max_concurrency
            ),
        )
        # This supplies only a deterministic public-observation commitment for
        # the exact scheduler-selected candidate and cannot trigger a sidecar.
        observation_ledger_sha256 = (
            self._auxiliary_observation_ledger_sha256(
                state,
                auxiliary_frontier.game,
            )
            if auxiliary_frontier is not None
            else Scheduler.sha256_json({
                "schema": 1,
                "kind": "no_auxiliary_candidate",
            })
        )
        try:
            decision = Scheduler.build_auxiliary_decision(
                snapshot,
                decision_id=identities[0],
                assignment_id=identities[1],
                reservation_id=identities[2],
                expert_id=identities[3],
                thread_id=identities[4],
                observation_ledger_sha256=observation_ledger_sha256,
                launch_configuration=configuration,
            )
        except Scheduler.SchedulerError as exc:
            raise ContiguousRunnerError(
                "canonical scheduler could not build an auxiliary decision"
            ) from exc
        if decision is None:
            return None
        self._append_auxiliary_event(
            event_id=f"{decision.decision_id}:auxiliary-decision",
            kind="AUXILIARY_DECISION",
            payload={
                "decision":
                    Scheduler.auxiliary_decision_to_dict(decision)
            },
        )
        return decision

    def _consume_auxiliary_decision(
        self,
        decision: Scheduler.AuxiliaryDecision,
        state: dict[str, Any],
    ) -> str:
        pending_raw = state.get("pending_auxiliary_decision")
        try:
            pending = Scheduler.auxiliary_decision_from_dict(pending_raw)
        except Scheduler.SchedulerError as exc:
            raise ContiguousRunnerError(
                "pending auxiliary decision cannot be recovered"
            ) from exc
        if pending != decision:
            raise ContiguousRunnerError(
                "scheduler attempted to consume a different auxiliary "
                "decision"
            )
        assignment_root = self.auxiliary / decision.assignment_id
        if assignment_root.exists() or assignment_root.is_symlink():
            raise ContiguousRunnerError(
                "auxiliary assignment directory exists before reservation"
            )
        self._append_auxiliary_event(
            event_id=f"{decision.assignment_id}:reserved",
            kind="AUXILIARY_RESERVED",
            payload={
                "assignment_id": decision.assignment_id,
                "reservation":
                    Scheduler.auxiliary_reservation_projection(decision),
            },
        )
        return decision.assignment_id

    def _reserve_auxiliary(
        self, state: dict[str, Any]
    ) -> str | None:
        if (
            state.get("operator_incident") is not None
            or state.get("substrate_incident") is not None
            or state.get("storage_incident") is not None
        ):
            return None
        pending_raw = state.get("pending_auxiliary_decision")
        if pending_raw is not None:
            try:
                decision = Scheduler.auxiliary_decision_from_dict(
                    pending_raw
                )
            except Scheduler.SchedulerError as exc:
                raise ContiguousRunnerError(
                    "pending auxiliary decision is malformed"
                ) from exc
            return self._consume_auxiliary_decision(decision, state)
        decision = self._append_auxiliary_decision(state)
        if decision is None:
            return None
        return self._consume_auxiliary_decision(
            decision, self.state()
        )

    @staticmethod
    def _validate_auxiliary_value(
        value: object, expected_type: type, label: str
    ) -> Any:
        if not isinstance(value, expected_type):
            raise ContiguousRunnerError(
                f"auxiliary backend returned invalid {label}"
            )
        return value

    def _call_auxiliary_backend(
        self,
        operation: str,
        assignment_id: str,
        callback: Callable[[], Any],
        *,
        cleanup: bool = False,
    ) -> Any:
        try:
            return callback()
        except AuxiliaryBackendFatalError as exc:
            if not cleanup:
                self._record_circuit_failure(
                    attempt_id=assignment_id,
                    operation=f"auxiliary_{operation}",
                    fault_domain=self._classify_fault_domain(
                        f"auxiliary_{operation}", exc
                    ),
                )
            raise
        except Exception as exc:
            if not cleanup:
                self._record_circuit_failure(
                    attempt_id=assignment_id,
                    operation=f"auxiliary_{operation}",
                    fault_domain=self._classify_fault_domain(
                        f"auxiliary_{operation}", exc
                    ),
                )
            raise ContiguousRunnerError(
                f"auxiliary backend {operation} failed"
            ) from exc

    @staticmethod
    def _auxiliary_fatal_reason(
        exc: AuxiliaryBackendFatalError,
        fallback: str,
    ) -> str:
        candidate = str(exc)
        if (
            len(candidate) <= 128
            and re.fullmatch(
                r"[a-z][a-z0-9_]{0,127}", candidate
            )
            is not None
        ):
            return candidate
        return fallback

    def _verify_auxiliary_receipt(
        self,
        decision: Scheduler.AuxiliaryDecision,
        path_value: object,
        digest_value: object,
        *,
        expected: Mapping[str, object],
        label: str,
        canonical_newline: bool = True,
    ) -> None:
        if (
            not isinstance(path_value, str)
            or not Path(path_value).is_absolute()
            or not _is_sha256(digest_value)
        ):
            raise ContiguousRunnerError(
                f"{label} path/digest is malformed"
            )
        backend = self.auxiliary_backend
        reader = (
            None
            if backend is None
            else getattr(backend, "read_confined_receipt", None)
        )
        if not callable(reader):
            raise ContiguousRunnerError(
                f"{label} lacks descriptor-confined read authority"
            )
        try:
            raw = reader(
                decision,
                path_value,
                maximum=MAX_AUXILIARY_RECEIPT_BYTES,
            )
        except AuxiliaryBackendFatalError as exc:
            raise ContiguousRunnerError(
                f"{label} descriptor confinement failed"
            ) from exc
        except Exception as exc:
            raise ContiguousRunnerError(
                f"{label} descriptor-confined read failed"
            ) from exc
        if not isinstance(raw, bytes) or not raw:
            raise ContiguousRunnerError(
                f"{label} descriptor-confined bytes are malformed"
            )
        if hashlib.sha256(raw).hexdigest() != digest_value:
            raise ContiguousRunnerError(f"{label} digest changed")
        try:
            value = Transport.strict_json_loads(raw)
        except Exception as exc:
            raise ContiguousRunnerError(
                f"{label} is not strict JSON"
            ) from exc
        canonical = _canonical_json(value)
        if canonical_newline:
            canonical += b"\n"
        if raw != canonical:
            raise ContiguousRunnerError(
                f"{label} encoding is not canonical"
            )
        if not isinstance(value, dict) or value != dict(expected):
            raise ContiguousRunnerError(
                f"{label} is not the exact host-bound receipt"
            )

    def _verify_auxiliary_prepared_receipts(
        self,
        decision: Scheduler.AuxiliaryDecision,
        prepared: AuxiliaryPreparedInput,
    ) -> None:
        self._verify_auxiliary_receipt(
            decision,
            prepared.input_manifest_path,
            prepared.input_manifest_sha256,
            expected=json.loads(
                _canonical_json(asdict(decision.input_manifest))
            ),
            label="auxiliary input manifest",
            canonical_newline=False,
        )
        if (
            prepared.input_manifest_sha256
            != decision.input_manifest_sha256
        ):
            raise ContiguousRunnerError(
                "materialized auxiliary manifest differs from its decision"
            )
        bundle_expected = {
            "schema": 1,
            "kind": "auxiliary_private_input_bundle",
            "assignment_id": decision.assignment_id,
            "frontier_sha256": decision.frontier_sha256,
            "parent_checkpoint_sha256":
                decision.parent_checkpoint_sha256,
            "input_manifest_sha256": decision.input_manifest_sha256,
            "observation_ledger_sha256":
                decision.observation_ledger_sha256,
            "input_bundle_contract_sha256":
                decision.input_bundle_contract_sha256,
            "immutable_inputs": True,
            "live_lineage_mounted": False,
            "public_observations_only": True,
        }
        self._verify_auxiliary_receipt(
            decision,
            prepared.input_bundle_receipt_path,
            prepared.input_bundle_receipt_sha256,
            expected=bundle_expected,
            label="auxiliary private input bundle",
        )

    def _verify_auxiliary_launch_receipt(
        self,
        decision: Scheduler.AuxiliaryDecision,
        launched: AuxiliaryLaunch,
    ) -> None:
        expected = {
            "schema": 1,
            "kind": "auxiliary_backend_launch",
            "assignment_id": decision.assignment_id,
            "backend_contract_sha256":
                decision.backend_contract_sha256,
            "expert_id": decision.expert_id,
            "thread_id": decision.thread_id,
            "model": decision.model,
            "reasoning_effort": decision.reasoning_effort,
            "fresh_context": True,
            "live_lineage_write_authority": False,
        }
        self._verify_auxiliary_receipt(
            decision,
            launched.launch_receipt_path,
            launched.launch_receipt_sha256,
            expected=expected,
            label="auxiliary backend launch",
        )

    def _verify_auxiliary_teardown_receipt(
        self,
        decision: Scheduler.AuxiliaryDecision,
        teardown: AuxiliaryTeardown,
        output: Scheduler.AuxiliaryOutputEvidence,
    ) -> None:
        expected = {
            "schema": 1,
            "kind": "auxiliary_backend_teardown",
            "assignment_id": decision.assignment_id,
            "backend_contract_sha256":
                decision.backend_contract_sha256,
            "output_manifest_sha256":
                output.output_manifest_sha256,
            "descendants_absent": True,
            "live_lineage_mutated": False,
        }
        self._verify_auxiliary_receipt(
            decision,
            teardown.teardown_receipt_path,
            teardown.teardown_receipt_sha256,
            expected=expected,
            label="auxiliary backend teardown",
        )

    def _rebind_auxiliary_prerequisites(
        self,
        assignment: Mapping[str, Any],
        *,
        include_teardown: bool = False,
    ) -> None:
        """Reopen journaled prerequisites before every driver operation.

        This is intentionally repeated rather than cached by the runner.
        After an operator restart the production backend's first exact-byte
        read rebinds each path beneath the pinned assignment-root descriptor;
        subsequent reads also detect component replacement.
        """

        decision = assignment["decision"]
        prepared = assignment["prepared"]
        launched = assignment["launched"]
        phase = assignment["state"].phase
        if phase != "RESERVED":
            if not isinstance(prepared, AuxiliaryPreparedInput):
                raise ContiguousRunnerError(
                    "auxiliary phase lacks prepared prerequisite"
                )
            self._verify_auxiliary_prepared_receipts(
                decision, prepared
            )
        if phase in {"RUNNING", "QUARANTINED"}:
            if not isinstance(launched, AuxiliaryLaunch):
                raise ContiguousRunnerError(
                    "auxiliary phase lacks launch prerequisite"
                )
            self._verify_auxiliary_launch_receipt(decision, launched)
        if include_teardown:
            teardown = assignment["teardown"]
            output = assignment["state"].output
            if (
                phase != "QUARANTINED"
                or not isinstance(teardown, AuxiliaryTeardown)
                or output is None
            ):
                raise ContiguousRunnerError(
                    "auxiliary admission lacks teardown prerequisite"
                )
            self._verify_auxiliary_teardown_receipt(
                decision, teardown, output
            )

    def _write_auxiliary_host_receipt(
        self,
        assignment_id: str,
        filename: str,
        body: Mapping[str, object],
    ) -> tuple[str, str]:
        if not _safe_identifier(assignment_id):
            raise ContiguousRunnerError(
                "auxiliary receipt assignment is malformed"
            )
        host = self.auxiliary / assignment_id / "host"
        if host.is_symlink() or (
            host.exists() and not host.is_dir()
        ):
            raise ContiguousRunnerError(
                "auxiliary host receipt directory is unsafe"
            )
        host.mkdir(parents=True, exist_ok=True)
        os.chmod(host.parent, 0o700, follow_symlinks=False)
        os.chmod(host, 0o700, follow_symlinks=False)
        path = host / filename
        if path.exists():
            if path.is_symlink() or _read_json_file(path) != dict(body):
                raise ContiguousRunnerError(
                    "idempotent auxiliary receipt differs"
                )
        else:
            _write_new_file(path, dict(body))
            os.chmod(path, 0o400, follow_symlinks=False)
        return str(path), _sha256_file(path)

    def _prepare_auxiliary(
        self, assignment: Mapping[str, Any]
    ) -> None:
        backend = self.auxiliary_backend
        if backend is None:
            raise ContiguousRunnerError(
                "cannot prepare auxiliary input without a backend"
            )
        decision = assignment["decision"]
        try:
            prepared_value = self._call_auxiliary_backend(
                "prepare",
                decision.assignment_id,
                lambda: backend.prepare(decision),
            )
        except AuxiliaryBackendFatalError as exc:
            self._abort_auxiliary(
                assignment,
                reason=self._auxiliary_fatal_reason(
                    exc, "fatal_input_builder_failure"
                ),
            )
            return
        prepared = self._validate_auxiliary_value(
            prepared_value,
            AuxiliaryPreparedInput,
            "prepared input",
        )
        self._verify_auxiliary_prepared_receipts(
            decision, prepared
        )
        self._append_auxiliary_event(
            event_id=f"{decision.assignment_id}:input-prepared",
            kind="AUXILIARY_INPUT_PREPARED",
            payload={
                "assignment_id": decision.assignment_id,
                **asdict(prepared),
            },
        )
        self._record_circuit_success(
            attempt_id=decision.assignment_id,
            operation="auxiliary_prepare",
            evidence_kind="auxiliary_input_prepared",
        )

    def _launch_auxiliary(
        self, assignment: Mapping[str, Any]
    ) -> None:
        backend = self.auxiliary_backend
        if backend is None:
            raise ContiguousRunnerError(
                "cannot launch auxiliary without a backend"
            )
        decision = assignment["decision"]
        prepared = assignment["prepared"]
        if not isinstance(prepared, AuxiliaryPreparedInput):
            raise ContiguousRunnerError(
                "auxiliary launch lacks prepared private input"
            )
        self._rebind_auxiliary_prerequisites(assignment)
        try:
            launched_value = self._call_auxiliary_backend(
                "launch",
                decision.assignment_id,
                lambda: backend.launch(decision, prepared),
            )
        except AuxiliaryBackendFatalError as exc:
            self._abort_auxiliary(
                assignment,
                reason=self._auxiliary_fatal_reason(
                    exc, "fatal_launch_failure"
                ),
            )
            return
        launched = self._validate_auxiliary_value(
            launched_value,
            AuxiliaryLaunch,
            "launch proof",
        )
        self._verify_auxiliary_launch_receipt(decision, launched)
        self._append_auxiliary_event(
            event_id=f"{decision.assignment_id}:launched",
            kind="AUXILIARY_LAUNCHED",
            payload={
                "assignment_id": decision.assignment_id,
                **asdict(launched),
            },
        )
        self._record_circuit_success(
            attempt_id=decision.assignment_id,
            operation="auxiliary_launch",
            evidence_kind="auxiliary_launched",
        )

    def _abort_auxiliary(
        self,
        assignment: Mapping[str, Any],
        *,
        reason: str,
    ) -> None:
        backend = self.auxiliary_backend
        if backend is None:
            raise ContiguousRunnerError(
                "cannot abort auxiliary without a backend"
            )
        if (
            not isinstance(reason, str)
            or not reason
            or len(reason) > 4096
            or "\x00" in reason
        ):
            raise ContiguousRunnerError(
                "auxiliary abort reason is malformed"
            )
        state = assignment["state"]
        prior_phase = state.phase
        if prior_phase not in Scheduler.AUXILIARY_ACTIVE_PHASES:
            raise ContiguousRunnerError(
                "auxiliary abort targets a terminal assignment"
            )
        decision = assignment["decision"]
        prepared = assignment["prepared"]
        launched = assignment["launched"]
        self._rebind_auxiliary_prerequisites(assignment)
        aborted = self._validate_auxiliary_value(
            self._call_auxiliary_backend(
                "abort",
                decision.assignment_id,
                lambda: backend.abort(
                    decision,
                    prepared,
                    launched,
                    prior_phase=prior_phase,
                    reason=reason,
                ),
                cleanup=True,
            ),
            AuxiliaryAbort,
            "abort proof",
        )
        try:
            charged_units = Scheduler.charge_to_units(
                aborted.cost_used
            )
        except Scheduler.SchedulerError as exc:
            raise ContiguousRunnerError(
                "auxiliary abort returned invalid authenticated usage"
            ) from exc
        teardown_path: str | None = None
        teardown_sha256: str | None = None
        if prior_phase == "RUNNING":
            teardown = self._validate_auxiliary_value(
                aborted.teardown,
                AuxiliaryTeardown,
                "abort teardown",
            )
            expected_teardown = {
                "schema": 1,
                "kind": "auxiliary_backend_abort_teardown",
                "assignment_id": decision.assignment_id,
                "backend_contract_sha256":
                    decision.backend_contract_sha256,
                "prior_phase": prior_phase,
                "descendants_absent": True,
                "live_lineage_mutated": False,
            }
            self._verify_auxiliary_receipt(
                decision,
                teardown.teardown_receipt_path,
                teardown.teardown_receipt_sha256,
                expected=expected_teardown,
                label="auxiliary abort teardown",
            )
            teardown_path = teardown.teardown_receipt_path
            teardown_sha256 = teardown.teardown_receipt_sha256
        elif aborted.teardown is not None:
            raise ContiguousRunnerError(
                "unlaunched auxiliary abort returned teardown evidence"
            )
        abort_body = {
            "schema": 1,
            "kind": "auxiliary_assignment_abort",
            "authority": "host_only",
            "assignment_id": decision.assignment_id,
            "frontier_sha256": decision.frontier_sha256,
            "parent_checkpoint_sha256":
                decision.parent_checkpoint_sha256,
            "prior_phase": prior_phase,
            "reason": reason,
            "invalidated": state.invalidated,
            "backend_contract_sha256":
                decision.backend_contract_sha256,
            "teardown_receipt_sha256": teardown_sha256,
            "verdict": "ABORTED",
        }
        abort_path, abort_sha256 = (
            self._write_auxiliary_host_receipt(
                decision.assignment_id,
                "abort_receipt.json",
                abort_body,
            )
        )
        self._append_auxiliary_event(
            event_id=f"{decision.assignment_id}:aborted",
            kind="AUXILIARY_ABORTED",
            payload={
                "assignment_id": decision.assignment_id,
                "prior_phase": prior_phase,
                "reason": reason,
                "cost_used": float(aborted.cost_used),
                "authenticated_cost_units": charged_units,
                "budget_reservation_id": decision.reservation_id,
                "auxiliary_decision_id": decision.decision_id,
                "abort_receipt_path": abort_path,
                "abort_receipt_sha256": abort_sha256,
                "teardown_receipt_path": teardown_path,
                "teardown_receipt_sha256": teardown_sha256,
            },
        )

    def _poll_auxiliary(
        self, assignment: Mapping[str, Any]
    ) -> bool:
        backend = self.auxiliary_backend
        if backend is None:
            raise ContiguousRunnerError(
                "cannot poll auxiliary without a backend"
            )
        decision = assignment["decision"]
        prepared = assignment["prepared"]
        launched = assignment["launched"]
        if (
            not isinstance(prepared, AuxiliaryPreparedInput)
            or not isinstance(launched, AuxiliaryLaunch)
        ):
            raise ContiguousRunnerError(
                "running auxiliary lacks prepare/launch evidence"
            )
        self._rebind_auxiliary_prerequisites(assignment)
        try:
            terminal = self._validate_auxiliary_value(
                self._call_auxiliary_backend(
                    "poll",
                    decision.assignment_id,
                    lambda: backend.poll(
                        decision,
                        prepared,
                        launched,
                        timeout_seconds=POLL_TIMEOUT_SECONDS,
                    ),
                ),
                AuxiliaryPoll,
                "poll result",
            )
        except AuxiliaryBackendFatalError as exc:
            self._abort_auxiliary(
                assignment,
                reason=self._auxiliary_fatal_reason(
                    exc, "fatal_poll_failure"
                ),
            )
            return True
        if (
            terminal.status
            not in {"running", "exited", "containment_fault"}
            or not _is_sha256(terminal.observation_sha256)
        ):
            raise ContiguousRunnerError(
                "auxiliary poll result is malformed"
            )
        if terminal.status == "running":
            return False
        if terminal.status == "containment_fault":
            self._abort_auxiliary(
                assignment,
                reason=terminal.reason or "containment_fault",
            )
            return True
        self._rebind_auxiliary_prerequisites(assignment)
        try:
            collection = self._validate_auxiliary_value(
                self._call_auxiliary_backend(
                    "collect",
                    decision.assignment_id,
                    lambda: backend.collect(
                        decision,
                        prepared,
                        launched,
                        terminal,
                    ),
                ),
                AuxiliaryCollection,
                "collection",
            )
        except AuxiliaryBackendFatalError as exc:
            self._abort_auxiliary(
                assignment,
                reason=self._auxiliary_fatal_reason(
                    exc, "fatal_collection_failure"
                ),
            )
            return True
        if collection.output is None:
            self._abort_auxiliary(
                assignment,
                reason=collection.abort_reason or "unusable_output",
            )
            return True
        if collection.abort_reason is not None:
            raise ContiguousRunnerError(
                "usable auxiliary output also claims an abort"
            )
        try:
            output = Scheduler.validate_auxiliary_output(
                collection.output,
                assignment=assignment["state"],
            )
            charged_units = Scheduler.charge_to_units(
                collection.cost_used
            )
        except Scheduler.SchedulerError as exc:
            raise ContiguousRunnerError(
                "auxiliary collection is not quarantine-only evidence"
            ) from exc
        self._rebind_auxiliary_prerequisites(assignment)
        try:
            teardown = self._validate_auxiliary_value(
                self._call_auxiliary_backend(
                    "teardown",
                    decision.assignment_id,
                    lambda: backend.teardown(
                        decision,
                        prepared,
                        launched,
                        collection,
                    ),
                ),
                AuxiliaryTeardown,
                "teardown proof",
            )
        except AuxiliaryBackendFatalError as exc:
            self._abort_auxiliary(
                assignment,
                reason=self._auxiliary_fatal_reason(
                    exc, "fatal_teardown_failure"
                ),
            )
            return True
        self._verify_auxiliary_teardown_receipt(
            decision, teardown, output
        )
        self._append_auxiliary_event(
            event_id=f"{decision.assignment_id}:quarantined",
            kind="AUXILIARY_RESULT_QUARANTINED",
            payload={
                "assignment_id": decision.assignment_id,
                "output": asdict(output),
                "cost_used": float(collection.cost_used),
                "authenticated_cost_units": charged_units,
                "budget_reservation_id": decision.reservation_id,
                "auxiliary_decision_id": decision.decision_id,
                **asdict(teardown),
            },
        )
        self._record_circuit_success(
            attempt_id=decision.assignment_id,
            operation="auxiliary_collect",
            evidence_kind="auxiliary_result_quarantined",
        )
        self._record_circuit_success(
            attempt_id=decision.assignment_id,
            operation="auxiliary_teardown",
            evidence_kind="auxiliary_result_quarantined",
        )
        return True

    def _admit_auxiliary(
        self, assignment: Mapping[str, Any]
    ) -> None:
        backend = self.auxiliary_backend
        if backend is None:
            raise ContiguousRunnerError(
                "cannot admit auxiliary output without a backend"
            )
        decision = assignment["decision"]
        output = assignment["state"].output
        if output is None:
            raise ContiguousRunnerError(
                "auxiliary admission lacks quarantined output"
            )
        if output.supervisory_handoff is not None:
            try:
                semantic_identity = (
                    Scheduler.supervisory_handoff_semantic_sha256(
                        output
                    )
                )
                duplicate = any(
                    item["state"].phase == "ADMITTED"
                    and not item["state"].invalidated
                    and item["state"].game == decision.game
                    and item["state"].frontier_sha256
                    == decision.frontier_sha256
                    and item["state"].parent_checkpoint_sha256
                    == decision.parent_checkpoint_sha256
                    and item["state"].output is not None
                    and item["state"].output.supervisory_handoff
                    is not None
                    and Scheduler
                    .supervisory_handoff_semantic_sha256(
                        item["state"].output
                    )
                    == semantic_identity
                    for item in self.state()[
                        "auxiliary_assignments"
                    ].values()
                    if item["state"].assignment_id
                    != decision.assignment_id
                )
            except Scheduler.SchedulerError as exc:
                raise ContiguousRunnerError(
                    "supervisory handoff semantic identity is malformed"
                ) from exc
            if duplicate:
                self._reject_auxiliary_host(
                    assignment,
                    reason="duplicate_supervisory_semantic_content",
                )
                return
        self._rebind_auxiliary_prerequisites(
            assignment, include_teardown=True
        )
        try:
            admission = self._validate_auxiliary_value(
                self._call_auxiliary_backend(
                    "admit",
                    decision.assignment_id,
                    lambda: backend.admit(decision, output),
                ),
                AuxiliaryAdmission,
                "admission result",
            )
        except AuxiliaryBackendFatalError as exc:
            self._reject_auxiliary_host(
                assignment,
                reason=self._auxiliary_fatal_reason(
                    exc, "fatal_admission_failure"
                ),
            )
            return
        if admission.verdict == "REJECTED":
            reason = admission.reason
            if (
                not isinstance(reason, str)
                or not reason
                or len(reason) > 4096
                or "\x00" in reason
                or admission.profile is not None
                or any(
                    value is not None
                    for value in (
                        admission.fresh_replay_receipt_path,
                        admission.fresh_replay_receipt_sha256,
                        admission.taint_receipt_path,
                        admission.taint_receipt_sha256,
                        admission.provenance_receipt_path,
                        admission.provenance_receipt_sha256,
                    )
                )
            ):
                raise ContiguousRunnerError(
                    "auxiliary rejection result is malformed"
                )
            rejection_expected = {
                "schema": 1,
                "kind": "auxiliary_output_rejection",
                "authority": "host_only",
                "assignment_id": decision.assignment_id,
                "frontier_sha256": decision.frontier_sha256,
                "parent_checkpoint_sha256":
                    decision.parent_checkpoint_sha256,
                "output_manifest_sha256":
                    output.output_manifest_sha256,
                "admission_contract_sha256":
                    decision.admission_contract_sha256,
                "reason": reason,
                "verdict": "REJECTED",
            }
            self._verify_auxiliary_receipt(
                decision,
                admission.admission_receipt_path,
                admission.admission_receipt_sha256,
                expected=rejection_expected,
                label="auxiliary host rejection",
            )
            self._append_auxiliary_event(
                event_id=f"{decision.assignment_id}:rejected",
                kind="AUXILIARY_OUTPUT_REJECTED",
                payload={
                    "assignment_id": decision.assignment_id,
                    "reason": reason,
                    "admission_receipt_path":
                        admission.admission_receipt_path,
                    "admission_receipt_sha256":
                        admission.admission_receipt_sha256,
                },
            )
            self._record_circuit_success(
                attempt_id=decision.assignment_id,
                operation="auxiliary_admit",
                evidence_kind="auxiliary_output_rejected",
            )
            return
        if admission.verdict != "ADMITTED" or admission.reason is not None:
            raise ContiguousRunnerError(
                "auxiliary admission verdict is malformed"
            )
        common = {
            "schema": 1,
            "assignment_id": decision.assignment_id,
            "frontier_sha256": decision.frontier_sha256,
            "parent_checkpoint_sha256":
                decision.parent_checkpoint_sha256,
            "output_manifest_sha256":
                output.output_manifest_sha256,
        }
        gate_fields = (
            (
                "fresh_replay",
                admission.fresh_replay_receipt_path,
                admission.fresh_replay_receipt_sha256,
                {
                    **common,
                    "kind": "auxiliary_fresh_public_replay",
                    "status": "PASS",
                },
            ),
            (
                "taint",
                admission.taint_receipt_path,
                admission.taint_receipt_sha256,
                {
                    **common,
                    "kind": "auxiliary_taint_scan",
                    "status": "CLEAN",
                },
            ),
            (
                "provenance",
                admission.provenance_receipt_path,
                admission.provenance_receipt_sha256,
                {
                    **common,
                    "kind": "auxiliary_provenance_scan",
                    "status": "PASS",
                },
            ),
        )
        for label, path, digest, expected in gate_fields:
            self._verify_auxiliary_receipt(
                decision,
                path,
                digest,
                expected=expected,
                label=f"auxiliary {label} gate",
            )
        if decision.specialization == "complexity_diagnosis":
            if admission.profile is None:
                raise ContiguousRunnerError(
                    "diagnosis admission lacks a complexity profile"
                )
            try:
                Scheduler.validate_complexity_profile(
                    admission.profile,
                    frontier_sha256=decision.frontier_sha256,
                )
                if (
                    admission.profile.round_index
                    != decision.round_index
                    or admission.profile
                    .observation_receipt_sha256
                    not in output.public_observation_receipt_sha256s
                    or admission.profile.taint_scan_receipt_sha256
                    != admission.taint_receipt_sha256
                ):
                    raise Scheduler.SchedulerError(
                        "complexity profile is not bound to exact admitted "
                        "evidence"
                    )
            except Scheduler.SchedulerError as exc:
                raise ContiguousRunnerError(
                    "diagnosis returned an invalid complexity profile"
                ) from exc
            admitted_evidence_sha256 = Scheduler.sha256_json(
                asdict(admission.profile)
            )
            admission_kind = "auxiliary_profile_admission"
            event_kind = "AUXILIARY_PROFILE_ADMITTED"
        else:
            if admission.profile is not None:
                raise ContiguousRunnerError(
                    "specialist admission unexpectedly returns a profile"
                )
            admitted_evidence_sha256 = Scheduler.sha256_json(
                asdict(output)
            )
            admission_kind = "auxiliary_output_admission"
            event_kind = "AUXILIARY_OUTPUT_ADMITTED"
        admission_expected = {
            **common,
            "kind": admission_kind,
            "authority": "host_only",
            "admission_contract_sha256":
                decision.admission_contract_sha256,
            "fresh_replay_receipt_sha256":
                admission.fresh_replay_receipt_sha256,
            "taint_receipt_sha256":
                admission.taint_receipt_sha256,
            "provenance_receipt_sha256":
                admission.provenance_receipt_sha256,
            "admitted_evidence_sha256": admitted_evidence_sha256,
            "verdict": "ADMITTED",
        }
        self._verify_auxiliary_receipt(
            decision,
            admission.admission_receipt_path,
            admission.admission_receipt_sha256,
            expected=admission_expected,
            label="auxiliary host admission",
        )
        payload: dict[str, object] = {
            "assignment_id": decision.assignment_id,
            "admitted_evidence_sha256": admitted_evidence_sha256,
            "fresh_replay_receipt_path":
                admission.fresh_replay_receipt_path,
            "fresh_replay_receipt_sha256":
                admission.fresh_replay_receipt_sha256,
            "taint_receipt_path": admission.taint_receipt_path,
            "taint_receipt_sha256":
                admission.taint_receipt_sha256,
            "provenance_receipt_path":
                admission.provenance_receipt_path,
            "provenance_receipt_sha256":
                admission.provenance_receipt_sha256,
            "admission_receipt_path":
                admission.admission_receipt_path,
            "admission_receipt_sha256":
                admission.admission_receipt_sha256,
        }
        if admission.profile is not None:
            payload["profile"] = asdict(admission.profile)
        self._append_auxiliary_event(
            event_id=f"{decision.assignment_id}:admitted",
            kind=event_kind,
            payload=payload,
        )
        self._record_circuit_success(
            attempt_id=decision.assignment_id,
            operation="auxiliary_admit",
            evidence_kind="auxiliary_output_admitted",
        )

    def _reject_auxiliary_host(
        self,
        assignment: Mapping[str, Any],
        *,
        reason: str,
    ) -> None:
        decision = assignment["decision"]
        output = assignment["state"].output
        if output is None:
            raise ContiguousRunnerError(
                "auxiliary host rejection lacks quarantined output"
            )
        if (
            not isinstance(reason, str)
            or not reason
            or len(reason) > 128
            or re.fullmatch(r"[a-z][a-z0-9_]{0,127}", reason)
            is None
        ):
            raise ContiguousRunnerError(
                "auxiliary host rejection reason is malformed"
            )
        body = {
            "schema": 1,
            "kind": "auxiliary_output_rejection",
            "authority": "host_only",
            "assignment_id": decision.assignment_id,
            "frontier_sha256": decision.frontier_sha256,
            "parent_checkpoint_sha256":
                decision.parent_checkpoint_sha256,
            "output_manifest_sha256": output.output_manifest_sha256,
            "admission_contract_sha256":
                decision.admission_contract_sha256,
            "reason": reason,
            "verdict": "REJECTED",
        }
        path, digest = self._write_auxiliary_host_receipt(
            decision.assignment_id,
            "stale_rejection_receipt.json",
            body,
        )
        self._append_auxiliary_event(
            event_id=f"{decision.assignment_id}:stale-rejected",
            kind="AUXILIARY_OUTPUT_REJECTED",
            payload={
                "assignment_id": decision.assignment_id,
                "reason": reason,
                "admission_receipt_path": path,
                "admission_receipt_sha256": digest,
            },
        )

    def _reject_auxiliary_stale(
        self, assignment: Mapping[str, Any]
    ) -> None:
        self._reject_auxiliary_host(
            assignment, reason="frontier_promoted"
        )

    @staticmethod
    def _classify_fault_domain(
        operation: str, exc: Exception
    ) -> str:
        """Map one typed failure to a small policy-hashed fault domain.

        Exception messages are deliberately ignored: they are untrusted,
        unstable, and may contain credentials.  The operation plus the class
        lineage is sufficient for deterministic safety throttling.
        """

        class_lineage = "_".join(
            cls.__name__.lower() for cls in type(exc).__mro__
        )
        provider_boundary = (
            operation.startswith("backend_")
            or operation.startswith("auxiliary_")
        )
        if provider_boundary and any(
            marker in class_lineage
            for marker in (
                "authentication",
                "authorization",
                "unauthorized",
                "credential",
                "forbidden",
                "token",
            )
        ):
            return "provider_auth"
        if any(
            marker in class_lineage
            for marker in (
                "containment",
                "container",
                "docker",
                "cgroup",
                "guardian",
            )
        ):
            return "containment_infrastructure"
        if provider_boundary and any(
            marker in class_lineage
            for marker in (
                "ratelimit",
                "rate_limit",
                "capacity",
                "quota",
                "unavailable",
                "timeout",
                "connection",
            )
        ):
            return "provider_availability"
        if provider_boundary and "provider" in class_lineage:
            return "provider_failure"
        return "operation_error"

    def _record_circuit_failure(
        self,
        *,
        attempt_id: str | None,
        operation: str,
        fault_domain: str,
    ) -> None:
        if (
            (attempt_id is not None and not _safe_identifier(attempt_id))
            or not _safe_identifier(operation)
            or fault_domain not in FAILURE_FAULT_DOMAINS
        ):
            raise ContiguousRunnerError(
                "failure circuit identity is malformed"
            )
        state = self.state()
        operation_key = f"{operation}:{fault_domain}"
        operation_state = state["failure_operation_circuits"].get(
            operation_key,
            {
                "consecutive": 0,
                "failure_index": 0,
                "retry_not_before": None,
            },
        )
        domain_state = state["failure_domain_circuits"].get(
            fault_domain,
            {
                "consecutive": 0,
                "failure_index": 0,
                "retry_not_before": None,
                "last_operation": None,
            },
        )
        operation_consecutive = int(
            operation_state["consecutive"]
        ) + 1
        domain_consecutive = int(domain_state["consecutive"]) + 1
        operation_failure_index = int(
            operation_state["failure_index"]
        ) + 1
        domain_failure_index = int(domain_state["failure_index"]) + 1
        backoff_schedule = (
            SUBSTRATE_HEALTH_REPROBE_BACKOFF_SECONDS
            if operation == "substrate_health_reprobe"
            else OPERATION_RETRY_BACKOFF_SECONDS
        )
        backoff_seconds = backoff_schedule[
            min(
                max(operation_consecutive, domain_consecutive),
                len(backoff_schedule),
            )
            - 1
        ]
        recorded_at = float(self.clock())
        self.journal.append(
            event_id=(
                f"failure-circuit:{fault_domain}:"
                f"{domain_failure_index:08d}"
            ),
            kind="FAILURE_CIRCUIT_FAILURE",
            payload={
                "attempt_id": attempt_id,
                "operation": operation,
                "fault_domain": fault_domain,
                "operation_consecutive": operation_consecutive,
                "operation_failure_index":
                    operation_failure_index,
                "domain_consecutive": domain_consecutive,
                "domain_failure_index": domain_failure_index,
                "backoff_seconds": backoff_seconds,
                "retry_not_before": recorded_at + backoff_seconds,
            },
            recorded_at=recorded_at,
        )
        if (
            state["operator_incident"] is None
            and max(operation_consecutive, domain_consecutive)
            >= FAILURE_CIRCUIT_THRESHOLD
        ):
            self.journal.append(
                event_id="campaign:operator-incident",
                kind="OPERATOR_INCIDENT",
                payload={
                    "attempt_id": attempt_id,
                    "operation": operation,
                    "fault_domain": fault_domain,
                    "operation_consecutive":
                        operation_consecutive,
                    "domain_consecutive": domain_consecutive,
                    "threshold": FAILURE_CIRCUIT_THRESHOLD,
                    "reason_code": "failure_circuit_exhausted",
                },
                recorded_at=recorded_at,
            )

    def _record_circuit_success(
        self,
        *,
        attempt_id: str | None,
        operation: str,
        evidence_kind: str,
    ) -> None:
        """Reset only exact operation scopes proven by typed success.

        A global domain resets only when the successful operation is also the
        last operation that failed in that domain.  Thus successes on unrelated
        live lanes cannot mask an alternating-operation outage.
        """

        if (
            (attempt_id is not None and not _safe_identifier(attempt_id))
            or not _safe_identifier(operation)
            or not _safe_identifier(evidence_kind)
        ):
            raise ContiguousRunnerError(
                "failure circuit success identity is malformed"
            )
        state = self.state()
        prefix = operation + ":"
        for operation_key, operation_state in sorted(
            state["failure_operation_circuits"].items()
        ):
            if (
                not operation_key.startswith(prefix)
                or int(operation_state["consecutive"]) == 0
            ):
                continue
            fault_domain = operation_key[len(prefix):]
            domain_state = state["failure_domain_circuits"].get(
                fault_domain
            )
            if not isinstance(domain_state, dict):
                raise ContiguousRunnerError(
                    "failure circuit domain state is missing"
                )
            reset_domain = (
                int(domain_state["consecutive"]) > 0
                and domain_state.get("last_operation") == operation
            )
            self.journal.append(
                event_id=(
                    f"failure-circuit:{fault_domain}:"
                    f"{int(operation_state['failure_index']):08d}:"
                    f"reset:{operation}"
                ),
                kind="FAILURE_CIRCUIT_RESET",
                payload={
                    "attempt_id": attempt_id,
                    "operation": operation,
                    "fault_domain": fault_domain,
                    "operation_consecutive":
                        operation_state["consecutive"],
                    "domain_consecutive":
                        domain_state["consecutive"],
                    "evidence_kind": evidence_kind,
                    "reset_operation": True,
                    "reset_domain": reset_domain,
                },
                recorded_at=self.clock(),
            )

    def _record_retry(
        self, attempt_id: str, operation: str, exc: Exception
    ) -> None:
        attempt = self.state()["attempts"].get(attempt_id)
        if attempt is None:
            raise ContiguousRunnerError(
                "cannot record retry for unknown attempt"
            )
        retry_index = attempt["retry_count"] + 1
        operation_retry_index = (
            attempt["operation_retry_counts"].get(operation, 0) + 1
        )
        bounded_index = min(
            operation_retry_index,
            len(OPERATION_RETRY_BACKOFF_SECONDS),
        )
        backoff_seconds = OPERATION_RETRY_BACKOFF_SECONDS[
            bounded_index - 1
        ]
        error_type = type(exc).__name__
        if not _safe_identifier(error_type):
            error_type = "BackendError"
        self._record_circuit_failure(
            attempt_id=attempt_id,
            operation=operation,
            fault_domain=self._classify_fault_domain(
                operation, exc
            ),
        )
        recorded_at = float(self.clock())
        retry_not_before = recorded_at + backoff_seconds
        self.journal.append(
            event_id=f"{attempt_id}:retry:{retry_index:08d}",
            kind="ATTEMPT_RETRY",
            payload={
                "attempt_id": attempt_id,
                "retry_index": retry_index,
                "operation": operation,
                "operation_retry_index": operation_retry_index,
                "error_type": error_type,
                "backoff_seconds": backoff_seconds,
                "retry_not_before": retry_not_before,
            },
            recorded_at=recorded_at,
        )

    @staticmethod
    def _attempt_operation_ready(
        attempt: Mapping[str, Any],
        operation: str,
        *,
        now: float,
    ) -> bool:
        retry_not_before = attempt.get(
            "operation_retry_not_before", {}
        ).get(operation)
        return (
            retry_not_before is None
            or float(now) >= float(retry_not_before)
        )

    @staticmethod
    def _failure_circuit_operation_ready(
        state: Mapping[str, Any],
        operation: str,
        *,
        now: float,
        cleanup: bool = False,
    ) -> bool:
        if cleanup:
            return True
        prefix = operation + ":"
        retry_not_before = [
            value.get("retry_not_before")
            for key, value in state[
                "failure_operation_circuits"
            ].items()
            if key.startswith(prefix)
        ]
        # Domain backoff is campaign-global by design, so changing attempt,
        # exception subclass, or operation cannot hammer the same outage.
        retry_not_before.extend(
            value.get("retry_not_before")
            for value in state[
                "failure_domain_circuits"
            ].values()
        )
        return all(
            cutoff is None or float(now) >= float(cutoff)
            for cutoff in retry_not_before
        )

    @classmethod
    def _operation_ready(
        cls,
        state: Mapping[str, Any],
        attempt: Mapping[str, Any],
        operation: str,
        *,
        now: float,
        cleanup: bool = False,
    ) -> bool:
        return cls._attempt_operation_ready(
            attempt, operation, now=now
        ) and cls._failure_circuit_operation_ready(
            state, operation, now=now, cleanup=cleanup
        )

    def _materialize_reserved(
        self, reservation: AttemptReservation
    ) -> AttemptSpec:
        generation = Path(reservation.generation_dir)
        if generation.is_symlink() or (
            generation.exists() and not generation.is_dir()
        ):
            raise ContiguousRunnerError(
                "reserved generation path is not a regular directory"
            )
        generation.mkdir(mode=0o700, exist_ok=True)
        os.chmod(generation, 0o700, follow_symlinks=False)
        children = (
            Path(reservation.input_dir),
            Path(reservation.scratch_dir),
            Path(reservation.output_dir),
            Path(reservation.arena_socket_path).parent,
            Path(reservation.bridge_dir),
            Path(reservation.host_transcript_path).parent,
            Path(reservation.app_server_state_dir).parent,
        )
        for path in children:
            if path.is_symlink() or (
                path.exists() and not path.is_dir()
            ):
                raise ContiguousRunnerError(
                    f"reserved child is not a regular directory: {path}"
                )
            path.mkdir(mode=0o700, exist_ok=True)
            os.chmod(path, 0o700, follow_symlinks=False)
        app_server_state = Path(
            reservation.app_server_state_dir
        )
        if app_server_state.is_symlink() or (
            app_server_state.exists()
            and not app_server_state.is_dir()
        ):
            raise ContiguousRunnerError(
                "staged app-server state root is unsafe"
            )
        app_server_state.mkdir(mode=0o700, exist_ok=True)
        os.chmod(app_server_state, 0o700, follow_symlinks=False)
        if reservation.wip is not None:
            prior_state = Path(
                reservation.wip.app_server_state_dir or ""
            )
            if (
                not prior_state.is_absolute()
                or prior_state == app_server_state
            ):
                raise ContiguousRunnerError(
                    "resume state is not a distinct immutable generation"
                )
            try:
                Contract._validate_regular_tree(
                    prior_state, label="committed app-server state"
                )
            except Exception as exc:
                raise ContiguousRunnerError(
                    "committed app-server state is unsafe"
                ) from exc
            if (
                Contract._tree_hash(prior_state)
                != reservation.wip.app_server_state_tree_sha256
            ):
                raise ContiguousRunnerError(
                    "committed app-server state changed before staging"
                )
            _copy_regular_tree(
                prior_state,
                app_server_state,
                overwrite=False,
                maximum_files=MAX_APP_SERVER_STATE_FILES,
                maximum_file_bytes=
                    MAX_APP_SERVER_STATE_FILE_BYTES,
                maximum_total_bytes=
                    MAX_APP_SERVER_STATE_TOTAL_BYTES,
                allow_hidden_state_paths=True,
            )
            if (
                Contract._tree_hash(app_server_state)
                != reservation.wip.app_server_state_tree_sha256
            ):
                raise ContiguousRunnerError(
                    "staged app-server state copy is incomplete"
                )
        layout = AttemptLayout(
            campaign_id=reservation.campaign_id,
            generation_id=reservation.generation_id,
            attempt_id=reservation.attempt_id,
            game=reservation.game,
            target_level=reservation.target_level,
            authoritative_target=reservation.authoritative_target,
            parent_checkpoint_path=reservation.parent_checkpoint_path,
            parent_checkpoint_sha256=(
                reservation.parent_checkpoint_sha256
            ),
            frontier_sha256=reservation.frontier_sha256,
            generation_dir=reservation.generation_dir,
            input_dir=reservation.input_dir,
            scratch_dir=reservation.scratch_dir,
            workspace_dir=reservation.workspace_dir,
            output_dir=reservation.output_dir,
            arena_socket_path=reservation.arena_socket_path,
            arena_token_file_path=reservation.arena_token_file_path,
            bridge_dir=reservation.bridge_dir,
            bridge_socket_path=reservation.bridge_socket_path,
            bridge_token_file_path=reservation.bridge_token_file_path,
            bridge_policy_receipt_path=(
                reservation.bridge_policy_receipt_path
            ),
            host_transcript_path=reservation.host_transcript_path,
            app_server_transcript_path=(
                reservation.app_server_transcript_path
            ),
            neutral_host_cwd_path=reservation.neutral_host_cwd_path,
            app_server_state_dir=reservation.app_server_state_dir,
            app_server_control_dir=(
                reservation.app_server_control_dir
            ),
            parent_source_path=reservation.parent_source_path,
            parent_source_tree_sha256=(
                reservation.parent_source_tree_sha256
            ),
            effort=reservation.effort,
            soft_allocation_seconds=(
                reservation.soft_allocation_seconds
            ),
            wip_mode=reservation.wip_mode,
            thread_mode=reservation.thread_mode,
            resume_thread_id=reservation.resume_thread_id,
            resume_thread_binding_sha256=(
                reservation.resume_thread_binding_sha256
            ),
            proposer_transport=reservation.proposer_transport,
            wip=reservation.wip,
            supervisory_handoff=reservation.supervisory_handoff,
        )
        try:
            receipt = self.input_builder.prepare(layout)
        except BaseException as exc:
            if isinstance(exc, SimulatedCrash):
                raise
            if not isinstance(exc, Exception):
                raise
            self._record_retry(
                reservation.attempt_id, "input_materialize", exc
            )
            raise ContiguousRunnerError(
                "trusted input-bundle preparation remains recoverable"
            ) from exc
        if not isinstance(receipt, InputBundleReceipt):
            raise ContiguousRunnerError(
                "input builder returned the wrong receipt type"
            )
        arena_socket = Path(reservation.arena_socket_path)
        arena_token = Path(reservation.arena_token_file_path)
        host_transcript = Path(reservation.host_transcript_path)
        scratch_root = Path(reservation.scratch_dir)
        output_root = Path(reservation.output_dir)
        rpc_root = arena_socket.parent
        host_root = host_transcript.parent
        bridge_root = Path(reservation.bridge_dir)
        allowed_generation_entries = {
            "input",
            "scratch",
            "output",
            "rpc",
            "bridge",
            "host",
            "state",
            "input_bundle_receipt.json",
            "attempt_spec.json",
        }
        if (
            Path(receipt.receipt_path) != generation
            / "input_bundle_receipt.json"
            or not _is_sha256(receipt.receipt_sha256)
            or not _is_sha256(receipt.input_tree_sha256)
            or not _is_sha256(
                receipt.parent_source_tree_sha256
            )
            or not _is_sha256(
                receipt.initial_workspace_tree_sha256
            )
            or Path(receipt.frontier_brief_path)
            != Path(reservation.input_dir) / "frontier_brief.json"
            or not _is_sha256(receipt.frontier_brief_sha256)
            or Path(receipt.bridge_policy_path)
            != Path(reservation.input_dir) / "bridge_policy.json"
            or not _is_sha256(receipt.bridge_policy_sha256)
            or not isinstance(receipt.parent_action_count, int)
            or isinstance(receipt.parent_action_count, bool)
            or not 0 <= receipt.parent_action_count <= 600
            or not isinstance(receipt.remaining_action_budget, int)
            or isinstance(receipt.remaining_action_budget, bool)
            or receipt.remaining_action_budget
            != 600 - receipt.parent_action_count
            or not isinstance(receipt.fresh_prefix_required, bool)
            or receipt.fresh_prefix_required
            is not (receipt.remaining_action_budget == 0)
            or receipt.parent_checkpoint_sha256
            != reservation.parent_checkpoint_sha256
            or receipt.parent_source_tree_sha256
            != reservation.parent_source_tree_sha256
            or receipt.wip_tree_sha256
            != (
                reservation.wip.wip_tree_sha256
                if reservation.wip else None
            )
            or receipt.wip_solver_source_tree_sha256
            != (
                reservation.wip.solver_source_tree_sha256
                if reservation.wip else None
            )
            or (
                reservation.supervisory_handoff is None
                and any(
                    item is not None
                    for item in (
                        receipt.supervisory_handoff_path,
                        receipt.supervisory_handoff_sha256,
                        receipt
                        .supervisory_handoff_binding_receipt_path,
                        receipt
                        .supervisory_handoff_binding_receipt_sha256,
                    )
                )
            )
            or (
                reservation.supervisory_handoff is not None
                and (
                    Path(
                        str(receipt.supervisory_handoff_path)
                    )
                    != Path(reservation.input_dir)
                    / "supervisory_handoff.json"
                    or not _is_sha256(
                        receipt.supervisory_handoff_sha256
                    )
                    or Path(
                        str(
                            receipt
                            .supervisory_handoff_binding_receipt_path
                        )
                    )
                    != Path(reservation.input_dir)
                    / "supervisory_handoff_binding_receipt.json"
                    or not _is_sha256(
                        receipt
                        .supervisory_handoff_binding_receipt_sha256
                    )
                )
            )
            or arena_socket.exists()
            or arena_token.exists()
            or host_transcript.exists()
            or Contract._tree_hash(scratch_root)
            != receipt.initial_workspace_tree_sha256
            or any(output_root.iterdir())
            or any(rpc_root.iterdir())
            or any(bridge_root.iterdir())
            or any(host_root.iterdir())
            or any(
                entry.name not in allowed_generation_entries
                for entry in generation.iterdir()
            )
        ):
            raise ContiguousRunnerError(
                "input builder violated the fresh-generation contract"
            )
        common = {
            name: getattr(reservation, name)
            for name in AttemptReservation.__dataclass_fields__
        }
        initial_app_server_state_tree_sha256 = Contract._tree_hash(
            app_server_state
        )
        spec = AttemptSpec(
            **common,
            input_tree_sha256=receipt.input_tree_sha256,
            initial_workspace_tree_sha256=
                receipt.initial_workspace_tree_sha256,
            initial_app_server_state_tree_sha256=(
                initial_app_server_state_tree_sha256
            ),
            hard_safety_seconds=(
                Taint.APP_SERVER_HARD_SAFETY_SECONDS
            ),
            max_auth_refreshes=Taint.MAX_AUTH_REFRESHES,
            input_bundle_receipt_path=receipt.receipt_path,
            input_bundle_receipt_sha256=receipt.receipt_sha256,
            frontier_brief_path=receipt.frontier_brief_path,
            frontier_brief_sha256=receipt.frontier_brief_sha256,
            supervisory_handoff_path=(
                receipt.supervisory_handoff_path
            ),
            supervisory_handoff_sha256=(
                receipt.supervisory_handoff_sha256
            ),
            supervisory_handoff_binding_receipt_path=(
                receipt
                .supervisory_handoff_binding_receipt_path
            ),
            supervisory_handoff_binding_receipt_sha256=(
                receipt
                .supervisory_handoff_binding_receipt_sha256
            ),
            bridge_policy_path=receipt.bridge_policy_path,
            bridge_policy_sha256=receipt.bridge_policy_sha256,
            parent_action_count=receipt.parent_action_count,
            remaining_action_budget=(
                receipt.remaining_action_budget
            ),
            fresh_prefix_required=receipt.fresh_prefix_required,
        )
        self._validate_prepared_input(spec)
        spec_path = generation / "attempt_spec.json"
        if spec_path.exists():
            if _read_json_file(spec_path) != json.loads(
                _canonical_json(_spec_to_dict(spec))
            ):
                raise ContiguousRunnerError(
                    "recovered attempt spec differs from reservation"
                )
        else:
            _write_new_file(spec_path, _spec_to_dict(spec))
        _fsync_directory(generation)
        self.journal.append(
            event_id=f"{reservation.attempt_id}:prepared",
            kind="ATTEMPT_PREPARED",
            payload={
                "attempt_id": reservation.attempt_id,
                "spec": _spec_to_dict(spec),
            },
            recorded_at=self.clock(),
        )
        self._record_circuit_success(
            attempt_id=reservation.attempt_id,
            operation="input_materialize",
            evidence_kind="attempt_prepared",
        )
        return spec

    def _prepare_spec(
        self, state: dict[str, Any]
    ) -> AttemptSpec:
        reservation = self._reserve_attempt(state)
        if reservation is None:
            raise ContiguousRunnerError(
                "canonical scheduler found no eligible dispatch"
            )
        return self._materialize_reserved(reservation)

    def _prepare_backend(
        self, attempt_id: str, spec: AttemptSpec
    ) -> None:
        try:
            prepared = self.backend.prepare(spec)
        except BaseException as exc:
            if isinstance(exc, SimulatedCrash):
                raise
            if not isinstance(exc, Exception):
                raise
            self._record_retry(
                attempt_id, "backend_prepare", exc
            )
            raise ContiguousRunnerError(
                "backend preparation remains recoverable"
            ) from exc
        try:
            if not isinstance(prepared, BackendPreparation):
                raise ContiguousRunnerError(
                    "backend returned invalid preparation type"
                )
            prepared = _backend_preparation_from_dict(asdict(prepared))
            if prepared.observed_image_digest != spec.image_digest:
                raise ContiguousRunnerError(
                    "backend observed a different image digest"
                )
            attestation = Path(prepared.launch_attestation_path)
            host_root = Path(spec.host_transcript_path).parent
            if (
                attestation.parent
                != host_root
                or attestation.name != "launch_attestation.json"
                or _sha256_file(attestation)
                != prepared.launch_attestation_sha256
            ):
                raise ContiguousRunnerError(
                    "backend preparation did not yield the bound attestation"
                )
            _validate_bound_receipt(
                prepared.bridge_policy_receipt_path,
                prepared.bridge_policy_receipt_sha256,
                expected_path=Path(
                    spec.bridge_policy_receipt_path
                ),
                expected_kind="contiguous_bridge_policy",
                spec=spec,
            )
            _validate_bound_receipt(
                prepared.neutral_cwd_attestation_path,
                prepared.neutral_cwd_attestation_sha256,
                expected_path=host_root
                / "neutral_cwd_attestation.json",
                expected_kind="contiguous_neutral_cwd_attestation",
                spec=spec,
            )
            config_receipt = _validate_bound_receipt(
                prepared.app_server_config_receipt_path,
                prepared.app_server_config_receipt_sha256,
                expected_path=host_root
                / "app_server_config_receipt.json",
                expected_kind="contiguous_app_server_config",
                spec=spec,
            )
            if (
                config_receipt.get("model")
                != spec.proposer_transport.model
                or config_receipt.get("model_provider")
                != spec.proposer_transport.model_provider
                or config_receipt.get(
                    "allow_provider_model_fallback"
                )
                is not False
                or config_receipt.get("reasoning_effort")
                != spec.effort
                or config_receipt.get("environments") != []
                or config_receipt.get(
                    "selected_capability_roots"
                )
                != []
                or config_receipt.get("runtime_workspace_roots")
                != ["/controller-neutral"]
                or config_receipt.get("dynamic_tool_namespace")
                != spec.proposer_transport.dynamic_tool_namespace
                or config_receipt.get("dynamic_tool_names")
                != list(
                    spec.proposer_transport.dynamic_tool_names
                )
                or config_receipt.get(
                    "controller_method_policy"
                )
                != {
                    "preflight_requests": list(
                        spec.proposer_transport
                        .controller_preflight_request_allowlist
                    ),
                    "preflight_notifications": list(
                        spec.proposer_transport
                        .controller_preflight_notification_allowlist
                    ),
                    "turn_requests": list(
                        spec.proposer_transport
                        .controller_turn_request_allowlist
                    ),
                }
                or config_receipt.get("builtin_tool_names") != []
                or config_receipt.get("approval_policy") != "never"
                or config_receipt.get("sandbox_policy")
                != {
                    "type": "readOnly",
                    "networkAccess": False,
                }
                or config_receipt.get("state_root")
                != "/controller-state"
                or config_receipt.get("state_host_staging_root")
                != spec.app_server_state_dir
                or config_receipt.get("state_mode")
                != (
                    "resume_staged_copy"
                    if spec.thread_mode == "resume"
                    else "new_reset"
                )
                or config_receipt.get("prior_state_root")
                != (
                    spec.wip.app_server_state_dir
                    if spec.wip is not None
                    else None
                )
                or config_receipt.get(
                    "prior_state_tree_sha256"
                )
                != (
                    spec.wip.app_server_state_tree_sha256
                    if spec.wip is not None
                    else None
                )
                or config_receipt.get("staged_state_root")
                != spec.app_server_state_dir
                or config_receipt.get(
                    "staged_initial_state_tree_sha256"
                )
                != spec.initial_app_server_state_tree_sha256
                or config_receipt.get(
                    "ambient_state_access_status"
                )
                != "DENIED"
                or config_receipt.get(
                    "state_root_write_probe_status"
                )
                != "PENDING_REAL_CONTROLLER_PREFLIGHT"
                or config_receipt.get(
                    "ambient_environment_names_stripped"
                )
                != [
                    "CODEX_HOME",
                    "HOME",
                    "XDG_CONFIG_HOME",
                    "XDG_DATA_HOME",
                    "XDG_STATE_HOME",
                ]
            ):
                raise ContiguousRunnerError(
                    "app-server config receipt has ambient capability"
                )
            binary_receipt = _validate_bound_receipt(
                prepared.codex_binary_receipt_path,
                prepared.codex_binary_receipt_sha256,
                expected_path=host_root
                / "codex_binary_receipt.json",
                expected_kind="contiguous_codex_binary",
                spec=spec,
            )
            if (
                binary_receipt.get("launcher_path")
                != spec.proposer_transport.codex_launcher_path
                or binary_receipt.get("launcher_sha256")
                != spec.proposer_transport.codex_launcher_sha256
                or binary_receipt.get("package_manifest_path")
                != spec.proposer_transport.codex_package_manifest_path
                or binary_receipt.get("package_manifest_sha256")
                != spec.proposer_transport
                .codex_package_manifest_sha256
                or binary_receipt.get("native_binary_path")
                != spec.proposer_transport.codex_binary_path
                or binary_receipt.get("native_binary_sha256")
                != spec.proposer_transport.codex_binary_sha256
                or binary_receipt.get("native_binary_bytes")
                != spec.proposer_transport.codex_binary_bytes
                or binary_receipt.get("version")
                != spec.proposer_transport.codex_cli_version
            ):
                raise ContiguousRunnerError(
                    "Codex binary receipt differs from pinned config"
                )
            schema_receipt = _validate_bound_receipt(
                prepared.protocol_schema_receipt_path,
                prepared.protocol_schema_receipt_sha256,
                expected_path=host_root
                / "app_server_protocol_schema_receipt.json",
                expected_kind="contiguous_app_server_protocol_schema",
                spec=spec,
            )
            if (
                schema_receipt.get("path")
                != spec.proposer_transport
                .app_server_protocol_schema_path
                or schema_receipt.get("sha256")
                != spec.proposer_transport
                .app_server_protocol_schema_sha256
                or schema_receipt.get("bundle_path")
                != spec.proposer_transport
                .app_server_protocol_schema_bundle_path
                or schema_receipt.get("bundle_sha256")
                != spec.proposer_transport
                .app_server_protocol_schema_bundle_sha256
            ):
                raise ContiguousRunnerError(
                    "app-server schema receipt differs from pinned config"
                )
            _validate_preparation_receipts(spec, prepared)
        except ContiguousRunnerError as exc:
            self._record_retry(
                attempt_id, "backend_prepare", exc
            )
            raise
        self.journal.append(
            event_id=f"{attempt_id}:backend_prepared",
            kind="BACKEND_PREPARED",
            payload={
                "attempt_id": attempt_id,
                "prepared": asdict(prepared),
            },
            recorded_at=self.clock(),
        )
        self._record_circuit_success(
            attempt_id=attempt_id,
            operation="backend_prepare",
            evidence_kind="backend_prepared",
        )

    def _record_substrate_infrastructure(
        self,
        attempt_id: str,
        exc: BackendSubstratePreflightError,
    ) -> None:
        attempt = self.state()["attempts"].get(attempt_id)
        if (
            attempt is None
            or attempt["phase"] != "BACKEND_PREPARED"
        ):
            raise ContiguousRunnerError(
                "substrate failure targets the wrong attempt phase"
            )
        result = AttemptResult(
            kind="infrastructure",
            cost_used=0.0,
            reason="codex_substrate_preflight_failed",
        )
        result_payload = self._result_payload(attempt_id, result)
        result_payload.pop("attempt_id")
        self.journal.append(
            event_id=f"{attempt_id}:substrate-infrastructure",
            kind="ATTEMPT_SUBSTRATE_INFRASTRUCTURE",
            payload={
                "attempt_id": attempt_id,
                "substrate_identity_sha256":
                    exc.substrate_identity_sha256,
                "failure_receipt_path":
                    exc.failure_receipt_path,
                "failure_receipt_sha256":
                    exc.failure_receipt_sha256,
                "result": result_payload,
                "authenticated_cost_units": 0,
                "budget_reservation_id":
                    attempt["budget_reservation_id"],
                "scheduler_decision_id":
                    attempt["scheduler_decision_id"],
            },
            recorded_at=self.clock(),
        )

    def _launch_backend(
        self,
        attempt_id: str,
        spec: AttemptSpec,
        prepared: BackendPreparation,
    ) -> bool:
        try:
            launched = self.backend.launch(spec, prepared)
        except BaseException as exc:
            if isinstance(exc, SimulatedCrash):
                raise
            if not isinstance(exc, Exception):
                raise
            if isinstance(exc, BackendSubstratePreflightError):
                self._record_substrate_infrastructure(
                    attempt_id, exc
                )
                return False
            self._record_retry(
                attempt_id, "backend_launch", exc
            )
            # Launch may have succeeded externally.  Never call it a failure or
            # free the lane: recovery repeats the idempotent attempt identity.
            raise ContiguousRunnerError(
                "backend launch acknowledgement uncertain and recoverable"
            ) from exc
        try:
            if not isinstance(launched, BackendLaunch):
                raise ContiguousRunnerError(
                    "backend returned invalid launch type"
                )
            launched = _backend_launch_from_dict(asdict(launched))
            host_root = Path(spec.host_transcript_path).parent
            substrate = _validate_bound_receipt(
                launched.substrate_preflight_receipt_path,
                launched.substrate_preflight_receipt_sha256,
                expected_path=host_root
                / "substrate_preflight_receipt.json",
                expected_kind="contiguous_substrate_preflight",
                spec=spec,
            )
            if (
                substrate.get("substrate_identity_sha256")
                != launched.substrate_identity_sha256
                or substrate.get("state_root")
                != spec.app_server_state_dir
                or substrate.get("state_root_write_probe_status")
                != "PASS"
                or substrate.get("state_database_initialized")
                is not True
                or substrate.get("path_alias_setup_status") != "PASS"
                or substrate.get("preflight_stderr_bytes") != 0
                or substrate.get("preflight_stderr_sha256")
                != hashlib.sha256(b"").hexdigest()
                or substrate.get("proposer_container_started")
                is not False
                or substrate.get("bridge_connected") is not False
                or substrate.get("thread_started") is not False
                or substrate.get("turn_started") is not False
                or substrate.get("controller_inspect_absent")
                is not True
                or substrate.get("controller_identity_query_empty")
                is not True
                or substrate.get("controller_no_descendants")
                is not True
                or substrate.get("egress_proxy_inspect_absent")
                is not True
                or substrate.get("egress_proxy_identity_query_empty")
                is not True
                or substrate.get("egress_proxy_no_descendants")
                is not True
                or substrate.get("status") != "PASS"
            ):
                raise ContiguousRunnerError(
                    "substrate preflight receipt lacks real pre-turn "
                    "initialization/absence authority"
                )
            bridge_runtime = _validate_bound_receipt(
                launched.bridge_runtime_attestation_path,
                launched.bridge_runtime_attestation_sha256,
                expected_path=host_root
                / "bridge_runtime_attestation.json",
                expected_kind="contiguous_bridge_runtime",
                spec=spec,
            )
            if (
                bridge_runtime.get("container_id")
                != launched.container_id
                or bridge_runtime.get("socket_path")
                != spec.bridge_socket_path
                or bridge_runtime.get("token_file_path")
                != spec.bridge_token_file_path
                or not isinstance(
                    bridge_runtime.get("socket_inode"), int
                )
                or isinstance(
                    bridge_runtime.get("socket_inode"), bool
                )
                or bridge_runtime["socket_inode"] <= 0
                or not isinstance(
                    bridge_runtime.get("token_inode"), int
                )
                or isinstance(
                    bridge_runtime.get("token_inode"), bool
                )
                or bridge_runtime["token_inode"] <= 0
                or not _is_sha256(
                    bridge_runtime.get("token_sha256")
                )
                or not _is_sha256(
                    bridge_runtime.get("handshake_nonce_sha256")
                )
                or bridge_runtime.get("policy_receipt_sha256")
                != prepared.bridge_policy_receipt_sha256
            ):
                raise ContiguousRunnerError(
                    "bridge runtime does not bind the launched container"
                )
            app_runtime = _validate_bound_receipt(
                launched.app_server_runtime_receipt_path,
                launched.app_server_runtime_receipt_sha256,
                expected_path=host_root
                / "app_server_runtime_receipt.json",
                expected_kind="contiguous_app_server_runtime",
                spec=spec,
            )
            if (
                app_runtime.get("pid") != launched.app_server_pid
                or app_runtime.get("process_start")
                != launched.app_server_process_start
                or app_runtime.get("process_group_id")
                != launched.app_server_process_group_id
                or app_runtime.get("state_root")
                != spec.app_server_state_dir
                or app_runtime.get("neutral_cwd")
                != "/controller-neutral"
                or app_runtime.get("neutral_host_staging_cwd")
                != spec.neutral_host_cwd_path
                or app_runtime.get("thread_id")
                != launched.codex_thread_id
                or app_runtime.get("turn_id")
                != launched.codex_turn_id
                or app_runtime.get("thread_mode")
                != spec.thread_mode
                or app_runtime.get("model")
                != spec.proposer_transport.model
                or app_runtime.get("model_provider")
                != spec.proposer_transport.model_provider
                or app_runtime.get("reasoning_effort")
                != spec.effort
                or app_runtime.get(
                    "allow_provider_model_fallback"
                )
                is not False
                or app_runtime.get("builtin_tool_names") != []
                or app_runtime.get("dynamic_tool_namespace")
                != spec.proposer_transport.dynamic_tool_namespace
                or app_runtime.get("dynamic_tool_names")
                != list(
                    spec.proposer_transport.dynamic_tool_names
                )
                or app_runtime.get("controller_method_policy")
                != {
                    "preflight_requests": list(
                        spec.proposer_transport
                        .controller_preflight_request_allowlist
                    ),
                    "preflight_notifications": list(
                        spec.proposer_transport
                        .controller_preflight_notification_allowlist
                    ),
                    "turn_requests": list(
                        spec.proposer_transport
                        .controller_turn_request_allowlist
                    ),
                }
                or app_runtime.get("startup_probe_status") != "PASS"
                or app_runtime.get("auth_probe_status") != "PASS"
                or app_runtime.get("model_probe_status") != "PASS"
                or app_runtime.get("bridge_probe_status") != "PASS"
                or app_runtime.get("substrate_identity_sha256")
                != launched.substrate_identity_sha256
                or app_runtime.get(
                    "substrate_preflight_receipt_path"
                )
                != launched.substrate_preflight_receipt_path
                or app_runtime.get(
                    "substrate_preflight_receipt_sha256"
                )
                != launched.substrate_preflight_receipt_sha256
                or app_runtime.get(
                    "state_root_write_probe_status"
                )
                != "PASS"
                or app_runtime.get(
                    "state_database_initialized"
                )
                is not True
                or app_runtime.get("path_alias_setup_status")
                != "PASS"
                or app_runtime.get("ambient_state_loaded") is not False
            ):
                raise ContiguousRunnerError(
                    "app-server runtime capability projection is unsafe"
                )
            thread_binding = _validate_bound_receipt(
                launched.thread_binding_path,
                launched.thread_binding_sha256,
                expected_path=host_root
                / "turn_start_binding.json",
                expected_kind="contiguous_turn_start_binding",
                spec=spec,
            )
            if (
                thread_binding.get("thread_id")
                != launched.codex_thread_id
                or thread_binding.get("turn_id")
                != launched.codex_turn_id
                or thread_binding.get("thread_mode")
                != spec.thread_mode
                or thread_binding.get(
                    "bridge_runtime_attestation_sha256"
                )
                != launched.bridge_runtime_attestation_sha256
                or thread_binding.get(
                    "app_server_runtime_receipt_sha256"
                )
                != launched.app_server_runtime_receipt_sha256
                or thread_binding.get("reasoning_effort")
                != spec.effort
                or thread_binding.get("model")
                != spec.proposer_transport.model
                or thread_binding.get("transcript_chain_sha256")
                != launched.transcript_chain_sha256
            ):
                raise ContiguousRunnerError(
                    "turn-start binding is incomplete or substituted"
                )
            transcript_receipt = _validate_bound_receipt(
                launched.transcript_chain_receipt_path,
                launched.transcript_chain_receipt_sha256,
                expected_path=host_root
                / "turn_start_transcript_chain_receipt.json",
                expected_kind="contiguous_turn_start_transcript_chain",
                spec=spec,
            )
            if (
                transcript_receipt.get("thread_id")
                != launched.codex_thread_id
                or transcript_receipt.get("turn_id")
                != launched.codex_turn_id
                or transcript_receipt.get("chain_head_sha256")
                != launched.transcript_chain_sha256
            ):
                raise ContiguousRunnerError(
                    "turn-start transcript chain receipt is mismatched"
                )
            if spec.thread_mode == "new":
                if (
                    launched.thread_rebinding_receipt_path is not None
                    or launched.thread_rebinding_receipt_sha256 is not None
                    or launched.codex_thread_id
                    == spec.resume_thread_id
                ):
                    raise ContiguousRunnerError(
                        "new thread carries a rebinding receipt"
                    )
            else:
                if (
                    launched.codex_thread_id
                    != spec.resume_thread_id
                    or launched.thread_rebinding_receipt_path is None
                    or launched.thread_rebinding_receipt_sha256 is None
                    or spec.wip is None
                ):
                    raise ContiguousRunnerError(
                        "resume lacks the exact prior thread binding"
                    )
                rebinding = _validate_bound_receipt(
                    launched.thread_rebinding_receipt_path,
                    launched.thread_rebinding_receipt_sha256,
                    expected_path=host_root
                    / "thread_rebinding_receipt.json",
                    expected_kind="contiguous_thread_rebinding",
                    spec=spec,
                )
                if (
                    rebinding.get("thread_id")
                    != spec.resume_thread_id
                    or rebinding.get("prior_thread_binding_sha256")
                    != spec.resume_thread_binding_sha256
                    or rebinding.get(
                        "prior_transcript_chain_sha256"
                    )
                    != spec.wip.transcript_chain_sha256
                    or rebinding.get(
                        "prior_app_server_state_tree_sha256"
                    )
                    != spec.wip.app_server_state_tree_sha256
                    or rebinding.get(
                        "prior_app_server_state_dir"
                    )
                    != spec.wip.app_server_state_dir
                    or rebinding.get(
                        "staged_app_server_state_dir"
                    )
                    != spec.app_server_state_dir
                    or rebinding.get(
                        "staged_initial_state_tree_sha256"
                    )
                    != spec.wip.app_server_state_tree_sha256
                    or rebinding.get("new_container_id")
                    != launched.container_id
                    or rebinding.get(
                        "new_bridge_runtime_attestation_sha256"
                    )
                    != launched.bridge_runtime_attestation_sha256
                    or rebinding.get("old_bridge_revoked")
                    is not True
                    or rebinding.get("no_binding_overlap")
                    is not True
                ):
                    raise ContiguousRunnerError(
                        "thread rebinding is stale, replayed, or overlapping"
                    )
            _validate_launch_receipts(spec, prepared, launched)
        except ContiguousRunnerError as exc:
            self._record_retry(
                attempt_id, "backend_launch", exc
            )
            raise
        launched_at = self.clock()
        self.journal.append(
            event_id=f"{attempt_id}:launched",
            kind="ATTEMPT_LAUNCHED",
            payload={
                "attempt_id": attempt_id,
                "launched": asdict(launched),
                "launched_at": launched_at,
            },
            recorded_at=launched_at,
        )
        self._record_circuit_success(
            attempt_id=attempt_id,
            operation="backend_launch",
            evidence_kind="attempt_launched",
        )
        return True

    def _poll_attempt(
        self,
        attempt_id: str,
        attempt: dict[str, Any],
        *,
        now: float,
    ) -> None:
        try:
            observation = self.backend.poll(
                spec=attempt["spec"],
                prepared=attempt["prepared"],
                launched=attempt["launched"],
                timeout_seconds=POLL_TIMEOUT_SECONDS,
            )
            if not isinstance(observation, BackendPoll):
                raise ContiguousRunnerError(
                    "backend returned invalid poll observation"
                )
        except BaseException as exc:
            if isinstance(exc, SimulatedCrash):
                raise
            if not isinstance(exc, Exception):
                raise
            self._record_retry(attempt_id, "backend_poll", exc)
            raise ContiguousRunnerError(
                "backend poll remains recoverable"
            ) from exc
        try:
            observation = _backend_poll_from_dict(asdict(observation))
        except ContiguousRunnerError as exc:
            self._record_retry(attempt_id, "backend_poll", exc)
            raise
        if observation.status == "running":
            last_observation_at = attempt["last_observation_at"]
            changed = (
                attempt["last_observation_sha256"]
                != observation.observation_sha256
            )
            if (
                changed
                and
                attempt["observation_count"]
                < Scheduler.MAX_JOURNALED_OBSERVATIONS_PER_ATTEMPT
                and (
                    last_observation_at is None
                    or now - last_observation_at
                    >= Scheduler
                    .MIN_JOURNALED_OBSERVATION_INTERVAL_SECONDS
                )
            ):
                observation_index = attempt["observation_count"] + 1
                self.journal.append(
                    event_id=(
                        f"{attempt_id}:observation:"
                        f"{observation_index:08d}"
                    ),
                    kind="ATTEMPT_OBSERVED",
                    payload={
                        "attempt_id": attempt_id,
                        "observation_index": observation_index,
                        "observation": asdict(observation),
                    },
                    recorded_at=now,
                )
                self._record_circuit_success(
                    attempt_id=attempt_id,
                    operation="backend_poll",
                    evidence_kind="backend_poll_observation",
                )
            deadline = (
                attempt["launched_at"]
                + attempt["spec"].soft_allocation_seconds
            )
            if attempt["phase"] == "RUNNING" and now >= deadline:
                self.journal.append(
                    event_id=f"{attempt_id}:draining",
                    kind="ATTEMPT_DRAINING",
                    payload={
                        "attempt_id": attempt_id,
                        "soft_deadline": deadline,
                    },
                    recorded_at=now,
                )
            return
        self.journal.append(
            event_id=f"{attempt_id}:terminal",
            kind="ATTEMPT_EXITED",
            payload={
                "attempt_id": attempt_id,
                "terminal": asdict(observation),
            },
            recorded_at=now,
        )
        self._record_circuit_success(
            attempt_id=attempt_id,
            operation="backend_poll",
            evidence_kind="backend_poll_observation",
        )

    def _collect_exited(
        self, attempt_id: str, attempt: dict[str, Any], *, now: float
    ) -> None:
        try:
            collection = self.backend.collect(
                spec=attempt["spec"],
                prepared=attempt["prepared"],
                launched=attempt["launched"],
                terminal=attempt["terminal"],
            )
        except BackendPublicActionProtocolInvalidError as exc:
            spec = attempt["spec"]
            try:
                receipt = _validate_bound_receipt(
                    exc.receipt_path,
                    exc.receipt_sha256,
                    expected_path=Path(spec.host_transcript_path).parent
                    / "arena_public_action_protocol_invalid_receipt.json",
                    expected_kind=(
                        "contiguous_arena_public_action_protocol_invalid"
                    ),
                    spec=spec,
                )
                terminal_evidence = {
                    "controller_state_scan_receipt_path":
                        exc.controller_state_scan_receipt_path,
                    "controller_state_scan_receipt_sha256":
                        exc.controller_state_scan_receipt_sha256,
                    "retained_canary_scan_receipt_path":
                        exc.retained_canary_scan_receipt_path,
                    "retained_canary_scan_receipt_sha256":
                        exc.retained_canary_scan_receipt_sha256,
                    "partial_taint_scan_receipt_path":
                        exc.partial_taint_scan_receipt_path,
                    "partial_taint_scan_receipt_sha256":
                        exc.partial_taint_scan_receipt_sha256,
                    "partial_usage_receipt_path":
                        exc.partial_usage_receipt_path,
                    "partial_usage_receipt_sha256":
                        exc.partial_usage_receipt_sha256,
                }
                _validate_protocol_invalid_terminal_evidence(
                    spec=spec,
                    receipt=receipt,
                    evidence=terminal_evidence,
                )
                violation = receipt.get("protocol_violation")
                if (
                    attempt["terminal"].status
                    != "containment_fault"
                    or not isinstance(violation, dict)
                    or receipt.get("protocol_violation_sha256")
                    != Scheduler.sha256_json(violation)
                    or not _is_sha256(
                        receipt.get(
                            "proposer_containment_sha256"
                        )
                    )
                    or not _is_sha256(
                        receipt.get(
                            "controller_absence_receipt_sha256"
                        )
                    )
                    or receipt.get("cost_used")
                    != float(exc.cost_used)
                    or receipt.get("cost_authority")
                    not in {
                        "full_finite_reservation",
                        "explicit_unlimited_no_local_charge",
                    }
                    or any(
                        receipt.get(name) is not False
                        for name in (
                            "candidate_admissible",
                            "wip_admissible",
                            "public_observation_admissible",
                            "sidecar_request_admissible",
                            "supervisory_handoff_admissible",
                            "promotion_admissible",
                            "restart_restoration_admissible",
                        )
                    )
                    or receipt.get("status") != "PROTOCOL_INVALID"
                ):
                    raise ContiguousRunnerError(
                        "public-action protocol-invalid receipt is malformed"
                    )
            except Exception as receipt_error:
                self._record_retry(
                    attempt_id, "backend_collect", receipt_error
                )
                raise ContiguousRunnerError(
                    "public-action protocol-invalid evidence remains "
                    "recoverable"
                ) from receipt_error
            protocol_invalid = AttemptResult(
                kind="protocol_invalid",
                cost_used=float(exc.cost_used),
                reason="public_action_protocol_invalid",
            )
            result_value = self._result_payload(
                attempt_id, protocol_invalid
            )
            result_value.pop("attempt_id")
            self.journal.append(
                event_id=f"{attempt_id}:public_action_protocol_invalid",
                kind="ATTEMPT_PUBLIC_ACTION_PROTOCOL_INVALID",
                payload={
                    "attempt_id": attempt_id,
                    "protocol_invalid_receipt_path":
                        exc.receipt_path,
                    "protocol_invalid_receipt_sha256":
                        exc.receipt_sha256,
                    "terminal_evidence": terminal_evidence,
                    "result": result_value,
                },
                recorded_at=now,
            )
            return
        except BaseException as exc:
            if isinstance(exc, SimulatedCrash):
                raise
            if not isinstance(exc, Exception):
                raise
            self._record_retry(
                attempt_id, "backend_collect", exc
            )
            raise ContiguousRunnerError(
                "backend collection remains recoverable"
            ) from exc
        try:
            if not isinstance(collection, BackendCollection):
                raise ContiguousRunnerError(
                    "backend returned invalid collection type"
                )
            collection = _backend_collection_from_dict(
                _backend_collection_to_dict(collection)
            )
            sanitized_result = self._sanitize_result(
                attempt["spec"], collection.result
            )
            if sanitized_result != collection.result:
                collection = replace(
                    collection, result=sanitized_result
                )
            self._validate_collection(
                attempt["spec"],
                attempt["prepared"],
                attempt["launched"],
                collection,
            )
        except ContiguousRunnerError as exc:
            rejected = AttemptResult(
                kind="infrastructure",
                reason=f"backend collection rejected: {exc}",
            )
            result_value = self._result_payload(
                attempt_id, rejected
            )
            result_value.pop("attempt_id")
            self.journal.append(
                event_id=f"{attempt_id}:collection_rejected",
                kind="ATTEMPT_COLLECTION_REJECTED",
                payload={
                    "attempt_id": attempt_id,
                    "reason": str(exc),
                    "result": result_value,
                },
                recorded_at=now,
            )
            return
        try:
            public_observation_transition = (
                Scheduler.public_observation_transition(
                    attempt_id=attempt_id,
                    generation_id=attempt["spec"].generation_id,
                    game=attempt["spec"].game,
                    frontier_sha256=attempt["spec"].frontier_sha256,
                    parent_checkpoint_sha256=(
                        attempt["spec"].parent_checkpoint_sha256
                    ),
                    host_transcript_path=(
                        attempt["spec"].host_transcript_path
                    ),
                    result_kind=collection.result.kind,
                    receipt_sha256s=(
                        collection
                        .native_public_observation_receipt_sha256s
                    ),
                )
            )
        except Scheduler.SchedulerError as exc:
            raise ContiguousRunnerError(
                "backend collection has no canonical observation transition"
            ) from exc
        self.journal.append(
            event_id=f"{attempt_id}:public_observations_staging",
            kind="ATTEMPT_PUBLIC_OBSERVATIONS_STAGING",
            payload={
                "attempt_id": attempt_id,
                "transition": public_observation_transition,
            },
            recorded_at=now,
        )
        if collection.result.kind in {
            "clean_no_progress",
            "candidate",
        }:
            try:
                self._register_native_public_observation_receipts(
                    attempt["spec"], collection
                )
            except ContiguousRunnerError as exc:
                self._record_retry(
                    attempt_id,
                    "backend_collect",
                    exc,
                )
                raise ContiguousRunnerError(
                    "public observation registry installation remains "
                    "recoverable"
                ) from exc
        self.journal.append(
            event_id=f"{attempt_id}:collected",
            kind="ATTEMPT_COLLECTED",
            payload={
                "attempt_id": attempt_id,
                "collection": _backend_collection_to_dict(collection),
                "public_observation_transition_sha256":
                    Scheduler.sha256_json(
                        public_observation_transition
                    ),
            },
            recorded_at=now,
        )
        self._record_circuit_success(
            attempt_id=attempt_id,
            operation="backend_collect",
            evidence_kind="attempt_collected",
        )

    def _teardown_collected(
        self, attempt_id: str, attempt: dict[str, Any], *, now: float
    ) -> None:
        cause: Literal["normal_exit", "containment_fault"] = (
            "containment_fault"
            if attempt["terminal"].status == "containment_fault"
            else "normal_exit"
        )
        try:
            proof = self.backend.teardown(
                spec=attempt["spec"],
                prepared=attempt["prepared"],
                launched=attempt["launched"],
                cause=cause,
            )
        except BaseException as exc:
            if isinstance(exc, SimulatedCrash):
                raise
            if not isinstance(exc, Exception):
                raise
            self._record_retry(
                attempt_id, "backend_teardown", exc
            )
            raise ContiguousRunnerError(
                "backend teardown remains mandatory/recoverable"
            ) from exc
        try:
            if not isinstance(proof, BackendTeardownProof):
                raise ContiguousRunnerError(
                    "backend returned invalid teardown proof type"
                )
            proof = _backend_teardown_from_dict(asdict(proof))
            if (
                proof.container_id != attempt["launched"].container_id
                or proof.cause != cause
                or Path(
                    attempt["spec"].arena_socket_path
                ).exists()
                or Path(
                    attempt["spec"].arena_socket_path
                ).is_symlink()
                or Path(
                    attempt["spec"].arena_token_file_path
                ).exists()
                or Path(
                    attempt["spec"].arena_token_file_path
                ).is_symlink()
                or Path(
                    attempt["spec"].bridge_socket_path
                ).exists()
                or Path(
                    attempt["spec"].bridge_socket_path
                ).is_symlink()
                or Path(
                    attempt["spec"].bridge_token_file_path
                ).exists()
                or Path(
                    attempt["spec"].bridge_token_file_path
                ).is_symlink()
                or Path(
                    attempt["spec"].app_server_control_dir
                ).exists()
                or Path(
                    attempt["spec"].app_server_control_dir
                ).is_symlink()
            ):
                raise ContiguousRunnerError(
                    "backend teardown proof/cleanup is incomplete"
                )
            _validate_arena_volume_teardown(
                spec=attempt["spec"],
                prepared=attempt["prepared"],
                proof=proof,
            )
            _validate_terminal_canary_reveal(
                spec=attempt["spec"],
                prepared=attempt["prepared"],
                launched=attempt["launched"],
                proof=proof,
                canaries=self._controller_state_canaries,
            )
            _validate_terminal_canary_cleanup(
                spec=attempt["spec"],
                prepared=attempt["prepared"],
                proof=proof,
            )
        except ContiguousRunnerError as exc:
            self._record_retry(
                attempt_id, "backend_teardown", exc
            )
            raise
        self.journal.append(
            event_id=f"{attempt_id}:torn_down",
            kind="ATTEMPT_TORN_DOWN",
            payload={"attempt_id": attempt_id, "teardown": asdict(proof)},
            recorded_at=now,
        )
        self._record_circuit_success(
            attempt_id=attempt_id,
            operation="backend_teardown",
            evidence_kind="attempt_torn_down",
        )

    @staticmethod
    def _terminal_fault_domain(
        attempt: Mapping[str, Any],
        result: AttemptResult,
    ) -> str | None:
        if attempt["terminal"].status == "containment_fault":
            return "containment_infrastructure"
        collection = attempt.get("collection")
        if isinstance(collection, BackendCollection):
            provider_outcome = collection.structured_provider_outcome
            if provider_outcome in {"capacity", "rate_limit"}:
                return "provider_availability"
            if provider_outcome == "provider_failure":
                return "provider_failure"
            if provider_outcome == "containment_fault":
                return "containment_infrastructure"
        if result.kind == "infrastructure":
            return "terminal_infrastructure"
        return None

    def _record_torn_down_result(
        self, attempt_id: str, attempt: dict[str, Any], *, now: float
    ) -> None:
        collected_result = self._sanitize_result(
            attempt["spec"], attempt["outcome"]
        )
        result = apply_terminal_result_precedence(
            attempt["terminal"].status,
            collected_result,
        )
        fault_domain = self._terminal_fault_domain(attempt, result)
        if (
            fault_domain is not None
            and not attempt[
                "terminal_failure_circuit_recorded"
            ]
        ):
            self._record_circuit_failure(
                attempt_id=attempt_id,
                operation="backend_terminal",
                fault_domain=fault_domain,
            )
        if attempt["collection"] is None:
            allowed = (
                result.kind == "infrastructure"
                or (
                    result.kind == "protocol_invalid"
                    and attempt.get("protocol_invalid") is not None
                    and result.reason
                    == "public_action_protocol_invalid"
                )
            )
            if (
                not allowed
                or result.wip is not None
                or result.candidate is not None
                or result.blocker is not None
            ):
                raise ContiguousRunnerError(
                    "rejected collection retained an admissible result"
                )
            self.journal.append(
                event_id=f"{attempt_id}:result",
                kind="ATTEMPT_RESULT",
                payload=self._result_payload(
                    attempt_id, result, attempt=attempt
                ),
                recorded_at=now,
            )
            return
        state_root = Path(attempt["spec"].app_server_state_dir)
        expected_state_sha = (
            attempt["collection"].app_server_state_tree_sha256
        )
        if Contract._tree_hash(state_root) != expected_state_sha:
            raise ContiguousRunnerError(
                "staged app-server state changed before commit/seal"
            )
        _seal_regular_tree(state_root)
        if Contract._tree_hash(state_root) != expected_state_sha:
            raise ContiguousRunnerError(
                "sealed app-server state changed content"
            )
        if result.wip is not None:
            wip_root = Path(result.wip.wip_root_path)
            if (
                Contract._tree_hash(wip_root)
                != result.wip.wip_tree_sha256
            ):
                raise ContiguousRunnerError(
                    "clean WIP changed before commit/seal"
                )
            _seal_regular_tree(wip_root)
            if (
                Contract._tree_hash(wip_root)
                != result.wip.wip_tree_sha256
            ):
                raise ContiguousRunnerError(
                    "sealed WIP changed content"
                )
        self.journal.append(
            event_id=f"{attempt_id}:result",
            kind="ATTEMPT_RESULT",
            payload=self._result_payload(
                attempt_id, result, attempt=attempt
            ),
            recorded_at=now,
        )

    def _commit_pending(
        self, attempt_id: str, attempt: dict[str, Any], *, now: float
    ) -> None:
        spec = attempt["spec"]
        candidate = attempt["candidate"]
        assert isinstance(candidate, PromotionCandidate)
        collection = attempt.get("collection")
        if not isinstance(collection, BackendCollection):
            raise ContiguousRunnerError(
                "promotion lacks its authenticated backend collection"
            )
        # Reopen the conservative handoff-exposure boundary immediately
        # before the independent promotion gate.  Collection-time validation
        # alone is not promotion authority.
        self._validate_supervisory_reproduction_gate(
            spec, collection
        )

        def reject_integrity(code: str) -> None:
            if code not in Scheduler.PROMOTION_FAILURE_CODES:
                raise ContiguousRunnerError(
                    "promotion failure code is outside policy"
                )
            self.journal.append(
                event_id=f"{attempt_id}:promotion_failed",
                kind="PROMOTION_FAILED",
                payload={
                    "attempt_id": attempt_id,
                    "code": code,
                },
                recorded_at=now,
            )

        try:
            commit = self.promotion_gate.commit(
                spec=spec, candidate=candidate
            )
        except PromotionRejected as exc:
            del exc
            reject_integrity("promotion_gate_rejected")
            return
        except BaseException as exc:
            if isinstance(exc, SimulatedCrash):
                raise
            if not isinstance(exc, Exception):
                raise
            recover = getattr(self.promotion_gate, "recover", None)
            if not callable(recover):
                self._record_retry(
                    attempt_id, "promotion_recover", exc
                )
                raise ContiguousRunnerError(
                    "promotion acknowledgement is ambiguous; "
                    "durable reconciliation is unavailable"
                ) from exc
            try:
                commit = recover(spec=spec, candidate=candidate)
                if commit is None:
                    self._record_retry(
                        attempt_id, "promotion_commit", exc
                    )
                    raise ContiguousRunnerError(
                        "promotion outcome remains ambiguous and recoverable"
                    )
            except PromotionRejected as recovery_exc:
                del recovery_exc
                reject_integrity("promotion_gate_rejected")
                return
            except BaseException as recovery_exc:
                if isinstance(recovery_exc, SimulatedCrash):
                    raise
                if not isinstance(recovery_exc, Exception):
                    raise
                self._record_retry(
                    attempt_id, "promotion_recover", recovery_exc
                )
                raise ContiguousRunnerError(
                    "promotion reconciliation remains recoverable"
                ) from recovery_exc
        try:
            if not isinstance(commit, PromotionCommit):
                raise ContiguousRunnerError(
                    "promotion gate returned an invalid commit"
                )
            self._validate_commit(
                spec,
                commit,
                self.state()["lanes"],
                candidate,
            )
        except ContiguousRunnerError as exc:
            # A gate response is an integrity claim, not an ambiguous
            # transport acknowledgement.  Once bytes were returned (either
            # directly or by recovery), an invalid K→K+1 edge is terminal for
            # the lane.  The unselected external artifact remains available
            # for quarantine/forensics but can never become the lane parent.
            del exc
            reject_integrity("promotion_commit_invalid")
            return
        payload = asdict(commit)
        payload["attempt_id"] = attempt_id
        payload["exact_path"] = list(commit.exact_path)
        payload["candidate_manifest_sha256"] = (
            candidate.candidate_manifest_sha256
        )
        payload["source_path"] = str(
            self._commit_source_path(commit)
        )
        self.journal.append(
            event_id=f"{attempt_id}:promotion",
            kind="PROMOTION_COMMITTED",
            payload=payload,
            recorded_at=now,
        )
        self._record_circuit_success(
            attempt_id=attempt_id,
            operation="promotion_commit",
            evidence_kind="promotion_committed",
        )
        self._record_circuit_success(
            attempt_id=attempt_id,
            operation="promotion_recover",
            evidence_kind="promotion_committed",
        )

    def _authorize_substrate_health_reprobe_locked(
        self,
        state: Mapping[str, Any],
        *,
        issued_at: float,
        authorization_mode: Literal[
            "sealed_autonomous_circuit",
            "trusted_operator_early_override",
        ],
        reason_code: str,
    ) -> dict[str, Any]:
        """Commit one single-use non-game probe authorization."""

        incident = state["substrate_incident"]
        if (
            incident is None
            or incident["pending_reprobe"] is not None
            or state["operator_incident"] is not None
            or not incident["circuit_failure_recorded"]
        ):
            raise ContiguousRunnerError(
                "substrate incident is not eligible for one reprobe"
            )
        operation_state = state["failure_operation_circuits"].get(
            "substrate_health_reprobe:controller_substrate"
        )
        if not isinstance(operation_state, dict):
            raise ContiguousRunnerError(
                "substrate reprobe lacks its durable circuit deadline"
            )
        retry_not_before = operation_state.get("retry_not_before")
        if not _is_finite_number(retry_not_before):
            raise ContiguousRunnerError(
                "substrate reprobe deadline is malformed"
            )
        if (
            authorization_mode == "sealed_autonomous_circuit"
            and issued_at < float(retry_not_before)
        ):
            raise ContiguousRunnerError(
                "autonomous substrate reprobe preceded its deadline"
            )
        if not _safe_identifier(reason_code):
            raise ContiguousRunnerError(
                "substrate reprobe reason is malformed"
            )
        authorization_id = self._new_identifier(
            "substrate reprobe authorization_id"
        )
        probe_index = int(incident["health_probe_count"]) + 1
        operator_configuration_sha256 = state[
            "operator_configuration_sha256"
        ]
        if not _is_sha256(operator_configuration_sha256):
            raise ContiguousRunnerError(
                "substrate circuit lacks sealed operator configuration"
            )
        unsigned = {
            "schema": 2,
            "kind":
                "contiguous_substrate_health_reprobe_authorization",
            "campaign_id": state["campaign_id"],
            "authorization_id": authorization_id,
            "attempt_id": incident["attempt_id"],
            "substrate_identity_sha256":
                incident["substrate_identity_sha256"],
            "incident_failure_receipt_sha256":
                incident["failure_receipt_sha256"],
            "probe_index": probe_index,
            "reason_code": reason_code,
            "authorization_mode": authorization_mode,
            "operator_configuration_sha256":
                operator_configuration_sha256,
            "retry_not_before": float(retry_not_before),
            "issued_at": float(issued_at),
            "single_use": True,
            "sealed_supervisor_authority":
                authorization_mode
                == "sealed_autonomous_circuit",
            "trusted_operator_authority":
                authorization_mode
                == "trusted_operator_early_override",
            "game_scheduler_authority": False,
            "meta_scheduler_authority": False,
        }
        receipt = {
            **unsigned,
            "authorization_binding_sha256":
                hashlib.sha256(_canonical_json(unsigned)).hexdigest(),
        }
        authorization_root = (
            self.root / SUBSTRATE_REPROBE_AUTHORIZATION_ROOT
        )
        if (
            authorization_root.exists()
            or authorization_root.is_symlink()
        ):
            if (
                authorization_root.is_symlink()
                or not authorization_root.is_dir()
            ):
                raise ContiguousRunnerError(
                    "substrate authorization root is unsafe"
                )
        else:
            authorization_root.mkdir(mode=0o700)
            _fsync_directory(self.root)
        authorization_path = (
            authorization_root / f"{authorization_id}.json"
        )
        _write_new_file(authorization_path, receipt)
        _fsync_directory(authorization_root)
        authorization_sha256 = _sha256_file(authorization_path)
        payload = {
            "authorization_id": authorization_id,
            "attempt_id": incident["attempt_id"],
            "substrate_identity_sha256":
                incident["substrate_identity_sha256"],
            "incident_failure_receipt_sha256":
                incident["failure_receipt_sha256"],
            "probe_index": probe_index,
            "reason_code": reason_code,
            "authorization_mode": authorization_mode,
            "retry_not_before": float(retry_not_before),
            "authorization_receipt_path": str(authorization_path),
            "authorization_receipt_sha256": authorization_sha256,
        }
        self.journal.append(
            event_id=(
                f"substrate-reprobe-authorized:{authorization_id}"
            ),
            kind="SUBSTRATE_HEALTH_REPROBE_AUTHORIZED",
            payload=payload,
            recorded_at=issued_at,
        )
        return dict(payload)

    def authorize_substrate_health_reprobe(
        self,
        *,
        reason_code: str,
    ) -> dict[str, Any]:
        """Trusted operator early override; no caller supplies an epoch."""

        lock_path = self.root / ".cycle.lock"
        descriptor = _open_unaliased(
            lock_path, os.O_RDWR | os.O_CREAT
        )
        handle = os.fdopen(descriptor, "r+", encoding="utf-8")
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            return self._authorize_substrate_health_reprobe_locked(
                self.state(),
                issued_at=float(self.clock()),
                authorization_mode=(
                    "trusted_operator_early_override"
                ),
                reason_code=reason_code,
            )
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            handle.close()

    def _authorize_meta_substrate_recovery_locked(
        self,
        state: Mapping[str, Any],
        *,
        meta_request_sha256: str,
        meta_response_sha256: str,
        meta_terminal_sha256: str,
        recommendation: str,
        issued_at: float,
    ) -> dict[str, Any]:
        incident = state["substrate_incident"]
        operator_incident = state["operator_incident"]
        if (
            incident is None
            or operator_incident is None
            or incident["pending_reprobe"] is not None
            or incident["meta_recovery"] is not None
            or incident["meta_recovery_invocation_count"] != 0
            or recommendation
            != META_SUBSTRATE_RECOVERY_RECOMMENDATION
            or any(
                not _is_sha256(value)
                for value in (
                    meta_request_sha256,
                    meta_response_sha256,
                    meta_terminal_sha256,
                )
            )
        ):
            raise ContiguousRunnerError(
                "meta recommendation is not eligible for the "
                "single-use substrate recovery"
            )
        operator_configuration_sha256 = state[
            "operator_configuration_sha256"
        ]
        if not _is_sha256(operator_configuration_sha256):
            raise ContiguousRunnerError(
                "meta recovery lacks a sealed operator configuration"
            )
        authorization_id = self._new_identifier(
            "meta substrate recovery authorization_id"
        )
        receipt_body = {
            "schema": 1,
            "kind":
                "contiguous_meta_substrate_recovery_authorization",
            "campaign_id": state["campaign_id"],
            "authorization_id": authorization_id,
            "attempt_id": incident["attempt_id"],
            "substrate_identity_sha256":
                incident["substrate_identity_sha256"],
            "incident_failure_receipt_sha256":
                incident["failure_receipt_sha256"],
            "incident_event_sequence":
                incident["incident_event_sequence"],
            "incident_event_digest":
                incident["incident_event_digest"],
            "incident_identity_sha256":
                incident["incident_identity_sha256"],
            "meta_request_sha256": meta_request_sha256,
            "meta_response_sha256": meta_response_sha256,
            "meta_terminal_sha256": meta_terminal_sha256,
            "recommendation": recommendation,
            "operator_configuration_sha256":
                operator_configuration_sha256,
            "invocation_index": 1,
            "single_use": True,
            "solver_authority": False,
            "wip_authority": False,
            "cost_authority": False,
            "promotion_authority": False,
        }
        authentication_sha256 = (
            meta_substrate_recovery_authentication_sha256(
                receipt_body,
                operator_configuration_sha256=(
                    operator_configuration_sha256
                ),
            )
        )
        receipt = {
            **receipt_body,
            "authorization_authentication_sha256":
                authentication_sha256,
        }
        receipt_root = (
            self.root
            / META_SUBSTRATE_RECOVERY_AUTHORIZATION_ROOT
        )
        if receipt_root.exists() or receipt_root.is_symlink():
            if receipt_root.is_symlink() or not receipt_root.is_dir():
                raise ContiguousRunnerError(
                    "meta substrate authorization root is unsafe"
                )
        else:
            receipt_root.mkdir(mode=0o700)
            _fsync_directory(self.root)
        receipt_path = receipt_root / f"{authorization_id}.json"
        _write_new_immutable_file_atomic(
            receipt_path, receipt
        )
        payload = {
            "authorization_id": authorization_id,
            "attempt_id": incident["attempt_id"],
            "substrate_identity_sha256":
                incident["substrate_identity_sha256"],
            "incident_failure_receipt_sha256":
                incident["failure_receipt_sha256"],
            "incident_event_sequence":
                incident["incident_event_sequence"],
            "incident_event_digest":
                incident["incident_event_digest"],
            "incident_identity_sha256":
                incident["incident_identity_sha256"],
            "meta_request_sha256": meta_request_sha256,
            "meta_response_sha256": meta_response_sha256,
            "meta_terminal_sha256": meta_terminal_sha256,
            "recommendation": recommendation,
            "operator_configuration_sha256":
                operator_configuration_sha256,
            "authorization_receipt_path": str(receipt_path),
            "authorization_receipt_sha256":
                _sha256_file(receipt_path),
            "authorization_authentication_sha256":
                authentication_sha256,
            "invocation_index": 1,
        }
        self.journal.append(
            event_id=(
                "meta-substrate-recovery-authorized:"
                + authorization_id
            ),
            kind="META_SUBSTRATE_RECOVERY_AUTHORIZED",
            payload=payload,
            recorded_at=issued_at,
        )
        return payload

    def _advance_meta_substrate_recovery_locked(
        self,
        state: Mapping[str, Any],
    ) -> dict[str, Any]:
        incident = state["substrate_incident"]
        if incident is None:
            return dict(state)
        meta = incident.get("meta_recovery")
        if not isinstance(meta, dict):
            return dict(state)
        if meta["phase"] == "AUTHORIZED":
            attempt = state["attempts"].get(incident["attempt_id"])
            if (
                attempt is None
                or attempt["phase"] != "CLOSED"
                or attempt["substrate_failure"] is None
            ):
                raise ContiguousRunnerError(
                    "meta recovery lost its failed closed attempt"
                )
            probe = self.backend.probe_substrate_health(
                spec=attempt["spec"],
                prepared=attempt["prepared"],
                authorization_id=meta["authorization_id"],
                authorization_receipt_sha256=(
                    meta["authorization_receipt_sha256"]
                ),
                probe_index=meta["probe_index"],
                failed_substrate_identity_sha256=(
                    meta["substrate_identity_sha256"]
                ),
                incident_failure_receipt_sha256=(
                    meta["incident_failure_receipt_sha256"]
                ),
            )
            if (
                not isinstance(probe, BackendSubstrateHealthProbe)
                or probe.authorization_id
                != meta["authorization_id"]
                or probe.probe_index != meta["probe_index"]
                or probe.failed_substrate_identity_sha256
                != meta["substrate_identity_sha256"]
                or probe.incident_failure_receipt_sha256
                != meta["incident_failure_receipt_sha256"]
                or not _is_sha256(
                    probe.remediation_epoch_sha256
                )
                or probe.status not in {"PASS", "FAILED"}
                or (
                    probe.status == "PASS"
                    and (
                        not _is_sha256(
                            probe.healthy_substrate_identity_sha256
                        )
                        or probe.failure_class is not None
                        or probe.failure_code is not None
                    )
                )
                or (
                    probe.status == "FAILED"
                    and (
                        probe.healthy_substrate_identity_sha256
                        is not None
                        or probe.failure_class not in {
                            "DETERMINISTIC_CONFIGURATION",
                            "TRANSIENT_INFRASTRUCTURE",
                        }
                        or not _safe_identifier(
                            probe.failure_code
                        )
                    )
                )
                or not _safe_path_string(probe.receipt_path)
                or not Path(probe.receipt_path).is_absolute()
                or not _is_sha256(probe.receipt_sha256)
            ):
                raise ContiguousRunnerError(
                    "meta recovery backend returned malformed health "
                    "evidence"
                )
            health = _read_json_file(Path(probe.receipt_path))
            rematerialization_path = health.get(
                "rematerialization_evidence_path"
            )
            rematerialization_sha256 = health.get(
                "rematerialization_evidence_sha256"
            )
            if (
                not _safe_path_string(rematerialization_path)
                or not Path(rematerialization_path).is_absolute()
                or not _is_sha256(rematerialization_sha256)
            ):
                raise ContiguousRunnerError(
                    "meta recovery health evidence lacks "
                    "rematerialization"
                )
            self.journal.append(
                event_id=(
                    "meta-substrate-recovery-result:"
                    + meta["authorization_id"]
                ),
                kind=(
                    "META_SUBSTRATE_HEALTH_RESTORED"
                    if probe.status == "PASS"
                    else "META_SUBSTRATE_RECOVERY_FAILED"
                ),
                payload={
                    "authorization_id": probe.authorization_id,
                    "attempt_id": meta["attempt_id"],
                    "substrate_identity_sha256":
                        probe.failed_substrate_identity_sha256,
                    "incident_failure_receipt_sha256":
                        probe.incident_failure_receipt_sha256,
                    "incident_event_sequence":
                        meta["incident_event_sequence"],
                    "incident_event_digest":
                        meta["incident_event_digest"],
                    "incident_identity_sha256":
                        meta["incident_identity_sha256"],
                    "probe_index": probe.probe_index,
                    "remediation_epoch_sha256":
                        probe.remediation_epoch_sha256,
                    "healthy_substrate_identity_sha256":
                        probe.healthy_substrate_identity_sha256,
                    "failure_class": probe.failure_class,
                    "failure_code": probe.failure_code,
                    "health_receipt_path": probe.receipt_path,
                    "health_receipt_sha256": probe.receipt_sha256,
                    "status": probe.status,
                    "meta_request_sha256":
                        meta["meta_request_sha256"],
                    "meta_response_sha256":
                        meta["meta_response_sha256"],
                    "meta_terminal_sha256":
                        meta["meta_terminal_sha256"],
                    "recommendation": meta["recommendation"],
                    "authorization_receipt_sha256":
                        meta["authorization_receipt_sha256"],
                    "authorization_authentication_sha256":
                        meta[
                            "authorization_authentication_sha256"
                        ],
                    "rematerialization_evidence_path":
                        rematerialization_path,
                    "rematerialization_evidence_sha256":
                        rematerialization_sha256,
                    "invocation_index": 1,
                },
                recorded_at=self.clock(),
            )
            state = self.state()
            incident = state["substrate_incident"]
            if incident is None:
                raise ContiguousRunnerError(
                    "meta recovery result unexpectedly cleared latch"
                )
            meta = incident["meta_recovery"]
        if meta["phase"] == "HEALTH_RESTORED":
            result = meta["result"]
            resume_body = {
                "schema": 1,
                "kind":
                    "contiguous_meta_substrate_resume_authorization",
                "campaign_id": state["campaign_id"],
                "authorization_id": meta["authorization_id"],
                "attempt_id": meta["attempt_id"],
                "substrate_identity_sha256":
                    meta["substrate_identity_sha256"],
                "incident_failure_receipt_sha256":
                    meta["incident_failure_receipt_sha256"],
                "incident_event_sequence":
                    meta["incident_event_sequence"],
                "incident_event_digest":
                    meta["incident_event_digest"],
                "incident_identity_sha256":
                    meta["incident_identity_sha256"],
                "meta_request_sha256":
                    meta["meta_request_sha256"],
                "meta_response_sha256":
                    meta["meta_response_sha256"],
                "meta_terminal_sha256":
                    meta["meta_terminal_sha256"],
                "recommendation": meta["recommendation"],
                "operator_configuration_sha256":
                    meta["operator_configuration_sha256"],
                "recovery_result_event_sequence":
                    result["recovery_result_event_sequence"],
                "recovery_result_event_digest":
                    result["recovery_result_event_digest"],
                "health_receipt_sha256":
                    result["health_receipt_sha256"],
                "rematerialization_evidence_sha256":
                    result[
                        "rematerialization_evidence_sha256"
                    ],
                "remediation_epoch_sha256":
                    result["remediation_epoch_sha256"],
                "healthy_substrate_identity_sha256":
                    result[
                        "healthy_substrate_identity_sha256"
                    ],
                "invocation_index": 1,
                "single_use": True,
                "solver_authority": False,
                "wip_authority": False,
                "cost_authority": False,
                "promotion_authority": False,
            }
            resume_authentication = (
                meta_substrate_resume_authentication_sha256(
                    resume_body,
                    operator_configuration_sha256=(
                        meta["operator_configuration_sha256"]
                    ),
                )
            )
            resume_receipt = {
                **resume_body,
                "resume_authentication_sha256":
                    resume_authentication,
            }
            receipt_root = (
                self.root
                / META_SUBSTRATE_RECOVERY_AUTHORIZATION_ROOT
            )
            resume_path = (
                receipt_root
                / f"{meta['authorization_id']}-resume.json"
            )
            if resume_path.exists() or resume_path.is_symlink():
                if _read_json_file(resume_path) != resume_receipt:
                    raise ContiguousRunnerError(
                        "meta substrate resume receipt changed"
                    )
            else:
                _write_new_immutable_file_atomic(
                    resume_path, resume_receipt
                )
            self.journal.append(
                event_id=(
                    "meta-substrate-resume-authorized:"
                    + meta["authorization_id"]
                ),
                kind="META_SUBSTRATE_RESUME_AUTHORIZED",
                payload={
                    "authorization_id": meta["authorization_id"],
                    "attempt_id": meta["attempt_id"],
                    "incident_event_sequence":
                        meta["incident_event_sequence"],
                    "incident_event_digest":
                        meta["incident_event_digest"],
                    "incident_identity_sha256":
                        meta["incident_identity_sha256"],
                    "recovery_result_event_sequence":
                        result[
                            "recovery_result_event_sequence"
                        ],
                    "recovery_result_event_digest":
                        result["recovery_result_event_digest"],
                    "health_receipt_sha256":
                        result["health_receipt_sha256"],
                    "rematerialization_evidence_sha256":
                        result[
                            "rematerialization_evidence_sha256"
                        ],
                    "healthy_substrate_identity_sha256":
                        result[
                            "healthy_substrate_identity_sha256"
                        ],
                    "resume_receipt_path": str(resume_path),
                    "resume_receipt_sha256":
                        _sha256_file(resume_path),
                    "resume_authentication_sha256":
                        resume_authentication,
                    "invocation_index": 1,
                },
                recorded_at=self.clock(),
            )
            state = self.state()
        return dict(state)

    def apply_meta_substrate_recovery(
        self,
        *,
        meta_request_sha256: str,
        meta_response_sha256: str,
        meta_terminal_sha256: str,
        recommendation: str,
    ) -> dict[str, Any]:
        """Apply the sole allowlisted meta recommendation exactly once."""

        lock_path = self.root / ".cycle.lock"
        descriptor = _open_unaliased(
            lock_path, os.O_RDWR | os.O_CREAT
        )
        handle = os.fdopen(descriptor, "r+", encoding="utf-8")
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            state = self.state()
            incident = state["substrate_incident"]
            meta = (
                None
                if incident is None
                else incident.get("meta_recovery")
            )
            if meta is None:
                self._authorize_meta_substrate_recovery_locked(
                    state,
                    meta_request_sha256=meta_request_sha256,
                    meta_response_sha256=meta_response_sha256,
                    meta_terminal_sha256=meta_terminal_sha256,
                    recommendation=recommendation,
                    issued_at=float(self.clock()),
                )
                state = self.state()
            elif any(
                meta.get(name) != value
                for name, value in (
                    ("meta_request_sha256", meta_request_sha256),
                    ("meta_response_sha256", meta_response_sha256),
                    ("meta_terminal_sha256", meta_terminal_sha256),
                    ("recommendation", recommendation),
                )
            ):
                raise ContiguousRunnerError(
                    "meta recovery replay changed its recommendation"
                )
            return self._advance_meta_substrate_recovery_locked(
                state
            )
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            handle.close()

    def _execute_pending_substrate_reprobe(
        self, state: dict[str, Any]
    ) -> None:
        incident = state["substrate_incident"]
        if incident is None:
            return
        pending = incident["pending_reprobe"]
        if pending is None:
            return
        attempt = state["attempts"].get(incident["attempt_id"])
        if (
            attempt is None
            or attempt["phase"] != "CLOSED"
            or attempt["substrate_failure"] is None
        ):
            raise ContiguousRunnerError(
                "substrate reprobe lost its closed failed attempt"
            )
        try:
            probe = self.backend.probe_substrate_health(
                spec=attempt["spec"],
                prepared=attempt["prepared"],
                authorization_id=pending["authorization_id"],
                authorization_receipt_sha256=(
                    pending["authorization_receipt_sha256"]
                ),
                probe_index=pending["probe_index"],
                failed_substrate_identity_sha256=(
                    pending["substrate_identity_sha256"]
                ),
                incident_failure_receipt_sha256=(
                    pending["incident_failure_receipt_sha256"]
                ),
            )
            if (
                not isinstance(
                    probe, BackendSubstrateHealthProbe
                )
                or probe.authorization_id
                != pending["authorization_id"]
                or probe.probe_index != pending["probe_index"]
                or not _is_sha256(
                    probe.remediation_epoch_sha256
                )
                or probe.failed_substrate_identity_sha256
                != pending["substrate_identity_sha256"]
                or probe.incident_failure_receipt_sha256
                != pending["incident_failure_receipt_sha256"]
                or probe.status not in {"PASS", "FAILED"}
                or (
                    probe.status == "PASS"
                    and not _is_sha256(
                        probe.healthy_substrate_identity_sha256
                    )
                )
                or (
                    probe.status == "FAILED"
                    and (
                        probe.healthy_substrate_identity_sha256
                        is not None
                        or probe.failure_class not in {
                            "DETERMINISTIC_CONFIGURATION",
                            "TRANSIENT_INFRASTRUCTURE",
                        }
                        or not _safe_identifier(
                            probe.failure_code
                        )
                    )
                )
                or (
                    probe.status == "PASS"
                    and (
                        probe.failure_class is not None
                        or probe.failure_code is not None
                    )
                )
                or not _safe_path_string(probe.receipt_path)
                or not Path(probe.receipt_path).is_absolute()
                or not _is_sha256(probe.receipt_sha256)
            ):
                raise ContiguousRunnerError(
                    "backend returned malformed substrate health "
                    "evidence"
                )
        except SimulatedCrash:
            raise
        except Exception as exc:
            error_type = type(exc).__name__
            if not _safe_identifier(error_type):
                error_type = "SubstrateHealthProbeError"
            self.journal.append(
                event_id=(
                    f"substrate-reprobe-aborted:"
                    f"{pending['authorization_id']}"
                ),
                kind="SUBSTRATE_HEALTH_REPROBE_ABORTED",
                payload={
                    "authorization_id":
                        pending["authorization_id"],
                    "attempt_id": pending["attempt_id"],
                    "substrate_identity_sha256":
                        pending["substrate_identity_sha256"],
                    "incident_failure_receipt_sha256":
                        pending[
                            "incident_failure_receipt_sha256"
                        ],
                    "probe_index": pending["probe_index"],
                    "error_type": error_type,
                    "status": "ABORTED",
                },
                recorded_at=self.clock(),
            )
            return
        self.journal.append(
            event_id=(
                f"substrate-reprobe-result:"
                f"{pending['authorization_id']}"
            ),
            kind=(
                "SUBSTRATE_HEALTH_RESTORED"
                if probe.status == "PASS"
                else "SUBSTRATE_HEALTH_REPROBE_FAILED"
            ),
            payload={
                "authorization_id": probe.authorization_id,
                "attempt_id": pending["attempt_id"],
                "substrate_identity_sha256":
                    probe.failed_substrate_identity_sha256,
                "incident_failure_receipt_sha256":
                    probe.incident_failure_receipt_sha256,
                "probe_index": probe.probe_index,
                "remediation_epoch_sha256":
                    probe.remediation_epoch_sha256,
                "healthy_substrate_identity_sha256":
                    probe.healthy_substrate_identity_sha256,
                "failure_class": (
                    probe.failure_class
                ),
                "failure_code": (
                    probe.failure_code
                ),
                "health_receipt_path": probe.receipt_path,
                "health_receipt_sha256": probe.receipt_sha256,
                "status": probe.status,
            },
            recorded_at=self.clock(),
        )
        if probe.status == "PASS":
            self._record_circuit_success(
                attempt_id=pending["attempt_id"],
                operation="substrate_health_reprobe",
                evidence_kind="substrate_health_restored",
            )

    def _advance_substrate_health_circuit(
        self,
        state: dict[str, Any],
        *,
        now: float,
    ) -> None:
        """Advance at most one durable substrate-circuit boundary."""

        incident = state["substrate_incident"]
        if incident is None or state["operator_incident"] is not None:
            return
        if not incident["circuit_failure_recorded"]:
            last = incident["last_health_probe"]
            repeated_deterministic = (
                incident["failure_class"]
                == "DETERMINISTIC_CONFIGURATION"
                and incident["health_probe_count"] == 1
                and isinstance(last, dict)
                and last.get("status") == "FAILED"
                and last.get("failure_class")
                == "DETERMINISTIC_CONFIGURATION"
                and last.get("failure_code")
                == incident["failure_code"]
            )
            if repeated_deterministic:
                self.journal.append(
                    event_id="campaign:operator-incident",
                    kind="OPERATOR_INCIDENT",
                    payload={
                        "attempt_id": incident["attempt_id"],
                        "operation": "substrate_health_reprobe",
                        "fault_domain": "controller_substrate",
                        "operation_consecutive": 2,
                        "domain_consecutive": 2,
                        "threshold": 2,
                        "reason_code": (
                            "deterministic_substrate_"
                            "configuration_repeated"
                        ),
                    },
                    recorded_at=now,
                )
                return
            self._record_circuit_failure(
                attempt_id=incident["attempt_id"],
                operation="substrate_health_reprobe",
                fault_domain="controller_substrate",
            )
            state = self.state()
            incident = state["substrate_incident"]
            if (
                incident is None
                or state["operator_incident"] is not None
            ):
                return
        if incident["pending_reprobe"] is None:
            operation_state = state[
                "failure_operation_circuits"
            ].get(
                "substrate_health_reprobe:controller_substrate"
            )
            if (
                not isinstance(operation_state, dict)
                or not _is_finite_number(
                    operation_state.get("retry_not_before")
                )
                or now
                < float(operation_state["retry_not_before"])
            ):
                return
            self._authorize_substrate_health_reprobe_locked(
                state,
                issued_at=now,
                authorization_mode="sealed_autonomous_circuit",
                reason_code="substrate_circuit_deadline_reached",
            )
            state = self.state()
        if (
            state["operator_incident"] is None
            and state["substrate_incident"] is not None
            and state["substrate_incident"]["pending_reprobe"]
            is not None
        ):
            self._execute_pending_substrate_reprobe(state)

    def cycle(self, *, now: float | None = None) -> dict[str, Any]:
        """Serialize one recovery/poll/dispatch pass across host processes."""
        lock_path = self.root / ".cycle.lock"
        descriptor = _open_unaliased(
            lock_path, os.O_RDWR | os.O_CREAT
        )
        handle = os.fdopen(descriptor, "r+", encoding="utf-8")
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            try:
                return self._cycle_locked(now=now)
            except JournalStorageExhausted as exc:
                recorded_at = float(
                    self.clock() if now is None else now
                )
                self.journal.commit_storage_incident(
                    exc, recorded_at=recorded_at
                )
                return self._cycle_locked(now=now)
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            handle.close()

    @staticmethod
    def _storage_emergency_report(
        state: Mapping[str, Any],
        *,
        newly_quiesced: bool,
    ) -> dict[str, Any]:
        raw_primary = sum(
            lane["active"] is not None
            for lane in state["lanes"].values()
        )
        raw_auxiliary = sum(
            item["state"].phase
            in Scheduler.AUXILIARY_ACTIVE_PHASES
            for item in state["auxiliary_assignments"].values()
        )
        return {
            "supervision_stage_trace": [],
            "storage_emergency_stage_trace": (
                [
                    "release_storage_quiescence_reserve",
                    "contain_all_primary_children",
                    "abort_all_auxiliary_children",
                    "quarantine_ambiguous_promotions",
                    "commit_zero_authority_quiescence",
                ]
                if newly_quiesced
                else ["verify_storage_emergency_quiescence"]
            ),
            "started_attempts": [],
            "started_auxiliary_assignments": [],
            "solved_levels": state["solved_levels"],
            "total_levels": state["total_levels"],
            # These retain the reducer's historical occupancy coordinates.
            # Effective live occupancy is zero only because the authenticated
            # quiescence receipt proves every descendant absent.
            "active_lanes": raw_primary,
            "active_auxiliary_assignments": raw_auxiliary,
            "effective_live_primary_children": 0,
            "effective_live_auxiliary_children": 0,
            "draining": state["draining"],
            "complete": state["complete"],
            "operator_incident": state["operator_incident"],
            "substrate_incident": state["substrate_incident"],
            "storage_incident": state["storage_incident"],
            "storage_quiescence": state["storage_quiescence"],
            "cost_control_enabled": state["limit"] is not None,
            "recoverable_errors": [],
        }

    def _validate_storage_primary_containment(
        self,
        *,
        attempt: Mapping[str, Any],
        prior_phase: str,
        containment: BackendEmergencyContainment,
    ) -> None:
        spec = attempt["spec"]
        if (
            not isinstance(containment, BackendEmergencyContainment)
            or (
                containment.launched_container_id is not None
                and (
                    not isinstance(
                        containment.launched_container_id, str
                    )
                    or not containment.launched_container_id
                    or len(containment.launched_container_id) > 256
                    or "\x00" in containment.launched_container_id
                )
            )
            or any(
                getattr(containment, name) is not True
                for name in (
                    "attempt_container_absent",
                    "controller_roles_absent",
                    "arena_resources_absent",
                    "rpc_endpoints_absent",
                    "workspace_probe_containers_absent",
                    "host_process_groups_absent",
                    "containment_canaries_absent",
                    "no_descendants",
                )
            )
        ):
            raise ContiguousRunnerError(
                "storage emergency backend did not prove exact absence"
            )
        receipt = _validate_bound_receipt(
            containment.containment_receipt_path,
            containment.containment_receipt_sha256,
            expected_path=(
                Path(spec.host_transcript_path).parent
                / "storage_emergency_containment.json"
            ),
            expected_kind=(
                "contiguous_storage_emergency_containment"
            ),
            spec=spec,
        )
        expected = {
            "schema": RUNNER_SCHEMA,
            "kind": "contiguous_storage_emergency_containment",
            "campaign_id": spec.campaign_id,
            "generation_id": spec.generation_id,
            "attempt_id": spec.attempt_id,
            "attempt_spec_sha256":
                proposer_attempt_binding_sha256(spec),
            "prior_phase": prior_phase,
            "reason": "journal_or_storage_exhausted",
            "launched_container_id":
                containment.launched_container_id,
            "attempt_container_absent": True,
            "controller_roles_absent": True,
            "arena_resources_absent": True,
            "rpc_endpoints_absent": True,
            "workspace_probe_containers_absent": True,
            "host_process_groups_absent": True,
            "containment_canaries_absent": True,
            "no_descendants": True,
            "solver_authority": False,
            "wip_authority": False,
            "cost_authority": False,
            "promotion_authority": False,
            "status": "QUIESCED",
        }
        if receipt != expected:
            raise ContiguousRunnerError(
                "storage emergency containment receipt changed or grants "
                "authority"
            )

    def _emergency_quiesce_storage(
        self,
        state: Mapping[str, Any],
        *,
        now: float,
    ) -> dict[str, Any]:
        """Contain all live work without collection, admission, or promotion."""

        incident = state.get("storage_incident")
        if not isinstance(incident, dict):
            raise ContiguousRunnerError(
                "storage emergency quiescence lacks its incident latch"
            )
        if state.get("storage_quiescence") is not None:
            return self._storage_emergency_report(
                state, newly_quiesced=False
            )
        self.journal.release_quiescence_reserve()

        primary_containments: list[dict[str, Any]] = []
        contained_phases = {
            "PREPARED",
            "BACKEND_PREPARED",
            "RUNNING",
            "DRAINING",
            "EXITED",
            "COLLECTED",
            "COLLECTION_REJECTED",
        }
        for attempt_id, attempt in sorted(
            state["attempts"].items()
        ):
            prior_phase = attempt["phase"]
            if prior_phase not in contained_phases:
                continue
            prepared = attempt.get("prepared")
            launched = attempt.get("launched")
            if not isinstance(prepared, BackendPreparation):
                prepared = None
            if not isinstance(launched, BackendLaunch):
                launched = None
            try:
                containment = self.backend.emergency_contain(
                    spec=attempt["spec"],
                    prepared=prepared,
                    launched=launched,
                    prior_phase=prior_phase,
                    reason="journal_or_storage_exhausted",
                )
            except BaseException as exc:
                if isinstance(exc, SimulatedCrash):
                    raise
                if not isinstance(exc, Exception):
                    raise
                raise ContiguousRunnerError(
                    "storage emergency primary containment failed"
                ) from exc
            self._validate_storage_primary_containment(
                attempt=attempt,
                prior_phase=prior_phase,
                containment=containment,
            )
            primary_containments.append({
                "attempt_id": attempt_id,
                "prior_phase": prior_phase,
                "containment": asdict(containment),
            })

        auxiliary_aborts: list[dict[str, Any]] = []
        for assignment_id, assignment in sorted(
            state["auxiliary_assignments"].items()
        ):
            prior_phase = assignment["state"].phase
            if prior_phase not in Scheduler.AUXILIARY_ACTIVE_PHASES:
                continue
            backend = self.auxiliary_backend
            if backend is None:
                raise ContiguousRunnerError(
                    "storage emergency found auxiliary work without its "
                    "backend"
                )
            self._rebind_auxiliary_prerequisites(assignment)
            try:
                aborted = self._validate_auxiliary_value(
                    backend.abort(
                        assignment["decision"],
                        assignment["prepared"],
                        assignment["launched"],
                        prior_phase=prior_phase,
                        reason="journal_or_storage_exhausted",
                    ),
                    AuxiliaryAbort,
                    "storage emergency auxiliary abort proof",
                )
                Scheduler.charge_to_units(aborted.cost_used)
            except BaseException as exc:
                if isinstance(exc, SimulatedCrash):
                    raise
                if not isinstance(exc, Exception):
                    raise
                raise ContiguousRunnerError(
                    "storage emergency auxiliary containment failed"
                ) from exc
            teardown_path: str | None = None
            teardown_sha256: str | None = None
            if prior_phase == "RUNNING":
                teardown = self._validate_auxiliary_value(
                    aborted.teardown,
                    AuxiliaryTeardown,
                    "storage emergency auxiliary teardown",
                )
                self._verify_auxiliary_receipt(
                    assignment["decision"],
                    teardown.teardown_receipt_path,
                    teardown.teardown_receipt_sha256,
                    expected={
                        "schema": 1,
                        "kind":
                            "auxiliary_backend_abort_teardown",
                        "assignment_id": assignment_id,
                        "backend_contract_sha256":
                            assignment["decision"]
                            .backend_contract_sha256,
                        "prior_phase": "RUNNING",
                        "descendants_absent": True,
                        "live_lineage_mutated": False,
                    },
                    label=(
                        "storage emergency auxiliary teardown"
                    ),
                )
                teardown_path = teardown.teardown_receipt_path
                teardown_sha256 = (
                    teardown.teardown_receipt_sha256
                )
            elif aborted.teardown is not None:
                raise ContiguousRunnerError(
                    "unlaunched storage emergency auxiliary returned "
                    "teardown evidence"
                )
            auxiliary_aborts.append({
                "assignment_id": assignment_id,
                "prior_phase": prior_phase,
                "teardown_receipt_path": teardown_path,
                "teardown_receipt_sha256": teardown_sha256,
                "no_descendants": True,
                "cost_authority": False,
            })

        promotion_quarantines: list[dict[str, Any]] = []
        for attempt_id, attempt in sorted(
            state["attempts"].items()
        ):
            if attempt["phase"] != "PROMOTING":
                continue
            recover = getattr(self.promotion_gate, "recover", None)
            if not callable(recover):
                raise ContiguousRunnerError(
                    "storage emergency cannot reconcile an ambiguous "
                    "promotion"
                )
            try:
                commit = recover(
                    spec=attempt["spec"],
                    candidate=attempt["candidate"],
                )
            except BaseException as exc:
                if isinstance(exc, SimulatedCrash):
                    raise
                if not isinstance(exc, Exception):
                    raise
                raise ContiguousRunnerError(
                    "storage emergency promotion reconciliation failed"
                ) from exc
            if commit is not None and not isinstance(
                commit, PromotionCommit
            ):
                raise ContiguousRunnerError(
                    "storage emergency promotion recovery returned an "
                    "invalid value"
                )
            promotion_quarantines.append({
                "attempt_id": attempt_id,
                "external_commit_observed": commit is not None,
                "external_commit_sha256": (
                    hashlib.sha256(
                        _canonical_json(asdict(commit))
                    ).hexdigest()
                    if commit is not None
                    else None
                ),
                "promotion_authority": False,
            })

        payload = {
            "storage_incident_event_sequence":
                incident["incident_event_sequence"],
            "storage_incident_event_digest":
                incident["incident_event_digest"],
            "primary_containments": primary_containments,
            "auxiliary_aborts": auxiliary_aborts,
            "promotion_quarantines": promotion_quarantines,
            "all_primary_children_absent": True,
            "all_auxiliary_children_absent": True,
            "all_promotions_non_authoritative": True,
            "solver_authority": False,
            "wip_authority": False,
            "cost_authority": False,
            "promotion_authority": False,
            "status": "QUIESCED",
        }
        self.journal.append(
            event_id="campaign:storage-emergency-quiesced",
            kind="STORAGE_EMERGENCY_QUIESCED",
            payload=payload,
            recorded_at=now,
        )
        quiesced = self.state()
        if quiesced["storage_quiescence"] != payload:
            raise ContiguousRunnerError(
                "storage emergency event did not reduce to exact "
                "quiescence"
            )
        return self._storage_emergency_report(
            quiesced, newly_quiesced=True
        )

    def _cycle_locked(
        self, *, now: float | None = None
    ) -> dict[str, Any]:
        """Advance recovery/polling once, then fill eligible independent lanes."""
        selected_now = float(self.clock() if now is None else now)
        state = self.state()
        if state["storage_incident"] is not None:
            return self._emergency_quiesce_storage(
                state, now=selected_now
            )
        touched_games: set[str] = set()
        recoverable_errors: list[str] = []
        stage_trace: list[str] = []

        def enter_stage(stage: str) -> None:
            index = len(stage_trace)
            if (
                index >= len(Scheduler.SUPERVISION_CYCLE_STAGES)
                or Scheduler.SUPERVISION_CYCLE_STAGES[index] != stage
            ):
                raise ContiguousRunnerError(
                    "supervision cycle departed from policy order"
                )
            stage_trace.append(stage)

        # A durable decision must be consumed before *any* other journal event.
        # Recovery performs no external side effect: reservation still precedes
        # generation-directory creation, process creation, or network access.
        enter_stage("consume_durable_decisions")
        if (
            state["storage_incident"] is None
            and state["pending_auxiliary_decision"] is not None
        ):
            self._reserve_auxiliary(state)
            state = self.state()
        if (
            state["storage_incident"] is None
            and state["pending_scheduler_decision"] is not None
        ):
            self._reserve_attempt(state)
            state = self.state()
        if state["substrate_incident"] is not None:
            meta_recovery = state["substrate_incident"].get(
                "meta_recovery"
            )
            if (
                isinstance(meta_recovery, dict)
                and meta_recovery.get("phase")
                in {"AUTHORIZED", "HEALTH_RESTORED"}
            ):
                state = self._advance_meta_substrate_recovery_locked(
                    state
                )
            elif state["operator_incident"] is None:
                self._advance_substrate_health_circuit(
                    state, now=selected_now
                )
            state = self.state()

        # Poll durable live containers before admitting any unlaunched turn.  A
        # process restart may occur after a soft deadline, and polling first must
        # establish the lane-local DRAINING identity before dispatch decisions.
        enter_stage("poll_live_attempts")
        for attempt_id, attempt in sorted(state["attempts"].items()):
            if (
                attempt["phase"] in {"RUNNING", "DRAINING"}
                and self._operation_ready(
                    state,
                    attempt,
                    "backend_poll",
                    now=selected_now,
                    cleanup=state["operator_incident"] is not None,
                )
            ):
                touched_games.add(attempt["spec"].game)
                try:
                    self._poll_attempt(
                        attempt_id, attempt, now=selected_now
                    )
                except JournalStorageExhausted:
                    raise
                except ContiguousRunnerError as exc:
                    recoverable_errors.append(
                        f"{attempt_id}: poll: {exc}"
                    )

        state = self.state()
        enter_stage("collect_terminal_evidence")
        for attempt_id, attempt in sorted(state["attempts"].items()):
            if (
                attempt["phase"] == "EXITED"
                and self._operation_ready(
                    state,
                    attempt,
                    "backend_collect",
                    now=selected_now,
                    cleanup=state["operator_incident"] is not None,
                )
            ):
                touched_games.add(attempt["spec"].game)
                try:
                    self._collect_exited(
                        attempt_id, attempt, now=selected_now
                    )
                except JournalStorageExhausted:
                    raise
                except ContiguousRunnerError as exc:
                    recoverable_errors.append(
                        f"{attempt_id}: collect: {exc}"
                    )

        state = self.state()
        enter_stage("prove_teardown")
        for attempt_id, attempt in sorted(state["attempts"].items()):
            if attempt["phase"] in {
                "COLLECTED", "COLLECTION_REJECTED"
            } and self._operation_ready(
                state,
                attempt,
                "backend_teardown",
                now=selected_now,
                cleanup=True,
            ):
                touched_games.add(attempt["spec"].game)
                try:
                    self._teardown_collected(
                        attempt_id, attempt, now=selected_now
                    )
                except JournalStorageExhausted:
                    raise
                except ContiguousRunnerError as exc:
                    recoverable_errors.append(
                        f"{attempt_id}: teardown: {exc}"
                    )

        state = self.state()
        enter_stage("classify_and_settle")
        for attempt_id, attempt in sorted(state["attempts"].items()):
            if attempt["phase"] == "TORN_DOWN":
                touched_games.add(attempt["spec"].game)
                try:
                    self._record_torn_down_result(
                        attempt_id, attempt, now=selected_now
                    )
                except JournalStorageExhausted:
                    raise
                except ContiguousRunnerError as exc:
                    recoverable_errors.append(
                        f"{attempt_id}: result: {exc}"
                    )
        try:
            self._admit_pending_native_sidecar_requests(
                self.state()
            )
        except JournalStorageExhausted:
            raise
        except ContiguousRunnerError as exc:
            recoverable_errors.append(
                f"native sidecar request admission: {exc}"
            )

        state = self.state()
        enter_stage("commit_exact_promotions")
        for attempt_id, attempt in sorted(state["attempts"].items()):
            if (
                attempt["phase"] == "PROMOTING"
                and state["operator_incident"] is None
                and state["storage_incident"] is None
                and self._operation_ready(
                    state,
                    attempt,
                    "promotion_commit",
                    now=selected_now,
                )
                and self._operation_ready(
                    state,
                    attempt,
                    "promotion_recover",
                    now=selected_now,
                )
            ):
                touched_games.add(attempt["spec"].game)
                try:
                    self._commit_pending(
                        attempt_id, attempt, now=selected_now
                    )
                except JournalStorageExhausted:
                    raise
                except ContiguousRunnerError as exc:
                    recoverable_errors.append(
                        f"{attempt_id}: promotion: {exc}"
                    )

        started_auxiliary: list[str] = []
        # Promotion never waits for a sidecar.  Its journal transition marks
        # old-frontier assignments invalid, after which containment/teardown and
        # authenticated usage settlement run before any capacity is reused.
        state = self.state()
        enter_stage("invalidate_or_admit_auxiliary_outputs")
        for assignment_id, assignment in sorted(
            state["auxiliary_assignments"].items()
        ):
            phase = assignment["state"].phase
            try:
                if (
                    (
                        assignment["state"].invalidated
                        or state["operator_incident"] is not None
                        or state["storage_incident"] is not None
                    )
                    and phase in Scheduler.AUXILIARY_ACTIVE_PHASES
                ):
                    self._abort_auxiliary(
                        assignment,
                        reason=(
                            "frontier_promoted"
                            if assignment["state"].invalidated
                            else "operator_incident"
                        ),
                    )
                elif (
                    assignment["state"].invalidated
                    and phase == "QUARANTINED"
                ):
                    self._reject_auxiliary_stale(assignment)
                elif (
                    phase == "RUNNING"
                    and self._failure_circuit_operation_ready(
                        state,
                        "auxiliary_poll",
                        now=selected_now,
                    )
                ):
                    self._poll_auxiliary(assignment)
                elif (
                    phase == "QUARANTINED"
                    and self._failure_circuit_operation_ready(
                        state,
                        "auxiliary_admit",
                        now=selected_now,
                    )
                ):
                    self._admit_auxiliary(assignment)
            except JournalStorageExhausted:
                raise
            except ContiguousRunnerError as exc:
                recoverable_errors.append(
                    f"{assignment_id}: auxiliary {phase.lower()}: {exc}"
                )
        try:
            self._admit_pending_supervisory_sidecar_requests(
                self.state()
            )
        except JournalStorageExhausted:
            raise
        except ContiguousRunnerError as exc:
            recoverable_errors.append(
                f"supervisory sidecar request admission: {exc}"
            )

        # Recover reserved and prepared sidecars with the same idempotent
        # reservation-before-side-effect discipline as proposer attempts.
        enter_stage("recover_reserved_auxiliary_work")
        for phase in ("RESERVED", "INPUT_PREPARED"):
            current = self.state()
            if (
                current["operator_incident"] is not None
                or current["substrate_incident"] is not None
                or current["storage_incident"] is not None
            ):
                break
            for assignment_id, assignment in sorted(
                current["auxiliary_assignments"].items()
            ):
                live_state = self.state()
                if (
                    live_state["operator_incident"] is not None
                    or live_state["substrate_incident"] is not None
                    or live_state["storage_incident"] is not None
                ):
                    break
                if assignment["state"].phase != phase:
                    continue
                operation = (
                    "auxiliary_prepare"
                    if phase == "RESERVED"
                    else "auxiliary_launch"
                )
                if not self._failure_circuit_operation_ready(
                    current,
                    operation,
                    now=selected_now,
                ):
                    continue
                try:
                    if assignment["state"].invalidated:
                        self._abort_auxiliary(
                            assignment, reason="frontier_promoted"
                        )
                    elif phase == "RESERVED":
                        self._prepare_auxiliary(assignment)
                    else:
                        self._launch_auxiliary(assignment)
                        if (
                            self.state()["auxiliary_assignments"][
                                assignment_id
                            ]["state"].phase
                            == "RUNNING"
                        ):
                            started_auxiliary.append(assignment_id)
                except JournalStorageExhausted:
                    raise
                except ContiguousRunnerError as exc:
                    recoverable_errors.append(
                        f"{assignment_id}: auxiliary "
                        f"{phase.lower()}: {exc}"
                    )
        # A just-prepared sidecar can be launched in this pass.
        current = self.state()
        for assignment_id, assignment in sorted(
            current["auxiliary_assignments"].items()
        ):
            live_state = self.state()
            if (
                live_state["operator_incident"] is not None
                or live_state["substrate_incident"] is not None
                or live_state["storage_incident"] is not None
                or
                assignment["state"].phase != "INPUT_PREPARED"
                or assignment_id in started_auxiliary
                or not self._failure_circuit_operation_ready(
                    current,
                    "auxiliary_launch",
                    now=selected_now,
                )
            ):
                continue
            try:
                if assignment["state"].invalidated:
                    self._abort_auxiliary(
                        assignment, reason="frontier_promoted"
                    )
                else:
                    self._launch_auxiliary(assignment)
                    if (
                        self.state()["auxiliary_assignments"][
                            assignment_id
                        ]["state"].phase
                        == "RUNNING"
                    ):
                        started_auxiliary.append(assignment_id)
            except JournalStorageExhausted:
                raise
            except ContiguousRunnerError as exc:
                recoverable_errors.append(
                    f"{assignment_id}: auxiliary launch: {exc}"
                )

        state = self.state()
        started: list[str] = []
        enter_stage("recover_or_dispatch_distinct_primary_frontiers")
        # DRAINING is lane-local: that game's active identity prevents overlap,
        # while unrelated games remain dispatchable to preserve useful capacity.
        if (
            not state["complete"]
            and state["operator_incident"] is None
            and state["substrate_incident"] is None
            and state["storage_incident"] is None
        ):
            active_count = sum(
                lane["active"] is not None
                for lane in state["lanes"].values()
            )
            # Cost admission applies only to new reservations.  An identity
            # already admitted before the cutoff must finish materialization
            # and launch; abandoning it would make recovery non-idempotent.
            if not state["complete"]:
                current = self.state()
                for attempt_id, attempt in sorted(
                    current["attempts"].items()
                ):
                    if attempt["phase"] != "RESERVED":
                        continue
                    if not self._operation_ready(
                        current,
                        attempt,
                        "input_materialize",
                        now=selected_now,
                    ):
                        continue
                    touched_games.add(
                        attempt["reservation"].game
                    )
                    try:
                        self._materialize_reserved(
                            attempt["reservation"]
                        )
                    except JournalStorageExhausted:
                        raise
                    except ContiguousRunnerError as exc:
                        recoverable_errors.append(
                            f"{attempt_id}: materialize: {exc}"
                        )

                # Recover prepared identities first.  Both methods are required
                # to be idempotent because a crash can occur after the external
                # transition and before its journal acknowledgement.
                for phase in ("PREPARED", "BACKEND_PREPARED"):
                    current = self.state()
                    for attempt_id, attempt in sorted(
                        current["attempts"].items()
                    ):
                        live_state = self.state()
                        if (
                            live_state["operator_incident"] is not None
                            or live_state["substrate_incident"]
                            is not None
                            or live_state["storage_incident"]
                            is not None
                        ):
                            break
                        if attempt["phase"] != phase:
                            continue
                        operation = (
                            "backend_prepare"
                            if phase == "PREPARED"
                            else "backend_launch"
                        )
                        if not self._operation_ready(
                            current,
                            attempt,
                            operation,
                            now=selected_now,
                        ):
                            continue
                        touched_games.add(attempt["spec"].game)
                        try:
                            if phase == "PREPARED":
                                self._prepare_backend(
                                    attempt_id, attempt["spec"]
                                )
                            else:
                                launched_now = self._launch_backend(
                                    attempt_id,
                                    attempt["spec"],
                                    attempt["prepared"],
                                )
                                if launched_now:
                                    started.append(attempt_id)
                        except JournalStorageExhausted:
                            raise
                        except ContiguousRunnerError as exc:
                            recoverable_errors.append(
                                f"{attempt_id}: {phase.lower()}: {exc}"
                            )

                current = self.state()
                # PREPARED just acknowledged above becomes BACKEND_PREPARED.
                for attempt_id, attempt in sorted(
                    current["attempts"].items()
                ):
                    live_state = self.state()
                    if (
                        live_state["operator_incident"] is not None
                        or live_state["substrate_incident"] is not None
                        or live_state["storage_incident"] is not None
                    ):
                        break
                    if attempt["phase"] != "BACKEND_PREPARED":
                        continue
                    if attempt_id in started:
                        continue
                    if not self._operation_ready(
                        current,
                        attempt,
                        "backend_launch",
                        now=selected_now,
                    ):
                        continue
                    try:
                        launched_now = self._launch_backend(
                            attempt_id,
                            attempt["spec"],
                            attempt["prepared"],
                        )
                        if launched_now:
                            started.append(attempt_id)
                    except JournalStorageExhausted:
                        raise
                    except ContiguousRunnerError as exc:
                        recoverable_errors.append(
                            f"{attempt_id}: launch: {exc}"
                        )

                while True:
                    current = self.state()
                    active_count = sum(
                        lane["active"] is not None
                        for lane in current["lanes"].values()
                    ) + sum(
                        item["state"].phase
                        in Scheduler.AUXILIARY_ACTIVE_PHASES
                        for item in current[
                            "auxiliary_assignments"
                        ].values()
                    )
                    if (
                        current["complete"]
                        or current["operator_incident"] is not None
                        or current["substrate_incident"] is not None
                        or current["storage_incident"] is not None
                        or not self._failure_circuit_operation_ready(
                            current,
                            "dispatch_primary",
                            now=selected_now,
                        )
                        or active_count >= current["max_lanes"]
                    ):
                        break
                    reservation = self._reserve_attempt(current)
                    if reservation is None:
                        break
                    touched_games.add(reservation.game)
                    try:
                        spec = self._materialize_reserved(reservation)
                        self._prepare_backend(spec.attempt_id, spec)
                        prepared = self.state()["attempts"][
                            spec.attempt_id
                        ]["prepared"]
                        launched_now = self._launch_backend(
                            spec.attempt_id, spec, prepared
                        )
                        if launched_now:
                            started.append(spec.attempt_id)
                    except JournalStorageExhausted:
                        raise
                    except ContiguousRunnerError as exc:
                        recoverable_errors.append(
                            f"{reservation.game}: dispatch: {exc}"
                        )
                # Only after every eligible primary proposer has been reserved
                # may otherwise-idle capacity become an independent sidecar.
                enter_stage("dispatch_eligible_auxiliary_analysis")
                while True:
                    current = self.state()
                    active_total = sum(
                        lane["active"] is not None
                        for lane in current["lanes"].values()
                    ) + sum(
                        item["state"].phase
                        in Scheduler.AUXILIARY_ACTIVE_PHASES
                        for item in current[
                            "auxiliary_assignments"
                        ].values()
                    )
                    if active_total >= current["max_lanes"]:
                        break
                    if (
                        current["operator_incident"] is not None
                        or current["substrate_incident"] is not None
                        or current["storage_incident"] is not None
                        or not self._failure_circuit_operation_ready(
                            current,
                            "dispatch_auxiliary",
                            now=selected_now,
                        )
                    ):
                        break
                    assignment_id = self._reserve_auxiliary(current)
                    if assignment_id is None:
                        break
                    try:
                        assignment = self.state()[
                            "auxiliary_assignments"
                        ][assignment_id]
                        self._prepare_auxiliary(assignment)
                        assignment = self.state()[
                            "auxiliary_assignments"
                        ][assignment_id]
                        if (
                            assignment["state"].phase
                            == "INPUT_PREPARED"
                        ):
                            self._launch_auxiliary(assignment)
                            if (
                                self.state()[
                                    "auxiliary_assignments"
                                ][assignment_id]["state"].phase
                                == "RUNNING"
                            ):
                                started_auxiliary.append(
                                    assignment_id
                                )
                    except JournalStorageExhausted:
                        raise
                    except ContiguousRunnerError as exc:
                        recoverable_errors.append(
                            f"{assignment_id}: auxiliary dispatch: {exc}"
                        )
                        break
        if len(stage_trace) == len(Scheduler.SUPERVISION_CYCLE_STAGES) - 1:
            # A completed campaign performs no auxiliary dispatch, but the
            # no-op policy stage remains explicit and auditable.
            enter_stage("dispatch_eligible_auxiliary_analysis")
        if tuple(stage_trace) != Scheduler.SUPERVISION_CYCLE_STAGES:
            raise ContiguousRunnerError(
                "supervision cycle did not execute the complete policy"
            )
        final = self.state()
        return {
            "supervision_stage_trace": stage_trace,
            "started_attempts": started,
            "started_auxiliary_assignments": started_auxiliary,
            "solved_levels": final["solved_levels"],
            "total_levels": final["total_levels"],
            "active_lanes": sum(
                lane["active"] is not None
                for lane in final["lanes"].values()
            ),
            "active_auxiliary_assignments": sum(
                item["state"].phase
                in Scheduler.AUXILIARY_ACTIVE_PHASES
                for item in final["auxiliary_assignments"].values()
            ),
            "draining": final["draining"],
            "complete": final["complete"],
            "operator_incident": final["operator_incident"],
            "substrate_incident": final["substrate_incident"],
            "storage_incident": final["storage_incident"],
            "cost_control_enabled": final["limit"] is not None,
            "recoverable_errors": recoverable_errors,
        }


def _terminal_retention_root(campaign_root: Path) -> Path:
    return campaign_root / TERMINAL_RETENTION_EVIDENCE_NAME


def _terminal_retention_intent_path(campaign_root: Path) -> Path:
    return campaign_root / TERMINAL_RETENTION_INTENT_NAME


def _terminal_retention_receipt_path(campaign_root: Path) -> Path:
    return campaign_root / TERMINAL_RETENTION_RECEIPT_NAME


def _terminal_retention_runner_body(
    runner_state_receipt: object,
) -> dict[str, Any]:
    if not isinstance(runner_state_receipt, dict):
        raise ContiguousRunnerError(
            "terminal retention requires a runner audit receipt"
        )
    body = {
        key: value
        for key, value in runner_state_receipt.items()
        if key != "receipt_sha256"
    }
    if (
        runner_state_receipt.get("status") != "PASS"
        or runner_state_receipt.get("receipt_sha256")
        != hashlib.sha256(_canonical_json(body)).hexdigest()
        or not _is_sha256(
            runner_state_receipt.get("journal_head_digest")
        )
        or not isinstance(
            runner_state_receipt.get("journal_head_sequence"), int
        )
        or isinstance(
            runner_state_receipt.get("journal_head_sequence"), bool
        )
        or runner_state_receipt["journal_head_sequence"] < 1
    ):
        raise ContiguousRunnerError(
            "terminal retention runner receipt is malformed"
        )
    return runner_state_receipt


def _terminal_retention_prerequisites(
    value: Mapping[str, str] | None,
) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ContiguousRunnerError(
            "terminal retention prerequisite audits are malformed"
        )
    result: dict[str, str] = {}
    for key, digest in sorted(
        value.items(), key=lambda pair: str(pair[0])
    ):
        if (
            not _safe_identifier(key)
            or not _is_sha256(digest)
            or key in result
        ):
            raise ContiguousRunnerError(
                "terminal retention prerequisite audit is malformed"
            )
        result[str(key)] = str(digest)
    return result


def _terminal_retention_state(
    campaign_root: Path,
    *,
    secret_sentinels: tuple[str, ...],
    controller_state_canaries: tuple[Taint.LiveCanary, ...],
) -> dict[str, Any]:
    """Return the independently replayed state used only to build an intent."""

    root = Path(campaign_root)
    reducer = object.__new__(ContiguousCampaignRunner)
    reducer.root = root
    reducer.journal = ReadOnlyAttemptJournal(
        root / "attempt_journal"
    )
    reducer.generations = root / "generations"
    reducer.auxiliary = root / "auxiliary"
    reducer._secret_sentinels = secret_sentinels
    reducer._controller_state_canaries = controller_state_canaries
    events = reducer.journal.read()
    if not events:
        raise ContiguousRunnerError(
            "terminal retention found no journal history"
        )
    try:
        reducer.auxiliary_launch_configuration = (
            Scheduler.auxiliary_launch_configuration_from_dict(
                events[0]["payload"].get(
                    "auxiliary_launch_configuration"
                )
            )
        )
    except Scheduler.SchedulerError as exc:
        raise ContiguousRunnerError(
            "terminal retention found invalid auxiliary configuration"
        ) from exc
    return ContiguousCampaignRunner.state(reducer)


def _terminal_retention_is_compact_path_field(name: str) -> bool:
    if name == "controller_canary_escrow_path":
        return False
    return (
        name.endswith("_receipt_path")
        or name
        in {
            "launch_attestation_path",
            "thread_binding_path",
            "final_thread_binding_path",
            "canary_reveal_path",
            "candidate_manifest_path",
            "frontier_brief_path",
            "bridge_policy_path",
        }
    )


def _terminal_retention_path_pairs(
    value: object,
    *,
    prefix: str,
) -> list[tuple[str, str, str]]:
    """Collect explicitly hash-bound compact evidence path fields."""

    if is_dataclass(value):
        value = asdict(value)
    if not isinstance(value, Mapping):
        return []
    result: list[tuple[str, str, str]] = []
    for key, item in sorted(value.items(), key=lambda pair: str(pair[0])):
        name = str(key)
        role = f"{prefix}.{name}" if prefix else name
        if (
            name.endswith("_path")
            and _terminal_retention_is_compact_path_field(name)
        ):
            hash_name = name[:-5] + "_sha256"
            if hash_name not in value:
                continue
            digest = value.get(hash_name)
            if item is not None or digest is not None:
                if (
                    not _safe_path_string(item)
                    or not _is_sha256(digest)
                ):
                    raise ContiguousRunnerError(
                        f"terminal compact evidence pair is malformed: {role}"
                    )
                result.append((role[:-5], str(item), str(digest)))
        if is_dataclass(item) or isinstance(item, Mapping):
            result.extend(
                _terminal_retention_path_pairs(item, prefix=role)
            )
    return result


def _terminal_retention_generation_path(
    generations: Path,
    generation_id: str,
) -> Path:
    if not _is_uuid4(generation_id):
        raise ContiguousRunnerError(
            "terminal retention generation identity is malformed"
        )
    selected = generations / generation_id
    if selected.parent != generations:
        raise ContiguousRunnerError(
            "terminal retention generation escaped its root"
        )
    return selected


def _terminal_retention_regular_json(
    path: Path,
    *,
    expected_sha256: str,
) -> tuple[bytes, int]:
    if not _is_sha256(expected_sha256):
        raise ContiguousRunnerError(
            "terminal compact evidence hash is malformed"
        )
    payload = _bounded_regular_bytes(
        path, maximum=MAX_TERMINAL_COMPACT_EVIDENCE_BYTES
    )
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ContiguousRunnerError(
            f"terminal compact evidence changed: {path}"
        )
    try:
        json.loads(payload)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ContiguousRunnerError(
            f"terminal compact evidence is not JSON: {path}"
        ) from exc
    return payload, len(payload)


def _terminal_retention_plan(
    campaign_root: Path,
    *,
    state: Mapping[str, Any],
    runner_state_receipt: Mapping[str, Any],
    pre_cleanup_audits: Mapping[str, str],
) -> dict[str, Any]:
    if (
        state.get("complete") is not True
        or state.get("solved_levels") != state.get("total_levels")
        or state.get("pending_scheduler_decision") is not None
        or state.get("pending_auxiliary_decision") is not None
    ):
        raise ContiguousRunnerError(
            "terminal retention is final-campaign-only"
        )
    attempts = state.get("attempts")
    if not isinstance(attempts, Mapping):
        raise ContiguousRunnerError(
            "terminal retention attempts are malformed"
        )
    generations = campaign_root / "generations"
    generation_ids: list[str] = []
    compact_exports: list[dict[str, Any]] = []
    aggregate_bytes = 0
    for attempt_id, attempt_value in sorted(
        attempts.items(), key=lambda pair: str(pair[0])
    ):
        if (
            not _safe_identifier(attempt_id)
            or not isinstance(attempt_value, Mapping)
            or attempt_value.get("phase") != "CLOSED"
        ):
            raise ContiguousRunnerError(
                "terminal retention requires every attempt to be closed"
            )
        spec = attempt_value.get("spec")
        reservation = attempt_value.get("reservation")
        if (
            not isinstance(spec, AttemptSpec)
            or not isinstance(reservation, AttemptReservation)
            or spec.attempt_id != attempt_id
            or reservation.attempt_id != attempt_id
            or spec.generation_id != reservation.generation_id
        ):
            raise ContiguousRunnerError(
                "terminal retention attempt binding is incomplete"
            )
        generation_id = spec.generation_id
        generation = _terminal_retention_generation_path(
            generations, generation_id
        )
        if generation.is_symlink() or not generation.is_dir():
            raise ContiguousRunnerError(
                "terminal retention requires every generation before intent"
            )
        generation_ids.append(generation_id)
        settled_result = attempt_value.get("settled_result")
        if not isinstance(settled_result, AttemptResult):
            raise ContiguousRunnerError(
                "terminal retention attempt lacks typed settlement"
            )
        # Tainted, protocol-invalid, and infrastructure turns retain only
        # their authenticated journal/retention-ledger metadata.  Their whole
        # generation (including raw turn bytes) is purged after the terminal
        # audit and none of it is copied into the compact archive.
        if settled_result.kind in {
            "tainted",
            "protocol_invalid",
            "infrastructure",
        }:
            continue
        candidates: list[tuple[str, str, str]] = []
        spec_path = generation / "attempt_spec.json"
        expected_spec_file = json.loads(
            _canonical_json(_spec_to_dict(spec))
        )
        if _read_json_file(spec_path) != expected_spec_file:
            raise ContiguousRunnerError(
                "terminal attempt spec differs from its journaled value"
            )
        candidates.append(
            (
                "attempt_spec",
                str(spec_path),
                _sha256_file(spec_path),
            )
        )
        collection = attempt_value.get("collection")
        if isinstance(collection, BackendCollection):
            candidates.append(
                (
                    "worker_outcome",
                    str(Path(spec.output_dir) / WORKER_OUTCOME_NAME),
                    collection.worker_outcome_sha256,
                )
            )
        for record_name in (
            "spec",
            "prepared",
            "launched",
            "collection",
            "teardown",
            "settled_result",
        ):
            record = attempt_value.get(record_name)
            if record is not None:
                candidates.extend(
                    _terminal_retention_path_pairs(
                        record, prefix=record_name
                    )
                )

        grouped: dict[str, dict[str, Any]] = {}
        for role, source_string, digest in candidates:
            source = Path(source_string)
            if not source.is_absolute():
                raise ContiguousRunnerError(
                    "terminal compact evidence path is not absolute"
                )
            try:
                if source.resolve(strict=True) != source:
                    raise ContiguousRunnerError(
                        "terminal compact evidence path is aliased"
                    )
                _regular_file_pointer(source)
            except OSError as exc:
                raise ContiguousRunnerError(
                    "terminal compact evidence path cannot be reopened"
                ) from exc
            generation_relative: Path | None = None
            campaign_relative: Path | None = None
            absolute_source: str | None = None
            try:
                generation_relative = source.relative_to(generation)
            except ValueError:
                try:
                    campaign_relative = source.relative_to(campaign_root)
                except ValueError:
                    absolute_source = str(source)
            relative = (
                generation_relative
                if generation_relative is not None
                else campaign_relative
            )
            if (
                relative is not None
                and (
                    not relative.parts
                    or any(
                        part in {"", ".", ".."}
                        for part in relative.parts
                    )
                )
            ):
                raise ContiguousRunnerError(
                    "terminal compact evidence relative path is unsafe"
                )
            payload, byte_count = _terminal_retention_regular_json(
                source, expected_sha256=digest
            )
            del payload
            aggregate_bytes += byte_count
            if (
                aggregate_bytes
                > MAX_TERMINAL_COMPACT_EVIDENCE_TOTAL_BYTES
            ):
                raise ContiguousRunnerError(
                    "terminal compact evidence exceeds aggregate bound"
                )
            selected = grouped.setdefault(
                digest,
                {
                    "attempt_id": attempt_id,
                    "generation_id": generation_id,
                    "evidence_sha256": digest,
                    "byte_count": byte_count,
                    "references": [],
                    "source_relative_paths": [],
                    "source_campaign_relative_paths": [],
                    "source_absolute_paths": [],
                    "retained_relative_path":
                        f"{attempt_id}/{digest}.json",
                },
            )
            if selected["byte_count"] != byte_count:
                raise ContiguousRunnerError(
                    "equal evidence hashes report unequal byte counts"
                )
            selected["references"].append(role)
            if generation_relative is not None:
                selected["source_relative_paths"].append(
                    generation_relative.as_posix()
                )
            elif campaign_relative is not None:
                selected["source_campaign_relative_paths"].append(
                    campaign_relative.as_posix()
                )
            elif absolute_source is not None:
                selected["source_absolute_paths"].append(absolute_source)
        for selected in grouped.values():
            selected["references"] = sorted(set(selected["references"]))
            selected["source_relative_paths"] = sorted(
                set(selected["source_relative_paths"])
            )
            selected["source_campaign_relative_paths"] = sorted(
                set(selected["source_campaign_relative_paths"])
            )
            selected["source_absolute_paths"] = sorted(
                set(selected["source_absolute_paths"])
            )
            compact_exports.append(selected)

    if len(generation_ids) != len(set(generation_ids)):
        raise ContiguousRunnerError(
            "terminal retention generation identity was reused"
        )
    if (
        sorted(str(item) for item in attempts)
        != runner_state_receipt.get("attempt_ids")
        or sorted(generation_ids)
        != runner_state_receipt.get("generation_ids")
    ):
        raise ContiguousRunnerError(
            "terminal retention state differs from runner audit identities"
        )
    compact_exports.sort(
        key=lambda item: (
            item["attempt_id"],
            item["retained_relative_path"],
        )
    )
    lane_authorities = []
    for item in runner_state_receipt.get("lane_boundaries", ()):
        if not isinstance(item, Mapping):
            raise ContiguousRunnerError(
                "terminal runner lane authority is malformed"
            )
        lane_authorities.append(
            {
                "game": item.get("game"),
                "target": item.get("target"),
                "reached": item.get("reached"),
                "checkpoint_path": item.get("checkpoint_path"),
                "checkpoint_sha256": item.get("checkpoint_sha256"),
                "source_path": item.get("source_path"),
                "source_tree_sha256": item.get(
                    "source_tree_sha256"
                ),
            }
        )
    lane_authorities.sort(key=lambda item: str(item["game"]))
    body = {
        "schema": TERMINAL_RETENTION_SCHEMA,
        "kind": "arc_agi3_terminal_attempt_retention_intent",
        "campaign_root": str(campaign_root),
        "campaign_id": runner_state_receipt["campaign_id"],
        "runner_state_receipt": dict(runner_state_receipt),
        "runner_state_receipt_sha256":
            runner_state_receipt["receipt_sha256"],
        "journal_head_sequence":
            runner_state_receipt["journal_head_sequence"],
        "journal_head_digest":
            runner_state_receipt["journal_head_digest"],
        "solved_levels": runner_state_receipt["solved_levels"],
        "total_levels": runner_state_receipt["total_levels"],
        "complete": True,
        "generation_ids": sorted(generation_ids),
        "attempt_ids": sorted(str(item) for item in attempts),
        "compact_evidence_root":
            TERMINAL_RETENTION_EVIDENCE_NAME,
        "compact_exports": compact_exports,
        "compact_exports_sha256": hashlib.sha256(
            _canonical_json(compact_exports)
        ).hexdigest(),
        "compact_export_bytes": sum(
            item["byte_count"] for item in compact_exports
        ),
        "lane_authorities": lane_authorities,
        "lane_authorities_sha256": hashlib.sha256(
            _canonical_json(lane_authorities)
        ).hexdigest(),
        "pre_cleanup_audits": dict(pre_cleanup_audits),
        "pre_cleanup_audits_sha256": hashlib.sha256(
            _canonical_json(dict(pre_cleanup_audits))
        ).hexdigest(),
        "retention_policy": {
            "phase": "final_campaign_only",
            "copy_all_compact_exports_before_first_purge": True,
            "generation_scratch_retained": False,
            "workspace_retained": False,
            "cache_retained": False,
            "raw_transcripts_retained": False,
            "stdout_stderr_retained": False,
            "invalid_attempt_raw_bytes_retained": False,
            "promotion_and_replay_authority":
                "external_unified_promotion_audit",
            "wip_needed_midcampaign": True,
        },
    }
    return {
        **body,
        "intent_sha256": hashlib.sha256(
            _canonical_json(body)
        ).hexdigest(),
    }


def _validate_terminal_retention_intent(
    value: object,
    *,
    campaign_root: Path,
    runner_state_receipt: Mapping[str, Any],
    pre_cleanup_audits: Mapping[str, str],
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ContiguousRunnerError(
            "terminal retention intent must be an object"
        )
    body = {
        key: item
        for key, item in value.items()
        if key != "intent_sha256"
    }
    exports = value.get("compact_exports")
    generation_ids = value.get("generation_ids")
    attempt_ids = value.get("attempt_ids")
    expected_fields = {
        "schema",
        "kind",
        "campaign_root",
        "campaign_id",
        "runner_state_receipt",
        "runner_state_receipt_sha256",
        "journal_head_sequence",
        "journal_head_digest",
        "solved_levels",
        "total_levels",
        "complete",
        "generation_ids",
        "attempt_ids",
        "compact_evidence_root",
        "compact_exports",
        "compact_exports_sha256",
        "compact_export_bytes",
        "lane_authorities",
        "lane_authorities_sha256",
        "pre_cleanup_audits",
        "pre_cleanup_audits_sha256",
        "retention_policy",
        "intent_sha256",
    }
    expected_lane_authorities = [
        {
            "game": item.get("game"),
            "target": item.get("target"),
            "reached": item.get("reached"),
            "checkpoint_path": item.get("checkpoint_path"),
            "checkpoint_sha256": item.get("checkpoint_sha256"),
            "source_path": item.get("source_path"),
            "source_tree_sha256": item.get("source_tree_sha256"),
        }
        for item in runner_state_receipt.get("lane_boundaries", ())
        if isinstance(item, Mapping)
    ]
    expected_lane_authorities.sort(
        key=lambda item: str(item["game"])
    )
    if (
        set(value) != expected_fields
        or value.get("schema") != TERMINAL_RETENTION_SCHEMA
        or value.get("kind")
        != "arc_agi3_terminal_attempt_retention_intent"
        or value.get("campaign_root") != str(campaign_root)
        or value.get("campaign_id")
        != runner_state_receipt.get("campaign_id")
        or value.get("runner_state_receipt")
        != dict(runner_state_receipt)
        or value.get("runner_state_receipt_sha256")
        != runner_state_receipt.get("receipt_sha256")
        or value.get("journal_head_sequence")
        != runner_state_receipt.get("journal_head_sequence")
        or value.get("journal_head_digest")
        != runner_state_receipt.get("journal_head_digest")
        or value.get("solved_levels")
        != runner_state_receipt.get("solved_levels")
        or value.get("total_levels")
        != runner_state_receipt.get("total_levels")
        or value.get("complete") is not True
        or value.get("compact_evidence_root")
        != TERMINAL_RETENTION_EVIDENCE_NAME
        or value.get("intent_sha256")
        != hashlib.sha256(_canonical_json(body)).hexdigest()
        or not isinstance(exports, list)
        or value.get("compact_exports_sha256")
        != hashlib.sha256(_canonical_json(exports)).hexdigest()
        or not isinstance(generation_ids, list)
        or generation_ids != sorted(set(generation_ids))
        or generation_ids
        != runner_state_receipt.get("generation_ids")
        or any(not _is_uuid4(item) for item in generation_ids)
        or not isinstance(attempt_ids, list)
        or attempt_ids != sorted(set(attempt_ids))
        or attempt_ids != runner_state_receipt.get("attempt_ids")
        or any(not _safe_identifier(item) for item in attempt_ids)
        or value.get("lane_authorities")
        != expected_lane_authorities
        or value.get("lane_authorities_sha256")
        != hashlib.sha256(
            _canonical_json(expected_lane_authorities)
        ).hexdigest()
        or value.get("pre_cleanup_audits")
        != dict(pre_cleanup_audits)
        or value.get("pre_cleanup_audits_sha256")
        != hashlib.sha256(
            _canonical_json(dict(pre_cleanup_audits))
        ).hexdigest()
    ):
        raise ContiguousRunnerError(
            "terminal retention intent is stale or malformed"
        )
    total = 0
    retained_paths: set[str] = set()
    for item in exports:
        if not isinstance(item, dict):
            raise ContiguousRunnerError(
                "terminal compact export is malformed"
            )
        expected_fields = {
            "attempt_id",
            "generation_id",
            "evidence_sha256",
            "byte_count",
            "references",
            "source_relative_paths",
            "source_campaign_relative_paths",
            "source_absolute_paths",
            "retained_relative_path",
        }
        retained = item.get("retained_relative_path")
        expected_retained = (
            f"{item.get('attempt_id')}/"
            f"{item.get('evidence_sha256')}.json"
        )
        if (
            set(item) != expected_fields
            or item.get("attempt_id") not in attempt_ids
            or item.get("generation_id") not in generation_ids
            or not _is_sha256(item.get("evidence_sha256"))
            or not isinstance(item.get("byte_count"), int)
            or isinstance(item.get("byte_count"), bool)
            or not 0 <= item["byte_count"] <= (
                MAX_TERMINAL_COMPACT_EVIDENCE_BYTES
            )
            or not isinstance(item.get("references"), list)
            or item["references"]
            != sorted(set(item["references"]))
            or not item["references"]
            or any(
                not isinstance(role, str) or not role
                for role in item["references"]
            )
            or not isinstance(
                item.get("source_relative_paths"), list
            )
            or item["source_relative_paths"]
            != sorted(set(item["source_relative_paths"]))
            or any(
                not isinstance(relative, str)
                or PurePosixPath(relative).is_absolute()
                or str(PurePosixPath(relative)) != relative
                or any(
                    part in {"", ".", ".."}
                    for part in PurePosixPath(relative).parts
                )
                for relative in item["source_relative_paths"]
            )
            or not isinstance(
                item.get("source_campaign_relative_paths"), list
            )
            or item["source_campaign_relative_paths"]
            != sorted(set(item["source_campaign_relative_paths"]))
            or any(
                not isinstance(relative, str)
                or PurePosixPath(relative).is_absolute()
                or str(PurePosixPath(relative)) != relative
                or any(
                    part in {"", ".", ".."}
                    for part in PurePosixPath(relative).parts
                )
                for relative
                in item["source_campaign_relative_paths"]
            )
            or not isinstance(item.get("source_absolute_paths"), list)
            or item["source_absolute_paths"]
            != sorted(set(item["source_absolute_paths"]))
            or any(
                not _safe_path_string(absolute)
                or not Path(absolute).is_absolute()
                or any(
                    part in {"", ".", ".."}
                    for part in Path(absolute).parts
                )
                for absolute in item["source_absolute_paths"]
            )
            or not (
                item["source_relative_paths"]
                or item["source_campaign_relative_paths"]
                or item["source_absolute_paths"]
            )
            or retained != expected_retained
            or retained in retained_paths
        ):
            raise ContiguousRunnerError(
                "terminal compact export is stale or malformed"
            )
        retained_paths.add(retained)
        total += item["byte_count"]
    if (
        total != value.get("compact_export_bytes")
        or total > MAX_TERMINAL_COMPACT_EVIDENCE_TOTAL_BYTES
    ):
        raise ContiguousRunnerError(
            "terminal compact export byte accounting is invalid"
        )
    expected_policy = {
        "phase": "final_campaign_only",
        "copy_all_compact_exports_before_first_purge": True,
        "generation_scratch_retained": False,
        "workspace_retained": False,
        "cache_retained": False,
        "raw_transcripts_retained": False,
        "stdout_stderr_retained": False,
        "invalid_attempt_raw_bytes_retained": False,
        "promotion_and_replay_authority":
            "external_unified_promotion_audit",
        "wip_needed_midcampaign": True,
    }
    policy = value.get("retention_policy")
    boolean_policy = {
        key: expected
        for key, expected in expected_policy.items()
        if type(expected) is bool
    }
    string_policy = {
        key: expected
        for key, expected in expected_policy.items()
        if type(expected) is str
    }
    if (
        type(policy) is not dict
        or set(policy) != set(expected_policy)
        or any(type(key) is not str for key in policy)
        or any(
            type(policy[key]) is not bool
            or policy[key] is not expected
            for key, expected in boolean_policy.items()
        )
        or any(
            type(policy[key]) is not str
            or policy[key] != expected
            for key, expected in string_policy.items()
        )
    ):
        raise ContiguousRunnerError(
            "terminal retention policy is malformed"
        )
    return value


def _terminal_retention_recovery_runner_receipt(
    campaign_root: Path,
    *,
    expected_receipt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Reopen the pre-purge runner audit after generations start vanishing.

    A normal runner audit deliberately reopens every generation receipt.  That
    is the right rule until the durable retention intent and complete compact
    archive exist, but it made a real SIGKILL during generation deletion
    impossible to resume: the next audit failed on the first already-purged
    receipt.  The intent is therefore the recovery boundary.  It embeds the
    exact previously verified runner receipt and binds it to the still
    independently replayable immutable journal head.  No missing generation is
    treated as new evidence.
    """

    root = Path(campaign_root)
    intent_path = _terminal_retention_intent_path(root)
    if (
        intent_path.is_symlink()
        or not intent_path.is_file()
        or stat.S_IMODE(
            intent_path.stat(follow_symlinks=False).st_mode
        )
        & 0o222
    ):
        raise ContiguousRunnerError(
            "terminal retention recovery lacks a sealed intent"
        )
    value = _read_json_file(intent_path)
    embedded = value.get("runner_state_receipt")
    if not isinstance(embedded, dict):
        raise ContiguousRunnerError(
            "terminal retention intent lacks its runner receipt"
        )
    runner_receipt = _terminal_retention_runner_body(embedded)
    prerequisites = _terminal_retention_prerequisites(
        value.get("pre_cleanup_audits")
    )
    _validate_terminal_retention_intent(
        value,
        campaign_root=root,
        runner_state_receipt=runner_receipt,
        pre_cleanup_audits=prerequisites,
    )
    if _sha256_file(intent_path) != hashlib.sha256(
        _canonical_json(value) + b"\n"
    ).hexdigest():
        raise ContiguousRunnerError(
            "terminal retention intent encoding changed"
        )
    if (
        expected_receipt is not None
        and dict(expected_receipt) != runner_receipt
    ):
        raise ContiguousRunnerError(
            "terminal retention recovery received another runner receipt"
        )

    required_runner_fields = {
        "schema",
        "kind",
        "status",
        "campaign_root",
        "attempt_journal_path",
        "campaign_id",
        "inventory_sha256",
        "scheduler_policy_sha256",
        "operator_configuration_sha256",
        "journal_event_count",
        "journal_head_sequence",
        "journal_head_digest",
        "journal_prefix",
        "state_sha256",
        "solved_levels",
        "total_levels",
        "complete",
        "attempt_ids",
        "generation_ids",
        "lane_boundaries",
        "receipt_sha256",
    }
    inventory = Contract.authoritative_inventory()
    lane_boundaries = runner_receipt.get("lane_boundaries")
    if (
        set(runner_receipt) != required_runner_fields
        or runner_receipt.get("schema") != 1
        or runner_receipt.get("kind")
        != "arc_agi3_contiguous_runner_state_audit"
        or runner_receipt.get("status") != "PASS"
        or runner_receipt.get("campaign_root") != str(root)
        or runner_receipt.get("attempt_journal_path")
        != str(root / "attempt_journal")
        or runner_receipt.get("inventory_sha256")
        != Contract.authoritative_inventory_sha256(inventory)
        or runner_receipt.get("scheduler_policy_sha256")
        != SCHEDULER_POLICY_SHA256
        or runner_receipt.get("complete") is not True
        or runner_receipt.get("solved_levels") != sum(inventory.values())
        or runner_receipt.get("total_levels") != sum(inventory.values())
        or not isinstance(lane_boundaries, list)
        or {
            item.get("game"): item.get("reached")
            for item in lane_boundaries
            if isinstance(item, Mapping)
        }
        != inventory
        or any(
            not isinstance(item, Mapping)
            or item.get("target") != inventory.get(item.get("game"))
            for item in lane_boundaries
        )
    ):
        raise ContiguousRunnerError(
            "terminal retention runner receipt is not exact complete authority"
        )

    journal = ReadOnlyAttemptJournal(root / "attempt_journal")
    events = journal.read()
    scheduler_digest = prerequisites.get("scheduler")
    scheduler_path = root / "terminal_audits" / "scheduler.json"
    try:
        if not _is_sha256(scheduler_digest):
            raise Scheduler.SchedulerError(
                "retention intent lacks its scheduler audit binding"
            )
        scheduler_receipt = (
            Scheduler.verify_pre_retention_audit_receipt(
                root,
                scheduler_path,
                expected_receipt_sha256=scheduler_digest,
            )
        )
        summary = scheduler_receipt["summary"]
        prefix = Scheduler.journal_prefix_status(root)
    except (KeyError, Scheduler.SchedulerError) as exc:
        raise ContiguousRunnerError(
            "terminal retention journal recovery failed"
        ) from exc
    if (
        not events
        or len(events) != runner_receipt["journal_event_count"]
        or events[-1]["sequence"]
        != runner_receipt["journal_head_sequence"]
        or events[-1]["digest"]
        != runner_receipt["journal_head_digest"]
        or prefix != runner_receipt["journal_prefix"]
        or summary.get("policy_promoted_levels")
        != runner_receipt["solved_levels"]
        or summary.get("total_levels")
        != runner_receipt["total_levels"]
        or summary.get("live_reservation_units") != 0
        or summary.get("pending_decision") is not None
    ):
        raise ContiguousRunnerError(
            "terminal retention journal differs from its runner receipt"
        )

    allowed_generations = set(runner_receipt["generation_ids"])
    generations = root / "generations"
    if generations.is_symlink() or not generations.is_dir():
        raise ContiguousRunnerError(
            "terminal retention generations root is unavailable"
        )
    for entry in generations.iterdir():
        if (
            entry.name not in allowed_generations
            or entry.is_symlink()
            or not entry.is_dir()
        ):
            raise ContiguousRunnerError(
                "terminal retention recovery found an unknown generation"
            )
    return runner_receipt


def _ensure_terminal_retention_directory(path: Path) -> None:
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_dir():
            raise ContiguousRunnerError(
                f"terminal retention directory is aliased: {path}"
            )
        return
    os.mkdir(path, 0o700)
    _fsync_directory(path.parent)


def _stage_terminal_retention_exports(
    campaign_root: Path,
    intent: Mapping[str, Any],
) -> None:
    evidence_root = _terminal_retention_root(campaign_root)
    _ensure_terminal_retention_directory(evidence_root)
    generations = campaign_root / "generations"
    for item in intent["compact_exports"]:
        destination = evidence_root / item["retained_relative_path"]
        if destination.exists() or destination.is_symlink():
            payload, byte_count = _terminal_retention_regular_json(
                destination,
                expected_sha256=item["evidence_sha256"],
            )
            del payload
            if byte_count != item["byte_count"]:
                raise ContiguousRunnerError(
                    "retained compact evidence byte count changed"
                )
            continue
        parent = destination.parent
        _ensure_terminal_retention_directory(parent)
        generation = _terminal_retention_generation_path(
            generations, item["generation_id"]
        )
        source_candidates: list[Path] = []
        if item["source_relative_paths"]:
            if generation.is_symlink() or not generation.is_dir():
                raise ContiguousRunnerError(
                    "compact evidence source generation is unavailable"
                )
            source_candidates.extend(
                generation / PurePosixPath(relative)
                for relative in item["source_relative_paths"]
            )
        source_candidates.extend(
            campaign_root / PurePosixPath(relative)
            for relative in item["source_campaign_relative_paths"]
        )
        source_candidates.extend(
            Path(absolute)
            for absolute in item["source_absolute_paths"]
        )
        payload: bytes | None = None
        for source in source_candidates:
            if source.exists() and not source.is_symlink():
                try:
                    if source.resolve(strict=True) != source:
                        raise ContiguousRunnerError(
                            "compact evidence source is aliased"
                        )
                    _regular_file_pointer(source)
                except OSError as exc:
                    raise ContiguousRunnerError(
                        "compact evidence source cannot be reopened"
                    ) from exc
                candidate, byte_count = _terminal_retention_regular_json(
                    source,
                    expected_sha256=item["evidence_sha256"],
                )
                if byte_count != item["byte_count"]:
                    raise ContiguousRunnerError(
                        "compact evidence source byte count changed"
                    )
                payload = candidate
                break
        if payload is None:
            raise ContiguousRunnerError(
                "compact evidence is missing before generation purge"
            )
        _install_regular_bytes(destination, payload)
        if _sha256_file(destination) != item["evidence_sha256"]:
            raise ContiguousRunnerError(
                "compact evidence installation changed bytes"
            )
    expected_files = {
        item["retained_relative_path"]
        for item in intent["compact_exports"]
    }
    actual_files = {
        path.relative_to(evidence_root).as_posix()
        for path in evidence_root.rglob("*")
        if path.is_file()
    }
    if actual_files != expected_files:
        raise ContiguousRunnerError(
            "terminal compact evidence archive has extra/missing files"
        )
    _seal_regular_tree(evidence_root)


def _make_terminal_generation_removable(generation: Path) -> None:
    try:
        metadata = generation.stat(follow_symlinks=False)
    except OSError as exc:
        raise ContiguousRunnerError(
            "terminal purge generation root cannot be inspected"
        ) from exc
    if (
        generation.is_symlink()
        or not stat.S_ISDIR(metadata.st_mode)
    ):
        raise ContiguousRunnerError(
            "terminal purge generation root is aliased or irregular"
        )
    try:
        walker = os.fwalk(
            generation,
            topdown=False,
            follow_symlinks=False,
        )
        for _path, directory_names, _file_names, directory_fd in walker:
            for name in directory_names:
                metadata = os.stat(
                    name,
                    dir_fd=directory_fd,
                    follow_symlinks=False,
                )
                if stat.S_ISLNK(metadata.st_mode):
                    continue
                if not stat.S_ISDIR(metadata.st_mode):
                    raise ContiguousRunnerError(
                        "terminal purge directory changed during thaw"
                    )
            metadata = os.fstat(directory_fd)
            if not stat.S_ISDIR(metadata.st_mode):
                raise ContiguousRunnerError(
                    "terminal purge opened a non-directory"
                )
            os.fchmod(directory_fd, 0o700)
            os.fsync(directory_fd)
    except OSError as exc:
        raise ContiguousRunnerError(
            "terminal purge could not thaw exact generation directories"
        ) from exc


def _terminal_retention_archive_inventory(
    campaign_root: Path,
    intent: Mapping[str, Any],
) -> list[dict[str, Any]]:
    evidence_root = _terminal_retention_root(campaign_root)
    try:
        Contract._validate_regular_tree(
            evidence_root, label="terminal compact evidence"
        )
    except Exception as exc:
        raise ContiguousRunnerError(
            "terminal compact evidence tree is irregular"
        ) from exc
    expected_files = {
        item["retained_relative_path"]: item
        for item in intent["compact_exports"]
    }
    actual_entries = list(evidence_root.rglob("*"))
    actual_files = {
        path.relative_to(evidence_root).as_posix(): path
        for path in actual_entries
        if path.is_file()
    }
    actual_directories = {
        path.relative_to(evidence_root).as_posix()
        for path in actual_entries
        if path.is_dir()
    }
    expected_directories = {
        str(item["attempt_id"])
        for item in intent["compact_exports"]
    }
    if (
        set(actual_files) != set(expected_files)
        or actual_directories != expected_directories
    ):
        raise ContiguousRunnerError(
            "terminal compact evidence tree has extra/missing entries"
        )
    root_mode = stat.S_IMODE(
        evidence_root.stat(follow_symlinks=False).st_mode
    )
    if root_mode & 0o222:
        raise ContiguousRunnerError(
            "terminal compact evidence root remains writable"
        )
    inventory: list[dict[str, Any]] = []
    for relative, item in sorted(expected_files.items()):
        path = actual_files[relative]
        metadata = path.stat(follow_symlinks=False)
        if stat.S_IMODE(metadata.st_mode) & 0o222:
            raise ContiguousRunnerError(
                "terminal compact evidence file remains writable"
            )
        payload, byte_count = _terminal_retention_regular_json(
            path, expected_sha256=item["evidence_sha256"]
        )
        del payload
        if byte_count != item["byte_count"]:
            raise ContiguousRunnerError(
                "terminal compact evidence byte count changed"
            )
        inventory.append(
            {
                "path": relative,
                "sha256": item["evidence_sha256"],
                "bytes": byte_count,
            }
        )
    for relative in sorted(expected_directories):
        directory = evidence_root / relative
        if stat.S_IMODE(
            directory.stat(follow_symlinks=False).st_mode
        ) & 0o222:
            raise ContiguousRunnerError(
                "terminal compact evidence directory remains writable"
            )
    forbidden_names = {
        "scratch",
        "workspace",
        "cache",
        "transcript",
        "stdout",
        "stderr",
        "jsonl",
        "log",
    }
    for item in inventory:
        filename_tokens = {
            token
            for token in re.split(r"[^a-z0-9]+", Path(item["path"]).name.lower())
            if token
        }
        if filename_tokens & forbidden_names:
            raise ContiguousRunnerError(
                "terminal archive retained a forbidden raw artifact"
            )
    return inventory


def _terminal_retention_receipt_value(
    campaign_root: Path,
    *,
    runner_state_receipt: Mapping[str, Any],
    intent: Mapping[str, Any],
) -> dict[str, Any]:
    generations = campaign_root / "generations"
    remaining = sorted(path.name for path in generations.iterdir())
    if remaining:
        raise ContiguousRunnerError(
            "terminal generation purge is incomplete"
        )
    inventory = _terminal_retention_archive_inventory(
        campaign_root, intent
    )
    body = {
        "schema": TERMINAL_RETENTION_SCHEMA,
        "kind": "arc_agi3_terminal_attempt_retention",
        "status": "PASS",
        "campaign_root": str(campaign_root),
        "campaign_id": runner_state_receipt["campaign_id"],
        "runner_state_receipt_sha256":
            runner_state_receipt["receipt_sha256"],
        "journal_head_sequence":
            runner_state_receipt["journal_head_sequence"],
        "journal_head_digest":
            runner_state_receipt["journal_head_digest"],
        "intent_sha256": intent["intent_sha256"],
        "intent_file_sha256": _sha256_file(
            _terminal_retention_intent_path(campaign_root)
        ),
        "compact_evidence_root":
            TERMINAL_RETENTION_EVIDENCE_NAME,
        "compact_evidence_tree_sha256": Contract._tree_hash(
            _terminal_retention_root(campaign_root)
        ),
        "compact_evidence_inventory": inventory,
        "compact_evidence_inventory_sha256": hashlib.sha256(
            _canonical_json(inventory)
        ).hexdigest(),
        "compact_export_count": len(inventory),
        "compact_export_bytes": sum(
            item["bytes"] for item in inventory
        ),
        "removed_generation_ids": intent["generation_ids"],
        "remaining_generation_entries": [],
        "generation_scratch_survivors": 0,
        "workspace_survivors": 0,
        "cache_survivors": 0,
        "raw_stream_survivors": 0,
        "promotion_and_replay_authority":
            "external_unified_promotion_audit",
        "lane_authorities_sha256":
            intent["lane_authorities_sha256"],
        "pre_cleanup_audits_sha256":
            intent["pre_cleanup_audits_sha256"],
    }
    return {
        **body,
        "receipt_sha256": hashlib.sha256(
            _canonical_json(body)
        ).hexdigest(),
    }


def finalize_terminal_attempt_retention(
    campaign_root: Path,
    runner_state_receipt: object,
    *,
    secret_sentinels: tuple[str, ...] = (),
    controller_state_canaries: tuple[Taint.LiveCanary, ...] = (),
    pre_cleanup_audits: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Crash-recoverably retain compact evidence and purge attempt workspaces.

    This operation is intentionally forbidden before exact 183/183 completion.
    Every retained receipt is copied and sealed before the first generation is
    removed.  The immutable journal-head-bound intent makes a partial purge
    idempotently resumable without treating missing scratch as new evidence.
    """

    requested = Path(campaign_root)
    runner_receipt = _terminal_retention_runner_body(
        runner_state_receipt
    )
    prerequisites = _terminal_retention_prerequisites(
        pre_cleanup_audits
    )
    intent_path = _terminal_retention_intent_path(requested)
    if intent_path.exists() or intent_path.is_symlink():
        verified = _terminal_retention_recovery_runner_receipt(
            requested,
            expected_receipt=runner_receipt,
        )
    else:
        verified = verify_runner_state_audit(
            runner_receipt,
            campaign_root=requested,
            secret_sentinels=secret_sentinels,
            controller_state_canaries=controller_state_canaries,
        )
    if (
        verified.get("complete") is not True
        or verified.get("solved_levels")
        != verified.get("total_levels")
    ):
        raise ContiguousRunnerError(
            "terminal retention is forbidden before complete coverage"
        )
    root = Path(verified["campaign_root"])
    generations = root / "generations"
    intent_path = _terminal_retention_intent_path(root)
    receipt_path = _terminal_retention_receipt_path(root)
    if receipt_path.exists() or receipt_path.is_symlink():
        if (
            receipt_path.is_symlink()
            or not receipt_path.is_file()
            or intent_path.is_symlink()
            or not intent_path.is_file()
        ):
            raise ContiguousRunnerError(
                "terminal retention receipt recovery found an alias"
            )
        retained_intent = _validate_terminal_retention_intent(
            _read_json_file(intent_path),
            campaign_root=root,
            runner_state_receipt=verified,
            pre_cleanup_audits=prerequisites,
        )
        expected_receipt = _terminal_retention_receipt_value(
            root,
            runner_state_receipt=verified,
            intent=retained_intent,
        )
        if _read_json_file(receipt_path) != expected_receipt:
            raise ContiguousRunnerError(
                "terminal retention receipt recovery found wrong bytes"
            )
        # These mode changes close only crash windows after exact durable
        # bytes were installed; malformed bytes are never normalized.
        os.chmod(intent_path, 0o400, follow_symlinks=False)
        os.chmod(receipt_path, 0o400, follow_symlinks=False)
        _fsync_directory(root)
        return audit_terminal_attempt_retention(
            root,
            verified,
            secret_sentinels=secret_sentinels,
            controller_state_canaries=controller_state_canaries,
            pre_cleanup_audits=prerequisites,
        )

    if intent_path.exists() or intent_path.is_symlink():
        intent = _validate_terminal_retention_intent(
            _read_json_file(intent_path),
            campaign_root=root,
            runner_state_receipt=verified,
            pre_cleanup_audits=prerequisites,
        )
        if _sha256_file(intent_path) != hashlib.sha256(
            _canonical_json(intent) + b"\n"
        ).hexdigest():
            raise ContiguousRunnerError(
                "terminal retention intent file changed"
            )
        os.chmod(intent_path, 0o400, follow_symlinks=False)
        _fsync_directory(root)
    else:
        evidence_root = _terminal_retention_root(root)
        if evidence_root.exists() or evidence_root.is_symlink():
            raise ContiguousRunnerError(
                "terminal evidence exists without a durable intent"
            )
        state = _terminal_retention_state(
            root,
            secret_sentinels=secret_sentinels,
            controller_state_canaries=controller_state_canaries,
        )
        intent = _terminal_retention_plan(
            root,
            state=state,
            runner_state_receipt=verified,
            pre_cleanup_audits=prerequisites,
        )
        actual_generations = sorted(
            path.name for path in generations.iterdir()
        )
        if actual_generations != intent["generation_ids"]:
            raise ContiguousRunnerError(
                "generation disappeared before retention intent"
            )
        _write_new_file(intent_path, intent)
        os.chmod(intent_path, 0o400, follow_symlinks=False)
        _fsync_directory(root)

    _stage_terminal_retention_exports(root, intent)
    if not getattr(shutil.rmtree, "avoids_symlink_attacks", False):
        raise ContiguousRunnerError(
            "platform lacks descriptor-safe recursive deletion"
        )
    allowed_generations = set(intent["generation_ids"])
    for entry in generations.iterdir():
        if (
            entry.name not in allowed_generations
            or entry.is_symlink()
            or not entry.is_dir()
        ):
            raise ContiguousRunnerError(
                "terminal purge encountered an unexpected generation"
            )
    for generation_id in intent["generation_ids"]:
        generation = _terminal_retention_generation_path(
            generations, generation_id
        )
        if not generation.exists() and not generation.is_symlink():
            continue
        if generation.is_symlink() or not generation.is_dir():
            raise ContiguousRunnerError(
                "terminal purge generation was substituted"
            )
        _make_terminal_generation_removable(generation)
        shutil.rmtree(generation)
        _fsync_directory(generations)
    if any(generations.iterdir()):
        raise ContiguousRunnerError(
            "terminal purge left generation entries"
        )
    receipt = _terminal_retention_receipt_value(
        root,
        runner_state_receipt=verified,
        intent=intent,
    )
    _write_new_file(receipt_path, receipt)
    os.chmod(receipt_path, 0o400, follow_symlinks=False)
    _fsync_directory(root)
    return audit_terminal_attempt_retention(
        root,
        verified,
        secret_sentinels=secret_sentinels,
        controller_state_canaries=controller_state_canaries,
        pre_cleanup_audits=prerequisites,
    )


def audit_terminal_attempt_retention(
    campaign_root: Path,
    runner_state_receipt: object,
    *,
    secret_sentinels: tuple[str, ...] = (),
    controller_state_canaries: tuple[Taint.LiveCanary, ...] = (),
    pre_cleanup_audits: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Fail closed on missing compact evidence or any terminal workspace."""

    requested = Path(campaign_root)
    runner_receipt = _terminal_retention_runner_body(
        runner_state_receipt
    )
    prerequisites = _terminal_retention_prerequisites(
        pre_cleanup_audits
    )
    intent_path = _terminal_retention_intent_path(requested)
    if intent_path.exists() or intent_path.is_symlink():
        verified = _terminal_retention_recovery_runner_receipt(
            requested,
            expected_receipt=runner_receipt,
        )
    else:
        verified = verify_runner_state_audit(
            runner_receipt,
            campaign_root=requested,
            secret_sentinels=secret_sentinels,
            controller_state_canaries=controller_state_canaries,
        )
    root = Path(verified["campaign_root"])
    intent_path = _terminal_retention_intent_path(root)
    receipt_path = _terminal_retention_receipt_path(root)
    evidence_root = _terminal_retention_root(root)
    if verified.get("complete") is not True:
        if any(
            path.exists() or path.is_symlink()
            for path in (intent_path, receipt_path, evidence_root)
        ):
            raise ContiguousRunnerError(
                "terminal retention mutated an incomplete campaign"
            )
        body = {
            "schema": TERMINAL_RETENTION_SCHEMA,
            "kind": "arc_agi3_terminal_attempt_retention",
            "status": "NOT_REQUIRED",
            "campaign_root": str(root),
            "campaign_id": verified["campaign_id"],
            "runner_state_receipt_sha256":
                verified["receipt_sha256"],
            "journal_head_sequence":
                verified["journal_head_sequence"],
            "journal_head_digest":
                verified["journal_head_digest"],
            "reason": "campaign_incomplete",
        }
        return {
            **body,
            "receipt_sha256": hashlib.sha256(
                _canonical_json(body)
            ).hexdigest(),
        }
    if (
        intent_path.is_symlink()
        or receipt_path.is_symlink()
        or evidence_root.is_symlink()
        or not intent_path.is_file()
        or not receipt_path.is_file()
        or not evidence_root.is_dir()
    ):
        raise ContiguousRunnerError(
            "complete campaign lacks terminal retention evidence"
        )
    intent = _validate_terminal_retention_intent(
        _read_json_file(intent_path),
        campaign_root=root,
        runner_state_receipt=verified,
        pre_cleanup_audits=prerequisites,
    )
    actual_receipt = _read_json_file(receipt_path)
    expected_receipt = _terminal_retention_receipt_value(
        root,
        runner_state_receipt=verified,
        intent=intent,
    )
    if actual_receipt != expected_receipt:
        raise ContiguousRunnerError(
            "terminal retention receipt is stale or forged"
        )
    if (
        stat.S_IMODE(
            intent_path.stat(follow_symlinks=False).st_mode
        ) & 0o222
        or stat.S_IMODE(
            receipt_path.stat(follow_symlinks=False).st_mode
        ) & 0o222
    ):
        raise ContiguousRunnerError(
            "terminal retention control receipt remains writable"
        )
    return expected_receipt


def _audit_json_value(value: Any) -> Any:
    if is_dataclass(value):
        return _audit_json_value(asdict(value))
    if isinstance(value, Mapping):
        return {
            str(key): _audit_json_value(item)
            for key, item in sorted(
                value.items(), key=lambda pair: str(pair[0])
            )
        }
    if isinstance(value, (tuple, list)):
        return [_audit_json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise ContiguousRunnerError(
        "runner state contains a noncanonical audit value"
    )


def audit_runner_state_read_only(
    campaign_root: Path,
    *,
    secret_sentinels: tuple[str, ...] = (),
    controller_state_canaries: tuple[Taint.LiveCanary, ...] = (),
) -> dict[str, Any]:
    """Replay and fully validate an existing campaign without any mutation.

    This deliberately bypasses :meth:`ContiguousCampaignRunner.__init__` and
    :class:`DurableAttemptJournal` construction.  It creates no directories,
    lock files, journal events, receipts, or permission changes.
    """

    requested = Path(campaign_root)
    if (
        not requested.is_absolute()
        or requested.is_symlink()
        or not requested.exists()
        or not requested.is_dir()
    ):
        raise ContiguousRunnerError(
            "read-only audit requires an absolute regular campaign root"
        )
    try:
        root = requested.resolve(strict=True)
    except OSError as exc:
        raise ContiguousRunnerError(
            "read-only campaign root cannot be resolved"
        ) from exc
    if root != requested:
        raise ContiguousRunnerError(
            "read-only campaign root is aliased"
        )
    for child_name in (
        "attempt_journal",
        "auxiliary",
        "generations",
        "public_observation_registry",
        "zero_checkpoints",
        "zero_sources",
    ):
        child = root / child_name
        if child.is_symlink() or not child.is_dir():
            raise ContiguousRunnerError(
                f"read-only campaign child is unavailable: {child_name}"
            )
    if (
        not isinstance(secret_sentinels, tuple)
        or any(
            not isinstance(item, str)
            or not item
            or item == "REDACTED"
            for item in secret_sentinels
        )
        or len(secret_sentinels) != len(set(secret_sentinels))
    ):
        raise ContiguousRunnerError(
            "read-only audit secret sentinels are malformed"
        )
    try:
        canaries = (
            Taint.validate_live_canaries(
                controller_state_canaries,
                require_complete=True,
            )
            if controller_state_canaries
            else ()
        )
    except Exception as exc:
        raise ContiguousRunnerError(
            "read-only audit canaries are malformed"
        ) from exc

    retention_intent = _terminal_retention_intent_path(root)
    if retention_intent.exists() or retention_intent.is_symlink():
        return _terminal_retention_recovery_runner_receipt(root)

    reducer = object.__new__(ContiguousCampaignRunner)
    reducer.root = root
    reducer.journal = ReadOnlyAttemptJournal(
        root / "attempt_journal"
    )
    reducer.generations = root / "generations"
    reducer.auxiliary = root / "auxiliary"
    reducer.public_observation_registry = (
        root / "public_observation_registry"
    )
    reducer._secret_sentinels = secret_sentinels
    reducer._controller_state_canaries = canaries
    reducer_events = reducer.journal.read()
    if not reducer_events:
        raise ContiguousRunnerError(
            "read-only runner audit found no journal history"
        )
    try:
        reducer.auxiliary_launch_configuration = (
            Scheduler.auxiliary_launch_configuration_from_dict(
                reducer_events[0]["payload"].get(
                    "auxiliary_launch_configuration"
                )
            )
        )
    except Scheduler.SchedulerError as exc:
        raise ContiguousRunnerError(
            "read-only audit found invalid auxiliary configuration"
        ) from exc
    state = ContiguousCampaignRunner.state(reducer)
    events = reducer_events
    normalized_state = _audit_json_value(state)
    lane_boundaries = [
        {
            "game": game,
            "target": lane["target"],
            "reached": lane["reached"],
            "checkpoint_path": lane["checkpoint_path"],
            "checkpoint_sha256": lane["checkpoint_sha256"],
            "source_path": lane["source_path"],
            "source_tree_sha256": lane["source_tree_sha256"],
        }
        for game, lane in sorted(state["lanes"].items())
    ]
    attempt_ids = sorted(state["attempts"])
    generation_ids = sorted(
        attempt["reservation"].generation_id
        for attempt in state["attempts"].values()
    )
    body = {
        "schema": 1,
        "kind": "arc_agi3_contiguous_runner_state_audit",
        "status": "PASS",
        "campaign_root": str(root),
        "attempt_journal_path": str(root / "attempt_journal"),
        "campaign_id": state["campaign_id"],
        "inventory_sha256":
            Contract.authoritative_inventory_sha256(
                state["inventory"]
            ),
        "scheduler_policy_sha256":
            state["scheduler_policy_sha256"],
        "operator_configuration_sha256":
            state["operator_configuration_sha256"],
        "journal_event_count": len(events),
        "journal_head_sequence": events[-1]["sequence"],
        "journal_head_digest": events[-1]["digest"],
        "journal_prefix": state["journal_prefix"],
        "public_observation_registry_sha256":
            state["public_observation_registry_sha256"],
        "state_sha256": hashlib.sha256(
            _canonical_json(normalized_state)
        ).hexdigest(),
        "solved_levels": state["solved_levels"],
        "total_levels": state["total_levels"],
        "complete": state["complete"],
        "attempt_ids": attempt_ids,
        "generation_ids": generation_ids,
        "lane_boundaries": lane_boundaries,
    }
    return {
        **body,
        "receipt_sha256": hashlib.sha256(
            _canonical_json(body)
        ).hexdigest(),
    }


def verify_runner_state_audit(
    receipt: object,
    *,
    campaign_root: Path,
    secret_sentinels: tuple[str, ...] = (),
    controller_state_canaries: tuple[Taint.LiveCanary, ...] = (),
) -> dict[str, Any]:
    if not isinstance(receipt, dict):
        raise ContiguousRunnerError(
            "runner-state audit receipt must be an object"
        )
    expected = audit_runner_state_read_only(
        campaign_root,
        secret_sentinels=secret_sentinels,
        controller_state_canaries=controller_state_canaries,
    )
    if receipt != expected:
        raise ContiguousRunnerError(
            "runner-state audit receipt is stale or forged"
        )
    return expected


__all__ = [
    "AttemptBackend",
    "AttemptLayout",
    "AttemptResult",
    "AttemptSpec",
    "BackendCollection",
    "BackendConfiguration",
    "BackendLaunch",
    "BackendPoll",
    "BackendPreparation",
    "BackendTeardownProof",
    "AuxiliaryAbort",
    "AuxiliaryAdmission",
    "AuxiliaryBackend",
    "AuxiliaryBackendFatalError",
    "AuxiliaryCollection",
    "AuxiliaryLaunch",
    "AuxiliaryPoll",
    "AuxiliaryPreparedInput",
    "AuxiliaryTeardown",
    "ContiguousCampaignRunner",
    "CONTIGUOUS_AUXILIARY_LAUNCH_READY",
    "CONTIGUOUS_RUNNER_LAUNCH_READY",
    "ContiguousRunnerError",
    "DurableAttemptJournal",
    "InputBundleBuilder",
    "InputBundleReceipt",
    "PromotionCandidate",
    "PromotionCommit",
    "PromotionGate",
    "ReadOnlyAttemptJournal",
    "ResourceLimitsProjection",
    "SimulatedCrash",
    "TERMINAL_RETENTION_EVIDENCE_NAME",
    "TERMINAL_RETENTION_INTENT_NAME",
    "TERMINAL_RETENTION_RECEIPT_NAME",
    "TERMINAL_RETENTION_SCHEMA",
    "WipSnapshot",
    "audit_terminal_attempt_retention",
    "audit_runner_state_read_only",
    "advance_exact_frontier_clean_no_progress",
    "apply_terminal_result_precedence",
    "escalation",
    "finalize_terminal_attempt_retention",
    "frontier_sha256",
    "should_restore_wip",
    "verify_runner_state_audit",
]
