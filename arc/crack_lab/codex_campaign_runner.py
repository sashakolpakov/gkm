#!/usr/bin/env python3
"""Execute the guarded exact-frontier ARC-AGI-3 compatibility campaign.

Dry-run is the default.  Every dispatch is rederived from the live authoritative
checkpoint and journal-reconstructed retry coordinate, then checked against the
single medium→high→xhigh→max policy before process launch.  Explicit provider
``unlimited`` disables cost cutoffs but never correctness, isolation, taint,
replay, provenance, or containment controls.
"""

from __future__ import annotations

import argparse
import ast
import base64
import binascii
import ctypes
import errno
import fcntl
import glob
import hashlib
import json
import os
import re
import shlex
import shutil
import signal
import stat
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import arc_agi3_proposer_boundary as Boundary
import codex_campaign_policy as Policy
import codex_campaign_reboot_recovery as RebootRecovery
import codex_campaign_status as Status
import codex_usage_guard as Guard
import arc_agi3_contiguous_supervisor as Contiguous
import gkm_legs as Legs


HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
DEFAULT_PLAN = HERE / "ARC_AGI3_CAMPAIGN_QUEUE.json"
SHA256_RE = re.compile(r"[0-9a-f]{64}")
SAFE_COMPONENT_RE = re.compile(r"[A-Za-z0-9_.-]+")
GIT_COMMIT_RE = re.compile(r"[0-9a-f]{40}")
RUNNER_RECEIPT_SCHEMA = 1
DISPATCH_QUARANTINE_SCHEMA = "scheduler_dispatch_quarantine_v1"
WIP_ROLLBACK_CAPSULE_SCHEMA_V1 = "scheduler_wip_rollback_capsule_v1"
WIP_ROLLBACK_CAPSULE_SCHEMA = "scheduler_wip_rollback_capsule_v2"
WIP_ROLLBACK_CAPSULE_SCHEMAS = frozenset({
    WIP_ROLLBACK_CAPSULE_SCHEMA_V1,
    WIP_ROLLBACK_CAPSULE_SCHEMA,
})
WIP_LOGICAL_RESTORE_SCHEMA_V1 = (
    RebootRecovery.WIP_LOGICAL_RESTORE_SCHEMA_V1
)
WIP_LOGICAL_RESTORE_SCHEMA = RebootRecovery.WIP_LOGICAL_RESTORE_SCHEMA
DISPATCH_RELEASE_INTENT_SCHEMA = "scheduler_dispatch_release_intent_v1"
MAX_DISPATCH_RELEASE_INTENT_BYTES = 64 * 1024
SAFE_RELEASE_RECOVERY_ARM_SCHEMA = (
    "scheduler_safe_release_recovery_arm_v1"
)
SAFE_RELEASE_RECOVERY_RECEIPT_SCHEMA = (
    "scheduler_safe_release_recovery_receipt_v1"
)
SAFE_RELEASE_RECOVERY_ARM_EVENT = "dispatch_safe_release_recovery_armed"
SAFE_RELEASE_RECOVERY_RECEIPT_EVENT = (
    "dispatch_safe_release_recovery_completed"
)
MAX_WIP_ROLLBACK_CAPSULE_BYTES = 24 * 1024 * 1024 * 1024
RECOVERY_PHASE_INTENT_SCHEMA = "scheduler_recovery_phase_intent_v1"
RECOVERY_PHASE_EVENTS = {
    "codex_exec_classification_correction": "correction",
    "codex_taint_cleanup_completed": "cleanup",
    "codex_post_reboot_operator_recovery_completed": "operator",
    "codex_infrastructure_generation_quarantined": "zero_ledger",
    "codex_sandbox_isolated_generation_abandoned": "sandbox_abandon",
}
ZERO_LEDGER_EVENT = "codex_infrastructure_generation_quarantined"
ZERO_LEDGER_EVENT_SCHEMA = "scheduler_zero_ledger_generation_quarantine_v1"
SANDBOX_ABANDON_EVENT = "codex_sandbox_isolated_generation_abandoned"
SANDBOX_ABANDON_EVENT_SCHEMA = (
    "scheduler_sandbox_isolated_generation_abandoned_v1"
)
SANDBOX_RELEASE_AUTHORITY_KIND = "sandbox_isolated_operator_terminal_v1"
SANDBOX_ABSENCE_SAMPLES = 3
SANDBOX_ABSENCE_INTERVAL_SECONDS = 0.5
SANDBOX_CONTRACTS = {
    "bb3474290d3411f980d53ffcee75be8234e634d478b1136677b9c6a93fe9ec64":
        hashlib.sha256(
            b"historical-gkm-codex-exec-ephemeral-strict-config-"
            b"sandbox-workspace-write-cd-exact-workspace/v1"
        ).hexdigest(),
}
SANDBOX_ABANDON_EVENT_KEYS = frozenset({
    "event", "schema", "recorded_at", "isolation_authority",
    "operator_provenance_assumption", "sandbox_contract_sha256",
    "dispatch_id", "recovery_nonce", "game", "target_level", "reached",
    "parent_action_count", "retry_complexity_n", "scratch_root",
    "scratch_root_identity", "scratch_root_disposition",
    "required_retry_scratch_relation", "workspace", "workspace_identity",
    "protected_identity", "transcript", "child_returncode",
    "failure_class", "failure_detail_class", "terminal_errors",
    "taint_verdict", "retry_increment", "codex_exec_appended",
    "process_tree_quiesced", "detached_processes_proven_absent",
    "wip_restore_logical_state_sha256", "canonical_digest",
    *Status.FRONTIER_BINDING_FIELDS,
})
SANDBOX_ISOLATION_RESULT_KEYS = frozenset({
    "game", "target_level", "reached", "result", "reason",
    "dispatch_id", "retry_complexity_n", "seed_mode", "wip_mode",
    "lineage_input_mode", "scratch_root", "scratch_root_disposition",
    "process_tree_quiesced", "detached_processes_proven_absent",
})
ZERO_LEDGER_EVENT_BASE_KEYS = frozenset({
    "event", "schema", "recorded_at", "infrastructure_authority",
    "dispatch_id", "game", "target_level", "reached",
    "parent_action_count", "retry_complexity_n", "workspace",
    "workspace_identity", "protected_identity", "transcript",
    "protected_transcript_sha256", "child_returncode", "failure_class",
    "failure_detail_class", "terminal_errors", "taint_verdict",
    "retry_increment", "codex_exec_appended", "process_tree_quiesced",
    "wip_restore_logical_state_sha256", "canonical_digest",
    *Status.FRONTIER_BINDING_FIELDS,
})
ZERO_LEDGER_EVENT_DIAGNOSTICS_KEYS = frozenset({
    "diagnostics", "protected_diagnostics_sha256",
})
ZERO_LEDGER_RESULT_KEYS = frozenset({
    "game", "target_level", "reached", "result", "reason",
    "child_returncode", "retry_complexity_n", "seed_mode", "wip_mode",
    "lineage_input_mode", "zero_ledger_replayed",
})
RUNNER_RECEIPT_KEYS = frozenset({
    "schema",
    "worktree",
    "cwd",
    "interpreter",
    "head_commit",
    "source_sha256",
    "artifacts_root",
    "scratch_root",
    "ledger",
    "evidence_schema",
    "lock_schema",
})
EVIDENCE_SCHEMAS = frozenset({
    "sealed_transcript_only_v1",
    "sealed_transcript_diagnostics_v1",
})
LOCK_SCHEMAS = frozenset({"in_workspace_v1", "hashed_external_v1"})
PINNED_HISTORICAL_RUNNERS = {
    # Exact submitted-benchmark protocol used for the LF52 continuation.
    "bb3474290d3411f980d53ffcee75be8234e634d478b1136677b9c6a93fe9ec64": {
        "head_commit": "c1f8168f230732f2d745c234555b3e3dfcb8aefa",
        "evidence_schema": "sealed_transcript_only_v1",
        "lock_schema": "in_workspace_v1",
    },
}
MAX_WIP_SNAPSHOT_ENTRIES = 200_000
MAX_WIP_SNAPSHOT_BYTES = 16 * 1024 * 1024 * 1024
MAX_CANONICAL_ROLLBACK_ENTRIES = 100_000
MAX_CANONICAL_ROLLBACK_BYTES = 512 * 1024 * 1024
LIVE_BOUNDARY_POLL_SECONDS = 0.25
EXACT_CHILD_TERMINATE_SECONDS = 30.0
UNQUIESCED_BOUNDARY_CODES = frozenset({
    "detached_process_escape",
    "shell_or_subprocess_escape",
    "dynamic_execution",
})


class CampaignPlanError(RuntimeError):
    pass


class UnquiescedChildError(CampaignPlanError):
    """The exact runner's complete descendant absence is not proven."""

    def __init__(
        self, message: str, *, details: dict[str, Any] | None = None
    ) -> None:
        super().__init__(message)
        self.details = dict(details or {})


class ConfirmedGenerationTaint(CampaignPlanError):
    """Terminal authentication independently confirmed current-policy taint."""


class NoDispatchQuarantine(CampaignPlanError):
    """No marker exists; a completed recovery receipt may still prove success."""


class MissingWipRollbackCapsule(CampaignPlanError):
    """A v2 marker remains after its bound capsule was durably retired."""


class IncompleteDispatchReleaseAuthority(CampaignPlanError):
    """A release intent lacks its complete durable host authority row."""


@dataclass(frozen=True)
class GuardedChildResult:
    returncode: int
    taint_reason: str | None = None
    workspace: str | None = None
    transcript: str | None = None
    workspace_identity: tuple[int, int] | None = None
    protected_identity: tuple[int, int] | None = None
    descendant_quiescence_unproven: bool = False
    process_tree_quiesced: bool = False
    detached_processes_proven_absent: bool = False


@dataclass(frozen=True)
class WipAbsenceCustody:
    """Persisted metadata for the namespace that proves an absent WIP root."""

    parent: Path
    name: str
    parent_identity: tuple[int, int]
    parent_mode: int
    parent_uid: int
    parent_gid: int
    parent_xattrs: tuple[tuple[str, bytes], ...]
    parent_atime_ns: int
    parent_mtime_ns: int
    parent_ctime_ns: int


@dataclass(frozen=True)
class WipRollbackState:
    level: Path
    baseline_snapshot: tuple[str, str | None]
    existed: bool
    level_identity: tuple[int, int] | None
    level_mode: int | None
    level_uid: int | None
    level_gid: int | None
    level_xattrs: tuple[tuple[str, bytes], ...] | None
    level_atime_ns: int | None
    level_mtime_ns: int | None
    level_ctime_ns: int | None
    entries: dict[str, tuple[Any, ...]]
    latest_bytes: bytes | None
    absence_custody: WipAbsenceCustody | None


@dataclass(frozen=True)
class CanonicalEntry:
    kind: str
    mode: int
    atime_ns: int
    mtime_ns: int
    payload: bytes | None = None
    uid: int = 0
    gid: int = 0
    xattrs: tuple[tuple[str, bytes], ...] = ()


@dataclass(frozen=True)
class CanonicalRollbackState:
    root: Path
    root_identity: tuple[int, int]
    root_mode: int
    root_uid: int
    root_gid: int
    root_xattrs: tuple[tuple[str, bytes], ...]
    root_atime_ns: int
    root_mtime_ns: int
    entries: dict[str, CanonicalEntry]
    digest: str
    excluded_prefixes: frozenset[str]


@dataclass
class SchedulerArtifactLock:
    handle: Any
    root: Path
    root_identity: tuple[int, int]
    path: Path
    lock_identity: tuple[int, int]


@dataclass
class DispatchQuarantine:
    root: Path
    root_fd: int
    root_identity: tuple[int, int]
    name: str
    path: Path
    marker_fd: int
    marker_identity: tuple[int, int]
    dispatch_id: str
    schema: str = DISPATCH_QUARANTINE_SCHEMA
    capsule_name: str | None = None
    capsule_identity: tuple[int, int] | None = None
    capsule_state: WipRollbackState | None = None
    capsule_record: dict[str, Any] | None = None
    capsule_missing: bool = False
    recovery_sealed_size: int | None = None
    recovery_sealed_sha256: str | None = None


@dataclass(frozen=True)
class LedgerPrefixState:
    path: Path
    parent_identity: tuple[int, int]
    file_identity: tuple[int, int] | None
    raw_prefix: bytes
    records: list[dict[str, Any]]


@dataclass(frozen=True)
class PostRebootLedgerState:
    dispatch_id: str
    intent_root: Path | None
    intent_root_identity: tuple[int, int] | None
    ledger: Path
    baseline: LedgerPrefixState
    record: dict[str, Any]
    correction: dict[str, Any] | None
    cleanup: dict[str, Any] | None
    operator: dict[str, Any] | None

    @property
    def phase(self) -> int:
        return 1 + sum(
            row is not None
            for row in (self.correction, self.cleanup, self.operator)
        )


def _single_cli_value(argv: list[str], option: str) -> str | None:
    prefix = f"{option}="
    values = [argument[len(prefix):] for argument in argv if argument.startswith(prefix)]
    if len(values) > 1:
        raise CampaignPlanError(f"duplicate campaign argument: {option}")
    return values[0] if values else None


def _normalized_absolute_path(value: object, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise CampaignPlanError(f"runner receipt {label} is malformed")
    path = Path(value)
    if not path.is_absolute() or Path(os.path.abspath(value)) != path:
        raise CampaignPlanError(
            f"runner receipt {label} must be an absolute normalized path"
        )
    return path


def _reject_symlinked_ancestry(path: Path, label: str) -> None:
    """Reject a host path reached through any symlinked directory component."""

    if not path.is_absolute():
        raise CampaignPlanError(f"{label} is not absolute")
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        try:
            metadata = current.stat(follow_symlinks=False)
        except OSError as exc:
            raise CampaignPlanError(f"{label} is unavailable: {current}") from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise CampaignPlanError(
                f"{label} has symlinked ancestry: {current}"
            )


def _runner_receipt(
    plan: dict[str, Any], *, allow_abandoned_scratch: bool = False
) -> dict[str, Any] | None:
    receipt = plan.get("runner_receipt")
    if receipt is None:
        return None
    if (
        not isinstance(receipt, dict)
        or set(receipt) != RUNNER_RECEIPT_KEYS
        or receipt.get("schema") != RUNNER_RECEIPT_SCHEMA
    ):
        raise CampaignPlanError("plan runner_receipt has an invalid schema")
    worktree = _normalized_absolute_path(receipt.get("worktree"), "worktree")
    cwd = _normalized_absolute_path(receipt.get("cwd"), "cwd")
    interpreter = _normalized_absolute_path(
        receipt.get("interpreter"), "interpreter"
    )
    artifacts_root = _normalized_absolute_path(
        receipt.get("artifacts_root"), "artifacts_root"
    )
    scratch_root = _normalized_absolute_path(
        receipt.get("scratch_root"), "scratch_root"
    )
    ledger = _normalized_absolute_path(receipt.get("ledger"), "ledger")
    if worktree != cwd:
        raise CampaignPlanError(
            "historical runner cwd must equal its pinned worktree"
        )
    if artifacts_root != (HERE / "agent_solutions").absolute():
        raise CampaignPlanError(
            "historical runner artifacts_root is not the canonical artifact root"
        )
    if scratch_root != Path(Legs.SCRATCH).absolute():
        raise CampaignPlanError(
            "historical runner scratch_root is not the scheduler scratch root"
        )
    if ledger != Path(Guard.DEFAULT_LEDGER).absolute():
        raise CampaignPlanError(
            "historical runner ledger is not the canonical Codex ledger"
        )
    if receipt.get("evidence_schema") not in EVIDENCE_SCHEMAS:
        raise CampaignPlanError("historical runner evidence_schema is invalid")
    if receipt.get("lock_schema") not in LOCK_SCHEMAS:
        raise CampaignPlanError("historical runner lock_schema is invalid")
    try:
        if (
            interpreter.resolve(strict=True)
            != Path(sys.executable).resolve(strict=True)
        ):
            raise CampaignPlanError(
                "historical runner interpreter is not the scheduler interpreter"
            )
    except OSError as exc:
        raise CampaignPlanError(
            "historical runner interpreter is unavailable"
        ) from exc
    head_commit = receipt.get("head_commit")
    source_sha256 = receipt.get("source_sha256")
    if (
        not isinstance(head_commit, str)
        or GIT_COMMIT_RE.fullmatch(head_commit) is None
        or not isinstance(source_sha256, str)
        or SHA256_RE.fullmatch(source_sha256) is None
    ):
        raise CampaignPlanError("plan runner_receipt hashes are malformed")
    pinned = PINNED_HISTORICAL_RUNNERS.get(source_sha256)
    if pinned is None or any(
        receipt.get(field) != expected
        for field, expected in pinned.items()
    ):
        raise CampaignPlanError(
            "historical runner is not an approved exact protocol receipt"
        )
    if not allow_abandoned_scratch:
        _reject_abandoned_scratch_root(scratch_root, ledger)
    return dict(receipt)


def _reject_abandoned_scratch_root(scratch_root: Path, ledger: Path) -> None:
    """Never dispatch into a namespace abandoned by operator isolation."""

    try:
        records = Guard.read_ledger(ledger)
    except (OSError, ValueError, Guard.CodexUsageGuardError) as exc:
        raise CampaignPlanError(
            "cannot authenticate abandoned scratch namespaces"
        ) from exc
    try:
        current = scratch_root.stat(follow_symlinks=False)
        current_identity: tuple[int, int] | None = (
            current.st_dev, current.st_ino
        )
    except FileNotFoundError:
        current_identity = None
    except OSError as exc:
        raise CampaignPlanError("runner scratch root is unavailable") from exc
    try:
        _reject_symlinked_ancestry(scratch_root, "runner scratch root")
        resolved_scratch = scratch_root.resolve(strict=True)
    except OSError as exc:
        raise CampaignPlanError("runner scratch root is unavailable") from exc
    for record in records:
        if (
            record.get("event") != SANDBOX_ABANDON_EVENT
            or record.get("schema") != SANDBOX_ABANDON_EVENT_SCHEMA
        ):
            continue
        abandoned = _normalized_absolute_path(
            record.get("scratch_root"), "abandoned scratch_root"
        )
        identity = _marker_identity(
            record.get("scratch_root_identity"), "abandoned scratch root"
        )
        try:
            _reject_symlinked_ancestry(abandoned, "abandoned scratch root")
            resolved_abandoned = abandoned.resolve(strict=True)
        except OSError as exc:
            raise CampaignPlanError(
                "abandoned scratch root is unavailable"
            ) from exc
        if (
            scratch_root == abandoned
            or scratch_root.is_relative_to(abandoned)
            or abandoned.is_relative_to(scratch_root)
            or resolved_scratch == resolved_abandoned
            or resolved_scratch.is_relative_to(resolved_abandoned)
            or resolved_abandoned.is_relative_to(resolved_scratch)
            or current_identity == identity
        ):
            raise CampaignPlanError(
                "runner scratch root belongs to an abandoned sandbox namespace"
            )


def _project_runner_receipt(
    plan: dict[str, Any], item: dict[str, Any], *,
    allow_abandoned_scratch: bool = False,
) -> dict[str, Any]:
    """Project one plan-level historical runner onto any policy-built item."""

    receipt = _runner_receipt(
        plan, allow_abandoned_scratch=allow_abandoned_scratch
    )
    if receipt is None:
        if item.get("historical_runner") is not None:
            raise CampaignPlanError(
                "item-level historical runner lacks a plan-level receipt"
            )
        return item
    argv = item.get("argv")
    if not isinstance(argv, list) or not all(isinstance(x, str) for x in argv):
        raise CampaignPlanError("cannot project runner receipt onto malformed argv")
    source = os.fspath(
        Path(receipt["worktree"]) / "arc" / "crack_lab" / "gkm_legs.py"
    )
    historical_prefix = [receipt["interpreter"], "-u", source]
    current_prefix = ["python3", "-u", "arc/crack_lab/gkm_legs.py"]
    if argv[:3] not in (current_prefix, historical_prefix):
        raise CampaignPlanError(
            "plan runner receipt cannot replace an unrelated command prefix"
        )
    existing = item.get("historical_runner")
    if existing is not None and existing != receipt:
        raise CampaignPlanError(
            "item historical runner disagrees with plan runner receipt"
        )
    projected_argv = [*historical_prefix, *argv[3:]]
    pinned_options = {
        "--artifacts-root": receipt["artifacts_root"],
        "--codex-ledger": receipt["ledger"],
    }
    for option, value in pinned_options.items():
        observed = _single_cli_value(projected_argv, option)
        if observed is not None and observed != value:
            raise CampaignPlanError(
                f"item {option} disagrees with plan runner receipt"
            )
        if observed is None:
            projected_argv.append(f"{option}={value}")
    projected = dict(item)
    projected["historical_runner"] = receipt
    projected["argv"] = projected_argv
    projected["command"] = shlex.join(projected_argv)
    return projected


def _runner_cwd(item: dict[str, Any]) -> Path:
    authority = item.get("historical_runner")
    if authority is None:
        return REPO
    if not isinstance(authority, dict):
        raise CampaignPlanError("historical runner receipt is malformed")
    return _normalized_absolute_path(authority.get("cwd"), "cwd")


def _runner_env(item: dict[str, Any]) -> dict[str, str] | None:
    authority = item.get("historical_runner")
    if authority is None:
        return None
    if not isinstance(authority, dict):
        raise CampaignPlanError("historical runner receipt is malformed")
    scratch = _normalized_absolute_path(
        authority.get("scratch_root"), "scratch_root"
    )
    environment = os.environ.copy()
    environment["GKM_SCRATCH"] = os.fspath(scratch)
    return environment


def _evidence_schema(item: dict[str, Any]) -> str:
    authority = item.get("historical_runner")
    if authority is None:
        return "sealed_transcript_diagnostics_v1"
    schema = authority.get("evidence_schema") if isinstance(authority, dict) else None
    if schema not in EVIDENCE_SCHEMAS:
        raise CampaignPlanError("historical runner evidence schema is unavailable")
    return str(schema)


def _lock_schema(item: dict[str, Any]) -> str:
    authority = item.get("historical_runner")
    if authority is None:
        return "hashed_external_v1"
    schema = authority.get("lock_schema") if isinstance(authority, dict) else None
    if schema not in LOCK_SCHEMAS:
        raise CampaignPlanError("historical runner lock schema is unavailable")
    return str(schema)


def _ledger_path(argv: list[str], *, cwd: Path) -> Path:
    selected = _single_cli_value(argv, "--codex-ledger")
    path = Path(selected) if selected is not None else Path(Guard.DEFAULT_LEDGER)
    if not path.is_absolute():
        path = cwd / path
    return path.absolute()


def _historical_runner_git_state(worktree: Path) -> tuple[str, bool]:
    head = subprocess.run(
        ["git", "-C", os.fspath(worktree), "rev-parse", "--verify", "HEAD"],
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    clean = subprocess.run(
        [
            "git", "-C", os.fspath(worktree), "status", "--porcelain=v1",
            "--untracked-files=all",
        ],
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return (
        head.stdout.strip().lower() if head.returncode == 0 else "",
        clean.returncode == 0 and not clean.stdout,
    )


def _validate_runner_prefix(
    item: dict[str, Any], argv: list[str], *,
    allow_abandoned_scratch: bool = False,
) -> None:
    """Admit the current runner or one receipt-bound historical worktree."""

    if argv[:3] == ["python3", "-u", "arc/crack_lab/gkm_legs.py"]:
        if item.get("historical_runner") is not None:
            raise CampaignPlanError(
                "relative current runner carries unexpected historical authority"
            )
        return
    authority = item.get("historical_runner")
    if (
        not isinstance(authority, dict)
        or set(authority) != RUNNER_RECEIPT_KEYS
        or authority.get("schema") != RUNNER_RECEIPT_SCHEMA
    ):
        raise CampaignPlanError(
            f"refusing non-GKM command prefix: {argv[:3]!r}"
        )
    authority = _runner_receipt(
        {"runner_receipt": authority},
        allow_abandoned_scratch=allow_abandoned_scratch,
    )
    if authority is None:  # pragma: no cover - guarded by the branch above.
        raise CampaignPlanError("historical runner authority is missing")
    worktree_raw = authority["worktree"]
    head_commit = authority["head_commit"]
    source_sha256 = authority["source_sha256"]
    worktree = _normalized_absolute_path(worktree_raw, "worktree")
    if _normalized_absolute_path(authority["cwd"], "cwd") != worktree:
        raise CampaignPlanError("historical runner cwd does not match worktree")
    if (
        not isinstance(head_commit, str)
        or GIT_COMMIT_RE.fullmatch(head_commit) is None
        or not isinstance(source_sha256, str)
        or SHA256_RE.fullmatch(source_sha256) is None
    ):
        raise CampaignPlanError("historical runner authority is malformed")
    _reject_symlinked_ancestry(worktree, "historical runner worktree")
    _host_directory_identity(worktree, "historical runner worktree")
    source = worktree / "arc" / "crack_lab" / "gkm_legs.py"
    _reject_symlinked_ancestry(source, "historical runner source")
    if len(argv) < 3 or argv[1] != "-u" or Path(argv[2]) != source:
        raise CampaignPlanError(
            "absolute runner path does not match its pinned worktree"
        )
    try:
        invoked_python = Path(argv[0]).resolve(strict=True)
        scheduler_python = Path(sys.executable).resolve(strict=True)
    except OSError as exc:
        raise CampaignPlanError("historical runner Python is unavailable") from exc
    if (
        argv[0] != authority["interpreter"]
        or not Path(argv[0]).is_absolute()
        or invoked_python != scheduler_python
    ):
        raise CampaignPlanError(
            "historical runner must use the scheduler's absolute Python"
        )
    if _single_cli_value(argv, "--artifacts-root") != authority["artifacts_root"]:
        raise CampaignPlanError(
            "historical runner does not consume its pinned artifacts root"
        )
    if _single_cli_value(argv, "--codex-ledger") != authority["ledger"]:
        raise CampaignPlanError(
            "historical runner does not consume its pinned Codex ledger"
        )
    try:
        source_bytes = Legs._read_single_link_regular(os.fspath(source))
    except (OSError, Legs.WorkspaceTainted) as exc:
        raise CampaignPlanError("historical runner source is unstable") from exc
    if hashlib.sha256(source_bytes).hexdigest() != source_sha256:
        raise CampaignPlanError("historical runner source hash mismatch")
    actual_head, tracked_clean = _historical_runner_git_state(worktree)
    if actual_head != head_commit or not tracked_clean:
        raise CampaignPlanError(
            "historical runner worktree does not match its pinned clean HEAD"
        )


def _revalidate_historical_control(
    item: dict[str, Any], *, allow_abandoned_scratch: bool = False
) -> None:
    """Recheck every pinned worktree byte surface at a terminal boundary."""

    authority = item.get("historical_runner")
    if authority is None:
        return
    # Production dispatches can reach this helper only after validate_item has
    # required the complete receipt.  Small direct unit fixtures intentionally
    # exercise lower-level recovery with a reduced authority object.
    if isinstance(authority, dict) and set(authority) == RUNNER_RECEIPT_KEYS:
        _validate_runner_prefix(
            item,
            item["argv"],
            allow_abandoned_scratch=allow_abandoned_scratch,
        )


def _read_ledger_locked(path: Path) -> list[dict[str, Any]]:
    with Guard.ledger_append_lock(path):
        return Guard.read_ledger(path)


def _strict_ledger_records(raw: bytes, *, label: str) -> list[dict[str, Any]]:
    """Decode an exact append-only JSONL surface without dropping any row."""

    if raw and not raw.endswith(b"\n"):
        raise CampaignPlanError(f"{label} lacks a final line boundary")
    records: list[dict[str, Any]] = []
    for index, line in enumerate(raw.splitlines(), 1):
        if not line:
            raise CampaignPlanError(f"{label} contains a blank JSONL row")
        try:
            record = json.loads(line)
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise CampaignPlanError(
                f"{label} contains malformed JSON at row {index}"
            ) from exc
        if not isinstance(record, dict):
            raise CampaignPlanError(
                f"{label} contains a non-object row at {index}"
            )
        records.append(record)
    return records


def _capture_ledger_prefix(path: Path) -> LedgerPrefixState:
    """Bind the pre-dispatch ledger inode and exact append prefix."""

    selected = Path(path).absolute()
    _reject_symlinked_ancestry(selected.parent, "Codex ledger parent")
    with Guard.ledger_append_lock(selected):
        parent_identity = _host_directory_identity(
            selected.parent, "Codex ledger parent"
        )
        if os.path.lexists(selected):
            metadata = selected.stat(follow_symlinks=False)
            if (
                selected.is_symlink()
                or not stat.S_ISREG(metadata.st_mode)
                or metadata.st_nlink != 1
            ):
                raise CampaignPlanError("Codex ledger is not an unaliased file")
            try:
                raw = Legs._read_single_link_regular(os.fspath(selected))
            except (OSError, Legs.WorkspaceTainted) as exc:
                raise CampaignPlanError("Codex ledger prefix is unstable") from exc
            file_identity: tuple[int, int] | None = (
                metadata.st_dev,
                metadata.st_ino,
            )
        else:
            raw = b""
            file_identity = None
        records = _strict_ledger_records(raw, label="Codex ledger prefix")
    return LedgerPrefixState(
        path=selected,
        parent_identity=parent_identity,
        file_identity=file_identity,
        raw_prefix=raw,
        records=records,
    )


def _ensure_durable_ledger_file(path: Path) -> None:
    """Ensure a dispatch can bind one stable ledger inode before arming."""

    selected = Path(path).absolute()
    _reject_symlinked_ancestry(selected.parent, "Codex ledger parent")
    with Guard.ledger_append_lock(selected):
        if os.path.lexists(selected):
            metadata = selected.stat(follow_symlinks=False)
            if (
                selected.is_symlink()
                or not stat.S_ISREG(metadata.st_mode)
                or metadata.st_nlink != 1
            ):
                raise CampaignPlanError("Codex ledger is not an unaliased file")
            return
        descriptor: int | None = None
        try:
            descriptor = os.open(
                selected,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL
                | getattr(os, "O_NOFOLLOW", 0),
                0o600,
            )
            os.fchmod(descriptor, 0o600)
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise CampaignPlanError(
                    "new Codex ledger is not an unaliased file"
                )
            os.fsync(descriptor)
            _fsync_directory(selected.parent)
        except OSError as exc:
            raise CampaignPlanError(
                "could not durably create the Codex ledger"
            ) from exc
        finally:
            if descriptor is not None:
                os.close(descriptor)


def _ledger_records_after_prefix(
    state: LedgerPrefixState,
) -> list[dict[str, Any]]:
    with Guard.ledger_append_lock(state.path):
        if (
            _host_directory_identity(state.path.parent, "Codex ledger parent")
            != state.parent_identity
            or not os.path.lexists(state.path)
        ):
            raise CampaignPlanError("Codex ledger parent/file identity changed")
        metadata = state.path.stat(follow_symlinks=False)
        if (
            state.path.is_symlink()
            or not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or (
                state.file_identity is not None
                and (metadata.st_dev, metadata.st_ino) != state.file_identity
            )
        ):
            raise CampaignPlanError("Codex ledger identity changed after dispatch")
        try:
            raw = Legs._read_single_link_regular(os.fspath(state.path))
        except (OSError, Legs.WorkspaceTainted) as exc:
            raise CampaignPlanError("Codex ledger is unstable after dispatch") from exc
        if not raw.startswith(state.raw_prefix):
            raise CampaignPlanError("Codex ledger rewrote its sealed byte prefix")
        suffix = raw[len(state.raw_prefix):]
        suffix_records = _strict_ledger_records(
            suffix, label="Codex ledger dispatch suffix"
        )
        return [*state.records, *suffix_records]


def _assert_no_incomplete_taint_cleanup(
    records: list[dict[str, Any]],
) -> None:
    """Fail closed if a prior scheduler quarantine crashed before completion."""

    pending: set[tuple[str, str, str]] = set()
    for record in records:
        if (
            record.get("event") == "codex_taint_cleanup_completed"
            and record.get("cleanup_authority")
            == "scheduler_exact_generation_cleanup_v1"
        ):
            pending.discard((
                _safe_component(record.get("thread_id"), "thread id"),
                _safe_component(record.get("transcript"), "transcript"),
                _safe_component(record.get("workspace"), "workspace"),
            ))
        elif (
            record.get("event")
            == "codex_exec_classification_correction"
            and record.get("classification_authority")
            == "scheduler_exact_generation_taint_scan_v1"
            and record.get("failure_class") == "taint"
        ):
            pending.add((
                _safe_component(record.get("thread_id"), "thread id"),
                _safe_component(record.get("transcript"), "transcript"),
                _safe_component(record.get("workspace"), "workspace"),
            ))
    if pending:
        raise CampaignPlanError(
            "prior exact-generation taint correction lacks cleanup completion; "
            "refusing a new dispatch"
        )


def _expected_exec_record(
    item: dict[str, Any],
    before: list[dict[str, Any]] | LedgerPrefixState,
    after: list[dict[str, Any]] | None = None,
    *,
    clean_terminal: bool | None = False,
) -> dict[str, Any]:
    """Select exactly one newly appended exec bound to this dispatch."""

    if isinstance(before, LedgerPrefixState):
        before_records = before.records
        after_records = _ledger_records_after_prefix(before)
    else:
        if after is None:
            raise CampaignPlanError("Codex ledger suffix is unavailable")
        before_records = before
        after_records = after
    if (
        len(after_records) < len(before_records)
        or after_records[:len(before_records)] != before_records
    ):
        raise CampaignPlanError("Codex ledger changed other than by append")
    expected = {
        "event": "codex_exec",
        "game": item["game"],
        "target_level": item["target_level"],
        "run_label": f"{item['game']}:L{item['target_level']}:propose",
        "model": "gpt-5.6-sol",
        "reasoning_effort": item["effort"],
        "minutes_limit": item["minutes"],
        "allocation_policy": "drain",
        "reached": item["reached"],
        "parent_action_count": item["parent_action_count"],
        **{
            field: item[field]
            for field in Status.FRONTIER_BINDING_FIELDS
        },
    }
    suffix = after_records[len(before_records):]
    if clean_terminal is None:
        if len(suffix) not in {1, 2}:
            raise CampaignPlanError(
                "exact child appended an ambiguous recovery ledger suffix: "
                f"found {len(suffix)} row(s)"
            )
        expected_length = len(suffix)
    else:
        expected_length = 2 if clean_terminal else 1
    if len(suffix) != expected_length:
        raise CampaignPlanError(
            "exact child appended an ambiguous ledger suffix: expected "
            f"{expected_length} row(s), found {len(suffix)}"
        )
    record = suffix[0]
    if not all(record.get(key) == value for key, value in expected.items()):
        raise CampaignPlanError(
            "exact child did not append the bound Codex exec record"
        )
    if clean_terminal is True or (
        clean_terminal is None and expected_length == 2
    ):
        outcome = suffix[1]
        if outcome.get("event") != "codex_level_outcome":
            raise CampaignPlanError(
                "exact child appended an ambiguous recovery ledger suffix"
            )
        expected_outcome = {
            "event": "codex_level_outcome",
            "codex_exec_transcript": record.get("transcript"),
            "thread_id": record.get("thread_id"),
            "game": item["game"],
            "target_level": item["target_level"],
            "run_label": expected["run_label"],
            "model": expected["model"],
            "reasoning_effort": item["effort"],
            "reached": item["reached"],
            "reached_before": item["reached"],
            "taint_verdict": "clean",
            **{
                field: item[field]
                for field in Status.FRONTIER_BINDING_FIELDS
            },
        }
        if not all(
            outcome.get(key) == value
            for key, value in expected_outcome.items()
        ):
            raise CampaignPlanError(
                "clean child outcome does not bind its exact Codex exec"
            )
        reached_after = outcome.get("reached_after")
        solved_target = outcome.get("solved_target")
        if (
            not isinstance(reached_after, int)
            or isinstance(reached_after, bool)
            or not item["reached"] <= reached_after <= item["target_level"]
            or solved_target is not (reached_after >= item["target_level"])
        ):
            raise CampaignPlanError(
                "clean child outcome has an invalid reached transition"
            )
    return record


def _safe_component(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value in {".", ".."}
        or Path(value).name != value
        or SAFE_COMPONENT_RE.fullmatch(value) is None
    ):
        raise CampaignPlanError(f"unsafe {label} component in Codex exec record")
    return value


def _safe_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise CampaignPlanError(f"missing or malformed sealed {label} hash")
    return value


def _host_directory_identity(path: Path, label: str) -> tuple[int, int]:
    try:
        metadata = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise CampaignPlanError(f"{label} is unavailable: {path}") from exc
    if path.is_symlink() or not stat.S_ISDIR(metadata.st_mode):
        raise CampaignPlanError(f"{label} is not an unaliased directory: {path}")
    return metadata.st_dev, metadata.st_ino


def _sealed_file_bytes(path: Path, expected_sha256: str, label: str) -> bytes:
    if path.parent / path.name != path or path.is_symlink():
        raise CampaignPlanError(f"unsafe sealed {label} path: {path}")
    try:
        payload = Legs._read_single_link_regular(os.fspath(path))
    except (OSError, Legs.WorkspaceTainted) as exc:
        raise CampaignPlanError(f"sealed {label} is unavailable: {path}") from exc
    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected_sha256:
        raise CampaignPlanError(f"sealed {label} hash does not match ledger")
    return payload


def _artifact_root(item: dict[str, Any]) -> Path:
    canonical = (HERE / "agent_solutions").absolute()
    selected = _single_cli_value(item["argv"], "--artifacts-root")
    inherited = os.environ.get("GKM_ARTIFACTS_ROOT") if selected is None else None
    if selected is None and inherited is None:
        return canonical
    effective = _normalized_absolute_path(
        selected if selected is not None else inherited,
        "artifacts_root",
    )
    if effective != canonical:
        raise CampaignPlanError(
            "dispatch artifacts root is not the canonical artifact root"
        )
    return canonical


def _stable_regular_sha256(path: Path, metadata: os.stat_result) -> str:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise CampaignPlanError(f"WIP file is unavailable: {path}") from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            != (
                metadata.st_dev,
                metadata.st_ino,
                metadata.st_size,
                metadata.st_mtime_ns,
            )
        ):
            raise CampaignPlanError(f"WIP file changed before hashing: {path}")
        digest = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        after = os.fstat(descriptor)
        if (
            after.st_nlink != 1
            or (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
            != (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        ):
            raise CampaignPlanError(f"WIP file changed during hashing: {path}")
        return digest.hexdigest()
    finally:
        os.close(descriptor)


def _target_wip_snapshot(
    item: dict[str, Any],
) -> tuple[str, str | None]:
    """Fingerprint the target-level WIP inventory without deleting forensics."""

    level = _target_wip_level(item)
    if not os.path.lexists(level):
        return os.fspath(level), None
    _reject_symlinked_ancestry(level, "target WIP directory")
    root_identity = _host_directory_identity(level, "target WIP directory")
    manifest = hashlib.sha256()
    entry_count = 0
    total_bytes = 0
    try:
        walker = os.walk(level, topdown=True, followlinks=False)
        for directory_raw, dirs, files in walker:
            directory = Path(directory_raw)
            directory_metadata = directory.stat(follow_symlinks=False)
            if directory.is_symlink() or not stat.S_ISDIR(
                directory_metadata.st_mode
            ):
                raise CampaignPlanError(
                    "target WIP inventory contains an aliased directory"
                )
            relative_directory = directory.relative_to(level).as_posix()
            manifest.update(json.dumps([
                relative_directory,
                "directory",
                directory_metadata.st_dev,
                directory_metadata.st_ino,
                directory_metadata.st_mode,
                directory_metadata.st_mtime_ns,
            ], separators=(",", ":")).encode("utf-8"))
            entry_count += 1
            dirs.sort()
            files.sort()
            for name in dirs:
                child = directory / name
                child_metadata = child.stat(follow_symlinks=False)
                if child.is_symlink() or not stat.S_ISDIR(child_metadata.st_mode):
                    raise CampaignPlanError(
                        "target WIP inventory contains an unsafe directory"
                    )
            for name in files:
                child = directory / name
                metadata = child.stat(follow_symlinks=False)
                if (
                    child.is_symlink()
                    or not stat.S_ISREG(metadata.st_mode)
                    or metadata.st_nlink != 1
                ):
                    raise CampaignPlanError(
                        "target WIP inventory contains an unsafe file"
                    )
                entry_count += 1
                total_bytes += metadata.st_size
                if (
                    entry_count > MAX_WIP_SNAPSHOT_ENTRIES
                    or total_bytes > MAX_WIP_SNAPSHOT_BYTES
                ):
                    raise CampaignPlanError(
                        "target WIP inventory exceeds the scheduler scan bound"
                    )
                manifest.update(json.dumps([
                    child.relative_to(level).as_posix(),
                    "file",
                    metadata.st_dev,
                    metadata.st_ino,
                    metadata.st_mode,
                    metadata.st_size,
                    metadata.st_mtime_ns,
                    _stable_regular_sha256(child, metadata),
                ], separators=(",", ":")).encode("utf-8"))
    except OSError as exc:
        raise CampaignPlanError("target WIP inventory is unstable") from exc
    if entry_count > MAX_WIP_SNAPSHOT_ENTRIES:
        raise CampaignPlanError(
            "target WIP inventory exceeds the scheduler scan bound"
        )
    if _host_directory_identity(level, "target WIP directory") != root_identity:
        raise CampaignPlanError("target WIP inventory changed during snapshot")
    return os.fspath(level), manifest.hexdigest()


def _target_wip_level(item: dict[str, Any]) -> Path:
    return (
        _artifact_root(item)
        / f"{item['game']}_legs"
        / "wip_context"
        / f"level_{int(item['target_level']):02d}"
    )


def _wip_entry_inventory(level: Path) -> dict[str, tuple[Any, ...]]:
    """Seal every WIP entry so rollback cannot mask an unrelated mutation."""

    entries: dict[str, tuple[Any, ...]] = {}
    if not os.path.lexists(level):
        return entries
    try:
        for directory_raw, dirs, files in os.walk(
            level, topdown=True, followlinks=False
        ):
            directory = Path(directory_raw)
            dirs.sort()
            files.sort()
            for name in dirs:
                child = directory / name
                metadata = child.stat(follow_symlinks=False)
                if child.is_symlink() or not stat.S_ISDIR(metadata.st_mode):
                    raise CampaignPlanError(
                        "target WIP inventory contains an unsafe directory"
                    )
                entries[child.relative_to(level).as_posix()] = (
                    "directory",
                    metadata.st_dev,
                    metadata.st_ino,
                    metadata.st_mode,
                    metadata.st_nlink,
                    metadata.st_size,
                    metadata.st_mtime_ns,
                    metadata.st_ctime_ns,
                    metadata.st_uid,
                    metadata.st_gid,
                    _canonical_xattrs(child),
                )
            for name in files:
                child = directory / name
                metadata = child.stat(follow_symlinks=False)
                if (
                    child.is_symlink()
                    or not stat.S_ISREG(metadata.st_mode)
                    or metadata.st_nlink != 1
                ):
                    raise CampaignPlanError(
                        "target WIP inventory contains an unsafe file"
                    )
                entries[child.relative_to(level).as_posix()] = (
                    "file",
                    metadata.st_dev,
                    metadata.st_ino,
                    metadata.st_mode,
                    metadata.st_nlink,
                    metadata.st_size,
                    metadata.st_mtime_ns,
                    metadata.st_ctime_ns,
                    _stable_regular_sha256(child, metadata),
                    metadata.st_uid,
                    metadata.st_gid,
                    _canonical_xattrs(child),
                    metadata.st_atime_ns,
                )
    except OSError as exc:
        raise CampaignPlanError("target WIP inventory is unstable") from exc
    return entries


def _capture_wip_absence_custody(level: Path) -> WipAbsenceCustody:
    """Bind the nearest extant directory whose child proves WIP absence."""

    parent = level.parent
    name = level.name
    while not os.path.lexists(parent):
        name = parent.name
        next_parent = parent.parent
        if next_parent == parent:
            raise CampaignPlanError(
                "target WIP absence has no reachable custody parent"
            )
        parent = next_parent
    _reject_symlinked_ancestry(parent, "target WIP absence custody parent")
    descriptor: int | None = None
    try:
        descriptor = os.open(
            parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        opened = os.fstat(descriptor)
        path_opened = parent.stat(follow_symlinks=False)
        identity = (opened.st_dev, opened.st_ino)
        if (
            not stat.S_ISDIR(opened.st_mode)
            or not stat.S_ISDIR(path_opened.st_mode)
            or identity != (path_opened.st_dev, path_opened.st_ino)
        ):
            raise CampaignPlanError(
                "target WIP absence custody parent is unsafe"
            )
        try:
            os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise CampaignPlanError(
                "target WIP absence custody name appeared during capture"
            )
        xattrs = _canonical_xattrs(parent)
        after = os.fstat(descriptor)
        path_after = parent.stat(follow_symlinks=False)
        if any((
            (after.st_dev, after.st_ino) != identity,
            (path_after.st_dev, path_after.st_ino) != identity,
            after.st_mode != opened.st_mode,
            after.st_uid != opened.st_uid,
            after.st_gid != opened.st_gid,
            after.st_mtime_ns != opened.st_mtime_ns,
            after.st_ctime_ns != opened.st_ctime_ns,
            _canonical_xattrs(parent) != xattrs,
        )):
            raise CampaignPlanError(
                "target WIP absence custody changed during capture"
            )
        try:
            os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise CampaignPlanError(
                "target WIP absence custody name appeared during capture"
            )
        return WipAbsenceCustody(
            parent=parent,
            name=name,
            parent_identity=identity,
            parent_mode=opened.st_mode,
            parent_uid=opened.st_uid,
            parent_gid=opened.st_gid,
            parent_xattrs=xattrs,
            parent_atime_ns=opened.st_atime_ns,
            parent_mtime_ns=opened.st_mtime_ns,
            parent_ctime_ns=opened.st_ctime_ns,
        )
    except OSError as exc:
        raise CampaignPlanError(
            "target WIP absence custody is unstable"
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _ensure_durable_wip_context_parent(item: dict[str, Any]) -> None:
    """Materialize the host-owned WIP wrapper before sealing dispatch state."""

    level = _target_wip_level(item)
    parent = level.parent
    canonical_root = _artifact_root(item) / f"{item['game']}_legs"
    if parent.parent != canonical_root:
        raise CampaignPlanError("target WIP parent escaped the canonical root")
    _reject_symlinked_ancestry(canonical_root, "canonical WIP parent root")
    root_fd: int | None = None
    parent_fd: int | None = None
    try:
        root_fd = os.open(
            canonical_root,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        root_before = os.fstat(root_fd)
        created = False
        try:
            os.mkdir(parent.name, 0o700, dir_fd=root_fd)
            created = True
        except FileExistsError:
            pass
        parent_fd = os.open(
            parent.name,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=root_fd,
        )
        opened = os.fstat(parent_fd)
        path_opened = os.stat(
            parent.name, dir_fd=root_fd, follow_symlinks=False
        )
        if (
            not stat.S_ISDIR(opened.st_mode)
            or (opened.st_dev, opened.st_ino)
            != (path_opened.st_dev, path_opened.st_ino)
        ):
            raise CampaignPlanError("target WIP parent is unsafe")
        os.fsync(parent_fd)
        os.fsync(root_fd)
        if created:
            # The wrapper is part of the excluded WIP namespace, not a
            # canonical model mutation.  Keep the durable directory entry but
            # restore the canonical root's logical metadata baseline.
            os.utime(
                canonical_root,
                ns=(root_before.st_atime_ns, root_before.st_mtime_ns),
                follow_symlinks=False,
            )
            os.fsync(root_fd)
            root_after = os.fstat(root_fd)
            if (
                (root_after.st_dev, root_after.st_ino)
                != (root_before.st_dev, root_before.st_ino)
                or root_after.st_mode != root_before.st_mode
                or root_after.st_uid != root_before.st_uid
                or root_after.st_gid != root_before.st_gid
                or root_after.st_mtime_ns != root_before.st_mtime_ns
            ):
                raise CampaignPlanError(
                    "canonical WIP parent root changed during materialization"
                )
    except OSError as exc:
        raise CampaignPlanError(
            "could not durably materialize the WIP context parent"
        ) from exc
    finally:
        if parent_fd is not None:
            os.close(parent_fd)
        if root_fd is not None:
            os.close(root_fd)


def _capture_wip_rollback(item: dict[str, Any]) -> WipRollbackState:
    """Capture the only artifact mutation a tainted legacy exit may undo."""

    baseline = _target_wip_snapshot(item)
    level = _target_wip_level(item)
    if not os.path.lexists(level):
        absence_custody = _capture_wip_absence_custody(level)
        if _target_wip_snapshot(item) != baseline:
            raise CampaignPlanError(
                "target WIP inventory changed during capture"
            )
        return WipRollbackState(
            level=level,
            baseline_snapshot=baseline,
            existed=False,
            level_identity=None,
            level_mode=None,
            level_uid=None,
            level_gid=None,
            level_xattrs=None,
            level_atime_ns=None,
            level_mtime_ns=None,
            level_ctime_ns=None,
            entries={},
            latest_bytes=None,
            absence_custody=absence_custody,
        )
    metadata = level.stat(follow_symlinks=False)
    if level.is_symlink() or not stat.S_ISDIR(metadata.st_mode):
        raise CampaignPlanError("target WIP root is unsafe")
    entries = _wip_entry_inventory(level)
    latest = level / "latest.json"
    latest_bytes: bytes | None = None
    if os.path.lexists(latest):
        try:
            latest_bytes = Legs._read_single_link_regular(os.fspath(latest))
        except (OSError, Legs.WorkspaceTainted) as exc:
            raise CampaignPlanError("target WIP latest.json is unstable") from exc
    # Recheck after the detailed walk: this receipt must describe one instant.
    if _target_wip_snapshot(item) != baseline:
        raise CampaignPlanError("target WIP inventory changed during capture")
    level_xattrs = _canonical_xattrs(level)
    after = level.stat(follow_symlinks=False)
    if (
        (after.st_dev, after.st_ino) != (metadata.st_dev, metadata.st_ino)
        or after.st_mode != metadata.st_mode
        or after.st_uid != metadata.st_uid
        or after.st_gid != metadata.st_gid
        or after.st_mtime_ns != metadata.st_mtime_ns
        or after.st_ctime_ns != metadata.st_ctime_ns
        or _canonical_xattrs(level) != level_xattrs
    ):
        raise CampaignPlanError("target WIP root changed during capture")
    return WipRollbackState(
        level=level,
        baseline_snapshot=baseline,
        existed=True,
        level_identity=(metadata.st_dev, metadata.st_ino),
        level_mode=metadata.st_mode,
        level_uid=metadata.st_uid,
        level_gid=metadata.st_gid,
        level_xattrs=level_xattrs,
        level_atime_ns=metadata.st_atime_ns,
        level_mtime_ns=metadata.st_mtime_ns,
        level_ctime_ns=metadata.st_ctime_ns,
        entries=entries,
        latest_bytes=latest_bytes,
        absence_custody=None,
    )


def _same_wip_entry(
    before: tuple[Any, ...], after: tuple[Any, ...]
) -> bool:
    # Reads performed by the watchdog may update atime.  Every authoritative
    # property (type, identity, mode, size, mtime, ctime, and bytes) remains
    # exact; atime is restored for latest.json but is not an integrity signal.
    if before[0] == after[0] == "file":
        return before[:-1] == after[:-1]
    return before == after


def _restore_latest_in_place(
    path: Path,
    state: WipRollbackState,
) -> None:
    baseline = state.entries.get("latest.json")
    if baseline is None or baseline[0] != "file" or state.latest_bytes is None:
        raise CampaignPlanError("WIP rollback lacks the prior latest.json receipt")
    flags = os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise CampaignPlanError("WIP latest.json is unavailable for rollback") from exc
    try:
        current = os.fstat(descriptor)
        if (
            not stat.S_ISREG(current.st_mode)
            or current.st_nlink != 1
            or (current.st_dev, current.st_ino) != (baseline[1], baseline[2])
        ):
            raise CampaignPlanError(
                "WIP latest.json changed identity before rollback"
            )
        os.lseek(descriptor, 0, os.SEEK_SET)
        os.ftruncate(descriptor, 0)
        payload = state.latest_bytes
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        if current.st_uid != baseline[9] or current.st_gid != baseline[10]:
            os.fchown(descriptor, baseline[9], baseline[10])
        os.fchmod(descriptor, stat.S_IMODE(baseline[3]))
        _restore_canonical_xattrs(path, baseline[11])
        os.fsync(descriptor)
        os.utime(descriptor, ns=(baseline[12], baseline[6]))
    except OSError as exc:
        raise CampaignPlanError("WIP latest.json rollback failed") from exc
    finally:
        os.close(descriptor)


def _rollback_tainted_wip(
    item: dict[str, Any],
    state: WipRollbackState,
    record: dict[str, Any],
    transcript_sha: str,
) -> None:
    """Delete one authenticated tainted capsule and restore prior latest.json."""

    level = state.level
    if not os.path.lexists(level):
        raise CampaignPlanError("tainted child did not leave a WIP level directory")
    _reject_symlinked_ancestry(level, "target WIP rollback directory")
    current_level_identity = _host_directory_identity(
        level, "target WIP rollback directory"
    )
    if state.existed and current_level_identity != state.level_identity:
        raise CampaignPlanError("target WIP root changed identity")
    current = _wip_entry_inventory(level)
    baseline_names = set(state.entries)
    current_names = set(current)
    missing = baseline_names - current_names
    if missing:
        raise CampaignPlanError("tainted child removed preexisting WIP evidence")
    for name in sorted(baseline_names - {"latest.json"}):
        if not _same_wip_entry(state.entries[name], current[name]):
            raise CampaignPlanError(
                f"tainted child changed preexisting WIP evidence: {name}"
            )
    baseline_top = {name.split("/", 1)[0] for name in baseline_names}
    current_top = {name.split("/", 1)[0] for name in current_names}
    extras = current_top - baseline_top
    if state.latest_bytes is None:
        extras.discard("latest.json")
    if len(extras) != 1:
        raise CampaignPlanError(
            "tainted child did not create exactly one isolated WIP attempt"
        )
    attempt = next(iter(extras))
    if SAFE_COMPONENT_RE.fullmatch(attempt) is None:
        raise CampaignPlanError("tainted WIP attempt has an unsafe name")
    attempt_dir = level / attempt
    if not attempt_dir.is_dir() or attempt_dir.is_symlink():
        raise CampaignPlanError("tainted WIP attempt is not a physical directory")
    unexpected = {
        name for name in current_names - baseline_names
        if name != "latest.json"
        and name != attempt
        and not name.startswith(f"{attempt}/")
    }
    if unexpected:
        raise CampaignPlanError("tainted child created ambiguous WIP evidence")
    try:
        metadata_raw = Legs._read_single_link_regular(
            os.fspath(attempt_dir / "metadata.json")
        )
        metadata = json.loads(metadata_raw)
        latest_raw = Legs._read_single_link_regular(
            os.fspath(level / "latest.json")
        )
        latest = json.loads(latest_raw)
    except (OSError, UnicodeError, json.JSONDecodeError, Legs.WorkspaceTainted) as exc:
        raise CampaignPlanError("tainted WIP receipt is unavailable") from exc
    if not isinstance(metadata, dict) or any((
        metadata.get("attempt") != attempt,
        metadata.get("game") != item["game"],
        metadata.get("level") != item["target_level"],
        not isinstance(metadata.get("phase"), str),
        not isinstance(metadata.get("files"), list),
        record["transcript"] not in metadata.get("files", []),
        latest != {"attempt": attempt, "metadata": metadata},
    )):
        raise CampaignPlanError("tainted WIP receipt does not bind the generation")
    _sealed_file_bytes(
        attempt_dir / "files" / record["transcript"],
        transcript_sha,
        "WIP transcript",
    )
    if not getattr(shutil.rmtree, "avoids_symlink_attacks", False):
        raise CampaignPlanError("platform lacks symlink-safe recursive deletion")
    try:
        shutil.rmtree(attempt_dir)
        if state.latest_bytes is None:
            latest_path = level / "latest.json"
            latest_metadata = latest_path.stat(follow_symlinks=False)
            if (
                latest_path.is_symlink()
                or not stat.S_ISREG(latest_metadata.st_mode)
                or latest_metadata.st_nlink != 1
            ):
                raise CampaignPlanError("new WIP latest.json is unsafe")
            os.unlink(latest_path)
        else:
            _restore_latest_in_place(level / "latest.json", state)
        if state.existed:
            assert state.level_mode is not None
            assert state.level_uid is not None
            assert state.level_gid is not None
            assert state.level_xattrs is not None
            assert state.level_atime_ns is not None
            assert state.level_mtime_ns is not None
            level_metadata = level.stat(follow_symlinks=False)
            if (
                level_metadata.st_uid != state.level_uid
                or level_metadata.st_gid != state.level_gid
            ):
                os.chown(
                    level,
                    state.level_uid,
                    state.level_gid,
                    follow_symlinks=False,
                )
            os.chmod(
                level,
                stat.S_IMODE(state.level_mode),
                follow_symlinks=False,
            )
            _restore_canonical_xattrs(level, state.level_xattrs)
            os.utime(
                level,
                ns=(state.level_atime_ns, state.level_mtime_ns),
                follow_symlinks=False,
            )
            _fsync_directory(level)
        else:
            if any(level.iterdir()):
                raise CampaignPlanError(
                    "new tainted WIP level contains unexpected evidence"
                )
            if state.absence_custody is None:
                raise CampaignPlanError(
                    "tainted WIP rollback lacks absence custody"
                )
            _durably_restore_wip_absence_custody(
                level, state.absence_custody
            )
    except OSError as exc:
        raise CampaignPlanError("tainted WIP rollback failed") from exc
    if _target_wip_snapshot(item) != state.baseline_snapshot:
        raise CampaignPlanError("tainted WIP rollback did not restore the baseline")
    if state.existed:
        assert state.level_mode is not None
        assert state.level_uid is not None
        assert state.level_gid is not None
        assert state.level_xattrs is not None
        assert state.level_mtime_ns is not None
        restored_level = level.stat(follow_symlinks=False)
        if (
            (restored_level.st_dev, restored_level.st_ino)
            != state.level_identity
            or restored_level.st_mode != state.level_mode
            or restored_level.st_uid != state.level_uid
            or restored_level.st_gid != state.level_gid
            or restored_level.st_mtime_ns != state.level_mtime_ns
            or _canonical_xattrs(level) != state.level_xattrs
        ):
            raise CampaignPlanError(
                "tainted WIP rollback did not restore root metadata"
            )
    else:
        restored = _capture_wip_rollback(item)
        if _wip_logical_restore_state_sha256(restored) != (
            _wip_logical_restore_state_sha256(state)
        ):
            raise CampaignPlanError(
                "tainted WIP rollback did not restore absence custody"
            )
        baseline_latest = state.entries.get("latest.json")
        if baseline_latest is not None:
            restored_latest = _wip_entry_inventory(level).get("latest.json")
            if (
                restored_latest is None
                or restored_latest[0:7] != baseline_latest[0:7]
                or restored_latest[8:12] != baseline_latest[8:12]
            ):
                raise CampaignPlanError(
                    "tainted WIP rollback did not restore latest.json metadata"
                )


def _canonical_path_excluded(
    relative: str, excluded_prefixes: frozenset[str]
) -> bool:
    return any(
        relative == prefix or relative.startswith(f"{prefix}/")
        for prefix in excluded_prefixes
    )


def _canonical_excluded_ancestor(
    relative: str, excluded_prefixes: frozenset[str]
) -> bool:
    return any(
        prefix.startswith(f"{relative}/") for prefix in excluded_prefixes
    )


_DARWIN_XATTR_NOFOLLOW = 0x0001
_DARWIN_XATTR_LIBC: Any | None = None


def _darwin_xattr_libc() -> Any:
    """Return typed Darwin xattr entry points absent from some CPython builds."""

    global _DARWIN_XATTR_LIBC
    if _DARWIN_XATTR_LIBC is not None:
        return _DARWIN_XATTR_LIBC
    if sys.platform != "darwin":
        raise OSError(
            errno.ENOSYS,
            "this Python has no extended-attribute API",
        )
    libc = ctypes.CDLL(None, use_errno=True)
    libc.listxattr.argtypes = [
        ctypes.c_char_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
    ]
    libc.listxattr.restype = ctypes.c_ssize_t
    libc.getxattr.argtypes = [
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_uint32,
        ctypes.c_int,
    ]
    libc.getxattr.restype = ctypes.c_ssize_t
    libc.setxattr.argtypes = [
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_uint32,
        ctypes.c_int,
    ]
    libc.setxattr.restype = ctypes.c_int
    libc.removexattr.argtypes = [
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_int,
    ]
    libc.removexattr.restype = ctypes.c_int
    _DARWIN_XATTR_LIBC = libc
    return libc


def _darwin_xattr_error(operation: str, path: Path) -> OSError:
    code = ctypes.get_errno() or errno.EIO
    return OSError(code, f"{operation}: {os.strerror(code)}", os.fspath(path))


def _portable_listxattr(path: Path) -> tuple[str, ...]:
    if hasattr(os, "listxattr"):
        return tuple(os.listxattr(path, follow_symlinks=False))
    libc = _darwin_xattr_libc()
    encoded_path = os.fsencode(path)
    for _attempt in range(4):
        ctypes.set_errno(0)
        size = libc.listxattr(
            encoded_path, None, 0, _DARWIN_XATTR_NOFOLLOW
        )
        if size < 0:
            raise _darwin_xattr_error("listxattr", path)
        if size == 0:
            return ()
        buffer = ctypes.create_string_buffer(size)
        ctypes.set_errno(0)
        result = libc.listxattr(
            encoded_path, buffer, size, _DARWIN_XATTR_NOFOLLOW
        )
        if result < 0 and ctypes.get_errno() == errno.ERANGE:
            continue
        if result < 0:
            raise _darwin_xattr_error("listxattr", path)
        raw_names = bytes(buffer.raw[:result]).split(b"\0")
        return tuple(os.fsdecode(name) for name in raw_names if name)
    raise OSError(
        errno.ERANGE,
        "extended-attribute inventory changed repeatedly",
        os.fspath(path),
    )


def _portable_getxattr(path: Path, name: str) -> bytes:
    if hasattr(os, "getxattr"):
        return os.getxattr(path, name, follow_symlinks=False)
    libc = _darwin_xattr_libc()
    encoded_path = os.fsencode(path)
    encoded_name = os.fsencode(name)
    for _attempt in range(4):
        ctypes.set_errno(0)
        size = libc.getxattr(
            encoded_path,
            encoded_name,
            None,
            0,
            0,
            _DARWIN_XATTR_NOFOLLOW,
        )
        if size < 0:
            raise _darwin_xattr_error("getxattr", path)
        if size == 0:
            return b""
        buffer = ctypes.create_string_buffer(size)
        ctypes.set_errno(0)
        result = libc.getxattr(
            encoded_path,
            encoded_name,
            buffer,
            size,
            0,
            _DARWIN_XATTR_NOFOLLOW,
        )
        if result < 0 and ctypes.get_errno() == errno.ERANGE:
            continue
        if result < 0:
            raise _darwin_xattr_error("getxattr", path)
        return bytes(buffer.raw[:result])
    raise OSError(
        errno.ERANGE,
        "extended-attribute value changed repeatedly",
        os.fspath(path),
    )


def _portable_setxattr(path: Path, name: str, value: bytes) -> None:
    if hasattr(os, "setxattr"):
        os.setxattr(path, name, value, follow_symlinks=False)
        return
    libc = _darwin_xattr_libc()
    buffer = ctypes.create_string_buffer(value, len(value)) if value else None
    ctypes.set_errno(0)
    result = libc.setxattr(
        os.fsencode(path),
        os.fsencode(name),
        buffer,
        len(value),
        0,
        _DARWIN_XATTR_NOFOLLOW,
    )
    if result != 0:
        raise _darwin_xattr_error("setxattr", path)


def _portable_removexattr(path: Path, name: str) -> None:
    if hasattr(os, "removexattr"):
        os.removexattr(path, name, follow_symlinks=False)
        return
    libc = _darwin_xattr_libc()
    ctypes.set_errno(0)
    result = libc.removexattr(
        os.fsencode(path), os.fsencode(name), _DARWIN_XATTR_NOFOLLOW
    )
    if result != 0:
        raise _darwin_xattr_error("removexattr", path)


def _canonical_xattrs(path: Path) -> tuple[tuple[str, bytes], ...]:
    try:
        names = sorted(_portable_listxattr(path))
        return tuple(
            (name, _portable_getxattr(path, name))
            for name in names
        )
    except OSError as exc:
        raise CampaignPlanError(
            f"canonical extended attributes are unavailable: {path}"
        ) from exc


def _restore_canonical_xattrs(
    path: Path, expected: tuple[tuple[str, bytes], ...]
) -> None:
    expected_map = dict(expected)
    try:
        current = set(_portable_listxattr(path))
        for name in sorted(current - set(expected_map)):
            _portable_removexattr(path, name)
        for name, value in expected:
            if (
                name not in current
                or _portable_getxattr(path, name) != value
            ):
                _portable_setxattr(path, name, value)
    except OSError as exc:
        raise CampaignPlanError(
            f"canonical extended-attribute rollback failed: {path}"
        ) from exc


def _update_canonical_digest(
    digest: Any, relative: str, entry: CanonicalEntry
) -> None:
    fields: list[Any] = [
        relative,
        entry.kind,
        entry.mode,
        entry.uid,
        entry.gid,
        entry.mtime_ns,
    ]
    if entry.kind == "file":
        if entry.payload is None:
            raise CampaignPlanError("canonical file receipt lacks payload")
        fields.extend((
            len(entry.payload),
            hashlib.sha256(entry.payload).hexdigest(),
        ))
    fields.append([
        [name, hashlib.sha256(value).hexdigest()]
        for name, value in entry.xattrs
    ])
    digest.update(json.dumps(
        fields, separators=(",", ":")
    ).encode("utf-8"))


def _canonical_entries_digest(entries: dict[str, CanonicalEntry]) -> str:
    """Recompute inventory order without consulting mutable paths."""

    digest = hashlib.sha256()

    def visit(prefix: str) -> None:
        depth = 0 if not prefix else prefix.count("/") + 1
        immediate = [
            (relative, entry)
            for relative, entry in entries.items()
            if relative.count("/") == depth
            and (
                not prefix
                or relative.startswith(f"{prefix}/")
            )
        ]
        directories = sorted(
            (
                (relative, entry)
                for relative, entry in immediate
                if entry.kind == "directory"
            ),
            key=lambda pair: pair[0],
        )
        files = sorted(
            (
                (relative, entry)
                for relative, entry in immediate
                if entry.kind == "file"
            ),
            key=lambda pair: pair[0],
        )
        for relative, entry in (*directories, *files):
            _update_canonical_digest(digest, relative, entry)
        for relative, _entry in directories:
            visit(relative)

    visit("")
    return digest.hexdigest()


def _canonical_inventory(
    root: Path,
    *,
    excluded_prefixes: frozenset[str] = frozenset(),
) -> tuple[dict[str, CanonicalEntry], str]:
    """Seal canonical bytes plus every non-target WIP subtree."""

    entries: dict[str, CanonicalEntry] = {}
    digest = hashlib.sha256()
    entry_count = 0
    total_bytes = 0
    try:
        for directory_raw, dirs, files in os.walk(
            root, topdown=True, followlinks=False
        ):
            directory = Path(directory_raw)
            dirs.sort()
            files.sort()
            kept_dirs: list[str] = []
            for name in dirs:
                child = directory / name
                relative = child.relative_to(root).as_posix()
                if _canonical_path_excluded(relative, excluded_prefixes):
                    continue
                metadata = child.stat(follow_symlinks=False)
                if child.is_symlink() or not stat.S_ISDIR(metadata.st_mode):
                    raise CampaignPlanError(
                        "canonical artifact contains an unsafe directory"
                    )
                kept_dirs.append(name)
                entry = CanonicalEntry(
                    "directory",
                    metadata.st_mode,
                    metadata.st_atime_ns,
                    metadata.st_mtime_ns,
                    uid=metadata.st_uid,
                    gid=metadata.st_gid,
                    xattrs=_canonical_xattrs(child),
                )
                entries[relative] = entry
                _update_canonical_digest(digest, relative, entry)
                entry_count += 1
                total_bytes += sum(len(value) for _, value in entry.xattrs)
            dirs[:] = kept_dirs
            for name in files:
                child = directory / name
                relative = child.relative_to(root).as_posix()
                if _canonical_path_excluded(relative, excluded_prefixes):
                    continue
                metadata = child.stat(follow_symlinks=False)
                if (
                    child.is_symlink()
                    or not stat.S_ISREG(metadata.st_mode)
                    or metadata.st_nlink != 1
                ):
                    raise CampaignPlanError(
                        "canonical artifact contains an unsafe file"
                    )
                try:
                    payload = Legs._read_single_link_regular(os.fspath(child))
                except (OSError, Legs.WorkspaceTainted) as exc:
                    raise CampaignPlanError(
                        "canonical artifact file is unstable"
                    ) from exc
                entry = CanonicalEntry(
                    "file",
                    metadata.st_mode,
                    metadata.st_atime_ns,
                    metadata.st_mtime_ns,
                    payload,
                    uid=metadata.st_uid,
                    gid=metadata.st_gid,
                    xattrs=_canonical_xattrs(child),
                )
                entries[relative] = entry
                _update_canonical_digest(digest, relative, entry)
                entry_count += 1
                total_bytes += len(payload) + sum(
                    len(value) for _, value in entry.xattrs
                )
            if (
                entry_count > MAX_CANONICAL_ROLLBACK_ENTRIES
                or total_bytes > MAX_CANONICAL_ROLLBACK_BYTES
            ):
                raise CampaignPlanError(
                    "canonical artifact exceeds the rollback receipt bound"
                )
    except OSError as exc:
        raise CampaignPlanError("canonical artifact inventory is unstable") from exc
    return entries, digest.hexdigest()


def _capture_canonical_rollback(
    item: dict[str, Any],
) -> CanonicalRollbackState:
    root = _artifact_root(item) / f"{item['game']}_legs"
    excluded_prefixes = frozenset({
        f"wip_context/level_{int(item['target_level']):02d}"
    })
    _reject_symlinked_ancestry(root, "canonical rollback root")
    metadata = root.stat(follow_symlinks=False)
    if root.is_symlink() or not stat.S_ISDIR(metadata.st_mode):
        raise CampaignPlanError("canonical rollback root is unsafe")
    root_xattrs = _canonical_xattrs(root)
    entries, digest = _canonical_inventory(
        root, excluded_prefixes=excluded_prefixes
    )
    after = root.stat(follow_symlinks=False)
    if (
        (after.st_dev, after.st_ino) != (metadata.st_dev, metadata.st_ino)
        or after.st_mtime_ns != metadata.st_mtime_ns
        or _canonical_xattrs(root) != root_xattrs
    ):
        raise CampaignPlanError(
            "canonical artifact changed during rollback capture"
        )
    check_entries, check_digest = _canonical_inventory(
        root, excluded_prefixes=excluded_prefixes
    )
    if check_digest != digest or set(check_entries) != set(entries):
        raise CampaignPlanError(
            "canonical artifact changed during rollback capture"
        )
    return CanonicalRollbackState(
        root=root,
        root_identity=(metadata.st_dev, metadata.st_ino),
        root_mode=metadata.st_mode,
        root_uid=metadata.st_uid,
        root_gid=metadata.st_gid,
        root_xattrs=root_xattrs,
        root_atime_ns=metadata.st_atime_ns,
        root_mtime_ns=metadata.st_mtime_ns,
        entries=entries,
        digest=digest,
        excluded_prefixes=excluded_prefixes,
    )


def _canonical_tree_names(
    root: Path, *, excluded_prefixes: frozenset[str]
) -> set[str]:
    names: set[str] = set()
    try:
        for directory_raw, dirs, files in os.walk(
            root, topdown=True, followlinks=False
        ):
            directory = Path(directory_raw)
            dirs.sort()
            files.sort()
            for name in tuple(dirs):
                child = directory / name
                relative = child.relative_to(root).as_posix()
                if _canonical_path_excluded(relative, excluded_prefixes):
                    dirs.remove(name)
                    continue
                excluded_ancestor = _canonical_excluded_ancestor(
                    relative, excluded_prefixes
                )
                excluded_descendant_present = excluded_ancestor and any(
                    os.path.lexists(root / prefix)
                    for prefix in excluded_prefixes
                    if prefix.startswith(f"{relative}/")
                )
                # Preserve an excluded target subtree during ambiguous
                # recovery.  Once exact WIP recovery has removed it, include
                # a newly created ancestor in the removal inventory.
                if not excluded_descendant_present:
                    names.add(relative)
                if child.is_symlink():
                    dirs.remove(name)
            for name in files:
                relative = (directory / name).relative_to(root).as_posix()
                if not _canonical_path_excluded(relative, excluded_prefixes):
                    names.add(relative)
    except OSError as exc:
        raise CampaignPlanError("canonical artifact tree is unstable") from exc
    return names


def _remove_canonical_node(path: Path) -> None:
    try:
        metadata = path.stat(follow_symlinks=False)
        if stat.S_ISDIR(metadata.st_mode) and not path.is_symlink():
            if not getattr(shutil.rmtree, "avoids_symlink_attacks", False):
                raise CampaignPlanError(
                    "platform lacks symlink-safe recursive deletion"
                )
            shutil.rmtree(path)
        else:
            os.unlink(path)
    except OSError as exc:
        raise CampaignPlanError("canonical rollback could not remove a node") from exc


def _restore_canonical_file(path: Path, entry: CanonicalEntry) -> None:
    if entry.kind != "file" or entry.payload is None:
        raise CampaignPlanError("canonical file rollback receipt is malformed")
    if os.path.lexists(path):
        metadata = path.stat(follow_symlinks=False)
        if (
            path.is_symlink()
            or not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
        ):
            _remove_canonical_node(path)
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_TRUNC
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(path, flags, stat.S_IMODE(entry.mode))
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1:
            raise CampaignPlanError("canonical rollback file is aliased")
        offset = 0
        while offset < len(entry.payload):
            offset += os.write(descriptor, entry.payload[offset:])
        if opened.st_uid != entry.uid or opened.st_gid != entry.gid:
            os.fchown(descriptor, entry.uid, entry.gid)
        os.fchmod(descriptor, stat.S_IMODE(entry.mode))
        _restore_canonical_xattrs(path, entry.xattrs)
        os.fsync(descriptor)
        os.utime(descriptor, ns=(entry.atime_ns, entry.mtime_ns))
    except OSError as exc:
        raise CampaignPlanError("canonical file rollback failed") from exc
    finally:
        try:
            os.close(descriptor)
        except (NameError, OSError):
            pass


def _canonical_matches(state: CanonicalRollbackState) -> bool:
    try:
        metadata = state.root.stat(follow_symlinks=False)
        if (
            state.root.is_symlink()
            or (metadata.st_dev, metadata.st_ino) != state.root_identity
            or metadata.st_mode != state.root_mode
            or metadata.st_uid != state.root_uid
            or metadata.st_gid != state.root_gid
            or metadata.st_mtime_ns != state.root_mtime_ns
            or _canonical_xattrs(state.root) != state.root_xattrs
        ):
            return False
        current_entries, digest = _canonical_inventory(
            state.root, excluded_prefixes=state.excluded_prefixes
        )
    except (OSError, CampaignPlanError):
        return False
    if digest == state.digest:
        return True
    # If the target subtree did not exist at baseline, preserving ambiguous
    # target evidence may necessarily leave only its newly created ancestor
    # directory visible.  Treat that wrapper as part of the exclusion iff it
    # has no other included descendants; any additional key still fails.
    extra_wrappers = {
        relative
        for relative in set(current_entries) - set(state.entries)
        if _canonical_excluded_ancestor(
            relative, state.excluded_prefixes
        )
    }
    reduced = {
        relative: entry
        for relative, entry in current_entries.items()
        if relative not in extra_wrappers
    }
    if set(reduced) != set(state.entries):
        return False
    for relative, expected in state.entries.items():
        observed = reduced[relative]
        if (
            observed.kind != expected.kind
            or observed.mode != expected.mode
            or observed.mtime_ns != expected.mtime_ns
            or observed.payload != expected.payload
            or observed.uid != expected.uid
            or observed.gid != expected.gid
            or observed.xattrs != expected.xattrs
        ):
            return False
    return True


def _rollback_tainted_canonical(state: CanonicalRollbackState) -> None:
    """Restore every non-WIP canonical byte after a raced tainted promotion."""

    root = state.root
    if _host_directory_identity(root, "canonical rollback root") != state.root_identity:
        raise CampaignPlanError("canonical rollback root changed identity")
    baseline_names = set(state.entries)
    current_names = _canonical_tree_names(
        root, excluded_prefixes=state.excluded_prefixes
    )
    for relative in sorted(
        current_names - baseline_names,
        key=lambda value: (value.count("/"), value),
        reverse=True,
    ):
        path = root / relative
        if os.path.lexists(path):
            _remove_canonical_node(path)
    directories = sorted(
        (
            (relative, entry)
            for relative, entry in state.entries.items()
            if entry.kind == "directory"
        ),
        key=lambda value: (value[0].count("/"), value[0]),
    )
    for relative, entry in directories:
        path = root / relative
        if os.path.lexists(path):
            metadata = path.stat(follow_symlinks=False)
            if path.is_symlink() or not stat.S_ISDIR(metadata.st_mode):
                _remove_canonical_node(path)
        if not os.path.lexists(path):
            path.mkdir()
        os.chmod(path, stat.S_IMODE(entry.mode), follow_symlinks=False)
    for relative, entry in sorted(state.entries.items()):
        if entry.kind == "file":
            _restore_canonical_file(root / relative, entry)
    for relative, entry in reversed(directories):
        path = root / relative
        metadata = path.stat(follow_symlinks=False)
        if metadata.st_uid != entry.uid or metadata.st_gid != entry.gid:
            os.chown(
                path, entry.uid, entry.gid, follow_symlinks=False
            )
        os.chmod(path, stat.S_IMODE(entry.mode), follow_symlinks=False)
        _restore_canonical_xattrs(path, entry.xattrs)
        os.utime(
            path,
            ns=(entry.atime_ns, entry.mtime_ns),
            follow_symlinks=False,
        )
        _fsync_directory(path)
    root_metadata = root.stat(follow_symlinks=False)
    if (
        root_metadata.st_uid != state.root_uid
        or root_metadata.st_gid != state.root_gid
    ):
        os.chown(
            root,
            state.root_uid,
            state.root_gid,
            follow_symlinks=False,
        )
    os.chmod(root, stat.S_IMODE(state.root_mode), follow_symlinks=False)
    _restore_canonical_xattrs(root, state.root_xattrs)
    os.utime(
        root,
        ns=(state.root_atime_ns, state.root_mtime_ns),
        follow_symlinks=False,
    )
    _fsync_directory(root)
    if not _canonical_matches(state):
        raise CampaignPlanError(
            "tainted canonical rollback did not restore the baseline"
        )


def _canonical_frontier_binding(item: dict[str, Any]) -> dict[str, Any]:
    artifact = _artifact_root(item) / f"{item['game']}_legs"
    try:
        return Status.exact_frontier_binding(
            artifact,
            game=item["game"],
            target_level=item["target_level"],
        )
    except (OSError, TypeError, ValueError) as exc:
        raise CampaignPlanError(
            "could not rederive the canonical frontier after taint"
        ) from exc


def _workspace_lock_is_active(workspace: Path) -> bool:
    for path in (
        Legs._workspace_lock_path(os.fspath(workspace)),
        workspace / ".orchestrate.lock",
    ):
        if not path.is_file():
            continue
        try:
            lock = Legs._open_unaliased_lock(os.fspath(path), create=False)
        except RuntimeError as exc:
            raise CampaignPlanError(f"unsafe workspace lock path: {path}") from exc
        try:
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return True
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        finally:
            lock.close()
    return False


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(
        os, "O_NOFOLLOW", 0
    )
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _exact_tainted_generation(
    item: dict[str, Any],
    record: dict[str, Any],
    *,
    require_taint: bool = True,
) -> tuple[Path, Path, str, str, str | None, bool]:
    """Authenticate and independently rescan one exact generation pair."""

    _revalidate_historical_control(item)
    argv = item["argv"]
    tag = _single_cli_value(argv, "--tag")
    if (
        tag is None
        or SAFE_COMPONENT_RE.fullmatch(tag) is None
        or not tag
    ):
        raise CampaignPlanError("taint recovery requires one safe campaign tag")
    workspace_name = _safe_component(record.get("workspace"), "workspace")
    expected_prefix = f"gkm_legs_ws_{item['game']}_{tag}_"
    if not workspace_name.startswith(expected_prefix) or workspace_name == expected_prefix:
        raise CampaignPlanError("Codex exec workspace does not match dispatch tag")
    transcript_name = _safe_component(record.get("transcript"), "transcript")
    if not transcript_name.startswith("codex_turn_") or not (
        transcript_name.endswith(".jsonl")
    ):
        raise CampaignPlanError("Codex exec transcript name is not canonical")
    if record.get("protected_transcript_status") != "sealed":
        raise CampaignPlanError("Codex exec transcript is not sealed")
    transcript_sha = _safe_sha256(
        record.get("protected_transcript_sha256"), "transcript"
    )
    evidence_schema = _evidence_schema(item)
    diagnostics_name: str | None = None
    diagnostics_sha: str | None = None
    if evidence_schema == "sealed_transcript_diagnostics_v1":
        diagnostics_name = _safe_component(
            record.get("diagnostics"), "diagnostics"
        )
        if not diagnostics_name.startswith("codex_turn_") or not (
            diagnostics_name.endswith(".stderr.log")
        ):
            raise CampaignPlanError(
                "Codex exec diagnostics name is not canonical"
            )
        if record.get("protected_diagnostics_status") != "sealed":
            raise CampaignPlanError("Codex exec diagnostics are not sealed")
        diagnostics_sha = _safe_sha256(
            record.get("protected_diagnostics_sha256"), "diagnostics"
        )
    else:
        forbidden_diagnostics_fields = {
            "diagnostics",
            "protected_diagnostics_status",
            "protected_diagnostics_sha256",
        }
        if forbidden_diagnostics_fields & set(record):
            raise CampaignPlanError(
                "transcript-only runner emitted ambiguous diagnostics fields"
            )

    scratch = Path(Legs.SCRATCH).absolute()
    _host_directory_identity(scratch, "scratch root")
    protected_root = scratch / ".proposer_transcripts"
    _host_directory_identity(protected_root, "protected transcript root")
    workspace = scratch / workspace_name
    protected = protected_root / workspace_name
    if workspace.parent != scratch or protected.parent != protected_root:
        raise CampaignPlanError("tainted generation path escaped scratch")
    _host_directory_identity(workspace, "tainted workspace")
    _host_directory_identity(protected, "paired transcript directory")
    inventory = sorted(entry.name for entry in protected.iterdir())
    expected_inventory = [transcript_name]
    if diagnostics_name is not None:
        expected_inventory.append(diagnostics_name)
    if inventory != sorted(expected_inventory):
        raise CampaignPlanError("protected evidence directory has an ambiguous inventory")
    _sealed_file_bytes(protected / transcript_name, transcript_sha, "transcript")
    if diagnostics_name is not None and diagnostics_sha is not None:
        _sealed_file_bytes(
            protected / diagnostics_name, diagnostics_sha, "diagnostics"
        )
    try:
        historical = evidence_schema == "sealed_transcript_only_v1"
        module_root = (
            Path(item["historical_runner"]["worktree"])
            / "arc"
            / "crack_lab"
            if historical
            else HERE
        )
        trusted = _historical_tester_scaffolds(item, workspace)
        boundary_findings = (
            *Boundary.scan_workspace(
                workspace,
                arena_module_root=module_root,
                trusted_host_scaffolds=trusted,
            ),
            *Boundary.scan_codex_transcript(
                protected / transcript_name,
                workspace_root=workspace,
                arena_module_root=module_root,
                accepted_workspace_root=os.fspath(workspace),
                allow_historical_transport_banner=historical,
            ),
        )
        boundary_findings = Legs._filter_trusted_scaffold_root_literal(
            workspace, boundary_findings, trusted=trusted
        )
        if evidence_schema == "sealed_transcript_only_v1":
            reason = (
                boundary_findings[0].describe()
                if boundary_findings
                else None
            )
        else:
            reason = Legs._workspace_or_protected_taint_reason(
                os.fspath(workspace)
            )
        descendant_unproven = any(
            finding.code in UNQUIESCED_BOUNDARY_CODES
            for finding in boundary_findings
        )
    except Exception as exc:
        raise CampaignPlanError("exact generation taint rescan failed") from exc
    if require_taint and (not isinstance(reason, str) or not reason):
        raise CampaignPlanError(
            "failed/tainted child has no independently confirmed generation taint"
        )
    return (
        workspace,
        protected,
        reason if isinstance(reason, str) else "",
        transcript_sha,
        diagnostics_sha,
        descendant_unproven,
    )


def _zero_ledger_suffix_is_exact(
    state: LedgerPrefixState,
) -> bool:
    records = _ledger_records_after_prefix(state)
    return len(records) == len(state.records)


def _seal_zero_ledger_observation(
    item: dict[str, Any], observed: GuardedChildResult
) -> tuple[dict[str, Any], Path, Path]:
    """Seal a complete quiesced generation that has no ``codex_exec`` row."""

    if (
        observed.process_tree_quiesced is not True
        or observed.descendant_quiescence_unproven
        or observed.workspace is None
        or observed.transcript is None
        or observed.workspace_identity is None
        or observed.protected_identity is None
    ):
        raise CampaignPlanError(
            "zero-ledger recovery lacks one complete quiesced observation"
        )
    workspace_name = _safe_component(observed.workspace, "workspace")
    transcript_name = _safe_component(observed.transcript, "transcript")
    if (
        not workspace_name.startswith(_dispatch_workspace_prefix(item))
        or workspace_name == _dispatch_workspace_prefix(item)
        or not transcript_name.startswith("codex_turn_")
        or not transcript_name.endswith(".jsonl")
    ):
        raise CampaignPlanError(
            "zero-ledger observation does not bind the dispatch generation"
        )
    scratch = Path(Legs.SCRATCH).absolute()
    protected_root = scratch / ".proposer_transcripts"
    _host_directory_identity(scratch, "scratch root")
    _host_directory_identity(protected_root, "protected transcript root")
    workspace = scratch / workspace_name
    protected = protected_root / workspace_name
    if (
        workspace.parent != scratch
        or protected.parent != protected_root
        or _host_directory_identity(workspace, "zero-ledger workspace")
        != observed.workspace_identity
        or _host_directory_identity(protected, "zero-ledger evidence")
        != observed.protected_identity
    ):
        raise CampaignPlanError(
            "zero-ledger generation identity changed after observation"
        )
    evidence_schema = _evidence_schema(item)
    diagnostics_name = transcript_name.removesuffix(
        ".jsonl"
    ) + ".stderr.log"
    expected_inventory = {transcript_name}
    if evidence_schema == "sealed_transcript_diagnostics_v1":
        expected_inventory.add(diagnostics_name)
    try:
        inventory = {entry.name for entry in protected.iterdir()}
    except OSError as exc:
        raise CampaignPlanError(
            "zero-ledger protected evidence inventory is unavailable"
        ) from exc
    if inventory != expected_inventory:
        raise CampaignPlanError(
            "zero-ledger protected evidence inventory is ambiguous"
        )
    try:
        transcript_bytes = Legs._read_single_link_regular(
            os.fspath(protected / transcript_name)
        )
        diagnostics_bytes = (
            Legs._read_single_link_regular(
                os.fspath(protected / diagnostics_name)
            )
            if evidence_schema == "sealed_transcript_diagnostics_v1"
            else None
        )
    except (OSError, Legs.WorkspaceTainted) as exc:
        raise CampaignPlanError(
            "zero-ledger protected evidence is unstable"
        ) from exc
    evidence_record: dict[str, Any] = {
        "workspace": workspace_name,
        "transcript": transcript_name,
        "protected_transcript_status": "sealed",
        "protected_transcript_sha256": hashlib.sha256(
            transcript_bytes
        ).hexdigest(),
    }
    if diagnostics_bytes is not None:
        evidence_record.update({
            "diagnostics": diagnostics_name,
            "protected_diagnostics_status": "sealed",
            "protected_diagnostics_sha256": hashlib.sha256(
                diagnostics_bytes
            ).hexdigest(),
        })
    (
        rescanned_workspace,
        rescanned_protected,
        authenticated_reason,
        transcript_sha,
        diagnostics_sha,
        descendant_unproven,
    ) = _exact_tainted_generation(
        item, evidence_record, require_taint=False
    )
    if (
        rescanned_workspace != workspace
        or rescanned_protected != protected
        or descendant_unproven
        and not observed.detached_processes_proven_absent
    ):
        raise CampaignPlanError(
            "zero-ledger terminal rescan lacks complete generation custody"
        )
    lock_schema, lock_path, lock_identity = (
        _capture_recovery_workspace_lock(item, workspace)
    )
    reason = (
        observed.taint_reason
        or authenticated_reason
        or "exact child exited before appending its Codex exec record"
    )
    marker_record: dict[str, Any] = {
        "event": "dispatch_zero_ledger_quarantined",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "exception_type": "ZeroLedgerSuffixInfrastructure",
        "reason": reason,
        "child_returncode": observed.returncode,
        "workspace": workspace_name,
        "protected": workspace_name,
        "transcript": transcript_name,
        "workspace_identity": list(observed.workspace_identity),
        "protected_identity": list(observed.protected_identity),
        "process_tree_quiesced": True,
        "descendant_quiescence_unproven": False,
        "detached_processes_proven_absent": (
            observed.detached_processes_proven_absent
        ),
        "ledger_suffix_rows": 0,
        "evidence_schema": evidence_schema,
        "protected_transcript_sha256": transcript_sha,
        "workspace_lock_schema": lock_schema,
        "workspace_lock_path": os.fspath(lock_path),
        "workspace_lock_identity": list(lock_identity),
    }
    if diagnostics_sha is not None:
        marker_record.update({
            "diagnostics": diagnostics_name,
            "protected_diagnostics_sha256": diagnostics_sha,
        })
    return marker_record, workspace, protected


def _append_taint_correction(
    ledger: Path,
    item: dict[str, Any],
    record: dict[str, Any],
    *,
    reason: str,
    transcript_sha: str,
    diagnostics_sha: str | None,
) -> None:
    correction = {
        "event": "codex_exec_classification_correction",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "classification_authority": "scheduler_exact_generation_taint_scan_v1",
        "thread_id": record.get("thread_id"),
        "transcript": record["transcript"],
        "workspace": record["workspace"],
        "game": item["game"],
        "target_level": item["target_level"],
        "failure_class": "taint",
        "failure_detail_class": (
            "host_process_introspection"
            if (
                "host process introspection" in reason.lower()
                or "host_process_introspection" in reason.lower()
            )
            else "post_proposer_workspace_taint"
        ),
        "terminal_errors": [reason],
        "solved_target": None,
        "taint_verdict": "tainted",
        "retry_increment": 0,
        "protected_transcript_sha256": transcript_sha,
        **{
            field: item[field]
            for field in (
                *Status.FRONTIER_BINDING_FIELDS,
                "reached",
                "parent_action_count",
            )
        },
    }
    if diagnostics_sha is not None:
        correction["diagnostics"] = record["diagnostics"]
        correction["protected_diagnostics_sha256"] = diagnostics_sha
    Guard.append_ledger(correction, ledger)


def _cleanup_exact_generation(
    item: dict[str, Any], workspace: Path, protected: Path
) -> None:
    if not getattr(shutil.rmtree, "avoids_symlink_attacks", False):
        raise CampaignPlanError("platform lacks symlink-safe recursive deletion")
    schema = _lock_schema(item)
    if schema == "in_workspace_v1":
        lock_path = workspace / ".orchestrate.lock"
        create_lock = False
        lock_root: Path | None = None
    else:
        lock_path = Path(Legs._workspace_lock_path(os.fspath(workspace)))
        lock_root = workspace.parent / ".workspace_locks"
        if lock_path.parent != lock_root:
            raise CampaignPlanError("exact workspace lock escaped scratch")
        _host_directory_identity(lock_root, "workspace lock root")
        create_lock = True
    try:
        cleanup_lock = Legs._open_unaliased_lock(
            os.fspath(lock_path), create=create_lock
        )
    except RuntimeError as exc:
        raise CampaignPlanError(
            f"unsafe exact workspace lock path: {lock_path}"
        ) from exc
    lock_identity: tuple[int, int] | None = None
    try:
        try:
            fcntl.flock(
                cleanup_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB
            )
        except BlockingIOError as exc:
            raise CampaignPlanError(
                "exact workspace lock became active before cleanup"
            ) from exc
        descriptor_metadata = os.fstat(cleanup_lock.fileno())
        path_metadata = lock_path.stat(follow_symlinks=False)
        lock_identity = (descriptor_metadata.st_dev, descriptor_metadata.st_ino)
        if (
            lock_path.is_symlink()
            or not stat.S_ISREG(path_metadata.st_mode)
            or path_metadata.st_nlink != 1
            or lock_identity != (path_metadata.st_dev, path_metadata.st_ino)
        ):
            raise CampaignPlanError("exact workspace lock changed before cleanup")
        identities = {
            workspace: _host_directory_identity(workspace, "tainted workspace"),
            protected: _host_directory_identity(
                protected, "paired transcript directory"
            ),
        }
        for path in (workspace, protected):
            if (
                _host_directory_identity(path, "tainted generation target")
                != identities[path]
            ):
                raise CampaignPlanError("tainted generation changed before cleanup")
            shutil.rmtree(path)
            _fsync_directory(path.parent)
        if any(os.path.lexists(path) for path in identities):
            raise CampaignPlanError("tainted generation cleanup was incomplete")
        if schema == "hashed_external_v1":
            current_lock = lock_path.stat(follow_symlinks=False)
            if (
                lock_path.is_symlink()
                or lock_identity != (current_lock.st_dev, current_lock.st_ino)
                or not stat.S_ISREG(current_lock.st_mode)
                or current_lock.st_nlink != 1
            ):
                raise CampaignPlanError(
                    "exact workspace lock changed during cleanup"
                )
            os.unlink(lock_path)
            if lock_root is None:  # pragma: no cover - schema fixes this.
                raise CampaignPlanError("exact workspace lock root is missing")
            _fsync_directory(lock_root)
            if os.path.lexists(lock_path):
                raise CampaignPlanError(
                    "exact workspace lock cleanup was incomplete"
                )
    except OSError as exc:
        raise CampaignPlanError("exact generation cleanup failed") from exc
    finally:
        cleanup_lock.close()


def _append_cleanup_completion(
    ledger: Path,
    item: dict[str, Any],
    record: dict[str, Any],
) -> None:
    completion = {
        "event": "codex_taint_cleanup_completed",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "cleanup_authority": "scheduler_exact_generation_cleanup_v1",
        "thread_id": record.get("thread_id"),
        "transcript": record["transcript"],
        "workspace": record["workspace"],
        "game": item["game"],
        "target_level": item["target_level"],
        "retry_increment": 0,
        **{
            field: item[field]
            for field in (
                *Status.FRONTIER_BINDING_FIELDS,
                "reached",
                "parent_action_count",
            )
        },
    }
    if record.get("diagnostics") is not None:
        completion["diagnostics"] = record["diagnostics"]
    Guard.append_ledger(completion, ledger)


def _assert_same_retry_coordinate(
    ledger: Path,
    item: dict[str, Any],
    record: dict[str, Any],
) -> None:
    """Prove the quarantined turn did not advance the clean-retry ladder."""

    turns = Status.joined_turns(_read_ledger_locked(ledger))
    matching = [
        turn for turn in turns
        if turn.get("thread_id") == record.get("thread_id")
        and turn.get("transcript") == record.get("transcript")
    ]
    if len(matching) != 1 or any((
        matching[0].get("failure_class") != "taint",
        matching[0].get("taint_verdict") != "tainted",
        matching[0].get("solved_target") is not None,
        matching[0].get("retry_increment") != 0,
    )):
        raise CampaignPlanError("taint correction did not reduce to noncounting")
    frontier = {
        "game": item["game"],
        "next_level": item["target_level"],
        "target_level": item["target_level"],
        "reached": item["reached"],
        "parent_action_count": item["parent_action_count"],
        "priority_score": 0.0,
        **{
            field: item[field]
            for field in Status.FRONTIER_BINDING_FIELDS
        },
    }
    ranked = Status.ranked_frontiers([frontier], turns)
    if (
        len(ranked) != 1
        or ranked[0].get("retry_complexity_n")
        != item.get("retry_complexity_n")
    ):
        raise CampaignPlanError(
            "tainted generation changed the exact-frontier retry coordinate"
        )


def _authenticate_clean_generation(
    item: dict[str, Any],
    *,
    ledger: Path,
    ledger_before: list[dict[str, Any]] | LedgerPrefixState,
    observed: GuardedChildResult,
) -> dict[str, Any]:
    """Bind a nominally clean exit to one sealed, current-policy generation."""

    if (
        observed.process_tree_quiesced is not True
        or observed.descendant_quiescence_unproven
    ):
        raise UnquiescedChildError(
            "clean generation authentication lacks a scoped process-tree "
            "quiescence proof; preserving quarantine"
        )
    record = _expected_exec_record(
        item,
        ledger_before,
        None
        if isinstance(ledger_before, LedgerPrefixState)
        else _read_ledger_locked(ledger),
        clean_terminal=True,
    )
    _safe_component(record.get("thread_id"), "thread id")
    if (
        observed.workspace is None
        or observed.transcript is None
        or record.get("workspace") != observed.workspace
        or record.get("transcript") != observed.transcript
    ):
        raise CampaignPlanError(
            "sealed clean ledger generation differs from the live watchdog"
        )
    (
        workspace,
        protected,
        reason,
        _transcript_sha,
        _diagnostics_sha,
        descendant_unproven,
    ) = _exact_tainted_generation(item, record, require_taint=False)
    if (
        observed.workspace_identity is None
        or observed.protected_identity is None
        or _host_directory_identity(workspace, "clean live workspace")
        != observed.workspace_identity
        or _host_directory_identity(protected, "clean protected evidence")
        != observed.protected_identity
    ):
        raise CampaignPlanError(
            "sealed clean generation directory identity changed after observation"
        )
    if (
        descendant_unproven
        and not observed.detached_processes_proven_absent
    ):
        raise UnquiescedChildError(
            "terminal generation scan found a detaching/process capability; "
            "preserving quarantine"
        )
    if reason:
        raise ConfirmedGenerationTaint(
            "nominally clean child contains independently confirmed taint: "
            f"{reason}"
        )
    if _workspace_lock_is_active(workspace):
        raise CampaignPlanError(
            "nominally clean generation remains active after child exit"
        )
    if (
        record.get("failure_class") == "taint"
        or record.get("public_action_protocol_violation") is True
    ):
        raise CampaignPlanError(
            "nominally clean child ledger classifies the generation as tainted"
        )
    return record


def _recover_confirmed_taint(
    item: dict[str, Any],
    *,
    ledger: Path,
    ledger_before: list[dict[str, Any]] | LedgerPrefixState,
    reached_before: int,
    wip_rollback_before: WipRollbackState,
    child_returncode: int,
    canonical_rollback_before: CanonicalRollbackState | None = None,
    observed_workspace: str | None = None,
    observed_transcript: str | None = None,
    observed_workspace_identity: tuple[int, int] | None = None,
    observed_protected_identity: tuple[int, int] | None = None,
    clean_terminal_suffix: bool | None = None,
    process_tree_quiesced: bool = False,
    detached_processes_proven_absent: bool = False,
) -> dict[str, Any]:
    """Quarantine one exact tainted generation and keep its frontier retryable."""

    if process_tree_quiesced is not True:
        raise UnquiescedChildError(
            "tainted generation recovery lacks a scoped process-tree "
            "quiescence proof; preserving quarantine"
        )
    record = _expected_exec_record(
        item,
        ledger_before,
        None
        if isinstance(ledger_before, LedgerPrefixState)
        else _read_ledger_locked(ledger),
        clean_terminal=clean_terminal_suffix,
    )
    _safe_component(record.get("thread_id"), "thread id")
    if (
        observed_workspace is not None
        and record.get("workspace") != observed_workspace
    ) or (
        observed_transcript is not None
        and record.get("transcript") != observed_transcript
    ):
        raise CampaignPlanError(
            "sealed tainted ledger generation differs from the live watchdog"
        )
    (
        workspace,
        protected,
        reason,
        transcript_sha,
        diagnostics_sha,
        descendant_unproven,
    ) = _exact_tainted_generation(item, record)
    if (
        descendant_unproven
        and not detached_processes_proven_absent
    ):
        raise UnquiescedChildError(
            "exact taint scan found a detaching/process capability; "
            "preserving quarantine"
        )
    if (
        observed_workspace_identity is not None
        and _host_directory_identity(workspace, "tainted live workspace")
        != observed_workspace_identity
    ) or (
        observed_protected_identity is not None
        and _host_directory_identity(protected, "tainted protected evidence")
        != observed_protected_identity
    ):
        raise CampaignPlanError(
            "sealed tainted generation directory identity changed after observation"
        )
    if _workspace_lock_is_active(workspace):
        raise CampaignPlanError(
            "refusing cleanup while the exact tainted workspace remains active"
        )
    lineage_lock = _acquire_scheduler_lineage_lock(item)
    try:
        return _complete_taint_recovery_locked(
            item,
            ledger=ledger,
            record=record,
            workspace=workspace,
            protected=protected,
            reason=reason,
            transcript_sha=transcript_sha,
            diagnostics_sha=diagnostics_sha,
            reached_before=reached_before,
            wip_rollback_before=wip_rollback_before,
            child_returncode=child_returncode,
            canonical_rollback_before=canonical_rollback_before,
        )
    finally:
        _release_scheduler_artifact_lock(lineage_lock)


def _rollback_control_failure_canonical(
    item: dict[str, Any],
    *,
    state: CanonicalRollbackState,
    reached_before: int,
) -> None:
    """Restore the sealed canonical baseline after a quiesced gate failure.

    A terminal watchdog/control exception can occur after the historical child
    has promoted but before a generation can be authenticated well enough for
    exact WIP/evidence cleanup.  Preserve that ambiguous evidence, but never
    leave its canonical side effects installed.  This helper is deliberately
    forbidden for an unquiesced SIGKILL path.
    """

    lineage_lock = _acquire_scheduler_lineage_lock(item)
    try:
        if not _canonical_matches(state):
            _rollback_tainted_canonical(state)
        if (
            _checkpoint_reached(item["game"]) != reached_before
            or not _canonical_matches(state)
        ):
            raise CampaignPlanError(
                "terminal control failure did not restore the canonical baseline"
            )
    finally:
        _release_scheduler_artifact_lock(lineage_lock)


def _complete_taint_recovery_locked(
    item: dict[str, Any],
    *,
    ledger: Path,
    record: dict[str, Any],
    workspace: Path,
    protected: Path,
    reason: str,
    transcript_sha: str,
    diagnostics_sha: str | None,
    reached_before: int,
    wip_rollback_before: WipRollbackState,
    child_returncode: int,
    canonical_rollback_before: CanonicalRollbackState | None,
) -> dict[str, Any]:
    """Mutate recovery state only while the canonical lineage lock is held."""

    _append_taint_correction(
        ledger,
        item,
        record,
        reason=reason,
        transcript_sha=transcript_sha,
        diagnostics_sha=diagnostics_sha,
    )
    if _target_wip_snapshot(item) != wip_rollback_before.baseline_snapshot:
        _rollback_tainted_wip(
            item,
            wip_rollback_before,
            record,
            transcript_sha,
        )
    if (
        canonical_rollback_before is not None
        and not _canonical_matches(canonical_rollback_before)
    ):
        _rollback_tainted_canonical(canonical_rollback_before)
    if _checkpoint_reached(item["game"]) != reached_before:
        raise CampaignPlanError(
            "tainted child changed the canonical checkpoint"
        )
    live_binding = _canonical_frontier_binding(item)
    expected_binding = Status.validate_frontier_binding({
        field: item[field]
        for field in (
            *Status.FRONTIER_BINDING_FIELDS,
            "game",
            "reached",
            "target_level",
            "parent_action_count",
        )
    })
    if live_binding != expected_binding:
        raise CampaignPlanError(
            "tainted child changed the canonical exact frontier"
        )

    _cleanup_exact_generation(item, workspace, protected)
    _taint_gate()
    if (
        _checkpoint_reached(item["game"]) != reached_before
        or _canonical_frontier_binding(item) != expected_binding
        or _target_wip_snapshot(item)
        != wip_rollback_before.baseline_snapshot
        or (
            canonical_rollback_before is not None
            and not _canonical_matches(canonical_rollback_before)
        )
    ):
        raise CampaignPlanError(
            "canonical frontier changed during tainted generation cleanup"
        )
    _assert_same_retry_coordinate(ledger, item, record)
    # Completion is the final durable commit marker.  Any failure above leaves
    # the correction pending so the next dispatch remains fail-closed.
    _append_cleanup_completion(ledger, item, record)
    return {
        "game": item["game"],
        "target_level": item["target_level"],
        "reached": reached_before,
        "result": "tainted_noncounting",
        "reason": reason,
        "child_returncode": child_returncode,
        "retry_complexity_n": item["retry_complexity_n"],
        "seed_mode": item["seed_mode"],
        "wip_mode": item["wip_mode"],
        "lineage_input_mode": item["lineage_input_mode"],
    }


def _effective_retry_inputs(
    item: dict[str, Any], policy: dict[str, Any]
) -> tuple[str, str]:
    """Return the only admissible effective WIP/dispatch projection.

    Clean infrastructure interruption is not a solver retry.  It may override
    a policy reset only when the plan pins the sealed same-frontier capsule and
    advertises the matching infrastructure phase.  Effort, allocation, retry
    coordinate, and auxiliary policy remain those of the versioned retry row.
    """
    recovery = item.get("warm_wip_recovery_required") is True
    if not recovery:
        if (
            policy.get("wip_mode") == "restore_clean_same_frontier"
            and item.get("warm_wip_available") is not True
            and item.get("warm_wip_validation")
            == Policy.BOUNDARY_POLICY_WIP_REJECTION
        ):
            return "exclude", "filesystem_boundary_clean_reset"
        return str(policy["wip_mode"]), str(policy["dispatch_mode"])
    attempt = item.get("expected_wip_attempt")
    phase = item.get("warm_wip_phase")
    if (
        item.get("warm_wip_available") is not True
        or not isinstance(attempt, str)
        or not attempt
        or Path(attempt).name != attempt
        or phase not in Status.INFRASTRUCTURE_WIP_PHASES
    ):
        raise CampaignPlanError(
            "infrastructure recovery lacks one sealed exact-frontier capsule"
        )
    return (
        "restore_clean_same_frontier",
        "recover_clean_infrastructure_wip",
    )


def _authoritative_targets() -> dict[str, int]:
    targets = Status._authoritative_inventory()
    if len(targets) != 25 or sum(targets.values()) != 183:
        raise CampaignPlanError(
            "authoritative inventory gate failed: expected 25 games / "
            f"183 levels, found {len(targets)} / {sum(targets.values())}"
        )
    return targets


def validate_inventory_item(
    item: dict[str, Any], targets: dict[str, int], reached: int
) -> None:
    game = item.get("game")
    target = item.get("target_level")
    if (
        not isinstance(game, str)
        or not game
        or not isinstance(target, int)
        or isinstance(target, bool)
        or target <= 0
        or not isinstance(reached, int)
        or isinstance(reached, bool)
        or reached < 0
    ):
        raise CampaignPlanError("plan item has invalid game or target_level")
    authoritative = targets.get(game)
    if authoritative is None:
        raise CampaignPlanError(f"game is absent from authoritative inventory: {game}")
    if reached > authoritative:
        raise CampaignPlanError(
            f"checkpoint exceeds authoritative target: {game} "
            f"{reached}/{authoritative}"
        )
    if target > authoritative:
        raise CampaignPlanError(
            f"refusing nonexistent level: {game} L{target}; "
            f"authoritative target is {authoritative}"
        )
    if reached < target and target != reached + 1:
        raise CampaignPlanError(
            f"refusing nonsequential target: {game} reached={reached}, "
            f"requested L{target}"
        )
    if item.get("reached") != reached:
        raise CampaignPlanError(
            "plan item exact-parent reached value is stale"
        )
    seed_mode = item.get("seed_mode")
    expected_seed = "verified_parent" if reached > 0 else "zero_seed"
    if seed_mode != expected_seed:
        raise CampaignPlanError(
            f"lineage seed mismatch: {game} reached={reached} requires "
            f"{expected_seed}, item requested {seed_mode!r}"
        )
    wip_mode = item.get("wip_mode")
    if wip_mode not in {"exclude", "restore_clean_same_frontier"}:
        raise CampaignPlanError(f"invalid WIP mode: {wip_mode!r}")
    if (
        wip_mode == "restore_clean_same_frontier"
        and item.get("warm_wip_available") is not True
    ):
        raise CampaignPlanError(
            "WIP restore requested without a recorded clean same-frontier snapshot"
        )


def validate_item(
    item: dict[str, Any], plan: dict[str, Any] | None = None, *,
    allow_abandoned_scratch: bool = False,
) -> list[str]:
    if plan is not None:
        projected = _project_runner_receipt(
            plan,
            item,
            allow_abandoned_scratch=allow_abandoned_scratch,
        )
        if (
            projected.get("historical_runner")
            != item.get("historical_runner")
            or projected.get("argv") != item.get("argv")
        ):
            raise CampaignPlanError(
                "plan item has not consumed its plan-level runner receipt"
            )
    argv = item.get("argv")
    if not isinstance(argv, list) or not argv or not all(isinstance(x, str) for x in argv):
        raise CampaignPlanError("plan item argv must be a nonempty string list")
    _validate_runner_prefix(
        item, argv, allow_abandoned_scratch=allow_abandoned_scratch
    )
    _artifact_root(item)
    if "--proposer=codex" not in argv or "--model=gpt-5.6-sol" not in argv:
        raise CampaignPlanError("plan item must pin the isolated Codex proposer and model")
    if "--codex-allocation-policy=drain" not in argv:
        raise CampaignPlanError(
            "plan item must use the non-interrupting drain allocation policy"
        )
    if "--debrief-policy=never" not in argv:
        raise CampaignPlanError("campaign items must disable extra debrief turns")
    if "--transient-retries=0" not in argv:
        raise CampaignPlanError("budgeted campaign items must admit at most one proposal turn")
    game = item.get("game")
    target = item.get("target_level")
    if not isinstance(game, str) or not game or f"--game={game}" not in argv:
        raise CampaignPlanError("command game does not match plan item")
    if (
        not isinstance(target, int)
        or isinstance(target, bool)
        or target <= 0
        or f"--max-level={target}" not in argv
    ):
        raise CampaignPlanError("command max-level does not match plan target")
    try:
        binding = Status.validate_frontier_binding({
            field: item.get(field)
            for field in (
                *Status.FRONTIER_BINDING_FIELDS,
                "game",
                "reached",
                "target_level",
                "parent_action_count",
            )
        })
    except ValueError as exc:
        raise CampaignPlanError(
            f"plan item has invalid exact-frontier binding: {exc}"
        ) from exc
    required_binding_args = {
        f"--expected-parent-reached={binding['reached']}",
        (
            "--expected-parent-action-count="
            f"{binding['parent_action_count']}"
        ),
        (
            "--expected-parent-checkpoint-sha256="
            f"{binding['parent_checkpoint_sha256']}"
        ),
        (
            "--expected-parent-source-tree-sha256="
            f"{binding['parent_source_tree_sha256']}"
        ),
        f"--expected-frontier-sha256={binding['frontier_sha256']}",
    }
    missing_binding_args = sorted(required_binding_args - set(argv))
    if missing_binding_args:
        raise CampaignPlanError(
            "command does not consume its exact-frontier binding: "
            f"{missing_binding_args}"
        )
    seed_mode = item.get("seed_mode")
    wip_mode = item.get("wip_mode")
    if seed_mode not in {"zero_seed", "verified_parent"}:
        raise CampaignPlanError(f"invalid lineage seed mode: {seed_mode!r}")
    if wip_mode not in {"exclude", "restore_clean_same_frontier"}:
        raise CampaignPlanError(f"invalid lineage WIP mode: {wip_mode!r}")
    if f"--seed-mode={seed_mode}" not in argv:
        raise CampaignPlanError("command seed mode does not match item")
    if f"--wip-mode={wip_mode}" not in argv:
        raise CampaignPlanError("command WIP mode does not match item")
    expected_wip_attempt = item.get("expected_wip_attempt")
    expected_wip_args = [
        argument for argument in argv
        if argument.startswith("--expected-wip-attempt=")
    ]
    if wip_mode == "restore_clean_same_frontier":
        if (
            not isinstance(expected_wip_attempt, str)
            or not expected_wip_attempt
            or Path(expected_wip_attempt).name != expected_wip_attempt
            or expected_wip_args != [
                f"--expected-wip-attempt={expected_wip_attempt}"
            ]
        ):
            raise CampaignPlanError(
                "WIP restore does not pin one scheduler-selected capsule"
            )
    elif expected_wip_attempt is not None or expected_wip_args:
        raise CampaignPlanError(
            "excluded WIP item carries an unexpected capsule selector"
        )
    expected_composite = f"{seed_mode}+{wip_mode}"
    if item.get("lineage_input_mode") != expected_composite:
        raise CampaignPlanError("composite lineage input mode does not match item")
    if not any(arg.startswith("--codex-weekly-reserve=") for arg in argv):
        raise CampaignPlanError("plan item has no weekly reserve")
    if not any(arg.startswith("--codex-weekly-headroom=") for arg in argv):
        raise CampaignPlanError("plan item has no per-turn weekly headroom")
    n = item.get("retry_complexity_n")
    if not isinstance(n, int) or isinstance(n, bool) or n < 0:
        raise CampaignPlanError(
            "plan item has no valid retry_complexity_n"
        )
    policy = Status.retry_policy(n)
    effective_wip, effective_dispatch = _effective_retry_inputs(item, policy)
    expected_fields = {
        "effort": policy["effort"],
        "minutes": policy["minutes"],
        "wip_mode": effective_wip,
        "dispatch_mode": effective_dispatch,
        "recommended_auxiliary_parallelism": policy[
            "auxiliary_parallelism"
        ],
    }
    for key, expected_value in expected_fields.items():
        if item.get(key) != expected_value:
            raise CampaignPlanError(
                f"plan item {key} does not match retry policy"
            )
    if f"--codex-effort={policy['effort']}" not in argv:
        raise CampaignPlanError(
            "command effort does not match retry policy"
        )
    if f"--minutes={policy['minutes']}" not in argv:
        raise CampaignPlanError(
            "command allocation does not match retry policy"
        )
    cost_control_enabled = item.get("cost_control_enabled")
    if not isinstance(cost_control_enabled, bool):
        raise CampaignPlanError(
            "plan item has no explicit cost-control mode"
        )
    max_runs = item.get("max_campaign_runs")
    max_tokens = item.get("max_campaign_tokens")
    if (
        not isinstance(max_runs, int)
        or isinstance(max_runs, bool)
        or not isinstance(max_tokens, int)
        or isinstance(max_tokens, bool)
        or f"--codex-max-campaign-runs={max_runs}" not in argv
        or f"--codex-max-campaign-tokens={max_tokens}" not in argv
    ):
        raise CampaignPlanError(
            "command local cost caps do not match plan item"
        )
    if not cost_control_enabled and (max_runs != -1 or max_tokens != -1):
        raise CampaignPlanError(
            "unlimited item retains a local run or token cutoff"
        )
    if plan is not None:
        reserve = plan.get("reserve_percent")
        headroom = item.get("required_headroom_percent")
        if (
            not isinstance(reserve, int)
            or f"--codex-weekly-reserve={reserve}" not in argv
        ):
            raise CampaignPlanError("command reserve does not match plan reserve")
        if (
            not isinstance(headroom, int)
            or f"--codex-weekly-headroom={headroom}" not in argv
        ):
            raise CampaignPlanError("command headroom does not match item headroom")
        plan_cost_control = plan.get("cost_control_enabled")
        if (
            not isinstance(plan_cost_control, bool)
            or plan_cost_control != cost_control_enabled
        ):
            raise CampaignPlanError(
                "item cost-control mode does not match plan"
            )
    return argv


def item_is_admissible(plan: dict[str, Any], item: dict[str, Any], *,
                       now: float, allowance: Guard.WeeklyAllowance) -> tuple[bool, str]:
    if getattr(allowance, "window_name", None) == "unlimited":
        if item.get("cost_control_enabled") is not False:
            return False, "provider is unlimited but item enables cost controls"
        return True, "admissible: provider pool is unlimited"
    if item.get("cost_control_enabled") is not True:
        return False, "finite or unknown provider limit requires cost controls"
    not_before = plan.get("not_before_epoch")
    if isinstance(not_before, int) and now < not_before:
        return False, f"plan is held until weekly reset epoch {not_before}"
    reserve = plan.get("reserve_percent")
    headroom = item.get("required_headroom_percent")
    if not isinstance(reserve, int) or not isinstance(headroom, int):
        return False, "plan has no integer reserve/headroom"
    available = allowance.remaining_percent - reserve
    if allowance.remaining_percent <= reserve or available < headroom:
        return False, (
            f"only {available}% above the {reserve}% reserve; "
            f"item requires {headroom}%"
        )
    return True, "admissible"


def active_workspace_lock(game: str) -> Path | None:
    """Return an actively locked tagged workspace for ``game``, if any."""
    scratch = Path(Legs.SCRATCH).absolute()
    pattern = os.fspath(scratch / f"gkm_legs_ws_{game}*")
    for workspace in sorted(glob.glob(pattern)):
        # Check both the current protected sibling and the legacy in-workspace
        # location so a runner upgrade cannot overlap an already live turn.
        for path in (
            Legs._workspace_lock_path(workspace),
            Path(workspace) / ".orchestrate.lock",
        ):
            if not path.is_file():
                continue
            try:
                lock = Legs._open_unaliased_lock(
                    os.fspath(path), create=False
                )
            except RuntimeError as exc:
                raise CampaignPlanError(
                    f"unsafe workspace lock path: {path}"
                ) from exc
            try:
                try:
                    fcntl.flock(
                        lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB
                    )
                except BlockingIOError:
                    return path
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
            finally:
                lock.close()
    return None


def _acquire_scheduler_artifact_lock(
    item: dict[str, Any], *, name: str, purpose: str
) -> SchedulerArtifactLock:
    artifact_root = _artifact_root(item)
    lock_root = artifact_root / ".campaign_locks"
    try:
        lock_root.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise CampaignPlanError("could not create the artifact lock root") from exc
    _reject_symlinked_ancestry(lock_root, "scheduler artifact lock root")
    root_identity = _host_directory_identity(
        lock_root, "scheduler artifact lock root"
    )
    lock_path = lock_root / name
    try:
        lock = Legs._open_unaliased_lock(os.fspath(lock_path))
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (OSError, RuntimeError, BlockingIOError) as exc:
        try:
            lock.close()
        except (NameError, OSError):
            pass
        raise CampaignPlanError(
            f"another writer still owns the exact artifact {purpose}"
        ) from exc
    try:
        descriptor = os.fstat(lock.fileno())
        path_metadata = lock_path.stat(follow_symlinks=False)
        lock_identity = (descriptor.st_dev, descriptor.st_ino)
        if (
            lock_path.is_symlink()
            or not stat.S_ISREG(path_metadata.st_mode)
            or path_metadata.st_nlink != 1
            or lock_identity != (path_metadata.st_dev, path_metadata.st_ino)
            or _host_directory_identity(
                lock_root, "scheduler artifact lock root"
            ) != root_identity
        ):
            raise CampaignPlanError("scheduler artifact lock identity changed")
        lock.seek(0)
        lock.truncate()
        artifact = artifact_root / f"{item['game']}_legs"
        lock.write(
            f"pid={os.getpid()}\nartifact={artifact}\npurpose={purpose}\n"
        )
        lock.flush()
        os.fsync(lock.fileno())
    except BaseException:
        Legs._release_workspace_lock(lock)
        raise
    return SchedulerArtifactLock(
        handle=lock,
        root=lock_root,
        root_identity=root_identity,
        path=lock_path,
        lock_identity=lock_identity,
    )


def _validate_scheduler_artifact_lock(lock: SchedulerArtifactLock) -> None:
    descriptor = os.fstat(lock.handle.fileno())
    path_metadata = lock.path.stat(follow_symlinks=False)
    if (
        _host_directory_identity(lock.root, "scheduler artifact lock root")
        != lock.root_identity
        or lock.path.is_symlink()
        or not stat.S_ISREG(path_metadata.st_mode)
        or path_metadata.st_nlink != 1
        or (descriptor.st_dev, descriptor.st_ino) != lock.lock_identity
        or (path_metadata.st_dev, path_metadata.st_ino) != lock.lock_identity
    ):
        raise CampaignPlanError("scheduler artifact lock identity changed")


def _release_scheduler_artifact_lock(lock: SchedulerArtifactLock) -> None:
    failure: BaseException | None = None
    try:
        _validate_scheduler_artifact_lock(lock)
    except BaseException as exc:
        failure = exc
    try:
        Legs._release_workspace_lock(lock.handle)
    finally:
        if failure is not None:
            raise failure


def _acquire_scheduler_lineage_lock(
    item: dict[str, Any],
) -> SchedulerArtifactLock:
    """Serialize scheduler recovery with every canonical artifact/WIP writer."""

    return _acquire_scheduler_artifact_lock(
        item, name=f"{item['game']}.lock", purpose="lineage"
    )


def _acquire_scheduler_dispatch_lock(
    item: dict[str, Any],
) -> SchedulerArtifactLock:
    """Serialize current watchdog baselines and terminal authentication."""

    return _acquire_scheduler_artifact_lock(
        item, name=f"{item['game']}.scheduler.lock", purpose="dispatch"
    )


def _dispatch_quarantine_name(item: dict[str, Any]) -> str:
    game = item.get("game")
    if not isinstance(game, str) or SAFE_COMPONENT_RE.fullmatch(game) is None:
        raise CampaignPlanError("dispatch quarantine requires one safe game")
    return f"{game}.jsonl"


def _dispatch_release_intent_names(marker_name: str) -> tuple[str, str]:
    if Path(marker_name).name != marker_name:
        raise CampaignPlanError("dispatch release marker name is unsafe")
    return (
        f".{marker_name}.release_intent",
        f".{marker_name}.release_preparing",
    )


def _safe_release_recovery_arm_name(marker_name: str) -> str:
    if Path(marker_name).name != marker_name:
        raise CampaignPlanError("safe-release marker name is unsafe")
    return f".{marker_name}.safe_release_recovery_arm"


def _safe_release_recovery_receipt_name(
    marker_name: str, dispatch_id: str
) -> str:
    if (
        Path(marker_name).name != marker_name
        or RebootRecovery.DISPATCH_ID_RE.fullmatch(dispatch_id) is None
    ):
        raise CampaignPlanError("safe-release receipt coordinate is unsafe")
    return f".{marker_name}.{dispatch_id}.safe_release_recovery_receipt"


def _durable_recovery_record_preparing_name(final_name: str) -> str:
    """Return the one reserved staging name for a durable recovery record."""

    if (
        not isinstance(final_name, str)
        or not final_name
        or Path(final_name).name != final_name
        or SAFE_COMPONENT_RE.fullmatch(final_name) is None
    ):
        raise CampaignPlanError("durable recovery record name is unsafe")
    return f"{final_name}.preparing"


def _read_bound_release_file_at(
    root_fd: int,
    name: str,
    identity: tuple[int, int],
    *,
    label: str,
    maximum_bytes: int,
) -> bytes:
    descriptor: int | None = None
    try:
        descriptor = os.open(
            name,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=root_fd,
        )
        before = os.fstat(descriptor)
        path_before = os.stat(
            name, dir_fd=root_fd, follow_symlinks=False
        )
        if (
            not stat.S_ISREG(before.st_mode)
            or not stat.S_ISREG(path_before.st_mode)
            or before.st_nlink != 1
            or path_before.st_nlink != 1
            or before.st_uid != os.geteuid()
            or path_before.st_uid != os.geteuid()
            or before.st_uid != path_before.st_uid
            or before.st_gid != path_before.st_gid
            or stat.S_IMODE(before.st_mode) != 0o600
            or stat.S_IMODE(path_before.st_mode) != 0o600
            or (before.st_dev, before.st_ino) != identity
            or (path_before.st_dev, path_before.st_ino) != identity
            or before.st_size < 0
            or before.st_size > maximum_bytes
        ):
            raise CampaignPlanError(f"{label} has unsafe custody")
        payload = bytearray()
        offset = 0
        while offset < before.st_size:
            chunk = os.pread(
                descriptor,
                min(1024 * 1024, before.st_size - offset),
                offset,
            )
            if not chunk:
                raise CampaignPlanError(f"{label} was truncated")
            payload.extend(chunk)
            offset += len(chunk)
        after = os.fstat(descriptor)
        path_after = os.stat(
            name, dir_fd=root_fd, follow_symlinks=False
        )
        if (
            (after.st_dev, after.st_ino) != identity
            or (path_after.st_dev, path_after.st_ino) != identity
            or after.st_nlink != 1
            or path_after.st_nlink != 1
            or after.st_size != before.st_size
            or after.st_mtime_ns != before.st_mtime_ns
            or after.st_uid != before.st_uid
            or after.st_gid != before.st_gid
            or after.st_mode != before.st_mode
            or path_after.st_uid != before.st_uid
            or path_after.st_gid != before.st_gid
            or path_after.st_mode != before.st_mode
        ):
            raise CampaignPlanError(f"{label} changed during read")
        return bytes(payload)
    except FileNotFoundError:
        raise
    except CampaignPlanError:
        raise
    except OSError as exc:
        raise CampaignPlanError(f"{label} is unreadable") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _release_regular_binding_at(
    root_fd: int,
    name: str,
    identity: tuple[int, int],
    *,
    label: str,
    maximum_bytes: int,
) -> tuple[int, str]:
    payload = _read_bound_release_file_at(
        root_fd,
        name,
        identity,
        label=label,
        maximum_bytes=maximum_bytes,
    )
    return len(payload), hashlib.sha256(payload).hexdigest()


_DISPATCH_RELEASE_INTENT_KEYS = frozenset({
    "schema",
    "event",
    "dispatch_id",
    "intent_name",
    "intent_identity",
    "quarantine_root_identity",
    "marker_name",
    "marker_identity",
    "marker_bytes",
    "marker_sha256",
    "capsule_name",
    "capsule_identity",
    "capsule_present_at_intent",
    "capsule_bytes",
    "capsule_sha256",
    "release_authority",
})

_DISPATCH_RELEASE_AUTHORITY_BASE_KEYS = frozenset({
    "schema",
    "kind",
    "projected_item_sha256",
    "game",
    "target_level",
    "retry_complexity_n",
    *Status.FRONTIER_BINDING_FIELDS,
    "reached",
    "parent_action_count",
    "ledger",
    "ledger_parent_identity",
    "ledger_file_identity",
    "ledger_prefix_bytes",
    "ledger_prefix_sha256",
    "dispatch_ledger_prefix_bytes",
    "dispatch_ledger_prefix_sha256",
    "terminal_event",
    "terminal_record_sha256",
    "terminal_result",
    "terminal_result_sha256",
})
_DISPATCH_RELEASE_AUTHORITY_KEYS = (
    _DISPATCH_RELEASE_AUTHORITY_BASE_KEYS
    | frozenset({
        "release_nonce",
        "intent_core_sha256",
        "authority_record",
    })
)

_DISPATCH_RELEASE_AUTHORITY_RECORD_KEYS = frozenset({
    "event",
    "schema",
    "recorded_at",
    "dispatch_id",
    "release_nonce",
    "intent_name",
    "intent_identity",
    "intent_core_sha256",
    "projected_item_sha256",
    "game",
    "target_level",
    "retry_complexity_n",
    *Status.FRONTIER_BINDING_FIELDS,
    "reached",
    "parent_action_count",
    "terminal_kind",
    "terminal_event",
    "terminal_record_sha256",
    "ledger",
    "ledger_parent_identity",
    "ledger_file_identity",
    "ledger_prefix_bytes",
    "ledger_prefix_sha256",
})

_SAFE_RELEASE_RECOVERY_ARM_KEYS = frozenset({
    "schema",
    "event",
    "recorded_at",
    "dispatch_id",
    "recovery_nonce",
    "boot_identity_source",
    "boot_identity",
    "marker_root",
    "marker_root_identity",
    "marker_name",
    "marker_identity",
    "marker_bytes",
    "marker_sha256",
    "projected_item_sha256",
    "release_wal_name",
    "release_wal_role",
    "release_wal_identity",
    "release_wal_bytes",
    "release_wal_sha256",
    "authority_tail_bytes",
    "authority_tail_sha256",
    "release_intent",
})

_SAFE_RELEASE_RECOVERY_RECEIPT_KEYS = frozenset({
    "schema",
    "event",
    "recorded_at",
    "recovery_authority",
    "dispatch_id",
    "recovery_nonce",
    "arm_record",
    "arm_record_sha256",
    "current_boot_identity",
    "release_intent",
    "release_intent_sha256",
    "terminal_result",
    "terminal_result_sha256",
})


def _dispatch_release_intent_core_sha256(
    record: dict[str, Any], authority: dict[str, Any]
) -> str:
    base_authority = {
        key: authority[key]
        for key in _DISPATCH_RELEASE_AUTHORITY_BASE_KEYS
    }
    core = {
        "intent": {
            key: value
            for key, value in record.items()
            if key != "release_authority"
        },
        "terminal_authority": base_authority,
    }
    return hashlib.sha256(json.dumps(
        core, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")).hexdigest()


def _validate_dispatch_release_intent_record(
    record: dict[str, Any], *, marker_name: str
) -> None:
    intent_name, _preparing_name = _dispatch_release_intent_names(marker_name)
    if set(record) != _DISPATCH_RELEASE_INTENT_KEYS or any((
        record.get("schema") != DISPATCH_RELEASE_INTENT_SCHEMA,
        record.get("event") != "dispatch_release_intent",
        record.get("intent_name") != intent_name,
        record.get("marker_name") != marker_name,
        not isinstance(record.get("dispatch_id"), str),
        isinstance(record.get("dispatch_id"), str)
        and RebootRecovery.DISPATCH_ID_RE.fullmatch(
            record["dispatch_id"]
        ) is None,
    )):
        raise CampaignPlanError("dispatch release intent schema is invalid")
    for field in (
        "intent_identity", "quarantine_root_identity", "marker_identity"
    ):
        _marker_identity(record.get(field), f"release {field}")
    authority = record.get("release_authority")
    if (
        not isinstance(authority, dict)
        or set(authority) != _DISPATCH_RELEASE_AUTHORITY_KEYS
        or authority.get("schema")
        != "scheduler_dispatch_release_authority_v1"
        or authority.get("kind") not in {
            "ordinary_safe_terminal_v1",
            "post_reboot_operator_terminal_v1",
            SANDBOX_RELEASE_AUTHORITY_KIND,
        }
        or not isinstance(authority.get("ledger"), str)
        or not isinstance(authority.get("terminal_event"), str)
        or not isinstance(authority.get("terminal_result"), dict)
    ):
        raise CampaignPlanError(
            "dispatch release terminal authority is invalid"
        )
    for field in (
        "projected_item_sha256",
        "ledger_prefix_sha256",
        "dispatch_ledger_prefix_sha256",
        "terminal_record_sha256",
        "terminal_result_sha256",
    ):
        value = authority.get(field)
        if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
            raise CampaignPlanError(
                "dispatch release terminal authority hash is invalid"
            )
    _marker_identity(
        authority.get("ledger_parent_identity"), "release ledger parent"
    )
    _marker_identity(
        authority.get("ledger_file_identity"), "release ledger file"
    )
    ledger_bytes = authority.get("ledger_prefix_bytes")
    dispatch_ledger_bytes = authority.get("dispatch_ledger_prefix_bytes")
    if (
        not isinstance(ledger_bytes, int)
        or isinstance(ledger_bytes, bool)
        or ledger_bytes <= 0
        or not isinstance(dispatch_ledger_bytes, int)
        or isinstance(dispatch_ledger_bytes, bool)
        or not 0 <= dispatch_ledger_bytes < ledger_bytes
    ):
        raise CampaignPlanError(
            "dispatch release terminal ledger size is invalid"
        )
    result_payload = json.dumps(
        authority["terminal_result"],
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    if hashlib.sha256(result_payload).hexdigest() != authority.get(
        "terminal_result_sha256"
    ):
        raise CampaignPlanError(
            "dispatch release terminal result seal changed"
        )
    try:
        Status.validate_frontier_binding({
            field: authority[field]
            for field in (
                *Status.FRONTIER_BINDING_FIELDS,
                "game",
                "reached",
                "target_level",
                "parent_action_count",
            )
        })
    except (KeyError, TypeError, ValueError) as exc:
        raise CampaignPlanError(
            "dispatch release terminal frontier is invalid"
        ) from exc
    retry_n = authority.get("retry_complexity_n")
    if (
        not isinstance(retry_n, int)
        or isinstance(retry_n, bool)
        or retry_n < 0
    ):
        raise CampaignPlanError(
            "dispatch release retry coordinate is invalid"
        )
    nonce = authority.get("release_nonce")
    core_sha256 = authority.get("intent_core_sha256")
    authority_record = authority.get("authority_record")
    if (
        not isinstance(nonce, str)
        or SHA256_RE.fullmatch(nonce) is None
        or not isinstance(core_sha256, str)
        or SHA256_RE.fullmatch(core_sha256) is None
        or core_sha256
        != _dispatch_release_intent_core_sha256(record, authority)
        or not isinstance(authority_record, dict)
        or set(authority_record)
        != _DISPATCH_RELEASE_AUTHORITY_RECORD_KEYS
    ):
        raise CampaignPlanError(
            "dispatch release authorization row binding is invalid"
        )
    expected_authority_fields = {
        "event": "codex_dispatch_release_authorized",
        "schema": "scheduler_dispatch_release_authorized_v1",
        "dispatch_id": record["dispatch_id"],
        "release_nonce": nonce,
        "intent_name": record["intent_name"],
        "intent_identity": record["intent_identity"],
        "intent_core_sha256": core_sha256,
        "projected_item_sha256": authority["projected_item_sha256"],
        "game": authority["game"],
        "target_level": authority["target_level"],
        "retry_complexity_n": authority["retry_complexity_n"],
        "reached": authority["reached"],
        "parent_action_count": authority["parent_action_count"],
        "terminal_kind": authority["kind"],
        "terminal_event": authority["terminal_event"],
        "terminal_record_sha256": authority["terminal_record_sha256"],
        "ledger": authority["ledger"],
        "ledger_parent_identity": authority["ledger_parent_identity"],
        "ledger_file_identity": authority["ledger_file_identity"],
        "ledger_prefix_bytes": authority["ledger_prefix_bytes"],
        "ledger_prefix_sha256": authority["ledger_prefix_sha256"],
        **{
            field: authority[field]
            for field in Status.FRONTIER_BINDING_FIELDS
        },
    }
    if any(
        authority_record.get(field) != value
        for field, value in expected_authority_fields.items()
    ):
        raise CampaignPlanError(
            "dispatch release authorization row changed"
        )
    _recovery_recorded_at(
        authority_record, "dispatch release authorization"
    )
    for field in ("marker_bytes",):
        value = record.get(field)
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or not 0 < value <= RebootRecovery.MAX_MARKER_BYTES
        ):
            raise CampaignPlanError(
                "dispatch release intent marker size is invalid"
            )
    marker_sha = record.get("marker_sha256")
    if not isinstance(marker_sha, str) or SHA256_RE.fullmatch(marker_sha) is None:
        raise CampaignPlanError(
            "dispatch release intent marker hash is invalid"
        )
    capsule_name = record.get("capsule_name")
    capsule_identity = record.get("capsule_identity")
    capsule_present = record.get("capsule_present_at_intent")
    capsule_bytes = record.get("capsule_bytes")
    capsule_sha = record.get("capsule_sha256")
    if not isinstance(capsule_present, bool):
        raise CampaignPlanError(
            "dispatch release intent capsule disposition is invalid"
        )
    if capsule_name is None:
        if any((
            capsule_identity is not None,
            capsule_present,
            capsule_bytes is not None,
            capsule_sha is not None,
        )):
            raise CampaignPlanError(
                "dispatch release intent capsule binding is inconsistent"
            )
        return
    if (
        not isinstance(capsule_name, str)
        or not capsule_name
        or Path(capsule_name).name != capsule_name
    ):
        raise CampaignPlanError(
            "dispatch release intent capsule name is invalid"
        )
    _marker_identity(capsule_identity, "release capsule")
    if capsule_present:
        if (
            not isinstance(capsule_bytes, int)
            or isinstance(capsule_bytes, bool)
            or not 0 < capsule_bytes <= MAX_WIP_ROLLBACK_CAPSULE_BYTES
            or not isinstance(capsule_sha, str)
            or SHA256_RE.fullmatch(capsule_sha) is None
        ):
            raise CampaignPlanError(
                "dispatch release intent capsule seal is invalid"
            )
    elif capsule_bytes is not None or capsule_sha is not None:
        raise CampaignPlanError(
            "dispatch release intent retired capsule has a payload seal"
        )


def _read_dispatch_release_intent_at(
    root_fd: int,
    actual_name: str,
    *,
    marker_name: str,
) -> tuple[dict[str, Any], tuple[int, int]]:
    metadata = os.stat(
        actual_name, dir_fd=root_fd, follow_symlinks=False
    )
    identity = (metadata.st_dev, metadata.st_ino)
    payload = _read_bound_release_file_at(
        root_fd,
        actual_name,
        identity,
        label="dispatch release intent",
        maximum_bytes=MAX_DISPATCH_RELEASE_INTENT_BYTES,
    )
    try:
        rows = RebootRecovery.parse_canonical_jsonl(
            payload, label="dispatch release intent"
        )
    except RebootRecovery.RecoveryEvidenceError as exc:
        raise CampaignPlanError(str(exc)) from exc
    if len(rows) != 1:
        raise CampaignPlanError("dispatch release intent row count is invalid")
    record = dict(rows[0])
    _validate_dispatch_release_intent_record(
        record, marker_name=marker_name
    )
    if _marker_identity(
        record.get("intent_identity"), "release intent"
    ) != identity:
        raise CampaignPlanError("dispatch release intent identity changed")
    return record, identity


def _record_sha256(record: dict[str, Any]) -> str:
    return hashlib.sha256(
        RebootRecovery.canonical_json_line(record)
    ).hexdigest()


def _validate_safe_release_recovery_arm_record(
    record: dict[str, Any], *, marker_name: str
) -> None:
    if set(record) != _SAFE_RELEASE_RECOVERY_ARM_KEYS or any((
        record.get("schema") != SAFE_RELEASE_RECOVERY_ARM_SCHEMA,
        record.get("event") != SAFE_RELEASE_RECOVERY_ARM_EVENT,
        record.get("marker_name") != marker_name,
        not isinstance(record.get("dispatch_id"), str),
        isinstance(record.get("dispatch_id"), str)
        and RebootRecovery.DISPATCH_ID_RE.fullmatch(record["dispatch_id"])
        is None,
        not isinstance(record.get("recovery_nonce"), str),
        isinstance(record.get("recovery_nonce"), str)
        and RebootRecovery.DISPATCH_ID_RE.fullmatch(record["recovery_nonce"])
        is None,
    )):
        raise CampaignPlanError("safe-release recovery arm schema is invalid")
    _recovery_recorded_at(record, "safe-release recovery arm")
    try:
        RebootRecovery.validate_boot_identity(RebootRecovery.BootIdentity(
            str(record.get("boot_identity_source")),
            str(record.get("boot_identity")),
        ))
    except RebootRecovery.RecoveryEvidenceError as exc:
        raise CampaignPlanError(str(exc)) from exc
    marker_root = record.get("marker_root")
    if (
        not isinstance(marker_root, str)
        or not Path(marker_root).is_absolute()
        or Path(os.path.abspath(marker_root)) != Path(marker_root)
    ):
        raise CampaignPlanError("safe-release arm marker root is invalid")
    for field in (
        "marker_root_identity", "marker_identity", "release_wal_identity"
    ):
        _marker_identity(record.get(field), f"safe-release {field}")
    for field in ("marker_bytes", "release_wal_bytes", "authority_tail_bytes"):
        value = record.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise CampaignPlanError("safe-release arm byte count is invalid")
    if not 0 < int(record["marker_bytes"]) <= RebootRecovery.MAX_MARKER_BYTES:
        raise CampaignPlanError("safe-release arm marker size is invalid")
    if not 0 < int(record["release_wal_bytes"]) <= (
        MAX_DISPATCH_RELEASE_INTENT_BYTES
    ):
        raise CampaignPlanError("safe-release arm WAL size is invalid")
    for field in (
        "marker_sha256", "projected_item_sha256", "release_wal_sha256",
        "authority_tail_sha256",
    ):
        value = record.get(field)
        if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
            raise CampaignPlanError("safe-release arm hash is invalid")
    intent = record.get("release_intent")
    if not isinstance(intent, dict):
        raise CampaignPlanError("safe-release arm lacks its release intent")
    _validate_dispatch_release_intent_record(intent, marker_name=marker_name)
    authority = intent["release_authority"]
    assert isinstance(authority, dict)
    intent_name, preparing_name = _dispatch_release_intent_names(marker_name)
    role = record.get("release_wal_role")
    expected_name = intent_name if role == "intent" else preparing_name
    payload = RebootRecovery.canonical_json_line(intent)
    authority_line = RebootRecovery.canonical_json_line(
        authority["authority_record"]
    )
    if any((
        role not in {"intent", "preparing"},
        record.get("release_wal_name") != expected_name,
        record.get("dispatch_id") != intent.get("dispatch_id"),
        record.get("marker_root_identity")
        != intent.get("quarantine_root_identity"),
        record.get("marker_identity") != intent.get("marker_identity"),
        record.get("marker_bytes") != intent.get("marker_bytes"),
        record.get("marker_sha256") != intent.get("marker_sha256"),
        record.get("release_wal_identity") != intent.get("intent_identity"),
        record.get("release_wal_bytes") != len(payload),
        record.get("release_wal_sha256")
        != hashlib.sha256(payload).hexdigest(),
        record.get("projected_item_sha256")
        != authority.get("projected_item_sha256"),
        authority.get("kind") != "ordinary_safe_terminal_v1",
        int(record["authority_tail_bytes"]) >= len(authority_line),
    )):
        raise CampaignPlanError("safe-release recovery arm binding is invalid")


def _validate_safe_release_recovery_receipt_record(
    record: dict[str, Any], *, marker_name: str
) -> None:
    if set(record) != _SAFE_RELEASE_RECOVERY_RECEIPT_KEYS or any((
        record.get("schema") != SAFE_RELEASE_RECOVERY_RECEIPT_SCHEMA,
        record.get("event") != SAFE_RELEASE_RECOVERY_RECEIPT_EVENT,
        record.get("recovery_authority")
        != "scheduler_authenticated_safe_release_v1",
    )):
        raise CampaignPlanError(
            "safe-release recovery receipt schema is invalid"
        )
    _recovery_recorded_at(record, "safe-release recovery receipt")
    arm = record.get("arm_record")
    if not isinstance(arm, dict):
        raise CampaignPlanError("safe-release receipt lacks its arm")
    _validate_safe_release_recovery_arm_record(arm, marker_name=marker_name)
    if any((
        record.get("dispatch_id") != arm.get("dispatch_id"),
        record.get("recovery_nonce") != arm.get("recovery_nonce"),
        record.get("arm_record_sha256") != _record_sha256(arm),
    )):
        raise CampaignPlanError("safe-release receipt arm binding changed")
    current_boot = record.get("current_boot_identity")
    armed_boot = RebootRecovery.boot_identity_receipt(
        RebootRecovery.BootIdentity(
            str(arm["boot_identity_source"]), str(arm["boot_identity"])
        )
    )
    if (
        not isinstance(current_boot, dict)
        or set(current_boot) != {"source", "identity_sha256"}
        or current_boot.get("source") != armed_boot["source"]
        or not isinstance(current_boot.get("identity_sha256"), str)
        or SHA256_RE.fullmatch(current_boot["identity_sha256"]) is None
        or current_boot == armed_boot
    ):
        raise CampaignPlanError(
            "safe-release receipt lacks a changed boot identity"
        )
    intent = record.get("release_intent")
    if not isinstance(intent, dict):
        raise CampaignPlanError("safe-release receipt lacks its fresh intent")
    _validate_dispatch_release_intent_record(intent, marker_name=marker_name)
    payload = RebootRecovery.canonical_json_line(intent)
    result = record.get("terminal_result")
    if not isinstance(result, dict):
        raise CampaignPlanError("safe-release receipt result is invalid")
    result_payload = json.dumps(
        result, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    old_authority = arm["release_intent"]["release_authority"]
    new_authority = intent["release_authority"]
    stable_keys = _DISPATCH_RELEASE_AUTHORITY_BASE_KEYS - {
        "terminal_result", "terminal_result_sha256"
    }
    if any((
        record.get("release_intent_sha256")
        != hashlib.sha256(payload).hexdigest(),
        record.get("terminal_result_sha256")
        != hashlib.sha256(result_payload).hexdigest(),
        new_authority.get("terminal_result") != result,
        any(
            new_authority.get(field) != old_authority.get(field)
            for field in stable_keys
        ),
    )):
        raise CampaignPlanError(
            "safe-release receipt fresh release binding changed"
        )


def _validate_durable_recovery_root_binding(
    root_fd: int,
    root_path: Path,
    root_identity: tuple[int, int],
    *,
    label: str,
) -> None:
    """Bind an opened quarantine directory back to its canonical pathname."""

    try:
        opened = os.fstat(root_fd)
        path_opened = root_path.stat(follow_symlinks=False)
    except OSError as exc:
        raise CampaignPlanError(f"{label} root identity changed") from exc
    if (
        not stat.S_ISDIR(opened.st_mode)
        or not stat.S_ISDIR(path_opened.st_mode)
        or opened.st_uid != os.geteuid()
        or path_opened.st_uid != os.geteuid()
        or stat.S_IMODE(opened.st_mode) != 0o700
        or stat.S_IMODE(path_opened.st_mode) != 0o700
        or (opened.st_dev, opened.st_ino) != root_identity
        or (path_opened.st_dev, path_opened.st_ino) != root_identity
    ):
        raise CampaignPlanError(f"{label} root identity changed")


def _strict_durable_recovery_record_at(
    root_fd: int,
    name: str,
    *,
    root_path: Path,
    root_identity: tuple[int, int],
    label: str,
) -> tuple[dict[str, Any], tuple[int, int], bytes]:
    """Re-fsync and re-read one exact installed or staged generation."""

    descriptor: int | None = None
    try:
        _validate_durable_recovery_root_binding(
            root_fd, root_path, root_identity, label=label
        )
        descriptor = os.open(
            name,
            os.O_RDWR | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=root_fd,
        )
        before = os.fstat(descriptor)
        path_before = os.stat(
            name, dir_fd=root_fd, follow_symlinks=False
        )
        root_before = os.fstat(root_fd)
        identity = (before.st_dev, before.st_ino)
        if (
            not stat.S_ISREG(before.st_mode)
            or not stat.S_ISREG(path_before.st_mode)
            or before.st_nlink != 1
            or path_before.st_nlink != 1
            or before.st_uid != os.geteuid()
            or path_before.st_uid != os.geteuid()
            or before.st_gid != path_before.st_gid
            or stat.S_IMODE(before.st_mode) != 0o600
            or stat.S_IMODE(path_before.st_mode) != 0o600
            or identity != (path_before.st_dev, path_before.st_ino)
            or not 0 <= before.st_size <= RebootRecovery.MAX_MARKER_BYTES
        ):
            raise CampaignPlanError(f"{label} has unsafe inode custody")
        os.fsync(descriptor)
        os.fsync(root_fd)
        payload = bytearray()
        offset = 0
        while offset < before.st_size:
            chunk = os.pread(
                descriptor,
                min(1024 * 1024, before.st_size - offset),
                offset,
            )
            if not chunk:
                raise CampaignPlanError(f"{label} was truncated")
            payload.extend(chunk)
            offset += len(chunk)
        after = os.fstat(descriptor)
        path_after = os.stat(
            name, dir_fd=root_fd, follow_symlinks=False
        )
        root_after = os.fstat(root_fd)
        if (
            (
                after.st_dev,
                after.st_ino,
                after.st_nlink,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
                after.st_uid,
                after.st_gid,
                stat.S_IMODE(after.st_mode),
            )
            != (
                before.st_dev,
                before.st_ino,
                before.st_nlink,
                before.st_size,
                before.st_mtime_ns,
                before.st_ctime_ns,
                before.st_uid,
                before.st_gid,
                stat.S_IMODE(before.st_mode),
            )
            or (path_after.st_dev, path_after.st_ino) != identity
            or path_after.st_nlink != 1
            or path_after.st_uid != before.st_uid
            or path_after.st_gid != before.st_gid
            or stat.S_IMODE(path_after.st_mode) != 0o600
            or (root_after.st_dev, root_after.st_ino)
            != (root_before.st_dev, root_before.st_ino)
        ):
            raise CampaignPlanError(f"{label} changed during durable reread")
        _validate_durable_recovery_root_binding(
            root_fd, root_path, root_identity, label=label
        )
        raw = bytes(payload)
        try:
            rows = RebootRecovery.parse_canonical_jsonl(raw, label=label)
        except RebootRecovery.RecoveryEvidenceError as exc:
            raise CampaignPlanError(str(exc)) from exc
        if len(rows) != 1:
            raise CampaignPlanError(f"{label} row count is invalid")
        if RebootRecovery.canonical_json_line(rows[0]) != raw:
            raise CampaignPlanError(f"{label} is not canonical")
        return dict(rows[0]), identity, raw
    except FileNotFoundError:
        raise
    except CampaignPlanError:
        raise
    except OSError as exc:
        raise CampaignPlanError(f"could not durably revalidate {label}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _retire_malformed_durable_recovery_preparing_at(
    root_fd: int,
    name: str,
    identity: tuple[int, int],
    *,
    root_path: Path,
    root_identity: tuple[int, int],
    label: str,
) -> None:
    """Remove only the exact, single-link staging inode just inspected."""

    descriptor: int | None = None
    try:
        _validate_durable_recovery_root_binding(
            root_fd, root_path, root_identity, label=label
        )
        descriptor = os.open(
            name,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=root_fd,
        )
        opened = os.fstat(descriptor)
        path_opened = os.stat(
            name, dir_fd=root_fd, follow_symlinks=False
        )
        if (
            not stat.S_ISREG(opened.st_mode)
            or not stat.S_ISREG(path_opened.st_mode)
            or opened.st_nlink != 1
            or path_opened.st_nlink != 1
            or opened.st_uid != os.geteuid()
            or path_opened.st_uid != os.geteuid()
            or stat.S_IMODE(opened.st_mode) != 0o600
            or stat.S_IMODE(path_opened.st_mode) != 0o600
            or (opened.st_dev, opened.st_ino) != identity
            or (path_opened.st_dev, path_opened.st_ino) != identity
        ):
            raise CampaignPlanError(
                f"{label} malformed staging inode cannot be retired"
            )
        os.unlink(name, dir_fd=root_fd)
        os.fsync(root_fd)
        _validate_durable_recovery_root_binding(
            root_fd, root_path, root_identity, label=label
        )
        retired = os.fstat(descriptor)
        if (
            (retired.st_dev, retired.st_ino) != identity
            or retired.st_nlink != 0
        ):
            raise CampaignPlanError(
                f"{label} malformed staging retirement is ambiguous"
            )
        try:
            os.stat(name, dir_fd=root_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise CampaignPlanError(
                f"{label} malformed staging name was replaced"
            )
    except CampaignPlanError:
        raise
    except OSError as exc:
        raise CampaignPlanError(
            f"could not durably retire malformed {label} staging"
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _durably_promote_recovery_preparing_at(
    root_fd: int,
    final_name: str,
    preparing_name: str,
    identity: tuple[int, int],
    *,
    root_path: Path,
    root_identity: tuple[int, int],
    label: str,
) -> tuple[dict[str, Any], tuple[int, int], bytes]:
    """Atomically publish one completely sealed staging generation."""

    _validate_durable_recovery_root_binding(
        root_fd, root_path, root_identity, label=label
    )
    try:
        os.stat(final_name, dir_fd=root_fd, follow_symlinks=False)
    except FileNotFoundError:
        pass
    else:
        raise CampaignPlanError(f"{label} final name appeared before publish")
    current = os.stat(
        preparing_name, dir_fd=root_fd, follow_symlinks=False
    )
    if (current.st_dev, current.st_ino) != identity:
        raise CampaignPlanError(f"{label} staging identity changed")
    try:
        os.replace(
            preparing_name,
            final_name,
            src_dir_fd=root_fd,
            dst_dir_fd=root_fd,
        )
        installed = os.stat(
            final_name, dir_fd=root_fd, follow_symlinks=False
        )
        if (installed.st_dev, installed.st_ino) != identity:
            raise CampaignPlanError(f"{label} installed identity changed")
        try:
            os.stat(
                preparing_name, dir_fd=root_fd, follow_symlinks=False
            )
        except FileNotFoundError:
            pass
        else:
            raise CampaignPlanError(f"{label} staging name survived publish")
        os.fsync(root_fd)
        _validate_durable_recovery_root_binding(
            root_fd, root_path, root_identity, label=label
        )
    except CampaignPlanError:
        raise
    except OSError as exc:
        raise CampaignPlanError(f"could not atomically publish {label}") from exc
    record, installed_identity, payload = (
        _strict_durable_recovery_record_at(
            root_fd,
            final_name,
            root_path=root_path,
            root_identity=root_identity,
            label=label,
        )
    )
    if installed_identity != identity:
        raise CampaignPlanError(f"{label} changed after publish")
    return record, installed_identity, payload


def _read_durable_recovery_record_at(
    root_fd: int,
    name: str,
    *,
    root_path: Path,
    root_identity: tuple[int, int],
    label: str,
) -> tuple[dict[str, Any], tuple[int, int], bytes]:
    """Read a final record, or finish publishing its durable staging inode."""

    preparing_name = _durable_recovery_record_preparing_name(name)
    try:
        final = _strict_durable_recovery_record_at(
            root_fd,
            name,
            root_path=root_path,
            root_identity=root_identity,
            label=label,
        )
    except FileNotFoundError:
        final = None
    if final is not None:
        try:
            preparing = os.stat(
                preparing_name, dir_fd=root_fd, follow_symlinks=False
            )
        except FileNotFoundError:
            return final
        _retire_malformed_durable_recovery_preparing_at(
            root_fd,
            preparing_name,
            (preparing.st_dev, preparing.st_ino),
            root_path=root_path,
            root_identity=root_identity,
            label=label,
        )
        return _strict_durable_recovery_record_at(
            root_fd,
            name,
            root_path=root_path,
            root_identity=root_identity,
            label=label,
        )
    try:
        preparing = os.stat(
            preparing_name, dir_fd=root_fd, follow_symlinks=False
        )
    except FileNotFoundError:
        raise FileNotFoundError(name)
    identity = (preparing.st_dev, preparing.st_ino)
    try:
        staged_payload = _read_bound_release_file_at(
            root_fd,
            preparing_name,
            identity,
            label=f"{label} preparing",
            maximum_bytes=RebootRecovery.MAX_MARKER_BYTES,
        )
    except CampaignPlanError:
        # Custody or I/O uncertainty is not proof of a malformed staging
        # generation.  Preserve it and fail closed.
        raise
    malformed = False
    try:
        rows = RebootRecovery.parse_canonical_jsonl(
            staged_payload, label=f"{label} preparing"
        )
    except RebootRecovery.RecoveryEvidenceError:
        malformed = True
    else:
        malformed = (
            len(rows) != 1
            or RebootRecovery.canonical_json_line(rows[0]) != staged_payload
        )
    if malformed:
        _retire_malformed_durable_recovery_preparing_at(
            root_fd,
            preparing_name,
            identity,
            root_path=root_path,
            root_identity=root_identity,
            label=label,
        )
        raise FileNotFoundError(name)
    _strict_durable_recovery_record_at(
        root_fd,
        preparing_name,
        root_path=root_path,
        root_identity=root_identity,
        label=f"{label} preparing",
    )
    return _durably_promote_recovery_preparing_at(
        root_fd,
        name,
        preparing_name,
        identity,
        root_path=root_path,
        root_identity=root_identity,
        label=label,
    )


def _install_durable_recovery_record_at(
    root_fd: int,
    name: str,
    record: dict[str, Any],
    *,
    root_path: Path,
    root_identity: tuple[int, int],
    label: str,
) -> tuple[int, int]:
    payload = RebootRecovery.canonical_json_line(record)
    if len(payload) > RebootRecovery.MAX_MARKER_BYTES:
        raise CampaignPlanError(f"{label} exceeds its durable bound")
    try:
        installed, identity, installed_payload = (
            _read_durable_recovery_record_at(
                root_fd,
                name,
                root_path=root_path,
                root_identity=root_identity,
                label=label,
            )
        )
    except FileNotFoundError:
        pass
    else:
        if installed != record or installed_payload != payload:
            raise CampaignPlanError(f"{label} installed generation changed")
        return identity
    preparing_name = _durable_recovery_record_preparing_name(name)
    descriptor: int | None = None
    identity: tuple[int, int] | None = None
    try:
        descriptor = os.open(
            preparing_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=root_fd,
        )
        os.fchmod(descriptor, 0o600)
        opened = os.fstat(descriptor)
        identity = (opened.st_dev, opened.st_ino)
        path_opened = os.stat(
            preparing_name, dir_fd=root_fd, follow_symlinks=False
        )
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or opened.st_uid != os.geteuid()
            or stat.S_IMODE(opened.st_mode) != 0o600
            or identity != (path_opened.st_dev, path_opened.st_ino)
        ):
            raise CampaignPlanError(f"{label} staging custody is unsafe")
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise OSError(errno.EIO, f"short {label} write")
            offset += written
        os.fsync(descriptor)
        sealed = os.fstat(descriptor)
        path_sealed = os.stat(
            preparing_name, dir_fd=root_fd, follow_symlinks=False
        )
        if (
            (sealed.st_dev, sealed.st_ino) != identity
            or sealed.st_nlink != 1
            or sealed.st_uid != os.geteuid()
            or stat.S_IMODE(sealed.st_mode) != 0o600
            or sealed.st_size != len(payload)
            or (path_sealed.st_dev, path_sealed.st_ino) != identity
        ):
            raise CampaignPlanError(f"{label} changed while sealing")
    except CampaignPlanError:
        raise
    except OSError as exc:
        raise CampaignPlanError(f"could not stage {label}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
    assert identity is not None
    staged, staged_identity, staged_payload = (
        _strict_durable_recovery_record_at(
            root_fd,
            preparing_name,
            root_path=root_path,
            root_identity=root_identity,
            label=f"{label} preparing",
        )
    )
    if (
        staged != record
        or staged_payload != payload
        or staged_identity != identity
    ):
        raise CampaignPlanError(f"{label} staging generation changed")
    installed, installed_identity, installed_payload = (
        _durably_promote_recovery_preparing_at(
            root_fd,
            name,
            preparing_name,
            identity,
            root_path=root_path,
            root_identity=root_identity,
            label=label,
        )
    )
    if installed != record or installed_payload != payload:
        raise CampaignPlanError(f"{label} installed generation changed")
    return installed_identity


def _open_safe_release_marker(
    item: dict[str, Any],
    root: Path,
    root_fd: int,
    root_identity: tuple[int, int],
    intent: dict[str, Any],
) -> tuple[DispatchQuarantine, dict[str, Any], bytes]:
    marker_name = str(intent["marker_name"])
    marker_identity = _marker_identity(
        intent.get("marker_identity"), "safe-release marker"
    )
    marker_fd: int | None = None
    try:
        marker_fd = os.open(
            marker_name,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=root_fd,
        )
        opened = os.fstat(marker_fd)
        if (opened.st_dev, opened.st_ino) != marker_identity:
            raise CampaignPlanError("safe-release marker identity changed")
        payload = _read_bound_release_file_at(
            root_fd,
            marker_name,
            marker_identity,
            label="safe-release dispatch marker",
            maximum_bytes=RebootRecovery.MAX_MARKER_BYTES,
        )
        try:
            rows = RebootRecovery.parse_canonical_jsonl(
                payload, label="safe-release dispatch marker"
            )
        except RebootRecovery.RecoveryEvidenceError as exc:
            raise CampaignPlanError(str(exc)) from exc
        if len(rows) != 1:
            raise CampaignPlanError(
                "safe-release recovery requires the exact armed-only marker"
            )
        armed = dict(rows[0])
        marker = DispatchQuarantine(
            root=root,
            root_fd=root_fd,
            root_identity=root_identity,
            name=marker_name,
            path=root / marker_name,
            marker_fd=marker_fd,
            marker_identity=marker_identity,
            dispatch_id=str(intent["dispatch_id"]),
            schema=str(armed.get("schema")),
            capsule_name=intent.get("capsule_name"),
            capsule_identity=(
                _marker_identity(intent.get("capsule_identity"), "release capsule")
                if intent.get("capsule_name") is not None else None
            ),
            recovery_sealed_size=len(payload),
            recovery_sealed_sha256=hashlib.sha256(payload).hexdigest(),
        )
        _validate_dispatch_quarantine(marker)
        authority = intent["release_authority"]
        assert isinstance(authority, dict)
        _read_dispatch_release_marker_phase(
            root_fd, intent, authority, allow_absent=False
        )
        if (
            os.fspath(root) != os.fspath(_artifact_root(item) / ".campaign_quarantine")
            or armed.get("dispatch_id") != marker.dispatch_id
        ):
            raise CampaignPlanError("safe-release marker belongs to another lane")
        return marker, armed, payload
    except BaseException:
        if marker_fd is not None:
            os.close(marker_fd)
        raise


def _safe_release_wal_inventory(
    root_fd: int, marker_name: str
) -> tuple[str, str] | None:
    intent_name, preparing_name = _dispatch_release_intent_names(marker_name)
    present: list[tuple[str, str]] = []
    for role, name in (("intent", intent_name), ("preparing", preparing_name)):
        try:
            os.stat(name, dir_fd=root_fd, follow_symlinks=False)
        except FileNotFoundError:
            continue
        present.append((role, name))
    if len(present) > 1:
        raise CampaignPlanError("safe-release recovery found ambiguous WALs")
    return present[0] if present else None


def _validate_safe_release_arm_context(
    item: dict[str, Any],
    root_fd: int,
    root_identity: tuple[int, int],
    arm: dict[str, Any],
    *,
    require_original_wal: bool,
) -> tuple[bytes, bytes, bytes]:
    marker_name = _dispatch_quarantine_name(item)
    _validate_safe_release_recovery_arm_record(arm, marker_name=marker_name)
    item_sha = hashlib.sha256(json.dumps(
        item, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")).hexdigest()
    if any((
        arm.get("projected_item_sha256") != item_sha,
        arm.get("marker_root")
        != os.fspath(_artifact_root(item) / ".campaign_quarantine"),
        _marker_identity(arm.get("marker_root_identity"), "safe-release root")
        != root_identity,
    )):
        raise CampaignPlanError("safe-release arm belongs to another plan item")
    intent = arm["release_intent"]
    assert isinstance(intent, dict)
    authority = intent["release_authority"]
    assert isinstance(authority, dict)
    _dispatch_release_item_ledger(item, authority)
    _read_dispatch_release_marker_phase(
        root_fd, intent, authority, allow_absent=False
    )
    marker_payload = _read_bound_release_file_at(
        root_fd,
        marker_name,
        _marker_identity(arm["marker_identity"], "safe-release marker"),
        label="safe-release dispatch marker",
        maximum_bytes=RebootRecovery.MAX_MARKER_BYTES,
    )
    if any((
        len(marker_payload) != arm.get("marker_bytes"),
        hashlib.sha256(marker_payload).hexdigest() != arm.get("marker_sha256"),
    )):
        raise CampaignPlanError("safe-release arm marker seal changed")
    if require_original_wal:
        role_and_name = _safe_release_wal_inventory(root_fd, marker_name)
        if role_and_name != (
            arm.get("release_wal_role"), arm.get("release_wal_name")
        ):
            raise CampaignPlanError("safe-release arm WAL generation changed")
        wal_payload = _read_bound_release_file_at(
            root_fd,
            str(arm["release_wal_name"]),
            _marker_identity(arm["release_wal_identity"], "safe-release WAL"),
            label="safe-release WAL",
            maximum_bytes=MAX_DISPATCH_RELEASE_INTENT_BYTES,
        )
        if any((
            len(wal_payload) != arm.get("release_wal_bytes"),
            hashlib.sha256(wal_payload).hexdigest()
            != arm.get("release_wal_sha256"),
        )):
            raise CampaignPlanError("safe-release arm WAL seal changed")
    ledger = _dispatch_release_item_ledger(item, authority)
    with Guard.ledger_append_lock(ledger):
        raw, _parent, _file = _read_dispatch_release_ledger_locked(
            ledger, authority
        )
        prefix, line, tail = _dispatch_release_authority_tail(
            raw, authority, dispatch_id=str(arm["dispatch_id"])
        )
    if any((
        len(tail) != arm.get("authority_tail_bytes"),
        hashlib.sha256(tail).hexdigest() != arm.get("authority_tail_sha256"),
        tail == line,
    )):
        raise CampaignPlanError("safe-release arm authority suffix changed")
    return prefix, line, tail


def _build_dispatch_release_authority(
    item: dict[str, Any],
    marker: DispatchQuarantine,
    ledger: Path,
    terminal_result: dict[str, Any],
    *,
    kind: str,
) -> dict[str, Any]:
    if kind not in {
        "ordinary_safe_terminal_v1",
        "post_reboot_operator_terminal_v1",
        SANDBOX_RELEASE_AUTHORITY_KIND,
    }:
        raise CampaignPlanError("dispatch release authority kind is invalid")
    prefix = _capture_ledger_prefix(ledger)
    if prefix.file_identity is None or not prefix.records:
        raise CampaignPlanError(
            "dispatch release lacks a durable terminal ledger"
        )
    terminal = prefix.records[-1]
    marker_payload = _read_bound_release_file_at(
        marker.root_fd,
        marker.name,
        marker.marker_identity,
        label="dispatch quarantine marker",
        maximum_bytes=RebootRecovery.MAX_MARKER_BYTES,
    )
    try:
        marker_rows = RebootRecovery.parse_canonical_jsonl(
            marker_payload, label="dispatch release marker"
        )
    except RebootRecovery.RecoveryEvidenceError as exc:
        raise CampaignPlanError(str(exc)) from exc
    if not marker_rows or marker_rows[0].get("dispatch_id") != marker.dispatch_id:
        raise CampaignPlanError(
            "dispatch release marker binding is invalid"
        )
    armed = marker_rows[0]
    dispatch_prefix_bytes = armed.get("ledger_prefix_bytes")
    dispatch_prefix_sha256 = armed.get("ledger_prefix_sha256")
    if (
        not isinstance(dispatch_prefix_bytes, int)
        or isinstance(dispatch_prefix_bytes, bool)
        or dispatch_prefix_bytes < 0
        or dispatch_prefix_bytes > len(prefix.raw_prefix)
        or not isinstance(dispatch_prefix_sha256, str)
        or SHA256_RE.fullmatch(dispatch_prefix_sha256) is None
        or hashlib.sha256(
            prefix.raw_prefix[:dispatch_prefix_bytes]
        ).hexdigest() != dispatch_prefix_sha256
        or armed.get("projected_item_sha256")
        != hashlib.sha256(json.dumps(
            item, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")).hexdigest()
    ):
        raise CampaignPlanError(
            "dispatch release marker ledger baseline changed"
        )
    terminal_event = terminal.get("event")
    if kind == "ordinary_safe_terminal_v1":
        if terminal_result.get("result") == "tainted_noncounting":
            expected_events = {"codex_taint_cleanup_completed"}
        elif terminal_result.get("result") == "infrastructure_noncounting":
            expected_events = {ZERO_LEDGER_EVENT}
        else:
            expected_events = {"codex_level_outcome"}
    elif kind == SANDBOX_RELEASE_AUTHORITY_KIND:
        expected_events = {SANDBOX_ABANDON_EVENT}
        if any((
            terminal_result.get("result") != "sandbox_isolated_noncounting",
            terminal_result.get("dispatch_id") != marker.dispatch_id,
            terminal_result.get("process_tree_quiesced") is not False,
            terminal_result.get("detached_processes_proven_absent") is not False,
        )):
            raise CampaignPlanError(
                "sandbox-isolated terminal result is malformed"
            )
    elif kind == "post_reboot_operator_terminal_v1":
        expected_events = {
            "codex_post_reboot_operator_recovery_completed"
        }
    else:  # pragma: no cover - kind allowlist above is exhaustive.
        raise CampaignPlanError("dispatch release authority kind is invalid")
    if terminal_event not in expected_events:
        raise CampaignPlanError(
            "dispatch release ledger lacks its terminal phase"
        )
    if kind == SANDBOX_RELEASE_AUTHORITY_KIND:
        _validate_sandbox_abandon_event(item, terminal)
        _validate_sandbox_isolation_result(
            item, terminal, terminal_result
        )
    if (
        kind == "ordinary_safe_terminal_v1"
        and terminal_result.get("result") == "infrastructure_noncounting"
    ):
        try:
            parsed_zero = RebootRecovery.parse_dispatch_marker(
                marker_payload, require_recovery_arm=False
            )
        except RebootRecovery.RecoveryEvidenceError as exc:
            raise CampaignPlanError(str(exc)) from exc
        if (
            len(marker_rows) != 2
            or parsed_zero.dispatch_id != marker.dispatch_id
            or parsed_zero.unquiesced.get("event")
            != "dispatch_zero_ledger_quarantined"
        ):
            raise CampaignPlanError(
                "zero-ledger release lacks its exact durable marker"
            )
        _validate_zero_ledger_event(item, parsed_zero, terminal)
        _validate_zero_ledger_result(
            item, parsed_zero, terminal, terminal_result
        )
    result_payload = json.dumps(
        terminal_result,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    item_payload = json.dumps(
        item, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    authority = {
        "schema": "scheduler_dispatch_release_authority_v1",
        "kind": kind,
        "projected_item_sha256": hashlib.sha256(item_payload).hexdigest(),
        "game": item["game"],
        "target_level": item["target_level"],
        "retry_complexity_n": item["retry_complexity_n"],
        **{
            field: item[field]
            for field in (
                *Status.FRONTIER_BINDING_FIELDS,
                "reached",
                "parent_action_count",
            )
        },
        "ledger": os.fspath(prefix.path),
        "ledger_parent_identity": list(prefix.parent_identity),
        "ledger_file_identity": list(prefix.file_identity),
        "ledger_prefix_bytes": len(prefix.raw_prefix),
        "ledger_prefix_sha256": hashlib.sha256(
            prefix.raw_prefix
        ).hexdigest(),
        "dispatch_ledger_prefix_bytes": dispatch_prefix_bytes,
        "dispatch_ledger_prefix_sha256": dispatch_prefix_sha256,
        "terminal_event": terminal_event,
        "terminal_record_sha256": _recovery_record_sha256(terminal),
        "terminal_result": terminal_result,
        "terminal_result_sha256": hashlib.sha256(result_payload).hexdigest(),
    }
    marker_probe = {
        "dispatch_id": marker.dispatch_id,
        "marker_name": marker.name,
        "marker_identity": list(marker.marker_identity),
        "marker_bytes": len(marker_payload),
        "marker_sha256": hashlib.sha256(marker_payload).hexdigest(),
        "capsule_name": marker.capsule_name,
        "capsule_identity": (
            list(marker.capsule_identity)
            if marker.capsule_identity is not None else None
        ),
        "capsule_present_at_intent": not marker.capsule_missing
        and marker.capsule_name is not None,
        "capsule_bytes": (
            armed.get("wip_rollback_capsule_bytes")
            if not marker.capsule_missing else None
        ),
        "capsule_sha256": (
            armed.get("wip_rollback_capsule_sha256")
            if not marker.capsule_missing else None
        ),
    }
    _read_dispatch_release_marker_phase(
        marker.root_fd,
        marker_probe,
        authority,
        allow_absent=False,
    )
    _validate_dispatch_release_terminal_prefix(
        authority,
        prefix.raw_prefix,
        dispatch_id=marker.dispatch_id,
    )
    return authority


def _dispatch_release_item_ledger(
    item: dict[str, Any], authority: dict[str, Any]
) -> Path:
    """Select the current host ledger without rebinding an old retry row.

    A clean ``not_solved`` outcome advances the next retry coordinate, and a
    solved outcome can advance the checkpoint before release is replayed.  The
    old intent therefore authenticates its own projected item; the new plan is
    used only to select the same game, target, and host ledger.
    """

    ledger = _ledger_path(item["argv"], cwd=_runner_cwd(item))
    authority_ledger = authority.get("ledger")
    if (
        not isinstance(authority_ledger, str)
        or not Path(authority_ledger).is_absolute()
        or Path(os.path.abspath(authority_ledger)) != Path(authority_ledger)
        or authority_ledger != os.fspath(ledger)
        or authority.get("game") != item.get("game")
        or authority.get("target_level") != item.get("target_level")
    ):
        raise CampaignPlanError(
            "dispatch release authority belongs to a different dispatch lane"
        )
    return ledger


def _read_dispatch_release_ledger_locked(
    ledger: Path, authority: dict[str, Any]
) -> tuple[bytes, tuple[int, int], tuple[int, int]]:
    """Read the exact release ledger generation while its append lock is held."""

    _reject_symlinked_ancestry(ledger.parent, "release ledger parent")
    parent_identity = _host_directory_identity(
        ledger.parent, "release ledger parent"
    )
    expected_parent = _marker_identity(
        authority.get("ledger_parent_identity"), "release ledger parent"
    )
    expected_file = _marker_identity(
        authority.get("ledger_file_identity"), "release ledger file"
    )
    try:
        metadata = ledger.stat(follow_symlinks=False)
    except OSError as exc:
        raise CampaignPlanError("dispatch release ledger is unavailable") from exc
    file_identity = (metadata.st_dev, metadata.st_ino)
    if (
        parent_identity != expected_parent
        or ledger.is_symlink()
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or file_identity != expected_file
    ):
        raise CampaignPlanError(
            "dispatch release ledger generation changed"
        )
    try:
        raw = Legs._read_single_link_regular(os.fspath(ledger))
    except (OSError, Legs.WorkspaceTainted) as exc:
        raise CampaignPlanError("dispatch release ledger is unstable") from exc
    after = ledger.stat(follow_symlinks=False)
    if (
        ledger.is_symlink()
        or not stat.S_ISREG(after.st_mode)
        or after.st_nlink != 1
        or (after.st_dev, after.st_ino) != file_identity
        or after.st_size != len(raw)
        or after.st_mtime_ns != metadata.st_mtime_ns
    ):
        raise CampaignPlanError(
            "dispatch release ledger changed during read"
        )
    return raw, parent_identity, file_identity


def _validate_dispatch_release_armed_row(
    armed: dict[str, Any],
    *,
    record: dict[str, Any],
    authority: dict[str, Any],
) -> None:
    """Bind the one-row ordinary safe-terminal marker exactly."""

    keys = set(armed)
    is_v1 = keys == RebootRecovery.ARMED_V1_KEYS
    is_v2 = keys == RebootRecovery.ARMED_V2_KEYS
    expected_frontier = {
        "game": authority["game"],
        "reached": authority["reached"],
        "target_level": authority["target_level"],
        "parent_action_count": authority["parent_action_count"],
        **{
            field: authority[field]
            for field in Status.FRONTIER_BINDING_FIELDS
        },
    }
    if (
        not (is_v1 or is_v2)
        or armed.get("event") != "dispatch_armed"
        or armed.get("dispatch_id") != record.get("dispatch_id")
        or armed.get("game") != authority.get("game")
        or armed.get("target_level") != authority.get("target_level")
        or armed.get("retry_complexity_n")
        != authority.get("retry_complexity_n")
        or armed.get("frontier_binding") != expected_frontier
        or armed.get("projected_item_sha256")
        != authority.get("projected_item_sha256")
        or armed.get("ledger") != authority.get("ledger")
        or armed.get("ledger_parent_identity")
        != authority.get("ledger_parent_identity")
        or armed.get("ledger_prefix_bytes")
        != authority.get("dispatch_ledger_prefix_bytes")
        or armed.get("ledger_prefix_sha256")
        != authority.get("dispatch_ledger_prefix_sha256")
        or (is_v1 and armed.get("schema") != DISPATCH_QUARANTINE_SCHEMA)
        or (
            is_v2
            and (
                armed.get("schema")
                != RebootRecovery.DISPATCH_QUARANTINE_SCHEMA_V2
                or armed.get("armed_schema")
                != RebootRecovery.DISPATCH_ARMED_SCHEMA_V2
            )
        )
    ):
        raise CampaignPlanError(
            "dispatch release marker is not the exact armed phase"
        )
    _recovery_recorded_at(armed, "dispatch release armed marker")
    marker_ledger_identity = armed.get("ledger_file_identity")
    if marker_ledger_identity is not None and marker_ledger_identity != (
        authority.get("ledger_file_identity")
    ):
        raise CampaignPlanError(
            "dispatch release marker ledger inode changed"
        )
    if marker_ledger_identity is None and armed.get("ledger_prefix_bytes") != 0:
        raise CampaignPlanError(
            "dispatch release marker has an impossible missing ledger inode"
        )
    capsule_name = record.get("capsule_name")
    if is_v1:
        if capsule_name is not None:
            raise CampaignPlanError(
                "legacy dispatch release marker gained a WIP capsule"
            )
        return
    if (
        armed.get("wip_rollback_capsule_name") != capsule_name
        or armed.get("wip_rollback_capsule_identity")
        != record.get("capsule_identity")
    ):
        raise CampaignPlanError(
            "dispatch release WIP capsule binding changed"
        )
    if record.get("capsule_present_at_intent") is True and any((
        armed.get("wip_rollback_capsule_bytes")
        != record.get("capsule_bytes"),
        armed.get("wip_rollback_capsule_sha256")
        != record.get("capsule_sha256"),
    )):
        raise CampaignPlanError(
            "dispatch release WIP capsule seal changed"
        )


def _read_dispatch_release_marker_phase(
    root_fd: int,
    record: dict[str, Any],
    authority: dict[str, Any],
    *,
    allow_absent: bool,
) -> bool:
    """Read and parse the exact marker phase bound into an installed intent."""

    marker_name = str(record["marker_name"])
    marker_identity = _marker_identity(
        record.get("marker_identity"), "release marker"
    )
    try:
        payload = _read_bound_release_file_at(
            root_fd,
            marker_name,
            marker_identity,
            label="dispatch quarantine marker",
            maximum_bytes=RebootRecovery.MAX_MARKER_BYTES,
        )
    except FileNotFoundError:
        if allow_absent:
            return False
        raise CampaignPlanError(
            "dispatch release marker disappeared before authorization"
        ) from None
    if (
        len(payload) != record.get("marker_bytes")
        or hashlib.sha256(payload).hexdigest()
        != record.get("marker_sha256")
    ):
        raise CampaignPlanError(
            "dispatch release marker changed after intent installation"
        )
    try:
        rows = RebootRecovery.parse_canonical_jsonl(
            payload, label="dispatch release marker"
        )
    except RebootRecovery.RecoveryEvidenceError as exc:
        raise CampaignPlanError(str(exc)) from exc
    if authority.get("kind") == "ordinary_safe_terminal_v1":
        result = authority.get("terminal_result")
        operator_discard = (
            isinstance(result, dict)
            and result.get("result") == "tainted_noncounting"
            and result.get("operator_recovery")
            == "discarded_unpublished_same_boot_generation"
            and result.get("detached_processes_proven_absent") is False
            and result.get("published_frontier_unchanged") is True
        )
        zero_ledger_terminal = (
            isinstance(result, dict)
            and result.get("result") == "infrastructure_noncounting"
            and isinstance(result.get("zero_ledger_replayed"), bool)
        )
        if zero_ledger_terminal:
            if len(rows) != 2:
                raise CampaignPlanError(
                    "zero-ledger release requires its exact two-row marker"
                )
            try:
                parsed = RebootRecovery.parse_dispatch_marker(
                    payload, require_recovery_arm=False
                )
            except RebootRecovery.RecoveryEvidenceError as exc:
                raise CampaignPlanError(str(exc)) from exc
            if (
                parsed.dispatch_id != record.get("dispatch_id")
                or parsed.unquiesced.get("event")
                != "dispatch_zero_ledger_quarantined"
            ):
                raise CampaignPlanError(
                    "zero-ledger release marker authority changed"
                )
            armed = dict(parsed.armed)
        elif len(rows) == 1:
            armed = dict(rows[0])
        elif len(rows) == 3 and operator_discard:
            try:
                parsed = RebootRecovery.parse_dispatch_marker(
                    payload, require_recovery_arm=True
                )
            except RebootRecovery.RecoveryEvidenceError as exc:
                raise CampaignPlanError(str(exc)) from exc
            arm = parsed.recovery_arm
            if (
                parsed.dispatch_id != record.get("dispatch_id")
                or arm is None
                or arm.get("wip_recovery_authority")
                != "operator_confirmed_quarantined_wip_v1"
                or arm.get("wip_disposition")
                != "discard_latest_pointer"
                or arm.get("restored_wip_logical_state_sha256") is not None
            ):
                raise CampaignPlanError(
                    "operator discard marker authority changed"
                )
            armed = dict(parsed.armed)
        else:
            raise CampaignPlanError(
                "ordinary release marker has a later failure phase"
            )
        _validate_dispatch_release_armed_row(
            armed, record=record, authority=authority
        )
    elif authority.get("kind") == SANDBOX_RELEASE_AUTHORITY_KIND:
        try:
            parsed = RebootRecovery.parse_sandboxed_generation_marker(
                payload, require_recovery_arm=True
            )
        except RebootRecovery.RecoveryEvidenceError as exc:
            raise CampaignPlanError(str(exc)) from exc
        if parsed.recovery_arm is None:
            raise CampaignPlanError(
                "sandbox-isolated release lacks its operator arm"
            )
        _validate_dispatch_release_armed_row(
            dict(parsed.armed), record=record, authority=authority
        )
        if (
            parsed.armed.get("wip_rollback_capsule_name")
            != record.get("capsule_name")
            or parsed.armed.get("wip_rollback_capsule_identity")
            != record.get("capsule_identity")
            or (
                record.get("capsule_present_at_intent") is True
                and any((
                    parsed.armed.get("wip_rollback_capsule_bytes")
                    != record.get("capsule_bytes"),
                    parsed.armed.get("wip_rollback_capsule_sha256")
                    != record.get("capsule_sha256"),
                ))
            )
        ):
            raise CampaignPlanError(
                "sandbox-isolated release capsule binding changed"
            )
    elif authority.get("kind") == "post_reboot_operator_terminal_v1":
        try:
            parsed = RebootRecovery.parse_dispatch_marker(
                payload, require_recovery_arm=True
            )
        except RebootRecovery.RecoveryEvidenceError as exc:
            raise CampaignPlanError(str(exc)) from exc
        expected_frontier = {
            "game": authority["game"],
            "reached": authority["reached"],
            "target_level": authority["target_level"],
            "parent_action_count": authority["parent_action_count"],
            **{
                field: authority[field]
                for field in Status.FRONTIER_BINDING_FIELDS
            },
        }
        marker_ledger_identity = parsed.armed.get("ledger_file_identity")
        if (
            parsed.dispatch_id != record.get("dispatch_id")
            or parsed.armed.get("game") != authority.get("game")
            or parsed.armed.get("target_level")
            != authority.get("target_level")
            or parsed.armed.get("retry_complexity_n")
            != authority.get("retry_complexity_n")
            or parsed.armed.get("frontier_binding") != expected_frontier
            or parsed.armed.get("projected_item_sha256")
            != authority.get("projected_item_sha256")
            or parsed.armed.get("ledger") != authority.get("ledger")
            or parsed.armed.get("ledger_parent_identity")
            != authority.get("ledger_parent_identity")
            or parsed.armed.get("ledger_prefix_bytes")
            != authority.get("dispatch_ledger_prefix_bytes")
            or parsed.armed.get("ledger_prefix_sha256")
            != authority.get("dispatch_ledger_prefix_sha256")
            or (
                marker_ledger_identity is not None
                and marker_ledger_identity
                != authority.get("ledger_file_identity")
            )
            or (
                marker_ledger_identity is None
                and parsed.armed.get("ledger_prefix_bytes") != 0
            )
            or parsed.armed.get("wip_rollback_capsule_name")
            != record.get("capsule_name")
            or parsed.armed.get("wip_rollback_capsule_identity")
            != record.get("capsule_identity")
        ):
            raise CampaignPlanError(
                "post-reboot release marker binding changed"
            )
        if record.get("capsule_present_at_intent") is True and any((
            parsed.armed.get("wip_rollback_capsule_bytes")
            != record.get("capsule_bytes"),
            parsed.armed.get("wip_rollback_capsule_sha256")
            != record.get("capsule_sha256"),
        )):
            raise CampaignPlanError(
                "post-reboot release capsule seal changed"
            )
    else:
        raise CampaignPlanError(
            "dispatch release marker authority kind is unsupported"
        )
    return True


def _release_coordinate_item(authority: dict[str, Any]) -> dict[str, Any]:
    return {
        "game": authority["game"],
        "target_level": authority["target_level"],
        "retry_complexity_n": authority["retry_complexity_n"],
        "reached": authority["reached"],
        "parent_action_count": authority["parent_action_count"],
        **{
            field: authority[field]
            for field in Status.FRONTIER_BINDING_FIELDS
        },
    }


def _validate_release_exec_coordinate(
    row: dict[str, Any], authority: dict[str, Any]
) -> None:
    expected = {
        "event": "codex_exec",
        "game": authority["game"],
        "target_level": authority["target_level"],
        "run_label": (
            f"{authority['game']}:L{authority['target_level']}:propose"
        ),
        "model": "gpt-5.6-sol",
        "reached": authority["reached"],
        "parent_action_count": authority["parent_action_count"],
        **{
            field: authority[field]
            for field in Status.FRONTIER_BINDING_FIELDS
        },
    }
    if any(row.get(field) != value for field, value in expected.items()):
        raise CampaignPlanError(
            "dispatch release exec row changed coordinate"
        )
    _safe_component(row.get("thread_id"), "release thread id")
    _safe_component(row.get("transcript"), "release transcript")
    _safe_component(row.get("workspace"), "release workspace")


def _validate_dispatch_release_terminal_prefix(
    authority: dict[str, Any], raw_prefix: bytes, *, dispatch_id: str
) -> list[dict[str, Any]]:
    """Authenticate the exact pre-authorization terminal ledger prefix."""

    prefix_bytes = authority.get("ledger_prefix_bytes")
    dispatch_bytes = authority.get("dispatch_ledger_prefix_bytes")
    if (
        not isinstance(prefix_bytes, int)
        or isinstance(prefix_bytes, bool)
        or not isinstance(dispatch_bytes, int)
        or isinstance(dispatch_bytes, bool)
        or len(raw_prefix) != prefix_bytes
        or not 0 <= dispatch_bytes < prefix_bytes
        or hashlib.sha256(raw_prefix).hexdigest()
        != authority.get("ledger_prefix_sha256")
        or hashlib.sha256(raw_prefix[:dispatch_bytes]).hexdigest()
        != authority.get("dispatch_ledger_prefix_sha256")
    ):
        raise CampaignPlanError(
            "dispatch release terminal ledger prefix changed"
        )
    _strict_ledger_records(
        raw_prefix[:dispatch_bytes], label="dispatch release ledger baseline"
    )
    suffix = _strict_ledger_records(
        raw_prefix[dispatch_bytes:],
        label="dispatch release terminal suffix",
    )
    result = authority.get("terminal_result")
    if (
        not isinstance(result, dict)
        or result.get("game") != authority.get("game")
        or result.get("target_level") != authority.get("target_level")
    ):
        raise CampaignPlanError(
            "dispatch release terminal result changed"
        )
    kind = authority.get("kind")
    if kind == "ordinary_safe_terminal_v1":
        if result.get("result") in {"solved", "not_solved"}:
            if [row.get("event") for row in suffix] != [
                "codex_exec", "codex_level_outcome"
            ]:
                raise CampaignPlanError(
                    "ordinary clean release lacks its exact terminal chain"
                )
            execution, outcome = suffix
            _validate_release_exec_coordinate(execution, authority)
            reached_after = outcome.get("reached_after")
            expected_outcome = {
                "event": "codex_level_outcome",
                "codex_exec_transcript": execution.get("transcript"),
                "thread_id": execution.get("thread_id"),
                "game": authority["game"],
                "target_level": authority["target_level"],
                "run_label": execution.get("run_label"),
                "model": execution.get("model"),
                "reasoning_effort": execution.get("reasoning_effort"),
                "reached": authority["reached"],
                "reached_before": authority["reached"],
                "taint_verdict": "clean",
                **{
                    field: authority[field]
                    for field in Status.FRONTIER_BINDING_FIELDS
                },
            }
            if (
                any(
                    outcome.get(field) != value
                    for field, value in expected_outcome.items()
                )
                or not isinstance(reached_after, int)
                or isinstance(reached_after, bool)
                or not authority["reached"] <= reached_after <= authority[
                    "target_level"
                ]
                or outcome.get("solved_target")
                is not (reached_after >= authority["target_level"])
                or result.get("reached") != reached_after
                or (result.get("result") == "solved")
                is not bool(outcome.get("solved_target"))
            ):
                raise CampaignPlanError(
                    "ordinary clean release outcome changed"
                )
        elif result.get("result") == "infrastructure_noncounting":
            if len(suffix) != 1 or suffix[0].get("event") != ZERO_LEDGER_EVENT:
                raise CampaignPlanError(
                    "zero-ledger release lacks its exact infrastructure event"
                )
            infrastructure = suffix[0]
            event_keys = ZERO_LEDGER_EVENT_BASE_KEYS | (
                ZERO_LEDGER_EVENT_DIAGNOSTICS_KEYS
                if "diagnostics" in infrastructure
                else frozenset()
            )
            expected = {
                "event": ZERO_LEDGER_EVENT,
                "schema": ZERO_LEDGER_EVENT_SCHEMA,
                "infrastructure_authority": (
                    "scheduler_quiesced_zero_ledger_suffix_v1"
                ),
                "dispatch_id": dispatch_id,
                "game": authority["game"],
                "target_level": authority["target_level"],
                "reached": authority["reached"],
                "parent_action_count": authority["parent_action_count"],
                "retry_complexity_n": authority["retry_complexity_n"],
                "failure_class": "infrastructure",
                "failure_detail_class": (
                    "interrupted_before_codex_exec_append"
                ),
                "taint_verdict": "quarantined",
                "retry_increment": 0,
                "codex_exec_appended": False,
                "process_tree_quiesced": True,
                **{
                    field: authority[field]
                    for field in Status.FRONTIER_BINDING_FIELDS
                },
            }
            if (
                set(infrastructure) != event_keys
                or set(result) != ZERO_LEDGER_RESULT_KEYS
                or not isinstance(result.get("zero_ledger_replayed"), bool)
                or any(
                    infrastructure.get(field) != value
                    for field, value in expected.items()
                )
                or result.get("reached") != authority.get("reached")
                or result.get("retry_complexity_n")
                != authority.get("retry_complexity_n")
                or infrastructure.get("terminal_errors")
                != [result.get("reason")]
                or infrastructure.get("child_returncode")
                != result.get("child_returncode")
                or any(
                    not isinstance(result.get(field), str)
                    or not result[field]
                    for field in (
                        "seed_mode", "wip_mode", "lineage_input_mode"
                    )
                )
            ):
                raise CampaignPlanError(
                    "zero-ledger infrastructure release binding changed"
                )
        elif result.get("result") == "tainted_noncounting":
            events = [row.get("event") for row in suffix]
            if events not in (
                [
                    "codex_exec",
                    "codex_exec_classification_correction",
                    "codex_taint_cleanup_completed",
                ],
                [
                    "codex_exec",
                    "codex_level_outcome",
                    "codex_exec_classification_correction",
                    "codex_taint_cleanup_completed",
                ],
            ):
                raise CampaignPlanError(
                    "ordinary taint release lacks its exact terminal chain"
                )
            execution = suffix[0]
            correction, cleanup = suffix[-2:]
            _validate_release_exec_coordinate(execution, authority)
            if len(suffix) == 4:
                outcome = suffix[1]
                reached_after = outcome.get("reached_after")
                expected_outcome = {
                    "event": "codex_level_outcome",
                    "codex_exec_transcript": execution.get("transcript"),
                    "thread_id": execution.get("thread_id"),
                    "game": authority["game"],
                    "target_level": authority["target_level"],
                    "run_label": execution.get("run_label"),
                    "model": execution.get("model"),
                    "reasoning_effort": execution.get(
                        "reasoning_effort"
                    ),
                    "reached": authority["reached"],
                    "reached_before": authority["reached"],
                    "taint_verdict": "clean",
                    **{
                        field: authority[field]
                        for field in Status.FRONTIER_BINDING_FIELDS
                    },
                }
                if (
                    any(
                        outcome.get(field) != value
                        for field, value in expected_outcome.items()
                    )
                    or not isinstance(reached_after, int)
                    or isinstance(reached_after, bool)
                    or not authority["reached"] <= reached_after <= (
                        authority["target_level"]
                    )
                    or outcome.get("solved_target")
                    is not (reached_after >= authority["target_level"])
                ):
                    raise CampaignPlanError(
                        "corrected clean outcome changed before taint release"
                    )
            coordinate = _release_coordinate_item(authority)
            _validate_recovery_correction(
                coordinate,
                execution,
                correction,
                not_before=datetime.min.replace(tzinfo=timezone.utc),
            )
            _validate_recovery_cleanup(
                coordinate,
                execution,
                cleanup,
                not_before=_recovery_recorded_at(
                    correction, "release correction"
                ),
            )
            if (
                result.get("reached") != authority.get("reached")
                or cleanup.get("retry_increment") != 0
            ):
                raise CampaignPlanError(
                    "ordinary taint release result changed"
                )
        else:
            raise CampaignPlanError(
                "ordinary release result is not terminal"
            )
    elif kind == SANDBOX_RELEASE_AUTHORITY_KIND:
        if (
            len(suffix) != 1
            or suffix[0].get("event") != SANDBOX_ABANDON_EVENT
            or result.get("result") != "sandbox_isolated_noncounting"
            or result.get("dispatch_id") != dispatch_id
            or result.get("process_tree_quiesced") is not False
            or result.get("detached_processes_proven_absent") is not False
        ):
            raise CampaignPlanError(
                "sandbox-isolated release lacks its exact nonquiescent event"
            )
        _validate_sandbox_abandon_event(
            _release_coordinate_item(authority), suffix[0]
        )
        _validate_sandbox_isolation_result(
            _release_coordinate_item(authority), suffix[0], result
        )
    elif kind == "post_reboot_operator_terminal_v1":
        if [row.get("event") for row in suffix] != [
            "codex_exec",
            "codex_exec_classification_correction",
            "codex_taint_cleanup_completed",
            "codex_post_reboot_operator_recovery_completed",
        ]:
            raise CampaignPlanError(
                "post-reboot release lacks its exact terminal chain"
            )
        execution, correction, cleanup, operator = suffix
        _validate_release_exec_coordinate(execution, authority)
        coordinate = _release_coordinate_item(authority)
        _validate_recovery_correction(
            coordinate,
            execution,
            correction,
            not_before=datetime.min.replace(tzinfo=timezone.utc),
        )
        _validate_recovery_cleanup(
            coordinate,
            execution,
            cleanup,
            not_before=_recovery_recorded_at(
                correction, "release correction"
            ),
        )
        expected_operator = {
            "event": "codex_post_reboot_operator_recovery_completed",
            "schema": RebootRecovery.OPERATOR_RECOVERY_SCHEMA,
            "recovery_authority": "scheduler_authenticated_post_reboot_v1",
            "dispatch_id": dispatch_id,
            "game": authority["game"],
            "target_level": authority["target_level"],
            "retry_increment": 0,
            "projected_item_sha256": authority[
                "projected_item_sha256"
            ],
            "exec_record_sha256": _recovery_record_sha256(execution),
            "correction_record_sha256": _recovery_record_sha256(correction),
            "cleanup_record_sha256": _recovery_record_sha256(cleanup),
            **{
                field: authority[field]
                for field in (
                    *Status.FRONTIER_BINDING_FIELDS,
                    "reached",
                    "parent_action_count",
                )
            },
        }
        if (
            any(
                operator.get(field) != value
                for field, value in expected_operator.items()
            )
            or result.get("result") != "tainted_noncounting"
            or result.get("dispatch_id") != dispatch_id
        ):
            raise CampaignPlanError(
                "post-reboot release operator receipt changed"
            )
    else:
        raise CampaignPlanError(
            "dispatch release terminal authority kind is unsupported"
        )
    terminal = suffix[-1]
    if (
        terminal.get("event") != authority.get("terminal_event")
        or _recovery_record_sha256(terminal)
        != authority.get("terminal_record_sha256")
    ):
        raise CampaignPlanError(
            "dispatch release terminal ledger phase changed"
        )
    return suffix


def _validate_dispatch_release_authority(
    item: dict[str, Any],
    root_fd: int,
    record: dict[str, Any],
    intent_identity: tuple[int, int],
    *,
    allow_missing_marker: bool,
) -> None:
    """Require the exact durable host authorization row before deletion."""

    _validate_dispatch_release_intent_record(
        record, marker_name=str(record["marker_name"])
    )
    if _marker_identity(
        record.get("intent_identity"), "release intent"
    ) != intent_identity:
        raise CampaignPlanError("dispatch release intent identity changed")
    authority = record["release_authority"]
    assert isinstance(authority, dict)
    ledger = _dispatch_release_item_ledger(item, authority)
    authority_line = RebootRecovery.canonical_json_line(
        authority["authority_record"]
    )
    with Guard.ledger_append_lock(ledger):
        raw, _parent_identity, _file_identity = (
            _read_dispatch_release_ledger_locked(ledger, authority)
        )
        prefix_bytes = int(authority["ledger_prefix_bytes"])
        expected = raw[:prefix_bytes] + authority_line
        if len(raw) != len(expected) or raw != expected:
            raise CampaignPlanError(
                "dispatch release lacks its exact host authorization row"
            )
        _validate_dispatch_release_terminal_prefix(
            authority,
            raw[:prefix_bytes],
            dispatch_id=str(record["dispatch_id"]),
        )
    _read_dispatch_release_marker_phase(
        root_fd,
        record,
        authority,
        allow_absent=allow_missing_marker,
    )


def _open_dispatch_quarantine_root(
    item: dict[str, Any], *, create: bool
) -> tuple[Path, int, tuple[int, int]] | None:
    artifact_root = _artifact_root(item)
    artifact_identity = _host_directory_identity(
        artifact_root, "canonical artifact root"
    )
    parent_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    parent_fd: int | None = None
    try:
        parent_fd = os.open(artifact_root, parent_flags)
        parent_metadata = os.fstat(parent_fd)
        if (
            not stat.S_ISDIR(parent_metadata.st_mode)
            or (parent_metadata.st_dev, parent_metadata.st_ino)
            != artifact_identity
        ):
            raise CampaignPlanError(
                "canonical artifact root identity changed"
            )
    except BaseException as exc:
        if parent_fd is not None:
            try:
                os.close(parent_fd)
            except OSError:
                pass
        if isinstance(exc, CampaignPlanError):
            raise
        raise CampaignPlanError(
            "canonical artifact root is unavailable for quarantine"
        ) from exc
    root = artifact_root / ".campaign_quarantine"
    descriptor: int | None = None
    try:
        created = False
        if not os.path.lexists(root):
            if not create:
                return None
            try:
                os.mkdir(".campaign_quarantine", 0o700, dir_fd=parent_fd)
                os.chmod(root, 0o700, follow_symlinks=False)
                created = True
            except FileExistsError:
                created = False
            except OSError as exc:
                raise CampaignPlanError(
                    "could not create the dispatch quarantine root"
                ) from exc
        _reject_symlinked_ancestry(root, "dispatch quarantine root")
        metadata = root.stat(follow_symlinks=False)
        if (
            root.is_symlink()
            or not stat.S_ISDIR(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or stat.S_IMODE(metadata.st_mode) != 0o700
        ):
            raise CampaignPlanError(
                "dispatch quarantine root has unsafe custody"
            )
        descriptor = os.open(root, parent_flags)
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISDIR(opened.st_mode)
            or opened.st_uid != os.geteuid()
            or stat.S_IMODE(opened.st_mode) != 0o700
            or (opened.st_dev, opened.st_ino)
            != (metadata.st_dev, metadata.st_ino)
        ):
            raise CampaignPlanError(
                "dispatch quarantine root identity changed"
            )
        if created:
            os.fsync(descriptor)
            if (
                _host_directory_identity(
                    artifact_root, "canonical artifact root"
                )
                != artifact_identity
            ):
                raise CampaignPlanError(
                    "canonical artifact root changed during quarantine creation"
                )
            os.fsync(parent_fd)
        return root, descriptor, (opened.st_dev, opened.st_ino)
    except BaseException as exc:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass
        if isinstance(exc, CampaignPlanError):
            raise
        raise CampaignPlanError(
            "dispatch quarantine root is unavailable"
        ) from exc
    finally:
        os.close(parent_fd)


def _assert_no_dispatch_quarantine(item: dict[str, Any]) -> None:
    opened = _open_dispatch_quarantine_root(item, create=False)
    if opened is None:
        return
    root, descriptor, _identity = opened
    name = _dispatch_quarantine_name(item)
    try:
        reconciled = _reconcile_dispatch_release_intent_at(
            item, descriptor, marker_name=name
        )
        intent_name, preparing_name = _dispatch_release_intent_names(name)
        safe_release_arm_name = _safe_release_recovery_arm_name(name)

        def residue() -> list[str]:
            names = set(os.listdir(descriptor))
            target = {
                candidate
                for candidate in (
                    name,
                    intent_name,
                    preparing_name,
                    safe_release_arm_name,
                )
                if candidate in names
            }
            capsule_prefix = f".{name}."
            target.update(
                candidate
                for candidate in names
                if candidate.startswith(capsule_prefix)
                and candidate.endswith(".wip_rollback_capsule")
            )
            return sorted(target)

        remaining = residue()
        if reconciled:
            if remaining:
                raise CampaignPlanError(
                    "dispatch release reconciliation left residue"
                )
            os.fsync(descriptor)
            if residue():
                raise CampaignPlanError(
                    "dispatch release residue reappeared after fsync"
                )
            return
        try:
            os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        except FileNotFoundError:
            # Required even for an apparently absent marker: a previous
            # process may have crashed after unlink and before parent fsync.
            remaining = residue()
            if remaining:
                raise CampaignPlanError(
                    "orphan dispatch quarantine release residue exists"
                )
            os.fsync(descriptor)
            if residue():
                raise CampaignPlanError(
                    "dispatch quarantine marker reappeared after fsync"
                )
            return
        except OSError as exc:
            raise CampaignPlanError(
                "dispatch quarantine marker inventory is unreadable"
            ) from exc
        raise CampaignPlanError(
            "prior dispatch quarantine requires explicit operator release: "
            f"{root / name}"
        )
    finally:
        os.close(descriptor)


def _validate_dispatch_quarantine(marker: DispatchQuarantine) -> None:
    root_fd_metadata = os.fstat(marker.root_fd)
    root_path_metadata = marker.root.stat(follow_symlinks=False)
    marker_fd_metadata = os.fstat(marker.marker_fd)
    marker_path_metadata = os.stat(
        marker.name,
        dir_fd=marker.root_fd,
        follow_symlinks=False,
    )
    if (
        marker.root.is_symlink()
        or not stat.S_ISDIR(root_fd_metadata.st_mode)
        or not stat.S_ISDIR(root_path_metadata.st_mode)
        or root_fd_metadata.st_uid != os.geteuid()
        or stat.S_IMODE(root_fd_metadata.st_mode) != 0o700
        or (root_fd_metadata.st_dev, root_fd_metadata.st_ino)
        != marker.root_identity
        or (root_path_metadata.st_dev, root_path_metadata.st_ino)
        != marker.root_identity
        or not stat.S_ISREG(marker_fd_metadata.st_mode)
        or not stat.S_ISREG(marker_path_metadata.st_mode)
        or marker_fd_metadata.st_nlink != 1
        or marker_path_metadata.st_nlink != 1
        or marker_fd_metadata.st_uid != os.geteuid()
        or stat.S_IMODE(marker_fd_metadata.st_mode) != 0o600
        or (marker_fd_metadata.st_dev, marker_fd_metadata.st_ino)
        != marker.marker_identity
        or (marker_path_metadata.st_dev, marker_path_metadata.st_ino)
        != marker.marker_identity
    ):
        raise CampaignPlanError("dispatch quarantine identity changed")


def _write_dispatch_quarantine_record(
    marker: DispatchQuarantine, record: dict[str, Any]
) -> None:
    payload = json.dumps(
        {
            **record,
            "schema": marker.schema,
            "dispatch_id": marker.dispatch_id,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8") + b"\n"
    try:
        _validate_dispatch_quarantine(marker)
        offset = 0
        while offset < len(payload):
            written = os.write(marker.marker_fd, payload[offset:])
            if written <= 0:
                raise OSError(
                    errno.EIO,
                    "dispatch quarantine write made no progress",
                )
            offset += written
        os.fsync(marker.marker_fd)
        os.fsync(marker.root_fd)
        _validate_dispatch_quarantine(marker)
    except (OSError, TypeError, ValueError) as exc:
        raise CampaignPlanError(
            "could not seal the dispatch quarantine receipt"
        ) from exc


def _remove_failed_dispatch_quarantine(
    *,
    root_fd: int,
    name: str,
    marker_fd: int,
    marker_identity: tuple[int, int] | None,
) -> None:
    """Remove only the exact marker inode created by a failed arm.

    No caller owns the quarantine until ``_arm_dispatch_quarantine`` returns.
    Consequently every pre-return failure must either durably remove that
    exact inode or fail closed without touching a replacement/alias.
    """

    try:
        fd_metadata = os.fstat(marker_fd)
        observed_identity = (
            fd_metadata.st_dev,
            fd_metadata.st_ino,
        )
        if marker_identity is None:
            marker_identity = observed_identity
        try:
            path_metadata = os.stat(
                name, dir_fd=root_fd, follow_symlinks=False
            )
        except FileNotFoundError:
            # An independently removed marker is already absent, but make
            # that namespace transition durable before returning control.
            os.fsync(root_fd)
            return
        safe_mode = (
            stat.S_IMODE(fd_metadata.st_mode) & ~0o600 == 0
            and stat.S_IMODE(path_metadata.st_mode) & ~0o600 == 0
        )
        if (
            observed_identity != marker_identity
            or (path_metadata.st_dev, path_metadata.st_ino)
            != marker_identity
            or not stat.S_ISREG(fd_metadata.st_mode)
            or not stat.S_ISREG(path_metadata.st_mode)
            or fd_metadata.st_nlink != 1
            or path_metadata.st_nlink != 1
            or fd_metadata.st_uid != os.geteuid()
            or path_metadata.st_uid != os.geteuid()
            or not safe_mode
        ):
            raise CampaignPlanError(
                "failed dispatch arm left a replaced or aliased quarantine"
            )
        os.unlink(name, dir_fd=root_fd)
        os.fsync(root_fd)
    except CampaignPlanError:
        raise
    except OSError as exc:
        raise CampaignPlanError(
            "could not durably remove the failed dispatch quarantine"
        ) from exc


def _install_wip_rollback_capsule(
    marker: DispatchQuarantine,
    item: dict[str, Any],
    state: WipRollbackState,
) -> dict[str, Any]:
    record = _wip_rollback_capsule_record(item, state, marker.dispatch_id)
    payload = RebootRecovery.canonical_json_line(record)
    if len(payload) > MAX_WIP_ROLLBACK_CAPSULE_BYTES:
        raise CampaignPlanError("WIP rollback capsule exceeds the durable bound")
    if _wip_capsule_state_sha256(
        _capture_wip_rollback(item), str(record["schema"])
    ) != record["state_sha256"]:
        raise CampaignPlanError(
            "WIP changed while preparing its rollback capsule"
        )
    name = f".{marker.name}.{marker.dispatch_id}.wip_rollback_capsule"
    descriptor: int | None = None
    try:
        descriptor = os.open(
            name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=marker.root_fd,
        )
        os.fchmod(descriptor, 0o600)
        before = os.fstat(descriptor)
        identity = (before.st_dev, before.st_ino)
        marker.capsule_name = name
        marker.capsule_identity = identity
        marker.capsule_state = state
        marker.capsule_record = record
        path_before = os.stat(
            name, dir_fd=marker.root_fd, follow_symlinks=False
        )
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_uid != os.geteuid()
            or stat.S_IMODE(before.st_mode) != 0o600
            or identity != (path_before.st_dev, path_before.st_ino)
        ):
            raise CampaignPlanError("new WIP rollback capsule is not unaliased")
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise OSError("short write preparing WIP rollback capsule")
            offset += written
        os.fsync(descriptor)
        after = os.fstat(descriptor)
        path_after = os.stat(
            name, dir_fd=marker.root_fd, follow_symlinks=False
        )
        if (
            (after.st_dev, after.st_ino) != identity
            or after.st_nlink != 1
            or after.st_size != len(payload)
            or after.st_uid != os.geteuid()
            or stat.S_IMODE(after.st_mode) != 0o600
            or (path_after.st_dev, path_after.st_ino) != identity
        ):
            raise CampaignPlanError("WIP rollback capsule changed while sealing")
        os.fsync(marker.root_fd)
        return {
            "armed_schema": RebootRecovery.DISPATCH_ARMED_SCHEMA_V2,
            "wip_rollback_capsule_name": name,
            "wip_rollback_capsule_identity": list(identity),
            "wip_rollback_capsule_bytes": len(payload),
            "wip_rollback_capsule_sha256": hashlib.sha256(payload).hexdigest(),
            "wip_rollback_capsule_state_sha256": record["state_sha256"],
            "wip_restore_logical_state_schema": record[
                "restore_logical_state_schema"
            ],
            "wip_restore_logical_state_sha256": record[
                "restore_logical_state_sha256"
            ],
        }
    except OSError as exc:
        raise CampaignPlanError(
            "could not durably install the WIP rollback capsule"
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _read_bound_wip_rollback_capsule(
    marker: DispatchQuarantine,
    parsed: RebootRecovery.ParsedMarker,
    item: dict[str, Any],
) -> tuple[WipRollbackState, dict[str, Any]]:
    armed = parsed.armed
    name = armed.get("wip_rollback_capsule_name")
    if not isinstance(name, str) or Path(name).name != name:
        raise CampaignPlanError("dispatch WIP rollback capsule name is malformed")
    expected_identity = _marker_identity(
        armed.get("wip_rollback_capsule_identity"), "WIP rollback capsule"
    )
    expected_size = armed.get("wip_rollback_capsule_bytes")
    if (
        not isinstance(expected_size, int)
        or isinstance(expected_size, bool)
        or not 0 < expected_size <= MAX_WIP_ROLLBACK_CAPSULE_BYTES
    ):
        raise CampaignPlanError("dispatch WIP rollback capsule size is malformed")
    marker.capsule_name = name
    marker.capsule_identity = expected_identity
    descriptor: int | None = None
    try:
        descriptor = os.open(
            name,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=marker.root_fd,
        )
        before = os.fstat(descriptor)
        path_before = os.stat(
            name, dir_fd=marker.root_fd, follow_symlinks=False
        )
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_uid != os.geteuid()
            or stat.S_IMODE(before.st_mode) != 0o600
            or (before.st_dev, before.st_ino) != expected_identity
            or (path_before.st_dev, path_before.st_ino) != expected_identity
            or before.st_size != expected_size
        ):
            raise CampaignPlanError("WIP rollback capsule has unsafe custody")
        payload = bytearray()
        offset = 0
        while offset < expected_size:
            chunk = os.pread(
                descriptor, min(1024 * 1024, expected_size - offset), offset
            )
            if not chunk:
                raise CampaignPlanError("WIP rollback capsule was truncated")
            payload.extend(chunk)
            offset += len(chunk)
        after = os.fstat(descriptor)
        if (
            (after.st_dev, after.st_ino) != expected_identity
            or after.st_size != expected_size
            or after.st_mtime_ns != before.st_mtime_ns
            or after.st_nlink != 1
            or hashlib.sha256(payload).hexdigest()
            != armed.get("wip_rollback_capsule_sha256")
        ):
            raise CampaignPlanError("WIP rollback capsule changed during read")
    except FileNotFoundError as exc:
        raise MissingWipRollbackCapsule(
            "bound WIP rollback capsule is missing"
        ) from exc
    except OSError as exc:
        raise CampaignPlanError("WIP rollback capsule is unavailable") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
    try:
        rows = RebootRecovery.parse_canonical_jsonl(
            bytes(payload), label="WIP rollback capsule"
        )
    except RebootRecovery.RecoveryEvidenceError as exc:
        raise CampaignPlanError(str(exc)) from exc
    if len(rows) != 1:
        raise CampaignPlanError("WIP rollback capsule row count is invalid")
    record = dict(rows[0])
    if set(record) != {
        "schema", "dispatch_id", "game", "target_level", "state_sha256",
        "restore_logical_state_schema", "restore_logical_state_sha256",
        "state", "file_payloads_base64",
    } or any((
        record.get("schema") not in WIP_ROLLBACK_CAPSULE_SCHEMAS,
        record.get("dispatch_id") != marker.dispatch_id,
        record.get("game") != item["game"],
        record.get("target_level") != item["target_level"],
        record.get("state_sha256")
        != armed.get("wip_rollback_capsule_state_sha256"),
        record.get("restore_logical_state_schema")
        != armed.get("wip_restore_logical_state_schema"),
        record.get("restore_logical_state_sha256")
        != armed.get("wip_restore_logical_state_sha256"),
    )):
        raise CampaignPlanError("WIP rollback capsule binding is invalid")
    state = _state_from_wip_rollback_capsule(record, item)
    if list(state.baseline_snapshot) != armed.get("target_wip_snapshot"):
        raise CampaignPlanError(
            "WIP rollback capsule historical snapshot changed"
        )
    marker.capsule_name = name
    marker.capsule_identity = expected_identity
    marker.capsule_state = state
    marker.capsule_record = record
    return state, record


def _arm_dispatch_quarantine(
    item: dict[str, Any],
    *,
    ledger_before: LedgerPrefixState,
    wip_before: WipRollbackState,
    canonical_before: CanonicalRollbackState,
    ownership: list[DispatchQuarantine | None] | None = None,
    durable_wip_capsule: bool = True,
) -> DispatchQuarantine:
    if (
        durable_wip_capsule
        and not wip_before.existed
        and (
            wip_before.absence_custody is None
            or wip_before.absence_custody.parent != wip_before.level.parent
            or wip_before.absence_custody.name != wip_before.level.name
        )
    ):
        raise CampaignPlanError(
            "durable dispatch requires a preexisting WIP context parent"
        )
    if ownership is not None and (
        len(ownership) != 1 or ownership[0] is not None
    ):
        raise CampaignPlanError(
            "dispatch quarantine ownership handoff is malformed"
        )
    root: Path | None = None
    root_fd: int | None = None
    root_identity: tuple[int, int] | None = None
    name: str | None = None
    marker_fd: int | None = None
    marker_identity: tuple[int, int] | None = None
    marker: DispatchQuarantine | None = None
    previous_mask: set[Any] | None = None
    try:
        previous_mask = Contiguous._block_scoped_spawn_signals()
        opened = _open_dispatch_quarantine_root(item, create=True)
        assert opened is not None
        root, root_fd, root_identity = opened
        name = _dispatch_quarantine_name(item)
        try:
            os.stat(name, dir_fd=root_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise CampaignPlanError(
                "prior dispatch quarantine requires explicit operator release: "
                f"{root / name}"
            )
        flags = (
            os.O_WRONLY
            | os.O_APPEND
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0)
        )
        marker_fd = os.open(name, flags, 0o600, dir_fd=root_fd)
        # Bind the just-created object before any further fallible mutation.
        marker_metadata = os.fstat(marker_fd)
        marker_identity = (marker_metadata.st_dev, marker_metadata.st_ino)
        os.fchmod(marker_fd, 0o600)
        marker_metadata = os.fstat(marker_fd)
        path_metadata = os.stat(
            name, dir_fd=root_fd, follow_symlinks=False
        )
        if marker_identity != (
            marker_metadata.st_dev,
            marker_metadata.st_ino,
        ):
            raise CampaignPlanError(
                "new dispatch quarantine descriptor identity changed"
            )
        if (
            not stat.S_ISREG(marker_metadata.st_mode)
            or marker_metadata.st_nlink != 1
            or marker_metadata.st_uid != os.geteuid()
            or marker_identity != (path_metadata.st_dev, path_metadata.st_ino)
        ):
            raise CampaignPlanError(
                "new dispatch quarantine marker is not unaliased"
            )
        dispatch_id = os.urandom(16).hex()
        marker = DispatchQuarantine(
            root=root,
            root_fd=root_fd,
            root_identity=root_identity,
            name=name,
            path=root / name,
            marker_fd=marker_fd,
            marker_identity=marker_identity,
            dispatch_id=dispatch_id,
            schema=(
                RebootRecovery.DISPATCH_QUARANTINE_SCHEMA_V2
                if durable_wip_capsule
                else DISPATCH_QUARANTINE_SCHEMA
            ),
        )
        capsule_binding = (
            _install_wip_rollback_capsule(marker, item, wip_before)
            if durable_wip_capsule
            else {}
        )
        artifact_root = _artifact_root(item)
        artifact_identity = _host_directory_identity(
            artifact_root, "canonical artifact root"
        )
        item_payload = json.dumps(
            item, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        frontier = {
            field: item[field]
            for field in (
                *Status.FRONTIER_BINDING_FIELDS,
                "game",
                "reached",
                "target_level",
                "parent_action_count",
            )
        }
        _write_dispatch_quarantine_record(marker, {
            "event": "dispatch_armed",
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "pid": os.getpid(),
            "game": item["game"],
            "target_level": item["target_level"],
            "tag": _single_cli_value(item["argv"], "--tag"),
            "run_label": f"{item['game']}:L{item['target_level']}:propose",
            "retry_complexity_n": item["retry_complexity_n"],
            "artifact_root": os.fspath(artifact_root),
            "artifact_root_identity": list(artifact_identity),
            "canonical_root": os.fspath(canonical_before.root),
            "canonical_root_identity": list(canonical_before.root_identity),
            "canonical_digest": canonical_before.digest,
            "frontier_binding": frontier,
            "target_wip_snapshot": list(wip_before.baseline_snapshot),
            "ledger": os.fspath(ledger_before.path),
            "ledger_parent_identity": list(ledger_before.parent_identity),
            "ledger_file_identity": (
                list(ledger_before.file_identity)
                if ledger_before.file_identity is not None
                else None
            ),
            "ledger_prefix_bytes": len(ledger_before.raw_prefix),
            "ledger_prefix_sha256": hashlib.sha256(
                ledger_before.raw_prefix
            ).hexdigest(),
            "projected_item_sha256": hashlib.sha256(item_payload).hexdigest(),
            "historical_runner": item.get("historical_runner"),
            **capsule_binding,
        })
        if ownership is not None:
            # STORE_SUBSCR is the single ownership-transfer bytecode.  Before
            # it, this frame cleans a failed arm; after it, the encompassing
            # caller can close or preserve the marker even if an async
            # BaseException lands before the call result is assigned.
            ownership[0] = marker
        Contiguous._restore_scoped_spawn_signals(previous_mask)
        previous_mask = None
        return marker
    except BaseException as failure:
        transferred = (
            ownership is not None
            and marker is not None
            and ownership[0] is marker
        )
        cleanup_failure: BaseException | None = None
        if marker_fd is not None and not transferred:
            assert root_fd is not None and name is not None
            try:
                if marker is not None and marker.capsule_name is not None:
                    capsule_metadata = os.stat(
                        marker.capsule_name,
                        dir_fd=root_fd,
                        follow_symlinks=False,
                    )
                    if (
                        marker.capsule_identity is None
                        or (capsule_metadata.st_dev, capsule_metadata.st_ino)
                        != marker.capsule_identity
                        or capsule_metadata.st_nlink != 1
                    ):
                        raise CampaignPlanError(
                            "failed dispatch arm left a replaced WIP capsule"
                        )
                    os.unlink(marker.capsule_name, dir_fd=root_fd)
                    os.fsync(root_fd)
                _remove_failed_dispatch_quarantine(
                    root_fd=root_fd,
                    name=name,
                    marker_fd=marker_fd,
                    marker_identity=marker_identity,
                )
            except BaseException as exc:
                cleanup_failure = exc
            finally:
                try:
                    os.close(marker_fd)
                except OSError as exc:
                    cleanup_failure = cleanup_failure or exc
        if root_fd is not None and not transferred:
            try:
                os.close(root_fd)
            except OSError as exc:
                cleanup_failure = cleanup_failure or exc
        restore_failure: BaseException | None = None
        if previous_mask is not None:
            try:
                Contiguous._restore_scoped_spawn_signals(previous_mask)
            except BaseException as exc:
                restore_failure = exc
        if transferred:
            # The caller's encompassing try owns both descriptors now.
            if restore_failure is not None:
                raise restore_failure
            raise failure
        if cleanup_failure is not None:
            raise CampaignPlanError(
                "failed dispatch arm could not prove quarantine cleanup"
            ) from cleanup_failure
        if restore_failure is not None:
            raise restore_failure
        raise failure


def _close_dispatch_quarantine(marker: DispatchQuarantine) -> None:
    for descriptor in (marker.marker_fd, marker.root_fd):
        try:
            os.close(descriptor)
        except OSError:
            pass


def _release_binding_is_present(
    root_fd: int,
    *,
    name: str,
    identity: tuple[int, int],
    expected_bytes: int,
    expected_sha256: str,
    label: str,
    maximum_bytes: int,
) -> bool:
    try:
        metadata = os.stat(
            name, dir_fd=root_fd, follow_symlinks=False
        )
    except FileNotFoundError:
        return False
    if (metadata.st_dev, metadata.st_ino) != identity:
        raise CampaignPlanError(f"{label} identity changed during release")
    observed_bytes, observed_sha256 = _release_regular_binding_at(
        root_fd,
        name,
        identity,
        label=label,
        maximum_bytes=maximum_bytes,
    )
    if (
        observed_bytes != expected_bytes
        or observed_sha256 != expected_sha256
    ):
        raise CampaignPlanError(f"{label} bytes changed during release")
    return True


def _install_dispatch_release_intent(
    marker: DispatchQuarantine,
    release_authority: dict[str, Any],
) -> tuple[dict[str, Any], tuple[int, int]]:
    _validate_dispatch_quarantine(marker)
    if set(release_authority) != _DISPATCH_RELEASE_AUTHORITY_BASE_KEYS:
        raise CampaignPlanError(
            "dispatch release base authority has an invalid exact schema"
        )
    intent_name, preparing_name = _dispatch_release_intent_names(marker.name)
    for candidate in (intent_name, preparing_name):
        try:
            os.stat(candidate, dir_fd=marker.root_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise CampaignPlanError(
                "prior dispatch release intent requires reconciliation"
            )
    marker_bytes, marker_sha256 = _release_regular_binding_at(
        marker.root_fd,
        marker.name,
        marker.marker_identity,
        label="dispatch quarantine marker",
        maximum_bytes=RebootRecovery.MAX_MARKER_BYTES,
    )
    capsule_name = marker.capsule_name
    capsule_identity = marker.capsule_identity
    capsule_present = False
    capsule_bytes: int | None = None
    capsule_sha256: str | None = None
    if capsule_name is not None:
        if capsule_identity is None:
            raise CampaignPlanError(
                "dispatch release lacks a capsule identity"
            )
        if marker.capsule_missing:
            try:
                os.stat(
                    capsule_name,
                    dir_fd=marker.root_fd,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                pass
            else:
                raise CampaignPlanError(
                    "retired WIP rollback capsule name reappeared"
                )
        else:
            capsule_bytes, capsule_sha256 = _release_regular_binding_at(
                marker.root_fd,
                capsule_name,
                capsule_identity,
                label="WIP rollback capsule",
                maximum_bytes=MAX_WIP_ROLLBACK_CAPSULE_BYTES,
            )
            capsule_present = True
    descriptor: int | None = None
    renamed = False
    try:
        descriptor = os.open(
            preparing_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=marker.root_fd,
        )
        os.fchmod(descriptor, 0o600)
        preparing = os.fstat(descriptor)
        intent_identity = (preparing.st_dev, preparing.st_ino)
        record: dict[str, Any] = {
            "schema": DISPATCH_RELEASE_INTENT_SCHEMA,
            "event": "dispatch_release_intent",
            "dispatch_id": marker.dispatch_id,
            "intent_name": intent_name,
            "intent_identity": list(intent_identity),
            "quarantine_root_identity": list(marker.root_identity),
            "marker_name": marker.name,
            "marker_identity": list(marker.marker_identity),
            "marker_bytes": marker_bytes,
            "marker_sha256": marker_sha256,
            "capsule_name": capsule_name,
            "capsule_identity": (
                list(capsule_identity)
                if capsule_identity is not None else None
            ),
            "capsule_present_at_intent": capsule_present,
            "capsule_bytes": capsule_bytes,
            "capsule_sha256": capsule_sha256,
            # Replaced below after the newly created intent inode and fresh
            # post-terminal nonce have both been bound.
            "release_authority": dict(release_authority),
        }
        release_nonce = os.urandom(32).hex()
        intent_core_sha256 = _dispatch_release_intent_core_sha256(
            record, release_authority
        )
        authority_record = {
            "event": "codex_dispatch_release_authorized",
            "schema": "scheduler_dispatch_release_authorized_v1",
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "dispatch_id": marker.dispatch_id,
            "release_nonce": release_nonce,
            "intent_name": intent_name,
            "intent_identity": list(intent_identity),
            "intent_core_sha256": intent_core_sha256,
            "projected_item_sha256": release_authority[
                "projected_item_sha256"
            ],
            "game": release_authority["game"],
            "target_level": release_authority["target_level"],
            "retry_complexity_n": release_authority[
                "retry_complexity_n"
            ],
            "reached": release_authority["reached"],
            "parent_action_count": release_authority[
                "parent_action_count"
            ],
            "terminal_kind": release_authority["kind"],
            "terminal_event": release_authority["terminal_event"],
            "terminal_record_sha256": release_authority[
                "terminal_record_sha256"
            ],
            "ledger": release_authority["ledger"],
            "ledger_parent_identity": release_authority[
                "ledger_parent_identity"
            ],
            "ledger_file_identity": release_authority[
                "ledger_file_identity"
            ],
            "ledger_prefix_bytes": release_authority[
                "ledger_prefix_bytes"
            ],
            "ledger_prefix_sha256": release_authority[
                "ledger_prefix_sha256"
            ],
            **{
                field: release_authority[field]
                for field in Status.FRONTIER_BINDING_FIELDS
            },
        }
        final_authority = {
            **release_authority,
            "release_nonce": release_nonce,
            "intent_core_sha256": intent_core_sha256,
            "authority_record": authority_record,
        }
        record["release_authority"] = final_authority
        _validate_dispatch_release_intent_record(
            record, marker_name=marker.name
        )
        payload = RebootRecovery.canonical_json_line(record)
        if len(payload) > MAX_DISPATCH_RELEASE_INTENT_BYTES:
            raise CampaignPlanError(
                "dispatch release intent exceeds the durable bound"
            )
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise OSError("short dispatch release intent write")
            offset += written
        os.fsync(descriptor)
        sealed = os.fstat(descriptor)
        preparing_path = os.stat(
            preparing_name,
            dir_fd=marker.root_fd,
            follow_symlinks=False,
        )
        if (
            (sealed.st_dev, sealed.st_ino) != intent_identity
            or (preparing_path.st_dev, preparing_path.st_ino)
            != intent_identity
            or sealed.st_nlink != 1
            or sealed.st_uid != os.geteuid()
            or stat.S_IMODE(sealed.st_mode) != 0o600
            or sealed.st_size != len(payload)
        ):
            raise CampaignPlanError(
                "dispatch release intent changed while sealing"
            )
        os.replace(
            preparing_name,
            intent_name,
            src_dir_fd=marker.root_fd,
            dst_dir_fd=marker.root_fd,
        )
        renamed = True
        os.fsync(marker.root_fd)
        installed, installed_identity = _read_dispatch_release_intent_at(
            marker.root_fd, intent_name, marker_name=marker.name
        )
        if installed != record or installed_identity != intent_identity:
            raise CampaignPlanError(
                "installed dispatch release intent changed"
            )
        return record, intent_identity
    except OSError as exc:
        raise CampaignPlanError(
            "could not durably install the dispatch release intent"
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        # A visible, fully written preparing file is deliberately retained on
        # pre-rename failure so the reconciler can validate and promote it.
        del renamed


def _dispatch_release_authority_tail(
    raw: bytes,
    authority: dict[str, Any],
    *,
    dispatch_id: str,
) -> tuple[bytes, bytes, bytes]:
    prefix_bytes = authority.get("ledger_prefix_bytes")
    if (
        not isinstance(prefix_bytes, int)
        or isinstance(prefix_bytes, bool)
        or prefix_bytes <= 0
        or len(raw) < prefix_bytes
    ):
        raise CampaignPlanError(
            "dispatch release ledger was truncated before authorization"
        )
    prefix = raw[:prefix_bytes]
    _validate_dispatch_release_terminal_prefix(
        authority, prefix, dispatch_id=dispatch_id
    )
    line = RebootRecovery.canonical_json_line(
        authority["authority_record"]
    )
    tail = raw[prefix_bytes:]
    if len(tail) > len(line) or not line.startswith(tail):
        raise CampaignPlanError(
            "dispatch release ledger has a conflicting authorization tail"
        )
    return prefix, line, tail


def _prevalidate_dispatch_release_preparing(
    item: dict[str, Any],
    root_fd: int,
    record: dict[str, Any],
    intent_identity: tuple[int, int],
) -> str:
    """Prove a complete staging record is eligible for promotion only."""

    if _marker_identity(
        record.get("intent_identity"), "release intent"
    ) != intent_identity:
        raise CampaignPlanError(
            "dispatch release preparing identity changed"
        )
    authority = record["release_authority"]
    assert isinstance(authority, dict)
    root_metadata = os.fstat(root_fd)
    if (
        not stat.S_ISDIR(root_metadata.st_mode)
        or (root_metadata.st_dev, root_metadata.st_ino)
        != _marker_identity(
            record.get("quarantine_root_identity"),
            "release quarantine root",
        )
    ):
        raise CampaignPlanError(
            "dispatch release quarantine root changed"
        )
    # A preparing file is not deletion authority.  It may be promoted only
    # while the exact safe-terminal marker is still present.
    _read_dispatch_release_marker_phase(
        root_fd, record, authority, allow_absent=False
    )
    ledger = _dispatch_release_item_ledger(item, authority)
    with Guard.ledger_append_lock(ledger):
        raw, _parent_identity, _file_identity = (
            _read_dispatch_release_ledger_locked(ledger, authority)
        )
        _prefix, line, tail = _dispatch_release_authority_tail(
            raw, authority, dispatch_id=str(record["dispatch_id"])
        )
    # A staging record by itself is never host deletion authority.  Promotion
    # is replayable only if the exact authorization row was already durable.
    if tail == line:
        return "authorized"
    if not tail:
        return "pre_authority"
    return "partial_authority"


def _ensure_dispatch_release_authority_row(
    item: dict[str, Any],
    root_fd: int,
    record: dict[str, Any],
    intent_identity: tuple[int, int],
    *,
    allow_new_authority_append: bool = False,
) -> None:
    """Append or repair the exact host release authorization using CAS."""

    marker_name = str(record["marker_name"])
    intent_name = str(record["intent_name"])
    # Re-fsync and re-read an installed intent before it can authorize any
    # ledger mutation.  This also resolves rename-before-parent-fsync crashes.
    os.fsync(root_fd)
    installed, installed_identity = _read_dispatch_release_intent_at(
        root_fd, intent_name, marker_name=marker_name
    )
    if installed != record or installed_identity != intent_identity:
        raise CampaignPlanError(
            "dispatch release intent changed before authorization"
        )
    authority = record["release_authority"]
    assert isinstance(authority, dict)
    ledger = _dispatch_release_item_ledger(item, authority)
    marker_present = _read_dispatch_release_marker_phase(
        root_fd, record, authority, allow_absent=True
    )
    with Guard.ledger_append_lock(ledger):
        raw, _parent_identity, file_identity = (
            _read_dispatch_release_ledger_locked(ledger, authority)
        )
        prefix, line, tail = _dispatch_release_authority_tail(
            raw, authority, dispatch_id=str(record["dispatch_id"])
        )
        if not marker_present and tail != line:
            raise CampaignPlanError(
                "dispatch marker disappeared without durable release authority"
            )
        if tail != line and not allow_new_authority_append:
            raise IncompleteDispatchReleaseAuthority(
                "installed dispatch release intent lacks its complete host "
                "authorization row"
            )
        if tail and tail != line and allow_new_authority_append:
            raise IncompleteDispatchReleaseAuthority(
                "live release encountered a preexisting partial authority row"
            )
        if tail != line:
            descriptor: int | None = None
            try:
                descriptor = os.open(
                    ledger,
                    os.O_WRONLY | os.O_APPEND
                    | getattr(os, "O_NOFOLLOW", 0),
                )
                opened = os.fstat(descriptor)
                if (
                    not stat.S_ISREG(opened.st_mode)
                    or opened.st_nlink != 1
                    or (opened.st_dev, opened.st_ino) != file_identity
                    or opened.st_size != len(raw)
                ):
                    raise CampaignPlanError(
                        "dispatch release ledger descriptor changed"
                    )
                missing = line[len(tail):]
                offset = 0
                while offset < len(missing):
                    written = os.write(descriptor, missing[offset:])
                    if written <= 0:
                        raise OSError(
                            errno.EIO,
                            "short dispatch release authority write",
                        )
                    offset += written
                os.fsync(descriptor)
                sealed = os.fstat(descriptor)
                if (
                    (sealed.st_dev, sealed.st_ino) != file_identity
                    or sealed.st_size != len(prefix) + len(line)
                ):
                    raise CampaignPlanError(
                        "dispatch release authorization append was not exact"
                    )
            except OSError as exc:
                raise CampaignPlanError(
                    "could not durably append dispatch release authority"
                ) from exc
            finally:
                if descriptor is not None:
                    os.close(descriptor)
        sealed_raw, _parent_identity, sealed_identity = (
            _read_dispatch_release_ledger_locked(ledger, authority)
        )
        if sealed_identity != file_identity or sealed_raw != prefix + line:
            raise CampaignPlanError(
                "dispatch release authorization row is not exact"
            )
    # If an unquiesced/failed marker row raced the append, the authority row
    # remains an audit receipt but cannot authorize deletion of that marker.
    _read_dispatch_release_marker_phase(
        root_fd, record, authority, allow_absent=not marker_present
    )


def _finish_dispatch_release_intent(
    item: dict[str, Any],
    root_fd: int,
    record: dict[str, Any],
    intent_identity: tuple[int, int],
) -> None:
    marker_name = str(record["marker_name"])
    intent_name = str(record["intent_name"])
    expected_root_identity = _marker_identity(
        record.get("quarantine_root_identity"), "release quarantine root"
    )

    _ensure_dispatch_release_authority_row(
        item, root_fd, record, intent_identity
    )

    def revalidate_intent() -> None:
        root_metadata = os.fstat(root_fd)
        if (
            not stat.S_ISDIR(root_metadata.st_mode)
            or (root_metadata.st_dev, root_metadata.st_ino)
            != expected_root_identity
        ):
            raise CampaignPlanError(
                "dispatch release quarantine root identity changed"
            )
        observed, observed_identity = _read_dispatch_release_intent_at(
            root_fd, intent_name, marker_name=marker_name
        )
        if observed != record or observed_identity != intent_identity:
            raise CampaignPlanError(
                "dispatch release intent changed during retirement"
            )
        _validate_dispatch_release_authority(
            item,
            root_fd,
            observed,
            observed_identity,
            allow_missing_marker=True,
        )

    revalidate_intent()
    marker_present = _release_binding_is_present(
        root_fd,
        name=marker_name,
        identity=_marker_identity(
            record["marker_identity"], "release marker"
        ),
        expected_bytes=int(record["marker_bytes"]),
        expected_sha256=str(record["marker_sha256"]),
        label="dispatch quarantine marker",
        maximum_bytes=RebootRecovery.MAX_MARKER_BYTES,
    )
    capsule_name = record.get("capsule_name")
    if isinstance(capsule_name, str):
        capsule_identity = _marker_identity(
            record.get("capsule_identity"), "release capsule"
        )
        try:
            capsule_metadata = os.stat(
                capsule_name, dir_fd=root_fd, follow_symlinks=False
            )
        except FileNotFoundError:
            capsule_present = False
        else:
            capsule_present = True
            if (capsule_metadata.st_dev, capsule_metadata.st_ino) != (
                capsule_identity
            ):
                raise CampaignPlanError(
                    "WIP rollback capsule identity changed during release"
                )
        if capsule_present:
            if record.get("capsule_present_at_intent") is not True:
                raise CampaignPlanError(
                    "retired WIP rollback capsule name reappeared"
                )
            if not _release_binding_is_present(
                root_fd,
                name=capsule_name,
                identity=capsule_identity,
                expected_bytes=int(record["capsule_bytes"]),
                expected_sha256=str(record["capsule_sha256"]),
                label="WIP rollback capsule",
                maximum_bytes=MAX_WIP_ROLLBACK_CAPSULE_BYTES,
            ):
                raise CampaignPlanError(
                    "WIP rollback capsule disappeared during validation"
                )
            os.unlink(capsule_name, dir_fd=root_fd)
        # Required even if the capsule is already absent: a prior process may
        # have crashed after unlink and before the namespace reached storage.
        os.fsync(root_fd)
        revalidate_intent()
    if marker_present:
        os.unlink(marker_name, dir_fd=root_fd)
    # Likewise, seal (or re-seal) the exact marker absence before retiring the
    # authority that permits it.
    os.fsync(root_fd)
    revalidate_intent()
    current_intent = os.stat(
        intent_name, dir_fd=root_fd, follow_symlinks=False
    )
    if (current_intent.st_dev, current_intent.st_ino) != intent_identity:
        raise CampaignPlanError(
            "dispatch release intent identity changed before retirement"
        )
    os.unlink(intent_name, dir_fd=root_fd)
    os.fsync(root_fd)


def _reconcile_dispatch_release_intent_at(
    item: dict[str, Any],
    root_fd: int,
    *,
    marker_name: str,
) -> bool:
    intent_name, preparing_name = _dispatch_release_intent_names(marker_name)
    present: list[str] = []
    for name in (intent_name, preparing_name):
        try:
            os.stat(name, dir_fd=root_fd, follow_symlinks=False)
        except FileNotFoundError:
            continue
        present.append(name)
    if not present:
        return False
    if len(present) != 1:
        raise CampaignPlanError(
            "dispatch release has ambiguous intent generations"
        )
    # A visible intent or preparing name may have been renamed/created just
    # before a crash.  First make that namespace observation durable.
    os.fsync(root_fd)
    actual_name = present[0]
    try:
        record, intent_identity = _read_dispatch_release_intent_at(
            root_fd, actual_name, marker_name=marker_name
        )
    except CampaignPlanError:
        if actual_name != preparing_name:
            raise
        # A partial/noncanonical staging file has never authorized deletion.
        # Preserve it as fail-closed evidence: if the marker was concurrently
        # removed, retiring this file could erase the final visible residue.
        return False
    if actual_name == preparing_name:
        try:
            preparing_state = _prevalidate_dispatch_release_preparing(
                item, root_fd, record, intent_identity
            )
        except CampaignPlanError:
            # Keep a complete but ineligible staging generation for the same
            # reason: without durable authority it cannot prove that marker
            # absence, if observed, was legitimate.
            return False
        if preparing_state != "authorized":
            return False
        os.replace(
            preparing_name,
            intent_name,
            src_dir_fd=root_fd,
            dst_dir_fd=root_fd,
        )
        os.fsync(root_fd)
        installed, installed_identity = _read_dispatch_release_intent_at(
            root_fd, intent_name, marker_name=marker_name
        )
        if installed != record or installed_identity != intent_identity:
            raise CampaignPlanError(
                "promoted dispatch release intent changed"
            )
    _finish_dispatch_release_intent(
        item, root_fd, record, intent_identity
    )
    return True


def _preflight_post_reboot_release_reconciliation(
    item: dict[str, Any],
) -> bool:
    """Finish an already-installed release before replaying recovery phases."""

    opened = _open_dispatch_quarantine_root(item, create=False)
    if opened is None:
        return False
    _root, root_fd, _root_identity = opened
    try:
        try:
            return _reconcile_dispatch_release_intent_at(
                item,
                root_fd,
                marker_name=_dispatch_quarantine_name(item),
            )
        except IncompleteDispatchReleaseAuthority:
            return False
    finally:
        os.close(root_fd)


def _retire_incomplete_release_for_operator(
    item: dict[str, Any], marker: DispatchQuarantine
) -> None:
    """Let authenticated post-reboot recovery replace an incomplete WAL.

    Ordinary campaign admission must preserve this residue and fail closed.
    Here the complete post-reboot recovery chain has just been revalidated by
    the explicit dispatch ID, nonce, changed-boot proof, and lineage locks.
    It may therefore roll back only an exact empty/partial authority suffix and
    retire only the exact intent/staging inode before installing a fresh WAL.
    """

    intent_name, preparing_name = _dispatch_release_intent_names(marker.name)
    present: list[str] = []
    for candidate in (intent_name, preparing_name):
        try:
            os.stat(
                candidate, dir_fd=marker.root_fd, follow_symlinks=False
            )
        except FileNotFoundError:
            continue
        present.append(candidate)
    if not present:
        return
    if len(present) != 1:
        raise CampaignPlanError(
            "operator recovery found ambiguous release WAL generations"
        )
    actual_name = present[0]
    os.fsync(marker.root_fd)
    record, intent_identity = _read_dispatch_release_intent_at(
        marker.root_fd,
        actual_name,
        marker_name=marker.name,
    )
    authority_state = _prevalidate_dispatch_release_preparing(
        item, marker.root_fd, record, intent_identity
    )
    if authority_state == "authorized":
        raise CampaignPlanError(
            "complete release authority must be reconciled before recovery"
        )
    authority = record["release_authority"]
    assert isinstance(authority, dict)
    ledger = _dispatch_release_item_ledger(item, authority)
    with Guard.ledger_append_lock(ledger):
        raw, _parent_identity, file_identity = (
            _read_dispatch_release_ledger_locked(ledger, authority)
        )
        prefix, line, tail = _dispatch_release_authority_tail(
            raw, authority, dispatch_id=str(record["dispatch_id"])
        )
        if tail == line:
            raise CampaignPlanError(
                "release authority completed during operator recovery"
            )
        if tail:
            descriptor: int | None = None
            try:
                descriptor = os.open(
                    ledger,
                    os.O_RDWR | getattr(os, "O_NOFOLLOW", 0),
                )
                opened = os.fstat(descriptor)
                if (
                    not stat.S_ISREG(opened.st_mode)
                    or opened.st_nlink != 1
                    or (opened.st_dev, opened.st_ino) != file_identity
                    or opened.st_size != len(raw)
                ):
                    raise CampaignPlanError(
                        "operator release ledger descriptor changed"
                    )
                os.ftruncate(descriptor, len(prefix))
                os.fsync(descriptor)
                sealed = os.fstat(descriptor)
                if (
                    (sealed.st_dev, sealed.st_ino) != file_identity
                    or sealed.st_size != len(prefix)
                ):
                    raise CampaignPlanError(
                        "operator release authority rollback was not exact"
                    )
            except OSError as exc:
                raise CampaignPlanError(
                    "operator could not roll back partial release authority"
                ) from exc
            finally:
                if descriptor is not None:
                    os.close(descriptor)
            sealed_raw, _parent_identity, sealed_identity = (
                _read_dispatch_release_ledger_locked(ledger, authority)
            )
            if sealed_identity != file_identity or sealed_raw != prefix:
                raise CampaignPlanError(
                    "operator release authority rollback changed"
                )
    current = os.stat(
        actual_name,
        dir_fd=marker.root_fd,
        follow_symlinks=False,
    )
    if (current.st_dev, current.st_ino) != intent_identity:
        raise CampaignPlanError(
            "dispatch release WAL changed before operator retirement"
        )
    os.unlink(actual_name, dir_fd=marker.root_fd)
    os.fsync(marker.root_fd)
    _validate_recovery_marker_seal(marker)


def _read_or_retire_authenticated_fresh_release_wal(
    item: dict[str, Any],
    marker: DispatchQuarantine,
    arm: dict[str, Any],
    *,
    role: str,
    wal_name: str,
) -> tuple[dict[str, Any], tuple[int, int]] | None:
    """Read a recovery WAL or retire one provably fresh malformed staging.

    The post-reboot arm binds the complete pre-reboot WAL inode and bytes.  A
    later recovery may durably retire that WAL and then crash while writing a
    new ``release_preparing`` record.  That new record has no deletion
    authority.  Under the authenticated changed-boot recovery context, only a
    safe staging inode distinct from the arm-bound WAL may be removed; a
    malformed final intent or the old bound inode always remains fail-closed.
    """

    try:
        return _read_dispatch_release_intent_at(
            marker.root_fd, wal_name, marker_name=marker.name
        )
    except CampaignPlanError as exc:
        if role != "preparing":
            raise
        read_failure = exc
    intent_name, preparing_name = _dispatch_release_intent_names(marker.name)
    if wal_name != preparing_name:
        raise read_failure
    try:
        metadata = os.stat(
            preparing_name,
            dir_fd=marker.root_fd,
            follow_symlinks=False,
        )
    except OSError:
        raise read_failure
    staging_identity = (metadata.st_dev, metadata.st_ino)
    try:
        payload = _read_bound_release_file_at(
            marker.root_fd,
            preparing_name,
            staging_identity,
            label="authenticated fresh dispatch release preparing",
            maximum_bytes=MAX_DISPATCH_RELEASE_INTENT_BYTES,
        )
    except (CampaignPlanError, OSError):
        # A second custody/read failure is uncertainty, not proof that the
        # staging generation is malformed.
        raise read_failure
    malformed = False
    try:
        rows = RebootRecovery.parse_canonical_jsonl(
            payload,
            label="authenticated fresh dispatch release preparing",
        )
    except RebootRecovery.RecoveryEvidenceError:
        malformed = True
    else:
        if (
            len(rows) != 1
            or RebootRecovery.canonical_json_line(rows[0]) != payload
        ):
            malformed = True
        else:
            candidate = dict(rows[0])
            try:
                _validate_dispatch_release_intent_record(
                    candidate, marker_name=marker.name
                )
                candidate_identity = _marker_identity(
                    candidate.get("intent_identity"), "release intent"
                )
            except CampaignPlanError:
                malformed = True
            else:
                if candidate_identity == staging_identity:
                    # The first read failed for an I/O/race reason even though
                    # a subsequent exact read is valid.  Preserve it and make
                    # the caller retry from a fresh observation.
                    raise read_failure
                malformed = True
    if not malformed:
        raise read_failure
    old_intent = arm.get("release_intent")
    if not isinstance(old_intent, dict):
        raise CampaignPlanError(
            "safe-release arm lacks its original WAL"
        )
    old_authority = old_intent.get("release_authority")
    if not isinstance(old_authority, dict):
        raise CampaignPlanError(
            "safe-release arm lacks its original authority"
        )
    old_identity = _marker_identity(
        arm.get("release_wal_identity"), "safe-release WAL"
    )
    if staging_identity == old_identity:
        raise CampaignPlanError(
            "arm-bound safe-release WAL became malformed"
        ) from read_failure

    def validate_authenticated_progress() -> None:
        _validate_dispatch_quarantine(marker)
        _validate_recovery_marker_seal(marker)
        if _safe_release_wal_inventory(
            marker.root_fd, marker.name
        ) != ("preparing", preparing_name):
            raise CampaignPlanError(
                "fresh safe-release preparing generation changed"
            )
        current = os.stat(
            preparing_name,
            dir_fd=marker.root_fd,
            follow_symlinks=False,
        )
        if (current.st_dev, current.st_ino) != staging_identity:
            raise CampaignPlanError(
                "fresh safe-release preparing identity changed"
            )
        _read_dispatch_release_marker_phase(
            marker.root_fd,
            old_intent,
            old_authority,
            allow_absent=False,
        )
        ledger = _dispatch_release_item_ledger(item, old_authority)
        with Guard.ledger_append_lock(ledger):
            raw, _parent, _file = _read_dispatch_release_ledger_locked(
                ledger, old_authority
            )
            _prefix, _line, tail = _dispatch_release_authority_tail(
                raw,
                old_authority,
                dispatch_id=str(arm["dispatch_id"]),
            )
        if tail:
            raise CampaignPlanError(
                "original safe-release authority was not fully retired"
            )
        capsule_name = old_intent.get("capsule_name")
        if isinstance(capsule_name, str):
            capsule_identity = _marker_identity(
                old_intent.get("capsule_identity"), "release capsule"
            )
            if old_intent.get("capsule_present_at_intent") is not True:
                raise CampaignPlanError(
                    "fresh safe-release cleanup lacks capsule custody"
                )
            if not _release_binding_is_present(
                marker.root_fd,
                name=capsule_name,
                identity=capsule_identity,
                expected_bytes=int(old_intent["capsule_bytes"]),
                expected_sha256=str(old_intent["capsule_sha256"]),
                label="WIP rollback capsule",
                maximum_bytes=MAX_WIP_ROLLBACK_CAPSULE_BYTES,
            ):
                raise CampaignPlanError(
                    "fresh safe-release cleanup lost its WIP capsule"
                )
        receipt_name = _safe_release_recovery_receipt_name(
            marker.name, marker.dispatch_id
        )
        for residue in (
            receipt_name,
            _durable_recovery_record_preparing_name(receipt_name),
            intent_name,
        ):
            try:
                os.stat(
                    residue,
                    dir_fd=marker.root_fd,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                continue
            raise CampaignPlanError(
                "malformed fresh safe-release WAL has conflicting authority"
            )

    validate_authenticated_progress()
    _retire_malformed_durable_recovery_preparing_at(
        marker.root_fd,
        preparing_name,
        staging_identity,
        root_path=marker.root,
        root_identity=marker.root_identity,
        label="fresh dispatch release intent",
    )
    # Revalidate every authority surface after the sole namespace mutation.
    # The inventory check is intentionally omitted because the exact staging
    # name is now required to be absent.
    _validate_dispatch_quarantine(marker)
    _validate_recovery_marker_seal(marker)
    if _safe_release_wal_inventory(marker.root_fd, marker.name) is not None:
        raise CampaignPlanError(
            "retired fresh safe-release WAL reappeared"
        )
    _read_dispatch_release_marker_phase(
        marker.root_fd,
        old_intent,
        old_authority,
        allow_absent=False,
    )
    ledger = _dispatch_release_item_ledger(item, old_authority)
    with Guard.ledger_append_lock(ledger):
        raw, _parent, _file = _read_dispatch_release_ledger_locked(
            ledger, old_authority
        )
        _prefix, _line, tail = _dispatch_release_authority_tail(
            raw,
            old_authority,
            dispatch_id=str(arm["dispatch_id"]),
        )
    if tail:
        raise CampaignPlanError(
            "safe-release authority changed during fresh WAL retirement"
        )
    return None


def _release_dispatch_quarantine(
    marker: DispatchQuarantine,
    item: dict[str, Any],
    release_authority: dict[str, Any],
    *,
    before_authority_append: Any | None = None,
) -> None:
    try:
        record, intent_identity = _install_dispatch_release_intent(
            marker, release_authority
        )
        if before_authority_append is not None:
            before_authority_append(record, intent_identity)
        _ensure_dispatch_release_authority_row(
            item,
            marker.root_fd,
            record,
            intent_identity,
            allow_new_authority_append=True,
        )
        _finish_dispatch_release_intent(
            item, marker.root_fd, record, intent_identity
        )
    except OSError as exc:
        raise CampaignPlanError(
            "could not durably release the dispatch quarantine"
        ) from exc
    finally:
        _close_dispatch_quarantine(marker)


def _read_existing_dispatch_quarantine(
    item: dict[str, Any], *, require_recovery_arm: bool | None = None,
    allow_missing_capsule: bool = False,
    marker_parser: Any = RebootRecovery.parse_dispatch_marker,
) -> tuple[DispatchQuarantine, RebootRecovery.ParsedMarker]:
    """Open and parse one preexisting marker without following aliases."""

    opened = _open_dispatch_quarantine_root(item, create=False)
    if opened is None:
        raise NoDispatchQuarantine(
            "no dispatch quarantine exists for recovery"
        )
    root, root_fd, root_identity = opened
    name = _dispatch_quarantine_name(item)
    marker_fd: int | None = None
    try:
        flags = (
            os.O_RDWR
            | os.O_APPEND
            | getattr(os, "O_NOFOLLOW", 0)
        )
        try:
            marker_fd = os.open(name, flags, dir_fd=root_fd)
        except FileNotFoundError as exc:
            raise NoDispatchQuarantine(
                "no dispatch quarantine exists for recovery"
            ) from exc
        before = os.fstat(marker_fd)
        path_metadata = os.stat(
            name, dir_fd=root_fd, follow_symlinks=False
        )
        marker_identity = (before.st_dev, before.st_ino)
        if (
            not stat.S_ISREG(before.st_mode)
            or not stat.S_ISREG(path_metadata.st_mode)
            or before.st_nlink != 1
            or path_metadata.st_nlink != 1
            or before.st_uid != os.geteuid()
            or path_metadata.st_uid != os.geteuid()
            or stat.S_IMODE(before.st_mode) != 0o600
            or stat.S_IMODE(path_metadata.st_mode) != 0o600
            or marker_identity
            != (path_metadata.st_dev, path_metadata.st_ino)
            or before.st_size <= 0
            or before.st_size > RebootRecovery.MAX_MARKER_BYTES
        ):
            raise CampaignPlanError(
                "dispatch quarantine marker has unsafe inode custody"
            )
        payload = bytearray()
        offset = 0
        while offset < before.st_size:
            chunk = os.pread(
                marker_fd,
                min(1024 * 1024, before.st_size - offset),
                offset,
            )
            if not chunk:
                raise CampaignPlanError(
                    "dispatch quarantine marker was truncated during read"
                )
            payload.extend(chunk)
            offset += len(chunk)
        after = os.fstat(marker_fd)
        path_after = os.stat(
            name, dir_fd=root_fd, follow_symlinks=False
        )
        if (
            (after.st_dev, after.st_ino, after.st_nlink, after.st_size,
             after.st_mtime_ns)
            != (before.st_dev, before.st_ino, before.st_nlink, before.st_size,
                before.st_mtime_ns)
            or (path_after.st_dev, path_after.st_ino)
            != marker_identity
        ):
            raise CampaignPlanError(
                "dispatch quarantine marker changed during read"
            )
        try:
            parsed = marker_parser(
                bytes(payload), require_recovery_arm=require_recovery_arm
            )
        except RebootRecovery.RecoveryEvidenceError as exc:
            raise CampaignPlanError(str(exc)) from exc
        marker = DispatchQuarantine(
            root=root,
            root_fd=root_fd,
            root_identity=root_identity,
            name=name,
            path=root / name,
            marker_fd=marker_fd,
            marker_identity=marker_identity,
            dispatch_id=parsed.dispatch_id,
            schema=str(parsed.armed["schema"]),
            recovery_sealed_size=len(payload),
            recovery_sealed_sha256=hashlib.sha256(payload).hexdigest(),
        )
        _validate_dispatch_quarantine(marker)
        if marker.schema == RebootRecovery.DISPATCH_QUARANTINE_SCHEMA_V2:
            try:
                _read_bound_wip_rollback_capsule(marker, parsed, item)
            except MissingWipRollbackCapsule:
                if not allow_missing_capsule:
                    raise
                marker.capsule_missing = True
        return marker, parsed
    except BaseException:
        if marker_fd is not None:
            try:
                os.close(marker_fd)
            except OSError:
                pass
        try:
            os.close(root_fd)
        except OSError:
            pass
        raise


def _marker_identity(value: object, label: str) -> tuple[int, int]:
    try:
        left, right = value  # type: ignore[misc]
    except (TypeError, ValueError) as exc:
        raise CampaignPlanError(f"dispatch marker {label} identity is malformed") from exc
    if any(
        not isinstance(part, int) or isinstance(part, bool) or part < 0
        for part in (left, right)
    ):
        raise CampaignPlanError(f"dispatch marker {label} identity is malformed")
    return int(left), int(right)


def _validate_recovery_marker_seal(marker: DispatchQuarantine) -> None:
    """Re-read the exact marker bytes so inode-preserving edits cannot race."""

    expected_size = marker.recovery_sealed_size
    expected_sha = marker.recovery_sealed_sha256
    if expected_size is None or expected_sha is None:
        raise CampaignPlanError("dispatch recovery marker lacks a byte seal")
    _validate_dispatch_quarantine(marker)
    metadata = os.fstat(marker.marker_fd)
    if metadata.st_size != expected_size:
        raise CampaignPlanError("dispatch recovery marker size changed")
    payload = bytearray()
    offset = 0
    while offset < expected_size:
        try:
            chunk = os.pread(
                marker.marker_fd,
                min(1024 * 1024, expected_size - offset),
                offset,
            )
        except OSError as exc:
            raise CampaignPlanError(
                "dispatch recovery marker is unreadable"
            ) from exc
        if not chunk:
            raise CampaignPlanError("dispatch recovery marker was truncated")
        payload.extend(chunk)
        offset += len(chunk)
    after = os.fstat(marker.marker_fd)
    if (
        after.st_size != expected_size
        or after.st_mtime_ns != metadata.st_mtime_ns
        or hashlib.sha256(payload).hexdigest() != expected_sha
    ):
        raise CampaignPlanError("dispatch recovery marker bytes changed")
    _validate_dispatch_quarantine(marker)


def _sealed_recovery_marker_bytes(marker: DispatchQuarantine) -> bytes:
    _validate_recovery_marker_seal(marker)
    assert marker.recovery_sealed_size is not None
    payload = bytearray()
    offset = 0
    while offset < marker.recovery_sealed_size:
        chunk = os.pread(
            marker.marker_fd,
            marker.recovery_sealed_size - offset,
            offset,
        )
        if not chunk:
            raise CampaignPlanError("dispatch recovery marker was truncated")
        payload.extend(chunk)
        offset += len(chunk)
    _validate_recovery_marker_seal(marker)
    return bytes(payload)


def _read_unaliased_small_file_at(
    root_fd: int, name: str, *, label: str
) -> tuple[int, bytes, tuple[int, int]]:
    descriptor: int | None = None
    try:
        descriptor = os.open(
            name,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=root_fd,
        )
        metadata = os.fstat(descriptor)
        path_metadata = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
        identity = (metadata.st_dev, metadata.st_ino)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or metadata.st_uid != os.geteuid()
            or stat.S_IMODE(metadata.st_mode) != 0o600
            or identity != (path_metadata.st_dev, path_metadata.st_ino)
            or not 0 <= metadata.st_size <= RebootRecovery.MAX_MARKER_BYTES
        ):
            raise CampaignPlanError(f"{label} has unsafe inode custody")
        payload = bytearray()
        offset = 0
        while offset < metadata.st_size:
            chunk = os.pread(descriptor, metadata.st_size - offset, offset)
            if not chunk:
                raise CampaignPlanError(f"{label} was truncated")
            payload.extend(chunk)
            offset += len(chunk)
        after = os.fstat(descriptor)
        if (
            after.st_size != metadata.st_size
            or after.st_mtime_ns != metadata.st_mtime_ns
            or after.st_nlink != 1
        ):
            raise CampaignPlanError(f"{label} changed during read")
        return descriptor, bytes(payload), identity
    except BaseException:
        if descriptor is not None:
            os.close(descriptor)
        raise


def _durably_reseal_staged_file_at(
    root_fd: int,
    name: str,
    *,
    expected_payload: bytes,
    expected_identity: tuple[int, int],
    label: str,
) -> tuple[int, int]:
    """Re-fsync and byte-rebind one exact staging inode before rename."""

    descriptor: int | None = None
    try:
        descriptor = os.open(
            name,
            os.O_RDWR | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=root_fd,
        )
        before = os.fstat(descriptor)
        path_before = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
        identity = (before.st_dev, before.st_ino)
        if (
            not stat.S_ISREG(before.st_mode)
            or not stat.S_ISREG(path_before.st_mode)
            or before.st_nlink != 1
            or path_before.st_nlink != 1
            or before.st_uid != os.geteuid()
            or path_before.st_uid != os.geteuid()
            or before.st_gid != path_before.st_gid
            or stat.S_IMODE(before.st_mode) != 0o600
            or stat.S_IMODE(path_before.st_mode) != 0o600
            or identity != expected_identity
            or identity != (path_before.st_dev, path_before.st_ino)
            or before.st_size != len(expected_payload)
        ):
            raise CampaignPlanError(f"{label} has unsafe staging custody")
        os.fsync(descriptor)
        payload = bytearray()
        offset = 0
        while offset < before.st_size:
            chunk = os.pread(descriptor, before.st_size - offset, offset)
            if not chunk:
                raise CampaignPlanError(f"{label} was truncated after fsync")
            payload.extend(chunk)
            offset += len(chunk)
        after = os.fstat(descriptor)
        path_after = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
        if (
            (
                after.st_dev,
                after.st_ino,
                after.st_nlink,
                after.st_size,
                after.st_mtime_ns,
                after.st_uid,
                after.st_gid,
                stat.S_IMODE(after.st_mode),
            )
            != (
                before.st_dev,
                before.st_ino,
                before.st_nlink,
                before.st_size,
                before.st_mtime_ns,
                before.st_uid,
                before.st_gid,
                stat.S_IMODE(before.st_mode),
            )
            or (path_after.st_dev, path_after.st_ino) != expected_identity
            or path_after.st_nlink != 1
            or path_after.st_uid != os.geteuid()
            or path_after.st_gid != before.st_gid
            or stat.S_IMODE(path_after.st_mode) != 0o600
            or bytes(payload) != expected_payload
        ):
            raise CampaignPlanError(f"{label} changed while being re-sealed")
        return identity
    except OSError as exc:
        raise CampaignPlanError(
            f"could not durably re-seal {label}"
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _atomic_recovery_arm_replace(
    marker: DispatchQuarantine,
    arm_record: dict[str, Any],
    *,
    marker_parser: Any = RebootRecovery.parse_dispatch_marker,
    sidecar_suffix: str = "post_reboot_arm",
) -> dict[str, Any]:
    """Install the third row atomically; a partial sidecar is never authority."""

    prefix = _sealed_recovery_marker_bytes(marker)
    if arm_record.get("pre_arm_marker_identity") != list(
        marker.marker_identity
    ):
        raise CampaignPlanError(
            "recovery arm does not bind the pre-arm marker inode"
        )
    if SAFE_COMPONENT_RE.fullmatch(sidecar_suffix) is None:
        raise CampaignPlanError("recovery arm sidecar suffix is unsafe")
    sidecar = f".{marker.name}.{sidecar_suffix}"
    sidecar_payload: bytes | None = None
    installed_arm: dict[str, Any] | None = None
    sidecar_identity: tuple[int, int] | None = None
    try:
        sidecar_fd, sidecar_payload, sidecar_identity = (
            _read_unaliased_small_file_at(
            marker.root_fd, sidecar, label="post-reboot arm sidecar"
            )
        )
    except FileNotFoundError:
        sidecar_fd = None
    except OSError as exc:
        raise CampaignPlanError(
            "could not safely inspect the post-reboot arm sidecar"
        ) from exc
    else:
        os.close(sidecar_fd)
        try:
            parsed = marker_parser(
                sidecar_payload, require_recovery_arm=True
            )
        except RebootRecovery.RecoveryEvidenceError:
            _validate_recovery_marker_seal(marker)
            try:
                os.unlink(sidecar, dir_fd=marker.root_fd)
                os.fsync(marker.root_fd)
            except OSError as exc:
                raise CampaignPlanError(
                    "could not discard an incomplete recovery arm sidecar"
                ) from exc
            sidecar_payload = None
        else:
            if (
                not sidecar_payload.startswith(prefix)
                or parsed.dispatch_id != marker.dispatch_id
                or parsed.recovery_arm is None
                or parsed.recovery_arm.get("armed_marker_identity")
                != list(sidecar_identity)
                or any(
                    parsed.recovery_arm.get(field) != value
                    for field, value in arm_record.items()
                    if field not in {
                        "recorded_at",
                        "recovery_nonce",
                        "absence_window_ns",
                        "absence_first_at",
                        "absence_last_at",
                    }
                )
            ):
                raise CampaignPlanError(
                    "recovery arm sidecar does not bind the marker prefix"
                )
            installed_arm = dict(parsed.recovery_arm)
    if sidecar_payload is None:
        descriptor: int | None = None
        try:
            descriptor = os.open(
                sidecar,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL
                | getattr(os, "O_NOFOLLOW", 0),
                0o600,
                dir_fd=marker.root_fd,
            )
            os.fchmod(descriptor, 0o600)
            metadata = os.fstat(descriptor)
            path_metadata = os.stat(
                sidecar, dir_fd=marker.root_fd, follow_symlinks=False
            )
            sidecar_identity = (metadata.st_dev, metadata.st_ino)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_nlink != 1
                or metadata.st_uid != os.geteuid()
                or stat.S_IMODE(metadata.st_mode) != 0o600
                or sidecar_identity
                != (path_metadata.st_dev, path_metadata.st_ino)
            ):
                raise CampaignPlanError(
                    "new recovery arm sidecar is not unaliased"
                )
            installed_arm = {
                **arm_record,
                "armed_marker_identity": list(sidecar_identity),
                "schema": marker.schema,
                "dispatch_id": marker.dispatch_id,
            }
            desired = prefix + RebootRecovery.canonical_json_line(
                installed_arm
            )
            offset = 0
            while offset < len(desired):
                written = os.write(descriptor, desired[offset:])
                if written <= 0:
                    raise OSError("short write preparing recovery arm")
                offset += written
            os.fsync(descriptor)
        except OSError as exc:
            raise CampaignPlanError(
                "could not durably prepare the recovery arm sidecar"
            ) from exc
        finally:
            if descriptor is not None:
                os.close(descriptor)
        sidecar_payload = desired
    if installed_arm is None or sidecar_identity is None:
        raise CampaignPlanError("recovery arm sidecar authority is unavailable")
    try:
        _durably_reseal_staged_file_at(
            marker.root_fd,
            sidecar,
            expected_payload=sidecar_payload,
            expected_identity=sidecar_identity,
            label="post-reboot arm sidecar",
        )
        descriptor, observed, observed_identity = _read_unaliased_small_file_at(
            marker.root_fd, sidecar, label="post-reboot arm sidecar"
        )
        os.close(descriptor)
        if (
            observed != sidecar_payload
            or observed_identity != sidecar_identity
        ):
            raise CampaignPlanError("post-reboot arm sidecar bytes changed")
        reparsed = marker_parser(
            observed, require_recovery_arm=True
        )
        if reparsed.recovery_arm != installed_arm:
            raise CampaignPlanError(
                "post-reboot arm sidecar receipt changed"
            )
        _validate_recovery_marker_seal(marker)
        os.replace(
            sidecar,
            marker.name,
            src_dir_fd=marker.root_fd,
            dst_dir_fd=marker.root_fd,
        )
        installed = os.stat(
            marker.name, dir_fd=marker.root_fd, follow_symlinks=False
        )
        if (
            not stat.S_ISREG(installed.st_mode)
            or installed.st_nlink != 1
            or installed.st_uid != os.geteuid()
            or stat.S_IMODE(installed.st_mode) != 0o600
            or (installed.st_dev, installed.st_ino) != sidecar_identity
        ):
            raise CampaignPlanError(
                "installed recovery arm marker identity changed"
            )
        os.fsync(marker.root_fd)
        sealed = os.stat(
            marker.name, dir_fd=marker.root_fd, follow_symlinks=False
        )
        if (sealed.st_dev, sealed.st_ino) != sidecar_identity:
            raise CampaignPlanError(
                "installed recovery arm marker changed after fsync"
            )
    except (OSError, RebootRecovery.RecoveryEvidenceError) as exc:
        if isinstance(exc, CampaignPlanError):
            raise
        raise CampaignPlanError(
            "could not atomically install the recovery arm"
        ) from exc
    return installed_arm


def _validate_post_reboot_dispatch_binding(
    item: dict[str, Any], parsed: RebootRecovery.ParsedMarker
) -> None:
    """Bind the immutable marker receipt to exactly one projected plan item."""

    armed = parsed.armed
    unquiesced = parsed.unquiesced
    item_payload = json.dumps(
        item, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    artifact_root = _artifact_root(item)
    canonical_root = artifact_root / f"{item['game']}_legs"
    ledger = _ledger_path(item["argv"], cwd=_runner_cwd(item))
    frontier = {
        field: item[field]
        for field in (
            *Status.FRONTIER_BINDING_FIELDS,
            "game",
            "reached",
            "target_level",
            "parent_action_count",
        )
    }
    # The dispatch row seals the target WIP *before* the child starts.  An
    # unquiesced child may subsequently leave an isolated attempt and replace
    # latest.json, so recovery must never re-project this historical field
    # from the live post-incident WIP.  The marker is the authority for the
    # digest; the immutable plan still determines the one permissible path.
    marker_wip = armed.get("target_wip_snapshot")
    historical_wip = [
        os.fspath(_target_wip_level(item)),
        marker_wip[1] if isinstance(marker_wip, list) and len(marker_wip) == 2
        else None,
    ]
    expected = {
        "game": item["game"],
        "target_level": item["target_level"],
        "tag": _single_cli_value(item["argv"], "--tag"),
        "run_label": f"{item['game']}:L{item['target_level']}:propose",
        "retry_complexity_n": item["retry_complexity_n"],
        "artifact_root": os.fspath(artifact_root),
        "canonical_root": os.fspath(canonical_root),
        "frontier_binding": frontier,
        "target_wip_snapshot": historical_wip,
        "ledger": os.fspath(ledger),
        "projected_item_sha256": hashlib.sha256(item_payload).hexdigest(),
        "historical_runner": item.get("historical_runner"),
    }
    mismatched = sorted(
        field for field, value in expected.items() if armed.get(field) != value
    )
    if mismatched:
        raise CampaignPlanError(
            "dispatch quarantine does not bind the projected plan item: "
            f"{mismatched}"
        )
    workspace = _safe_component(unquiesced.get("workspace"), "workspace")
    protected = _safe_component(
        unquiesced.get("protected"), "protected workspace"
    )
    transcript = _safe_component(unquiesced.get("transcript"), "transcript")
    expected_prefix = _dispatch_workspace_prefix(item)
    if (
        not workspace.startswith(expected_prefix)
        or workspace == expected_prefix
        or protected != workspace
        or not transcript.startswith("codex_turn_")
        or not transcript.endswith(".jsonl")
    ):
        raise CampaignPlanError(
            "dispatch quarantine generation names do not bind the dispatch"
        )


def _reconstruct_historical_recovery_item(
    item: dict[str, Any], authority: dict[str, Any], *,
    allow_abandoned_scratch: bool = False,
) -> dict[str, Any]:
    """Rebuild the exact item committed by a marker/operator receipt.

    Plan-level runner projection is operational metadata and can be absent
    from a later plan file.  Recovery therefore takes that projection only
    from the authenticated historical receipt, while every other item field
    remains supplied by (and immutable in) the selected plan item.  The
    marker's canonical item hash is the final equality check.
    """

    if "historical_runner" not in authority:
        # Completed v1 operator receipts commit the already-projected item by
        # hash but intentionally do not duplicate the runner receipt.  The
        # selected plan must therefore still supply that exact projection.
        projected = item
    else:
        historical_runner = authority.get("historical_runner")
        projected = _project_runner_receipt(
            (
                {"runner_receipt": historical_runner}
                if historical_runner is not None
                else {}
            ),
            item,
            allow_abandoned_scratch=allow_abandoned_scratch,
        )
    expected_sha = authority.get("projected_item_sha256")
    observed_sha = hashlib.sha256(json.dumps(
        projected, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()
    if observed_sha != expected_sha:
        raise CampaignPlanError(
            "historical recovery projection does not match the sealed "
            "projected item"
        )
    return projected


def _recovery_record_sha256(record: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(
        record, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()


def _recovery_recorded_at(record: dict[str, Any], label: str) -> datetime:
    value = record.get("recorded_at")
    if not isinstance(value, str) or not value:
        raise CampaignPlanError(f"{label} recorded_at is malformed")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise CampaignPlanError(f"{label} recorded_at is malformed") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise CampaignPlanError(f"{label} recorded_at is not timezone-aware")
    return parsed.astimezone(timezone.utc)


def _build_zero_ledger_event(
    item: dict[str, Any], parsed: RebootRecovery.ParsedMarker
) -> dict[str, Any]:
    observed = parsed.unquiesced
    armed = parsed.armed
    event: dict[str, Any] = {
        "event": ZERO_LEDGER_EVENT,
        "schema": ZERO_LEDGER_EVENT_SCHEMA,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "infrastructure_authority": (
            "scheduler_quiesced_zero_ledger_suffix_v1"
        ),
        "dispatch_id": parsed.dispatch_id,
        "game": item["game"],
        "target_level": item["target_level"],
        "reached": item["reached"],
        "parent_action_count": item["parent_action_count"],
        "retry_complexity_n": item["retry_complexity_n"],
        "workspace": observed["workspace"],
        "workspace_identity": observed["workspace_identity"],
        "protected_identity": observed["protected_identity"],
        "transcript": observed["transcript"],
        "protected_transcript_sha256": observed[
            "protected_transcript_sha256"
        ],
        "child_returncode": observed["child_returncode"],
        "failure_class": "infrastructure",
        "failure_detail_class": "interrupted_before_codex_exec_append",
        "terminal_errors": [observed["reason"]],
        "taint_verdict": "quarantined",
        "retry_increment": 0,
        "codex_exec_appended": False,
        "process_tree_quiesced": True,
        "wip_restore_logical_state_sha256": armed[
            "wip_restore_logical_state_sha256"
        ],
        "canonical_digest": armed["canonical_digest"],
        **{
            field: item[field]
            for field in Status.FRONTIER_BINDING_FIELDS
        },
    }
    if observed.get("diagnostics") is not None:
        event.update({
            "diagnostics": observed["diagnostics"],
            "protected_diagnostics_sha256": observed[
                "protected_diagnostics_sha256"
            ],
        })
    return event


def _validate_zero_ledger_event(
    item: dict[str, Any],
    parsed: RebootRecovery.ParsedMarker,
    record: dict[str, Any],
) -> None:
    expected_keys = ZERO_LEDGER_EVENT_BASE_KEYS | (
        ZERO_LEDGER_EVENT_DIAGNOSTICS_KEYS
        if parsed.unquiesced.get("diagnostics") is not None
        else frozenset()
    )
    if set(record) != expected_keys:
        raise CampaignPlanError(
            "zero-ledger infrastructure event has an invalid exact schema"
        )
    expected = _build_zero_ledger_event(item, parsed)
    expected.pop("recorded_at")
    observed = dict(record)
    observed.pop("recorded_at", None)
    if observed != expected:
        raise CampaignPlanError(
            "zero-ledger infrastructure event binding changed"
        )
    _recovery_recorded_at(record, "zero-ledger infrastructure event")


def _build_sandbox_abandon_event(
    item: dict[str, Any], parsed: RebootRecovery.ParsedMarker
) -> dict[str, Any]:
    arm = parsed.recovery_arm
    if arm is None:
        raise CampaignPlanError(
            "sandbox-isolated generation lacks its operator arm"
        )
    failed = parsed.unquiesced
    return {
        "event": SANDBOX_ABANDON_EVENT,
        "schema": SANDBOX_ABANDON_EVENT_SCHEMA,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "isolation_authority": (
            "explicit_operator_assumed_artifact_isolation_v1"
        ),
        "operator_provenance_assumption": arm[
            "operator_provenance_assumption"
        ],
        "sandbox_contract_sha256": arm["sandbox_contract_sha256"],
        "dispatch_id": parsed.dispatch_id,
        "recovery_nonce": arm["recovery_nonce"],
        "game": item["game"],
        "target_level": item["target_level"],
        "reached": item["reached"],
        "parent_action_count": item["parent_action_count"],
        "retry_complexity_n": item["retry_complexity_n"],
        "scratch_root": arm["scratch_root"],
        "scratch_root_identity": arm["scratch_root_identity"],
        "scratch_root_disposition": "abandoned_in_place",
        "required_retry_scratch_relation": (
            "outside_abandoned_path_and_inode"
        ),
        "workspace": failed["workspace"],
        "workspace_identity": failed["workspace_identity"],
        "protected_identity": failed["protected_identity"],
        "transcript": failed["transcript"],
        "child_returncode": None,
        "failure_class": "infrastructure",
        "failure_detail_class": "sandbox_isolated_nonquiescent",
        "terminal_errors": [failed["reason"]],
        "taint_verdict": "quarantined",
        "retry_increment": 0,
        "codex_exec_appended": False,
        "process_tree_quiesced": False,
        "detached_processes_proven_absent": False,
        "wip_restore_logical_state_sha256": parsed.armed[
            "wip_restore_logical_state_sha256"
        ],
        "canonical_digest": parsed.armed["canonical_digest"],
        **{
            field: item[field]
            for field in Status.FRONTIER_BINDING_FIELDS
        },
    }


def _validate_sandbox_abandon_event(
    item: dict[str, Any],
    parsed_or_record: RebootRecovery.ParsedMarker | dict[str, Any],
    record: dict[str, Any] | None = None,
) -> None:
    """Validate the exact nonquiescent, noncounting isolation receipt."""

    if record is None:
        # Release-WAL replay has the event and authority coordinate but no
        # mutable marker parser object.  Validate its exact public semantics.
        observed = parsed_or_record
        if not isinstance(observed, dict):
            raise CampaignPlanError("sandbox isolation event is malformed")
        expected_coordinate = {
            field: item.get(field)
            for field in (
                "game",
                "target_level",
                "reached",
                "parent_action_count",
                "retry_complexity_n",
                *Status.FRONTIER_BINDING_FIELDS,
            )
        }
        fixed = {
            "event": SANDBOX_ABANDON_EVENT,
            "schema": SANDBOX_ABANDON_EVENT_SCHEMA,
            "isolation_authority": (
                "explicit_operator_assumed_artifact_isolation_v1"
            ),
            "operator_provenance_assumption": (
                "historical_codex_workspace_write_effective_as_invoked"
            ),
            "scratch_root_disposition": "abandoned_in_place",
            "required_retry_scratch_relation": (
                "outside_abandoned_path_and_inode"
            ),
            "child_returncode": None,
            "failure_class": "infrastructure",
            "failure_detail_class": "sandbox_isolated_nonquiescent",
            "taint_verdict": "quarantined",
            "retry_increment": 0,
            "codex_exec_appended": False,
            "process_tree_quiesced": False,
            "detached_processes_proven_absent": False,
        }
        historical = item.get("historical_runner")
        expected_scratch: str | None = None
        expected_contract: str | None = None
        if isinstance(historical, dict):
            scratch_value = historical.get("scratch_root")
            source_sha256 = historical.get("source_sha256")
            if isinstance(scratch_value, str):
                expected_scratch = os.fspath(
                    _normalized_absolute_path(
                        scratch_value, "sandbox event scratch_root"
                    )
                )
            if isinstance(source_sha256, str):
                expected_contract = SANDBOX_CONTRACTS.get(source_sha256)
        terminal_errors = observed.get("terminal_errors")
        if set(observed) != SANDBOX_ABANDON_EVENT_KEYS or any((
            any(
                observed.get(field) != value
                for field, value in expected_coordinate.items()
            ),
            any(
                observed.get(field) != value
                for field, value in fixed.items()
            ),
            not isinstance(observed.get("dispatch_id"), str),
            isinstance(observed.get("dispatch_id"), str)
            and RebootRecovery.DISPATCH_ID_RE.fullmatch(
                observed["dispatch_id"]
            ) is None,
            not isinstance(observed.get("recovery_nonce"), str),
            isinstance(observed.get("recovery_nonce"), str)
            and RebootRecovery.DISPATCH_ID_RE.fullmatch(
                observed["recovery_nonce"]
            ) is None,
            observed.get("sandbox_contract_sha256")
            not in set(SANDBOX_CONTRACTS.values()),
            expected_contract is not None
            and observed.get("sandbox_contract_sha256")
            != expected_contract,
            expected_scratch is not None
            and observed.get("scratch_root") != expected_scratch,
            not isinstance(terminal_errors, list),
            isinstance(terminal_errors, list)
            and (
                len(terminal_errors) != 1
                or not isinstance(terminal_errors[0], str)
                or not terminal_errors[0]
            ),
            not isinstance(observed.get("wip_restore_logical_state_sha256"), str),
            isinstance(observed.get("wip_restore_logical_state_sha256"), str)
            and SHA256_RE.fullmatch(
                observed["wip_restore_logical_state_sha256"]
            ) is None,
            not isinstance(observed.get("canonical_digest"), str),
            isinstance(observed.get("canonical_digest"), str)
            and SHA256_RE.fullmatch(observed["canonical_digest"]) is None,
        )):
            raise CampaignPlanError(
                "sandbox-isolated infrastructure event binding changed"
            )
        _normalized_absolute_path(
            observed.get("scratch_root"), "sandbox event scratch_root"
        )
        _marker_identity(
            observed.get("scratch_root_identity"), "sandbox scratch root"
        )
        _safe_component(observed.get("workspace"), "sandbox workspace")
        _marker_identity(
            observed.get("workspace_identity"), "sandbox workspace"
        )
        _marker_identity(
            observed.get("protected_identity"), "sandbox protected evidence"
        )
        _safe_component(observed.get("transcript"), "sandbox transcript")
        try:
            Status.validate_frontier_binding({
                field: observed[field]
                for field in (
                    *Status.FRONTIER_BINDING_FIELDS,
                    "game",
                    "reached",
                    "target_level",
                    "parent_action_count",
                )
            })
        except (KeyError, TypeError, ValueError) as exc:
            raise CampaignPlanError(
                "sandbox isolation event frontier is invalid"
            ) from exc
        _recovery_recorded_at(observed, "sandbox isolation event")
        return
    parsed = parsed_or_record
    if not isinstance(parsed, RebootRecovery.ParsedMarker):
        raise CampaignPlanError("sandbox isolation marker is malformed")
    if set(record) != SANDBOX_ABANDON_EVENT_KEYS:
        raise CampaignPlanError(
            "sandbox-isolated infrastructure event has an invalid schema"
        )
    expected = _build_sandbox_abandon_event(item, parsed)
    expected.pop("recorded_at")
    observed = dict(record)
    observed.pop("recorded_at", None)
    if observed != expected:
        raise CampaignPlanError(
            "sandbox-isolated infrastructure event binding changed"
        )
    _recovery_recorded_at(record, "sandbox isolation event")


def _sandbox_isolation_result(
    item: dict[str, Any], parsed: RebootRecovery.ParsedMarker
) -> dict[str, Any]:
    arm = parsed.recovery_arm
    if arm is None:
        raise CampaignPlanError("sandbox isolation result lacks its arm")
    return {
        "game": item["game"],
        "target_level": item["target_level"],
        "reached": item["reached"],
        "result": "sandbox_isolated_noncounting",
        "reason": parsed.unquiesced["reason"],
        "dispatch_id": parsed.dispatch_id,
        "retry_complexity_n": item["retry_complexity_n"],
        "seed_mode": item["seed_mode"],
        "wip_mode": item["wip_mode"],
        "lineage_input_mode": item["lineage_input_mode"],
        "scratch_root": arm["scratch_root"],
        "scratch_root_disposition": "abandoned_in_place",
        "process_tree_quiesced": False,
        "detached_processes_proven_absent": False,
    }


def _validate_sandbox_isolation_result(
    item: dict[str, Any],
    event: dict[str, Any],
    result: dict[str, Any],
) -> None:
    """Cross-bind the noncounting result to its exact abandonment row."""

    terminal_errors = event.get("terminal_errors")
    reason = (
        terminal_errors[0]
        if isinstance(terminal_errors, list) and len(terminal_errors) == 1
        else None
    )
    expected = {
        "game": event.get("game"),
        "target_level": event.get("target_level"),
        "reached": event.get("reached"),
        "result": "sandbox_isolated_noncounting",
        "reason": reason,
        "dispatch_id": event.get("dispatch_id"),
        "retry_complexity_n": event.get("retry_complexity_n"),
        "scratch_root": event.get("scratch_root"),
        "scratch_root_disposition": event.get("scratch_root_disposition"),
        "process_tree_quiesced": False,
        "detached_processes_proven_absent": False,
    }
    seed_mode = result.get("seed_mode")
    wip_mode = result.get("wip_mode")
    if (
        set(result) != SANDBOX_ISOLATION_RESULT_KEYS
        or reason is None
        or any(result.get(field) != value for field, value in expected.items())
        or seed_mode not in {"zero_seed", "verified_parent"}
        or wip_mode not in {"exclude", "restore_clean_same_frontier"}
        or result.get("lineage_input_mode") != f"{seed_mode}+{wip_mode}"
        or any(
            item.get(field) is not None
            and result.get(field) != item.get(field)
            for field in ("seed_mode", "wip_mode", "lineage_input_mode")
        )
    ):
        raise CampaignPlanError(
            "sandbox isolation terminal result binding changed"
        )


def _validate_recovery_correction(
    item: dict[str, Any],
    record: dict[str, Any],
    correction: dict[str, Any],
    *,
    not_before: datetime,
) -> None:
    keys = {
        "event", "recorded_at", "classification_authority", "thread_id",
        "transcript", "workspace", "game", "target_level",
        "failure_class", "failure_detail_class", "terminal_errors",
        "solved_target", "taint_verdict", "retry_increment",
        "protected_transcript_sha256", *Status.FRONTIER_BINDING_FIELDS,
        "reached", "parent_action_count",
    }
    diagnostics = record.get("diagnostics")
    if diagnostics is not None:
        keys.update({"diagnostics", "protected_diagnostics_sha256"})
    if set(correction) != keys:
        raise CampaignPlanError(
            "post-reboot correction row has an invalid exact schema"
        )
    errors = correction.get("terminal_errors")
    if (
        not isinstance(errors, list)
        or len(errors) != 1
        or not isinstance(errors[0], str)
        or not errors[0]
    ):
        raise CampaignPlanError("post-reboot correction reason is malformed")
    reason = errors[0]
    expected_detail = (
        "host_process_introspection"
        if (
            "host process introspection" in reason.lower()
            or "host_process_introspection" in reason.lower()
        )
        else "post_proposer_workspace_taint"
    )
    expected = {
        "event": "codex_exec_classification_correction",
        "classification_authority": "scheduler_exact_generation_taint_scan_v1",
        "thread_id": record.get("thread_id"),
        "transcript": record.get("transcript"),
        "workspace": record.get("workspace"),
        "game": item["game"],
        "target_level": item["target_level"],
        "failure_class": "taint",
        "failure_detail_class": expected_detail,
        "solved_target": None,
        "taint_verdict": "tainted",
        "retry_increment": 0,
        "protected_transcript_sha256": record.get(
            "protected_transcript_sha256"
        ),
        **{
            field: item[field]
            for field in (
                *Status.FRONTIER_BINDING_FIELDS,
                "reached",
                "parent_action_count",
            )
        },
    }
    if diagnostics is not None:
        expected.update({
            "diagnostics": diagnostics,
            "protected_diagnostics_sha256": record.get(
                "protected_diagnostics_sha256"
            ),
        })
    if any(correction.get(field) != value for field, value in expected.items()):
        raise CampaignPlanError(
            "post-reboot correction does not bind the exact generation"
        )
    _recovery_recorded_at(correction, "post-reboot correction")


def _build_post_reboot_correction(
    item: dict[str, Any],
    record: dict[str, Any],
    *,
    reason: str,
    transcript_sha: str,
    diagnostics_sha: str | None,
) -> dict[str, Any]:
    correction = {
        "event": "codex_exec_classification_correction",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "classification_authority": "scheduler_exact_generation_taint_scan_v1",
        "thread_id": record.get("thread_id"),
        "transcript": record["transcript"],
        "workspace": record["workspace"],
        "game": item["game"],
        "target_level": item["target_level"],
        "failure_class": "taint",
        "failure_detail_class": (
            "host_process_introspection"
            if (
                "host process introspection" in reason.lower()
                or "host_process_introspection" in reason.lower()
            )
            else "post_proposer_workspace_taint"
        ),
        "terminal_errors": [reason],
        "solved_target": None,
        "taint_verdict": "tainted",
        "retry_increment": 0,
        "protected_transcript_sha256": transcript_sha,
        **{
            field: item[field]
            for field in (
                *Status.FRONTIER_BINDING_FIELDS,
                "reached",
                "parent_action_count",
            )
        },
    }
    if diagnostics_sha is not None:
        correction["diagnostics"] = record["diagnostics"]
        correction["protected_diagnostics_sha256"] = diagnostics_sha
    return correction


def _validate_recovery_cleanup(
    item: dict[str, Any],
    record: dict[str, Any],
    cleanup: dict[str, Any],
    *,
    not_before: datetime,
) -> None:
    keys = {
        "event", "recorded_at", "cleanup_authority", "thread_id",
        "transcript", "workspace", "game", "target_level",
        "retry_increment", *Status.FRONTIER_BINDING_FIELDS,
        "reached", "parent_action_count",
    }
    if record.get("diagnostics") is not None:
        keys.add("diagnostics")
    discard_keys = {
        "wip_recovery_authority",
        "confirmed_current_wip_state_sha256",
        "wip_disposition",
        "discard_survivor_sha256",
        "restored_wip_logical_state_sha256",
    }
    has_wip_resolution = discard_keys.issubset(cleanup)
    if has_wip_resolution:
        keys.update(discard_keys)
    if set(cleanup) != keys:
        raise CampaignPlanError(
            "post-reboot cleanup row has an invalid exact schema"
        )
    expected = {
        "event": "codex_taint_cleanup_completed",
        "cleanup_authority": "scheduler_exact_generation_cleanup_v1",
        "thread_id": record.get("thread_id"),
        "transcript": record.get("transcript"),
        "workspace": record.get("workspace"),
        "game": item["game"],
        "target_level": item["target_level"],
        "retry_increment": 0,
        **{
            field: item[field]
            for field in (
                *Status.FRONTIER_BINDING_FIELDS,
                "reached",
                "parent_action_count",
            )
        },
    }
    if record.get("diagnostics") is not None:
        expected["diagnostics"] = record["diagnostics"]
    if has_wip_resolution:
        authority = cleanup.get("wip_recovery_authority")
        disposition = cleanup.get("wip_disposition")
        if (authority, disposition) not in {
            (
                "operator_confirmed_quarantined_wip_v1",
                "discard_latest_pointer",
            ),
            (
                "dispatch_full_wip_rollback_capsule_v1",
                "restore_historical_baseline",
            ),
        }:
            raise CampaignPlanError(
                "post-reboot cleanup WIP authority is malformed"
            )
        expected.update({
            "wip_recovery_authority": authority,
            "wip_disposition": disposition,
        })
        for field in ("confirmed_current_wip_state_sha256",):
            value = cleanup.get(field)
            if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
                raise CampaignPlanError(
                    "post-reboot cleanup WIP hash is malformed"
                )
        survivor = cleanup.get("discard_survivor_sha256")
        if disposition == "discard_latest_pointer":
            if not isinstance(survivor, str) or SHA256_RE.fullmatch(
                survivor
            ) is None:
                raise CampaignPlanError(
                    "post-reboot cleanup discard hash is malformed"
                )
        elif survivor is not None:
            raise CampaignPlanError(
                "post-reboot capsule cleanup has a discard hash"
            )
        restored_logical = cleanup.get(
            "restored_wip_logical_state_sha256"
        )
        if disposition == "restore_historical_baseline":
            if (
                not isinstance(restored_logical, str)
                or SHA256_RE.fullmatch(restored_logical) is None
            ):
                raise CampaignPlanError(
                    "post-reboot cleanup logical restore hash is malformed"
                )
        elif restored_logical is not None:
            raise CampaignPlanError(
                "legacy cleanup has a logical restore hash"
            )
    if any(cleanup.get(field) != value for field, value in expected.items()):
        raise CampaignPlanError(
            "post-reboot cleanup does not bind the exact generation"
        )
    _recovery_recorded_at(cleanup, "post-reboot cleanup")


def _build_post_reboot_cleanup(
    item: dict[str, Any],
    record: dict[str, Any],
    parsed: RebootRecovery.ParsedMarker,
) -> dict[str, Any]:
    cleanup = {
        "event": "codex_taint_cleanup_completed",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "cleanup_authority": "scheduler_exact_generation_cleanup_v1",
        "thread_id": record.get("thread_id"),
        "transcript": record["transcript"],
        "workspace": record["workspace"],
        "game": item["game"],
        "target_level": item["target_level"],
        "retry_increment": 0,
        **{
            field: item[field]
            for field in (
                *Status.FRONTIER_BINDING_FIELDS,
                "reached",
                "parent_action_count",
            )
        },
    }
    if record.get("diagnostics") is not None:
        cleanup["diagnostics"] = record["diagnostics"]
    arm = parsed.recovery_arm
    assert arm is not None
    if arm.get("wip_disposition") in {
        "discard_latest_pointer", "restore_historical_baseline"
    }:
        cleanup.update({
            "wip_recovery_authority": arm["wip_recovery_authority"],
            "confirmed_current_wip_state_sha256": arm[
                "confirmed_current_wip_state_sha256"
            ],
            "wip_disposition": arm["wip_disposition"],
            "discard_survivor_sha256": arm["discard_survivor_sha256"],
            "restored_wip_logical_state_sha256": arm[
                "restored_wip_logical_state_sha256"
            ],
        })
    return cleanup


RECOVERY_PHASE_INTENT_KEYS = frozenset({
    "schema", "dispatch_id", "event", "ledger",
    "intent_root", "intent_root_identity",
    "ledger_parent_identity", "ledger_file_identity",
    "expected_prefix_bytes", "expected_prefix_sha256",
    "record_sha256", "record",
})


def _recovery_phase_intent_names(
    ledger: Path, dispatch_id: str
) -> dict[str, str]:
    if RebootRecovery.DISPATCH_ID_RE.fullmatch(dispatch_id) is None:
        raise CampaignPlanError("recovery phase intent dispatch ID is malformed")
    ledger_key = hashlib.sha256(
        os.fsencode(os.fspath(ledger))
    ).hexdigest()[:16]
    return {
        event: (
            f".codex_recovery_{dispatch_id}_{ledger_key}_{phase}.intent"
        )
        for event, phase in RECOVERY_PHASE_EVENTS.items()
    }


def _open_recovery_intent_parent(
    intent_root: Path, expected_identity: tuple[int, int]
) -> int:
    descriptor = os.open(
        intent_root,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    metadata = os.fstat(descriptor)
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or (metadata.st_dev, metadata.st_ino) != expected_identity
    ):
        os.close(descriptor)
        raise CampaignPlanError("recovery intent parent identity changed")
    return descriptor


def _parse_recovery_phase_intent(
    raw: bytes, *, label: str
) -> dict[str, Any]:
    try:
        rows = RebootRecovery.parse_canonical_jsonl(raw, label=label)
    except RebootRecovery.RecoveryEvidenceError as exc:
        raise CampaignPlanError(str(exc)) from exc
    if len(rows) != 1 or set(rows[0]) != RECOVERY_PHASE_INTENT_KEYS:
        raise CampaignPlanError(f"{label} has an invalid exact schema")
    intent = dict(rows[0])
    record = intent.get("record")
    event = intent.get("event")
    prefix_bytes = intent.get("expected_prefix_bytes")
    if (
        intent.get("schema") != RECOVERY_PHASE_INTENT_SCHEMA
        or not isinstance(intent.get("dispatch_id"), str)
        or RebootRecovery.DISPATCH_ID_RE.fullmatch(intent["dispatch_id"])
        is None
        or event not in RECOVERY_PHASE_EVENTS
        or not isinstance(intent.get("ledger"), str)
        or not isinstance(intent.get("intent_root"), str)
        or not isinstance(prefix_bytes, int)
        or isinstance(prefix_bytes, bool)
        or prefix_bytes < 0
        or not isinstance(record, dict)
        or record.get("event") != event
        or not isinstance(intent.get("expected_prefix_sha256"), str)
        or SHA256_RE.fullmatch(intent["expected_prefix_sha256"]) is None
        or not isinstance(intent.get("record_sha256"), str)
        or SHA256_RE.fullmatch(intent["record_sha256"]) is None
        or intent["record_sha256"] != _recovery_record_sha256(record)
    ):
        raise CampaignPlanError(f"{label} binding is malformed")
    _marker_identity(
        intent.get("ledger_parent_identity"), "intent ledger parent"
    )
    _marker_identity(intent.get("ledger_file_identity"), "intent ledger")
    _marker_identity(intent.get("intent_root_identity"), "intent root")
    _recovery_recorded_at(record, "recovery phase intent record")
    return intent


def _fsync_and_revalidate_installed_intent(
    parent_fd: int,
    name: str,
    *,
    payload: bytes,
    identity: tuple[int, int],
    label: str,
) -> None:
    """Make a visible installed intent durable before it authorizes mutation."""

    try:
        os.fsync(parent_fd)
    except OSError as exc:
        raise CampaignPlanError(
            f"could not durably confirm {label}"
        ) from exc
    descriptor, observed, observed_identity = _read_unaliased_small_file_at(
        parent_fd, name, label=label
    )
    os.close(descriptor)
    if observed != payload or observed_identity != identity:
        raise CampaignPlanError(f"{label} changed after parent fsync")


def _phase_intent_envelope(
    state: PostRebootLedgerState,
    record: dict[str, Any],
    expected_raw: bytes,
) -> dict[str, Any]:
    if record.get("event") not in RECOVERY_PHASE_EVENTS:
        raise CampaignPlanError("recovery phase event is not journalled")
    if state.baseline.file_identity is None:
        raise CampaignPlanError("recovery phase ledger identity is unavailable")
    if state.intent_root is None or state.intent_root_identity is None:
        raise CampaignPlanError("recovery phase intent custody is unavailable")
    return {
        "schema": RECOVERY_PHASE_INTENT_SCHEMA,
        "dispatch_id": state.dispatch_id,
        "event": record["event"],
        "ledger": os.fspath(state.ledger),
        "intent_root": os.fspath(state.intent_root),
        "intent_root_identity": list(state.intent_root_identity),
        "ledger_parent_identity": list(state.baseline.parent_identity),
        "ledger_file_identity": list(state.baseline.file_identity),
        "expected_prefix_bytes": len(expected_raw),
        "expected_prefix_sha256": hashlib.sha256(expected_raw).hexdigest(),
        "record_sha256": _recovery_record_sha256(record),
        "record": record,
    }


def _phase_intents_equivalent(
    observed: dict[str, Any], expected: dict[str, Any]
) -> bool:
    ignored_envelope_fields = {"record", "record_sha256"}
    if {
        key: value for key, value in observed.items()
        if key not in ignored_envelope_fields
    } != {
        key: value for key, value in expected.items()
        if key not in ignored_envelope_fields
    }:
        return False
    observed_record = dict(observed.get("record", {}))
    expected_record = dict(expected.get("record", {}))
    observed_record.pop("recorded_at", None)
    expected_record.pop("recorded_at", None)
    if observed.get("event") != (
        "codex_post_reboot_operator_recovery_completed"
    ):
        return observed_record == expected_record
    observed_boot = observed_record.pop("current_boot_identity", None)
    expected_boot = expected_record.pop("current_boot_identity", None)
    if observed_record != expected_record:
        return False
    armed_boot = expected_record.get("armed_boot_identity")

    def changed_boot_receipt(value: object) -> bool:
        return (
            isinstance(armed_boot, dict)
            and set(armed_boot) == {"source", "identity_sha256"}
            and isinstance(armed_boot.get("source"), str)
            and isinstance(armed_boot.get("identity_sha256"), str)
            and SHA256_RE.fullmatch(armed_boot["identity_sha256"]) is not None
            and isinstance(value, dict)
            and set(value) == {"source", "identity_sha256"}
            and value.get("source") == armed_boot.get("source")
            and isinstance(value.get("identity_sha256"), str)
            and SHA256_RE.fullmatch(value["identity_sha256"]) is not None
            and value != armed_boot
        )

    return changed_boot_receipt(observed_boot) and changed_boot_receipt(
        expected_boot
    )


def _prepare_recovery_phase_intent_locked(
    state: PostRebootLedgerState,
    record: dict[str, Any],
    expected_raw: bytes,
) -> tuple[str, dict[str, Any]]:
    expected = _phase_intent_envelope(state, record, expected_raw)
    names = _recovery_phase_intent_names(state.ledger, state.dispatch_id)
    intent_name = names[record["event"]]
    temporary_name = f"{intent_name}.preparing"
    assert state.intent_root is not None
    assert state.intent_root_identity is not None
    parent_fd = _open_recovery_intent_parent(
        state.intent_root, state.intent_root_identity
    )
    try:
        for other_name in names.values():
            try:
                os.stat(other_name, dir_fd=parent_fd, follow_symlinks=False)
            except FileNotFoundError:
                continue
            raise CampaignPlanError(
                "a recovery phase intent already requires reconciliation"
            )
        selected = expected
        payload: bytes | None = None
        try:
            descriptor, payload, _identity = _read_unaliased_small_file_at(
                parent_fd,
                temporary_name,
                label="recovery phase preparing intent",
            )
        except FileNotFoundError:
            descriptor = None
        else:
            os.close(descriptor)
            try:
                candidate = _parse_recovery_phase_intent(
                    payload, label="recovery phase preparing intent"
                )
            except CampaignPlanError:
                try:
                    os.unlink(temporary_name, dir_fd=parent_fd)
                    os.fsync(parent_fd)
                except OSError as exc:
                    raise CampaignPlanError(
                        "could not discard a partial recovery phase intent"
                    ) from exc
                payload = None
            else:
                if not _phase_intents_equivalent(candidate, expected):
                    raise CampaignPlanError(
                        "recovery phase preparing intent changed"
                    )
                selected = candidate
        if payload is None:
            desired = RebootRecovery.canonical_json_line(selected)
            descriptor = None
            try:
                descriptor = os.open(
                    temporary_name,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL
                    | getattr(os, "O_NOFOLLOW", 0),
                    0o600,
                    dir_fd=parent_fd,
                )
                os.fchmod(descriptor, 0o600)
                offset = 0
                while offset < len(desired):
                    written = os.write(descriptor, desired[offset:])
                    if written <= 0:
                        raise OSError("short write preparing phase intent")
                    offset += written
                os.fsync(descriptor)
            except OSError as exc:
                raise CampaignPlanError(
                    "could not durably prepare the recovery phase intent"
                ) from exc
            finally:
                if descriptor is not None:
                    os.close(descriptor)
            payload = desired
        descriptor, observed, temporary_identity = (
            _read_unaliased_small_file_at(
                parent_fd,
                temporary_name,
                label="recovery phase preparing intent",
            )
        )
        os.close(descriptor)
        if observed != payload:
            raise CampaignPlanError("recovery phase preparing intent changed")
        selected = _parse_recovery_phase_intent(
            observed, label="recovery phase preparing intent"
        )
        if not _phase_intents_equivalent(selected, expected):
            raise CampaignPlanError("recovery phase preparing intent changed")
        _durably_reseal_staged_file_at(
            parent_fd,
            temporary_name,
            expected_payload=observed,
            expected_identity=temporary_identity,
            label="recovery phase preparing intent",
        )
        try:
            os.replace(
                temporary_name,
                intent_name,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
            )
            os.fsync(parent_fd)
        except OSError as exc:
            raise CampaignPlanError(
                "could not install the recovery phase intent"
            ) from exc
        descriptor, installed, installed_identity = (
            _read_unaliased_small_file_at(
                parent_fd, intent_name, label="recovery phase intent"
            )
        )
        os.close(descriptor)
        if installed != observed or installed_identity != temporary_identity:
            raise CampaignPlanError("installed recovery phase intent changed")
        return intent_name, selected
    finally:
        os.close(parent_fd)


def _reconcile_recovery_phase_intent_locked(
    *,
    ledger: Path,
    binding: dict[str, Any],
    intent_root: Path,
    intent_root_identity: tuple[int, int],
    parent_identity: tuple[int, int],
    file_identity: tuple[int, int],
    raw: bytes,
) -> bytes:
    dispatch_id = binding.get("dispatch_id")
    if not isinstance(dispatch_id, str):
        raise CampaignPlanError("recovery ledger dispatch ID is malformed")
    names = _recovery_phase_intent_names(ledger, dispatch_id)
    parent_fd = _open_recovery_intent_parent(
        intent_root, intent_root_identity
    )
    try:
        present: list[str] = []
        for name in names.values():
            try:
                os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
            except FileNotFoundError:
                continue
            present.append(name)
        if not present:
            return raw
        if len(present) != 1:
            raise CampaignPlanError("multiple recovery phase intents are present")
        intent_name = present[0]
        descriptor, payload, intent_identity = _read_unaliased_small_file_at(
            parent_fd, intent_name, label="recovery phase intent"
        )
        os.close(descriptor)
        intent = _parse_recovery_phase_intent(
            payload, label="recovery phase intent"
        )
        expected_binding = {
            "dispatch_id": dispatch_id,
            "ledger": os.fspath(ledger),
            "intent_root": os.fspath(intent_root),
            "intent_root_identity": list(intent_root_identity),
            "ledger_parent_identity": list(parent_identity),
            "ledger_file_identity": list(file_identity),
        }
        if any(
            intent.get(field) != value
            for field, value in expected_binding.items()
        ) or names.get(intent["event"]) != intent_name:
            raise CampaignPlanError("recovery phase intent binding changed")
        _fsync_and_revalidate_installed_intent(
            parent_fd,
            intent_name,
            payload=payload,
            identity=intent_identity,
            label="recovery phase intent",
        )
        expected_bytes = int(intent["expected_prefix_bytes"])
        if expected_bytes > len(raw):
            raise CampaignPlanError("recovery phase ledger prefix was truncated")
        expected_prefix = raw[:expected_bytes]
        if hashlib.sha256(expected_prefix).hexdigest() != intent.get(
            "expected_prefix_sha256"
        ):
            raise CampaignPlanError("recovery phase ledger prefix changed")
        baseline_bytes = binding.get("ledger_prefix_bytes")
        if (
            not isinstance(baseline_bytes, int)
            or isinstance(baseline_bytes, bool)
            or not 0 <= baseline_bytes <= expected_bytes
        ):
            raise CampaignPlanError("recovery phase baseline length changed")
        try:
            phase_rows = RebootRecovery.parse_canonical_jsonl(
                expected_prefix[baseline_bytes:],
                label="recovery phase intent prefix",
            )
        except RebootRecovery.RecoveryEvidenceError as exc:
            raise CampaignPlanError(str(exc)) from exc
        expected_events = [
            "codex_exec",
            "codex_exec_classification_correction",
            "codex_taint_cleanup_completed",
            "codex_post_reboot_operator_recovery_completed",
        ]
        zero_ledger_phase = (
            intent.get("event") in {ZERO_LEDGER_EVENT, SANDBOX_ABANDON_EVENT}
            and not phase_rows
        )
        ordinary_phase = (
            1 <= len(phase_rows) < len(expected_events)
            and [row.get("event") for row in phase_rows]
            == expected_events[:len(phase_rows)]
            and intent.get("event") == expected_events[len(phase_rows)]
        )
        if not (zero_ledger_phase or ordinary_phase):
            raise CampaignPlanError(
                "recovery phase intent does not follow the ledger phase"
            )
        line = RebootRecovery.canonical_json_line(intent["record"])
        tail = raw[expected_bytes:]
        if len(tail) > len(line) or not line.startswith(tail):
            raise CampaignPlanError(
                "recovery ledger tail is not the intended phase prefix"
            )
        ledger_fd: int | None = None
        try:
            ledger_fd = os.open(
                ledger, os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
            )
            opened = os.fstat(ledger_fd)
            if (
                not stat.S_ISREG(opened.st_mode)
                or opened.st_nlink != 1
                or (opened.st_dev, opened.st_ino) != file_identity
            ):
                raise CampaignPlanError(
                    "recovery phase ledger descriptor changed"
                )
            if tail != line:
                os.ftruncate(ledger_fd, expected_bytes)
                os.fsync(ledger_fd)
                os.lseek(ledger_fd, 0, os.SEEK_END)
                offset = 0
                while offset < len(line):
                    written = os.write(ledger_fd, line[offset:])
                    if written <= 0:
                        raise OSError("short write reconciling recovery phase")
                    offset += written
            os.fsync(ledger_fd)
        except OSError as exc:
            raise CampaignPlanError(
                "could not reconcile the recovery phase ledger"
            ) from exc
        finally:
            if ledger_fd is not None:
                os.close(ledger_fd)
        sealed = Legs._read_single_link_regular(os.fspath(ledger))
        expected_sealed = expected_prefix + line
        if sealed != expected_sealed:
            raise CampaignPlanError("reconciled recovery phase is not exact")
        try:
            os.unlink(intent_name, dir_fd=parent_fd)
            os.fsync(parent_fd)
        except OSError as exc:
            raise CampaignPlanError(
                "could not retire the recovery phase intent"
            ) from exc
        return sealed
    finally:
        os.close(parent_fd)


def _read_post_reboot_ledger_surface(
    item: dict[str, Any],
    binding: dict[str, Any],
    *,
    intent_root: Path | None = None,
    intent_root_identity: tuple[int, int] | None = None,
) -> tuple[Path, LedgerPrefixState, list[dict[str, Any]]]:
    ledger = _ledger_path(item["argv"], cwd=_runner_cwd(item))
    if binding.get("ledger") != os.fspath(ledger):
        raise CampaignPlanError("dispatch quarantine ledger path changed")
    _reject_symlinked_ancestry(ledger.parent, "Codex ledger parent")
    with Guard.ledger_append_lock(ledger):
        parent_identity = _host_directory_identity(
            ledger.parent, "Codex ledger parent"
        )
        if parent_identity != _marker_identity(
            binding.get("ledger_parent_identity"), "ledger parent"
        ):
            raise CampaignPlanError("dispatch quarantine ledger parent changed")
        try:
            metadata = ledger.stat(follow_symlinks=False)
        except OSError as exc:
            raise CampaignPlanError(
                "dispatch quarantine ledger is unavailable"
            ) from exc
        if (
            ledger.is_symlink()
            or not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
        ):
            raise CampaignPlanError("dispatch quarantine ledger is aliased")
        recorded_identity = binding.get("ledger_file_identity")
        if recorded_identity is None or (
            metadata.st_dev,
            metadata.st_ino,
        ) != _marker_identity(recorded_identity, "ledger file"):
            raise CampaignPlanError("dispatch quarantine ledger inode changed")
        try:
            raw = Legs._read_single_link_regular(os.fspath(ledger))
        except (OSError, Legs.WorkspaceTainted) as exc:
            raise CampaignPlanError(
                "dispatch quarantine ledger is unstable"
            ) from exc
        prefix_bytes = binding.get("ledger_prefix_bytes")
        if (
            not isinstance(prefix_bytes, int)
            or isinstance(prefix_bytes, bool)
            or prefix_bytes < 0
            or prefix_bytes > len(raw)
        ):
            raise CampaignPlanError(
                "dispatch quarantine ledger prefix length is invalid"
            )
        prefix = raw[:prefix_bytes]
        if hashlib.sha256(prefix).hexdigest() != binding.get(
            "ledger_prefix_sha256"
        ):
            raise CampaignPlanError(
                "dispatch quarantine ledger baseline digest changed"
            )
        if (intent_root is None) != (intent_root_identity is None):
            raise CampaignPlanError(
                "recovery phase intent custody is incomplete"
            )
        if intent_root is not None and intent_root_identity is not None:
            raw = _reconcile_recovery_phase_intent_locked(
                ledger=ledger,
                binding=binding,
                intent_root=intent_root,
                intent_root_identity=intent_root_identity,
                parent_identity=parent_identity,
                file_identity=(metadata.st_dev, metadata.st_ino),
                raw=raw,
            )
        prefix = raw[:prefix_bytes]
        before_records = _strict_ledger_records(
            prefix, label="post-reboot ledger baseline"
        )
        try:
            suffix_records = RebootRecovery.parse_canonical_jsonl(
                raw[prefix_bytes:], label="post-reboot ledger suffix"
            )
        except RebootRecovery.RecoveryEvidenceError as exc:
            raise CampaignPlanError(str(exc)) from exc
    baseline = LedgerPrefixState(
        path=ledger,
        parent_identity=parent_identity,
        file_identity=(metadata.st_dev, metadata.st_ino),
        raw_prefix=prefix,
        records=before_records,
    )
    return ledger, baseline, [dict(row) for row in suffix_records]


def _post_reboot_state_rows(state: PostRebootLedgerState) -> list[dict[str, Any]]:
    rows = [state.record]
    for row in (state.correction, state.cleanup, state.operator):
        if row is not None:
            rows.append(row)
    return rows


def _append_recovery_phase_cas(
    state: PostRebootLedgerState,
    record: dict[str, Any],
    *,
    after_intent: Any | None = None,
) -> dict[str, Any]:
    """Compare the exact phase and append under one ledger-lock critical section."""

    expected_raw = state.baseline.raw_prefix + b"".join(
        RebootRecovery.canonical_json_line(row)
        for row in _post_reboot_state_rows(state)
    )
    with Guard.ledger_append_lock(state.ledger):
        if _host_directory_identity(
            state.ledger.parent, "Codex ledger parent"
        ) != state.baseline.parent_identity:
            raise CampaignPlanError(
                "post-reboot ledger parent changed before phase append"
            )
        try:
            metadata = state.ledger.stat(follow_symlinks=False)
        except OSError as exc:
            raise CampaignPlanError(
                "post-reboot ledger disappeared before phase append"
            ) from exc
        if (
            state.ledger.is_symlink()
            or not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or state.baseline.file_identity is None
            or (metadata.st_dev, metadata.st_ino)
            != state.baseline.file_identity
        ):
            raise CampaignPlanError(
                "post-reboot ledger inode changed before phase append"
            )
        try:
            raw = Legs._read_single_link_regular(os.fspath(state.ledger))
        except (OSError, Legs.WorkspaceTainted) as exc:
            raise CampaignPlanError(
                "post-reboot ledger is unstable before phase append"
            ) from exc
        if raw != expected_raw:
            raise CampaignPlanError(
                "post-reboot ledger phase changed before append"
            )
        intent_name, selected_intent = _prepare_recovery_phase_intent_locked(
            state, record, expected_raw
        )
        selected_record = dict(selected_intent["record"])
        line = RebootRecovery.canonical_json_line(selected_record)
        if after_intent is not None:
            after_intent()
        descriptor: int | None = None
        try:
            descriptor = os.open(
                state.ledger,
                os.O_WRONLY | os.O_APPEND | getattr(os, "O_NOFOLLOW", 0),
            )
            opened = os.fstat(descriptor)
            if (
                not stat.S_ISREG(opened.st_mode)
                or opened.st_nlink != 1
                or (opened.st_dev, opened.st_ino)
                != state.baseline.file_identity
            ):
                raise CampaignPlanError(
                    "post-reboot ledger descriptor changed before append"
                )
            offset = 0
            while offset < len(line):
                written = os.write(descriptor, line[offset:])
                if written <= 0:
                    raise OSError("short write appending recovery phase")
                offset += written
            os.fsync(descriptor)
            after = os.fstat(descriptor)
            if (
                (after.st_dev, after.st_ino) != state.baseline.file_identity
                or after.st_size != len(expected_raw) + len(line)
            ):
                raise CampaignPlanError(
                    "post-reboot ledger phase append was not exact"
                )
        except OSError as exc:
            raise CampaignPlanError(
                "post-reboot ledger phase append failed"
            ) from exc
        finally:
            if descriptor is not None:
                os.close(descriptor)
        try:
            sealed = Legs._read_single_link_regular(os.fspath(state.ledger))
        except (OSError, Legs.WorkspaceTainted) as exc:
            raise CampaignPlanError(
                "post-reboot ledger is unstable after phase append"
            ) from exc
        if sealed != expected_raw + line:
            raise CampaignPlanError(
                "post-reboot ledger phase append bytes are not exact"
            )
        assert state.intent_root is not None
        assert state.intent_root_identity is not None
        parent_fd = _open_recovery_intent_parent(
            state.intent_root, state.intent_root_identity
        )
        try:
            os.unlink(intent_name, dir_fd=parent_fd)
            os.fsync(parent_fd)
        except OSError as exc:
            raise CampaignPlanError(
                "could not retire the appended recovery phase intent"
            ) from exc
        finally:
            os.close(parent_fd)
    return selected_record


def _append_zero_ledger_event_cas(
    *,
    marker: DispatchQuarantine,
    baseline: LedgerPrefixState,
    record: dict[str, Any],
    after_intent: Any | None = None,
) -> dict[str, Any]:
    """Append the first zero-ledger phase behind the recovery intent WAL."""

    state = PostRebootLedgerState(
        dispatch_id=marker.dispatch_id,
        intent_root=marker.root,
        intent_root_identity=marker.root_identity,
        ledger=baseline.path,
        baseline=baseline,
        record={},
        correction=None,
        cleanup=None,
        operator=None,
    )
    expected_raw = baseline.raw_prefix
    with Guard.ledger_append_lock(baseline.path):
        if (
            _host_directory_identity(
                baseline.path.parent, "Codex ledger parent"
            ) != baseline.parent_identity
            or baseline.file_identity is None
        ):
            raise CampaignPlanError(
                "zero-ledger event lost its ledger custody"
            )
        metadata = baseline.path.stat(follow_symlinks=False)
        if (
            baseline.path.is_symlink()
            or not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or (metadata.st_dev, metadata.st_ino) != baseline.file_identity
        ):
            raise CampaignPlanError(
                "zero-ledger event ledger identity changed"
            )
        try:
            raw = Legs._read_single_link_regular(os.fspath(baseline.path))
        except (OSError, Legs.WorkspaceTainted) as exc:
            raise CampaignPlanError(
                "zero-ledger event ledger is unstable"
            ) from exc
        if raw != expected_raw:
            raise CampaignPlanError(
                "zero-ledger event no longer has an empty dispatch suffix"
            )
        intent_name, selected_intent = _prepare_recovery_phase_intent_locked(
            state, record, expected_raw
        )
        selected = dict(selected_intent["record"])
        line = RebootRecovery.canonical_json_line(selected)
        if after_intent is not None:
            after_intent()
        descriptor: int | None = None
        try:
            descriptor = os.open(
                baseline.path,
                os.O_WRONLY | os.O_APPEND | getattr(os, "O_NOFOLLOW", 0),
            )
            opened = os.fstat(descriptor)
            if (
                not stat.S_ISREG(opened.st_mode)
                or opened.st_nlink != 1
                or (opened.st_dev, opened.st_ino) != baseline.file_identity
            ):
                raise CampaignPlanError(
                    "zero-ledger event descriptor identity changed"
                )
            offset = 0
            while offset < len(line):
                written = os.write(descriptor, line[offset:])
                if written <= 0:
                    raise OSError("short write appending zero-ledger event")
                offset += written
            os.fsync(descriptor)
        except OSError as exc:
            raise CampaignPlanError(
                "zero-ledger infrastructure event append failed"
            ) from exc
        finally:
            if descriptor is not None:
                os.close(descriptor)
        if Legs._read_single_link_regular(
            os.fspath(baseline.path)
        ) != expected_raw + line:
            raise CampaignPlanError(
                "zero-ledger infrastructure event append was not exact"
            )
        parent_fd = _open_recovery_intent_parent(
            marker.root, marker.root_identity
        )
        try:
            os.unlink(intent_name, dir_fd=parent_fd)
            os.fsync(parent_fd)
        except OSError as exc:
            raise CampaignPlanError(
                "could not retire the zero-ledger event intent"
            ) from exc
        finally:
            os.close(parent_fd)
    return selected


def _has_exact_pending_cleanup_intent(
    state: PostRebootLedgerState,
    cleanup: dict[str, Any],
) -> bool:
    """Authenticate, without reconciling, a cleanup mutation authority."""

    if state.intent_root is None or state.intent_root_identity is None:
        raise CampaignPlanError("recovery cleanup intent custody is unavailable")
    names = _recovery_phase_intent_names(state.ledger, state.dispatch_id)
    parent_fd = _open_recovery_intent_parent(
        state.intent_root, state.intent_root_identity
    )
    try:
        present: list[str] = []
        for name in names.values():
            try:
                os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
            except FileNotFoundError:
                continue
            present.append(name)
        if not present:
            return False
        cleanup_name = names["codex_taint_cleanup_completed"]
        if present != [cleanup_name]:
            raise CampaignPlanError(
                "unexpected recovery phase intent blocks WIP replay"
            )
        descriptor, payload, intent_identity = _read_unaliased_small_file_at(
            parent_fd, cleanup_name, label="recovery cleanup intent"
        )
        os.close(descriptor)
        intent = _parse_recovery_phase_intent(
            payload, label="recovery cleanup intent"
        )
        expected_raw = state.baseline.raw_prefix + b"".join(
            RebootRecovery.canonical_json_line(row)
            for row in _post_reboot_state_rows(state)
        )
        expected = _phase_intent_envelope(state, cleanup, expected_raw)
        if not _phase_intents_equivalent(intent, expected):
            raise CampaignPlanError("recovery cleanup intent changed")
        _fsync_and_revalidate_installed_intent(
            parent_fd,
            cleanup_name,
            payload=payload,
            identity=intent_identity,
            label="recovery cleanup intent",
        )
        return True
    finally:
        os.close(parent_fd)


def _exec_record_binds_unquiesced_marker(
    record: dict[str, Any], parsed: RebootRecovery.ParsedMarker
) -> bool:
    """Bind the wrapper status without confusing it with the inner CLI status."""

    exec_returncode = record.get("returncode")
    child_returncode = parsed.unquiesced.get("child_returncode")
    if (
        not isinstance(exec_returncode, int)
        or isinstance(exec_returncode, bool)
        or not isinstance(child_returncode, int)
        or isinstance(child_returncode, bool)
    ):
        return False
    if exec_returncode == child_returncode:
        return True
    # A legacy interruption race could leave the exec row recording an
    # interrupted but cleanly exited inner Codex CLI while the exact historical
    # runner remained alive until the scheduler watchdog escalated to SIGKILL.
    # These statuses belong to different processes; accept only that complete
    # legacy shape.
    return all((
        parsed.armed.get("schema")
        == RebootRecovery.DISPATCH_QUARANTINE_SCHEMA,
        child_returncode == -signal.SIGKILL,
        exec_returncode == 0,
        record.get("interrupted") is True,
        record.get("surviving_process_group") is False,
        record.get("timed_out") is False,
        record.get("allocation_expired") is False,
        "launch_error" in record and record["launch_error"] is None,
        (
            "postflight_error" in record
            and record["postflight_error"] is None
        ),
        "failure_class" in record and record["failure_class"] is None,
        (
            "failure_detail_class" in record
            and record["failure_detail_class"] is None
        ),
        record.get("protected_transcript_status") == "sealed",
        (
            "protected_transcript_error" in record
            and record["protected_transcript_error"] is None
        ),
        record.get("public_action_protocol_violation") is False,
        record.get("terminal_errors") == [],
    ))


def _rebind_post_reboot_ledger(
    item: dict[str, Any],
    parsed: RebootRecovery.ParsedMarker,
    *,
    marker: DispatchQuarantine,
    current_boot: RebootRecovery.BootIdentity,
    reconcile_intent: bool = True,
) -> PostRebootLedgerState:
    ledger, baseline, suffix = _read_post_reboot_ledger_surface(
        item,
        parsed.armed,
        intent_root=(marker.root if reconcile_intent else None),
        intent_root_identity=(
            marker.root_identity if reconcile_intent else None
        ),
    )
    expected_events = [
        "codex_exec",
        "codex_exec_classification_correction",
        "codex_taint_cleanup_completed",
        "codex_post_reboot_operator_recovery_completed",
    ]
    if (
        not 1 <= len(suffix) <= len(expected_events)
        or [row.get("event") for row in suffix]
        != expected_events[:len(suffix)]
    ):
        raise CampaignPlanError(
            "post-reboot ledger has an invalid recovery phase prefix"
        )
    record = _expected_exec_record(
        item,
        baseline.records,
        [*baseline.records, suffix[0]],
        clean_terminal=False,
    )
    if any((
        record.get("workspace") != parsed.unquiesced.get("workspace"),
        record.get("transcript") != parsed.unquiesced.get("transcript"),
        not _exec_record_binds_unquiesced_marker(record, parsed),
        parsed.unquiesced.get("exception_type") != "UnquiescedChildError",
    )):
        raise CampaignPlanError(
            "dispatch quarantine does not bind its exact ledger generation"
        )
    marker_time = _recovery_recorded_at(
        parsed.unquiesced, "dispatch unquiesced"
    )
    correction = suffix[1] if len(suffix) >= 2 else None
    cleanup = suffix[2] if len(suffix) >= 3 else None
    operator = suffix[3] if len(suffix) >= 4 else None
    if correction is not None:
        _validate_recovery_correction(
            item, record, correction, not_before=marker_time
        )
    if cleanup is not None:
        assert correction is not None
        _validate_recovery_cleanup(
            item,
            record,
            cleanup,
            not_before=_recovery_recorded_at(
                correction, "post-reboot correction"
            ),
        )
    return PostRebootLedgerState(
        dispatch_id=parsed.dispatch_id,
        intent_root=marker.root,
        intent_root_identity=marker.root_identity,
        ledger=ledger,
        baseline=baseline,
        record=record,
        correction=correction,
        cleanup=cleanup,
        operator=operator,
    )


def _xattr_receipt_sha256(values: tuple[tuple[str, bytes], ...]) -> str:
    return hashlib.sha256(json.dumps(
        [[name, payload.hex()] for name, payload in values],
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")).hexdigest()


def _canonical_root_recovery_metadata(
    state: CanonicalRollbackState,
) -> dict[str, Any]:
    return {
        "identity": list(state.root_identity),
        "mode": state.root_mode,
        "uid": state.root_uid,
        "gid": state.root_gid,
        "mtime_ns": state.root_mtime_ns,
        "xattrs_sha256": _xattr_receipt_sha256(state.root_xattrs),
    }


def _wip_root_recovery_metadata(state: WipRollbackState) -> dict[str, Any]:
    return {
        "existed": state.existed,
        "identity": (
            list(state.level_identity)
            if state.level_identity is not None else None
        ),
        "mode": state.level_mode,
        "uid": state.level_uid,
        "gid": state.level_gid,
        "mtime_ns": state.level_mtime_ns,
        "xattrs_sha256": (
            _xattr_receipt_sha256(state.level_xattrs)
            if state.level_xattrs is not None else None
        ),
    }


def _recovery_jsonable(value: Any) -> Any:
    if isinstance(value, bytes):
        return {"bytes_hex": value.hex()}
    if isinstance(value, tuple):
        return [_recovery_jsonable(part) for part in value]
    if isinstance(value, dict):
        return {
            str(key): _recovery_jsonable(item)
            for key, item in sorted(value.items())
        }
    return value


def _wip_absence_custody_payload(
    custody: WipAbsenceCustody | None,
    *,
    include_atime: bool,
    include_ctime: bool,
) -> dict[str, Any] | None:
    if custody is None:
        return None
    payload: dict[str, Any] = {
        "parent": os.fspath(custody.parent),
        "name": custody.name,
        "parent_identity": list(custody.parent_identity),
        "parent_mode": custody.parent_mode,
        "parent_uid": custody.parent_uid,
        "parent_gid": custody.parent_gid,
        "parent_xattrs": _recovery_jsonable(custody.parent_xattrs),
        "parent_mtime_ns": custody.parent_mtime_ns,
    }
    if include_atime:
        payload["parent_atime_ns"] = custody.parent_atime_ns
    if include_ctime:
        payload["parent_ctime_ns"] = custody.parent_ctime_ns
    return payload


def _wip_recovery_state_sha256(state: WipRollbackState) -> str:
    # File atime is retained in the rollback receipt so latest.json can be
    # restored exactly, but reads made while authenticating recovery may update
    # it.  It is deliberately excluded from the cross-boot integrity seal.
    recovery_entries = {
        relative: entry[:-1] if entry and entry[0] == "file" else entry
        for relative, entry in state.entries.items()
    }
    payload = {
        "level": os.fspath(state.level),
        "baseline_snapshot": list(state.baseline_snapshot),
        "existed": state.existed,
        "level_identity": (
            list(state.level_identity)
            if state.level_identity is not None else None
        ),
        "level_mode": state.level_mode,
        "level_uid": state.level_uid,
        "level_gid": state.level_gid,
        "level_xattrs": _recovery_jsonable(state.level_xattrs),
        "level_mtime_ns": state.level_mtime_ns,
        "level_ctime_ns": state.level_ctime_ns,
        "entries": _recovery_jsonable(recovery_entries),
        "latest_bytes": _recovery_jsonable(state.latest_bytes),
    }
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")).hexdigest()


def _wip_capsule_state_sha256(
    state: WipRollbackState, capsule_schema: str
) -> str:
    """Version the durable capsule hash without changing live-arm v1 hashes."""

    legacy_sha256 = _wip_recovery_state_sha256(state)
    if capsule_schema == WIP_ROLLBACK_CAPSULE_SCHEMA_V1:
        return legacy_sha256
    if capsule_schema != WIP_ROLLBACK_CAPSULE_SCHEMA:
        raise CampaignPlanError("WIP rollback capsule schema is unsupported")
    payload = {
        "schema": capsule_schema,
        "legacy_state_sha256": legacy_sha256,
        "absence_custody": _wip_absence_custody_payload(
            state.absence_custody,
            include_atime=False,
            include_ctime=True,
        ),
    }
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")).hexdigest()


def _wip_logical_restore_state_sha256(
    state: WipRollbackState,
    schema: str = WIP_LOGICAL_RESTORE_SCHEMA,
) -> str:
    """Seal the exact v2 state that rollback can honestly reproduce.

    The capsule separately preserves the full historical provenance hash,
    including inode and ctime.  This logical target retains every achievable
    authoritative property while excluding only filesystem-generated values:
    the replaceable ``latest.json`` inode, all ctimes/atimes, and directory
    allocation sizes.  Immutable baseline entry identities remain sealed.
    """

    logical_entries: dict[str, Any] = {}
    for relative, entry in sorted(state.entries.items()):
        if not entry:
            raise CampaignPlanError("WIP logical restore entry is malformed")
        if entry[0] == "file":
            logical_entries[relative] = {
                "kind": "file",
                "identity": (
                    None
                    if relative == "latest.json"
                    else [entry[1], entry[2]]
                ),
                "mode": entry[3],
                "nlink": entry[4],
                "size": entry[5],
                "mtime_ns": entry[6],
                "sha256": entry[8],
                "uid": entry[9],
                "gid": entry[10],
                "xattrs": _recovery_jsonable(entry[11]),
            }
        elif entry[0] == "directory":
            logical_entries[relative] = {
                "kind": "directory",
                "identity": [entry[1], entry[2]],
                "mode": entry[3],
                "nlink": entry[4],
                "mtime_ns": entry[6],
                "uid": entry[8],
                "gid": entry[9],
                "xattrs": _recovery_jsonable(entry[10]),
            }
        else:
            raise CampaignPlanError("WIP logical restore entry is malformed")
    if schema not in {
        WIP_LOGICAL_RESTORE_SCHEMA_V1,
        WIP_LOGICAL_RESTORE_SCHEMA,
    }:
        raise CampaignPlanError("WIP logical restore schema is unsupported")
    payload = {
        "schema": schema,
        "level": os.fspath(state.level),
        "existed": state.existed,
        "level_identity": (
            list(state.level_identity)
            if state.level_identity is not None else None
        ),
        "level_mode": state.level_mode,
        "level_uid": state.level_uid,
        "level_gid": state.level_gid,
        "level_xattrs": _recovery_jsonable(state.level_xattrs),
        "level_mtime_ns": state.level_mtime_ns,
        "entries": logical_entries,
    }
    if schema == WIP_LOGICAL_RESTORE_SCHEMA:
        payload["absence_custody"] = _wip_absence_custody_payload(
            state.absence_custody,
            include_atime=False,
            include_ctime=False,
        )
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")).hexdigest()


def _recovery_unjsonable(value: Any) -> Any:
    if isinstance(value, dict) and set(value) == {"bytes_hex"}:
        encoded = value["bytes_hex"]
        if not isinstance(encoded, str):
            raise CampaignPlanError("WIP rollback capsule bytes are malformed")
        try:
            return bytes.fromhex(encoded)
        except ValueError as exc:
            raise CampaignPlanError(
                "WIP rollback capsule bytes are malformed"
            ) from exc
    if isinstance(value, list):
        return tuple(_recovery_unjsonable(part) for part in value)
    if isinstance(value, dict):
        return {
            str(key): _recovery_unjsonable(item)
            for key, item in value.items()
        }
    return value


def _wip_rollback_capsule_record(
    item: dict[str, Any], state: WipRollbackState, dispatch_id: str
) -> dict[str, Any]:
    payloads: dict[str, str] = {}
    for relative, entry in sorted(state.entries.items()):
        if not entry or entry[0] != "file":
            continue
        path = state.level / relative
        try:
            payload = Legs._read_single_link_regular(os.fspath(path))
        except (OSError, Legs.WorkspaceTainted) as exc:
            raise CampaignPlanError(
                "WIP rollback capsule source changed"
            ) from exc
        if (
            len(payload) != entry[5]
            or hashlib.sha256(payload).hexdigest() != entry[8]
        ):
            raise CampaignPlanError(
                "WIP rollback capsule source hash changed"
            )
        payloads[relative] = base64.b64encode(payload).decode("ascii")
    state_payload = {
        "level": os.fspath(state.level),
        "baseline_snapshot": list(state.baseline_snapshot),
        "existed": state.existed,
        "level_identity": (
            list(state.level_identity)
            if state.level_identity is not None else None
        ),
        "level_mode": state.level_mode,
        "level_uid": state.level_uid,
        "level_gid": state.level_gid,
        "level_xattrs": _recovery_jsonable(state.level_xattrs),
        "level_atime_ns": state.level_atime_ns,
        "level_mtime_ns": state.level_mtime_ns,
        "level_ctime_ns": state.level_ctime_ns,
        "entries": _recovery_jsonable(state.entries),
        "latest_bytes": _recovery_jsonable(state.latest_bytes),
        "absence_custody": _wip_absence_custody_payload(
            state.absence_custody,
            include_atime=True,
            include_ctime=True,
        ),
    }
    return {
        "schema": WIP_ROLLBACK_CAPSULE_SCHEMA,
        "dispatch_id": dispatch_id,
        "game": item["game"],
        "target_level": item["target_level"],
        "state_sha256": _wip_capsule_state_sha256(
            state, WIP_ROLLBACK_CAPSULE_SCHEMA
        ),
        "restore_logical_state_schema": WIP_LOGICAL_RESTORE_SCHEMA,
        "restore_logical_state_sha256": (
            _wip_logical_restore_state_sha256(state)
        ),
        "state": state_payload,
        "file_payloads_base64": payloads,
    }


def _state_from_wip_rollback_capsule(
    record: dict[str, Any], item: dict[str, Any]
) -> WipRollbackState:
    capsule_schema = record.get("schema")
    if capsule_schema not in WIP_ROLLBACK_CAPSULE_SCHEMAS:
        raise CampaignPlanError("WIP rollback capsule schema is unsupported")
    state = record.get("state")
    legacy_state_keys = {
        "level", "baseline_snapshot", "existed", "level_identity",
        "level_mode", "level_uid", "level_gid", "level_xattrs",
        "level_atime_ns", "level_mtime_ns", "level_ctime_ns", "entries",
        "latest_bytes",
    }
    expected_state_keys = (
        legacy_state_keys
        if capsule_schema == WIP_ROLLBACK_CAPSULE_SCHEMA_V1
        else legacy_state_keys | {"absence_custody"}
    )
    if not isinstance(state, dict) or set(state) != expected_state_keys:
        raise CampaignPlanError("WIP rollback capsule state schema is invalid")
    expected_level = _target_wip_level(item)
    if state.get("level") != os.fspath(expected_level):
        raise CampaignPlanError("WIP rollback capsule level changed")
    baseline = state.get("baseline_snapshot")
    identity = state.get("level_identity")
    entries_raw = state.get("entries")
    existed = state.get("existed")
    if (
        not isinstance(baseline, list)
        or len(baseline) != 2
        or baseline[0] != os.fspath(expected_level)
        or not isinstance(existed, bool)
        or not isinstance(entries_raw, dict)
        or (identity is not None and (
            not isinstance(identity, list) or len(identity) != 2
        ))
    ):
        raise CampaignPlanError("WIP rollback capsule state is malformed")
    entries_decoded = _recovery_unjsonable(entries_raw)
    if not isinstance(entries_decoded, dict) or any(
        not isinstance(value, tuple) for value in entries_decoded.values()
    ):
        raise CampaignPlanError("WIP rollback capsule entries are malformed")
    absence_custody: WipAbsenceCustody | None = None
    if capsule_schema == WIP_ROLLBACK_CAPSULE_SCHEMA:
        custody_raw = state.get("absence_custody")
        if custody_raw is not None:
            custody_keys = {
                "parent", "name", "parent_identity", "parent_mode",
                "parent_uid", "parent_gid", "parent_xattrs",
                "parent_atime_ns", "parent_mtime_ns", "parent_ctime_ns",
            }
            if not isinstance(custody_raw, dict) or set(custody_raw) != (
                custody_keys
            ):
                raise CampaignPlanError(
                    "WIP rollback absence custody schema is invalid"
                )
            parent = _normalized_absolute_path(
                custody_raw.get("parent"), "WIP absence custody parent"
            )
            name = custody_raw.get("name")
            parent_identity = custody_raw.get("parent_identity")
            integer_fields = (
                "parent_mode", "parent_uid", "parent_gid",
                "parent_atime_ns", "parent_mtime_ns", "parent_ctime_ns",
            )
            decoded_xattrs = _recovery_unjsonable(
                custody_raw.get("parent_xattrs")
            )
            if any((
                not isinstance(name, str),
                isinstance(name, str) and (
                    not name or name in {".", ".."} or Path(name).name != name
                ),
                not isinstance(parent_identity, list),
                isinstance(parent_identity, list)
                and (
                    len(parent_identity) != 2
                    or any(
                        not isinstance(value, int) or isinstance(value, bool)
                        or value < 0
                        for value in parent_identity
                    )
                ),
                any(
                    not isinstance(custody_raw.get(field), int)
                    or isinstance(custody_raw.get(field), bool)
                    or custody_raw.get(field) < 0
                    for field in integer_fields
                ),
                not isinstance(decoded_xattrs, tuple),
                isinstance(decoded_xattrs, tuple) and any(
                    not isinstance(pair, tuple)
                    or len(pair) != 2
                    or not isinstance(pair[0], str)
                    or not isinstance(pair[1], bytes)
                    for pair in decoded_xattrs
                ),
            )):
                raise CampaignPlanError(
                    "WIP rollback absence custody is malformed"
                )
            assert isinstance(name, str)
            branch = parent / name
            try:
                parent.relative_to(
                    _artifact_root(item) / f"{item['game']}_legs"
                )
                expected_level.relative_to(branch)
            except ValueError as exc:
                raise CampaignPlanError(
                    "WIP rollback absence custody does not bind the target"
                ) from exc
            absence_custody = WipAbsenceCustody(
                parent=parent,
                name=name,
                parent_identity=tuple(parent_identity),
                parent_mode=int(custody_raw["parent_mode"]),
                parent_uid=int(custody_raw["parent_uid"]),
                parent_gid=int(custody_raw["parent_gid"]),
                parent_xattrs=decoded_xattrs,
                parent_atime_ns=int(custody_raw["parent_atime_ns"]),
                parent_mtime_ns=int(custody_raw["parent_mtime_ns"]),
                parent_ctime_ns=int(custody_raw["parent_ctime_ns"]),
            )
            if not stat.S_ISDIR(absence_custody.parent_mode):
                raise CampaignPlanError(
                    "WIP rollback absence custody parent is not a directory"
                )
    if existed and absence_custody is not None:
        raise CampaignPlanError(
            "WIP rollback capsule absence custody is inconsistent"
        )
    if (
        not existed
        and capsule_schema == WIP_ROLLBACK_CAPSULE_SCHEMA
        and absence_custody is None
    ):
        raise CampaignPlanError(
            "WIP rollback capsule lacks absence custody"
        )
    if not existed and any((
        identity is not None,
        state.get("level_mode") is not None,
        state.get("level_uid") is not None,
        state.get("level_gid") is not None,
        state.get("level_xattrs") is not None,
        state.get("level_atime_ns") is not None,
        state.get("level_mtime_ns") is not None,
        state.get("level_ctime_ns") is not None,
        bool(entries_decoded),
        state.get("latest_bytes") is not None,
    )):
        raise CampaignPlanError(
            "WIP rollback absent-root state is internally inconsistent"
        )
    restored = WipRollbackState(
        level=expected_level,
        baseline_snapshot=(str(baseline[0]), baseline[1]),
        existed=existed,
        level_identity=(tuple(identity) if identity is not None else None),
        level_mode=state.get("level_mode"),
        level_uid=state.get("level_uid"),
        level_gid=state.get("level_gid"),
        level_xattrs=_recovery_unjsonable(state.get("level_xattrs")),
        level_atime_ns=state.get("level_atime_ns"),
        level_mtime_ns=state.get("level_mtime_ns"),
        level_ctime_ns=state.get("level_ctime_ns"),
        entries={str(key): value for key, value in entries_decoded.items()},
        latest_bytes=_recovery_unjsonable(state.get("latest_bytes")),
        absence_custody=absence_custody,
    )
    if _wip_capsule_state_sha256(
        restored, str(capsule_schema)
    ) != record.get("state_sha256"):
        raise CampaignPlanError("WIP rollback capsule state hash changed")
    logical_schema = record.get("restore_logical_state_schema")
    expected_logical_schema = (
        WIP_LOGICAL_RESTORE_SCHEMA_V1
        if capsule_schema == WIP_ROLLBACK_CAPSULE_SCHEMA_V1
        else WIP_LOGICAL_RESTORE_SCHEMA
    )
    if (
        logical_schema != expected_logical_schema
        or _wip_logical_restore_state_sha256(
            restored, schema=str(logical_schema)
        )
        != record.get("restore_logical_state_sha256")
    ):
        raise CampaignPlanError(
            "WIP rollback capsule logical restore seal changed"
        )
    payloads = record.get("file_payloads_base64")
    if not isinstance(payloads, dict):
        raise CampaignPlanError("WIP rollback capsule payloads are malformed")
    expected_files = {
        relative for relative, entry in restored.entries.items()
        if entry and entry[0] == "file"
    }
    if set(payloads) != expected_files:
        raise CampaignPlanError("WIP rollback capsule payload inventory changed")
    for relative, encoded in payloads.items():
        if not isinstance(encoded, str):
            raise CampaignPlanError("WIP rollback capsule payload is malformed")
        try:
            payload = base64.b64decode(encoded, validate=True)
        except (ValueError, binascii.Error) as exc:
            raise CampaignPlanError(
                "WIP rollback capsule payload is malformed"
            ) from exc
        entry = restored.entries[relative]
        if (
            len(payload) != entry[5]
            or hashlib.sha256(payload).hexdigest() != entry[8]
        ):
            raise CampaignPlanError("WIP rollback capsule payload hash changed")
    return restored


def _wip_discard_survivor_sha256(state: WipRollbackState) -> str:
    """Seal everything except the intentionally discarded latest pointer.

    Removing ``latest.json`` changes the level directory's mtime/ctime, so
    those two timestamps cannot be part of the post-discard equality target.
    Every other root attribute and every surviving entry remains exact.
    """

    recovery_entries = {
        relative: entry[:-1] if entry and entry[0] == "file" else entry
        for relative, entry in state.entries.items()
        if relative != "latest.json"
    }
    payload = {
        "level": os.fspath(state.level),
        "existed": state.existed,
        "level_identity": (
            list(state.level_identity)
            if state.level_identity is not None else None
        ),
        "level_mode": state.level_mode,
        "level_uid": state.level_uid,
        "level_gid": state.level_gid,
        "level_xattrs": _recovery_jsonable(state.level_xattrs),
        "entries": _recovery_jsonable(recovery_entries),
    }
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")).hexdigest()


def _legacy_wip_requires_confirmation(
    parsed: RebootRecovery.ParsedMarker, state: WipRollbackState
) -> bool:
    del state
    # Dispatch v1 sealed only the weaker inventory snapshot, not the complete
    # recovery state (ownership/xattrs/ctime/latest bytes).  It can never
    # silently authorize current WIP, even when that weaker digest matches.
    return "wip_rollback_capsule_name" not in parsed.armed


def _validate_legacy_wip_exclusion_item(item: dict[str, Any]) -> None:
    if any((
        item.get("seed_mode") != "verified_parent",
        item.get("wip_mode") != "exclude",
        item.get("lineage_input_mode") != "verified_parent+exclude",
        item.get("expected_wip_attempt") is not None,
    )):
        raise CampaignPlanError(
            "operator-confirmed quarantined WIP requires verified-parent "
            "seeding, excluded WIP, and no selected WIP attempt"
        )


def _validate_discarded_wip_state(
    item: dict[str, Any], arm: dict[str, Any], state: WipRollbackState
) -> None:
    if (
        os.path.lexists(_target_wip_level(item) / "latest.json")
        or "latest.json" in state.entries
        or state.latest_bytes is not None
        or _wip_discard_survivor_sha256(state)
        != arm.get("discard_survivor_sha256")
    ):
        raise CampaignPlanError(
            "operator-confirmed quarantined WIP discard state changed"
        )


def _validate_capsule_restored_wip_state(
    current: WipRollbackState,
    baseline: WipRollbackState,
    expected_logical_sha256: str | None = None,
    logical_schema: str = WIP_LOGICAL_RESTORE_SCHEMA,
) -> None:
    baseline_logical = _wip_logical_restore_state_sha256(
        baseline, schema=logical_schema
    )
    expected = expected_logical_sha256 or baseline_logical
    if (
        not isinstance(expected, str)
        or SHA256_RE.fullmatch(expected) is None
        or baseline_logical != expected
        or _wip_logical_restore_state_sha256(
            current, schema=logical_schema
        ) != expected
    ):
        raise CampaignPlanError(
            "WIP rollback capsule logical restore state changed"
        )


def _validate_capsule_restore_progress(
    current: WipRollbackState, baseline: WipRollbackState
) -> None:
    """Accept only a tree on the authorized idempotent restore envelope."""

    if not baseline.existed:
        # The pre-dispatch target was absent.  Any safely inventoried current
        # root is entirely rollback-owned once the exact cleanup intent exists.
        return
    if (
        not current.existed
        or current.level_identity != baseline.level_identity
    ):
        raise CampaignPlanError("WIP restore progress changed the baseline root")
    for relative, expected in baseline.entries.items():
        observed = current.entries.get(relative)
        if relative == "latest.json":
            if observed is not None and observed[0] != "file":
                raise CampaignPlanError(
                    "WIP restore progress has an unsafe latest pointer"
                )
            continue
        if (
            observed is None
            or observed[0] != expected[0]
            or observed[1:3] != expected[1:3]
        ):
            raise CampaignPlanError(
                "WIP restore progress changed an immutable baseline inode"
            )


def _discard_confirmed_wip_latest_pointer(
    item: dict[str, Any], arm: dict[str, Any], state: WipRollbackState
) -> WipRollbackState:
    """Durably remove only the confirmed untrusted latest.json authority."""

    expected_state = arm.get("confirmed_current_wip_state_sha256")
    if _wip_recovery_state_sha256(state) != expected_state:
        raise CampaignPlanError(
            "confirmed quarantined WIP changed before discard"
        )
    latest = state.entries.get("latest.json")
    if latest is None or latest[0] != "file" or state.latest_bytes is None:
        raise CampaignPlanError(
            "confirmed quarantined WIP lacks one discardable latest pointer"
        )
    level = _target_wip_level(item)
    descriptor: int | None = None
    try:
        descriptor = os.open(
            level,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        root = os.fstat(descriptor)
        latest_stat = os.stat(
            "latest.json", dir_fd=descriptor, follow_symlinks=False
        )
        if (
            not stat.S_ISDIR(root.st_mode)
            or (root.st_dev, root.st_ino) != state.level_identity
            or not stat.S_ISREG(latest_stat.st_mode)
            or latest_stat.st_nlink != 1
            or (latest_stat.st_dev, latest_stat.st_ino) != (latest[1], latest[2])
            or latest_stat.st_mode != latest[3]
            or latest_stat.st_size != latest[5]
            or latest_stat.st_mtime_ns != latest[6]
            or latest_stat.st_ctime_ns != latest[7]
            or latest_stat.st_uid != latest[9]
            or latest_stat.st_gid != latest[10]
            or _canonical_xattrs(level / "latest.json") != latest[11]
        ):
            raise CampaignPlanError(
                "confirmed quarantined WIP latest pointer changed"
            )
        os.unlink("latest.json", dir_fd=descriptor)
        os.fsync(descriptor)
    except OSError as exc:
        raise CampaignPlanError(
            "could not durably discard quarantined WIP latest pointer"
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
    discarded = _capture_wip_rollback(item)
    _validate_discarded_wip_state(item, arm, discarded)
    return discarded


def _durably_confirm_discarded_wip_state(
    item: dict[str, Any], arm: dict[str, Any], state: WipRollbackState
) -> WipRollbackState:
    """Resolve an unlink-before-fsync crash without reauthorizing WIP input."""

    _validate_discarded_wip_state(item, arm, state)
    level = _target_wip_level(item)
    descriptor: int | None = None
    try:
        descriptor = os.open(
            level,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISDIR(opened.st_mode)
            or state.level_identity is None
            or (opened.st_dev, opened.st_ino) != state.level_identity
        ):
            raise CampaignPlanError(
                "discarded quarantined WIP root identity changed"
            )
        try:
            os.stat(
                "latest.json", dir_fd=descriptor, follow_symlinks=False
            )
        except FileNotFoundError:
            pass
        else:
            raise CampaignPlanError(
                "discarded quarantined WIP latest pointer reappeared"
            )
        # Required even when unlink already returned: a prior process may have
        # died before the directory-entry removal reached stable storage.
        os.fsync(descriptor)
    except OSError as exc:
        raise CampaignPlanError(
            "could not durably confirm quarantined WIP discard"
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
    confirmed = _capture_wip_rollback(item)
    _validate_discarded_wip_state(item, arm, confirmed)
    return confirmed


def _capsule_file_payloads(record: dict[str, Any]) -> dict[str, bytes]:
    encoded = record.get("file_payloads_base64")
    if not isinstance(encoded, dict):
        raise CampaignPlanError("WIP rollback capsule payloads are malformed")
    payloads: dict[str, bytes] = {}
    for relative, value in encoded.items():
        if not isinstance(relative, str) or not isinstance(value, str):
            raise CampaignPlanError(
                "WIP rollback capsule payload is malformed"
            )
        try:
            payloads[relative] = base64.b64decode(value, validate=True)
        except (ValueError, binascii.Error) as exc:
            raise CampaignPlanError(
                "WIP rollback capsule payload is malformed"
            ) from exc
    return payloads


def _durably_remove_wip_entry_at(
    parent_fd: int,
    name: str,
    *,
    expected_identity: tuple[int, int] | None = None,
) -> None:
    """Remove one unaliased entry and fsync every changed directory."""

    metadata = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    identity = (metadata.st_dev, metadata.st_ino)
    if expected_identity is not None and identity != expected_identity:
        raise CampaignPlanError("WIP rollback extra entry identity changed")
    if stat.S_ISDIR(metadata.st_mode):
        descriptor = os.open(
            name,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_fd,
        )
        try:
            opened = os.fstat(descriptor)
            path_opened = os.stat(
                name, dir_fd=parent_fd, follow_symlinks=False
            )
            if (
                not stat.S_ISDIR(opened.st_mode)
                or (opened.st_dev, opened.st_ino) != identity
                or (path_opened.st_dev, path_opened.st_ino) != identity
            ):
                raise CampaignPlanError(
                    "WIP rollback extra directory changed"
                )
            for child in sorted(os.listdir(descriptor)):
                child_metadata = os.stat(
                    child, dir_fd=descriptor, follow_symlinks=False
                )
                _durably_remove_wip_entry_at(
                    descriptor,
                    child,
                    expected_identity=(
                        child_metadata.st_dev,
                        child_metadata.st_ino,
                    ),
                )
            # Seal all child namespace removals while this directory inode is
            # still reachable, then seal its removal in the parent.
            os.fsync(descriptor)
            path_before_remove = os.stat(
                name, dir_fd=parent_fd, follow_symlinks=False
            )
            if (path_before_remove.st_dev, path_before_remove.st_ino) != identity:
                raise CampaignPlanError(
                    "WIP rollback extra directory was replaced"
                )
            os.rmdir(name, dir_fd=parent_fd)
            os.fsync(parent_fd)
        finally:
            os.close(descriptor)
        return
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise CampaignPlanError("WIP rollback extra entry is unsafe")
    descriptor = os.open(
        name,
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
        dir_fd=parent_fd,
    )
    try:
        opened = os.fstat(descriptor)
        path_opened = os.stat(
            name, dir_fd=parent_fd, follow_symlinks=False
        )
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or (opened.st_dev, opened.st_ino) != identity
            or (path_opened.st_dev, path_opened.st_ino) != identity
        ):
            raise CampaignPlanError("WIP rollback extra file changed")
        os.unlink(name, dir_fd=parent_fd)
        os.fsync(parent_fd)
    finally:
        os.close(descriptor)


def _open_wip_inventory_directory(
    root_fd: int,
    relative: str,
    inventory: dict[str, tuple[Any, ...]],
) -> int:
    """Open one baseline parent through identity-bound directory components."""

    descriptor = os.dup(root_fd)
    prefix: list[str] = []
    try:
        if not relative:
            return descriptor
        for component in Path(relative).parts:
            prefix.append(component)
            key = "/".join(prefix)
            expected = inventory.get(key)
            if expected is None or expected[0] != "directory":
                raise CampaignPlanError(
                    "WIP rollback extra parent is not a baseline directory"
                )
            next_fd = os.open(
                component,
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=descriptor,
            )
            opened = os.fstat(next_fd)
            path_opened = os.stat(
                component, dir_fd=descriptor, follow_symlinks=False
            )
            if (
                not stat.S_ISDIR(opened.st_mode)
                or (opened.st_dev, opened.st_ino)
                != (expected[1], expected[2])
                or (path_opened.st_dev, path_opened.st_ino)
                != (expected[1], expected[2])
            ):
                os.close(next_fd)
                raise CampaignPlanError(
                    "WIP rollback extra parent identity changed"
                )
            os.close(descriptor)
            descriptor = next_fd
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _validate_wip_absence_branch_at(
    parent_fd: int,
    custody: WipAbsenceCustody,
    level: Path,
) -> tuple[int, int] | None:
    """Prove a newly-created wrapper has no siblings outside target custody."""

    try:
        metadata = os.stat(
            custody.name, dir_fd=parent_fd, follow_symlinks=False
        )
    except FileNotFoundError:
        return None
    branch_identity = (metadata.st_dev, metadata.st_ino)
    if not stat.S_ISDIR(metadata.st_mode):
        raise CampaignPlanError("WIP absence branch is not a directory")
    branch = custody.parent / custody.name
    remaining = level.relative_to(branch).parts
    if not remaining:
        return branch_identity
    descriptor = os.open(
        custody.name,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        dir_fd=parent_fd,
    )
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino) != branch_identity:
            raise CampaignPlanError("WIP absence branch identity changed")
        for index, component in enumerate(remaining):
            names = set(os.listdir(descriptor))
            if component not in names:
                if names:
                    raise CampaignPlanError(
                        "new WIP wrapper contains out-of-scope evidence"
                    )
                break
            if names != {component}:
                raise CampaignPlanError(
                    "new WIP wrapper contains out-of-scope evidence"
                )
            if index == len(remaining) - 1:
                break
            child = os.open(
                component,
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=descriptor,
            )
            child_stat = os.fstat(child)
            path_stat = os.stat(
                component, dir_fd=descriptor, follow_symlinks=False
            )
            if (
                not stat.S_ISDIR(child_stat.st_mode)
                or (child_stat.st_dev, child_stat.st_ino)
                != (path_stat.st_dev, path_stat.st_ino)
            ):
                os.close(child)
                raise CampaignPlanError("new WIP wrapper is unsafe")
            os.close(descriptor)
            descriptor = child
        return branch_identity
    finally:
        os.close(descriptor)


def _durably_restore_wip_absence_custody(
    level: Path,
    custody: WipAbsenceCustody,
) -> None:
    """Remove an authorized absent-root branch and restore its parent inode."""

    branch = custody.parent / custody.name
    try:
        level.relative_to(branch)
    except ValueError as exc:
        raise CampaignPlanError(
            "WIP absence custody does not bind the target level"
        ) from exc
    ancestor = custody.parent
    absent_name = custody.name
    _reject_symlinked_ancestry(ancestor, "WIP absence custody parent")
    descriptor: int | None = None
    try:
        descriptor = os.open(
            ancestor,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        opened = os.fstat(descriptor)
        path_opened = ancestor.stat(follow_symlinks=False)
        identity = custody.parent_identity
        if (
            not stat.S_ISDIR(opened.st_mode)
            or not stat.S_ISDIR(path_opened.st_mode)
            or (opened.st_dev, opened.st_ino) != identity
            or (path_opened.st_dev, path_opened.st_ino) != identity
        ):
            raise CampaignPlanError(
                "WIP absence custody parent identity changed"
            )
        branch_identity = _validate_wip_absence_branch_at(
            descriptor, custody, level
        )
        if branch_identity is not None:
            _durably_remove_wip_entry_at(
                descriptor,
                absent_name,
                expected_identity=branch_identity,
            )
        try:
            os.stat(absent_name, dir_fd=descriptor, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise CampaignPlanError(
                "historically absent WIP namespace reappeared"
            )
        current = os.fstat(descriptor)
        if (
            current.st_uid != custody.parent_uid
            or current.st_gid != custody.parent_gid
        ):
            os.fchown(
                descriptor, custody.parent_uid, custody.parent_gid
            )
        os.fchmod(descriptor, stat.S_IMODE(custody.parent_mode))
        _restore_canonical_xattrs(ancestor, custody.parent_xattrs)
        os.utime(
            descriptor,
            ns=(custody.parent_atime_ns, custody.parent_mtime_ns),
        )
        os.fsync(descriptor)
        after = os.fstat(descriptor)
        path_after = ancestor.stat(follow_symlinks=False)
        if any((
            (after.st_dev, after.st_ino) != identity
            , (path_after.st_dev, path_after.st_ino) != identity
            , after.st_mode != custody.parent_mode
            , after.st_uid != custody.parent_uid
            , after.st_gid != custody.parent_gid
            , after.st_mtime_ns != custody.parent_mtime_ns
            , _canonical_xattrs(ancestor) != custody.parent_xattrs
        )):
            raise CampaignPlanError(
                "WIP absence custody parent changed during fsync"
            )
    except OSError as exc:
        raise CampaignPlanError(
            "could not durably restore absent WIP namespace"
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _restore_wip_from_rollback_capsule(
    item: dict[str, Any],
    baseline: WipRollbackState,
    record: dict[str, Any],
) -> WipRollbackState:
    """Restore the exact pre-dispatch tree while preserving baseline inodes."""

    level = _target_wip_level(item)
    payloads = _capsule_file_payloads(record)
    if not baseline.existed:
        if baseline.absence_custody is None:
            # Capsule-v1 did not preserve the parent metadata needed to undo
            # an absent-root namespace transition.  It remains readable for
            # existing-level compatibility but cannot authorize this mutation.
            raise CampaignPlanError(
                "legacy WIP rollback capsule lacks absence custody"
            )
        try:
            _durably_restore_wip_absence_custody(
                level, baseline.absence_custody
            )
        except CampaignPlanError as exc:
            raise CampaignPlanError(
                "WIP rollback capsule restore failed"
            ) from exc
        restored = _capture_wip_rollback(item)
        if restored.baseline_snapshot != baseline.baseline_snapshot:
            raise CampaignPlanError(
                "WIP capsule rollback did not restore absent-root baseline"
            )
        return restored
    if (
        not os.path.lexists(level)
        or level.is_symlink()
        or not level.is_dir()
        or _host_directory_identity(level, "WIP capsule rollback root")
        != baseline.level_identity
    ):
        raise CampaignPlanError("preexisting WIP root identity changed")
    current = _wip_entry_inventory(level)
    missing = set(baseline.entries) - set(current)
    if missing - {"latest.json"}:
        raise CampaignPlanError(
            "WIP capsule rollback cannot restore missing baseline inodes"
        )
    for relative, expected in baseline.entries.items():
        if relative == "latest.json" and relative not in current:
            continue
        observed = current[relative]
        if (
            relative == "latest.json"
            and expected[0] == observed[0] == "file"
        ):
            # ``latest.json`` is the one designated replaceable pointer.  A
            # normal atomic publish changes its inode; the inventory above
            # has already proved that the replacement is a single-link
            # regular file, and the open/unlink path below revalidates it.
            continue
        if (
            observed[0] != expected[0]
            or observed[1:3] != expected[1:3]
        ):
            raise CampaignPlanError(
                "WIP capsule rollback baseline inode changed"
            )
    extras = set(current) - set(baseline.entries)
    extra_roots = sorted(
        (
            relative for relative in extras
            if not any(
                relative.startswith(f"{parent}/")
                for parent in extras
                if parent != relative
            )
        ),
        key=lambda value: (value.count("/"), value),
        reverse=True,
    )
    root_fd: int | None = None
    try:
        root_fd = os.open(
            level,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        opened_root = os.fstat(root_fd)
        if (opened_root.st_dev, opened_root.st_ino) != baseline.level_identity:
            raise CampaignPlanError("preexisting WIP root identity changed")
        for relative in extra_roots:
            entry = current[relative]
            relative_path = Path(relative)
            parent_relative = relative_path.parent.as_posix()
            if parent_relative == ".":
                parent_relative = ""
            parent_fd = _open_wip_inventory_directory(
                root_fd, parent_relative, current
            )
            try:
                _durably_remove_wip_entry_at(
                    parent_fd,
                    relative_path.name,
                    expected_identity=(entry[1], entry[2]),
                )
            finally:
                os.close(parent_fd)
        for relative, expected in sorted(baseline.entries.items()):
            if expected[0] != "file":
                continue
            payload = payloads.get(relative)
            if payload is None:
                raise CampaignPlanError(
                    "WIP capsule rollback lacks baseline file bytes"
                )
            path = level / relative
            recreated_latest = False
            if relative == "latest.json" and (
                relative not in current
                or current[relative][1:3] != expected[1:3]
            ):
                if os.path.lexists(path):
                    replacement = path.stat(follow_symlinks=False)
                    if (
                        path.is_symlink()
                        or not stat.S_ISREG(replacement.st_mode)
                        or replacement.st_nlink != 1
                    ):
                        raise CampaignPlanError(
                            "WIP capsule latest replacement is unsafe"
                        )
                    os.unlink(path)
                    _fsync_directory(path.parent)
                descriptor = os.open(
                    path,
                    os.O_RDWR | os.O_CREAT | os.O_EXCL
                    | getattr(os, "O_NOFOLLOW", 0),
                    stat.S_IMODE(expected[3]),
                )
                recreated_latest = True
            else:
                descriptor = os.open(
                    path, os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
                )
            try:
                opened = os.fstat(descriptor)
                if (
                    not stat.S_ISREG(opened.st_mode)
                    or opened.st_nlink != 1
                    or (
                        not recreated_latest
                        and (opened.st_dev, opened.st_ino)
                        != (expected[1], expected[2])
                    )
                ):
                    raise CampaignPlanError(
                        "WIP capsule rollback file identity changed"
                    )
                os.ftruncate(descriptor, 0)
                offset = 0
                while offset < len(payload):
                    written = os.write(descriptor, payload[offset:])
                    if written <= 0:
                        raise OSError("short WIP capsule rollback write")
                    offset += written
                os.fchown(descriptor, expected[9], expected[10])
                os.fchmod(descriptor, stat.S_IMODE(expected[3]))
                _restore_canonical_xattrs(path, expected[11])
                os.utime(descriptor, ns=(expected[12], expected[6]))
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        directories = [
            (relative, entry)
            for relative, entry in baseline.entries.items()
            if entry[0] == "directory"
        ]
        for relative, expected in sorted(
            directories, key=lambda pair: pair[0].count("/"), reverse=True
        ):
            path = level / relative
            os.chown(
                path, expected[8], expected[9], follow_symlinks=False
            )
            os.chmod(
                path, stat.S_IMODE(expected[3]), follow_symlinks=False
            )
            _restore_canonical_xattrs(path, expected[10])
            os.utime(
                path,
                ns=(expected[6], expected[6]),
                follow_symlinks=False,
            )
            _fsync_directory(path)
        assert baseline.level_uid is not None
        assert baseline.level_gid is not None
        assert baseline.level_mode is not None
        assert baseline.level_xattrs is not None
        assert baseline.level_atime_ns is not None
        assert baseline.level_mtime_ns is not None
        os.chown(
            level,
            baseline.level_uid,
            baseline.level_gid,
            follow_symlinks=False,
        )
        os.chmod(
            level, stat.S_IMODE(baseline.level_mode), follow_symlinks=False
        )
        _restore_canonical_xattrs(level, baseline.level_xattrs)
        os.utime(
            level,
            ns=(baseline.level_atime_ns, baseline.level_mtime_ns),
            follow_symlinks=False,
        )
        _fsync_directory(level)
    except OSError as exc:
        raise CampaignPlanError("WIP rollback capsule restore failed") from exc
    finally:
        if root_fd is not None:
            os.close(root_fd)
    restored = _capture_wip_rollback(item)
    _validate_capsule_restored_wip_state(
        restored,
        baseline,
        str(record.get("restore_logical_state_sha256")),
        logical_schema=str(record.get("restore_logical_state_schema")),
    )
    return restored


def _pre_reboot_exec_state(
    item: dict[str, Any], parsed: RebootRecovery.ParsedMarker
) -> PostRebootLedgerState:
    ledger, baseline, suffix = _read_post_reboot_ledger_surface(
        item, parsed.armed
    )
    if len(suffix) != 1 or suffix[0].get("event") != "codex_exec":
        raise CampaignPlanError(
            "pre-reboot arm requires exactly one unclassified exec row"
        )
    record = _expected_exec_record(
        item,
        baseline.records,
        [*baseline.records, suffix[0]],
        clean_terminal=False,
    )
    if any((
        record.get("workspace") != parsed.unquiesced.get("workspace"),
        record.get("transcript") != parsed.unquiesced.get("transcript"),
        not _exec_record_binds_unquiesced_marker(record, parsed),
        parsed.unquiesced.get("exception_type") != "UnquiescedChildError",
    )):
        raise CampaignPlanError(
            "pre-reboot marker does not bind its exact ledger generation"
        )
    return PostRebootLedgerState(
        dispatch_id=parsed.dispatch_id,
        intent_root=None,
        intent_root_identity=None,
        ledger=ledger,
        baseline=baseline,
        record=record,
        correction=None,
        cleanup=None,
        operator=None,
    )


def _canonical_absence_restore_progress_is_bound(
    canonical: CanonicalRollbackState,
    parsed: RebootRecovery.ParsedMarker,
    baseline: WipRollbackState,
    *,
    allow_metadata_restore_progress: bool,
) -> tuple[bool, bool, CanonicalRollbackState | None]:
    """Normalize only persisted custody metadata for pre-restore auth."""

    custody = baseline.absence_custody
    arm = parsed.recovery_arm
    if baseline.existed or custody is None:
        return False, False, None
    try:
        metadata = custody.parent.stat(follow_symlinks=False)
    except OSError:
        return False, False, None
    if (
        custody.parent.is_symlink()
        or not stat.S_ISDIR(metadata.st_mode)
        or (metadata.st_dev, metadata.st_ino) != custody.parent_identity
    ):
        return False, False, None
    expected_entry = CanonicalEntry(
        kind="directory",
        mode=custody.parent_mode,
        atime_ns=custody.parent_atime_ns,
        mtime_ns=custody.parent_mtime_ns,
        uid=custody.parent_uid,
        gid=custody.parent_gid,
        xattrs=custody.parent_xattrs,
    )
    expected_root_metadata = (
        arm.get("canonical_root_metadata") if arm is not None else None
    )
    if custody.parent == canonical.root:
        normalized_root_metadata = {
            "identity": list(custody.parent_identity),
            "mode": custody.parent_mode,
            "uid": custody.parent_uid,
            "gid": custody.parent_gid,
            "mtime_ns": custody.parent_mtime_ns,
            "xattrs_sha256": _xattr_receipt_sha256(
                custody.parent_xattrs
            ),
        }
        digest_valid = (
            canonical.digest == parsed.armed.get("canonical_digest")
        )
        return (
            digest_valid,
            normalized_root_metadata == expected_root_metadata,
            canonical if digest_valid else None,
        )
    try:
        relative = custody.parent.relative_to(canonical.root).as_posix()
    except ValueError:
        return False, False, None
    observed = canonical.entries.get(relative)
    if observed is None or observed.kind != "directory":
        return False, False, None
    if not allow_metadata_restore_progress and any((
        observed.mode != custody.parent_mode,
        observed.uid != custody.parent_uid,
        observed.gid != custody.parent_gid,
        observed.xattrs != custody.parent_xattrs,
    )):
        raise CampaignPlanError(
            "WIP absence custody parent has unauthorized metadata drift"
        )
    normalized = dict(canonical.entries)
    normalized[relative] = expected_entry
    normalized_digest = _canonical_entries_digest(normalized)
    digest_valid = normalized_digest == parsed.armed.get("canonical_digest")
    normalized_state = CanonicalRollbackState(
        root=canonical.root,
        root_identity=canonical.root_identity,
        root_mode=canonical.root_mode,
        root_uid=canonical.root_uid,
        root_gid=canonical.root_gid,
        root_xattrs=canonical.root_xattrs,
        root_atime_ns=canonical.root_atime_ns,
        root_mtime_ns=canonical.root_mtime_ns,
        entries=normalized,
        digest=normalized_digest,
        excluded_prefixes=canonical.excluded_prefixes,
    )
    return (
        digest_valid,
        (
            arm is not None
            and _canonical_root_recovery_metadata(canonical)
            == expected_root_metadata
        ),
        normalized_state if digest_valid else None,
    )


def _rebind_recovery_baselines(
    item: dict[str, Any],
    parsed: RebootRecovery.ParsedMarker,
    *,
    capsule_baseline: WipRollbackState | None = None,
    allow_capsule_restore_progress: bool = False,
) -> tuple[Path, CanonicalRollbackState, WipRollbackState, int]:
    _validate_post_reboot_dispatch_binding(item, parsed)
    artifact_root = _artifact_root(item)
    if _host_directory_identity(
        artifact_root, "canonical artifact root"
    ) != _marker_identity(
        parsed.armed.get("artifact_root_identity"), "artifact root"
    ):
        raise CampaignPlanError(
            "dispatch quarantine artifact root identity changed"
        )
    canonical = _capture_canonical_rollback(item)
    canonical_digest_valid = (
        canonical.digest == parsed.armed.get("canonical_digest")
    )
    canonical_root_metadata_valid = False
    if (
        capsule_baseline is not None
        and not capsule_baseline.existed
        and capsule_baseline.absence_custody is not None
    ):
        if capsule_baseline.absence_custody.parent == canonical.root:
            raise CampaignPlanError(
                "post-reboot absent WIP wrapper lacks a stable parent boundary"
            )
        (
            progress_digest_valid,
            canonical_root_metadata_valid,
            normalized_canonical,
        ) = (
            _canonical_absence_restore_progress_is_bound(
                canonical,
                parsed,
                capsule_baseline,
                allow_metadata_restore_progress=(
                    allow_capsule_restore_progress
                ),
            )
        )
        canonical_digest_valid = (
            canonical_digest_valid or progress_digest_valid
        )
        if progress_digest_valid and normalized_canonical is not None:
            canonical = normalized_canonical
    if any((
        os.fspath(canonical.root) != parsed.armed.get("canonical_root"),
        canonical.root_identity != _marker_identity(
            parsed.armed.get("canonical_root_identity"), "canonical root"
        ),
        not canonical_digest_valid,
    )):
        raise CampaignPlanError("dispatch quarantine canonical baseline changed")
    arm = parsed.recovery_arm
    if (
        arm is not None
        and _canonical_root_recovery_metadata(canonical)
        != arm.get("canonical_root_metadata")
        and not canonical_root_metadata_valid
    ):
        raise CampaignPlanError(
            "post-reboot canonical/WIP metadata differs from the armed state"
        )
    wip = _capture_wip_rollback(item)
    # ``wip`` is the exact post-incident state that pre-reboot arm seals.  It
    # is deliberately distinct from the pre-dispatch historical snapshot in
    # ``dispatch_armed`` when the interrupted child wrote a WIP capsule.
    wip_state_valid = True
    if arm is not None:
        wip_state_valid = (
            _wip_recovery_state_sha256(wip)
            == arm.get("wip_state_sha256")
        )
        if (
            not wip_state_valid
            and arm.get("wip_disposition") == "discard_latest_pointer"
        ):
            try:
                _validate_discarded_wip_state(item, arm, wip)
            except CampaignPlanError:
                wip_state_valid = False
            else:
                wip_state_valid = True
        if (
            not wip_state_valid
            and arm.get("wip_disposition") == "restore_historical_baseline"
            and capsule_baseline is not None
        ):
            try:
                _validate_capsule_restored_wip_state(
                    wip,
                    capsule_baseline,
                    parsed.armed.get("wip_restore_logical_state_sha256"),
                    logical_schema=str(parsed.armed.get(
                        "wip_restore_logical_state_schema"
                    )),
                )
            except CampaignPlanError:
                wip_state_valid = False
            else:
                wip_state_valid = True
        if (
            not wip_state_valid
            and allow_capsule_restore_progress
            and arm.get("wip_disposition") == "restore_historical_baseline"
            and capsule_baseline is not None
        ):
            _validate_capsule_restore_progress(wip, capsule_baseline)
            wip_state_valid = True
    if arm is not None and not wip_state_valid:
        raise CampaignPlanError(
            "post-reboot canonical/WIP metadata differs from the armed state"
        )
    reached = _checkpoint_reached(item["game"])
    expected_frontier = Status.validate_frontier_binding({
        field: item[field]
        for field in (
            *Status.FRONTIER_BINDING_FIELDS,
            "game",
            "reached",
            "target_level",
            "parent_action_count",
        )
    })
    if reached != item["reached"] or (
        _canonical_frontier_binding(item) != expected_frontier
    ):
        raise CampaignPlanError(
            "dispatch quarantine canonical frontier changed"
        )
    return artifact_root, canonical, wip, reached


def _validate_recovery_arm(
    *,
    item: dict[str, Any],
    marker: DispatchQuarantine,
    parsed: RebootRecovery.ParsedMarker,
    boot_identity: RebootRecovery.BootIdentity,
    record: dict[str, Any],
    canonical: CanonicalRollbackState,
    wip: WipRollbackState,
    capsule_baseline: WipRollbackState | None = None,
) -> str:
    arm = parsed.recovery_arm
    if arm is None:
        raise CampaignPlanError("dispatch marker is not armed for reboot recovery")
    RebootRecovery.validate_boot_identity(boot_identity)
    observed_wip_sha256 = _wip_recovery_state_sha256(wip)
    expected = {
        "dispatch_id": marker.dispatch_id,
        "boot_identity_source": boot_identity.source,
        "boot_identity": boot_identity.value,
        "marker_root_identity": list(marker.root_identity),
        "armed_marker_identity": list(marker.marker_identity),
        "projected_item_sha256": hashlib.sha256(json.dumps(
            item, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")).hexdigest(),
        "exec_record_sha256": _recovery_record_sha256(record),
        "canonical_root_metadata": _canonical_root_recovery_metadata(
            canonical
        ),
    }
    valid_wip_state = observed_wip_sha256 == arm.get("wip_state_sha256")
    if (
        not valid_wip_state
        and arm.get("wip_disposition") == "discard_latest_pointer"
    ):
        try:
            _validate_discarded_wip_state(item, arm, wip)
        except CampaignPlanError:
            valid_wip_state = False
        else:
            valid_wip_state = True
    if (
        not valid_wip_state
        and arm.get("wip_disposition") == "restore_historical_baseline"
        and capsule_baseline is not None
    ):
        try:
            _validate_capsule_restored_wip_state(
                wip,
                capsule_baseline,
                parsed.armed.get("wip_restore_logical_state_sha256"),
                logical_schema=str(parsed.armed.get(
                    "wip_restore_logical_state_schema"
                )),
            )
        except CampaignPlanError:
            valid_wip_state = False
        else:
            valid_wip_state = True
    v2_expected = {}
    if arm.get("recovery_arm_schema") is not None:
        v2_expected = {
            "recovery_arm_schema": RebootRecovery.RECOVERY_ARM_SCHEMA_V2,
            "historical_wip_snapshot": parsed.armed["target_wip_snapshot"],
            "confirmed_current_wip_state_sha256": arm.get(
                "wip_state_sha256"
            ),
        }
        if marker.schema == RebootRecovery.DISPATCH_QUARANTINE_SCHEMA_V2:
            v2_expected.update({
                "wip_recovery_authority": (
                    "dispatch_full_wip_rollback_capsule_v1"
                ),
                "wip_disposition": "restore_historical_baseline",
                "discard_survivor_sha256": None,
                "restored_wip_logical_state_sha256": parsed.armed[
                    "wip_restore_logical_state_sha256"
                ],
            })
        else:
            v2_expected["wip_recovery_authority"] = (
                "operator_confirmed_quarantined_wip_v1"
            )
            v2_expected["restored_wip_logical_state_sha256"] = None
    if (
        not valid_wip_state
        or any(arm.get(field) != value for field, value in expected.items())
        or any(arm.get(field) != value for field, value in v2_expected.items())
    ):
        raise CampaignPlanError(
            "post-reboot recovery arm does not bind the exact dispatch"
        )
    nonce = arm.get("recovery_nonce")
    if not isinstance(nonce, str):  # Parser already enforces the full shape.
        raise CampaignPlanError("post-reboot recovery arm nonce is malformed")
    return nonce


def _capture_recovery_workspace_lock(
    item: dict[str, Any], workspace: Path
) -> tuple[str, Path, tuple[int, int]]:
    schema = _lock_schema(item)
    lock_path = (
        workspace / ".orchestrate.lock"
        if schema == "in_workspace_v1"
        else Path(Legs._workspace_lock_path(os.fspath(workspace)))
    )
    try:
        lock = Legs._open_unaliased_lock(os.fspath(lock_path), create=False)
    except RuntimeError as exc:
        raise CampaignPlanError(
            "pre-reboot workspace lock is unavailable"
        ) from exc
    try:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise CampaignPlanError(
                "pre-reboot workspace lock remains active"
            ) from exc
        descriptor = os.fstat(lock.fileno())
        path_metadata = lock_path.stat(follow_symlinks=False)
        identity = (descriptor.st_dev, descriptor.st_ino)
        if (
            lock_path.is_symlink()
            or not stat.S_ISREG(path_metadata.st_mode)
            or path_metadata.st_nlink != 1
            or identity != (path_metadata.st_dev, path_metadata.st_ino)
        ):
            raise CampaignPlanError("pre-reboot workspace lock is aliased")
        return schema, lock_path, identity
    finally:
        lock.close()


def _open_held_recovery_workspace_lock(
    item: dict[str, Any], workspace: Path
) -> tuple[Any, str, Path, tuple[int, int]]:
    """Acquire the exact inactive lock without mutating abandoned metadata."""

    schema = _lock_schema(item)
    lock_path = (
        workspace / ".orchestrate.lock"
        if schema == "in_workspace_v1"
        else Path(Legs._workspace_lock_path(os.fspath(workspace)))
    )
    descriptor: int | None = None
    try:
        descriptor = os.open(
            lock_path,
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
        )
        lock = os.fdopen(descriptor, "rb")
        descriptor = None
    except OSError as exc:
        if descriptor is not None:
            os.close(descriptor)
        raise CampaignPlanError(
            "sandboxed-generation workspace lock is unavailable"
        ) from exc
    try:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise CampaignPlanError(
                "sandboxed-generation workspace lock remains active"
            ) from exc
        descriptor = os.fstat(lock.fileno())
        path_metadata = lock_path.stat(follow_symlinks=False)
        identity = (descriptor.st_dev, descriptor.st_ino)
        if (
            lock_path.is_symlink()
            or not stat.S_ISREG(descriptor.st_mode)
            or not stat.S_ISREG(path_metadata.st_mode)
            or descriptor.st_nlink != 1
            or path_metadata.st_nlink != 1
            or identity != (path_metadata.st_dev, path_metadata.st_ino)
            or descriptor.st_uid != os.geteuid()
            or path_metadata.st_uid != os.geteuid()
            or stat.S_IMODE(descriptor.st_mode) != 0o600
            or stat.S_IMODE(path_metadata.st_mode) != 0o600
        ):
            raise CampaignPlanError(
                "sandboxed-generation workspace lock is aliased"
            )
        return lock, schema, lock_path, identity
    except BaseException:
        lock.close()
        raise


def _generation_tree_observation_sha256(path: Path, *, label: str) -> str:
    """Hash one generation for audit only; this is never mutation authority."""

    root_identity = _host_directory_identity(path, label)
    entries: list[dict[str, Any]] = [{
        "path": ".", "kind": "directory", "identity": list(root_identity),
    }]
    total_bytes = 0
    try:
        for current, directories, files in os.walk(path, followlinks=False):
            base = Path(current)
            for name in sorted(directories):
                selected = base / name
                metadata = selected.stat(follow_symlinks=False)
                if selected.is_symlink() or not stat.S_ISDIR(metadata.st_mode):
                    raise CampaignPlanError(f"{label} contains an unsafe directory")
                entries.append({
                    "path": selected.relative_to(path).as_posix(),
                    "kind": "directory",
                    "identity": [metadata.st_dev, metadata.st_ino],
                })
            for name in sorted(files):
                selected = base / name
                metadata = selected.stat(follow_symlinks=False)
                if (
                    selected.is_symlink()
                    or not stat.S_ISREG(metadata.st_mode)
                    or metadata.st_nlink != 1
                ):
                    raise CampaignPlanError(f"{label} contains an unsafe file")
                payload = Legs._read_single_link_regular(os.fspath(selected))
                total_bytes += len(payload)
                entries.append({
                    "path": selected.relative_to(path).as_posix(),
                    "kind": "file",
                    "bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                })
                if total_bytes > MAX_WIP_SNAPSHOT_BYTES:
                    raise CampaignPlanError(f"{label} exceeds its byte bound")
            if len(entries) > MAX_WIP_SNAPSHOT_ENTRIES:
                raise CampaignPlanError(f"{label} exceeds its entry bound")
    except (OSError, Legs.WorkspaceTainted) as exc:
        raise CampaignPlanError(f"{label} observation is unstable") from exc
    if _host_directory_identity(path, label) != root_identity:
        raise CampaignPlanError(f"{label} root identity changed during observation")
    return hashlib.sha256(json.dumps(
        entries, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")).hexdigest()


def _observe_named_root_group_absence(
    *, scheduler_pid: int, child_pid: int
) -> dict[str, Any]:
    """Observe named roots/owned PGID only; never signal or claim descendants."""

    if sys.platform != "darwin":
        raise CampaignPlanError(
            "sandboxed-generation release is restricted to the Darwin incident"
        )
    first_at = datetime.now(timezone.utc)
    started_ns = time.monotonic_ns()
    for index in range(SANDBOX_ABSENCE_SAMPLES):
        for pid, label in ((scheduler_pid, "scheduler"), (child_pid, "child")):
            identity = Contiguous._process_identity(pid)
            if identity is not None and not identity[3].startswith("Z"):
                raise CampaignPlanError(
                    f"sandboxed-generation {label} PID remains live"
                )
        if Contiguous._process_group_has_live_members(child_pid):
            raise CampaignPlanError(
                "sandboxed-generation owned process group remains live"
            )
        if index + 1 < SANDBOX_ABSENCE_SAMPLES:
            time.sleep(SANDBOX_ABSENCE_INTERVAL_SECONDS)
    last_at = datetime.now(timezone.utc)
    return {
        "absence_sample_count": SANDBOX_ABSENCE_SAMPLES,
        "absence_window_ns": max(1, time.monotonic_ns() - started_ns),
        "absence_first_at": first_at.isoformat(),
        "absence_last_at": last_at.isoformat(),
    }


def _sandboxed_generation_paths(
    item: dict[str, Any], parsed: RebootRecovery.ParsedMarker
) -> tuple[Path, tuple[int, int], Path, Path]:
    historical = parsed.armed.get("historical_runner")
    if (
        not isinstance(historical, dict)
        or historical.get("source_sha256") not in SANDBOX_CONTRACTS
    ):
        raise CampaignPlanError(
            "sandboxed-generation release lacks an approved historical runner"
        )
    _revalidate_historical_control(item, allow_abandoned_scratch=True)
    scratch = _normalized_absolute_path(
        historical.get("scratch_root"), "scratch_root"
    )
    scratch_identity = _host_directory_identity(scratch, "abandoned scratch root")
    workspace, protected = _post_reboot_generation_paths(item, parsed.unquiesced)
    _validate_post_reboot_generation_identities(
        parsed, workspace, protected, require_both=True
    )
    return scratch, scratch_identity, workspace, protected


def _validate_sandboxed_generation_baseline(
    item: dict[str, Any],
    marker: DispatchQuarantine,
    parsed: RebootRecovery.ParsedMarker,
) -> tuple[LedgerPrefixState, CanonicalRollbackState, WipRollbackState]:
    _validate_post_reboot_dispatch_binding(item, parsed)
    if marker.capsule_state is None or marker.capsule_record is None:
        raise CampaignPlanError(
            "sandboxed-generation release requires its full WIP capsule"
        )
    ledger, baseline, suffix = _read_post_reboot_ledger_surface(
        item,
        parsed.armed,
        intent_root=marker.root,
        intent_root_identity=marker.root_identity,
    )
    if suffix:
        raise CampaignPlanError(
            "sandboxed-generation release requires an exact zero ledger suffix"
        )
    canonical = _capture_canonical_rollback(item)
    if any((
        list(canonical.root_identity)
        != parsed.armed.get("canonical_root_identity"),
        canonical.digest != parsed.armed.get("canonical_digest"),
        _checkpoint_reached(item["game"]) != item["reached"],
        _canonical_frontier_binding(item)
        != Status.validate_frontier_binding(dict(parsed.armed["frontier_binding"])),
    )):
        raise CampaignPlanError(
            "sandboxed-generation canonical/frontier baseline changed"
        )
    return baseline, canonical, _capture_wip_rollback(item)


def _arm_sandboxed_generation_release_locked(
    item: dict[str, Any],
    *,
    confirm_dispatch_id: str,
    boot_identity_provider: RebootRecovery.BootIdentityProvider,
) -> dict[str, Any]:
    marker, parsed = _read_existing_dispatch_quarantine(
        item,
        require_recovery_arm=None,
        marker_parser=RebootRecovery.parse_sandboxed_generation_marker,
    )
    held_lock: Any | None = None
    try:
        item = _reconstruct_historical_recovery_item(
            item, parsed.armed, allow_abandoned_scratch=True
        )
        if confirm_dispatch_id != marker.dispatch_id:
            raise CampaignPlanError(
                "operator confirmation does not match the sandboxed dispatch"
            )
        boot = RebootRecovery.validate_boot_identity(boot_identity_provider())
        baseline, canonical, current_wip = _validate_sandboxed_generation_baseline(
            item, marker, parsed
        )
        scratch, scratch_identity, workspace, protected = (
            _sandboxed_generation_paths(item, parsed)
        )
        held_lock, lock_schema, lock_path, lock_identity = (
            _open_held_recovery_workspace_lock(item, workspace)
        )
        absence = _observe_named_root_group_absence(
            scheduler_pid=int(parsed.armed["pid"]),
            child_pid=int(parsed.unquiesced["child_pid"]),
        )
        if parsed.recovery_arm is not None:
            arm = parsed.recovery_arm
            if any((
                arm.get("boot_identity_source") != boot.source,
                arm.get("boot_identity") != boot.value,
                arm.get("scratch_root") != os.fspath(scratch),
                arm.get("scratch_root_identity") != list(scratch_identity),
                arm.get("workspace_lock_identity") != list(lock_identity),
                arm.get("wip_state_sha256")
                != _wip_recovery_state_sha256(current_wip),
            )):
                raise CampaignPlanError(
                    "installed sandboxed-generation arm context changed"
                )
            return {
                "game": item["game"], "target_level": item["target_level"],
                "result": "sandboxed_generation_release_already_armed",
                "dispatch_id": marker.dispatch_id,
                "recovery_nonce": arm["recovery_nonce"],
            }
        _taint_gate()
        pre_arm_size = marker.recovery_sealed_size
        pre_arm_sha = marker.recovery_sealed_sha256
        if pre_arm_size is None or pre_arm_sha is None:
            raise CampaignPlanError("sandboxed-generation marker seal is unavailable")
        historical = parsed.armed["historical_runner"]
        assert isinstance(historical, dict)
        arm_record = {
            "event": RebootRecovery.SANDBOXED_GENERATION_ARM_EVENT,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "recovery_arm_schema": RebootRecovery.SANDBOXED_GENERATION_ARM_SCHEMA,
            "recovery_nonce": os.urandom(16).hex(),
            "boot_identity_source": boot.source,
            "boot_identity": boot.value,
            "marker_root_identity": list(marker.root_identity),
            "pre_arm_marker_identity": list(marker.marker_identity),
            "pre_arm_marker_bytes": pre_arm_size,
            "pre_arm_marker_sha256": pre_arm_sha,
            "projected_item_sha256": parsed.armed["projected_item_sha256"],
            "historical_runner_sha256": hashlib.sha256(
                RebootRecovery.canonical_json_line(historical)
            ).hexdigest(),
            "authority_kind": "explicit_operator_assumed_artifact_isolation_v1",
            "operator_provenance_assumption": (
                "historical_codex_workspace_write_effective_as_invoked"
            ),
            "sandbox_contract_sha256": SANDBOX_CONTRACTS[
                str(historical["source_sha256"])
            ],
            "process_claim": "named_root_and_owned_group_absent_only",
            "process_tree_quiesced": False,
            "detached_processes_proven_absent": False,
            "isolation_claim": (
                "published_artifact_namespace_unreachable_by_assumption"
            ),
            "scheduler_pid": parsed.armed["pid"],
            "child_pid": parsed.unquiesced["child_pid"],
            "child_pgid": parsed.unquiesced["child_pid"],
            **absence,
            "scratch_root": os.fspath(scratch),
            "scratch_root_identity": list(scratch_identity),
            "scratch_root_disposition": "abandoned_in_place",
            "required_retry_scratch_relation": "outside_abandoned_path_and_inode",
            "workspace": parsed.unquiesced["workspace"],
            "workspace_identity": parsed.unquiesced["workspace_identity"],
            "workspace_tree_observation_sha256": (
                _generation_tree_observation_sha256(
                    workspace, label="workspace audit observation"
                )
            ),
            "protected": parsed.unquiesced["protected"],
            "protected_identity": parsed.unquiesced["protected_identity"],
            "protected_tree_sha256": _generation_tree_observation_sha256(
                protected, label="protected generation"
            ),
            "workspace_lock_schema": lock_schema,
            "workspace_lock_path": os.fspath(lock_path),
            "workspace_lock_identity": list(lock_identity),
            "canonical_digest": canonical.digest,
            "ledger_prefix_bytes": len(baseline.raw_prefix),
            "ledger_prefix_sha256": hashlib.sha256(baseline.raw_prefix).hexdigest(),
            "wip_state_sha256": _wip_recovery_state_sha256(current_wip),
            **{
                field: parsed.armed[field]
                for field in (
                    "wip_rollback_capsule_name",
                    "wip_rollback_capsule_identity",
                    "wip_rollback_capsule_bytes",
                    "wip_rollback_capsule_sha256",
                    "wip_rollback_capsule_state_sha256",
                    "wip_restore_logical_state_schema",
                    "wip_restore_logical_state_sha256",
                )
            },
            "operator_artifact_scanner_assumption": (
                "current_full_artifact_scanner_reported_pass"
            ),
        }
        # The scanner/tree observations above are deliberately not mutation
        # authority.  Rebind every authoritative surface immediately before
        # the one arm installation while retaining the inode-bound flock.
        rebound_baseline, rebound_canonical, rebound_wip = (
            _validate_sandboxed_generation_baseline(item, marker, parsed)
        )
        (
            rebound_scratch,
            rebound_scratch_identity,
            rebound_workspace,
            rebound_protected,
        ) = _sandboxed_generation_paths(item, parsed)
        lock_stat = os.fstat(held_lock.fileno())
        lock_path_stat = lock_path.stat(follow_symlinks=False)
        rebound_absence = _observe_named_root_group_absence(
            scheduler_pid=int(parsed.armed["pid"]),
            child_pid=int(parsed.unquiesced["child_pid"]),
        )
        if any((
            rebound_baseline.raw_prefix != baseline.raw_prefix,
            rebound_baseline.file_identity != baseline.file_identity,
            rebound_canonical.digest != canonical.digest,
            _wip_recovery_state_sha256(rebound_wip)
            != arm_record["wip_state_sha256"],
            rebound_scratch != scratch,
            rebound_scratch_identity != scratch_identity,
            rebound_workspace != workspace,
            rebound_protected != protected,
            (lock_stat.st_dev, lock_stat.st_ino) != lock_identity,
            (lock_path_stat.st_dev, lock_path_stat.st_ino) != lock_identity,
            rebound_absence["absence_sample_count"]
            != SANDBOX_ABSENCE_SAMPLES,
        )):
            raise CampaignPlanError(
                "sandboxed-generation authority changed before arm install"
            )
        installed = _atomic_recovery_arm_replace(
            marker,
            arm_record,
            marker_parser=RebootRecovery.parse_sandboxed_generation_marker,
            sidecar_suffix="sandboxed_generation_arm",
        )
        return {
            "game": item["game"], "target_level": item["target_level"],
            "result": "sandboxed_generation_release_armed",
            "dispatch_id": marker.dispatch_id,
            "recovery_nonce": installed["recovery_nonce"],
            "process_tree_quiesced": False,
            "detached_processes_proven_absent": False,
        }
    except RebootRecovery.RecoveryEvidenceError as exc:
        raise CampaignPlanError(str(exc)) from exc
    finally:
        if held_lock is not None:
            held_lock.close()
        _close_dispatch_quarantine(marker)


def _arm_sandboxed_generation_release(
    item: dict[str, Any],
    *,
    confirm_dispatch_id: str,
    boot_identity_provider: RebootRecovery.BootIdentityProvider = (
        RebootRecovery.authoritative_boot_identity
    ),
) -> dict[str, Any]:
    dispatch_lock = _acquire_scheduler_dispatch_lock(item)
    try:
        lineage_lock = _acquire_scheduler_lineage_lock(item)
        try:
            return _arm_sandboxed_generation_release_locked(
                item,
                confirm_dispatch_id=confirm_dispatch_id,
                boot_identity_provider=boot_identity_provider,
            )
        finally:
            _release_scheduler_artifact_lock(lineage_lock)
    finally:
        _release_scheduler_artifact_lock(dispatch_lock)


def _sandbox_wip_is_restored(
    current: WipRollbackState,
    marker: DispatchQuarantine,
    parsed: RebootRecovery.ParsedMarker,
) -> bool:
    if marker.capsule_state is None:
        raise CampaignPlanError("sandbox isolation lost its WIP capsule")
    try:
        _validate_capsule_restored_wip_state(
            current,
            marker.capsule_state,
            parsed.armed.get("wip_restore_logical_state_sha256"),
            logical_schema=str(parsed.armed.get(
                "wip_restore_logical_state_schema"
            )),
        )
    except CampaignPlanError:
        return False
    return True


def _recover_sandboxed_generation_release_locked(
    item: dict[str, Any],
    *,
    confirm_dispatch_id: str,
    confirm_recovery_nonce: str,
    boot_identity_provider: RebootRecovery.BootIdentityProvider,
) -> dict[str, Any]:
    marker, parsed = _read_existing_dispatch_quarantine(
        item,
        require_recovery_arm=True,
        marker_parser=RebootRecovery.parse_sandboxed_generation_marker,
    )
    held_lock: Any | None = None
    released = False
    try:
        item = _reconstruct_historical_recovery_item(
            item, parsed.armed, allow_abandoned_scratch=True
        )
        arm = parsed.recovery_arm
        if arm is None or any((
            confirm_dispatch_id != marker.dispatch_id,
            confirm_recovery_nonce != arm.get("recovery_nonce"),
        )):
            raise CampaignPlanError(
                "operator confirmation does not match the sandboxed arm"
            )
        boot = RebootRecovery.validate_boot_identity(boot_identity_provider())
        if (
            arm.get("boot_identity_source") != boot.source
            or arm.get("boot_identity") != boot.value
        ):
            raise CampaignPlanError(
                "sandboxed-generation recovery changed boot session"
            )
        scratch, scratch_identity, workspace, protected = (
            _sandboxed_generation_paths(item, parsed)
        )
        if any((
            os.fspath(scratch) != arm.get("scratch_root"),
            list(scratch_identity) != arm.get("scratch_root_identity"),
        )):
            raise CampaignPlanError(
                "abandoned scratch namespace changed before recovery"
            )
        held_lock, lock_schema, lock_path, lock_identity = (
            _open_held_recovery_workspace_lock(item, workspace)
        )
        if any((
            lock_schema != arm.get("workspace_lock_schema"),
            os.fspath(lock_path) != arm.get("workspace_lock_path"),
            list(lock_identity) != arm.get("workspace_lock_identity"),
        )):
            raise CampaignPlanError(
                "sandboxed-generation lock binding changed before recovery"
            )
        _observe_named_root_group_absence(
            scheduler_pid=int(arm["scheduler_pid"]),
            child_pid=int(arm["child_pid"]),
        )
        # Reconcile runs before marker recovery.  If it found a validated WAL
        # whose authorization row was empty or only a canonical prefix, this
        # authenticated operator path can now roll back exactly that suffix
        # and retire exactly that WAL inode before strict ledger parsing.
        _retire_incomplete_release_for_operator(item, marker)
        ledger, baseline, suffix = _read_post_reboot_ledger_surface(
            item,
            parsed.armed,
            intent_root=marker.root,
            intent_root_identity=marker.root_identity,
        )
        if not (
            not suffix
            or (
                len(suffix) == 1
                and suffix[0].get("event") == SANDBOX_ABANDON_EVENT
            )
        ):
            raise CampaignPlanError(
                "sandbox isolation has an ambiguous ledger suffix"
            )
        canonical = _capture_canonical_rollback(item)
        if any((
            list(canonical.root_identity)
            != parsed.armed.get("canonical_root_identity"),
            canonical.digest != arm.get("canonical_digest"),
            _checkpoint_reached(item["game"]) != item["reached"],
            _canonical_frontier_binding(item)
            != Status.validate_frontier_binding(
                dict(parsed.armed["frontier_binding"])
            ),
        )):
            raise CampaignPlanError(
                "sandbox isolation canonical/frontier baseline changed"
            )
        current_wip = _capture_wip_rollback(item)
        restored = _sandbox_wip_is_restored(current_wip, marker, parsed)
        if (
            not suffix
            and not restored
            and _wip_recovery_state_sha256(current_wip)
            != arm.get("wip_state_sha256")
        ):
            raise CampaignPlanError(
                "sandbox isolation disposable WIP state changed"
            )
        if not suffix:
            event = _build_sandbox_abandon_event(item, parsed)
            _validate_sandbox_abandon_event(item, parsed, event)
            committed = _append_zero_ledger_event_cas(
                marker=marker, baseline=baseline, record=event
            )
            _validate_sandbox_abandon_event(item, parsed, committed)
            suffix = [committed]
        else:
            _validate_sandbox_abandon_event(item, parsed, suffix[0])

        # The durable abandonment row is the mutation authority for an
        # idempotent capsule restore.  A crash at any later point can resume
        # only from the capsule's narrowly validated restore envelope.
        current_wip = _capture_wip_rollback(item)
        restored = _sandbox_wip_is_restored(current_wip, marker, parsed)
        if not restored:
            if marker.capsule_state is None or marker.capsule_record is None:
                raise CampaignPlanError(
                    "sandbox isolation lost its rollback capsule"
                )
            if (
                _wip_recovery_state_sha256(current_wip)
                != arm.get("wip_state_sha256")
            ):
                _validate_capsule_restore_progress(
                    current_wip, marker.capsule_state
                )
            current_wip = _restore_wip_from_rollback_capsule(
                item, marker.capsule_state, marker.capsule_record
            )
            restored = _sandbox_wip_is_restored(
                current_wip, marker, parsed
            )
        if not restored:
            raise CampaignPlanError(
                "sandbox isolation failed to restore sealed WIP"
            )
        if (
            _capture_canonical_rollback(item).digest
            != arm.get("canonical_digest")
        ):
            raise CampaignPlanError(
                "canonical artifact changed during WIP restore"
            )
        # The old scratch root is never renamed, deleted, or otherwise
        # traversed for cleanup.  Its durable ledger receipt now blocks reuse.
        if any((
            _host_directory_identity(scratch, "abandoned scratch root")
            != scratch_identity,
            _host_directory_identity(workspace, "abandoned workspace")
            != _marker_identity(
                parsed.unquiesced["workspace_identity"], "workspace"
            ),
            _host_directory_identity(protected, "abandoned protected evidence")
            != _marker_identity(
                parsed.unquiesced["protected_identity"], "protected"
            ),
        )):
            raise CampaignPlanError(
                "abandoned sandbox namespace changed during artifact restore"
            )
        result = _sandbox_isolation_result(item, parsed)
        if set(result) != SANDBOX_ISOLATION_RESULT_KEYS:
            raise CampaignPlanError(
                "sandbox isolation terminal result schema changed"
            )
        release_authority = _build_dispatch_release_authority(
            item,
            marker,
            ledger,
            result,
            kind=SANDBOX_RELEASE_AUTHORITY_KIND,
        )
        _release_dispatch_quarantine(marker, item, release_authority)
        released = True
        return result
    except RebootRecovery.RecoveryEvidenceError as exc:
        raise CampaignPlanError(str(exc)) from exc
    finally:
        if held_lock is not None:
            held_lock.close()
        if not released:
            _close_dispatch_quarantine(marker)


def _completed_sandbox_isolation_result(
    item: dict[str, Any],
    *,
    confirm_dispatch_id: str,
    confirm_recovery_nonce: str,
) -> dict[str, Any] | None:
    historical = item.get("historical_runner")
    cwd = (
        Path(str(historical["cwd"]))
        if isinstance(historical, dict) and historical.get("cwd") is not None
        else REPO
    )
    ledger = _ledger_path(item["argv"], cwd=cwd)
    captured = _capture_ledger_prefix(ledger)
    records = captured.records
    matches = [
        (index, record) for index, record in enumerate(records)
        if (
            record.get("event") == SANDBOX_ABANDON_EVENT
            and record.get("schema") == SANDBOX_ABANDON_EVENT_SCHEMA
            and record.get("dispatch_id") == confirm_dispatch_id
            and record.get("recovery_nonce") == confirm_recovery_nonce
            and record.get("game") == item.get("game")
            and record.get("target_level") == item.get("target_level")
        )
    ]
    if not matches:
        return None
    if len(matches) != 1:
        raise CampaignPlanError(
            "sandbox isolation completion receipt is ambiguous"
        )
    event_index, event = matches[0]
    _validate_sandbox_abandon_event(item, event)
    if event_index + 1 >= len(records):
        raise CampaignPlanError(
            "sandbox isolation completion lacks release authorization"
        )
    authority = records[event_index + 1]
    raw_lines = captured.raw_prefix.splitlines(keepends=True)
    if len(raw_lines) != len(records):
        raise CampaignPlanError(
            "sandbox isolation completion ledger framing changed"
        )
    event_line = RebootRecovery.canonical_json_line(event)
    authority_line = RebootRecovery.canonical_json_line(authority)
    prefix_through_event = b"".join(raw_lines[:event_index + 1])
    marker_name = _dispatch_quarantine_name(item)
    intent_name, _preparing_name = _dispatch_release_intent_names(
        marker_name
    )
    expected_authority = {
        "event": "codex_dispatch_release_authorized",
        "schema": "scheduler_dispatch_release_authorized_v1",
        "dispatch_id": confirm_dispatch_id,
        "intent_name": intent_name,
        "projected_item_sha256": hashlib.sha256(json.dumps(
            item, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")).hexdigest(),
        "game": item["game"],
        "target_level": item["target_level"],
        "retry_complexity_n": item["retry_complexity_n"],
        "reached": item["reached"],
        "parent_action_count": item["parent_action_count"],
        "terminal_kind": SANDBOX_RELEASE_AUTHORITY_KIND,
        "terminal_event": SANDBOX_ABANDON_EVENT,
        "terminal_record_sha256": _recovery_record_sha256(event),
        "ledger": os.fspath(ledger),
        "ledger_parent_identity": list(captured.parent_identity),
        "ledger_file_identity": (
            list(captured.file_identity)
            if captured.file_identity is not None else None
        ),
        "ledger_prefix_bytes": len(prefix_through_event),
        "ledger_prefix_sha256": hashlib.sha256(
            prefix_through_event
        ).hexdigest(),
        **{
            field: item[field]
            for field in Status.FRONTIER_BINDING_FIELDS
        },
    }
    matching_authorities = [
        row for row in records
        if (
            row.get("event") == "codex_dispatch_release_authorized"
            and row.get("dispatch_id") == confirm_dispatch_id
            and row.get("terminal_kind") == SANDBOX_RELEASE_AUTHORITY_KIND
            and row.get("terminal_event") == SANDBOX_ABANDON_EVENT
        )
    ]
    if (
        len(matching_authorities) != 1
        or authority is not matching_authorities[0]
        or set(authority) != _DISPATCH_RELEASE_AUTHORITY_RECORD_KEYS
        or raw_lines[event_index] != event_line
        or raw_lines[event_index + 1] != authority_line
        or any(
            authority.get(field) != value
            for field, value in expected_authority.items()
        )
        or not isinstance(authority.get("release_nonce"), str)
        or SHA256_RE.fullmatch(authority["release_nonce"]) is None
        or not isinstance(authority.get("intent_core_sha256"), str)
        or SHA256_RE.fullmatch(authority["intent_core_sha256"]) is None
    ):
        raise CampaignPlanError(
            "sandbox isolation release authorization is invalid"
        )
    _marker_identity(
        authority.get("intent_identity"), "sandbox release intent"
    )
    event_at = _recovery_recorded_at(event, "sandbox isolation event")
    authority_at = _recovery_recorded_at(
        authority, "sandbox release authorization"
    )
    if authority_at < event_at:
        raise CampaignPlanError(
            "sandbox release authorization predates its terminal event"
        )
    return {
        "game": item["game"],
        "target_level": item["target_level"],
        "reached": item["reached"],
        "result": "sandbox_isolated_noncounting_already_completed",
        "dispatch_id": confirm_dispatch_id,
        "scratch_root": event["scratch_root"],
        "scratch_root_disposition": "abandoned_in_place",
        "process_tree_quiesced": False,
        "detached_processes_proven_absent": False,
    }


def _recover_sandboxed_generation_release(
    item: dict[str, Any],
    *,
    confirm_dispatch_id: str,
    confirm_recovery_nonce: str,
    boot_identity_provider: RebootRecovery.BootIdentityProvider = (
        RebootRecovery.authoritative_boot_identity
    ),
) -> dict[str, Any]:
    dispatch_lock = _acquire_scheduler_dispatch_lock(item)
    try:
        lineage_lock = _acquire_scheduler_lineage_lock(item)
        try:
            reconciled = _preflight_post_reboot_release_reconciliation(item)
            if reconciled:
                completed = _completed_sandbox_isolation_result(
                    item,
                    confirm_dispatch_id=confirm_dispatch_id,
                    confirm_recovery_nonce=confirm_recovery_nonce,
                )
                if completed is None:
                    raise CampaignPlanError(
                        "reconciled sandbox release lacks its completion pair"
                    )
                return completed
            try:
                return _recover_sandboxed_generation_release_locked(
                    item,
                    confirm_dispatch_id=confirm_dispatch_id,
                    confirm_recovery_nonce=confirm_recovery_nonce,
                    boot_identity_provider=boot_identity_provider,
                )
            except NoDispatchQuarantine:
                completed = _completed_sandbox_isolation_result(
                    item,
                    confirm_dispatch_id=confirm_dispatch_id,
                    confirm_recovery_nonce=confirm_recovery_nonce,
                )
                if completed is None:
                    raise
                return completed
        finally:
            _release_scheduler_artifact_lock(lineage_lock)
    finally:
        _release_scheduler_artifact_lock(dispatch_lock)


def _arm_incomplete_safe_release_recovery(
    item: dict[str, Any],
    *,
    confirm_dispatch_id: str,
    boot_identity_provider: RebootRecovery.BootIdentityProvider,
) -> dict[str, Any] | None:
    """Arm an ordinary terminal whose release authority is incomplete."""

    opened = _open_dispatch_quarantine_root(item, create=False)
    if opened is None:
        return None
    root, root_fd, root_identity = opened
    marker_name = _dispatch_quarantine_name(item)
    arm_name = _safe_release_recovery_arm_name(marker_name)
    marker: DispatchQuarantine | None = None
    try:
        wal = _safe_release_wal_inventory(root_fd, marker_name)
        try:
            arm, _arm_identity, _arm_payload = (
                _read_durable_recovery_record_at(
                    root_fd,
                    arm_name,
                    root_path=root,
                    root_identity=root_identity,
                    label="safe-release recovery arm",
                )
            )
        except FileNotFoundError:
            arm = None
        if wal is None and arm is None:
            return None
        try:
            boot = RebootRecovery.validate_boot_identity(
                boot_identity_provider()
            )
        except RebootRecovery.RecoveryEvidenceError as exc:
            raise CampaignPlanError(str(exc)) from exc
        if arm is not None:
            _validate_safe_release_arm_context(
                item,
                root_fd,
                root_identity,
                arm,
                require_original_wal=True,
            )
            if any((
                confirm_dispatch_id != arm.get("dispatch_id"),
                boot.source != arm.get("boot_identity_source"),
                boot.value != arm.get("boot_identity"),
            )):
                raise CampaignPlanError(
                    "safe-release arm confirmation or boot identity changed"
                )
            _validate_durable_recovery_root_binding(
                root_fd,
                root,
                root_identity,
                label="safe-release recovery arm",
            )
            return {
                "game": item["game"],
                "target_level": item["target_level"],
                "result": "post_reboot_safe_release_already_armed",
                "dispatch_id": arm["dispatch_id"],
                "recovery_nonce": arm["recovery_nonce"],
            }
        assert wal is not None
        role, wal_name = wal
        intent, intent_identity = _read_dispatch_release_intent_at(
            root_fd, wal_name, marker_name=marker_name
        )
        if (
            RebootRecovery.DISPATCH_ID_RE.fullmatch(confirm_dispatch_id) is None
            or confirm_dispatch_id != intent.get("dispatch_id")
        ):
            raise CampaignPlanError(
                "operator confirmation does not match the safe terminal"
            )
        marker, _armed, marker_payload = _open_safe_release_marker(
            item, root, root_fd, root_identity, intent
        )
        state = _prevalidate_dispatch_release_preparing(
            item, root_fd, intent, intent_identity
        )
        if state == "authorized":
            raise CampaignPlanError(
                "complete release authority requires ordinary reconciliation"
            )
        authority = intent["release_authority"]
        assert isinstance(authority, dict)
        ledger = _dispatch_release_item_ledger(item, authority)
        with Guard.ledger_append_lock(ledger):
            raw, _parent, _file = _read_dispatch_release_ledger_locked(
                ledger, authority
            )
            _prefix, line, tail = _dispatch_release_authority_tail(
                raw, authority, dispatch_id=confirm_dispatch_id
            )
        wal_payload = RebootRecovery.canonical_json_line(intent)
        arm = {
            "schema": SAFE_RELEASE_RECOVERY_ARM_SCHEMA,
            "event": SAFE_RELEASE_RECOVERY_ARM_EVENT,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "dispatch_id": confirm_dispatch_id,
            "recovery_nonce": os.urandom(16).hex(),
            "boot_identity_source": boot.source,
            "boot_identity": boot.value,
            "marker_root": os.fspath(root),
            "marker_root_identity": list(root_identity),
            "marker_name": marker_name,
            "marker_identity": list(marker.marker_identity),
            "marker_bytes": len(marker_payload),
            "marker_sha256": hashlib.sha256(marker_payload).hexdigest(),
            "projected_item_sha256": authority["projected_item_sha256"],
            "release_wal_name": wal_name,
            "release_wal_role": role,
            "release_wal_identity": list(intent_identity),
            "release_wal_bytes": len(wal_payload),
            "release_wal_sha256": hashlib.sha256(wal_payload).hexdigest(),
            "authority_tail_bytes": len(tail),
            "authority_tail_sha256": hashlib.sha256(tail).hexdigest(),
            "release_intent": intent,
        }
        _validate_safe_release_recovery_arm_record(
            arm, marker_name=marker_name
        )
        _install_durable_recovery_record_at(
            root_fd,
            arm_name,
            arm,
            root_path=root,
            root_identity=root_identity,
            label="safe-release recovery arm",
        )
        installed, _identity, _payload = _read_durable_recovery_record_at(
            root_fd,
            arm_name,
            root_path=root,
            root_identity=root_identity,
            label="safe-release recovery arm",
        )
        if installed != arm:
            raise CampaignPlanError("installed safe-release arm changed")
        _validate_safe_release_arm_context(
            item,
            root_fd,
            root_identity,
            arm,
            require_original_wal=True,
        )
        _validate_durable_recovery_root_binding(
            root_fd,
            root,
            root_identity,
            label="safe-release recovery arm",
        )
        return {
            "game": item["game"],
            "target_level": item["target_level"],
            "result": "post_reboot_safe_release_armed",
            "dispatch_id": confirm_dispatch_id,
            "recovery_nonce": arm["recovery_nonce"],
        }
    finally:
        if marker is not None:
            _close_dispatch_quarantine(marker)
        else:
            os.close(root_fd)


def _arm_post_reboot_recovery_locked(
    item: dict[str, Any],
    *,
    confirm_dispatch_id: str,
    confirm_current_wip_state_sha256: str | None,
    boot_identity_provider: RebootRecovery.BootIdentityProvider,
) -> dict[str, Any]:
    safe_release = _arm_incomplete_safe_release_recovery(
        item,
        confirm_dispatch_id=confirm_dispatch_id,
        boot_identity_provider=boot_identity_provider,
    )
    if safe_release is not None:
        return safe_release
    marker, parsed = _read_existing_dispatch_quarantine(
        item, require_recovery_arm=None
    )
    try:
        item = _reconstruct_historical_recovery_item(item, parsed.armed)
        if (
            RebootRecovery.DISPATCH_ID_RE.fullmatch(confirm_dispatch_id) is None
            or confirm_dispatch_id != marker.dispatch_id
        ):
            raise CampaignPlanError(
                "operator confirmation does not match the quarantined dispatch"
            )
        try:
            boot_identity = RebootRecovery.validate_boot_identity(
                boot_identity_provider()
            )
        except RebootRecovery.RecoveryEvidenceError as exc:
            raise CampaignPlanError(str(exc)) from exc
        _validate_recovery_marker_seal(marker)
        _artifact_root_bound, canonical, wip, _reached = (
            _rebind_recovery_baselines(
                item, parsed, capsule_baseline=marker.capsule_state
            )
        )
        state = _pre_reboot_exec_state(item, parsed)
        (
            workspace,
            protected,
            _reason,
            _transcript_sha,
            _diagnostics_sha,
            _descendant_unproven,
        ) = _exact_tainted_generation(item, state.record, require_taint=False)
        if (
            _host_directory_identity(workspace, "quarantined workspace")
            != _marker_identity(
                parsed.unquiesced.get("workspace_identity"), "workspace"
            )
            or _host_directory_identity(
                protected, "quarantined protected evidence"
            ) != _marker_identity(
                parsed.unquiesced.get("protected_identity"), "protected"
            )
        ):
            raise CampaignPlanError(
                "dispatch quarantine generation identity changed"
            )
        lock_schema, lock_path, lock_identity = (
            _capture_recovery_workspace_lock(item, workspace)
        )
        if parsed.recovery_arm is not None:
            if parsed.recovery_arm.get("recovery_arm_schema") != (
                RebootRecovery.RECOVERY_ARM_SCHEMA_V2
            ):
                raise CampaignPlanError(
                    "legacy recovery arm lacks explicit current-WIP authority"
                )
            nonce = _validate_recovery_arm(
                item=item,
                marker=marker,
                parsed=parsed,
                boot_identity=boot_identity,
                record=state.record,
                canonical=canonical,
                wip=wip,
                capsule_baseline=marker.capsule_state,
            )
            if any((
                parsed.recovery_arm.get("workspace_lock_schema")
                != lock_schema,
                parsed.recovery_arm.get("workspace_lock_path")
                != os.fspath(lock_path),
                _marker_identity(
                    parsed.recovery_arm.get("workspace_lock_identity"),
                    "workspace lock",
                ) != lock_identity,
            )):
                raise CampaignPlanError(
                    "post-reboot recovery arm workspace lock changed"
                )
            try:
                os.fsync(marker.root_fd)
            except OSError as exc:
                raise CampaignPlanError(
                    "could not durably confirm the installed recovery arm"
                ) from exc
            _validate_recovery_marker_seal(marker)
            return {
                "game": item["game"],
                "target_level": item["target_level"],
                "result": "post_reboot_recovery_already_armed",
                "dispatch_id": marker.dispatch_id,
                "recovery_nonce": nonce,
            }
        current_wip_sha256 = _wip_recovery_state_sha256(wip)
        requires_wip_confirmation = _legacy_wip_requires_confirmation(
            parsed, wip
        )
        if requires_wip_confirmation:
            _validate_legacy_wip_exclusion_item(item)
            if confirm_current_wip_state_sha256 is None:
                return {
                    "game": item["game"],
                    "target_level": item["target_level"],
                    "result": "post_reboot_wip_confirmation_required",
                    "dispatch_id": marker.dispatch_id,
                    "current_wip_state_sha256": current_wip_sha256,
                    "wip_recovery_authority": (
                        "operator_confirmed_quarantined_wip_v1"
                    ),
                    "wip_disposition": (
                        "discard_latest_pointer"
                        if wip.entries.get("latest.json") is not None
                        else "confirmed_latest_absent"
                    ),
                }
            if confirm_current_wip_state_sha256 != current_wip_sha256:
                raise CampaignPlanError(
                    "operator current-WIP confirmation does not match; "
                    f"observed {current_wip_sha256}"
                )
            wip_recovery_authority = (
                "operator_confirmed_quarantined_wip_v1"
            )
            if (
                wip.entries.get("latest.json") is not None
                and wip.latest_bytes is not None
            ):
                wip_disposition = "discard_latest_pointer"
                discard_survivor_sha256: str | None = (
                    _wip_discard_survivor_sha256(wip)
                )
            elif (
                wip.entries.get("latest.json") is None
                and wip.latest_bytes is None
            ):
                wip_disposition = "confirmed_latest_absent"
                discard_survivor_sha256 = None
            else:
                raise CampaignPlanError(
                    "legacy WIP latest-pointer state is internally inconsistent"
                )
        else:
            if confirm_current_wip_state_sha256 is not None:
                raise CampaignPlanError(
                    "current-WIP confirmation was supplied but the historical "
                    "snapshot is unchanged"
                )
            if marker.capsule_state is None or marker.capsule_record is None:
                raise CampaignPlanError(
                    "dispatch v2 WIP rollback capsule is unavailable"
                )
            wip_recovery_authority = "dispatch_full_wip_rollback_capsule_v1"
            wip_disposition = "restore_historical_baseline"
            discard_survivor_sha256 = None
        restored_wip_logical_state_sha256 = (
            parsed.armed.get("wip_restore_logical_state_sha256")
            if wip_disposition == "restore_historical_baseline"
            else None
        )
        pre_arm_size = marker.recovery_sealed_size
        pre_arm_sha = marker.recovery_sealed_sha256
        if pre_arm_size is None or pre_arm_sha is None:
            raise CampaignPlanError("pre-reboot marker byte seal is unavailable")
        nonce = os.urandom(16).hex()
        installed_arm = _atomic_recovery_arm_replace(marker, {
            "event": "post_reboot_recovery_armed",
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "recovery_nonce": nonce,
            "boot_identity_source": boot_identity.source,
            "boot_identity": boot_identity.value,
            "marker_root_identity": list(marker.root_identity),
            "pre_arm_marker_identity": list(marker.marker_identity),
            "pre_arm_marker_bytes": pre_arm_size,
            "pre_arm_marker_sha256": pre_arm_sha,
            "projected_item_sha256": parsed.armed["projected_item_sha256"],
            "exec_record_sha256": _recovery_record_sha256(state.record),
            "canonical_root_metadata": _canonical_root_recovery_metadata(
                canonical
            ),
            "wip_state_sha256": current_wip_sha256,
            "recovery_arm_schema": RebootRecovery.RECOVERY_ARM_SCHEMA_V2,
            "wip_recovery_authority": wip_recovery_authority,
            "historical_wip_snapshot": parsed.armed["target_wip_snapshot"],
            "confirmed_current_wip_state_sha256": current_wip_sha256,
            "wip_disposition": wip_disposition,
            "discard_survivor_sha256": discard_survivor_sha256,
            "restored_wip_logical_state_sha256": (
                restored_wip_logical_state_sha256
            ),
            "workspace_lock_schema": lock_schema,
            "workspace_lock_path": os.fspath(lock_path),
            "workspace_lock_identity": list(lock_identity),
        })
        _close_dispatch_quarantine(marker)
        marker, parsed = _read_existing_dispatch_quarantine(
            item, require_recovery_arm=True
        )
        if parsed.recovery_arm != installed_arm:
            raise CampaignPlanError(
                "installed recovery arm differs from its sealed sidecar"
            )
        _validate_recovery_marker_seal(marker)
        state = _pre_reboot_exec_state(item, parsed)
        nonce = _validate_recovery_arm(
            item=item,
            marker=marker,
            parsed=parsed,
            boot_identity=boot_identity,
            record=state.record,
            canonical=canonical,
            wip=wip,
            capsule_baseline=marker.capsule_state,
        )
        return {
            "game": item["game"],
            "target_level": item["target_level"],
            "result": "post_reboot_recovery_armed",
            "dispatch_id": marker.dispatch_id,
            "recovery_nonce": nonce,
        }
    finally:
        _close_dispatch_quarantine(marker)


def _arm_post_reboot_recovery(
    item: dict[str, Any],
    *,
    confirm_dispatch_id: str,
    confirm_current_wip_state_sha256: str | None = None,
    boot_identity_provider: RebootRecovery.BootIdentityProvider = (
        RebootRecovery.authoritative_boot_identity
    ),
) -> dict[str, Any]:
    dispatch_lock = _acquire_scheduler_dispatch_lock(item)
    try:
        lineage_lock = _acquire_scheduler_lineage_lock(item)
        try:
            return _arm_post_reboot_recovery_locked(
                item,
                confirm_dispatch_id=confirm_dispatch_id,
                confirm_current_wip_state_sha256=(
                    confirm_current_wip_state_sha256
                ),
                boot_identity_provider=boot_identity_provider,
            )
        finally:
            _release_scheduler_artifact_lock(lineage_lock)
    finally:
        _release_scheduler_artifact_lock(dispatch_lock)


def _append_post_reboot_recovery_receipt(
    *,
    state: PostRebootLedgerState,
    item: dict[str, Any],
    marker: DispatchQuarantine,
    parsed: RebootRecovery.ParsedMarker,
    current_boot: RebootRecovery.BootIdentity,
    record: dict[str, Any],
    correction: dict[str, Any],
    cleanup: dict[str, Any],
    canonical: CanonicalRollbackState,
    wip: WipRollbackState,
) -> dict[str, Any]:
    receipt = {
        "event": "codex_post_reboot_operator_recovery_completed",
        "schema": RebootRecovery.OPERATOR_RECOVERY_SCHEMA,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "recovery_authority": "scheduler_authenticated_post_reboot_v1",
        "dispatch_id": marker.dispatch_id,
        "marker_root_identity": list(marker.root_identity),
        "pre_arm_marker_identity": parsed.recovery_arm[
            "pre_arm_marker_identity"
        ],
        "armed_marker_identity": parsed.recovery_arm[
            "armed_marker_identity"
        ],
        "marker_root": os.fspath(marker.root),
        "marker_name": marker.name,
        "dispatch_unquiesced_at": parsed.unquiesced["recorded_at"],
        "recovery_nonce": parsed.recovery_arm["recovery_nonce"],
        "armed_boot_identity": RebootRecovery.boot_identity_receipt(
            RebootRecovery.BootIdentity(
                str(parsed.recovery_arm["boot_identity_source"]),
                str(parsed.recovery_arm["boot_identity"]),
            )
        ),
        "current_boot_identity": RebootRecovery.boot_identity_receipt(
            current_boot
        ),
        "thread_id": record.get("thread_id"),
        "transcript": record["transcript"],
        "workspace": record["workspace"],
        "game": item["game"],
        "target_level": item["target_level"],
        "failure_class": "taint",
        "taint_verdict": "tainted",
        "solved_target": None,
        "retry_increment": 0,
        "artifact_root": parsed.armed["artifact_root"],
        "artifact_root_identity": parsed.armed["artifact_root_identity"],
        "canonical_root": parsed.armed["canonical_root"],
        "canonical_root_identity": parsed.armed["canonical_root_identity"],
        "canonical_digest": parsed.armed["canonical_digest"],
        "canonical_root_metadata": _canonical_root_recovery_metadata(
            canonical
        ),
        "target_wip_snapshot": parsed.armed["target_wip_snapshot"],
        "wip_root_metadata": _wip_root_recovery_metadata(wip),
        "wip_state_sha256": _wip_recovery_state_sha256(wip),
        "wip_recovery_authority": parsed.recovery_arm[
            "wip_recovery_authority"
        ],
        "confirmed_current_wip_state_sha256": parsed.recovery_arm[
            "confirmed_current_wip_state_sha256"
        ],
        "wip_disposition": parsed.recovery_arm["wip_disposition"],
        "discard_survivor_sha256": parsed.recovery_arm[
            "discard_survivor_sha256"
        ],
        "restored_wip_logical_state_sha256": parsed.recovery_arm[
            "restored_wip_logical_state_sha256"
        ],
        "wip_restore_logical_state_schema": parsed.armed.get(
            "wip_restore_logical_state_schema"
        ),
        "wip_rollback_capsule_name": parsed.armed.get(
            "wip_rollback_capsule_name"
        ),
        "wip_rollback_capsule_identity": parsed.armed.get(
            "wip_rollback_capsule_identity"
        ),
        "wip_rollback_capsule_bytes": parsed.armed.get(
            "wip_rollback_capsule_bytes"
        ),
        "wip_rollback_capsule_sha256": parsed.armed.get(
            "wip_rollback_capsule_sha256"
        ),
        "wip_rollback_capsule_state_sha256": parsed.armed.get(
            "wip_rollback_capsule_state_sha256"
        ),
        "ledger": parsed.armed["ledger"],
        "ledger_parent_identity": parsed.armed["ledger_parent_identity"],
        "ledger_file_identity": parsed.armed["ledger_file_identity"],
        "ledger_prefix_bytes": parsed.armed["ledger_prefix_bytes"],
        "ledger_prefix_sha256": parsed.armed["ledger_prefix_sha256"],
        "projected_item_sha256": parsed.armed["projected_item_sha256"],
        "workspace_identity": parsed.unquiesced["workspace_identity"],
        "protected_identity": parsed.unquiesced["protected_identity"],
        "workspace_lock_schema": parsed.recovery_arm[
            "workspace_lock_schema"
        ],
        "workspace_lock_path": parsed.recovery_arm["workspace_lock_path"],
        "workspace_lock_identity": parsed.recovery_arm[
            "workspace_lock_identity"
        ],
        "exec_record_sha256": _recovery_record_sha256(record),
        "correction_record_sha256": _recovery_record_sha256(correction),
        "cleanup_record_sha256": _recovery_record_sha256(cleanup),
        **{
            field: item[field]
            for field in (
                *Status.FRONTIER_BINDING_FIELDS,
                "reached",
                "parent_action_count",
            )
        },
    }
    return _append_recovery_phase_cas(state, receipt)


def _validate_post_reboot_operator_receipt(
    *,
    item: dict[str, Any],
    marker: DispatchQuarantine,
    parsed: RebootRecovery.ParsedMarker,
    current_boot: RebootRecovery.BootIdentity,
    record: dict[str, Any],
    correction: dict[str, Any],
    cleanup: dict[str, Any],
    receipt: dict[str, Any],
    canonical: CanonicalRollbackState,
    wip: WipRollbackState,
) -> None:
    expected = {
        "event": "codex_post_reboot_operator_recovery_completed",
        "schema": RebootRecovery.OPERATOR_RECOVERY_SCHEMA,
        "recovery_authority": "scheduler_authenticated_post_reboot_v1",
        "dispatch_id": marker.dispatch_id,
        "marker_root_identity": list(marker.root_identity),
        "pre_arm_marker_identity": parsed.recovery_arm[
            "pre_arm_marker_identity"
        ],
        "armed_marker_identity": parsed.recovery_arm[
            "armed_marker_identity"
        ],
        "marker_root": os.fspath(marker.root),
        "marker_name": marker.name,
        "dispatch_unquiesced_at": parsed.unquiesced["recorded_at"],
        "recovery_nonce": parsed.recovery_arm["recovery_nonce"],
        "armed_boot_identity": RebootRecovery.boot_identity_receipt(
            RebootRecovery.BootIdentity(
                str(parsed.recovery_arm["boot_identity_source"]),
                str(parsed.recovery_arm["boot_identity"]),
            )
        ),
        "thread_id": record.get("thread_id"),
        "transcript": record["transcript"],
        "workspace": record["workspace"],
        "game": item["game"],
        "target_level": item["target_level"],
        "failure_class": "taint",
        "taint_verdict": "tainted",
        "solved_target": None,
        "retry_increment": 0,
        "artifact_root": parsed.armed["artifact_root"],
        "artifact_root_identity": parsed.armed["artifact_root_identity"],
        "canonical_root": parsed.armed["canonical_root"],
        "canonical_root_identity": parsed.armed["canonical_root_identity"],
        "canonical_digest": parsed.armed["canonical_digest"],
        "canonical_root_metadata": _canonical_root_recovery_metadata(
            canonical
        ),
        "target_wip_snapshot": parsed.armed["target_wip_snapshot"],
        "wip_root_metadata": _wip_root_recovery_metadata(wip),
        "wip_state_sha256": _wip_recovery_state_sha256(wip),
        "wip_recovery_authority": parsed.recovery_arm[
            "wip_recovery_authority"
        ],
        "confirmed_current_wip_state_sha256": parsed.recovery_arm[
            "confirmed_current_wip_state_sha256"
        ],
        "wip_disposition": parsed.recovery_arm["wip_disposition"],
        "discard_survivor_sha256": parsed.recovery_arm[
            "discard_survivor_sha256"
        ],
        "restored_wip_logical_state_sha256": parsed.recovery_arm[
            "restored_wip_logical_state_sha256"
        ],
        "wip_restore_logical_state_schema": parsed.armed.get(
            "wip_restore_logical_state_schema"
        ),
        "wip_rollback_capsule_name": parsed.armed.get(
            "wip_rollback_capsule_name"
        ),
        "wip_rollback_capsule_identity": parsed.armed.get(
            "wip_rollback_capsule_identity"
        ),
        "wip_rollback_capsule_bytes": parsed.armed.get(
            "wip_rollback_capsule_bytes"
        ),
        "wip_rollback_capsule_sha256": parsed.armed.get(
            "wip_rollback_capsule_sha256"
        ),
        "wip_rollback_capsule_state_sha256": parsed.armed.get(
            "wip_rollback_capsule_state_sha256"
        ),
        "ledger": parsed.armed["ledger"],
        "ledger_parent_identity": parsed.armed["ledger_parent_identity"],
        "ledger_file_identity": parsed.armed["ledger_file_identity"],
        "ledger_prefix_bytes": parsed.armed["ledger_prefix_bytes"],
        "ledger_prefix_sha256": parsed.armed["ledger_prefix_sha256"],
        "projected_item_sha256": parsed.armed["projected_item_sha256"],
        "workspace_identity": parsed.unquiesced["workspace_identity"],
        "protected_identity": parsed.unquiesced["protected_identity"],
        "workspace_lock_schema": parsed.recovery_arm[
            "workspace_lock_schema"
        ],
        "workspace_lock_path": parsed.recovery_arm["workspace_lock_path"],
        "workspace_lock_identity": parsed.recovery_arm[
            "workspace_lock_identity"
        ],
        "exec_record_sha256": _recovery_record_sha256(record),
        "correction_record_sha256": _recovery_record_sha256(correction),
        "cleanup_record_sha256": _recovery_record_sha256(cleanup),
        **{
            field: item[field]
            for field in (
                *Status.FRONTIER_BINDING_FIELDS,
                "reached",
                "parent_action_count",
            )
        },
    }
    if set(receipt) != {
        *expected, "recorded_at", "current_boot_identity"
    } or any(
        receipt.get(field) != value for field, value in expected.items()
    ):
        raise CampaignPlanError(
            "post-reboot operator receipt does not bind the exact recovery"
        )
    armed_boot_receipt = expected["armed_boot_identity"]
    recovery_boot_receipt = receipt.get("current_boot_identity")
    current_boot_receipt = RebootRecovery.boot_identity_receipt(current_boot)
    if (
        not isinstance(recovery_boot_receipt, dict)
        or set(recovery_boot_receipt) != {"source", "identity_sha256"}
        or recovery_boot_receipt.get("source")
        != armed_boot_receipt.get("source")
        or not isinstance(
            recovery_boot_receipt.get("identity_sha256"), str
        )
        or SHA256_RE.fullmatch(
            recovery_boot_receipt["identity_sha256"]
        ) is None
        or recovery_boot_receipt == armed_boot_receipt
        or current_boot_receipt == armed_boot_receipt
    ):
        raise CampaignPlanError(
            "post-reboot operator receipt lacks a changed boot identity"
        )
    _recovery_recorded_at(receipt, "post-reboot operator receipt")


def _post_reboot_generation_paths(
    item: dict[str, Any], record: dict[str, Any]
) -> tuple[Path, Path]:
    workspace_name = _safe_component(record.get("workspace"), "workspace")
    scratch = Path(Legs.SCRATCH).absolute()
    authority = item.get("historical_runner")
    if isinstance(authority, dict) and scratch != _normalized_absolute_path(
        authority.get("scratch_root"), "scratch_root"
    ):
        raise CampaignPlanError(
            "post-reboot scratch root differs from the runner receipt"
        )
    _host_directory_identity(scratch, "scratch root")
    protected_root = scratch / ".proposer_transcripts"
    _host_directory_identity(protected_root, "protected transcript root")
    workspace = scratch / workspace_name
    protected = protected_root / workspace_name
    if workspace.parent != scratch or protected.parent != protected_root:
        raise CampaignPlanError("post-reboot generation path escaped scratch")
    return workspace, protected


def _validate_post_reboot_generation_identities(
    parsed: RebootRecovery.ParsedMarker,
    workspace: Path,
    protected: Path,
    *,
    require_both: bool,
) -> tuple[bool, bool]:
    workspace_exists = os.path.lexists(workspace)
    protected_exists = os.path.lexists(protected)
    if require_both and not (workspace_exists and protected_exists):
        raise CampaignPlanError(
            "post-reboot generation evidence disappeared before classification"
        )
    if workspace_exists and (
        _host_directory_identity(workspace, "quarantined workspace")
        != _marker_identity(
            parsed.unquiesced.get("workspace_identity"), "workspace"
        )
    ):
        raise CampaignPlanError("quarantined workspace identity changed")
    if protected_exists and (
        _host_directory_identity(protected, "quarantined protected evidence")
        != _marker_identity(
            parsed.unquiesced.get("protected_identity"), "protected"
        )
    ):
        raise CampaignPlanError("quarantined protected identity changed")
    return workspace_exists, protected_exists


def _post_reboot_tombstones(
    dispatch_id: str,
    workspace: Path,
    protected: Path,
) -> tuple[Path, Path]:
    if RebootRecovery.DISPATCH_ID_RE.fullmatch(dispatch_id) is None:
        raise CampaignPlanError("recovery tombstone dispatch ID is malformed")
    suffix = f"post_reboot_cleanup_{dispatch_id}_{workspace.name}"
    name = f".{suffix}"
    for parent in (workspace.parent, protected.parent):
        try:
            name_max = os.pathconf(parent, "PC_NAME_MAX")
        except OSError as exc:
            raise CampaignPlanError(
                "could not validate the recovery tombstone name"
            ) from exc
        if len(os.fsencode(name)) > name_max:
            raise CampaignPlanError("recovery tombstone name is too long")
    return workspace.parent / name, protected.parent / name


def _rename_recovery_tombstone(
    source: Path, target: Path, expected_identity: tuple[int, int]
) -> None:
    if source.parent != target.parent:
        raise CampaignPlanError("recovery tombstone escaped its source parent")
    parent_identity = _host_directory_identity(
        source.parent, "recovery tombstone parent"
    )
    descriptor = os.open(
        source.parent,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        metadata = os.stat(source.name, dir_fd=descriptor, follow_symlinks=False)
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or (metadata.st_dev, metadata.st_ino) != expected_identity
        ):
            raise CampaignPlanError("recovery source identity changed")
        try:
            os.stat(target.name, dir_fd=descriptor, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise CampaignPlanError("recovery tombstone already exists")
        os.rename(
            source.name,
            target.name,
            src_dir_fd=descriptor,
            dst_dir_fd=descriptor,
        )
        os.fsync(descriptor)
        moved = os.stat(target.name, dir_fd=descriptor, follow_symlinks=False)
        if (
            (moved.st_dev, moved.st_ino) != expected_identity
            or _host_directory_identity(
                source.parent, "recovery tombstone parent"
            ) != parent_identity
        ):
            raise CampaignPlanError("recovery tombstone identity changed")
    except OSError as exc:
        raise CampaignPlanError("recovery tombstone rename failed") from exc
    finally:
        os.close(descriptor)


VALID_RECOVERY_TOMBSTONE_INVENTORIES = frozenset({
    frozenset({"W", "P"}),
    frozenset({"Wt", "P"}),
    frozenset({"Wt", "Pt"}),
    frozenset({"Pt"}),
    frozenset(),
})


def _validate_recovery_tombstone_inventory(
    workspace_exists: bool,
    protected_exists: bool,
    workspace_tombstone_exists: bool,
    protected_tombstone_exists: bool,
) -> frozenset[str]:
    inventory = frozenset(
        label
        for label, present in (
            ("W", workspace_exists),
            ("P", protected_exists),
            ("Wt", workspace_tombstone_exists),
            ("Pt", protected_tombstone_exists),
        )
        if present
    )
    if inventory not in VALID_RECOVERY_TOMBSTONE_INVENTORIES:
        raise CampaignPlanError(
            "post-reboot cleanup inventory is not a reachable tombstone phase"
        )
    return inventory


def _open_post_reboot_cleanup_lock(
    item: dict[str, Any],
    parsed: RebootRecovery.ParsedMarker,
    workspace: Path,
    workspace_tombstone: Path,
    protected: Path,
    protected_tombstone: Path,
) -> tuple[Any | None, Path, tuple[int, int]]:
    arm = parsed.recovery_arm
    assert arm is not None
    schema = arm.get("workspace_lock_schema")
    recorded_path = Path(str(arm.get("workspace_lock_path")))
    recorded_identity = _marker_identity(
        arm.get("workspace_lock_identity"), "workspace lock"
    )
    expected_path = (
        workspace / ".orchestrate.lock"
        if schema == "in_workspace_v1"
        else Path(Legs._workspace_lock_path(os.fspath(workspace)))
    )
    if schema != _lock_schema(item) or recorded_path != expected_path:
        raise CampaignPlanError("armed workspace lock binding changed")
    candidates = [recorded_path]
    if schema == "in_workspace_v1":
        candidates.append(workspace_tombstone / ".orchestrate.lock")
    existing = [path for path in candidates if os.path.lexists(path)]
    if len(existing) > 1:
        raise CampaignPlanError("workspace lock exists at ambiguous paths")
    if not existing:
        if schema == "hashed_external_v1" and any(
            os.path.lexists(path) for path in (
                workspace, workspace_tombstone,
                protected, protected_tombstone,
            )
        ):
            raise CampaignPlanError(
                "external workspace lock disappeared before cleanup"
            )
        if os.path.lexists(workspace):
            raise CampaignPlanError(
                "original workspace remains but its armed lock disappeared"
            )
        return None, recorded_path, recorded_identity
    lock_path = existing[0]
    lock: Any | None = None
    try:
        lock = Legs._open_unaliased_lock(os.fspath(lock_path), create=False)
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (OSError, RuntimeError, BlockingIOError) as exc:
        try:
            if lock is not None:
                lock.close()
        except OSError:
            pass
        raise CampaignPlanError("armed workspace lock remains active") from exc
    metadata = os.fstat(lock.fileno())
    if (metadata.st_dev, metadata.st_ino) != recorded_identity:
        lock.close()
        raise CampaignPlanError("armed workspace lock identity changed")
    return lock, recorded_path, recorded_identity


def _resume_post_reboot_generation_cleanup(
    *,
    item: dict[str, Any],
    parsed: RebootRecovery.ParsedMarker,
    state: PostRebootLedgerState,
    workspace: Path,
    protected: Path,
) -> None:
    assert state.correction is not None
    workspace_tombstone, protected_tombstone = _post_reboot_tombstones(
        parsed.dispatch_id, workspace, protected
    )
    expected_workspace = _marker_identity(
        parsed.unquiesced.get("workspace_identity"), "workspace"
    )
    expected_protected = _marker_identity(
        parsed.unquiesced.get("protected_identity"), "protected"
    )
    workspace_exists, protected_exists = _validate_post_reboot_generation_identities(
        parsed, workspace, protected, require_both=False
    )
    workspace_tombstone_exists = os.path.lexists(workspace_tombstone)
    protected_tombstone_exists = os.path.lexists(protected_tombstone)
    _validate_recovery_tombstone_inventory(
        workspace_exists,
        protected_exists,
        workspace_tombstone_exists,
        protected_tombstone_exists,
    )
    if workspace_tombstone_exists and _host_directory_identity(
        workspace_tombstone, "workspace cleanup tombstone"
    ) != expected_workspace:
        raise CampaignPlanError("workspace cleanup tombstone identity changed")
    if protected_tombstone_exists and _host_directory_identity(
        protected_tombstone, "protected cleanup tombstone"
    ) != expected_protected:
        raise CampaignPlanError("protected cleanup tombstone identity changed")
    if workspace_exists and protected_exists:
        (
            observed_workspace,
            observed_protected,
            _reason,
            transcript_sha,
            diagnostics_sha,
            _descendant_unproven,
        ) = _exact_tainted_generation(
            item, state.record, require_taint=False
        )
        if (
            observed_workspace != workspace
            or observed_protected != protected
            or transcript_sha
            != state.correction.get("protected_transcript_sha256")
            or (
                diagnostics_sha is not None
                and diagnostics_sha
                != state.correction.get("protected_diagnostics_sha256")
            )
        ):
            raise CampaignPlanError(
                "resumed cleanup evidence differs from its correction"
            )
        if _workspace_lock_is_active(workspace):
            raise CampaignPlanError(
                "quarantined exact workspace remains active after reboot"
            )
    cleanup_lock, recorded_lock_path, _lock_identity = (
        _open_post_reboot_cleanup_lock(
            item,
            parsed,
            workspace,
            workspace_tombstone,
            protected,
            protected_tombstone,
        )
    )
    try:
        if workspace_exists:
            _rename_recovery_tombstone(
                workspace, workspace_tombstone, expected_workspace
            )
            workspace_tombstone_exists = True
        if protected_exists:
            _rename_recovery_tombstone(
                protected, protected_tombstone, expected_protected
            )
            protected_tombstone_exists = True
        if not getattr(shutil.rmtree, "avoids_symlink_attacks", False):
            raise CampaignPlanError(
                "platform lacks symlink-safe recursive deletion"
            )
        for tombstone in (workspace_tombstone, protected_tombstone):
            if os.path.lexists(tombstone):
                expected_identity = (
                    expected_workspace
                    if tombstone == workspace_tombstone
                    else expected_protected
                )
                if _host_directory_identity(
                    tombstone, "recovery cleanup tombstone"
                ) != expected_identity:
                    raise CampaignPlanError(
                        "recovery cleanup tombstone identity changed"
                    )
                shutil.rmtree(tombstone)
                _fsync_directory(tombstone.parent)
        if _lock_schema(item) == "hashed_external_v1":
            if os.path.lexists(recorded_lock_path):
                metadata = recorded_lock_path.stat(follow_symlinks=False)
                if (
                    recorded_lock_path.is_symlink()
                    or not stat.S_ISREG(metadata.st_mode)
                    or (metadata.st_dev, metadata.st_ino) != _lock_identity
                ):
                    raise CampaignPlanError("external workspace lock changed")
                os.unlink(recorded_lock_path)
            # This fsync is required even when the name is already absent: the
            # prior attempt may have crashed after unlink and before sealing
            # the lock-directory entry update.
            _fsync_directory(recorded_lock_path.parent)
    except OSError as exc:
        raise CampaignPlanError("post-reboot tombstone cleanup failed") from exc
    finally:
        if cleanup_lock is not None:
            cleanup_lock.close()
    if any(os.path.lexists(path) for path in (
        workspace, protected, workspace_tombstone, protected_tombstone
    )):
        raise CampaignPlanError("post-reboot tombstone cleanup was incomplete")
    if _lock_schema(item) == "hashed_external_v1" and os.path.lexists(
        recorded_lock_path
    ):
        raise CampaignPlanError("external workspace lock cleanup was incomplete")


def _open_zero_ledger_cleanup_lock(
    item: dict[str, Any],
    parsed: RebootRecovery.ParsedMarker,
    workspace: Path,
    workspace_tombstone: Path,
    protected: Path,
    protected_tombstone: Path,
) -> tuple[Any | None, Path, tuple[int, int]]:
    observed = parsed.unquiesced
    schema = observed.get("workspace_lock_schema")
    recorded_path = Path(str(observed.get("workspace_lock_path")))
    recorded_identity = _marker_identity(
        observed.get("workspace_lock_identity"), "workspace lock"
    )
    expected_path = (
        workspace / ".orchestrate.lock"
        if schema == "in_workspace_v1"
        else Path(Legs._workspace_lock_path(os.fspath(workspace)))
    )
    if schema != _lock_schema(item) or recorded_path != expected_path:
        raise CampaignPlanError("zero-ledger workspace lock binding changed")
    candidates = [recorded_path]
    if schema == "in_workspace_v1":
        candidates.append(workspace_tombstone / ".orchestrate.lock")
    existing = [path for path in candidates if os.path.lexists(path)]
    if len(existing) > 1:
        raise CampaignPlanError("zero-ledger workspace lock is ambiguous")
    if not existing:
        if schema == "hashed_external_v1" and any(
            os.path.lexists(path) for path in (
                workspace, workspace_tombstone,
                protected, protected_tombstone,
            )
        ):
            raise CampaignPlanError(
                "zero-ledger external workspace lock disappeared"
            )
        if os.path.lexists(workspace):
            raise CampaignPlanError(
                "zero-ledger workspace remains without its bound lock"
            )
        return None, recorded_path, recorded_identity
    lock_path = existing[0]
    lock: Any | None = None
    try:
        lock = Legs._open_unaliased_lock(os.fspath(lock_path), create=False)
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (OSError, RuntimeError, BlockingIOError) as exc:
        if lock is not None:
            lock.close()
        raise CampaignPlanError(
            "zero-ledger workspace lock remains active"
        ) from exc
    metadata = os.fstat(lock.fileno())
    if (metadata.st_dev, metadata.st_ino) != recorded_identity:
        lock.close()
        raise CampaignPlanError("zero-ledger workspace lock identity changed")
    return lock, recorded_path, recorded_identity


def _resume_zero_ledger_generation_cleanup(
    *,
    item: dict[str, Any],
    parsed: RebootRecovery.ParsedMarker,
    workspace: Path,
    protected: Path,
) -> None:
    """Idempotently retire only the two marker-bound generation directories."""

    workspace_tombstone, protected_tombstone = _post_reboot_tombstones(
        parsed.dispatch_id, workspace, protected
    )
    expected_workspace = _marker_identity(
        parsed.unquiesced.get("workspace_identity"), "workspace"
    )
    expected_protected = _marker_identity(
        parsed.unquiesced.get("protected_identity"), "protected"
    )
    workspace_exists, protected_exists = _validate_post_reboot_generation_identities(
        parsed, workspace, protected, require_both=False
    )
    workspace_tombstone_exists = os.path.lexists(workspace_tombstone)
    protected_tombstone_exists = os.path.lexists(protected_tombstone)
    _validate_recovery_tombstone_inventory(
        workspace_exists,
        protected_exists,
        workspace_tombstone_exists,
        protected_tombstone_exists,
    )
    if workspace_tombstone_exists and _host_directory_identity(
        workspace_tombstone, "zero-ledger workspace tombstone"
    ) != expected_workspace:
        raise CampaignPlanError(
            "zero-ledger workspace tombstone identity changed"
        )
    if protected_tombstone_exists and _host_directory_identity(
        protected_tombstone, "zero-ledger protected tombstone"
    ) != expected_protected:
        raise CampaignPlanError(
            "zero-ledger protected tombstone identity changed"
        )
    cleanup_lock, lock_path, lock_identity = _open_zero_ledger_cleanup_lock(
        item,
        parsed,
        workspace,
        workspace_tombstone,
        protected,
        protected_tombstone,
    )
    try:
        if workspace_exists:
            _rename_recovery_tombstone(
                workspace, workspace_tombstone, expected_workspace
            )
        if protected_exists:
            _rename_recovery_tombstone(
                protected, protected_tombstone, expected_protected
            )
        if not getattr(shutil.rmtree, "avoids_symlink_attacks", False):
            raise CampaignPlanError(
                "platform lacks symlink-safe recursive deletion"
            )
        for tombstone, identity in (
            (workspace_tombstone, expected_workspace),
            (protected_tombstone, expected_protected),
        ):
            if os.path.lexists(tombstone):
                if _host_directory_identity(
                    tombstone, "zero-ledger cleanup tombstone"
                ) != identity:
                    raise CampaignPlanError(
                        "zero-ledger cleanup tombstone identity changed"
                    )
                shutil.rmtree(tombstone)
            _fsync_directory(tombstone.parent)
        if _lock_schema(item) == "hashed_external_v1":
            if os.path.lexists(lock_path):
                metadata = lock_path.stat(follow_symlinks=False)
                if (
                    lock_path.is_symlink()
                    or not stat.S_ISREG(metadata.st_mode)
                    or (metadata.st_dev, metadata.st_ino) != lock_identity
                ):
                    raise CampaignPlanError(
                        "zero-ledger external lock identity changed"
                    )
                os.unlink(lock_path)
            _fsync_directory(lock_path.parent)
    except OSError as exc:
        raise CampaignPlanError(
            "zero-ledger tombstone cleanup failed"
        ) from exc
    finally:
        if cleanup_lock is not None:
            cleanup_lock.close()
    if any(os.path.lexists(path) for path in (
        workspace, protected, workspace_tombstone, protected_tombstone,
    )):
        raise CampaignPlanError(
            "zero-ledger tombstone cleanup was incomplete"
        )
    if _lock_schema(item) == "hashed_external_v1" and os.path.lexists(
        lock_path
    ):
        raise CampaignPlanError(
            "zero-ledger external lock cleanup was incomplete"
        )


def _zero_ledger_generation_paths(
    item: dict[str, Any], parsed: RebootRecovery.ParsedMarker
) -> tuple[Path, Path]:
    workspace, protected = _post_reboot_generation_paths(
        item, parsed.unquiesced
    )
    if parsed.unquiesced.get("protected") != workspace.name:
        raise CampaignPlanError(
            "zero-ledger protected generation name changed"
        )
    return workspace, protected


def _authenticate_zero_ledger_marker_evidence(
    item: dict[str, Any],
    parsed: RebootRecovery.ParsedMarker,
    workspace: Path,
    protected: Path,
) -> None:
    observed = parsed.unquiesced
    _validate_post_reboot_generation_identities(
        parsed, workspace, protected, require_both=True
    )
    evidence: dict[str, Any] = {
        "workspace": observed["workspace"],
        "transcript": observed["transcript"],
        "protected_transcript_status": "sealed",
        "protected_transcript_sha256": observed[
            "protected_transcript_sha256"
        ],
    }
    if observed.get("diagnostics") is not None:
        evidence.update({
            "diagnostics": observed["diagnostics"],
            "protected_diagnostics_status": "sealed",
            "protected_diagnostics_sha256": observed[
                "protected_diagnostics_sha256"
            ],
        })
    (
        scanned_workspace,
        scanned_protected,
        _reason,
        transcript_sha,
        diagnostics_sha,
        descendant_unproven,
    ) = _exact_tainted_generation(item, evidence, require_taint=False)
    if any((
        scanned_workspace != workspace,
        scanned_protected != protected,
        transcript_sha != observed.get("protected_transcript_sha256"),
        diagnostics_sha
        != observed.get("protected_diagnostics_sha256"),
        descendant_unproven
        and not observed.get("detached_processes_proven_absent"),
    )):
        raise CampaignPlanError(
            "zero-ledger protected evidence binding changed"
        )
    schema, lock_path, lock_identity = _capture_recovery_workspace_lock(
        item, workspace
    )
    if any((
        schema != observed.get("workspace_lock_schema"),
        os.fspath(lock_path) != observed.get("workspace_lock_path"),
        list(lock_identity) != observed.get("workspace_lock_identity"),
    )):
        raise CampaignPlanError("zero-ledger workspace lock binding changed")


def _zero_ledger_result(
    item: dict[str, Any],
    parsed: RebootRecovery.ParsedMarker,
    *,
    replayed: bool,
) -> dict[str, Any]:
    return {
        "game": item["game"],
        "target_level": item["target_level"],
        "reached": item["reached"],
        "result": "infrastructure_noncounting",
        "reason": parsed.unquiesced["reason"],
        "child_returncode": parsed.unquiesced["child_returncode"],
        "retry_complexity_n": item["retry_complexity_n"],
        "seed_mode": item["seed_mode"],
        "wip_mode": item["wip_mode"],
        "lineage_input_mode": item["lineage_input_mode"],
        "zero_ledger_replayed": replayed,
    }


def _validate_zero_ledger_result(
    item: dict[str, Any],
    parsed: RebootRecovery.ParsedMarker,
    event: dict[str, Any],
    result: dict[str, Any],
) -> None:
    """Cross-bind the public terminal result to its marker and ledger row."""

    if (
        set(result) != ZERO_LEDGER_RESULT_KEYS
        or not isinstance(result.get("zero_ledger_replayed"), bool)
    ):
        raise CampaignPlanError(
            "zero-ledger terminal result has an invalid exact schema"
        )
    expected = _zero_ledger_result(
        item,
        parsed,
        replayed=bool(result["zero_ledger_replayed"]),
    )
    if result != expected or any((
        event.get("child_returncode") != result["child_returncode"],
        event.get("retry_complexity_n") != result["retry_complexity_n"],
        event.get("terminal_errors") != [result["reason"]],
    )):
        raise CampaignPlanError(
            "zero-ledger terminal result binding changed"
        )


def _validate_zero_ledger_baseline(
    item: dict[str, Any],
    parsed: RebootRecovery.ParsedMarker,
    *,
    reached_before: int,
) -> None:
    canonical = _capture_canonical_rollback(item)
    if any((
        list(canonical.root_identity)
        != parsed.armed.get("canonical_root_identity"),
        canonical.digest != parsed.armed.get("canonical_digest"),
        _checkpoint_reached(item["game"]) != reached_before,
        _canonical_frontier_binding(item)
        != Status.validate_frontier_binding(
            dict(parsed.armed["frontier_binding"])
        ),
    )):
        raise CampaignPlanError(
            "zero-ledger recovery has not restored the sealed baseline"
        )


def _complete_zero_ledger_recovery(
    item: dict[str, Any],
    *,
    marker: DispatchQuarantine,
    parsed: RebootRecovery.ParsedMarker,
    ledger_before: LedgerPrefixState,
    reached_before: int,
    wip_before: WipRollbackState | None,
    canonical_before: CanonicalRollbackState | None,
    replayed: bool,
    release_marker: bool,
) -> dict[str, Any]:
    """Restore, journal, and retire one exact zero-``codex_exec`` generation."""

    _validate_post_reboot_dispatch_binding(item, parsed)
    if (
        parsed.armed.get("schema")
        != RebootRecovery.DISPATCH_QUARANTINE_SCHEMA_V2
        or marker.capsule_state is None
        or marker.capsule_record is None
    ):
        raise CampaignPlanError(
            "zero-ledger recovery requires the full v2 WIP capsule"
        )
    workspace, protected = _zero_ledger_generation_paths(item, parsed)
    ledger, baseline, suffix = _read_post_reboot_ledger_surface(
        item,
        parsed.armed,
        intent_root=marker.root,
        intent_root_identity=marker.root_identity,
    )
    if (
        baseline.raw_prefix != ledger_before.raw_prefix
        or baseline.file_identity != ledger_before.file_identity
        or baseline.parent_identity != ledger_before.parent_identity
    ):
        raise CampaignPlanError("zero-ledger baseline custody changed")
    if not suffix:
        _authenticate_zero_ledger_marker_evidence(
            item, parsed, workspace, protected
        )
        if replayed:
            # Canonical bytes have no durable payload capsule in marker v2.
            # Automatic replay is therefore authorized only when they already
            # match the sealed pre-dispatch digest; otherwise quarantine stays.
            _validate_zero_ledger_baseline(
                item, parsed, reached_before=reached_before
            )
            restored = _restore_wip_from_rollback_capsule(
                item, marker.capsule_state, marker.capsule_record
            )
            _validate_capsule_restored_wip_state(
                restored,
                marker.capsule_state,
                parsed.armed.get("wip_restore_logical_state_sha256"),
                logical_schema=str(parsed.armed.get(
                    "wip_restore_logical_state_schema"
                )),
            )
        else:
            if wip_before is None or canonical_before is None:
                raise CampaignPlanError(
                    "live zero-ledger recovery lost its rollback state"
                )
            evidence = {
                "transcript": parsed.unquiesced["transcript"],
            }
            if _target_wip_snapshot(item) != wip_before.baseline_snapshot:
                _rollback_tainted_wip(
                    item,
                    wip_before,
                    evidence,
                    str(parsed.unquiesced[
                        "protected_transcript_sha256"
                    ]),
                )
            if not _canonical_matches(canonical_before):
                _rollback_tainted_canonical(canonical_before)
            _validate_zero_ledger_baseline(
                item, parsed, reached_before=reached_before
            )
        event = _build_zero_ledger_event(item, parsed)
        _validate_zero_ledger_event(item, parsed, event)
        committed = _append_zero_ledger_event_cas(
            marker=marker, baseline=baseline, record=event
        )
        _validate_zero_ledger_event(item, parsed, committed)
    elif len(suffix) == 1 and suffix[0].get("event") == ZERO_LEDGER_EVENT:
        _validate_zero_ledger_event(item, parsed, suffix[0])
        _validate_zero_ledger_baseline(
            item, parsed, reached_before=reached_before
        )
        restored = _capture_wip_rollback(item)
        _validate_capsule_restored_wip_state(
            restored,
            marker.capsule_state,
            parsed.armed.get("wip_restore_logical_state_sha256"),
            logical_schema=str(parsed.armed.get(
                "wip_restore_logical_state_schema"
            )),
        )
    else:
        raise CampaignPlanError(
            "zero-ledger quarantine has an ambiguous ledger suffix"
        )
    _resume_zero_ledger_generation_cleanup(
        item=item,
        parsed=parsed,
        workspace=workspace,
        protected=protected,
    )
    restored_wip = _capture_wip_rollback(item)
    _validate_capsule_restored_wip_state(
        restored_wip,
        marker.capsule_state,
        parsed.armed.get("wip_restore_logical_state_sha256"),
        logical_schema=str(parsed.armed.get(
            "wip_restore_logical_state_schema"
        )),
    )
    _validate_zero_ledger_baseline(
        item, parsed, reached_before=reached_before
    )
    result = _zero_ledger_result(item, parsed, replayed=replayed)
    if release_marker:
        release_authority = _build_dispatch_release_authority(
            item,
            marker,
            ledger,
            result,
            kind="ordinary_safe_terminal_v1",
        )
        _release_dispatch_quarantine(marker, item, release_authority)
    return result


def _resume_existing_zero_ledger_quarantine(
    item: dict[str, Any],
) -> dict[str, Any] | None:
    """Auto-resume only the exact durable quiesced zero-ledger marker phase."""

    opened = _open_dispatch_quarantine_root(item, create=False)
    if opened is not None:
        _root, root_fd, _root_identity = opened
        try:
            if _reconcile_dispatch_release_intent_at(
                item,
                root_fd,
                marker_name=_dispatch_quarantine_name(item),
            ):
                return None
        finally:
            os.close(root_fd)
    try:
        marker, parsed = _read_existing_dispatch_quarantine(
            item, require_recovery_arm=False
        )
    except (NoDispatchQuarantine, CampaignPlanError, OSError):
        return None
    if parsed.unquiesced.get("event") != (
        "dispatch_zero_ledger_quarantined"
    ):
        _close_dispatch_quarantine(marker)
        return None
    released = False
    try:
        _ledger, baseline, _suffix = _read_post_reboot_ledger_surface(
            item,
            parsed.armed,
            intent_root=marker.root,
            intent_root_identity=marker.root_identity,
        )
        result = _complete_zero_ledger_recovery(
            item,
            marker=marker,
            parsed=parsed,
            ledger_before=baseline,
            reached_before=int(parsed.armed["frontier_binding"]["reached"]),
            wip_before=None,
            canonical_before=None,
            replayed=True,
            release_marker=True,
        )
        released = True
        return result
    finally:
        if not released:
            _close_dispatch_quarantine(marker)


def _validate_post_reboot_clean_state(
    *,
    item: dict[str, Any],
    artifact_root: Path,
    parsed: RebootRecovery.ParsedMarker,
    canonical: CanonicalRollbackState,
    wip: WipRollbackState,
    reached: int,
    workspace: Path,
    protected: Path,
    capsule_baseline: WipRollbackState | None = None,
) -> None:
    _taint_gate()
    workspace_tombstone, protected_tombstone = _post_reboot_tombstones(
        parsed.dispatch_id, workspace, protected
    )
    arm = parsed.recovery_arm
    assert arm is not None
    external_lock_remains = (
        arm.get("workspace_lock_schema") == "hashed_external_v1"
        and os.path.lexists(Path(str(arm.get("workspace_lock_path"))))
    )
    expected_frontier = Status.validate_frontier_binding({
        field: item[field]
        for field in (
            *Status.FRONTIER_BINDING_FIELDS,
            "game",
            "reached",
            "target_level",
            "parent_action_count",
        )
    })
    wip_preserved = _target_wip_snapshot(item) == wip.baseline_snapshot
    if arm.get("wip_disposition") == "discard_latest_pointer":
        try:
            _validate_discarded_wip_state(item, arm, wip)
        except CampaignPlanError:
            wip_preserved = False
        else:
            wip_preserved = True
    if (
        arm.get("wip_disposition") == "restore_historical_baseline"
    ):
        logical_target = parsed.armed.get(
            "wip_restore_logical_state_sha256"
        )
        if capsule_baseline is None:
            wip_preserved = (
                isinstance(logical_target, str)
                and SHA256_RE.fullmatch(logical_target) is not None
                and _wip_logical_restore_state_sha256(
                    wip,
                    schema=str(parsed.armed.get(
                        "wip_restore_logical_state_schema"
                    )),
                )
                == logical_target
            )
        else:
            try:
                _validate_capsule_restored_wip_state(
                    wip,
                    capsule_baseline,
                    logical_target,
                    logical_schema=str(parsed.armed.get(
                        "wip_restore_logical_state_schema"
                    )),
                )
            except CampaignPlanError:
                wip_preserved = False
            else:
                wip_preserved = True
    if any((
        _host_directory_identity(
            artifact_root, "canonical artifact root"
        ) != _marker_identity(
            parsed.armed.get("artifact_root_identity"), "artifact root"
        ),
        not _canonical_matches(canonical),
        not wip_preserved,
        _checkpoint_reached(item["game"]) != reached,
        _canonical_frontier_binding(item) != expected_frontier,
        os.path.lexists(workspace),
        os.path.lexists(protected),
        os.path.lexists(workspace_tombstone),
        os.path.lexists(protected_tombstone),
        external_lock_remains,
    )):
        raise CampaignPlanError(
            "post-reboot recovery does not preserve the authenticated baseline"
        )


def _finish_recovery_after_capsule_retirement(
    *,
    item: dict[str, Any],
    marker: DispatchQuarantine,
    parsed: RebootRecovery.ParsedMarker,
    current_boot: RebootRecovery.BootIdentity,
    confirm_recovery_nonce: str,
) -> dict[str, Any]:
    """Finish only the capsule-unlink/marker-unlink crash boundary.

    A missing v2 capsule is acceptable only after the exact operator receipt
    has committed the restored WIP state.  Until then, the missing capsule is
    a hard failure and this path performs no mutation.
    """

    if (
        not marker.capsule_missing
        or marker.schema != RebootRecovery.DISPATCH_QUARANTINE_SCHEMA_V2
        or marker.capsule_name is None
        or marker.capsule_identity is None
    ):
        raise CampaignPlanError(
            "retired-capsule recovery lacks an exact v2 marker binding"
        )
    state = _rebind_post_reboot_ledger(
        item, parsed, marker=marker, current_boot=current_boot
    )
    if (
        state.operator is None
        or state.correction is None
        or state.cleanup is None
    ):
        raise CampaignPlanError(
            "WIP rollback capsule disappeared before operator completion"
        )
    artifact_root = _artifact_root(item)
    if _host_directory_identity(
        artifact_root, "canonical artifact root"
    ) != _marker_identity(
        parsed.armed.get("artifact_root_identity"), "artifact root"
    ):
        raise CampaignPlanError(
            "dispatch quarantine artifact root identity changed"
        )
    canonical = _capture_canonical_rollback(item)
    arm = parsed.recovery_arm
    assert arm is not None
    if any((
        os.fspath(canonical.root) != parsed.armed.get("canonical_root"),
        canonical.root_identity != _marker_identity(
            parsed.armed.get("canonical_root_identity"), "canonical root"
        ),
        canonical.digest != parsed.armed.get("canonical_digest"),
        _canonical_root_recovery_metadata(canonical)
        != arm.get("canonical_root_metadata"),
    )):
        raise CampaignPlanError(
            "completed capsule recovery canonical baseline changed"
        )
    wip = _capture_wip_rollback(item)
    workspace, protected = _post_reboot_generation_paths(item, state.record)
    reached = _checkpoint_reached(item["game"])
    _validate_post_reboot_operator_receipt(
        item=item,
        marker=marker,
        parsed=parsed,
        current_boot=current_boot,
        record=state.record,
        correction=state.correction,
        cleanup=state.cleanup,
        receipt=state.operator,
        canonical=canonical,
        wip=wip,
    )
    _validate_post_reboot_clean_state(
        item=item,
        artifact_root=artifact_root,
        parsed=parsed,
        canonical=canonical,
        wip=wip,
        reached=reached,
        workspace=workspace,
        protected=protected,
    )
    _assert_same_retry_coordinate(state.ledger, item, state.record)
    try:
        os.stat(
            marker.capsule_name,
            dir_fd=marker.root_fd,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        pass
    else:
        raise CampaignPlanError(
            "retired WIP rollback capsule name reappeared"
        )
    _validate_recovery_marker_seal(marker)
    outcome = {
        "game": item["game"],
        "target_level": item["target_level"],
        "reached": reached,
        "result": "tainted_noncounting",
        "reason": state.correction["terminal_errors"][0],
        "child_returncode": int(parsed.unquiesced["child_returncode"]),
        "retry_complexity_n": item["retry_complexity_n"],
        "seed_mode": item["seed_mode"],
        "wip_mode": item["wip_mode"],
        "lineage_input_mode": item["lineage_input_mode"],
        "operator_recovery": "post_reboot_already_completed",
        "dispatch_id": marker.dispatch_id,
        "recovery_nonce": confirm_recovery_nonce,
        "recovery_receipt_event": state.operator["event"],
    }
    _retire_incomplete_release_for_operator(item, marker)
    release_authority = _build_dispatch_release_authority(
        item,
        marker,
        state.ledger,
        outcome,
        kind="post_reboot_operator_terminal_v1",
    )
    _release_dispatch_quarantine(
        marker, item, release_authority
    )
    return outcome


def _recover_completed_safe_release_without_marker_locked(
    item: dict[str, Any],
    *,
    confirm_dispatch_id: str,
    confirm_recovery_nonce: str,
    boot_identity_provider: RebootRecovery.BootIdentityProvider,
    allow_arm_residue: bool = False,
) -> dict[str, Any] | None:
    marker_name = _dispatch_quarantine_name(item)
    receipt_name = _safe_release_recovery_receipt_name(
        marker_name, confirm_dispatch_id
    )
    opened = _open_dispatch_quarantine_root(item, create=False)
    if opened is None:
        return None
    root, root_fd, root_identity = opened
    try:
        try:
            receipt, _identity, _payload = _read_durable_recovery_record_at(
                root_fd,
                receipt_name,
                root_path=root,
                root_identity=root_identity,
                label="safe-release recovery receipt",
            )
        except FileNotFoundError:
            return None
        _validate_safe_release_recovery_receipt_record(
            receipt, marker_name=marker_name
        )
        arm = receipt["arm_record"]
        assert isinstance(arm, dict)
        if any((
            receipt.get("dispatch_id") != confirm_dispatch_id,
            receipt.get("recovery_nonce") != confirm_recovery_nonce,
            arm.get("marker_root_identity") != list(root_identity),
            arm.get("marker_root")
            != os.fspath(_artifact_root(item) / ".campaign_quarantine"),
            arm.get("projected_item_sha256")
            != hashlib.sha256(json.dumps(
                item, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode("utf-8")).hexdigest(),
        )):
            raise CampaignPlanError(
                "safe-release completion belongs to another operator request"
            )
        try:
            RebootRecovery.require_changed_boot_identity(
                arm.get("boot_identity_source"),
                arm.get("boot_identity"),
                boot_identity_provider(),
            )
        except RebootRecovery.RecoveryEvidenceError as exc:
            raise CampaignPlanError(str(exc)) from exc
        intent = receipt["release_intent"]
        assert isinstance(intent, dict)
        authority = intent["release_authority"]
        assert isinstance(authority, dict)
        ledger = _dispatch_release_item_ledger(item, authority)
        with Guard.ledger_append_lock(ledger):
            raw, _parent, _file = _read_dispatch_release_ledger_locked(
                ledger, authority
            )
            prefix_bytes = int(authority["ledger_prefix_bytes"])
            line = RebootRecovery.canonical_json_line(
                authority["authority_record"]
            )
            if (
                len(raw) < prefix_bytes + len(line)
                or raw[prefix_bytes:prefix_bytes + len(line)] != line
            ):
                raise CampaignPlanError(
                    "safe-release completion lacks durable release authority"
                )
            _validate_dispatch_release_terminal_prefix(
                authority,
                raw[:prefix_bytes],
                dispatch_id=confirm_dispatch_id,
            )
            later = _strict_ledger_records(
                raw[prefix_bytes + len(line):],
                label="safe-release later ledger tail",
            )
        if any(
            row.get("dispatch_id") == confirm_dispatch_id for row in later
        ):
            raise CampaignPlanError(
                "safe-release completion has a conflicting later generation"
            )
        intent_name, preparing_name = _dispatch_release_intent_names(marker_name)
        arm_name = _safe_release_recovery_arm_name(marker_name)
        for residue in (marker_name, intent_name, preparing_name):
            try:
                os.stat(residue, dir_fd=root_fd, follow_symlinks=False)
            except FileNotFoundError:
                continue
            raise CampaignPlanError(
                "safe-release completion still has mutable quarantine residue"
            )
        if allow_arm_residue:
            try:
                installed_arm, arm_identity, _arm_payload = (
                    _read_durable_recovery_record_at(
                        root_fd,
                        arm_name,
                        root_path=root,
                        root_identity=root_identity,
                        label="safe-release recovery arm",
                    )
                )
            except FileNotFoundError:
                pass
            else:
                if installed_arm != arm:
                    raise CampaignPlanError(
                        "safe-release completion arm residue changed"
                    )
                current_arm = os.stat(
                    arm_name, dir_fd=root_fd, follow_symlinks=False
                )
                if (current_arm.st_dev, current_arm.st_ino) != arm_identity:
                    raise CampaignPlanError(
                        "safe-release completion arm identity changed"
                    )
                _validate_durable_recovery_root_binding(
                    root_fd,
                    root,
                    root_identity,
                    label="safe-release recovery arm",
                )
                os.unlink(arm_name, dir_fd=root_fd)
                os.fsync(root_fd)
        else:
            try:
                os.stat(arm_name, dir_fd=root_fd, follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                raise CampaignPlanError(
                    "safe-release completion still has its recovery arm"
                )
        capsule_name = intent.get("capsule_name")
        if isinstance(capsule_name, str):
            try:
                os.stat(capsule_name, dir_fd=root_fd, follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                raise CampaignPlanError(
                    "safe-release completion capsule reappeared"
                )
        _validate_durable_recovery_root_binding(
            root_fd,
            root,
            root_identity,
            label="safe-release recovery receipt",
        )
        os.fsync(root_fd)
        _validate_durable_recovery_root_binding(
            root_fd,
            root,
            root_identity,
            label="safe-release recovery receipt",
        )
        outcome = dict(receipt["terminal_result"])
        outcome["operator_recovery"] = (
            "post_reboot_safe_release_already_completed"
        )
        return outcome
    finally:
        os.close(root_fd)


def _recover_incomplete_safe_release(
    item: dict[str, Any],
    *,
    confirm_dispatch_id: str,
    confirm_recovery_nonce: str,
    boot_identity_provider: RebootRecovery.BootIdentityProvider,
) -> dict[str, Any] | None:
    marker_name = _dispatch_quarantine_name(item)
    arm_name = _safe_release_recovery_arm_name(marker_name)
    receipt_name = _safe_release_recovery_receipt_name(
        marker_name, confirm_dispatch_id
    )
    opened = _open_dispatch_quarantine_root(item, create=False)
    if opened is None:
        return None
    root, root_fd, root_identity = opened
    marker: DispatchQuarantine | None = None
    root_owned = True
    try:
        try:
            arm, arm_identity, _arm_payload = _read_durable_recovery_record_at(
                root_fd,
                arm_name,
                root_path=root,
                root_identity=root_identity,
                label="safe-release recovery arm",
            )
        except FileNotFoundError:
            os.close(root_fd)
            root_owned = False
            return _recover_completed_safe_release_without_marker_locked(
                item,
                confirm_dispatch_id=confirm_dispatch_id,
                confirm_recovery_nonce=confirm_recovery_nonce,
                boot_identity_provider=boot_identity_provider,
            )
        _validate_safe_release_recovery_arm_record(
            arm, marker_name=marker_name
        )
        item_sha = hashlib.sha256(json.dumps(
            item, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")).hexdigest()
        if any((
            confirm_dispatch_id != arm.get("dispatch_id"),
            confirm_recovery_nonce != arm.get("recovery_nonce"),
            arm.get("projected_item_sha256") != item_sha,
            arm.get("marker_root") != os.fspath(root),
            _marker_identity(
                arm.get("marker_root_identity"), "safe-release root"
            ) != root_identity,
        )):
            raise CampaignPlanError(
                "operator confirmation does not match the safe-release arm"
            )
        try:
            current_boot = RebootRecovery.require_changed_boot_identity(
                arm.get("boot_identity_source"),
                arm.get("boot_identity"),
                boot_identity_provider(),
            )
        except RebootRecovery.RecoveryEvidenceError as exc:
            raise CampaignPlanError(str(exc)) from exc
        old_intent = arm["release_intent"]
        assert isinstance(old_intent, dict)
        try:
            os.stat(marker_name, dir_fd=root_fd, follow_symlinks=False)
        except FileNotFoundError:
            os.close(root_fd)
            root_owned = False
            return _recover_completed_safe_release_without_marker_locked(
                item,
                confirm_dispatch_id=confirm_dispatch_id,
                confirm_recovery_nonce=confirm_recovery_nonce,
                boot_identity_provider=boot_identity_provider,
                allow_arm_residue=True,
            )
        marker, _armed, _marker_payload = _open_safe_release_marker(
            item, root, root_fd, root_identity, old_intent
        )
        old_authority = old_intent["release_authority"]
        assert isinstance(old_authority, dict)
        outcome = {
            **dict(old_authority["terminal_result"]),
            "operator_recovery": "post_reboot_safe_release_authenticated",
            "dispatch_id": confirm_dispatch_id,
            "recovery_nonce": confirm_recovery_nonce,
            "recovery_receipt_event": SAFE_RELEASE_RECOVERY_RECEIPT_EVENT,
        }
        ledger = _dispatch_release_item_ledger(item, old_authority)

        try:
            receipt, receipt_identity, _receipt_payload = (
                _read_durable_recovery_record_at(
                    root_fd,
                    receipt_name,
                    root_path=root,
                    root_identity=root_identity,
                    label="safe-release recovery receipt",
                )
            )
        except FileNotFoundError:
            receipt = None
            receipt_identity = None
        if receipt is not None:
            _validate_safe_release_recovery_receipt_record(
                receipt, marker_name=marker_name
            )
            if any((
                receipt.get("arm_record") != arm,
                receipt.get("dispatch_id") != confirm_dispatch_id,
                receipt.get("recovery_nonce") != confirm_recovery_nonce,
                receipt.get("terminal_result") != outcome,
            )):
                raise CampaignPlanError(
                    "safe-release receipt does not bind this recovery"
                )
            fresh_intent = receipt["release_intent"]
            assert isinstance(fresh_intent, dict)
            wal = _safe_release_wal_inventory(root_fd, marker_name)
            if wal != ("intent", str(fresh_intent["intent_name"])):
                raise CampaignPlanError(
                    "safe-release receipt lost its fresh intent"
                )
            installed, installed_identity = _read_dispatch_release_intent_at(
                root_fd, wal[1], marker_name=marker_name
            )
            if installed != fresh_intent:
                raise CampaignPlanError(
                    "safe-release receipt fresh intent changed"
                )
            state = _prevalidate_dispatch_release_preparing(
                item, root_fd, installed, installed_identity
            )
            if state == "authorized":
                _finish_dispatch_release_intent(
                    item, root_fd, installed, installed_identity
                )
                current_arm = os.stat(
                    arm_name, dir_fd=root_fd, follow_symlinks=False
                )
                if (current_arm.st_dev, current_arm.st_ino) != arm_identity:
                    raise CampaignPlanError("safe-release arm identity changed")
                os.unlink(arm_name, dir_fd=root_fd)
                os.fsync(root_fd)
                result = dict(outcome)
                _close_dispatch_quarantine(marker)
                marker = None
                root_owned = False
                return result
            _retire_incomplete_release_for_operator(item, marker)
            assert receipt_identity is not None
            current_receipt = os.stat(
                receipt_name, dir_fd=root_fd, follow_symlinks=False
            )
            if (
                current_receipt.st_dev,
                current_receipt.st_ino,
            ) != receipt_identity:
                raise CampaignPlanError(
                    "safe-release receipt identity changed"
                )
            os.unlink(receipt_name, dir_fd=root_fd)
            os.fsync(root_fd)
        else:
            wal = _safe_release_wal_inventory(root_fd, marker_name)
            if wal is not None:
                role, wal_name = wal
                observed = _read_or_retire_authenticated_fresh_release_wal(
                    item,
                    marker,
                    arm,
                    role=role,
                    wal_name=wal_name,
                )
                if observed is None:
                    wal = None
                else:
                    installed, installed_identity = observed
                if wal is not None:
                    if (
                        role == arm.get("release_wal_role")
                        and wal_name == arm.get("release_wal_name")
                        and installed == old_intent
                        and list(installed_identity)
                        == arm.get("release_wal_identity")
                    ):
                        _validate_safe_release_arm_context(
                            item,
                            root_fd,
                            root_identity,
                            arm,
                            require_original_wal=True,
                        )
                    else:
                        candidate_authority = installed["release_authority"]
                        assert isinstance(candidate_authority, dict)
                        stable_keys = _DISPATCH_RELEASE_AUTHORITY_BASE_KEYS - {
                            "terminal_result", "terminal_result_sha256"
                        }
                        if any(
                            candidate_authority.get(field)
                            != old_authority.get(field)
                            for field in stable_keys
                        ) or candidate_authority.get(
                            "terminal_result"
                        ) != outcome:
                            raise CampaignPlanError(
                                "unreceipted safe-release intent changed authority"
                            )
                        state = _prevalidate_dispatch_release_preparing(
                            item, root_fd, installed, installed_identity
                        )
                        if state == "authorized":
                            raise CampaignPlanError(
                                "unreceipted safe-release intent has full authority"
                            )
                    _retire_incomplete_release_for_operator(item, marker)
            if wal is None:
                with Guard.ledger_append_lock(ledger):
                    raw, _parent, _file = _read_dispatch_release_ledger_locked(
                        ledger, old_authority
                    )
                    prefix, line, tail = _dispatch_release_authority_tail(
                        raw,
                        old_authority,
                        dispatch_id=confirm_dispatch_id,
                    )
                if tail:
                    raise CampaignPlanError(
                        "safe-release WAL disappeared before suffix rollback"
                    )
        release_authority = _build_dispatch_release_authority(
            item,
            marker,
            ledger,
            outcome,
            kind="ordinary_safe_terminal_v1",
        )

        def persist_receipt(
            fresh_intent: dict[str, Any], _identity: tuple[int, int]
        ) -> None:
            receipt = {
                "schema": SAFE_RELEASE_RECOVERY_RECEIPT_SCHEMA,
                "event": SAFE_RELEASE_RECOVERY_RECEIPT_EVENT,
                "recorded_at": datetime.now(timezone.utc).isoformat(),
                "recovery_authority": (
                    "scheduler_authenticated_safe_release_v1"
                ),
                "dispatch_id": confirm_dispatch_id,
                "recovery_nonce": confirm_recovery_nonce,
                "arm_record": arm,
                "arm_record_sha256": _record_sha256(arm),
                "current_boot_identity": (
                    RebootRecovery.boot_identity_receipt(current_boot)
                ),
                "release_intent": fresh_intent,
                "release_intent_sha256": _record_sha256(fresh_intent),
                "terminal_result": outcome,
                "terminal_result_sha256": hashlib.sha256(json.dumps(
                    outcome,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("utf-8")).hexdigest(),
            }
            _validate_safe_release_recovery_receipt_record(
                receipt, marker_name=marker_name
            )
            _install_durable_recovery_record_at(
                marker.root_fd,
                receipt_name,
                receipt,
                root_path=marker.root,
                root_identity=marker.root_identity,
                label="safe-release recovery receipt",
            )

        _release_dispatch_quarantine(
            marker,
            item,
            release_authority,
            before_authority_append=persist_receipt,
        )
        marker = None
        root_owned = False
        reopened = _open_dispatch_quarantine_root(item, create=False)
        if reopened is None:
            raise CampaignPlanError(
                "safe-release root disappeared before arm retirement"
            )
        _reopened_root, reopened_fd, reopened_identity = reopened
        try:
            if reopened_identity != root_identity:
                raise CampaignPlanError("safe-release root identity changed")
            current_arm = os.stat(
                arm_name, dir_fd=reopened_fd, follow_symlinks=False
            )
            if (current_arm.st_dev, current_arm.st_ino) != arm_identity:
                raise CampaignPlanError("safe-release arm identity changed")
            os.unlink(arm_name, dir_fd=reopened_fd)
            os.fsync(reopened_fd)
        finally:
            os.close(reopened_fd)
        return outcome
    finally:
        if marker is not None:
            _close_dispatch_quarantine(marker)
        elif root_owned:
            try:
                os.close(root_fd)
            except OSError:
                pass


def _recover_post_reboot_quarantine_locked(
    item: dict[str, Any],
    *,
    confirm_dispatch_id: str,
    confirm_recovery_nonce: str,
    boot_identity_provider: RebootRecovery.BootIdentityProvider,
) -> dict[str, Any]:
    if (
        not isinstance(confirm_dispatch_id, str)
        or RebootRecovery.DISPATCH_ID_RE.fullmatch(confirm_dispatch_id) is None
    ):
        raise CampaignPlanError(
            "completed recovery dispatch confirmation is malformed"
        )
    if (
        not isinstance(confirm_recovery_nonce, str)
        or RebootRecovery.DISPATCH_ID_RE.fullmatch(confirm_recovery_nonce)
        is None
    ):
        raise CampaignPlanError(
            "completed recovery nonce confirmation is malformed"
        )
    safe_release = _recover_incomplete_safe_release(
        item,
        confirm_dispatch_id=confirm_dispatch_id,
        confirm_recovery_nonce=confirm_recovery_nonce,
        boot_identity_provider=boot_identity_provider,
    )
    if safe_release is not None:
        return safe_release
    if _preflight_post_reboot_release_reconciliation(item):
        raise NoDispatchQuarantine(
            "post-reboot release was already durably reconciled"
        )
    marker, parsed = _read_existing_dispatch_quarantine(
        item, require_recovery_arm=True, allow_missing_capsule=True
    )
    released = False
    try:
        item = _reconstruct_historical_recovery_item(item, parsed.armed)
        if (
            RebootRecovery.DISPATCH_ID_RE.fullmatch(confirm_dispatch_id) is None
            or confirm_dispatch_id != marker.dispatch_id
        ):
            raise CampaignPlanError(
                "operator confirmation does not match the quarantined dispatch"
            )
        arm = parsed.recovery_arm
        if arm is None:  # Enforced by the exact parser.
            raise CampaignPlanError("dispatch marker lacks a recovery arm")
        if arm.get("recovery_arm_schema") != (
            RebootRecovery.RECOVERY_ARM_SCHEMA_V2
        ):
            raise CampaignPlanError(
                "legacy recovery arm lacks explicit current-WIP authority"
            )
        if (
            RebootRecovery.DISPATCH_ID_RE.fullmatch(confirm_recovery_nonce)
            is None
            or confirm_recovery_nonce != arm.get("recovery_nonce")
        ):
            raise CampaignPlanError(
                "operator nonce does not match the post-reboot recovery arm"
            )
        try:
            current_boot = RebootRecovery.require_changed_boot_identity(
                arm.get("boot_identity_source"),
                arm.get("boot_identity"),
                boot_identity_provider(),
            )
        except RebootRecovery.RecoveryEvidenceError as exc:
            raise CampaignPlanError(str(exc)) from exc
        _validate_recovery_marker_seal(marker)
        _retire_incomplete_release_for_operator(item, marker)
        if marker.capsule_missing:
            outcome = _finish_recovery_after_capsule_retirement(
                item=item,
                marker=marker,
                parsed=parsed,
                current_boot=current_boot,
                confirm_recovery_nonce=confirm_recovery_nonce,
            )
            released = True
            return outcome
        try:
            unreconciled_state = _rebind_post_reboot_ledger(
                item,
                parsed,
                marker=marker,
                current_boot=current_boot,
                reconcile_intent=False,
            )
        except CampaignPlanError as exc:
            # A phase intent may accompany a deliberately partial ledger-row
            # append.  That case needs the ordinary reconciler, not WIP
            # side-effect replay.  Only the exact framing failure is deferred.
            if "lacks a final line boundary" not in str(exc):
                raise
            unreconciled_state = None
        if (
            unreconciled_state is not None
            and unreconciled_state.correction is not None
            and unreconciled_state.cleanup is None
            and arm.get("wip_disposition") in {
                "restore_historical_baseline",
                "discard_latest_pointer",
            }
        ):
            pending_cleanup = _build_post_reboot_cleanup(
                item, unreconciled_state.record, parsed
            )
            if _has_exact_pending_cleanup_intent(
                unreconciled_state, pending_cleanup
            ):
                if arm.get("wip_disposition") == "restore_historical_baseline":
                    if (
                        marker.capsule_state is None
                        or marker.capsule_record is None
                    ):
                        raise CampaignPlanError(
                            "pending WIP restore lacks capsule authority"
                        )
                    _rebind_recovery_baselines(
                        item,
                        parsed,
                        capsule_baseline=marker.capsule_state,
                        allow_capsule_restore_progress=True,
                    )
                    _restore_wip_from_rollback_capsule(
                        item, marker.capsule_state, marker.capsule_record
                    )
                else:
                    _artifact, _canonical, current_wip, _reached = (
                        _rebind_recovery_baselines(item, parsed)
                    )
                    if _wip_recovery_state_sha256(
                        current_wip
                    ) == arm.get("confirmed_current_wip_state_sha256"):
                        _discard_confirmed_wip_latest_pointer(
                            item, arm, current_wip
                        )
                    else:
                        _durably_confirm_discarded_wip_state(
                            item, arm, current_wip
                        )
                _validate_recovery_marker_seal(marker)
        artifact_root, canonical_before, wip_before, reached_before = (
            _rebind_recovery_baselines(
                item, parsed, capsule_baseline=marker.capsule_state
            )
        )
        state = _rebind_post_reboot_ledger(
            item, parsed, marker=marker, current_boot=current_boot
        )
        armed_boot = RebootRecovery.BootIdentity(
            str(arm["boot_identity_source"]), str(arm["boot_identity"])
        )
        _validate_recovery_arm(
            item=item,
            marker=marker,
            parsed=parsed,
            boot_identity=armed_boot,
            record=state.record,
            canonical=canonical_before,
            wip=wip_before,
            capsule_baseline=marker.capsule_state,
        )
        workspace, protected = _post_reboot_generation_paths(
            item, state.record
        )
        if state.correction is None:
            (
                observed_workspace,
                observed_protected,
                scan_reason,
                transcript_sha,
                diagnostics_sha,
                _descendant_unproven,
            ) = _exact_tainted_generation(
                item, state.record, require_taint=False
            )
            if observed_workspace != workspace or observed_protected != protected:
                raise CampaignPlanError(
                    "post-reboot generation path changed during authentication"
                )
            _validate_post_reboot_generation_identities(
                parsed, workspace, protected, require_both=True
            )
            if _workspace_lock_is_active(workspace):
                raise CampaignPlanError(
                    "quarantined exact workspace remains active after reboot"
                )
            reason = scan_reason or (
                "post_reboot_unquiesced_dispatch: "
                f"{parsed.unquiesced['reason']}"
            )
            correction = _build_post_reboot_correction(
                item,
                state.record,
                reason=reason,
                transcript_sha=transcript_sha,
                diagnostics_sha=diagnostics_sha,
            )
            _append_recovery_phase_cas(state, correction)
            _validate_recovery_marker_seal(marker)
            state = _rebind_post_reboot_ledger(
                item, parsed, marker=marker, current_boot=current_boot
            )
        if state.cleanup is None:
            assert state.correction is not None
            cleanup = _build_post_reboot_cleanup(
                item, state.record, parsed
            )
            _resume_post_reboot_generation_cleanup(
                item=item,
                parsed=parsed,
                state=state,
                workspace=workspace,
                protected=protected,
            )
            arm_wip = parsed.recovery_arm
            assert arm_wip is not None
            discarded_holder: list[WipRollbackState] = []
            if arm_wip.get("wip_disposition") == "discard_latest_pointer":
                def discard_after_intent() -> None:
                    current_wip = _capture_wip_rollback(item)
                    if _wip_recovery_state_sha256(current_wip) == arm_wip.get(
                        "confirmed_current_wip_state_sha256"
                    ):
                        current_wip = _discard_confirmed_wip_latest_pointer(
                            item, arm_wip, current_wip
                        )
                    else:
                        current_wip = _durably_confirm_discarded_wip_state(
                            item, arm_wip, current_wip
                        )
                    discarded_holder[:] = [current_wip]
                    _validate_post_reboot_clean_state(
                        item=item,
                        artifact_root=artifact_root,
                        parsed=parsed,
                        canonical=canonical_before,
                        wip=current_wip,
                        reached=reached_before,
                        workspace=workspace,
                        protected=protected,
                        capsule_baseline=marker.capsule_state,
                    )
                after_intent = discard_after_intent
            elif arm_wip.get("wip_disposition") == (
                "restore_historical_baseline"
            ):
                capsule_baseline = marker.capsule_state
                capsule_record = marker.capsule_record
                if capsule_baseline is None or capsule_record is None:
                    raise CampaignPlanError(
                        "WIP rollback capsule authority is unavailable"
                    )

                def restore_after_intent() -> None:
                    current_wip = _capture_wip_rollback(item)
                    if _wip_recovery_state_sha256(current_wip) == arm_wip.get(
                        "confirmed_current_wip_state_sha256"
                    ):
                        current_wip = _restore_wip_from_rollback_capsule(
                            item, capsule_baseline, capsule_record
                        )
                    else:
                        _validate_capsule_restored_wip_state(
                            current_wip,
                            capsule_baseline,
                            parsed.armed.get(
                                "wip_restore_logical_state_sha256"
                            ),
                            logical_schema=str(parsed.armed.get(
                                "wip_restore_logical_state_schema"
                            )),
                        )
                    discarded_holder[:] = [current_wip]
                    _validate_post_reboot_clean_state(
                        item=item,
                        artifact_root=artifact_root,
                        parsed=parsed,
                        canonical=canonical_before,
                        wip=current_wip,
                        reached=reached_before,
                        workspace=workspace,
                        protected=protected,
                        capsule_baseline=capsule_baseline,
                    )
                after_intent = restore_after_intent
            else:
                after_intent = None
                _validate_post_reboot_clean_state(
                    item=item,
                    artifact_root=artifact_root,
                    parsed=parsed,
                    canonical=canonical_before,
                    wip=wip_before,
                    reached=reached_before,
                    workspace=workspace,
                    protected=protected,
                    capsule_baseline=marker.capsule_state,
                )
            _assert_same_retry_coordinate(state.ledger, item, state.record)
            if after_intent is None:
                _append_recovery_phase_cas(state, cleanup)
            else:
                _append_recovery_phase_cas(
                    state, cleanup, after_intent=after_intent
                )
            if discarded_holder:
                wip_before = discarded_holder[0]
            _validate_recovery_marker_seal(marker)
            state = _rebind_post_reboot_ledger(
                item, parsed, marker=marker, current_boot=current_boot
            )
            wip_before = _capture_wip_rollback(item)
        if state.operator is None:
            assert state.correction is not None and state.cleanup is not None
            _validate_post_reboot_clean_state(
                item=item,
                artifact_root=artifact_root,
                parsed=parsed,
                canonical=canonical_before,
                wip=wip_before,
                reached=reached_before,
                workspace=workspace,
                protected=protected,
                capsule_baseline=marker.capsule_state,
            )
            _assert_same_retry_coordinate(state.ledger, item, state.record)
            _append_post_reboot_recovery_receipt(
                state=state,
                item=item,
                marker=marker,
                parsed=parsed,
                current_boot=current_boot,
                record=state.record,
                correction=state.correction,
                cleanup=state.cleanup,
                canonical=canonical_before,
                wip=wip_before,
            )
            _validate_recovery_marker_seal(marker)
            state = _rebind_post_reboot_ledger(
                item, parsed, marker=marker, current_boot=current_boot
            )
        assert state.operator is not None
        assert state.correction is not None and state.cleanup is not None
        _validate_post_reboot_operator_receipt(
            item=item,
            marker=marker,
            parsed=parsed,
            current_boot=current_boot,
            record=state.record,
            correction=state.correction,
            cleanup=state.cleanup,
            receipt=state.operator,
            canonical=canonical_before,
            wip=wip_before,
        )
        _validate_post_reboot_clean_state(
            item=item,
            artifact_root=artifact_root,
            parsed=parsed,
            canonical=canonical_before,
            wip=wip_before,
            reached=reached_before,
            workspace=workspace,
            protected=protected,
            capsule_baseline=marker.capsule_state,
        )
        _validate_recovery_marker_seal(marker)
        outcome = {
            "game": item["game"],
            "target_level": item["target_level"],
            "reached": reached_before,
            "result": "tainted_noncounting",
            "reason": state.correction["terminal_errors"][0],
            "child_returncode": int(parsed.unquiesced["child_returncode"]),
            "retry_complexity_n": item["retry_complexity_n"],
            "seed_mode": item["seed_mode"],
            "wip_mode": item["wip_mode"],
            "lineage_input_mode": item["lineage_input_mode"],
            "operator_recovery": "post_reboot_authenticated",
            "dispatch_id": marker.dispatch_id,
            "recovery_nonce": confirm_recovery_nonce,
            "recovery_receipt_event": state.operator["event"],
        }
        _retire_incomplete_release_for_operator(item, marker)
        release_authority = _build_dispatch_release_authority(
            item,
            marker,
            state.ledger,
            outcome,
            kind="post_reboot_operator_terminal_v1",
        )
        _release_dispatch_quarantine(
            marker, item, release_authority
        )
        released = True
        return outcome
    finally:
        if not released:
            _close_dispatch_quarantine(marker)


def _markerless_release_authority_tail_length(
    *,
    ledger: Path,
    baseline: LedgerPrefixState,
    suffix: list[dict[str, Any]],
    receipt: dict[str, Any],
) -> int:
    """Validate the optional release-WAL commit following an operator row."""

    if len(suffix) < 5 or suffix[4].get("event") != (
        "codex_dispatch_release_authorized"
    ):
        return 0
    row = suffix[4]
    expected_prefix = baseline.raw_prefix + b"".join(
        RebootRecovery.canonical_json_line(entry)
        for entry in suffix[:4]
    )
    marker_name = receipt.get("marker_name")
    if not isinstance(marker_name, str):
        raise CampaignPlanError(
            "completed recovery release authority lacks a marker name"
        )
    intent_name, _preparing_name = _dispatch_release_intent_names(marker_name)
    expected = {
        "event": "codex_dispatch_release_authorized",
        "schema": "scheduler_dispatch_release_authorized_v1",
        "dispatch_id": receipt.get("dispatch_id"),
        "intent_name": intent_name,
        "projected_item_sha256": receipt.get("projected_item_sha256"),
        "game": receipt.get("game"),
        "target_level": receipt.get("target_level"),
        "reached": receipt.get("reached"),
        "parent_action_count": receipt.get("parent_action_count"),
        "terminal_kind": "post_reboot_operator_terminal_v1",
        "terminal_event": (
            "codex_post_reboot_operator_recovery_completed"
        ),
        "terminal_record_sha256": _recovery_record_sha256(receipt),
        "ledger": os.fspath(ledger),
        "ledger_parent_identity": list(baseline.parent_identity),
        "ledger_file_identity": (
            list(baseline.file_identity)
            if baseline.file_identity is not None else None
        ),
        "ledger_prefix_bytes": len(expected_prefix),
        "ledger_prefix_sha256": hashlib.sha256(
            expected_prefix
        ).hexdigest(),
        **{
            field: receipt.get(field)
            for field in Status.FRONTIER_BINDING_FIELDS
        },
    }
    if (
        set(row) != _DISPATCH_RELEASE_AUTHORITY_RECORD_KEYS
        or any(row.get(field) != value for field, value in expected.items())
        or not isinstance(row.get("retry_complexity_n"), int)
        or isinstance(row.get("retry_complexity_n"), bool)
        or row["retry_complexity_n"] < 0
        or not isinstance(row.get("release_nonce"), str)
        or SHA256_RE.fullmatch(row["release_nonce"]) is None
        or not isinstance(row.get("intent_core_sha256"), str)
        or SHA256_RE.fullmatch(row["intent_core_sha256"]) is None
    ):
        raise CampaignPlanError(
            "completed recovery release authority row is invalid"
        )
    _marker_identity(row.get("intent_identity"), "completed release intent")
    _recovery_recorded_at(row, "completed release authorization")
    return 1


def _recover_completed_without_marker_locked(
    item: dict[str, Any],
    *,
    confirm_dispatch_id: str,
    confirm_recovery_nonce: str,
    boot_identity_provider: RebootRecovery.BootIdentityProvider,
) -> dict[str, Any]:
    """Resolve unlink/fsync ambiguity from a fully committed ledger chain."""

    if (
        not isinstance(confirm_dispatch_id, str)
        or RebootRecovery.DISPATCH_ID_RE.fullmatch(confirm_dispatch_id) is None
    ):
        raise CampaignPlanError(
            "completed recovery dispatch confirmation is malformed"
        )
    if (
        not isinstance(confirm_recovery_nonce, str)
        or RebootRecovery.DISPATCH_ID_RE.fullmatch(confirm_recovery_nonce)
        is None
    ):
        raise CampaignPlanError(
            "completed recovery nonce confirmation is malformed"
        )
    try:
        current_boot = RebootRecovery.validate_boot_identity(
            boot_identity_provider()
        )
    except RebootRecovery.RecoveryEvidenceError as exc:
        raise CampaignPlanError(str(exc)) from exc
    ledger = _ledger_path(item["argv"], cwd=_runner_cwd(item))
    records = _read_ledger_locked(ledger)
    operator_receipts = [
        row for row in records
        if (
            row.get("event")
            == "codex_post_reboot_operator_recovery_completed"
            and row.get("schema") in {
                RebootRecovery.OPERATOR_RECOVERY_SCHEMA,
                RebootRecovery.LEGACY_OPERATOR_RECOVERY_SCHEMA,
            }
        )
    ]
    if any(
        not isinstance(row.get("dispatch_id"), str)
        or RebootRecovery.DISPATCH_ID_RE.fullmatch(row["dispatch_id"]) is None
        for row in operator_receipts
    ):
        raise CampaignPlanError(
            "completed recovery receipt dispatch ID is malformed"
        )
    matches = [
        row for row in operator_receipts
        if row.get("dispatch_id") == confirm_dispatch_id
    ]
    if len(matches) != 1:
        raise NoDispatchQuarantine(
            "no marker or unique completed recovery receipt exists"
        )
    receipt = matches[0]
    operator_keys_v2 = {
        "event", "schema", "recorded_at", "recovery_authority",
        "dispatch_id", "marker_root_identity",
        "pre_arm_marker_identity", "armed_marker_identity",
        "marker_root", "marker_name", "dispatch_unquiesced_at",
        "recovery_nonce", "armed_boot_identity", "current_boot_identity",
        "thread_id", "transcript", "workspace", "game", "target_level",
        "failure_class", "taint_verdict", "solved_target",
        "retry_increment", "artifact_root", "artifact_root_identity",
        "canonical_root", "canonical_root_identity", "canonical_digest",
        "canonical_root_metadata", "target_wip_snapshot",
        "wip_root_metadata", "wip_state_sha256", "ledger",
        "wip_recovery_authority",
        "confirmed_current_wip_state_sha256",
        "wip_disposition", "discard_survivor_sha256",
        "restored_wip_logical_state_sha256",
        "wip_restore_logical_state_schema",
        "wip_rollback_capsule_name", "wip_rollback_capsule_identity",
        "wip_rollback_capsule_bytes", "wip_rollback_capsule_sha256",
        "wip_rollback_capsule_state_sha256",
        "ledger_parent_identity",
        "ledger_file_identity", "ledger_prefix_bytes",
        "ledger_prefix_sha256", "projected_item_sha256",
        "workspace_identity", "protected_identity", "exec_record_sha256",
        "workspace_lock_schema", "workspace_lock_path",
        "workspace_lock_identity",
        "correction_record_sha256", "cleanup_record_sha256",
        *Status.FRONTIER_BINDING_FIELDS, "reached", "parent_action_count",
    }
    wip_v2_keys = {
        "wip_recovery_authority",
        "confirmed_current_wip_state_sha256",
        "wip_disposition",
        "discard_survivor_sha256",
        "restored_wip_logical_state_sha256",
        "wip_restore_logical_state_schema",
        "wip_rollback_capsule_name",
        "wip_rollback_capsule_identity",
        "wip_rollback_capsule_bytes",
        "wip_rollback_capsule_sha256",
        "wip_rollback_capsule_state_sha256",
    }
    operator_keys_v1 = operator_keys_v2 - wip_v2_keys
    expected_operator_keys = (
        operator_keys_v2
        if receipt.get("schema") == RebootRecovery.OPERATOR_RECOVERY_SCHEMA
        else operator_keys_v1
    )
    if set(receipt) != expected_operator_keys or receipt.get(
        "recovery_authority"
    ) != "scheduler_authenticated_post_reboot_v1":
        raise CampaignPlanError(
            "completed recovery receipt has an invalid exact schema"
        )
    if receipt.get("schema") == RebootRecovery.OPERATOR_RECOVERY_SCHEMA:
        authority = receipt.get("wip_recovery_authority")
        disposition = receipt.get("wip_disposition")
        if (authority, disposition) not in {
            (
                "operator_confirmed_quarantined_wip_v1",
                "discard_latest_pointer",
            ),
            (
                "operator_confirmed_quarantined_wip_v1",
                "confirmed_latest_absent",
            ),
            (
                "dispatch_full_wip_rollback_capsule_v1",
                "restore_historical_baseline",
            ),
        }:
            raise CampaignPlanError(
                "completed recovery receipt has invalid WIP authority"
            )
        confirmed = receipt.get("confirmed_current_wip_state_sha256")
        if not isinstance(confirmed, str) or SHA256_RE.fullmatch(
            confirmed
        ) is None:
            raise CampaignPlanError(
                "completed recovery receipt has an invalid WIP state seal"
            )
        survivor = receipt.get("discard_survivor_sha256")
        if disposition == "discard_latest_pointer":
            if not isinstance(survivor, str) or SHA256_RE.fullmatch(
                survivor
            ) is None:
                raise CampaignPlanError(
                    "completed recovery receipt has an invalid survivor seal"
                )
        elif survivor is not None:
            raise CampaignPlanError(
                "completed recovery receipt has an unexpected survivor seal"
            )
        restored_logical = receipt.get(
            "restored_wip_logical_state_sha256"
        )
        if disposition == "restore_historical_baseline":
            if (
                not isinstance(restored_logical, str)
                or SHA256_RE.fullmatch(restored_logical) is None
                or receipt.get("wip_restore_logical_state_schema")
                not in {
                    WIP_LOGICAL_RESTORE_SCHEMA_V1,
                    WIP_LOGICAL_RESTORE_SCHEMA,
                }
            ):
                raise CampaignPlanError(
                    "completed recovery receipt has an invalid logical seal"
                )
        elif (
            restored_logical is not None
            or receipt.get("wip_restore_logical_state_schema") is not None
        ):
            raise CampaignPlanError(
                "completed legacy recovery has a logical restore seal"
            )
        capsule_fields = (
            receipt.get("wip_rollback_capsule_name"),
            receipt.get("wip_rollback_capsule_identity"),
            receipt.get("wip_rollback_capsule_bytes"),
            receipt.get("wip_rollback_capsule_sha256"),
            receipt.get("wip_rollback_capsule_state_sha256"),
        )
        if disposition == "restore_historical_baseline":
            capsule_name, capsule_identity, capsule_bytes, capsule_sha, state_sha = (
                capsule_fields
            )
            if any((
                not isinstance(capsule_name, str),
                isinstance(capsule_name, str)
                and Path(capsule_name).name != capsule_name,
                not isinstance(capsule_bytes, int),
                isinstance(capsule_bytes, bool),
                isinstance(capsule_bytes, int)
                and not 0 < capsule_bytes <= MAX_WIP_ROLLBACK_CAPSULE_BYTES,
                not isinstance(capsule_sha, str),
                isinstance(capsule_sha, str)
                and SHA256_RE.fullmatch(capsule_sha) is None,
                not isinstance(state_sha, str),
                isinstance(state_sha, str)
                and SHA256_RE.fullmatch(state_sha) is None,
            )):
                raise CampaignPlanError(
                    "completed recovery receipt has an invalid capsule binding"
                )
            _marker_identity(capsule_identity, "WIP rollback capsule")
        elif any(value is not None for value in capsule_fields):
            raise CampaignPlanError(
                "completed legacy recovery unexpectedly binds a capsule"
            )
    _recovery_recorded_at(
        receipt, "completed post-reboot operator receipt"
    )
    receipt_nonce = receipt.get("recovery_nonce")
    if (
        not isinstance(receipt_nonce, str)
        or RebootRecovery.DISPATCH_ID_RE.fullmatch(receipt_nonce) is None
    ):
        raise CampaignPlanError(
            "completed recovery receipt nonce is malformed"
        )
    if receipt_nonce != confirm_recovery_nonce:
        raise CampaignPlanError(
            "operator nonce does not match the completed recovery"
        )
    item = _reconstruct_historical_recovery_item(item, receipt)
    projected_ledger = _ledger_path(item["argv"], cwd=_runner_cwd(item))
    if projected_ledger != ledger:
        raise CampaignPlanError(
            "completed recovery receipt projects a different Codex ledger"
        )
    ledger = projected_ledger
    item_sha = hashlib.sha256(json.dumps(
        item, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()
    expected_binding = {
        "game": item["game"],
        "target_level": item["target_level"],
        "projected_item_sha256": item_sha,
        "artifact_root": os.fspath(_artifact_root(item)),
        "canonical_root": os.fspath(
            _artifact_root(item) / f"{item['game']}_legs"
        ),
        "ledger": os.fspath(ledger),
        "failure_class": "taint",
        "taint_verdict": "tainted",
        "solved_target": None,
        "retry_increment": 0,
        **{
            field: item[field]
            for field in (
                *Status.FRONTIER_BINDING_FIELDS,
                "reached",
                "parent_action_count",
            )
        },
    }
    if any(receipt.get(field) != value for field, value in expected_binding.items()):
        raise CampaignPlanError(
            "completed recovery receipt does not bind the projected item"
        )
    armed_boot = receipt.get("armed_boot_identity")
    recovery_boot = receipt.get("current_boot_identity")
    current_boot_receipt = RebootRecovery.boot_identity_receipt(current_boot)
    if (
        not isinstance(armed_boot, dict)
        or set(armed_boot) != {"source", "identity_sha256"}
        or armed_boot.get("source") != current_boot.source
        or not isinstance(armed_boot.get("identity_sha256"), str)
        or SHA256_RE.fullmatch(armed_boot["identity_sha256"]) is None
        or not isinstance(recovery_boot, dict)
        or set(recovery_boot) != {"source", "identity_sha256"}
        or recovery_boot.get("source") != armed_boot.get("source")
        or not isinstance(recovery_boot.get("identity_sha256"), str)
        or SHA256_RE.fullmatch(recovery_boot["identity_sha256"]) is None
        or armed_boot == recovery_boot
        or current_boot_receipt == armed_boot
    ):
        raise CampaignPlanError(
            "completed recovery receipt lacks a changed boot identity"
        )
    ledger_path, baseline, suffix = _read_post_reboot_ledger_surface(
        item, receipt
    )
    expected_events = [
        "codex_exec",
        "codex_exec_classification_correction",
        "codex_taint_cleanup_completed",
        "codex_post_reboot_operator_recovery_completed",
    ]
    if (
        len(suffix) < 4
        or [row.get("event") for row in suffix[:4]] != expected_events
        or suffix[3] != receipt
    ):
        raise CampaignPlanError(
            "completed recovery ledger chain lacks its exact four-row prefix"
        )
    release_tail_length = _markerless_release_authority_tail_length(
        ledger=ledger,
        baseline=baseline,
        suffix=suffix,
        receipt=receipt,
    )
    # V2 was introduced together with the durable release-WAL authority row.
    # Only the explicitly supported V1 compatibility receipt may predate that
    # row; accepting a current receipt without it would let marker absence act
    # as deletion authority.
    if (
        receipt.get("schema") == RebootRecovery.OPERATOR_RECOVERY_SCHEMA
        and release_tail_length != 1
    ):
        raise CampaignPlanError(
            "current completed recovery lacks durable release authority"
        )
    conflicting_tail = [
        row for row in suffix[4 + release_tail_length:]
        if (
            row.get("dispatch_id") == confirm_dispatch_id
            or row.get("thread_id") == receipt.get("thread_id")
            or row.get("transcript") == receipt.get("transcript")
        )
    ]
    if conflicting_tail:
        raise CampaignPlanError(
            "completed recovery ledger has a conflicting later generation row"
        )
    record = _expected_exec_record(
        item,
        baseline.records,
        [*baseline.records, suffix[0]],
        clean_terminal=False,
    )
    correction, cleanup = suffix[1], suffix[2]
    if any((
        receipt.get("thread_id") != record.get("thread_id"),
        receipt.get("transcript") != record.get("transcript"),
        receipt.get("workspace") != record.get("workspace"),
    )):
        raise CampaignPlanError(
            "completed recovery receipt does not bind its exec generation"
        )
    pre_arm_marker_identity = _marker_identity(
        receipt.get("pre_arm_marker_identity"), "pre-arm marker"
    )
    armed_marker_identity = _marker_identity(
        receipt.get("armed_marker_identity"), "armed marker"
    )
    if pre_arm_marker_identity == armed_marker_identity:
        raise CampaignPlanError(
            "completed recovery marker identities are not distinct"
        )
    _marker_identity(receipt.get("workspace_identity"), "workspace")
    _marker_identity(receipt.get("protected_identity"), "protected")
    workspace_lock_identity = _marker_identity(
        receipt.get("workspace_lock_identity"), "workspace lock"
    )
    marker_time = _recovery_recorded_at(
        {"recorded_at": receipt.get("dispatch_unquiesced_at")},
        "dispatch unquiesced",
    )
    _validate_recovery_correction(
        item, record, correction, not_before=marker_time
    )
    _validate_recovery_cleanup(
        item,
        record,
        cleanup,
        not_before=_recovery_recorded_at(correction, "post-reboot correction"),
    )
    if any((
        receipt.get("exec_record_sha256")
        != _recovery_record_sha256(record),
        receipt.get("correction_record_sha256")
        != _recovery_record_sha256(correction),
        receipt.get("cleanup_record_sha256")
        != _recovery_record_sha256(cleanup),
    )):
        raise CampaignPlanError(
            "completed recovery receipt chain hashes do not match"
        )
    artifact_root = _artifact_root(item)
    if _host_directory_identity(
        artifact_root, "canonical artifact root"
    ) != _marker_identity(
        receipt.get("artifact_root_identity"), "artifact root"
    ):
        raise CampaignPlanError("completed recovery artifact root changed")
    canonical = _capture_canonical_rollback(item)
    if any((
        canonical.root_identity != _marker_identity(
            receipt.get("canonical_root_identity"), "canonical root"
        ),
        canonical.digest != receipt.get("canonical_digest"),
        _canonical_root_recovery_metadata(canonical)
        != receipt.get("canonical_root_metadata"),
        not isinstance(receipt.get("target_wip_snapshot"), list)
        or len(receipt["target_wip_snapshot"]) != 2
        or receipt["target_wip_snapshot"][0]
        != os.fspath(_target_wip_level(item)),
        _checkpoint_reached(item["game"]) != item["reached"],
    )):
        raise CampaignPlanError(
            "completed recovery canonical baseline changed"
        )
    wip = _capture_wip_rollback(item)
    if _wip_root_recovery_metadata(wip) != receipt.get("wip_root_metadata"):
        raise CampaignPlanError(
            "completed recovery WIP root metadata changed"
        )
    if _wip_recovery_state_sha256(wip) != receipt.get("wip_state_sha256"):
        raise CampaignPlanError(
            "completed recovery WIP state changed"
        )
    if receipt.get("wip_disposition") == "discard_latest_pointer":
        _validate_discarded_wip_state(item, receipt, wip)
    elif receipt.get("wip_disposition") == "restore_historical_baseline":
        logical_schema = receipt.get("wip_restore_logical_state_schema")
        if (
            logical_schema not in {
                WIP_LOGICAL_RESTORE_SCHEMA_V1,
                WIP_LOGICAL_RESTORE_SCHEMA,
            }
            or _wip_logical_restore_state_sha256(
                wip, schema=str(logical_schema)
            )
            != receipt.get("restored_wip_logical_state_sha256")
        ):
            raise CampaignPlanError(
                "completed recovery logical WIP state changed"
            )
    workspace, protected = _post_reboot_generation_paths(item, record)
    workspace_tombstone, protected_tombstone = _post_reboot_tombstones(
        confirm_dispatch_id,
        workspace,
        protected,
    )
    lock_schema = receipt.get("workspace_lock_schema")
    expected_lock_path = (
        workspace / ".orchestrate.lock"
        if lock_schema == "in_workspace_v1"
        else Path(Legs._workspace_lock_path(os.fspath(workspace)))
    )
    if (
        lock_schema != _lock_schema(item)
        or receipt.get("workspace_lock_path") != os.fspath(expected_lock_path)
    ):
        raise CampaignPlanError(
            "completed recovery workspace lock binding changed"
        )
    if any(os.path.lexists(path) for path in (
        workspace,
        protected,
        workspace_tombstone,
        protected_tombstone,
    )):
        raise CampaignPlanError(
            "completed recovery generation unexpectedly reappeared"
        )
    if lock_schema == "hashed_external_v1" and os.path.lexists(
        expected_lock_path
    ):
        metadata = expected_lock_path.stat(follow_symlinks=False)
        if (
            expected_lock_path.is_symlink()
            or not stat.S_ISREG(metadata.st_mode)
            or (metadata.st_dev, metadata.st_ino) != workspace_lock_identity
        ):
            raise CampaignPlanError(
                "completed recovery workspace lock identity changed"
            )
        raise CampaignPlanError(
            "completed recovery external workspace lock reappeared"
        )
    _assert_same_retry_coordinate(ledger_path, item, record)
    opened = _open_dispatch_quarantine_root(item, create=False)
    if opened is None:
        raise CampaignPlanError(
            "completed recovery quarantine root is unavailable for fsync"
        )
    root, root_fd, root_identity = opened
    try:
        if (
            os.fspath(root) != receipt.get("marker_root")
            or root_identity != _marker_identity(
                receipt.get("marker_root_identity"), "marker root"
            )
            or receipt.get("marker_name") != _dispatch_quarantine_name(item)
        ):
            raise CampaignPlanError(
                "completed recovery quarantine root identity changed"
            )
        try:
            os.stat(
                _dispatch_quarantine_name(item),
                dir_fd=root_fd,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            pass
        else:
            raise CampaignPlanError(
                "completed recovery marker unexpectedly reappeared"
            )
        capsule_name = receipt.get("wip_rollback_capsule_name")
        if capsule_name is not None:
            assert isinstance(capsule_name, str)
            try:
                os.stat(
                    capsule_name,
                    dir_fd=root_fd,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                pass
            else:
                raise CampaignPlanError(
                    "completed recovery capsule unexpectedly reappeared"
                )
        os.fsync(root_fd)
    except OSError as exc:
        raise CampaignPlanError(
            "could not resolve completed marker fsync ambiguity"
        ) from exc
    finally:
        os.close(root_fd)
    return {
        "game": item["game"],
        "target_level": item["target_level"],
        "reached": item["reached"],
        "result": "tainted_noncounting",
        "retry_complexity_n": item["retry_complexity_n"],
        "operator_recovery": "post_reboot_already_completed",
        "dispatch_id": confirm_dispatch_id,
        "recovery_nonce": confirm_recovery_nonce,
        "recovery_receipt_event": receipt["event"],
    }


def _recover_post_reboot_quarantine(
    item: dict[str, Any],
    *,
    confirm_dispatch_id: str,
    confirm_recovery_nonce: str,
    boot_identity_provider: RebootRecovery.BootIdentityProvider = (
        RebootRecovery.authoritative_boot_identity
    ),
) -> dict[str, Any]:
    """Explicitly recover one marker only across a proven reboot boundary."""

    dispatch_lock = _acquire_scheduler_dispatch_lock(item)
    try:
        lineage_lock = _acquire_scheduler_lineage_lock(item)
        try:
            try:
                return _recover_post_reboot_quarantine_locked(
                    item,
                    confirm_dispatch_id=confirm_dispatch_id,
                    confirm_recovery_nonce=confirm_recovery_nonce,
                    boot_identity_provider=boot_identity_provider,
                )
            except NoDispatchQuarantine:
                return _recover_completed_without_marker_locked(
                    item,
                    confirm_dispatch_id=confirm_dispatch_id,
                    confirm_recovery_nonce=confirm_recovery_nonce,
                    boot_identity_provider=boot_identity_provider,
                )
        finally:
            _release_scheduler_artifact_lock(lineage_lock)
    finally:
        _release_scheduler_artifact_lock(dispatch_lock)


def _dispatch_workspace_prefix(item: dict[str, Any]) -> str:
    tag = _single_cli_value(item["argv"], "--tag")
    if (
        not isinstance(tag, str)
        or not tag
        or SAFE_COMPONENT_RE.fullmatch(tag) is None
    ):
        raise CampaignPlanError("live boundary watchdog requires one safe tag")
    return f"gkm_legs_ws_{item['game']}_{tag}_"


def _physical_directory_names(
    root: Path, prefix: str
) -> dict[str, tuple[int, int]]:
    if not os.path.lexists(root):
        return {}
    _reject_symlinked_ancestry(root, "live boundary inventory root")
    _host_directory_identity(root, "live boundary inventory root")
    names: dict[str, tuple[int, int]] = {}
    try:
        for entry in root.iterdir():
            if not entry.name.startswith(prefix):
                continue
            metadata = entry.stat(follow_symlinks=False)
            if entry.is_symlink() or not stat.S_ISDIR(metadata.st_mode):
                raise CampaignPlanError(
                    f"unsafe live generation inventory entry: {entry}"
                )
            names[entry.name] = (metadata.st_dev, metadata.st_ino)
    except OSError as exc:
        raise CampaignPlanError("live generation inventory is unstable") from exc
    return names


def _historical_tester_scaffolds(
    item: dict[str, Any], workspace: Path
) -> dict[str, frozenset[str]]:
    """Render TESTER from the authenticated historical source without importing it."""

    receipt = item.get("historical_runner")
    if not isinstance(receipt, dict):
        return Legs._trusted_host_scaffold_hashes(os.fspath(workspace))
    source = (
        Path(receipt["worktree"])
        / "arc"
        / "crack_lab"
        / "gkm_legs.py"
    )
    try:
        raw = Legs._read_single_link_regular(os.fspath(source))
        expected_sha = receipt.get("source_sha256")
        if (
            not isinstance(expected_sha, str)
            or SHA256_RE.fullmatch(expected_sha) is None
            or hashlib.sha256(raw).hexdigest() != expected_sha
        ):
            raise CampaignPlanError(
                "authenticated historical runner source hash drifted"
            )
        tree = ast.parse(raw, filename=os.fspath(source))
    except (OSError, SyntaxError, Legs.WorkspaceTainted) as exc:
        raise CampaignPlanError(
            "authenticated historical TESTER template is unavailable"
        ) from exc
    templates: list[str] = []
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if not any(
            isinstance(target, ast.Name) and target.id == "TESTER"
            for target in targets
        ):
            continue
        value = node.value
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            templates.append(value.value)
    if len(templates) != 1:
        raise CampaignPlanError(
            "authenticated historical runner has an ambiguous TESTER template"
        )
    try:
        payload = templates[0].format(
            labdir=os.fspath(source.parent), game=item["game"]
        ).encode("utf-8")
    except (KeyError, ValueError) as exc:
        raise CampaignPlanError(
            "authenticated historical TESTER template cannot be rendered"
        ) from exc
    return {"gkm_try.py": frozenset({hashlib.sha256(payload).hexdigest()})}


def _live_transcript_inventory(
    item: dict[str, Any], protected: Path
) -> tuple[Path | None, str | None]:
    if not os.path.lexists(protected):
        return None, None
    _reject_symlinked_ancestry(protected, "live protected transcript directory")
    _host_directory_identity(protected, "live protected transcript directory")
    transcript_paths: list[Path] = []
    allowed_suffixes = {".jsonl"}
    if _evidence_schema(item) == "sealed_transcript_diagnostics_v1":
        allowed_suffixes.add(".log")
    try:
        for entry in protected.iterdir():
            metadata = entry.stat(follow_symlinks=False)
            if (
                entry.is_symlink()
                or not stat.S_ISREG(metadata.st_mode)
                or metadata.st_nlink != 1
            ):
                return None, f"unsafe live protected evidence node: {entry.name}"
            if (
                entry.name.startswith("codex_turn_")
                and entry.name.endswith(".jsonl")
            ):
                transcript_paths.append(entry)
            elif (
                entry.suffix not in allowed_suffixes
                or not entry.name.startswith("codex_turn_")
                or not entry.name.endswith(".stderr.log")
            ):
                return None, (
                    "ambiguous live protected evidence inventory: "
                    f"{entry.name}"
                )
    except OSError as exc:
        raise CampaignPlanError(
            "live protected evidence inventory is unstable"
        ) from exc
    if len(transcript_paths) > 1:
        return None, "multiple live Codex transcripts appeared for one generation"
    return (transcript_paths[0] if transcript_paths else None), None


def _launch_exact_child(
    argv: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None,
    ownership: list[Contiguous.ScopedProcessTree | None] | None = None,
) -> Contiguous.ScopedProcessTree:
    """Launch one exact runner under the shared scoped-tree supervisor."""

    try:
        return Contiguous.ScopedProcessTree.launch(
            argv,
            cwd=cwd,
            environment=env,
            ownership=ownership,
        )
    except Contiguous.ScopedProcessContainmentError as exc:
        raise UnquiescedChildError(
            "exact child launch left process containment unproven"
        ) from exc
    except Contiguous.SupervisorContractError as exc:
        raise CampaignPlanError(
            f"exact child supervision rejected launch: {exc}"
        ) from exc


def _run_guarded_child(
    item: dict[str, Any],
    argv: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None,
) -> GuardedChildResult:
    """Run one exact dispatch behind a current-policy live boundary watchdog."""

    scratch = Path(Legs.SCRATCH).absolute()
    prefix = _dispatch_workspace_prefix(item)
    protected_root = scratch / ".proposer_transcripts"
    workspaces_before = _physical_directory_names(scratch, prefix)
    protected_before = _physical_directory_names(protected_root, prefix)
    historical = _evidence_schema(item) == "sealed_transcript_only_v1"
    _revalidate_historical_control(item)
    process_tree_owner: list[Contiguous.ScopedProcessTree | None] = [None]
    process_tree: Contiguous.ScopedProcessTree | None = None
    terminal_returncode: int | None = None
    detached_processes_proven_absent = False

    def stop_for_quarantine() -> int:
        nonlocal terminal_returncode, detached_processes_proven_absent
        if process_tree is None:
            raise CampaignPlanError(
                "exact child custody handoff was lost before quarantine"
            )
        if process_tree.sealed:
            if terminal_returncode is None:
                raise CampaignPlanError(
                    "sealed exact process tree lacks a terminal status"
                )
            return terminal_returncode
        try:
            terminal = process_tree.seal(
                stop_requested=True,
                grace_seconds=EXACT_CHILD_TERMINATE_SECONDS,
            )
        except Contiguous.ScopedProcessContainmentError as exc:
            raise UnquiescedChildError(
                "exact child stop left descendant containment unproven"
            ) from exc
        except Contiguous.SupervisorContractError as exc:
            if not process_tree.sealed:
                raise UnquiescedChildError(
                    "exact child stop left descendant containment unproven"
                ) from exc
            raise CampaignPlanError(
                f"exact child process-tree stop failed: {exc}"
            ) from exc
        terminal_returncode = terminal.returncode
        detached_processes_proven_absent = bool(
            getattr(
                terminal, "detached_processes_proven_absent", False
            )
        )
        return terminal_returncode

    def finish_normal_exit() -> int:
        nonlocal terminal_returncode, detached_processes_proven_absent
        if process_tree is None:
            raise CampaignPlanError(
                "exact child custody handoff was lost before finalization"
            )
        if process_tree.sealed:
            if terminal_returncode is None:
                raise CampaignPlanError(
                    "sealed exact process tree lacks a terminal status"
                )
            return terminal_returncode
        try:
            terminal = process_tree.seal(
                stop_requested=False,
                grace_seconds=0,
            )
        except Contiguous.ScopedProcessContainmentError as exc:
            raise UnquiescedChildError(
                "exact child exit left descendant containment unproven"
            ) from exc
        except Contiguous.SupervisorContractError as exc:
            if not process_tree.sealed:
                raise UnquiescedChildError(
                    "exact child exit left descendant containment unproven"
                ) from exc
            raise CampaignPlanError(
                f"exact child process-tree finalization failed: {exc}"
            ) from exc
        terminal_returncode = terminal.returncode
        detached_processes_proven_absent = bool(
            getattr(
                terminal, "detached_processes_proven_absent", False
            )
        )
        return terminal_returncode

    workspace: Path | None = None
    protected: Path | None = None
    monitor: Boundary.LiveBoundaryMonitor | None = None
    transcript: Path | None = None
    transcript_seen = False
    workspace_identity: tuple[int, int] | None = None
    protected_identity: tuple[int, int] | None = None
    taint_reason: str | None = None
    descendant_quiescence_unproven = False
    try:
        process_tree = _launch_exact_child(
            argv,
            cwd=cwd,
            env=env,
            ownership=process_tree_owner,
        )
        if process_tree_owner[0] is None:
            # Compatibility with synthetic launch fakes; the production
            # launcher publishes before returning.
            process_tree_owner[0] = process_tree
        while True:
            try:
                final = process_tree.observe_exit()
            except Contiguous.ScopedProcessContainmentError as exc:
                raise UnquiescedChildError(
                    "live exact child observation lost process containment"
                ) from exc
            except Contiguous.SupervisorContractError as exc:
                raise CampaignPlanError(
                    f"live exact child observation failed: {exc}"
                ) from exc
            current_workspaces = _physical_directory_names(scratch, prefix)
            new_workspaces = set(current_workspaces) - set(workspaces_before)
            if len(new_workspaces) > 1:
                if not final:
                    stop_for_quarantine()
                raise CampaignPlanError(
                    "one dispatch created multiple candidate workspaces"
                )
            current_protected = _physical_directory_names(protected_root, prefix)
            new_protected = set(current_protected) - set(protected_before)
            if len(new_protected) > 1:
                if not final:
                    stop_for_quarantine()
                raise CampaignPlanError(
                    "one dispatch created multiple protected evidence roots"
                )
            if new_workspaces:
                name = next(iter(new_workspaces))
                if new_protected and new_protected != {name}:
                    if not final:
                        stop_for_quarantine()
                    raise CampaignPlanError(
                        "live workspace and protected evidence identities diverged"
                    )
                candidate = scratch / name
                if workspace is not None and candidate != workspace:
                    if not final:
                        stop_for_quarantine()
                    raise CampaignPlanError("live workspace identity changed")
                workspace = candidate
                protected = protected_root / name
                observed_workspace_identity = current_workspaces[name]
                if (
                    workspace_identity is not None
                    and observed_workspace_identity != workspace_identity
                ):
                    if not final:
                        stop_for_quarantine()
                    raise CampaignPlanError(
                        "live workspace directory identity changed"
                    )
                workspace_identity = observed_workspace_identity
                if name in current_protected:
                    observed_protected_identity = current_protected[name]
                    if (
                        protected_identity is not None
                        and observed_protected_identity != protected_identity
                    ):
                        if not final:
                            stop_for_quarantine()
                        raise CampaignPlanError(
                            "live protected directory identity changed"
                        )
                    protected_identity = observed_protected_identity
                # Workspace creation precedes the legacy lock and host scaffold
                # writes.  Do not scan a partially written host template; model
                # authority begins only after the exact runner lock is active.
                if monitor is None and (
                    _workspace_lock_is_active(workspace) or final
                ):
                    trusted = _historical_tester_scaffolds(item, workspace)
                    module_root = (
                        Path(item["historical_runner"]["worktree"])
                        / "arc"
                        / "crack_lab"
                        if isinstance(item.get("historical_runner"), dict)
                        else HERE
                    )
                    monitor = Boundary.LiveBoundaryMonitor(
                        workspace,
                        arena_module_root=module_root,
                        trusted_host_scaffolds=trusted,
                        allow_historical_transport_banner=historical,
                    )
                if monitor is not None:
                    findings = monitor.scan_workspace()
                    findings = Legs._filter_trusted_scaffold_root_literal(
                        workspace,
                        findings,
                        trusted=monitor.trusted_host_scaffolds,
                    )
                    if any(
                        finding.code in UNQUIESCED_BOUNDARY_CODES
                        for finding in findings
                    ):
                        descendant_quiescence_unproven = True
                    if findings and taint_reason is None:
                        taint_reason = findings[0].describe()
                    assert protected is not None
                    selected, inventory_reason = _live_transcript_inventory(
                        item, protected
                    )
                    if inventory_reason is not None and taint_reason is None:
                        taint_reason = inventory_reason
                    if selected is not None:
                        if transcript is not None and selected != transcript:
                            taint_reason = taint_reason or (
                                "live transcript identity changed"
                            )
                        transcript = selected
                        transcript_seen = True
                        transcript_findings = monitor.scan_transcript(
                            selected, final=final
                        )
                        if any(
                            finding.code in UNQUIESCED_BOUNDARY_CODES
                            for finding in transcript_findings
                        ):
                            descendant_quiescence_unproven = True
                        if transcript_findings and taint_reason is None:
                            taint_reason = transcript_findings[0].describe()
                    elif final and transcript_seen and taint_reason is None:
                        taint_reason = "live transcript disappeared before sealing"
            elif new_protected:
                if not final:
                    stop_for_quarantine()
                raise CampaignPlanError(
                    "protected evidence appeared without its exact workspace"
                )
            if (
                workspace is not None
                and workspace.name not in current_workspaces
            ):
                if not final:
                    stop_for_quarantine()
                raise CampaignPlanError("live workspace directory disappeared")
            if (
                protected_identity is not None
                and protected is not None
                and protected.name not in current_protected
            ):
                if not final:
                    stop_for_quarantine()
                raise CampaignPlanError(
                    "live protected directory disappeared"
                )
            if taint_reason is not None:
                returncode = stop_for_quarantine()
                # The legacy runner seals its ledger/WIP receipt while handling
                # SIGTERM.  Re-poll the final append-only state after it exits.
                if monitor is not None and transcript is not None:
                    terminal_findings = monitor.scan_transcript(
                        transcript, final=True
                    )
                    if any(
                        finding.code in UNQUIESCED_BOUNDARY_CODES
                        for finding in terminal_findings
                    ):
                        descendant_quiescence_unproven = True
                    if terminal_findings and taint_reason is None:
                        taint_reason = terminal_findings[0].describe()
                return GuardedChildResult(
                    int(returncode),
                    taint_reason,
                    workspace.name if workspace is not None else None,
                    transcript.name if transcript is not None else None,
                    workspace_identity,
                    protected_identity,
                    (
                        descendant_quiescence_unproven
                        and not detached_processes_proven_absent
                    ),
                    True,
                    detached_processes_proven_absent,
                )
            if final:
                returncode = finish_normal_exit()
                _revalidate_historical_control(item)
                if int(returncode) == 0 and any((
                    workspace is None,
                    monitor is None,
                    protected is None,
                    not os.path.lexists(protected)
                    if protected is not None else True,
                    not transcript_seen,
                    transcript is None,
                    workspace_identity is None,
                    protected_identity is None,
                )):
                    raise CampaignPlanError(
                        "successful dispatch lacks one complete live generation "
                        "and sealed transcript inventory"
                    )
                return GuardedChildResult(
                    int(returncode),
                    None,
                    workspace.name if workspace is not None else None,
                    transcript.name if transcript is not None else None,
                    workspace_identity,
                    protected_identity,
                    (
                        descendant_quiescence_unproven
                        and not detached_processes_proven_absent
                    ),
                    True,
                    detached_processes_proven_absent,
                )
            time.sleep(LIVE_BOUNDARY_POLL_SECONDS)
    except BaseException as failure:
        process_tree = process_tree_owner[0] or process_tree
        details = {
            "child_pid": (
                process_tree.pid if process_tree is not None else None
            ),
            "child_returncode": terminal_returncode,
            "workspace": workspace.name if workspace is not None else None,
            "protected": protected.name if protected is not None else None,
            "transcript": transcript.name if transcript is not None else None,
            "workspace_identity": workspace_identity,
            "protected_identity": protected_identity,
        }
        if isinstance(failure, UnquiescedChildError):
            failure.details.update(details)
            raise
        if process_tree is not None and not process_tree.sealed:
            try:
                stop_for_quarantine()
            except UnquiescedChildError as exc:
                exc.details.update(details)
                raise
        raise


def _checkpoint_reached(game: str) -> int:
    path = HERE / "agent_solutions" / f"{game}_legs" / "checkpoint.json"
    if not path.exists():
        return 0
    targets = _authoritative_targets()
    target = targets.get(game)
    if target is None:
        raise CampaignPlanError(
            f"game is absent from authoritative inventory: {game}"
        )
    try:
        checkpoint = Contiguous.load_trusted_checkpoint(
            path,
            expected_game=game,
            authoritative_target=target,
        )
    except Contiguous.SupervisorContractError as exc:
        raise CampaignPlanError(
            f"refusing malformed or untrusted checkpoint for {game}: {exc}"
        ) from exc
    return checkpoint.reached


def validate_live_policy_item(item: dict[str, Any]) -> None:
    """Reject a queue item whose exact frontier or retry row has gone stale."""

    game = item.get("game")
    target = item.get("target_level")
    if not isinstance(game, str) or not isinstance(target, int):
        raise CampaignPlanError("plan item has invalid game or target_level")
    artifact = HERE / "agent_solutions" / f"{game}_legs"
    if artifact.exists():
        boundary_reason = Legs.promoted_artifact_taint_reason(
            os.fspath(artifact)
        )
        if boundary_reason:
            raise CampaignPlanError(
                "canonical parent fails the current clean-room boundary: "
                f"{boundary_reason}"
            )
    report = Status.campaign_report(
        reserve=0,
        medium_headroom=1,
        high_headroom=1,
        max_runs=-1,
        max_tokens=-1,
    )
    matches = [
        row
        for row in report.get("frontiers", [])
        if row.get("game") == game and row.get("next_level") == target
    ]
    if len(matches) != 1:
        raise CampaignPlanError(
            "plan item is not the unique live exact frontier"
        )
    row = matches[0]
    for key in (
        *Status.FRONTIER_BINDING_FIELDS,
        "reached",
        "parent_action_count",
    ):
        if item.get(key) != row.get(key):
            raise CampaignPlanError(
                f"plan item exact-frontier field {key} is stale"
            )
    n = row.get("retry_complexity_n")
    if not isinstance(n, int) or isinstance(n, bool) or n < 0:
        raise CampaignPlanError(
            "live frontier has no valid retry coordinate"
        )
    if item.get("retry_complexity_n") != n:
        raise CampaignPlanError(
            "plan item retry coordinate is stale"
        )
    policy = Status.retry_policy(n)
    effective_wip, effective_dispatch = _effective_retry_inputs(item, policy)
    comparisons = {
        "effort": policy["effort"],
        "minutes": policy["minutes"],
        "wip_mode": effective_wip,
        "dispatch_mode": effective_dispatch,
        "recommended_auxiliary_parallelism": policy[
            "auxiliary_parallelism"
        ],
    }
    for key, expected in comparisons.items():
        if item.get(key) != expected:
            raise CampaignPlanError(
                f"plan item {key} is stale at the live frontier"
            )
    if item.get("warm_wip_available") != bool(
        row.get("warm_wip_available")
    ):
        raise CampaignPlanError(
            "plan item WIP availability is stale"
        )
    if item.get("warm_wip_validation") != row.get("warm_wip_validation"):
        raise CampaignPlanError(
            "plan item WIP boundary-policy validation is stale"
        )
    # A reset lane deliberately excludes the latest same-frontier WIP capsule,
    # so its plan carries no ``expected_wip_attempt`` selector even when an
    # eligible capsule exists.  Capsule identity is live policy state only for
    # a restore lane; requiring it for an exclude lane makes every reset after
    # a clean no-progress turn impossible to launch.  Availability, phase, and
    # infrastructure-recovery status remain live-checked in both modes.
    if effective_wip == "restore_clean_same_frontier":
        if item.get("expected_wip_attempt") != row.get("warm_wip_attempt"):
            raise CampaignPlanError(
                "plan item warm_wip_attempt is stale at the live frontier"
            )
    for key in ("warm_wip_phase", "warm_wip_recovery_required"):
        if item.get(key) != row.get(key):
            raise CampaignPlanError(
                f"plan item {key} is stale at the live frontier"
            )


def _taint_gate() -> None:
    proc = subprocess.run(
        [sys.executable, "arc/audit_submission_taint.py",
         "arc/crack_lab/agent_solutions"],
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    try:
        result = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise CampaignPlanError("taint gate returned non-JSON output") from exc
    if proc.returncode != 0 or result.get("automated_verdict") != "PASS":
        raise CampaignPlanError("post-turn taint gate failed; campaign stopped")
    # The standalone auditor deliberately keeps forensic WIP findings separate
    # from its canonical automated verdict.  A continuing campaign has a
    # stronger requirement: no hit may be eligible to enter a future clean
    # workspace.  Reject such evidence without deleting historical WIP.
    for category in ("successful_candidate_wip", "discarded_wip"):
        section = result.get(category)
        hits = section.get("hits") if isinstance(section, dict) else None
        if not isinstance(hits, list):
            raise CampaignPlanError(
                f"post-turn taint gate omitted the {category} hit ledger"
            )
        if hits:
            raise CampaignPlanError(
                "post-turn taint gate found forensic WIP taint; campaign "
                f"stopped without deleting it ({category}: {len(hits)} hit(s))"
            )
    for artifact in sorted((HERE / "agent_solutions").glob("*_legs")):
        reason = Legs.promoted_artifact_taint_reason(os.fspath(artifact))
        if reason:
            raise CampaignPlanError(
                "post-turn clean-room boundary failed; campaign stopped: "
                f"{artifact.name}: {reason}"
            )


def _refresh_solver_audits() -> None:
    """Refresh exact GKM checkpoints and the cross-system marginal comparator."""
    commands = [
        [
            sys.executable, "arc/audit_gkm_solved_checkpoints.py",
            "arc/crack_lab/agent_solutions",
            "--csv", "arc/audit_results/gkm-solved-checkpoints.csv",
            "--json", "arc/audit_results/gkm-solved-checkpoints.json",
        ],
        [
            sys.executable, "arc/audit_marginal_literal_reuse.py",
            "--reuse-non-gkm-from-json",
            "arc/audit_results/marginal-literal-reuse.json",
            "--json", "arc/audit_results/marginal-literal-reuse.json",
        ],
    ]
    for argv in commands:
        proc = subprocess.run(
            argv, cwd=REPO, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, check=False,
        )
        if proc.returncode != 0:
            raise CampaignPlanError(
                f"post-turn solver audit failed: {' '.join(argv)}\n{proc.stdout}"
            )


def _run_item_locked(
    plan: dict[str, Any], item: dict[str, Any], *, allowance: Guard.WeeklyAllowance
) -> dict[str, Any]:
    item = _project_runner_receipt(plan, item)
    argv = validate_item(item, plan)
    game = item.get("game")
    target = item.get("target_level")
    if not isinstance(game, str) or not isinstance(target, int):
        raise CampaignPlanError("plan item has invalid game or target_level")
    reached_before = _checkpoint_reached(game)
    validate_inventory_item(item, _authoritative_targets(), reached_before)
    # Reconcile a previously authorized release before the solved-target
    # short-circuit.  Otherwise a crash after the checkpoint became durable
    # could strand the final target's marker forever.
    resumed_zero_ledger = _resume_existing_zero_ledger_quarantine(item)
    if resumed_zero_ledger is not None:
        return resumed_zero_ledger
    _assert_no_dispatch_quarantine(item)
    if reached_before >= target:
        return {
            "game": game, "target_level": target, "result": "already_solved",
            "seed_mode": item["seed_mode"], "wip_mode": item["wip_mode"],
            "lineage_input_mode": item["lineage_input_mode"],
        }
    validate_live_policy_item(item)
    active_lock = active_workspace_lock(game)
    if active_lock is not None:
        raise CampaignPlanError(
            f"refusing duplicate active game lineage for {game}: {active_lock}"
        )
    admissible, reason = item_is_admissible(
        plan, item, now=time.time(), allowance=allowance
    )
    if not admissible:
        return {
            "game": game, "target_level": target,
            "result": "reserve_stop", "reason": reason,
            "seed_mode": item["seed_mode"], "wip_mode": item["wip_mode"],
            "lineage_input_mode": item["lineage_input_mode"],
        }
    lineage_probe = _acquire_scheduler_lineage_lock(item)
    try:
        _ensure_durable_wip_context_parent(item)
    finally:
        _release_scheduler_artifact_lock(lineage_probe)
    runner_cwd = _runner_cwd(item)
    ledger = _ledger_path(argv, cwd=runner_cwd)
    _ensure_durable_ledger_file(ledger)
    ledger_before = _capture_ledger_prefix(ledger)
    _assert_no_incomplete_taint_cleanup(ledger_before.records)
    wip_rollback_before = _capture_wip_rollback(item)
    canonical_rollback_before = _capture_canonical_rollback(item)
    _taint_gate()
    quarantine_owner: list[DispatchQuarantine | None] = [None]
    quarantine: DispatchQuarantine | None = None
    safe_terminal = False
    quarantine_failure_recorded = False
    try:
        _arm_dispatch_quarantine(
            item,
            ledger_before=ledger_before,
            wip_before=wip_rollback_before,
            canonical_before=canonical_rollback_before,
            ownership=quarantine_owner,
        )
        quarantine = quarantine_owner[0]
        if quarantine is None:
            raise CampaignPlanError(
                "dispatch quarantine ownership handoff was lost"
            )
        child = _run_guarded_child(
            item,
            argv,
            cwd=runner_cwd,
            env=_runner_env(item),
        )
        if (
            not child.process_tree_quiesced
            or child.descendant_quiescence_unproven
        ):
            raise UnquiescedChildError(
                "the exact runner lacks a complete scoped process-tree "
                "quiescence proof; preserving quarantine",
                details={
                    "child_returncode": child.returncode,
                    "workspace": child.workspace,
                    "protected": child.workspace,
                    "transcript": child.transcript,
                    "workspace_identity": child.workspace_identity,
                    "protected_identity": child.protected_identity,
                },
            )
        if _zero_ledger_suffix_is_exact(ledger_before):
            zero_marker_record, _workspace, _protected = (
                _seal_zero_ledger_observation(item, child)
            )
            _write_dispatch_quarantine_record(
                quarantine, zero_marker_record
            )
            quarantine_failure_recorded = True
            marker_payload = _read_bound_release_file_at(
                quarantine.root_fd,
                quarantine.name,
                quarantine.marker_identity,
                label="zero-ledger dispatch quarantine marker",
                maximum_bytes=RebootRecovery.MAX_MARKER_BYTES,
            )
            try:
                zero_parsed = RebootRecovery.parse_dispatch_marker(
                    marker_payload, require_recovery_arm=False
                )
            except RebootRecovery.RecoveryEvidenceError as exc:
                raise CampaignPlanError(str(exc)) from exc
            result = _complete_zero_ledger_recovery(
                item,
                marker=quarantine,
                parsed=zero_parsed,
                ledger_before=ledger_before,
                reached_before=reached_before,
                wip_before=wip_rollback_before,
                canonical_before=canonical_rollback_before,
                replayed=False,
                release_marker=False,
            )
            safe_terminal = True
            return result
        if child.returncode != 0 or child.taint_reason is not None:
            result = _recover_confirmed_taint(
                item,
                ledger=ledger,
                ledger_before=ledger_before,
                reached_before=reached_before,
                wip_rollback_before=wip_rollback_before,
                child_returncode=child.returncode,
                canonical_rollback_before=canonical_rollback_before,
                observed_workspace=child.workspace,
                observed_transcript=child.transcript,
                observed_workspace_identity=child.workspace_identity,
                observed_protected_identity=child.protected_identity,
                process_tree_quiesced=child.process_tree_quiesced,
                detached_processes_proven_absent=(
                    child.detached_processes_proven_absent
                ),
            )
            safe_terminal = True
            return result
        try:
            _authenticate_clean_generation(
                item,
                ledger=ledger,
                ledger_before=ledger_before,
                observed=child,
            )
        except ConfirmedGenerationTaint:
            result = _recover_confirmed_taint(
                item,
                ledger=ledger,
                ledger_before=ledger_before,
                reached_before=reached_before,
                wip_rollback_before=wip_rollback_before,
                child_returncode=child.returncode,
                canonical_rollback_before=canonical_rollback_before,
                observed_workspace=child.workspace,
                observed_transcript=child.transcript,
                observed_workspace_identity=child.workspace_identity,
                observed_protected_identity=child.protected_identity,
                process_tree_quiesced=child.process_tree_quiesced,
                detached_processes_proven_absent=(
                    child.detached_processes_proven_absent
                ),
            )
            safe_terminal = True
            return result
        _taint_gate()
        reached = _checkpoint_reached(game)
        if reached >= target:
            _refresh_solver_audits()
        result = {
            "game": game,
            "target_level": target,
            "reached": reached,
            "result": "solved" if reached >= target else "not_solved",
            "seed_mode": item["seed_mode"],
            "wip_mode": item["wip_mode"],
            "lineage_input_mode": item["lineage_input_mode"],
        }
        safe_terminal = True
        return result
    except UnquiescedChildError as exc:
        quarantine = quarantine_owner[0]
        # A surviving descendant could still race any filesystem restoration.
        # Preserve every artifact and require operator quarantine instead.
        observed_child = locals().get("child")
        if isinstance(observed_child, GuardedChildResult):
            exc.details.setdefault(
                "child_returncode", observed_child.returncode
            )
            exc.details.setdefault("workspace", observed_child.workspace)
            exc.details.setdefault("protected", observed_child.workspace)
            exc.details.setdefault("transcript", observed_child.transcript)
            exc.details.setdefault(
                "workspace_identity", observed_child.workspace_identity
            )
            exc.details.setdefault(
                "protected_identity", observed_child.protected_identity
            )
        recovery_ready = (
            set(exc.details).issubset({
                "child_pid",
                "child_returncode",
                "workspace",
                "protected",
                "transcript",
                "workspace_identity",
                "protected_identity",
            })
            and isinstance(exc.details.get("child_returncode"), int)
            and not isinstance(exc.details.get("child_returncode"), bool)
            and all(
                isinstance(exc.details.get(field), str)
                and bool(exc.details[field])
                for field in ("workspace", "protected", "transcript")
            )
            and all(
                isinstance(exc.details.get(field), tuple)
                and len(exc.details[field]) == 2
                and all(
                    isinstance(value, int) and not isinstance(value, bool)
                    for value in exc.details[field]
                )
                for field in ("workspace_identity", "protected_identity")
            )
            and (
                "child_pid" not in exc.details
                or (
                    isinstance(exc.details["child_pid"], int)
                    and not isinstance(exc.details["child_pid"], bool)
                    and exc.details["child_pid"] > 0
                )
            )
        )
        if quarantine is not None:
            try:
                _write_dispatch_quarantine_record(quarantine, {
                    "event": (
                        "dispatch_unquiesced"
                        if recovery_ready
                        else "dispatch_failed"
                    ),
                    "recorded_at": datetime.now(timezone.utc).isoformat(),
                    "exception_type": type(exc).__name__,
                    "reason": str(exc),
                    **exc.details,
                })
            except CampaignPlanError:
                pass
        raise
    except BaseException as exc:
        quarantine = quarantine_owner[0]
        if quarantine is None:
            # A failed arm owns no child or caller-visible marker.  The arm
            # routine has already durably removed its exact created inode, or
            # replaced this exception with a fail-closed cleanup error.
            raise
        if not quarantine_failure_recorded:
            observed_child = locals().get("child")
            observed_fields: dict[str, Any] = {}
            if isinstance(observed_child, GuardedChildResult):
                observed_fields = {
                    "child_returncode": observed_child.returncode,
                    "workspace": observed_child.workspace,
                    "protected": observed_child.workspace,
                    "transcript": observed_child.transcript,
                    "workspace_identity": observed_child.workspace_identity,
                    "protected_identity": observed_child.protected_identity,
                    "process_tree_quiesced": (
                        observed_child.process_tree_quiesced
                    ),
                    "descendant_quiescence_unproven": (
                        observed_child.descendant_quiescence_unproven
                    ),
                    "detached_processes_proven_absent": (
                        observed_child.detached_processes_proven_absent
                    ),
                }
            try:
                _write_dispatch_quarantine_record(quarantine, {
                    "event": "dispatch_failed",
                    "recorded_at": datetime.now(timezone.utc).isoformat(),
                    "exception_type": type(exc).__name__,
                    "reason": str(exc),
                    **observed_fields,
                })
            except CampaignPlanError:
                pass
        try:
            _rollback_control_failure_canonical(
                item,
                state=canonical_rollback_before,
                reached_before=reached_before,
            )
        except BaseException as rollback_exc:
            raise CampaignPlanError(
                "post-launch failure also prevented canonical rollback"
            ) from rollback_exc
        raise
    finally:
        quarantine = quarantine_owner[0]
        if quarantine is not None:
            if safe_terminal:
                terminal_result = locals().get("result")
                if not isinstance(terminal_result, dict):
                    raise CampaignPlanError(
                        "safe terminal lacks a release result"
                    )
                release_authority = _build_dispatch_release_authority(
                    item,
                    quarantine,
                    ledger,
                    terminal_result,
                    kind="ordinary_safe_terminal_v1",
                )
                _release_dispatch_quarantine(
                    quarantine, item, release_authority
                )
            else:
                _close_dispatch_quarantine(quarantine)


def _run_item(
    plan: dict[str, Any], item: dict[str, Any], *, allowance: Guard.WeeklyAllowance
) -> dict[str, Any]:
    """Run one item while holding the current scheduler's baseline lock."""

    projected = _project_runner_receipt(plan, item)
    validate_item(projected, plan)
    dispatch_lock = _acquire_scheduler_dispatch_lock(projected)
    try:
        return _run_item_locked(plan, projected, allowance=allowance)
    finally:
        _release_scheduler_artifact_lock(dispatch_lock)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--execute", action="store_true")
    actions.add_argument(
        "--arm-post-reboot-recovery",
        metavar="GAME",
        help=(
            "authenticate an unquiesced dispatch and seal the current kernel "
            "boot identity before reboot"
        ),
    )
    actions.add_argument(
        "--recover-post-reboot-quarantine",
        metavar="GAME",
        help=(
            "recover one pre-armed unquiesced dispatch after a different "
            "kernel boot session"
        ),
    )
    actions.add_argument(
        "--arm-sandboxed-generation-release",
        metavar="GAME",
        help=(
            "arm explicit same-boot artifact isolation for one exact "
            "Darwin sandbox generation"
        ),
    )
    actions.add_argument(
        "--recover-sandboxed-generation-release",
        metavar="GAME",
        help=(
            "restore and release one explicitly armed sandbox-isolated "
            "generation without claiming process quiescence"
        ),
    )
    parser.add_argument("--max-items", type=int, default=Policy.DEFAULT_MAX_RUNS)
    parser.add_argument("--calibration-only", action="store_true")
    parser.add_argument(
        "--confirm-dispatch-id",
        metavar="HEX",
        help="exact 128-bit dispatch ID printed in the quarantine marker",
    )
    parser.add_argument(
        "--confirm-recovery-nonce",
        metavar="HEX",
        help="exact 128-bit nonce printed by --arm-post-reboot-recovery",
    )
    parser.add_argument(
        "--confirm-current-wip-state-sha256",
        metavar="HEX",
        help=(
            "exact full current-WIP state hash printed by legacy recovery "
            "preflight; accepted only while arming"
        ),
    )
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    targets = _authoritative_targets()
    items = plan.get("initial_queue")
    if items is None:
        # Backward compatibility for plans generated before the adaptive queue
        # replaced the completed cold-L1 screen.
        items = plan.get("cold_screen_cohort")
    if not isinstance(items, list):
        raise CampaignPlanError("plan has no initial_queue list")
    if not all(isinstance(item, dict) for item in items):
        raise CampaignPlanError("plan initial_queue contains a non-object item")
    sandbox_arm_game = args.arm_sandboxed_generation_release
    sandbox_recovery_game = args.recover_sandboxed_generation_release
    sandbox_operator = (
        sandbox_arm_game is not None or sandbox_recovery_game is not None
    )
    items = [
        _project_runner_receipt(
            plan, item, allow_abandoned_scratch=sandbox_operator
        )
        for item in items
    ]
    arm_game = args.arm_post_reboot_recovery
    recovery_game = args.recover_post_reboot_quarantine
    operator_game = next((
        selected for selected in (
            arm_game, recovery_game, sandbox_arm_game, sandbox_recovery_game
        )
        if selected is not None
    ), None)
    if (
        operator_game is not None
        or args.confirm_dispatch_id is not None
        or args.confirm_recovery_nonce is not None
        or args.confirm_current_wip_state_sha256 is not None
    ):
        if operator_game is None or args.confirm_dispatch_id is None:
            raise CampaignPlanError(
                "operator recovery action requires one game and "
                "--confirm-dispatch-id"
            )
        if (
            arm_game is not None or sandbox_arm_game is not None
        ) and args.confirm_recovery_nonce is not None:
            raise CampaignPlanError(
                "recovery arm does not accept a recovery nonce"
            )
        if (
            (recovery_game is not None or sandbox_recovery_game is not None)
            and args.confirm_current_wip_state_sha256 is not None
        ):
            raise CampaignPlanError(
                "post-reboot recovery does not accept current-WIP confirmation"
            )
        if (
            args.confirm_current_wip_state_sha256 is not None
            and SHA256_RE.fullmatch(
                args.confirm_current_wip_state_sha256
            ) is None
        ):
            raise CampaignPlanError(
                "current-WIP confirmation must be one SHA-256 hex digest"
            )
        if (
            recovery_game is not None or sandbox_recovery_game is not None
        ) and args.confirm_recovery_nonce is None:
            raise CampaignPlanError(
                "recovery requires --confirm-recovery-nonce"
            )
        if args.calibration_only:
            raise CampaignPlanError(
                "operator recovery cannot be combined with calibration"
            )
        selected = [item for item in items if item.get("game") == operator_game]
        if len(selected) != 1:
            raise CampaignPlanError(
                "operator recovery requires exactly one matching plan item"
            )
        item = selected[0]
        validate_item(
            item,
            None if sandbox_operator else plan,
            allow_abandoned_scratch=sandbox_operator,
        )
        reached = _checkpoint_reached(operator_game)
        validate_inventory_item(item, targets, reached)
        if sandbox_arm_game is not None:
            outcome = _arm_sandboxed_generation_release(
                item,
                confirm_dispatch_id=args.confirm_dispatch_id,
            )
        elif sandbox_recovery_game is not None:
            assert args.confirm_recovery_nonce is not None
            outcome = _recover_sandboxed_generation_release(
                item,
                confirm_dispatch_id=args.confirm_dispatch_id,
                confirm_recovery_nonce=args.confirm_recovery_nonce,
            )
        elif arm_game is not None:
            outcome = _arm_post_reboot_recovery(
                item,
                confirm_dispatch_id=args.confirm_dispatch_id,
                confirm_current_wip_state_sha256=(
                    args.confirm_current_wip_state_sha256
                ),
            )
        else:
            assert args.confirm_recovery_nonce is not None
            outcome = _recover_post_reboot_quarantine(
                item,
                confirm_dispatch_id=args.confirm_dispatch_id,
                confirm_recovery_nonce=args.confirm_recovery_nonce,
            )
        print(json.dumps({"outcomes": [outcome]}, indent=2, sort_keys=True))
        return 0
    for item in items:
        argv = validate_item(item, plan)
        game = item.get("game")
        reached = _checkpoint_reached(game) if isinstance(game, str) else 0
        validate_inventory_item(item, targets, reached)
        # A dry run is the operator's review surface for the command that may
        # subsequently be executed.  It must therefore reject a queue frozen
        # at an older retry coordinate just as strictly as ``_run_item`` does;
        # printing a stale command as "DRY" is misleading even though the
        # execution path would later fail closed.
        validate_live_policy_item(item)
        print("DRY" if not args.execute else "QUEUE", item.get("game"), " ".join(argv))
    if not args.execute:
        print(
            "No model turn started; pass --execute only after reviewing the "
            "fresh policy-derived queue."
        )
        return 0

    outcomes = []
    for item in items:
        if len(outcomes) >= args.max_items:
            break
        allowance = Guard.weekly_allowance(Guard.query_rate_limits())
        outcome = _run_item(plan, item, allowance=allowance)
        outcomes.append(outcome)
        if outcome["result"] == "reserve_stop":
            print(json.dumps({"outcomes": outcomes}, indent=2, sort_keys=True))
            return 0

    while not args.calibration_only and len(outcomes) < args.max_items:
        snapshot = Guard.query_rate_limits()
        allowance = Guard.weekly_allowance(snapshot)
        report = Status.campaign_report(
            live_snapshot=snapshot,
            reserve=int(plan["reserve_percent"]),
            medium_headroom=5,
            high_headroom=6,
            max_runs=Policy.DEFAULT_MAX_RUNS,
            max_tokens=Policy.DEFAULT_MAX_TOKENS,
        )
        if not report["readiness"]["local_budget_ok"]:
            outcomes.append({
                "result": "local_budget_stop",
                "local_window": report["local_window"],
            })
            break
        item = Policy.adaptive_campaign_item(
            report, reserve=int(plan["reserve_percent"])
        )
        if item is None:
            outcomes.append({
                "result": "adaptation_stop",
                "reason": "matched evidence or remaining frontier unavailable",
            })
            break
        item = _project_runner_receipt(plan, item)
        print("ADAPT", item["game"], " ".join(validate_item(item, plan)))
        outcome = _run_item(plan, item, allowance=allowance)
        outcomes.append(outcome)
        if outcome["result"] == "reserve_stop":
            break
    print(json.dumps({"outcomes": outcomes}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
