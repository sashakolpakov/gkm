"""Strict evidence parsing for explicit post-reboot campaign recovery.

This module deliberately has no campaign mutation capabilities.  It parses the
append-only dispatch marker and obtains a kernel boot proof; the scheduler owns
all locks, identity rebinding, ledger writes, rollback, cleanup, and release.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import re
import signal
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Callable


DISPATCH_QUARANTINE_SCHEMA = "scheduler_dispatch_quarantine_v1"
DISPATCH_QUARANTINE_SCHEMA_V2 = "scheduler_dispatch_quarantine_v2"
DISPATCH_ARMED_SCHEMA_V2 = "scheduler_dispatch_armed_v2"
WIP_LOGICAL_RESTORE_SCHEMA_V1 = "scheduler_wip_logical_restore_v1"
WIP_LOGICAL_RESTORE_SCHEMA = "scheduler_wip_logical_restore_v2"
WIP_LOGICAL_RESTORE_SCHEMAS = frozenset({
    WIP_LOGICAL_RESTORE_SCHEMA_V1,
    WIP_LOGICAL_RESTORE_SCHEMA,
})
LEGACY_OPERATOR_RECOVERY_SCHEMA = "scheduler_post_reboot_recovery_v1"
OPERATOR_RECOVERY_SCHEMA = "scheduler_post_reboot_recovery_v2"
RECOVERY_ARM_SCHEMA_V2 = "scheduler_post_reboot_recovery_arm_v2"
SANDBOXED_GENERATION_ARM_SCHEMA = (
    "scheduler_sandboxed_generation_release_arm_v1"
)
SANDBOXED_GENERATION_EXEC_ARM_SCHEMA = (
    "scheduler_sandboxed_generation_release_arm_v2"
)
SANDBOXED_GENERATION_ARM_EVENT = (
    "sandboxed_generation_release_armed"
)
INTERRUPTED_GENERATION_ARM_SCHEMA = (
    "scheduler_interrupted_generation_release_arm_v1"
)
INTERRUPTED_GENERATION_TRANSITION_ARM_SCHEMA = (
    "scheduler_interrupted_generation_release_arm_v2"
)
INTERRUPTED_GENERATION_ARM_EVENT = (
    "interrupted_generation_release_armed"
)
CANONICAL_DIGEST_TRANSITION_SCHEMA = (
    "scheduler_audited_canonical_digest_transition_v1"
)
MAX_MARKER_BYTES = 1024 * 1024
SHA256_RE = re.compile(r"[0-9a-f]{64}")
DISPATCH_ID_RE = re.compile(r"[0-9a-f]{32}")
SAFE_COMPONENT_RE = re.compile(r"[A-Za-z0-9_.-]+")
SANDBOX_CONTRACT_SHA256 = hashlib.sha256(
    b"historical-gkm-codex-exec-ephemeral-strict-config-"
    b"sandbox-workspace-write-cd-exact-workspace/v1"
).hexdigest()
APPROVED_SANDBOXED_GENERATION_SOURCES = frozenset({
    "bb3474290d3411f980d53ffcee75be8234e634d478b1136677b9c6a93fe9ec64",
    "7455d304c96f5b070ecb4e62a45bcca21e4d5faf52027b8c3434dc094f7e7b0b",
    "18b5a3f1da18d10e9f7dba2c73b5d097abe691bd1b2cdfad3f3dcdf99d6a9fc0",
    "3bbd7ca93c9d74eef0b532ca8159283ce6d7fa81b6be316f0792a72ccd054398",
})
QUIESCED_INCOMPLETE_RUNNER_HEADS = {
    "bb3474290d3411f980d53ffcee75be8234e634d478b1136677b9c6a93fe9ec64": (
        "c1f8168f230732f2d745c234555b3e3dfcb8aefa"
    ),
    "7455d304c96f5b070ecb4e62a45bcca21e4d5faf52027b8c3434dc094f7e7b0b": (
        "246405c1cd903e1dcde9d3a4c6eed1ec93cf2c1f"
    ),
    "18b5a3f1da18d10e9f7dba2c73b5d097abe691bd1b2cdfad3f3dcdf99d6a9fc0": (
        "aa666cc3ff4c2167e12ce32b317bc3fe6c45a867"
    ),
    "3bbd7ca93c9d74eef0b532ca8159283ce6d7fa81b6be316f0792a72ccd054398": (
        "b37d0a0bece4c18da5cdc37f88f829e3a491fee9"
    ),
}
QUIESCED_INCOMPLETE_RUNNER_KEYS = frozenset({
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
BOOT_ID_RE = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-"
    r"[89ab][0-9a-f]{3}-[0-9a-f]{12}"
)


class RecoveryEvidenceError(ValueError):
    """Recovery evidence is missing, ambiguous, or not authoritative."""


@dataclass(frozen=True)
class BootIdentity:
    source: str
    value: str


@dataclass(frozen=True)
class ParsedMarker:
    dispatch_id: str
    armed: dict[str, object]
    unquiesced: dict[str, object]
    recovery_arm: dict[str, object] | None


@dataclass(frozen=True)
class AuditedCanonicalDigestTransition:
    """One removable, incident-bound legacy canonical migration authority."""

    transition_id: str
    dispatch_id: str
    game: str
    target_level: int
    reached: int
    parent_action_count: int
    retry_complexity_n: int
    projected_item_sha256: str
    runner_source_sha256: str
    runner_head_commit: str
    canonical_root_identity: tuple[int, int]
    canonical_digest: str
    observed_canonical_digest: str
    frontier_binding_schema: int
    parent_checkpoint_sha256: str
    parent_source_tree_sha256: str
    frontier_sha256: str


# Incident pins are local migration authority, never permanent scheduler
# policy.  Tests inject synthetic pins to exercise the fail-closed machinery;
# production ships with none after the audited migrations are retired.
AUDITED_CANONICAL_DIGEST_TRANSITIONS = MappingProxyType({})


def audited_canonical_digest_transition(
    armed: dict[str, object], observed_canonical_digest: object
) -> AuditedCanonicalDigestTransition | None:
    """Match every immutable coordinate of one approved legacy transition."""

    dispatch_id = armed.get("dispatch_id")
    if not isinstance(dispatch_id, str):
        return None
    pin = AUDITED_CANONICAL_DIGEST_TRANSITIONS.get(dispatch_id)
    historical = armed.get("historical_runner")
    frontier = armed.get("frontier_binding")
    if (
        pin is None
        or not isinstance(historical, dict)
        or not isinstance(frontier, dict)
        or observed_canonical_digest != pin.observed_canonical_digest
    ):
        return None
    expected_armed = {
        "dispatch_id": pin.dispatch_id,
        "game": pin.game,
        "target_level": pin.target_level,
        "retry_complexity_n": pin.retry_complexity_n,
        "projected_item_sha256": pin.projected_item_sha256,
        "canonical_root_identity": list(pin.canonical_root_identity),
        "canonical_digest": pin.canonical_digest,
    }
    expected_runner = {
        "source_sha256": pin.runner_source_sha256,
        "head_commit": pin.runner_head_commit,
    }
    expected_frontier = {
        "game": pin.game,
        "target_level": pin.target_level,
        "reached": pin.reached,
        "parent_action_count": pin.parent_action_count,
        "frontier_binding_schema": pin.frontier_binding_schema,
        "parent_checkpoint_sha256": pin.parent_checkpoint_sha256,
        "parent_source_tree_sha256": pin.parent_source_tree_sha256,
        "frontier_sha256": pin.frontier_sha256,
    }
    if any(
        armed.get(field) != expected
        for field, expected in expected_armed.items()
    ) or any(
        historical.get(field) != expected
        for field, expected in expected_runner.items()
    ) or any(
        frontier.get(field) != expected
        for field, expected in expected_frontier.items()
    ):
        return None
    return pin


ARMED_V1_KEYS = frozenset({
    "schema",
    "dispatch_id",
    "event",
    "recorded_at",
    "pid",
    "game",
    "target_level",
    "tag",
    "run_label",
    "retry_complexity_n",
    "artifact_root",
    "artifact_root_identity",
    "canonical_root",
    "canonical_root_identity",
    "canonical_digest",
    "frontier_binding",
    "target_wip_snapshot",
    "ledger",
    "ledger_parent_identity",
    "ledger_file_identity",
    "ledger_prefix_bytes",
    "ledger_prefix_sha256",
    "projected_item_sha256",
    "historical_runner",
})
ARMED_V2_KEYS = ARMED_V1_KEYS | frozenset({
    "armed_schema",
    "wip_rollback_capsule_name",
    "wip_rollback_capsule_identity",
    "wip_rollback_capsule_bytes",
    "wip_rollback_capsule_sha256",
    "wip_rollback_capsule_state_sha256",
    "wip_restore_logical_state_schema",
    "wip_restore_logical_state_sha256",
})
UNQUIESCED_REQUIRED_KEYS = frozenset({
    "schema",
    "dispatch_id",
    "event",
    "recorded_at",
    "exception_type",
    "reason",
    "child_returncode",
    "workspace",
    "protected",
    "transcript",
    "workspace_identity",
    "protected_identity",
})
UNQUIESCED_OPTIONAL_KEYS = frozenset({"child_pid"})
UNQUIESCED_V2_OPTIONAL_KEYS = UNQUIESCED_OPTIONAL_KEYS | frozenset({
    "boundary_finding_counts",
})
BOUNDARY_FINDING_COUNT_CODES = frozenset({
    "detached_process_escape",
    "shell_or_subprocess_escape",
    "dynamic_execution",
})
ZERO_LEDGER_REQUIRED_KEYS = frozenset({
    "schema",
    "dispatch_id",
    "event",
    "recorded_at",
    "exception_type",
    "reason",
    "child_returncode",
    "workspace",
    "protected",
    "transcript",
    "workspace_identity",
    "protected_identity",
    "process_tree_quiesced",
    "descendant_quiescence_unproven",
    "detached_processes_proven_absent",
    "ledger_suffix_rows",
    "evidence_schema",
    "protected_transcript_sha256",
    "workspace_lock_schema",
    "workspace_lock_path",
    "workspace_lock_identity",
})
ZERO_LEDGER_DIAGNOSTICS_KEYS = frozenset({
    "diagnostics", "protected_diagnostics_sha256",
})
QUIESCED_INCOMPLETE_EVIDENCE_REASON = (
    "zero-ledger recovery lacks one complete quiesced observation"
)
QUIESCED_INCOMPLETE_EVIDENCE_FAILED_KEYS = frozenset({
    "schema",
    "dispatch_id",
    "event",
    "recorded_at",
    "exception_type",
    "reason",
    "child_returncode",
    "workspace",
    "protected",
    "transcript",
    "workspace_identity",
    "protected_identity",
    "process_tree_quiesced",
    "descendant_quiescence_unproven",
    "detached_processes_proven_absent",
    "normal_exit_left_captured_descendants",
})
RECOVERY_ARM_V1_KEYS = frozenset({
    "schema",
    "dispatch_id",
    "event",
    "recorded_at",
    "recovery_nonce",
    "boot_identity_source",
    "boot_identity",
    "marker_root_identity",
    "pre_arm_marker_identity",
    "armed_marker_identity",
    "pre_arm_marker_bytes",
    "pre_arm_marker_sha256",
    "projected_item_sha256",
    "exec_record_sha256",
    "canonical_root_metadata",
    "wip_state_sha256",
    "workspace_lock_schema",
    "workspace_lock_path",
    "workspace_lock_identity",
})
RECOVERY_ARM_V2_KEYS = RECOVERY_ARM_V1_KEYS | frozenset({
    "recovery_arm_schema",
    "wip_recovery_authority",
    "historical_wip_snapshot",
    "confirmed_current_wip_state_sha256",
    "wip_disposition",
    "discard_survivor_sha256",
    "restored_wip_logical_state_sha256",
})

# This is deliberately a separate authority from post-reboot containment.
# It accepts the exact failure row emitted when Darwin could not prove a
# detached process terminal, but only so a caller can retire a generation
# whose writable namespace is independently confined.  It never upgrades the
# row to ``dispatch_unquiesced`` and never asserts descendant absence.
SANDBOXED_FAILED_KEYS = UNQUIESCED_REQUIRED_KEYS | frozenset({"child_pid"})
SANDBOXED_GENERATION_ARM_KEYS = frozenset({
    "schema",
    "dispatch_id",
    "event",
    "recorded_at",
    "recovery_arm_schema",
    "recovery_nonce",
    "boot_identity_source",
    "boot_identity",
    "marker_root_identity",
    "pre_arm_marker_identity",
    "armed_marker_identity",
    "pre_arm_marker_bytes",
    "pre_arm_marker_sha256",
    "projected_item_sha256",
    "historical_runner_sha256",
    "authority_kind",
    "operator_provenance_assumption",
    "sandbox_contract_sha256",
    "process_claim",
    "process_tree_quiesced",
    "detached_processes_proven_absent",
    "isolation_claim",
    "scheduler_pid",
    "child_pid",
    "child_pgid",
    "absence_sample_count",
    "absence_window_ns",
    "absence_first_at",
    "absence_last_at",
    "scratch_root",
    "scratch_root_identity",
    "scratch_root_disposition",
    "required_retry_scratch_relation",
    "workspace",
    "workspace_identity",
    "workspace_tree_observation_sha256",
    "protected",
    "protected_identity",
    "protected_tree_sha256",
    "workspace_lock_schema",
    "workspace_lock_path",
    "workspace_lock_identity",
    "canonical_digest",
    "ledger_prefix_bytes",
    "ledger_prefix_sha256",
    "wip_state_sha256",
    "wip_rollback_capsule_name",
    "wip_rollback_capsule_identity",
    "wip_rollback_capsule_bytes",
    "wip_rollback_capsule_sha256",
    "wip_rollback_capsule_state_sha256",
    "wip_restore_logical_state_schema",
    "wip_restore_logical_state_sha256",
    "operator_artifact_scanner_assumption",
})
SANDBOXED_GENERATION_EXEC_ARM_KEYS = (
    SANDBOXED_GENERATION_ARM_KEYS | frozenset({"exec_record_sha256"})
)
INTERRUPTED_GENERATION_ARM_KEYS = (
    SANDBOXED_GENERATION_EXEC_ARM_KEYS
    - frozenset({"child_pid", "child_pgid"})
)
INTERRUPTED_GENERATION_TRANSITION_ARM_KEYS = (
    INTERRUPTED_GENERATION_ARM_KEYS | frozenset({
        "observed_canonical_digest",
        "canonical_digest_transition_schema",
        "canonical_digest_transition_id",
    })
)


def _reject_constant(value: str) -> None:
    raise RecoveryEvidenceError(f"non-standard JSON constant is forbidden: {value}")


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise RecoveryEvidenceError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def canonical_json_line(record: dict[str, object]) -> bytes:
    return json.dumps(
        record, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8") + b"\n"


def parse_canonical_jsonl(raw: bytes, *, label: str) -> list[dict[str, object]]:
    if raw and not raw.endswith(b"\n"):
        raise RecoveryEvidenceError(f"{label} lacks a final line boundary")
    records: list[dict[str, object]] = []
    for index, line in enumerate(raw.splitlines(keepends=True), 1):
        if line == b"\n":
            raise RecoveryEvidenceError(f"{label} contains a blank row")
        try:
            text = line[:-1].decode("utf-8")
            record = json.loads(
                text,
                object_pairs_hook=_unique_object,
                parse_constant=_reject_constant,
            )
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise RecoveryEvidenceError(
                f"{label} row {index} is malformed"
            ) from exc
        if not isinstance(record, dict):
            raise RecoveryEvidenceError(f"{label} row {index} is not an object")
        if line != canonical_json_line(record):
            raise RecoveryEvidenceError(
                f"{label} row {index} is not canonically encoded"
            )
        records.append(record)
    return records


def _utc_timestamp(value: object, label: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise RecoveryEvidenceError(f"{label} timestamp is malformed")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise RecoveryEvidenceError(f"{label} timestamp is malformed") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise RecoveryEvidenceError(f"{label} timestamp is not timezone-aware")
    return parsed.astimezone(timezone.utc)


def _identity(value: object, label: str) -> tuple[int, int]:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(
            not isinstance(part, int) or isinstance(part, bool) or part < 0
            for part in value
        )
    ):
        raise RecoveryEvidenceError(f"{label} identity is malformed")
    return value[0], value[1]


def _normalized_absolute(value: object, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise RecoveryEvidenceError(f"{label} path is malformed")
    path = Path(value)
    if not path.is_absolute() or Path(os.path.abspath(value)) != path:
        raise RecoveryEvidenceError(f"{label} path is not normalized absolute")
    return path


def _nonnegative_integer(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise RecoveryEvidenceError(f"{label} is malformed")
    return value


def validate_boundary_finding_counts(value: object) -> dict[str, int]:
    """Return one description-free count map over the fixed boundary enum."""

    if (
        not isinstance(value, dict)
        or not value
        or not set(value).issubset(BOUNDARY_FINDING_COUNT_CODES)
        or any(
            not isinstance(count, int)
            or isinstance(count, bool)
            or count <= 0
            for count in value.values()
        )
    ):
        raise RecoveryEvidenceError(
            "dispatch boundary finding counts are malformed"
        )
    return dict(value)


def parse_dispatch_marker(
    raw: bytes, *, require_recovery_arm: bool | None = None
) -> ParsedMarker:
    """Parse an exact two-row pre-arm or three-row armed marker."""

    if not raw or len(raw) > MAX_MARKER_BYTES or not raw.endswith(b"\n"):
        raise RecoveryEvidenceError("dispatch marker framing is malformed")
    rows = parse_canonical_jsonl(raw, label="dispatch marker")
    expected_lengths = (
        {3} if require_recovery_arm is True
        else {2} if require_recovery_arm is False
        else {2, 3}
    )
    if len(rows) not in expected_lengths:
        raise RecoveryEvidenceError(
            "dispatch marker has an invalid recovery phase row count"
        )
    armed, unquiesced = rows[:2]
    armed_keys = set(armed)
    armed_v1 = armed_keys == ARMED_V1_KEYS
    armed_v2 = armed_keys == ARMED_V2_KEYS
    if not (armed_v1 or armed_v2) or armed.get(
        "event"
    ) != "dispatch_armed":
        raise RecoveryEvidenceError("dispatch armed row has an invalid schema")
    allowed_unquiesced_keys = UNQUIESCED_REQUIRED_KEYS | (
        UNQUIESCED_V2_OPTIONAL_KEYS
        if armed_v2 else UNQUIESCED_OPTIONAL_KEYS
    )
    unquiesced_phase = (
        UNQUIESCED_REQUIRED_KEYS.issubset(unquiesced)
        and set(unquiesced).issubset(allowed_unquiesced_keys)
        and unquiesced.get("event") == "dispatch_unquiesced"
    )
    zero_keys = set(unquiesced)
    zero_phase = (
        ZERO_LEDGER_REQUIRED_KEYS.issubset(zero_keys)
        and (
            zero_keys == ZERO_LEDGER_REQUIRED_KEYS
            or zero_keys
            == ZERO_LEDGER_REQUIRED_KEYS | ZERO_LEDGER_DIAGNOSTICS_KEYS
        )
        and unquiesced.get("event") == "dispatch_zero_ledger_quarantined"
    )
    if not (unquiesced_phase or zero_phase):
        raise RecoveryEvidenceError(
            "dispatch terminal quarantine row has an invalid schema"
        )
    dispatch_id = armed.get("dispatch_id")
    if (
        not isinstance(dispatch_id, str)
        or DISPATCH_ID_RE.fullmatch(dispatch_id) is None
        or unquiesced.get("dispatch_id") != dispatch_id
        or armed.get("schema") not in {
            DISPATCH_QUARANTINE_SCHEMA,
            DISPATCH_QUARANTINE_SCHEMA_V2,
        }
        or unquiesced.get("schema") != armed.get("schema")
        or armed_v1
        != (armed.get("schema") == DISPATCH_QUARANTINE_SCHEMA)
        or armed_v2
        != (armed.get("schema") == DISPATCH_QUARANTINE_SCHEMA_V2)
    ):
        raise RecoveryEvidenceError("dispatch marker binding is malformed")
    if armed_v2:
        if armed.get("armed_schema") != DISPATCH_ARMED_SCHEMA_V2:
            raise RecoveryEvidenceError("dispatch v2 armed schema is malformed")
        capsule_name = armed.get("wip_rollback_capsule_name")
        if (
            not isinstance(capsule_name, str)
            or not capsule_name
            or Path(capsule_name).name != capsule_name
        ):
            raise RecoveryEvidenceError("dispatch WIP capsule name is malformed")
        _identity(
            armed.get("wip_rollback_capsule_identity"), "WIP capsule"
        )
        _nonnegative_integer(
            armed.get("wip_rollback_capsule_bytes"), "WIP capsule bytes"
        )
        for field in (
            "wip_rollback_capsule_sha256",
            "wip_rollback_capsule_state_sha256",
            "wip_restore_logical_state_sha256",
        ):
            value = armed.get(field)
            if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
                raise RecoveryEvidenceError(
                    f"dispatch armed {field} is malformed"
                )
        if armed.get("wip_restore_logical_state_schema") not in (
            WIP_LOGICAL_RESTORE_SCHEMAS
        ):
            raise RecoveryEvidenceError(
                "dispatch WIP logical restore schema is malformed"
            )
    armed_at = _utc_timestamp(armed.get("recorded_at"), "dispatch armed")
    unquiesced_at = _utc_timestamp(
        unquiesced.get("recorded_at"), "dispatch unquiesced"
    )
    if unquiesced_at < armed_at:
        raise RecoveryEvidenceError("dispatch marker timestamps are reversed")
    for field in ("pid", "target_level", "retry_complexity_n"):
        _nonnegative_integer(armed.get(field), field)
    if armed["pid"] == 0 or armed["target_level"] == 0:
        raise RecoveryEvidenceError("dispatch armed row has an invalid process/level")
    for field in ("game", "tag", "run_label"):
        if not isinstance(armed.get(field), str) or not armed[field]:
            raise RecoveryEvidenceError(f"dispatch armed {field} is malformed")
    _normalized_absolute(armed.get("artifact_root"), "artifact root")
    _normalized_absolute(armed.get("canonical_root"), "canonical root")
    _normalized_absolute(armed.get("ledger"), "ledger")
    for field in (
        "artifact_root_identity",
        "canonical_root_identity",
        "ledger_parent_identity",
    ):
        _identity(armed.get(field), field)
    ledger_identity = armed.get("ledger_file_identity")
    if ledger_identity is not None:
        _identity(ledger_identity, "ledger file")
    for field in (
        "canonical_digest",
        "ledger_prefix_sha256",
        "projected_item_sha256",
    ):
        value = armed.get(field)
        if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
            raise RecoveryEvidenceError(f"dispatch armed {field} is malformed")
    prefix_bytes = _nonnegative_integer(
        armed.get("ledger_prefix_bytes"), "ledger prefix bytes"
    )
    if ledger_identity is None and prefix_bytes != 0:
        raise RecoveryEvidenceError("absent ledger has a nonempty prefix")
    if not isinstance(armed.get("frontier_binding"), dict):
        raise RecoveryEvidenceError("dispatch frontier binding is malformed")
    wip = armed.get("target_wip_snapshot")
    if (
        not isinstance(wip, list)
        or len(wip) != 2
        or not isinstance(wip[0], str)
        or not Path(wip[0]).is_absolute()
        or (wip[1] is not None and (
            not isinstance(wip[1], str) or SHA256_RE.fullmatch(wip[1]) is None
        ))
    ):
        raise RecoveryEvidenceError("dispatch WIP snapshot is malformed")
    if armed.get("historical_runner") is not None and not isinstance(
        armed.get("historical_runner"), dict
    ):
        raise RecoveryEvidenceError("dispatch runner receipt is malformed")
    for field in ("exception_type", "reason", "workspace", "protected", "transcript"):
        if not isinstance(unquiesced.get(field), str) or not unquiesced[field]:
            raise RecoveryEvidenceError(
                f"dispatch unquiesced {field} is malformed"
            )
    returncode = unquiesced.get("child_returncode")
    if not isinstance(returncode, int) or isinstance(returncode, bool):
        raise RecoveryEvidenceError("dispatch child return code is malformed")
    _identity(unquiesced.get("workspace_identity"), "workspace")
    _identity(unquiesced.get("protected_identity"), "protected")
    if unquiesced_phase and "child_pid" in unquiesced:
        child_pid = _nonnegative_integer(unquiesced["child_pid"], "child pid")
        if child_pid == 0:
            raise RecoveryEvidenceError("dispatch child pid is malformed")
    if "boundary_finding_counts" in unquiesced:
        if not armed_v2:
            raise RecoveryEvidenceError(
                "dispatch v1 cannot carry boundary finding counts"
            )
        validate_boundary_finding_counts(
            unquiesced["boundary_finding_counts"]
        )
    if zero_phase:
        if (
            armed_v2 is not True
            or unquiesced.get("process_tree_quiesced") is not True
            or unquiesced.get("descendant_quiescence_unproven") is not False
            or not isinstance(
                unquiesced.get("detached_processes_proven_absent"), bool
            )
            or unquiesced.get("ledger_suffix_rows") != 0
            or unquiesced.get("evidence_schema") not in {
                "sealed_transcript_only_v1",
                "sealed_transcript_diagnostics_v1",
            }
            or unquiesced.get("workspace_lock_schema") not in {
                "in_workspace_v1", "hashed_external_v1",
            }
        ):
            raise RecoveryEvidenceError(
                "zero-ledger quarantine proof is malformed"
            )
        _normalized_absolute(
            unquiesced.get("workspace_lock_path"), "workspace lock"
        )
        _identity(unquiesced.get("workspace_lock_identity"), "workspace lock")
        for field in (
            "protected_transcript_sha256",
            "protected_diagnostics_sha256",
        ):
            if field not in unquiesced:
                continue
            value = unquiesced[field]
            if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
                raise RecoveryEvidenceError(
                    f"zero-ledger quarantine {field} is malformed"
                )
        diagnostics_schema = (
            unquiesced.get("evidence_schema")
            == "sealed_transcript_diagnostics_v1"
        )
        if diagnostics_schema != ZERO_LEDGER_DIAGNOSTICS_KEYS.issubset(
            unquiesced
        ):
            raise RecoveryEvidenceError(
                "zero-ledger quarantine diagnostics binding is malformed"
            )
        if diagnostics_schema and (
            not isinstance(unquiesced.get("diagnostics"), str)
            or not unquiesced["diagnostics"]
        ):
            raise RecoveryEvidenceError(
                "zero-ledger quarantine diagnostics name is malformed"
            )
    recovery_arm = rows[2] if len(rows) == 3 else None
    if zero_phase and recovery_arm is not None:
        raise RecoveryEvidenceError(
            "zero-ledger quarantine cannot carry a reboot recovery arm"
        )
    if recovery_arm is not None:
        arm_keys = set(recovery_arm)
        is_v1 = arm_keys == RECOVERY_ARM_V1_KEYS
        is_v2 = arm_keys == RECOVERY_ARM_V2_KEYS
        if (
            not (is_v1 or is_v2)
            or recovery_arm.get("event") != "post_reboot_recovery_armed"
            or recovery_arm.get("schema") != armed.get("schema")
            or recovery_arm.get("dispatch_id") != dispatch_id
        ):
            raise RecoveryEvidenceError(
                "post-reboot recovery arm row has an invalid schema"
            )
        if is_v2:
            authority = recovery_arm.get("wip_recovery_authority")
            disposition = recovery_arm.get("wip_disposition")
            if (
                recovery_arm.get("recovery_arm_schema")
                != RECOVERY_ARM_SCHEMA_V2
                or authority not in {
                    "operator_confirmed_quarantined_wip_v1",
                    "dispatch_full_wip_rollback_capsule_v1",
                }
                or disposition not in {
                    "discard_latest_pointer",
                    "confirmed_latest_absent",
                    "restore_historical_baseline",
                }
            ):
                raise RecoveryEvidenceError(
                    "post-reboot recovery WIP authority is malformed"
                )
            historical_wip = recovery_arm.get("historical_wip_snapshot")
            if historical_wip != armed.get("target_wip_snapshot"):
                raise RecoveryEvidenceError(
                    "recovery arm historical WIP snapshot changed"
                )
            confirmed_wip = recovery_arm.get(
                "confirmed_current_wip_state_sha256"
            )
            if (
                not isinstance(confirmed_wip, str)
                or SHA256_RE.fullmatch(confirmed_wip) is None
                or confirmed_wip != recovery_arm.get("wip_state_sha256")
            ):
                raise RecoveryEvidenceError(
                    "recovery arm confirmed WIP hash is malformed"
                )
            survivor = recovery_arm.get("discard_survivor_sha256")
            restored_logical = recovery_arm.get(
                "restored_wip_logical_state_sha256"
            )
            if (
                authority == "dispatch_full_wip_rollback_capsule_v1"
            ) != (disposition == "restore_historical_baseline"):
                raise RecoveryEvidenceError(
                    "recovery arm capsule authority/disposition disagree"
                )
            if disposition == "discard_latest_pointer":
                if (
                    authority != "operator_confirmed_quarantined_wip_v1"
                    or not isinstance(survivor, str)
                    or SHA256_RE.fullmatch(survivor) is None
                ):
                    raise RecoveryEvidenceError(
                        "recovery arm discard binding is malformed"
                    )
            elif survivor is not None:
                raise RecoveryEvidenceError(
                    "preserved recovery arm has a discard binding"
                )
            if disposition == "restore_historical_baseline":
                if restored_logical != armed.get(
                    "wip_restore_logical_state_sha256"
                ):
                    raise RecoveryEvidenceError(
                        "recovery arm logical restore seal changed"
                    )
            elif restored_logical is not None:
                raise RecoveryEvidenceError(
                    "legacy recovery arm has a logical restore seal"
                )
        _utc_timestamp(recovery_arm.get("recorded_at"), "recovery arm")
        nonce = recovery_arm.get("recovery_nonce")
        if not isinstance(nonce, str) or DISPATCH_ID_RE.fullmatch(nonce) is None:
            raise RecoveryEvidenceError("recovery arm nonce is malformed")
        _validate_boot_identity_fields(
            recovery_arm.get("boot_identity_source"),
            recovery_arm.get("boot_identity"),
        )
        _identity(recovery_arm.get("marker_root_identity"), "marker root")
        _identity(
            recovery_arm.get("pre_arm_marker_identity"), "pre-arm marker"
        )
        _identity(
            recovery_arm.get("armed_marker_identity"), "armed marker"
        )
        if recovery_arm.get("pre_arm_marker_identity") == (
            recovery_arm.get("armed_marker_identity")
        ):
            raise RecoveryEvidenceError(
                "recovery arm marker identities are not distinct"
            )
        pre_arm_bytes = _nonnegative_integer(
            recovery_arm.get("pre_arm_marker_bytes"), "pre-arm marker bytes"
        )
        pre_arm_raw = b"".join(
            canonical_json_line(row)
            for row in rows[:2]
        )
        # Every scheduler marker row is written in this canonical encoding.
        if (
            pre_arm_bytes != len(pre_arm_raw)
            or recovery_arm.get("pre_arm_marker_sha256")
            != hashlib.sha256(pre_arm_raw).hexdigest()
        ):
            raise RecoveryEvidenceError(
                "recovery arm does not seal the two-row marker prefix"
            )
        for field in ("projected_item_sha256", "exec_record_sha256"):
            value = recovery_arm.get(field)
            if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
                raise RecoveryEvidenceError(f"recovery arm {field} is malformed")
        root_metadata = recovery_arm.get("canonical_root_metadata")
        if not isinstance(root_metadata, dict) or set(root_metadata) != {
            "identity", "mode", "uid", "gid", "mtime_ns", "xattrs_sha256"
        }:
            raise RecoveryEvidenceError(
                "recovery arm canonical root metadata is malformed"
            )
        _identity(root_metadata.get("identity"), "armed canonical root")
        for field in ("mode", "uid", "gid", "mtime_ns"):
            _nonnegative_integer(root_metadata.get(field), f"canonical {field}")
        for field in ("xattrs_sha256",):
            value = root_metadata.get(field)
            if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
                raise RecoveryEvidenceError(
                    "recovery arm canonical metadata hash is malformed"
                )
        wip_state_sha = recovery_arm.get("wip_state_sha256")
        if (
            not isinstance(wip_state_sha, str)
            or SHA256_RE.fullmatch(wip_state_sha) is None
        ):
            raise RecoveryEvidenceError("recovery arm WIP state hash is malformed")
        if recovery_arm.get("workspace_lock_schema") not in {
            "in_workspace_v1", "hashed_external_v1"
        }:
            raise RecoveryEvidenceError("recovery arm lock schema is malformed")
        _normalized_absolute(
            recovery_arm.get("workspace_lock_path"), "workspace lock"
        )
        _identity(recovery_arm.get("workspace_lock_identity"), "workspace lock")
    return ParsedMarker(dispatch_id, armed, unquiesced, recovery_arm)


def parse_quiesced_incomplete_evidence_marker(
    raw: bytes, *, require_recovery_arm: bool | None = None
) -> ParsedMarker:
    """Parse only the exact v2 quiesced/incomplete-evidence incident.

    This is deliberately separate from both ordinary zero-ledger recovery and
    sandboxed nonquiescent release.  The failed row proves that the guarded
    child's captured process tree was quiesced, but it carries no protected
    transcript authority and must never be upgraded into one.
    """

    if require_recovery_arm is True:
        raise RecoveryEvidenceError(
            "quiesced incomplete-evidence recovery has no arm phase"
        )
    if not raw or len(raw) > MAX_MARKER_BYTES or not raw.endswith(b"\n"):
        raise RecoveryEvidenceError(
            "quiesced incomplete-evidence marker framing is malformed"
        )
    rows = parse_canonical_jsonl(
        raw, label="quiesced incomplete-evidence marker"
    )
    if len(rows) != 2:
        raise RecoveryEvidenceError(
            "quiesced incomplete-evidence marker must have exactly two rows"
        )
    armed, failed = rows
    if (
        set(armed) != ARMED_V2_KEYS
        or armed.get("schema") != DISPATCH_QUARANTINE_SCHEMA_V2
        or armed.get("armed_schema") != DISPATCH_ARMED_SCHEMA_V2
        or armed.get("event") != "dispatch_armed"
    ):
        raise RecoveryEvidenceError(
            "quiesced incomplete-evidence armed row is not exact v2"
        )
    dispatch_id = armed.get("dispatch_id")
    if (
        not isinstance(dispatch_id, str)
        or DISPATCH_ID_RE.fullmatch(dispatch_id) is None
        or failed.get("dispatch_id") != dispatch_id
        or failed.get("schema") != armed.get("schema")
    ):
        raise RecoveryEvidenceError(
            "quiesced incomplete-evidence marker binding is malformed"
        )

    capsule_name = armed.get("wip_rollback_capsule_name")
    if (
        not isinstance(capsule_name, str)
        or not capsule_name
        or Path(capsule_name).name != capsule_name
    ):
        raise RecoveryEvidenceError("dispatch WIP capsule name is malformed")
    _identity(armed.get("wip_rollback_capsule_identity"), "WIP capsule")
    _nonnegative_integer(
        armed.get("wip_rollback_capsule_bytes"), "WIP capsule bytes"
    )
    for field in (
        "wip_rollback_capsule_sha256",
        "wip_rollback_capsule_state_sha256",
        "wip_restore_logical_state_sha256",
    ):
        value = armed.get(field)
        if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
            raise RecoveryEvidenceError(f"dispatch armed {field} is malformed")
    if armed.get("wip_restore_logical_state_schema") not in (
        WIP_LOGICAL_RESTORE_SCHEMAS
    ):
        raise RecoveryEvidenceError(
            "dispatch WIP logical restore schema is malformed"
        )
    armed_at = _utc_timestamp(armed.get("recorded_at"), "dispatch armed")
    failed_at = _utc_timestamp(
        failed.get("recorded_at"), "quiesced incomplete-evidence failure"
    )
    if failed_at < armed_at:
        raise RecoveryEvidenceError("dispatch marker timestamps are reversed")
    for field in ("pid", "target_level", "retry_complexity_n"):
        _nonnegative_integer(armed.get(field), field)
    if armed["pid"] <= 1 or armed["target_level"] == 0:
        raise RecoveryEvidenceError(
            "dispatch armed row has an invalid process/level"
        )
    for field in ("game", "tag", "run_label"):
        if not isinstance(armed.get(field), str) or not armed[field]:
            raise RecoveryEvidenceError(f"dispatch armed {field} is malformed")
    _normalized_absolute(armed.get("artifact_root"), "artifact root")
    _normalized_absolute(armed.get("canonical_root"), "canonical root")
    _normalized_absolute(armed.get("ledger"), "ledger")
    for field in (
        "artifact_root_identity",
        "canonical_root_identity",
        "ledger_parent_identity",
    ):
        _identity(armed.get(field), field)
    ledger_identity = armed.get("ledger_file_identity")
    if ledger_identity is not None:
        _identity(ledger_identity, "ledger file")
    for field in (
        "canonical_digest",
        "ledger_prefix_sha256",
        "projected_item_sha256",
    ):
        value = armed.get(field)
        if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
            raise RecoveryEvidenceError(f"dispatch armed {field} is malformed")
    prefix_bytes = _nonnegative_integer(
        armed.get("ledger_prefix_bytes"), "ledger prefix bytes"
    )
    if ledger_identity is None and prefix_bytes != 0:
        raise RecoveryEvidenceError("absent ledger has a nonempty prefix")
    if not isinstance(armed.get("frontier_binding"), dict):
        raise RecoveryEvidenceError("dispatch frontier binding is malformed")
    wip = armed.get("target_wip_snapshot")
    if (
        not isinstance(wip, list)
        or len(wip) != 2
        or not isinstance(wip[0], str)
        or not Path(wip[0]).is_absolute()
        or (
            wip[1] is not None
            and (
                not isinstance(wip[1], str)
                or SHA256_RE.fullmatch(wip[1]) is None
            )
        )
    ):
        raise RecoveryEvidenceError("dispatch WIP snapshot is malformed")
    historical = armed.get("historical_runner")
    source_sha256 = (
        historical.get("source_sha256")
        if isinstance(historical, dict)
        else None
    )
    if (
        not isinstance(historical, dict)
        or set(historical) != QUIESCED_INCOMPLETE_RUNNER_KEYS
        or source_sha256 not in QUIESCED_INCOMPLETE_RUNNER_HEADS
        or historical.get("schema") != 1
        or historical.get("head_commit")
        != QUIESCED_INCOMPLETE_RUNNER_HEADS.get(source_sha256)
        or historical.get("evidence_schema") != "sealed_transcript_only_v1"
        or historical.get("lock_schema") != "in_workspace_v1"
    ):
        raise RecoveryEvidenceError(
            "quiesced incomplete-evidence runner receipt is not approved"
        )
    for field in (
        "worktree",
        "cwd",
        "interpreter",
        "artifacts_root",
        "scratch_root",
        "ledger",
    ):
        _normalized_absolute(historical.get(field), f"runner {field}")

    if (
        set(failed) != QUIESCED_INCOMPLETE_EVIDENCE_FAILED_KEYS
        or failed.get("event") != "dispatch_failed"
        or failed.get("exception_type") != "CampaignPlanError"
        or failed.get("reason") != QUIESCED_INCOMPLETE_EVIDENCE_REASON
        or failed.get("child_returncode") != 1
        or failed.get("transcript") is not None
        or failed.get("protected_identity") is not None
        or failed.get("process_tree_quiesced") is not True
        or failed.get("descendant_quiescence_unproven") is not False
        or failed.get("normal_exit_left_captured_descendants") is not False
        or failed.get("detached_processes_proven_absent") is not False
    ):
        raise RecoveryEvidenceError(
            "quiesced incomplete-evidence failure row is not the exact incident"
        )
    workspace = failed.get("workspace")
    protected = failed.get("protected")
    workspace_identity = failed.get("workspace_identity")
    pre_workspace = (
        workspace is None
        and protected is None
        and workspace_identity is None
    )
    if not pre_workspace:
        if (
            not isinstance(workspace, str)
            or not workspace
            or SAFE_COMPONENT_RE.fullmatch(workspace) is None
            or protected != workspace
        ):
            raise RecoveryEvidenceError(
                "quiesced incomplete-evidence workspace name is malformed"
            )
        _identity(workspace_identity, "workspace")
    return ParsedMarker(dispatch_id, armed, failed, None)


def _parse_isolated_generation_marker(
    raw: bytes,
    *,
    require_recovery_arm: bool | None,
    interrupted_exec: bool,
) -> ParsedMarker:
    """Parse one exact explicit artifact-isolation marker profile."""

    label = (
        "interrupted-generation"
        if interrupted_exec else "sandboxed-generation"
    )
    if not raw or len(raw) > MAX_MARKER_BYTES or not raw.endswith(b"\n"):
        raise RecoveryEvidenceError(f"{label} marker framing is malformed")
    rows = parse_canonical_jsonl(raw, label=f"{label} marker")
    expected_lengths = (
        {3} if require_recovery_arm is True
        else {2} if require_recovery_arm is False
        else {2, 3}
    )
    if len(rows) not in expected_lengths:
        raise RecoveryEvidenceError(
            f"{label} marker has an invalid phase row count"
        )
    armed, failed = rows[:2]
    if interrupted_exec:
        failed_keys = frozenset(failed)
        if (
            failed_keys not in {
                UNQUIESCED_REQUIRED_KEYS,
                UNQUIESCED_REQUIRED_KEYS
                | frozenset({"boundary_finding_counts"}),
            }
            or failed.get("event") != "dispatch_unquiesced"
            or failed.get("exception_type") != "UnquiescedChildError"
            or failed.get("child_returncode") != -signal.SIGINT
        ):
            raise RecoveryEvidenceError(
                "interrupted-generation row is not the exact v2 incident"
            )
        counts = failed.get("boundary_finding_counts")
        if counts is not None and counts != {"dynamic_execution": 1}:
            raise RecoveryEvidenceError(
                "interrupted-generation boundary findings are not the exact "
                "workspace-only incident"
            )
        validated = parse_dispatch_marker(
            canonical_json_line(armed) + canonical_json_line(failed),
            require_recovery_arm=False,
        )
    else:
        if (
            set(failed) != SANDBOXED_FAILED_KEYS
            or failed.get("event") != "dispatch_failed"
            or failed.get("exception_type") != "UnquiescedChildError"
            or failed.get("child_returncode") is not None
        ):
            raise RecoveryEvidenceError(
                "sandboxed-generation failure row is not the exact "
                "incident shape"
            )
        for field in ("reason", "workspace", "protected", "transcript"):
            if not isinstance(failed.get(field), str) or not failed[field]:
                raise RecoveryEvidenceError(
                    f"sandboxed-generation {field} is malformed"
                )
        child_pid = _nonnegative_integer(
            failed.get("child_pid"), "child pid"
        )
        if child_pid <= 1:
            raise RecoveryEvidenceError(
                "sandboxed-generation child PID is malformed"
            )
        for field in ("workspace_identity", "protected_identity"):
            _identity(failed.get(field), field)
        validation_row = dict(failed)
        validation_row["event"] = "dispatch_unquiesced"
        validation_row["child_returncode"] = -1
        validated = parse_dispatch_marker(
            canonical_json_line(armed)
            + canonical_json_line(validation_row),
            require_recovery_arm=False,
        )
    if armed.get("schema") != DISPATCH_QUARANTINE_SCHEMA_V2:
        raise RecoveryEvidenceError(
            f"{label} release requires a v2 rollback capsule"
        )

    historical = armed.get("historical_runner")
    source_sha256 = (
        historical.get("source_sha256")
        if isinstance(historical, dict) else None
    )
    if interrupted_exec:
        if (
            not isinstance(historical, dict)
            or set(historical) != QUIESCED_INCOMPLETE_RUNNER_KEYS
            or historical.get("schema") != 1
            or source_sha256
            != "7455d304c96f5b070ecb4e62a45bcca21e4d5faf52027b8c3434dc094f7e7b0b"
            or historical.get("head_commit")
            != "246405c1cd903e1dcde9d3a4c6eed1ec93cf2c1f"
            or historical.get("evidence_schema")
            != "sealed_transcript_only_v1"
            or historical.get("lock_schema") != "in_workspace_v1"
        ):
            raise RecoveryEvidenceError(
                "interrupted-generation runner receipt is not the exact "
                "v2 runner"
            )
        for field in (
            "worktree", "cwd", "interpreter", "artifacts_root",
            "scratch_root", "ledger",
        ):
            _normalized_absolute(historical.get(field), f"runner {field}")
    elif (
        not isinstance(historical, dict)
        or not historical
        or source_sha256 not in APPROVED_SANDBOXED_GENERATION_SOURCES
    ):
        raise RecoveryEvidenceError(
            "sandboxed-generation historical isolation binding is malformed"
        )

    arm = rows[2] if len(rows) == 3 else None
    if arm is None:
        return ParsedMarker(validated.dispatch_id, armed, failed, None)
    arm_keys = set(arm)
    transition_arm = (
        interrupted_exec
        and arm_keys == INTERRUPTED_GENERATION_TRANSITION_ARM_KEYS
    )
    one_exec_arm = (
        arm_keys in (
            INTERRUPTED_GENERATION_ARM_KEYS,
            INTERRUPTED_GENERATION_TRANSITION_ARM_KEYS,
        )
        if interrupted_exec
        else arm_keys == SANDBOXED_GENERATION_EXEC_ARM_KEYS
    )
    zero_exec_arm = (
        False
        if interrupted_exec
        else arm_keys == SANDBOXED_GENERATION_ARM_KEYS
    )
    expected_arm_schema = (
        INTERRUPTED_GENERATION_TRANSITION_ARM_SCHEMA
        if transition_arm
        else INTERRUPTED_GENERATION_ARM_SCHEMA
        if interrupted_exec
        else SANDBOXED_GENERATION_EXEC_ARM_SCHEMA
        if one_exec_arm
        else SANDBOXED_GENERATION_ARM_SCHEMA
    )
    if (
        not (zero_exec_arm or one_exec_arm)
        or arm.get("schema") != armed.get("schema")
        or arm.get("dispatch_id") != validated.dispatch_id
        or arm.get("event") != (
            INTERRUPTED_GENERATION_ARM_EVENT
            if interrupted_exec else SANDBOXED_GENERATION_ARM_EVENT
        )
        or arm.get("recovery_arm_schema") != expected_arm_schema
        or arm.get("authority_kind") != (
            "explicit_operator_assumed_interrupted_artifact_isolation_v1"
            if interrupted_exec
            else "explicit_operator_assumed_artifact_isolation_v1"
        )
        or arm.get("operator_provenance_assumption")
        != "historical_codex_workspace_write_effective_as_invoked"
        or arm.get("sandbox_contract_sha256") != SANDBOX_CONTRACT_SHA256
        or arm.get("process_claim") != (
            "same_boot_scheduler_pid_absent_only"
            if interrupted_exec
            else "named_root_and_owned_group_absent_only"
        )
        or arm.get("process_tree_quiesced") is not False
        or arm.get("detached_processes_proven_absent") is not False
        or arm.get("isolation_claim")
        != "published_artifact_namespace_unreachable_by_assumption"
        or arm.get("scratch_root_disposition") != "abandoned_in_place"
        or arm.get("required_retry_scratch_relation")
        != "outside_abandoned_path_and_inode"
        or arm.get("operator_artifact_scanner_assumption")
        != "current_full_artifact_scanner_reported_pass"
    ):
        raise RecoveryEvidenceError(
            f"{label} recovery arm schema is malformed"
        )

    armed_at = _utc_timestamp(armed.get("recorded_at"), "dispatch armed")
    failed_at = _utc_timestamp(failed.get("recorded_at"), f"dispatch {label}")
    absence_first = _utc_timestamp(
        arm.get("absence_first_at"), "absence first"
    )
    absence_last = _utc_timestamp(
        arm.get("absence_last_at"), "absence last"
    )
    arm_at = _utc_timestamp(arm.get("recorded_at"), f"{label} arm")
    nonce = arm.get("recovery_nonce")
    if not isinstance(nonce, str) or DISPATCH_ID_RE.fullmatch(nonce) is None:
        raise RecoveryEvidenceError(f"{label} nonce is malformed")
    _validate_boot_identity_fields(
        arm.get("boot_identity_source"), arm.get("boot_identity")
    )
    for field in (
        "marker_root_identity", "pre_arm_marker_identity",
        "armed_marker_identity", "scratch_root_identity",
        "workspace_identity", "protected_identity",
        "workspace_lock_identity", "wip_rollback_capsule_identity",
    ):
        _identity(arm.get(field), field)
    if arm.get("pre_arm_marker_identity") == arm.get(
        "armed_marker_identity"
    ):
        raise RecoveryEvidenceError(f"{label} marker identities alias")
    numeric_fields = [
        "scheduler_pid", "absence_sample_count", "absence_window_ns",
        "pre_arm_marker_bytes", "ledger_prefix_bytes",
        "wip_rollback_capsule_bytes",
    ]
    if not interrupted_exec:
        numeric_fields.extend(("child_pid", "child_pgid"))
    for field in numeric_fields:
        _nonnegative_integer(arm.get(field), field)
    invalid_absence = any((
        arm.get("scheduler_pid") != armed.get("pid"),
        int(arm["scheduler_pid"]) <= 1,
        not 2 <= int(arm["absence_sample_count"]) <= 16,
        not 0 < int(arm["absence_window_ns"]) <= 60_000_000_000,
        not (
            armed_at <= failed_at <= absence_first <= absence_last <= arm_at
        ),
    ))
    if not interrupted_exec:
        invalid_absence = invalid_absence or any((
            arm.get("child_pid") != failed.get("child_pid"),
            arm.get("child_pgid") != failed.get("child_pid"),
            int(arm["child_pid"]) <= 1,
            arm.get("scheduler_pid") == arm.get("child_pid"),
        ))
    if invalid_absence:
        raise RecoveryEvidenceError(f"{label} absence scope is malformed")

    hash_fields = [
        "pre_arm_marker_sha256", "projected_item_sha256",
        "historical_runner_sha256", "sandbox_contract_sha256",
        "workspace_tree_observation_sha256", "protected_tree_sha256",
        "canonical_digest", "ledger_prefix_sha256", "wip_state_sha256",
        "wip_rollback_capsule_sha256",
        "wip_rollback_capsule_state_sha256",
        "wip_restore_logical_state_sha256",
    ]
    if one_exec_arm:
        hash_fields.append("exec_record_sha256")
    if transition_arm:
        hash_fields.append("observed_canonical_digest")
    for field in hash_fields:
        value = arm.get(field)
        if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
            raise RecoveryEvidenceError(f"{label} {field} is malformed")
    for field in ("scratch_root", "workspace_lock_path"):
        _normalized_absolute(arm.get(field), field)
    for field in ("workspace", "protected", "workspace_lock_schema"):
        if not isinstance(arm.get(field), str) or not arm[field]:
            raise RecoveryEvidenceError(f"{label} {field} is malformed")

    pre_arm = b"".join(canonical_json_line(row) for row in rows[:2])
    historical_sha = hashlib.sha256(
        canonical_json_line(historical)
    ).hexdigest()
    mirrored = {
        "projected_item_sha256": armed.get("projected_item_sha256"),
        "canonical_digest": armed.get("canonical_digest"),
        "ledger_prefix_bytes": armed.get("ledger_prefix_bytes"),
        "ledger_prefix_sha256": armed.get("ledger_prefix_sha256"),
        "workspace": failed.get("workspace"),
        "workspace_identity": failed.get("workspace_identity"),
        "protected": failed.get("protected"),
        "protected_identity": failed.get("protected_identity"),
        "wip_rollback_capsule_name": armed.get(
            "wip_rollback_capsule_name"
        ),
        "wip_rollback_capsule_identity": armed.get(
            "wip_rollback_capsule_identity"
        ),
        "wip_rollback_capsule_bytes": armed.get(
            "wip_rollback_capsule_bytes"
        ),
        "wip_rollback_capsule_sha256": armed.get(
            "wip_rollback_capsule_sha256"
        ),
        "wip_rollback_capsule_state_sha256": armed.get(
            "wip_rollback_capsule_state_sha256"
        ),
        "wip_restore_logical_state_schema": armed.get(
            "wip_restore_logical_state_schema"
        ),
        "wip_restore_logical_state_sha256": armed.get(
            "wip_restore_logical_state_sha256"
        ),
    }
    if any((
        arm.get("scratch_root") != historical.get("scratch_root"),
        arm.get("workspace_lock_schema") != historical.get("lock_schema"),
        arm.get("protected") != arm.get("workspace"),
        SAFE_COMPONENT_RE.fullmatch(str(arm.get("workspace"))) is None,
        arm.get("workspace") in {".", ".."},
        Path(str(arm.get("workspace"))).name != arm.get("workspace"),
        arm.get("pre_arm_marker_bytes") != len(pre_arm),
        arm.get("pre_arm_marker_sha256")
        != hashlib.sha256(pre_arm).hexdigest(),
        arm.get("historical_runner_sha256") != historical_sha,
        any(arm.get(field) != value for field, value in mirrored.items()),
    )):
        raise RecoveryEvidenceError(f"{label} recovery arm binding changed")
    if transition_arm:
        pin = audited_canonical_digest_transition(
            armed, arm.get("observed_canonical_digest")
        )
        if (
            pin is None
            or arm.get("canonical_digest_transition_schema")
            != CANONICAL_DIGEST_TRANSITION_SCHEMA
            or arm.get("canonical_digest_transition_id")
            != pin.transition_id
        ):
            raise RecoveryEvidenceError(
                "interrupted-generation canonical transition is not the "
                "exact audited incident"
            )
    return ParsedMarker(validated.dispatch_id, armed, failed, arm)


def parse_sandboxed_generation_marker(
    raw: bytes, *, require_recovery_arm: bool | None = None
) -> ParsedMarker:
    """Parse the original child-PID sandbox-isolation incident."""

    return _parse_isolated_generation_marker(
        raw,
        require_recovery_arm=require_recovery_arm,
        interrupted_exec=False,
    )


def parse_interrupted_generation_marker(
    raw: bytes, *, require_recovery_arm: bool | None = None
) -> ParsedMarker:
    """Parse only the v2 outer-SIGINT/sealed-exec scheduler incident."""

    return _parse_isolated_generation_marker(
        raw,
        require_recovery_arm=require_recovery_arm,
        interrupted_exec=True,
    )


def _validate_boot_identity_fields(source: object, value: object) -> None:
    if source not in {"darwin_kern_bootsessionuuid", "linux_proc_boot_id"}:
        raise RecoveryEvidenceError("boot identity source is not authoritative")
    if (
        not isinstance(value, str)
        or value != value.lower()
        or BOOT_ID_RE.fullmatch(value) is None
    ):
        raise RecoveryEvidenceError("kernel boot identity is malformed")


def validate_boot_identity(identity: BootIdentity) -> BootIdentity:
    _validate_boot_identity_fields(identity.source, identity.value)
    return identity


def require_changed_boot_identity(
    armed_source: object,
    armed_value: object,
    current: BootIdentity,
) -> BootIdentity:
    _validate_boot_identity_fields(armed_source, armed_value)
    validate_boot_identity(current)
    if current.source != armed_source:
        raise RecoveryEvidenceError(
            "kernel boot identity source changed across recovery"
        )
    if current.value == armed_value:
        raise RecoveryEvidenceError(
            "post-reboot recovery is still running in the armed boot session"
        )
    return current


def _darwin_boot_identity() -> BootIdentity:
    libc = ctypes.CDLL(None, use_errno=True)
    sysctlbyname = libc.sysctlbyname
    sysctlbyname.argtypes = [
        ctypes.c_char_p,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_void_p,
        ctypes.c_size_t,
    ]
    sysctlbyname.restype = ctypes.c_int
    size = ctypes.c_size_t()
    if sysctlbyname(
        b"kern.bootsessionuuid", None, ctypes.byref(size), None, 0
    ) != 0 or not 1 <= size.value <= 128:
        error = ctypes.get_errno()
        raise RecoveryEvidenceError(
            f"Darwin kern.bootsessionuuid is unavailable (errno={error})"
        )
    buffer = ctypes.create_string_buffer(size.value)
    if sysctlbyname(
        b"kern.bootsessionuuid",
        buffer,
        ctypes.byref(size),
        None,
        0,
    ) != 0:
        error = ctypes.get_errno()
        raise RecoveryEvidenceError(
            f"Darwin kern.bootsessionuuid is unavailable (errno={error})"
        )
    try:
        value = buffer.raw[:size.value].rstrip(b"\0").decode("ascii").lower()
    except UnicodeError as exc:
        raise RecoveryEvidenceError(
            "Darwin kern.bootsessionuuid is malformed"
        ) from exc
    identity = BootIdentity("darwin_kern_bootsessionuuid", value)
    return validate_boot_identity(identity)


def _stable_kernel_text(path: Path, label: str) -> str:
    try:
        before = path.stat(follow_symlinks=False)
        if not path.is_file() or path.is_symlink():
            raise RecoveryEvidenceError(f"{label} is not a kernel file")
        raw = path.read_bytes()
        after = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise RecoveryEvidenceError(f"{label} is unavailable") from exc
    if (
        (before.st_dev, before.st_ino, before.st_size)
        != (after.st_dev, after.st_ino, after.st_size)
    ):
        raise RecoveryEvidenceError(f"{label} changed during read")
    try:
        return raw.decode("ascii").strip()
    except UnicodeError as exc:
        raise RecoveryEvidenceError(f"{label} is malformed") from exc


def _linux_boot_identity() -> BootIdentity:
    boot_id = _stable_kernel_text(
        Path("/proc/sys/kernel/random/boot_id"), "Linux boot ID"
    ).lower()
    return validate_boot_identity(BootIdentity("linux_proc_boot_id", boot_id))


def authoritative_boot_identity() -> BootIdentity:
    """Read one narrow kernel boot identity; never enumerate processes."""

    if sys.platform == "darwin":
        return _darwin_boot_identity()
    if sys.platform.startswith("linux"):
        return _linux_boot_identity()
    raise RecoveryEvidenceError(
        "post-reboot recovery has no authoritative boot proof on this platform"
    )


def boot_identity_receipt(identity: BootIdentity) -> dict[str, object]:
    validate_boot_identity(identity)
    return {
        "source": identity.source,
        "identity_sha256": hashlib.sha256(
            identity.value.encode("ascii")
        ).hexdigest(),
    }


BootIdentityProvider = Callable[[], BootIdentity]
