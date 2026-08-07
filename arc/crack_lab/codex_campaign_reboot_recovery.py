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
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
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
MAX_MARKER_BYTES = 1024 * 1024
SHA256_RE = re.compile(r"[0-9a-f]{64}")
DISPATCH_ID_RE = re.compile(r"[0-9a-f]{32}")
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
    unquiesced_phase = (
        UNQUIESCED_REQUIRED_KEYS.issubset(unquiesced)
        and set(unquiesced).issubset(
            UNQUIESCED_REQUIRED_KEYS | UNQUIESCED_OPTIONAL_KEYS
        )
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
