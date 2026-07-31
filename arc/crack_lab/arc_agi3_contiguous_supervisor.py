#!/usr/bin/env python3
"""Fail-closed primitives for the isolated ARC-AGI-3 contiguous supervisor.

This module deliberately does not launch a proposer.  It defines the boundary
that the later automated scheduler must use: authoritative inventory admission,
strict host-owned checkpoints, a narrow proposer-output schema, immutable
versioned artifacts, atomic current-version selection, and a signed-off launch
attestation.  A scheduler that cannot pass these gates must not start.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import hmac
import json
import math
import os
import re
import resource
import secrets
import shutil
import signal
import stat
import subprocess
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Sequence

import codex_campaign_status as Status
import arc_agi3_contiguous_conformance as Conformance
import arc_agi3_python_runtime_manifest as RuntimeManifest
import arc_agi3_source_schema as SourceSchema


EXPECTED_GAMES = 25
EXPECTED_LEVELS = 183
MAX_REPLAY_ACTIONS = 600
MAX_CANDIDATE_FILES = 256
MAX_CANDIDATE_FILE_BYTES = 16 * 1024 * 1024
MAX_CANDIDATE_TOTAL_BYTES = 64 * 1024 * 1024
MAX_CANDIDATE_DEPTH = 8
# Deliberately false until the container scheduler, observed attestation
# producer, Arena RPC boundary, and canonical 183/183 release gate exist.
# Schema-valid, self-asserted JSON must never authorize an early launch.
CONTIGUOUS_LAUNCH_READY = False
CHECKPOINT_NAME = "checkpoint.json"
CANDIDATE_NAME = "candidate_path.json"
CANDIDATE_EVIDENCE_NAME = "host_candidate_path.json"
POINTER_NAME = "current.json"
HOST_RECEIPT_NAME = "host_promotion_receipt.json"
WINNING_SOURCE_NAME = "winning_source"
OPERATOR_LEASE_SCHEMA = 1
OPERATOR_LEASE_ROOT_NAME = "operator_lease"
OPERATOR_LEASE_ACQUIRE_TIMEOUT_SECONDS = 5.0
OPERATOR_LEASE_HEARTBEAT_SECONDS = 10.0
MAX_OPERATOR_LEASE_ACQUISITIONS = 1024
POST_INCIDENT_META_ROOT_NAME = "post_incident_meta_diagnostic"
POST_INCIDENT_META_SCHEMA = 1
POST_INCIDENT_META_MAX_EPISODES = 8
POST_INCIDENT_META_MAX_CONTROL_BYTES = 1024 * 1024
POST_INCIDENT_META_MAX_STREAM_BYTES = 1024 * 1024
POST_INCIDENT_META_PROTOCOL_TEXT = """\
ARC-AGI-3 contiguous post-incident meta-diagnostic protocol v1
Exactly one sealed, bounded diagnosis may follow each distinct authenticated
controller-substrate OPERATOR_INCIDENT, with at most eight episodes per
campaign. The request contains only a host-selected incident projection and
control hashes. The response is quarantined advice: it has no
scheduler, solver, WIP, cost, retry, dispatch, or promotion authority. The
runner remains paused and latched. Missing, malformed, interrupted, or failed
diagnosis is terminal and requires surfaced human intervention.
"""
POST_INCIDENT_META_PROTOCOL_SHA256 = hashlib.sha256(
    POST_INCIDENT_META_PROTOCOL_TEXT.encode("ascii")
).hexdigest()
POST_INCIDENT_META_RECOMMENDATIONS = frozenset({
    "REMATERIALIZE_AND_REPROBE_CONTROLLER_SUBSTRATE",
    "NO_SAFE_AUTOMATED_RECOVERY",
})
FORBIDDEN_EXPORT_NAMES = {
    CHECKPOINT_NAME,
    HOST_RECEIPT_NAME,
    "campaign_usage.jsonl",
    "accounting.json",
    "promotion_manifest.json",
}
REQUIRED_PROMOTION_CHECKS = {
    "candidate_schema",
    "transcript_taint",
    "output_taint",
    "probe_isolation_binding",
    "replay_from_parent",
    "path_replay_from_zero",
    "source_replay_from_zero",
    "exact_boundary",
    "winning_source_snapshot",
    "manifest_chain",
    "checkpoint_rebuilt",
}
PROMOTION_RECEIPT_FIELDS = {
    "schema",
    "game",
    "target_level",
    "authoritative_target",
    "parent_checkpoint_sha256",
    "parent_action_count",
    "remaining_action_budget",
    "fresh_prefix_required",
    "candidate_manifest_sha256",
    "checkpoint_sha256",
    "source_tree_sha256",
    "winning_source_path",
    "winning_source_sha256",
    "promotion_manifest_path",
    "promotion_manifest_sha256",
    "exact_path",
    "checks",
}
CONTROL_CONTRACT_FILES = Conformance.CONTROL_CONTRACT_FILES


class SupervisorContractError(RuntimeError):
    """A fail-closed supervisor admission or integrity error."""


PROBE_ISOLATION_SCHEMA = 1
PROBE_ISOLATION_KIND = "arc_agi3_probe_isolation_decision"
PROBE_ISOLATION_AUTHORITY = "trusted_arena_host_controller"
PROBE_ISOLATION_CANARY = (
    "sibling_same_action_and_mutable_graph/v1"
)
VERIFIED_ISOLATED_CLONE_MODE = "verified_isolated_clone"
FRESH_PROCESS_PER_CANDIDATE_MODE = "fresh_process_per_candidate"
PROBE_ISOLATION_MODES = frozenset({
    VERIFIED_ISOLATED_CLONE_MODE,
    FRESH_PROCESS_PER_CANDIDATE_MODE,
})
PROBE_CANARY_STATUSES = frozenset({"PASS", "LEAK", "INCONCLUSIVE"})
PROBE_CANARY_FAILURE_STAGES = frozenset({
    "NONE",
    "SEED",
    "CLONE",
    "LEFT_STEP",
    "LEFT_OBSERVATION",
    "RIGHT_STEP",
    "RIGHT_OBSERVATION",
    "COMPARE",
})
PROBE_ISOLATION_FIELDS = frozenset({
    "schema",
    "kind",
    "authority",
    "algorithm",
    "mode",
    "seed_snapshot_sha256",
    "seed_path_sha256",
    "canary_status",
    "failure_stage",
    "canary_action",
    "canary_action_sha256",
    "mutable_graph_status",
    "shared_mutable_identity_count",
    "mutable_graph_observation_sha256",
    "seed_before_sha256",
    "left_before_sha256",
    "right_before_sha256",
    "left_after_sha256",
    "right_after_left_sha256",
    "seed_after_left_sha256",
    "right_after_sha256",
    "seed_after_right_sha256",
    "mutation_observed",
    "sibling_unchanged",
    "matching_trajectory",
    "fallback_process_ready",
    "fallback_process_identity_sha256",
})


def _probe_isolation_json_sha256(value: object) -> str:
    try:
        raw = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise SupervisorContractError(
            "probe-isolation evidence is not canonical JSON"
        ) from exc
    return hashlib.sha256(raw).hexdigest()


def validate_probe_isolation_evidence(
    value: object,
    *,
    expected_seed_snapshot_sha256: str | None = None,
    expected_seed_path_sha256: str | None = None,
) -> tuple[
    Literal[
        "verified_isolated_clone",
        "fresh_process_per_candidate",
    ],
    str,
]:
    """Validate the host-only probe-substrate decision.

    The decision is deliberately finite and machine-derived.  Solver text,
    worker outcomes, and candidate manifests have no field that can select or
    weaken the mode.  A clone mode requires the complete deterministic sibling
    canary to pass; every leak or inconclusive observation requires a ready,
    authenticated fresh-process substrate.
    """

    if not isinstance(value, Mapping) or set(value) != PROBE_ISOLATION_FIELDS:
        raise SupervisorContractError(
            "probe-isolation evidence schema mismatch"
        )
    evidence = dict(value)
    if (
        evidence.get("schema") != PROBE_ISOLATION_SCHEMA
        or isinstance(evidence.get("schema"), bool)
        or evidence.get("kind") != PROBE_ISOLATION_KIND
        or evidence.get("authority") != PROBE_ISOLATION_AUTHORITY
        or evidence.get("algorithm") != PROBE_ISOLATION_CANARY
        or evidence.get("mode") not in PROBE_ISOLATION_MODES
        or evidence.get("canary_status") not in PROBE_CANARY_STATUSES
        or evidence.get("failure_stage")
        not in PROBE_CANARY_FAILURE_STAGES
        or evidence.get("mutable_graph_status")
        not in PROBE_CANARY_STATUSES
    ):
        raise SupervisorContractError(
            "probe-isolation decision is not controller-owned and finite"
        )
    hash_fields = (
        "seed_snapshot_sha256",
        "seed_path_sha256",
        "canary_action_sha256",
        "mutable_graph_observation_sha256",
    )
    if any(
        not isinstance(evidence.get(field), str)
        or not _is_sha256_hex(evidence[field])
        for field in hash_fields
    ):
        raise SupervisorContractError(
            "probe-isolation binding hashes are malformed"
        )
    observation_fields = (
        "seed_before_sha256",
        "left_before_sha256",
        "right_before_sha256",
        "left_after_sha256",
        "right_after_left_sha256",
        "seed_after_left_sha256",
        "right_after_sha256",
        "seed_after_right_sha256",
    )
    if any(
        digest is not None
        and (
            not isinstance(digest, str)
            or not _is_sha256_hex(digest)
        )
        for digest in (
            evidence.get(field) for field in observation_fields
        )
    ):
        raise SupervisorContractError(
            "probe-isolation canary observation hash is malformed"
        )
    booleans = (
        "mutation_observed",
        "sibling_unchanged",
        "matching_trajectory",
        "fallback_process_ready",
    )
    if any(not isinstance(evidence.get(field), bool) for field in booleans):
        raise SupervisorContractError(
            "probe-isolation verdict fields must be booleans"
        )
    shared_count = evidence.get("shared_mutable_identity_count")
    if (
        not isinstance(shared_count, int)
        or isinstance(shared_count, bool)
        or shared_count < 0
    ):
        raise SupervisorContractError(
            "probe-isolation mutable graph count is malformed"
        )
    action = evidence.get("canary_action")
    action_valid = (
        isinstance(action, int)
        and not isinstance(action, bool)
        and action in {1, 2, 3, 4, 5, 7}
    ) or (
        isinstance(action, list)
        and len(action) == 3
        and action[0] == 6
        and all(
            isinstance(item, int) and not isinstance(item, bool)
            for item in action
        )
        and 0 <= action[1] < 64
        and 0 <= action[2] < 64
    )
    if (
        not action_valid
        or _probe_isolation_json_sha256(action)
        != evidence["canary_action_sha256"]
    ):
        raise SupervisorContractError(
            "probe-isolation canary action is malformed"
        )
    if (
        expected_seed_snapshot_sha256 is not None
        and evidence["seed_snapshot_sha256"]
        != expected_seed_snapshot_sha256
    ) or (
        expected_seed_path_sha256 is not None
        and evidence["seed_path_sha256"] != expected_seed_path_sha256
    ):
        raise SupervisorContractError(
            "probe-isolation decision targets another exploration seed"
        )

    mode = evidence["mode"]
    fallback_digest = evidence.get(
        "fallback_process_identity_sha256"
    )
    if mode == VERIFIED_ISOLATED_CLONE_MODE:
        before = evidence["seed_before_sha256"]
        if (
            evidence["canary_status"] != "PASS"
            or evidence["failure_stage"] != "NONE"
            or evidence["mutable_graph_status"] != "PASS"
            or shared_count != 0
            or any(evidence[field] is None for field in observation_fields)
            or not evidence["mutation_observed"]
            or not evidence["sibling_unchanged"]
            or not evidence["matching_trajectory"]
            or evidence["fallback_process_ready"]
            or fallback_digest is not None
            or not (
                before
                == evidence["left_before_sha256"]
                == evidence["right_before_sha256"]
                == evidence["right_after_left_sha256"]
                == evidence["seed_after_left_sha256"]
                == evidence["seed_after_right_sha256"]
            )
            or evidence["left_after_sha256"]
            != evidence["right_after_sha256"]
            or evidence["left_after_sha256"] == before
        ):
            raise SupervisorContractError(
                "clone probing lacks a passing sibling-isolation canary"
            )
    else:
        if (
            evidence["canary_status"] not in {"LEAK", "INCONCLUSIVE"}
            or evidence["failure_stage"] == "NONE"
            or not evidence["fallback_process_ready"]
            or not isinstance(fallback_digest, str)
            or not _is_sha256_hex(fallback_digest)
        ):
            raise SupervisorContractError(
                "unsafe or inconclusive clones lack a ready authenticated "
                "fresh-process substrate"
            )
    return mode, _probe_isolation_json_sha256(evidence)


@dataclass(frozen=True)
class TurnDrainDecision:
    """One scheduler decision at a proposer-turn lifecycle boundary."""

    phase: str
    launch_new_turn: bool
    request_container_stop: bool
    force_container_teardown: bool


def decide_turn_drain(
    *,
    elapsed_seconds: float,
    soft_allocation_seconds: float,
    proposer_active: bool,
    containment_fault: bool = False,
    containment_grace_expired: bool = False,
) -> TurnDrainDecision:
    """Keep healthy active turns alive past their nominal allocation.

    The effort ladder's 90/120/180-minute values are scheduling allocations,
    not kill deadlines.  Crossing one while a proposer is active moves the lane
    to ``draining`` and forbids another dispatch, but emits no stop signal.
    Process termination is reachable only through the independent containment
    fault path.
    """
    for label, value, allow_zero in (
        ("elapsed_seconds", elapsed_seconds, True),
        ("soft_allocation_seconds", soft_allocation_seconds, False),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
            or (not allow_zero and value == 0)
        ):
            raise SupervisorContractError(
                f"{label} must be a {'nonnegative' if allow_zero else 'positive'} "
                "number"
            )
    for label, value in (
        ("proposer_active", proposer_active),
        ("containment_fault", containment_fault),
        ("containment_grace_expired", containment_grace_expired),
    ):
        if not isinstance(value, bool):
            raise SupervisorContractError(f"{label} must be boolean")
    if containment_grace_expired and not containment_fault:
        raise SupervisorContractError(
            "containment grace cannot expire without a containment fault"
        )

    allocation_expired = elapsed_seconds >= soft_allocation_seconds
    if containment_fault:
        if not proposer_active:
            return TurnDrainDecision(
                phase="containment_fault_after_exit",
                launch_new_turn=False,
                request_container_stop=False,
                force_container_teardown=False,
            )
        if containment_grace_expired:
            return TurnDrainDecision(
                phase="containment_teardown",
                launch_new_turn=False,
                request_container_stop=True,
                force_container_teardown=True,
            )
        return TurnDrainDecision(
            phase="containment_stopping",
            launch_new_turn=False,
            request_container_stop=True,
            force_container_teardown=False,
        )

    if proposer_active:
        return TurnDrainDecision(
            phase="draining" if allocation_expired else "proposing",
            launch_new_turn=False,
            request_container_stop=False,
            force_container_teardown=False,
        )
    if allocation_expired:
        return TurnDrainDecision(
            phase="allocation_complete",
            launch_new_turn=False,
            request_container_stop=False,
            force_container_teardown=False,
        )
    return TurnDrainDecision(
        phase="ready",
        launch_new_turn=True,
        request_container_stop=False,
        force_container_teardown=False,
    )


def _reject_symlinked_path_components(path: Path, *, label: str) -> None:
    """Reject every existing symlink from the filesystem anchor to ``path``."""
    absolute = Path(os.path.abspath(path))
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current /= component
        if os.path.lexists(current) and current.is_symlink():
            raise SupervisorContractError(
                f"{label} contains a symlinked path component: {current}"
            )


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(_read_regular_bytes(path).decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SupervisorContractError(f"invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise SupervisorContractError(f"expected JSON object: {path}")
    return value


def _read_regular_bytes(path: Path) -> bytes:
    """Read one unaliased regular file without following a final symlink."""
    _reject_symlinked_path_components(path, label="host-owned file")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise SupervisorContractError(
                f"expected an unaliased regular host-owned file: {path}"
            )
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            return handle.read()
    finally:
        os.close(descriptor)


def _sha256_file(path: Path) -> str:
    _reject_symlinked_path_components(path, label="hashed file")
    digest = hashlib.sha256()
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise SupervisorContractError(
            f"cannot hash a regular file without following links: {path}"
        ) from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise SupervisorContractError(
                f"cannot hash a non-regular or hard-linked file: {path}"
            )
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
    finally:
        os.close(descriptor)


def _tree_hash(
    root: Path,
    *,
    exclude_relative: frozenset[str] = frozenset(),
    exclude_prefixes: tuple[str, ...] = (),
) -> str:
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(root).as_posix()
        if rel in exclude_relative or any(
            rel == prefix.rstrip("/")
            or rel.startswith(prefix.rstrip("/") + "/")
            for prefix in exclude_prefixes
        ):
            continue
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _winning_source_payloads(source: Path) -> dict[str, bytes]:
    """Read the canonical flat solver-source view from a regular source tree."""

    payloads: dict[str, bytes] = {}
    for entry in source.iterdir():
        if (
            entry.is_file()
            and entry.name not in SourceSchema.FORBIDDEN_FILES
            and entry.suffix in SourceSchema.ALLOWED_SUFFIXES
        ):
            payloads[entry.name] = _read_regular_bytes(entry)
    try:
        SourceSchema.validate_source_payloads(payloads)
    except SourceSchema.SourceSchemaError as exc:
        raise SupervisorContractError(
            "winning solver source violates the shared source schema"
        ) from exc
    return payloads


def validate_winning_source_tree(root: Path) -> str:
    """Validate one immutable winning-source view and return its tree hash."""

    _validate_regular_tree(root, label="winning source")
    if any(not entry.is_file() for entry in root.iterdir()):
        raise SupervisorContractError(
            "winning source must be a flat regular-file tree"
        )
    payloads = {
        entry.name: _read_regular_bytes(entry)
        for entry in root.iterdir()
    }
    try:
        SourceSchema.validate_source_payloads(payloads)
    except SourceSchema.SourceSchemaError as exc:
        raise SupervisorContractError(
            "winning source violates the shared source schema"
        ) from exc
    return _tree_hash(root)


def _validate_regular_tree(root: Path, *, label: str) -> None:
    """Reject aliases and special files before hashing or publishing a tree."""
    if root.is_symlink() or not root.is_dir():
        raise SupervisorContractError(f"{label} is not a regular directory")
    for path in root.rglob("*"):
        if path.is_symlink():
            raise SupervisorContractError(f"{label} contains a symlink: {path}")
        if path.is_file():
            if path.stat(follow_symlinks=False).st_nlink != 1:
                raise SupervisorContractError(
                    f"{label} contains a hard-linked file: {path}"
                )
        elif not path.is_dir():
            raise SupervisorContractError(
                f"{label} contains a non-regular entry: {path}"
            )


def _validate_candidate_output_quota(root: Path) -> None:
    """Bound host work before parsing or hashing proposer-exported bytes."""
    files = 0
    total_bytes = 0
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        files += 1
        if files > MAX_CANDIDATE_FILES:
            raise SupervisorContractError(
                "candidate output exceeds the file-count quota"
            )
        if len(relative.parts) > MAX_CANDIDATE_DEPTH:
            raise SupervisorContractError(
                "candidate output exceeds the path-depth quota"
            )
        metadata = path.stat(follow_symlinks=False)
        if metadata.st_size > MAX_CANDIDATE_FILE_BYTES:
            raise SupervisorContractError(
                f"candidate output file exceeds the byte quota: {relative}"
            )
        blocks = getattr(metadata, "st_blocks", None)
        allocated = blocks * 512 if blocks is not None else None
        if (
            metadata.st_size
            and allocated is not None
            and allocated < metadata.st_size
        ):
            raise SupervisorContractError(
                f"candidate output contains a sparse file: {relative}"
            )
        total_bytes += metadata.st_size
        if total_bytes > MAX_CANDIDATE_TOTAL_BYTES:
            raise SupervisorContractError(
                "candidate output exceeds the total-byte quota"
            )


def _write_new_regular_bytes(path: Path, payload: bytes, *, label: str) -> None:
    """Create a new host file without following or replacing a raced-in path."""
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as exc:
        raise SupervisorContractError(
            f"{label} path appeared or is not a safe regular file: {path}"
        ) from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise SupervisorContractError(
                f"{label} is not an unaliased regular file: {path}"
            )
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)


def _remove_host_staging_tree(path: Path) -> None:
    """Remove one supervisor-owned stage, including sealed subdirectories.

    Winning-source directories are intentionally made read-only before the
    atomic version move.  A rejection or crash recovery can therefore reach
    cleanup after that seal.  Reopen each real directory without following
    symlinks, restore owner write permission on the opened descriptor, and
    then use Python's symlink-safe ``rmtree`` implementation.
    """
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return
    if (
        path.is_symlink()
        or not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.getuid()
    ):
        raise SupervisorContractError(
            f"refusing to remove an unsafe staging tree: {path}"
        )
    try:
        for _root, _directories, _files, descriptor in os.fwalk(
            path, topdown=True, follow_symlinks=False
        ):
            current = os.fstat(descriptor)
            if (
                not stat.S_ISDIR(current.st_mode)
                or current.st_uid != os.getuid()
            ):
                raise SupervisorContractError(
                    f"staging tree contains an unowned directory: {path}"
                )
            os.fchmod(descriptor, 0o700)
        shutil.rmtree(path)
        _fsync_directory(path.parent)
    except SupervisorContractError:
        raise
    except OSError as exc:
        raise SupervisorContractError(
            f"cannot remove supervisor staging tree: {path}"
        ) from exc


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


def _fsync_tree(root: Path) -> None:
    """Durably flush a staged tree before it becomes pointer-addressable."""
    directories = [root]
    for path in root.rglob("*"):
        if path.is_file():
            descriptor = os.open(
                path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            )
            try:
                if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                    raise SupervisorContractError(
                        f"cannot fsync a non-regular file: {path}"
                    )
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        elif path.is_dir():
            directories.append(path)
    for directory in sorted(
        directories, key=lambda item: len(item.parts), reverse=True
    ):
        _fsync_directory(directory)


def control_contract_sha256() -> str:
    """Hash the exact host code/tests whose passing attestation authorizes launch."""
    try:
        return Conformance.control_contract_sha256()
    except Conformance.ConformanceError as exc:
        raise SupervisorContractError(str(exc)) from exc


def authoritative_inventory() -> dict[str, int]:
    targets = Status._authoritative_inventory()
    validate_inventory(targets)
    return targets


def authoritative_inventory_sha256(
    targets: dict[str, int] | None = None,
) -> str:
    """Bind launch authorization to the exact tested per-game target map."""
    selected = authoritative_inventory() if targets is None else targets
    validate_inventory(selected)
    payload = json.dumps(
        selected, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_inventory(targets: dict[str, int]) -> None:
    expected = Status._authoritative_inventory()
    if not isinstance(targets, dict):
        raise SupervisorContractError("authoritative inventory must be an object")
    malformed = {
        game: target
        for game, target in targets.items()
        if (
            not isinstance(game, str)
            or not game
            or not isinstance(target, int)
            or isinstance(target, bool)
            or target <= 0
        )
    }
    if malformed:
        raise SupervisorContractError(
            f"inventory contains invalid entries: {malformed}"
        )
    total = sum(targets.values())
    if len(targets) != EXPECTED_GAMES or total != EXPECTED_LEVELS:
        raise SupervisorContractError(
            "authoritative inventory must contain exactly "
            f"{EXPECTED_GAMES} games / {EXPECTED_LEVELS} levels; found "
            f"{len(targets)} / {total}"
        )
    if targets != expected:
        missing = sorted(set(expected) - set(targets))
        unexpected = sorted(set(targets) - set(expected))
        changed = {
            game: {"expected": expected[game], "found": targets[game]}
            for game in sorted(set(expected) & set(targets))
            if expected[game] != targets[game]
        }
        raise SupervisorContractError(
            "inventory does not exactly match authoritative per-game targets; "
            f"missing={missing}, unexpected={unexpected}, changed={changed}"
        )


@dataclass(frozen=True)
class TrustedCheckpoint:
    game: str
    reached: int
    total_marginal_C: int
    records: tuple[dict[str, Any], ...]
    final_path: tuple[Any, ...]
    validated: bool


def load_trusted_checkpoint(
    path: Path, *, expected_game: str, authoritative_target: int
) -> TrustedCheckpoint:
    if path.is_symlink() or not path.is_file():
        raise SupervisorContractError(
            f"checkpoint must be a regular host-owned file: {path}"
        )
    data = _read_json(path)
    required = {
        "game",
        "reached",
        "total_marginal_C",
        "records",
        "final_path",
        "validated",
    }
    if set(data) != required:
        missing = sorted(required - set(data))
        extra = sorted(set(data) - required)
        raise SupervisorContractError(
            f"checkpoint schema mismatch; missing={missing}, extra={extra}"
        )
    game = data["game"]
    reached = data["reached"]
    total = data["total_marginal_C"]
    records = data["records"]
    final_path = data["final_path"]
    validated = data["validated"]
    if game != expected_game:
        raise SupervisorContractError(
            f"checkpoint game mismatch: {game!r} != {expected_game!r}"
        )
    if (
        not isinstance(reached, int)
        or isinstance(reached, bool)
        or not 0 <= reached <= authoritative_target
    ):
        raise SupervisorContractError(
            f"checkpoint reached is outside 0..{authoritative_target}: {reached!r}"
        )
    if not isinstance(total, int) or isinstance(total, bool) or total < 0:
        raise SupervisorContractError("checkpoint has invalid total_marginal_C")
    if not isinstance(validated, bool):
        raise SupervisorContractError("checkpoint validated must be boolean")
    if reached and not validated:
        raise SupervisorContractError("nonzero checkpoint is not replay-validated")
    if not isinstance(final_path, list) or not all(
        _valid_replay_action(action) for action in final_path
    ):
        raise SupervisorContractError(
            "checkpoint final_path contains an invalid replay action"
        )
    if len(final_path) > MAX_REPLAY_ACTIONS:
        raise SupervisorContractError(
            "checkpoint final_path exceeds the public 600-action cap"
        )
    if reached > 0 and not final_path:
        raise SupervisorContractError(
            "nonzero checkpoint has no replay path"
        )
    if reached == 0 and (final_path or total or records):
        raise SupervisorContractError(
            "zero checkpoint must have empty path, records, and marginal total"
        )
    if not isinstance(records, list):
        raise SupervisorContractError("checkpoint records must be a list")
    levels: list[int] = []
    marginal_sum = 0
    for record in records:
        if not isinstance(record, dict) or set(record) != {
            "level", "marginal_C", "reached"
        }:
            raise SupervisorContractError("checkpoint has malformed level record")
        level = record["level"]
        marginal = record["marginal_C"]
        if (
            not isinstance(level, int)
            or isinstance(level, bool)
            or not isinstance(marginal, int)
            or isinstance(marginal, bool)
            or marginal < 0
            or record["reached"] is not True
        ):
            raise SupervisorContractError("checkpoint has invalid level record")
        levels.append(level)
        marginal_sum += marginal
    if levels != list(range(1, reached + 1)):
        raise SupervisorContractError(
            f"checkpoint record levels are not exactly 1..{reached}: {levels}"
        )
    if marginal_sum != total:
        raise SupervisorContractError(
            f"checkpoint marginal total mismatch: records={marginal_sum}, cached={total}"
        )
    return TrustedCheckpoint(
        game=game,
        reached=reached,
        total_marginal_C=total,
        records=tuple(records),
        final_path=tuple(final_path),
        validated=validated,
    )


@dataclass(frozen=True)
class FrontierAdmission:
    game: str
    reached: int
    next_level: int
    authoritative_target: int
    parent_checkpoint_sha256: str
    parent_action_count: int
    remaining_action_budget: int
    fresh_prefix_required: bool


def admit_next_frontier(
    checkpoint_path: Path,
    *,
    expected_game: str,
    requested_level: int,
) -> FrontierAdmission:
    """Admit exactly one sequential, authoritative frontier or fail closed."""
    targets = authoritative_inventory()
    if expected_game not in targets:
        raise SupervisorContractError(
            f"unknown game outside authoritative inventory: {expected_game!r}"
        )
    target = targets[expected_game]
    checkpoint = load_trusted_checkpoint(
        checkpoint_path,
        expected_game=expected_game,
        authoritative_target=target,
    )
    if checkpoint.reached >= target:
        raise SupervisorContractError(
            f"{expected_game} is already complete at authoritative target {target}"
        )
    next_level = checkpoint.reached + 1
    if (
        not isinstance(requested_level, int)
        or isinstance(requested_level, bool)
        or requested_level != next_level
    ):
        raise SupervisorContractError(
            "refusing nonsequential or wrong-authority frontier: "
            f"{expected_game} reached={checkpoint.reached}, "
            f"requested={requested_level!r}, required={next_level}, target={target}"
        )
    return FrontierAdmission(
        game=expected_game,
        reached=checkpoint.reached,
        next_level=next_level,
        authoritative_target=target,
        parent_checkpoint_sha256=_sha256_file(checkpoint_path),
        parent_action_count=len(checkpoint.final_path),
        remaining_action_budget=MAX_REPLAY_ACTIONS - len(checkpoint.final_path),
        fresh_prefix_required=(
            len(checkpoint.final_path) >= MAX_REPLAY_ACTIONS
        ),
    )


@dataclass(frozen=True)
class CandidateOutput:
    game: str
    target_level: int
    parent_checkpoint_sha256: str
    candidate_path: tuple[Any, ...]
    exported_files_sha256: dict[str, str]


def _safe_relative_file(value: str) -> bool:
    path = Path(value)
    return (
        bool(value)
        and not path.is_absolute()
        and ".." not in path.parts
        and path.name not in FORBIDDEN_EXPORT_NAMES
    )


def _is_sha256_hex(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _paths_overlap(left: Path, right: Path) -> bool:
    left_resolved = left.resolve(strict=False)
    right_resolved = right.resolve(strict=False)
    return (
        left_resolved == right_resolved
        or left_resolved in right_resolved.parents
        or right_resolved in left_resolved.parents
    )


def _valid_replay_action(action: object) -> bool:
    """Match the public Arena replay token grammar.

    Key actions are integers 1..5 or 7. Coordinate actions are JSON arrays
    ``[6, x, y]``. Booleans are rejected even though Python considers them ints.
    """
    if isinstance(action, int) and not isinstance(action, bool):
        return 1 <= action <= 7 and action != 6
    return (
        isinstance(action, list)
        and len(action) == 3
        and action[0] == 6
        and all(isinstance(value, int) and not isinstance(value, bool)
                for value in action)
        and all(0 <= value < 64 for value in action[1:])
    )


def validate_candidate_output(
    output_root: Path,
    *,
    expected_game: str,
    expected_level: int,
    parent_checkpoint_sha256: str,
) -> CandidateOutput:
    """Validate every exported byte; undeclared files and symlinks are fatal."""
    _validate_regular_tree(output_root, label="candidate output")
    _validate_candidate_output_quota(output_root)
    manifest_path = output_root / CANDIDATE_NAME
    data = _read_json(manifest_path)
    required = {
        "schema",
        "game",
        "target_level",
        "parent_checkpoint_sha256",
        "candidate_path",
        "exported_files_sha256",
    }
    if (
        set(data) != required
        or not isinstance(data["schema"], int)
        or isinstance(data["schema"], bool)
        or data["schema"] != 1
    ):
        raise SupervisorContractError("candidate output schema mismatch")
    if (
        data["game"] != expected_game
        or not isinstance(data["target_level"], int)
        or isinstance(data["target_level"], bool)
        or data["target_level"] != expected_level
    ):
        raise SupervisorContractError("candidate output targets the wrong frontier")
    if not _is_sha256_hex(parent_checkpoint_sha256):
        raise SupervisorContractError("host supplied an invalid parent checkpoint hash")
    if data["parent_checkpoint_sha256"] != parent_checkpoint_sha256:
        raise SupervisorContractError("candidate output has a stale/wrong parent")
    actions = data["candidate_path"]
    if (
        not isinstance(actions, list)
        or not actions
        or len(actions) > MAX_REPLAY_ACTIONS
        or not all(_valid_replay_action(action) for action in actions)
    ):
        raise SupervisorContractError(
            "candidate_path must contain at most 600 valid replay actions"
        )
    declared = data["exported_files_sha256"]
    if not isinstance(declared, dict):
        raise SupervisorContractError("exported_files_sha256 must be an object")
    for rel, expected_hash in declared.items():
        if (
            not isinstance(rel, str)
            or not _safe_relative_file(rel)
            or not _is_sha256_hex(expected_hash)
        ):
            raise SupervisorContractError(f"invalid declared export: {rel!r}")
        path = output_root / rel
        if not path.is_file() or _sha256_file(path) != expected_hash:
            raise SupervisorContractError(f"export hash mismatch: {rel}")
    actual = {
        path.relative_to(output_root).as_posix()
        for path in output_root.rglob("*")
        if path.is_file() and path != manifest_path
    }
    if actual != set(declared):
        raise SupervisorContractError(
            "undeclared or missing candidate exports: "
            f"actual={sorted(actual)}, declared={sorted(declared)}"
        )
    return CandidateOutput(
        game=expected_game,
        target_level=expected_level,
        parent_checkpoint_sha256=parent_checkpoint_sha256,
        candidate_path=tuple(actions),
        exported_files_sha256=dict(declared),
    )


@dataclass(frozen=True)
class PromotionAdmission:
    game: str
    target_level: int
    authoritative_target: int
    parent_checkpoint_sha256: str
    checkpoint_sha256: str
    source_tree_sha256: str
    receipt_sha256: str
    receipt_bytes: bytes
    candidate_manifest_bytes: bytes


def _safe_host_relative_file(value: object) -> bool:
    if not isinstance(value, str) or not value:
        return False
    path = Path(value)
    return (
        not path.is_absolute()
        and ".." not in path.parts
        and path.name != HOST_RECEIPT_NAME
    )


def _validate_manifest_chain(
    source: Path,
    *,
    game: str,
    reached: int,
    current_checkpoint_sha256: str,
) -> dict[str, Any]:
    """Recompute the complete, contiguous promotion-manifest chain.

    A boolean ``manifest_chain`` receipt check is necessary provenance, but it
    is not sufficient recovery evidence.  Re-parse and hash every retained
    boundary so a buggy receipt writer cannot turn a missing historical level
    into an apparently complete contiguous lineage.
    """
    evidence_root = source / "promotion_evidence"
    if evidence_root.is_symlink() or not evidence_root.is_dir():
        raise SupervisorContractError(
            "promotion evidence root is missing or non-regular"
        )
    manifests = {
        path.parent.name: path
        for path in evidence_root.glob("level_*/manifest.json")
        if path.is_file()
    }
    expected_names = {f"level_{level:02d}" for level in range(1, reached + 1)}
    if set(manifests) != expected_names:
        raise SupervisorContractError(
            "promotion manifest chain is not exactly complete through "
            f"level {reached}; present={sorted(manifests)}, "
            f"expected={sorted(expected_names)}"
        )

    previous_path: str | None = None
    previous_hash: str | None = None
    current: dict[str, Any] | None = None
    for level in range(1, reached + 1):
        relative = Path(
            "promotion_evidence", f"level_{level:02d}", "manifest.json"
        )
        path = source / relative
        manifest = _read_json(path)
        promoted_files = manifest.get("promoted_files_sha256")
        if (
            manifest.get("schema") != 1
            or isinstance(manifest.get("schema"), bool)
            or manifest.get("game") != game
            or not isinstance(manifest.get("level"), int)
            or isinstance(manifest.get("level"), bool)
            or manifest.get("level") != level
            or manifest.get("validated") is not True
            or manifest.get("taint_verdict") != "clean"
            or not isinstance(promoted_files, dict)
            or not _is_sha256_hex(promoted_files.get(CHECKPOINT_NAME))
            or manifest.get("parent_manifest") != previous_path
            or manifest.get("parent_manifest_sha256") != previous_hash
        ):
            raise SupervisorContractError(
                f"promotion manifest chain is invalid at level {level}"
            )
        files_root = path.parent / "files"
        if files_root.is_symlink() or not files_root.is_dir():
            raise SupervisorContractError(
                f"promotion evidence files are missing at level {level}"
            )
        for relative_name, expected_hash in promoted_files.items():
            if (
                not isinstance(relative_name, str)
                or not _safe_host_relative_file(relative_name)
                or not _is_sha256_hex(expected_hash)
            ):
                raise SupervisorContractError(
                    f"promotion manifest has an invalid promoted file "
                    f"at level {level}: {relative_name!r}"
                )
            evidence_file = files_root / relative_name
            if (
                evidence_file.is_symlink()
                or not evidence_file.is_file()
                or _sha256_file(evidence_file) != expected_hash
            ):
                raise SupervisorContractError(
                    f"promotion manifest evidence mismatch at level "
                    f"{level}: {relative_name}"
                )
        boundary_path = files_root / CHECKPOINT_NAME
        boundary = load_trusted_checkpoint(
            boundary_path,
            expected_game=game,
            authoritative_target=reached,
        )
        if boundary.reached != level:
            raise SupervisorContractError(
                f"promotion manifest checkpoint is not the exact level "
                f"{level} boundary"
            )
        previous_path = relative.as_posix()
        previous_hash = _sha256_file(path)
        current = manifest

    assert current is not None
    if (
        current["promoted_files_sha256"].get(CHECKPOINT_NAME)
        != current_checkpoint_sha256
    ):
        raise SupervisorContractError(
            "current promotion manifest does not bind the clean exact checkpoint"
        )
    return current


def validate_promotion_receipt(
    receipt_path: Path,
    source: Path,
    *,
    frontier: FrontierAdmission,
    candidate_output_root: Path,
) -> PromotionAdmission:
    """Bind every host verification gate to the exact bytes being published."""
    if receipt_path.is_symlink() or not receipt_path.is_file():
        raise SupervisorContractError(
            "promotion receipt must be a regular host-owned file"
        )
    _validate_regular_tree(source, label="promotion source")
    if (source / HOST_RECEIPT_NAME).exists():
        raise SupervisorContractError(
            "promotion source must not preinstall a host receipt"
        )

    receipt_bytes = _read_regular_bytes(receipt_path)
    try:
        data = json.loads(receipt_bytes)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise SupervisorContractError(
            f"invalid JSON: {receipt_path}"
        ) from exc
    if not isinstance(data, dict):
        raise SupervisorContractError(
            f"expected JSON object: {receipt_path}"
        )

    candidate = validate_candidate_output(
        candidate_output_root,
        expected_game=frontier.game,
        expected_level=frontier.next_level,
        parent_checkpoint_sha256=frontier.parent_checkpoint_sha256,
    )
    candidate_manifest_path = candidate_output_root / CANDIDATE_NAME
    if (
        set(data) != PROMOTION_RECEIPT_FIELDS
        or not isinstance(data["schema"], int)
        or isinstance(data["schema"], bool)
        or data["schema"] != 1
    ):
        raise SupervisorContractError("promotion receipt schema mismatch")
    if (
        data["game"] != frontier.game
        or not isinstance(data["target_level"], int)
        or isinstance(data["target_level"], bool)
        or data["target_level"] != frontier.next_level
        or not isinstance(data["authoritative_target"], int)
        or isinstance(data["authoritative_target"], bool)
        or data["authoritative_target"] != frontier.authoritative_target
        or data["parent_checkpoint_sha256"]
        != frontier.parent_checkpoint_sha256
        or not isinstance(data["parent_action_count"], int)
        or isinstance(data["parent_action_count"], bool)
        or data["parent_action_count"] != frontier.parent_action_count
        or not isinstance(data["remaining_action_budget"], int)
        or isinstance(data["remaining_action_budget"], bool)
        or data["remaining_action_budget"] != frontier.remaining_action_budget
        or not isinstance(data["fresh_prefix_required"], bool)
        or data["fresh_prefix_required"] != frontier.fresh_prefix_required
    ):
        raise SupervisorContractError(
            "promotion receipt targets a stale or wrong frontier"
        )
    for field in (
        "candidate_manifest_sha256",
        "checkpoint_sha256",
        "source_tree_sha256",
        "winning_source_sha256",
        "promotion_manifest_sha256",
    ):
        if not _is_sha256_hex(data[field]):
            raise SupervisorContractError(
                f"promotion receipt has invalid {field}"
            )
    candidate_manifest_bytes = _read_regular_bytes(candidate_manifest_path)
    if data["candidate_manifest_sha256"] != hashlib.sha256(
        candidate_manifest_bytes
    ).hexdigest():
        raise SupervisorContractError("promotion receipt candidate hash mismatch")

    checks = data["checks"]
    if not isinstance(checks, dict) or set(checks) != REQUIRED_PROMOTION_CHECKS:
        raise SupervisorContractError("promotion receipt check schema mismatch")
    failed = sorted(
        name for name in REQUIRED_PROMOTION_CHECKS
        if checks.get(name) is not True
    )
    if failed:
        raise SupervisorContractError(
            f"promotion blocked by failed host checks: {failed}"
        )

    exact_path = data["exact_path"]
    if (
        not isinstance(exact_path, list)
        or not exact_path
        or len(exact_path) > MAX_REPLAY_ACTIONS
        or not all(_valid_replay_action(action) for action in exact_path)
    ):
        raise SupervisorContractError(
            "promotion receipt contains an invalid exact path"
        )
    candidate_actions = list(candidate.candidate_path)
    if (
        len(exact_path) > len(candidate_actions)
        or candidate_actions[:len(exact_path)] != exact_path
    ):
        raise SupervisorContractError(
            "exact replay boundary is not a prefix of the candidate path"
        )

    checkpoint_path = source / CHECKPOINT_NAME
    checkpoint = load_trusted_checkpoint(
        checkpoint_path,
        expected_game=frontier.game,
        authoritative_target=frontier.authoritative_target,
    )
    if (
        checkpoint.reached != frontier.next_level
        or list(checkpoint.final_path) != exact_path
    ):
        raise SupervisorContractError(
            "rebuilt checkpoint does not equal the exact next-level boundary"
        )
    checkpoint_hash = _sha256_file(checkpoint_path)
    if data["checkpoint_sha256"] != checkpoint_hash:
        raise SupervisorContractError("promotion receipt checkpoint hash mismatch")

    for path_field, hash_field in (
        ("winning_source_path", "winning_source_sha256"),
        ("promotion_manifest_path", "promotion_manifest_sha256"),
    ):
        relative = data[path_field]
        if not _safe_host_relative_file(relative):
            raise SupervisorContractError(
                f"promotion receipt has invalid {path_field}"
            )
        path = source / relative
        if (
            path.is_symlink()
            or not path.is_file()
            or _sha256_file(path) != data[hash_field]
        ):
            raise SupervisorContractError(
                f"promotion receipt evidence mismatch: {relative}"
            )
    winning_relative = Path(data["winning_source_path"])
    manifest_relative = Path(data["promotion_manifest_path"])
    expected_evidence_dir = Path(
        "promotion_evidence", f"level_{frontier.next_level:02d}"
    )
    if (
        winning_relative.suffix != ".py"
        or expected_evidence_dir not in winning_relative.parents
        or manifest_relative.name != "manifest.json"
        or manifest_relative.parent != expected_evidence_dir
    ):
        raise SupervisorContractError(
            "promotion receipt points to the wrong evidence boundary"
        )
    if (
        data["winning_source_sha256"]
        not in candidate.exported_files_sha256.values()
    ):
        raise SupervisorContractError(
            "winning source snapshot is not one of the declared "
            "candidate exports"
        )
    _validate_manifest_chain(
        source,
        game=frontier.game,
        reached=frontier.next_level,
        current_checkpoint_sha256=checkpoint_hash,
    )

    source_hash = _tree_hash(source)
    if data["source_tree_sha256"] != source_hash:
        raise SupervisorContractError("promotion receipt source-tree mismatch")
    return PromotionAdmission(
        game=frontier.game,
        target_level=frontier.next_level,
        authoritative_target=frontier.authoritative_target,
        parent_checkpoint_sha256=frontier.parent_checkpoint_sha256,
        checkpoint_sha256=checkpoint_hash,
        source_tree_sha256=source_hash,
        receipt_sha256=hashlib.sha256(receipt_bytes).hexdigest(),
        receipt_bytes=receipt_bytes,
        candidate_manifest_bytes=candidate_manifest_bytes,
    )


def _validate_embedded_promotion_receipt(
    version: Path,
    *,
    pointer: dict[str, Any],
    authoritative_target: int,
    checkpoint: TrustedCheckpoint,
) -> None:
    """Revalidate durable receipt/evidence instead of trusting pointer hashes.

    A pointer hash only proves that selected bytes are self-consistent.  It must
    not let a recovery path accept a rewritten receipt whose own gate schema,
    exact boundary, or evidence hashes are incomplete.  The host-captured
    candidate manifest is retained beside the exact winning boundary so
    recovery can re-establish candidate-prefix and winning-export lineage.
    """
    receipt_path = version / HOST_RECEIPT_NAME
    receipt = _read_json(receipt_path)
    if (
        set(receipt) != PROMOTION_RECEIPT_FIELDS
        or not isinstance(receipt["schema"], int)
        or isinstance(receipt["schema"], bool)
        or receipt["schema"] != 1
    ):
        raise SupervisorContractError(
            "embedded promotion receipt schema mismatch"
        )
    if (
        not isinstance(receipt["target_level"], int)
        or isinstance(receipt["target_level"], bool)
        or not isinstance(receipt["authoritative_target"], int)
        or isinstance(receipt["authoritative_target"], bool)
        or receipt["game"] != pointer["game"]
        or receipt["target_level"] != pointer["target_level"]
        or receipt["authoritative_target"] != authoritative_target
        or receipt["parent_checkpoint_sha256"]
        != pointer["parent_checkpoint_sha256"]
        or receipt["checkpoint_sha256"] != pointer["checkpoint_sha256"]
        or not isinstance(receipt["parent_action_count"], int)
        or isinstance(receipt["parent_action_count"], bool)
        or not 0 <= receipt["parent_action_count"] <= MAX_REPLAY_ACTIONS
        or not isinstance(receipt["remaining_action_budget"], int)
        or isinstance(receipt["remaining_action_budget"], bool)
        or receipt["remaining_action_budget"]
        != MAX_REPLAY_ACTIONS - receipt["parent_action_count"]
        or not isinstance(receipt["fresh_prefix_required"], bool)
        or receipt["fresh_prefix_required"]
        != (receipt["remaining_action_budget"] == 0)
    ):
        raise SupervisorContractError(
            "embedded promotion receipt disagrees with its selected frontier"
        )
    for field in (
        "candidate_manifest_sha256",
        "checkpoint_sha256",
        "source_tree_sha256",
        "winning_source_sha256",
        "promotion_manifest_sha256",
    ):
        if not _is_sha256_hex(receipt[field]):
            raise SupervisorContractError(
                f"embedded promotion receipt has invalid {field}"
            )
    checks = receipt["checks"]
    if (
        not isinstance(checks, dict)
        or set(checks) != REQUIRED_PROMOTION_CHECKS
        or any(checks[name] is not True for name in REQUIRED_PROMOTION_CHECKS)
    ):
        raise SupervisorContractError(
            "embedded promotion receipt has missing or failed host checks"
        )
    exact_path = receipt["exact_path"]
    if (
        not isinstance(exact_path, list)
        or not exact_path
        or len(exact_path) > MAX_REPLAY_ACTIONS
        or not all(_valid_replay_action(action) for action in exact_path)
        or list(checkpoint.final_path) != exact_path
    ):
        raise SupervisorContractError(
            "embedded promotion receipt does not bind the exact checkpoint path"
        )
    for path_field, hash_field in (
        ("winning_source_path", "winning_source_sha256"),
        ("promotion_manifest_path", "promotion_manifest_sha256"),
    ):
        relative = receipt[path_field]
        if not _safe_host_relative_file(relative):
            raise SupervisorContractError(
                f"embedded promotion receipt has invalid {path_field}"
            )
        path = version / relative
        if (
            path.is_symlink()
            or not path.is_file()
            or _sha256_file(path) != receipt[hash_field]
        ):
            raise SupervisorContractError(
                f"embedded promotion evidence mismatch: {relative}"
            )
    winning_relative = Path(receipt["winning_source_path"])
    manifest_relative = Path(receipt["promotion_manifest_path"])
    expected_evidence_dir = Path(
        "promotion_evidence", f"level_{pointer['target_level']:02d}"
    )
    candidate_relative = expected_evidence_dir / CANDIDATE_EVIDENCE_NAME
    candidate_path = version / candidate_relative
    if (
        candidate_path.is_symlink()
        or not candidate_path.is_file()
        or _sha256_file(candidate_path)
        != receipt["candidate_manifest_sha256"]
    ):
        raise SupervisorContractError(
            "embedded promotion candidate-manifest evidence mismatch"
        )
    candidate = _read_json(candidate_path)
    candidate_fields = {
        "schema",
        "game",
        "target_level",
        "parent_checkpoint_sha256",
        "candidate_path",
        "exported_files_sha256",
    }
    candidate_actions = candidate.get("candidate_path")
    declared_exports = candidate.get("exported_files_sha256")
    if (
        set(candidate) != candidate_fields
        or candidate.get("schema") != 1
        or isinstance(candidate.get("schema"), bool)
        or candidate.get("game") != pointer["game"]
        or candidate.get("target_level") != pointer["target_level"]
        or isinstance(candidate.get("target_level"), bool)
        or candidate.get("parent_checkpoint_sha256")
        != pointer["parent_checkpoint_sha256"]
        or not isinstance(candidate_actions, list)
        or not candidate_actions
        or len(candidate_actions) > MAX_REPLAY_ACTIONS
        or not all(_valid_replay_action(action) for action in candidate_actions)
        or len(receipt["exact_path"]) > len(candidate_actions)
        or candidate_actions[:len(receipt["exact_path"])]
        != receipt["exact_path"]
        or not isinstance(declared_exports, dict)
        or receipt["winning_source_sha256"]
        not in declared_exports.values()
    ):
        raise SupervisorContractError(
            "embedded candidate manifest disagrees with the selected boundary"
        )
    if (
        winning_relative.suffix != ".py"
        or expected_evidence_dir not in winning_relative.parents
        or manifest_relative.name != "manifest.json"
        or manifest_relative.parent != expected_evidence_dir
    ):
        raise SupervisorContractError(
            "embedded promotion receipt points to the wrong evidence boundary"
        )
    _validate_manifest_chain(
        version,
        game=pointer["game"],
        reached=pointer["target_level"],
        current_checkpoint_sha256=pointer["checkpoint_sha256"],
    )
    source_hash = _tree_hash(
        version,
        exclude_relative=frozenset({
            HOST_RECEIPT_NAME,
            candidate_relative.as_posix(),
        }),
        exclude_prefixes=(WINNING_SOURCE_NAME,),
    )
    if source_hash != receipt["source_tree_sha256"]:
        raise SupervisorContractError(
            "embedded promotion receipt source-tree mismatch"
        )


def _operator_lease_canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise SupervisorContractError(
            "operator-lease evidence is not canonical JSON"
        ) from exc


def _operator_lease_strict_json(
    raw: bytes, *, label: str
) -> dict[str, Any]:
    if not raw or len(raw) > 64 * 1024:
        raise SupervisorContractError(
            f"{label} is empty or exceeds its byte bound"
        )

    def no_duplicates(
        pairs: list[tuple[str, object]],
    ) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise SupervisorContractError(
                    f"{label} contains a duplicate JSON key"
                )
            result[key] = value
        return result

    try:
        value = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=no_duplicates,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(token)
            ),
        )
    except SupervisorContractError:
        raise
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise SupervisorContractError(
            f"{label} is not strict canonical JSON"
        ) from exc
    if (
        not isinstance(value, dict)
        or _operator_lease_canonical_json(value) + b"\n" != raw
    ):
        raise SupervisorContractError(
            f"{label} is not a canonical JSON object"
        )
    return value


def _operator_lease_hmac(
    key: bytes, body: Mapping[str, object]
) -> str:
    return hmac.new(
        key,
        _operator_lease_canonical_json(dict(body)),
        hashlib.sha256,
    ).hexdigest()


def _operator_lease_authenticated(
    key: bytes,
    value: Mapping[str, object],
    *,
    label: str,
) -> dict[str, object]:
    if set(value) <= {"host_authentication_sha256"}:
        raise SupervisorContractError(
            f"{label} lacks authenticated fields"
        )
    selected = dict(value)
    observed = selected.pop("host_authentication_sha256", None)
    if (
        not _is_sha256_hex(observed)
        or not hmac.compare_digest(
            str(observed),
            _operator_lease_hmac(key, selected),
        )
    ):
        raise SupervisorContractError(
            f"{label} is not host authenticated"
        )
    return selected


def _operator_lease_process_start_identity(pid: int) -> str:
    """Bind one live PID to its observed OS start time without signalling it."""

    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 1:
        raise SupervisorContractError(
            "operator lease requires a non-init process PID"
        )
    try:
        observed = subprocess.run(
            (
                "/bin/ps",
                "-p",
                str(pid),
                "-o",
                "pid=,lstart=",
            ),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env={"LANG": "C", "LC_ALL": "C"},
            text=True,
            timeout=10,
            check=False,
            shell=False,
            close_fds=True,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise SupervisorContractError(
            "cannot observe operator process-start identity"
        ) from exc
    rows = [row.strip() for row in observed.stdout.splitlines() if row.strip()]
    if (
        observed.returncode != 0
        or observed.stderr
        or len(rows) != 1
    ):
        raise SupervisorContractError(
            "operator process-start identity observation failed"
        )
    parts = rows[0].split(None, 1)
    if len(parts) != 2 or parts[0] != str(pid) or not parts[1]:
        raise SupervisorContractError(
            "operator process-start identity response is malformed"
        )
    return hashlib.sha256(
        _operator_lease_canonical_json({
            "pid": pid,
            "os_process_start": parts[1],
        })
    ).hexdigest()


def _operator_lease_private_directory(
    path: Path, *, create: bool, label: str
) -> None:
    selected = Path(path)
    if not selected.is_absolute():
        raise SupervisorContractError(
            f"{label} must be an absolute path"
        )
    _reject_symlinked_path_components(
        selected.parent, label=f"{label} parent"
    )
    if create:
        try:
            selected.mkdir(mode=0o700)
            _fsync_directory(selected.parent)
        except FileExistsError:
            pass
        except OSError as exc:
            raise SupervisorContractError(
                f"cannot create {label}"
            ) from exc
    try:
        metadata = selected.stat(follow_symlinks=False)
    except OSError as exc:
        raise SupervisorContractError(
            f"{label} is unavailable"
        ) from exc
    if (
        selected.is_symlink()
        or not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.getuid()
        or stat.S_IMODE(metadata.st_mode) != 0o700
    ):
        raise SupervisorContractError(
            f"{label} must be an owner-held mode-0700 directory"
        )


def _operator_lease_atomic_json(
    directory: Path,
    name: str,
    value: Mapping[str, object],
) -> tuple[Path, str]:
    if re.fullmatch(r"[A-Za-z0-9_.-]{1,160}", name) is None:
        raise SupervisorContractError(
            "operator lease target name is malformed"
        )
    raw = _operator_lease_canonical_json(dict(value)) + b"\n"
    temporary = directory / (
        f".{name}.{secrets.token_hex(8)}.tmp"
    )
    target = directory / name
    _write_new_regular_bytes(
        temporary, raw, label="operator lease temporary receipt"
    )
    try:
        if target.is_symlink():
            raise SupervisorContractError(
                "operator lease target is a symlink"
            )
        os.replace(temporary, target)
        _fsync_directory(directory)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise
    reopened = _read_regular_bytes(target)
    if reopened != raw:
        raise SupervisorContractError(
            "operator lease receipt changed after publication"
        )
    return target, hashlib.sha256(raw).hexdigest()


class OperatorLease:
    """Kernel-held, host-authenticated ownership for one campaign operator.

    The stable lock inode is never replaced.  Authenticated acquisition and
    rotating heartbeat receipts are separate, so a crash cannot make a new
    inode appear unlocked while an old process still owns the original lock.
    Stale takeover relies only on acquiring that kernel lock; it never signals
    or otherwise trusts the prior PID, and therefore cannot kill a reused PID.
    """

    def __init__(
        self,
        campaign_root: Path,
        *,
        operator_configuration_sha256: str,
        acquire_timeout_seconds: float = (
            OPERATOR_LEASE_ACQUIRE_TIMEOUT_SECONDS
        ),
        heartbeat_interval_seconds: float = (
            OPERATOR_LEASE_HEARTBEAT_SECONDS
        ),
        clock_ns: Any = time.time_ns,
        monotonic: Any = time.monotonic,
        sleeper: Any = time.sleep,
    ):
        if (
            not _is_sha256_hex(operator_configuration_sha256)
            or isinstance(acquire_timeout_seconds, bool)
            or not isinstance(acquire_timeout_seconds, (int, float))
            or not math.isfinite(float(acquire_timeout_seconds))
            or not 0 <= acquire_timeout_seconds <= 60
            or isinstance(heartbeat_interval_seconds, bool)
            or not isinstance(heartbeat_interval_seconds, (int, float))
            or not math.isfinite(float(heartbeat_interval_seconds))
            or not 0.05 <= heartbeat_interval_seconds <= 60
        ):
            raise SupervisorContractError(
                "operator lease configuration is malformed"
            )
        self.campaign_root = Path(campaign_root)
        self.operator_configuration_sha256 = (
            operator_configuration_sha256
        )
        self.acquire_timeout_seconds = float(acquire_timeout_seconds)
        self.heartbeat_interval_seconds = float(
            heartbeat_interval_seconds
        )
        self.clock_ns = clock_ns
        self.monotonic = monotonic
        self.sleeper = sleeper
        self.root = (
            self.campaign_root / OPERATOR_LEASE_ROOT_NAME
        )
        self.acquisitions = self.root / "acquisitions"
        self.lock_path = self.root / "lease.lock"
        self.key_path = self.root / "host_authentication.key"
        self.current_path = self.root / "current.json"
        self._lock_descriptor: int | None = None
        self._authentication_key: bytes | None = None
        self._owner_instance_id: str | None = None
        self._owner_pid: int | None = None
        self._owner_process_start_identity: str | None = None
        self._acquisition_sequence: int | None = None
        self._acquisition_path: Path | None = None
        self._acquisition_sha256: str | None = None
        self._heartbeat_sequence = -1
        self._thread: threading.Thread | None = None
        self._thread_stop = threading.Event()
        self._thread_error: BaseException | None = None
        self._state_lock = threading.Lock()
        self._released = False

    @property
    def owner_instance_id(self) -> str:
        if self._owner_instance_id is None:
            raise SupervisorContractError(
                "operator lease has not been acquired"
            )
        return self._owner_instance_id

    @property
    def acquisition_sha256(self) -> str:
        if self._acquisition_sha256 is None:
            raise SupervisorContractError(
                "operator lease has no acquisition receipt"
            )
        return self._acquisition_sha256

    @classmethod
    def observe_current(
        cls,
        campaign_root: Path,
        *,
        operator_configuration_sha256: str,
    ) -> dict[str, object]:
        """Read one authenticated heartbeat without acquiring or mutating."""

        observer = cls(
            campaign_root,
            operator_configuration_sha256=(
                operator_configuration_sha256
            ),
        )
        _operator_lease_private_directory(
            observer.campaign_root,
            create=False,
            label="operator campaign root",
        )
        _operator_lease_private_directory(
            observer.root,
            create=False,
            label="operator lease root",
        )
        _operator_lease_private_directory(
            observer.acquisitions,
            create=False,
            label="operator lease acquisition root",
        )
        if observer.key_path.is_symlink():
            raise SupervisorContractError(
                "operator lease authentication key is a symlink"
            )
        key = _read_regular_bytes(observer.key_path)
        metadata = observer.key_path.stat(follow_symlinks=False)
        if (
            len(key) != 32
            or metadata.st_uid != os.getuid()
            or stat.S_IMODE(metadata.st_mode) != 0o600
        ):
            raise SupervisorContractError(
                "operator lease authentication key is malformed"
            )
        observer._authentication_key = key
        (
            acquisition,
            acquisition_sha256,
            heartbeat,
            heartbeat_sha256,
        ) = observer._load_prior()
        if (
            acquisition is None
            or acquisition_sha256 is None
            or heartbeat is None
            or heartbeat_sha256 is None
        ):
            raise SupervisorContractError(
                "operator lease has no current authenticated owner"
            )
        return {
            "acquisition": acquisition,
            "acquisition_sha256": acquisition_sha256,
            "heartbeat": heartbeat,
            "heartbeat_sha256": heartbeat_sha256,
        }

    def _open_lock(self) -> None:
        _operator_lease_private_directory(
            self.campaign_root,
            create=True,
            label="operator campaign root",
        )
        _operator_lease_private_directory(
            self.root,
            create=True,
            label="operator lease root",
        )
        _operator_lease_private_directory(
            self.acquisitions,
            create=True,
            label="operator lease acquisition root",
        )
        try:
            descriptor = os.open(
                self.lock_path,
                os.O_RDWR
                | os.O_CREAT
                | getattr(os, "O_NOFOLLOW", 0),
                0o600,
            )
        except OSError as exc:
            raise SupervisorContractError(
                "operator lease lock cannot be opened"
            ) from exc
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or metadata.st_uid != os.getuid()
            or stat.S_IMODE(metadata.st_mode) != 0o600
        ):
            os.close(descriptor)
            raise SupervisorContractError(
                "operator lease lock is not an unaliased owner-held file"
            )
        deadline = self.monotonic() + self.acquire_timeout_seconds
        while True:
            try:
                fcntl.flock(
                    descriptor,
                    fcntl.LOCK_EX | fcntl.LOCK_NB,
                )
                self._lock_descriptor = descriptor
                return
            except BlockingIOError:
                remaining = deadline - self.monotonic()
                if remaining <= 0:
                    os.close(descriptor)
                    raise SupervisorContractError(
                        "another live contiguous operator owns the "
                        "campaign lease"
                    )
                self.sleeper(min(0.05, remaining))

    def _load_or_create_key(self) -> bytes:
        if self.key_path.exists() or self.key_path.is_symlink():
            if self.key_path.is_symlink():
                raise SupervisorContractError(
                    "operator lease authentication key is a symlink"
                )
            raw = _read_regular_bytes(self.key_path)
            metadata = self.key_path.stat(follow_symlinks=False)
            if (
                len(raw) != 32
                or metadata.st_uid != os.getuid()
                or stat.S_IMODE(metadata.st_mode) != 0o600
            ):
                raise SupervisorContractError(
                    "operator lease authentication key is malformed"
                )
            return raw
        key = secrets.token_bytes(32)
        _write_new_regular_bytes(
            self.key_path,
            key,
            label="operator lease authentication key",
        )
        _fsync_directory(self.root)
        return key

    def _read_authenticated(
        self, path: Path, *, label: str
    ) -> tuple[dict[str, object], str]:
        if self._authentication_key is None:
            raise SupervisorContractError(
                "operator lease authentication key is unavailable"
            )
        raw = _read_regular_bytes(path)
        value = _operator_lease_strict_json(raw, label=label)
        body = _operator_lease_authenticated(
            self._authentication_key,
            value,
            label=label,
        )
        return body, hashlib.sha256(raw).hexdigest()

    def _load_prior(
        self,
    ) -> tuple[
        dict[str, object] | None,
        str | None,
        dict[str, object] | None,
        str | None,
    ]:
        if not (self.current_path.exists() or self.current_path.is_symlink()):
            return None, None, None, None
        if self.current_path.is_symlink():
            raise SupervisorContractError(
                "operator lease current pointer is a symlink"
            )
        pointer, _pointer_sha256 = self._read_authenticated(
            self.current_path,
            label="operator lease current pointer",
        )
        expected_pointer_keys = {
            "schema",
            "kind",
            "campaign_root",
            "operator_configuration_sha256",
            "owner_instance_id",
            "acquisition_sequence",
            "acquisition_path",
            "acquisition_sha256",
            "heartbeat_sequence",
            "heartbeat_path",
            "heartbeat_sha256",
            "status",
        }
        if (
            set(pointer) != expected_pointer_keys
            or pointer["schema"] != OPERATOR_LEASE_SCHEMA
            or pointer["kind"]
            != "arc_agi3_contiguous_operator_lease_current"
            or pointer["campaign_root"] != str(self.campaign_root)
            or pointer["operator_configuration_sha256"]
            != self.operator_configuration_sha256
            or not isinstance(pointer["owner_instance_id"], str)
            or re.fullmatch(
                r"[0-9a-f]{32}",
                str(pointer["owner_instance_id"]),
            )
            is None
            or isinstance(pointer["acquisition_sequence"], bool)
            or not isinstance(pointer["acquisition_sequence"], int)
            or pointer["acquisition_sequence"] < 0
            or isinstance(pointer["heartbeat_sequence"], bool)
            or not isinstance(pointer["heartbeat_sequence"], int)
            or pointer["heartbeat_sequence"] < 0
            or not _is_sha256_hex(pointer["acquisition_sha256"])
            or not _is_sha256_hex(pointer["heartbeat_sha256"])
            or pointer["status"] not in {"ACTIVE", "RELEASED"}
        ):
            raise SupervisorContractError(
                "operator lease current pointer is malformed"
            )
        acquisition_path = Path(str(pointer["acquisition_path"]))
        heartbeat_path = Path(str(pointer["heartbeat_path"]))
        expected_acquisition = self.acquisitions / (
            f"{int(pointer['acquisition_sequence']):08d}-"
            f"{pointer['owner_instance_id']}.json"
        )
        expected_heartbeat = self.root / (
            f"heartbeat_{int(pointer['heartbeat_sequence']) % 2}.json"
        )
        if (
            acquisition_path != expected_acquisition
            or heartbeat_path != expected_heartbeat
        ):
            raise SupervisorContractError(
                "operator lease current pointer escapes its exact paths"
            )
        acquisition, acquisition_sha256 = self._read_authenticated(
            acquisition_path,
            label="operator lease acquisition receipt",
        )
        heartbeat, heartbeat_sha256 = self._read_authenticated(
            heartbeat_path,
            label="operator lease heartbeat receipt",
        )
        expected_acquisition_keys = {
            "schema",
            "kind",
            "campaign_root",
            "operator_configuration_sha256",
            "owner_instance_id",
            "owner_pid",
            "owner_process_start_identity_sha256",
            "acquisition_sequence",
            "acquired_at_ns",
            "takeover",
            "takeover_authority",
            "prior_owner_instance_id",
            "prior_acquisition_sha256",
            "prior_heartbeat_sha256",
            "prior_heartbeat_status",
            "recovered_orphan_acquisition_sha256s",
            "signals_prior_pid",
            "lock_device",
            "lock_inode",
            "status",
        }
        expected_heartbeat_keys = {
            "schema",
            "kind",
            "campaign_root",
            "operator_configuration_sha256",
            "owner_instance_id",
            "owner_pid",
            "owner_process_start_identity_sha256",
            "acquisition_sequence",
            "acquisition_path",
            "acquisition_sha256",
            "heartbeat_sequence",
            "heartbeat_at_ns",
            "status",
        }
        if (
            set(acquisition) != expected_acquisition_keys
            or set(heartbeat) != expected_heartbeat_keys
            or acquisition.get("schema") != OPERATOR_LEASE_SCHEMA
            or acquisition.get("kind")
            != "arc_agi3_contiguous_operator_lease_acquisition"
            or heartbeat.get("schema") != OPERATOR_LEASE_SCHEMA
            or heartbeat.get("kind")
            != "arc_agi3_contiguous_operator_lease_heartbeat"
            or acquisition.get("campaign_root")
            != str(self.campaign_root)
            or heartbeat.get("campaign_root")
            != str(self.campaign_root)
            or acquisition.get("operator_configuration_sha256")
            != self.operator_configuration_sha256
            or heartbeat.get("operator_configuration_sha256")
            != self.operator_configuration_sha256
            or not _is_sha256_hex(
                acquisition.get(
                    "owner_process_start_identity_sha256"
                )
            )
            or heartbeat.get(
                "owner_process_start_identity_sha256"
            )
            != acquisition.get(
                "owner_process_start_identity_sha256"
            )
            or heartbeat.get("owner_pid")
            != acquisition.get("owner_pid")
            or heartbeat.get("acquisition_sequence")
            != acquisition.get("acquisition_sequence")
            or heartbeat.get("acquisition_path")
            != str(acquisition_path)
            or acquisition.get("status") != "ACTIVE"
            or acquisition.get("signals_prior_pid") is not False
            or acquisition.get("takeover_authority")
            != "kernel_lock_absence_plus_authenticated_prior_receipt"
            or not isinstance(
                acquisition.get(
                    "recovered_orphan_acquisition_sha256s"
                ),
                list,
            )
            or any(
                not _is_sha256_hex(item)
                for item in acquisition.get(
                    "recovered_orphan_acquisition_sha256s", []
                )
            )
            or acquisition.get(
                "recovered_orphan_acquisition_sha256s"
            )
            != sorted(set(acquisition.get(
                "recovered_orphan_acquisition_sha256s", []
            )))
            or not isinstance(acquisition.get("takeover"), bool)
            or isinstance(acquisition.get("owner_pid"), bool)
            or not isinstance(acquisition.get("owner_pid"), int)
            or int(acquisition.get("owner_pid")) <= 1
            or isinstance(acquisition.get("acquired_at_ns"), bool)
            or not isinstance(acquisition.get("acquired_at_ns"), int)
            or int(acquisition.get("acquired_at_ns")) <= 0
            or isinstance(heartbeat.get("heartbeat_at_ns"), bool)
            or not isinstance(heartbeat.get("heartbeat_at_ns"), int)
            or int(heartbeat.get("heartbeat_at_ns")) <= 0
            or acquisition_sha256
            != pointer["acquisition_sha256"]
            or heartbeat_sha256 != pointer["heartbeat_sha256"]
            or acquisition.get("owner_instance_id")
            != pointer["owner_instance_id"]
            or acquisition.get("acquisition_sequence")
            != pointer["acquisition_sequence"]
            or heartbeat.get("owner_instance_id")
            != pointer["owner_instance_id"]
            or heartbeat.get("acquisition_sha256")
            != pointer["acquisition_sha256"]
            or heartbeat.get("heartbeat_sequence")
            != pointer["heartbeat_sequence"]
            or heartbeat.get("status") != pointer["status"]
        ):
            raise SupervisorContractError(
                "operator lease current pointer has substituted evidence"
            )
        return (
            acquisition,
            acquisition_sha256,
            heartbeat,
            heartbeat_sha256,
        )

    def _unaccounted_acquisition_receipts(
        self,
        current_acquisition_sha256: str | None,
    ) -> tuple[str, ...]:
        """Authenticate the history and surface crash-orphaned acquisitions."""

        rows: dict[str, dict[str, object]] = {}
        paths: dict[str, Path] = {}
        for path in sorted(self.acquisitions.iterdir()):
            if (
                path.is_symlink()
                or not path.is_file()
                or re.fullmatch(
                    r"[0-9]{8}-[0-9a-f]{32}\.json",
                    path.name,
                )
                is None
            ):
                raise SupervisorContractError(
                    "operator lease acquisition history contains an "
                    "unexpected entry"
                )
            row, digest = self._read_authenticated(
                path,
                label="operator lease acquisition history receipt",
            )
            if (
                digest in rows
                or row.get("kind")
                != "arc_agi3_contiguous_operator_lease_acquisition"
                or row.get("schema") != OPERATOR_LEASE_SCHEMA
                or row.get("campaign_root") != str(self.campaign_root)
                or row.get("operator_configuration_sha256")
                != self.operator_configuration_sha256
                or isinstance(
                    row.get("acquisition_sequence"), bool
                )
                or not isinstance(
                    row.get("acquisition_sequence"), int
                )
                or int(row.get("acquisition_sequence")) < 0
                or not isinstance(
                    row.get("owner_instance_id"), str
                )
                or re.fullmatch(
                    r"[0-9a-f]{32}",
                    str(row.get("owner_instance_id")),
                )
                is None
                or path.name
                != (
                    f"{int(row.get('acquisition_sequence', -1)):08d}-"
                    f"{row.get('owner_instance_id')}.json"
                )
                or not isinstance(
                    row.get(
                        "recovered_orphan_acquisition_sha256s"
                    ),
                    list,
                )
                or any(
                    not _is_sha256_hex(item)
                    for item in row.get(
                        "recovered_orphan_acquisition_sha256s", []
                    )
                )
                or row.get(
                    "recovered_orphan_acquisition_sha256s"
                )
                != sorted(set(row.get(
                    "recovered_orphan_acquisition_sha256s", []
                )))
            ):
                raise SupervisorContractError(
                    "operator lease acquisition history is malformed"
                )
            rows[digest] = row
            paths[digest] = path
        del paths
        chain: set[str] = set()
        acknowledged: set[str] = set()
        cursor = current_acquisition_sha256
        while cursor is not None:
            if cursor in chain or cursor not in rows:
                raise SupervisorContractError(
                    "operator lease acquisition history chain is broken"
                )
            chain.add(cursor)
            row = rows[cursor]
            acknowledged.update(row[
                "recovered_orphan_acquisition_sha256s"
            ])
            prior = row.get("prior_acquisition_sha256")
            if prior is not None and not _is_sha256_hex(prior):
                raise SupervisorContractError(
                    "operator lease acquisition predecessor is malformed"
                )
            cursor = None if prior is None else str(prior)
        if (
            not acknowledged.issubset(rows)
            or acknowledged & chain
        ):
            raise SupervisorContractError(
                "operator lease orphan acknowledgements are forged"
            )
        return tuple(sorted(set(rows) - chain - acknowledged))

    def _publish_acquisition(
        self,
        *,
        prior_acquisition: Mapping[str, object] | None,
        prior_acquisition_sha256: str | None,
        prior_heartbeat: Mapping[str, object] | None,
        prior_heartbeat_sha256: str | None,
        recovered_orphan_acquisition_sha256s: Sequence[str],
    ) -> None:
        assert self._authentication_key is not None
        assert self._lock_descriptor is not None
        acquisition_sequence = (
            0
            if prior_acquisition is None
            else int(prior_acquisition["acquisition_sequence"]) + 1
        )
        if acquisition_sequence >= MAX_OPERATOR_LEASE_ACQUISITIONS:
            raise SupervisorContractError(
                "operator lease acquisition history is exhausted"
            )
        owner_instance_id = secrets.token_hex(16)
        owner_pid = os.getpid()
        process_start = _operator_lease_process_start_identity(
            owner_pid
        )
        lock_metadata = os.fstat(self._lock_descriptor)
        body: dict[str, object] = {
            "schema": OPERATOR_LEASE_SCHEMA,
            "kind": "arc_agi3_contiguous_operator_lease_acquisition",
            "campaign_root": str(self.campaign_root),
            "operator_configuration_sha256":
                self.operator_configuration_sha256,
            "owner_instance_id": owner_instance_id,
            "owner_pid": owner_pid,
            "owner_process_start_identity_sha256": process_start,
            "acquisition_sequence": acquisition_sequence,
            "acquired_at_ns": int(self.clock_ns()),
            "takeover": prior_acquisition is not None,
            "takeover_authority":
                "kernel_lock_absence_plus_authenticated_prior_receipt",
            "prior_owner_instance_id": (
                None
                if prior_acquisition is None
                else prior_acquisition["owner_instance_id"]
            ),
            "prior_acquisition_sha256": prior_acquisition_sha256,
            "prior_heartbeat_sha256": prior_heartbeat_sha256,
            "prior_heartbeat_status": (
                None
                if prior_heartbeat is None
                else prior_heartbeat["status"]
            ),
            "recovered_orphan_acquisition_sha256s": list(
                recovered_orphan_acquisition_sha256s
            ),
            "signals_prior_pid": False,
            "lock_device": lock_metadata.st_dev,
            "lock_inode": lock_metadata.st_ino,
            "status": "ACTIVE",
        }
        value = {
            **body,
            "host_authentication_sha256":
                _operator_lease_hmac(
                    self._authentication_key, body
                ),
        }
        path = self.acquisitions / (
            f"{acquisition_sequence:08d}-{owner_instance_id}.json"
        )
        raw = _operator_lease_canonical_json(value) + b"\n"
        _write_new_regular_bytes(
            path,
            raw,
            label="operator lease acquisition receipt",
        )
        _fsync_directory(self.acquisitions)
        self._owner_instance_id = owner_instance_id
        self._owner_pid = owner_pid
        self._owner_process_start_identity = process_start
        self._acquisition_sequence = acquisition_sequence
        self._acquisition_path = path
        self._acquisition_sha256 = hashlib.sha256(raw).hexdigest()

    def _publish_heartbeat(self, *, status: str) -> dict[str, object]:
        if status not in {"ACTIVE", "RELEASED"}:
            raise SupervisorContractError(
                "operator lease heartbeat status is invalid"
            )
        if (
            self._authentication_key is None
            or self._owner_instance_id is None
            or self._owner_pid is None
            or self._owner_process_start_identity is None
            or self._acquisition_sequence is None
            or self._acquisition_path is None
            or self._acquisition_sha256 is None
        ):
            raise SupervisorContractError(
                "operator lease heartbeat precedes acquisition"
            )
        if os.getpid() != self._owner_pid:
            raise SupervisorContractError(
                "operator lease cannot survive a process fork"
            )
        # A live process cannot have its own PID reused.  The independent
        # watchdog re-observes the OS start identity before acting on a PID;
        # avoiding a new ``ps`` subprocess on every heartbeat keeps a
        # multi-day campaign bounded.
        self._heartbeat_sequence += 1
        body: dict[str, object] = {
            "schema": OPERATOR_LEASE_SCHEMA,
            "kind": "arc_agi3_contiguous_operator_lease_heartbeat",
            "campaign_root": str(self.campaign_root),
            "operator_configuration_sha256":
                self.operator_configuration_sha256,
            "owner_instance_id": self._owner_instance_id,
            "owner_pid": self._owner_pid,
            "owner_process_start_identity_sha256":
                self._owner_process_start_identity,
            "acquisition_sequence": self._acquisition_sequence,
            "acquisition_path": str(self._acquisition_path),
            "acquisition_sha256": self._acquisition_sha256,
            "heartbeat_sequence": self._heartbeat_sequence,
            "heartbeat_at_ns": int(self.clock_ns()),
            "status": status,
        }
        heartbeat = {
            **body,
            "host_authentication_sha256":
                _operator_lease_hmac(
                    self._authentication_key, body
                ),
        }
        heartbeat_path, heartbeat_sha256 = (
            _operator_lease_atomic_json(
                self.root,
                f"heartbeat_{self._heartbeat_sequence % 2}.json",
                heartbeat,
            )
        )
        pointer_body: dict[str, object] = {
            "schema": OPERATOR_LEASE_SCHEMA,
            "kind": "arc_agi3_contiguous_operator_lease_current",
            "campaign_root": str(self.campaign_root),
            "operator_configuration_sha256":
                self.operator_configuration_sha256,
            "owner_instance_id": self._owner_instance_id,
            "acquisition_sequence": self._acquisition_sequence,
            "acquisition_path": str(self._acquisition_path),
            "acquisition_sha256": self._acquisition_sha256,
            "heartbeat_sequence": self._heartbeat_sequence,
            "heartbeat_path": str(heartbeat_path),
            "heartbeat_sha256": heartbeat_sha256,
            "status": status,
        }
        pointer = {
            **pointer_body,
            "host_authentication_sha256":
                _operator_lease_hmac(
                    self._authentication_key, pointer_body
                ),
        }
        _operator_lease_atomic_json(
            self.root, "current.json", pointer
        )
        return pointer

    def _heartbeat_main(self) -> None:
        while not self._thread_stop.wait(
            self.heartbeat_interval_seconds
        ):
            try:
                with self._state_lock:
                    if self._released:
                        return
                    self._publish_heartbeat(status="ACTIVE")
            except BaseException as exc:
                self._thread_error = exc
                self._thread_stop.set()
                return

    def acquire(self) -> "OperatorLease":
        if self._lock_descriptor is not None:
            raise SupervisorContractError(
                "operator lease cannot be acquired twice"
            )
        try:
            self._open_lock()
            self._authentication_key = self._load_or_create_key()
            (
                prior_acquisition,
                prior_acquisition_sha256,
                prior_heartbeat,
                prior_heartbeat_sha256,
            ) = self._load_prior()
            recovered_orphans = (
                self._unaccounted_acquisition_receipts(
                    prior_acquisition_sha256
                )
            )
            self._publish_acquisition(
                prior_acquisition=prior_acquisition,
                prior_acquisition_sha256=(
                    prior_acquisition_sha256
                ),
                prior_heartbeat=prior_heartbeat,
                prior_heartbeat_sha256=prior_heartbeat_sha256,
                recovered_orphan_acquisition_sha256s=(
                    recovered_orphans
                ),
            )
            self._publish_heartbeat(status="ACTIVE")
            self._thread = threading.Thread(
                target=self._heartbeat_main,
                name="arc-agi3-contiguous-operator-heartbeat",
                daemon=True,
            )
            self._thread.start()
            return self
        except BaseException:
            self._unlock()
            raise

    def assert_healthy(self) -> None:
        if self._lock_descriptor is None or self._released:
            raise SupervisorContractError(
                "operator lease is not active"
            )
        if self._thread_error is not None:
            raise SupervisorContractError(
                "operator lease heartbeat failed"
            ) from self._thread_error
        if self._thread is None or not self._thread.is_alive():
            raise SupervisorContractError(
                "operator lease heartbeat thread is absent"
            )

    def _unlock(self) -> None:
        descriptor = self._lock_descriptor
        self._lock_descriptor = None
        if descriptor is not None:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)

    def release(self) -> None:
        if self._lock_descriptor is None:
            return
        self._thread_stop.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(
                timeout=max(
                    1.0, self.heartbeat_interval_seconds * 2
                )
            )
            if thread.is_alive():
                raise SupervisorContractError(
                    "operator lease heartbeat did not quiesce"
                )
        try:
            if self._thread_error is not None:
                raise SupervisorContractError(
                    "operator lease heartbeat failed before release"
                ) from self._thread_error
            with self._state_lock:
                self._publish_heartbeat(status="RELEASED")
                self._released = True
        finally:
            self._unlock()

    def __enter__(self) -> "OperatorLease":
        return self.acquire()

    def __exit__(self, *_args: object) -> None:
        self.release()


def _post_incident_meta_read(
    path: Path,
    *,
    label: str,
    maximum: int,
    allow_empty: bool = False,
) -> bytes:
    """Read one bounded owner-held diagnostic artifact without aliases."""

    _reject_symlinked_path_components(path, label=label)
    try:
        descriptor = os.open(
            path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        )
    except OSError as exc:
        raise SupervisorContractError(
            f"{label} is unavailable as an unaliased file"
        ) from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_uid != os.getuid()
            or stat.S_IMODE(before.st_mode) not in {0o400, 0o600}
            or before.st_size > maximum
            or (not allow_empty and before.st_size == 0)
        ):
            raise SupervisorContractError(
                f"{label} is not a bounded owner-held regular file"
            )
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            block = os.read(descriptor, min(64 * 1024, remaining))
            if not block:
                raise SupervisorContractError(
                    f"{label} changed while being read"
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
            raise SupervisorContractError(
                f"{label} changed while being read"
            )
        if stat.S_IMODE(after.st_mode) != 0o400:
            os.fchmod(descriptor, 0o400)
            os.fsync(descriptor)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _post_incident_meta_write(
    path: Path, value: Mapping[str, object], *, label: str
) -> tuple[Path, str]:
    raw = _operator_lease_canonical_json(dict(value)) + b"\n"
    if len(raw) > POST_INCIDENT_META_MAX_CONTROL_BYTES:
        raise SupervisorContractError(
            f"{label} exceeds its byte bound"
        )
    _write_new_regular_bytes(path, raw, label=label)
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        os.fchmod(descriptor, 0o400)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent)
    if _post_incident_meta_read(
        path,
        label=label,
        maximum=POST_INCIDENT_META_MAX_CONTROL_BYTES,
    ) != raw:
        raise SupervisorContractError(
            f"{label} changed after publication"
        )
    return path, hashlib.sha256(raw).hexdigest()


def _validate_post_incident_meta_projection(
    value: object,
) -> dict[str, object]:
    """Admit only a redacted controller-substrate incident projection."""

    if not isinstance(value, Mapping):
        raise SupervisorContractError(
            "post-incident meta projection is not an object"
        )
    selected = dict(value)
    expected = {
        "schema",
        "kind",
        "operator_incident",
        "substrate_incident",
        "incident_event_sequence",
        "incident_event_digest",
    }
    operator_fields = {
        "attempt_id",
        "operation",
        "fault_domain",
        "operation_consecutive",
        "domain_consecutive",
        "threshold",
        "reason_code",
    }
    substrate_fields = {
        "attempt_id",
        "substrate_identity_sha256",
        "failure_receipt_sha256",
        "failure_class",
        "failure_code",
        "health_probe_count",
        "attempted_remediation_epochs_sha256",
        "last_health_probe_sha256",
    }
    operator = selected.get("operator_incident")
    substrate = selected.get("substrate_incident")
    if (
        set(selected) != expected
        or selected.get("schema") != POST_INCIDENT_META_SCHEMA
        or selected.get("kind")
        != "arc_agi3_contiguous_substrate_incident_projection"
        or not isinstance(operator, Mapping)
        or set(operator) != operator_fields
        or not isinstance(substrate, Mapping)
        or set(substrate) != substrate_fields
        or operator.get("operation")
        != "substrate_health_reprobe"
        or operator.get("fault_domain") != "controller_substrate"
        or operator.get("attempt_id") != substrate.get("attempt_id")
        or not _is_sha256_hex(
            substrate.get("substrate_identity_sha256")
        )
        or not _is_sha256_hex(
            substrate.get("failure_receipt_sha256")
        )
        or not _is_sha256_hex(
            substrate.get("attempted_remediation_epochs_sha256")
        )
        or (
            substrate.get("last_health_probe_sha256") is not None
            and not _is_sha256_hex(
                substrate.get("last_health_probe_sha256")
            )
        )
        or isinstance(selected.get("incident_event_sequence"), bool)
        or not isinstance(selected.get("incident_event_sequence"), int)
        or int(selected.get("incident_event_sequence")) < 0
        or not _is_sha256_hex(selected.get("incident_event_digest"))
    ):
        raise SupervisorContractError(
            "post-incident meta projection is not the exact sealed "
            "controller-substrate incident"
        )
    for field in (
        "operation_consecutive",
        "domain_consecutive",
        "threshold",
    ):
        item = operator.get(field)
        if (
            isinstance(item, bool)
            or not isinstance(item, int)
            or item <= 0
        ):
            raise SupervisorContractError(
                "post-incident meta circuit coordinate is malformed"
            )
    for item in (
        operator.get("attempt_id"),
        operator.get("reason_code"),
        substrate.get("attempt_id"),
        substrate.get("failure_class"),
        substrate.get("failure_code"),
    ):
        if (
            not isinstance(item, str)
            or re.fullmatch(r"[A-Za-z0-9_.:-]{1,160}", item) is None
        ):
            raise SupervisorContractError(
                "post-incident meta identifier is malformed"
            )
    health_probe_count = substrate.get("health_probe_count")
    if (
        isinstance(health_probe_count, bool)
        or not isinstance(health_probe_count, int)
        or health_probe_count < 1
    ):
        raise SupervisorContractError(
            "post-incident meta health-probe count is malformed"
        )
    return selected


def _post_incident_meta_identity(
    projection: Mapping[str, object],
) -> str:
    """Content identity for one immutable incident, excluding probe progress."""

    selected = _validate_post_incident_meta_projection(projection)
    operator = dict(selected["operator_incident"])
    substrate = dict(selected["substrate_incident"])
    identity = {
        "schema": POST_INCIDENT_META_SCHEMA,
        "operator_incident": operator,
        "substrate_incident": {
            name: substrate[name]
            for name in (
                "attempt_id",
                "substrate_identity_sha256",
                "failure_receipt_sha256",
                "failure_class",
                "failure_code",
            )
        },
        "incident_event_sequence":
            selected["incident_event_sequence"],
        "incident_event_digest":
            selected["incident_event_digest"],
    }
    return hashlib.sha256(
        _operator_lease_canonical_json(identity)
    ).hexdigest()


class PostIncidentMetaDiagnostic:
    """At-most-once quarantined diagnosis after a durable substrate incident.

    The lifecycle deliberately cannot return a scheduler decision, retry
    authorization, WIP, cost settlement, or promotion.  Its sole output is an
    immutable diagnostic receipt that the formal operator binds into the
    already-paused OPERATOR_INCIDENT.
    """

    def __init__(
        self,
        campaign_root: Path,
        *,
        operator_configuration_sha256: str,
        driver_executable: Path,
        driver_executable_sha256: str,
        driver_configuration: Path,
        driver_configuration_sha256: str,
        driver_attestation_sha256: str,
        operation_timeout_seconds: int,
        command_runner: object,
    ) -> None:
        self.campaign_root = Path(campaign_root).resolve()
        self.collection_root = (
            self.campaign_root / POST_INCIDENT_META_ROOT_NAME
        )
        self.incident_identity_sha256: str | None = None
        self.root = self.collection_root
        self.invocation_root = self.root / "invocation_0001"
        self.request_path = self.root / "request.json"
        self.intent_path = self.root / "invocation_intent.json"
        self.terminal_path = self.root / "terminal.json"
        self.response_path = self.invocation_root / "response.json"
        self.stdout_path = self.invocation_root / "stdout.bin"
        self.stderr_path = self.invocation_root / "stderr.bin"
        self.invocation_receipt_path = (
            self.invocation_root / "invocation_receipt.json"
        )
        self.operator_configuration_sha256 = (
            operator_configuration_sha256
        )
        self.driver_executable = Path(driver_executable)
        self.driver_executable_sha256 = driver_executable_sha256
        self.driver_configuration = Path(driver_configuration)
        self.driver_configuration_sha256 = (
            driver_configuration_sha256
        )
        self.driver_attestation_sha256 = (
            driver_attestation_sha256
        )
        self.operation_timeout_seconds = operation_timeout_seconds
        self.command_runner = command_runner
        if (
            not _is_sha256_hex(operator_configuration_sha256)
            or not _is_sha256_hex(driver_executable_sha256)
            or not _is_sha256_hex(driver_configuration_sha256)
            or not _is_sha256_hex(driver_attestation_sha256)
            or isinstance(operation_timeout_seconds, bool)
            or not isinstance(operation_timeout_seconds, int)
            or not 5 <= operation_timeout_seconds <= 3600
            or not callable(
                getattr(command_runner, "run_attached_stream", None)
            )
        ):
            raise SupervisorContractError(
                "post-incident meta driver configuration is malformed"
            )

    def _select_episode(
        self, projection: Mapping[str, object]
    ) -> str:
        identity = _post_incident_meta_identity(projection)
        if (
            self.incident_identity_sha256 is not None
            and self.incident_identity_sha256 != identity
        ):
            raise SupervisorContractError(
                "post-incident meta object cannot change incident identity"
            )
        self.incident_identity_sha256 = identity
        self.root = self.collection_root / identity
        self.invocation_root = self.root / "invocation_0001"
        self.request_path = self.root / "request.json"
        self.intent_path = self.root / "invocation_intent.json"
        self.terminal_path = self.root / "terminal.json"
        self.response_path = self.invocation_root / "response.json"
        self.stdout_path = self.invocation_root / "stdout.bin"
        self.stderr_path = self.invocation_root / "stderr.bin"
        self.invocation_receipt_path = (
            self.invocation_root / "invocation_receipt.json"
        )
        return identity

    @staticmethod
    def _response(
        raw: bytes,
        *,
        request_sha256: str,
    ) -> dict[str, object]:
        value = _operator_lease_strict_json(
            raw, label="post-incident meta response"
        )
        fields = {
            "schema",
            "kind",
            "protocol_sha256",
            "request_sha256",
            "status",
            "diagnosis_code",
            "diagnosis_summary",
            "socratic_challenge",
            "recommended_operator_action",
            "scheduler_authority",
            "solver_authority",
            "wip_authority",
            "cost_authority",
            "retry_authority",
            "dispatch_authority",
            "promotion_authority",
        }
        if (
            set(value) != fields
            or value.get("schema") != POST_INCIDENT_META_SCHEMA
            or value.get("kind")
            != "arc_agi3_contiguous_post_incident_meta_response"
            or value.get("protocol_sha256")
            != POST_INCIDENT_META_PROTOCOL_SHA256
            or value.get("request_sha256") != request_sha256
            or value.get("status") != "DIAGNOSED"
            or not isinstance(value.get("diagnosis_code"), str)
            or re.fullmatch(
                r"[a-z][a-z0-9_]{0,127}",
                str(value.get("diagnosis_code")),
            )
            is None
            or value.get("recommended_operator_action")
            not in POST_INCIDENT_META_RECOMMENDATIONS
            or any(
                value.get(name) is not False
                for name in (
                    "scheduler_authority",
                    "solver_authority",
                    "wip_authority",
                    "cost_authority",
                    "retry_authority",
                    "dispatch_authority",
                    "promotion_authority",
                )
            )
        ):
            raise SupervisorContractError(
                "post-incident meta response is not the exact "
                "quarantine-only schema"
            )
        for field in ("diagnosis_summary", "socratic_challenge"):
            text = value.get(field)
            if (
                not isinstance(text, str)
                or not text
                or "\x00" in text
                or len(text.encode("utf-8")) > 4096
            ):
                raise SupervisorContractError(
                    f"post-incident meta {field} is malformed"
                )
        return value

    def _request(
        self, incident_projection: Mapping[str, object]
    ) -> tuple[dict[str, object], bytes, str]:
        projection = _validate_post_incident_meta_projection(
            incident_projection
        )
        incident_identity_sha256 = self._select_episode(projection)
        request = {
            "schema": POST_INCIDENT_META_SCHEMA,
            "kind":
                "arc_agi3_contiguous_post_incident_meta_request",
            "protocol_sha256":
                POST_INCIDENT_META_PROTOCOL_SHA256,
            "operator_configuration_sha256":
                self.operator_configuration_sha256,
            "driver_executable_sha256":
                self.driver_executable_sha256,
            "driver_configuration_sha256":
                self.driver_configuration_sha256,
            "driver_attestation_sha256":
                self.driver_attestation_sha256,
            "invocation_sequence": 1,
            "incident_identity_sha256":
                incident_identity_sha256,
            "incident_projection": projection,
            "incident_projection_sha256": hashlib.sha256(
                _operator_lease_canonical_json(projection)
            ).hexdigest(),
            "input_contains_game_source": False,
            "input_contains_wip": False,
            "input_contains_candidate": False,
            "result_authority": "quarantine_only",
        }
        raw = _operator_lease_canonical_json(request) + b"\n"
        return request, raw, hashlib.sha256(raw).hexdigest()

    def _validate_request_raw(
        self, raw: bytes
    ) -> dict[str, object]:
        value = _operator_lease_strict_json(
            raw, label="post-incident meta request"
        )
        projection = value.get("incident_projection")
        if (
            set(value)
            != {
                "schema",
                "kind",
                "protocol_sha256",
                "operator_configuration_sha256",
                "driver_executable_sha256",
                "driver_configuration_sha256",
                "driver_attestation_sha256",
                "invocation_sequence",
                "incident_identity_sha256",
                "incident_projection",
                "incident_projection_sha256",
                "input_contains_game_source",
                "input_contains_wip",
                "input_contains_candidate",
                "result_authority",
            }
            or value.get("schema") != POST_INCIDENT_META_SCHEMA
            or value.get("kind")
            != "arc_agi3_contiguous_post_incident_meta_request"
            or value.get("protocol_sha256")
            != POST_INCIDENT_META_PROTOCOL_SHA256
            or value.get("operator_configuration_sha256")
            != self.operator_configuration_sha256
            or value.get("driver_executable_sha256")
            != self.driver_executable_sha256
            or value.get("driver_configuration_sha256")
            != self.driver_configuration_sha256
            or value.get("driver_attestation_sha256")
            != self.driver_attestation_sha256
            or value.get("invocation_sequence") != 1
            or value.get("incident_identity_sha256")
            != self.incident_identity_sha256
            or value.get("input_contains_game_source") is not False
            or value.get("input_contains_wip") is not False
            or value.get("input_contains_candidate") is not False
            or value.get("result_authority") != "quarantine_only"
            or not isinstance(projection, Mapping)
        ):
            raise SupervisorContractError(
                "post-incident meta request is malformed"
            )
        validated_projection = (
            _validate_post_incident_meta_projection(projection)
        )
        if (
            _post_incident_meta_identity(validated_projection)
            != self.incident_identity_sha256
            or value.get("incident_projection_sha256")
            != hashlib.sha256(
                _operator_lease_canonical_json(
                    validated_projection
                )
            ).hexdigest()
        ):
            raise SupervisorContractError(
                "post-incident meta request incident binding changed"
            )
        return value

    def _ensure_layout(self) -> None:
        _operator_lease_private_directory(
            self.collection_root,
            create=True,
            label="post-incident meta collection root",
        )
        episode_names = []
        for path in self.collection_root.iterdir():
            if (
                path.is_symlink()
                or not path.is_dir()
                or re.fullmatch(r"[0-9a-f]{64}", path.name) is None
            ):
                raise SupervisorContractError(
                    "post-incident meta collection contains an "
                    "unexpected entry"
                )
            episode_names.append(path.name)
        if (
            self.incident_identity_sha256 is None
            or (
                self.incident_identity_sha256 not in episode_names
                and len(episode_names)
                >= POST_INCIDENT_META_MAX_EPISODES
            )
        ):
            raise SupervisorContractError(
                "post-incident meta episode bound is exhausted"
            )
        _operator_lease_private_directory(
            self.root,
            create=True,
            label="post-incident meta episode root",
        )
        _operator_lease_private_directory(
            self.invocation_root,
            create=True,
            label="post-incident meta invocation root",
        )
        allowed_root = {
            "request.json",
            "invocation_intent.json",
            "terminal.json",
            "invocation_0001",
        }
        allowed_invocation = {
            "response.json",
            "stdout.bin",
            "stderr.bin",
            "invocation_receipt.json",
        }
        if (
            any(path.name not in allowed_root for path in self.root.iterdir())
            or any(
                path.name not in allowed_invocation
                for path in self.invocation_root.iterdir()
            )
        ):
            raise SupervisorContractError(
                "post-incident meta evidence contains an unexpected entry"
            )

    def _publish_terminal(
        self,
        *,
        request_sha256: str,
        status: str,
        failure_code: str | None,
        invocation_receipt_sha256: str | None,
        response_sha256: str | None,
        recommended_operator_action: str | None,
    ) -> dict[str, object]:
        if status not in {
            "DIAGNOSED",
            "FAILED",
            "AMBIGUOUS_INTERRUPTION",
        }:
            raise SupervisorContractError(
                "post-incident meta terminal status is malformed"
            )
        body: dict[str, object] = {
            "schema": POST_INCIDENT_META_SCHEMA,
            "kind":
                "arc_agi3_contiguous_post_incident_meta_terminal",
            "status": status,
            "failure_code": failure_code,
            "protocol_sha256":
                POST_INCIDENT_META_PROTOCOL_SHA256,
            "operator_configuration_sha256":
                self.operator_configuration_sha256,
            "incident_identity_sha256":
                self.incident_identity_sha256,
            "request_path": str(self.request_path),
            "request_sha256": request_sha256,
            "invocation_sequence": 1,
            "invocation_receipt_path": (
                None
                if invocation_receipt_sha256 is None
                else str(self.invocation_receipt_path)
            ),
            "invocation_receipt_sha256":
                invocation_receipt_sha256,
            "response_path": (
                None
                if response_sha256 is None
                else str(self.response_path)
            ),
            "response_sha256": response_sha256,
            "recommended_operator_action":
                recommended_operator_action,
            "diagnostic_available": status == "DIAGNOSED",
            "human_intervention_required": True,
            "runner_remained_paused": True,
            "meta_proposer_invocation_count": 1,
            "scheduler_authority": False,
            "solver_authority": False,
            "wip_authority": False,
            "cost_authority": False,
            "retry_authority": False,
            "dispatch_authority": False,
            "promotion_authority": False,
        }
        value = {
            **body,
            "receipt_sha256": hashlib.sha256(
                _operator_lease_canonical_json(body)
            ).hexdigest(),
        }
        _post_incident_meta_write(
            self.terminal_path,
            value,
            label="post-incident meta terminal receipt",
        )
        return self._load_terminal(request_sha256=request_sha256)

    def _load_terminal(
        self, *, request_sha256: str
    ) -> dict[str, object]:
        raw = _post_incident_meta_read(
            self.terminal_path,
            label="post-incident meta terminal receipt",
            maximum=POST_INCIDENT_META_MAX_CONTROL_BYTES,
        )
        value = _operator_lease_strict_json(
            raw, label="post-incident meta terminal receipt"
        )
        selected = dict(value)
        observed_receipt = selected.pop("receipt_sha256", None)
        authority_fields = (
            "scheduler_authority",
            "solver_authority",
            "wip_authority",
            "cost_authority",
            "retry_authority",
            "dispatch_authority",
            "promotion_authority",
        )
        expected_fields = {
            "schema",
            "kind",
            "status",
            "failure_code",
            "protocol_sha256",
            "operator_configuration_sha256",
            "incident_identity_sha256",
            "request_path",
            "request_sha256",
            "invocation_sequence",
            "invocation_receipt_path",
            "invocation_receipt_sha256",
            "response_path",
            "response_sha256",
            "recommended_operator_action",
            "diagnostic_available",
            "human_intervention_required",
            "runner_remained_paused",
            "meta_proposer_invocation_count",
            *authority_fields,
            "receipt_sha256",
        }
        if (
            set(value) != expected_fields
            or value.get("schema") != POST_INCIDENT_META_SCHEMA
            or value.get("kind")
            != "arc_agi3_contiguous_post_incident_meta_terminal"
            or value.get("status")
            not in {
                "DIAGNOSED",
                "FAILED",
                "AMBIGUOUS_INTERRUPTION",
            }
            or value.get("protocol_sha256")
            != POST_INCIDENT_META_PROTOCOL_SHA256
            or value.get("operator_configuration_sha256")
            != self.operator_configuration_sha256
            or value.get("incident_identity_sha256")
            != self.incident_identity_sha256
            or not _is_sha256_hex(
                value.get("incident_identity_sha256")
            )
            or value.get("request_path") != str(self.request_path)
            or value.get("request_sha256") != request_sha256
            or value.get("invocation_sequence") != 1
            or value.get("human_intervention_required") is not True
            or value.get("runner_remained_paused") is not True
            or value.get("meta_proposer_invocation_count") != 1
            or any(value.get(name) is not False for name in authority_fields)
            or value.get("diagnostic_available")
            is not (value.get("status") == "DIAGNOSED")
            or (
                value.get("status") == "DIAGNOSED"
                and (
                    value.get("failure_code") is not None
                    or value.get("recommended_operator_action")
                    not in POST_INCIDENT_META_RECOMMENDATIONS
                    or not _is_sha256_hex(
                        value.get("invocation_receipt_sha256")
                    )
                    or not _is_sha256_hex(
                        value.get("response_sha256")
                    )
                )
            )
            or (
                value.get("status") == "FAILED"
                and (
                    value.get("recommended_operator_action") is not None
                    or not isinstance(value.get("failure_code"), str)
                    or re.fullmatch(
                        r"[a-z][a-z0-9_]{0,127}",
                        str(value.get("failure_code")),
                    )
                    is None
                    or not _is_sha256_hex(
                        value.get("invocation_receipt_sha256")
                    )
                )
            )
            or (
                value.get("status") == "AMBIGUOUS_INTERRUPTION"
                and (
                    value.get("failure_code")
                    != "operator_interrupted_during_driver"
                    or value.get("invocation_receipt_path") is not None
                    or value.get("invocation_receipt_sha256") is not None
                    or value.get("response_path") is not None
                    or value.get("response_sha256") is not None
                    or value.get("recommended_operator_action") is not None
                )
            )
            or not _is_sha256_hex(observed_receipt)
            or not hmac.compare_digest(
                str(observed_receipt),
                hashlib.sha256(
                    _operator_lease_canonical_json(selected)
                ).hexdigest(),
            )
        ):
            raise SupervisorContractError(
                "post-incident meta terminal receipt is malformed"
            )
        request_raw = _post_incident_meta_read(
            self.request_path,
            label="post-incident meta request",
            maximum=POST_INCIDENT_META_MAX_CONTROL_BYTES,
        )
        if hashlib.sha256(request_raw).hexdigest() != request_sha256:
            raise SupervisorContractError(
                "post-incident meta terminal request binding changed"
            )
        invocation_sha256 = value.get(
            "invocation_receipt_sha256"
        )
        if invocation_sha256 is not None:
            invocation_raw = _post_incident_meta_read(
                self.invocation_receipt_path,
                label="post-incident meta invocation receipt",
                maximum=POST_INCIDENT_META_MAX_CONTROL_BYTES,
            )
            if (
                value.get("invocation_receipt_path")
                != str(self.invocation_receipt_path)
                or hashlib.sha256(invocation_raw).hexdigest()
                != invocation_sha256
            ):
                raise SupervisorContractError(
                    "post-incident meta invocation binding changed"
                )
        response_sha256 = value.get("response_sha256")
        if response_sha256 is not None:
            response_raw = _post_incident_meta_read(
                self.response_path,
                label="post-incident meta response",
                maximum=POST_INCIDENT_META_MAX_CONTROL_BYTES,
            )
            if (
                value.get("response_path") != str(self.response_path)
                or hashlib.sha256(response_raw).hexdigest()
                != response_sha256
            ):
                raise SupervisorContractError(
                    "post-incident meta response binding changed"
                )
            if value.get("diagnostic_available") is True:
                validated_response = self._response(
                    response_raw, request_sha256=request_sha256
                )
                if (
                    validated_response[
                        "recommended_operator_action"
                    ]
                    != value.get("recommended_operator_action")
                ):
                    raise SupervisorContractError(
                        "post-incident meta recommendation binding changed"
                    )
        return value

    def _finish_invocation(
        self,
        *,
        request_sha256: str,
        observed: object | None,
        failure_code: str | None,
    ) -> dict[str, object]:
        stream_bindings: dict[str, dict[str, object]] = {}
        for name, path in (
            ("stdout", self.stdout_path),
            ("stderr", self.stderr_path),
        ):
            if not (path.exists() or path.is_symlink()):
                _write_new_regular_bytes(
                    path,
                    b"",
                    label=f"post-incident meta {name}",
                )
            raw = _post_incident_meta_read(
                path,
                label=f"post-incident meta {name}",
                maximum=POST_INCIDENT_META_MAX_STREAM_BYTES,
                allow_empty=True,
            )
            stream_bindings[name] = {
                "path": str(path),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "bytes": len(raw),
            }
        response_sha256: str | None = None
        response_valid = False
        recommended_operator_action: str | None = None
        if self.response_path.exists() or self.response_path.is_symlink():
            response_raw = _post_incident_meta_read(
                self.response_path,
                label="post-incident meta response",
                maximum=POST_INCIDENT_META_MAX_CONTROL_BYTES,
            )
            response_sha256 = hashlib.sha256(response_raw).hexdigest()
            try:
                validated_response = self._response(
                    response_raw, request_sha256=request_sha256
                )
                response_valid = True
                recommended_operator_action = str(
                    validated_response["recommended_operator_action"]
                )
            except SupervisorContractError:
                failure_code = failure_code or "invalid_response"
        invocation_body = {
            "schema": POST_INCIDENT_META_SCHEMA,
            "kind":
                "arc_agi3_contiguous_post_incident_meta_invocation",
            "protocol_sha256":
                POST_INCIDENT_META_PROTOCOL_SHA256,
            "request_sha256": request_sha256,
            "incident_identity_sha256":
                self.incident_identity_sha256,
            "invocation_sequence": 1,
            "argv_shape": [
                "driver",
                "--configuration",
                "configuration",
                "--request",
                "request",
                "--response",
                "response",
            ],
            "returncode": (
                None if observed is None else observed.returncode
            ),
            "timed_out": (
                None if observed is None else observed.timed_out
            ),
            "output_overflow": (
                None if observed is None else observed.output_overflow
            ),
            "stdout": stream_bindings["stdout"],
            "stderr": stream_bindings["stderr"],
            "response_sha256": response_sha256,
            "response_valid": response_valid,
            "failure_code": failure_code,
            "raw_streams_host_only": True,
            "result_authority": "quarantine_only",
        }
        _, invocation_sha256 = _post_incident_meta_write(
            self.invocation_receipt_path,
            invocation_body,
            label="post-incident meta invocation receipt",
        )
        successful_observation = (
            observed is not None
            and observed.returncode == 0
            and observed.timed_out is False
            and observed.output_overflow is False
        )
        if successful_observation and response_valid:
            return self._publish_terminal(
                request_sha256=request_sha256,
                status="DIAGNOSED",
                failure_code=None,
                invocation_receipt_sha256=invocation_sha256,
                response_sha256=response_sha256,
                recommended_operator_action=(
                    recommended_operator_action
                ),
            )
        return self._publish_terminal(
            request_sha256=request_sha256,
            status="FAILED",
            failure_code=(
                failure_code
                or (
                    "driver_failed"
                    if observed is not None
                    else "driver_interrupted"
                )
            ),
            invocation_receipt_sha256=invocation_sha256,
            response_sha256=response_sha256,
            recommended_operator_action=None,
        )

    def _validate_intent(self, *, request_sha256: str) -> None:
        raw = _post_incident_meta_read(
            self.intent_path,
            label="post-incident meta invocation intent",
            maximum=POST_INCIDENT_META_MAX_CONTROL_BYTES,
        )
        value = _operator_lease_strict_json(
            raw, label="post-incident meta invocation intent"
        )
        if (
            set(value)
            != {
                "schema",
                "kind",
                "protocol_sha256",
                "request_sha256",
                "incident_identity_sha256",
                "invocation_sequence",
                "maximum_invocations",
                "timeout_seconds",
                "result_authority",
            }
            or value.get("schema") != POST_INCIDENT_META_SCHEMA
            or value.get("kind")
            != (
                "arc_agi3_contiguous_post_incident_meta_"
                "invocation_intent"
            )
            or value.get("protocol_sha256")
            != POST_INCIDENT_META_PROTOCOL_SHA256
            or value.get("request_sha256") != request_sha256
            or value.get("incident_identity_sha256")
            != self.incident_identity_sha256
            or value.get("invocation_sequence") != 1
            or value.get("maximum_invocations") != 1
            or value.get("timeout_seconds")
            != self.operation_timeout_seconds
            or value.get("result_authority") != "quarantine_only"
        ):
            raise SupervisorContractError(
                "post-incident meta invocation intent is malformed"
            )

    def run_once(
        self, incident_projection: Mapping[str, object]
    ) -> dict[str, object]:
        """Run or recover the one allowed quarantined diagnostic."""

        if (
            _sha256_file(self.driver_executable)
            != self.driver_executable_sha256
            or _sha256_file(self.driver_configuration)
            != self.driver_configuration_sha256
        ):
            raise SupervisorContractError(
                "post-incident meta driver controls changed"
            )
        executable_metadata = self.driver_executable.stat(
            follow_symlinks=False
        )
        if (
            not stat.S_ISREG(executable_metadata.st_mode)
            or executable_metadata.st_nlink != 1
            or not executable_metadata.st_mode & stat.S_IXUSR
        ):
            raise SupervisorContractError(
                "post-incident meta driver is not executable authority"
            )
        _operator_lease_private_directory(
            self.campaign_root,
            create=False,
            label="post-incident meta campaign root",
        )
        request, request_raw, request_sha256 = self._request(
            incident_projection
        )
        self._ensure_layout()
        if self.request_path.exists() or self.request_path.is_symlink():
            retained_request_raw = _post_incident_meta_read(
                self.request_path,
                label="post-incident meta request",
                maximum=POST_INCIDENT_META_MAX_CONTROL_BYTES,
            )
            self._validate_request_raw(retained_request_raw)
            if (
                not (
                    self.terminal_path.exists()
                    or self.terminal_path.is_symlink()
                )
                and retained_request_raw != request_raw
            ):
                raise SupervisorContractError(
                    "post-incident meta request changed across recovery"
                )
        else:
            _post_incident_meta_write(
                self.request_path,
                request,
                label="post-incident meta request",
            )
        if self.terminal_path.exists() or self.terminal_path.is_symlink():
            request_sha256 = hashlib.sha256(
                _post_incident_meta_read(
                    self.request_path,
                    label="post-incident meta request",
                    maximum=POST_INCIDENT_META_MAX_CONTROL_BYTES,
                )
            ).hexdigest()
            return self._load_terminal(
                request_sha256=request_sha256
            )
        if self.intent_path.exists() or self.intent_path.is_symlink():
            # The durable intent is written immediately before external
            # execution.  Its presence makes a second invocation illegal.  A
            # complete response can still be sealed; otherwise the outcome is
            # explicitly ambiguous and remains paused.
            self._validate_intent(request_sha256=request_sha256)
            if self.invocation_receipt_path.exists():
                raise SupervisorContractError(
                    "post-incident meta invocation receipt lacks terminal"
                )
            if (
                self.response_path.exists()
                and self.stdout_path.exists()
                and self.stderr_path.exists()
            ):
                return self._finish_invocation(
                    request_sha256=request_sha256,
                    observed=None,
                    failure_code="operator_interrupted_after_driver",
                )
            return self._publish_terminal(
                request_sha256=request_sha256,
                status="AMBIGUOUS_INTERRUPTION",
                failure_code="operator_interrupted_during_driver",
                invocation_receipt_sha256=None,
                response_sha256=None,
                recommended_operator_action=None,
            )
        intent = {
            "schema": POST_INCIDENT_META_SCHEMA,
            "kind":
                "arc_agi3_contiguous_post_incident_meta_invocation_intent",
            "protocol_sha256":
                POST_INCIDENT_META_PROTOCOL_SHA256,
            "request_sha256": request_sha256,
            "incident_identity_sha256":
                self.incident_identity_sha256,
            "invocation_sequence": 1,
            "maximum_invocations": 1,
            "timeout_seconds": self.operation_timeout_seconds,
            "result_authority": "quarantine_only",
        }
        _post_incident_meta_write(
            self.intent_path,
            intent,
            label="post-incident meta invocation intent",
        )
        argv = (
            str(self.driver_executable),
            "--configuration",
            str(self.driver_configuration),
            "--request",
            str(self.request_path),
            "--response",
            str(self.response_path),
        )
        observed: object | None = None
        failure_code: str | None = None
        try:
            observed = self.command_runner.run_attached_stream(
                argv,
                timeout_seconds=self.operation_timeout_seconds,
                stdout_path=self.stdout_path,
                stderr_path=self.stderr_path,
                stdout_limit_bytes=POST_INCIDENT_META_MAX_STREAM_BYTES,
                stderr_limit_bytes=POST_INCIDENT_META_MAX_STREAM_BYTES,
            )
        except BaseException:
            failure_code = "driver_invocation_exception"
        return self._finish_invocation(
            request_sha256=request_sha256,
            observed=observed,
            failure_code=failure_code,
        )


def verify_post_incident_meta_terminal_receipt(
    terminal_path: Path,
    *,
    expected_campaign_root: Path,
    expected_operator_configuration_sha256: str,
) -> dict[str, object]:
    """Reopen a complete real meta episode without invoking its driver."""

    campaign_root = Path(expected_campaign_root).resolve()
    terminal = Path(terminal_path).resolve()
    if (
        terminal.name != "terminal.json"
        or terminal.parent.parent
        != campaign_root / POST_INCIDENT_META_ROOT_NAME
        or re.fullmatch(r"[0-9a-f]{64}", terminal.parent.name) is None
        or not _is_sha256_hex(
            expected_operator_configuration_sha256
        )
    ):
        raise SupervisorContractError(
            "post-incident meta terminal escaped its exact episode"
        )
    request_path = terminal.parent / "request.json"
    intent_path = terminal.parent / "invocation_intent.json"
    request_raw = _post_incident_meta_read(
        request_path,
        label="post-incident meta request",
        maximum=POST_INCIDENT_META_MAX_CONTROL_BYTES,
    )
    request = _operator_lease_strict_json(
        request_raw, label="post-incident meta request"
    )
    intent = _operator_lease_strict_json(
        _post_incident_meta_read(
            intent_path,
            label="post-incident meta invocation intent",
            maximum=POST_INCIDENT_META_MAX_CONTROL_BYTES,
        ),
        label="post-incident meta invocation intent",
    )
    projection = request.get("incident_projection")
    timeout_seconds = intent.get("timeout_seconds")
    if (
        not isinstance(projection, Mapping)
        or request.get("operator_configuration_sha256")
        != expected_operator_configuration_sha256
        or not _is_sha256_hex(
            request.get("driver_executable_sha256")
        )
        or not _is_sha256_hex(
            request.get("driver_configuration_sha256")
        )
        or not _is_sha256_hex(
            request.get("driver_attestation_sha256")
        )
        or isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, int)
        or not 5 <= timeout_seconds <= 3600
    ):
        raise SupervisorContractError(
            "post-incident meta retained controls are malformed"
        )

    class _VerificationOnlyRunner:
        @staticmethod
        def run_attached_stream(*_args: object, **_kwargs: object) -> None:
            raise AssertionError("verification cannot invoke the meta driver")

    diagnostic = PostIncidentMetaDiagnostic(
        campaign_root,
        operator_configuration_sha256=(
            expected_operator_configuration_sha256
        ),
        driver_executable=campaign_root / ".verification-only-driver",
        driver_executable_sha256=str(
            request["driver_executable_sha256"]
        ),
        driver_configuration=(
            campaign_root / ".verification-only-configuration"
        ),
        driver_configuration_sha256=str(
            request["driver_configuration_sha256"]
        ),
        driver_attestation_sha256=str(
            request["driver_attestation_sha256"]
        ),
        operation_timeout_seconds=timeout_seconds,
        command_runner=_VerificationOnlyRunner(),
    )
    diagnostic._select_episode(projection)
    if (
        diagnostic.terminal_path != terminal
        or diagnostic.incident_identity_sha256 != terminal.parent.name
    ):
        raise SupervisorContractError(
            "post-incident meta episode identity changed"
        )
    diagnostic._validate_request_raw(request_raw)
    diagnostic._validate_intent(
        request_sha256=hashlib.sha256(request_raw).hexdigest()
    )
    value = diagnostic._load_terminal(
        request_sha256=hashlib.sha256(request_raw).hexdigest()
    )
    if (
        value.get("status") != "DIAGNOSED"
        or value.get("recommended_operator_action")
        != "REMATERIALIZE_AND_REPROBE_CONTROLLER_SUBSTRATE"
    ):
        raise SupervisorContractError(
            "post-incident meta terminal did not prove safe remediation"
        )
    return value


class GameLock:
    """Nonblocking host lock for one game in one contiguous lineage."""

    def __init__(self, root: Path, game: str):
        self.root = root
        self.game = game
        self.path = root / "locks" / f"{game}.lock"
        self.handle = None

    def __enter__(self) -> "GameLock":
        _reject_symlinked_path_components(
            self.root, label="game-lock root"
        )
        if self.root.is_symlink() or (
            self.root.exists() and not self.root.is_dir()
        ):
            raise SupervisorContractError(
                "game-lock root must be a regular host-owned directory"
            )
        if self.game not in authoritative_inventory():
            raise SupervisorContractError(
                f"cannot lock a game outside the authoritative inventory: "
                f"{self.game!r}"
            )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.parent.is_symlink() or not self.path.parent.is_dir():
            raise SupervisorContractError(
                "game-lock directory must be a regular host-owned directory"
            )
        try:
            descriptor = os.open(
                self.path,
                os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
                0o600,
            )
        except OSError as exc:
            raise SupervisorContractError(
                f"game lock must be a regular host-owned file: {self.path}"
            ) from exc
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            os.close(descriptor)
            raise SupervisorContractError(
                f"game lock must be an unaliased regular host-owned file: "
                f"{self.path}"
            )
        self.handle = os.fdopen(descriptor, "r+", encoding="utf-8")
        try:
            fcntl.flock(
                self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB
            )
        except BlockingIOError as exc:
            self.handle.close()
            self.handle = None
            raise SupervisorContractError(
                f"game already has a live supervisor: {self.path.stem}"
            ) from exc
        self.handle.seek(0)
        self.handle.truncate()
        self.handle.write(f"pid={os.getpid()}\n")
        self.handle.flush()
        return self

    def __exit__(self, *_args: object) -> None:
        assert self.handle is not None
        fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        self.handle.close()
        self.handle = None


class StoreLock:
    """Nonblocking lock that serializes pointer admission and replacement."""

    def __init__(self, root: Path):
        self.path = root / ".promotion.lock"
        self.handle = None

    def __enter__(self) -> "StoreLock":
        _reject_symlinked_path_components(
            self.path.parent, label="artifact-store lock directory"
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.parent.is_symlink() or not self.path.parent.is_dir():
            raise SupervisorContractError(
                "artifact-store lock directory must be regular and host-owned"
            )
        try:
            descriptor = os.open(
                self.path,
                os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
                0o600,
            )
        except OSError as exc:
            raise SupervisorContractError(
                "artifact-store lock must be a regular host-owned file"
            ) from exc
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            os.close(descriptor)
            raise SupervisorContractError(
                "artifact-store lock must be an unaliased regular "
                "host-owned file"
            )
        self.handle = os.fdopen(descriptor, "r+", encoding="utf-8")
        try:
            fcntl.flock(
                self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB
            )
        except BlockingIOError as exc:
            self.handle.close()
            self.handle = None
            raise SupervisorContractError(
                "artifact store already has a live publisher"
            ) from exc
        return self

    def __exit__(self, *_args: object) -> None:
        assert self.handle is not None
        fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        self.handle.close()
        self.handle = None


class VersionedArtifactStore:
    """Immutable versions selected by one atomically replaced pointer file."""

    def __init__(self, root: Path):
        self.root = root
        self.versions = root / "versions"
        self.staging = root / "staging"
        self.pointer = root / POINTER_NAME

    def _validate_layout(self) -> None:
        _reject_symlinked_path_components(
            self.root, label="artifact store root"
        )
        if self.root.is_symlink() or (
            self.root.exists() and not self.root.is_dir()
        ):
            raise SupervisorContractError(
                "artifact store root must be a regular host-owned directory"
            )
        for path, label in (
            (self.versions, "versions"),
            (self.staging, "staging"),
        ):
            if path.is_symlink() or (path.exists() and not path.is_dir()):
                raise SupervisorContractError(
                    f"artifact store {label} must be a regular directory"
                )

    def current(self) -> dict[str, Any] | None:
        self._validate_layout()
        if self.pointer.is_symlink():
            raise SupervisorContractError(
                "current pointer must be a regular host-owned file"
            )
        if not self.pointer.exists():
            return None
        if not self.pointer.is_file():
            raise SupervisorContractError(
                "current pointer must be a regular host-owned file"
            )
        pointer = _read_json(self.pointer)
        required = {
            "schema",
            "version",
            "tree_sha256",
            "game",
            "target_level",
            "parent_checkpoint_sha256",
            "checkpoint_sha256",
            "promotion_receipt_sha256",
        }
        if (
            set(pointer) != required
            or not isinstance(pointer["schema"], int)
            or isinstance(pointer["schema"], bool)
            or pointer["schema"] != 1
            or not isinstance(pointer["version"], str)
            or re.fullmatch(r"[0-9a-f]{32}", pointer["version"]) is None
            or not _is_sha256_hex(pointer["tree_sha256"])
            or not isinstance(pointer["game"], str)
            or not pointer["game"]
            or not isinstance(pointer["target_level"], int)
            or isinstance(pointer["target_level"], bool)
            or pointer["target_level"] <= 0
            or not _is_sha256_hex(pointer["parent_checkpoint_sha256"])
            or not _is_sha256_hex(pointer["checkpoint_sha256"])
            or not _is_sha256_hex(pointer["promotion_receipt_sha256"])
        ):
            raise SupervisorContractError("current pointer schema mismatch")
        version = self.versions / pointer["version"]
        if version.is_symlink() or not version.is_dir():
            raise SupervisorContractError("current pointer references missing version")
        for path in version.rglob("*"):
            if path.is_symlink():
                raise SupervisorContractError(
                    f"current version contains a symlink: {path}"
                )
            if not path.is_dir() and not path.is_file():
                raise SupervisorContractError(
                    f"current version contains a non-regular entry: {path}"
                )
        if _tree_hash(version) != pointer["tree_sha256"]:
            raise SupervisorContractError("current pointer references corrupt version")

        targets = authoritative_inventory()
        game = pointer["game"]
        if game not in targets or pointer["target_level"] > targets[game]:
            raise SupervisorContractError(
                "current pointer is outside the authoritative inventory"
            )
        checkpoint = version / CHECKPOINT_NAME
        receipt = version / HOST_RECEIPT_NAME
        if (
            not checkpoint.is_file()
            or _sha256_file(checkpoint) != pointer["checkpoint_sha256"]
            or not receipt.is_file()
            or _sha256_file(receipt) != pointer["promotion_receipt_sha256"]
        ):
            raise SupervisorContractError(
                "current version evidence does not match its pointer"
            )
        selected_checkpoint = load_trusted_checkpoint(
            checkpoint,
            expected_game=game,
            authoritative_target=targets[game],
        )
        if selected_checkpoint.reached != pointer["target_level"]:
            raise SupervisorContractError(
                "current pointer disagrees with its trusted checkpoint level"
            )
        _validate_embedded_promotion_receipt(
            version,
            pointer=pointer,
            authoritative_target=targets[game],
            checkpoint=selected_checkpoint,
        )
        validate_winning_source_tree(
            version / WINNING_SOURCE_NAME
        )
        return pointer

    def publish(
        self,
        source: Path,
        *,
        receipt_path: Path,
        frontier: FrontierAdmission,
        parent_checkpoint_path: Path,
        candidate_output_root: Path,
        fault_at: str | None = None,
    ) -> dict[str, Any]:
        """Publish without exposing a partial version.

        ``fault_at`` exists solely for deterministic integration fault tests.
        Before pointer replacement, readers keep seeing the previous version.
        After pointer replacement, readers see a complete immutable version.
        """
        self._validate_layout()
        if (
            _paths_overlap(self.root, source)
            or _paths_overlap(self.root, candidate_output_root)
            or _paths_overlap(source, candidate_output_root)
        ):
            raise SupervisorContractError(
                "store, promotion source, and candidate output roots "
                "must be pairwise disjoint"
            )
        parent_resolved = parent_checkpoint_path.resolve(strict=False)
        for controlled_root in (source, candidate_output_root):
            controlled_resolved = controlled_root.resolve(strict=False)
            if (
                parent_resolved == controlled_resolved
                or controlled_resolved in parent_resolved.parents
            ):
                raise SupervisorContractError(
                    "host parent checkpoint must be outside "
                    "proposer-controlled roots"
                )
        receipt_resolved = receipt_path.resolve(strict=False)
        if (
            source.resolve(strict=False) in receipt_resolved.parents
            or candidate_output_root.resolve(strict=False)
            in receipt_resolved.parents
        ):
            raise SupervisorContractError(
                "host promotion receipt must be outside proposer-controlled roots"
            )
        with StoreLock(self.root):
            revalidated_frontier = admit_next_frontier(
                parent_checkpoint_path,
                expected_game=frontier.game,
                requested_level=frontier.next_level,
            )
            if revalidated_frontier != frontier:
                raise SupervisorContractError(
                    "frontier admission does not match the current "
                    "host-owned parent checkpoint"
                )
            selected = self.current()
            if selected is None:
                if self.root.resolve(strict=False) in parent_resolved.parents:
                    raise SupervisorContractError(
                        "initial parent checkpoint must be outside the "
                        "artifact store"
                    )
            else:
                selected_parent = (
                    self.versions
                    / selected["version"]
                    / CHECKPOINT_NAME
                ).resolve(strict=True)
                if parent_resolved != selected_parent:
                    raise SupervisorContractError(
                        "parent checkpoint is not the exact checkpoint of "
                        "the currently selected immutable version"
                    )
            admission = validate_promotion_receipt(
                receipt_path,
                source,
                frontier=revalidated_frontier,
                candidate_output_root=candidate_output_root,
            )
            return self._publish_admitted(
                source,
                admission=admission,
                frontier=revalidated_frontier,
                fault_at=fault_at,
            )

    def _publish_admitted(
        self,
        source: Path,
        *,
        admission: PromotionAdmission,
        frontier: FrontierAdmission,
        fault_at: str | None,
    ) -> dict[str, Any]:
        selected = self.current()
        if selected is None:
            if frontier.reached != 0 or admission.target_level != 1:
                raise SupervisorContractError(
                    "first version must promote level 1 from a zero checkpoint"
                )
        elif (
            selected["game"] != admission.game
            or selected["target_level"] + 1 != admission.target_level
            or selected["checkpoint_sha256"]
            != admission.parent_checkpoint_sha256
        ):
            raise SupervisorContractError(
                "promotion does not extend the currently selected version"
            )
        self.versions.mkdir(parents=True, exist_ok=True)
        self.staging.mkdir(parents=True, exist_ok=True)
        version_id = uuid.uuid4().hex
        stage = self.staging / version_id
        final = self.versions / version_id
        try:
            # Preserve links as links so a raced-in link can be rejected rather
            # than followed into the host filesystem by copytree.
            shutil.copytree(source, stage, symlinks=True)
            if fault_at == "partial_copy":
                raise RuntimeError("injected partial-copy failure")
            _validate_regular_tree(stage, label="staged promotion")
            if _tree_hash(stage) != admission.source_tree_sha256:
                raise SupervisorContractError(
                    "promotion source changed after host admission"
                )
            winning_payloads = _winning_source_payloads(stage)
            winning_source = stage / WINNING_SOURCE_NAME
            if winning_source.exists() or winning_source.is_symlink():
                raise SupervisorContractError(
                    "promotion source pre-creates reserved winning-source view"
                )
            winning_source.mkdir(mode=0o700)
            for name, payload in sorted(winning_payloads.items()):
                _write_new_regular_bytes(
                    winning_source / name,
                    payload,
                    label="winning source file",
                )
                os.chmod(
                    winning_source / name,
                    0o400,
                    follow_symlinks=False,
                )
            winning_source_tree_sha256 = (
                validate_winning_source_tree(winning_source)
            )
            os.chmod(
                winning_source, 0o500, follow_symlinks=False
            )
            _fsync_tree(winning_source)
            if (
                validate_winning_source_tree(winning_source)
                != winning_source_tree_sha256
            ):
                raise SupervisorContractError(
                    "winning source changed while staging publication"
                )
            _write_new_regular_bytes(
                stage / HOST_RECEIPT_NAME,
                admission.receipt_bytes,
                label="host promotion receipt",
            )
            candidate_evidence = (
                stage
                / "promotion_evidence"
                / f"level_{admission.target_level:02d}"
                / CANDIDATE_EVIDENCE_NAME
            )
            _write_new_regular_bytes(
                candidate_evidence,
                admission.candidate_manifest_bytes,
                label="host candidate-manifest evidence",
            )
            tree_hash = _tree_hash(stage)
            _fsync_tree(stage)
            os.replace(stage, final)
            _fsync_directory(self.versions)
            if fault_at == "after_version":
                raise RuntimeError("injected pre-pointer failure")
            pointer = {
                "schema": 1,
                "version": version_id,
                "tree_sha256": tree_hash,
                "game": admission.game,
                "target_level": admission.target_level,
                "parent_checkpoint_sha256":
                    admission.parent_checkpoint_sha256,
                "checkpoint_sha256": admission.checkpoint_sha256,
                "promotion_receipt_sha256": admission.receipt_sha256,
            }
            fd, temporary_name = tempfile.mkstemp(
                prefix=".current.", dir=self.root
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    json.dump(pointer, handle, sort_keys=True)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary_name, self.pointer)
                _fsync_directory(self.root)
            finally:
                if os.path.exists(temporary_name):
                    os.unlink(temporary_name)
            if fault_at == "after_pointer":
                raise RuntimeError("injected post-pointer failure")
            return pointer
        finally:
            if stage.exists():
                _remove_host_staging_tree(stage)

    def recover(self) -> dict[str, Any] | None:
        self._validate_layout()
        with StoreLock(self.root):
            return self._recover_locked()

    def _recover_locked(self) -> dict[str, Any] | None:
        self.staging.mkdir(parents=True, exist_ok=True)
        for path in self.staging.iterdir():
            if path.is_symlink():
                raise SupervisorContractError(
                    f"staging contains an unexpected symlink: {path}"
                )
            if path.is_dir():
                _remove_host_staging_tree(path)
            elif path.is_file():
                path.unlink()
            else:
                raise SupervisorContractError(
                    f"staging contains a non-regular entry: {path}"
                )
        for path in self.root.glob(".current.*"):
            if path.is_symlink() or not path.is_file():
                raise SupervisorContractError(
                    f"pointer staging contains a non-regular entry: {path}"
                )
            path.unlink()
        return self.current()


def validate_launch_attestation(
    path: Path,
    *,
    canonical_root: Path,
    environments_root: Path,
    repository: Path,
) -> dict[str, Any]:
    """Validate the one canonical terminal conformance receipt.

    There is deliberately no second caller-authored boolean attestation.
    Isolation and fault authority must come from the production S01--S12
    receipts already reopened by terminal conformance validation.
    """

    try:
        result = Conformance.load_result(
            Path(path),
            repository=Path(repository),
        )
        terminal = Conformance.validate_launch_authority_result(
            result,
            canonical_root=canonical_root,
            environments_root=environments_root,
            repository=Path(repository),
        )
    except Exception as exc:
        raise SupervisorContractError(
            "launch attestation is not genuine terminal conformance"
        ) from exc
    if (
        terminal["inventory_sha256"]
        != authoritative_inventory_sha256()
        or terminal["control_contract_sha256"]
        != Conformance.control_contract_sha256(Path(repository))
        or not isinstance(
            terminal["production_scenario_driver_receipt_path"],
            str,
        )
        or not _is_sha256_hex(
            terminal[
                "production_scenario_driver_receipt_sha256"
            ]
        )
        or not _is_sha256_hex(
            terminal["production_scenario_receipts_sha256"]
        )
        or not _is_sha256_hex(
            terminal[
                "production_scenario_verification_environment_sha256"
            ]
        )
    ):
        raise SupervisorContractError(
            "terminal conformance targets another control or inventory"
        )
    return terminal

MAX_CONTROL_SUITE_STREAM_BYTES = 128 * 1024 * 1024
CONTROL_SUITE_TERM_GRACE_SECONDS = 2.0


@dataclass(frozen=True)
class _BoundedProcessResult:
    argv: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str
    started_at_ns: int
    ended_at_ns: int
    timed_out: bool
    captured_descendant_count: int
    captured_descendants_absent: bool


def _limit_control_suite_child() -> None:
    resource.setrlimit(
        resource.RLIMIT_FSIZE,
        (
            MAX_CONTROL_SUITE_STREAM_BYTES,
            MAX_CONTROL_SUITE_STREAM_BYTES,
        ),
    )


def _process_identity_table(
) -> dict[int, tuple[int, int, int, str, str]]:
    """Return PID -> (PPID, PGID, SID, state, start identity)."""

    try:
        observed = subprocess.run(
            (
                "/bin/ps",
                "-axo",
                "pid=,ppid=,pgid=,stat=,lstart=",
            ),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env={"LANG": "C", "LC_ALL": "C"},
            text=True,
            timeout=10,
            check=False,
            shell=False,
            close_fds=True,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise SupervisorContractError(
            "cannot inventory control-suite process identities"
        ) from exc
    if observed.returncode != 0:
        raise SupervisorContractError(
            "control-suite process identity inventory failed"
        )
    result: dict[int, tuple[int, int, int, str, str]] = {}
    for raw in observed.stdout.splitlines():
        parts = raw.strip().split(None, 4)
        if len(parts) != 5:
            continue
        try:
            pid, ppid, pgid = (int(parts[index]) for index in range(3))
            sid = os.getsid(pid)
        except (ValueError, ProcessLookupError, PermissionError):
            continue
        if (
            pid > 1
            and ppid >= 0
            and pgid >= 0
            and parts[3]
            and parts[4]
        ):
            result[pid] = (
                ppid,
                pgid,
                sid,
                parts[3],
                parts[4],
            )
    return result


def _descendant_identities(
    root_pid: int,
    table: dict[int, tuple[int, int, int, str, str]],
) -> dict[int, str]:
    selected: dict[int, str] = {}
    frontier = [root_pid]
    while frontier:
        parent = frontier.pop()
        for pid, (
            ppid,
            _pgid,
            _sid,
            _state,
            started,
        ) in table.items():
            if ppid == parent and pid not in selected:
                selected[pid] = started
                frontier.append(pid)
    return selected


def _accumulate_related_identities(
    root_pid: int,
    root_sid: int,
    identities: dict[int, str],
    table: dict[int, tuple[int, int, int, str, str]],
) -> None:
    """Accumulate descendants plus exact process-group/session members."""

    related = {
        pid
        for pid, (_ppid, pgid, sid, _state, _started)
        in table.items()
        if pid != root_pid and (pgid == root_pid or sid == root_sid)
    }
    frontier = {root_pid, *identities, *related}
    changed = True
    while changed:
        changed = False
        for pid, (ppid, _pgid, _sid, _state, _started) in (
            table.items()
        ):
            if pid != root_pid and ppid in frontier and pid not in frontier:
                frontier.add(pid)
                related.add(pid)
                changed = True
    for pid in related:
        current = table.get(pid)
        if current is not None:
            identities.setdefault(pid, current[4])


def _signal_exact_processes(
    identities: dict[int, str],
    signum: int,
) -> None:
    live = _process_identity_table()
    for pid, started in sorted(identities.items(), reverse=True):
        current = live.get(pid)
        if (
            current is None
            or current[4] != started
            or current[3].startswith("Z")
        ):
            continue
        try:
            os.kill(pid, signum)
        except ProcessLookupError:
            pass
        except PermissionError as exc:
            raise SupervisorContractError(
                "cannot contain an exact control-suite descendant"
            ) from exc


def _identities_absent(identities: dict[int, str]) -> bool:
    live = _process_identity_table()
    return all(
        pid not in live
        or live[pid][4] != started
        or live[pid][3].startswith("Z")
        for pid, started in identities.items()
    )


def _read_bounded_process_stream(
    stream: Any, *, label: str
) -> str:
    stream.flush()
    size = os.fstat(stream.fileno()).st_size
    if size > MAX_CONTROL_SUITE_STREAM_BYTES:
        raise SupervisorContractError(
            f"control-suite {label} exceeded its hard byte bound"
        )
    stream.seek(0)
    raw = stream.read(MAX_CONTROL_SUITE_STREAM_BYTES + 1)
    if len(raw) != size:
        raise SupervisorContractError(
            f"control-suite {label} changed during capture"
        )
    try:
        return raw.decode("utf-8")
    except UnicodeError as exc:
        raise SupervisorContractError(
            f"control-suite {label} is not UTF-8"
        ) from exc


def _run_bounded_process_group(
    argv: tuple[str, ...],
    *,
    cwd: Path,
    environment: dict[str, str],
    timeout_seconds: float,
    scratch_root: Path | None = None,
) -> _BoundedProcessResult:
    """Run one exact command and contain its process group and descendants."""

    if (
        not argv
        or not all(isinstance(part, str) and part for part in argv)
        or not Path(argv[0]).is_absolute()
        or timeout_seconds <= 0
    ):
        raise SupervisorContractError(
            "bounded control-suite command is malformed"
        )
    temporary_root = None
    if scratch_root is not None:
        temporary_root = Path(scratch_root)
        try:
            metadata = temporary_root.lstat()
        except OSError as exc:
            raise SupervisorContractError(
                "bounded control-suite scratch root is unavailable"
            ) from exc
        if (
            not temporary_root.is_absolute()
            or temporary_root.is_symlink()
            or not stat.S_ISDIR(metadata.st_mode)
            or metadata.st_uid != os.getuid()
            or stat.S_IMODE(metadata.st_mode) != 0o700
        ):
            raise SupervisorContractError(
                "bounded control-suite scratch root is not private"
            )
    # Do not start a process tree unless the supervisor can enumerate exact
    # descendant identities for the timeout path.
    _process_identity_table()
    started_at_ns = time.time_ns()
    with tempfile.TemporaryFile(
        mode="w+b", dir=temporary_root
    ) as stdout_file, (
        tempfile.TemporaryFile(mode="w+b", dir=temporary_root)
    ) as stderr_file:
        try:
            process = subprocess.Popen(
                argv,
                cwd=cwd,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=stdout_file,
                stderr=stderr_file,
                shell=False,
                close_fds=True,
                start_new_session=True,
                preexec_fn=_limit_control_suite_child,
            )
        except OSError as exc:
            raise SupervisorContractError(
                "runtime control-suite process could not start"
            ) from exc
        timed_out = False
        descendants: dict[int, str] = {}
        root_start: str | None = None
        deadline = time.monotonic() + timeout_seconds
        while True:
            table = _process_identity_table()
            root_identity = table.get(process.pid)
            if root_identity is not None:
                root_start = root_start or root_identity[4]
            _accumulate_related_identities(
                process.pid,
                process.pid,
                descendants,
                table,
            )
            polled = process.poll()
            if polled is not None:
                returncode = polled
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                timed_out = True
                returncode = -signal.SIGTERM
                break
            time.sleep(min(0.05, remaining))

        # Capture children that exited their parent concurrently with the
        # direct child's state transition.  Same-session/group processes are
        # retained even after reparenting.
        table = _process_identity_table()
        _accumulate_related_identities(
            process.pid,
            process.pid,
            descendants,
            table,
        )
        captured = {
            **descendants,
            **(
                {process.pid: root_start}
                if root_start is not None
                else {}
            ),
        }
        residual_after_normal_exit = (
            not timed_out
            and not _identities_absent(descendants)
        )
        if timed_out or residual_after_normal_exit:
            _signal_exact_processes(descendants, signal.SIGTERM)
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            except PermissionError:
                # macOS may report EPERM after the group leader is reaped
                # and only already-signalled zombies remain.  Exact
                # PID/start identities are still signalled below and their
                # absence is mandatory before return.
                pass

            # Continue accumulating exact identities throughout the grace
            # period; a child may fork, setsid, or reparent while teardown is
            # underway.
            grace_deadline = (
                time.monotonic()
                + CONTROL_SUITE_TERM_GRACE_SECONDS
            )
            while time.monotonic() < grace_deadline:
                table = _process_identity_table()
                _accumulate_related_identities(
                    process.pid,
                    process.pid,
                    descendants,
                    table,
                )
                _signal_exact_processes(
                    descendants, signal.SIGTERM
                )
                if (
                    process.poll() is not None
                    and _identities_absent(descendants)
                ):
                    break
                time.sleep(0.05)
            captured = {
                **descendants,
                **(
                    {process.pid: root_start}
                    if root_start is not None
                    else {}
                ),
            }
            _signal_exact_processes(captured, signal.SIGKILL)
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except PermissionError:
                pass
            try:
                final_returncode = process.wait(timeout=10)
            except subprocess.TimeoutExpired as exc:
                raise SupervisorContractError(
                    "control-suite direct child could not be reaped"
                ) from exc
            if timed_out:
                returncode = final_returncode
            final_table = _process_identity_table()
            _accumulate_related_identities(
                process.pid,
                process.pid,
                descendants,
                final_table,
            )
            _signal_exact_processes(descendants, signal.SIGKILL)
            if not _identities_absent(captured):
                raise SupervisorContractError(
                    "control-suite descendant survived containment"
                )
            if not _identities_absent(descendants):
                raise SupervisorContractError(
                    "late control-suite descendant survived containment"
                )
            if residual_after_normal_exit:
                raise SupervisorContractError(
                    "normally exited control suite left a descendant"
                )
        stdout = _read_bounded_process_stream(
            stdout_file, label="stdout"
        )
        stderr = _read_bounded_process_stream(
            stderr_file, label="stderr"
        )
    return _BoundedProcessResult(
        argv=argv,
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
        started_at_ns=started_at_ns,
        ended_at_ns=time.time_ns(),
        timed_out=timed_out,
        captured_descendant_count=len(descendants),
        captured_descendants_absent=True,
    )


def _private_system_scratch() -> Path:
    private_tmp = Path("/private/tmp")
    parent = private_tmp if private_tmp.exists() else Path("/tmp")
    repository = Path(__file__).resolve().parents[2]
    if not parent.is_absolute():
        raise SupervisorContractError(
            "control-suite system scratch parent is unsafe"
        )
    parent_descriptor = -1
    scratch_descriptor = -1
    scratch_name: str | None = None
    root: Path | None = None
    try:
        named_parent = os.lstat(parent)
        canonical_parent = parent.resolve(strict=True)
        parent_descriptor = os.open(
            parent,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
        )
        opened_parent = os.fstat(parent_descriptor)
        parent_identity = (
            named_parent.st_dev,
            named_parent.st_ino,
            named_parent.st_mode,
            named_parent.st_uid,
            named_parent.st_gid,
        )
        if (
            canonical_parent != parent
            or stat.S_ISLNK(named_parent.st_mode)
            or not stat.S_ISDIR(named_parent.st_mode)
            or (
                stat.S_IMODE(named_parent.st_mode) & 0o022
                and not named_parent.st_mode & stat.S_ISVTX
            )
            or (
                opened_parent.st_dev,
                opened_parent.st_ino,
                opened_parent.st_mode,
                opened_parent.st_uid,
                opened_parent.st_gid,
            )
            != parent_identity
            or canonical_parent == repository
            or canonical_parent.is_relative_to(repository)
        ):
            raise SupervisorContractError(
                "control-suite system scratch parent is unsafe"
            )
        for _ in range(32):
            candidate = f"a3c_{secrets.token_hex(6)}"
            try:
                os.mkdir(
                    candidate,
                    mode=0o700,
                    dir_fd=parent_descriptor,
                )
            except FileExistsError:
                continue
            scratch_name = candidate
            break
        if scratch_name is None:
            raise SupervisorContractError(
                "control-suite private scratch names are exhausted"
            )
        created = os.stat(
            scratch_name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        scratch_descriptor = os.open(
            scratch_name,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
            dir_fd=parent_descriptor,
        )
        opened_scratch = os.fstat(scratch_descriptor)
        os.fchmod(scratch_descriptor, 0o700)
        sealed = os.fstat(scratch_descriptor)
        rebound_parent = os.lstat(parent)
        root = parent / scratch_name
        rebound_root = os.lstat(root)
        scratch_identity = (
            created.st_dev,
            created.st_ino,
            created.st_uid,
            created.st_gid,
        )
        if (
            not root.is_absolute()
            or root.parent != parent
            or re.fullmatch(r"a3c_[0-9a-f]{12}", root.name) is None
            or stat.S_ISLNK(created.st_mode)
            or not stat.S_ISDIR(created.st_mode)
            or created.st_uid != os.getuid()
            or (
                opened_scratch.st_dev,
                opened_scratch.st_ino,
                opened_scratch.st_uid,
                opened_scratch.st_gid,
            )
            != scratch_identity
            or (
                sealed.st_dev,
                sealed.st_ino,
                sealed.st_uid,
                sealed.st_gid,
            )
            != scratch_identity
            or stat.S_IMODE(sealed.st_mode) != 0o700
            or (
                rebound_root.st_dev,
                rebound_root.st_ino,
                rebound_root.st_uid,
                rebound_root.st_gid,
            )
            != scratch_identity
            or (
                rebound_parent.st_dev,
                rebound_parent.st_ino,
                rebound_parent.st_mode,
                rebound_parent.st_uid,
                rebound_parent.st_gid,
            )
            != parent_identity
        ):
            raise SupervisorContractError(
                "control-suite private scratch is not a shallow owned "
                "directory"
            )
    except BaseException as exc:
        cleanup_error: BaseException | None = None
        if scratch_name is not None and parent_descriptor >= 0:
            try:
                residue = os.stat(
                    scratch_name,
                    dir_fd=parent_descriptor,
                    follow_symlinks=False,
                )
                if (
                    stat.S_ISDIR(residue.st_mode)
                    and not stat.S_ISLNK(residue.st_mode)
                    and residue.st_uid == os.getuid()
                ):
                    os.rmdir(
                        scratch_name,
                        dir_fd=parent_descriptor,
                    )
                else:
                    raise SupervisorContractError(
                        "failed control-suite scratch residue is unsafe"
                    )
            except FileNotFoundError:
                pass
            except BaseException as cleanup_exc:
                cleanup_error = cleanup_exc
        if cleanup_error is not None:
            raise SupervisorContractError(
                "control-suite private scratch creation failed and "
                "could not remove its residue"
            ) from cleanup_error
        if isinstance(exc, SupervisorContractError):
            raise
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        raise SupervisorContractError(
            "control-suite private scratch creation failed"
        ) from exc
    finally:
        if scratch_descriptor >= 0:
            os.close(scratch_descriptor)
        if parent_descriptor >= 0:
            os.close(parent_descriptor)
    assert root is not None
    return root


def _run_control_suite_with_scratch(
    *,
    python_executable: Path,
    python_executable_sha256: str,
    runtime_control_snapshot_root: Path,
    python_runtime_manifest: Path | None = None,
    python_runtime_manifest_sha256: str | None = None,
    scratch_root: Path,
) -> dict[str, Any]:
    """Execute sealed controls under one fully manifest-bound Python runtime."""

    repository = Path(__file__).resolve().parents[2]
    if (
        python_runtime_manifest is None
        or python_runtime_manifest_sha256 is None
    ):
        raise SupervisorContractError(
            "runtime control suite requires a pinned Python runtime manifest"
        )
    interpreter = Path(python_executable)
    try:
        runtime_manifest = RuntimeManifest.load_runtime_manifest(
            Path(python_runtime_manifest),
            expected_sha256=python_runtime_manifest_sha256,
            python_executable=interpreter,
            python_executable_sha256=python_executable_sha256,
        )
    except RuntimeManifest.RuntimeManifestError as exc:
        raise SupervisorContractError(
            "Python runtime manifest failed its pre-execution recheck"
        ) from exc
    live_start = Conformance.control_contract_snapshot(
        repository=repository
    )
    try:
        snapshot = Conformance.materialize_immutable_control_snapshot(
            repository,
            runtime_control_snapshot_root,
        )
    except Exception as exc:
        raise SupervisorContractError(
            "immutable runtime control snapshot could not be materialized"
        ) from exc
    if snapshot != live_start:
        raise SupervisorContractError(
            "immutable runtime snapshot differs from live preflight bytes"
        )
    snapshot_root = Path(runtime_control_snapshot_root)
    suite_path = snapshot_root / Conformance.SUITE_CONTROL_PATH
    environment = {
        "HOME": str(snapshot_root / ".neutral"),
        "LANG": "C",
        "LC_ALL": "C",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        "TMPDIR": str(scratch_root),
    }
    for package, command in (
        (
            False,
            RuntimeManifest.base_probe_command(interpreter),
        ),
        (
            True,
            RuntimeManifest.package_probe_command(
                interpreter,
                Path(
                    runtime_manifest[
                        "base_runtime_probe"
                    ]["purelib"]
                ),
            ),
        ),
    ):
        observed = _run_bounded_process_group(
            command,
            cwd=snapshot_root / ".neutral",
            environment=environment,
            timeout_seconds=60,
            scratch_root=scratch_root,
        )
        if observed.timed_out or observed.returncode != 0:
            raise SupervisorContractError(
                "manifest-bound Python runtime probe failed"
            )
        try:
            probe = RuntimeManifest.parse_probe(
                observed.stdout.encode("utf-8"),
                package=package,
            )
        except RuntimeManifest.RuntimeManifestError as exc:
            raise SupervisorContractError(
                "manifest-bound Python runtime probe is malformed"
            ) from exc
        expected_probe = runtime_manifest[
            (
                "package_runtime_probe"
                if package
                else "base_runtime_probe"
            )
        ]
        if probe != expected_probe:
            raise SupervisorContractError(
                "live Python runtime identity differs from its manifest"
            )
    try:
        RuntimeManifest.revalidate_runtime_files(runtime_manifest)
    except RuntimeManifest.RuntimeManifestError as exc:
        raise SupervisorContractError(
            "Python runtime changed during its live identity probes"
        ) from exc
    command = RuntimeManifest.suite_command(
        interpreter,
        site_root=Path(
            runtime_manifest["base_runtime_probe"]["purelib"]
        ),
        suite_path=suite_path,
        runtime_manifest_path=Path(python_runtime_manifest),
        runtime_manifest_sha256=python_runtime_manifest_sha256,
    )
    completed: _BoundedProcessResult | None = None
    suite_error: BaseException | None = None
    try:
        completed = _run_bounded_process_group(
            command,
            cwd=snapshot_root / ".neutral",
            environment=environment,
            timeout_seconds=15 * 60,
            scratch_root=scratch_root,
        )
    except BaseException as exc:
        suite_error = exc
    try:
        RuntimeManifest.revalidate_runtime_files(runtime_manifest)
        reopened = RuntimeManifest.load_runtime_manifest(
            Path(python_runtime_manifest),
            expected_sha256=python_runtime_manifest_sha256,
            python_executable=interpreter,
            python_executable_sha256=python_executable_sha256,
        )
        if reopened != runtime_manifest:
            raise RuntimeManifest.RuntimeManifestError(
                "runtime manifest changed across suite execution"
            )
    except RuntimeManifest.RuntimeManifestError as exc:
        raise SupervisorContractError(
            "Python runtime or manifest changed during control execution"
        ) from exc
    if suite_error is not None:
        raise suite_error
    assert completed is not None
    if completed.timed_out:
        raise SupervisorContractError(
            "runtime control-suite recheck timed out after exact "
            "descendant containment"
        )
    if completed.returncode != 0:
        combined = (completed.stdout or "") + (completed.stderr or "")
        tail = combined[-4000:]
        raise SupervisorContractError(
            "runtime control-suite recheck failed before launch"
            + (f":\n{tail}" if tail else "")
        )
    try:
        raw = completed.stdout.encode("utf-8")
        value = json.loads(raw)
        if raw != Conformance._canonical_json(value) + b"\n":
            raise Conformance.ConformanceError(
                "runtime result is not canonical JSON"
            )
        result = Conformance.validate_result(
            value, repository=snapshot_root
        )
        if (
            result["suite_interpreter_path"]
            != str(interpreter.resolve())
            or result["suite_interpreter_sha256"]
            != python_executable_sha256
            or result["suite_runtime_manifest_path"]
            != str(Path(python_runtime_manifest))
            or result["suite_runtime_manifest_sha256"]
            != python_runtime_manifest_sha256
            or result["execution_control_root"]
            != str(snapshot_root.resolve())
            or result["execution_control_snapshot_immutable"] is not True
            or Conformance.validate_immutable_control_snapshot(
                snapshot_root,
                expected_sha256=live_start["sha256"],
            )
            != live_start
            or Conformance.control_contract_snapshot(
                repository=repository
            )
            != live_start
        ):
            raise Conformance.ConformanceError(
                "runtime result differs from interpreter/snapshot/live "
                "control evidence"
            )
        return result
    except Exception as exc:
        raise SupervisorContractError(
            "runtime unified conformance result is not exact PASS"
        ) from exc


def _run_control_suite(
    *,
    python_executable: Path,
    python_executable_sha256: str,
    runtime_control_snapshot_root: Path,
    python_runtime_manifest: Path | None = None,
    python_runtime_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    scratch_root = _private_system_scratch()
    scratch_metadata = os.stat(
        scratch_root, follow_symlinks=False
    )
    scratch_identity = (
        scratch_metadata.st_dev,
        scratch_metadata.st_ino,
        scratch_metadata.st_uid,
        scratch_metadata.st_gid,
    )
    result: dict[str, Any] | None = None
    error: BaseException | None = None
    cleanup_error: BaseException | None = None
    leaked: list[str] = []
    try:
        result = _run_control_suite_with_scratch(
            python_executable=python_executable,
            python_executable_sha256=python_executable_sha256,
            runtime_control_snapshot_root=runtime_control_snapshot_root,
            python_runtime_manifest=python_runtime_manifest,
            python_runtime_manifest_sha256=(
                python_runtime_manifest_sha256
            ),
            scratch_root=scratch_root,
        )
        leaked = sorted(path.name for path in scratch_root.iterdir())
    except BaseException as exc:
        error = exc
        try:
            leaked = sorted(
                path.name for path in scratch_root.iterdir()
            )
        except OSError:
            leaked = ["<unreadable>"]
    finally:
        try:
            Conformance._remove_owned_private_tree(
                scratch_root,
                expected_identity=scratch_identity,
                label="control-suite scratch",
            )
        except BaseException as exc:
            cleanup_error = exc
    if cleanup_error is not None:
        raise SupervisorContractError(
            "runtime control-suite scratch cleanup failed closed"
        ) from cleanup_error
    if leaked:
        raise SupervisorContractError(
            f"runtime control suite leaked scratch entries: {leaked}"
        ) from error
    if error is not None:
        raise error
    assert result is not None
    return result


def launch_preflight(
    attestation: Path,
    *,
    requested_image_digest: str,
    conformance_result: Path,
    canonical_root: Path,
    environments_root: Path,
    python_executable: Path,
    python_executable_sha256: str,
    runtime_control_snapshot_root: Path,
    pilot_gate_receipt: Path | None = None,
    pilot_authentication_key: Path | None = None,
    pilot_production_stack_attestation_sha256: str | None = None,
    python_runtime_manifest: Path | None = None,
    python_runtime_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    targets = authoritative_inventory()
    if (
        Path(os.path.abspath(attestation))
        != Path(os.path.abspath(conformance_result))
    ):
        raise SupervisorContractError(
            "launch requires one terminal conformance receipt, not a "
            "separate caller attestation"
        )
    prior_conformance = validate_launch_attestation(
        attestation,
        canonical_root=canonical_root,
        environments_root=environments_root,
        repository=runtime_control_snapshot_root,
    )
    if (
        prior_conformance["container_image_digest"]
        != requested_image_digest
    ):
        raise SupervisorContractError(
            "requested container image does not match the tested image digest"
        )
    if (
        prior_conformance["suite_runtime_manifest_path"]
        != (
            None
            if python_runtime_manifest is None
            else str(Path(python_runtime_manifest))
        )
        or prior_conformance["suite_runtime_manifest_sha256"]
        != python_runtime_manifest_sha256
    ):
        raise SupervisorContractError(
            "terminal conformance targets another Python runtime manifest"
        )
    runtime_prelaunch = _run_control_suite(
        python_executable=python_executable,
        python_executable_sha256=python_executable_sha256,
        python_runtime_manifest=python_runtime_manifest,
        python_runtime_manifest_sha256=(
            python_runtime_manifest_sha256
        ),
        runtime_control_snapshot_root=runtime_control_snapshot_root,
    )
    try:
        runtime_conformance = Conformance.bind_terminal_launch_authority(
            runtime_prelaunch,
            container_image_digest=requested_image_digest,
            release_receipt_path=Path(
                prior_conformance["frozen_release_receipt_path"]
            ),
            scenario_driver_receipt_path=Path(
                prior_conformance[
                    "production_scenario_driver_receipt_path"
                ]
            ),
            canonical_root=canonical_root,
            environments_root=environments_root,
            repository=Path(
                runtime_prelaunch["execution_control_root"]
            ),
        )
    except Exception as exc:
        raise SupervisorContractError(
            "runtime conformance could not bind live terminal launch "
            "authority"
        ) from exc
    if any(
        runtime_conformance[field] != prior_conformance[field]
        for field in (
            "registry_sha256",
            "control_contract_sha256",
            "inventory_sha256",
            "container_image_digest",
            "frozen_release_receipt_path",
            "frozen_release_receipt_sha256",
            "frozen_release_levels",
            "production_scenario_driver_receipt_path",
            "production_scenario_driver_receipt_sha256",
            "production_scenario_receipts_sha256",
            "production_scenario_verification_environment_sha256",
            "suite_runtime_manifest_path",
            "suite_runtime_manifest_sha256",
        )
    ):
        raise SupervisorContractError(
            "runtime terminal authority differs from the supplied "
            "conformance receipt"
        )
    if not CONTIGUOUS_LAUNCH_READY:
        raise SupervisorContractError(
            "contiguous launch is fail-closed until the real container "
            "scheduler, observed attestation producer, Arena RPC boundary, "
            "and canonical 183/183 release gate are implemented"
        )
    try:
        import arc_agi3_contiguous_pilot as Pilot

        pilot_gate = Pilot.verify_pilot_gate_receipt(
            Path(pilot_gate_receipt),
            authentication_key_path=Path(
                pilot_authentication_key
            ),
            expected_image_digest=requested_image_digest,
            expected_control_contract_sha256=(
                runtime_conformance["control_contract_sha256"]
            ),
            expected_production_stack_attestation_sha256=str(
                pilot_production_stack_attestation_sha256
            ),
        )
    except Exception as exc:
        raise SupervisorContractError(
            "full launch requires exact live ft09 then lp85 pilot PASS"
        ) from exc
    return {
        "status": "PASS",
        "runtime_contiguous_conformance": "PASS",
        "conformance_result": str(conformance_result),
        "conformance_registry_sha256":
            prior_conformance["registry_sha256"],
        "runtime_conformance_output_sha256":
            runtime_conformance["pytest_output_sha256"],
        "games": len(targets),
        "levels": sum(targets.values()),
        "attestation": str(attestation),
        "image_digest": requested_image_digest,
        "authoritative_inventory_sha256":
            prior_conformance["inventory_sha256"],
        "control_contract_sha256":
            prior_conformance["control_contract_sha256"],
        "python_runtime_manifest":
            str(python_runtime_manifest),
        "python_runtime_manifest_sha256":
            python_runtime_manifest_sha256,
        "pilot_gate_receipt": str(
            Path(pilot_gate_receipt).resolve()
        ),
        "pilot_gate_receipt_sha256":
            pilot_gate["file_sha256"],
        "pilot_manifest_sha256":
            pilot_gate["pilot_manifest_sha256"],
        "pilot_meta_handoff_count":
            pilot_gate["meta_handoff_count"],
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--image-digest", required=True)
    parser.add_argument("--conformance-result", type=Path, required=True)
    parser.add_argument("--canonical-root", type=Path, required=True)
    parser.add_argument("--environments-root", type=Path, required=True)
    parser.add_argument("--python-executable", type=Path, required=True)
    parser.add_argument("--python-sha256", required=True)
    parser.add_argument(
        "--python-runtime-manifest",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--python-runtime-manifest-sha256",
        required=True,
    )
    parser.add_argument(
        "--runtime-control-snapshot-root",
        type=Path,
        required=True,
    )
    args = parser.parse_args(argv)
    print(json.dumps(
        launch_preflight(
            args.preflight,
            requested_image_digest=args.image_digest,
            conformance_result=args.conformance_result,
            canonical_root=args.canonical_root,
            environments_root=args.environments_root,
            python_executable=args.python_executable,
            python_executable_sha256=args.python_sha256,
            python_runtime_manifest=args.python_runtime_manifest,
            python_runtime_manifest_sha256=(
                args.python_runtime_manifest_sha256
            ),
            runtime_control_snapshot_root=(
                args.runtime_control_snapshot_root
            ),
        ),
        indent=2,
        sort_keys=True,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
