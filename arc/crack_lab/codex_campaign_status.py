#!/usr/bin/env python3
"""Report headless Codex allowance, solve efficiency, and next GKM frontiers.

Without ``--live`` this is entirely local and uses the last postflight snapshot
in the durable ledger.  ``--live`` performs only ``account/rateLimits/read``;
it does not start a model turn or consume a reset credit.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import stat
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Optional

import codex_usage_guard as Guard
import arc_agi3_proposer_boundary as Boundary


HERE = Path(__file__).resolve().parent
ARC_ROOT = HERE.parent
ARTIFACTS = HERE / "agent_solutions"
AUDITS = ARC_ROOT / "audit_results"
ENVIRONMENTS = HERE.parents[1] / "environment_files"
_INVENTORY_CACHE_MAX_ENTRIES = 16
_inventory_cache: dict[
    tuple[str, tuple[tuple[object, ...], ...]], dict[str, int]
] = {}
ZERO_SHA256 = "0" * 64
FRONTIER_BINDING_SCHEMA = 1
FRONTIER_BINDING_FIELDS = (
    "frontier_binding_schema",
    "parent_checkpoint_sha256",
    "parent_source_tree_sha256",
    "frontier_sha256",
)
FRONTIER_BINDING_CORRECTION_SCHEMA = 1
FRONTIER_BINDING_CORRECTION_AUTHORITY = (
    "receipt_backed_exact_launch_parent_claim"
)
FRONTIER_BINDING_CORRECTION_EVIDENCE_FIELDS = (
    "workspace_baseline_commit",
    "baseline_checkpoint_sha256",
    "baseline_source_tree_sha256",
    "protected_transcript_sha256",
    "audit_receipt_relpath",
    "audit_receipt_sha256",
    "baseline_checkpoint_replay_verified",
    "workspace_git_history_unmodified",
    "terminal_turn_audited",
    "taint_scan_passed",
)
INFRASTRUCTURE_WIP_PHASES = frozenset({
    "infrastructure_failure",
    "infrastructure_failure_transport",
    "containment_timeout",
})
INFRASTRUCTURE_NONCOUNTING_EVENT = (
    "codex_infrastructure_generation_quarantined"
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_json(value: Any) -> str:
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _read_stable_regular(path: Path) -> bytes:
    """Read one immutable-identity input, rejecting aliases and read races."""
    if path.is_symlink():
        raise ValueError(f"frontier identity input is a symlink: {path}")
    before = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(before.st_mode):
        raise ValueError(f"frontier identity input is not regular: {path}")
    raw = path.read_bytes()
    after = path.stat(follow_symlinks=False)
    pointer_before = (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    pointer_after = (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if pointer_before != pointer_after:
        raise ValueError(
            f"frontier identity input changed while read: {path}"
        )
    return raw


def _read_receipt_regular(root: Path, name: str) -> bytes:
    """Open one receipt beneath a stable, unaliased directory descriptor."""
    if Path(name).name != name:
        raise ValueError("receipt name is not one path component")
    required_flags = ("O_DIRECTORY", "O_NOFOLLOW")
    if any(not hasattr(os, flag) for flag in required_flags):
        raise ValueError("platform lacks descriptor-safe receipt flags")
    directory_flags = (
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        | getattr(os, "O_CLOEXEC", 0)
    )
    file_flags = (
        os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    )
    root_fd = os.open(os.fspath(root), directory_flags)
    try:
        root_stat = os.fstat(root_fd)
        if not stat.S_ISDIR(root_stat.st_mode):
            raise ValueError("receipt root is not a directory")
        file_fd = os.open(name, file_flags, dir_fd=root_fd)
        try:
            before = os.fstat(file_fd)
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_nlink != 1
                or before.st_size > 1_000_000
            ):
                raise ValueError(
                    "receipt is aliased, non-regular, or unreasonably large"
                )
            chunks = []
            remaining = before.st_size
            while remaining:
                chunk = os.read(file_fd, min(remaining, 65_536))
                if not chunk:
                    raise ValueError("receipt became short while read")
                chunks.append(chunk)
                remaining -= len(chunk)
            if os.read(file_fd, 1):
                raise ValueError("receipt grew while read")
            after = os.fstat(file_fd)
            entry_after = os.stat(
                name, dir_fd=root_fd, follow_symlinks=False
            )
            file_identity = (
                before.st_dev,
                before.st_ino,
                before.st_mode,
                before.st_nlink,
                before.st_size,
                before.st_mtime_ns,
                before.st_ctime_ns,
            )
            if file_identity != (
                after.st_dev,
                after.st_ino,
                after.st_mode,
                after.st_nlink,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            ) or (
                entry_after.st_dev,
                entry_after.st_ino,
                entry_after.st_mode,
                entry_after.st_nlink,
                entry_after.st_size,
                entry_after.st_mtime_ns,
                entry_after.st_ctime_ns,
            ) != file_identity:
                raise ValueError(
                    "receipt changed or was replaced while read"
                )
        finally:
            os.close(file_fd)
        root_after = os.stat(root, follow_symlinks=False)
        if (
            root_after.st_dev,
            root_after.st_ino,
            root_after.st_mode,
        ) != (
            root_stat.st_dev,
            root_stat.st_ino,
            root_stat.st_mode,
        ):
            raise ValueError("receipt root changed while read")
        return b"".join(chunks)
    finally:
        os.close(root_fd)


def _source_tree_sha256(payloads: dict[str, bytes]) -> str:
    """Use the same named-file digest construction as the contiguous runner."""
    digest = hashlib.sha256()
    for name, raw in sorted(payloads.items()):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(raw).hexdigest().encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def validate_frontier_binding(
    value: dict[str, Any],
    *,
    expected_game: str | None = None,
    expected_target_level: int | None = None,
) -> dict[str, Any]:
    """Validate a path-free exact-parent binding for ledger/queue transport."""
    required = {
        *FRONTIER_BINDING_FIELDS,
        "game",
        "reached",
        "target_level",
        "parent_action_count",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("frontier binding has a noncanonical field set")
    game = value["game"]
    reached = value["reached"]
    target_level = value["target_level"]
    action_count = value["parent_action_count"]
    if (
        value["frontier_binding_schema"] != FRONTIER_BINDING_SCHEMA
        or not isinstance(game, str)
        or not game
        or not isinstance(reached, int)
        or isinstance(reached, bool)
        or reached < 0
        or not isinstance(target_level, int)
        or isinstance(target_level, bool)
        or target_level != reached + 1
        or not isinstance(action_count, int)
        or isinstance(action_count, bool)
        or action_count < 0
        or any(
            not _is_sha256(value[field])
            for field in (
                "parent_checkpoint_sha256",
                "parent_source_tree_sha256",
                "frontier_sha256",
            )
        )
    ):
        raise ValueError("frontier binding is malformed")
    if expected_game is not None and game != expected_game:
        raise ValueError("frontier binding game does not match dispatch")
    if (
        expected_target_level is not None
        and target_level != expected_target_level
    ):
        raise ValueError("frontier binding target does not match dispatch")
    expected_frontier = _sha256_json(
        {
            "game": game,
            "reached": reached,
            "parent_checkpoint_sha256":
                value["parent_checkpoint_sha256"],
        }
    )
    if value["frontier_sha256"] != expected_frontier:
        raise ValueError("frontier digest does not match its exact parent")
    if reached == 0 and (
        value["parent_checkpoint_sha256"] != ZERO_SHA256
        or value["parent_source_tree_sha256"] != ZERO_SHA256
        or action_count != 0
    ):
        raise ValueError("cold frontier must use the canonical zero parent")
    if reached > 0 and (
        value["parent_checkpoint_sha256"] == ZERO_SHA256
        or value["parent_source_tree_sha256"] == ZERO_SHA256
    ):
        raise ValueError("promoted frontier cannot use a zero parent")
    return dict(value)


def _record_frontier_binding(
    record: dict[str, Any],
) -> tuple[dict[str, Any] | None, bool]:
    """Return a complete binding claim and whether any such claim was made."""
    claim_fields = (
        *FRONTIER_BINDING_FIELDS,
        "reached",
        "parent_action_count",
    )
    declared = any(field in record for field in claim_fields)
    if not declared:
        return None, False
    try:
        binding = validate_frontier_binding({
            field: record.get(field)
            for field in (
                *FRONTIER_BINDING_FIELDS,
                "game",
                "reached",
                "target_level",
                "parent_action_count",
            )
        })
    except ValueError:
        return None, True
    return binding, True


def _binding_correction_receipt_payload(
    record: dict[str, Any],
) -> dict[str, Any]:
    """Project a correction onto its separately sealed audit receipt."""
    evidence = record["evidence"]
    return {
        "receipt_schema": FRONTIER_BINDING_CORRECTION_SCHEMA,
        "binding_authority": record["binding_authority"],
        "thread_id": record["thread_id"],
        "transcript": record["transcript"],
        "binding": {
            field: record[field]
            for field in (
                *FRONTIER_BINDING_FIELDS,
                "game",
                "reached",
                "target_level",
                "parent_action_count",
            )
        },
        "evidence": {
            field: evidence[field]
            for field in FRONTIER_BINDING_CORRECTION_EVIDENCE_FIELDS
            if field not in {
                "audit_receipt_relpath",
                "audit_receipt_sha256",
            }
        },
    }


def validate_frontier_binding_correction(
    record: dict[str, Any],
    *,
    exec_record: dict[str, Any],
    receipt_root: Path | None = None,
) -> dict[str, Any]:
    """Validate a receipt-backed append-only binding of a pre-schema turn.

    This reducer does not treat host-asserted booleans in the ledger as proof.
    It independently reopens a canonical, content-addressed audit receipt and
    checks that the receipt seals the exact binding and evidence assertions.
    The receipt producer remains responsible for reconstructing the launch
    parent, replaying its checkpoint, checking the protected transcript/Git
    baseline/taint status, and writing the receipt. Ordinary failure-class
    corrections cannot supply or override frontier identity.
    """
    required = {
        "event",
        "binding_correction_schema",
        "binding_authority",
        "recorded_at",
        "thread_id",
        "transcript",
        "evidence",
        *FRONTIER_BINDING_FIELDS,
        "game",
        "reached",
        "target_level",
        "parent_action_count",
    }
    if not isinstance(record, dict) or set(record) != required:
        raise ValueError(
            "frontier binding correction has a noncanonical field set"
        )
    if (
        record["event"] != "codex_frontier_binding_correction"
        or record["binding_correction_schema"]
        != FRONTIER_BINDING_CORRECTION_SCHEMA
        or record["binding_authority"]
        != FRONTIER_BINDING_CORRECTION_AUTHORITY
        or not isinstance(record["recorded_at"], str)
        or not record["recorded_at"]
        or not isinstance(record["thread_id"], str)
        or not record["thread_id"]
        or not isinstance(record["transcript"], str)
        or not record["transcript"]
        or record["thread_id"] != exec_record.get("thread_id")
        or record["transcript"] != exec_record.get("transcript")
        or record["game"] != exec_record.get("game")
        or record["target_level"] != exec_record.get("target_level")
    ):
        raise ValueError(
            "frontier binding correction does not identify one exact exec"
        )
    binding = validate_frontier_binding({
        field: record[field]
        for field in (
            *FRONTIER_BINDING_FIELDS,
            "game",
            "reached",
            "target_level",
            "parent_action_count",
        )
    })
    evidence = record["evidence"]
    if (
        not isinstance(evidence, dict)
        or set(evidence) != set(
            FRONTIER_BINDING_CORRECTION_EVIDENCE_FIELDS
        )
    ):
        raise ValueError(
            "frontier binding correction has noncanonical evidence"
        )
    baseline_commit = evidence["workspace_baseline_commit"]
    if (
        not isinstance(baseline_commit, str)
        or len(baseline_commit) not in {40, 64}
        or any(character not in "0123456789abcdef"
               for character in baseline_commit)
        or not all(
            _is_sha256(evidence[field])
            for field in (
                "baseline_checkpoint_sha256",
                "baseline_source_tree_sha256",
                "protected_transcript_sha256",
                "audit_receipt_sha256",
            )
        )
        or evidence["baseline_checkpoint_sha256"]
        != binding["parent_checkpoint_sha256"]
        or evidence["baseline_source_tree_sha256"]
        != binding["parent_source_tree_sha256"]
        or any(
            evidence[field] is not True
            for field in (
                "baseline_checkpoint_replay_verified",
                "workspace_git_history_unmodified",
                "terminal_turn_audited",
                "taint_scan_passed",
            )
        )
    ):
        raise ValueError(
            "frontier binding correction evidence is incomplete or inconsistent"
        )
    receipt_relpath = evidence["audit_receipt_relpath"]
    receipt_sha256 = evidence["audit_receipt_sha256"]
    if (
        not isinstance(receipt_relpath, str)
        or receipt_relpath != f"{receipt_sha256}.json"
        or Path(receipt_relpath).name != receipt_relpath
    ):
        raise ValueError(
            "frontier binding correction receipt path is not content-addressed"
        )
    root = (
        Path(receipt_root)
        if receipt_root is not None
        else HERE / "frontier_binding_receipts"
    )
    try:
        receipt_raw = _read_receipt_regular(root, receipt_relpath)
        receipt = json.loads(receipt_raw.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(
            "frontier binding correction receipt cannot be reopened safely"
        ) from exc
    expected_receipt = _binding_correction_receipt_payload(record)
    canonical_receipt = (
        json.dumps(
            expected_receipt,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        + b"\n"
    )
    if (
        receipt != expected_receipt
        or receipt_raw != canonical_receipt
        or hashlib.sha256(receipt_raw).hexdigest() != receipt_sha256
    ):
        raise ValueError(
            "frontier binding correction receipt is not canonical or "
            "does not seal the advertised evidence"
        )
    return binding


def exact_frontier_binding(
    artifact: Path,
    *,
    game: str,
    target_level: int,
) -> dict[str, Any]:
    """Bind a dispatch to the exact promoted checkpoint and source parent."""
    artifact = Path(artifact)
    checkpoint_path = artifact / "checkpoint.json"
    if not checkpoint_path.exists():
        reached = 0
        action_count = 0
        checkpoint_sha256 = ZERO_SHA256
        source_tree_sha256 = ZERO_SHA256
    else:
        checkpoint_raw = _read_stable_regular(checkpoint_path)
        try:
            checkpoint = json.loads(checkpoint_raw.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"invalid exact-parent checkpoint: {checkpoint_path}"
            ) from exc
        if not isinstance(checkpoint, dict):
            raise ValueError(
                f"exact-parent checkpoint is not an object: {checkpoint_path}"
            )
        reached = checkpoint.get("reached")
        final_path = checkpoint.get("final_path")
        if (
            checkpoint.get("game") != game
            or not isinstance(reached, int)
            or isinstance(reached, bool)
            or reached < 0
            or not isinstance(final_path, list)
        ):
            raise ValueError(
                f"exact-parent checkpoint has invalid lineage: {checkpoint_path}"
            )
        action_count = len(final_path)
        if reached == 0:
            checkpoint_sha256 = ZERO_SHA256
            source_tree_sha256 = ZERO_SHA256
            action_count = 0
        else:
            if checkpoint.get("validated") is not True:
                raise ValueError(
                    f"exact-parent checkpoint is not validated: "
                    f"{checkpoint_path}"
                )
            checkpoint_sha256 = hashlib.sha256(checkpoint_raw).hexdigest()
            payloads = {}
            for name in ("legs.py", "players.py", "solve.py"):
                path = artifact / name
                if not path.exists():
                    raise ValueError(
                        f"exact parent lacks required source file: {path}"
                    )
                payloads[name] = _read_stable_regular(path)
            source_tree_sha256 = _source_tree_sha256(payloads)
    binding = {
        "frontier_binding_schema": FRONTIER_BINDING_SCHEMA,
        "game": game,
        "reached": reached,
        "target_level": target_level,
        "parent_checkpoint_sha256": checkpoint_sha256,
        "parent_source_tree_sha256": source_tree_sha256,
        "parent_action_count": action_count,
        "frontier_sha256": _sha256_json(
            {
                "game": game,
                "reached": reached,
                "parent_checkpoint_sha256": checkpoint_sha256,
            }
        ),
    }
    return validate_frontier_binding(
        binding,
        expected_game=game,
        expected_target_level=target_level,
    )


def latest_wip_descriptor(
    artifact: Path,
    *,
    game: str,
    reached: int,
    target_level: int,
    frontier_binding: dict[str, Any],
) -> dict[str, Any]:
    """Validate the host-sealed latest WIP pointer for one exact frontier.

    Merely finding a ``wip_context`` directory is not enough to authorize a
    continuation.  The pointer, metadata, complete file inventory, and the
    snapshot-time exact-parent binding must all agree with the current
    promoted frontier.  The proposer runner re-runs the current taint scanner
    when it restores the selected capsule; this descriptor is the scheduler's
    path/binding gate, not a substitute for that dispatch-time scan.
    """
    unavailable = {
        "warm_wip_available": False,
        "warm_wip_attempt": None,
        "warm_wip_phase": None,
        "warm_wip_recovery_required": False,
        "warm_wip_validation": "unavailable",
    }
    level_dir = (
        Path(artifact) / "wip_context" / f"level_{target_level:02d}"
    )
    latest_path = level_dir / "latest.json"
    if not latest_path.exists():
        return unavailable
    try:
        latest_raw = _read_stable_regular(latest_path)
        latest = json.loads(latest_raw.decode("utf-8"))
        if not isinstance(latest, dict) or set(latest) != {
            "attempt", "metadata"
        }:
            raise ValueError("latest pointer has a noncanonical field set")
        attempt = latest["attempt"]
        if (
            not isinstance(attempt, str)
            or not attempt
            or Path(attempt).name != attempt
        ):
            raise ValueError("latest attempt is not one path component")
        attempt_dir = level_dir / attempt
        files_dir = attempt_dir / "files"
        metadata_path = attempt_dir / "metadata.json"
        if (
            attempt_dir.is_symlink()
            or files_dir.is_symlink()
            or not files_dir.is_dir()
        ):
            raise ValueError("latest attempt directory is absent or aliased")
        metadata_raw = _read_stable_regular(metadata_path)
        metadata = json.loads(metadata_raw.decode("utf-8"))
        if not isinstance(metadata, dict):
            raise ValueError("latest metadata is not an object")
        if metadata != latest["metadata"]:
            raise ValueError("latest pointer does not seal its metadata")
        if (
            metadata.get("filesystem_boundary_policy_schema")
            != Boundary.POLICY_SCHEMA
            or metadata.get("filesystem_boundary_policy_sha256")
            != Boundary.policy_sha256()
            or metadata.get("compatibility_arena_module_sha256")
            != Boundary.arena_module_sha256(HERE)
            or metadata.get("compatibility_boundary_authority")
            != "behavioral_defense_in_depth"
        ):
            return {
                **unavailable,
                "warm_wip_validation": (
                    "rejected:filesystem_boundary_policy_binding"
                ),
            }
        if (
            metadata.get("attempt") != attempt
            or metadata.get("game") != game
            or metadata.get("level") != target_level
            or metadata.get("reached") != reached
            or metadata.get("taint_verdict") != "clean"
        ):
            raise ValueError("latest metadata is not for this clean frontier")
        phase = metadata.get("phase")
        if not isinstance(phase, str) or not phase:
            raise ValueError("latest metadata lacks a phase")
        sealed_binding = validate_frontier_binding(
            metadata.get("frontier_binding"),
            expected_game=game,
            expected_target_level=target_level,
        )
        if sealed_binding != frontier_binding:
            raise ValueError("latest WIP belongs to a different exact parent")
        advertised_files = metadata.get("files")
        if (
            not isinstance(advertised_files, list)
            or not advertised_files
            or any(not isinstance(name, str) or not name for name in advertised_files)
            or len(set(advertised_files)) != len(advertised_files)
        ):
            raise ValueError("latest WIP has an invalid file inventory")
        actual_files = []
        for path in sorted(files_dir.rglob("*")):
            if path.is_symlink():
                raise ValueError("latest WIP inventory contains a symlink")
            if not path.is_file():
                continue
            relative = path.relative_to(files_dir).as_posix()
            if (
                Path(relative).is_absolute()
                or ".." in Path(relative).parts
            ):
                raise ValueError("latest WIP inventory escapes its capsule")
            _read_stable_regular(path)
            actual_files.append(relative)
        if sorted(advertised_files) != actual_files:
            raise ValueError("latest WIP file inventory is stale")
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        TypeError,
        ValueError,
    ) as exc:
        return {
            **unavailable,
            "warm_wip_validation": f"rejected:{exc}",
        }
    return {
        "warm_wip_available": True,
        "warm_wip_attempt": attempt,
        "warm_wip_phase": phase,
        "warm_wip_recovery_required": phase in INFRASTRUCTURE_WIP_PHASES,
        "warm_wip_validation": "exact_frontier_capsule",
    }


def _external_profiles(audits: Path = AUDITS) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = defaultdict(dict)
    for system, name in (
        ("baseline1", "baseline1_gpt55_xhigh_solved_checkpoints.json"),
        ("Retrodict", "retrodict-solved-checkpoint-memory.json"),
    ):
        path = audits / name
        if not path.exists():
            continue
        for row in _read_json(path).get("rows", []):
            game = row.get("game")
            completed = row.get("completed_levels")
            if isinstance(game, str) and isinstance(completed, int):
                result[game][system] = max(
                    result[game].get(system, 0), completed
                )
    return dict(result)


def _external_ceilings(audits: Path = AUDITS) -> dict[str, int]:
    return {
        game: max(profile.values())
        for game, profile in _external_profiles(audits).items()
        if profile
    }


def _authoritative_inventory(
    environments: Path = ENVIRONMENTS,
) -> dict[str, int]:
    """Return public game level counts from the downloaded toolkit metadata.

    ``baseline_actions`` has one entry per level.  Only its length is used;
    action values and game implementation files never enter proposer context.
    Comparator achievements are evidence about other systems, not an inventory,
    and therefore must not create nonexistent targets or hide real levels.
    """
    root = Path(environments).resolve()

    def pointer() -> tuple[tuple[object, ...], ...]:
        rows: list[tuple[object, ...]] = []
        for path in sorted(root.glob("*/*/metadata.json")):
            metadata = path.stat(follow_symlinks=False)
            if (
                path.is_symlink()
                or not stat.S_ISREG(metadata.st_mode)
                or metadata.st_nlink != 1
            ):
                raise ValueError(
                    f"unsafe authoritative metadata pointer: {path}"
                )
            rows.append((
                str(path),
                metadata.st_dev,
                metadata.st_ino,
                metadata.st_mode,
                metadata.st_nlink,
                metadata.st_size,
                metadata.st_mtime_ns,
                metadata.st_ctime_ns,
            ))
        return tuple(rows)

    before = pointer()
    cache_key = (str(root), before)
    cached = _inventory_cache.get(cache_key)
    if cached is not None:
        if pointer() != before:
            raise ValueError(
                "authoritative inventory changed while cached"
            )
        return dict(cached)
    result: dict[str, int] = {}
    for path in (Path(str(row[0])) for row in before):
        payload = _read_json(path)
        game = path.parents[1].name
        actions = payload.get("baseline_actions")
        if not isinstance(actions, list) or not actions:
            raise ValueError(f"missing authoritative baseline_actions: {path}")
        count = len(actions)
        previous = result.get(game)
        if previous is not None and previous != count:
            raise ValueError(
                f"conflicting authoritative level counts for {game}: "
                f"{previous} vs {count}"
            )
        result[game] = count
    if pointer() != before:
        raise ValueError(
            "authoritative inventory changed while it was inspected"
        )
    if len(_inventory_cache) >= _INVENTORY_CACHE_MAX_ENTRIES:
        _inventory_cache.pop(next(iter(_inventory_cache)))
    _inventory_cache[cache_key] = dict(result)
    return result


def _definition_count(path: Path) -> int:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError):
        return 0
    return sum(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        for node in tree.body
    )


def frontier_rows(artifacts: Path = ARTIFACTS,
                  audits: Path = AUDITS,
                  environments: Path = ENVIRONMENTS) -> list[dict[str, Any]]:
    profiles = _external_profiles(audits)
    inventory = _authoritative_inventory(environments)
    local: dict[str, tuple[Path, dict[str, Any]]] = {}
    for artifact in sorted(artifacts.glob("*_legs")):
        checkpoint_path = artifact / "checkpoint.json"
        if not checkpoint_path.exists():
            continue
        checkpoint = _read_json(checkpoint_path)
        game = checkpoint.get("game")
        if isinstance(game, str):
            local[game] = (artifact, checkpoint)

    rows = []
    for game in sorted(inventory):
        artifact, checkpoint = local.get(game, (None, {}))
        candidate_dir = artifacts / f"{game}_legs"
        reached = checkpoint.get("reached", 0)
        if not isinstance(reached, int):
            reached = 0
        next_level = reached + 1
        scaffold_path = (
            candidate_dir / "wip_context" / f"level_{next_level:02d}"
            / "frontier_scaffold.json"
        )
        scaffold = _read_json(scaffold_path) if scaffold_path.exists() else {}
        external = profiles.get(game, {})
        target = inventory[game]
        if reached > target:
            raise ValueError(
                f"canonical checkpoint exceeds authoritative inventory: "
                f"{game} reached={reached} target={target}"
            )
        if reached >= target:
            continue
        binding = exact_frontier_binding(
            candidate_dir,
            game=game,
            target_level=next_level,
        )
        wip = latest_wip_descriptor(
            candidate_dir,
            game=game,
            reached=reached,
            target_level=next_level,
            frontier_binding=binding,
        )
        warm_wip = wip["warm_wip_available"]
        sources = (
            [artifact / name for name in ("legs.py", "players.py", "solve.py")]
            if artifact is not None else []
        )
        source_bytes = sum(path.stat().st_size for path in sources if path.exists())
        definitions = (
            sum(_definition_count(artifact / name)
                for name in ("legs.py", "players.py"))
            if artifact is not None else 0
        )
        gap = target - reached
        # Operational heuristic only: reward a mature incumbent and a one-level
        # completion opportunity, while penalizing context that will be replayed
        # through every agent/tool iteration.
        if artifact is None:
            values = list(external.values())
            consensus_floor = min(values, default=0)
            spread = max(values, default=0) - consensus_floor
            # Cold L1 trials have essentially no retained-source context.  Rank
            # games solved deeply by both external systems ahead of repeatedly
            # paying for a stalled mature frontier.
            priority_score = 1.15 + 0.12 * consensus_floor - 0.08 * spread
            if warm_wip:
                priority_score += 0.2
        else:
            priority_score = reached / (1.0 + source_bytes / 10_000.0)
            if gap == 1:
                priority_score += 1.0
        rows.append({
            **binding,
            **wip,
            "game": game,
            "incumbent_kind": "promoted" if artifact is not None else "cold_start",
            "current_level": reached,
            "next_level": next_level,
            "frontier_scaffold_version": scaffold.get("version"),
            "frontier_scaffold_created_at": scaffold.get("created_at"),
            "authoritative_level_count": target,
            "external_evidence": external,
            "levels_to_authoritative_completion": gap,
            "solver_source_bytes": source_bytes,
            "top_level_definitions": definitions,
            "priority_score": round(priority_score, 3),
        })
    return sorted(rows, key=lambda row: (-row["priority_score"], row["game"]))


def effort_efficiency(turns: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Summarize proposal yield by effort without implying a randomized comparison."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for turn in turns:
        if not str(turn.get("run_label") or "").endswith(":propose"):
            continue
        effort = turn.get("reasoning_effort")
        if isinstance(effort, str):
            grouped[effort].append(turn)

    result: dict[str, dict[str, Any]] = {}
    for effort, all_rows in sorted(grouped.items()):
        infrastructure = [
            row for row in all_rows
            if row.get("failure_class") == "infrastructure"
        ]
        rows = [
            row for row in all_rows
            if _is_solver_attempt(row)
        ]
        non_solver = [row for row in all_rows if not _is_solver_attempt(row)]
        solved = [row for row in rows if row.get("solved_target") is True]
        failed = [row for row in rows if _is_clean_no_progress(row)]
        points = [row.get("displayed_weekly_points_used") for row in rows]
        known_points = [value for value in points if isinstance(value, int)]
        success_points = [
            row.get("displayed_weekly_points_used") for row in solved
            if isinstance(row.get("displayed_weekly_points_used"), int)
        ]
        duration = sum(float(row.get("duration_seconds") or 0.0) for row in rows)
        missing_usage = sum(
            1 for row in rows
            if not isinstance(row.get("observed_tokens"), int)
            or row.get("observed_tokens") == 0
        )
        result[effort] = {
            "proposal_attempts": len(rows),
            "infrastructure_turns_excluded": len(infrastructure),
            "non_solver_turns_excluded": len(non_solver),
            "solved_levels": len(solved),
            "failed_levels": len(failed),
            "unknown_outcomes": len(rows) - len(solved) - len(failed),
            "timed_out_turns": sum(bool(row.get("timed_out")) for row in rows),
            "displayed_weekly_points": sum(known_points),
            "displayed_points_on_successes": sum(success_points),
            "displayed_points_per_solved_level": (
                round(sum(known_points) / len(solved), 3) if solved else None
            ),
            "success_only_points_per_solved_level": (
                round(sum(success_points) / len(solved), 3) if solved else None
            ),
            "displayed_points_per_wall_minute": (
                round(sum(known_points) / (duration / 60.0), 3)
                if duration else None
            ),
            "turns_with_missing_token_usage": missing_usage,
            "observed_tokens_are_complete": missing_usage == 0,
        }
    return result


def _is_clean_no_progress(turn: dict[str, Any]) -> bool:
    explicit = turn.get("clean_no_progress")
    if isinstance(explicit, bool):
        return explicit
    taint = turn.get("taint_verdict")
    return bool(
        turn.get("solved_target") is False
        and turn.get("failure_class") is None
        and not turn.get("timed_out")
        and not turn.get("interrupted")
        and taint in {None, "clean"}
    )


def _is_solver_attempt(turn: dict[str, Any]) -> bool:
    if turn.get("failure_class") is not None:
        return False
    if turn.get("solved_target") is True:
        return True
    return _is_clean_no_progress(turn)


def effort_efficiency_by_phase(
    turns: list[dict[str, Any]],
) -> dict[str, dict[str, dict[str, Any]]]:
    """Separate cold game entry from retained-solver continuation attempts.

    The two phases have visibly different difficulty and context sizes.  A
    pooled medium/high comparison is therefore useful for bookkeeping but not
    for deciding which arm is cheaper on a continuation frontier.
    """
    phases = {
        "cold_L1": [
            turn for turn in turns
            if turn.get("target_level") == 1
        ],
        "continuation_L2_plus": [
            turn for turn in turns
            if isinstance(turn.get("target_level"), int)
            and turn["target_level"] >= 2
        ],
    }
    return {
        phase: effort_efficiency(rows)
        for phase, rows in phases.items()
    }


def effort_solve_quality(
    turns: list[dict[str, Any]], audits: Path = AUDITS
) -> dict[str, dict[str, Any]]:
    """Join paid solves to exact GKM checkpoints and summarize solver structure.

    The conditional normalized-AST marginal is an executable-artifact proxy for
    description length. It is not inferred from transcript or episode length.
    Existing turns were not matched by frontier difficulty, so the result is
    descriptive rather than a causal high-versus-medium estimate.
    """
    path = audits / "marginal-literal-reuse.json"
    audit_rows: dict[tuple[str, int], dict[str, Any]] = {}
    if path.exists():
        for row in _read_json(path).get("rows", []):
            game, level = row.get("game"), row.get("completed_level")
            if (
                row.get("system") == "GKM"
                and row.get("source_checkpoint_exact") is True
                and isinstance(game, str)
                and isinstance(level, int)
            ):
                audit_rows[(game, level)] = row

    grouped: dict[str, list[tuple[dict[str, Any], Optional[dict[str, Any]]]]] = (
        defaultdict(list)
    )
    for turn in turns:
        if (
            not str(turn.get("run_label") or "").endswith(":propose")
            or turn.get("solved_target") is not True
        ):
            continue
        effort, game, level = (
            turn.get("reasoning_effort"),
            turn.get("game"),
            turn.get("target_level"),
        )
        if isinstance(effort, str) and isinstance(game, str) and isinstance(level, int):
            grouped[effort].append((turn, audit_rows.get((game, level))))

    result: dict[str, dict[str, Any]] = {}
    for effort, pairs in sorted(grouped.items()):
        audited = [row for _, row in pairs if row is not None]
        ast_marginals = [
            row.get("marginal_ast_zlib_bytes")
            for row in audited
            if isinstance(row.get("marginal_ast_zlib_bytes"), int)
        ]
        acquisition_charges = [
            turn.get("winning_marginal_C")
            for turn, _ in pairs
            if isinstance(turn.get("winning_marginal_C"), int)
        ]
        result[effort] = {
            "solved_levels": len(pairs),
            "exact_checkpoint_coverage": len(audited),
            "median_conditional_ast_zlib_bytes": (
                float(median(ast_marginals)) if ast_marginals else None
            ),
            "median_pre_debrief_acquisition_charge": (
                float(median(acquisition_charges))
                if acquisition_charges else None
            ),
            "literal_reuse_wins": sum(
                row.get("hard_literal_reuse_witness") is True for row in audited
            ),
            "sharp_marginal_drop_wins": sum(
                row.get("sharp_marginal_drop") is True for row in audited
            ),
            "sharp_drop_with_literal_reuse_wins": sum(
                row.get("sharp_drop_with_literal_reuse") is True for row in audited
            ),
            "checkpoint_details": [
                {
                    "game": turn["game"],
                    "completed_level": turn["target_level"],
                    "marginal_ast_zlib_bytes": (
                        row.get("marginal_ast_zlib_bytes") if row else None
                    ),
                    "pre_debrief_acquisition_charge": turn.get("winning_marginal_C"),
                    "literal_reuse": (
                        row.get("hard_literal_reuse_witness") if row else None
                    ),
                    "sharp_marginal_drop": (
                        row.get("sharp_marginal_drop") if row else None
                    ),
                }
                for turn, row in pairs
            ],
        }
    return result


def _iso_epoch(value: Any) -> Optional[float]:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.timestamp()


# Pre-adaptive static wall-time caps, kept as the conservative fallback when the
# ledger has too few replay-validated solves in an arm to size it empirically.
STATIC_WALL_MINUTES = {
    ("cold_L1", "medium"): 6,
    ("cold_L1", "high"): 6,
    ("continuation_L2+", "medium"): 8,
    ("continuation_L2+", "high"): 8,
}
# An arm needs at least this many replay-validated solves before its own solve-time
# distribution overrides the static cap.  Below it, thin/noisy evidence is not trusted.
MIN_SOLVES_TO_SIZE = 3
# Solve-preserving margin over the slowest observed solve.  Solves cluster against
# the historical cap (right-censored), so the true tail can exceed what we have seen;
# the margin decensors without wildly over-allocating.  The user's rule: never
# truncate good continuation WIP to save a few minutes.
WALL_SAFETY_FACTOR = 1.15
# Per-effort floors mirror the headroom floors; the ceiling is a secondary safety cap.
WALL_MINUTES_FLOOR = {"medium": 5, "high": 6}
WALL_MINUTES_CEILING = 15


def _phase_of_level(level: Any) -> Optional[str]:
    """Map a target level to the cold-entry vs retained-WIP continuation phase."""
    if level == 1:
        return "cold_L1"
    if isinstance(level, int) and level >= 2:
        return "continuation_L2+"
    return None


def _validated_solve_minutes(
    phase: str, effort: str, turns: list[dict[str, Any]]
) -> list[float]:
    """Wall minutes of replay-validated proposal solves in one (phase, effort) arm."""
    minutes = []
    for turn in turns:
        if not str(turn.get("run_label") or "").endswith(":propose"):
            continue
        if turn.get("solved_target") is not True:
            continue
        if turn.get("reasoning_effort") != effort:
            continue
        if _phase_of_level(turn.get("target_level")) != phase:
            continue
        duration = turn.get("duration_seconds")
        if isinstance(duration, (int, float)) and duration > 0:
            minutes.append(float(duration) / 60.0)
    return minutes


def recommend_minutes(
    phase: Optional[str], effort: str, turns: list[dict[str, Any]]
) -> dict[str, Any]:
    """Solve-preserving adaptive wall-time for a (phase, effort) arm.

    The binding constraint is that no historically replay-validated solve would
    have been truncated: the recommendation covers the slowest such solve plus a
    censoring margin.  With fewer than ``MIN_SOLVES_TO_SIZE`` solves the arm keeps
    its conservative static cap.  Returns the minutes plus provenance so the plan
    is auditable and the recommendation is never a bare unexplained number.
    """
    static = STATIC_WALL_MINUTES.get((phase, effort), 8)
    solves = (
        _validated_solve_minutes(phase, effort, turns)
        if phase is not None else []
    )
    if len(solves) < MIN_SOLVES_TO_SIZE:
        return {
            "minutes": static,
            "basis": "static_fallback",
            "solve_samples": len(solves),
            "slowest_solve_minutes": (
                round(max(solves), 2) if solves else None
            ),
        }
    slowest = max(solves)
    floor = WALL_MINUTES_FLOOR.get(effort, 5)
    needed = math.ceil(slowest * WALL_SAFETY_FACTOR)
    minutes = max(floor, min(WALL_MINUTES_CEILING, needed))
    # Hard guarantee of the solve-preserving property even if the ceiling binds.
    minutes = max(minutes, math.ceil(slowest))
    return {
        "minutes": minutes,
        "basis": "empirical_solve_preserving",
        "solve_samples": len(solves),
        "slowest_solve_minutes": round(slowest, 2),
    }


def retry_policy(n: int) -> dict[str, Any]:
    """Project one exact-frontier clean-retry coordinate onto both ladders."""
    if not isinstance(n, int) or isinstance(n, bool) or n < 0:
        raise ValueError("retry coordinate n must be a nonnegative integer")
    table = {
        0: ("medium", 15, "exclude", "fresh_frontier"),
        1: ("high", 20, "restore_clean_same_frontier", "continue_clean_wip"),
        2: ("xhigh", 25, "restore_clean_same_frontier", "continue_clean_wip"),
        3: ("xhigh", 40, "restore_clean_same_frontier", "warm_hard_frontier"),
        4: ("max", 60, "restore_clean_same_frontier", "first_max"),
        5: ("max", 90, "exclude", "max_coherence_reset"),
        6: ("max", 120, "restore_clean_same_frontier", "max_cumulative"),
        7: ("max", 180, "exclude", "repeated_hard_frontier_reset"),
        8: (
            "max",
            180,
            "restore_clean_same_frontier",
            "repeated_hard_frontier_continuation",
        ),
    }
    if n in table:
        effort, minutes, wip_mode, mode = table[n]
    elif n % 2:
        effort, minutes, wip_mode, mode = (
            "max",
            300,
            "exclude",
            "long_coherence_reset",
        )
    else:
        effort, minutes, wip_mode, mode = (
            "max",
            300,
            "restore_clean_same_frontier",
            "long_coherence_cumulative",
        )
    sidecars = 0 if n < 5 else 1 if n < 7 else 2
    return {
        "n": n,
        "effort": effort,
        "minutes": minutes,
        "wip_mode": wip_mode,
        "dispatch_mode": mode,
        "auxiliary_parallelism": sidecars,
    }


def ranked_frontiers(frontiers: list[dict[str, Any]],
                     turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Project exact-parent-bound clean retries onto the unified effort policy."""
    attempts: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for turn in turns:
        game, level = turn.get("game"), turn.get("target_level")
        if isinstance(game, str) and isinstance(level, int):
            attempts[(game, level)].append(turn)

    ranked = []
    for row in frontiers:
        row_binding = validate_frontier_binding({
            field: row.get(field)
            for field in (
                *FRONTIER_BINDING_FIELDS,
                "game",
                "reached",
                "target_level",
                "parent_action_count",
            )
        })
        all_history = attempts[(row["game"], row["next_level"])]
        exact_history = []
        unbound_history = []
        superseded_history = []
        for turn in all_history:
            try:
                turn_binding = validate_frontier_binding({
                    field: turn.get(field)
                    for field in (
                        *FRONTIER_BINDING_FIELDS,
                        "game",
                        "reached",
                        "target_level",
                        "parent_action_count",
                    )
                })
            except ValueError:
                # Pre-binding turns and malformed/partial retrospective
                # corrections remain visible for diagnosis, but they have no
                # authority to move the retry coordinate.
                unbound_history.append(turn)
                continue
            if turn_binding == row_binding:
                exact_history.append(turn)
            else:
                superseded_history.append(turn)
        infrastructure_history = [
            turn for turn in exact_history
            if turn.get("failure_class") == "infrastructure"
        ]
        prior = [turn for turn in exact_history if _is_solver_attempt(turn)]
        failures = [turn for turn in prior if _is_clean_no_progress(turn)]
        non_solver = [
            turn for turn in exact_history if not _is_solver_attempt(turn)
        ]
        failed_efforts = sorted({
            str(turn.get("reasoning_effort")) for turn in failures
            if turn.get("reasoning_effort")
        })
        policy = retry_policy(len(failures))
        adjusted = float(row["priority_score"]) - 0.8 * len(failures)
        ranked.append({
            **row,
            "paid_attempts_at_frontier": len(prior),
            "infrastructure_turns_at_frontier": len(
                infrastructure_history
            ),
            "non_solver_turns_at_frontier": len(non_solver),
            "superseded_attempts_at_frontier": len(superseded_history),
            "unbound_legacy_turns_for_game_level": len(unbound_history),
            "exact_bound_turns_at_frontier": len(exact_history),
            "game_level_history_turns": len(all_history),
            "failed_attempts_at_frontier": len(failures),
            "retry_complexity_n": policy["n"],
            "retry_history_authority": "exact_parent_bound_only",
            "failed_efforts": failed_efforts,
            "quarantined_after_escalation_failure": False,
            "recommended_effort": policy["effort"],
            "recommended_minutes": policy["minutes"],
            "recommended_wip_mode": policy["wip_mode"],
            "recommended_auxiliary_parallelism": policy[
                "auxiliary_parallelism"
            ],
            "recommended_minutes_basis": (
                "versioned_exact_frontier_clean_retry_ladder"
            ),
            "recommended_minutes_solve_samples": None,
            "slowest_validated_solve_minutes": None,
            "dispatch_mode": policy["dispatch_mode"],
            "adjusted_priority_score": round(adjusted, 3),
        })
    return sorted(
        ranked,
        key=lambda row: (-row["adjusted_priority_score"], row["game"]),
    )


def _transcript_counts(record: dict[str, Any]) -> dict[str, int]:
    workspace = record.get("workspace")
    transcript = record.get("transcript")
    result = {
        "command_executions": 0,
        "file_changes": 0,
        "turn_completed_events": 0,
    }
    if not isinstance(workspace, str) or not isinstance(transcript, str):
        return result
    # ``gkm_legs`` seals the authoritative transcript outside the proposer-
    # writable workspace before that workspace may be retired.  Retry
    # accounting must follow the sealed copy; otherwise ordinary cleanup turns
    # a completed clean attempt back into an unknown outcome and silently
    # lowers the escalation coordinate.  Accept only single path components so
    # a malformed ledger entry cannot escape either transcript root.
    if (
        not workspace
        or Path(workspace).name != workspace
        or not transcript
        or Path(transcript).name != transcript
    ):
        return result
    scratch = HERE / "runs" / "scratch"
    protected = scratch / ".proposer_transcripts" / workspace / transcript
    live = scratch / workspace / transcript
    path = protected if protected.exists() else live
    if not path.exists():
        return result
    try:
        raw_transcript = _read_stable_regular(path).decode(
            "utf-8", errors="ignore"
        )
    except OSError:
        return result
    for raw in raw_transcript.splitlines():
        try:
            event = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "turn.completed":
            result["turn_completed_events"] += 1
        item = event.get("item", {})
        if event.get("type") != "item.completed" or not isinstance(item, dict):
            continue
        if item.get("type") == "command_execution":
            result["command_executions"] += 1
        elif item.get("type") == "file_change":
            result["file_changes"] += 1
    return result


def joined_turns(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    outcomes = {
        row.get("thread_id"): row
        for row in records
        if row.get("event") == "codex_level_outcome" and row.get("thread_id")
    }
    corrections_by_thread = {
        row.get("thread_id"): row
        for row in records
        if (
            row.get("event") == "codex_exec_classification_correction"
            and row.get("thread_id")
        )
    }
    corrections_by_transcript = {
        row.get("transcript"): row
        for row in records
        if (
            row.get("event") == "codex_exec_classification_correction"
            and row.get("transcript")
        )
    }
    binding_corrections_by_thread: dict[
        str, list[dict[str, Any]]
    ] = defaultdict(list)
    for record in records:
        if (
            record.get("event") == "codex_frontier_binding_correction"
            and isinstance(record.get("thread_id"), str)
        ):
            binding_corrections_by_thread[record["thread_id"]].append(
                record
            )
    result = []
    for row in records:
        if row.get("event") != "codex_exec":
            continue
        outcome = outcomes.get(row.get("thread_id"), {})
        correction = (
            corrections_by_thread.get(row.get("thread_id"))
            or corrections_by_transcript.get(row.get("transcript"))
            or {}
        )
        transcript = _transcript_counts(row)
        failure_class = correction.get(
            "failure_class", row.get("failure_class")
        )
        solved_target = (
            correction["solved_target"]
            if "solved_target" in correction
            else outcome.get("solved_target")
        )
        taint_verdict = correction.get(
            "taint_verdict", outcome.get("taint_verdict")
        )
        transcript_complete = transcript["turn_completed_events"] == 1
        clean_no_progress = bool(
            solved_target is False
            and failure_class is None
            and not row.get("timed_out")
            and not row.get("interrupted")
            and taint_verdict == "clean"
            and transcript_complete
        )
        before, after = row.get("weekly_remaining_before"), row.get("weekly_remaining_after")
        weekly_delta = before - after if isinstance(before, int) and isinstance(after, int) else None
        binding_claims = []
        binding_claim_invalid = False
        for binding_record in (row, outcome):
            binding, declared = _record_frontier_binding(binding_record)
            if declared and binding is None:
                binding_claim_invalid = True
            elif binding is not None:
                binding_claims.append(binding)
        binding_corrections = binding_corrections_by_thread.get(
            row.get("thread_id"), []
        )
        correction_binding = None
        if binding_corrections:
            # One canonical append is deliberate. Multiple, malformed, or
            # conflicting retrospective claims leave the turn unbound.
            if len(binding_corrections) != 1:
                binding_claim_invalid = True
            else:
                try:
                    correction_binding = (
                        validate_frontier_binding_correction(
                            binding_corrections[0],
                            exec_record=row,
                        )
                    )
                except ValueError:
                    binding_claim_invalid = True
                else:
                    binding_claims.append(correction_binding)
        distinct_bindings = {
            json.dumps(binding, sort_keys=True, separators=(",", ":"))
            for binding in binding_claims
        }
        if binding_claim_invalid or len(distinct_bindings) > 1:
            selected_binding = None
            binding_authority = "unbound_conflicting_or_malformed"
        elif binding_claims:
            selected_binding = binding_claims[0]
            binding_authority = (
                "retrospective_receipt_backed_claim"
                if correction_binding is not None
                else "prospective_exec"
            )
        else:
            selected_binding = None
            binding_authority = "unbound_legacy"
        frontier_binding = {
            field: (
                selected_binding.get(field)
                if selected_binding is not None
                else None
            )
            for field in (
                *FRONTIER_BINDING_FIELDS,
                "reached",
                "parent_action_count",
            )
        }
        result.append({
            "thread_id": row.get("thread_id"),
            "transcript": row.get("transcript"),
            "started_at": row.get("started_at"),
            "run_label": row.get("run_label"),
            "failure_class": failure_class,
            "failure_detail_class": correction.get(
                "failure_detail_class", row.get("failure_detail_class")
            ),
            "terminal_errors": correction.get(
                "terminal_errors", row.get("terminal_errors")
            ),
            "model": row.get("model"),
            "reasoning_effort": row.get("reasoning_effort"),
            "duration_seconds": row.get("duration_seconds"),
            "minutes_limit": row.get("minutes_limit"),
            "allocation_policy": row.get("allocation_policy"),
            "allocation_expired": row.get("allocation_expired"),
            "timed_out": row.get("timed_out"),
            "interrupted": row.get("interrupted"),
            "returncode": row.get("returncode"),
            "observed_tokens": row.get("observed_tokens"),
            "cached_input_tokens": row.get("cached_input_tokens"),
            "reasoning_output_tokens": row.get("reasoning_output_tokens"),
            "weekly_remaining_before": before,
            "weekly_remaining_after": after,
            "displayed_weekly_points_used": weekly_delta,
            "solved_target": solved_target,
            "clean_no_progress": clean_no_progress,
            "retry_increment": correction.get(
                "retry_increment", int(clean_no_progress)
            ),
            "game": outcome.get(
                "game", correction.get("game", row.get("game"))
            ),
            "target_level": outcome.get(
                "target_level",
                correction.get("target_level", row.get("target_level")),
            ),
            "winning_marginal_C": outcome.get("winning_marginal_C"),
            "taint_verdict": taint_verdict,
            "transcript_complete": transcript_complete,
            "frontier_binding_authority": binding_authority,
            **frontier_binding,
            **transcript,
        })
    return result


def infrastructure_noncounting_events(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return scheduler failures that deliberately are not solver turns."""

    return [
        row for row in records
        if (
            row.get("event") == INFRASTRUCTURE_NONCOUNTING_EVENT
            and row.get("schema")
            == "scheduler_zero_ledger_generation_quarantine_v1"
            and row.get("failure_class") == "infrastructure"
            and row.get("retry_increment") == 0
            and row.get("codex_exec_appended") is False
        )
    ]


def _joined_window_turns(
    records: list[dict[str, Any]], exec_records: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    thread_ids = {
        row.get("thread_id") for row in exec_records if row.get("thread_id")
    }
    outcomes = [
        row for row in records
        if row.get("event") == "codex_level_outcome"
        and row.get("thread_id") in thread_ids
    ]
    corrections = [
        row for row in records
        if row.get("event") == "codex_exec_classification_correction"
        and row.get("thread_id") in thread_ids
    ]
    return joined_turns([*exec_records, *outcomes, *corrections])


def _allowance_from_records(records: list[dict[str, Any]],
                            turns: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
    # Prefer the newest record that explicitly identifies the provider window.
    # A legacy postflight containing only ``remaining=100`` must not erase a
    # newer semantic fact that the provider called the pool ``unlimited``.
    for record in reversed(records):
        if record.get("event") == "rate_limit_snapshot":
            allowance = record.get("allowance")
            if isinstance(allowance, dict) and isinstance(
                allowance.get("remaining_percent"), int
            ):
                return {**allowance, "source": "cached_live_rate_limit_read"}
        if record.get("event") == "codex_exec":
            window_name = record.get("weekly_window_after")
            remaining = record.get("weekly_remaining_after")
            if isinstance(window_name, str) and isinstance(remaining, int):
                return {
                    "remaining_percent": remaining,
                    "resets_at": record.get("weekly_resets_at"),
                    "window_name": window_name,
                    "limit_id": record.get("weekly_limit_id_after"),
                    "source": "explicit_postflight",
                }
    # Compatibility fallback for historical finite-pool records that predate
    # explicit window metadata.
    for record in reversed(records):
        if record.get("event") == "codex_exec":
            remaining = record.get("weekly_remaining_after")
            if isinstance(remaining, int):
                return {
                    "remaining_percent": remaining,
                    "resets_at": record.get("weekly_resets_at"),
                    "source": "last_postflight",
                }
    # Retain compatibility for synthetic callers that pass joined turns only.
    for turn in reversed(turns):
        remaining = turn.get("weekly_remaining_after")
        if isinstance(remaining, int):
            return {"remaining_percent": remaining, "source": "last_postflight"}
    return None


def _readiness(remaining: Optional[int], reserve: int,
               medium_headroom: int, high_headroom: int,
               totals: dict[str, int], max_runs: int,
               max_tokens: int) -> dict[str, Any]:
    local_budget_ok = (
        (max_runs < 0 or totals["runs"] < max_runs)
        and (max_tokens < 0 or totals["observed_tokens"] < max_tokens)
    )

    def ready(required: int) -> bool:
        return bool(
            remaining is not None
            and remaining > reserve
            and remaining - reserve >= required
            and local_budget_ok
        )

    return {
        "reserve_percent": reserve,
        "medium_required_headroom_percent": medium_headroom,
        "high_required_headroom_percent": high_headroom,
        "available_headroom_percent": remaining - reserve if remaining is not None else None,
        "local_budget_ok": local_budget_ok,
        "medium_admissible": ready(medium_headroom),
        "high_admissible": ready(high_headroom),
    }


def campaign_report(*, ledger: Path = Guard.DEFAULT_LEDGER,
                    artifacts: Path = ARTIFACTS, audits: Path = AUDITS,
                    environments: Path = ENVIRONMENTS,
                    live_snapshot: Optional[dict[str, Any]] = None,
                    reserve: int = 20, medium_headroom: int = 4,
                    high_headroom: int = 6, max_runs: int = 60,
                    max_tokens: int = 32_000_000) -> dict[str, Any]:
    records = Guard.read_ledger(ledger)
    turns = joined_turns(records)
    window_turns: list[dict[str, Any]] = []
    allowance = None
    local_totals = Guard.local_window_totals([])
    if live_snapshot is not None:
        live = Guard.weekly_allowance(live_snapshot)
        allowance = {**live.as_dict(), "source": "live_rate_limit_read"}
        current = Guard.current_window_records(records, live)
        local_totals = Guard.local_window_totals(current)
        window_turns = _joined_window_turns(records, current)
    else:
        allowance = _allowance_from_records(records, turns)
        if turns:
            reset = allowance.get("resets_at") if allowance else None
            if not isinstance(reset, int):
                reset = next(
                    (row.get("weekly_resets_at") for row in reversed(records)
                     if row.get("event") == "codex_exec"),
                    None,
                )
            if allowance is not None:
                allowance.setdefault("resets_at", reset)
                allowance.setdefault(
                    "resets_at_iso",
                    datetime.fromtimestamp(reset, timezone.utc).isoformat()
                    if isinstance(reset, int) else None,
                )
            current = [
                row for row in records
                if row.get("event") == "codex_exec"
                and isinstance(row.get("weekly_resets_at"), int)
                and isinstance(reset, int)
                and abs(row["weekly_resets_at"] - reset)
                <= Guard.RESET_EPOCH_TOLERANCE_SECONDS
            ]
            local_totals = Guard.local_window_totals(current)
            window_turns = _joined_window_turns(records, current)

    unlimited = bool(allowance and allowance.get("window_name") == "unlimited")
    remaining = allowance.get("remaining_percent") if allowance else None
    inventory = _authoritative_inventory(environments)
    if len(inventory) != 25 or sum(inventory.values()) != 183:
        raise ValueError(
            "authoritative ARC inventory must contain 25 games / 183 levels, "
            f"found {len(inventory)} games / {sum(inventory.values())} levels"
        )
    frontiers = ranked_frontiers(
        frontier_rows(artifacts, audits, environments), turns
    )
    remaining_levels = sum(
        inventory[row["game"]] - int(row["current_level"])
        for row in frontiers
    )
    solved_levels = sum(inventory.values()) - remaining_levels
    readiness = _readiness(
        remaining,
        0 if unlimited else reserve,
        medium_headroom,
        high_headroom,
        local_totals,
        -1 if unlimited else max_runs,
        -1 if unlimited else max_tokens,
    )
    readiness["cost_control_enabled"] = not unlimited
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "allowance": allowance,
        "local_window": local_totals,
        "readiness": readiness,
        "turns": turns,
        "infrastructure_noncounting_events": (
            infrastructure_noncounting_events(records)
        ),
        "effort_efficiency": effort_efficiency(turns),
        "window_effort_efficiency": effort_efficiency(window_turns),
        "effort_efficiency_by_phase": effort_efficiency_by_phase(turns),
        "window_effort_efficiency_by_phase": effort_efficiency_by_phase(
            window_turns
        ),
        "solver_quality_by_effort": effort_solve_quality(turns, audits),
        "window_solver_quality_by_effort": effort_solve_quality(
            window_turns, audits
        ),
        "effort_comparison_identified": False,
        "effort_comparison_note": (
            "medium and high were not randomized or matched by frontier difficulty; "
            "cost and exact-checkpoint solver-quality summaries are descriptive, "
            "not a causal estimate"
        ),
        "authoritative_inventory": {
            "games": len(inventory),
            "levels": sum(inventory.values()),
            "per_game": inventory,
        },
        "canonical_progress": {
            "solved_levels": solved_levels,
            "total_levels": sum(inventory.values()),
            "remaining_levels": remaining_levels,
            "percent": round(100.0 * solved_levels / sum(inventory.values()), 4),
        },
        "frontiers": frontiers,
        "recommended_frontier": frontiers[0] if frontiers else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--ledger", type=Path, default=Guard.DEFAULT_LEDGER)
    parser.add_argument("--reserve-percent", type=int, default=20)
    parser.add_argument("--medium-headroom-percent", type=int, default=4)
    parser.add_argument("--high-headroom-percent", type=int, default=6)
    parser.add_argument("--max-campaign-runs", type=int, default=60)
    parser.add_argument("--max-campaign-tokens", type=int, default=32_000_000)
    args = parser.parse_args()
    snapshot = Guard.query_rate_limits() if args.live else None
    if snapshot is not None:
        live_allowance = Guard.weekly_allowance(snapshot)
        with Guard.campaign_lock(args.ledger):
            Guard.append_ledger({
                "event": "rate_limit_snapshot",
                "checked_at": datetime.now(timezone.utc).isoformat(),
                "allowance": live_allowance.as_dict(),
            }, args.ledger)
    report = campaign_report(
        ledger=args.ledger,
        live_snapshot=snapshot,
        reserve=args.reserve_percent,
        medium_headroom=args.medium_headroom_percent,
        high_headroom=args.high_headroom_percent,
        max_runs=args.max_campaign_runs,
        max_tokens=args.max_campaign_tokens,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
