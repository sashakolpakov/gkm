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
import fcntl
import glob
import hashlib
import json
import os
import re
import shlex
import shutil
import stat
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import codex_campaign_policy as Policy
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


class CampaignPlanError(RuntimeError):
    pass


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


def _runner_receipt(plan: dict[str, Any]) -> dict[str, Any] | None:
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
    return dict(receipt)


def _project_runner_receipt(
    plan: dict[str, Any], item: dict[str, Any]
) -> dict[str, Any]:
    """Project one plan-level historical runner onto any policy-built item."""

    receipt = _runner_receipt(plan)
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


def _validate_runner_prefix(item: dict[str, Any], argv: list[str]) -> None:
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
    authority = _runner_receipt({"runner_receipt": authority})
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


def _read_ledger_locked(path: Path) -> list[dict[str, Any]]:
    with Guard.ledger_append_lock(path):
        return Guard.read_ledger(path)


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
    before: list[dict[str, Any]],
    after: list[dict[str, Any]],
) -> dict[str, Any]:
    """Select exactly one newly appended exec bound to this dispatch."""

    if len(after) < len(before) or after[:len(before)] != before:
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
    matches = [
        record for record in after[len(before):]
        if all(record.get(key) == value for key, value in expected.items())
    ]
    if len(matches) != 1:
        raise CampaignPlanError(
            "nonzero child did not append exactly one exact-dispatch Codex exec "
            f"record (found {len(matches)})"
        )
    return matches[0]


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

    level = (
        _artifact_root(item)
        / f"{item['game']}_legs"
        / "wip_context"
        / f"level_{int(item['target_level']):02d}"
    )
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
    item: dict[str, Any], record: dict[str, Any]
) -> tuple[Path, Path, str, str, str | None]:
    """Authenticate and independently rescan one failed generation pair."""

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
        if evidence_schema == "sealed_transcript_only_v1":
            # The submitted historical runner predates the current exact host
            # scaffold hashes.  Its receipt authenticates those host-owned
            # bytes, so use the legacy marker surface plus the exact immutable
            # execution transcript and require a concrete forbidden finding.
            reason = Legs._file_taint_reason(
                os.fspath(protected / transcript_name), transcript_name
            ) or Legs._workspace_marker_taint_reason(os.fspath(workspace))
        else:
            reason = Legs._workspace_or_protected_taint_reason(
                os.fspath(workspace)
            )
    except Exception as exc:
        raise CampaignPlanError("exact generation taint rescan failed") from exc
    if not isinstance(reason, str) or not reason:
        raise CampaignPlanError(
            "nonzero child has no independently confirmed generation taint"
        )
    return workspace, protected, reason, transcript_sha, diagnostics_sha


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


def _recover_confirmed_taint(
    item: dict[str, Any],
    *,
    ledger: Path,
    ledger_before: list[dict[str, Any]],
    reached_before: int,
    wip_snapshot_before: tuple[str, str | None],
    child_returncode: int,
) -> dict[str, Any]:
    """Quarantine one exact tainted generation and keep its frontier retryable."""

    record = _expected_exec_record(
        item, ledger_before, _read_ledger_locked(ledger)
    )
    _safe_component(record.get("thread_id"), "thread id")
    workspace, protected, reason, transcript_sha, diagnostics_sha = (
        _exact_tainted_generation(item, record)
    )
    if _workspace_lock_is_active(workspace):
        raise CampaignPlanError(
            "refusing cleanup while the exact tainted workspace remains active"
        )
    if _checkpoint_reached(item["game"]) != reached_before:
        raise CampaignPlanError(
            "nonzero tainted child changed the canonical checkpoint"
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
            "nonzero tainted child changed the canonical exact frontier"
        )
    if _target_wip_snapshot(item) != wip_snapshot_before:
        raise CampaignPlanError(
            "tainted generation changed the exact-frontier WIP inventory"
        )

    _append_taint_correction(
        ledger,
        item,
        record,
        reason=reason,
        transcript_sha=transcript_sha,
        diagnostics_sha=diagnostics_sha,
    )
    _cleanup_exact_generation(item, workspace, protected)
    _append_cleanup_completion(ledger, item, record)
    _taint_gate()
    if (
        _checkpoint_reached(item["game"]) != reached_before
        or _canonical_frontier_binding(item) != expected_binding
        or _target_wip_snapshot(item) != wip_snapshot_before
    ):
        raise CampaignPlanError(
            "canonical frontier changed during tainted generation cleanup"
        )
    _assert_same_retry_coordinate(ledger, item, record)
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
    item: dict[str, Any], plan: dict[str, Any] | None = None
) -> list[str]:
    if plan is not None:
        projected = _project_runner_receipt(plan, item)
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
    _validate_runner_prefix(item, argv)
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


def _run_item(
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
    runner_cwd = _runner_cwd(item)
    ledger = _ledger_path(argv, cwd=runner_cwd)
    ledger_before = _read_ledger_locked(ledger)
    _assert_no_incomplete_taint_cleanup(ledger_before)
    wip_snapshot_before = _target_wip_snapshot(item)
    _taint_gate()
    proc = subprocess.run(
        argv,
        cwd=runner_cwd,
        env=_runner_env(item),
        check=False,
    )
    if proc.returncode != 0:
        return _recover_confirmed_taint(
            item,
            ledger=ledger,
            ledger_before=ledger_before,
            reached_before=reached_before,
            wip_snapshot_before=wip_snapshot_before,
            child_returncode=proc.returncode,
        )
    _taint_gate()
    reached = _checkpoint_reached(game)
    if reached >= target:
        _refresh_solver_audits()
    return {
        "game": game,
        "target_level": target,
        "reached": reached,
        "result": "solved" if reached >= target else "not_solved",
        "seed_mode": item["seed_mode"],
        "wip_mode": item["wip_mode"],
        "lineage_input_mode": item["lineage_input_mode"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--max-items", type=int, default=Policy.DEFAULT_MAX_RUNS)
    parser.add_argument("--calibration-only", action="store_true")
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
    items = [_project_runner_receipt(plan, item) for item in items]
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
