#!/usr/bin/env python3
"""Fail-closed release gate for a frozen ARC-AGI-3 canonical campaign.

This module deliberately does not solve games, launch proposers, or promote
artifacts.  It admits one already-frozen canonical tree and emits a
content-addressed receipt only when all 25 authoritative games and all 183
exact level boundaries have independently hash-bound evidence.

The release schema is stricter than the historical schema-1 promotion
manifests.  A manifest that merely says ``validated: true`` or
``taint_verdict: clean`` is not evidence.  Every boundary instead carries
separate, machine-readable taint, path-from-zero replay, source-from-zero
replay, and hash-audit records.  The gate cross-checks their subjects against
the exact checkpoint, winning source, host transcript, parent checkpoint, and
manifest bytes.

Trust model
-----------
The gate is intended to run on a trusted host after the attempt containers are
gone.  Content addressing detects later mutation but is not a digital
signature.  Authenticity therefore still depends on control of the host and
receipt store.  Receipt creation uses no-follow exclusive creation, rejects
hard links and special files, fsyncs the result, and never overwrites an
existing receipt.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import arc_agi3_contiguous_conformance as Conformance
import arc_agi3_source_schema as SourceSchema

EXPECTED_GAMES = 25
EXPECTED_LEVELS = 183
MAX_REPLAY_ACTIONS = 600
BOUNDARY_MANIFEST_SCHEMA = 2
RELEASE_RECEIPT_SCHEMA = 1
PARTIAL_RELEASE_RECEIPT_SCHEMA = 1

REQUIRED_SOURCE_FILES = frozenset({"legs.py", "players.py", "solve.py"})
AUDIT_PATHS = {
    "taint": "audits/taint.json",
    "action_protocol": "audits/action_protocol.json",
    "path_replay": "audits/path_replay.json",
    "source_replay": "audits/source_replay.json",
    "hash": "audits/hash_audit.json",
}

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GAME_RE = re.compile(r"^[a-z0-9]{4}$")
_IDENTITY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_REVISION_RE = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
_MAX_JSON_BYTES = 32 * 1024 * 1024
_HASH_CHUNK_BYTES = 1024 * 1024

_CHECKPOINT_FIELDS = {
    "game",
    "reached",
    "total_marginal_C",
    "records",
    "final_path",
    "validated",
}
_MANIFEST_FIELDS = {
    "schema",
    "game",
    "level",
    "frontier",
    "parent_manifest",
    "promoted_files_sha256",
    "winning_source_files",
    "transcripts",
    "audits",
}
_FRONTIER_FIELDS = {
    "parent_level",
    "target_level",
    "parent_checkpoint_sha256",
}
_PARENT_MANIFEST_FIELDS = {"path", "sha256"}
_HASHED_PATH_FIELDS = {"path", "sha256"}
_TAINT_FIELDS = {
    "schema",
    "kind",
    "game",
    "level",
    "scanner_sha256",
    "checked_files_sha256",
    "verdict",
    "findings",
}
_REPLAY_FIELDS = {
    "schema",
    "kind",
    "game",
    "target_level",
    "frontier_parent_level",
    "parent_checkpoint_sha256",
    "checkpoint_sha256",
    "winning_source_tree_sha256",
    "exact_path_sha256",
    "action_count",
    "observed_reached",
    "engine_sha256",
    "result",
}
_ACTION_PROTOCOL_FIELDS = {
    "schema",
    "kind",
    "game",
    "target_level",
    "checkpoint_sha256",
    "exact_path_sha256",
    "action_count",
    "runtime_enforcement",
    "source_protocol_latch",
    "path_protocol_latch",
    "engine_sha256",
    "result",
}
_HASH_AUDIT_FIELDS = {
    "schema",
    "kind",
    "game",
    "level",
    "hasher_sha256",
    "checked_files_sha256",
    "result",
}
_RELEASE_FIELDS = {
    "schema",
    "release_identity",
    "release_identity_sha256",
    "inventory",
    "inventory_sha256",
    "inventory_metadata_sha256",
    "canonical_game_count",
    "authoritative_level_count",
    "canonical_tree_sha256",
    "evidence",
    "evidence_sha256",
    "verifier",
    "control_contract",
}
_PARTIAL_RELEASE_FIELDS = _RELEASE_FIELDS | {
    "kind",
    "claimed_inventory",
    "claimed_inventory_sha256",
    "claimed_level_count",
    "unclaimed_boundaries",
    "complete",
}
_IDENTITY_FIELDS = {
    "campaign_id",
    "release_name",
    "source_revision",
    "created_at_utc",
}


class ReleaseGateError(RuntimeError):
    """The frozen canonical tree is not admissible for release."""


@dataclass(frozen=True)
class FileRecord:
    """One securely hashed, unaliased regular file."""

    relative_path: str
    sha256: str
    size: int
    mode: int


@dataclass(frozen=True)
class TreeSnapshot:
    """A deterministic snapshot of one regular directory tree."""

    root: Path
    files: Mapping[str, FileRecord]
    directories: Mapping[str, int]
    file_children: Mapping[str, frozenset[str]]
    directory_children: Mapping[str, frozenset[str]]
    sha256: str


@dataclass(frozen=True)
class ReleaseReceipt:
    """A verified immutable receipt already present in the receipt store."""

    path: Path
    sha256: str
    body: Mapping[str, Any]


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ReleaseGateError("value is not canonical JSON") from exc


def _json_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None


def _valid_action(action: object) -> bool:
    if _is_int(action):
        return 1 <= action <= 7 and action != 6
    return (
        isinstance(action, list)
        and len(action) == 3
        and action[0] == 6
        and all(_is_int(value) for value in action)
        and all(0 <= value < 64 for value in action[1:])
    )


def _safe_relative(value: object, *, prefix: str | None = None) -> bool:
    if not isinstance(value, str) or not value or "\\" in value:
        return False
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        return False
    return prefix is None or path.parts[0] == prefix


def _lstat_directory(path: Path, *, label: str) -> os.stat_result:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise ReleaseGateError(f"{label} is missing: {path}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise ReleaseGateError(
            f"{label} must be a non-symlink directory: {path}"
        )
    return metadata


def _directory_entries(path: Path, *, label: str) -> list[os.DirEntry[str]]:
    _lstat_directory(path, label=label)
    try:
        with os.scandir(path) as iterator:
            entries = sorted(iterator, key=lambda entry: entry.name)
    except OSError as exc:
        raise ReleaseGateError(f"cannot scan {label}: {path}") from exc
    return entries


def _entry_lstat(
    entry: os.DirEntry[str],
    *,
    label: str,
) -> os.stat_result:
    try:
        return entry.stat(follow_symlinks=False)
    except OSError as exc:
        raise ReleaseGateError(
            f"cannot inspect {label}: {entry.path}"
        ) from exc


def _file_record(path: Path, relative_path: str) -> FileRecord:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ReleaseGateError(
            f"cannot open evidence without following links: {path}"
        ) from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ReleaseGateError(f"non-regular file is forbidden: {path}")
        if before.st_nlink != 1:
            raise ReleaseGateError(f"hard-linked file is forbidden: {path}")
        digest = hashlib.sha256()
        size = 0
        while True:
            block = os.read(descriptor, _HASH_CHUNK_BYTES)
            if not block:
                break
            size += len(block)
            digest.update(block)
        after = os.fstat(descriptor)
        stable_fields = (
            "st_dev",
            "st_ino",
            "st_mode",
            "st_nlink",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
        )
        if any(getattr(before, field) != getattr(after, field)
               for field in stable_fields):
            raise ReleaseGateError(f"file changed while being hashed: {path}")
        if size != after.st_size:
            raise ReleaseGateError(f"short or unstable read: {path}")
        return FileRecord(
            relative_path=relative_path,
            sha256=digest.hexdigest(),
            size=size,
            mode=stat.S_IMODE(after.st_mode),
        )
    finally:
        os.close(descriptor)


def _snapshot_tree(root: Path) -> TreeSnapshot:
    _lstat_directory(root, label="canonical root")
    files: dict[str, FileRecord] = {}
    directories: dict[str, int] = {}

    def visit(directory: Path, relative: PurePosixPath | None) -> None:
        before = _lstat_directory(directory, label="canonical directory")
        if relative is not None:
            directories[relative.as_posix()] = stat.S_IMODE(before.st_mode)
        entries = _directory_entries(directory, label="canonical directory")
        for entry in entries:
            rel = (
                PurePosixPath(entry.name)
                if relative is None
                else relative / entry.name
            )
            metadata = _entry_lstat(entry, label="canonical entry")
            mode = metadata.st_mode
            if stat.S_ISLNK(mode):
                raise ReleaseGateError(
                    f"symlink is forbidden in canonical tree: {entry.path}"
                )
            if stat.S_ISDIR(mode):
                visit(Path(entry.path), rel)
            elif stat.S_ISREG(mode):
                record = _file_record(Path(entry.path), rel.as_posix())
                files[record.relative_path] = record
            else:
                raise ReleaseGateError(
                    f"non-regular canonical entry is forbidden: {entry.path}"
                )
        after = _lstat_directory(directory, label="canonical directory")
        for field in ("st_dev", "st_ino", "st_mode", "st_mtime_ns", "st_ctime_ns"):
            if getattr(before, field) != getattr(after, field):
                raise ReleaseGateError(
                    f"directory changed while being inspected: {directory}"
                )

    visit(root, None)
    digest_entries: list[dict[str, Any]] = [
        {"kind": "directory", "path": path, "mode": directories[path]}
        for path in sorted(directories)
    ]
    digest_entries.extend(
        {
            "kind": "file",
            "path": path,
            "mode": files[path].mode,
            "size": files[path].size,
            "sha256": files[path].sha256,
        }
        for path in sorted(files)
    )
    file_children_mutable: dict[str, set[str]] = {}
    directory_children_mutable: dict[str, set[str]] = {}
    for path in files:
        candidate = PurePosixPath(path)
        parent = (
            ""
            if candidate.parent == PurePosixPath(".")
            else candidate.parent.as_posix()
        )
        file_children_mutable.setdefault(parent, set()).add(candidate.name)
    for path in directories:
        candidate = PurePosixPath(path)
        parent = (
            ""
            if candidate.parent == PurePosixPath(".")
            else candidate.parent.as_posix()
        )
        directory_children_mutable.setdefault(parent, set()).add(candidate.name)
    return TreeSnapshot(
        root=root,
        files=files,
        directories=directories,
        file_children={
            path: frozenset(children)
            for path, children in file_children_mutable.items()
        },
        directory_children={
            path: frozenset(children)
            for path, children in directory_children_mutable.items()
        },
        sha256=_json_sha256(digest_entries),
    )


def _read_record_bytes(
    root: Path,
    record: FileRecord,
    *,
    max_bytes: int = _MAX_JSON_BYTES,
) -> bytes:
    if record.size > max_bytes:
        raise ReleaseGateError(
            f"evidence file exceeds {max_bytes} bytes: {record.relative_path}"
        )
    path = root / PurePosixPath(record.relative_path)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ReleaseGateError(
            f"cannot re-open snapshotted evidence: {record.relative_path}"
        ) from exc
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or metadata.st_size != record.size
            or stat.S_IMODE(metadata.st_mode) != record.mode
        ):
            raise ReleaseGateError(
                f"snapshotted evidence changed: {record.relative_path}"
            )
        data = bytearray()
        while True:
            block = os.read(descriptor, min(_HASH_CHUNK_BYTES, max_bytes + 1))
            if not block:
                break
            data.extend(block)
            if len(data) > max_bytes:
                raise ReleaseGateError(
                    f"evidence file exceeds {max_bytes} bytes: "
                    f"{record.relative_path}"
                )
        if hashlib.sha256(data).hexdigest() != record.sha256:
            raise ReleaseGateError(
                f"snapshotted evidence hash changed: {record.relative_path}"
            )
        return bytes(data)
    finally:
        os.close(descriptor)


def _read_json_record(
    snapshot: TreeSnapshot,
    relative_path: str,
) -> dict[str, Any]:
    record = snapshot.files.get(relative_path)
    if record is None:
        raise ReleaseGateError(f"required evidence is missing: {relative_path}")
    raw = _read_record_bytes(snapshot.root, record)
    try:
        value = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ReleaseGateError(f"invalid JSON evidence: {relative_path}") from exc
    if not isinstance(value, dict):
        raise ReleaseGateError(f"JSON evidence must be an object: {relative_path}")
    return value


def _direct_children(
    snapshot: TreeSnapshot,
    prefix: str,
) -> tuple[set[str], set[str]]:
    return (
        set(snapshot.file_children.get(prefix, frozenset())),
        set(snapshot.directory_children.get(prefix, frozenset())),
    )


def _require_exact_keys(
    value: Mapping[str, Any],
    expected: set[str],
    *,
    label: str,
) -> None:
    if set(value) != expected:
        raise ReleaseGateError(
            f"{label} schema mismatch; "
            f"missing={sorted(expected - set(value))}, "
            f"extra={sorted(set(value) - expected)}"
        )


def _validate_hash_map(value: object, *, label: str) -> dict[str, str]:
    if not isinstance(value, dict) or not value:
        raise ReleaseGateError(f"{label} must be a non-empty hash object")
    result: dict[str, str] = {}
    for path, digest in value.items():
        if not _safe_relative(path) or not _is_sha256(digest):
            raise ReleaseGateError(f"{label} has an invalid entry: {path!r}")
        result[path] = digest
    return result


def _validate_checkpoint(
    value: Mapping[str, Any],
    *,
    game: str,
    level: int,
) -> tuple[list[dict[str, Any]], list[Any]]:
    _require_exact_keys(value, _CHECKPOINT_FIELDS, label="checkpoint")
    if value["game"] != game:
        raise ReleaseGateError(
            f"checkpoint game mismatch at {game} L{level}: {value['game']!r}"
        )
    if not _is_int(value["reached"]) or value["reached"] != level:
        raise ReleaseGateError(
            f"checkpoint is not the exact {game} L{level} boundary"
        )
    if value["validated"] is not True:
        raise ReleaseGateError(f"checkpoint is not replay-marked at {game} L{level}")
    total = value["total_marginal_C"]
    records = value["records"]
    path = value["final_path"]
    if not _is_int(total) or total < 0:
        raise ReleaseGateError(f"invalid marginal total at {game} L{level}")
    if (
        not isinstance(path, list)
        or not path
        or len(path) > MAX_REPLAY_ACTIONS
        or not all(_valid_action(action) for action in path)
    ):
        raise ReleaseGateError(f"invalid exact replay path at {game} L{level}")
    if not isinstance(records, list) or len(records) != level:
        raise ReleaseGateError(f"checkpoint records do not cover 1..{level}")
    marginal_sum = 0
    normalized: list[dict[str, Any]] = []
    for expected_level, row in enumerate(records, start=1):
        if not isinstance(row, dict):
            raise ReleaseGateError(f"malformed checkpoint record at {game} L{level}")
        _require_exact_keys(
            row,
            {"level", "marginal_C", "reached"},
            label="checkpoint record",
        )
        if (
            not _is_int(row["level"])
            or row["level"] != expected_level
            or not _is_int(row["marginal_C"])
            or row["marginal_C"] < 0
            or row["reached"] is not True
        ):
            raise ReleaseGateError(
                f"invalid checkpoint record at {game} L{expected_level}"
            )
        marginal_sum += row["marginal_C"]
        normalized.append(dict(row))
    if marginal_sum != total:
        raise ReleaseGateError(f"marginal total mismatch at {game} L{level}")
    return normalized, list(path)


def _discover_inventory(
    environments_root: Path,
) -> tuple[dict[str, int], dict[str, str]]:
    entries = _directory_entries(
        environments_root, label="authoritative environment metadata root"
    )
    game_entries: dict[str, Path] = {}
    for entry in entries:
        metadata = _entry_lstat(
            entry, label="authoritative inventory entry"
        )
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or stat.S_ISLNK(metadata.st_mode)
            or _GAME_RE.fullmatch(entry.name) is None
        ):
            raise ReleaseGateError(
                f"unexpected authoritative inventory entry: {entry.path}"
            )
        game_entries[entry.name] = Path(entry.path)
    if len(game_entries) != EXPECTED_GAMES:
        raise ReleaseGateError(
            f"authoritative inventory must have exactly {EXPECTED_GAMES} games; "
            f"found {len(game_entries)}"
        )

    inventory: dict[str, int] = {}
    metadata_hashes: dict[str, str] = {}
    for game, game_root in sorted(game_entries.items()):
        versions = _directory_entries(game_root, label=f"{game} metadata root")
        version_dirs: list[Path] = []
        for entry in versions:
            metadata = _entry_lstat(entry, label=f"{game} metadata version")
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
                raise ReleaseGateError(
                    f"unexpected file in {game} metadata root: {entry.path}"
                )
            version_dirs.append(Path(entry.path))
        if len(version_dirs) != 1:
            raise ReleaseGateError(
                f"{game} must have exactly one authoritative metadata version; "
                f"found {len(version_dirs)}"
            )
        metadata_path = version_dirs[0] / "metadata.json"
        relative = metadata_path.relative_to(environments_root).as_posix()
        record = _file_record(metadata_path, relative)
        raw = _read_record_bytes(environments_root, record)
        try:
            payload = json.loads(raw)
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise ReleaseGateError(
                f"invalid authoritative metadata: {relative}"
            ) from exc
        if not isinstance(payload, dict):
            raise ReleaseGateError(
                f"authoritative metadata must be an object: {relative}"
            )
        actions = payload.get("baseline_actions")
        if not isinstance(actions, list) or not actions:
            raise ReleaseGateError(
                f"missing authoritative baseline_actions: {relative}"
            )
        inventory[game] = len(actions)
        metadata_hashes[relative] = record.sha256

    total = sum(inventory.values())
    if total != EXPECTED_LEVELS:
        raise ReleaseGateError(
            f"authoritative inventory must total {EXPECTED_LEVELS} levels; "
            f"found {total}"
        )
    return inventory, metadata_hashes


def _validate_release_identity(identity: object) -> dict[str, str]:
    if not isinstance(identity, dict):
        raise ReleaseGateError("release identity must be an object")
    _require_exact_keys(identity, _IDENTITY_FIELDS, label="release identity")
    campaign_id = identity["campaign_id"]
    release_name = identity["release_name"]
    revision = identity["source_revision"]
    created_at = identity["created_at_utc"]
    if (
        not isinstance(campaign_id, str)
        or _IDENTITY_RE.fullmatch(campaign_id) is None
        or not isinstance(release_name, str)
        or _IDENTITY_RE.fullmatch(release_name) is None
        or not isinstance(revision, str)
        or _REVISION_RE.fullmatch(revision) is None
        or not isinstance(created_at, str)
    ):
        raise ReleaseGateError("release identity contains an invalid field")
    try:
        instant = dt.datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ReleaseGateError(
            "created_at_utc must be an ISO-8601 UTC timestamp"
        ) from exc
    if instant.tzinfo is None or instant.utcoffset() != dt.timedelta(0):
        raise ReleaseGateError("created_at_utc must explicitly use UTC")
    return {
        "campaign_id": campaign_id,
        "release_name": release_name,
        "source_revision": revision,
        "created_at_utc": created_at,
    }


def _default_control_files() -> dict[str, Path]:
    repository = Path(__file__).resolve().parents[2]
    return {
        relative: repository / relative
        for relative in Conformance.CONTROL_CONTRACT_FILES
    }


def _control_contract_snapshot(
    control_files: Mapping[str, Path],
) -> dict[str, Any]:
    defaults = _default_control_files()
    if (
        set(control_files) == set(Conformance.CONTROL_CONTRACT_FILES)
        and all(
            Path(control_files[relative])
            == defaults[relative]
            for relative in Conformance.CONTROL_CONTRACT_FILES
        )
    ):
        try:
            return Conformance.control_contract_snapshot(
                repository=Path(__file__).resolve().parents[2]
            )
        except Conformance.ConformanceError as exc:
            raise ReleaseGateError(str(exc)) from exc
    expected_order = (
        Conformance.CONTROL_CONTRACT_FILES
        if set(control_files) == set(Conformance.CONTROL_CONTRACT_FILES)
        else None
    )
    try:
        return Conformance.control_contract_snapshot(
            control_files,
            expected_order=expected_order,
        )
    except Conformance.ConformanceError as exc:
        raise ReleaseGateError(str(exc)) from exc


def _verifier_snapshot() -> dict[str, Any]:
    repository = Path(__file__).resolve().parents[2]
    return _control_contract_snapshot({
        "arc/crack_lab/arc_agi3_release_gate.py": (
            repository / "arc/crack_lab/arc_agi3_release_gate.py"
        ),
        "arc/crack_lab/test_arc_agi3_release_gate.py": (
            repository / "arc/crack_lab/test_arc_agi3_release_gate.py"
        ),
    })


def _expect_hashed_path(
    value: object,
    *,
    expected_path: str,
    snapshot: TreeSnapshot,
    game_prefix: str,
    label: str,
) -> str:
    if not isinstance(value, dict):
        raise ReleaseGateError(f"{label} must be a hash-bound path object")
    _require_exact_keys(value, _HASHED_PATH_FIELDS, label=label)
    if value["path"] != expected_path or not _is_sha256(value["sha256"]):
        raise ReleaseGateError(f"{label} has an invalid path or hash")
    full_path = f"{game_prefix}/{expected_path}"
    record = snapshot.files.get(full_path)
    if record is None or record.sha256 != value["sha256"]:
        raise ReleaseGateError(f"{label} does not match frozen bytes")
    return record.sha256


def _validate_taint_audit(
    value: Mapping[str, Any],
    *,
    game: str,
    level: int,
    expected_checked: Mapping[str, str],
    allowed_tool_hashes: frozenset[str],
) -> None:
    _require_exact_keys(value, _TAINT_FIELDS, label="taint audit")
    if (
        value["schema"] != 1
        or isinstance(value["schema"], bool)
        or value["kind"] != "taint_audit"
        or value["game"] != game
        or not _is_int(value["level"])
        or value["level"] != level
        or value["scanner_sha256"] not in allowed_tool_hashes
        or value["verdict"] != "PASS"
        or value["findings"] != []
    ):
        raise ReleaseGateError(f"taint audit did not pass at {game} L{level}")
    checked = _validate_hash_map(
        value["checked_files_sha256"], label="taint checked-files"
    )
    if checked != dict(expected_checked):
        raise ReleaseGateError(
            f"taint audit subjects are incomplete or stale at {game} L{level}"
        )


def _validate_replay_audit(
    value: Mapping[str, Any],
    *,
    kind: str,
    game: str,
    level: int,
    parent_checkpoint_sha256: str | None,
    checkpoint_sha256: str,
    winning_source_tree_sha256: str,
    exact_path_sha256: str,
    action_count: int,
    allowed_tool_hashes: frozenset[str],
) -> None:
    _require_exact_keys(value, _REPLAY_FIELDS, label=f"{kind} replay audit")
    if (
        value["schema"] != 1
        or isinstance(value["schema"], bool)
        or value["kind"] != kind
        or value["game"] != game
        or not _is_int(value["target_level"])
        or value["target_level"] != level
        or not _is_int(value["frontier_parent_level"])
        or value["frontier_parent_level"] != level - 1
        or value["parent_checkpoint_sha256"] != parent_checkpoint_sha256
        or value["checkpoint_sha256"] != checkpoint_sha256
        or value["winning_source_tree_sha256"]
        != winning_source_tree_sha256
        or value["exact_path_sha256"] != exact_path_sha256
        or not _is_int(value["action_count"])
        or value["action_count"] != action_count
        or not _is_int(value["observed_reached"])
        or value["observed_reached"] != level
        or value["engine_sha256"] not in allowed_tool_hashes
        or value["result"] != "PASS"
    ):
        raise ReleaseGateError(
            f"{kind} replay is stale, inexact, or failed at {game} L{level}"
        )


def _validate_action_protocol_audit(
    value: Mapping[str, Any],
    *,
    game: str,
    level: int,
    checkpoint_sha256: str,
    exact_path_sha256: str,
    action_count: int,
    allowed_tool_hashes: frozenset[str],
) -> None:
    """Require the fresh source and path replays to share a fail-closed latch.

    This receipt is about the certification executions, not the historical
    acquisition turn.  ``gkm_arena`` shares one violation latch between the
    root environment and every clone, so catching an invalid-action exception
    inside retained solver code cannot turn the replay green.
    """
    _require_exact_keys(
        value,
        _ACTION_PROTOCOL_FIELDS,
        label="action-protocol audit",
    )
    if (
        value["schema"] != 1
        or isinstance(value["schema"], bool)
        or value["kind"] != "action_protocol_audit"
        or value["game"] != game
        or not _is_int(value["target_level"])
        or value["target_level"] != level
        or value["checkpoint_sha256"] != checkpoint_sha256
        or value["exact_path_sha256"] != exact_path_sha256
        or not _is_int(value["action_count"])
        or value["action_count"] != action_count
        or value["runtime_enforcement"]
        != "shared_violation_latch_across_root_and_clones"
        or value["source_protocol_latch"] != "PASS"
        or value["path_protocol_latch"] != "PASS"
        or value["engine_sha256"] not in allowed_tool_hashes
        or value["result"] != "PASS"
    ):
        raise ReleaseGateError(
            f"action-protocol audit is stale or failed at {game} L{level}"
        )


def _validate_hash_audit(
    value: Mapping[str, Any],
    *,
    game: str,
    level: int,
    expected_checked: Mapping[str, str],
    allowed_tool_hashes: frozenset[str],
) -> None:
    _require_exact_keys(value, _HASH_AUDIT_FIELDS, label="hash audit")
    if (
        value["schema"] != 1
        or isinstance(value["schema"], bool)
        or value["kind"] != "hash_audit"
        or value["game"] != game
        or not _is_int(value["level"])
        or value["level"] != level
        or value["hasher_sha256"] not in allowed_tool_hashes
        or value["result"] != "PASS"
    ):
        raise ReleaseGateError(f"hash audit did not pass at {game} L{level}")
    checked = _validate_hash_map(
        value["checked_files_sha256"], label="hash-audit checked-files"
    )
    if checked != dict(expected_checked):
        raise ReleaseGateError(
            f"hash audit subjects are incomplete or stale at {game} L{level}"
        )


def _boundary_tree_sha256(
    snapshot: TreeSnapshot,
    boundary_prefix: str,
) -> str:
    prefix = f"{boundary_prefix}/"
    entries: list[dict[str, Any]] = []
    for path, mode in sorted(snapshot.directories.items()):
        if path.startswith(prefix):
            entries.append({"kind": "directory", "path": path[len(prefix):],
                            "mode": mode})
    for path, record in sorted(snapshot.files.items()):
        if path.startswith(prefix):
            entries.append({
                "kind": "file",
                "path": path[len(prefix):],
                "mode": record.mode,
                "size": record.size,
                "sha256": record.sha256,
            })
    return _json_sha256(entries)


def _validate_boundary(
    snapshot: TreeSnapshot,
    *,
    game: str,
    level: int,
    previous_checkpoint_sha256: str | None,
    previous_manifest_sha256: str | None,
    previous_records: list[dict[str, Any]] | None,
    allowed_tool_hashes: frozenset[str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    game_prefix = f"{game}_legs"
    evidence_rel = f"promotion_evidence/level_{level:02d}"
    boundary_prefix = f"{game_prefix}/{evidence_rel}"
    direct_files, direct_dirs = _direct_children(snapshot, boundary_prefix)
    if direct_files != {"manifest.json"} or direct_dirs != {
        "files", "transcripts", "audits"
    }:
        raise ReleaseGateError(
            f"{game} L{level} evidence layout is not exact"
        )
    for child in ("files", "transcripts", "audits"):
        nested_files, nested_dirs = _direct_children(
            snapshot, f"{boundary_prefix}/{child}"
        )
        if nested_dirs:
            raise ReleaseGateError(
                f"nested evidence directories are forbidden at {game} L{level}"
            )
        if child == "transcripts" and not nested_files:
            raise ReleaseGateError(
                f"host transcript evidence is missing at {game} L{level}"
            )

    manifest_rel = f"{boundary_prefix}/manifest.json"
    manifest = _read_json_record(snapshot, manifest_rel)
    _require_exact_keys(manifest, _MANIFEST_FIELDS, label="boundary manifest")
    if (
        manifest["schema"] != BOUNDARY_MANIFEST_SCHEMA
        or isinstance(manifest["schema"], bool)
        or manifest["game"] != game
        or not _is_int(manifest["level"])
        or manifest["level"] != level
    ):
        raise ReleaseGateError(
            f"boundary manifest identity mismatch at {game} L{level}"
        )

    frontier = manifest["frontier"]
    if not isinstance(frontier, dict):
        raise ReleaseGateError(f"missing frontier binding at {game} L{level}")
    _require_exact_keys(frontier, _FRONTIER_FIELDS, label="frontier binding")
    if (
        not _is_int(frontier["parent_level"])
        or frontier["parent_level"] != level - 1
        or not _is_int(frontier["target_level"])
        or frontier["target_level"] != level
        or frontier["parent_checkpoint_sha256"]
        != previous_checkpoint_sha256
    ):
        raise ReleaseGateError(
            f"parent/frontier continuity failed at {game} L{level}"
        )

    parent_manifest = manifest["parent_manifest"]
    if level == 1:
        if parent_manifest is not None or previous_manifest_sha256 is not None:
            raise ReleaseGateError(f"L1 has a synthetic parent manifest: {game}")
    else:
        if not isinstance(parent_manifest, dict):
            raise ReleaseGateError(
                f"parent manifest is missing at {game} L{level}"
            )
        _require_exact_keys(
            parent_manifest,
            _PARENT_MANIFEST_FIELDS,
            label="parent manifest binding",
        )
        expected_parent_path = (
            f"promotion_evidence/level_{level - 1:02d}/manifest.json"
        )
        if (
            parent_manifest["path"] != expected_parent_path
            or parent_manifest["sha256"] != previous_manifest_sha256
        ):
            raise ReleaseGateError(
                f"manifest chain is stale or discontinuous at {game} L{level}"
            )

    promoted = _validate_hash_map(
        manifest["promoted_files_sha256"], label="promoted files"
    )
    if any(PurePosixPath(path).parent != PurePosixPath(".") for path in promoted):
        raise ReleaseGateError(
            f"promoted file paths must be direct basenames at {game} L{level}"
        )
    actual_promoted, promoted_dirs = _direct_children(
        snapshot, f"{boundary_prefix}/files"
    )
    if promoted_dirs or set(promoted) != actual_promoted:
        raise ReleaseGateError(
            f"promoted-file inventory is not exact at {game} L{level}"
        )
    for name, expected_hash in promoted.items():
        record = snapshot.files[f"{boundary_prefix}/files/{name}"]
        if record.sha256 != expected_hash:
            raise ReleaseGateError(
                f"promoted file hash mismatch at {game} L{level}: {name}"
            )
    if "checkpoint.json" not in promoted:
        raise ReleaseGateError(
            f"exact checkpoint evidence is missing at {game} L{level}"
        )

    winning_sources = manifest["winning_source_files"]
    if (
        not isinstance(winning_sources, list)
        or winning_sources != sorted(winning_sources)
        or len(set(winning_sources)) != len(winning_sources)
        or not REQUIRED_SOURCE_FILES.issubset(winning_sources)
        or any(name not in promoted for name in winning_sources)
    ):
        raise ReleaseGateError(
            f"winning-source snapshot is incomplete at {game} L{level}"
        )
    try:
        SourceSchema.validate_source_payloads({
            name: _read_record_bytes(
                snapshot.root,
                snapshot.files[f"{boundary_prefix}/files/{name}"],
            )
            for name in winning_sources
        })
    except SourceSchema.SourceSchemaError as exc:
        raise ReleaseGateError(
            f"winning-source snapshot violates the shared schema at "
            f"{game} L{level}"
        ) from exc
    winning_hashes = {name: promoted[name] for name in winning_sources}
    winning_source_tree_sha256 = _json_sha256(winning_hashes)

    checkpoint_relative = f"{boundary_prefix}/files/checkpoint.json"
    checkpoint = _read_json_record(snapshot, checkpoint_relative)
    records, exact_path = _validate_checkpoint(
        checkpoint, game=game, level=level
    )
    if previous_records is not None and records[:-1] != previous_records:
        raise ReleaseGateError(
            f"checkpoint record lineage forked at {game} L{level}"
        )
    checkpoint_sha256 = snapshot.files[checkpoint_relative].sha256
    exact_path_sha256 = _json_sha256(exact_path)

    transcript_entries = manifest["transcripts"]
    if not isinstance(transcript_entries, list) or not transcript_entries:
        raise ReleaseGateError(
            f"transcript manifest is missing at {game} L{level}"
        )
    transcript_hashes: dict[str, str] = {}
    for item in transcript_entries:
        if not isinstance(item, dict):
            raise ReleaseGateError(
                f"malformed transcript binding at {game} L{level}"
            )
        _require_exact_keys(item, _HASHED_PATH_FIELDS, label="transcript binding")
        path = item["path"]
        if (
            not _safe_relative(path, prefix="transcripts")
            or len(PurePosixPath(path).parts) != 2
            or not _is_sha256(item["sha256"])
            or path in transcript_hashes
        ):
            raise ReleaseGateError(
                f"invalid transcript binding at {game} L{level}"
            )
        record = snapshot.files.get(f"{boundary_prefix}/{path}")
        if (
            record is None
            or record.size == 0
            or record.sha256 != item["sha256"]
        ):
            raise ReleaseGateError(
                f"transcript evidence is missing or stale at {game} L{level}"
            )
        transcript_hashes[path] = item["sha256"]
    actual_transcripts, transcript_dirs = _direct_children(
        snapshot, f"{boundary_prefix}/transcripts"
    )
    if (
        transcript_dirs
        or set(transcript_hashes)
        != {f"transcripts/{name}" for name in actual_transcripts}
    ):
        raise ReleaseGateError(
            f"transcript inventory is not exact at {game} L{level}"
        )

    audits = manifest["audits"]
    if not isinstance(audits, dict) or set(audits) != set(AUDIT_PATHS):
        raise ReleaseGateError(f"audit manifest is incomplete at {game} L{level}")
    audit_hashes: dict[str, str] = {}
    for name, expected_path in AUDIT_PATHS.items():
        audit_hashes[name] = _expect_hashed_path(
            audits[name],
            expected_path=expected_path,
            snapshot=snapshot,
            game_prefix=boundary_prefix,
            label=f"{name} audit binding",
        )
    actual_audits, audit_dirs = _direct_children(
        snapshot, f"{boundary_prefix}/audits"
    )
    if (
        audit_dirs
        or actual_audits
        != {PurePosixPath(path).name for path in AUDIT_PATHS.values()}
    ):
        raise ReleaseGateError(
            f"audit file inventory is not exact at {game} L{level}"
        )

    primary_checked = {
        **{f"files/{name}": digest for name, digest in promoted.items()},
        **transcript_hashes,
    }
    taint = _read_json_record(
        snapshot, f"{boundary_prefix}/{AUDIT_PATHS['taint']}"
    )
    _validate_taint_audit(
        taint,
        game=game,
        level=level,
        expected_checked=primary_checked,
        allowed_tool_hashes=allowed_tool_hashes,
    )

    action_protocol = _read_json_record(
        snapshot,
        f"{boundary_prefix}/{AUDIT_PATHS['action_protocol']}",
    )
    _validate_action_protocol_audit(
        action_protocol,
        game=game,
        level=level,
        checkpoint_sha256=checkpoint_sha256,
        exact_path_sha256=exact_path_sha256,
        action_count=len(exact_path),
        allowed_tool_hashes=allowed_tool_hashes,
    )

    for audit_name, replay_kind in (
        ("path_replay", "path_replay"),
        ("source_replay", "source_replay"),
    ):
        replay = _read_json_record(
            snapshot, f"{boundary_prefix}/{AUDIT_PATHS[audit_name]}"
        )
        _validate_replay_audit(
            replay,
            kind=replay_kind,
            game=game,
            level=level,
            parent_checkpoint_sha256=previous_checkpoint_sha256,
            checkpoint_sha256=checkpoint_sha256,
            winning_source_tree_sha256=winning_source_tree_sha256,
            exact_path_sha256=exact_path_sha256,
            action_count=len(exact_path),
            allowed_tool_hashes=allowed_tool_hashes,
        )

    hash_checked = {
        **primary_checked,
        AUDIT_PATHS["taint"]: audit_hashes["taint"],
        AUDIT_PATHS["action_protocol"]:
            audit_hashes["action_protocol"],
        AUDIT_PATHS["path_replay"]: audit_hashes["path_replay"],
        AUDIT_PATHS["source_replay"]: audit_hashes["source_replay"],
    }
    hash_audit = _read_json_record(
        snapshot, f"{boundary_prefix}/{AUDIT_PATHS['hash']}"
    )
    _validate_hash_audit(
        hash_audit,
        game=game,
        level=level,
        expected_checked=hash_checked,
        allowed_tool_hashes=allowed_tool_hashes,
    )

    manifest_sha256 = snapshot.files[manifest_rel].sha256
    summary = {
        "level": level,
        "frontier_parent_level": level - 1,
        "parent_checkpoint_sha256": previous_checkpoint_sha256,
        "parent_manifest_sha256": previous_manifest_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "exact_path_sha256": exact_path_sha256,
        "action_count": len(exact_path),
        "winning_source_files_sha256": winning_hashes,
        "winning_source_tree_sha256": winning_source_tree_sha256,
        "transcripts_sha256": transcript_hashes,
        "audits_sha256": audit_hashes,
        "manifest_sha256": manifest_sha256,
        "boundary_tree_sha256": _boundary_tree_sha256(
            snapshot, boundary_prefix
        ),
    }
    return summary, records


def _validate_canonical(
    canonical_root: Path,
    inventory: Mapping[str, int],
    *,
    allowed_tool_hashes: frozenset[str],
) -> tuple[TreeSnapshot, dict[str, list[dict[str, Any]]]]:
    root_entries = _directory_entries(canonical_root, label="canonical root")
    expected_roots = {f"{game}_legs" for game in inventory}
    found_roots: set[str] = set()
    for entry in root_entries:
        metadata = _entry_lstat(entry, label="canonical game entry")
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise ReleaseGateError(
                f"canonical root contains a non-game entry: {entry.path}"
            )
        found_roots.add(entry.name)
    if found_roots != expected_roots:
        raise ReleaseGateError(
            "canonical root does not exactly match the authoritative games; "
            f"missing={sorted(expected_roots - found_roots)}, "
            f"extra={sorted(found_roots - expected_roots)}"
        )

    snapshot = _snapshot_tree(canonical_root)
    evidence: dict[str, list[dict[str, Any]]] = {}
    for game, target in sorted(inventory.items()):
        game_prefix = f"{game}_legs"
        top_files, top_dirs = _direct_children(snapshot, game_prefix)
        if top_dirs != {"promotion_evidence"}:
            raise ReleaseGateError(
                f"frozen {game} tree has mutable or unknown top-level directories"
            )
        if not {"checkpoint.json", *REQUIRED_SOURCE_FILES}.issubset(top_files):
            raise ReleaseGateError(
                f"canonical solver/checkpoint is incomplete for {game}"
            )
        evidence_files, evidence_dirs = _direct_children(
            snapshot, f"{game_prefix}/promotion_evidence"
        )
        expected_levels = {
            f"level_{level:02d}" for level in range(1, target + 1)
        }
        if evidence_files or evidence_dirs != expected_levels:
            raise ReleaseGateError(
                f"{game} evidence levels are not exactly 1..{target}; "
                f"missing={sorted(expected_levels - evidence_dirs)}, "
                f"extra={sorted(evidence_dirs - expected_levels)}"
            )

        previous_checkpoint: str | None = None
        previous_manifest: str | None = None
        previous_records: list[dict[str, Any]] | None = None
        boundaries: list[dict[str, Any]] = []
        for level in range(1, target + 1):
            summary, records = _validate_boundary(
                snapshot,
                game=game,
                level=level,
                previous_checkpoint_sha256=previous_checkpoint,
                previous_manifest_sha256=previous_manifest,
                previous_records=previous_records,
                allowed_tool_hashes=allowed_tool_hashes,
            )
            boundaries.append(summary)
            previous_checkpoint = summary["checkpoint_sha256"]
            previous_manifest = summary["manifest_sha256"]
            previous_records = records

        final = boundaries[-1]
        top_checkpoint = snapshot.files[f"{game_prefix}/checkpoint.json"]
        if top_checkpoint.sha256 != final["checkpoint_sha256"]:
            raise ReleaseGateError(
                f"canonical checkpoint is not the exact final boundary for {game}"
            )
        top_checkpoint_value = _read_json_record(
            snapshot, f"{game_prefix}/checkpoint.json"
        )
        _validate_checkpoint(top_checkpoint_value, game=game, level=target)

        top_python = sorted(
            name for name in top_files if PurePosixPath(name).suffix == ".py"
        )
        final_sources = sorted(final["winning_source_files_sha256"])
        if top_python != final_sources:
            raise ReleaseGateError(
                f"canonical source set differs from final winning source for {game}"
            )
        for source_name, source_hash in (
            final["winning_source_files_sha256"].items()
        ):
            if snapshot.files[f"{game_prefix}/{source_name}"].sha256 != source_hash:
                raise ReleaseGateError(
                    f"canonical source is stale for {game}: {source_name}"
                )
        evidence[game] = boundaries
    return snapshot, evidence


def _claimed_inventory(
    canonical_root: Path,
    authoritative_inventory: Mapping[str, int],
) -> dict[str, int]:
    """Derive the exact claimed frontier without trusting a caller-supplied map."""
    root = Path(canonical_root)
    entries = _directory_entries(root, label="canonical root")
    expected_roots = {
        f"{game}_legs" for game in authoritative_inventory
    }
    found_roots: set[str] = set()
    for entry in entries:
        metadata = _entry_lstat(entry, label="canonical game entry")
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise ReleaseGateError(
                f"canonical root contains a non-game entry: {entry.path}"
            )
        found_roots.add(entry.name)
    if found_roots != expected_roots:
        raise ReleaseGateError(
            "canonical root does not exactly match the authoritative games; "
            f"missing={sorted(expected_roots - found_roots)}, "
            f"extra={sorted(found_roots - expected_roots)}"
        )

    claimed: dict[str, int] = {}
    for game, authoritative_target in sorted(
        authoritative_inventory.items()
    ):
        checkpoint_path = root / f"{game}_legs" / "checkpoint.json"
        record = _file_record(
            checkpoint_path,
            f"{game}_legs/checkpoint.json",
        )
        raw = _read_record_bytes(root, record)
        try:
            checkpoint = json.loads(raw)
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise ReleaseGateError(
                f"invalid canonical checkpoint for {game}"
            ) from exc
        if not isinstance(checkpoint, dict):
            raise ReleaseGateError(
                f"canonical checkpoint must be an object for {game}"
            )
        reached = checkpoint.get("reached")
        if (
            not _is_int(reached)
            or reached < 1
            or reached > authoritative_target
        ):
            raise ReleaseGateError(
                f"claimed frontier for {game} must be within "
                f"1..{authoritative_target}"
            )
        _validate_checkpoint(checkpoint, game=game, level=reached)
        claimed[game] = reached
    return claimed


def _unclaimed_boundaries(
    authoritative_inventory: Mapping[str, int],
    claimed_inventory: Mapping[str, int],
) -> list[dict[str, Any]]:
    if set(authoritative_inventory) != set(claimed_inventory):
        raise ReleaseGateError(
            "claimed inventory does not cover the authoritative games"
        )
    missing: list[dict[str, Any]] = []
    for game, target in sorted(authoritative_inventory.items()):
        reached = claimed_inventory[game]
        if not _is_int(reached) or not 0 <= reached <= target:
            raise ReleaseGateError(
                f"claimed frontier is outside authoritative inventory: {game}"
            )
        missing.extend(
            {"game": game, "level": level}
            for level in range(reached + 1, target + 1)
        )
    return missing


def _diagnostic_issue(
    code: str,
    path: str,
    detail: str,
) -> dict[str, str]:
    return {"code": code, "path": path, "detail": detail}


def _sort_issues(issues: list[dict[str, str]]) -> list[dict[str, str]]:
    unique = {
        (issue["code"], issue["path"], issue["detail"]): issue
        for issue in issues
    }
    return [
        unique[key]
        for key in sorted(unique)
    ]


def _diagnostic_json_file(
    root: Path,
    relative_path: str,
    *,
    missing_code: str,
) -> tuple[dict[str, Any] | None, FileRecord | None, list[dict[str, str]]]:
    path = root / PurePosixPath(relative_path)
    issues: list[dict[str, str]] = []
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        issues.append(_diagnostic_issue(
            missing_code, relative_path, "required file is missing"
        ))
        return None, None, issues
    except OSError:
        issues.append(_diagnostic_issue(
            "unreadable_evidence", relative_path, "file metadata is unreadable"
        ))
        return None, None, issues
    if stat.S_ISLNK(metadata.st_mode):
        issues.append(_diagnostic_issue(
            "symlink_evidence", relative_path, "symlink evidence is forbidden"
        ))
        return None, None, issues
    if not stat.S_ISREG(metadata.st_mode):
        issues.append(_diagnostic_issue(
            "nonregular_evidence",
            relative_path,
            "evidence must be a regular file",
        ))
        return None, None, issues
    if metadata.st_nlink != 1:
        issues.append(_diagnostic_issue(
            "hardlinked_evidence",
            relative_path,
            "evidence must have exactly one hard link",
        ))
        return None, None, issues
    try:
        record = _file_record(path, relative_path)
        raw = _read_record_bytes(root, record)
        value = json.loads(raw)
    except (ReleaseGateError, UnicodeError, json.JSONDecodeError):
        issues.append(_diagnostic_issue(
            "invalid_json_evidence",
            relative_path,
            "evidence is not stable valid JSON",
        ))
        return None, None, issues
    if not isinstance(value, dict):
        issues.append(_diagnostic_issue(
            "invalid_json_evidence",
            relative_path,
            "evidence JSON must be an object",
        ))
        return None, record, issues
    return value, record, issues


def _diagnostic_regular_file(
    root: Path,
    relative_path: str,
    *,
    missing_code: str,
) -> tuple[FileRecord | None, list[dict[str, str]]]:
    path = root / PurePosixPath(relative_path)
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return None, [_diagnostic_issue(
            missing_code, relative_path, "required file is missing"
        )]
    except OSError:
        return None, [_diagnostic_issue(
            "unreadable_evidence", relative_path, "file metadata is unreadable"
        )]
    if stat.S_ISLNK(metadata.st_mode):
        return None, [_diagnostic_issue(
            "symlink_evidence", relative_path, "symlink evidence is forbidden"
        )]
    if not stat.S_ISREG(metadata.st_mode):
        return None, [_diagnostic_issue(
            "nonregular_evidence",
            relative_path,
            "evidence must be a regular file",
        )]
    if metadata.st_nlink != 1:
        return None, [_diagnostic_issue(
            "hardlinked_evidence",
            relative_path,
            "evidence must have exactly one hard link",
        )]
    try:
        return _file_record(path, relative_path), []
    except ReleaseGateError:
        return None, [_diagnostic_issue(
            "unstable_evidence", relative_path, "file changed while being hashed"
        )]


def _diagnose_level(
    canonical_root: Path,
    *,
    game: str,
    level: int,
    previous_checkpoint_sha256: str | None,
    previous_manifest_sha256: str | None,
) -> tuple[dict[str, Any], str | None, str | None]:
    game_prefix = f"{game}_legs"
    boundary_relative = (
        f"{game_prefix}/promotion_evidence/level_{level:02d}"
    )
    boundary = canonical_root / PurePosixPath(boundary_relative)
    issues: list[dict[str, str]] = []
    try:
        boundary_metadata = boundary.lstat()
    except FileNotFoundError:
        issues.append(_diagnostic_issue(
            "missing_boundary",
            boundary_relative,
            f"exact level {level} evidence directory is missing",
        ))
        return {
            "status": "missing",
            "issues": issues,
        }, None, None
    except OSError:
        issues.append(_diagnostic_issue(
            "unreadable_boundary",
            boundary_relative,
            "boundary directory metadata is unreadable",
        ))
        return {"status": "invalid", "issues": issues}, None, None
    if stat.S_ISLNK(boundary_metadata.st_mode):
        issues.append(_diagnostic_issue(
            "symlink_boundary",
            boundary_relative,
            "boundary directory must not be a symlink",
        ))
        return {"status": "invalid", "issues": issues}, None, None
    if not stat.S_ISDIR(boundary_metadata.st_mode):
        issues.append(_diagnostic_issue(
            "nonregular_boundary",
            boundary_relative,
            "boundary evidence must be a directory",
        ))
        return {"status": "invalid", "issues": issues}, None, None
    try:
        boundary_entries = _directory_entries(
            boundary, label=f"{game} L{level} boundary"
        )
    except ReleaseGateError:
        boundary_entries = []
        issues.append(_diagnostic_issue(
            "unreadable_boundary",
            boundary_relative,
            "boundary entries cannot be enumerated",
        ))
    allowed_boundary_entries = {"manifest.json", "files", "transcripts", "audits"}
    for entry in boundary_entries:
        if entry.name not in allowed_boundary_entries:
            issues.append(_diagnostic_issue(
                "unbound_boundary_entry",
                f"{boundary_relative}/{entry.name}",
                "entry is outside the schema-v2 evidence manifest",
            ))

    manifest_relative = f"{boundary_relative}/manifest.json"
    manifest, manifest_record, manifest_issues = _diagnostic_json_file(
        canonical_root,
        manifest_relative,
        missing_code="missing_manifest",
    )
    issues.extend(manifest_issues)
    manifest_schema: int | None = None
    legacy = False
    if manifest is not None:
        raw_schema = manifest.get("schema")
        if _is_int(raw_schema):
            manifest_schema = raw_schema
        if manifest_schema != BOUNDARY_MANIFEST_SCHEMA:
            legacy = manifest_schema == 1
            issues.append(_diagnostic_issue(
                (
                    "legacy_manifest_schema"
                    if legacy
                    else "unsupported_manifest_schema"
                ),
                manifest_relative,
                (
                    f"found schema {manifest_schema!r}; "
                    f"release requires schema {BOUNDARY_MANIFEST_SCHEMA}"
                ),
            ))
        if "validated" in manifest or "taint_verdict" in manifest:
            issues.append(_diagnostic_issue(
                "boolean_only_gate_claim",
                manifest_relative,
                "manifest gate claims do not replace machine audit receipts",
            ))
        if manifest.get("game") != game or manifest.get("level") != level:
            issues.append(_diagnostic_issue(
                "manifest_identity_mismatch",
                manifest_relative,
                f"manifest must identify {game} level {level}",
            ))
        if manifest_schema == BOUNDARY_MANIFEST_SCHEMA:
            frontier = manifest.get("frontier")
            if (
                not isinstance(frontier, dict)
                or frontier.get("parent_level") != level - 1
                or frontier.get("target_level") != level
                or frontier.get("parent_checkpoint_sha256")
                != previous_checkpoint_sha256
            ):
                issues.append(_diagnostic_issue(
                    "parent_frontier_discontinuity",
                    manifest_relative,
                    "frontier does not bind the exact preceding checkpoint",
                ))
            parent = manifest.get("parent_manifest")
            expected_parent_path = (
                None
                if level == 1
                else (
                    "promotion_evidence/"
                    f"level_{level - 1:02d}/manifest.json"
                )
            )
            if level == 1:
                parent_ok = parent is None
            else:
                parent_ok = (
                    isinstance(parent, dict)
                    and parent.get("path") == expected_parent_path
                    and parent.get("sha256") == previous_manifest_sha256
                )
            if not parent_ok:
                issues.append(_diagnostic_issue(
                    "parent_manifest_discontinuity",
                    manifest_relative,
                    "manifest does not hash-bind the exact preceding manifest",
                ))

    checkpoint_relative = f"{boundary_relative}/files/checkpoint.json"
    checkpoint, checkpoint_record, checkpoint_issues = _diagnostic_json_file(
        canonical_root,
        checkpoint_relative,
        missing_code="missing_exact_checkpoint",
    )
    issues.extend(checkpoint_issues)
    if checkpoint is not None:
        try:
            _validate_checkpoint(checkpoint, game=game, level=level)
        except ReleaseGateError:
            issues.append(_diagnostic_issue(
                "inexact_checkpoint",
                checkpoint_relative,
                f"checkpoint is not the exact {game} level {level} boundary",
            ))

    for source_name in sorted(REQUIRED_SOURCE_FILES):
        _, source_issues = _diagnostic_regular_file(
            canonical_root,
            f"{boundary_relative}/files/{source_name}",
            missing_code="missing_winning_source",
        )
        issues.extend(source_issues)

    transcripts_relative = f"{boundary_relative}/transcripts"
    transcripts_path = canonical_root / PurePosixPath(transcripts_relative)
    try:
        transcript_entries = _directory_entries(
            transcripts_path, label="host transcript evidence"
        )
    except ReleaseGateError:
        issues.append(_diagnostic_issue(
            "missing_host_transcript_evidence",
            transcripts_relative,
            "schema-v2 host transcript directory is missing or invalid",
        ))
    else:
        regular_transcripts = 0
        for entry in transcript_entries:
            metadata = _entry_lstat(entry, label="host transcript")
            relative = f"{transcripts_relative}/{entry.name}"
            if stat.S_ISREG(metadata.st_mode) and metadata.st_nlink == 1:
                regular_transcripts += 1
            else:
                issues.append(_diagnostic_issue(
                    "invalid_host_transcript_evidence",
                    relative,
                    "host transcript must be an unaliased regular file",
                ))
        if regular_transcripts == 0:
            issues.append(_diagnostic_issue(
                "missing_host_transcript_evidence",
                transcripts_relative,
                "at least one host-captured transcript is required",
            ))

    for audit_name, audit_relative_suffix in sorted(AUDIT_PATHS.items()):
        audit_relative = f"{boundary_relative}/{audit_relative_suffix}"
        audit, _, audit_issues = _diagnostic_json_file(
            canonical_root,
            audit_relative,
            missing_code=f"missing_{audit_name}_evidence",
        )
        issues.extend(audit_issues)
        if audit is not None:
            if audit.get("schema") != 1 or isinstance(
                audit.get("schema"), bool
            ):
                issues.append(_diagnostic_issue(
                    "invalid_audit_schema",
                    audit_relative,
                    "machine audit receipt must use schema 1",
                ))
            if any(
                audit.get(field) is True
                for field in ("passed", "validated", "clean")
            ):
                issues.append(_diagnostic_issue(
                    "boolean_only_gate_claim",
                    audit_relative,
                    "Boolean-only audit claims are not release evidence",
                ))

    status = (
        "legacy"
        if legacy
        else "schema2_candidate"
        if not issues
        else "invalid"
    )
    result: dict[str, Any] = {
        "status": status,
        "manifest_schema": manifest_schema,
        "issues": _sort_issues(issues),
    }
    return (
        result,
        checkpoint_record.sha256 if checkpoint_record is not None else None,
        manifest_record.sha256 if manifest_record is not None else None,
    )


def diagnose_release_migration(
    *,
    canonical_root: Path,
    environments_root: Path,
) -> dict[str, Any]:
    """Return a deterministic 183-boundary schema-migration diagnostic.

    This is intentionally non-admitting: ``schema2_candidate`` means only that
    the inexpensive migration scan found the expected shape.  Publication
    still requires the full hashing and semantic checks in
    :func:`build_release_receipt_body`.
    """
    root_issues: list[dict[str, str]] = []
    try:
        inventory, inventory_metadata = _discover_inventory(
            Path(environments_root)
        )
    except ReleaseGateError as exc:
        root_issues.append(_diagnostic_issue(
            "authoritative_inventory_invalid",
            "environment_files",
            str(exc),
        ))
        return {
            "schema": 1,
            "status": "FAIL",
            "inventory": {},
            "inventory_sha256": None,
            "inventory_metadata_sha256": {},
            "root_issues": root_issues,
            "games": {},
            "migration_queue": [],
            "summary": {
                "authoritative_games": 0,
                "authoritative_levels": 0,
                "canonical_reached_levels": 0,
                "schema2_candidate_boundaries": 0,
                "legacy_boundaries": 0,
                "missing_boundaries": 0,
                "invalid_boundaries": 0,
                "queued_boundaries": 0,
            },
        }

    canonical = Path(canonical_root)
    expected_game_roots = {f"{game}_legs" for game in inventory}
    found_game_roots: set[str] = set()
    try:
        root_entries = _directory_entries(canonical, label="canonical root")
    except ReleaseGateError as exc:
        root_entries = []
        root_issues.append(_diagnostic_issue(
            "canonical_root_invalid", ".", str(exc)
        ))
    for entry in root_entries:
        metadata = _entry_lstat(entry, label="canonical root entry")
        if (
            entry.name in expected_game_roots
            and stat.S_ISDIR(metadata.st_mode)
            and not stat.S_ISLNK(metadata.st_mode)
        ):
            found_game_roots.add(entry.name)
            continue
        code = (
            "mutable_non_evidence_root_entry"
            if stat.S_ISDIR(metadata.st_mode)
            else "unexpected_canonical_root_entry"
        )
        root_issues.append(_diagnostic_issue(
            code,
            entry.name,
            "entry is outside the frozen authoritative game trees",
        ))

    games: dict[str, Any] = {}
    migration_queue: list[dict[str, Any]] = []
    counts = {
        "schema2_candidate": 0,
        "legacy": 0,
        "missing": 0,
        "invalid": 0,
    }
    canonical_reached_total = 0

    for game, target in sorted(inventory.items()):
        game_root_name = f"{game}_legs"
        game_root = canonical / game_root_name
        game_issues: list[dict[str, str]] = []
        canonical_reached: int | None = None
        if game_root_name not in found_game_roots:
            game_issues.append(_diagnostic_issue(
                "missing_game_tree",
                game_root_name,
                "authoritative canonical game tree is missing or invalid",
            ))
        else:
            try:
                game_entries = _directory_entries(
                    game_root, label=f"{game} canonical tree"
                )
            except ReleaseGateError as exc:
                game_entries = []
                game_issues.append(_diagnostic_issue(
                    "invalid_game_tree", game_root_name, str(exc)
                ))
            for entry in game_entries:
                metadata = _entry_lstat(entry, label=f"{game} canonical entry")
                if stat.S_ISDIR(metadata.st_mode) and entry.name != (
                    "promotion_evidence"
                ):
                    game_issues.append(_diagnostic_issue(
                        "mutable_non_evidence_entry",
                        f"{game_root_name}/{entry.name}",
                        "mutable/non-evidence directory is forbidden in a freeze",
                    ))

            top_checkpoint, _, checkpoint_issues = _diagnostic_json_file(
                canonical,
                f"{game_root_name}/checkpoint.json",
                missing_code="missing_canonical_checkpoint",
            )
            game_issues.extend(checkpoint_issues)
            if top_checkpoint is not None and _is_int(
                top_checkpoint.get("reached")
            ):
                canonical_reached = top_checkpoint["reached"]
                if not 0 <= canonical_reached <= target:
                    game_issues.append(_diagnostic_issue(
                        "canonical_reached_out_of_range",
                        f"{game_root_name}/checkpoint.json",
                        f"reached must be within 0..{target}",
                    ))
                else:
                    canonical_reached_total += canonical_reached
                    if canonical_reached != target:
                        game_issues.append(_diagnostic_issue(
                            "canonical_game_incomplete",
                            f"{game_root_name}/checkpoint.json",
                            f"reached {canonical_reached} of {target}",
                        ))

            evidence_root = game_root / "promotion_evidence"
            try:
                evidence_entries = _directory_entries(
                    evidence_root, label=f"{game} promotion evidence"
                )
            except ReleaseGateError:
                evidence_entries = []
                game_issues.append(_diagnostic_issue(
                    "missing_promotion_evidence_root",
                    f"{game_root_name}/promotion_evidence",
                    "promotion evidence directory is missing or invalid",
                ))
            expected_level_names = {
                f"level_{level:02d}" for level in range(1, target + 1)
            }
            for entry in evidence_entries:
                metadata = _entry_lstat(
                    entry, label=f"{game} promotion evidence entry"
                )
                if (
                    entry.name not in expected_level_names
                    or not stat.S_ISDIR(metadata.st_mode)
                    or stat.S_ISLNK(metadata.st_mode)
                ):
                    game_issues.append(_diagnostic_issue(
                        "extra_or_invalid_level_entry",
                        f"{game_root_name}/promotion_evidence/{entry.name}",
                        f"expected only exact level_01..level_{target:02d} directories",
                    ))

        previous_checkpoint: str | None = None
        previous_manifest: str | None = None
        levels: dict[str, Any] = {}
        for level in range(1, target + 1):
            if game_root_name in found_game_roots:
                diagnostic, checkpoint_hash, manifest_hash = _diagnose_level(
                    canonical,
                    game=game,
                    level=level,
                    previous_checkpoint_sha256=previous_checkpoint,
                    previous_manifest_sha256=previous_manifest,
                )
            else:
                diagnostic = {
                    "status": "missing",
                    "issues": [_diagnostic_issue(
                        "missing_boundary",
                        (
                            f"{game_root_name}/promotion_evidence/"
                            f"level_{level:02d}"
                        ),
                        "game tree is missing",
                    )],
                }
                checkpoint_hash = None
                manifest_hash = None
            levels[f"{level:02d}"] = diagnostic
            status = diagnostic["status"]
            counts[status] += 1
            if status != "schema2_candidate":
                migration_queue.append({
                    "game": game,
                    "level": level,
                    "status": status,
                    "issue_codes": sorted({
                        issue["code"] for issue in diagnostic["issues"]
                    }),
                })
            previous_checkpoint = checkpoint_hash
            previous_manifest = manifest_hash

        if canonical_reached is None:
            game_status = "invalid"
        elif canonical_reached < target:
            game_status = "incomplete"
        elif any(
            level["status"] != "schema2_candidate"
            for level in levels.values()
        ):
            game_status = "migration_required"
        elif game_issues:
            game_status = "invalid"
        else:
            game_status = "schema2_candidate"
        games[game] = {
            "target_levels": target,
            "canonical_reached": canonical_reached,
            "status": game_status,
            "issues": _sort_issues(game_issues),
            "levels": levels,
        }

    root_issues = _sort_issues(root_issues)
    migration_queue.sort(
        key=lambda row: (row["game"], row["level"], row["status"])
    )
    queued = len(migration_queue)
    status = (
        "PASS"
        if (
            not root_issues
            and not any(game["issues"] for game in games.values())
            and queued == 0
        )
        else "FAIL"
    )
    return {
        "schema": 1,
        "status": status,
        "inventory": inventory,
        "inventory_sha256": _json_sha256(inventory),
        "inventory_metadata_sha256": inventory_metadata,
        "root_issues": root_issues,
        "games": games,
        "migration_queue": migration_queue,
        "summary": {
            "authoritative_games": len(inventory),
            "authoritative_levels": sum(inventory.values()),
            "canonical_reached_levels": canonical_reached_total,
            "schema2_candidate_boundaries": counts["schema2_candidate"],
            "legacy_boundaries": counts["legacy"],
            "missing_boundaries": counts["missing"],
            "invalid_boundaries": counts["invalid"],
            "queued_boundaries": queued,
        },
    }


def build_release_receipt_body(
    *,
    canonical_root: Path,
    environments_root: Path,
    release_identity: Mapping[str, Any],
    control_contract_files: Mapping[str, Path] | None = None,
) -> dict[str, Any]:
    """Audit live bytes and return the deterministic release-receipt body.

    This function performs no writes.  Use :func:`issue_release_receipt` for
    exclusive durable publication.
    """
    identity = _validate_release_identity(dict(release_identity))
    inventory, inventory_metadata = _discover_inventory(
        Path(environments_root)
    )
    controls = _control_contract_snapshot(
        _default_control_files()
        if control_contract_files is None
        else control_contract_files
    )
    verifier = _verifier_snapshot()
    allowed_tool_hashes = frozenset({
        controls["sha256"],
        verifier["sha256"],
        *controls["files_sha256"].values(),
        *verifier["files_sha256"].values(),
    })
    snapshot, evidence = _validate_canonical(
        Path(canonical_root),
        inventory,
        allowed_tool_hashes=allowed_tool_hashes,
    )
    body = {
        "schema": RELEASE_RECEIPT_SCHEMA,
        "release_identity": identity,
        "release_identity_sha256": _json_sha256(identity),
        "inventory": inventory,
        "inventory_sha256": _json_sha256(inventory),
        "inventory_metadata_sha256": inventory_metadata,
        "canonical_game_count": len(inventory),
        "authoritative_level_count": sum(inventory.values()),
        "canonical_tree_sha256": snapshot.sha256,
        "evidence": evidence,
        "evidence_sha256": _json_sha256(evidence),
        "verifier": verifier,
        "control_contract": controls,
    }
    _require_exact_keys(body, _RELEASE_FIELDS, label="release receipt")
    return body


def build_partial_release_receipt_body(
    *,
    canonical_root: Path,
    environments_root: Path,
    release_identity: Mapping[str, Any],
    expected_claimed_levels: int,
    control_contract_files: Mapping[str, Path] | None = None,
) -> dict[str, Any]:
    """Audit a strongest-known incomplete freeze without weakening 183 mode.

    The authoritative inventory remains 25 games / 183 levels.  The claimed
    inventory is derived from the frozen top-level checkpoints, never supplied
    by the caller, and every level from one through each claimed frontier must
    pass the same schema-v2 boundary validation used by the complete release.
    Missing suffixes are enumerated explicitly and cannot be mistaken for
    solved evidence.
    """
    if (
        not _is_int(expected_claimed_levels)
        or expected_claimed_levels <= 0
        or expected_claimed_levels >= EXPECTED_LEVELS
    ):
        raise ReleaseGateError(
            "partial release expected-claimed-levels must be within 1..182"
        )
    identity = _validate_release_identity(dict(release_identity))
    inventory, inventory_metadata = _discover_inventory(
        Path(environments_root)
    )
    claimed = _claimed_inventory(Path(canonical_root), inventory)
    claimed_total = sum(claimed.values())
    if claimed_total != expected_claimed_levels:
        raise ReleaseGateError(
            "partial release frontier count mismatch; "
            f"expected {expected_claimed_levels}, found {claimed_total}"
        )
    missing = _unclaimed_boundaries(inventory, claimed)
    if len(missing) != sum(inventory.values()) - claimed_total:
        raise ReleaseGateError("partial release gap accounting mismatch")

    controls = _control_contract_snapshot(
        _default_control_files()
        if control_contract_files is None
        else control_contract_files
    )
    verifier = _verifier_snapshot()
    allowed_tool_hashes = frozenset({
        controls["sha256"],
        verifier["sha256"],
        *controls["files_sha256"].values(),
        *verifier["files_sha256"].values(),
    })
    snapshot, evidence = _validate_canonical(
        Path(canonical_root),
        claimed,
        allowed_tool_hashes=allowed_tool_hashes,
    )
    body = {
        "schema": PARTIAL_RELEASE_RECEIPT_SCHEMA,
        "kind": "partial_campaign_freeze",
        "release_identity": identity,
        "release_identity_sha256": _json_sha256(identity),
        "inventory": inventory,
        "inventory_sha256": _json_sha256(inventory),
        "inventory_metadata_sha256": inventory_metadata,
        "claimed_inventory": claimed,
        "claimed_inventory_sha256": _json_sha256(claimed),
        "canonical_game_count": len(claimed),
        "authoritative_level_count": sum(inventory.values()),
        "claimed_level_count": claimed_total,
        "unclaimed_boundaries": missing,
        "complete": False,
        "canonical_tree_sha256": snapshot.sha256,
        "evidence": evidence,
        "evidence_sha256": _json_sha256(evidence),
        "verifier": verifier,
        "control_contract": controls,
    }
    _require_exact_keys(
        body,
        _PARTIAL_RELEASE_FIELDS,
        label="partial release receipt",
    )
    return body


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISDIR(metadata.st_mode):
            raise ReleaseGateError(f"cannot fsync non-directory: {path}")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_content_addressed(
    receipt_directory: Path,
    payload: bytes,
) -> tuple[Path, str]:
    if not payload.endswith(b"\n"):
        raise ReleaseGateError("receipt payload must have a canonical newline")
    directory = Path(receipt_directory)
    if not directory.exists():
        parent = directory.parent
        _lstat_directory(parent, label="receipt-store parent")
        try:
            directory.mkdir(mode=0o700)
        except OSError as exc:
            raise ReleaseGateError(
                f"cannot create receipt directory: {directory}"
            ) from exc
        _fsync_directory(parent)
    _lstat_directory(directory, label="receipt store")
    digest = hashlib.sha256(payload).hexdigest()
    target = directory / f"{digest}.json"
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(target, flags, 0o400)
    except FileExistsError as exc:
        raise ReleaseGateError(
            f"immutable release receipt already exists: {target}"
        ) from exc
    except OSError as exc:
        raise ReleaseGateError(f"cannot create release receipt: {target}") from exc
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise ReleaseGateError(f"short write for release receipt: {target}")
            offset += written
        os.fsync(descriptor)
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or metadata.st_size != len(payload)
        ):
            raise ReleaseGateError(
                f"release receipt inode is not immutable regular evidence: {target}"
            )
    except BaseException:
        os.close(descriptor)
        try:
            target.unlink()
        except OSError:
            pass
        raise
    else:
        os.close(descriptor)
    _fsync_directory(directory)
    return target, digest


def issue_release_receipt(
    *,
    canonical_root: Path,
    environments_root: Path,
    receipt_directory: Path,
    release_identity: Mapping[str, Any],
    control_contract_files: Mapping[str, Path] | None = None,
) -> ReleaseReceipt:
    """Audit twice, exclusively publish, then independently re-verify a receipt."""
    canonical = Path(canonical_root).resolve()
    receipt_dir = Path(receipt_directory)
    try:
        receipt_parent = receipt_dir.resolve(strict=False)
    except OSError as exc:
        raise ReleaseGateError("cannot resolve receipt directory") from exc
    if canonical == receipt_parent or canonical in receipt_parent.parents:
        raise ReleaseGateError(
            "receipt store must be outside the frozen canonical tree"
        )

    first = build_release_receipt_body(
        canonical_root=canonical_root,
        environments_root=environments_root,
        release_identity=release_identity,
        control_contract_files=control_contract_files,
    )
    second = build_release_receipt_body(
        canonical_root=canonical_root,
        environments_root=environments_root,
        release_identity=release_identity,
        control_contract_files=control_contract_files,
    )
    if first != second:
        raise ReleaseGateError(
            "canonical, inventory, verifier, or control bytes changed during release"
        )
    payload = _canonical_json(first) + b"\n"
    path, digest = _write_content_addressed(receipt_directory, payload)
    verified = verify_release_receipt(
        receipt_path=path,
        canonical_root=canonical_root,
        environments_root=environments_root,
        control_contract_files=control_contract_files,
    )
    if verified.sha256 != digest:
        raise ReleaseGateError("new release receipt failed content-addressing")
    return verified


def issue_partial_release_receipt(
    *,
    canonical_root: Path,
    environments_root: Path,
    receipt_directory: Path,
    release_identity: Mapping[str, Any],
    expected_claimed_levels: int,
    control_contract_files: Mapping[str, Path] | None = None,
) -> ReleaseReceipt:
    """Double-audit and immutably publish an explicitly incomplete freeze."""
    canonical = Path(canonical_root).resolve()
    receipt_dir = Path(receipt_directory)
    try:
        receipt_parent = receipt_dir.resolve(strict=False)
    except OSError as exc:
        raise ReleaseGateError("cannot resolve receipt directory") from exc
    if canonical == receipt_parent or canonical in receipt_parent.parents:
        raise ReleaseGateError(
            "receipt store must be outside the frozen canonical tree"
        )

    arguments = {
        "canonical_root": canonical_root,
        "environments_root": environments_root,
        "release_identity": release_identity,
        "expected_claimed_levels": expected_claimed_levels,
        "control_contract_files": control_contract_files,
    }
    first = build_partial_release_receipt_body(**arguments)
    second = build_partial_release_receipt_body(**arguments)
    if first != second:
        raise ReleaseGateError(
            "canonical, inventory, verifier, or control bytes changed "
            "during partial freeze"
        )
    payload = _canonical_json(first) + b"\n"
    path, digest = _write_content_addressed(receipt_directory, payload)
    verified = verify_partial_release_receipt(
        receipt_path=path,
        canonical_root=canonical_root,
        environments_root=environments_root,
        control_contract_files=control_contract_files,
    )
    if verified.sha256 != digest:
        raise ReleaseGateError(
            "new partial release receipt failed content-addressing"
        )
    return verified


def verify_release_receipt(
    *,
    receipt_path: Path,
    canonical_root: Path,
    environments_root: Path,
    control_contract_files: Mapping[str, Path] | None = None,
) -> ReleaseReceipt:
    """Re-audit live bytes and require exact equality with a frozen receipt."""
    path = Path(receipt_path)
    record = _file_record(path, path.name)
    if path.suffix != ".json" or path.stem != record.sha256:
        raise ReleaseGateError(
            "release receipt filename is not its exact content hash"
        )
    raw = _read_record_bytes(path.parent, record)
    try:
        body = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ReleaseGateError("release receipt is invalid JSON") from exc
    if raw != _canonical_json(body) + b"\n":
        raise ReleaseGateError("release receipt bytes are not canonical JSON")
    if not isinstance(body, dict):
        raise ReleaseGateError("release receipt must be a JSON object")
    _require_exact_keys(body, _RELEASE_FIELDS, label="release receipt")
    if (
        body["schema"] != RELEASE_RECEIPT_SCHEMA
        or isinstance(body["schema"], bool)
    ):
        raise ReleaseGateError("release receipt schema mismatch")
    identity = _validate_release_identity(body["release_identity"])
    if body["release_identity_sha256"] != _json_sha256(identity):
        raise ReleaseGateError("release identity hash mismatch")
    rebuilt = build_release_receipt_body(
        canonical_root=canonical_root,
        environments_root=environments_root,
        release_identity=identity,
        control_contract_files=control_contract_files,
    )
    if rebuilt != body:
        raise ReleaseGateError(
            "release receipt no longer matches frozen canonical/evidence bytes"
        )
    return ReleaseReceipt(path=path, sha256=record.sha256, body=body)


def verify_partial_release_receipt(
    *,
    receipt_path: Path,
    canonical_root: Path,
    environments_root: Path,
    control_contract_files: Mapping[str, Path] | None = None,
) -> ReleaseReceipt:
    """Re-audit an incomplete freeze and require its explicit gaps to match."""
    path = Path(receipt_path)
    record = _file_record(path, path.name)
    if path.suffix != ".json" or path.stem != record.sha256:
        raise ReleaseGateError(
            "partial release receipt filename is not its exact content hash"
        )
    raw = _read_record_bytes(path.parent, record)
    try:
        body = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ReleaseGateError(
            "partial release receipt is invalid JSON"
        ) from exc
    if raw != _canonical_json(body) + b"\n":
        raise ReleaseGateError(
            "partial release receipt bytes are not canonical JSON"
        )
    if not isinstance(body, dict):
        raise ReleaseGateError(
            "partial release receipt must be a JSON object"
        )
    _require_exact_keys(
        body,
        _PARTIAL_RELEASE_FIELDS,
        label="partial release receipt",
    )
    if (
        body["schema"] != PARTIAL_RELEASE_RECEIPT_SCHEMA
        or isinstance(body["schema"], bool)
        or body["kind"] != "partial_campaign_freeze"
        or body["complete"] is not False
        or not _is_int(body["claimed_level_count"])
    ):
        raise ReleaseGateError("partial release receipt schema mismatch")
    identity = _validate_release_identity(body["release_identity"])
    if body["release_identity_sha256"] != _json_sha256(identity):
        raise ReleaseGateError("partial release identity hash mismatch")
    rebuilt = build_partial_release_receipt_body(
        canonical_root=canonical_root,
        environments_root=environments_root,
        release_identity=identity,
        expected_claimed_levels=body["claimed_level_count"],
        control_contract_files=control_contract_files,
    )
    if rebuilt != body:
        raise ReleaseGateError(
            "partial release receipt no longer matches frozen "
            "canonical/evidence bytes"
        )
    return ReleaseReceipt(path=path, sha256=record.sha256, body=body)


def _load_identity_file(path: Path) -> dict[str, Any]:
    record = _file_record(path, path.name)
    raw = _read_record_bytes(path.parent, record)
    try:
        value = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ReleaseGateError("identity file is invalid JSON") from exc
    if not isinstance(value, dict):
        raise ReleaseGateError("identity file must contain an object")
    return value


def _summary(body: Mapping[str, Any], *, receipt: str | None = None) -> dict[str, Any]:
    result = {
        "status": "PASS",
        "games": body["canonical_game_count"],
        "levels": body["authoritative_level_count"],
        "inventory_sha256": body["inventory_sha256"],
        "canonical_tree_sha256": body["canonical_tree_sha256"],
        "evidence_sha256": body["evidence_sha256"],
        "verifier_sha256": body["verifier"]["sha256"],
        "control_contract_sha256": body["control_contract"]["sha256"],
    }
    if receipt is not None:
        result["receipt"] = receipt
    return result


def _partial_summary(
    body: Mapping[str, Any],
    *,
    receipt: str | None = None,
) -> dict[str, Any]:
    result = {
        "status": "PASS",
        "kind": body["kind"],
        "games": body["canonical_game_count"],
        "claimed_levels": body["claimed_level_count"],
        "authoritative_levels": body["authoritative_level_count"],
        "unclaimed_boundaries": body["unclaimed_boundaries"],
        "inventory_sha256": body["inventory_sha256"],
        "claimed_inventory_sha256": body["claimed_inventory_sha256"],
        "canonical_tree_sha256": body["canonical_tree_sha256"],
        "evidence_sha256": body["evidence_sha256"],
        "verifier_sha256": body["verifier"]["sha256"],
        "control_contract_sha256": body["control_contract"]["sha256"],
    }
    if receipt is not None:
        result["receipt"] = receipt
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit or receipt a frozen ARC-AGI-3 canonical campaign"
    )
    parser.add_argument(
        "--canonical-root",
        type=Path,
        default=Path(__file__).resolve().parent / "agent_solutions",
    )
    parser.add_argument(
        "--environments-root",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "environment_files",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("diagnose")

    audit = subparsers.add_parser("audit")
    audit.add_argument("--identity-json", type=Path, required=True)

    partial_audit = subparsers.add_parser("audit-partial")
    partial_audit.add_argument("--identity-json", type=Path, required=True)
    partial_audit.add_argument(
        "--expected-claimed-levels", type=int, required=True
    )

    issue = subparsers.add_parser("issue")
    issue.add_argument("--identity-json", type=Path, required=True)
    issue.add_argument("--receipt-directory", type=Path, required=True)

    partial_issue = subparsers.add_parser("issue-partial")
    partial_issue.add_argument("--identity-json", type=Path, required=True)
    partial_issue.add_argument(
        "--expected-claimed-levels", type=int, required=True
    )
    partial_issue.add_argument(
        "--receipt-directory", type=Path, required=True
    )

    verify = subparsers.add_parser("verify")
    verify.add_argument("--receipt", type=Path, required=True)

    partial_verify = subparsers.add_parser("verify-partial")
    partial_verify.add_argument("--receipt", type=Path, required=True)

    args = parser.parse_args(argv)
    try:
        if args.command == "diagnose":
            diagnostic = diagnose_release_migration(
                canonical_root=args.canonical_root,
                environments_root=args.environments_root,
            )
            print(json.dumps(diagnostic, sort_keys=True))
            return 0 if diagnostic["status"] == "PASS" else 1
        if args.command == "audit":
            body = build_release_receipt_body(
                canonical_root=args.canonical_root,
                environments_root=args.environments_root,
                release_identity=_load_identity_file(args.identity_json),
            )
            print(json.dumps(_summary(body), sort_keys=True))
        elif args.command == "audit-partial":
            body = build_partial_release_receipt_body(
                canonical_root=args.canonical_root,
                environments_root=args.environments_root,
                release_identity=_load_identity_file(args.identity_json),
                expected_claimed_levels=args.expected_claimed_levels,
            )
            print(json.dumps(_partial_summary(body), sort_keys=True))
        elif args.command == "issue":
            receipt = issue_release_receipt(
                canonical_root=args.canonical_root,
                environments_root=args.environments_root,
                receipt_directory=args.receipt_directory,
                release_identity=_load_identity_file(args.identity_json),
            )
            print(json.dumps(
                _summary(receipt.body, receipt=str(receipt.path)),
                sort_keys=True,
            ))
        elif args.command == "issue-partial":
            receipt = issue_partial_release_receipt(
                canonical_root=args.canonical_root,
                environments_root=args.environments_root,
                receipt_directory=args.receipt_directory,
                release_identity=_load_identity_file(args.identity_json),
                expected_claimed_levels=args.expected_claimed_levels,
            )
            print(json.dumps(
                _partial_summary(
                    receipt.body,
                    receipt=str(receipt.path),
                ),
                sort_keys=True,
            ))
        elif args.command == "verify":
            receipt = verify_release_receipt(
                receipt_path=args.receipt,
                canonical_root=args.canonical_root,
                environments_root=args.environments_root,
            )
            print(json.dumps(
                _summary(receipt.body, receipt=str(receipt.path)),
                sort_keys=True,
            ))
        else:
            receipt = verify_partial_release_receipt(
                receipt_path=args.receipt,
                canonical_root=args.canonical_root,
                environments_root=args.environments_root,
            )
            print(json.dumps(
                _partial_summary(
                    receipt.body,
                    receipt=str(receipt.path),
                ),
                sort_keys=True,
            ))
    except ReleaseGateError as exc:
        print(json.dumps({"status": "FAIL", "error": str(exc)}, sort_keys=True))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
