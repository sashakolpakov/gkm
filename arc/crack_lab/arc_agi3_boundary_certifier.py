#!/usr/bin/env python3
"""Build a uniform schema-v2 ARC-AGI-3 release tree by fresh replay.

The live ``agent_solutions`` tree is an acquisition archive.  It contains
legacy schema-1 promotions, resumable WIP, and a small number of older
boundaries whose exact source survived only in a WIP snapshot.  This tool
never edits that archive.  It selects one source snapshot per boundary,
replays the source from zero, replays the resulting exact path independently,
scans every admitted source/transcript byte for taint, and writes a separate
minimal release tree accepted by :mod:`arc_agi3_release_gate`.

Historical gaps may use harness-recorded deterministic auto-solve snapshots.
If every retained source for an otherwise replay-valid boundary fails fresh
source execution, the final fallback is a minimal source capsule generated
from the canonical exact action boundary.  Both forms carry explicit
deterministic-reconstruction provenance and are never relabeled as
contemporaneous pre-debrief acquisition boundaries.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import tempfile
import uuid
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

import arc_agi3_release_gate as release_gate
import arc_agi3_proposer_boundary as Boundary
import arc_agi3_source_schema as SourceSchema
import gkm_arena
import gkm_legs


SCHEMA = 1
REQUIRED_SOURCES = ("legs.py", "players.py", "solve.py")
PHASE_PRIORITY = {
    "reached_before_debrief": 0,
    "debrief_skipped_policy": 1,
    "recovered_path_artifact": 2,
    "after_auto_solve_debrief": 30,
    "after_propose": 40,
    "after_debrief": 50,
    # Schema-1 promotion files were copied only after the optional debrief.
    # Their manifest does not bind the retained pre-debrief snapshot, so their
    # acquisition phase is unresolved even when the source still replays.
    "legacy_schema1_promoted_source": 80,
    "deterministic_exact_path_reconstruction": 100,
}
HISTORICAL_SOURCE_PHASES = frozenset({
    "reached_before_debrief",
    "debrief_skipped_policy",
    "recovered_path_artifact",
})
HISTORICAL_TRANSCRIPT_EVENT_TYPES = frozenset({
    "thread.started",
    "turn.started",
    "item.started",
    "item.updated",
    "item.completed",
    "turn.completed",
})
_SOURCE_SCHEMA_PATH = Path(SourceSchema.__file__).resolve()
_LOADED_SOURCE_SCHEMA_SHA256 = hashlib.sha256(
    _SOURCE_SCHEMA_PATH.read_bytes()
).hexdigest()


class CertificationError(RuntimeError):
    """A source selection, replay, taint, or publication gate failed."""


@dataclass(frozen=True)
class SourceCapture:
    """One immutable byte boundary shared by scan, replay, and staging."""

    payloads: tuple[tuple[str, bytes], ...]
    origins: tuple[tuple[str, str], ...]
    transcript_payload: bytes | None
    transcript_name: str | None
    filesystem_policy_sha256: str
    source_schema_sha256: str
    allowed_import_roots_sha256: str


@dataclass(frozen=True)
class SourceCandidate:
    game: str
    level: int
    kind: str
    phase: str
    files_dir: Path
    dependency_dir: Path | None
    transcript: Path | None
    origin: Path
    historical_source_boundary: bool
    deterministic_reconstruction: bool
    reconstruction_path: tuple[Any, ...] | None = None
    retained_historical_phase: bool = False
    historical_transcript_complete: bool = False
    historical_transcript_failure: str | None = None
    capture: SourceCapture | None = None

    @property
    def priority(self) -> tuple[int, int, str]:
        kind_rank = {
            "legacy_promotion": 0,
            "wip_snapshot": 1,
            "exact_path_reconstruction": 2,
        }.get(self.kind, 99)
        return (
            PHASE_PRIORITY.get(self.phase, 99),
            kind_rank,
            self.origin.as_posix(),
        )


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _stat_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _read_regular_snapshot(
    path: Path,
    *,
    kind: str,
    max_bytes: int,
) -> bytes:
    """Read one regular file descriptor and reject an identity-changing read."""
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise CertificationError(
            f"cannot capture {kind} as a regular file: {path} "
            f"({type(exc).__name__})"
        ) from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size <= 0
            or before.st_size > max_bytes
        ):
            raise CertificationError(
                f"unsafe {kind} identity or size: {path}"
            )
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        after = os.fstat(descriptor)
        raw = b"".join(chunks)
        if (
            _stat_identity(before) != _stat_identity(after)
            or len(raw) != before.st_size
        ):
            raise CertificationError(
                f"{kind} changed while its byte boundary was captured: {path}"
            )
        return raw
    finally:
        os.close(descriptor)


def _regular_file(path: Path, *, nonempty: bool = False) -> bool:
    try:
        metadata = path.lstat()
    except OSError:
        return False
    return (
        stat.S_ISREG(metadata.st_mode)
        and not stat.S_ISLNK(metadata.st_mode)
        and metadata.st_nlink == 1
        and (not nonempty or metadata.st_size > 0)
    )


def _read_json(path: Path) -> dict[str, Any]:
    if not _regular_file(path, nonempty=True):
        raise CertificationError(f"JSON input is not a regular file: {path}")
    try:
        value = json.loads(path.read_bytes())
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise CertificationError(f"invalid JSON input: {path}") from exc
    if not isinstance(value, dict):
        raise CertificationError(f"JSON input is not an object: {path}")
    return value


def _write_new_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o644,
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)


def _write_new_json(path: Path, value: object) -> None:
    _write_new_bytes(path, _canonical_json(value))


def _copy_regular(source: Path, destination: Path) -> None:
    if not _regular_file(source, nonempty=True):
        raise CertificationError(f"source is not a nonempty regular file: {source}")
    _write_new_bytes(destination, source.read_bytes())


def _relative_to(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise CertificationError(
            f"candidate escaped the acquisition archive: {path}"
        ) from exc


def _candidate_transcript(
    *,
    files_dir: Path,
    evidence_dir: Path | None = None,
) -> Path | None:
    preferred: list[Path] = [files_dir / "proposer_last.log"]
    if evidence_dir is not None:
        preferred.append(evidence_dir / "proposer_last.log")
        turns = evidence_dir / "codex_turns"
        if turns.is_dir() and not turns.is_symlink():
            preferred.extend(sorted(turns.glob("*.jsonl"), reverse=True))
    for path in preferred:
        if _regular_file(path, nonempty=True):
            return path
    return None


def _historical_transcript_verdict(
    transcript: Path | None,
) -> tuple[bool, str | None]:
    """Validate the one successful proposer turn needed for historical credit.

    Legacy source remains eligible for fresh release replay when this check
    fails, but it cannot be represented as a contemporaneous acquisition
    boundary or contribute its old marginal.  In particular, raw CLI stderr
    interleaved into nominal JSONL is not silently ignored here.
    """
    if transcript is None:
        return False, "missing_transcript"
    try:
        raw = _read_regular_snapshot(
            transcript,
            kind="historical transcript",
            max_bytes=gkm_legs.MAX_TAINT_SCAN_BYTES,
        )
    except CertificationError:
        return False, "transcript_not_regular"
    return _historical_transcript_bytes_verdict(raw)


def _historical_transcript_bytes_verdict(
    raw: bytes,
) -> tuple[bool, str | None]:
    """Judge historical credit from the exact transcript bytes retained."""
    try:
        text = raw.decode("utf-8")
    except UnicodeError:
        return False, "transcript_not_utf8"
    if not raw.endswith(b"\n") or b"\r" in raw:
        return False, "transcript_not_exact_lf_terminated"

    events: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.split("\n")[:-1], 1):
        if not line:
            return False, f"empty_transcript_line_{line_number}"
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            return False, f"malformed_transcript_json_line_{line_number}"
        if not isinstance(event, dict):
            return False, f"nonobject_transcript_event_line_{line_number}"
        event_type = event.get("type")
        if event_type not in HISTORICAL_TRANSCRIPT_EVENT_TYPES:
            return False, f"unknown_transcript_event_line_{line_number}"
        events.append(event)
    if not events:
        return False, "empty_transcript"
    if sum(event.get("type") == "thread.started" for event in events) != 1:
        return False, "historical_transcript_thread_count"
    if sum(event.get("type") == "turn.started" for event in events) != 1:
        return False, "historical_transcript_turn_start_count"
    if sum(event.get("type") == "turn.completed" for event in events) != 1:
        return False, "historical_transcript_turn_completion_count"
    if events[-1].get("type") != "turn.completed":
        return False, "historical_transcript_not_terminal"

    started: dict[str, int] = {}
    completed: dict[str, int] = {}
    for line_number, event in enumerate(events, 1):
        if event.get("type") not in {
            "item.started", "item.updated", "item.completed"
        }:
            continue
        item = event.get("item")
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("id"), str)
            or not item["id"]
            or not isinstance(item.get("type"), str)
            or not item["type"]
        ):
            return False, f"malformed_transcript_item_line_{line_number}"
        item_id = item["id"]
        if event["type"] == "item.started":
            started[item_id] = started.get(item_id, 0) + 1
        elif event["type"] == "item.completed":
            completed[item_id] = completed.get(item_id, 0) + 1
    if any(count != 1 for count in started.values()):
        return False, "historical_transcript_duplicate_item_start"
    if any(completed.get(item_id) != 1 for item_id in started):
        return False, "historical_transcript_unclosed_item"
    return True, None


def _has_required_sources(files_dir: Path) -> bool:
    return all(_regular_file(files_dir / name, nonempty=True)
               for name in REQUIRED_SOURCES)


def boundary_candidates(
    source_root: Path,
    *,
    game: str,
    level: int,
    include_exact_path_reconstruction: bool = True,
) -> list[SourceCandidate]:
    """Return admissible-shaped candidates in deterministic preference order."""
    game_root = source_root / f"{game}_legs"
    candidates: list[SourceCandidate] = []

    evidence_dir = (
        game_root / "promotion_evidence" / f"level_{level:02d}"
    )
    legacy_files = evidence_dir / "files"
    legacy_transcript = _candidate_transcript(
        files_dir=legacy_files,
        evidence_dir=evidence_dir,
    )
    if _has_required_sources(legacy_files):
        transcript_complete, transcript_failure = (
            _historical_transcript_verdict(legacy_transcript)
        )
        candidates.append(SourceCandidate(
            game=game,
            level=level,
            kind="legacy_promotion",
            phase="legacy_schema1_promoted_source",
            files_dir=legacy_files,
            dependency_dir=None,
            transcript=legacy_transcript,
            origin=evidence_dir,
            # The schema-1 publisher copied these files after debrief and did
            # not bind the earlier reached-before-debrief snapshot.  A clean
            # transcript cannot recover that missing source-phase identity.
            historical_source_boundary=False,
            deterministic_reconstruction=False,
            retained_historical_phase=False,
            historical_transcript_complete=transcript_complete,
            historical_transcript_failure=transcript_failure,
        ))

    wip_level = game_root / "wip_context" / f"level_{level:02d}"
    if wip_level.is_dir() and not wip_level.is_symlink():
        for metadata_path in sorted(wip_level.glob("*/metadata.json")):
            try:
                metadata = _read_json(metadata_path)
            except CertificationError:
                continue
            phase = metadata.get("phase")
            if (
                metadata.get("game") != game
                or metadata.get("level") != level
                or metadata.get("reached") != level
                or not isinstance(phase, str)
                or phase not in PHASE_PRIORITY
            ):
                continue
            files_dir = metadata_path.parent / "files"
            transcript = _candidate_transcript(files_dir=files_dir)
            if not _has_required_sources(files_dir):
                continue
            retained_historical_phase = phase in HISTORICAL_SOURCE_PHASES
            transcript_complete, transcript_failure = (
                _historical_transcript_verdict(transcript)
            )
            candidates.append(SourceCandidate(
                game=game,
                level=level,
                kind="wip_snapshot",
                phase=phase,
                files_dir=files_dir,
                dependency_dir=files_dir,
                transcript=transcript,
                origin=metadata_path.parent,
                historical_source_boundary=(
                    retained_historical_phase and transcript_complete
                ),
                deterministic_reconstruction=(
                    phase == "after_auto_solve_debrief"
                ),
                retained_historical_phase=retained_historical_phase,
                historical_transcript_complete=transcript_complete,
                historical_transcript_failure=transcript_failure,
            ))

    wip_candidates = [
        candidate for candidate in candidates
        if candidate.kind == "wip_snapshot"
    ]
    enriched: list[SourceCandidate] = []
    for candidate in candidates:
        if candidate.kind != "legacy_promotion":
            enriched.append(candidate)
            continue
        core_hashes = {
            name: _sha256_file(candidate.files_dir / name)
            for name in REQUIRED_SOURCES
        }
        matching = next(
            (
                wip for wip in wip_candidates
                if all(
                    _sha256_file(wip.files_dir / name) == digest
                    for name, digest in core_hashes.items()
                )
            ),
            None,
        )
        enriched.append(replace(
            candidate,
            dependency_dir=matching.files_dir if matching else None,
        ))

    checkpoint_path = game_root / "checkpoint.json"
    if (
        include_exact_path_reconstruction
        and _regular_file(checkpoint_path, nonempty=True)
    ):
        try:
            checkpoint = _read_json(checkpoint_path)
            final_path = checkpoint.get("final_path")
            if (
                checkpoint.get("game") == game
                and checkpoint.get("validated") is True
                and isinstance(checkpoint.get("reached"), int)
                and checkpoint["reached"] >= level
                and isinstance(final_path, list)
            ):
                exact_path = gkm_legs.exact_level_boundary(
                    game, final_path, level
                )
                if (
                    exact_path
                    and gkm_arena.validate(game, exact_path, level)
                ):
                    enriched.append(SourceCandidate(
                        game=game,
                        level=level,
                        kind="exact_path_reconstruction",
                        phase="deterministic_exact_path_reconstruction",
                        files_dir=game_root,
                        dependency_dir=None,
                        transcript=None,
                        origin=checkpoint_path,
                        historical_source_boundary=False,
                        deterministic_reconstruction=True,
                        reconstruction_path=tuple(exact_path),
                        retained_historical_phase=False,
                        historical_transcript_complete=False,
                        historical_transcript_failure="missing_transcript",
                    ))
        except (CertificationError, OSError, ValueError, TypeError):
            # A malformed or non-replayable checkpoint is never admitted as a
            # reconstruction candidate.  Other retained candidates remain
            # independently eligible.
            pass
    return sorted(enriched, key=lambda candidate: candidate.priority)


def _exact_path_source_payloads(
    candidate: SourceCandidate,
) -> tuple[dict[str, bytes], dict[str, str]]:
    if candidate.reconstruction_path is None:
        raise CertificationError(
            "exact-path reconstruction candidate has no exact path"
        )
    actions = _normalized_actions(candidate.reconstruction_path)
    encoded_actions = repr(actions)
    payloads = {
        "legs.py": (
            "# Deterministic release-only exact-boundary reconstruction.\n"
            f"EXACT_PATH = {encoded_actions}\n\n"
            "def play_exact_path(env):\n"
            "    for action in EXACT_PATH:\n"
            "        if isinstance(action, list):\n"
            "            env.step(*action)\n"
            "        else:\n"
            "            env.step(action)\n"
        ).encode("utf-8"),
        "players.py": (
            "from legs import play_exact_path\n\n"
            "def play_level_1(env):\n"
            "    play_exact_path(env)\n"
        ).encode("utf-8"),
        "solve.py": (
            "import players\n\n"
            "def solve(env):\n"
            "    if env.levels_completed == 0:\n"
            "        players.play_level_1(env)\n"
        ).encode("utf-8"),
    }
    origins = {
        name: candidate.origin.as_posix()
        for name in payloads
    }
    return payloads, origins


def _source_payloads(
    candidate: SourceCandidate,
) -> tuple[dict[str, bytes], dict[str, str]]:
    if candidate.capture is not None:
        return (
            dict(candidate.capture.payloads),
            dict(candidate.capture.origins),
        )
    if candidate.kind == "exact_path_reconstruction":
        return _exact_path_source_payloads(candidate)
    payloads = {
        name: _read_regular_snapshot(
            candidate.files_dir / name,
            kind=f"winning source {name}",
            max_bytes=Boundary.MAX_SOURCE_BYTES,
        )
        for name in REQUIRED_SOURCES
    }
    origins = {
        name: (candidate.files_dir / name).as_posix()
        for name in REQUIRED_SOURCES
    }
    needs_perception = any(
        b"from perception import" in payload
        or b"import perception" in payload
        for payload in payloads.values()
    )
    if needs_perception:
        perception: Path | None = None
        for directory in (candidate.files_dir, candidate.dependency_dir):
            if (
                directory is not None
                and _regular_file(directory / "perception.py", nonempty=True)
            ):
                perception = directory / "perception.py"
                break
        if perception is None:
            payloads["perception.py"] = gkm_legs.PERCEPTION_SEED.encode("utf-8")
            origins["perception.py"] = "harness:PERCEPTION_SEED"
        else:
            payloads["perception.py"] = _read_regular_snapshot(
                perception,
                kind="winning source perception.py",
                max_bytes=Boundary.MAX_SOURCE_BYTES,
            )
            origins["perception.py"] = perception.as_posix()
    return payloads, origins


def _allowed_import_roots(
    payloads: Mapping[str, bytes],
) -> frozenset[str]:
    local_import_roots = frozenset({
        PurePosixPath(name).stem
        for name in payloads
        if PurePosixPath(name).suffix in {".py", ".pyw"}
    })
    return (
        SourceSchema.STDLIB_ROOTS
        | SourceSchema.ALLOWED_THIRD_PARTY_ROOTS
        | local_import_roots
    )


def _allowed_import_roots_sha256(
    payloads: Mapping[str, bytes],
) -> str:
    return _sha256_bytes(_canonical_json({
        "source_schema": SourceSchema.SCHEMA,
        "pinned_numpy_version": SourceSchema.PINNED_NUMPY_VERSION,
        "allowed_import_roots": sorted(_allowed_import_roots(payloads)),
    }))


def _source_schema_sha256() -> str:
    current = _sha256_file(_SOURCE_SCHEMA_PATH)
    if current != _LOADED_SOURCE_SCHEMA_SHA256:
        raise CertificationError(
            "shared source schema changed after certifier import"
        )
    return _LOADED_SOURCE_SCHEMA_SHA256


def _capture_candidate(candidate: SourceCandidate) -> SourceCandidate:
    """Freeze every admitted source/transcript byte before any scan or replay."""
    if candidate.capture is not None:
        return candidate
    payloads, origins = _source_payloads(candidate)
    try:
        policy_sha256 = Boundary.policy_sha256()
    except (OSError, RuntimeError) as exc:
        raise CertificationError(
            f"filesystem boundary policy identity is unavailable: {exc}"
        ) from exc
    schema_sha256 = _source_schema_sha256()

    transcript_payload: bytes | None = None
    transcript_name: str | None = None
    transcript_complete = False
    transcript_failure: str | None = "missing_transcript"
    if candidate.transcript is not None:
        transcript_payload = _read_regular_snapshot(
            candidate.transcript,
            kind="source transcript",
            max_bytes=gkm_legs.MAX_TAINT_SCAN_BYTES,
        )
        transcript_name = candidate.transcript.name
        transcript_complete, transcript_failure = (
            _historical_transcript_bytes_verdict(transcript_payload)
        )

    capture = SourceCapture(
        payloads=tuple(sorted(payloads.items())),
        origins=tuple(sorted(origins.items())),
        transcript_payload=transcript_payload,
        transcript_name=transcript_name,
        filesystem_policy_sha256=policy_sha256,
        source_schema_sha256=schema_sha256,
        allowed_import_roots_sha256=(
            _allowed_import_roots_sha256(payloads)
        ),
    )
    return replace(
        candidate,
        capture=capture,
        historical_transcript_complete=transcript_complete,
        historical_transcript_failure=transcript_failure,
        historical_source_boundary=(
            candidate.retained_historical_phase and transcript_complete
        ),
    )


def _source_filesystem_boundary_reason(
    payloads: Mapping[str, bytes],
) -> str | None:
    """Apply the clean-room import/filesystem policy to exact source bytes.

    A certified ``solve(env)`` receives its Arena object from the host and has
    no raw-Arena capability.  In particular, retained acquisition source may
    not recover one by importing a host path.  This check is separate from the
    semantic taint scanner and must run before any retained source executes.
    """
    try:
        _source_schema_sha256()
    except CertificationError as exc:
        return f"source_schema_violation: {exc}"
    for name, payload in sorted(payloads.items()):
        try:
            source = payload.decode("utf-8")
        except UnicodeError as exc:
            return (
                f"non_utf8_source in {name}: executable source is not UTF-8 "
                f"({type(exc).__name__})"
            )
        reason = Boundary.first_reason(Boundary.scan_python_source(
            source,
            logical_path=name,
            arena_module_root=None,
        ))
        if reason:
            return reason
    try:
        SourceSchema.validate_source_payloads(payloads)
    except SourceSchema.SourceSchemaError as exc:
        return f"source_schema_violation: {exc}"
    return None


def _source_taint_reason(candidate: SourceCandidate) -> str | None:
    frozen = _capture_candidate(candidate)
    assert frozen.capture is not None
    payloads, origins = _source_payloads(frozen)
    reason = _source_filesystem_boundary_reason(payloads)
    if reason:
        return reason
    with tempfile.TemporaryDirectory(
        prefix=(
            f"arc_agi3_scan_{candidate.game}_L{candidate.level:02d}_"
        )
    ) as temporary:
        snapshot_root = Path(temporary)
        for name, payload in sorted(payloads.items()):
            if origins[name] == "harness:PERCEPTION_SEED":
                continue
            path = snapshot_root / name
            _write_new_bytes(path, payload)
            reason = gkm_legs._file_taint_reason(str(path), name)
            if reason:
                return reason
        if frozen.capture.transcript_payload is not None:
            transcript_name = frozen.capture.transcript_name
            assert transcript_name is not None
            path = snapshot_root / transcript_name
            _write_new_bytes(path, frozen.capture.transcript_payload)
            return gkm_legs._file_taint_reason(
                str(path), transcript_name
            )
        return None


def _normalized_actions(actions: Iterable[Any]) -> list[Any]:
    normalized: list[Any] = []
    for action in actions:
        if isinstance(action, tuple):
            action = list(action)
        if not release_gate._valid_action(action):
            raise CertificationError(f"invalid replay action: {action!r}")
        normalized.append(action)
    return normalized


def _source_replay(
    candidate: SourceCandidate,
    *,
    time_cap: int,
) -> list[Any]:
    frozen = _capture_candidate(candidate)
    payloads, _origins = _source_payloads(frozen)
    boundary_reason = _source_filesystem_boundary_reason(payloads)
    if boundary_reason:
        raise CertificationError(
            "source filesystem boundary violation before replay: "
            f"{boundary_reason}"
        )
    with tempfile.TemporaryDirectory(
        prefix=f"arc_agi3_certify_{candidate.game}_L{candidate.level:02d}_"
    ) as temporary:
        workspace = Path(temporary)
        for name, payload in payloads.items():
            _write_new_bytes(workspace / name, payload)
        reached, path, error = gkm_legs.run_solve_file(
            candidate.game,
            str(workspace / "solve.py"),
            time_cap=time_cap,
            resume_checkpoint=False,
        )
        if error or reached != candidate.level or not path:
            raise CertificationError(
                f"source replay failed for {candidate.game} L{candidate.level}: "
                f"reached={reached!r} error={error!r}"
            )
        exact = gkm_legs.exact_level_boundary(
            candidate.game, path, candidate.level
        )
        if exact is None:
            raise CertificationError(
                f"source missed exact boundary for "
                f"{candidate.game} L{candidate.level}"
            )
        normalized = _normalized_actions(exact)
        if len(normalized) > release_gate.MAX_REPLAY_ACTIONS:
            raise CertificationError(
                f"source path exceeds {release_gate.MAX_REPLAY_ACTIONS} actions "
                f"for {candidate.game} L{candidate.level}"
            )
        if not gkm_arena.validate(
            candidate.game, normalized, candidate.level
        ):
            raise CertificationError(
                f"source exact path failed validation for "
                f"{candidate.game} L{candidate.level}"
            )
        return normalized


def _path_replay(game: str, level: int, path: Sequence[Any]) -> None:
    reached, observed, error = gkm_legs._run_candidate_replay(
        game, list(path)
    )
    normalized = _normalized_actions(observed)
    if (
        error
        or reached != level
        or normalized != list(path)
        or not gkm_arena.validate(game, normalized, level)
    ):
        raise CertificationError(
            f"independent path replay failed for {game} L{level}: "
            f"reached={reached!r} error={error!r}"
        )


def _canonical_records(
    source_root: Path,
    *,
    game: str,
    target: int,
) -> list[dict[str, Any]]:
    checkpoint = _read_json(source_root / f"{game}_legs" / "checkpoint.json")
    if (
        checkpoint.get("game") != game
        or checkpoint.get("reached") != target
        or checkpoint.get("validated") is not True
        or not isinstance(checkpoint.get("records"), list)
        or len(checkpoint["records"]) != target
    ):
        raise CertificationError(
            f"canonical {game} checkpoint is not complete at {target}"
        )
    records: list[dict[str, Any]] = []
    for expected, row in enumerate(checkpoint["records"], start=1):
        if (
            not isinstance(row, dict)
            or row.get("level") != expected
            or row.get("reached") is not True
            or not isinstance(row.get("marginal_C"), int)
            or isinstance(row.get("marginal_C"), bool)
            or row["marginal_C"] < 0
        ):
            raise CertificationError(
                f"invalid canonical marginal record at {game} L{expected}"
            )
        records.append({
            "level": expected,
            "marginal_C": row["marginal_C"],
            "reached": True,
        })
    return records


def _claimed_source_inventory(
    source_root: Path,
    authoritative_inventory: Mapping[str, int],
) -> dict[str, int]:
    """Derive claimed levels from canonical acquisition checkpoints."""
    claimed: dict[str, int] = {}
    for game, authoritative_target in sorted(
        authoritative_inventory.items()
    ):
        checkpoint = _read_json(
            source_root / f"{game}_legs" / "checkpoint.json"
        )
        reached = checkpoint.get("reached")
        if (
            checkpoint.get("game") != game
            or checkpoint.get("validated") is not True
            or not isinstance(reached, int)
            or isinstance(reached, bool)
            or reached < 1
            or reached > authoritative_target
        ):
            raise CertificationError(
                f"canonical claimed frontier is invalid for {game}"
            )
        # This also validates the exact sequential marginal record shape.
        _canonical_records(source_root, game=game, target=reached)
        claimed[game] = reached
    return claimed


def _select_and_replay(
    source_root: Path,
    *,
    game: str,
    level: int,
    time_cap: int,
) -> tuple[SourceCandidate, list[Any]]:
    failures: list[str] = []
    for candidate in boundary_candidates(
        source_root, game=game, level=level
    ):
        try:
            frozen = _capture_candidate(candidate)
        except CertificationError as exc:
            failures.append(f"{candidate.origin}: capture={exc}")
            continue
        taint = _source_taint_reason(frozen)
        if taint:
            failures.append(f"{candidate.origin}: taint={taint}")
            continue
        try:
            path = _source_replay(frozen, time_cap=time_cap)
            _path_replay(game, level, path)
        except CertificationError as exc:
            failures.append(f"{candidate.origin}: {exc}")
            continue
        return frozen, path
    detail = "; ".join(failures[-5:]) if failures else "no source candidate"
    raise CertificationError(
        f"no certifiable source for {game} L{level}: {detail}"
    )


def _audit_record(
    *,
    kind: str,
    game: str,
    level: int,
    parent_checkpoint_sha256: str | None,
    checkpoint_sha256: str,
    winning_source_tree_sha256: str,
    exact_path_sha256: str,
    action_count: int,
    engine_sha256: str,
) -> dict[str, Any]:
    return {
        "schema": 1,
        "kind": kind,
        "game": game,
        "target_level": level,
        "frontier_parent_level": level - 1,
        "parent_checkpoint_sha256": parent_checkpoint_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "winning_source_tree_sha256": winning_source_tree_sha256,
        "exact_path_sha256": exact_path_sha256,
        "action_count": action_count,
        "observed_reached": level,
        "engine_sha256": engine_sha256,
        "result": "PASS",
    }


def _certify_boundary(
    *,
    source_root: Path,
    game_root: Path,
    game: str,
    level: int,
    records: Sequence[Mapping[str, Any]],
    parent_checkpoint_sha256: str | None,
    parent_manifest_sha256: str | None,
    time_cap: int,
    scanner_sha256: str,
    engine_sha256: str,
    hasher_sha256: str,
) -> tuple[str, str, dict[str, Any]]:
    candidate, exact_path = _select_and_replay(
        source_root,
        game=game,
        level=level,
        time_cap=time_cap,
    )
    candidate = _capture_candidate(candidate)
    assert candidate.capture is not None
    boundary = game_root / "promotion_evidence" / f"level_{level:02d}"
    temporary = boundary.with_name(
        f".{boundary.name}.{uuid.uuid4().hex}.staging"
    )
    if boundary.exists():
        raise CertificationError(f"boundary already exists: {boundary}")
    (temporary / "files").mkdir(parents=True)
    (temporary / "transcripts").mkdir()
    (temporary / "audits").mkdir()
    try:
        source_payloads, source_origins = _source_payloads(candidate)
        source_snapshot_hashes = {
            name: _sha256_bytes(payload)
            for name, payload in sorted(source_payloads.items())
        }
        source_snapshot_tree_sha256 = release_gate._json_sha256(
            source_snapshot_hashes
        )
        boundary_reason = _source_filesystem_boundary_reason(source_payloads)
        if boundary_reason:
            raise CertificationError(
                f"staged source filesystem boundary at {game} L{level}: "
                f"{boundary_reason}"
            )
        for name, payload in source_payloads.items():
            _write_new_bytes(temporary / "files" / name, payload)

        checkpoint = {
            "game": game,
            "reached": level,
            "total_marginal_C": sum(
                int(row["marginal_C"]) for row in records[:level]
            ),
            "records": [dict(row) for row in records[:level]],
            "final_path": exact_path,
            "validated": True,
        }
        _write_new_json(temporary / "files" / "checkpoint.json", checkpoint)
        provenance = {
            "schema": SCHEMA,
            "kind": "boundary_certification_provenance",
            "game": game,
            "level": level,
            "source_kind": candidate.kind,
            "source_phase": candidate.phase,
            "source_origin": _relative_to(candidate.origin, source_root),
            "source_file_origins": {
                name: (
                    origin
                    if origin == "harness:PERCEPTION_SEED"
                    else _relative_to(Path(origin), source_root)
                )
                for name, origin in sorted(source_origins.items())
            },
            "historical_source_boundary":
                candidate.historical_source_boundary,
            "retained_historical_phase":
                candidate.retained_historical_phase,
            "historical_transcript_complete":
                candidate.historical_transcript_complete,
            "historical_transcript_failure":
                candidate.historical_transcript_failure,
            "deterministic_reconstruction":
                candidate.deterministic_reconstruction,
            "deterministic_reconstruction_basis": (
                "canonical_exact_action_boundary"
                if candidate.kind == "exact_path_reconstruction"
                else (
                    "retained_deterministic_source"
                    if candidate.deterministic_reconstruction
                    else None
                )
            ),
            "dependency_reconstructed_from_harness_seed": (
                "harness:PERCEPTION_SEED" in source_origins.values()
            ),
            "source_snapshot_files_sha256": source_snapshot_hashes,
            "source_snapshot_tree_sha256": source_snapshot_tree_sha256,
            "source_transcript_snapshot_sha256": (
                _sha256_bytes(candidate.capture.transcript_payload)
                if candidate.capture.transcript_payload is not None
                else None
            ),
            "filesystem_boundary_policy_sha256": (
                candidate.capture.filesystem_policy_sha256
            ),
            "source_schema": SourceSchema.SCHEMA,
            "source_schema_sha256": (
                candidate.capture.source_schema_sha256
            ),
            "pinned_numpy_version": SourceSchema.PINNED_NUMPY_VERSION,
            "allowed_import_roots_sha256": (
                candidate.capture.allowed_import_roots_sha256
            ),
            "checkpoint_reconstructed_by_fresh_replay": True,
            "posthoc_acquisition_marginal_admissible":
                (
                    candidate.historical_source_boundary
                    and "harness:PERCEPTION_SEED"
                    not in source_origins.values()
                ),
        }
        _write_new_json(
            temporary / "files" / "provenance.json", provenance
        )

        promoted = {
            path.name: _sha256_file(path)
            for path in sorted((temporary / "files").iterdir())
        }
        if candidate.capture.transcript_payload is not None:
            # Preserve the scanner's evidence type when copying a native Codex
            # transcript.  ``gkm_legs._file_taint_reason`` deliberately scans
            # only agent-authored commands in ``proposer_last.log``/JSONL and
            # not command output: a traceback from an allowed public
            # ``env.clone()`` call may mention private harness fields.  A
            # hash-stamped generic ``*.log`` name would silently change that
            # interpretation and create a false taint finding after an
            # otherwise byte-identical copy.  Each boundary has its own
            # transcript directory, so the canonical name cannot collide; its
            # content hash is already sealed in the manifest and taint audit.
            transcript_source_name = candidate.capture.transcript_name
            assert transcript_source_name is not None
            transcript_sha256 = _sha256_bytes(
                candidate.capture.transcript_payload
            )
            transcript_suffix = Path(transcript_source_name).suffix
            if transcript_source_name == "proposer_last.log":
                transcript_name = "proposer_last.log"
            elif transcript_suffix == ".jsonl":
                transcript_name = (
                    f"{game}_L{level:02d}_source_"
                    f"{transcript_sha256[:12]}.jsonl"
                )
            else:
                transcript_name = (
                    f"{game}_L{level:02d}_source_"
                    f"{transcript_sha256[:12]}"
                    f"{transcript_suffix or '.log'}"
                )
            _write_new_bytes(
                temporary / "transcripts" / transcript_name,
                candidate.capture.transcript_payload,
            )
        winning_hashes = {
            name: promoted[name] for name in sorted(source_payloads)
        }
        winning_source_tree_sha256 = release_gate._json_sha256(
            winning_hashes
        )
        if (
            winning_hashes != source_snapshot_hashes
            or winning_source_tree_sha256 != source_snapshot_tree_sha256
        ):
            raise CertificationError(
                f"source snapshot identity drift at {game} L{level}"
            )
        exact_path_sha256 = release_gate._json_sha256(exact_path)
        certification_transcript = {
            "schema": SCHEMA,
            "kind": "host_boundary_certification_transcript",
            "game": game,
            "level": level,
            "source_kind": candidate.kind,
            "source_phase": candidate.phase,
            "source_origin": _relative_to(candidate.origin, source_root),
            "historical_source_boundary":
                candidate.historical_source_boundary,
            "retained_historical_phase":
                candidate.retained_historical_phase,
            "historical_transcript_complete":
                candidate.historical_transcript_complete,
            "historical_transcript_failure":
                candidate.historical_transcript_failure,
            "winning_source_files_sha256": winning_hashes,
            "winning_source_tree_sha256": winning_source_tree_sha256,
            "source_snapshot_tree_sha256": source_snapshot_tree_sha256,
            "source_transcript_snapshot_sha256": (
                _sha256_bytes(candidate.capture.transcript_payload)
                if candidate.capture.transcript_payload is not None
                else None
            ),
            "filesystem_boundary_policy_sha256": (
                candidate.capture.filesystem_policy_sha256
            ),
            "source_schema": SourceSchema.SCHEMA,
            "source_schema_sha256": (
                candidate.capture.source_schema_sha256
            ),
            "pinned_numpy_version": SourceSchema.PINNED_NUMPY_VERSION,
            "allowed_import_roots_sha256": (
                candidate.capture.allowed_import_roots_sha256
            ),
            "exact_path_sha256": exact_path_sha256,
            "action_count": len(exact_path),
            "source_from_zero_replay": "PASS",
            "path_from_zero_replay": "PASS",
            "action_protocol_runtime_enforcement":
                "shared_violation_latch_across_root_and_clones",
            "source_action_protocol": "PASS",
            "path_action_protocol": "PASS",
            "original_source_transcript_available":
                candidate.capture.transcript_payload is not None,
        }
        _write_new_json(
            temporary / "transcripts" / "certification.json",
            certification_transcript,
        )
        transcripts = {
            f"transcripts/{path.name}": _sha256_file(path)
            for path in sorted((temporary / "transcripts").iterdir())
        }
        primary_checked = {
            **{f"files/{name}": digest
               for name, digest in promoted.items()},
            **transcripts,
        }
        for relative in sorted(primary_checked):
            reason = gkm_legs._file_taint_reason(
                str(temporary / relative), relative
            )
            if reason:
                raise CertificationError(
                    f"staged taint at {game} L{level}: {reason}"
                )

        checkpoint_sha256 = promoted["checkpoint.json"]
        taint_audit = {
            "schema": 1,
            "kind": "taint_audit",
            "game": game,
            "level": level,
            "scanner_sha256": scanner_sha256,
            "checked_files_sha256": primary_checked,
            "verdict": "PASS",
            "findings": [],
        }
        path_audit = _audit_record(
            kind="path_replay",
            game=game,
            level=level,
            parent_checkpoint_sha256=parent_checkpoint_sha256,
            checkpoint_sha256=checkpoint_sha256,
            winning_source_tree_sha256=winning_source_tree_sha256,
            exact_path_sha256=exact_path_sha256,
            action_count=len(exact_path),
            engine_sha256=engine_sha256,
        )
        source_audit = _audit_record(
            kind="source_replay",
            game=game,
            level=level,
            parent_checkpoint_sha256=parent_checkpoint_sha256,
            checkpoint_sha256=checkpoint_sha256,
            winning_source_tree_sha256=winning_source_tree_sha256,
            exact_path_sha256=exact_path_sha256,
            action_count=len(exact_path),
            engine_sha256=engine_sha256,
        )
        action_protocol_audit = {
            "schema": 1,
            "kind": "action_protocol_audit",
            "game": game,
            "target_level": level,
            "checkpoint_sha256": checkpoint_sha256,
            "exact_path_sha256": exact_path_sha256,
            "action_count": len(exact_path),
            "runtime_enforcement":
                "shared_violation_latch_across_root_and_clones",
            "source_protocol_latch": "PASS",
            "path_protocol_latch": "PASS",
            "engine_sha256": engine_sha256,
            "result": "PASS",
        }
        audit_values = {
            "taint": taint_audit,
            "action_protocol": action_protocol_audit,
            "path_replay": path_audit,
            "source_replay": source_audit,
        }
        audit_hashes: dict[str, str] = {}
        for name, value in audit_values.items():
            relative = release_gate.AUDIT_PATHS[name]
            path = temporary / relative
            _write_new_json(path, value)
            audit_hashes[name] = _sha256_file(path)

        hash_checked = {
            **primary_checked,
            release_gate.AUDIT_PATHS["taint"]: audit_hashes["taint"],
            release_gate.AUDIT_PATHS["action_protocol"]:
                audit_hashes["action_protocol"],
            release_gate.AUDIT_PATHS["path_replay"]:
                audit_hashes["path_replay"],
            release_gate.AUDIT_PATHS["source_replay"]:
                audit_hashes["source_replay"],
        }
        hash_audit = {
            "schema": 1,
            "kind": "hash_audit",
            "game": game,
            "level": level,
            "hasher_sha256": hasher_sha256,
            "checked_files_sha256": hash_checked,
            "result": "PASS",
        }
        hash_path = temporary / release_gate.AUDIT_PATHS["hash"]
        _write_new_json(hash_path, hash_audit)
        audit_hashes["hash"] = _sha256_file(hash_path)

        parent_manifest: dict[str, str] | None
        if level == 1:
            parent_manifest = None
        else:
            if parent_manifest_sha256 is None:
                raise CertificationError(
                    f"missing parent manifest at {game} L{level}"
                )
            parent_manifest = {
                "path": (
                    f"promotion_evidence/level_{level - 1:02d}/"
                    "manifest.json"
                ),
                "sha256": parent_manifest_sha256,
            }
        manifest = {
            "schema": release_gate.BOUNDARY_MANIFEST_SCHEMA,
            "game": game,
            "level": level,
            "frontier": {
                "parent_level": level - 1,
                "target_level": level,
                "parent_checkpoint_sha256": parent_checkpoint_sha256,
            },
            "parent_manifest": parent_manifest,
            "promoted_files_sha256": promoted,
            "winning_source_files": sorted(source_payloads),
            "transcripts": [
                {"path": path, "sha256": digest}
                for path, digest in transcripts.items()
            ],
            "audits": {
                name: {
                    "path": release_gate.AUDIT_PATHS[name],
                    "sha256": audit_hashes[name],
                }
                for name in release_gate.AUDIT_PATHS
            },
        }
        _write_new_json(temporary / "manifest.json", manifest)
        manifest_sha256 = _sha256_file(temporary / "manifest.json")
        os.rename(temporary, boundary)
        return checkpoint_sha256, manifest_sha256, provenance
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def _certify_game(
    *,
    source_root: Path,
    stage_root: Path,
    game: str,
    target: int,
    time_cap: int,
    scanner_sha256: str,
    engine_sha256: str,
    hasher_sha256: str,
) -> list[dict[str, Any]]:
    records = _canonical_records(
        source_root, game=game, target=target
    )
    temporary = stage_root / f".{game}_legs.{uuid.uuid4().hex}.staging"
    game_root = stage_root / f"{game}_legs"
    if game_root.exists():
        raise CertificationError(f"game already exists in stage: {game}")
    (temporary / "promotion_evidence").mkdir(parents=True)
    provenance: list[dict[str, Any]] = []
    parent_checkpoint: str | None = None
    parent_manifest: str | None = None
    try:
        for level in range(1, target + 1):
            parent_checkpoint, parent_manifest, record = _certify_boundary(
                source_root=source_root,
                game_root=temporary,
                game=game,
                level=level,
                records=records,
                parent_checkpoint_sha256=parent_checkpoint,
                parent_manifest_sha256=parent_manifest,
                time_cap=time_cap,
                scanner_sha256=scanner_sha256,
                engine_sha256=engine_sha256,
                hasher_sha256=hasher_sha256,
            )
            provenance.append(record)
        final_files = (
            temporary / "promotion_evidence"
            / f"level_{target:02d}" / "files"
        )
        final_manifest = _read_json(
            temporary / "promotion_evidence"
            / f"level_{target:02d}" / "manifest.json"
        )
        final_source_names = final_manifest.get("winning_source_files")
        if (
            not isinstance(final_source_names, list)
            or not all(isinstance(name, str) for name in final_source_names)
        ):
            raise CertificationError(
                f"invalid final source inventory for {game}"
            )
        for name in (*final_source_names, "checkpoint.json"):
            _copy_regular(final_files / name, temporary / name)
        os.rename(temporary, game_root)
        return provenance
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def plan_migration(
    *,
    source_root: Path,
    environments_root: Path,
) -> dict[str, Any]:
    inventory, _ = release_gate._discover_inventory(environments_root)
    rows: list[dict[str, Any]] = []
    counts = {
        "boundaries": 0,
        "with_candidate": 0,
        "legacy_first": 0,
        "historical_wip_first": 0,
        "deterministic_reconstruction_first": 0,
        "missing_candidate": 0,
    }
    for game, target in sorted(inventory.items()):
        for level in range(1, target + 1):
            candidates = boundary_candidates(
                source_root,
                game=game,
                level=level,
                include_exact_path_reconstruction=False,
            )
            if not candidates:
                candidates = boundary_candidates(
                    source_root,
                    game=game,
                    level=level,
                )
            first = candidates[0] if candidates else None
            counts["boundaries"] += 1
            if first is None:
                counts["missing_candidate"] += 1
            else:
                counts["with_candidate"] += 1
                if first.kind == "legacy_promotion":
                    counts["legacy_first"] += 1
                elif first.deterministic_reconstruction:
                    counts["deterministic_reconstruction_first"] += 1
                elif first.historical_source_boundary:
                    counts["historical_wip_first"] += 1
            rows.append({
                "game": game,
                "level": level,
                "candidate_count": len(candidates),
                "selected_kind": first.kind if first else None,
                "selected_phase": first.phase if first else None,
                "historical_source_boundary": (
                    first.historical_source_boundary if first else None
                ),
                "deterministic_reconstruction": (
                    first.deterministic_reconstruction if first else None
                ),
                "origin": (
                    _relative_to(first.origin, source_root)
                    if first else None
                ),
            })
    return {
        "schema": SCHEMA,
        "status": (
            "PASS" if counts["missing_candidate"] == 0 else "INCOMPLETE"
        ),
        "inventory": inventory,
        "summary": counts,
        "boundaries": rows,
    }


def plan_partial_migration(
    *,
    source_root: Path,
    environments_root: Path,
    expected_claimed_levels: int,
) -> dict[str, Any]:
    """Plan only the claimed prefix while retaining the full 183 inventory."""
    if (
        not isinstance(expected_claimed_levels, int)
        or isinstance(expected_claimed_levels, bool)
        or expected_claimed_levels <= 0
        or expected_claimed_levels >= release_gate.EXPECTED_LEVELS
    ):
        raise CertificationError(
            "partial expected-claimed-levels must be within 1..182"
        )
    inventory, _ = release_gate._discover_inventory(environments_root)
    claimed = _claimed_source_inventory(source_root, inventory)
    claimed_total = sum(claimed.values())
    if claimed_total != expected_claimed_levels:
        raise CertificationError(
            "partial claimed frontier count mismatch; "
            f"expected {expected_claimed_levels}, found {claimed_total}"
        )

    rows: list[dict[str, Any]] = []
    counts = {
        "boundaries": claimed_total,
        "with_candidate": 0,
        "historical_source_boundary": 0,
        "retrospective_certification": 0,
        "missing_candidate": 0,
    }
    for game, target in sorted(claimed.items()):
        for level in range(1, target + 1):
            candidates = boundary_candidates(
                source_root,
                game=game,
                level=level,
                include_exact_path_reconstruction=False,
            )
            if not candidates:
                candidates = boundary_candidates(
                    source_root,
                    game=game,
                    level=level,
                )
            first = candidates[0] if candidates else None
            if first is None:
                counts["missing_candidate"] += 1
            else:
                counts["with_candidate"] += 1
                if first.historical_source_boundary:
                    counts["historical_source_boundary"] += 1
                else:
                    counts["retrospective_certification"] += 1
            rows.append({
                "game": game,
                "level": level,
                "candidate_count": len(candidates),
                "selected_kind": first.kind if first else None,
                "selected_phase": first.phase if first else None,
                "historical_source_boundary": (
                    first.historical_source_boundary if first else None
                ),
                "deterministic_reconstruction": (
                    first.deterministic_reconstruction if first else None
                ),
                "origin": (
                    _relative_to(first.origin, source_root)
                    if first else None
                ),
            })
    unclaimed = release_gate._unclaimed_boundaries(inventory, claimed)
    return {
        "schema": SCHEMA,
        "status": (
            "PASS" if counts["missing_candidate"] == 0 else "INCOMPLETE"
        ),
        "authoritative_inventory": inventory,
        "claimed_inventory": claimed,
        "claimed_levels": claimed_total,
        "authoritative_levels": sum(inventory.values()),
        "unclaimed_boundaries": unclaimed,
        "summary": counts,
        "boundaries": rows,
    }


def build_release_tree(
    *,
    source_root: Path,
    output_root: Path,
    environments_root: Path,
    time_cap: int,
) -> dict[str, Any]:
    source_root = source_root.resolve()
    output_root = output_root.resolve()
    environments_root = environments_root.resolve()
    if output_root.exists() or output_root.is_symlink():
        raise CertificationError(
            f"refusing to overwrite release output: {output_root}"
        )
    stage_root = output_root.with_name(
        f".{output_root.name}.{uuid.uuid4().hex}.building"
    )
    if stage_root.exists():
        raise CertificationError(f"staging collision: {stage_root}")

    inventory, _ = release_gate._discover_inventory(environments_root)
    plan = plan_migration(
        source_root=source_root,
        environments_root=environments_root,
    )
    if plan["status"] != "PASS":
        missing = [
            (row["game"], row["level"])
            for row in plan["boundaries"]
            if row["candidate_count"] == 0
        ]
        raise CertificationError(
            f"release source selection is incomplete: {missing}"
        )

    module_root = Path(__file__).resolve().parent
    scanner_sha256 = _sha256_file(module_root / "gkm_legs.py")
    engine_sha256 = _sha256_file(module_root / "gkm_arena.py")
    hasher_sha256 = _sha256_file(Path(__file__).resolve())
    allowed_tool_hashes = frozenset({
        scanner_sha256, engine_sha256, hasher_sha256,
    })

    stage_root.mkdir(parents=True)
    provenance: dict[str, list[dict[str, Any]]] = {}
    try:
        for game, target in sorted(inventory.items()):
            provenance[game] = _certify_game(
                source_root=source_root,
                stage_root=stage_root,
                game=game,
                target=target,
                time_cap=time_cap,
                scanner_sha256=scanner_sha256,
                engine_sha256=engine_sha256,
                hasher_sha256=hasher_sha256,
            )

        diagnostic = release_gate.diagnose_release_migration(
            canonical_root=stage_root,
            environments_root=environments_root,
        )
        if diagnostic["status"] != "PASS":
            raise CertificationError(
                "schema-v2 migration diagnostic rejected the staged tree"
            )
        release_gate._validate_canonical(
            stage_root,
            inventory,
            allowed_tool_hashes=allowed_tool_hashes,
        )
        os.rename(stage_root, output_root)
    except Exception:
        # Preserve a failed build for forensic inspection; never make it appear
        # under the requested release path.
        raise

    return {
        "schema": SCHEMA,
        "status": "PASS",
        "source_root": str(source_root),
        "output_root": str(output_root),
        "games": len(inventory),
        "levels": sum(inventory.values()),
        "deterministic_reconstructions": [
            {"game": game, "level": row["level"], "source_phase": row["source_phase"]}
            for game, rows in provenance.items()
            for row in rows
            if row["deterministic_reconstruction"]
        ],
    }


def build_partial_release_tree(
    *,
    source_root: Path,
    output_root: Path,
    environments_root: Path,
    time_cap: int,
    expected_claimed_levels: int,
) -> dict[str, Any]:
    """Build a separate exact prefix freeze; never edit the acquisition tree."""
    source_root = source_root.resolve()
    output_root = output_root.resolve()
    environments_root = environments_root.resolve()
    if output_root.exists() or output_root.is_symlink():
        raise CertificationError(
            f"refusing to overwrite release output: {output_root}"
        )
    stage_root = output_root.with_name(
        f".{output_root.name}.{uuid.uuid4().hex}.building"
    )
    if stage_root.exists():
        raise CertificationError(f"staging collision: {stage_root}")

    plan = plan_partial_migration(
        source_root=source_root,
        environments_root=environments_root,
        expected_claimed_levels=expected_claimed_levels,
    )
    if plan["status"] != "PASS":
        missing = [
            (row["game"], row["level"])
            for row in plan["boundaries"]
            if row["candidate_count"] == 0
        ]
        raise CertificationError(
            f"partial release source selection is incomplete: {missing}"
        )
    claimed = dict(plan["claimed_inventory"])

    module_root = Path(__file__).resolve().parent
    scanner_sha256 = _sha256_file(module_root / "gkm_legs.py")
    engine_sha256 = _sha256_file(module_root / "gkm_arena.py")
    hasher_sha256 = _sha256_file(Path(__file__).resolve())
    allowed_tool_hashes = frozenset({
        scanner_sha256,
        engine_sha256,
        hasher_sha256,
    })

    stage_root.mkdir(parents=True)
    provenance: dict[str, list[dict[str, Any]]] = {}
    try:
        for game, target in sorted(claimed.items()):
            provenance[game] = _certify_game(
                source_root=source_root,
                stage_root=stage_root,
                game=game,
                target=target,
                time_cap=time_cap,
                scanner_sha256=scanner_sha256,
                engine_sha256=engine_sha256,
                hasher_sha256=hasher_sha256,
            )
        release_gate._validate_canonical(
            stage_root,
            claimed,
            allowed_tool_hashes=allowed_tool_hashes,
        )
        os.rename(stage_root, output_root)
    except Exception:
        # A failed staging tree is not a release and is never renamed into the
        # requested output path.
        raise

    return {
        "schema": SCHEMA,
        "status": "PASS",
        "kind": "partial_campaign_freeze",
        "source_root": str(source_root),
        "output_root": str(output_root),
        "games": len(claimed),
        "claimed_levels": sum(claimed.values()),
        "authoritative_levels": plan["authoritative_levels"],
        "unclaimed_boundaries": plan["unclaimed_boundaries"],
        "historical_source_boundaries": [
            {"game": game, "level": row["level"]}
            for game, rows in provenance.items()
            for row in rows
            if row["historical_source_boundary"]
        ],
        "retrospective_certifications": [
            {
                "game": game,
                "level": row["level"],
                "source_phase": row["source_phase"],
            }
            for game, rows in provenance.items()
            for row in rows
            if not row["historical_source_boundary"]
        ],
        "deterministic_reconstructions": [
            {
                "game": game,
                "level": row["level"],
                "source_phase": row["source_phase"],
            }
            for game, rows in provenance.items()
            for row in rows
            if row["deterministic_reconstruction"]
        ],
    }


def _default_source_root() -> Path:
    return Path(__file__).resolve().parent / "agent_solutions"


def _default_environments_root() -> Path:
    return Path(__file__).resolve().parents[2] / "environment_files"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root", type=Path, default=_default_source_root()
    )
    parser.add_argument(
        "--environments-root",
        type=Path,
        default=_default_environments_root(),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("plan")
    partial_plan = subparsers.add_parser("plan-partial")
    partial_plan.add_argument(
        "--expected-claimed-levels", type=int, required=True
    )
    build = subparsers.add_parser("build")
    build.add_argument("--output-root", type=Path, required=True)
    build.add_argument("--time-cap", type=int, default=600)
    partial_build = subparsers.add_parser("build-partial")
    partial_build.add_argument("--output-root", type=Path, required=True)
    partial_build.add_argument("--time-cap", type=int, default=600)
    partial_build.add_argument(
        "--expected-claimed-levels", type=int, required=True
    )
    args = parser.parse_args(argv)

    try:
        if args.command == "plan":
            result = plan_migration(
                source_root=args.source_root,
                environments_root=args.environments_root,
            )
            print(json.dumps(result, sort_keys=True))
            return 0 if result["status"] == "PASS" else 1
        if args.command == "plan-partial":
            result = plan_partial_migration(
                source_root=args.source_root,
                environments_root=args.environments_root,
                expected_claimed_levels=args.expected_claimed_levels,
            )
            print(json.dumps(result, sort_keys=True))
            return 0 if result["status"] == "PASS" else 1
        if args.time_cap <= 0:
            raise CertificationError("--time-cap must be positive")
        if args.command == "build":
            result = build_release_tree(
                source_root=args.source_root,
                output_root=args.output_root,
                environments_root=args.environments_root,
                time_cap=args.time_cap,
            )
        else:
            result = build_partial_release_tree(
                source_root=args.source_root,
                output_root=args.output_root,
                environments_root=args.environments_root,
                time_cap=args.time_cap,
                expected_claimed_levels=args.expected_claimed_levels,
            )
        print(json.dumps(result, sort_keys=True))
        return 0
    except (CertificationError, release_gate.ReleaseGateError) as exc:
        print(json.dumps({
            "schema": SCHEMA,
            "status": "FAIL",
            "error": str(exc),
        }, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
