#!/usr/bin/env python3
"""Production composition for the isolated ARC-AGI-3 contiguous campaign.

This module is the trusted join between the durable scheduler, the observed
Docker backend, isolated source replay, schema-v2 boundary certification, and
atomic per-game publication.  It intentionally contains no launch-ready
constant: launch authority is issued only by the independent contiguous
conformance ``verify`` path.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import dataclasses
import json
import math
import os
import re
import secrets
import shutil
import stat
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Literal, Mapping, Protocol, Sequence

# The documented production entry point is direct script execution from the
# repository checkout.  Add only that content-derived repository root so
# package imports work without trusting ambient PYTHONPATH entries.
if __package__ in {None, ""}:
    _DIRECT_SCRIPT = Path(__file__).resolve()
    _DIRECT_CONTROL_ROOT = str(_DIRECT_SCRIPT.parent)
    _DIRECT_REPOSITORY_ROOT = str(_DIRECT_SCRIPT.parents[2])
    if _DIRECT_CONTROL_ROOT not in sys.path:
        sys.path.insert(0, _DIRECT_CONTROL_ROOT)
    if _DIRECT_REPOSITORY_ROOT not in sys.path:
        # Keep the control-bound adjacent modules ahead of the package root;
        # the root is added only to resolve the ``arc`` package itself.
        sys.path.insert(1, _DIRECT_REPOSITORY_ROOT)

import arc_agi3_codex_app_server_transport as Transport
import arc_agi3_contiguous_runner as Runner
import arc_agi3_contiguous_scheduler as Scheduler
import arc_agi3_contiguous_supervisor as Supervisor
import arc_agi3_contiguous_taint as Taint
import arc_agi3_python_runtime_manifest as RuntimeManifest
import arc_agi3_release_gate as Release
import arc_agi3_source_schema as SourceSchema
import gkm_arena
import gkm_legs


SCHEMA = 1
POINTER_SCHEMA = 1
HOST_RECEIPT_SCHEMA = 2
MAX_JSON_BYTES = 32 * 1024 * 1024
MAX_TRANSCRIPT_BYTES = 64 * 1024 * 1024
MAX_JOURNAL_PREFIX_BYTES = 24 * 1024 * 1024
MAX_REPLAY_SECONDS = 20 * 60
MAX_OPERATOR_CONFIG_BYTES = 4 * 1024 * 1024
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
VERSION_RE = re.compile(r"^[0-9a-f]{32}$")
IMAGE_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
ENVIRONMENT_NAME_RE = re.compile(r"^[A-Z][A-Z0-9_]{0,127}$")
POINTER_NAME = "current.json"
VERSIONS_NAME = "versions"
STAGING_NAME = "staging"
QUARANTINE_NAME = "quarantine"
INTENTS_NAME = "intents"
HOST_RECEIPT_NAME = Supervisor.HOST_RECEIPT_NAME
WINNING_SOURCE_NAME = Supervisor.WINNING_SOURCE_NAME
CANDIDATE_NAME = Supervisor.CANDIDATE_NAME
UNIFIED_AUDIT_SCHEMA = 1
CANONICAL_TERMINAL_CONDITION = "complete"
SELECTIVE_TERMINAL_CONDITION = "selective_complete"
TERMINAL_PROMOTION_REPLAY_AUDIT_NAME = "promotion_replay.json"
AUXILIARY_DRIVER_PROTOCOL_TEXT = """\
ARC-AGI-3 contiguous auxiliary backend protocol v1
The host invokes one digest-pinned driver with exactly --configuration,
--request, and --response. The scheduler-authenticated request, never argv or
operator input, selects game, frontier, effort, round, and specialization.
Every output remains immutable-private-copy and quarantine-only. Diagnosis and
specialist evidence must include a Socratic challenge and may be admitted only
by the separately attested host replay, taint, and provenance gate. Driver
stdout and stderr are immutable host audit surfaces and are never copied into
proposer-visible context. Every driver path is component-walked beneath a
pinned assignment-root descriptor; every receipt is one descriptor-stable
canonical read, including restart rebinding against journaled digests. Stream
bytes must equal their reported digests and lengths, and every successful
canonical response has a durable raw-byte recovery binding before it is acted
upon.
"""
AUXILIARY_DRIVER_PROTOCOL_SHA256 = hashlib.sha256(
    AUXILIARY_DRIVER_PROTOCOL_TEXT.encode("utf-8")
).hexdigest()
MAX_AUXILIARY_DRIVER_CONTROL_BYTES = 4 * 1024 * 1024
MAX_AUXILIARY_DRIVER_RESPONSE_BYTES = 32 * 1024 * 1024
MAX_AUXILIARY_DRIVER_STREAM_BYTES = 8 * 1024 * 1024
_AUXILIARY_REASON_CODE_RE = re.compile(r"^[a-z][a-z0-9_]{0,127}$")
REPLAY_FILESYSTEM_PATH_FIELDS = frozenset({
    "arena_transcript_path",
    "worker_outcome_path",
    "stdout_path",
    "stderr_path",
})
HOST_RECEIPT_FIELDS = frozenset({
    "schema",
    "kind",
    "version",
    "campaign_id",
    "generation_id",
    "attempt_id",
    "game",
    "target_level",
    "authoritative_target",
    "parent_checkpoint_sha256",
    "candidate_manifest_sha256",
    "supervisory_handoff_sha256",
    "supervisory_native_reproduction_receipt_sha256",
    "candidate_path_sha256",
    "probe_isolation_mode",
    "probe_isolation_evidence_sha256",
    "probe_result_authority",
    "checkpoint_sha256",
    "exact_path",
    "exact_path_sha256",
    "schema_v2_manifest_path",
    "schema_v2_manifest_sha256",
    "schema_v2_audits_sha256",
    "winning_source_tree_sha256",
    "release_source_tree_sha256",
    "source_description_bytes_before",
    "source_description_bytes_after",
    "source_description_metric",
    "source_description_bytes_before_by_file",
    "source_description_bytes_after_by_file",
    "same_size_rewrite_novelty",
    "marginal_C",
    "isolated_source_replay",
    "attempt_evidence_event_sha256",
    "path_replay_from_zero",
    "source_replay_from_zero",
    "taint_scan",
    "publication_subject_tree_sha256",
    "control_tools_sha256",
})


class ContiguousOrchestratorError(RuntimeError):
    """A trusted collection, replay, publication, or composition check failed."""


@dataclass(frozen=True)
class IsolatedReplayEvidence:
    """Observed result of executing candidate source in the replay container."""

    schema: int
    replay_id: str
    game: str
    target_level: int
    observed_level: int
    observed_path: tuple[Any, ...]
    exact_path: tuple[Any, ...]
    source_tree_sha256: str
    replay_image_reference: str
    replay_image_digest: str
    container_id: str
    launch_attestation_sha256: str
    running_observation_sha256: str
    arena_transcript_path: str
    arena_transcript_sha256: str
    worker_outcome_path: str
    worker_outcome_sha256: str
    stdout_path: str
    stdout_sha256: str
    stderr_path: str
    stderr_sha256: str
    teardown_proof_sha256: str


class SourceReplayExecutor(Protocol):
    """Execute untrusted winning source from public zero in OS isolation."""

    def replay_from_zero(
        self,
        *,
        spec: Runner.AttemptSpec,
        source_payloads: Mapping[str, bytes],
    ) -> IsolatedReplayEvidence:
        ...


@dataclass(frozen=True)
class AttemptEvidenceBundle:
    """Journal-authenticated collection and teardown used by publication."""

    collection: Runner.BackendCollection
    teardown: Runner.BackendTeardownProof
    collected_sequence: int
    collected_event_sha256: str
    teardown_sequence: int
    teardown_event_sha256: str
    result_sequence: int
    result_event_sha256: str
    journal_prefix: tuple[Mapping[str, Any], ...]
    journal_prefix_sha256: str
    journal_genesis_sha256: str


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
        raise ContiguousOrchestratorError(
            "value is not canonical JSON"
        ) from exc


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _json_sha256(value: object) -> str:
    return _sha256(_canonical_json(value))


def _strict_json(raw: bytes, *, label: str) -> dict[str, Any]:
    if len(raw) > MAX_JSON_BYTES:
        raise ContiguousOrchestratorError(f"{label} exceeds its byte bound")
    try:
        value = Transport.strict_json_loads(raw)
    except Exception as exc:
        raise ContiguousOrchestratorError(
            f"{label} is not strict JSON"
        ) from exc
    if not isinstance(value, dict):
        raise ContiguousOrchestratorError(f"{label} is not a JSON object")
    return value


def _require_no_live_secret(
    raw: bytes,
    *,
    secret_sentinels: Sequence[str],
    label: str,
) -> None:
    """Reject live credential generations without serializing their values."""

    for sentinel in secret_sentinels:
        try:
            encoded = sentinel.encode("utf-8")
        except UnicodeError as exc:  # pragma: no cover - constructor guards str.
            raise ContiguousOrchestratorError(
                "live credential sentinel is not UTF-8"
            ) from exc
        if encoded and encoded in raw:
            raise ContiguousOrchestratorError(
                f"{label} contains a live credential sentinel"
            )


def _read_regular(
    path: Path,
    *,
    maximum: int,
    allow_empty: bool = False,
) -> bytes:
    selected = Path(path)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(selected, flags)
    except OSError as exc:
        raise ContiguousOrchestratorError(
            f"cannot descriptor-open {selected}"
        ) from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size > maximum
            or (before.st_size == 0 and not allow_empty)
        ):
            raise ContiguousOrchestratorError(
                f"file is aliased, nonregular, empty, or oversized: {selected}"
            )
        remaining = before.st_size
        chunks: list[bytes] = []
        while remaining:
            block = os.read(descriptor, min(1024 * 1024, remaining))
            if not block:
                raise ContiguousOrchestratorError(
                    f"file changed while reading: {selected}"
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
            raise ContiguousOrchestratorError(
                f"file metadata changed while reading: {selected}"
            )
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _load_attempt_evidence(
    spec: Runner.AttemptSpec,
) -> AttemptEvidenceBundle:
    generation = Path(spec.generation_dir)
    if (
        not generation.is_absolute()
        or generation.name != spec.generation_id
        or generation.parent.name != "generations"
    ):
        raise ContiguousOrchestratorError(
            "attempt generation path is not canonical"
        )
    journal_root = generation.parent.parent / "attempt_journal"
    try:
        events = Runner.DurableAttemptJournal(journal_root).read()
    except Runner.ContiguousRunnerError as exc:
        raise ContiguousOrchestratorError(
            "attempt journal failed immutable replay"
        ) from exc
    selected_events = [
        event
        for event in events
        if event["payload"].get("attempt_id") == spec.attempt_id
        and event["kind"]
        in {"ATTEMPT_COLLECTED", "ATTEMPT_TORN_DOWN", "ATTEMPT_RESULT"}
    ]
    selected = {
        event["kind"]: event for event in selected_events
    }
    if len(selected_events) != 3 or set(selected) != {
        "ATTEMPT_COLLECTED",
        "ATTEMPT_TORN_DOWN",
        "ATTEMPT_RESULT",
    }:
        raise ContiguousOrchestratorError(
            "promotion lacks its exact collected/teardown/result journal chain"
        )
    collected_event = selected["ATTEMPT_COLLECTED"]
    teardown_event = selected["ATTEMPT_TORN_DOWN"]
    result_event = selected["ATTEMPT_RESULT"]
    if not (
        collected_event["sequence"]
        < teardown_event["sequence"]
        < result_event["sequence"]
    ):
        raise ContiguousOrchestratorError(
            "attempt evidence events are reordered"
        )
    try:
        collection = Runner._backend_collection_from_dict(
            collected_event["payload"]["collection"]
        )
        teardown = Runner._backend_teardown_from_dict(
            teardown_event["payload"]["teardown"]
        )
        result = Runner.ContiguousCampaignRunner._result_from_payload(
            result_event["payload"]
        )
    except (KeyError, Runner.ContiguousRunnerError) as exc:
        raise ContiguousOrchestratorError(
            "attempt evidence payload failed typed replay"
        ) from exc
    if (
        collection.result != result
        or result.kind != "candidate"
        or result.candidate is None
        or result.candidate.candidate_manifest_sha256
        != (
            collection.result.candidate.candidate_manifest_sha256
            if collection.result.candidate is not None
            else None
        )
        or teardown.cause != "normal_exit"
        or teardown.container_inspect_absent is not True
        or teardown.container_top_absent is not True
        or teardown.identity_query_empty is not True
        or teardown.no_descendants is not True
    ):
        raise ContiguousOrchestratorError(
            "attempt journal evidence is not one clean torn-down candidate"
        )
    # Retain the complete canonical prefix, not merely an anchored suffix.
    # A bare previous-digest assertion cannot prove campaign-chain membership
    # to an offline verifier.
    prefix = tuple(events[:result_event["sequence"]])
    prefix_raw = _canonical_json(prefix)
    if len(prefix_raw) > MAX_JOURNAL_PREFIX_BYTES:
        raise ContiguousOrchestratorError(
            "attempt journal prefix exceeds the bounded v1 evidence format"
        )
    return AttemptEvidenceBundle(
        collection=collection,
        teardown=teardown,
        collected_sequence=collected_event["sequence"],
        collected_event_sha256=collected_event["digest"],
        teardown_sequence=teardown_event["sequence"],
        teardown_event_sha256=teardown_event["digest"],
        result_sequence=result_event["sequence"],
        result_event_sha256=result_event["digest"],
        journal_prefix=prefix,
        journal_prefix_sha256=_sha256(prefix_raw),
        journal_genesis_sha256=events[0]["digest"],
    )


def _validate_retained_journal_evidence(
    value: object,
    *,
    event_hashes: Mapping[str, str],
    attempt_id: str,
    campaign_id: str,
) -> None:
    if not isinstance(value, dict):
        raise ContiguousOrchestratorError(
            "retained attempt journal evidence is not an object"
        )
    prefix = value.get("journal_prefix")
    if (
        not isinstance(prefix, list)
        or not prefix
        or len(_canonical_json(prefix)) > MAX_JOURNAL_PREFIX_BYTES
        or _json_sha256(prefix) != value.get("journal_prefix_sha256")
        or SHA256_RE.fullmatch(
            str(value.get("journal_genesis_sha256"))
        )
        is None
        or not isinstance(attempt_id, str)
        or not attempt_id
        or not isinstance(campaign_id, str)
        or not campaign_id
        or set(event_hashes) != {"collected", "teardown", "result"}
        or any(
            SHA256_RE.fullmatch(str(digest)) is None
            for digest in event_hashes.values()
        )
    ):
        raise ContiguousOrchestratorError(
            "retained attempt journal prefix identity is invalid"
        )
    prior: str | None = None
    previous_sequence = 0
    observed: dict[str, str] = {}
    observed_sequences: dict[str, int] = {}
    event_ids: set[str] = set()
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
    for event in prefix:
        if not isinstance(event, dict) or set(event) != required:
            raise ContiguousOrchestratorError(
                "retained attempt journal event schema is invalid"
            )
        body = {key: event[key] for key in required - {"digest"}}
        if (
            event.get("schema") != Runner.JOURNAL_SCHEMA
            or isinstance(event.get("schema"), bool)
            or not isinstance(event.get("sequence"), int)
            or isinstance(event.get("sequence"), bool)
            or event["sequence"] != previous_sequence + 1
            or event.get("previous_digest") != prior
            or not isinstance(event.get("payload"), dict)
            or not isinstance(event.get("event_id"), str)
            or not event["event_id"]
            or event["event_id"] in event_ids
            or not isinstance(event.get("kind"), str)
            or not event["kind"]
            or not isinstance(event.get("recorded_at"), (int, float))
            or isinstance(event.get("recorded_at"), bool)
            or not math.isfinite(float(event["recorded_at"]))
            or SHA256_RE.fullmatch(str(event.get("digest"))) is None
            or Runner.DurableAttemptJournal._event_digest(body)
            != event["digest"]
        ):
            raise ContiguousOrchestratorError(
                "retained attempt journal hash chain is invalid"
            )
        if event["payload"].get("attempt_id") == attempt_id:
            selected_kind = {
                "ATTEMPT_COLLECTED": "collected",
                "ATTEMPT_TORN_DOWN": "teardown",
                "ATTEMPT_RESULT": "result",
            }.get(event["kind"])
            if selected_kind is not None:
                if selected_kind in observed:
                    raise ContiguousOrchestratorError(
                        "retained journal segment duplicates an attempt event"
                    )
                observed[selected_kind] = event["digest"]
                observed_sequences[selected_kind] = event["sequence"]
        event_ids.add(event["event_id"])
        prior = event["digest"]
        previous_sequence = event["sequence"]
    genesis = prefix[0]
    if (
        genesis["kind"] != "GENESIS"
        or genesis["event_id"] != "campaign:genesis"
        or genesis["previous_digest"] is not None
        or genesis["digest"] != value["journal_genesis_sha256"]
        or genesis["payload"].get("campaign_id") != campaign_id
        or observed != dict(event_hashes)
        or observed_sequences
        != {
            "collected": value.get("collected_sequence"),
            "teardown": value.get("teardown_sequence"),
            "result": value.get("result_sequence"),
        }
        or not (
            observed_sequences["collected"]
            < observed_sequences["teardown"]
            < observed_sequences["result"]
        )
        or prefix[-1]["digest"] != event_hashes["result"]
        or prefix[-1]["kind"] != "ATTEMPT_RESULT"
        or prefix[-1]["payload"].get("attempt_id") != attempt_id
    ):
        raise ContiguousOrchestratorError(
            "retained journal prefix omits authenticated campaign membership"
        )


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


def _require_owned_directory(path: Path, *, label: str) -> None:
    """Reject link traversal and non-private mutable store directories."""

    selected = Path(path)
    if not selected.is_absolute():
        raise ContiguousOrchestratorError(f"{label} is not absolute")
    descriptor = os.open(
        "/",
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        for part in selected.parts[1:]:
            if part in {"", ".", ".."}:
                raise ContiguousOrchestratorError(
                    f"{label} has an unsafe path component"
                )
            next_descriptor = os.open(
                part,
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=descriptor,
            )
            os.close(descriptor)
            descriptor = next_descriptor
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or metadata.st_uid != os.getuid()
            or stat.S_IMODE(metadata.st_mode) & 0o022
        ):
            raise ContiguousOrchestratorError(
                f"{label} is not an owned non-writable-by-others directory"
            )
    except OSError as exc:
        raise ContiguousOrchestratorError(
            f"{label} traverses an alias or non-directory"
        ) from exc
    finally:
        os.close(descriptor)


def _ensure_private_directory(path: Path, *, label: str) -> None:
    try:
        os.mkdir(path, 0o700)
    except FileExistsError:
        pass
    try:
        metadata = os.lstat(path)
    except OSError as exc:
        raise ContiguousOrchestratorError(
            f"{label} cannot be inspected"
        ) from exc
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or metadata.st_uid != os.getuid()
        or stat.S_IMODE(metadata.st_mode) & 0o022
    ):
        raise ContiguousOrchestratorError(
            f"{label} is aliased, unowned, or writable by others"
        )
    _require_owned_directory(path, label=label)


def _write_new(path: Path, raw: bytes, *, mode: int = 0o600) -> str:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor = os.open(
        path,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0),
        mode,
    )
    try:
        view = memoryview(raw)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise ContiguousOrchestratorError(
                    f"short write while creating {path}"
                )
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent)
    return _sha256(raw)


def _write_new_json(path: Path, value: object) -> str:
    return _write_new(path, _canonical_json(value) + b"\n")


def _replace_json(path: Path, value: object) -> str:
    raw = _canonical_json(value) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return _sha256(raw)


def _copy_regular(source: Path, destination: Path) -> str:
    raw = _read_regular(
        source, maximum=MAX_TRANSCRIPT_BYTES, allow_empty=True
    )
    return _write_new(destination, raw)


def _validate_flat_source_payloads(
    payloads: Mapping[str, bytes],
) -> tuple[str, ...]:
    try:
        return SourceSchema.validate_source_payloads(payloads)
    except SourceSchema.SourceSchemaError as exc:
        raise ContiguousOrchestratorError(
            "candidate source violates the shared source schema"
        ) from exc


def _source_payloads_from_manifest(
    output_root: Path,
    manifest: Mapping[str, Any],
) -> dict[str, bytes]:
    try:
        Supervisor._validate_regular_tree(
            output_root, label="candidate output"
        )
    except Supervisor.SupervisorContractError as exc:
        raise ContiguousOrchestratorError(
            "candidate output is not one regular tree"
        ) from exc
    direct = {entry.name: entry for entry in output_root.iterdir()}
    if set(direct) != {
        CANDIDATE_NAME,
        "worker_outcome.json",
        "source",
    } or not direct["source"].is_dir():
        raise ContiguousOrchestratorError(
            "candidate output top-level inventory is not exact"
        )
    declared = manifest.get("exported_files_sha256")
    if not isinstance(declared, dict):
        raise ContiguousOrchestratorError(
            "candidate export inventory is not an object"
        )
    source_names = {
        relative: PurePosixPath(relative).name
        for relative in declared
        if (
            isinstance(relative, str)
            and PurePosixPath(relative).parent
            == PurePosixPath("source")
            and len(PurePosixPath(relative).parts) == 2
        )
    }
    expected = {"worker_outcome.json", *source_names}
    if (
        set(declared) != expected
        or any(
            not isinstance(digest, str)
            or SHA256_RE.fullmatch(digest) is None
            for digest in declared.values()
        )
    ):
        raise ContiguousOrchestratorError(
            "candidate manifest does not declare one exact reusable source set"
        )
    payloads: dict[str, bytes] = {}
    for relative, name in sorted(source_names.items()):
        path = output_root / relative
        raw = _read_regular(
            path, maximum=SourceSchema.MAX_FILE_BYTES, allow_empty=True
        )
        if _sha256(raw) != declared[relative]:
            raise ContiguousOrchestratorError(
                f"candidate source hash mismatch: {relative}"
            )
        if name in payloads:
            raise ContiguousOrchestratorError(
                "candidate source basenames are duplicated"
            )
        payloads[name] = raw
    outcome_raw = _read_regular(
        output_root / "worker_outcome.json",
        maximum=MAX_JSON_BYTES,
    )
    if _sha256(outcome_raw) != declared["worker_outcome.json"]:
        raise ContiguousOrchestratorError(
            "candidate worker outcome hash mismatch"
        )
    outcome = _strict_json(
        outcome_raw, label="candidate worker outcome"
    )
    if (
        set(outcome)
        != {"schema", "kind", "attempt_id", "authoritative"}
        or outcome.get("schema") != 1
        or isinstance(outcome.get("schema"), bool)
        or outcome.get("kind")
        != "arc_agi3_contiguous_proposer_worker"
        or outcome.get("authoritative") is not False
    ):
        raise ContiguousOrchestratorError(
            "candidate worker outcome schema is invalid"
        )
    _validate_flat_source_payloads(payloads)
    actual = {
        path.relative_to(output_root).as_posix()
        for path in output_root.rglob("*")
        if path.is_file() and path.name != CANDIDATE_NAME
    }
    if actual != set(declared):
        raise ContiguousOrchestratorError(
            "candidate output contains undeclared or missing files"
        )
    return payloads


def _load_candidate(
    spec: Runner.AttemptSpec,
    candidate: Runner.PromotionCandidate,
) -> tuple[Path, bytes, dict[str, Any], dict[str, bytes]]:
    if (
        candidate.game != spec.game
        or candidate.from_level != spec.target_level - 1
        or candidate.to_level != spec.target_level
        or candidate.parent_checkpoint_sha256
        != spec.parent_checkpoint_sha256
    ):
        raise ContiguousOrchestratorError(
            "candidate identity differs from the admitted frontier"
        )
    manifest_path = Path(candidate.candidate_manifest_path)
    output_root = Path(spec.output_dir)
    if (
        not output_root.is_absolute()
        or output_root.is_symlink()
        or not manifest_path.is_absolute()
        or manifest_path != output_root / CANDIDATE_NAME
        or manifest_path.is_symlink()
    ):
        raise ContiguousOrchestratorError(
            "candidate manifest is not the canonical output manifest"
        )
    raw = _read_regular(manifest_path, maximum=MAX_JSON_BYTES)
    if _sha256(raw) != candidate.candidate_manifest_sha256:
        raise ContiguousOrchestratorError(
            "candidate manifest changed after collection"
        )
    manifest = _strict_json(raw, label="candidate manifest")
    required = {
        "schema",
        "game",
        "target_level",
        "parent_checkpoint_sha256",
        "candidate_path",
        "exported_files_sha256",
    }
    if (
        set(manifest) != required
        or manifest.get("schema") != 1
        or isinstance(manifest.get("schema"), bool)
        or manifest.get("game") != spec.game
        or manifest.get("target_level") != spec.target_level
        or isinstance(manifest.get("target_level"), bool)
        or manifest.get("parent_checkpoint_sha256")
        != spec.parent_checkpoint_sha256
    ):
        raise ContiguousOrchestratorError(
            "candidate manifest targets a stale/wrong frontier"
        )
    actions = manifest["candidate_path"]
    if (
        not isinstance(actions, list)
        or not actions
        or len(actions) > Supervisor.MAX_REPLAY_ACTIONS
        or not all(Release._valid_action(action) for action in actions)
    ):
        raise ContiguousOrchestratorError(
            "candidate manifest path is malformed or exhausted"
        )
    payloads = _source_payloads_from_manifest(output_root, manifest)
    outcome = _strict_json(
        _read_regular(
            output_root / "worker_outcome.json",
            maximum=MAX_JSON_BYTES,
        ),
        label="candidate worker outcome",
    )
    if outcome["attempt_id"] != spec.attempt_id:
        raise ContiguousOrchestratorError(
            "candidate worker outcome targets another attempt"
        )
    return output_root, raw, manifest, payloads


class TrustedCandidateCollector:
    """Convert one quiescent, host-observed attempt into a runner result.

    The Docker adapter independently authenticates the candidate publication
    after this conversion.  This collector therefore chooses only the canonical
    manifest and never trusts model prose or a container-authored PASS flag.
    """

    def __call__(
        self,
        runner_spec: Runner.AttemptSpec,
        terminal_poll: Runner.BackendPoll,
        arena_host_result: Any | None,
        worker_outcome: Mapping[str, Any] | None,
        output_root: Path,
    ) -> Runner.AttemptResult:
        if terminal_poll.status == "containment_fault":
            return Runner.AttemptResult(
                kind="infrastructure",
                reason="attempt containment failed before trusted collection",
            )
        if terminal_poll.status != "exited":
            raise ContiguousOrchestratorError(
                "trusted collection requires a terminal attempt"
            )
        if not isinstance(worker_outcome, Mapping):
            return Runner.AttemptResult(
                kind="infrastructure",
                reason="worker outcome is absent",
            )
        # A trusted Arena result exists only after an authenticated, durably
        # delivered clean close.  Require it even when the worker exported no
        # candidate: otherwise an out-of-range or malformed public action could
        # poison the RPC session, be caught by solver code, and be mislabeled as
        # ordinary clean no-progress with restorable WIP.
        if arena_host_result is None:
            return Runner.AttemptResult(
                kind="infrastructure",
                reason="attempt lacks a clean trusted Arena session",
            )
        manifest_path = Path(output_root) / CANDIDATE_NAME
        if not manifest_path.exists():
            return Runner.AttemptResult(
                kind="clean_no_progress",
                reason="no authenticated candidate was published",
            )
        raw = _read_regular(manifest_path, maximum=MAX_JSON_BYTES)
        manifest = _strict_json(raw, label="candidate manifest")
        try:
            observed_path = tuple(arena_host_result.path)
            observed_level = int(arena_host_result.levels_completed)
        except (AttributeError, TypeError, ValueError):
            return Runner.AttemptResult(
                kind="infrastructure",
                reason="trusted Arena result is malformed",
            )
        if (
            manifest.get("candidate_path") != [
                list(action) if isinstance(action, tuple) else action
                for action in observed_path
            ]
            or observed_level < runner_spec.target_level
        ):
            return Runner.AttemptResult(
                kind="infrastructure",
                reason="candidate differs from the trusted Arena path",
            )
        arena_receipt_path = (
            Path(runner_spec.host_transcript_path).parent
            / "arena_session_binding_receipt.json"
        )
        arena_receipt_raw = _read_regular(
            arena_receipt_path, maximum=MAX_JSON_BYTES
        )
        arena_receipt = Runner._validate_bound_receipt(
            str(arena_receipt_path),
            _sha256(arena_receipt_raw),
            expected_path=arena_receipt_path,
            expected_kind="contiguous_arena_session_binding",
            spec=runner_spec,
        )
        binding_event = arena_receipt.get("binding_event")
        if (
            not isinstance(binding_event, dict)
            or binding_event.get("game") != runner_spec.game
            or binding_event.get("attempt_id")
            != runner_spec.attempt_id
            or not isinstance(
                binding_event.get("probe_isolation_evidence"), dict
            )
        ):
            return Runner.AttemptResult(
                kind="infrastructure",
                reason="candidate lacks controller probe-isolation evidence",
            )
        try:
            probe_mode, probe_digest = (
                Supervisor.validate_probe_isolation_evidence(
                    binding_event["probe_isolation_evidence"],
                    expected_seed_snapshot_sha256=binding_event.get(
                        "exploration_seed_snapshot_sha256"
                    ),
                    expected_seed_path_sha256=binding_event.get(
                        "exploration_seed_path_sha256"
                    ),
                )
            )
        except Supervisor.SupervisorContractError:
            return Runner.AttemptResult(
                kind="infrastructure",
                reason="candidate probe-isolation evidence failed closed",
            )
        if (
            binding_event.get("probe_isolation_mode") != probe_mode
            or binding_event.get(
                "probe_isolation_evidence_sha256"
            )
            != probe_digest
        ):
            return Runner.AttemptResult(
                kind="infrastructure",
                reason="candidate probe-isolation binding is inconsistent",
            )
        candidate = Runner.PromotionCandidate(
            game=runner_spec.game,
            from_level=runner_spec.target_level - 1,
            to_level=runner_spec.target_level,
            parent_checkpoint_sha256=(
                runner_spec.parent_checkpoint_sha256
            ),
            candidate_manifest_path=str(manifest_path),
            candidate_manifest_sha256=_sha256(raw),
            probe_isolation_mode=probe_mode,
            probe_isolation_evidence_sha256=probe_digest,
            supervisory_handoff_sha256=(
                runner_spec.supervisory_handoff
                .supervisory_handoff_sha256
                if runner_spec.supervisory_handoff is not None
                else None
            ),
            supervisory_native_reproduction_receipt_sha256=None,
        )
        # Parse the exact declared source set before returning.  The adapter
        # repeats this under its authenticated bridge evidence.
        _load_candidate(runner_spec, candidate)
        return Runner.AttemptResult(
            kind="candidate",
            reason="canonical candidate awaits independent replay",
            candidate=candidate,
        )


def _tree_hash(
    root: Path,
    *,
    exclude_relative: frozenset[str] = frozenset(),
    exclude_prefixes: tuple[str, ...] = (),
) -> str:
    return Supervisor._tree_hash(
        root,
        exclude_relative=exclude_relative,
        exclude_prefixes=exclude_prefixes,
    )


def _seal_tree(root: Path) -> None:
    for path in sorted(
        root.rglob("*"), key=lambda item: len(item.parts), reverse=True
    ):
        if path.is_file():
            os.chmod(path, 0o400, follow_symlinks=False)
        elif path.is_dir():
            os.chmod(path, 0o500, follow_symlinks=False)
        else:
            raise ContiguousOrchestratorError(
                f"cannot seal nonregular version entry: {path}"
            )
    os.chmod(root, 0o500, follow_symlinks=False)


def _is_sealed_tree(root: Path) -> bool:
    try:
        Supervisor._validate_regular_tree(
            root, label="sealed promotion version"
        )
    except Supervisor.SupervisorContractError:
        return False
    for path in (root, *root.rglob("*")):
        mode = stat.S_IMODE(os.lstat(path).st_mode)
        if path.is_dir():
            if mode != 0o500:
                return False
        elif path.is_file():
            if mode != 0o400:
                return False
        else:
            return False
    return True


def _source_tree_sha256(payloads: Mapping[str, bytes]) -> str:
    digest = hashlib.sha256()
    for name, raw in sorted(payloads.items()):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256(raw).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _public_replay_evidence(
    replay: IsolatedReplayEvidence,
) -> dict[str, Any]:
    return {
        key: value
        for key, value in asdict(replay).items()
        if key not in REPLAY_FILESYSTEM_PATH_FIELDS
    }


def _source_description_bytes(payloads: Mapping[str, bytes]) -> int:
    # Every reusable byte participates, including modular JSON/TXT support.
    return sum(len(raw) for raw in payloads.values())


def _source_description_lengths(
    payloads: Mapping[str, bytes],
) -> dict[str, int]:
    """Versioned per-file byte-description metric used by this campaign.

    Positive growth is taken independently per reusable file, matching the
    historical no-cross-file-cancellation rule while extending it to every
    declared Python/JSON/TXT source byte.  Like the historical metric, a
    same-size rewrite is intentionally zero and is analyzed separately by the
    post-hoc normalized-AST novelty audit.
    """

    _validate_flat_source_payloads(payloads)
    return {name: len(payloads[name]) for name in sorted(payloads)}


def _marginal_description_growth(
    before: Mapping[str, bytes],
    after: Mapping[str, bytes],
) -> tuple[int, dict[str, int], dict[str, int]]:
    before_lengths = _source_description_lengths(before)
    after_lengths = _source_description_lengths(after)
    marginal = sum(
        max(
            0,
            after_lengths.get(name, 0)
            - before_lengths.get(name, 0),
        )
        for name in set(before_lengths) | set(after_lengths)
    )
    return marginal, before_lengths, after_lengths


def _normalize_actions(actions: Sequence[Any]) -> list[Any]:
    normalized: list[Any] = []
    for action in actions:
        if isinstance(action, tuple):
            action = list(action)
        if not Release._valid_action(action):
            raise ContiguousOrchestratorError(
                f"invalid replay action: {action!r}"
            )
        normalized.append(action)
    return normalized


def _exact_path(game: str, path: Sequence[Any], level: int) -> list[Any]:
    exact = gkm_legs.exact_level_boundary(game, path, level)
    if exact is None:
        raise ContiguousOrchestratorError(
            "replay did not reach the exact requested boundary"
        )
    normalized = _normalize_actions(exact)
    if (
        not normalized
        or len(normalized) > Supervisor.MAX_REPLAY_ACTIONS
        or not gkm_arena.validate(game, normalized, level)
    ):
        raise ContiguousOrchestratorError(
            "exact boundary failed independent Arena validation"
        )
    return normalized


def _path_replay(game: str, level: int, path: Sequence[Any]) -> None:
    reached, observed, error = gkm_legs._run_candidate_replay(
        game, list(path)
    )
    if (
        error
        or reached != level
        or _normalize_actions(observed) != list(path)
        or not gkm_arena.validate(game, list(path), level)
    ):
        raise ContiguousOrchestratorError(
            "independent path-from-zero replay failed"
        )


def _tool_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    components = {
        "scanner": Taint.source_sha256(),
        "arena": _sha256(_read_regular(
            root / "gkm_arena.py", maximum=32 * 1024 * 1024
        )),
        "legs_runtime": _sha256(_read_regular(
            root / "gkm_legs.py", maximum=32 * 1024 * 1024
        )),
        "arena_rpc": _sha256(_read_regular(
            root / "arc_agi3_arena_rpc.py",
            maximum=32 * 1024 * 1024,
        )),
        "arena_rpc_client": _sha256(_read_regular(
            root / "arc_agi3_arena_rpc_client.py",
            maximum=32 * 1024 * 1024,
        )),
        "replay_worker": _sha256(_read_regular(
            root / "arc_agi3_container_worker.py",
            maximum=32 * 1024 * 1024,
        )),
        "proposer_worker": _sha256(_read_regular(
            root / "arc_agi3_proposer_worker.py",
            maximum=32 * 1024 * 1024,
        )),
        "source_schema": _sha256(_read_regular(
            root / "arc_agi3_source_schema.py",
            maximum=32 * 1024 * 1024,
        )),
        "container_recipe": _sha256(_read_regular(
            root / "container" / "Containerfile.arc-agi3-contiguous",
            maximum=32 * 1024 * 1024,
        )),
        "solver_requirements": _sha256(_read_regular(
            root / "container" / "arc_agi3_solver_requirements.lock",
            maximum=32 * 1024 * 1024,
        )),
        "container_backend": _sha256(_read_regular(
            root / "arc_agi3_container_backend.py",
            maximum=32 * 1024 * 1024,
        )),
        "hasher": _sha256(
            _read_regular(
                Path(__file__).resolve(),
                maximum=32 * 1024 * 1024,
            )
        ),
    }
    components["engine"] = _json_sha256({
        name: components[name]
        for name in (
            "arena",
            "legs_runtime",
            "arena_rpc",
            "arena_rpc_client",
            "replay_worker",
            "proposer_worker",
            "source_schema",
            "container_recipe",
            "solver_requirements",
            "container_backend",
        )
    })
    return components


def _prefixed_snapshot(
    version: Path,
    *,
    game: str,
) -> Release.TreeSnapshot:
    """Snapshot a physical one-game release wrapper without virtual paths."""

    source = Release._snapshot_tree(version)
    prefix = f"{game}_legs"
    if (
        source.file_children.get("", frozenset())
        or source.directory_children.get("", frozenset())
        != frozenset({prefix})
    ):
        raise ContiguousOrchestratorError(
            "version is not an exact one-game release wrapper"
        )
    return source


def _validate_schema_v2_chain(
    version: Path,
    *,
    game: str,
    reached: int,
) -> list[dict[str, Any]]:
    snapshot = _prefixed_snapshot(version, game=game)
    tools = _tool_hashes()
    allowed = frozenset(tools.values())
    previous_checkpoint: str | None = None
    previous_manifest: str | None = None
    previous_records: list[dict[str, Any]] | None = None
    summaries: list[dict[str, Any]] = []
    try:
        for level in range(1, reached + 1):
            summary, records = Release._validate_boundary(
                snapshot,
                game=game,
                level=level,
                previous_checkpoint_sha256=previous_checkpoint,
                previous_manifest_sha256=previous_manifest,
                previous_records=previous_records,
                allowed_tool_hashes=allowed,
            )
            summaries.append(summary)
            previous_checkpoint = summary["checkpoint_sha256"]
            previous_manifest = summary["manifest_sha256"]
            previous_records = records
    except Release.ReleaseGateError as exc:
        raise ContiguousOrchestratorError(
            "schema-v2 promotion chain failed release validation"
        ) from exc
    return summaries


class DockerReplayExecutor:
    """Run candidate source from zero in the real digest-pinned replay role."""

    def __init__(
        self,
        backend: Any,
        *,
        replay_image_reference: str,
        evidence_root: Path,
        timeout_seconds: int = MAX_REPLAY_SECONDS,
    ) -> None:
        try:
            import arc_agi3_container_backend as Container
        except ImportError as exc:
            raise ContiguousOrchestratorError(
                "container backend is unavailable"
            ) from exc
        if not isinstance(backend, Container.DockerContainerBackend):
            raise ContiguousOrchestratorError(
                "Docker replay requires the observed container backend"
            )
        Container.parse_digest_reference(replay_image_reference)
        if (
            not isinstance(timeout_seconds, int)
            or isinstance(timeout_seconds, bool)
            or not 1 <= timeout_seconds <= MAX_REPLAY_SECONDS
        ):
            raise ContiguousOrchestratorError(
                "replay timeout is outside the production bound"
            )
        self.backend = backend
        self.replay_image_reference = replay_image_reference
        self.evidence_root = Path(evidence_root).resolve()
        self.timeout_seconds = timeout_seconds
        self.evidence_root.mkdir(parents=True, exist_ok=True, mode=0o700)

    def replay_from_zero(
        self,
        *,
        spec: Runner.AttemptSpec,
        source_payloads: Mapping[str, bytes],
    ) -> IsolatedReplayEvidence:
        import arc_agi3_arena_rpc as ArenaRpc
        import arc_agi3_container_backend as Container

        names = _validate_flat_source_payloads(source_payloads)
        replay_id = str(uuid.uuid4())
        root = self.evidence_root / replay_id
        if root.exists() or root.is_symlink():
            raise ContiguousOrchestratorError(
                "replay evidence identity collided"
            )
        input_root = root / "input"
        export_root = root / "export"
        rpc_root = root / "rpc"
        evidence = root / "evidence"
        for path in (input_root, export_root, rpc_root, evidence):
            path.mkdir(parents=True, mode=0o700)
        for name in names:
            _write_new(input_root / name, source_payloads[name])
        _seal_tree(input_root)
        token = uuid.uuid4().hex + uuid.uuid4().hex
        token_path = rpc_root / "token"
        _write_new(token_path, token.encode("ascii"), mode=0o400)
        socket_path = rpc_root / "arena.sock"
        transcript_path = evidence / "arena_replay.jsonl"
        zero_checkpoint = {
            "game": spec.game,
            "reached": 0,
            "total_marginal_C": 0,
            "records": [],
            "final_path": [],
            "validated": False,
        }
        zero_sha256 = _sha256(
            _canonical_json(zero_checkpoint) + b"\n"
        )
        binding = ArenaRpc.ArenaSessionBinding(
            campaign_id=spec.campaign_id,
            generation_id=str(uuid.uuid4()),
            attempt_id=str(uuid.uuid4()),
            game=spec.game,
            parent_level=0,
            target_level=spec.target_level,
            parent_checkpoint_sha256=zero_sha256,
            frontier_sha256=Runner.frontier_sha256(
                spec.game, 0, zero_sha256
            ),
            exploration_mode="continue_parent",
        )
        session = ArenaRpc.ArenaHostSession(
            spec.game,
            binding=binding,
            parent_path=(),
            token=token,
        )
        server = ArenaRpc.ArenaRpcServer(
            session, socket_path, transcript_path
        )
        server_thread = server.start_thread()
        identity = Container.AttemptIdentity(
            campaign_id=spec.campaign_id,
            generation_id=binding.generation_id,
            attempt_id=binding.attempt_id,
            game=spec.game,
            target_level=spec.target_level,
        )
        low_spec = Container.AttemptSpec(
            identity=identity,
            image_reference=self.replay_image_reference,
            parent_input=input_root,
            export_root=export_root,
            arena_socket=socket_path,
            arena_token_file=token_path,
            command=Container.expected_worker_command(),
            resource_limits=Container.ResourceLimits(
                cpus=2.0,
                memory_bytes=4 * 1024**3,
                pids=256,
                tmpfs_bytes=512 * 1024**2,
            ),
            soft_allocation_seconds=float(self.timeout_seconds),
            role="replay",
        )
        attestation = None
        running = None
        running_observation_sha256: str | None = None
        logs = None
        teardown = None
        terminal_fault: BaseException | None = None
        try:
            attestation = self.backend.build_launch_attestation(low_spec)
            running = self.backend.start_attested(
                attestation, low_spec
            )
            running_observation_sha256 = (
                running.running_observation_sha256
            )
            deadline = time.monotonic() + self.timeout_seconds
            while True:
                observed = self.backend.observe_container_state(
                    attestation, low_spec, timeout_seconds=30
                )
                if not observed.running:
                    if (
                        observed.status != "exited"
                        or observed.exit_code != 0
                        or observed.oom_killed
                        or observed.error
                    ):
                        raise ContiguousOrchestratorError(
                            "isolated source replay container failed"
                        )
                    break
                if time.monotonic() >= deadline:
                    raise ContiguousOrchestratorError(
                        "isolated source replay exceeded its hard bound"
                    )
                time.sleep(0.05)
            server.wait(timeout=30)
            server_thread.join(timeout=1)
            if server_thread.is_alive():
                raise ContiguousOrchestratorError(
                    "Arena replay server remained live"
                )
            logs = self.backend.collect_terminal_logs(
                attestation, low_spec
            )
            host_result = session.host_result()
            exact = _exact_path(
                spec.game,
                host_result.path,
                spec.target_level,
            )
            observed_path = _normalize_actions(host_result.path)
            if (
                host_result.levels_completed != spec.target_level
                or observed_path != exact
            ):
                raise ContiguousOrchestratorError(
                    "isolated source replay did not stop at the exact boundary"
                )
            outcome_path = export_root / "worker_outcome.json"
            outcome_raw = _read_regular(
                outcome_path, maximum=MAX_JSON_BYTES
            )
            outcome = _strict_json(
                outcome_raw, label="replay worker outcome"
            )
            if (
                set(outcome)
                != {
                    "schema",
                    "status",
                    "solver_sha256",
                    "elapsed_ns",
                    "error",
                    "authoritative",
                }
                or outcome.get("schema")
                != "arc-agi3-container-worker/v1"
                or outcome.get("status") != "completed"
                or outcome.get("authoritative") is not False
                or outcome.get("error") is not None
                or not isinstance(outcome.get("elapsed_ns"), int)
                or isinstance(outcome.get("elapsed_ns"), bool)
                or outcome["elapsed_ns"] < 0
                or outcome.get("solver_sha256")
                != _sha256(source_payloads["solve.py"])
            ):
                raise ContiguousOrchestratorError(
                    "replay worker did not complete the exact source"
                )
            stdout_path = evidence / "container_stdout.log"
            stderr_path = evidence / "container_stderr.log"
            _write_new(
                stdout_path, logs.stdout.encode("utf-8"), mode=0o600
            )
            _write_new(
                stderr_path, logs.stderr.encode("utf-8"), mode=0o600
            )
            teardown = self.backend.teardown(
                running,
                cause=Container.TeardownCause.NORMAL_EXIT,
                graceful_seconds=20,
            )
            if (
                teardown.container_id != attestation.container_id
                or teardown.cause != "normal_exit"
                or teardown.container_inspect_absent is not True
                or teardown.container_top_absent is not True
                or teardown.identity_label_query_empty is not True
                or teardown.no_descendants is not True
                or SHA256_RE.fullmatch(teardown.proof_sha256) is None
            ):
                raise ContiguousOrchestratorError(
                    "isolated replay teardown proof is incomplete"
                )
            running = None
            token_path.unlink()
            _fsync_directory(rpc_root)
            source_tree_sha256 = _source_tree_sha256(
                source_payloads
            )
            result = IsolatedReplayEvidence(
                schema=SCHEMA,
                replay_id=replay_id,
                game=spec.game,
                target_level=spec.target_level,
                observed_level=host_result.levels_completed,
                observed_path=tuple(observed_path),
                exact_path=tuple(exact),
                source_tree_sha256=source_tree_sha256,
                replay_image_reference=self.replay_image_reference,
                replay_image_digest=attestation.image.manifest_digest,
                container_id=attestation.container_id,
                launch_attestation_sha256=(
                    attestation.document_sha256
                ),
                running_observation_sha256=(
                    running_observation_sha256
                ),
                arena_transcript_path=str(transcript_path),
                arena_transcript_sha256=_sha256(
                    _read_regular(
                        transcript_path,
                        maximum=MAX_TRANSCRIPT_BYTES,
                    )
                ),
                worker_outcome_path=str(outcome_path),
                worker_outcome_sha256=_sha256(outcome_raw),
                stdout_path=str(stdout_path),
                stdout_sha256=logs.stdout_sha256,
                stderr_path=str(stderr_path),
                stderr_sha256=logs.stderr_sha256,
                teardown_proof_sha256=teardown.proof_sha256,
            )
            _write_new_json(evidence / "replay_receipt.json", asdict(result))
            _seal_tree(root)
            return result
        except BaseException as exc:
            terminal_fault = exc
            raise
        finally:
            if terminal_fault is not None:
                server.shutdown()
                try:
                    server.wait(timeout=5)
                except Exception:
                    pass
                if running is not None:
                    try:
                        self.backend.teardown(
                            running,
                            cause=Container.TeardownCause.CONTAINMENT_FAULT,
                            graceful_seconds=0,
                        )
                    except Exception:
                        pass
                elif attestation is not None:
                    try:
                        self.backend.teardown(
                            attestation,
                            cause=Container.TeardownCause.CONTAINMENT_FAULT,
                            graceful_seconds=0,
                        )
                    except Exception:
                        pass
                try:
                    token_path.unlink()
                except FileNotFoundError:
                    pass


def _scan_primary_files(
    root: Path,
    names: Sequence[str],
    *,
    secret_sentinels: Sequence[str],
) -> None:
    for name in names:
        raw = _read_regular(
            root / name,
            maximum=max(SourceSchema.MAX_FILE_BYTES, MAX_JSON_BYTES),
            allow_empty=True,
        )
        _require_no_live_secret(
            raw,
            secret_sentinels=secret_sentinels,
            label=f"promoted evidence {name}",
        )
        record = Taint.scan_regular_file(
            root / name, evidence_kind="candidate_output"
        )
        if record.hits:
            raise ContiguousOrchestratorError(
                f"promoted evidence is tainted: {name}: {record.hits}"
            )


def _app_scan_policy(
    spec: Runner.AttemptSpec,
    *,
    secret_sentinels: tuple[str, ...],
) -> Taint.AppServerScanPolicy:
    frontier = _strict_json(
        _read_regular(
            Path(spec.frontier_brief_path), maximum=MAX_JSON_BYTES
        ),
        label="frontier brief",
    )
    prompt = (
        "Solve exactly this receipt-bound ARC-AGI-3 frontier using only the "
        "contiguous_lane namespace. Immutable frontier:\n"
        + Transport.canonical_json(frontier).decode("ascii")
    )
    return Taint.AppServerScanPolicy(
        state_root=spec.app_server_state_dir,
        neutral_cwd=spec.neutral_host_cwd_path,
        model=spec.proposer_transport.model,
        model_provider=spec.proposer_transport.model_provider,
        reasoning_effort=spec.effort,
        thread_mode=spec.thread_mode,
        resume_thread_id=spec.resume_thread_id,
        prompt_sha256=_sha256(prompt.encode("utf-8")),
        hard_safety_seconds=spec.hard_safety_seconds,
        max_auth_refreshes=spec.max_auth_refreshes,
        secret_sentinels=secret_sentinels,
    )


class ProductionPromotionGate:
    """Independent schema-v2 replay gate and atomic per-game publisher."""

    def __init__(
        self,
        root: Path,
        *,
        replay_executor: SourceReplayExecutor,
        secret_sentinels: tuple[str, ...] = (),
        frontier_import_root: Path | None = None,
        selective_frontier_import:
            Runner.SelectiveFrontierImport | None = None,
        fault_at: str | None = None,
    ) -> None:
        requested_root = Path(root)
        if not requested_root.is_absolute():
            requested_root = Path(
                os.path.abspath(os.fspath(requested_root))
            )
        if requested_root.is_symlink():
            raise ContiguousOrchestratorError(
                "promotion store root cannot be a symlink"
            )
        resolved_import_root: Path | None = None
        if frontier_import_root is not None:
            requested_import_root = Path(frontier_import_root)
            if (
                not requested_import_root.is_absolute()
                or requested_import_root.is_symlink()
                or not requested_import_root.is_dir()
            ):
                raise ContiguousOrchestratorError(
                    "frontier import root must be an existing absolute "
                    "regular directory"
                )
            try:
                resolved_import_root = requested_import_root.resolve(
                    strict=True
                )
                prospective_root = requested_root.resolve(strict=False)
            except OSError as exc:
                raise ContiguousOrchestratorError(
                    "promotion/import roots cannot be resolved"
                ) from exc
            if (
                resolved_import_root != requested_import_root
                or prospective_root != requested_root
            ):
                raise ContiguousOrchestratorError(
                    "promotion/import root is aliased"
                )
            _require_owned_directory(
                resolved_import_root,
                label="frontier import authority root",
            )
            try:
                resolved_import_root.relative_to(prospective_root)
            except ValueError:
                pass
            else:
                raise ContiguousOrchestratorError(
                    "frontier import root is inside mutable promotion root"
                )
            try:
                prospective_root.relative_to(resolved_import_root)
            except ValueError:
                pass
            else:
                raise ContiguousOrchestratorError(
                    "mutable promotion root is inside frontier import root"
                )
        requested_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(requested_root, 0o700, follow_symlinks=False)
        _require_owned_directory(
            requested_root, label="promotion store root"
        )
        self.root = requested_root
        self.frontier_import_root = resolved_import_root
        if (
            selective_frontier_import is not None
            and resolved_import_root is None
        ):
            raise ContiguousOrchestratorError(
                "pinned selective import requires its authority root"
            )
        try:
            self.selective_frontier_import = (
                None
                if selective_frontier_import is None
                else Runner.selective_frontier_import_from_dict(
                    asdict(selective_frontier_import)
                )
            )
        except Runner.ContiguousRunnerError as exc:
            raise ContiguousOrchestratorError(
                "pinned selective frontier import is malformed"
            ) from exc
        if not callable(getattr(replay_executor, "replay_from_zero", None)):
            raise ContiguousOrchestratorError(
                "production promotion requires an isolated replay executor"
            )
        if (
            not isinstance(secret_sentinels, tuple)
            or any(
                not isinstance(item, str) or not item
                for item in secret_sentinels
            )
        ):
            raise ContiguousOrchestratorError(
                "promotion secret sentinels are malformed"
            )
        if fault_at not in {
            None,
            "after_version",
            "after_pointer",
        }:
            raise ContiguousOrchestratorError(
                "unknown promotion fault injection point"
            )
        self.replay_executor = replay_executor
        self.secret_sentinels = secret_sentinels
        self.fault_at = fault_at

    def _game_root(self, game: str) -> Path:
        if (
            not isinstance(game, str)
            or re.fullmatch(r"[a-z0-9]{4}", game) is None
        ):
            raise ContiguousOrchestratorError("invalid game identity")
        root = self.root / game
        _ensure_private_directory(root, label="per-game promotion root")
        for name in (
            VERSIONS_NAME,
            STAGING_NAME,
            QUARANTINE_NAME,
            INTENTS_NAME,
        ):
            _ensure_private_directory(
                root / name, label=f"promotion {name} directory"
            )
        return root

    def _import_game_root(self, game: str) -> Path:
        if (
            not isinstance(game, str)
            or re.fullmatch(r"[a-z0-9]{4}", game) is None
        ):
            raise ContiguousOrchestratorError("invalid game identity")
        if self.frontier_import_root is None:
            raise ContiguousOrchestratorError(
                "selective continuation has no frontier import root"
            )
        root = self.frontier_import_root / game
        if (
            root.is_symlink()
            or not root.is_dir()
            or root.resolve(strict=True) != root
        ):
            raise ContiguousOrchestratorError(
                "frontier import game root is unavailable or aliased"
            )
        _require_owned_directory(
            root, label="frontier import game root"
        )
        return root

    @staticmethod
    def _lock(game_root: Path):
        path = game_root / ".promotion.lock"
        descriptor = os.open(
            path,
            os.O_RDWR
            | os.O_CREAT
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            os.close(descriptor)
            raise ContiguousOrchestratorError(
                "promotion lock is aliased or nonregular"
            )
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        return descriptor

    @staticmethod
    def _unlock(descriptor: int) -> None:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)

    @staticmethod
    def _intent_document(
        *,
        version_id: str,
        spec: Runner.AttemptSpec,
        candidate: Runner.PromotionCandidate,
    ) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "kind": "arc_agi3_contiguous_promotion_intent",
            "version": version_id,
            "campaign_id": spec.campaign_id,
            "generation_id": spec.generation_id,
            "attempt_id": spec.attempt_id,
            "game": spec.game,
            "target_level": spec.target_level,
            "parent_checkpoint_sha256":
                spec.parent_checkpoint_sha256,
            "candidate_manifest_sha256":
                candidate.candidate_manifest_sha256,
        }

    @staticmethod
    def _read_intent(path: Path) -> dict[str, Any]:
        value = _strict_json(
            _read_regular(path, maximum=MAX_JSON_BYTES),
            label="promotion intent",
        )
        required = {
            "schema",
            "kind",
            "version",
            "campaign_id",
            "generation_id",
            "attempt_id",
            "game",
            "target_level",
            "parent_checkpoint_sha256",
            "candidate_manifest_sha256",
        }
        if (
            set(value) != required
            or value.get("schema") != SCHEMA
            or isinstance(value.get("schema"), bool)
            or value.get("kind")
            != "arc_agi3_contiguous_promotion_intent"
            or VERSION_RE.fullmatch(str(value.get("version"))) is None
            or re.fullmatch(r"[a-z0-9]{4}", str(value.get("game")))
            is None
            or not isinstance(value.get("target_level"), int)
            or isinstance(value.get("target_level"), bool)
            or value["target_level"] <= 0
            or SHA256_RE.fullmatch(
                str(value.get("parent_checkpoint_sha256"))
            )
            is None
            or SHA256_RE.fullmatch(
                str(value.get("candidate_manifest_sha256"))
            )
            is None
        ):
            raise ContiguousOrchestratorError(
                "promotion intent schema is invalid"
            )
        return value

    def _quarantine_orphan(
        self,
        game_root: Path,
        source: Path,
        *,
        suffix: str,
    ) -> None:
        destination = (
            game_root / QUARANTINE_NAME / f"{source.name}.{suffix}"
        )
        if destination.exists() or destination.is_symlink():
            raise ContiguousOrchestratorError(
                "promotion quarantine identity collided"
            )
        if stat.S_IMODE(os.lstat(source).st_mode) == 0o500:
            os.chmod(source, 0o700, follow_symlinks=False)
        os.replace(source, destination)
        _fsync_directory(game_root / QUARANTINE_NAME)
        _fsync_directory(source.parent)

    def _reconcile_locked(self, game_root: Path) -> None:
        """Reconcile every durable intent/staging/version state after death."""

        current_version: str | None = None
        pointer_path = game_root / POINTER_NAME
        try:
            pointer_metadata = os.lstat(pointer_path)
        except FileNotFoundError:
            pass
        else:
            if (
                not stat.S_ISREG(pointer_metadata.st_mode)
                or pointer_metadata.st_nlink != 1
            ):
                raise ContiguousOrchestratorError(
                    "promotion pointer is aliased or nonregular"
                )
            pointer = _strict_json(
                _read_regular(pointer_path, maximum=MAX_JSON_BYTES),
                label="promotion pointer",
            )
            version_value = pointer.get("version")
            if VERSION_RE.fullmatch(str(version_value)) is None:
                raise ContiguousOrchestratorError(
                    "promotion pointer has an invalid version identity"
                )
            current_version = str(version_value)

        intents: dict[str, dict[str, Any]] = {}
        for path in sorted((game_root / INTENTS_NAME).iterdir()):
            if (
                path.is_symlink()
                or not path.is_file()
                or path.suffix != ".json"
                or VERSION_RE.fullmatch(path.stem) is None
            ):
                raise ContiguousOrchestratorError(
                    "promotion intent store contains an invalid entry"
                )
            value = self._read_intent(path)
            if value["version"] != path.stem:
                raise ContiguousOrchestratorError(
                    "promotion intent filename differs from its identity"
                )
            intents[path.stem] = value

        for stage in sorted((game_root / STAGING_NAME).iterdir()):
            if (
                stage.is_symlink()
                or not stage.is_dir()
                or VERSION_RE.fullmatch(stage.name) is None
                or stage.name not in intents
            ):
                raise ContiguousOrchestratorError(
                    "promotion staging store contains an unauthenticated entry"
                )
            self._quarantine_orphan(
                game_root, stage, suffix="staging"
            )

        for version in sorted((game_root / VERSIONS_NAME).iterdir()):
            if (
                version.is_symlink()
                or not version.is_dir()
                or VERSION_RE.fullmatch(version.name) is None
                or version.name not in intents
            ):
                raise ContiguousOrchestratorError(
                    "promotion version store contains an unauthenticated entry"
                )
            subject = (
                version / f"{intents[version.name]['game']}_legs"
            )
            receipt = subject / HOST_RECEIPT_NAME
            complete = False
            if receipt.is_file() and not receipt.is_symlink():
                try:
                    value = _strict_json(
                        _read_regular(receipt, maximum=MAX_JSON_BYTES),
                        label="host promotion receipt",
                    )
                    complete = (
                        value.get("version") == version.name
                        and value.get("attempt_id")
                        == intents[version.name]["attempt_id"]
                        and value.get("candidate_manifest_sha256")
                        == intents[version.name][
                            "candidate_manifest_sha256"
                        ]
                        and _is_sealed_tree(version)
                    )
                except ContiguousOrchestratorError:
                    complete = False
            if not complete:
                if version.name == current_version:
                    raise ContiguousOrchestratorError(
                        "selected promotion version is incomplete"
                    )
                self._quarantine_orphan(
                    game_root, version, suffix="version"
                )

        for entry in sorted((game_root / QUARANTINE_NAME).iterdir()):
            if entry.is_symlink() or not entry.is_dir():
                raise ContiguousOrchestratorError(
                    "promotion quarantine contains an invalid entry"
                )

    @staticmethod
    def _validate_receipt(
        *,
        version: Path,
        subject: Path,
        pointer: Mapping[str, Any],
        receipt: Mapping[str, Any],
        summaries: Sequence[Mapping[str, Any]],
    ) -> None:
        if (
            set(receipt) != HOST_RECEIPT_FIELDS
            or receipt.get("schema") != HOST_RECEIPT_SCHEMA
            or isinstance(receipt.get("schema"), bool)
            or receipt.get("kind")
            != "arc_agi3_contiguous_schema_v2_promotion"
            or receipt.get("version") != version.name
            or receipt.get("game") != pointer["game"]
            or receipt.get("target_level") != pointer["target_level"]
            or isinstance(receipt.get("target_level"), bool)
            or not isinstance(receipt.get("authoritative_target"), int)
            or isinstance(receipt.get("authoritative_target"), bool)
            or receipt["authoritative_target"] < receipt["target_level"]
            or receipt.get("parent_checkpoint_sha256")
            != pointer["parent_checkpoint_sha256"]
            or receipt.get("candidate_manifest_sha256")
            != pointer["candidate_manifest_sha256"]
            or receipt.get("supervisory_handoff_sha256")
            != pointer["supervisory_handoff_sha256"]
            or receipt.get(
                "supervisory_native_reproduction_receipt_sha256"
            )
            != pointer[
                "supervisory_native_reproduction_receipt_sha256"
            ]
            or receipt.get("checkpoint_sha256")
            != pointer["checkpoint_sha256"]
            or receipt.get("winning_source_tree_sha256")
            != pointer["winning_source_tree_sha256"]
        ):
            raise ContiguousOrchestratorError(
                "selected host receipt identity differs from its pointer"
            )
        for field in (
            "campaign_id",
            "generation_id",
            "attempt_id",
        ):
            if (
                not isinstance(receipt.get(field), str)
                or not receipt[field]
                or len(receipt[field]) > 128
            ):
                raise ContiguousOrchestratorError(
                    "selected host receipt has an invalid execution identity"
                )
        hash_fields = (
            "candidate_path_sha256",
            "probe_isolation_evidence_sha256",
            "exact_path_sha256",
            "schema_v2_manifest_sha256",
            "release_source_tree_sha256",
            "publication_subject_tree_sha256",
        )
        if any(
            SHA256_RE.fullmatch(str(receipt.get(field))) is None
            for field in hash_fields
        ):
            raise ContiguousOrchestratorError(
                "selected host receipt has a malformed digest"
            )
        if (
            receipt.get("probe_isolation_mode")
            not in Supervisor.PROBE_ISOLATION_MODES
            or receipt.get("probe_result_authority")
            != "hypothesis_only"
        ):
            raise ContiguousOrchestratorError(
                "selected host receipt grants probe output promotion "
                "authority"
            )
        exact_path = receipt.get("exact_path")
        if (
            not isinstance(exact_path, list)
            or not exact_path
            or len(exact_path) > Supervisor.MAX_REPLAY_ACTIONS
            or not all(Release._valid_action(action) for action in exact_path)
            or _json_sha256(exact_path)
            != receipt["exact_path_sha256"]
            or not gkm_arena.validate(
                receipt["game"],
                exact_path,
                receipt["target_level"],
            )
        ):
            raise ContiguousOrchestratorError(
                "selected host receipt has an invalid exact path"
            )
        final = summaries[-1]
        transcripts_root = (
            subject
            / "promotion_evidence"
            / f"level_{receipt['target_level']:02d}"
            / "transcripts"
        )
        candidate_manifest_raw = _read_regular(
            transcripts_root / "candidate_manifest.json",
            maximum=MAX_JSON_BYTES,
        )
        candidate_manifest = _strict_json(
            candidate_manifest_raw,
            label="retained candidate manifest",
        )
        certification = _strict_json(
            _read_regular(
                transcripts_root / "certification.json",
                maximum=MAX_JSON_BYTES,
            ),
            label="retained replay certification",
        )
        expected_manifest_path = (
            "promotion_evidence/"
            f"level_{receipt['target_level']:02d}/manifest.json"
        )
        if (
            receipt.get("schema_v2_manifest_path")
            != expected_manifest_path
            or receipt["schema_v2_manifest_sha256"]
            != final["manifest_sha256"]
            or receipt.get("schema_v2_audits_sha256")
            != final["audits_sha256"]
            or receipt["checkpoint_sha256"]
            != final["checkpoint_sha256"]
            or receipt["exact_path_sha256"]
            != final["exact_path_sha256"]
            or receipt["release_source_tree_sha256"]
            != final["winning_source_tree_sha256"]
            or _sha256(candidate_manifest_raw)
            != receipt["candidate_manifest_sha256"]
            or _json_sha256(candidate_manifest.get("candidate_path"))
            != receipt["candidate_path_sha256"]
        ):
            raise ContiguousOrchestratorError(
                "selected host receipt differs from schema-v2 evidence"
            )
        source_payloads = {
            entry.name: _read_regular(
                entry,
                maximum=SourceSchema.MAX_FILE_BYTES,
                allow_empty=True,
            )
            for entry in (subject / WINNING_SOURCE_NAME).iterdir()
        }
        source_names = _validate_flat_source_payloads(source_payloads)
        after_by_file = _source_description_lengths(source_payloads)
        before_by_file = receipt.get(
            "source_description_bytes_before_by_file"
        )
        if (
            not isinstance(before_by_file, dict)
            or any(
                not isinstance(name, str)
                or not isinstance(value, int)
                or isinstance(value, bool)
                or value < 0
                for name, value in before_by_file.items()
            )
            or receipt.get("source_description_bytes_after_by_file")
            != after_by_file
            or receipt.get("source_description_bytes_before")
            != sum(before_by_file.values())
            or receipt.get("source_description_bytes_after")
            != sum(after_by_file.values())
            or receipt.get("source_description_metric")
            != "positive_per_file_utf8_bytes_v1"
            or receipt.get("same_size_rewrite_novelty")
            != "not_measured_use_posthoc_normalized_ast"
        ):
            raise ContiguousOrchestratorError(
                "selected host receipt description accounting is invalid"
            )
        expected_marginal = sum(
            max(
                0,
                after_by_file.get(name, 0)
                - before_by_file.get(name, 0),
            )
            for name in set(before_by_file) | set(after_by_file)
        )
        checkpoint = _strict_json(
            _read_regular(
                subject / Supervisor.CHECKPOINT_NAME,
                maximum=MAX_JSON_BYTES,
            ),
            label="selected checkpoint",
        )
        if (
            receipt.get("marginal_C") != expected_marginal
            or not checkpoint.get("records")
            or checkpoint["records"][-1].get("marginal_C")
            != expected_marginal
            or checkpoint.get("final_path") != exact_path
        ):
            raise ContiguousOrchestratorError(
                "selected host receipt marginal/path differs from checkpoint"
            )
        replay = receipt.get("isolated_source_replay")
        replay_fields = {
            field.name
            for field in dataclasses.fields(IsolatedReplayEvidence)
            if field.name not in REPLAY_FILESYSTEM_PATH_FIELDS
        }
        if (
            not isinstance(replay, dict)
            or set(replay) != replay_fields
            or replay.get("schema") != SCHEMA
            or replay.get("game") != receipt["game"]
            or replay.get("target_level") != receipt["target_level"]
            or replay.get("observed_level") != receipt["target_level"]
            or replay.get("observed_path") != exact_path
            or replay.get("exact_path") != exact_path
            or replay.get("source_tree_sha256")
            != _source_tree_sha256(source_payloads)
            or re.fullmatch(
                r"sha256:[0-9a-f]{64}",
                str(replay.get("replay_image_digest")),
            )
            is None
            or any(
                SHA256_RE.fullmatch(str(replay.get(field))) is None
                for field in (
                    "source_tree_sha256",
                    "launch_attestation_sha256",
                    "running_observation_sha256",
                    "arena_transcript_sha256",
                    "worker_outcome_sha256",
                    "stdout_sha256",
                    "stderr_sha256",
                    "teardown_proof_sha256",
                )
            )
        ):
            raise ContiguousOrchestratorError(
                "selected host receipt isolated replay is invalid"
            )
        event_hashes = receipt.get(
            "attempt_evidence_event_sha256"
        )
        certification_attempt_evidence = certification.get(
            "attempt_evidence"
        )
        if not isinstance(event_hashes, dict):
            raise ContiguousOrchestratorError(
                "selected host receipt journal hashes are invalid"
            )
        _validate_retained_journal_evidence(
            certification_attempt_evidence,
            event_hashes=event_hashes,
            attempt_id=receipt["attempt_id"],
            campaign_id=receipt["campaign_id"],
        )
        assert isinstance(certification_attempt_evidence, dict)
        retained_results = [
            event.get("payload", {}).get("candidate")
            for event in certification_attempt_evidence.get(
                "journal_prefix", []
            )
            if event.get("kind") == "ATTEMPT_RESULT"
            and isinstance(event.get("payload"), dict)
        ]
        if (
            not isinstance(event_hashes, dict)
            or set(event_hashes) != {"collected", "teardown", "result"}
            or any(
                SHA256_RE.fullmatch(str(value)) is None
                for value in event_hashes.values()
            )
            or certification.get("isolated_replay") != replay
            or certification_attempt_evidence.get(
                "collected_event_sha256"
            )
            != event_hashes.get("collected")
            or certification_attempt_evidence.get(
                "teardown_event_sha256"
            )
            != event_hashes.get("teardown")
            or certification_attempt_evidence.get(
                "result_event_sha256"
            )
            != event_hashes.get("result")
            or len(retained_results) != 1
            or not isinstance(retained_results[0], dict)
            or retained_results[0].get("probe_isolation_mode")
            != receipt.get("probe_isolation_mode")
            or retained_results[0].get(
                "probe_isolation_evidence_sha256"
            )
            != receipt.get("probe_isolation_evidence_sha256")
            or final["transcripts_sha256"].get(
                "transcripts/arena_source_replay.jsonl"
            )
            != replay.get("arena_transcript_sha256")
            or receipt.get("path_replay_from_zero") != "PASS"
            or receipt.get("source_replay_from_zero") != "PASS"
            or receipt.get("taint_scan") != "PASS"
            or receipt.get("control_tools_sha256") != _tool_hashes()
            or receipt["publication_subject_tree_sha256"]
            != _tree_hash(
                subject,
                exclude_relative=frozenset({HOST_RECEIPT_NAME}),
            )
            or sorted(source_names)
            != sorted(final["winning_source_files_sha256"])
        ):
            raise ContiguousOrchestratorError(
                "selected host receipt evidence/control binding is invalid"
            )

    def _current(self, game_root: Path) -> dict[str, Any] | None:
        pointer_path = game_root / POINTER_NAME
        try:
            pointer_metadata = os.lstat(pointer_path)
        except FileNotFoundError:
            return None
        if (
            not stat.S_ISREG(pointer_metadata.st_mode)
            or pointer_metadata.st_nlink != 1
        ):
            raise ContiguousOrchestratorError(
                "promotion pointer is aliased or nonregular"
            )
        pointer_raw = _read_regular(
            pointer_path, maximum=MAX_JSON_BYTES
        )
        pointer = _strict_json(pointer_raw, label="promotion pointer")
        return self._validate_selected_pointer(game_root, pointer)

    def _validate_selected_pointer(
        self,
        game_root: Path,
        pointer: Mapping[str, Any],
    ) -> dict[str, Any]:
        pointer = dict(pointer)
        required = {
            "schema",
            "kind",
            "version",
            "game",
            "target_level",
            "parent_checkpoint_sha256",
            "checkpoint_sha256",
            "winning_source_tree_sha256",
            "host_receipt_sha256",
            "candidate_manifest_sha256",
            "supervisory_handoff_sha256",
            "supervisory_native_reproduction_receipt_sha256",
            "version_tree_sha256",
        }
        if (
            set(pointer) != required
            or pointer.get("schema") != POINTER_SCHEMA
            or isinstance(pointer.get("schema"), bool)
            or pointer.get("kind")
            != "arc_agi3_contiguous_current_version"
            or VERSION_RE.fullmatch(str(pointer.get("version"))) is None
            or pointer.get("game") != game_root.name
            or not isinstance(pointer.get("target_level"), int)
            or isinstance(pointer.get("target_level"), bool)
            or pointer["target_level"] <= 0
            or any(
                SHA256_RE.fullmatch(str(pointer.get(field))) is None
                for field in (
                    "parent_checkpoint_sha256",
                    "checkpoint_sha256",
                    "winning_source_tree_sha256",
                    "host_receipt_sha256",
                    "candidate_manifest_sha256",
                    "version_tree_sha256",
                )
            )
            or (
                pointer.get("supervisory_handoff_sha256")
                is None
            )
            != (
                pointer.get(
                    "supervisory_native_reproduction_receipt_sha256"
                )
                is None
            )
            or any(
                value is not None
                and SHA256_RE.fullmatch(str(value)) is None
                for value in (
                    pointer.get("supervisory_handoff_sha256"),
                    pointer.get(
                        "supervisory_native_reproduction_receipt_sha256"
                    ),
                )
            )
        ):
            raise ContiguousOrchestratorError(
                "promotion pointer schema is invalid"
            )
        version = game_root / VERSIONS_NAME / pointer["version"]
        Supervisor._validate_regular_tree(
            version, label="selected contiguous version"
        )
        subject = version / f"{pointer['game']}_legs"
        if (
            set(entry.name for entry in version.iterdir())
            != {subject.name}
            or subject.is_symlink()
            or not subject.is_dir()
        ):
            raise ContiguousOrchestratorError(
                "selected version is not an exact one-game wrapper"
            )
        if _tree_hash(version) != pointer["version_tree_sha256"]:
            raise ContiguousOrchestratorError(
                "selected contiguous version changed"
            )
        receipt_path = subject / HOST_RECEIPT_NAME
        if _sha256(
            receipt_raw := _read_regular(
                receipt_path, maximum=MAX_JSON_BYTES
            )
        ) != pointer["host_receipt_sha256"]:
            raise ContiguousOrchestratorError(
                "selected promotion receipt changed"
            )
        checkpoint = subject / Supervisor.CHECKPOINT_NAME
        if _sha256(
            _read_regular(checkpoint, maximum=MAX_JSON_BYTES)
        ) != pointer["checkpoint_sha256"]:
            raise ContiguousOrchestratorError(
                "selected checkpoint changed"
            )
        winning = subject / WINNING_SOURCE_NAME
        if Supervisor.validate_winning_source_tree(
            winning
        ) != pointer["winning_source_tree_sha256"]:
            raise ContiguousOrchestratorError(
                "selected winning source changed"
            )
        summaries = _validate_schema_v2_chain(
            version,
            game=pointer["game"],
            reached=pointer["target_level"],
        )
        receipt = _strict_json(
            receipt_raw, label="selected host promotion receipt"
        )
        self._validate_receipt(
            version=version,
            subject=subject,
            pointer=pointer,
            receipt=receipt,
            summaries=summaries,
        )
        return pointer

    @staticmethod
    def _commit_from_pointer(
        game_root: Path,
        pointer: Mapping[str, Any],
    ) -> Runner.PromotionCommit:
        version = game_root / VERSIONS_NAME / pointer["version"]
        subject = version / f"{pointer['game']}_legs"
        receipt = _strict_json(
            _read_regular(
                subject / HOST_RECEIPT_NAME, maximum=MAX_JSON_BYTES
            ),
            label="host promotion receipt",
        )
        return Runner.PromotionCommit(
            game=pointer["game"],
            from_level=pointer["target_level"] - 1,
            to_level=pointer["target_level"],
            parent_checkpoint_sha256=(
                pointer["parent_checkpoint_sha256"]
            ),
            checkpoint_path=str(
                subject / Supervisor.CHECKPOINT_NAME
            ),
            checkpoint_sha256=pointer["checkpoint_sha256"],
            exact_path=tuple(receipt["exact_path"]),
            promotion_receipt_sha256=pointer["host_receipt_sha256"],
            source_version_id=pointer["version"],
            source_tree_sha256=pointer[
                "winning_source_tree_sha256"
            ],
            supervisory_handoff_sha256=pointer[
                "supervisory_handoff_sha256"
            ],
            supervisory_native_reproduction_receipt_sha256=pointer[
                "supervisory_native_reproduction_receipt_sha256"
            ],
        )

    def _selective_frontier_import_from_pointer(
        self,
        game_root: Path,
        pointer: Mapping[str, Any],
    ) -> Runner.SelectiveFrontierImport:
        selected = self._validate_selected_pointer(
            game_root, pointer
        )
        version = game_root / VERSIONS_NAME / selected["version"]
        subject = version / f"{selected['game']}_legs"
        receipt_path = subject / HOST_RECEIPT_NAME
        receipt = _strict_json(
            _read_regular(receipt_path, maximum=MAX_JSON_BYTES),
            label="selective frontier host receipt",
        )
        inventory = Supervisor.authoritative_inventory()
        Supervisor.validate_inventory(inventory)
        if (
            selected["game"] not in inventory
            or receipt.get("authoritative_target")
            != inventory[selected["game"]]
            or selected["target_level"]
            >= inventory[selected["game"]]
        ):
            raise ContiguousOrchestratorError(
                "selected version is not an incomplete authoritative frontier"
            )
        return Runner.build_selective_frontier_import(
            game=selected["game"],
            reached=selected["target_level"],
            authoritative_target=inventory[selected["game"]],
            parent_checkpoint_sha256=(
                selected["parent_checkpoint_sha256"]
            ),
            checkpoint_path=str(subject / Supervisor.CHECKPOINT_NAME),
            checkpoint_sha256=selected["checkpoint_sha256"],
            source_path=str(subject / WINNING_SOURCE_NAME),
            source_tree_sha256=selected[
                "winning_source_tree_sha256"
            ],
            promotion_receipt_path=str(receipt_path),
            promotion_receipt_sha256=selected[
                "host_receipt_sha256"
            ],
            source_version_id=selected["version"],
            version_tree_sha256=selected["version_tree_sha256"],
            selected_pointer_sha256=hashlib.sha256(
                _canonical_json(selected)
            ).hexdigest(),
        )

    def issue_selective_frontier_import(
        self, game: str
    ) -> Runner.SelectiveFrontierImport:
        """Issue the exact current frontier from a separate read-only root."""

        game_root = self._import_game_root(game)
        first = self._current(game_root)
        if first is None:
            raise ContiguousOrchestratorError(
                "frontier import game has no selected version"
            )
        binding = self._selective_frontier_import_from_pointer(
            game_root, first
        )
        second = self._current(game_root)
        if second != first:
            raise ContiguousOrchestratorError(
                "frontier import pointer changed during issuance"
            )
        return self.verify_selective_frontier_import(binding)

    def verify_selective_frontier_import(
        self,
        binding: Runner.SelectiveFrontierImport,
    ) -> Runner.SelectiveFrontierImport:
        """Reopen a sealed imported version after its current pointer moves."""

        if not isinstance(binding, Runner.SelectiveFrontierImport):
            raise ContiguousOrchestratorError(
                "frontier import binding is not typed"
            )
        try:
            canonical = Runner.selective_frontier_import_from_dict(
                asdict(binding)
            )
        except Runner.ContiguousRunnerError as exc:
            raise ContiguousOrchestratorError(
                "frontier import binding is malformed"
            ) from exc
        game_root = self._import_game_root(canonical.game)
        version = (
            game_root
            / VERSIONS_NAME
            / canonical.source_version_id
        )
        subject = version / f"{canonical.game}_legs"
        if (
            Path(canonical.checkpoint_path)
            != subject / Supervisor.CHECKPOINT_NAME
            or Path(canonical.source_path)
            != subject / WINNING_SOURCE_NAME
            or Path(canonical.promotion_receipt_path)
            != subject / HOST_RECEIPT_NAME
        ):
            raise ContiguousOrchestratorError(
                "frontier import escaped its configured authority root"
            )
        receipt = _strict_json(
            _read_regular(
                subject / HOST_RECEIPT_NAME,
                maximum=MAX_JSON_BYTES,
            ),
            label="selective frontier host receipt",
        )
        pointer = self._pointer_for_version(
            version, receipt=receipt
        )
        if hashlib.sha256(
            _canonical_json(pointer)
        ).hexdigest() != canonical.selected_pointer_sha256:
            raise ContiguousOrchestratorError(
                "frontier import selected-pointer identity changed"
            )
        reopened = self._selective_frontier_import_from_pointer(
            game_root, pointer
        )
        if reopened != canonical:
            raise ContiguousOrchestratorError(
                "frontier import version differs from its binding"
            )
        return reopened

    def _selective_parent_binding(
        self, spec: Runner.AttemptSpec
    ) -> Runner.SelectiveFrontierImport:
        binding = self.selective_frontier_import
        if binding is None:
            binding = self.issue_selective_frontier_import(spec.game)
        else:
            binding = self.verify_selective_frontier_import(binding)
        if (
            spec.game != binding.game
            or spec.authoritative_target != binding.authoritative_target
            or spec.target_level != binding.reached + 1
            or spec.parent_checkpoint_path != binding.checkpoint_path
            or spec.parent_checkpoint_sha256
            != binding.checkpoint_sha256
            or spec.parent_source_path != binding.source_path
            or spec.parent_source_tree_sha256
            != binding.source_tree_sha256
            or spec.frontier_sha256 != binding.frontier_sha256
        ):
            raise ContiguousOrchestratorError(
                "candidate does not extend the exact imported frontier"
            )
        return binding

    @staticmethod
    def _matching_receipt(
        version: Path,
        *,
        spec: Runner.AttemptSpec,
        candidate: Runner.PromotionCandidate,
    ) -> dict[str, Any] | None:
        receipt_path = (
            version / f"{spec.game}_legs" / HOST_RECEIPT_NAME
        )
        if not receipt_path.is_file() or receipt_path.is_symlink():
            return None
        try:
            receipt = _strict_json(
                _read_regular(receipt_path, maximum=MAX_JSON_BYTES),
                label="host promotion receipt",
            )
        except ContiguousOrchestratorError:
            return None
        if (
            receipt.get("schema") == HOST_RECEIPT_SCHEMA
            and receipt.get("kind")
            == "arc_agi3_contiguous_schema_v2_promotion"
            and receipt.get("attempt_id") == spec.attempt_id
            and receipt.get("game") == spec.game
            and receipt.get("target_level") == spec.target_level
            and receipt.get("parent_checkpoint_sha256")
            == spec.parent_checkpoint_sha256
            and receipt.get("candidate_manifest_sha256")
            == candidate.candidate_manifest_sha256
            and receipt.get("supervisory_handoff_sha256")
            == candidate.supervisory_handoff_sha256
            and receipt.get(
                "supervisory_native_reproduction_receipt_sha256"
            )
            == candidate
            .supervisory_native_reproduction_receipt_sha256
            and receipt.get("version") == version.name
        ):
            return receipt
        return None

    def _recover_locked(
        self,
        game_root: Path,
        *,
        spec: Runner.AttemptSpec,
        candidate: Runner.PromotionCandidate,
    ) -> Runner.PromotionCommit | None:
        self._reconcile_locked(game_root)
        current = self._current(game_root)
        matches: list[tuple[Path, dict[str, Any]]] = []
        for version in sorted((game_root / VERSIONS_NAME).iterdir()):
            if (
                version.is_symlink()
                or not version.is_dir()
                or VERSION_RE.fullmatch(version.name) is None
            ):
                raise ContiguousOrchestratorError(
                    "version store contains an invalid entry"
                )
            receipt = self._matching_receipt(
                version, spec=spec, candidate=candidate
            )
            if receipt is not None:
                matches.append((version, receipt))
        if len(matches) > 1:
            raise ContiguousOrchestratorError(
                "promotion recovery is ambiguous across versions"
            )
        if not matches:
            return None
        version, receipt = matches[0]
        if current is not None and current["version"] == version.name:
            return self._commit_from_pointer(game_root, current)
        if current is None:
            if self.frontier_import_root is None:
                if spec.target_level != 1:
                    raise ContiguousOrchestratorError(
                        "recovery cannot skip a missing selected parent"
                    )
            else:
                self._selective_parent_binding(spec)
        elif (
            current["target_level"] != spec.target_level - 1
            or current["checkpoint_sha256"]
            != spec.parent_checkpoint_sha256
        ):
            raise ContiguousOrchestratorError(
                "promotion recovery conflicts with the selected lineage"
            )
        pointer = self._pointer_for_version(
            version, receipt=receipt
        )
        _replace_json(game_root / POINTER_NAME, pointer)
        selected = self._current(game_root)
        if selected != pointer:
            raise ContiguousOrchestratorError(
                "recovered pointer did not become durable"
            )
        return self._commit_from_pointer(game_root, selected)

    def recover(
        self,
        *,
        spec: Runner.AttemptSpec,
        candidate: Runner.PromotionCandidate,
    ) -> Runner.PromotionCommit | None:
        game_root = self._game_root(spec.game)
        descriptor = self._lock(game_root)
        try:
            return self._recover_locked(
                game_root, spec=spec, candidate=candidate
            )
        finally:
            self._unlock(descriptor)

    @staticmethod
    def _pointer_for_version(
        version: Path,
        *,
        receipt: Mapping[str, Any],
    ) -> dict[str, Any]:
        return {
            "schema": POINTER_SCHEMA,
            "kind": "arc_agi3_contiguous_current_version",
            "version": version.name,
            "game": receipt["game"],
            "target_level": receipt["target_level"],
            "parent_checkpoint_sha256":
                receipt["parent_checkpoint_sha256"],
            "checkpoint_sha256": receipt["checkpoint_sha256"],
            "winning_source_tree_sha256":
                receipt["winning_source_tree_sha256"],
            "host_receipt_sha256": _sha256(
                _read_regular(
                    version
                    / f"{receipt['game']}_legs"
                    / HOST_RECEIPT_NAME,
                    maximum=MAX_JSON_BYTES,
                )
            ),
            "candidate_manifest_sha256":
                receipt["candidate_manifest_sha256"],
            "supervisory_handoff_sha256":
                receipt["supervisory_handoff_sha256"],
            "supervisory_native_reproduction_receipt_sha256":
                receipt[
                    "supervisory_native_reproduction_receipt_sha256"
                ],
            "version_tree_sha256": _tree_hash(version),
        }

    def _copy_attempt_transcripts(
        self,
        *,
        spec: Runner.AttemptSpec,
        manifest_raw: bytes,
        replay: IsolatedReplayEvidence,
        attempt_evidence: AttemptEvidenceBundle,
        destination: Path,
    ) -> dict[str, str]:
        destination.mkdir(parents=True, mode=0o700)
        records: dict[str, str] = {}

        def admit(
            name: str,
            raw: bytes,
            *,
            evidence_kind: str,
            app_policy: Taint.AppServerScanPolicy | None = None,
        ) -> str:
            _require_no_live_secret(
                raw,
                secret_sentinels=self.secret_sentinels,
                label=f"retained transcript {name}",
            )
            target = destination / name
            digest = _write_new(target, raw)
            if evidence_kind == "app_server_jsonl":
                scan = Taint.scan_evidence(
                    target,
                    evidence_kind="app_server_jsonl",
                    app_server_policy=app_policy,
                )
            else:
                scan = Taint.scan_evidence(
                    target, evidence_kind=evidence_kind
                )
            if scan.hits:
                raise ContiguousOrchestratorError(
                    f"promotion transcript is tainted: {name}: {scan.hits}"
                )
            records[f"transcripts/{name}"] = digest
            return digest

        app_policy = _app_scan_policy(
            spec, secret_sentinels=self.secret_sentinels
        )
        admit(
            "candidate_manifest.json",
            manifest_raw,
            evidence_kind="candidate_output",
        )
        transcript_sources = (
            (
                "app_server.jsonl",
                Path(spec.app_server_transcript_path),
                "app_server_jsonl",
            ),
            (
                "arena_attempt.jsonl",
                Path(spec.host_transcript_path),
                "backend_jsonl",
            ),
            (
                "arena_source_replay.jsonl",
                Path(replay.arena_transcript_path),
                "backend_jsonl",
            ),
        )
        for name, source, kind in transcript_sources:
            raw = _read_regular(
                source, maximum=MAX_TRANSCRIPT_BYTES
            )
            admit(
                name,
                raw,
                evidence_kind=kind,
                app_policy=app_policy if kind == "app_server_jsonl" else None,
            )

        collection = attempt_evidence.collection
        if (
            collection.host_transcript_path
            != spec.host_transcript_path
            or collection.app_server_transcript_path
            != spec.app_server_transcript_path
        ):
            raise ContiguousOrchestratorError(
                "journal collection references another attempt transcript"
            )
        collection_value = Runner._backend_collection_to_dict(collection)
        candidate_outcome_raw = _read_regular(
            Path(spec.output_dir) / "worker_outcome.json",
            maximum=MAX_JSON_BYTES,
        )
        if (
            _sha256(candidate_outcome_raw)
            != collection.worker_outcome_sha256
        ):
            raise ContiguousOrchestratorError(
                "candidate worker outcome changed after collection"
            )
        admit(
            "candidate_worker_outcome.json",
            candidate_outcome_raw,
            evidence_kind="candidate_output",
        )
        copied_receipts: dict[str, str] = {}
        skip_paths = {
            "host_transcript_path",
            "app_server_transcript_path",
            "container_stdout_path",
            "container_stderr_path",
        }
        for field in sorted(
            name
            for name in collection_value
            if name.endswith("_path") and name not in skip_paths
        ):
            digest_field = field[:-5] + "_sha256"
            expected = collection_value.get(digest_field)
            if not isinstance(expected, str):
                raise ContiguousOrchestratorError(
                    f"collection path lacks a digest: {field}"
                )
            raw = _read_regular(
                Path(collection_value[field]),
                maximum=MAX_TRANSCRIPT_BYTES,
            )
            if _sha256(raw) != expected:
                raise ContiguousOrchestratorError(
                    f"collection receipt changed: {field}"
                )
            name = f"collection_{field[:-5]}.json"
            copied_receipts[field] = admit(
                name, raw, evidence_kind="candidate_output"
            )

        stdout_raw = _read_regular(
            Path(collection.container_stdout_path),
            maximum=MAX_TRANSCRIPT_BYTES,
            allow_empty=True,
        )
        stderr_raw = _read_regular(
            Path(collection.container_stderr_path),
            maximum=MAX_TRANSCRIPT_BYTES,
            allow_empty=True,
        )
        if (
            _sha256(stdout_raw) != collection.container_stdout_sha256
            or _sha256(stderr_raw) != collection.container_stderr_sha256
        ):
            raise ContiguousOrchestratorError(
                "candidate container streams changed after collection"
            )
        stream_envelope = {
            "schema": SCHEMA,
            "kind": "arc_agi3_contiguous_candidate_streams",
            "stdout": stdout_raw.decode("utf-8"),
            "stdout_sha256": collection.container_stdout_sha256,
            "stderr": stderr_raw.decode("utf-8"),
            "stderr_sha256": collection.container_stderr_sha256,
        }
        admit(
            "candidate_container_streams.json",
            _canonical_json(stream_envelope) + b"\n",
            evidence_kind="candidate_output",
        )

        replay_outcome = _read_regular(
            Path(replay.worker_outcome_path), maximum=MAX_JSON_BYTES
        )
        replay_stdout = _read_regular(
            Path(replay.stdout_path),
            maximum=MAX_TRANSCRIPT_BYTES,
            allow_empty=True,
        )
        replay_stderr = _read_regular(
            Path(replay.stderr_path),
            maximum=MAX_TRANSCRIPT_BYTES,
            allow_empty=True,
        )
        if (
            _sha256(replay_outcome) != replay.worker_outcome_sha256
            or _sha256(replay_stdout) != replay.stdout_sha256
            or _sha256(replay_stderr) != replay.stderr_sha256
        ):
            raise ContiguousOrchestratorError(
                "isolated replay evidence changed before publication"
            )
        admit(
            "replay_worker_outcome.json",
            replay_outcome,
            evidence_kind="candidate_output",
        )
        replay_stream_envelope = {
            "schema": SCHEMA,
            "kind": "arc_agi3_contiguous_replay_streams",
            "stdout": replay_stdout.decode("utf-8"),
            "stdout_sha256": replay.stdout_sha256,
            "stderr": replay_stderr.decode("utf-8"),
            "stderr_sha256": replay.stderr_sha256,
        }
        admit(
            "replay_container_streams.json",
            _canonical_json(replay_stream_envelope) + b"\n",
            evidence_kind="candidate_output",
        )

        replay_public = _public_replay_evidence(replay)
        certification = {
            "schema": SCHEMA,
            "kind": "arc_agi3_contiguous_replay_certification",
            "game": spec.game,
            "target_level": spec.target_level,
            "attempt_id": spec.attempt_id,
            "candidate_manifest_sha256":
                _sha256(manifest_raw),
            "isolated_replay": replay_public,
            "attempt_evidence": {
                "collected_sequence":
                    attempt_evidence.collected_sequence,
                "collected_event_sha256":
                    attempt_evidence.collected_event_sha256,
                "teardown_sequence":
                    attempt_evidence.teardown_sequence,
                "teardown_event_sha256":
                    attempt_evidence.teardown_event_sha256,
                "result_sequence":
                    attempt_evidence.result_sequence,
                "result_event_sha256":
                    attempt_evidence.result_event_sha256,
                "journal_prefix":
                    list(attempt_evidence.journal_prefix),
                "journal_prefix_sha256":
                    attempt_evidence.journal_prefix_sha256,
                "journal_genesis_sha256":
                    attempt_evidence.journal_genesis_sha256,
                "teardown": asdict(attempt_evidence.teardown),
                "collection_receipts_sha256": copied_receipts,
                "output_tree_sha256":
                    collection.output_tree_sha256,
                "worker_outcome_sha256":
                    collection.worker_outcome_sha256,
                "token_usage_receipt_sha256":
                    collection.token_usage_receipt_sha256,
                "final_thread_binding_sha256":
                    collection.final_thread_binding_sha256,
                "final_transcript_chain_sha256":
                    collection.final_transcript_chain_sha256,
                "bridge_export_receipt_sha256":
                    collection.bridge_export_receipt_sha256,
                "secret_scan_receipt_sha256":
                    collection.secret_scan_receipt_sha256,
                "taint_scan_receipt_sha256":
                    collection.taint_scan_receipt_sha256,
                "app_server_state_tree_sha256":
                    collection.app_server_state_tree_sha256,
            },
        }
        admit(
            "certification.json",
            _canonical_json(certification) + b"\n",
            evidence_kind="candidate_output",
        )
        return records

    def _build_version(
        self,
        *,
        game_root: Path,
        version_id: str,
        spec: Runner.AttemptSpec,
        candidate: Runner.PromotionCandidate,
        parent: Supervisor.TrustedCheckpoint,
        manifest_raw: bytes,
        manifest: Mapping[str, Any],
        source_payloads: Mapping[str, bytes],
        replay: IsolatedReplayEvidence,
        attempt_evidence: AttemptEvidenceBundle,
        exact_path: Sequence[Any],
    ) -> tuple[Path, dict[str, Any]]:
        wrapper = game_root / STAGING_NAME / version_id
        game_stage = wrapper / f"{spec.game}_legs"
        if wrapper.exists() or wrapper.is_symlink():
            raise ContiguousOrchestratorError(
                "promotion staging identity collided"
            )
        game_stage.mkdir(parents=True, mode=0o700)
        try:
            current = self._current(game_root)
            if spec.target_level > 1:
                if current is None:
                    imported = self._selective_parent_binding(spec)
                    parent_version = Path(
                        imported.checkpoint_path
                    ).parent
                else:
                    parent_version = (
                        game_root / VERSIONS_NAME / current["version"]
                        / f"{spec.game}_legs"
                    )
                shutil.copytree(
                    parent_version / "promotion_evidence",
                    game_stage / "promotion_evidence",
                    symlinks=True,
                )
                Supervisor._validate_regular_tree(
                    game_stage / "promotion_evidence",
                    label="copied promotion evidence",
                )
            evidence = (
                game_stage
                / "promotion_evidence"
                / f"level_{spec.target_level:02d}"
            )
            files_dir = evidence / "files"
            transcripts_dir = evidence / "transcripts"
            audits_dir = evidence / "audits"
            for path in (files_dir, transcripts_dir, audits_dir):
                path.mkdir(parents=True, mode=0o700)
            source_names = _validate_flat_source_payloads(
                source_payloads
            )
            for name in source_names:
                _write_new(files_dir / name, source_payloads[name])
                _write_new(game_stage / name, source_payloads[name])
            parent_payloads = {
                entry.name: _read_regular(
                    entry,
                    maximum=SourceSchema.MAX_FILE_BYTES,
                    allow_empty=True,
                )
                for entry in Path(spec.parent_source_path).iterdir()
                if entry.is_file()
            }
            _validate_flat_source_payloads(parent_payloads)
            before_description = _source_description_bytes(
                parent_payloads
            )
            after_description = _source_description_bytes(
                source_payloads
            )
            (
                marginal,
                before_description_by_file,
                after_description_by_file,
            ) = _marginal_description_growth(
                parent_payloads, source_payloads
            )
            records = [dict(record) for record in parent.records]
            records.append(
                {
                    "level": spec.target_level,
                    "marginal_C": marginal,
                    "reached": True,
                }
            )
            checkpoint = {
                "game": spec.game,
                "reached": spec.target_level,
                "total_marginal_C":
                    parent.total_marginal_C + marginal,
                "records": records,
                "final_path": list(exact_path),
                "validated": True,
            }
            checkpoint_raw = _canonical_json(checkpoint) + b"\n"
            checkpoint_sha256 = _write_new(
                files_dir / Supervisor.CHECKPOINT_NAME,
                checkpoint_raw,
            )
            _write_new(
                game_stage / Supervisor.CHECKPOINT_NAME,
                checkpoint_raw,
            )
            promoted = {
                path.name: _sha256(
                    _read_regular(
                        path,
                        maximum=max(
                            SourceSchema.MAX_FILE_BYTES,
                            MAX_JSON_BYTES,
                        ),
                        allow_empty=True,
                    )
                )
                for path in sorted(files_dir.iterdir())
            }
            transcript_hashes = self._copy_attempt_transcripts(
                spec=spec,
                manifest_raw=manifest_raw,
                replay=replay,
                attempt_evidence=attempt_evidence,
                destination=transcripts_dir,
            )
            _scan_primary_files(
                files_dir,
                sorted(promoted),
                secret_sentinels=self.secret_sentinels,
            )
            tools = _tool_hashes()
            primary_checked = {
                **{
                    f"files/{name}": digest
                    for name, digest in promoted.items()
                },
                **transcript_hashes,
            }
            taint = {
                "schema": 1,
                "kind": "taint_audit",
                "game": spec.game,
                "level": spec.target_level,
                "scanner_sha256": tools["scanner"],
                "checked_files_sha256": primary_checked,
                "verdict": "PASS",
                "findings": [],
            }
            release_sources = sorted(source_names)
            release_source_tree_sha256 = Release._json_sha256(
                {name: promoted[name] for name in release_sources}
            )
            exact_path_sha256 = Release._json_sha256(
                list(exact_path)
            )
            release_parent_sha = (
                None
                if spec.target_level == 1
                else spec.parent_checkpoint_sha256
            )
            replay_base = {
                "schema": 1,
                "game": spec.game,
                "target_level": spec.target_level,
                "frontier_parent_level": spec.target_level - 1,
                "parent_checkpoint_sha256": release_parent_sha,
                "checkpoint_sha256": checkpoint_sha256,
                "winning_source_tree_sha256":
                    release_source_tree_sha256,
                "exact_path_sha256": exact_path_sha256,
                "action_count": len(exact_path),
                "observed_reached": spec.target_level,
                "engine_sha256": tools["engine"],
                "result": "PASS",
            }
            path_audit = {
                **replay_base,
                "kind": "path_replay",
            }
            source_audit = {
                **replay_base,
                "kind": "source_replay",
            }
            # Both replay authorities above are fail-closed on the public
            # action protocol.  The isolated source replay is admitted only
            # after the host-owned Arena RPC session closes cleanly; an
            # invalid call poisons that session even if solver code catches
            # the local exception.  ``_path_replay`` independently executes
            # the exact path through ``gkm_arena``, whose violation latch is
            # shared by the root Arena and all clones.  Bind those observed
            # successes into the schema-v2 action-protocol audit rather than
            # merely filling the release manifest with an asserted PASS.
            action_protocol_audit = {
                "schema": 1,
                "kind": "action_protocol_audit",
                "game": spec.game,
                "target_level": spec.target_level,
                "checkpoint_sha256": checkpoint_sha256,
                "exact_path_sha256": exact_path_sha256,
                "action_count": len(exact_path),
                "runtime_enforcement":
                    "shared_violation_latch_across_root_and_clones",
                "source_protocol_latch": "PASS",
                "path_protocol_latch": "PASS",
                "engine_sha256": tools["engine"],
                "result": "PASS",
            }
            audit_values = {
                "taint": taint,
                "action_protocol": action_protocol_audit,
                "path_replay": path_audit,
                "source_replay": source_audit,
            }
            audit_hashes: dict[str, str] = {}
            for name, value in audit_values.items():
                relative = Release.AUDIT_PATHS[name]
                audit_hashes[name] = _write_new_json(
                    evidence / relative, value
                )
            hash_checked = {
                **primary_checked,
                Release.AUDIT_PATHS["taint"]:
                    audit_hashes["taint"],
                Release.AUDIT_PATHS["action_protocol"]:
                    audit_hashes["action_protocol"],
                Release.AUDIT_PATHS["path_replay"]:
                    audit_hashes["path_replay"],
                Release.AUDIT_PATHS["source_replay"]:
                    audit_hashes["source_replay"],
            }
            hash_audit = {
                "schema": 1,
                "kind": "hash_audit",
                "game": spec.game,
                "level": spec.target_level,
                "hasher_sha256": tools["hasher"],
                "checked_files_sha256": hash_checked,
                "result": "PASS",
            }
            audit_hashes["hash"] = _write_new_json(
                evidence / Release.AUDIT_PATHS["hash"],
                hash_audit,
            )
            parent_manifest: dict[str, str] | None = None
            if spec.target_level > 1:
                prior = (
                    game_stage
                    / "promotion_evidence"
                    / f"level_{spec.target_level - 1:02d}"
                    / "manifest.json"
                )
                parent_manifest = {
                    "path": (
                        "promotion_evidence/"
                        f"level_{spec.target_level - 1:02d}/"
                        "manifest.json"
                    ),
                    "sha256": _sha256(
                        _read_regular(prior, maximum=MAX_JSON_BYTES)
                    ),
                }
            boundary_manifest = {
                "schema": Release.BOUNDARY_MANIFEST_SCHEMA,
                "game": spec.game,
                "level": spec.target_level,
                "frontier": {
                    "parent_level": spec.target_level - 1,
                    "target_level": spec.target_level,
                    "parent_checkpoint_sha256": release_parent_sha,
                },
                "parent_manifest": parent_manifest,
                "promoted_files_sha256": promoted,
                "winning_source_files": release_sources,
                "transcripts": [
                    {"path": path, "sha256": digest}
                    for path, digest in sorted(
                        transcript_hashes.items()
                    )
                ],
                "audits": {
                    name: {
                        "path": Release.AUDIT_PATHS[name],
                        "sha256": audit_hashes[name],
                    }
                    for name in Release.AUDIT_PATHS
                },
            }
            boundary_manifest_sha256 = _write_new_json(
                evidence / "manifest.json", boundary_manifest
            )
            summaries = _validate_schema_v2_chain(
                wrapper,
                game=spec.game,
                reached=spec.target_level,
            )
            if (
                summaries[-1]["checkpoint_sha256"]
                != checkpoint_sha256
                or summaries[-1]["manifest_sha256"]
                != boundary_manifest_sha256
            ):
                raise ContiguousOrchestratorError(
                    "schema-v2 summary differs from staged boundary"
                )
            winning = game_stage / WINNING_SOURCE_NAME
            winning.mkdir(mode=0o700)
            for name in source_names:
                _write_new(winning / name, source_payloads[name])
            winning_source_tree_sha256 = (
                Supervisor.validate_winning_source_tree(winning)
            )
            publication_subject_tree_sha256 = _tree_hash(game_stage)
            receipt = {
                "schema": HOST_RECEIPT_SCHEMA,
                "kind": "arc_agi3_contiguous_schema_v2_promotion",
                "version": version_id,
                "campaign_id": spec.campaign_id,
                "generation_id": spec.generation_id,
                "attempt_id": spec.attempt_id,
                "game": spec.game,
                "target_level": spec.target_level,
                "authoritative_target": spec.authoritative_target,
                "parent_checkpoint_sha256":
                    spec.parent_checkpoint_sha256,
                "candidate_manifest_sha256":
                    candidate.candidate_manifest_sha256,
                "supervisory_handoff_sha256":
                    candidate.supervisory_handoff_sha256,
                "supervisory_native_reproduction_receipt_sha256":
                    candidate
                    .supervisory_native_reproduction_receipt_sha256,
                "candidate_path_sha256": _json_sha256(
                    manifest["candidate_path"]
                ),
                "probe_isolation_mode":
                    candidate.probe_isolation_mode,
                "probe_isolation_evidence_sha256":
                    candidate.probe_isolation_evidence_sha256,
                "probe_result_authority": "hypothesis_only",
                "checkpoint_sha256": checkpoint_sha256,
                "exact_path": list(exact_path),
                "exact_path_sha256": exact_path_sha256,
                "schema_v2_manifest_path": (
                    "promotion_evidence/"
                    f"level_{spec.target_level:02d}/manifest.json"
                ),
                "schema_v2_manifest_sha256":
                    boundary_manifest_sha256,
                "schema_v2_audits_sha256": audit_hashes,
                "winning_source_tree_sha256":
                    winning_source_tree_sha256,
                "release_source_tree_sha256":
                    release_source_tree_sha256,
                "source_description_bytes_before":
                    before_description,
                "source_description_bytes_after":
                    after_description,
                "source_description_metric":
                    "positive_per_file_utf8_bytes_v1",
                "source_description_bytes_before_by_file":
                    before_description_by_file,
                "source_description_bytes_after_by_file":
                    after_description_by_file,
                "same_size_rewrite_novelty":
                    "not_measured_use_posthoc_normalized_ast",
                "marginal_C": marginal,
                "isolated_source_replay":
                    _public_replay_evidence(replay),
                "attempt_evidence_event_sha256": {
                    "collected":
                        attempt_evidence.collected_event_sha256,
                    "teardown":
                        attempt_evidence.teardown_event_sha256,
                    "result": attempt_evidence.result_event_sha256,
                },
                "path_replay_from_zero": "PASS",
                "source_replay_from_zero": "PASS",
                "taint_scan": "PASS",
                "publication_subject_tree_sha256":
                    publication_subject_tree_sha256,
                "control_tools_sha256": tools,
            }
            _write_new_json(
                game_stage / HOST_RECEIPT_NAME, receipt
            )
            _fsync_directory(game_stage)
            _fsync_directory(wrapper)
            return wrapper, receipt
        except BaseException:
            if wrapper.exists():
                quarantine = (
                    game_root / QUARANTINE_NAME / version_id
                )
                try:
                    os.replace(wrapper, quarantine)
                    _fsync_directory(game_root / QUARANTINE_NAME)
                except OSError:
                    pass
            raise

    def commit(
        self,
        *,
        spec: Runner.AttemptSpec,
        candidate: Runner.PromotionCandidate,
    ) -> Runner.PromotionCommit:
        try:
            attempt_evidence = _load_attempt_evidence(spec)
            if attempt_evidence.collection.result.candidate != candidate:
                raise ContiguousOrchestratorError(
                    "journal-authenticated candidate differs from commit input"
                )
            output_root, manifest_raw, manifest, payloads = (
                _load_candidate(spec, candidate)
            )
            del output_root
            parent = Supervisor.load_trusted_checkpoint(
                Path(spec.parent_checkpoint_path),
                expected_game=spec.game,
                authoritative_target=spec.authoritative_target,
            )
            if parent.reached != spec.target_level - 1:
                raise ContiguousOrchestratorError(
                    "promotion parent is not the exact K frontier"
                )
            candidate_exact = _exact_path(
                spec.game,
                manifest["candidate_path"],
                spec.target_level,
            )
            if (
                _normalize_actions(manifest["candidate_path"])
                != candidate_exact
            ):
                raise ContiguousOrchestratorError(
                    "candidate path continues past its first exact boundary"
                )
            replay = self.replay_executor.replay_from_zero(
                spec=spec, source_payloads=payloads
            )
            if (
                replay.game != spec.game
                or replay.target_level != spec.target_level
                or replay.observed_level != spec.target_level
                or replay.source_tree_sha256
                != _source_tree_sha256(payloads)
                or list(replay.observed_path) != candidate_exact
                or list(replay.exact_path) != candidate_exact
            ):
                raise ContiguousOrchestratorError(
                    "isolated source replay differs from the candidate boundary"
                )
            _path_replay(spec.game, spec.target_level, candidate_exact)
        except (
            ContiguousOrchestratorError,
            Supervisor.SupervisorContractError,
        ) as exc:
            raise Runner.PromotionRejected(str(exc)) from exc

        game_root = self._game_root(spec.game)
        descriptor = self._lock(game_root)
        final: Path | None = None
        try:
            recovered = self._recover_locked(
                game_root, spec=spec, candidate=candidate
            )
            if recovered is not None:
                return recovered
            current = self._current(game_root)
            if current is None:
                if self.frontier_import_root is not None:
                    try:
                        self._selective_parent_binding(spec)
                    except ContiguousOrchestratorError as exc:
                        raise Runner.PromotionRejected(str(exc)) from exc
                elif spec.target_level != 1:
                    raise Runner.PromotionRejected(
                        "candidate lacks its exact selected parent"
                    )
            elif spec.target_level == 1:
                raise Runner.PromotionRejected(
                    "L1 candidate conflicts with an existing selected version"
                )
            elif (
                current["target_level"] != spec.target_level - 1
                or current["checkpoint_sha256"]
                != spec.parent_checkpoint_sha256
                or Path(spec.parent_checkpoint_path)
                != (
                    game_root
                    / VERSIONS_NAME
                    / current["version"]
                    / f"{spec.game}_legs"
                    / Supervisor.CHECKPOINT_NAME
                )
            ):
                raise Runner.PromotionRejected(
                    "candidate does not extend the exact selected parent"
                )
            version_id = uuid.uuid4().hex
            intent = self._intent_document(
                version_id=version_id,
                spec=spec,
                candidate=candidate,
            )
            intent_path = (
                game_root / INTENTS_NAME / f"{version_id}.json"
            )
            _write_new_json(intent_path, intent)
            os.chmod(intent_path, 0o400, follow_symlinks=False)
            _fsync_directory(game_root / INTENTS_NAME)
            version_stage, receipt = self._build_version(
                game_root=game_root,
                version_id=version_id,
                spec=spec,
                candidate=candidate,
                parent=parent,
                manifest_raw=manifest_raw,
                manifest=manifest,
                source_payloads=payloads,
                replay=replay,
                attempt_evidence=attempt_evidence,
                exact_path=candidate_exact,
            )
            final = game_root / VERSIONS_NAME / version_id
            os.replace(version_stage, final)
            _seal_tree(final)
            _fsync_directory(game_root / VERSIONS_NAME)
            if self.fault_at == "after_version":
                raise OSError("injected loss after durable version")
            pointer = self._pointer_for_version(
                final, receipt=receipt
            )
            _replace_json(game_root / POINTER_NAME, pointer)
            if self.fault_at == "after_pointer":
                raise OSError("injected loss after durable pointer")
            selected = self._current(game_root)
            if selected != pointer:
                raise ContiguousOrchestratorError(
                    "published pointer failed post-commit validation"
                )
            return self._commit_from_pointer(game_root, selected)
        finally:
            self._unlock(descriptor)


def _read_only_promotion_records(
    promotion_root: Path,
    *,
    campaign_id: str,
    lane_boundaries: object,
) -> tuple[list[dict[str, Any]], int]:
    """Reopen every runner-claimed boundary without constructing a gate."""

    root = Path(promotion_root)
    if not root.is_absolute():
        raise ContiguousOrchestratorError(
            "promotion audit root must be absolute"
        )
    _require_owned_directory(root, label="promotion audit root")
    if not isinstance(lane_boundaries, list) or not lane_boundaries:
        raise ContiguousOrchestratorError(
            "runner audit has no lane boundaries"
        )
    lanes: dict[str, dict[str, Any]] = {}
    for value in lane_boundaries:
        if (
            not isinstance(value, dict)
            or set(value)
            != {
                "game",
                "target",
                "reached",
                "checkpoint_path",
                "checkpoint_sha256",
                "source_path",
                "source_tree_sha256",
            }
            or re.fullmatch(r"[a-z0-9]{4}", str(value.get("game")))
            is None
            or not isinstance(value.get("target"), int)
            or isinstance(value.get("target"), bool)
            or not isinstance(value.get("reached"), int)
            or isinstance(value.get("reached"), bool)
            or not 0 <= value["reached"] <= value["target"]
            or any(
                SHA256_RE.fullmatch(str(value.get(field))) is None
                for field in (
                    "checkpoint_sha256",
                    "source_tree_sha256",
                )
            )
            or not isinstance(value.get("checkpoint_path"), str)
            or not isinstance(value.get("source_path"), str)
            or value["game"] in lanes
        ):
            raise ContiguousOrchestratorError(
                "runner audit lane boundary is malformed"
            )
        lanes[value["game"]] = value

    for entry in root.iterdir():
        if (
            entry.is_symlink()
            or not entry.is_dir()
            or entry.name not in lanes
        ):
            raise ContiguousOrchestratorError(
                "promotion audit root has an unexpected game entry"
            )

    # Bypass the mutating ProductionPromotionGate constructor.  _current is a
    # pure verifier: it descriptor-opens and hashes the selected sealed
    # version, replays the full schema-v2 chain, and validates the host receipt.
    verifier = object.__new__(ProductionPromotionGate)
    records: list[dict[str, Any]] = []
    verified_boundaries = 0
    for game, lane in sorted(lanes.items()):
        game_root = root / game
        if not game_root.exists():
            if lane["reached"] != 0:
                raise ContiguousOrchestratorError(
                    f"{game} runner boundary has no promotion store"
                )
            continue
        _require_owned_directory(
            game_root, label=f"{game} promotion audit root"
        )
        pointer = ProductionPromotionGate._current(verifier, game_root)
        if pointer is None:
            if lane["reached"] != 0:
                raise ContiguousOrchestratorError(
                    f"{game} runner boundary has no selected promotion"
                )
            continue
        if lane["reached"] == 0:
            raise ContiguousOrchestratorError(
                f"{game} has a promotion absent from runner state"
            )
        commit = ProductionPromotionGate._commit_from_pointer(
            game_root, pointer
        )
        subject = (
            game_root
            / VERSIONS_NAME
            / pointer["version"]
            / f"{game}_legs"
        )
        receipt = _strict_json(
            _read_regular(
                subject / HOST_RECEIPT_NAME,
                maximum=MAX_JSON_BYTES,
            ),
            label=f"{game} unified host receipt",
        )
        if (
            receipt.get("campaign_id") != campaign_id
            or pointer["target_level"] != lane["reached"]
            or commit.to_level != lane["reached"]
            or commit.checkpoint_path != lane["checkpoint_path"]
            or commit.checkpoint_sha256
            != lane["checkpoint_sha256"]
            or str(
                Runner.ContiguousCampaignRunner._commit_source_path(
                    commit
                )
            )
            != lane["source_path"]
            or commit.source_tree_sha256
            != lane["source_tree_sha256"]
        ):
            raise ContiguousOrchestratorError(
                f"{game} promotion differs from runner boundary"
            )
        records.append(
            {
                "game": game,
                "reached": lane["reached"],
                "version": pointer["version"],
                "checkpoint_sha256": pointer["checkpoint_sha256"],
                "source_tree_sha256":
                    pointer["winning_source_tree_sha256"],
                "host_receipt_sha256":
                    pointer["host_receipt_sha256"],
                "candidate_manifest_sha256":
                    pointer["candidate_manifest_sha256"],
            }
        )
        verified_boundaries += lane["reached"]
    return records, verified_boundaries


def audit_contiguous_campaign_unified(
    *,
    campaign_root: Path,
    scheduler_audit_receipt_path: Path,
    runner_state_receipt: object,
    promotion_root: Path,
    secret_sentinels: tuple[str, ...] = (),
    controller_state_canaries: tuple[Taint.LiveCanary, ...] = (),
) -> dict[str, Any]:
    """Require scheduler policy, full runner replay, and promotion evidence."""

    campaign = Path(campaign_root).resolve()
    scheduler_path = Path(scheduler_audit_receipt_path).resolve()
    promotions = Path(promotion_root).resolve()
    try:
        runner_receipt = Runner.verify_runner_state_audit(
            runner_state_receipt,
            campaign_root=campaign,
            secret_sentinels=secret_sentinels,
            controller_state_canaries=controller_state_canaries,
        )
        promotion_replay_receipt: dict[str, Any] | None = None
        if runner_receipt.get("complete"):
            scheduler_candidate = json.loads(
                _read_regular(
                    scheduler_path, maximum=MAX_JSON_BYTES
                )
            )
            if not isinstance(scheduler_candidate, dict):
                raise ContiguousOrchestratorError(
                    "pre-retention scheduler receipt is malformed"
                )
            scheduler_sha256 = scheduler_candidate.get(
                "receipt_sha256"
            )
            if (
                not isinstance(scheduler_sha256, str)
                or SHA256_RE.fullmatch(scheduler_sha256) is None
            ):
                raise ContiguousOrchestratorError(
                    "pre-retention scheduler receipt hash is malformed"
                )
            promotion_replay_receipt = (
                _verify_terminal_promotion_replay_audit(
                    scheduler_path.parent
                    / TERMINAL_PROMOTION_REPLAY_AUDIT_NAME,
                    campaign_root=campaign,
                    promotion_root=promotions,
                    runner_state_receipt=runner_receipt,
                )
            )
            prerequisite_audits = {
                "promotion_replay":
                    promotion_replay_receipt["receipt_sha256"],
                "scheduler": scheduler_sha256,
            }
            terminal_retention = (
                Runner.audit_terminal_attempt_retention(
                    campaign,
                    runner_receipt,
                    secret_sentinels=secret_sentinels,
                    controller_state_canaries=(
                        controller_state_canaries
                    ),
                    pre_cleanup_audits=prerequisite_audits,
                )
            )
            scheduler_receipt = (
                Scheduler.verify_pre_retention_audit_receipt(
                    campaign,
                    scheduler_path,
                    expected_receipt_sha256=scheduler_sha256,
                )
            )
        else:
            scheduler_receipt = Scheduler.verify_audit_receipt(
                campaign, scheduler_path
            )
            terminal_retention = (
                Runner.audit_terminal_attempt_retention(
                    campaign,
                    runner_receipt,
                    secret_sentinels=secret_sentinels,
                    controller_state_canaries=(
                        controller_state_canaries
                    ),
                )
            )
        expected_retention_status = (
            "PASS" if runner_receipt.get("complete") else "NOT_REQUIRED"
        )
        if terminal_retention.get("status") != expected_retention_status:
            raise ContiguousOrchestratorError(
                "terminal attempt retention audit has wrong status"
            )
        scheduler_summary = scheduler_receipt.get("summary")
        if not isinstance(scheduler_summary, dict):
            raise ContiguousOrchestratorError(
                "scheduler audit summary is malformed"
            )
        live_units = sum(
            int(item["units"])
            for item in runner_receipt.get(
                "live_budget_reservations", ()
            )
        ) if "live_budget_reservations" in runner_receipt else None
        # The runner receipt deliberately exposes only the stable audit
        # projection.  All shared fields must agree exactly; solved status is
        # never inherited from the scheduler receipt alone.
        if (
            scheduler_receipt.get("campaign_root") != str(campaign)
            or runner_receipt.get("campaign_root") != str(campaign)
            or scheduler_receipt.get("policy_sha256")
            != runner_receipt.get("scheduler_policy_sha256")
            or scheduler_receipt.get("journal_events")
            != runner_receipt.get("journal_event_count")
            or scheduler_receipt.get("journal_head_sequence")
            != runner_receipt.get("journal_head_sequence")
            or scheduler_receipt.get("journal_head_digest")
            != runner_receipt.get("journal_head_digest")
            or scheduler_summary.get("journal_prefix")
            != runner_receipt.get("journal_prefix")
            or scheduler_summary.get("policy_promoted_levels")
            != runner_receipt.get("solved_levels")
            or scheduler_summary.get("total_levels")
            != runner_receipt.get("total_levels")
            or (
                live_units is not None
                and scheduler_summary.get("live_reservation_units")
                != live_units
            )
        ):
            raise ContiguousOrchestratorError(
                "scheduler and runner audit receipts disagree"
            )
        promotion_records, verified_boundaries = (
            _read_only_promotion_records(
                promotions,
                campaign_id=str(runner_receipt["campaign_id"]),
                lane_boundaries=runner_receipt["lane_boundaries"],
            )
        )
        if (
            promotion_replay_receipt is not None
            and (
                promotion_replay_receipt.get("promotion_records")
                != promotion_records
                or promotion_replay_receipt.get(
                    "verified_promotion_boundaries"
                )
                != verified_boundaries
            )
        ):
            raise ContiguousOrchestratorError(
                "post-retention promotion/replay evidence changed"
            )
        if verified_boundaries != runner_receipt["solved_levels"]:
            raise ContiguousOrchestratorError(
                "runner solved count lacks full promotion evidence"
            )
        body = {
            "schema": UNIFIED_AUDIT_SCHEMA,
            "kind": "arc_agi3_contiguous_unified_audit",
            "status": "PASS",
            "campaign_root": str(campaign),
            "promotion_root": str(promotions),
            "scheduler_audit_receipt_sha256":
                scheduler_receipt["receipt_sha256"],
            "runner_state_receipt_sha256":
                runner_receipt["receipt_sha256"],
            "terminal_retention_receipt_sha256":
                terminal_retention["receipt_sha256"],
            "pre_retention_promotion_replay_receipt_sha256": (
                promotion_replay_receipt["receipt_sha256"]
                if promotion_replay_receipt is not None
                else None
            ),
            "campaign_id": runner_receipt["campaign_id"],
            "inventory_sha256": runner_receipt["inventory_sha256"],
            "scheduler_policy_sha256":
                runner_receipt["scheduler_policy_sha256"],
            "operator_configuration_sha256":
                runner_receipt["operator_configuration_sha256"],
            "journal_event_count":
                runner_receipt["journal_event_count"],
            "journal_head_sequence":
                runner_receipt["journal_head_sequence"],
            "journal_head_digest":
                runner_receipt["journal_head_digest"],
            "solved_levels": verified_boundaries,
            "verified_promotion_boundaries": verified_boundaries,
            "total_levels": runner_receipt["total_levels"],
            "complete": (
                runner_receipt["complete"]
                and verified_boundaries
                == runner_receipt["total_levels"]
            ),
            "promotion_records": promotion_records,
            "promotion_records_sha256":
                _json_sha256(promotion_records),
            "findings": [],
        }
    except (
        ContiguousOrchestratorError,
        Runner.ContiguousRunnerError,
        Scheduler.SchedulerError,
        OSError,
        KeyError,
        TypeError,
        ValueError,
    ) as exc:
        body = {
            "schema": UNIFIED_AUDIT_SCHEMA,
            "kind": "arc_agi3_contiguous_unified_audit",
            "status": "FAIL",
            "campaign_root": str(campaign),
            "promotion_root": str(promotions),
            "scheduler_audit_receipt_sha256": None,
            "runner_state_receipt_sha256": None,
            "terminal_retention_receipt_sha256": None,
            "pre_retention_promotion_replay_receipt_sha256": None,
            "campaign_id": None,
            "inventory_sha256": None,
            "scheduler_policy_sha256": None,
            "operator_configuration_sha256": None,
            "journal_event_count": None,
            "journal_head_sequence": None,
            "journal_head_digest": None,
            "solved_levels": 0,
            "verified_promotion_boundaries": 0,
            "total_levels": None,
            "complete": False,
            "promotion_records": [],
            "promotion_records_sha256": _json_sha256([]),
            "findings": [f"{type(exc).__name__}: {exc}"],
        }
    return {
        **body,
        "receipt_sha256": _json_sha256(body),
    }


def verify_contiguous_campaign_unified_audit(
    receipt: object,
    *,
    campaign_root: Path,
    scheduler_audit_receipt_path: Path,
    runner_state_receipt: object,
    promotion_root: Path,
    secret_sentinels: tuple[str, ...] = (),
    controller_state_canaries: tuple[Taint.LiveCanary, ...] = (),
) -> dict[str, Any]:
    """Recompute the unified receipt and require an exact current PASS."""

    expected = audit_contiguous_campaign_unified(
        campaign_root=campaign_root,
        scheduler_audit_receipt_path=scheduler_audit_receipt_path,
        runner_state_receipt=runner_state_receipt,
        promotion_root=promotion_root,
        secret_sentinels=secret_sentinels,
        controller_state_canaries=controller_state_canaries,
    )
    if (
        not isinstance(receipt, dict)
        or receipt != expected
        or expected["status"] != "PASS"
    ):
        raise ContiguousOrchestratorError(
            "unified campaign audit is stale, forged, or not PASS"
        )
    return expected


def _read_only_selective_frontier_gate(
    frontier_import_root: Path,
) -> ProductionPromotionGate:
    """Build a verifier for a separate immutable promotion store.

    The production gate constructor intentionally creates its output root.
    Selective preflight and audit are read-only, so they bind only the
    already descriptor-validated authority root and reuse the gate's sealed
    version verification methods without invoking that constructor.
    """

    selected = Path(frontier_import_root)
    if (
        not selected.is_absolute()
        or selected.is_symlink()
        or not selected.is_dir()
        or selected.resolve(strict=True) != selected
    ):
        raise ContiguousOrchestratorError(
            "frontier import authority root is unavailable or aliased"
        )
    _require_owned_directory(
        selected, label="frontier import authority root"
    )
    verifier = object.__new__(ProductionPromotionGate)
    verifier.frontier_import_root = selected
    return verifier


def issue_selective_frontier_import_read_only(
    *, frontier_import_root: Path, game: str
) -> Runner.SelectiveFrontierImport:
    """Issue one twice-read current frontier without creating output state."""

    return _read_only_selective_frontier_gate(
        frontier_import_root
    ).issue_selective_frontier_import(game)


def _require_operator_authorized_selective_frontier(
    config: OperatorConfiguration,
    binding: Runner.SelectiveFrontierImport,
) -> Runner.SelectiveFrontierImport:
    """Require one import to equal the exact digest authorized by config."""

    expected = getattr(
        config, "selective_frontier_import_sha256", None
    )
    game = getattr(config, "selective_continuation_game", None)
    if (
        not isinstance(expected, str)
        or SHA256_RE.fullmatch(expected) is None
        or not isinstance(game, str)
        or binding.game != game
        or binding.import_sha256 != expected
    ):
        raise ContiguousOrchestratorError(
            "selective frontier import differs from operator-authorized "
            "digest"
        )
    return binding


def _selective_operator_genesis_frontier(
    config: OperatorConfiguration,
) -> Runner.SelectiveFrontierImport | None:
    """Reopen the earliest durable selective frontier authority, if any."""

    path = config.campaign_root / "operator_genesis.json"
    if not path.exists() and not path.is_symlink():
        return None
    if path.is_symlink() or not path.is_file():
        raise ContiguousOrchestratorError(
            "selective operator genesis is unavailable or aliased"
        )
    genesis = _strict_json(
        _read_regular(path, maximum=MAX_JSON_BYTES),
        label="selective operator genesis",
    )
    contract = genesis.get("selective_durable_launch_contract")
    required_contract_fields = {
        "schema",
        "kind",
        "operator_configuration_sha256",
        "frontier_import_root",
        "selective_continuation_game",
        "selective_frontier_import",
        "selective_frontier_import_sha256",
        "operator_authorized_selective_frontier_import_sha256",
        "conformance_registry_sha256",
        "control_contract_sha256",
        "supplied_prelaunch_sha256",
        "authoritative_inventory_sha256",
        "image_digest",
        "python_runtime_manifest_sha256",
        "pilot_gate_receipt_sha256",
        "pilot_manifest_sha256",
        "pilot_meta_handoff_count",
        "production_stack_attestation_sha256",
        "contract_sha256",
    }
    if not isinstance(contract, Mapping):
        raise ContiguousOrchestratorError(
            "selective operator genesis lacks its launch contract"
        )
    contract_dict = dict(contract)
    contract_body = {
        key: value
        for key, value in contract_dict.items()
        if key != "contract_sha256"
    }
    try:
        binding = Runner.selective_frontier_import_from_dict(
            contract_dict.get("selective_frontier_import")
        )
    except Runner.ContiguousRunnerError as exc:
        raise ContiguousOrchestratorError(
            "selective operator genesis import is malformed"
        ) from exc
    if (
        set(contract_dict) != required_contract_fields
        or contract_dict.get("schema") != 1
        or contract_dict.get("kind")
        != "arc_agi3_selective_durable_launch_contract"
        or contract_dict.get("contract_sha256")
        != _json_sha256(contract_body)
        or genesis.get("schema") != 1
        or genesis.get("kind")
        != "arc_agi3_contiguous_operator_genesis"
        or genesis.get("operator_config_sha256")
        != config.config_sha256
        or genesis.get("campaign_mode")
        != "selective_continuation"
        or genesis.get("terminal_condition")
        != SELECTIVE_TERMINAL_CONDITION
        or genesis.get("frontier_import_root")
        != str(config.frontier_import_root)
        or genesis.get("selective_continuation_game")
        != config.selective_continuation_game
        or genesis.get("selective_frontier_import_sha256")
        != binding.import_sha256
        or genesis.get(
            "operator_authorized_selective_frontier_import_sha256"
        )
        != config.selective_frontier_import_sha256
        or genesis.get("selective_launch_authority_sha256")
        != contract_dict.get("contract_sha256")
        or contract_dict.get("operator_configuration_sha256")
        != config.config_sha256
        or contract_dict.get("frontier_import_root")
        != str(config.frontier_import_root)
        or contract_dict.get("selective_continuation_game")
        != config.selective_continuation_game
        or contract_dict.get("selective_frontier_import_sha256")
        != binding.import_sha256
        or contract_dict.get(
            "operator_authorized_selective_frontier_import_sha256"
        )
        != config.selective_frontier_import_sha256
        or binding.import_sha256
        != config.selective_frontier_import_sha256
    ):
        raise ContiguousOrchestratorError(
            "selective operator genesis launch contract is stale or forged"
        )
    return _require_operator_authorized_selective_frontier(
        config, binding
    )


def _selective_preflight_frontier(
    config: OperatorConfiguration,
) -> tuple[Runner.SelectiveFrontierImport, str]:
    """Issue fresh current once, then recover only the durable pinned import."""

    if (
        config.frontier_import_root is None
        or config.selective_continuation_game is None
        or config.selective_frontier_import_sha256 is None
    ):
        raise ContiguousOrchestratorError(
            "selective operator lacks its import configuration"
        )
    operator_binding = _selective_operator_genesis_frontier(config)
    journal_path = config.campaign_root / "attempt_journal"
    if not journal_path.exists() and not journal_path.is_symlink():
        if operator_binding is not None:
            verified = _read_only_selective_frontier_gate(
                config.frontier_import_root
            ).verify_selective_frontier_import(operator_binding)
            if verified != operator_binding:
                raise ContiguousOrchestratorError(
                    "operator-genesis frontier changed during preflight"
                )
            return (
                operator_binding,
                "authenticated_durable_operator_genesis_frontier",
            )
        return (
            _require_operator_authorized_selective_frontier(
                config,
                issue_selective_frontier_import_read_only(
                    frontier_import_root=config.frontier_import_root,
                    game=config.selective_continuation_game,
                ),
            ),
            "authenticated_current_frontier",
        )
    if journal_path.is_symlink() or not journal_path.is_dir():
        raise ContiguousOrchestratorError(
            "selective campaign journal is unavailable or aliased"
        )
    try:
        events = Scheduler.read_journal(config.campaign_root)
    except Exception as exc:
        raise ContiguousOrchestratorError(
            "selective campaign journal cannot authenticate its import"
        ) from exc
    if not events:
        if operator_binding is not None:
            verified = _read_only_selective_frontier_gate(
                config.frontier_import_root
            ).verify_selective_frontier_import(operator_binding)
            if verified != operator_binding:
                raise ContiguousOrchestratorError(
                    "operator-genesis frontier changed during preflight"
                )
            return (
                operator_binding,
                "authenticated_durable_operator_genesis_frontier",
            )
        return (
            _require_operator_authorized_selective_frontier(
                config,
                issue_selective_frontier_import_read_only(
                    frontier_import_root=config.frontier_import_root,
                    game=config.selective_continuation_game,
                ),
            ),
            "authenticated_current_frontier_empty_journal_recovery",
        )
    if events[0].get("kind") != "GENESIS":
        raise ContiguousOrchestratorError(
            "selective campaign lacks authenticated genesis"
        )
    genesis = events[0].get("payload")
    if not isinstance(genesis, Mapping):
        raise ContiguousOrchestratorError(
            "selective campaign genesis is malformed"
        )
    try:
        binding = Runner.selective_frontier_import_from_dict(
            genesis.get("selective_frontier_import")
        )
    except Runner.ContiguousRunnerError as exc:
        raise ContiguousOrchestratorError(
            "selective campaign genesis lacks its pinned import"
        ) from exc
    if (
        genesis.get("campaign_mode") != "selective_continuation"
        or genesis.get("operator_configuration_sha256")
        != config.config_sha256
        or genesis.get("selective_continuation_game")
        != config.selective_continuation_game
        or genesis.get("selective_frontier_import_sha256")
        != binding.import_sha256
        or genesis.get(
            "operator_authorized_selective_frontier_import_sha256"
        )
        != config.selective_frontier_import_sha256
        or binding.import_sha256
        != config.selective_frontier_import_sha256
    ):
        raise ContiguousOrchestratorError(
            "selective campaign genesis disagrees with operator authority"
        )
    if operator_binding is not None and operator_binding != binding:
        raise ContiguousOrchestratorError(
            "operator and runner genesis frontier authorities disagree"
        )
    import_events = [
        event
        for event in events[1:]
        if event.get("kind") == "FRONTIER_IMPORTED"
    ]
    expected_payload = Runner.selective_frontier_import_to_dict(
        binding
    )
    missing_recoverable_import_event = (
        not import_events and len(events) == 1
    )
    if not missing_recoverable_import_event and (
        len(import_events) != 1
        or import_events[0].get("sequence") != 2
        or import_events[0].get("payload") != expected_payload
    ):
        raise ContiguousOrchestratorError(
            "selective campaign import event is missing or substituted"
        )
    verified = _read_only_selective_frontier_gate(
        config.frontier_import_root
    ).verify_selective_frontier_import(binding)
    if verified != binding:
        raise ContiguousOrchestratorError(
            "selective durable frontier changed during preflight"
        )
    return (
        _require_operator_authorized_selective_frontier(
            config, binding
        ),
        (
            "authenticated_durable_operator_and_runner_genesis_frontier"
            if operator_binding is not None
            else "authenticated_durable_genesis_frontier"
        ),
    )


def audit_selective_continuation_unified(
    *,
    campaign_root: Path,
    scheduler_audit_receipt_path: Path,
    runner_state_receipt: object,
    promotion_root: Path,
    frontier_import_root: Path,
    expected_selective_frontier_import_sha256: str,
    secret_sentinels: tuple[str, ...] = (),
    controller_state_canaries: tuple[Taint.LiveCanary, ...] = (),
) -> dict[str, Any]:
    """Audit exactly the imported K-to-target continuation scope.

    Imported evidence is reopened from its immutable authority store.  Fresh
    promotions are reopened from the distinct mutable output store.  Global
    183-level completion remains false and cannot be inferred by this audit.
    """

    campaign = Path(campaign_root).resolve()
    scheduler_path = Path(scheduler_audit_receipt_path).resolve()
    promotions = Path(promotion_root).resolve()
    import_root = Path(frontier_import_root)
    try:
        if (
            not isinstance(
                expected_selective_frontier_import_sha256, str
            )
            or SHA256_RE.fullmatch(
                expected_selective_frontier_import_sha256
            )
            is None
        ):
            raise ContiguousOrchestratorError(
                "operator-authorized selective import digest is malformed"
            )
        runner_receipt = Runner.verify_runner_state_audit(
            runner_state_receipt,
            campaign_root=campaign,
            secret_sentinels=secret_sentinels,
            controller_state_canaries=controller_state_canaries,
        )
        raw_import = runner_receipt.get("selective_frontier_import")
        if (
            runner_receipt.get("status") != "PASS"
            or runner_receipt.get("campaign_mode")
            != "selective_continuation"
            or runner_receipt.get("complete") is not False
            or runner_receipt.get("selective_complete") is not True
            or not isinstance(raw_import, dict)
        ):
            raise ContiguousOrchestratorError(
                "runner audit is not a selective-complete PASS"
            )
        selective_import = Runner.selective_frontier_import_from_dict(
            raw_import
        )
        if (
            runner_receipt.get("selective_continuation_game")
            != selective_import.game
            or runner_receipt.get(
                "selective_frontier_import_sha256"
            )
            != selective_import.import_sha256
            or runner_receipt.get(
                "operator_authorized_selective_frontier_import_sha256"
            )
            != expected_selective_frontier_import_sha256
            or selective_import.import_sha256
            != expected_selective_frontier_import_sha256
        ):
            raise ContiguousOrchestratorError(
                "runner audit changes its selective import identity"
            )
        reopened_import = _read_only_selective_frontier_gate(
            import_root
        ).verify_selective_frontier_import(selective_import)
        if reopened_import != selective_import:
            raise ContiguousOrchestratorError(
                "selective frontier import changed during audit"
            )
        scheduler_receipt = Scheduler.verify_audit_receipt(
            campaign, scheduler_path
        )
        terminal_retention = Runner.audit_terminal_attempt_retention(
            campaign,
            runner_receipt,
            secret_sentinels=secret_sentinels,
            controller_state_canaries=controller_state_canaries,
        )
        if terminal_retention.get("status") != "NOT_REQUIRED":
            raise ContiguousOrchestratorError(
                "selective audit cannot exercise global terminal retention"
            )
        scheduler_summary = scheduler_receipt.get("summary")
        if not isinstance(scheduler_summary, dict):
            raise ContiguousOrchestratorError(
                "selective scheduler audit summary is malformed"
            )
        scope_solved = runner_receipt.get(
            "selective_scope_solved_levels"
        )
        scope_total = runner_receipt.get(
            "selective_scope_total_levels"
        )
        if (
            scheduler_receipt.get("campaign_root") != str(campaign)
            or runner_receipt.get("campaign_root") != str(campaign)
            or scheduler_receipt.get("policy_sha256")
            != runner_receipt.get("scheduler_policy_sha256")
            or scheduler_receipt.get("journal_events")
            != runner_receipt.get("journal_event_count")
            or scheduler_receipt.get("journal_head_sequence")
            != runner_receipt.get("journal_head_sequence")
            or scheduler_receipt.get("journal_head_digest")
            != runner_receipt.get("journal_head_digest")
            or scheduler_summary.get("journal_prefix")
            != runner_receipt.get("journal_prefix")
            or scheduler_summary.get("campaign_mode")
            != "selective_continuation"
            or scheduler_summary.get("selective_continuation_game")
            != selective_import.game
            or scheduler_summary.get(
                "selective_frontier_import_sha256"
            )
            != selective_import.import_sha256
            or scheduler_summary.get(
                "operator_authorized_selective_frontier_import_sha256"
            )
            != expected_selective_frontier_import_sha256
            or scheduler_summary.get("policy_promoted_levels")
            != scope_solved
            or scheduler_summary.get(
                "selective_scope_promoted_levels"
            )
            != scope_solved
            or scheduler_summary.get(
                "selective_scope_total_levels"
            )
            != scope_total
            or scheduler_summary.get("selective_complete") is not True
            or scope_solved != scope_total
            or scheduler_summary.get("pending_decision") is not None
            or scheduler_summary.get("active_auxiliary_assignments")
            != []
            or scheduler_summary.get("live_reservation_units") != 0
        ):
            raise ContiguousOrchestratorError(
                "selective scheduler and runner receipts disagree"
            )
        promotion_records, verified_boundaries = (
            _read_only_promotion_records(
                promotions,
                campaign_id=str(runner_receipt["campaign_id"]),
                lane_boundaries=runner_receipt["lane_boundaries"],
            )
        )
        scope_verified = verified_boundaries - selective_import.reached
        if (
            scope_verified != scope_solved
            or verified_boundaries != runner_receipt["solved_levels"]
        ):
            raise ContiguousOrchestratorError(
                "selective scope lacks exact fresh promotion evidence"
            )
        body = {
            "schema": UNIFIED_AUDIT_SCHEMA,
            "kind": "arc_agi3_selective_continuation_unified_audit",
            "status": "PASS",
            "campaign_mode": "selective_continuation",
            "campaign_root": str(campaign),
            "promotion_root": str(promotions),
            "frontier_import_root": str(import_root),
            "scheduler_audit_receipt_sha256":
                scheduler_receipt["receipt_sha256"],
            "runner_state_receipt_sha256":
                runner_receipt["receipt_sha256"],
            "terminal_retention_receipt_sha256":
                terminal_retention["receipt_sha256"],
            "campaign_id": runner_receipt["campaign_id"],
            "inventory_sha256": runner_receipt["inventory_sha256"],
            "scheduler_policy_sha256":
                runner_receipt["scheduler_policy_sha256"],
            "operator_configuration_sha256":
                runner_receipt["operator_configuration_sha256"],
            "journal_event_count": runner_receipt[
                "journal_event_count"
            ],
            "journal_head_sequence": runner_receipt[
                "journal_head_sequence"
            ],
            "journal_head_digest": runner_receipt[
                "journal_head_digest"
            ],
            "selective_continuation_game": selective_import.game,
            "selective_frontier_import_sha256":
                selective_import.import_sha256,
            "operator_authorized_selective_frontier_import_sha256":
                expected_selective_frontier_import_sha256,
            "imported_frontier_levels": selective_import.reached,
            "selective_scope_solved_levels": scope_solved,
            "selective_scope_total_levels": scope_total,
            "selective_verified_promotion_boundaries": scope_verified,
            "selective_complete": True,
            "solved_levels": verified_boundaries,
            "total_levels": runner_receipt["total_levels"],
            "complete": False,
            "promotion_records": promotion_records,
            "promotion_records_sha256": _json_sha256(
                promotion_records
            ),
            "findings": [],
        }
    except (
        ContiguousOrchestratorError,
        Runner.ContiguousRunnerError,
        Scheduler.SchedulerError,
        OSError,
        KeyError,
        TypeError,
        ValueError,
    ) as exc:
        body = {
            "schema": UNIFIED_AUDIT_SCHEMA,
            "kind": "arc_agi3_selective_continuation_unified_audit",
            "status": "FAIL",
            "campaign_mode": "selective_continuation",
            "campaign_root": str(campaign),
            "promotion_root": str(promotions),
            "frontier_import_root": str(import_root),
            "operator_authorized_selective_frontier_import_sha256":
                expected_selective_frontier_import_sha256,
            "selective_complete": False,
            "complete": False,
            "findings": [f"{type(exc).__name__}: {exc}"],
        }
    return {**body, "receipt_sha256": _json_sha256(body)}


def verify_selective_continuation_unified_audit(
    receipt: object,
    *,
    campaign_root: Path,
    scheduler_audit_receipt_path: Path,
    runner_state_receipt: object,
    promotion_root: Path,
    frontier_import_root: Path,
    expected_selective_frontier_import_sha256: str,
    secret_sentinels: tuple[str, ...] = (),
    controller_state_canaries: tuple[Taint.LiveCanary, ...] = (),
) -> dict[str, Any]:
    """Recompute and require the exact selective continuation PASS."""

    expected = audit_selective_continuation_unified(
        campaign_root=campaign_root,
        scheduler_audit_receipt_path=scheduler_audit_receipt_path,
        runner_state_receipt=runner_state_receipt,
        promotion_root=promotion_root,
        frontier_import_root=frontier_import_root,
        expected_selective_frontier_import_sha256=(
            expected_selective_frontier_import_sha256
        ),
        secret_sentinels=secret_sentinels,
        controller_state_canaries=controller_state_canaries,
    )
    if (
        not isinstance(receipt, dict)
        or receipt != expected
        or expected.get("status") != "PASS"
        or expected.get("selective_complete") is not True
        or expected.get("complete") is not False
    ):
        raise ContiguousOrchestratorError(
            "selective unified audit is stale, forged, or not PASS"
        )
    return expected


@dataclass(frozen=True)
class AuxiliaryBackendDriverConfiguration:
    """Digest-bound host driver selected once by the launch manifest."""

    schema: Literal[1]
    driver_executable: Path
    driver_executable_sha256: str
    driver_configuration: Path
    driver_configuration_sha256: str
    backend_attestation: Path
    backend_attestation_sha256: str
    operation_timeout_seconds: int


_AUXILIARY_BACKEND_CONFIGURATION_FIELDS = frozenset({
    "schema",
    "driver_executable",
    "driver_executable_sha256",
    "driver_configuration",
    "driver_configuration_sha256",
    "backend_attestation",
    "backend_attestation_sha256",
    "operation_timeout_seconds",
})
_AUXILIARY_BACKEND_ATTESTATION_FIELDS = frozenset({
    "schema",
    "kind",
    "driver_protocol_sha256",
    "driver_executable_sha256",
    "driver_configuration_sha256",
    "backend_contract_sha256",
    "input_bundle_contract_sha256",
    "admission_contract_sha256",
    "model",
    "reasoning_effort",
    "production_isolation_attested",
    "immutable_private_input_attested",
    "host_admission_attested",
    "descriptor_confined_receipts_attested",
    "post_incident_meta_protocol_sha256",
    "post_incident_meta_diagnostic_attested",
    "post_incident_meta_result_authority",
})


def _owner_held_control_bytes(
    path: Path, *, label: str, maximum: int
) -> bytes:
    """Read one unaliased, owner-held, immutable control input."""

    selected = Path(path)
    try:
        descriptor = os.open(
            selected, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        )
    except OSError as exc:
        raise ContiguousOrchestratorError(
            f"{label} must be an unaliased regular file"
        ) from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_uid != os.getuid()
            or stat.S_IMODE(before.st_mode) != 0o400
            or not 0 < before.st_size <= maximum
        ):
            raise ContiguousOrchestratorError(
                f"{label} must be owner-held mode 0400"
            )
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            block = os.read(
                descriptor, min(1024 * 1024, remaining)
            )
            if not block:
                raise ContiguousOrchestratorError(
                    f"{label} changed while reading"
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
            raise ContiguousOrchestratorError(
                f"{label} changed while reading"
            )
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _verify_auxiliary_backend_configuration(
    configuration: AuxiliaryBackendDriverConfiguration,
    launch_configuration: Scheduler.AuxiliaryLaunchConfiguration,
) -> dict[str, object]:
    """Reopen all launch-manifest controls and return the exact attestation."""

    try:
        launch = Scheduler.validate_auxiliary_launch_configuration(
            launch_configuration
        )
    except Scheduler.SchedulerError as exc:
        raise ContiguousOrchestratorError(
            "auxiliary launch configuration is malformed"
        ) from exc
    if not launch.automatic_dispatch_enabled:
        raise ContiguousOrchestratorError(
            "the production operator requires automatic auxiliary dispatch"
        )
    if (
        not isinstance(configuration, AuxiliaryBackendDriverConfiguration)
        or configuration.schema != 1
        or isinstance(configuration.operation_timeout_seconds, bool)
        or not isinstance(
            configuration.operation_timeout_seconds, int
        )
        or not 5 <= configuration.operation_timeout_seconds <= 3600
    ):
        raise ContiguousOrchestratorError(
            "auxiliary backend configuration is malformed"
        )
    _verify_executable(
        configuration.driver_executable,
        configuration.driver_executable_sha256,
        label="auxiliary backend driver",
    )
    driver_configuration = _owner_held_control_bytes(
        configuration.driver_configuration,
        label="auxiliary driver configuration",
        maximum=MAX_AUXILIARY_DRIVER_CONTROL_BYTES,
    )
    attestation_raw = _owner_held_control_bytes(
        configuration.backend_attestation,
        label="auxiliary backend attestation",
        maximum=MAX_AUXILIARY_DRIVER_CONTROL_BYTES,
    )
    if (
        hashlib.sha256(driver_configuration).hexdigest()
        != configuration.driver_configuration_sha256
        or hashlib.sha256(attestation_raw).hexdigest()
        != configuration.backend_attestation_sha256
    ):
        raise ContiguousOrchestratorError(
            "auxiliary backend control digest differs"
        )
    attestation = _strict_json(
        attestation_raw, label="auxiliary backend attestation"
    )
    expected = {
        "schema": 1,
        "kind": "arc_agi3_contiguous_auxiliary_backend_attestation",
        "driver_protocol_sha256":
            AUXILIARY_DRIVER_PROTOCOL_SHA256,
        "driver_executable_sha256":
            configuration.driver_executable_sha256,
        "driver_configuration_sha256":
            configuration.driver_configuration_sha256,
        "backend_contract_sha256":
            launch.backend_contract_sha256,
        "input_bundle_contract_sha256":
            launch.input_bundle_contract_sha256,
        "admission_contract_sha256":
            launch.admission_contract_sha256,
        "model": launch.model,
        "reasoning_effort": launch.reasoning_effort,
        "production_isolation_attested": True,
        "immutable_private_input_attested": True,
        "host_admission_attested": True,
        "descriptor_confined_receipts_attested": True,
        "post_incident_meta_protocol_sha256":
            Supervisor.POST_INCIDENT_META_PROTOCOL_SHA256,
        "post_incident_meta_diagnostic_attested": True,
        "post_incident_meta_result_authority":
            "quarantine_only",
    }
    if (
        not isinstance(attestation, dict)
        or set(attestation)
        != _AUXILIARY_BACKEND_ATTESTATION_FIELDS
        or attestation != expected
    ):
        raise ContiguousOrchestratorError(
            "auxiliary backend attestation differs from the exact "
            "scheduler/driver contract"
        )
    return expected


def _parse_auxiliary_backend_configuration(
    value: object,
    launch_configuration: Scheduler.AuxiliaryLaunchConfiguration,
) -> AuxiliaryBackendDriverConfiguration:
    if (
        not isinstance(value, dict)
        or set(value) != _AUXILIARY_BACKEND_CONFIGURATION_FIELDS
        or value.get("schema") != 1
    ):
        raise ContiguousOrchestratorError(
            "auxiliary_backend_configuration schema is not exact"
        )
    try:
        configuration = AuxiliaryBackendDriverConfiguration(
            schema=1,
            driver_executable=_absolute_path(
                value["driver_executable"],
                label="auxiliary driver executable",
            ),
            driver_executable_sha256=value[
                "driver_executable_sha256"
            ],
            driver_configuration=_absolute_path(
                value["driver_configuration"],
                label="auxiliary driver configuration",
            ),
            driver_configuration_sha256=value[
                "driver_configuration_sha256"
            ],
            backend_attestation=_absolute_path(
                value["backend_attestation"],
                label="auxiliary backend attestation",
            ),
            backend_attestation_sha256=value[
                "backend_attestation_sha256"
            ],
            operation_timeout_seconds=value[
                "operation_timeout_seconds"
            ],
        )
    except (KeyError, TypeError) as exc:
        raise ContiguousOrchestratorError(
            "auxiliary_backend_configuration could not be typed"
        ) from exc
    _verify_auxiliary_backend_configuration(
        configuration, launch_configuration
    )
    return configuration


class ProductionAuxiliaryBackend:
    """Execute only scheduler-signed sidecar operations through one driver.

    The fixed argv deliberately contains no game, effort, round, or
    specialization switch.  Those values exist only in the canonical
    ``AuxiliaryDecision`` inside the immutable request.  Driver stdout/stderr
    are retained as host-only bytes and never become an abort reason or model
    prompt.
    """

    production_isolation_attested = True
    immutable_private_input_attested = True
    host_admission_attested = True
    descriptor_confined_receipts_attested = True

    def __init__(
        self,
        *,
        campaign_root: Path,
        command_runner: Any,
        configuration: AuxiliaryBackendDriverConfiguration,
        launch_configuration:
            Scheduler.AuxiliaryLaunchConfiguration,
    ) -> None:
        _verify_auxiliary_backend_configuration(
            configuration, launch_configuration
        )
        if not callable(
            getattr(command_runner, "run_attached_stream", None)
        ):
            raise ContiguousOrchestratorError(
                "auxiliary backend driver requires bounded stream capture"
            )
        self.campaign_root = Path(campaign_root).resolve()
        self.command_runner = command_runner
        self.driver_configuration = configuration
        self.launch_configuration = launch_configuration
        self.backend_contract_sha256 = str(
            launch_configuration.backend_contract_sha256
        )
        self.input_bundle_contract_sha256 = str(
            launch_configuration.input_bundle_contract_sha256
        )
        self.admission_contract_sha256 = str(
            launch_configuration.admission_contract_sha256
        )
        self._path_binding_lock = threading.Lock()
        self._auxiliary_root_identity: tuple[int, ...] | None = None
        self._assignment_root_identities: dict[
            str, tuple[int, ...]
        ] = {}
        self._driver_path_identities: dict[
            tuple[str, str], tuple[tuple[int, ...], ...]
        ] = {}

    def configuration(
        self,
    ) -> Scheduler.AuxiliaryLaunchConfiguration:
        return self.launch_configuration

    @staticmethod
    def _safe_reason(value: object, *, allow_none: bool) -> str | None:
        if value is None and allow_none:
            return None
        if (
            not isinstance(value, str)
            or _AUXILIARY_REASON_CODE_RE.fullmatch(value) is None
        ):
            raise Runner.AuxiliaryBackendFatalError(
                "driver_reason_code_invalid"
            )
        return value

    @staticmethod
    def _strict_result(
        value: object, fields: set[str] | frozenset[str], *, label: str
    ) -> dict[str, object]:
        if not isinstance(value, dict) or set(value) != set(fields):
            del label
            raise Runner.AuxiliaryBackendFatalError(
                "driver_result_schema_invalid"
            )
        return value

    def _assignment_root(
        self, decision: Scheduler.AuxiliaryDecision
    ) -> Path:
        return (
            self.campaign_root
            / "auxiliary"
            / decision.assignment_id
        )

    @staticmethod
    def _directory_identity(metadata: os.stat_result) -> tuple[int, ...]:
        return (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_mode,
            metadata.st_uid,
            metadata.st_gid,
        )

    @staticmethod
    def _file_identity(metadata: os.stat_result) -> tuple[int, ...]:
        return (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_mode,
            metadata.st_nlink,
            metadata.st_uid,
            metadata.st_gid,
            metadata.st_size,
            metadata.st_mtime_ns,
            metadata.st_ctime_ns,
        )

    @staticmethod
    def _private_directory_metadata(
        metadata: os.stat_result,
    ) -> bool:
        return (
            stat.S_ISDIR(metadata.st_mode)
            and metadata.st_uid == os.getuid()
            and stat.S_IMODE(metadata.st_mode) & 0o022 == 0
        )

    def _open_auxiliary_root(self) -> int:
        try:
            descriptor = os.open(
                self.campaign_root / "auxiliary",
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0),
            )
        except OSError as exc:
            raise Runner.AuxiliaryBackendFatalError(
                "driver_assignment_root_unavailable"
            ) from exc
        metadata = os.fstat(descriptor)
        identity = self._directory_identity(metadata)
        if not self._private_directory_metadata(metadata):
            os.close(descriptor)
            raise Runner.AuxiliaryBackendFatalError(
                "driver_assignment_root_inadmissible"
            )
        with self._path_binding_lock:
            existing = self._auxiliary_root_identity
            if existing is None:
                self._auxiliary_root_identity = identity
            elif existing != identity:
                os.close(descriptor)
                raise Runner.AuxiliaryBackendFatalError(
                    "driver_assignment_root_replaced"
                )
        return descriptor

    def _open_assignment_root(
        self, decision: Scheduler.AuxiliaryDecision
    ) -> int:
        auxiliary_descriptor = self._open_auxiliary_root()
        try:
            descriptor = os.open(
                decision.assignment_id,
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=auxiliary_descriptor,
            )
        except OSError as exc:
            raise Runner.AuxiliaryBackendFatalError(
                "driver_assignment_root_unavailable"
            ) from exc
        finally:
            os.close(auxiliary_descriptor)
        metadata = os.fstat(descriptor)
        identity = self._directory_identity(metadata)
        if not self._private_directory_metadata(metadata):
            os.close(descriptor)
            raise Runner.AuxiliaryBackendFatalError(
                "driver_assignment_root_inadmissible"
            )
        with self._path_binding_lock:
            existing = self._assignment_root_identities.get(
                decision.assignment_id
            )
            if existing is None:
                self._assignment_root_identities[
                    decision.assignment_id
                ] = identity
            elif existing != identity:
                os.close(descriptor)
                raise Runner.AuxiliaryBackendFatalError(
                    "driver_assignment_root_replaced"
                )
        return descriptor

    def _relative_driver_path(
        self,
        value: object,
        *,
        decision: Scheduler.AuxiliaryDecision,
    ) -> tuple[str, tuple[str, ...]]:
        if (
            not isinstance(value, str)
            or not Path(value).is_absolute()
            or "\x00" in value
        ):
            raise Runner.AuxiliaryBackendFatalError(
                "driver_result_path_invalid"
            )
        selected = Path(value)
        root = self._assignment_root(decision)
        root_parts = root.parts
        selected_parts = selected.parts
        relative = selected_parts[len(root_parts):]
        if (
            value != str(selected)
            or selected_parts[:len(root_parts)] != root_parts
            or not relative
            or any(part in {"", ".", ".."} for part in relative)
        ):
            raise Runner.AuxiliaryBackendFatalError(
                "driver_result_path_escape"
            )
        return value, tuple(relative)

    def _read_confined_driver_file(
        self,
        decision: Scheduler.AuxiliaryDecision,
        value: object,
        *,
        maximum: int,
        bind_path: bool,
        expected_mode: int | None = 0o400,
        allow_empty: bool = False,
    ) -> bytes:
        if (
            isinstance(maximum, bool)
            or not isinstance(maximum, int)
            or not 1 <= maximum <= MAX_AUXILIARY_DRIVER_RESPONSE_BYTES
        ):
            raise Runner.AuxiliaryBackendFatalError(
                "driver_result_read_bound_invalid"
            )
        canonical, relative = self._relative_driver_path(
            value, decision=decision
        )
        directory_descriptors: list[int] = []
        directory_identities: list[tuple[int, ...]] = []
        namespace_edges: list[
            tuple[int, str, tuple[int, ...]]
        ] = []
        file_descriptor: int | None = None
        raw = b""
        try:
            assignment_descriptor = self._open_assignment_root(decision)
            directory_descriptors.append(assignment_descriptor)
            assignment_metadata = os.fstat(assignment_descriptor)
            directory_identities.append(
                self._directory_identity(assignment_metadata)
            )
            for part in relative[:-1]:
                parent_descriptor = directory_descriptors[-1]
                child_descriptor = os.open(
                    part,
                    os.O_RDONLY
                    | getattr(os, "O_DIRECTORY", 0)
                    | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=parent_descriptor,
                )
                child_metadata = os.fstat(child_descriptor)
                if not self._private_directory_metadata(child_metadata):
                    os.close(child_descriptor)
                    raise Runner.AuxiliaryBackendFatalError(
                        "driver_result_path_alias"
                    )
                child_identity = self._directory_identity(
                    child_metadata
                )
                namespace_edges.append(
                    (parent_descriptor, part, child_identity)
                )
                directory_descriptors.append(child_descriptor)
                directory_identities.append(child_identity)
            leaf = relative[-1]
            file_descriptor = os.open(
                leaf,
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=directory_descriptors[-1],
            )
            before = os.fstat(file_descriptor)
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_nlink != 1
                or before.st_uid != os.getuid()
                or (before.st_size == 0 and not allow_empty)
                or before.st_size > maximum
                or (
                    expected_mode is not None
                    and stat.S_IMODE(before.st_mode) != expected_mode
                )
                or (
                    expected_mode is None
                    and stat.S_IMODE(before.st_mode) & 0o022
                )
            ):
                raise Runner.AuxiliaryBackendFatalError(
                    "driver_result_file_inadmissible"
                )
            chunks: list[bytes] = []
            remaining = before.st_size
            while remaining:
                block = os.read(
                    file_descriptor, min(1024 * 1024, remaining)
                )
                if not block:
                    raise Runner.AuxiliaryBackendFatalError(
                        "driver_result_file_changed"
                    )
                chunks.append(block)
                remaining -= len(block)
            after = os.fstat(file_descriptor)
            file_identity = self._file_identity(after)
            if self._file_identity(before) != file_identity:
                raise Runner.AuxiliaryBackendFatalError(
                    "driver_result_file_changed"
                )
            for (
                parent_descriptor,
                part,
                expected_identity,
            ) in namespace_edges:
                observed = os.stat(
                    part,
                    dir_fd=parent_descriptor,
                    follow_symlinks=False,
                )
                if (
                    not stat.S_ISDIR(observed.st_mode)
                    or self._directory_identity(observed)
                    != expected_identity
                ):
                    raise Runner.AuxiliaryBackendFatalError(
                        "driver_result_path_replaced"
                    )
            observed_file = os.stat(
                leaf,
                dir_fd=directory_descriptors[-1],
                follow_symlinks=False,
            )
            if self._file_identity(observed_file) != file_identity:
                raise Runner.AuxiliaryBackendFatalError(
                    "driver_result_path_replaced"
                )
            raw = b"".join(chunks)
            path_identity = (
                *directory_identities,
                file_identity,
            )
        except Runner.AuxiliaryBackendFatalError:
            raise
        except OSError as exc:
            raise Runner.AuxiliaryBackendFatalError(
                "driver_result_path_alias"
            ) from exc
        finally:
            if file_descriptor is not None:
                os.close(file_descriptor)
            for descriptor in reversed(directory_descriptors):
                os.close(descriptor)
        # Reopen through the pinned auxiliary-root descriptor after the read.
        # This detects an assignment-root rename/replacement while the file
        # descriptor was live; no absolute path is trusted as an authority.
        rebound = self._open_assignment_root(decision)
        os.close(rebound)
        key = (decision.assignment_id, canonical)
        with self._path_binding_lock:
            existing = self._driver_path_identities.get(key)
            if bind_path:
                if existing is None:
                    self._driver_path_identities[key] = path_identity
                elif existing != path_identity:
                    raise Runner.AuxiliaryBackendFatalError(
                        "driver_result_path_replaced"
                    )
            elif existing is None:
                raise Runner.AuxiliaryBackendFatalError(
                    "driver_result_path_unbound"
                )
            elif existing != path_identity:
                raise Runner.AuxiliaryBackendFatalError(
                    "driver_result_path_replaced"
                )
        return raw

    def _confined_path(
        self,
        value: object,
        *,
        decision: Scheduler.AuxiliaryDecision,
        label: str,
    ) -> str:
        del label
        canonical, _ = self._relative_driver_path(
            value, decision=decision
        )
        self._read_confined_driver_file(
            decision,
            canonical,
            maximum=MAX_AUXILIARY_DRIVER_RESPONSE_BYTES,
            bind_path=True,
        )
        return canonical

    def read_confined_receipt(
        self,
        decision: Scheduler.AuxiliaryDecision,
        path_value: str,
        *,
        maximum: int,
    ) -> bytes:
        return self._read_confined_driver_file(
            decision,
            path_value,
            maximum=maximum,
            # A fresh operator process has no in-memory path map.  The runner
            # immediately checks these exact bytes against the authenticated
            # journal digest and canonical expected body, so a first read is
            # an explicit restart rebind.  An existing binding remains an
            # immutable component-identity check.
            bind_path=True,
        )

    def _next_invocation_directory(self, operation_root: Path) -> Path:
        operation_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(operation_root, 0o700, follow_symlinks=False)
        for index in range(1, 10000):
            selected = operation_root / f"invocation-{index:04d}"
            try:
                selected.mkdir(mode=0o700)
            except FileExistsError:
                continue
            return selected
        raise ContiguousOrchestratorError(
            "auxiliary driver invocation bound exhausted"
        )

    @staticmethod
    def _poll_sample_identity(
        *,
        request_sha256: str,
        sample_sequence: int,
        previous_checkpoint_sha256: str | None,
    ) -> str:
        return Scheduler.sha256_json({
            "request_sha256": request_sha256,
            "sample_sequence": sample_sequence,
            "previous_checkpoint_sha256":
                previous_checkpoint_sha256,
        })

    def _forget_driver_path(
        self,
        decision: Scheduler.AuxiliaryDecision,
        path: Path,
    ) -> None:
        key = (decision.assignment_id, str(path))
        with self._path_binding_lock:
            self._driver_path_identities.pop(key, None)

    def _unlink_bound_driver_file(
        self,
        decision: Scheduler.AuxiliaryDecision,
        path: Path,
        *,
        maximum: int,
        expected_sha256: str | None = None,
        expected_mode: int | None = 0o400,
        allow_empty: bool = False,
    ) -> None:
        if not path.exists() and not path.is_symlink():
            return
        raw = self._read_confined_driver_file(
            decision,
            str(path),
            maximum=maximum,
            bind_path=True,
            expected_mode=expected_mode,
            allow_empty=allow_empty,
        )
        if (
            expected_sha256 is not None
            and hashlib.sha256(raw).hexdigest() != expected_sha256
        ):
            raise Runner.AuxiliaryBackendFatalError(
                "driver_poll_compaction_digest_invalid"
            )
        try:
            path.unlink()
        except OSError as exc:
            raise Runner.AuxiliaryBackendFatalError(
                "driver_poll_compaction_failed"
            ) from exc
        self._forget_driver_path(decision, path)
        _fsync_directory(path.parent)

    def _validate_poll_checkpoint(
        self,
        decision: Scheduler.AuxiliaryDecision,
        value: object,
        *,
        request_sha256: str,
    ) -> dict[str, object]:
        checkpoint = self._strict_result(
            value,
            {
                "schema",
                "kind",
                "operation",
                "assignment_id",
                "decision_sha256",
                "request_sha256",
                "sample_sequence",
                "previous_checkpoint_sha256",
                "sample_identity_sha256",
                "response_sha256",
                "response_bytes",
                "response_binding_sha256",
                "invocation_receipt_sha256",
                "result",
            },
            label="poll checkpoint",
        )
        sequence = checkpoint["sample_sequence"]
        previous = checkpoint["previous_checkpoint_sha256"]
        result = checkpoint["result"]
        if (
            checkpoint["schema"] != 1
            or checkpoint["kind"]
            != "arc_agi3_contiguous_auxiliary_poll_checkpoint"
            or checkpoint["operation"] != "poll"
            or checkpoint["assignment_id"] != decision.assignment_id
            or checkpoint["decision_sha256"]
            != decision.decision_sha256
            or checkpoint["request_sha256"] != request_sha256
            or isinstance(sequence, bool)
            or not isinstance(sequence, int)
            or sequence < 1
            or (
                previous is not None
                and SHA256_RE.fullmatch(str(previous)) is None
            )
            or ((sequence == 1) != (previous is None))
            or checkpoint["sample_identity_sha256"]
            != self._poll_sample_identity(
                request_sha256=request_sha256,
                sample_sequence=sequence,
                previous_checkpoint_sha256=(
                    str(previous) if previous is not None else None
                ),
            )
            or SHA256_RE.fullmatch(
                str(checkpoint["response_sha256"])
            ) is None
            or isinstance(checkpoint["response_bytes"], bool)
            or not isinstance(checkpoint["response_bytes"], int)
            or not 1 <= checkpoint["response_bytes"] <= (
                MAX_AUXILIARY_DRIVER_RESPONSE_BYTES
            )
            or SHA256_RE.fullmatch(
                str(checkpoint["response_binding_sha256"])
            ) is None
            or SHA256_RE.fullmatch(
                str(checkpoint["invocation_receipt_sha256"])
            ) is None
            or not isinstance(result, dict)
            or result.get("status")
            not in {"running", "exited", "containment_fault"}
        ):
            raise Runner.AuxiliaryBackendFatalError(
                "driver_poll_checkpoint_invalid"
            )
        return checkpoint

    def _load_poll_checkpoint(
        self,
        decision: Scheduler.AuxiliaryDecision,
        operation_root: Path,
        *,
        request_sha256: str,
    ) -> tuple[dict[str, object] | None, str | None]:
        observed: list[tuple[dict[str, object], str, Path]] = []
        for slot in (0, 1):
            path = operation_root / f"poll_checkpoint_{slot}.json"
            if not path.exists() and not path.is_symlink():
                continue
            raw = self._read_confined_driver_file(
                decision,
                str(path),
                maximum=MAX_AUXILIARY_DRIVER_CONTROL_BYTES,
                bind_path=True,
            )
            value = _strict_json(
                raw, label="auxiliary poll checkpoint"
            )
            if raw != _canonical_json(value) + b"\n":
                raise Runner.AuxiliaryBackendFatalError(
                    "driver_poll_checkpoint_encoding_invalid"
                )
            observed.append((
                self._validate_poll_checkpoint(
                    decision,
                    value,
                    request_sha256=request_sha256,
                ),
                hashlib.sha256(raw).hexdigest(),
                path,
            ))
            if int(observed[-1][0]["sample_sequence"]) % 2 != slot:
                raise Runner.AuxiliaryBackendFatalError(
                    "driver_poll_checkpoint_slot_invalid"
                )
        if not observed:
            return None, None
        observed.sort(key=lambda item: int(item[0]["sample_sequence"]))
        if len(observed) > 2:
            raise Runner.AuxiliaryBackendFatalError(
                "driver_poll_checkpoint_cardinality_invalid"
            )
        if len(observed) == 2:
            older, newer = observed
            if (
                int(newer[0]["sample_sequence"])
                != int(older[0]["sample_sequence"]) + 1
                or newer[0]["previous_checkpoint_sha256"] != older[1]
            ):
                raise Runner.AuxiliaryBackendFatalError(
                    "driver_poll_checkpoint_chain_invalid"
                )
        latest, latest_sha256, _ = observed[-1]
        return latest, latest_sha256

    def _poll_invocation_directory(
        self,
        operation_root: Path,
        sample_sequence: int,
    ) -> Path:
        selected = (
            operation_root
            / f"sample-{sample_sequence:016d}"
        )
        if selected.exists() or selected.is_symlink():
            metadata = os.lstat(selected)
            if (
                not stat.S_ISDIR(metadata.st_mode)
                or stat.S_ISLNK(metadata.st_mode)
                or metadata.st_uid != os.getuid()
                or stat.S_IMODE(metadata.st_mode) & 0o022
            ):
                raise Runner.AuxiliaryBackendFatalError(
                    "driver_poll_sample_directory_invalid"
                )
            return selected
        try:
            selected.mkdir(mode=0o700)
        except OSError as exc:
            raise Runner.AuxiliaryBackendFatalError(
                "driver_poll_sample_directory_unavailable"
            ) from exc
        _fsync_directory(operation_root)
        return selected

    def _poll_transient_invocations(
        self,
        operation_root: Path,
    ) -> tuple[Path, ...]:
        try:
            children = tuple(operation_root.iterdir())
        except OSError as exc:
            raise Runner.AuxiliaryBackendFatalError(
                "driver_poll_operation_root_unavailable"
            ) from exc
        result = []
        allowed_files = {
            "request.json",
            "response.json",
            "response_binding.json",
            "poll_checkpoint_0.json",
            "poll_checkpoint_1.json",
            "poll_checkpoint_pending.json",
        }
        for child in children:
            if re.fullmatch(r"sample-[0-9]{16}", child.name):
                result.append(child)
                continue
            if child.name not in allowed_files:
                raise Runner.AuxiliaryBackendFatalError(
                    "driver_poll_operation_root_has_unknown_evidence"
                )
            try:
                metadata = os.lstat(child)
            except OSError as exc:
                raise Runner.AuxiliaryBackendFatalError(
                    "driver_poll_operation_root_changed"
                ) from exc
            if (
                not stat.S_ISREG(metadata.st_mode)
                or stat.S_ISLNK(metadata.st_mode)
                or metadata.st_nlink != 1
                or metadata.st_uid != os.getuid()
            ):
                raise Runner.AuxiliaryBackendFatalError(
                    "driver_poll_operation_root_entry_invalid"
                )
        if len(result) > 1:
            raise Runner.AuxiliaryBackendFatalError(
                "driver_poll_sample_cardinality_invalid"
            )
        return tuple(sorted(result))

    def _compact_poll_sample(
        self,
        decision: Scheduler.AuxiliaryDecision,
        operation_root: Path,
        checkpoint: Mapping[str, object],
        *,
        keep_checkpoint_sha256: str,
    ) -> None:
        sequence = int(checkpoint["sample_sequence"])
        invocation = (
            operation_root / f"sample-{sequence:016d}"
        )
        response_path = operation_root / "response.json"
        binding_path = operation_root / "response_binding.json"
        receipt_path = invocation / "invocation_receipt.json"
        if response_path.exists() or response_path.is_symlink():
            self._unlink_bound_driver_file(
                decision,
                response_path,
                maximum=MAX_AUXILIARY_DRIVER_RESPONSE_BYTES,
                expected_sha256=str(checkpoint["response_sha256"]),
            )
        if binding_path.exists() or binding_path.is_symlink():
            self._unlink_bound_driver_file(
                decision,
                binding_path,
                maximum=MAX_AUXILIARY_DRIVER_CONTROL_BYTES,
                expected_sha256=str(
                    checkpoint["response_binding_sha256"]
                ),
            )
        if getattr(self, "_poll_crash_cut", None) == (
            "mid_compaction_after_response"
        ):
            self._poll_crash_cut = None
            raise Runner.SimulatedCrash(
                "during auxiliary poll response compaction"
            )
        if invocation.exists() or invocation.is_symlink():
            expected_files = {
                "stdout.bin",
                "stderr.bin",
                "stderr_visibility_receipt.json",
                "invocation_receipt.json",
            }
            try:
                observed_files = {item.name for item in invocation.iterdir()}
            except OSError as exc:
                raise Runner.AuxiliaryBackendFatalError(
                    "driver_poll_sample_directory_unavailable"
                ) from exc
            if not observed_files.issubset(expected_files):
                raise Runner.AuxiliaryBackendFatalError(
                    "driver_poll_sample_directory_has_unknown_evidence"
                )
            if receipt_path.exists() or receipt_path.is_symlink():
                self._unlink_bound_driver_file(
                    decision,
                    receipt_path,
                    maximum=MAX_AUXILIARY_DRIVER_CONTROL_BYTES,
                    expected_sha256=str(
                        checkpoint["invocation_receipt_sha256"]
                    ),
                )
            for name, allow_empty in (
                ("stdout.bin", True),
                ("stderr.bin", True),
                ("stderr_visibility_receipt.json", False),
            ):
                selected = invocation / name
                if selected.exists() or selected.is_symlink():
                    self._unlink_bound_driver_file(
                        decision,
                        selected,
                        maximum=MAX_AUXILIARY_DRIVER_STREAM_BYTES,
                        expected_mode=None if name.endswith(".bin") else 0o400,
                        allow_empty=allow_empty,
                    )
            _fsync_directory(invocation)
            if getattr(self, "_poll_crash_cut", None) == (
                "mid_compaction_before_sample_rmdir"
            ):
                self._poll_crash_cut = None
                raise Runner.SimulatedCrash(
                    "during auxiliary poll sample compaction"
                )
            try:
                invocation.rmdir()
            except OSError as exc:
                raise Runner.AuxiliaryBackendFatalError(
                    "driver_poll_sample_compaction_incomplete"
                ) from exc
            _fsync_directory(operation_root)
        for slot in (0, 1):
            selected = operation_root / f"poll_checkpoint_{slot}.json"
            if not selected.exists() and not selected.is_symlink():
                continue
            raw = self._read_confined_driver_file(
                decision,
                str(selected),
                maximum=MAX_AUXILIARY_DRIVER_CONTROL_BYTES,
                bind_path=True,
            )
            if hashlib.sha256(raw).hexdigest() == keep_checkpoint_sha256:
                continue
            self._unlink_bound_driver_file(
                decision,
                selected,
                maximum=MAX_AUXILIARY_DRIVER_CONTROL_BYTES,
            )
        _fsync_directory(operation_root)

    def _publish_poll_checkpoint(
        self,
        decision: Scheduler.AuxiliaryDecision,
        operation_root: Path,
        checkpoint: Mapping[str, object],
    ) -> tuple[str, Path]:
        sequence = int(checkpoint["sample_sequence"])
        target = (
            operation_root
            / f"poll_checkpoint_{sequence % 2}.json"
        )
        pending = operation_root / "poll_checkpoint_pending.json"
        if (
            target.exists()
            or target.is_symlink()
            or pending.exists()
            or pending.is_symlink()
        ):
            raise Runner.AuxiliaryBackendFatalError(
                "driver_poll_checkpoint_publish_collision"
            )
        raw = _canonical_json(checkpoint) + b"\n"
        _write_new(pending, raw, mode=0o400)
        if getattr(self, "_poll_crash_cut", None) == (
            "before_checkpoint_rename"
        ):
            self._poll_crash_cut = None
            raise Runner.SimulatedCrash(
                "before auxiliary poll checkpoint rename"
            )
        try:
            os.replace(pending, target)
        except OSError as exc:
            raise Runner.AuxiliaryBackendFatalError(
                "driver_poll_checkpoint_publish_failed"
            ) from exc
        self._forget_driver_path(decision, pending)
        _fsync_directory(operation_root)
        digest = hashlib.sha256(raw).hexdigest()
        retained = self._read_confined_driver_file(
            decision,
            str(target),
            maximum=MAX_AUXILIARY_DRIVER_CONTROL_BYTES,
            bind_path=True,
        )
        if retained != raw:
            raise Runner.AuxiliaryBackendFatalError(
                "driver_poll_checkpoint_publish_changed"
            )
        return digest, target

    def _invoke(
        self,
        operation: str,
        decision: Scheduler.AuxiliaryDecision,
        arguments: Mapping[str, object],
        *,
        cacheable: bool = True,
        timeout_seconds: int | None = None,
    ) -> dict[str, object]:
        _verify_auxiliary_backend_configuration(
            self.driver_configuration, self.launch_configuration
        )
        try:
            normalized = Scheduler.auxiliary_decision_from_dict(
                json.loads(_canonical_json(
                    Scheduler.auxiliary_decision_to_dict(decision)
                ))
            )
        except Scheduler.SchedulerError as exc:
            raise ContiguousOrchestratorError(
                "auxiliary backend received a malformed decision"
            ) from exc
        if normalized != decision:
            raise ContiguousOrchestratorError(
                "auxiliary decision normalization differs"
            )
        if (
            operation not in {
                "prepare",
                "launch",
                "poll",
                "collect",
                "teardown",
                "admit",
                "abort",
            }
            or not isinstance(arguments, Mapping)
        ):
            raise ContiguousOrchestratorError(
                "auxiliary driver operation is unsupported"
            )
        request_body = {
            "schema": 1,
            "kind": "arc_agi3_contiguous_auxiliary_driver_request",
            "driver_protocol_sha256":
                AUXILIARY_DRIVER_PROTOCOL_SHA256,
            "operation": operation,
            "assignment_id": decision.assignment_id,
            "decision_sha256": decision.decision_sha256,
            "decision":
                Scheduler.auxiliary_decision_to_dict(decision),
            "arguments": dict(arguments),
        }
        request_sha256 = Scheduler.sha256_json(request_body)
        sampled_poll = operation == "poll" and not cacheable
        operation_key = (
            request_sha256
            if cacheable or sampled_poll
            else f"{request_sha256}-{time.monotonic_ns()}"
        )
        auxiliary_root = self.campaign_root / "auxiliary"
        assignment_root = self._assignment_root(decision)
        host_root = assignment_root / "host"
        driver_root = host_root / "driver"
        for path, label in (
            (auxiliary_root, "auxiliary root"),
            (assignment_root, "auxiliary assignment root"),
            (host_root, "auxiliary host root"),
            (driver_root, "auxiliary driver root"),
        ):
            _ensure_private_directory(path, label=label)
        # Bind the exact assignment directory before the configured driver
        # receives control.  Every later path walk starts from this inode via
        # ``openat`` and rejects replacement.
        assignment_descriptor = self._open_assignment_root(decision)
        os.close(assignment_descriptor)
        operation_root = (
            driver_root
            / f"{operation}-{operation_key}"
        )
        _ensure_private_directory(
            operation_root, label="auxiliary driver operation root"
        )
        request_path = operation_root / "request.json"
        request_raw = _canonical_json(request_body) + b"\n"
        if request_path.exists() or request_path.is_symlink():
            if (
                self._read_confined_driver_file(
                    decision,
                    str(request_path),
                    maximum=MAX_AUXILIARY_DRIVER_RESPONSE_BYTES,
                    bind_path=True,
                )
                != request_raw
            ):
                raise ContiguousOrchestratorError(
                    "auxiliary driver request differs on recovery"
                )
        else:
            _write_new(request_path, request_raw, mode=0o400)
            self._read_confined_driver_file(
                decision,
                str(request_path),
                maximum=MAX_AUXILIARY_DRIVER_RESPONSE_BYTES,
                bind_path=True,
            )
        poll_checkpoint: dict[str, object] | None = None
        poll_checkpoint_sha256: str | None = None
        poll_sample_sequence: int | None = None
        if sampled_poll:
            (
                poll_checkpoint,
                poll_checkpoint_sha256,
            ) = self._load_poll_checkpoint(
                decision,
                operation_root,
                request_sha256=request_sha256,
            )
            pending_checkpoint_path = (
                operation_root / "poll_checkpoint_pending.json"
            )
            if (
                pending_checkpoint_path.exists()
                or pending_checkpoint_path.is_symlink()
            ):
                pending_raw = self._read_confined_driver_file(
                    decision,
                    str(pending_checkpoint_path),
                    maximum=MAX_AUXILIARY_DRIVER_CONTROL_BYTES,
                    bind_path=True,
                )
                pending_value = _strict_json(
                    pending_raw,
                    label="pending auxiliary poll checkpoint",
                )
                if pending_raw != _canonical_json(pending_value) + b"\n":
                    raise Runner.AuxiliaryBackendFatalError(
                        "driver_poll_pending_checkpoint_encoding_invalid"
                    )
                pending_checkpoint = self._validate_poll_checkpoint(
                    decision,
                    pending_value,
                    request_sha256=request_sha256,
                )
                expected_pending_sequence = (
                    1
                    if poll_checkpoint is None
                    else int(
                        poll_checkpoint["sample_sequence"]
                    ) + 1
                )
                if (
                    pending_checkpoint["sample_sequence"]
                    != expected_pending_sequence
                    or pending_checkpoint[
                        "previous_checkpoint_sha256"
                    ] != poll_checkpoint_sha256
                ):
                    raise Runner.AuxiliaryBackendFatalError(
                        "driver_poll_pending_checkpoint_chain_invalid"
                    )
                # The pending file is not authority until its rename has
                # completed and the operation root has been fsynced.  Keep
                # the already-complete transient sample, discard this
                # unpublished copy, and deterministically republish it after
                # all response/invocation bindings have been reopened.
                self._unlink_bound_driver_file(
                    decision,
                    pending_checkpoint_path,
                    maximum=MAX_AUXILIARY_DRIVER_CONTROL_BYTES,
                )
                _fsync_directory(operation_root)
            if poll_checkpoint is not None:
                response_candidate = operation_root / "response.json"
                response_is_committed = False
                if (
                    response_candidate.exists()
                    or response_candidate.is_symlink()
                ):
                    candidate_raw = (
                        self._read_confined_driver_file(
                            decision,
                            str(response_candidate),
                            maximum=(
                                MAX_AUXILIARY_DRIVER_RESPONSE_BYTES
                            ),
                            bind_path=True,
                        )
                    )
                    response_is_committed = (
                        hashlib.sha256(candidate_raw).hexdigest()
                        == poll_checkpoint["response_sha256"]
                    )
                committed_invocation = (
                    operation_root
                    / (
                        "sample-"
                        f"{int(poll_checkpoint['sample_sequence']):016d}"
                    )
                )
                committed_binding = (
                    operation_root / "response_binding.json"
                )
                transient_invocations = (
                    self._poll_transient_invocations(operation_root)
                )
                future_invocation = (
                    operation_root
                    / (
                        "sample-"
                        f"{int(poll_checkpoint['sample_sequence']) + 1:016d}"
                    )
                )
                has_uncommitted_future = (
                    future_invocation in transient_invocations
                )
                if (
                    not has_uncommitted_future
                    and (
                        response_is_committed
                        or committed_invocation.exists()
                        or committed_invocation.is_symlink()
                        or committed_binding.exists()
                        or committed_binding.is_symlink()
                    )
                ):
                    self._compact_poll_sample(
                        decision,
                        operation_root,
                        poll_checkpoint,
                        keep_checkpoint_sha256=str(
                            poll_checkpoint_sha256
                        ),
                    )
                if poll_checkpoint["result"].get("status") != "running":
                    return dict(poll_checkpoint["result"])
            poll_sample_sequence = (
                1
                if poll_checkpoint is None
                else int(poll_checkpoint["sample_sequence"]) + 1
            )
        response_path = operation_root / "response.json"
        response_binding_path = operation_root / "response_binding.json"
        invocation: Path | None = None
        invocation_was_run = False
        observed: object | None = None
        stderr_classification: dict[str, object] | None = None
        stderr_classification_sha256: str | None = None
        if not response_path.exists() and not response_path.is_symlink():
            if sampled_poll:
                assert poll_sample_sequence is not None
                existing_samples = self._poll_transient_invocations(
                    operation_root
                )
                if existing_samples:
                    raise Runner.AuxiliaryBackendFatalError(
                        "driver_poll_incomplete_sample_without_response"
                    )
                invocation = self._poll_invocation_directory(
                    operation_root, poll_sample_sequence
                )
            else:
                invocation = self._next_invocation_directory(
                    operation_root
                )
            invocation_was_run = True
            stdout_path = invocation / "stdout.bin"
            stderr_path = invocation / "stderr.bin"
            argv = (
                str(self.driver_configuration.driver_executable),
                "--configuration",
                str(self.driver_configuration.driver_configuration),
                "--request",
                str(request_path),
                "--response",
                str(response_path),
            )
            selected_timeout = (
                self.driver_configuration.operation_timeout_seconds
                if timeout_seconds is None
                else min(
                    self.driver_configuration.operation_timeout_seconds,
                    max(5, timeout_seconds),
                )
            )
            observed = self.command_runner.run_attached_stream(
                argv,
                timeout_seconds=selected_timeout,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                stdout_limit_bytes=MAX_AUXILIARY_DRIVER_STREAM_BYTES,
                stderr_limit_bytes=MAX_AUXILIARY_DRIVER_STREAM_BYTES,
            )
            stdout_raw = self._read_confined_driver_file(
                decision,
                str(stdout_path),
                maximum=MAX_AUXILIARY_DRIVER_STREAM_BYTES,
                bind_path=True,
                expected_mode=None,
                allow_empty=True,
            )
            stderr_raw = self._read_confined_driver_file(
                decision,
                str(stderr_path),
                maximum=MAX_AUXILIARY_DRIVER_STREAM_BYTES,
                bind_path=True,
                expected_mode=None,
                allow_empty=True,
            )
            if (
                hashlib.sha256(stdout_raw).hexdigest()
                != observed.stdout_sha256
                or len(stdout_raw) != observed.stdout_bytes
                or hashlib.sha256(stderr_raw).hexdigest()
                != observed.stderr_sha256
                or len(stderr_raw) != observed.stderr_bytes
            ):
                raise Runner.AuxiliaryBackendFatalError(
                    "driver_stream_binding_invalid"
                )
            _, stderr_classification = (
                Transport._probe_stderr_visibility_projection(stderr_raw)
            )
            proposed_visibility_path = stderr_path.with_name(
                "stderr_visibility_receipt.json"
            )
            if (
                proposed_visibility_path.exists()
                or proposed_visibility_path.is_symlink()
            ):
                raise Runner.AuxiliaryBackendFatalError(
                    "driver_stderr_visibility_preexisting"
                )
            (
                stderr_visibility_path,
                stderr_classification_sha256,
            ) = (
                Transport._retain_probe_stderr_visibility_receipt(
                    stderr_path, stderr_classification
                )
            )
            stderr_visibility_raw = (
                self._read_confined_driver_file(
                    decision,
                    stderr_visibility_path,
                    maximum=MAX_AUXILIARY_DRIVER_CONTROL_BYTES,
                    bind_path=True,
                )
            )
            if (
                hashlib.sha256(stderr_visibility_raw).hexdigest()
                != stderr_classification_sha256
            ):
                raise Runner.AuxiliaryBackendFatalError(
                    "driver_stderr_visibility_binding_invalid"
                )
            if (
                observed.timed_out
                or observed.output_overflow
                or observed.returncode != 0
            ):
                failure_receipt = {
                    "schema": 1,
                    "kind":
                        "arc_agi3_contiguous_auxiliary_driver_invocation",
                    "driver_protocol_sha256":
                        AUXILIARY_DRIVER_PROTOCOL_SHA256,
                    "operation": operation,
                    "assignment_id": decision.assignment_id,
                    "decision_sha256": decision.decision_sha256,
                    "request_sha256": request_sha256,
                    "argv_shape": [
                        "driver",
                        "--configuration",
                        "configuration",
                        "--request",
                        "request",
                        "--response",
                        "response",
                    ],
                    "caller_selected_scheduler_fields": False,
                    "returncode": observed.returncode,
                    "timed_out": observed.timed_out,
                    "output_overflow": observed.output_overflow,
                    "stdout_sha256": observed.stdout_sha256,
                    "stdout_bytes": observed.stdout_bytes,
                    "stderr_sha256": observed.stderr_sha256,
                    "stderr_bytes": observed.stderr_bytes,
                    "stderr_visibility_receipt_sha256":
                        stderr_classification_sha256,
                    "stderr_raw_surface_classification":
                        stderr_classification[
                            "raw_surface_classification"
                        ],
                    "response_sha256": None,
                    "response_bytes": None,
                    "response_binding_sha256": None,
                    "raw_streams_host_only": True,
                }
                _write_new(
                    invocation / "invocation_receipt.json",
                    _canonical_json(failure_receipt) + b"\n",
                    mode=0o400,
                )
                self._read_confined_driver_file(
                    decision,
                    str(invocation / "invocation_receipt.json"),
                    maximum=MAX_AUXILIARY_DRIVER_CONTROL_BYTES,
                    bind_path=True,
                )
                suffix = (
                    "timeout"
                    if observed.timed_out
                    else (
                        "overflow"
                        if observed.output_overflow
                        else "nonzero"
                    )
                )
                raise Runner.AuxiliaryBackendFatalError(
                    f"driver_{operation}_{suffix}"
                )
        elif sampled_poll:
            assert poll_sample_sequence is not None
            expected_invocation = (
                operation_root
                / f"sample-{poll_sample_sequence:016d}"
            )
            transient_invocations = self._poll_transient_invocations(
                operation_root
            )
            if transient_invocations != (expected_invocation,):
                raise Runner.AuxiliaryBackendFatalError(
                    "driver_poll_transient_sample_identity_invalid"
                )
            invocation = expected_invocation
        response_raw = self._read_confined_driver_file(
            decision,
            str(response_path),
            maximum=MAX_AUXILIARY_DRIVER_RESPONSE_BYTES,
            bind_path=True,
        )
        response = _strict_json(
            response_raw, label="auxiliary driver response"
        )
        if response_raw != _canonical_json(response) + b"\n":
            raise Runner.AuxiliaryBackendFatalError(
                "driver_response_encoding_invalid"
            )
        response_sha256 = hashlib.sha256(response_raw).hexdigest()
        response_binding = {
            "schema": 1,
            "kind":
                "arc_agi3_contiguous_auxiliary_driver_response_binding",
            "driver_protocol_sha256":
                AUXILIARY_DRIVER_PROTOCOL_SHA256,
            "operation": operation,
            "assignment_id": decision.assignment_id,
            "decision_sha256": decision.decision_sha256,
            "request_sha256": request_sha256,
            "response_sha256": response_sha256,
            "response_bytes": len(response_raw),
            "canonical_response": True,
        }
        response_binding_raw = (
            _canonical_json(response_binding) + b"\n"
        )
        if (
            response_binding_path.exists()
            or response_binding_path.is_symlink()
        ):
            if invocation is not None and invocation_was_run:
                raise Runner.AuxiliaryBackendFatalError(
                    "driver_response_binding_preexisting"
                )
            observed_binding_raw = self._read_confined_driver_file(
                decision,
                str(response_binding_path),
                maximum=MAX_AUXILIARY_DRIVER_CONTROL_BYTES,
                bind_path=True,
            )
            if observed_binding_raw != response_binding_raw:
                raise Runner.AuxiliaryBackendFatalError(
                    "driver_response_recovery_binding_invalid"
                )
        else:
            if invocation is None:
                raise Runner.AuxiliaryBackendFatalError(
                    "driver_response_recovery_binding_absent"
                )
            _write_new(
                response_binding_path,
                response_binding_raw,
                mode=0o400,
            )
            self._read_confined_driver_file(
                decision,
                str(response_binding_path),
                maximum=MAX_AUXILIARY_DRIVER_CONTROL_BYTES,
                bind_path=True,
            )
        response_binding_sha256 = hashlib.sha256(
            response_binding_raw
        ).hexdigest()
        if (
            self._read_confined_driver_file(
                decision,
                str(response_path),
                maximum=MAX_AUXILIARY_DRIVER_RESPONSE_BYTES,
                bind_path=True,
            )
            != response_raw
        ):
            raise Runner.AuxiliaryBackendFatalError(
                "driver_response_changed_after_binding"
            )
        invocation_receipt_sha256: str | None = None
        if invocation is not None:
            invocation_receipt_path = (
                invocation / "invocation_receipt.json"
            )
            if invocation_was_run:
                if (
                    observed is None
                    or stderr_classification is None
                    or stderr_classification_sha256 is None
                ):
                    raise Runner.AuxiliaryBackendFatalError(
                        "driver_invocation_evidence_incomplete"
                    )
                invocation_receipt = {
                    "schema": 1,
                    "kind":
                        "arc_agi3_contiguous_auxiliary_driver_invocation",
                    "driver_protocol_sha256":
                        AUXILIARY_DRIVER_PROTOCOL_SHA256,
                    "operation": operation,
                    "assignment_id": decision.assignment_id,
                    "decision_sha256": decision.decision_sha256,
                    "request_sha256": request_sha256,
                    "argv_shape": [
                        "driver",
                        "--configuration",
                        "configuration",
                        "--request",
                        "request",
                        "--response",
                        "response",
                    ],
                    "caller_selected_scheduler_fields": False,
                    "returncode": observed.returncode,
                    "timed_out": observed.timed_out,
                    "output_overflow": observed.output_overflow,
                    "stdout_sha256": observed.stdout_sha256,
                    "stdout_bytes": observed.stdout_bytes,
                    "stderr_sha256": observed.stderr_sha256,
                    "stderr_bytes": observed.stderr_bytes,
                    "stderr_visibility_receipt_sha256":
                        stderr_classification_sha256,
                    "stderr_raw_surface_classification":
                        stderr_classification[
                            "raw_surface_classification"
                        ],
                    "response_sha256": response_sha256,
                    "response_bytes": len(response_raw),
                    "response_binding_sha256":
                        response_binding_sha256,
                    "raw_streams_host_only": True,
                }
                if sampled_poll:
                    assert poll_sample_sequence is not None
                    invocation_receipt.update({
                        "poll_sample_sequence":
                            poll_sample_sequence,
                        "poll_sample_identity_sha256":
                            self._poll_sample_identity(
                                request_sha256=request_sha256,
                                sample_sequence=poll_sample_sequence,
                                previous_checkpoint_sha256=(
                                    poll_checkpoint_sha256
                                ),
                            ),
                        "previous_poll_checkpoint_sha256":
                            poll_checkpoint_sha256,
                    })
                expected_invocation_raw = (
                    _canonical_json(invocation_receipt) + b"\n"
                )
                _write_new(
                    invocation_receipt_path,
                    expected_invocation_raw,
                    mode=0o400,
                )
            else:
                expected_invocation_raw = (
                    self._read_confined_driver_file(
                        decision,
                        str(invocation_receipt_path),
                        maximum=MAX_AUXILIARY_DRIVER_CONTROL_BYTES,
                        bind_path=True,
                    )
                )
                invocation_receipt = _strict_json(
                    expected_invocation_raw,
                    label="recovered auxiliary invocation receipt",
                )
                expected_fields = {
                    "schema",
                    "kind",
                    "driver_protocol_sha256",
                    "operation",
                    "assignment_id",
                    "decision_sha256",
                    "request_sha256",
                    "argv_shape",
                    "caller_selected_scheduler_fields",
                    "returncode",
                    "timed_out",
                    "output_overflow",
                    "stdout_sha256",
                    "stdout_bytes",
                    "stderr_sha256",
                    "stderr_bytes",
                    "stderr_visibility_receipt_sha256",
                    "stderr_raw_surface_classification",
                    "response_sha256",
                    "response_bytes",
                    "response_binding_sha256",
                    "raw_streams_host_only",
                    "poll_sample_sequence",
                    "poll_sample_identity_sha256",
                    "previous_poll_checkpoint_sha256",
                }
                if (
                    not sampled_poll
                    or set(invocation_receipt) != expected_fields
                    or invocation_receipt["schema"] != 1
                    or invocation_receipt["kind"]
                    != (
                        "arc_agi3_contiguous_auxiliary_driver_"
                        "invocation"
                    )
                    or invocation_receipt["driver_protocol_sha256"]
                    != AUXILIARY_DRIVER_PROTOCOL_SHA256
                    or invocation_receipt["operation"] != operation
                    or invocation_receipt["assignment_id"]
                    != decision.assignment_id
                    or invocation_receipt["decision_sha256"]
                    != decision.decision_sha256
                    or invocation_receipt["request_sha256"]
                    != request_sha256
                    or invocation_receipt["returncode"] != 0
                    or invocation_receipt["timed_out"] is not False
                    or invocation_receipt["output_overflow"] is not False
                    or invocation_receipt["response_sha256"]
                    != response_sha256
                    or invocation_receipt["response_bytes"]
                    != len(response_raw)
                    or invocation_receipt[
                        "response_binding_sha256"
                    ] != response_binding_sha256
                    or invocation_receipt[
                        "poll_sample_sequence"
                    ] != poll_sample_sequence
                    or invocation_receipt[
                        "previous_poll_checkpoint_sha256"
                    ] != poll_checkpoint_sha256
                    or invocation_receipt[
                        "poll_sample_identity_sha256"
                    ] != self._poll_sample_identity(
                        request_sha256=request_sha256,
                        sample_sequence=int(poll_sample_sequence),
                        previous_checkpoint_sha256=(
                            poll_checkpoint_sha256
                        ),
                    )
                    or invocation_receipt[
                        "raw_streams_host_only"
                    ] is not True
                ):
                    raise Runner.AuxiliaryBackendFatalError(
                        "driver_poll_recovered_invocation_invalid"
                    )
                for name, digest_key, bytes_key in (
                    ("stdout.bin", "stdout_sha256", "stdout_bytes"),
                    ("stderr.bin", "stderr_sha256", "stderr_bytes"),
                ):
                    stream_raw = self._read_confined_driver_file(
                        decision,
                        str(invocation / name),
                        maximum=MAX_AUXILIARY_DRIVER_STREAM_BYTES,
                        bind_path=True,
                        expected_mode=None,
                        allow_empty=True,
                    )
                    if (
                        hashlib.sha256(stream_raw).hexdigest()
                        != invocation_receipt[digest_key]
                        or len(stream_raw)
                        != invocation_receipt[bytes_key]
                    ):
                        raise Runner.AuxiliaryBackendFatalError(
                            "driver_poll_recovered_stream_invalid"
                        )
                visibility_raw = self._read_confined_driver_file(
                    decision,
                    str(
                        invocation
                        / "stderr_visibility_receipt.json"
                    ),
                    maximum=MAX_AUXILIARY_DRIVER_CONTROL_BYTES,
                    bind_path=True,
                )
                if (
                    hashlib.sha256(visibility_raw).hexdigest()
                    != invocation_receipt[
                        "stderr_visibility_receipt_sha256"
                    ]
                ):
                    raise Runner.AuxiliaryBackendFatalError(
                        "driver_poll_recovered_visibility_invalid"
                    )
            retained_invocation = self._read_confined_driver_file(
                decision,
                str(invocation_receipt_path),
                maximum=MAX_AUXILIARY_DRIVER_CONTROL_BYTES,
                bind_path=True,
            )
            if retained_invocation != expected_invocation_raw:
                raise Runner.AuxiliaryBackendFatalError(
                    "driver_invocation_receipt_binding_invalid"
                )
            invocation_receipt_sha256 = hashlib.sha256(
                expected_invocation_raw
            ).hexdigest()
        envelope = self._strict_result(
            response,
            {
                "schema",
                "kind",
                "driver_protocol_sha256",
                "operation",
                "assignment_id",
                "decision_sha256",
                "request_sha256",
                "result",
            },
            label=operation,
        )
        if (
            envelope["schema"] != 1
            or envelope["kind"]
            != "arc_agi3_contiguous_auxiliary_driver_response"
            or envelope["driver_protocol_sha256"]
            != AUXILIARY_DRIVER_PROTOCOL_SHA256
            or envelope["operation"] != operation
            or envelope["assignment_id"] != decision.assignment_id
            or envelope["decision_sha256"]
            != decision.decision_sha256
            or envelope["request_sha256"] != request_sha256
            or not isinstance(envelope["result"], dict)
        ):
            raise Runner.AuxiliaryBackendFatalError(
                "driver_response_binding_invalid"
            )
        if sampled_poll:
            assert (
                poll_sample_sequence is not None
                and invocation_receipt_sha256 is not None
            )
            checkpoint = {
                "schema": 1,
                "kind":
                    "arc_agi3_contiguous_auxiliary_poll_checkpoint",
                "operation": "poll",
                "assignment_id": decision.assignment_id,
                "decision_sha256": decision.decision_sha256,
                "request_sha256": request_sha256,
                "sample_sequence": poll_sample_sequence,
                "previous_checkpoint_sha256":
                    poll_checkpoint_sha256,
                "sample_identity_sha256":
                    self._poll_sample_identity(
                        request_sha256=request_sha256,
                        sample_sequence=poll_sample_sequence,
                        previous_checkpoint_sha256=(
                            poll_checkpoint_sha256
                        ),
                    ),
                "response_sha256": response_sha256,
                "response_bytes": len(response_raw),
                "response_binding_sha256":
                    response_binding_sha256,
                "invocation_receipt_sha256":
                    invocation_receipt_sha256,
                "result": dict(envelope["result"]),
            }
            self._validate_poll_checkpoint(
                decision,
                checkpoint,
                request_sha256=request_sha256,
            )
            checkpoint_sha256, _ = self._publish_poll_checkpoint(
                decision, operation_root, checkpoint
            )
            if getattr(self, "_poll_crash_cut", None) == (
                "after_checkpoint_fsync"
            ):
                self._poll_crash_cut = None
                raise Runner.SimulatedCrash(
                    "after auxiliary poll checkpoint fsync"
                )
            self._compact_poll_sample(
                decision,
                operation_root,
                checkpoint,
                keep_checkpoint_sha256=checkpoint_sha256,
            )
            if getattr(self, "_poll_crash_cut", None) == (
                "after_transient_removal"
            ):
                self._poll_crash_cut = None
                raise Runner.SimulatedCrash(
                    "after auxiliary poll transient removal"
                )
        return envelope["result"]

    @staticmethod
    def _prepared_dict(
        prepared: Runner.AuxiliaryPreparedInput,
    ) -> dict[str, object]:
        return asdict(prepared)

    @staticmethod
    def _launched_dict(
        launched: Runner.AuxiliaryLaunch,
    ) -> dict[str, object]:
        return asdict(launched)

    def prepare(
        self, decision: Scheduler.AuxiliaryDecision
    ) -> Runner.AuxiliaryPreparedInput:
        result = self._strict_result(
            self._invoke("prepare", decision, {}),
            set(Runner.AuxiliaryPreparedInput.__dataclass_fields__),
            label="prepare result",
        )
        return Runner.AuxiliaryPreparedInput(
            **{
                key: self._confined_path(
                    value,
                    decision=decision,
                    label=key,
                )
                if key.endswith("_path")
                else value
                for key, value in result.items()
            }
        )

    def launch(
        self,
        decision: Scheduler.AuxiliaryDecision,
        prepared: Runner.AuxiliaryPreparedInput,
    ) -> Runner.AuxiliaryLaunch:
        result = self._strict_result(
            self._invoke(
                "launch",
                decision,
                {"prepared": self._prepared_dict(prepared)},
            ),
            set(Runner.AuxiliaryLaunch.__dataclass_fields__),
            label="launch result",
        )
        return Runner.AuxiliaryLaunch(
            launch_receipt_path=self._confined_path(
                result["launch_receipt_path"],
                decision=decision,
                label="launch receipt",
            ),
            launch_receipt_sha256=result["launch_receipt_sha256"],
        )

    def poll(
        self,
        decision: Scheduler.AuxiliaryDecision,
        prepared: Runner.AuxiliaryPreparedInput,
        launched: Runner.AuxiliaryLaunch,
        *,
        timeout_seconds: float,
    ) -> Runner.AuxiliaryPoll:
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(float(timeout_seconds))
            or timeout_seconds <= 0
        ):
            raise ContiguousOrchestratorError(
                "auxiliary poll timeout is malformed"
            )
        result = self._strict_result(
            self._invoke(
                "poll",
                decision,
                {
                    "prepared": self._prepared_dict(prepared),
                    "launched": self._launched_dict(launched),
                    "timeout_seconds": float(timeout_seconds),
                },
                cacheable=False,
                timeout_seconds=math.ceil(float(timeout_seconds)) + 5,
            ),
            set(Runner.AuxiliaryPoll.__dataclass_fields__),
            label="poll result",
        )
        reason = self._safe_reason(
            result["reason"], allow_none=False
        )
        status = result["status"]
        if (
            status not in {"running", "exited", "containment_fault"}
            or not isinstance(result["observation_sha256"], str)
            or SHA256_RE.fullmatch(result["observation_sha256"]) is None
            or (status != "containment_fault" and reason != "none")
        ):
            raise Runner.AuxiliaryBackendFatalError(
                "driver_poll_result_invalid"
            )
        return Runner.AuxiliaryPoll(
            status=status,
            observation_sha256=result["observation_sha256"],
            reason="" if reason == "none" else str(reason),
        )

    def collect(
        self,
        decision: Scheduler.AuxiliaryDecision,
        prepared: Runner.AuxiliaryPreparedInput,
        launched: Runner.AuxiliaryLaunch,
        terminal: Runner.AuxiliaryPoll,
    ) -> Runner.AuxiliaryCollection:
        result = self._strict_result(
            self._invoke(
                "collect",
                decision,
                {
                    "prepared": self._prepared_dict(prepared),
                    "launched": self._launched_dict(launched),
                    "terminal": asdict(terminal),
                },
            ),
            {"output", "cost_used", "abort_reason"},
            label="collect result",
        )
        try:
            output = (
                None
                if result["output"] is None
                else Scheduler.auxiliary_output_from_dict(
                    result["output"]
                )
            )
        except Scheduler.SchedulerError as exc:
            raise Runner.AuxiliaryBackendFatalError(
                "driver_output_evidence_invalid"
            ) from exc
        abort_reason = self._safe_reason(
            result["abort_reason"], allow_none=True
        )
        cost_used = result["cost_used"]
        if (
            isinstance(cost_used, bool)
            or not isinstance(cost_used, (int, float))
            or not math.isfinite(float(cost_used))
            or cost_used < 0
            or (output is None) == (abort_reason is None)
        ):
            raise Runner.AuxiliaryBackendFatalError(
                "driver_collection_invalid"
            )
        return Runner.AuxiliaryCollection(
            output=output,
            cost_used=float(cost_used),
            abort_reason=abort_reason,
        )

    def teardown(
        self,
        decision: Scheduler.AuxiliaryDecision,
        prepared: Runner.AuxiliaryPreparedInput,
        launched: Runner.AuxiliaryLaunch,
        collection: Runner.AuxiliaryCollection,
    ) -> Runner.AuxiliaryTeardown:
        result = self._strict_result(
            self._invoke(
                "teardown",
                decision,
                {
                    "prepared": self._prepared_dict(prepared),
                    "launched": self._launched_dict(launched),
                    "collection": {
                        "output": (
                            asdict(collection.output)
                            if collection.output is not None
                            else None
                        ),
                        "cost_used": collection.cost_used,
                        "abort_reason": collection.abort_reason,
                    },
                },
            ),
            set(Runner.AuxiliaryTeardown.__dataclass_fields__),
            label="teardown result",
        )
        return Runner.AuxiliaryTeardown(
            teardown_receipt_path=self._confined_path(
                result["teardown_receipt_path"],
                decision=decision,
                label="teardown receipt",
            ),
            teardown_receipt_sha256=result[
                "teardown_receipt_sha256"
            ],
        )

    def admit(
        self,
        decision: Scheduler.AuxiliaryDecision,
        output: Scheduler.AuxiliaryOutputEvidence,
    ) -> Runner.AuxiliaryAdmission:
        result = self._strict_result(
            self._invoke(
                "admit",
                decision,
                {"output": asdict(output)},
            ),
            set(Runner.AuxiliaryAdmission.__dataclass_fields__),
            label="admission result",
        )
        try:
            profile = (
                None
                if result["profile"] is None
                else Scheduler.complexity_profile_from_dict(
                    result["profile"]
                )
            )
        except Scheduler.SchedulerError as exc:
            raise Runner.AuxiliaryBackendFatalError(
                "driver_admission_profile_invalid"
            ) from exc
        reason = self._safe_reason(
            result["reason"], allow_none=True
        )
        normalized: dict[str, object] = dict(result)
        normalized["profile"] = profile
        normalized["reason"] = reason
        for key, value in tuple(normalized.items()):
            if key.endswith("_path") and value is not None:
                normalized[key] = self._confined_path(
                    value,
                    decision=decision,
                    label=key,
                )
        return Runner.AuxiliaryAdmission(**normalized)

    def abort(
        self,
        decision: Scheduler.AuxiliaryDecision,
        prepared: Runner.AuxiliaryPreparedInput | None,
        launched: Runner.AuxiliaryLaunch | None,
        *,
        prior_phase: Literal["RESERVED", "INPUT_PREPARED", "RUNNING"],
        reason: str,
    ) -> Runner.AuxiliaryAbort:
        # ``reason`` is runner-generated.  Convert it to a finite category
        # before the driver boundary so Python exception prose never becomes
        # driver/model input.
        reason_code = (
            reason
            if _AUXILIARY_REASON_CODE_RE.fullmatch(reason)
            else "host_backend_failure"
        )
        result = self._strict_result(
            self._invoke(
                "abort",
                decision,
                {
                    "prepared": (
                        self._prepared_dict(prepared)
                        if prepared is not None
                        else None
                    ),
                    "launched": (
                        self._launched_dict(launched)
                        if launched is not None
                        else None
                    ),
                    "prior_phase": prior_phase,
                    "reason": reason_code,
                },
            ),
            {"cost_used", "teardown"},
            label="abort result",
        )
        cost_used = result["cost_used"]
        if (
            isinstance(cost_used, bool)
            or not isinstance(cost_used, (int, float))
            or not math.isfinite(float(cost_used))
            or cost_used < 0
        ):
            raise ContiguousOrchestratorError(
                "auxiliary abort usage is malformed"
            )
        teardown = None
        if result["teardown"] is not None:
            teardown_raw = self._strict_result(
                result["teardown"],
                set(Runner.AuxiliaryTeardown.__dataclass_fields__),
                label="abort teardown",
            )
            teardown = Runner.AuxiliaryTeardown(
                teardown_receipt_path=self._confined_path(
                    teardown_raw["teardown_receipt_path"],
                    decision=decision,
                    label="abort teardown receipt",
                ),
                teardown_receipt_sha256=teardown_raw[
                    "teardown_receipt_sha256"
                ],
            )
        if (prior_phase == "RUNNING") != (teardown is not None):
            raise ContiguousOrchestratorError(
                "auxiliary abort teardown phase differs"
            )
        return Runner.AuxiliaryAbort(
            cost_used=float(cost_used), teardown=teardown
        )


@dataclass(frozen=True)
class OperatorConfiguration:
    """Strict, host-only input to the formal production entry point."""

    config_path: Path
    config_sha256: str
    campaign_root: Path
    promotion_root: Path
    replay_evidence_root: Path
    docker_binary: Path
    docker_binary_sha256: str
    docker_socket: Path
    docker_config_root: Path
    python_executable: Path
    python_executable_sha256: str
    python_runtime_manifest: Path
    python_runtime_manifest_sha256: str
    runtime_control_snapshot_root: Path
    credential_source: Path
    launch_attestation: Path
    conformance_result: Path
    pilot_gate_receipt: Path
    pilot_authentication_key: Path
    pilot_production_stack_attestation_sha256: str
    canonical_root: Path
    environments_root: Path
    workspace_probe_image_reference: str
    replay_image_reference: str
    backend_configuration: Runner.BackendConfiguration
    auxiliary_launch_configuration: Scheduler.AuxiliaryLaunchConfiguration
    auxiliary_backend_configuration: AuxiliaryBackendDriverConfiguration
    cost_window_id: str
    limit: float | None
    max_lanes: int
    poll_interval_seconds: float
    terminal_condition: str
    canary_placements: Mapping[str, str]
    path_relationships: Mapping[str, object]
    path_relationships_sha256: str
    frontier_import_root: Path | None = None
    selective_continuation_game: str | None = None
    selective_frontier_import_sha256: str | None = None


_OPERATOR_CONFIG_FIELDS = frozenset({
    "schema",
    "campaign_root",
    "promotion_root",
    "replay_evidence_root",
    "docker_binary",
    "docker_binary_sha256",
    "docker_socket",
    "docker_config_root",
    "python_executable",
    "python_executable_sha256",
    "python_runtime_manifest",
    "python_runtime_manifest_sha256",
    "runtime_control_snapshot_root",
    "credential_source",
    "launch_attestation",
    "conformance_result",
    "pilot_gate_receipt",
    "pilot_authentication_key",
    "pilot_production_stack_attestation_sha256",
    "canonical_root",
    "environments_root",
    "workspace_probe_image_reference",
    "replay_image_reference",
    "backend_configuration",
    "auxiliary_launch_configuration",
    "auxiliary_backend_configuration",
    "cost_window_id",
    "limit",
    "max_lanes",
    "poll_interval_seconds",
    "terminal_condition",
    "canary_placements",
})
_SELECTIVE_OPERATOR_CONFIG_FIELDS = _OPERATOR_CONFIG_FIELDS | frozenset({
    "frontier_import_root",
    "selective_continuation_game",
    "selective_frontier_import_sha256",
})
_BACKEND_CONFIGURATION_FIELDS = frozenset({
    "image_reference",
    "image_digest",
    "worker_command",
    "resource_limits",
    "proposer_transport",
})
_RESOURCE_LIMIT_FIELDS = frozenset({
    "cpus", "memory_bytes", "pids", "tmpfs_bytes",
})


def _private_operator_config_bytes(path: Path) -> bytes:
    selected = Path(path)
    try:
        descriptor = os.open(
            selected,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
        )
    except OSError as exc:
        raise ContiguousOrchestratorError(
            "operator config must be an unaliased regular file"
        ) from exc
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or metadata.st_uid != os.getuid()
            or stat.S_IMODE(metadata.st_mode) != 0o400
            or not 0 < metadata.st_size <= MAX_OPERATOR_CONFIG_BYTES
        ):
            raise ContiguousOrchestratorError(
                "operator config must be owner-held mode 0400"
            )
        remaining = metadata.st_size
        chunks: list[bytes] = []
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                raise ContiguousOrchestratorError(
                    "operator config changed while reading"
                )
            chunks.append(chunk)
            remaining -= len(chunk)
        after = os.fstat(descriptor)
        if any(
            getattr(metadata, name) != getattr(after, name)
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
            )
        ):
            raise ContiguousOrchestratorError(
                "operator config changed while reading"
            )
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _absolute_path(value: object, *, label: str) -> Path:
    if (
        not isinstance(value, str)
        or not value
        or "\x00" in value
        or not Path(value).is_absolute()
        or Path(value) == Path("/")
        or str(Path(value)) != value
        or os.path.normpath(value) != value
        or any(part in {"", ".", ".."} for part in Path(value).parts[1:])
    ):
        raise ContiguousOrchestratorError(
            f"{label} must be one canonical explicit absolute path"
        )
    return Path(value)


def _operator_path_metadata(
    metadata: os.stat_result,
) -> dict[str, object]:
    if stat.S_ISDIR(metadata.st_mode):
        kind = "directory"
    elif stat.S_ISREG(metadata.st_mode):
        kind = "regular"
    elif stat.S_ISSOCK(metadata.st_mode):
        kind = "socket"
    else:
        kind = "other"
    return {
        "device": metadata.st_dev,
        "inode": metadata.st_ino,
        "mode": metadata.st_mode,
        "links": metadata.st_nlink,
        "owner_uid": metadata.st_uid,
        "owner_gid": metadata.st_gid,
        "kind": kind,
    }


def _descriptor_walk_operator_path(
    path: Path,
    *,
    label: str,
    allow_missing_tail: bool,
    final_kind: Literal["directory", "file", "file_or_missing"],
) -> dict[str, object]:
    selected = _absolute_path(str(path), label=label)
    components = selected.parts[1:]
    descriptor = os.open(
        "/",
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    traversed: list[dict[str, object]] = []
    missing: tuple[str, ...] = ()
    prefix = Path("/")
    try:
        for index, component in enumerate(components):
            is_last = index == len(components) - 1
            try:
                before = os.stat(
                    component,
                    dir_fd=descriptor,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                if not allow_missing_tail:
                    raise ContiguousOrchestratorError(
                        f"{label} is missing"
                    )
                missing = tuple(components[index:])
                break
            except OSError as exc:
                raise ContiguousOrchestratorError(
                    f"{label} cannot be descriptor-walked"
                ) from exc
            if stat.S_ISLNK(before.st_mode):
                raise ContiguousOrchestratorError(
                    f"{label} traverses a symlink alias"
                )
            prefix = prefix / component
            identity = _operator_path_metadata(before)
            traversed.append({
                "path": str(prefix),
                **identity,
            })
            if not is_last:
                if not stat.S_ISDIR(before.st_mode):
                    raise ContiguousOrchestratorError(
                        f"{label} has a nondirectory ancestor"
                    )
                try:
                    next_descriptor = os.open(
                        component,
                        os.O_RDONLY
                        | getattr(os, "O_DIRECTORY", 0)
                        | getattr(os, "O_NOFOLLOW", 0),
                        dir_fd=descriptor,
                    )
                except OSError as exc:
                    raise ContiguousOrchestratorError(
                        f"{label} ancestor changed during descriptor walk"
                    ) from exc
                after = os.fstat(next_descriptor)
                rebound = os.stat(
                    component,
                    dir_fd=descriptor,
                    follow_symlinks=False,
                )
                if (
                    _operator_path_metadata(after) != identity
                    or _operator_path_metadata(rebound) != identity
                ):
                    os.close(next_descriptor)
                    raise ContiguousOrchestratorError(
                        f"{label} ancestor changed during descriptor walk"
                    )
                os.close(descriptor)
                descriptor = next_descriptor
            else:
                rebound = os.stat(
                    component,
                    dir_fd=descriptor,
                    follow_symlinks=False,
                )
                if _operator_path_metadata(rebound) != identity:
                    raise ContiguousOrchestratorError(
                        f"{label} changed during descriptor walk"
                    )
        if missing:
            if final_kind == "file" or (
                final_kind == "file_or_missing"
                and len(missing) != 1
            ):
                raise ContiguousOrchestratorError(
                    f"{label} lacks its exact existing parent"
                )
        elif traversed:
            observed_kind = traversed[-1]["kind"]
            if (
                final_kind == "directory"
                and observed_kind != "directory"
            ):
                raise ContiguousOrchestratorError(
                    f"{label} is not a directory"
                )
            if (
                final_kind in {"file", "file_or_missing"}
                and observed_kind == "directory"
            ):
                raise ContiguousOrchestratorError(
                    f"{label} is not a file endpoint"
                )
    finally:
        os.close(descriptor)
    nearest = (
        traversed[-1]["path"] if traversed else "/"
    )
    return {
        "path": str(selected),
        "existing_ancestors": traversed,
        "nearest_existing_path": nearest,
        "missing_suffix": list(missing),
        "final_kind": final_kind,
    }


def _path_contains_or_equals(left: Path, right: Path) -> bool:
    left_parts = left.parts
    right_parts = right.parts
    return (
        len(left_parts) <= len(right_parts)
        and right_parts[:len(left_parts)] == left_parts
    )


def _operator_path_relationship_projection(
    *,
    config_path: Path,
    paths: Mapping[str, Path],
    auxiliary_configuration: AuxiliaryBackendDriverConfiguration,
    canary_placements: Mapping[str, str],
    selective_frontier_import_sha256: str | None = None,
) -> dict[str, object]:
    if (
        ("frontier_import_root" in paths)
        != (selective_frontier_import_sha256 is not None)
        or (
            selective_frontier_import_sha256 is not None
            and (
                not isinstance(
                    selective_frontier_import_sha256, str
                )
                or SHA256_RE.fullmatch(
                    selective_frontier_import_sha256
                )
                is None
            )
        )
    ):
        raise ContiguousOrchestratorError(
            "selective path projection lacks its authorized import digest"
        )
    mutable = {
        name: paths[name]
        for name in (
            "campaign_root",
            "promotion_root",
            "replay_evidence_root",
        )
    }
    broad_mutation_targets = {
        Path("/"),
        Path.home().resolve(),
        Path(__file__).resolve().parents[2],
        Path.cwd().resolve(),
        Path("/tmp"),
        Path("/private/tmp"),
        Path(tempfile.gettempdir()).resolve(),
    }
    checkout_root = Path(__file__).resolve().parents[2]
    for name, selected in mutable.items():
        canonical = _absolute_path(str(selected), label=name)
        if canonical in broad_mutation_targets:
            raise ContiguousOrchestratorError(
                f"{name} is an ambient broad directory"
            )
        if _path_contains_or_equals(checkout_root, canonical):
            raise ContiguousOrchestratorError(
                f"{name} may not retain campaign bytes in the checkout"
            )
    authority_roots = {
        name: paths[name]
        for name in (
            "docker_config_root",
            "runtime_control_snapshot_root",
            "canonical_root",
            "environments_root",
        )
    }
    if "frontier_import_root" in paths:
        authority_roots["frontier_import_root"] = paths[
            "frontier_import_root"
        ]
    authority_files = {
        "config_path": config_path,
        **{
            name: paths[name]
            for name in (
                "docker_binary",
                "docker_socket",
                "python_executable",
                "python_runtime_manifest",
                "credential_source",
                "launch_attestation",
                "conformance_result",
                "pilot_gate_receipt",
                "pilot_authentication_key",
            )
        },
        "auxiliary_driver_executable":
            auxiliary_configuration.driver_executable,
        "auxiliary_driver_configuration":
            auxiliary_configuration.driver_configuration,
        "auxiliary_backend_attestation":
            auxiliary_configuration.backend_attestation,
    }
    canary_files = {
        f"canary_{category}": _absolute_path(
            canary_placements[category],
            label=f"{category} canary placement",
        )
        for category in (
            set(Taint.CONTROLLER_CANARY_CATEGORIES) - {"environment"}
        )
    }
    for name, selected in canary_files.items():
        if _path_contains_or_equals(checkout_root, selected):
            raise ContiguousOrchestratorError(
                f"{name} may not plant canary bytes in the checkout"
            )
    roles: dict[str, tuple[str, Path]] = {
        **{
            name: ("mutable_root", value)
            for name, value in mutable.items()
        },
        **{
            name: ("authority_root", value)
            for name, value in authority_roots.items()
        },
        **{
            name: ("authority_file", value)
            for name, value in authority_files.items()
        },
        **{
            name: ("canary_file", value)
            for name, value in canary_files.items()
        },
    }
    role_names = sorted(roles)
    for index, left_name in enumerate(role_names):
        left_class, left_path = roles[left_name]
        for right_name in role_names[index + 1:]:
            right_class, right_path = roles[right_name]
            shared_conformance_endpoint = (
                {left_name, right_name}
                == {"launch_attestation", "conformance_result"}
                and left_path == right_path
            )
            left_contains = _path_contains_or_equals(
                left_path, right_path
            )
            right_contains = _path_contains_or_equals(
                right_path, left_path
            )
            exact = left_path == right_path
            if (
                (exact and not shared_conformance_endpoint)
                or (
                    "mutable_root" in {left_class, right_class}
                    and (left_contains or right_contains)
                )
                or (
                    left_class == right_class == "canary_file"
                    and (left_contains or right_contains)
                )
                or (
                    {left_class, right_class}
                    == {"canary_file", "authority_file"}
                    and (left_contains or right_contains)
                )
            ):
                raise ContiguousOrchestratorError(
                    "operator path relationship is forbidden: "
                    f"{left_name} <-> {right_name}"
                )

    def observe() -> dict[str, object]:
        result: dict[str, object] = {}
        for name in role_names:
            role, selected = roles[name]
            if role == "mutable_root":
                allow_missing_tail = True
                final_kind: Literal[
                    "directory", "file", "file_or_missing"
                ] = "directory"
            elif role == "authority_root":
                allow_missing_tail = False
                final_kind = "directory"
            elif role == "canary_file":
                allow_missing_tail = True
                final_kind = "file_or_missing"
            else:
                allow_missing_tail = False
                final_kind = "file"
            result[name] = {
                "role": role,
                **_descriptor_walk_operator_path(
                    selected,
                    label=name,
                    allow_missing_tail=allow_missing_tail,
                    final_kind=final_kind,
                ),
            }
        return result

    first = observe()
    for index, left_name in enumerate(role_names):
        left_class, left_path = roles[left_name]
        left_observation = first[left_name]
        for right_name in role_names[index + 1:]:
            right_class, right_path = roles[right_name]
            right_observation = first[right_name]
            left_rows = left_observation["existing_ancestors"]
            right_rows = right_observation["existing_ancestors"]
            left_complete = not left_observation["missing_suffix"]
            right_complete = not right_observation["missing_suffix"]
            if left_complete and right_complete:
                left_final = left_rows[-1]
                right_final = right_rows[-1]
                if (
                    left_final["device"],
                    left_final["inode"],
                ) == (
                    right_final["device"],
                    right_final["inode"],
                ) and not (
                    {left_name, right_name}
                    == {"launch_attestation", "conformance_result"}
                    and left_path == right_path
                ):
                    raise ContiguousOrchestratorError(
                        "operator endpoint identity is aliased: "
                        f"{left_name} <-> {right_name}"
                    )
            identity_disjoint = (
                "mutable_root" in {left_class, right_class}
                or left_class == right_class == "canary_file"
                or {
                    left_class, right_class
                } == {"canary_file", "authority_file"}
            )
            if not identity_disjoint:
                continue
            common_parts = 0
            for left_part, right_part in zip(
                left_path.parts, right_path.parts
            ):
                if left_part != right_part:
                    break
                common_parts += 1
            left_divergent = {
                (row["device"], row["inode"])
                for row in left_rows
                if len(Path(str(row["path"])).parts)
                > common_parts
            }
            right_divergent = {
                (row["device"], row["inode"])
                for row in right_rows
                if len(Path(str(row["path"])).parts)
                > common_parts
            }
            if left_divergent & right_divergent:
                raise ContiguousOrchestratorError(
                    "operator disjoint paths share an aliased ancestor: "
                    f"{left_name} <-> {right_name}"
                )
    second = observe()
    if first != second:
        raise ContiguousOrchestratorError(
            "operator path identity changed during pre-mutation binding"
        )
    projection: dict[str, object] = {
        "schema": 1,
        "kind": "arc_agi3_operator_path_relationships",
        "allowed_matrix": {
            "mutable_root:mutable_root": "disjoint",
            "mutable_root:any_other": "disjoint",
            "authority_root:authority_root": "containment_allowed",
            "authority_root:authority_file": "containment_allowed",
            "authority_root:canary_file": "containment_allowed",
            "authority_file:authority_file": "distinct",
            "launch_attestation:conformance_result":
                "same_exact_receipt_required",
            "authority_file:canary_file": "disjoint",
            "canary_file:canary_file": "disjoint",
        },
        "roles": first,
        "environment_canary_name":
            canary_placements["environment"],
    }
    if selective_frontier_import_sha256 is not None:
        projection["selective_frontier_import_sha256"] = (
            selective_frontier_import_sha256
        )
    return projection


def _revalidate_operator_path_relationships(
    config: OperatorConfiguration,
) -> None:
    paths = {
        name: getattr(config, name)
        for name in (
            "campaign_root",
            "promotion_root",
            "replay_evidence_root",
            "docker_binary",
            "docker_socket",
            "docker_config_root",
            "python_executable",
            "python_runtime_manifest",
            "runtime_control_snapshot_root",
            "credential_source",
            "launch_attestation",
            "conformance_result",
            "pilot_gate_receipt",
            "pilot_authentication_key",
            "canonical_root",
            "environments_root",
        )
    }
    frontier_import_root = getattr(
        config, "frontier_import_root", None
    )
    if frontier_import_root is not None:
        paths["frontier_import_root"] = frontier_import_root
    observed = _operator_path_relationship_projection(
        config_path=config.config_path,
        paths=paths,
        auxiliary_configuration=(
            config.auxiliary_backend_configuration
        ),
        canary_placements=config.canary_placements,
        selective_frontier_import_sha256=getattr(
            config, "selective_frontier_import_sha256", None
        ),
    )
    if (
        observed != config.path_relationships
        or _json_sha256(observed)
        != config.path_relationships_sha256
    ):
        raise ContiguousOrchestratorError(
            "operator path identities changed before first mutation"
        )


def _verify_executable(
    path: Path, expected_sha256: str, *, label: str
) -> None:
    if SHA256_RE.fullmatch(expected_sha256) is None:
        raise ContiguousOrchestratorError(
            f"{label} digest is malformed"
        )
    raw = _read_regular(
        path, maximum=1024 * 1024 * 1024
    )
    metadata = os.stat(path, follow_symlinks=False)
    if (
        hashlib.sha256(raw).hexdigest() != expected_sha256
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or not metadata.st_mode & stat.S_IXUSR
    ):
        raise ContiguousOrchestratorError(
            f"{label} is not the exact executable bytes"
        )


def _parse_backend_configuration(
    value: object,
) -> Runner.BackendConfiguration:
    if (
        not isinstance(value, dict)
        or set(value) != _BACKEND_CONFIGURATION_FIELDS
        or not isinstance(value.get("resource_limits"), dict)
        or set(value["resource_limits"]) != _RESOURCE_LIMIT_FIELDS
        or not isinstance(value.get("worker_command"), list)
    ):
        raise ContiguousOrchestratorError(
            "backend_configuration schema is not exact"
        )
    try:
        transport = Runner._transport_from_dict(
            value["proposer_transport"]
        )
        limits = Runner.ResourceLimitsProjection(
            **value["resource_limits"]
        )
        result = Runner.BackendConfiguration(
            image_reference=value["image_reference"],
            image_digest=value["image_digest"],
            worker_command=tuple(value["worker_command"]),
            resource_limits=limits,
            proposer_transport=transport,
        )
    except (KeyError, TypeError, Runner.ContiguousRunnerError) as exc:
        raise ContiguousOrchestratorError(
            "backend_configuration could not be typed"
        ) from exc
    if (
        not Runner._valid_backend_configuration(result)
        or Runner._backend_configuration_to_dict(result) != value
    ):
        raise ContiguousOrchestratorError(
            "backend_configuration differs from the production contract"
        )
    return result


def _validate_image_reference(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9./:_-]*@sha256:[0-9a-f]{64}",
            value,
        )
        is None
    ):
        raise ContiguousOrchestratorError(
            f"{label} must be digest-pinned"
        )
    return value


def load_operator_configuration(path: Path) -> OperatorConfiguration:
    """Parse and fully validate the host-only config without mutation."""

    config_path = _absolute_path(
        str(path), label="operator config"
    )
    raw = _private_operator_config_bytes(config_path)
    value = _strict_json(raw, label="operator config")
    selected_fields = frozenset(value)
    selective_mode = selected_fields == _SELECTIVE_OPERATOR_CONFIG_FIELDS
    if (
        selected_fields not in {
            _OPERATOR_CONFIG_FIELDS,
            _SELECTIVE_OPERATOR_CONFIG_FIELDS,
        }
        or value["schema"] != 1
    ):
        raise ContiguousOrchestratorError(
            "operator config has missing, duplicate, or unknown fields"
        )
    backend_configuration = _parse_backend_configuration(
        value["backend_configuration"]
    )
    try:
        auxiliary_launch_configuration = (
            Scheduler.auxiliary_launch_configuration_from_dict(
                value["auxiliary_launch_configuration"]
            )
        )
    except Scheduler.SchedulerError as exc:
        raise ContiguousOrchestratorError(
            "auxiliary_launch_configuration is not canonical"
        ) from exc
    auxiliary_backend_configuration = (
        _parse_auxiliary_backend_configuration(
            value["auxiliary_backend_configuration"],
            auxiliary_launch_configuration,
        )
    )
    path_names = (
        "campaign_root",
        "promotion_root",
        "replay_evidence_root",
        "docker_binary",
        "docker_socket",
        "docker_config_root",
        "python_executable",
        "python_runtime_manifest",
        "runtime_control_snapshot_root",
        "credential_source",
        "launch_attestation",
        "conformance_result",
        "pilot_gate_receipt",
        "pilot_authentication_key",
        "canonical_root",
        "environments_root",
    )
    if selective_mode:
        path_names = (*path_names, "frontier_import_root")
    paths = {
        name: _absolute_path(value[name], label=name)
        for name in path_names
    }
    if paths["launch_attestation"] != paths["conformance_result"]:
        raise ContiguousOrchestratorError(
            "launch attestation and conformance result must be the same "
            "exact receipt"
        )
    _verify_executable(
        paths["docker_binary"],
        value["docker_binary_sha256"],
        label="Docker executable",
    )
    try:
        RuntimeManifest.load_runtime_manifest(
            paths["python_runtime_manifest"],
            expected_sha256=(
                value["python_runtime_manifest_sha256"]
            ),
            python_executable=paths["python_executable"],
            python_executable_sha256=(
                value["python_executable_sha256"]
            ),
        )
    except (
        KeyError,
        TypeError,
        RuntimeManifest.RuntimeManifestError,
    ) as exc:
        raise ContiguousOrchestratorError(
            "Python runtime manifest is not exact current evidence"
        ) from exc
    required_existing_paths = (
        "docker_socket",
        "docker_config_root",
        "credential_source",
        "launch_attestation",
        "conformance_result",
        "pilot_gate_receipt",
        "pilot_authentication_key",
        "canonical_root",
        "environments_root",
    )
    if selective_mode:
        required_existing_paths = (
            *required_existing_paths,
            "frontier_import_root",
        )
    for name in required_existing_paths:
        if not paths[name].exists() or paths[name].is_symlink():
            raise ContiguousOrchestratorError(
                f"{name} does not exist as an explicit unaliased path"
            )
    placements = value["canary_placements"]
    if (
        not isinstance(placements, dict)
        or set(placements) != set(Taint.CONTROLLER_CANARY_CATEGORIES)
        or any(
            not isinstance(item, str) or not item or "\x00" in item
            for item in placements.values()
        )
        or not ENVIRONMENT_NAME_RE.fullmatch(
            placements["environment"]
        )
    ):
        raise ContiguousOrchestratorError(
            "canary placement map must cover the exact six categories"
        )
    for category in (
        set(Taint.CONTROLLER_CANARY_CATEGORIES) - {"environment"}
    ):
        placement = _absolute_path(
            placements[category],
            label=f"{category} canary placement",
        )
        if not placement.parent.is_dir() or placement.parent.is_symlink():
            raise ContiguousOrchestratorError(
                f"{category} canary parent is unavailable"
            )
    path_relationships = _operator_path_relationship_projection(
        config_path=config_path,
        paths=paths,
        auxiliary_configuration=auxiliary_backend_configuration,
        canary_placements=placements,
        selective_frontier_import_sha256=(
            value["selective_frontier_import_sha256"]
            if selective_mode
            else None
        ),
    )
    path_relationships_sha256 = _json_sha256(
        path_relationships
    )
    if (
        not isinstance(value["cost_window_id"], str)
        or re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}",
            value["cost_window_id"],
        )
        is None
        or (
            value["limit"] is not None
            and (
                isinstance(value["limit"], bool)
                or not isinstance(value["limit"], (int, float))
                or not math.isfinite(float(value["limit"]))
                or value["limit"] < 0
            )
        )
        or isinstance(value["max_lanes"], bool)
        or not isinstance(value["max_lanes"], int)
        or not 1 <= value["max_lanes"] <= Runner.MAX_LANES
        or isinstance(value["poll_interval_seconds"], bool)
        or not isinstance(
            value["poll_interval_seconds"], (int, float)
        )
        or not math.isfinite(float(value["poll_interval_seconds"]))
        or not 0.05 <= value["poll_interval_seconds"] <= 30
        or not isinstance(
            value["pilot_production_stack_attestation_sha256"],
            str,
        )
        or SHA256_RE.fullmatch(
            value["pilot_production_stack_attestation_sha256"]
        )
        is None
        or value["terminal_condition"]
        != (
            SELECTIVE_TERMINAL_CONDITION
            if selective_mode
            else CANONICAL_TERMINAL_CONDITION
        )
        or (
            selective_mode
            and (
                not isinstance(
                    value["selective_continuation_game"], str
                )
                or re.fullmatch(
                    r"[a-z0-9]{4}",
                    value["selective_continuation_game"],
                )
                is None
                or value["selective_continuation_game"]
                not in Supervisor.authoritative_inventory()
                or not isinstance(
                    value["selective_frontier_import_sha256"], str
                )
                or SHA256_RE.fullmatch(
                    value["selective_frontier_import_sha256"]
                )
                is None
            )
        )
    ):
        raise ContiguousOrchestratorError(
            "operator scheduling controls are malformed"
        )
    if (
        backend_configuration.image_reference
        != value["backend_configuration"]["image_reference"]
        or not IMAGE_DIGEST_RE.fullmatch(
            backend_configuration.image_digest
        )
    ):
        raise ContiguousOrchestratorError(
            "operator proposer image binding differs"
        )
    return OperatorConfiguration(
        config_path=config_path,
        config_sha256=hashlib.sha256(raw).hexdigest(),
        campaign_root=paths["campaign_root"],
        promotion_root=paths["promotion_root"],
        replay_evidence_root=paths["replay_evidence_root"],
        docker_binary=paths["docker_binary"],
        docker_binary_sha256=value["docker_binary_sha256"],
        docker_socket=paths["docker_socket"],
        docker_config_root=paths["docker_config_root"],
        python_executable=paths["python_executable"],
        python_executable_sha256=value["python_executable_sha256"],
        python_runtime_manifest=paths["python_runtime_manifest"],
        python_runtime_manifest_sha256=(
            value["python_runtime_manifest_sha256"]
        ),
        runtime_control_snapshot_root=(
            paths["runtime_control_snapshot_root"]
        ),
        credential_source=paths["credential_source"],
        launch_attestation=paths["launch_attestation"],
        conformance_result=paths["conformance_result"],
        pilot_gate_receipt=paths["pilot_gate_receipt"],
        pilot_authentication_key=paths["pilot_authentication_key"],
        pilot_production_stack_attestation_sha256=value[
            "pilot_production_stack_attestation_sha256"
        ],
        canonical_root=paths["canonical_root"],
        environments_root=paths["environments_root"],
        workspace_probe_image_reference=_validate_image_reference(
            value["workspace_probe_image_reference"],
            label="workspace-probe image",
        ),
        replay_image_reference=_validate_image_reference(
            value["replay_image_reference"],
            label="replay image",
        ),
        backend_configuration=backend_configuration,
        auxiliary_launch_configuration=(
            auxiliary_launch_configuration
        ),
        auxiliary_backend_configuration=(
            auxiliary_backend_configuration
        ),
        cost_window_id=value["cost_window_id"],
        limit=(
            None
            if value["limit"] is None
            else float(value["limit"])
        ),
        max_lanes=value["max_lanes"],
        poll_interval_seconds=float(value["poll_interval_seconds"]),
        terminal_condition=value["terminal_condition"],
        canary_placements=dict(placements),
        path_relationships=path_relationships,
        path_relationships_sha256=path_relationships_sha256,
        frontier_import_root=(
            paths["frontier_import_root"]
            if selective_mode
            else None
        ),
        selective_continuation_game=(
            value["selective_continuation_game"]
            if selective_mode
            else None
        ),
        selective_frontier_import_sha256=(
            value["selective_frontier_import_sha256"]
            if selective_mode
            else None
        ),
    )


@dataclass(frozen=True)
class _CanaryPlanting:
    canaries: tuple[Taint.LiveCanary, ...]
    escrow_path: Path
    receipt_path: Path
    receipt_sha256: str
    placement_descriptors_json: str
    placement_descriptors_sha256: str
    file_paths: tuple[Path, ...]
    environment_name: str


def _canary_control_root(campaign_root: Path) -> Path:
    return campaign_root / "operator_canary_control"


def _load_or_create_canary_planting(
    config: OperatorConfiguration,
) -> _CanaryPlanting:
    """Generate/reopen and observably plant the six host-only markers."""

    root = _canary_control_root(config.campaign_root)
    _ensure_private_directory(
        config.campaign_root, label="campaign root"
    )
    _ensure_private_directory(root, label="operator canary control root")
    escrow = root / "master_escrow.json"
    receipt_path = root / "placement_receipt.json"
    placement_sha256 = _json_sha256(dict(config.canary_placements))
    if escrow.exists():
        metadata = os.stat(escrow, follow_symlinks=False)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or metadata.st_uid != os.getuid()
            or stat.S_IMODE(metadata.st_mode) != 0o400
        ):
            raise ContiguousOrchestratorError(
                "operator canary escrow is not private and unaliased"
            )
        saved = _strict_json(
            _read_regular(escrow, maximum=64 * 1024),
            label="operator canary escrow",
        )
        if (
            set(saved)
            != {
                "schema",
                "kind",
                "placement_sha256",
                "canaries",
            }
            or saved["schema"] != 1
            or saved["kind"] != "contiguous_operator_canary_escrow"
            or saved["placement_sha256"] != placement_sha256
            or not isinstance(saved["canaries"], list)
        ):
            raise ContiguousOrchestratorError(
                "operator canary escrow differs from configuration"
            )
        try:
            canaries = tuple(
                Taint.LiveCanary(**item)
                for item in saved["canaries"]
            )
        except (TypeError, AttributeError) as exc:
            raise ContiguousOrchestratorError(
                "operator canary escrow is malformed"
            ) from exc
    else:
        canaries = tuple(
            Taint.LiveCanary(
                category=category,
                location_name=(
                    "environment:"
                    + config.canary_placements[category]
                    if category == "environment"
                    else config.canary_placements[category]
                ),
                value=secrets.token_hex(32),
            )
            for category in Taint.CONTROLLER_CANARY_CATEGORIES
        )
        canaries = Taint.validate_live_canaries(
            canaries, require_complete=True
        )
        _write_new(
            escrow,
            _canonical_json({
                "schema": 1,
                "kind": "contiguous_operator_canary_escrow",
                "placement_sha256": placement_sha256,
                "canaries": [asdict(item) for item in canaries],
            })
            + b"\n",
            mode=0o400,
        )
    canaries = Taint.validate_live_canaries(
        tuple(canaries), require_complete=True
    )
    by_category = {item.category: item for item in canaries}
    file_paths: list[Path] = []
    observations: list[dict[str, Any]] = []
    for category in Taint.CONTROLLER_CANARY_CATEGORIES:
        item = by_category[category]
        if category == "environment":
            name = config.canary_placements[category]
            prior = os.environ.get(name)
            if prior not in (None, item.value):
                raise ContiguousOrchestratorError(
                    "host environment canary name is already occupied"
                )
            os.environ[name] = item.value
            observations.append({
                **item.commitment(),
                "placement_kind": "host_environment",
                "observed": os.environ.get(name) == item.value,
            })
            continue
        path = Path(config.canary_placements[category])
        raw = item.value.encode("ascii")
        if path.exists() or path.is_symlink():
            if _read_regular(path, maximum=1024) != raw:
                raise ContiguousOrchestratorError(
                    f"{category} canary placement was substituted"
                )
        else:
            _write_new(path, raw, mode=0o400)
        metadata = os.stat(path, follow_symlinks=False)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or metadata.st_uid != os.getuid()
            or stat.S_IMODE(metadata.st_mode) != 0o400
        ):
            raise ContiguousOrchestratorError(
                f"{category} canary placement is not private"
            )
        file_paths.append(path)
        observations.append({
            **item.commitment(),
            "placement_kind": (
                "credential_decoy_file"
                if category == "auth_source"
                else "host_file"
            ),
            "device": metadata.st_dev,
            "inode": metadata.st_ino,
            "mode": stat.S_IMODE(metadata.st_mode),
            "owner_uid": metadata.st_uid,
            "owner_gid": metadata.st_gid,
            "size": metadata.st_size,
            "environment_owner_pid": None,
            "observed": True,
        })
    environment_observation = next(
        row
        for row in observations
        if row["category"] == "environment"
    )
    environment_observation.update({
        "device": 0,
        "inode": 0,
        "mode": 0,
        "owner_uid": os.getuid(),
        "owner_gid": os.getgid(),
        "size": len(by_category["environment"].value),
        "environment_owner_pid": None,
    })
    observations.sort(key=lambda row: row["category"])
    placement_descriptors_json = _canonical_json(
        observations
    ).decode("ascii")
    credential_raw = _read_regular(
        config.credential_source,
        maximum=MAX_JSON_BYTES,
    )
    credential_metadata = os.stat(
        config.credential_source, follow_symlinks=False
    )
    credential_source_identity = {
        "path": str(config.credential_source),
        "device": credential_metadata.st_dev,
        "inode": credential_metadata.st_ino,
        "mode": stat.S_IMODE(credential_metadata.st_mode),
        "owner_uid": credential_metadata.st_uid,
        "owner_gid": credential_metadata.st_gid,
        "size": credential_metadata.st_size,
        "sha256": hashlib.sha256(credential_raw).hexdigest(),
    }
    receipt = {
        "schema": 1,
        "kind": "contiguous_operator_canary_placement",
        "placement_sha256": placement_sha256,
        "escrow_path": str(escrow),
        "escrow_sha256": hashlib.sha256(
            _read_regular(escrow, maximum=64 * 1024)
        ).hexdigest(),
        "commitments": [
            item.commitment() for item in canaries
        ],
        "credential_source_identity": credential_source_identity,
        "placement_descriptors": observations,
        "placement_descriptors_sha256": hashlib.sha256(
            placement_descriptors_json.encode("ascii")
        ).hexdigest(),
        "observations": observations,
        "status": "PASS",
    }
    raw_receipt = _canonical_json(receipt) + b"\n"
    if receipt_path.exists():
        if _read_regular(
            receipt_path, maximum=256 * 1024
        ) != raw_receipt:
            raise ContiguousOrchestratorError(
                "operator canary placement receipt was substituted"
            )
    else:
        _write_new(receipt_path, raw_receipt, mode=0o400)
    return _CanaryPlanting(
        canaries=canaries,
        escrow_path=escrow,
        receipt_path=receipt_path,
        receipt_sha256=hashlib.sha256(raw_receipt).hexdigest(),
        placement_descriptors_json=placement_descriptors_json,
        placement_descriptors_sha256=receipt[
            "placement_descriptors_sha256"
        ],
        file_paths=tuple(file_paths),
        environment_name=config.canary_placements["environment"],
    )


def _operator_terminal_cleanup_intent_path(
    campaign_root: Path,
) -> Path:
    return (
        _canary_control_root(campaign_root)
        / "terminal_cleanup_intent.json"
    )


def _operator_terminal_cleanup_receipt_path(
    campaign_root: Path,
) -> Path:
    return (
        _canary_control_root(campaign_root)
        / "terminal_cleanup_receipt.json"
    )


def _file_identity(path: Path) -> dict[str, int]:
    metadata = os.stat(path, follow_symlinks=False)
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or metadata.st_uid != os.getuid()
    ):
        raise ContiguousOrchestratorError(
            f"terminal cleanup target is not exact/unaliased: {path}"
        )
    return {
        "device": metadata.st_dev,
        "inode": metadata.st_ino,
        "mode": stat.S_IMODE(metadata.st_mode),
        "owner_uid": metadata.st_uid,
        "owner_gid": metadata.st_gid,
        "size": metadata.st_size,
    }


def _terminal_cleanup_intent_value(
    *,
    campaign_root: Path,
    planting: _CanaryPlanting,
    terminal_receipt_path: Path,
    terminal_receipt_sha256: str,
    terminal_status: str,
) -> dict[str, Any]:
    by_location = {
        item.location_name: item for item in planting.canaries
    }
    files = []
    for path in planting.file_paths:
        item = by_location.get(str(path))
        raw = _read_regular(path, maximum=1024)
        if (
            item is None
            or raw != item.value.encode("ascii")
        ):
            raise ContiguousOrchestratorError(
                "terminal cleanup cannot bind a substituted canary"
            )
        files.append({
            "path": str(path),
            "category": item.category,
            "commitment": item.commitment(),
            "identity": _file_identity(path),
        })
    environment = next(
        item
        for item in planting.canaries
        if item.category == "environment"
    )
    if os.environ.get(planting.environment_name) != environment.value:
        raise ContiguousOrchestratorError(
            "terminal cleanup cannot bind a substituted environment canary"
        )
    escrow_raw = _read_regular(
        planting.escrow_path, maximum=64 * 1024
    )
    return {
        "schema": 1,
        "kind": "arc_agi3_operator_terminal_cleanup_intent",
        "campaign_root": str(Path(campaign_root).resolve()),
        "terminal_status": terminal_status,
        "terminal_receipt_path": str(terminal_receipt_path),
        "terminal_receipt_sha256": terminal_receipt_sha256,
        "canary_placement_receipt_path": str(
            planting.receipt_path
        ),
        "canary_placement_receipt_sha256":
            planting.receipt_sha256,
        "files": files,
        "environment": {
            "name": planting.environment_name,
            "category": environment.category,
            "commitment": environment.commitment(),
        },
        "escrow": {
            "path": str(planting.escrow_path),
            "sha256": _sha256(escrow_raw),
            "identity": _file_identity(planting.escrow_path),
        },
        "cleanup_order": [
            *[f"file:{index}" for index in range(len(files))],
            "environment",
            "escrow",
        ],
    }


def _validate_terminal_cleanup_intent(
    campaign_root: Path,
    value: object,
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {
        "schema",
        "kind",
        "campaign_root",
        "terminal_status",
        "terminal_receipt_path",
        "terminal_receipt_sha256",
        "canary_placement_receipt_path",
        "canary_placement_receipt_sha256",
        "files",
        "environment",
        "escrow",
        "cleanup_order",
    }:
        raise ContiguousOrchestratorError(
            "operator terminal cleanup intent schema is malformed"
        )
    terminal_path = Path(str(value["terminal_receipt_path"]))
    placement_path = Path(
        str(value["canary_placement_receipt_path"])
    )
    files = value["files"]
    environment = value["environment"]
    escrow = value["escrow"]
    if (
        value["schema"] != 1
        or value["kind"]
        != "arc_agi3_operator_terminal_cleanup_intent"
        or value["campaign_root"]
        != str(Path(campaign_root).resolve())
        or value["terminal_status"]
        not in {"PASS", "BLOCKED", "OPERATOR_INCIDENT"}
        or not isinstance(files, list)
        or not isinstance(environment, dict)
        or not isinstance(escrow, dict)
        or value["cleanup_order"]
        != [
            *[f"file:{index}" for index in range(len(files))],
            "environment",
            "escrow",
        ]
        or SHA256_RE.fullmatch(
            str(value["terminal_receipt_sha256"])
        ) is None
        or SHA256_RE.fullmatch(
            str(value["canary_placement_receipt_sha256"])
        ) is None
        or _sha256(
            _read_regular(terminal_path, maximum=MAX_JSON_BYTES)
        )
        != value["terminal_receipt_sha256"]
        or _sha256(
            _read_regular(placement_path, maximum=MAX_JSON_BYTES)
        )
        != value["canary_placement_receipt_sha256"]
    ):
        raise ContiguousOrchestratorError(
            "operator terminal cleanup intent is stale or substituted"
        )
    return value


def _commitment_matches(
    *,
    category: str,
    location_name: str,
    raw_value: str,
    commitment: object,
) -> bool:
    try:
        observed = Taint.LiveCanary(
            category=category,
            location_name=location_name,
            value=raw_value,
        ).commitment()
    except Exception:
        return False
    return observed == commitment


def _resume_terminal_canary_cleanup(
    campaign_root: Path,
    *,
    fault_hook: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Idempotently finish a durable terminal cleanup intent."""

    hook = fault_hook or (lambda _point: None)
    intent_path = _operator_terminal_cleanup_intent_path(
        campaign_root
    )
    intent = _validate_terminal_cleanup_intent(
        campaign_root,
        _strict_json(
            _read_regular(intent_path, maximum=MAX_JSON_BYTES),
            label="operator terminal cleanup intent",
        ),
    )
    for index, row in enumerate(intent["files"]):
        if not isinstance(row, dict) or set(row) != {
            "path", "category", "commitment", "identity"
        }:
            raise ContiguousOrchestratorError(
                "terminal cleanup file row is malformed"
            )
        path = Path(str(row["path"]))
        if path.exists() or path.is_symlink():
            raw = _read_regular(path, maximum=1024)
            try:
                value = raw.decode("ascii")
            except UnicodeError as exc:
                raise ContiguousOrchestratorError(
                    "terminal cleanup canary is not ASCII"
                ) from exc
            if (
                _file_identity(path) != row["identity"]
                or not _commitment_matches(
                    category=str(row["category"]),
                    location_name=str(path),
                    raw_value=value,
                    commitment=row["commitment"],
                )
            ):
                raise ContiguousOrchestratorError(
                    "refusing to unlink a substituted canary"
                )
            hook(f"before_unlink:file:{index}")
            path.unlink()
            _fsync_directory(path.parent)
            hook(f"after_unlink:file:{index}")
    environment = intent["environment"]
    name = str(environment["name"])
    current = os.environ.get(name)
    if current is not None:
        if not _commitment_matches(
            category=str(environment["category"]),
            location_name="environment:" + name,
            raw_value=current,
            commitment=environment["commitment"],
        ):
            raise ContiguousOrchestratorError(
                "refusing to remove a substituted environment canary"
            )
        hook("before_unset:environment")
        os.environ.pop(name)
        hook("after_unset:environment")
    escrow = intent["escrow"]
    escrow_path = Path(str(escrow["path"]))
    if escrow_path.exists() or escrow_path.is_symlink():
        escrow_raw = _read_regular(
            escrow_path, maximum=64 * 1024
        )
        if (
            _file_identity(escrow_path) != escrow["identity"]
            or _sha256(escrow_raw) != escrow["sha256"]
        ):
            raise ContiguousOrchestratorError(
                "refusing to unlink a substituted canary escrow"
            )
        hook("before_unlink:escrow")
        escrow_path.unlink()
        _fsync_directory(escrow_path.parent)
        hook("after_unlink:escrow")
    if any(
        Path(str(row["path"])).exists()
        or Path(str(row["path"])).is_symlink()
        for row in intent["files"]
    ) or os.environ.get(name) is not None or (
        escrow_path.exists() or escrow_path.is_symlink()
    ):
        raise ContiguousOrchestratorError(
            "terminal canary cleanup did not reach exact absence"
        )
    intent_sha256 = _sha256(
        _read_regular(intent_path, maximum=MAX_JSON_BYTES)
    )
    receipt = {
        "schema": 1,
        "kind": "arc_agi3_operator_terminal_cleanup",
        "status": "PASS",
        "terminal_status": intent["terminal_status"],
        "terminal_receipt_path": intent["terminal_receipt_path"],
        "terminal_receipt_sha256":
            intent["terminal_receipt_sha256"],
        "cleanup_intent_path": str(intent_path),
        "cleanup_intent_sha256": intent_sha256,
        "file_canaries_absent": True,
        "environment_canary_absent": True,
        "escrow_absent": True,
    }
    receipt_path = _operator_terminal_cleanup_receipt_path(
        campaign_root
    )
    _ensure_receipt(receipt_path, receipt)
    return {
        "terminal_receipt": intent["terminal_receipt_path"],
        "terminal_receipt_sha256":
            intent["terminal_receipt_sha256"],
        "terminal_cleanup_intent": str(intent_path),
        "terminal_cleanup_intent_sha256": intent_sha256,
        "terminal_cleanup_receipt": str(receipt_path),
        "terminal_cleanup_receipt_sha256": _sha256(
            _read_regular(receipt_path, maximum=MAX_JSON_BYTES)
        ),
        "canary_live_values_cleaned": True,
    }


def _finalize_operator_terminal(
    *,
    campaign_root: Path,
    planting: _CanaryPlanting,
    terminal_receipt_path: Path,
    terminal_value: Mapping[str, Any],
    fault_hook: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Durably publish terminal authority before any canary removal."""

    hook = fault_hook or (lambda _point: None)
    hook("before_terminal_receipt_durable")
    _ensure_receipt(terminal_receipt_path, dict(terminal_value))
    terminal_sha256 = _sha256(
        _read_regular(terminal_receipt_path, maximum=MAX_JSON_BYTES)
    )
    hook("after_terminal_receipt_durable")
    intent = _terminal_cleanup_intent_value(
        campaign_root=campaign_root,
        planting=planting,
        terminal_receipt_path=terminal_receipt_path,
        terminal_receipt_sha256=terminal_sha256,
        terminal_status=str(terminal_value["status"]),
    )
    intent_path = _operator_terminal_cleanup_intent_path(
        campaign_root
    )
    _ensure_receipt(intent_path, intent)
    cleanup = _resume_terminal_canary_cleanup(
        campaign_root, fault_hook=fault_hook
    )
    return {**dict(terminal_value), **cleanup}


def _ensure_receipt(path: Path, value: object) -> None:
    raw = _canonical_json(value) + b"\n"
    if path.exists():
        if _read_regular(path, maximum=MAX_JSON_BYTES) != raw:
            raise ContiguousOrchestratorError(
                f"retained receipt differs: {path}"
            )
        return
    _write_new(path, raw, mode=0o400)


def _terminal_promotion_replay_audit_value(
    *,
    campaign_root: Path,
    promotion_root: Path,
    runner_state_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    records, verified_boundaries = _read_only_promotion_records(
        Path(promotion_root).resolve(),
        campaign_id=str(runner_state_receipt["campaign_id"]),
        lane_boundaries=runner_state_receipt["lane_boundaries"],
    )
    if (
        runner_state_receipt.get("complete") is not True
        or verified_boundaries
        != runner_state_receipt.get("solved_levels")
        or verified_boundaries
        != runner_state_receipt.get("total_levels")
    ):
        raise ContiguousOrchestratorError(
            "pre-retention promotion/replay evidence is incomplete"
        )
    body = {
        "schema": 1,
        "kind": "arc_agi3_pre_retention_promotion_replay_audit",
        "status": "PASS",
        "campaign_root": str(Path(campaign_root).resolve()),
        "promotion_root": str(Path(promotion_root).resolve()),
        "campaign_id": runner_state_receipt["campaign_id"],
        "runner_state_receipt_sha256":
            runner_state_receipt["receipt_sha256"],
        "journal_head_sequence":
            runner_state_receipt["journal_head_sequence"],
        "journal_head_digest":
            runner_state_receipt["journal_head_digest"],
        "verified_promotion_boundaries": verified_boundaries,
        "total_levels": runner_state_receipt["total_levels"],
        "promotion_records": records,
        "promotion_records_sha256": _json_sha256(records),
    }
    return {
        **body,
        "receipt_sha256": _json_sha256(body),
    }


def _verify_terminal_promotion_replay_audit(
    path: Path,
    *,
    campaign_root: Path,
    promotion_root: Path,
    runner_state_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    expected = _terminal_promotion_replay_audit_value(
        campaign_root=campaign_root,
        promotion_root=promotion_root,
        runner_state_receipt=runner_state_receipt,
    )
    raw = _read_regular(path, maximum=MAX_JSON_BYTES)
    if raw != _canonical_json(expected) + b"\n":
        raise ContiguousOrchestratorError(
            "pre-retention promotion/replay audit is stale or forged"
        )
    if (
        stat.S_IMODE(path.stat(follow_symlinks=False).st_mode)
        & 0o222
    ):
        raise ContiguousOrchestratorError(
            "pre-retention promotion/replay audit remains writable"
        )
    return expected


def _load_or_create_terminal_promotion_replay_audit(
    *,
    campaign_root: Path,
    promotion_root: Path,
    audit_root: Path,
    runner_state_receipt: Mapping[str, Any],
) -> tuple[dict[str, Any], Path]:
    path = audit_root / TERMINAL_PROMOTION_REPLAY_AUDIT_NAME
    intent = campaign_root / Runner.TERMINAL_RETENTION_INTENT_NAME
    retention_started = intent.exists() or intent.is_symlink()
    if retention_started and (path.is_symlink() or not path.is_file()):
        raise ContiguousOrchestratorError(
            "partial retention lacks its promotion/replay audit"
        )
    expected = _terminal_promotion_replay_audit_value(
        campaign_root=campaign_root,
        promotion_root=promotion_root,
        runner_state_receipt=runner_state_receipt,
    )
    _ensure_receipt(path, expected)
    return (
        _verify_terminal_promotion_replay_audit(
            path,
            campaign_root=campaign_root,
            promotion_root=promotion_root,
            runner_state_receipt=runner_state_receipt,
        ),
        path,
    )


def _load_or_create_terminal_scheduler_audit(
    *,
    campaign_root: Path,
    audit_root: Path,
) -> tuple[dict[str, object], Path]:
    """Recover the exact pre-cleanup scheduler PASS after a partial purge."""

    scheduler_path = audit_root / "scheduler.json"
    retention_intent = (
        campaign_root / Runner.TERMINAL_RETENTION_INTENT_NAME
    )
    retention_receipt = (
        campaign_root / Runner.TERMINAL_RETENTION_RECEIPT_NAME
    )
    retention_evidence = (
        campaign_root / Runner.TERMINAL_RETENTION_EVIDENCE_NAME
    )
    retention_started = (
        retention_intent.exists() or retention_intent.is_symlink()
    )
    if not retention_started and any(
        path.exists() or path.is_symlink()
        for path in (retention_receipt, retention_evidence)
    ):
        raise ContiguousOrchestratorError(
            "terminal retention artifacts exist without their intent"
        )
    if retention_started:
        if retention_intent.is_symlink() or not retention_intent.is_file():
            raise ContiguousOrchestratorError(
                "terminal retention intent is aliased"
            )
        if scheduler_path.is_symlink() or not scheduler_path.is_file():
            raise ContiguousOrchestratorError(
                "partial retention lacks its pre-cleanup scheduler audit"
            )
        try:
            candidate = json.loads(
                _read_regular(
                    scheduler_path, maximum=MAX_JSON_BYTES
                )
            )
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise ContiguousOrchestratorError(
                "pre-cleanup scheduler audit is invalid JSON"
            ) from exc
        if not isinstance(candidate, dict):
            raise ContiguousOrchestratorError(
                "pre-cleanup scheduler audit is malformed"
            )
        digest = candidate.get("receipt_sha256")
        if (
            not isinstance(digest, str)
            or SHA256_RE.fullmatch(digest) is None
        ):
            raise ContiguousOrchestratorError(
                "pre-cleanup scheduler audit hash is malformed"
            )
        verified = Scheduler.verify_pre_retention_audit_receipt(
            campaign_root,
            scheduler_path,
            expected_receipt_sha256=digest,
        )
        return verified, scheduler_path

    receipt = Scheduler.audit_campaign(campaign_root)
    if receipt.get("verdict") != "PASS":
        raise ContiguousOrchestratorError(
            "terminal scheduler audit is not PASS"
        )
    _ensure_receipt(scheduler_path, receipt)
    verified = Scheduler.verify_audit_receipt(
        campaign_root, scheduler_path
    )
    return verified, scheduler_path


def _validate_host_child_ledger_audit(
    value: object,
    *,
    campaign_root: Path,
    require_quiescent: bool,
) -> dict[str, Any]:
    """Validate the backend-authenticated complete host-child inventory."""

    if not isinstance(value, Mapping):
        raise ContiguousOrchestratorError(
            "host child ledger audit is not an object"
        )
    selected = dict(value)
    counts = selected.get("status_counts")
    records = selected.get("records")
    startup_recovery = selected.get("startup_recovery")
    expected_fields = {
        "schema",
        "kind",
        "ledger_root",
        "authentication_key_sha256",
        "invocation_count",
        "status_counts",
        "startup_recovered_count",
        "startup_recovery",
        "records",
        "all_receipts_authenticated",
        "external_absence_proof_required_count",
        "all_children_accounted_for",
        "authentication_sha256",
    }
    if (
        set(selected) != expected_fields
        or selected.get("schema") != 1
        or selected.get("kind")
        != "arc_agi3_managed_host_child_ledger_audit"
        or selected.get("ledger_root")
        != str(
            Path(campaign_root).resolve()
            / "host_child_invocations"
        )
        or any(
            SHA256_RE.fullmatch(str(selected.get(name))) is None
            for name in (
                "authentication_key_sha256",
                "authentication_sha256",
            )
        )
        or not isinstance(counts, Mapping)
        or set(counts) != {
            "PENDING",
            "ACTIVE",
            "TERMINAL",
            "CLEAN",
        }
        or any(
            isinstance(item, bool)
            or not isinstance(item, int)
            or item < 0
            for item in counts.values()
        )
        or isinstance(selected.get("invocation_count"), bool)
        or not isinstance(selected.get("invocation_count"), int)
        or selected.get("invocation_count") != sum(counts.values())
        or not isinstance(records, list)
        or len(records) != selected.get("invocation_count")
        or isinstance(
            selected.get("startup_recovered_count"), bool
        )
        or not isinstance(
            selected.get("startup_recovered_count"), int
        )
        or not isinstance(startup_recovery, list)
        or len(startup_recovery)
        != selected.get("startup_recovered_count")
        or selected.get("all_receipts_authenticated") is not True
        or selected.get(
            "external_absence_proof_required_count"
        )
        != 0
        or selected.get("all_children_accounted_for") is not True
        or (
            require_quiescent
            and (
                counts.get("PENDING") != 0
                or counts.get("ACTIVE") != 0
            )
        )
    ):
        raise ContiguousOrchestratorError(
            "host child ledger audit is incomplete or nonquiescent"
        )
    return selected


def _selective_campaign_audit(
    *,
    config: OperatorConfiguration,
    credentials: Transport.ExternalChatGptCredentials,
    planting: _CanaryPlanting,
    command_runner: Any,
) -> dict[str, Any]:
    """Finalize the exact imported continuation without a global purge."""

    if (
        config.frontier_import_root is None
        or config.selective_continuation_game is None
        or config.selective_frontier_import_sha256 is None
    ):
        raise ContiguousOrchestratorError(
            "selective terminal audit lacks its import authority"
        )
    audit_root = config.campaign_root / "terminal_audits"
    _ensure_private_directory(audit_root, label="terminal audit root")
    host_child_audit = _validate_host_child_ledger_audit(
        command_runner.audit_invocation_ledger(),
        campaign_root=config.campaign_root,
        require_quiescent=True,
    )
    host_child_path = audit_root / "host_children.json"
    _ensure_receipt(host_child_path, host_child_audit)
    scheduler_receipt, scheduler_path = (
        _load_or_create_terminal_scheduler_audit(
            campaign_root=config.campaign_root,
            audit_root=audit_root,
        )
    )
    runner_receipt = Runner.audit_runner_state_read_only(
        config.campaign_root,
        secret_sentinels=credentials.leak_sentinels,
        controller_state_canaries=planting.canaries,
    )
    if (
        runner_receipt.get("status") != "PASS"
        or runner_receipt.get("campaign_mode")
        != "selective_continuation"
        or runner_receipt.get("selective_continuation_game")
        != config.selective_continuation_game
        or runner_receipt.get(
            "operator_authorized_selective_frontier_import_sha256"
        )
        != config.selective_frontier_import_sha256
        or runner_receipt.get("selective_complete") is not True
        or runner_receipt.get("complete") is not False
    ):
        raise ContiguousOrchestratorError(
            "selective terminal runner audit is not an exact PASS"
        )
    retention_receipt = Runner.audit_terminal_attempt_retention(
        config.campaign_root,
        runner_receipt,
        secret_sentinels=credentials.leak_sentinels,
        controller_state_canaries=planting.canaries,
    )
    if retention_receipt.get("status") != "NOT_REQUIRED":
        raise ContiguousOrchestratorError(
            "selective terminal attempted a global retention purge"
        )
    runner_path = audit_root / "runner.json"
    _ensure_receipt(runner_path, runner_receipt)
    unified = audit_selective_continuation_unified(
        campaign_root=config.campaign_root,
        scheduler_audit_receipt_path=scheduler_path,
        runner_state_receipt=runner_receipt,
        promotion_root=config.promotion_root,
        frontier_import_root=config.frontier_import_root,
        expected_selective_frontier_import_sha256=(
            config.selective_frontier_import_sha256
        ),
        secret_sentinels=credentials.leak_sentinels,
        controller_state_canaries=planting.canaries,
    )
    verify_selective_continuation_unified_audit(
        unified,
        campaign_root=config.campaign_root,
        scheduler_audit_receipt_path=scheduler_path,
        runner_state_receipt=runner_receipt,
        promotion_root=config.promotion_root,
        frontier_import_root=config.frontier_import_root,
        expected_selective_frontier_import_sha256=(
            config.selective_frontier_import_sha256
        ),
        secret_sentinels=credentials.leak_sentinels,
        controller_state_canaries=planting.canaries,
    )
    unified_path = audit_root / "unified.json"
    _ensure_receipt(unified_path, unified)
    return {
        "campaign_mode": "selective_continuation",
        "host_child_ledger_audit": str(host_child_path),
        "host_child_ledger_audit_sha256":
            _sha256(_canonical_json(host_child_audit) + b"\n"),
        "scheduler_audit": str(scheduler_path),
        "runner_audit": str(runner_path),
        "terminal_retention_status": "NOT_REQUIRED",
        "terminal_retention_receipt_sha256":
            retention_receipt["receipt_sha256"],
        "unified_audit": str(unified_path),
        "unified_audit_sha256": unified["receipt_sha256"],
        "selective_continuation_game":
            unified["selective_continuation_game"],
        "selective_frontier_import_sha256":
            unified["selective_frontier_import_sha256"],
        "operator_authorized_selective_frontier_import_sha256":
            unified[
                "operator_authorized_selective_frontier_import_sha256"
            ],
        "selective_scope_solved_levels":
            unified["selective_scope_solved_levels"],
        "selective_scope_total_levels":
            unified["selective_scope_total_levels"],
        "selective_complete": True,
        "complete": False,
        "solved_levels": unified["solved_levels"],
        "total_levels": unified["total_levels"],
    }


def _terminal_campaign_audit(
    *,
    config: OperatorConfiguration,
    credentials: Transport.ExternalChatGptCredentials,
    planting: _CanaryPlanting,
    command_runner: Any,
) -> dict[str, Any]:
    if config.selective_continuation_game is not None:
        return _selective_campaign_audit(
            config=config,
            credentials=credentials,
            planting=planting,
            command_runner=command_runner,
        )
    audit_root = config.campaign_root / "terminal_audits"
    _ensure_private_directory(audit_root, label="terminal audit root")
    host_child_audit = _validate_host_child_ledger_audit(
        command_runner.audit_invocation_ledger(),
        campaign_root=config.campaign_root,
        require_quiescent=True,
    )
    host_child_path = audit_root / "host_children.json"
    _ensure_receipt(host_child_path, host_child_audit)
    scheduler_receipt, scheduler_path = (
        _load_or_create_terminal_scheduler_audit(
            campaign_root=config.campaign_root,
            audit_root=audit_root,
        )
    )
    pre_retention_runner_receipt = Runner.audit_runner_state_read_only(
        config.campaign_root,
        secret_sentinels=credentials.leak_sentinels,
        controller_state_canaries=planting.canaries,
    )
    if (
        pre_retention_runner_receipt.get("status") != "PASS"
        or pre_retention_runner_receipt.get("complete") is not True
    ):
        raise ContiguousOrchestratorError(
            "terminal runner audit is not a complete PASS"
        )
    promotion_replay_receipt, promotion_replay_path = (
        _load_or_create_terminal_promotion_replay_audit(
            campaign_root=config.campaign_root,
            promotion_root=config.promotion_root,
            audit_root=audit_root,
            runner_state_receipt=pre_retention_runner_receipt,
        )
    )
    pre_cleanup_audits = {
        "promotion_replay":
            promotion_replay_receipt["receipt_sha256"],
        "scheduler": str(scheduler_receipt["receipt_sha256"]),
    }
    retention_receipt = Runner.finalize_terminal_attempt_retention(
        config.campaign_root,
        pre_retention_runner_receipt,
        secret_sentinels=credentials.leak_sentinels,
        controller_state_canaries=planting.canaries,
        pre_cleanup_audits=pre_cleanup_audits,
    )
    if retention_receipt.get("status") != "PASS":
        raise ContiguousOrchestratorError(
            "terminal attempt retention is not PASS"
        )
    runner_receipt = Runner.audit_runner_state_read_only(
        config.campaign_root,
        secret_sentinels=credentials.leak_sentinels,
        controller_state_canaries=planting.canaries,
    )
    if runner_receipt != pre_retention_runner_receipt:
        raise ContiguousOrchestratorError(
            "terminal retention changed journal-derived runner state"
        )
    Runner.audit_terminal_attempt_retention(
        config.campaign_root,
        runner_receipt,
        secret_sentinels=credentials.leak_sentinels,
        controller_state_canaries=planting.canaries,
        pre_cleanup_audits=pre_cleanup_audits,
    )
    runner_path = audit_root / "runner.json"
    _ensure_receipt(runner_path, runner_receipt)
    unified = audit_contiguous_campaign_unified(
        campaign_root=config.campaign_root,
        scheduler_audit_receipt_path=scheduler_path,
        runner_state_receipt=runner_receipt,
        promotion_root=config.promotion_root,
        secret_sentinels=credentials.leak_sentinels,
        controller_state_canaries=planting.canaries,
    )
    verify_contiguous_campaign_unified_audit(
        unified,
        campaign_root=config.campaign_root,
        scheduler_audit_receipt_path=scheduler_path,
        runner_state_receipt=runner_receipt,
        promotion_root=config.promotion_root,
        secret_sentinels=credentials.leak_sentinels,
        controller_state_canaries=planting.canaries,
    )
    unified_path = audit_root / "unified.json"
    _ensure_receipt(unified_path, unified)
    return {
        "host_child_ledger_audit": str(host_child_path),
        "host_child_ledger_audit_sha256":
            _sha256(_canonical_json(host_child_audit) + b"\n"),
        "scheduler_audit": str(scheduler_path),
        "promotion_replay_audit": str(promotion_replay_path),
        "promotion_replay_audit_sha256":
            promotion_replay_receipt["receipt_sha256"],
        "runner_audit": str(runner_path),
        "terminal_retention_receipt": str(
            config.campaign_root
            / Runner.TERMINAL_RETENTION_RECEIPT_NAME
        ),
        "terminal_retention_receipt_sha256":
            retention_receipt["receipt_sha256"],
        "unified_audit": str(unified_path),
        "unified_audit_sha256": unified["receipt_sha256"],
        "complete": unified["complete"],
        "solved_levels": unified["solved_levels"],
        "total_levels": unified["total_levels"],
    }


def _quiescent_authenticated_blocked_projection(
    state: Mapping[str, Any],
    *,
    journal_head_sequence: int,
    journal_head_digest: str,
) -> dict[str, Any] | None:
    """Return one machine-terminal BLOCKED projection or ``None``.

    Completed lanes are deliberately excluded from the unresolved predicate.
    A BLOCKED terminal is admissible only when no primary, auxiliary, or
    pending scheduler identity remains live.
    """

    lanes = state.get("lanes")
    assignments = state.get("auxiliary_assignments")
    if not isinstance(lanes, dict) or not isinstance(assignments, dict):
        raise ContiguousOrchestratorError(
            "runner state lacks typed lane/auxiliary inventories"
        )
    selective = state.get("campaign_mode") == "selective_continuation"
    selected_game = state.get("selective_continuation_game")
    selective_import = state.get("selective_frontier_import")
    if selective:
        if (
            not isinstance(selected_game, str)
            or selected_game not in lanes
            or not isinstance(selective_import, dict)
            or SHA256_RE.fullmatch(
                str(selective_import.get("import_sha256"))
            )
            is None
        ):
            raise ContiguousOrchestratorError(
                "selective runner state lacks its exact selected lane"
            )
        for game, lane in lanes.items():
            if not isinstance(lane, dict):
                raise ContiguousOrchestratorError(
                    "selective runner state has an untyped lane"
                )
            if game == selected_game:
                if (
                    lane.get("blocked")
                    == Runner.SELECTIVE_SCOPE_BLOCKED_REASON
                ):
                    raise ContiguousOrchestratorError(
                        "selected lane is marked outside selective scope"
                    )
            elif (
                lane.get("blocked")
                != Runner.SELECTIVE_SCOPE_BLOCKED_REASON
                or lane.get("active") is not None
            ):
                raise ContiguousOrchestratorError(
                    "nonselected lane lacks exact scope exclusion"
                )
    unresolved = [
        (game, lane)
        for game, lane in sorted(lanes.items())
        if (
            isinstance(lane, dict)
            and lane.get("reached") < lane.get("target")
            and (not selective or game == selected_game)
        )
    ]
    active_auxiliary = [
        assignment_id
        for assignment_id, row in assignments.items()
        if (
            isinstance(row, dict)
            and getattr(row.get("state"), "phase", None)
            in Scheduler.AUXILIARY_ACTIVE_PHASES
        )
    ]
    if (
        state.get("complete") is True
        or (selective and state.get("selective_complete") is True)
        or not unresolved
        or state.get("pending_scheduler_decision") is not None
        or state.get("pending_auxiliary_decision") is not None
        or active_auxiliary
        or any(
            lane.get("active") is not None
            or lane.get("blocked") is None
            for _game, lane in unresolved
        )
    ):
        return None
    result = {
        "schema": 1,
        "kind": "arc_agi3_contiguous_operator_terminal",
        "status": "BLOCKED",
        "authority": "host_authenticated_frontier_blockers",
        "solved_levels": state.get("solved_levels"),
        "total_levels": state.get("total_levels"),
        "journal_head_sequence": journal_head_sequence,
        "journal_head_digest": journal_head_digest,
        "unresolved_frontiers": [
            {
                "game": game,
                "reached": lane["reached"],
                "target": lane["target"],
                "blocker": lane["blocked"],
            }
            for game, lane in unresolved
        ],
        "active_primary_attempts": [],
        "active_auxiliary_assignments": [],
        "pending_scheduler_decision": False,
        "pending_auxiliary_decision": False,
    }
    if selective:
        result.update({
            "campaign_mode": "selective_continuation",
            "selective_continuation_game": selected_game,
            "selective_frontier_import_sha256": selective_import[
                "import_sha256"
            ],
            "selective_scope_solved_levels": state.get(
                "selective_scope_solved_levels"
            ),
            "selective_scope_total_levels": state.get(
                "selective_scope_total_levels"
            ),
            "scope_excluded_lane_count": len(lanes) - 1,
        })
    return result


def _operator_journal_head(
    campaign_root: Path,
) -> dict[str, object]:
    journal_path = campaign_root / "attempt_journal"
    try:
        events = Scheduler.read_journal(campaign_root)
    except Exception:
        return {
            "attempt_journal_path": str(journal_path),
            "journal_status": "UNAVAILABLE",
            "journal_event_count": None,
            "journal_head_sequence": None,
            "journal_head_digest": None,
        }
    if not events:
        return {
            "attempt_journal_path": str(journal_path),
            "journal_status": "EMPTY",
            "journal_event_count": 0,
            "journal_head_sequence": None,
            "journal_head_digest": None,
        }
    return {
        "attempt_journal_path": str(journal_path),
        "journal_status": "AUTHENTICATED",
        "journal_event_count": len(events),
        "journal_head_sequence": events[-1]["sequence"],
        "journal_head_digest": events[-1]["digest"],
    }


def _post_incident_meta_projection(
    state: Mapping[str, Any],
    journal_events: Sequence[Mapping[str, Any]],
) -> dict[str, object]:
    """Reduce a latched substrate incident to the only meta-proposer input."""

    operator = state.get("operator_incident")
    substrate = state.get("substrate_incident")
    operator_fields = (
        "attempt_id",
        "operation",
        "fault_domain",
        "operation_consecutive",
        "domain_consecutive",
        "threshold",
        "reason_code",
    )
    if (
        not isinstance(operator, Mapping)
        or not isinstance(substrate, Mapping)
        or not journal_events
        or operator.get("operation") != "substrate_health_reprobe"
        or operator.get("fault_domain") != "controller_substrate"
        or operator.get("attempt_id") != substrate.get("attempt_id")
    ):
        raise ContiguousOrchestratorError(
            "post-incident meta diagnosis requires one durable "
            "controller-substrate incident"
        )
    attempted_epochs = substrate.get(
        "attempted_remediation_epochs"
    )
    if not isinstance(attempted_epochs, list):
        raise ContiguousOrchestratorError(
            "substrate incident lacks remediation history"
        )
    last_probe = substrate.get("last_health_probe")
    if last_probe is not None and not isinstance(last_probe, Mapping):
        raise ContiguousOrchestratorError(
            "substrate incident health probe is malformed"
        )
    incident_events = [
        event
        for event in journal_events
        if (
            event.get("kind") == "OPERATOR_INCIDENT"
            and event.get("payload") == operator
        )
    ]
    if len(incident_events) != 1:
        raise ContiguousOrchestratorError(
            "substrate incident lacks one exact authenticated event"
        )
    incident_event = incident_events[0]
    projection: dict[str, object] = {
        "schema": Supervisor.POST_INCIDENT_META_SCHEMA,
        "kind":
            "arc_agi3_contiguous_substrate_incident_projection",
        "operator_incident": {
            name: operator.get(name) for name in operator_fields
        },
        "substrate_incident": {
            "attempt_id": substrate.get("attempt_id"),
            "substrate_identity_sha256":
                substrate.get("substrate_identity_sha256"),
            "failure_receipt_sha256":
                substrate.get("failure_receipt_sha256"),
            "failure_class": substrate.get("failure_class"),
            "failure_code": substrate.get("failure_code"),
            "health_probe_count":
                substrate.get("health_probe_count"),
            "attempted_remediation_epochs_sha256":
                _json_sha256(attempted_epochs),
            "last_health_probe_sha256": (
                None
                if last_probe is None
                else _json_sha256(dict(last_probe))
            ),
        },
        "incident_event_sequence":
            incident_event.get("sequence"),
        "incident_event_digest":
            incident_event.get("digest"),
    }
    try:
        return Supervisor._validate_post_incident_meta_projection(
            projection
        )
    except Supervisor.SupervisorContractError as exc:
        raise ContiguousOrchestratorError(
            "substrate incident cannot enter the bounded meta diagnosis"
        ) from exc


def _post_incident_meta_protected_snapshot(
    *,
    campaign_root: Path,
    promotion_root: Path,
) -> str:
    """Hash every non-diagnostic campaign/promotion entry before advice."""

    rows: list[dict[str, object]] = []
    excluded_campaign_roots = {
        Supervisor.POST_INCIDENT_META_ROOT_NAME,
        Supervisor.OPERATOR_LEASE_ROOT_NAME,
    }
    for root_label, root in (
        ("campaign", Path(campaign_root)),
        ("promotion", Path(promotion_root)),
    ):
        if not root.exists():
            rows.append({
                "root": root_label,
                "relative": ".",
                "kind": "missing",
            })
            continue
        if root.is_symlink() or not root.is_dir():
            raise ContiguousOrchestratorError(
                "meta-protected root is not a regular directory"
            )
        for selected in sorted(
            root.rglob("*"),
            key=lambda item: item.relative_to(root).as_posix(),
        ):
            relative = selected.relative_to(root)
            if (
                root_label == "campaign"
                and relative.parts
                and relative.parts[0] in excluded_campaign_roots
            ):
                continue
            metadata = selected.lstat()
            if stat.S_ISLNK(metadata.st_mode):
                raise ContiguousOrchestratorError(
                    "meta-protected tree contains a symlink"
                )
            row: dict[str, object] = {
                "root": root_label,
                "relative": relative.as_posix(),
                "mode": stat.S_IMODE(metadata.st_mode),
            }
            if stat.S_ISDIR(metadata.st_mode):
                row["kind"] = "directory"
            elif (
                stat.S_ISREG(metadata.st_mode)
                and metadata.st_nlink == 1
            ):
                row.update({
                    "kind": "file",
                    "bytes": metadata.st_size,
                    "sha256": Supervisor._sha256_file(selected),
                })
            else:
                raise ContiguousOrchestratorError(
                    "meta-protected tree contains an aliased or special "
                    "entry"
                )
            rows.append(row)
    return _json_sha256(rows)


def _run_latched_substrate_meta_diagnostic(
    config: OperatorConfiguration,
    *,
    runner: Any,
    state: Mapping[str, Any],
    journal_events: Sequence[Mapping[str, Any]],
    command_runner: Any,
    operator_lease: Supervisor.OperatorLease,
) -> dict[str, Any]:
    """Execute the sole sealed meta handoff without opening runner authority."""

    projection = _post_incident_meta_projection(
        state, journal_events
    )
    _verify_auxiliary_backend_configuration(
        config.auxiliary_backend_configuration,
        config.auxiliary_launch_configuration,
    )
    protected_before = _post_incident_meta_protected_snapshot(
        campaign_root=config.campaign_root,
        promotion_root=config.promotion_root,
    )
    state_before = copy.deepcopy(state)
    journal_before = copy.deepcopy(journal_events)
    result = dict(
        Supervisor.PostIncidentMetaDiagnostic(
            config.campaign_root,
            operator_configuration_sha256=config.config_sha256,
            driver_executable=(
                config.auxiliary_backend_configuration.driver_executable
            ),
            driver_executable_sha256=(
                config.auxiliary_backend_configuration
                .driver_executable_sha256
            ),
            driver_configuration=(
                config.auxiliary_backend_configuration
                .driver_configuration
            ),
            driver_configuration_sha256=(
                config.auxiliary_backend_configuration
                .driver_configuration_sha256
            ),
            driver_attestation_sha256=(
                config.auxiliary_backend_configuration
                .backend_attestation_sha256
            ),
            operation_timeout_seconds=(
                config.auxiliary_backend_configuration
                .operation_timeout_seconds
            ),
            command_runner=command_runner,
        ).run_once(projection)
    )
    operator_lease.assert_healthy()
    if (
        runner.state() != state_before
        or runner.journal.read() != journal_before
        or _post_incident_meta_protected_snapshot(
            campaign_root=config.campaign_root,
            promotion_root=config.promotion_root,
        )
        != protected_before
    ):
        raise ContiguousOrchestratorError(
            "post-incident meta diagnostic mutated campaign authority"
        )
    return result


def _apply_latched_substrate_meta_recovery(
    *,
    runner: Any,
    latched_state: Mapping[str, Any],
    meta_diagnostic: Mapping[str, Any],
    operator_lease: Supervisor.OperatorLease,
) -> tuple[bool, dict[str, Any]]:
    """Mechanically apply the sole recommendation and prove its authority.

    The meta-proposer never resumes the campaign.  This trusted controller
    bridge accepts only the exact authenticated hashes exposed by the sealed
    diagnostic, invokes the runner's single-use recovery API, and then proves
    that no solver, WIP, cost, attempt, lane, or promotion state changed.  A
    fresh real substrate PASS is the only state that may clear both latches.
    """

    recommendation = meta_diagnostic.get(
        "recommended_operator_action"
    )
    if (
        meta_diagnostic.get("status") != "DIAGNOSED"
        or recommendation
        != Runner.META_SUBSTRATE_RECOVERY_RECOMMENDATION
    ):
        return False, dict(latched_state)
    hashes = {
        "meta_request_sha256":
            meta_diagnostic.get("request_sha256"),
        "meta_response_sha256":
            meta_diagnostic.get("response_sha256"),
        "meta_terminal_sha256":
            meta_diagnostic.get("receipt_sha256"),
    }
    if any(
        not isinstance(value, str)
        or re.fullmatch(r"[0-9a-f]{64}", value) is None
        for value in hashes.values()
    ):
        raise ContiguousOrchestratorError(
            "post-incident meta recovery lacks exact sealed hashes"
        )
    operator_lease.assert_healthy()
    recovered = runner.apply_meta_substrate_recovery(
        **hashes,
        recommendation=str(recommendation),
    )
    operator_lease.assert_healthy()
    if (
        not isinstance(recovered, Mapping)
        or dict(recovered) != runner.state()
    ):
        raise ContiguousOrchestratorError(
            "post-incident meta recovery did not return exact runner state"
        )
    before = copy.deepcopy(dict(latched_state))
    after = copy.deepcopy(dict(recovered))
    allowed_recovery_fields = {
        "operator_incident",
        "substrate_incident",
        "failure_operation_circuits",
        "failure_domain_circuits",
    }
    for field in allowed_recovery_fields:
        before.pop(field, None)
        after.pop(field, None)
    if before != after:
        raise ContiguousOrchestratorError(
            "post-incident meta recovery mutated campaign authority"
        )
    operator_incident = recovered.get("operator_incident")
    substrate_incident = recovered.get("substrate_incident")
    resumed = (
        operator_incident is None
        and substrate_incident is None
    )
    if resumed:
        events = runner.journal.read()
        if (
            len(events) < 3
            or [
                event.get("kind") for event in events[-3:]
            ]
            != [
                "META_SUBSTRATE_RECOVERY_AUTHORIZED",
                "META_SUBSTRATE_HEALTH_RESTORED",
                "META_SUBSTRATE_RESUME_AUTHORIZED",
            ]
        ):
            raise ContiguousOrchestratorError(
                "post-incident meta resume lacks its exact journal chain"
            )
        return True, dict(recovered)
    meta_recovery = (
        substrate_incident.get("meta_recovery")
        if isinstance(substrate_incident, Mapping)
        else None
    )
    if (
        not isinstance(operator_incident, Mapping)
        or not isinstance(substrate_incident, Mapping)
        or not isinstance(meta_recovery, Mapping)
        or meta_recovery.get("phase") != "FAILED"
    ):
        raise ContiguousOrchestratorError(
            "post-incident meta recovery ended in an untyped state"
        )
    events = runner.journal.read()
    if (
        len(events) < 2
        or [
            event.get("kind") for event in events[-2:]
        ]
        != [
            "META_SUBSTRATE_RECOVERY_AUTHORIZED",
            "META_SUBSTRATE_RECOVERY_FAILED",
        ]
    ):
        raise ContiguousOrchestratorError(
            "failed post-incident meta recovery lacks its journal chain"
        )
    return False, dict(recovered)


def _operator_incident_value(
    config: OperatorConfiguration,
    *,
    reason_code: str,
    error_class: str | None,
    runner_incident: Mapping[str, Any] | None = None,
    solved_levels: int | None = None,
    total_levels: int | None = None,
    meta_diagnostic: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if (
        re.fullmatch(r"[a-z][a-z0-9_]{0,127}", reason_code)
        is None
    ):
        raise ContiguousOrchestratorError(
            "operator incident reason code is malformed"
        )
    normalized_error_class = error_class
    if (
        normalized_error_class is not None
        and re.fullmatch(
            r"[A-Za-z_][A-Za-z0-9_]{0,127}",
            normalized_error_class,
        )
        is None
    ):
        normalized_error_class = "OperatorError"
    body = {
        "schema": 1,
        "kind": "arc_agi3_contiguous_operator_incident",
        "status": "OPERATOR_INCIDENT",
        "campaign_phase": "PAUSED",
        "paused": True,
        "human_intervention_required": True,
        "reason_code": reason_code,
        "error_class": normalized_error_class,
        "campaign_root": str(config.campaign_root),
        "operator_config_sha256": config.config_sha256,
        "runner_incident": (
            dict(runner_incident)
            if runner_incident is not None
            else None
        ),
        "solved_levels": solved_levels,
        "total_levels": total_levels,
        "meta_diagnostic": (
            None
            if meta_diagnostic is None
            else dict(meta_diagnostic)
        ),
        **_operator_journal_head(config.campaign_root),
    }
    return {
        **body,
        "receipt_sha256": _json_sha256(body),
    }


def _storage_exhausted_terminal_value(
    config: OperatorConfiguration,
    *,
    state: Mapping[str, Any],
    journal_events: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    incident = state.get("storage_incident")
    if (
        not isinstance(incident, Mapping)
        or not journal_events
        or journal_events[-1].get("kind")
        != "JOURNAL_OR_STORAGE_EXHAUSTED"
        or journal_events[-1].get("payload") != incident
        or incident.get("reason_code")
        != "journal_or_storage_exhausted"
        or incident.get("status") != "OPERATOR_INCIDENT"
        or any(
            incident.get(name) is not False
            for name in (
                "solver_authority",
                "wip_authority",
                "cost_authority",
                "promotion_authority",
            )
        )
    ):
        raise ContiguousOrchestratorError(
            "storage terminal lacks its exact authenticated incident"
        )
    body = {
        "schema": 1,
        "kind": "arc_agi3_contiguous_operator_terminal",
        "status": "JOURNAL_OR_STORAGE_EXHAUSTED",
        "terminal_condition": "journal_or_storage_exhausted",
        "campaign_phase": "STOPPED",
        "reason_code": "journal_or_storage_exhausted",
        "campaign_root": str(config.campaign_root),
        "operator_config_sha256": config.config_sha256,
        "storage_incident": dict(incident),
        "solved_levels": state.get("solved_levels"),
        "total_levels": state.get("total_levels"),
        "journal_head_sequence":
            journal_events[-1].get("sequence"),
        "journal_head_digest":
            journal_events[-1].get("digest"),
        "active_primary_attempts": [],
        "active_auxiliary_assignments": [],
        "pending_scheduler_decision": False,
        "pending_auxiliary_decision": False,
        "terminal_cleanup_intent": str(
            _operator_terminal_cleanup_intent_path(
                config.campaign_root
            )
        ),
        "canary_cleanup_required": True,
    }
    return {
        **body,
        "receipt_sha256": _json_sha256(body),
    }


def _selective_operator_preflight(
    config: OperatorConfiguration,
) -> dict[str, Any]:
    """Bind fresh controls and the exact current imported frontier."""

    if (
        config.frontier_import_root is None
        or config.selective_continuation_game is None
        or config.selective_frontier_import_sha256 is None
    ):
        raise ContiguousOrchestratorError(
            "selective operator lacks its exact import configuration"
        )
    first, frontier_authority_source = (
        _selective_preflight_frontier(config)
    )
    _require_operator_authorized_selective_frontier(config, first)
    control = Supervisor.selective_continuation_preflight(
        config.launch_attestation,
        requested_image_digest=(
            config.backend_configuration.image_digest
        ),
        conformance_result=config.conformance_result,
        python_executable=config.python_executable,
        python_executable_sha256=config.python_executable_sha256,
        python_runtime_manifest=config.python_runtime_manifest,
        python_runtime_manifest_sha256=(
            config.python_runtime_manifest_sha256
        ),
        runtime_control_snapshot_root=(
            config.runtime_control_snapshot_root
        ),
        pilot_gate_receipt=config.pilot_gate_receipt,
        pilot_authentication_key=config.pilot_authentication_key,
        pilot_production_stack_attestation_sha256=(
            config.pilot_production_stack_attestation_sha256
        ),
    )
    control_evidence = control.get("launch_authority_evidence")
    if (
        control.get("status") != "PASS"
        or control.get("launch_authority")
        != "SELECTIVE_CONTROL_RECEIPT_DERIVED"
        or control.get("launch_authority_kind")
        != "arc_agi3_selective_continuation_control_authority"
        or not isinstance(control_evidence, Mapping)
        or control_evidence.get("authority_sha256")
        != control.get("launch_authority_sha256")
        or control_evidence.get("terminal_release_authority") is not False
        or control.get("image_digest")
        != config.backend_configuration.image_digest
    ):
        raise ContiguousOrchestratorError(
            "selective control preflight did not issue exact authority"
        )
    second, reopened_frontier_authority_source = (
        _selective_preflight_frontier(config)
    )
    if (
        second != first
        or reopened_frontier_authority_source
        != frontier_authority_source
    ):
        raise ContiguousOrchestratorError(
            "current frontier changed during selective launch preflight"
        )
    frontier = Runner.selective_frontier_import_to_dict(first)
    body = {
        "schema": 1,
        "kind": "arc_agi3_selective_frontier_launch_authority",
        "status": "PASS",
        "authority_source":
            "verified_controls_pilot_and_authenticated_frontier",
        "frontier_authority_source": frontier_authority_source,
        "operator_configuration_sha256": config.config_sha256,
        "frontier_import_root": str(config.frontier_import_root),
        "selective_continuation_game":
            config.selective_continuation_game,
        "selective_frontier_import": frontier,
        "selective_frontier_import_sha256": first.import_sha256,
        "operator_authorized_selective_frontier_import_sha256":
            config.selective_frontier_import_sha256,
        "control_launch_authority_sha256": control[
            "launch_authority_sha256"
        ],
        "control_launch_authority_kind": control[
            "launch_authority_kind"
        ],
        "control_launch_authority_evidence": control[
            "launch_authority_evidence"
        ],
        "control_contract_sha256": control[
            "control_contract_sha256"
        ],
        "python_runtime_manifest_sha256":
            config.python_runtime_manifest_sha256,
        "image_digest": config.backend_configuration.image_digest,
        "pilot_gate_receipt_sha256": control[
            "pilot_gate_receipt_sha256"
        ],
        "pilot_manifest_sha256": control[
            "pilot_manifest_sha256"
        ],
    }
    authority_sha256 = _json_sha256(body)
    return {
        **control,
        "launch_authority": "SELECTIVE_FRONTIER_RECEIPT_DERIVED",
        "launch_authority_kind":
            "arc_agi3_selective_frontier_launch_authority",
        "launch_authority_sha256": authority_sha256,
        "launch_authority_evidence": {
            **body,
            "authority_sha256": authority_sha256,
        },
        "selective_continuation_game":
            config.selective_continuation_game,
        "selective_frontier_import": frontier,
        "selective_frontier_import_sha256": first.import_sha256,
        "operator_authorized_selective_frontier_import_sha256":
            config.selective_frontier_import_sha256,
    }


def _selective_durable_preflight_projection(
    config: OperatorConfiguration,
    preflight: Mapping[str, Any],
) -> dict[str, Any]:
    """Project restart-stable authority, excluding fresh-run timestamps."""

    outer = preflight.get("launch_authority_evidence")
    if not isinstance(outer, Mapping):
        raise ContiguousOrchestratorError(
            "selective preflight lacks outer authority evidence"
        )
    control = outer.get("control_launch_authority_evidence")
    if not isinstance(control, Mapping):
        raise ContiguousOrchestratorError(
            "selective preflight lacks control authority evidence"
        )
    if (
        preflight.get("selective_frontier_import_sha256")
        != config.selective_frontier_import_sha256
        or outer.get(
            "operator_authorized_selective_frontier_import_sha256"
        )
        != config.selective_frontier_import_sha256
    ):
        raise ContiguousOrchestratorError(
            "selective preflight differs from operator-authorized import"
        )
    body = {
        "schema": 1,
        "kind": "arc_agi3_selective_durable_launch_contract",
        "operator_configuration_sha256": config.config_sha256,
        "frontier_import_root": str(config.frontier_import_root),
        "selective_continuation_game":
            config.selective_continuation_game,
        "selective_frontier_import": preflight[
            "selective_frontier_import"
        ],
        "selective_frontier_import_sha256": preflight[
            "selective_frontier_import_sha256"
        ],
        "operator_authorized_selective_frontier_import_sha256":
            config.selective_frontier_import_sha256,
        "conformance_registry_sha256": preflight[
            "conformance_registry_sha256"
        ],
        "control_contract_sha256": preflight[
            "control_contract_sha256"
        ],
        "supplied_prelaunch_sha256": control[
            "supplied_prelaunch_sha256"
        ],
        "authoritative_inventory_sha256": preflight[
            "authoritative_inventory_sha256"
        ],
        "image_digest": preflight["image_digest"],
        "python_runtime_manifest_sha256": preflight[
            "python_runtime_manifest_sha256"
        ],
        "pilot_gate_receipt_sha256": preflight[
            "pilot_gate_receipt_sha256"
        ],
        "pilot_manifest_sha256": preflight[
            "pilot_manifest_sha256"
        ],
        "pilot_meta_handoff_count": preflight[
            "pilot_meta_handoff_count"
        ],
        "production_stack_attestation_sha256": control[
            "production_stack_attestation_sha256"
        ],
    }
    return {**body, "contract_sha256": _json_sha256(body)}


def _run_operator_impl(
    config: OperatorConfiguration,
) -> dict[str, Any]:
    """Run recovery/cycles until the exact declared audited terminal state."""

    selective_game = getattr(
        config, "selective_continuation_game", None
    )
    frontier_import_root = getattr(
        config, "frontier_import_root", None
    )
    selective_frontier_import_sha256 = getattr(
        config, "selective_frontier_import_sha256", None
    )
    selective_mode = (
        selective_game is not None
        or frontier_import_root is not None
        or selective_frontier_import_sha256 is not None
    )
    expected_terminal = (
        SELECTIVE_TERMINAL_CONDITION
        if selective_mode
        else CANONICAL_TERMINAL_CONDITION
    )
    if (
        getattr(config, "terminal_condition", None)
        != expected_terminal
        or selective_mode
        and (
            selective_game is None
            or frontier_import_root is None
            or selective_frontier_import_sha256 is None
        )
    ):
        raise ContiguousOrchestratorError(
            "operator terminal condition is fixed by campaign policy"
        )
    if isinstance(config, OperatorConfiguration):
        _revalidate_operator_path_relationships(config)

    # This is the last read-only boundary.  A failure here cannot create the
    # campaign root, plant canaries, or start any external process.
    if selective_mode:
        preflight = _selective_operator_preflight(config)
    else:
        preflight = Supervisor.launch_preflight(
            config.launch_attestation,
            requested_image_digest=(
                config.backend_configuration.image_digest
            ),
            conformance_result=config.conformance_result,
            canonical_root=config.canonical_root,
            environments_root=config.environments_root,
            python_executable=config.python_executable,
            python_executable_sha256=config.python_executable_sha256,
            python_runtime_manifest=config.python_runtime_manifest,
            python_runtime_manifest_sha256=(
                config.python_runtime_manifest_sha256
            ),
            runtime_control_snapshot_root=(
                config.runtime_control_snapshot_root
            ),
            pilot_gate_receipt=config.pilot_gate_receipt,
            pilot_authentication_key=(
                config.pilot_authentication_key
            ),
            pilot_production_stack_attestation_sha256=(
                config.pilot_production_stack_attestation_sha256
            ),
        )
    # Reopen the sidecar controls after the global release preflight and
    # before the first campaign mutation.  The formal operator has no
    # per-dispatch selector: this one manifest remains fixed for genesis.
    auxiliary_attestation = _verify_auxiliary_backend_configuration(
        config.auxiliary_backend_configuration,
        config.auxiliary_launch_configuration,
    )
    _ensure_private_directory(
        config.campaign_root, label="campaign root"
    )
    operator_lease = Supervisor.OperatorLease(
        config.campaign_root,
        operator_configuration_sha256=config.config_sha256,
    ).acquire()
    try:
        try:
            return _run_operator_owned_impl(
                config,
                preflight=preflight,
                auxiliary_attestation=auxiliary_attestation,
                operator_lease=operator_lease,
            )
        except Exception as exc:
            return _handle_operator_exception(config, exc)
    finally:
        operator_lease.release()


def _production_command_runner(
    config: OperatorConfiguration,
) -> Any:
    """Construct the sole host child runner on its campaign-bound ledger."""

    import arc_agi3_container_backend as Container

    return Container.SubprocessCommandRunner(
        docker_socket=config.docker_socket,
        docker_config=config.docker_config_root,
        invocation_ledger_root=(
            config.campaign_root / "host_child_invocations"
        ),
    )


def _runner_cleanup_pending(state: Mapping[str, Any]) -> bool:
    primary_phases = {
        "RUNNING",
        "DRAINING",
        "EXITED",
        "COLLECTED",
        "COLLECTION_REJECTED",
        "TORN_DOWN",
    }
    return any(
        attempt.get("phase") in primary_phases
        for attempt in state["attempts"].values()
    ) or any(
        row["state"].phase in Scheduler.AUXILIARY_ACTIVE_PHASES
        for row in state["auxiliary_assignments"].values()
    )


def _run_operator_owned_impl(
    config: OperatorConfiguration,
    *,
    preflight: Mapping[str, Any],
    auxiliary_attestation: Mapping[str, Any],
    operator_lease: Supervisor.OperatorLease,
) -> dict[str, Any]:
    """Execute every mutable/recovery phase under one live host lease."""

    operator_lease.assert_healthy()
    import arc_agi3_container_backend as Container

    selective_durable_preflight = (
        None
        if config.selective_continuation_game is None
        else _selective_durable_preflight_projection(
            config, preflight
        )
    )
    operator_genesis = {
        "schema": 1,
        "kind": "arc_agi3_contiguous_operator_genesis",
        "operator_config_path": str(config.config_path),
        "operator_config_sha256": config.config_sha256,
        "operator_path_relationships_sha256":
            config.path_relationships_sha256,
        "python_executable": str(config.python_executable),
        "python_executable_sha256":
            config.python_executable_sha256,
        "python_runtime_manifest":
            str(config.python_runtime_manifest),
        "python_runtime_manifest_sha256":
            config.python_runtime_manifest_sha256,
        "runtime_control_snapshot_root":
            str(config.runtime_control_snapshot_root),
        "runtime_control_snapshot_sha256":
            preflight["control_contract_sha256"],
        "container_image_digest":
            config.backend_configuration.image_digest,
        "auxiliary_driver_protocol_sha256":
            AUXILIARY_DRIVER_PROTOCOL_SHA256,
        "post_incident_meta_protocol_sha256":
            Supervisor.POST_INCIDENT_META_PROTOCOL_SHA256,
        "auxiliary_backend_attestation_sha256":
            config.auxiliary_backend_configuration
            .backend_attestation_sha256,
        "auxiliary_backend_contract_sha256":
            auxiliary_attestation["backend_contract_sha256"],
        "auxiliary_input_bundle_contract_sha256":
            auxiliary_attestation[
                "input_bundle_contract_sha256"
            ],
        "auxiliary_admission_contract_sha256":
            auxiliary_attestation["admission_contract_sha256"],
        "conformance_result": str(config.conformance_result),
        "conformance_registry_sha256":
            preflight["conformance_registry_sha256"],
        "preflight_sha256": (
            _json_sha256(preflight)
            if selective_durable_preflight is None
            else selective_durable_preflight["contract_sha256"]
        ),
        "pilot_gate_receipt":
            preflight["pilot_gate_receipt"],
        "pilot_gate_receipt_sha256":
            preflight["pilot_gate_receipt_sha256"],
        "pilot_manifest_sha256":
            preflight["pilot_manifest_sha256"],
        "pilot_meta_handoff_count":
            preflight["pilot_meta_handoff_count"],
    }
    if config.selective_continuation_game is not None:
        operator_genesis.update({
            "campaign_mode": "selective_continuation",
            "terminal_condition": SELECTIVE_TERMINAL_CONDITION,
            "frontier_import_root": str(
                config.frontier_import_root
            ),
            "selective_continuation_game":
                config.selective_continuation_game,
            "selective_frontier_import_sha256": preflight[
                "selective_frontier_import_sha256"
            ],
            "operator_authorized_selective_frontier_import_sha256":
                config.selective_frontier_import_sha256,
            "selective_launch_authority_sha256":
                selective_durable_preflight["contract_sha256"],
            "selective_durable_launch_contract":
                selective_durable_preflight,
        })
    _ensure_receipt(
        config.campaign_root / "operator_genesis.json",
        operator_genesis,
    )
    cleanup_intent_path = _operator_terminal_cleanup_intent_path(
        config.campaign_root
    )
    terminal_paths = (
        config.campaign_root / "operator_incident.json",
        config.campaign_root / "operator_storage_exhausted.json",
        config.campaign_root / "operator_terminal_blocked.json",
        config.campaign_root / "terminal_audits" / "operator.json",
    )
    existing_terminal_paths = [
        path
        for path in terminal_paths
        if path.exists() or path.is_symlink()
    ]
    if cleanup_intent_path.exists() or cleanup_intent_path.is_symlink():
        if len(existing_terminal_paths) != 1:
            raise ContiguousOrchestratorError(
                "terminal cleanup intent lacks one exact terminal receipt"
            )
        terminal_value = _strict_json(
            _read_regular(
                existing_terminal_paths[0], maximum=MAX_JSON_BYTES
            ),
            label="operator terminal receipt",
        )
        return {
            **terminal_value,
            **_resume_terminal_canary_cleanup(
                config.campaign_root
            ),
        }
    if len(existing_terminal_paths) > 1:
        raise ContiguousOrchestratorError(
            "operator has conflicting terminal receipts"
        )
    if existing_terminal_paths:
        if (
            existing_terminal_paths[0].name
            == "operator_incident.json"
            and not (
                _canary_control_root(config.campaign_root)
                / "placement_receipt.json"
            ).exists()
        ):
            return _strict_json(
                _read_regular(
                    existing_terminal_paths[0],
                    maximum=MAX_JSON_BYTES,
                ),
                label="operator incident receipt",
            )
        planting = _load_or_create_canary_planting(config)
        terminal_value = _strict_json(
            _read_regular(
                existing_terminal_paths[0], maximum=MAX_JSON_BYTES
            ),
            label="operator terminal receipt",
        )
        return _finalize_operator_terminal(
            campaign_root=config.campaign_root,
            planting=planting,
            terminal_receipt_path=existing_terminal_paths[0],
            terminal_value=terminal_value,
        )
    credentials = Transport.load_external_chatgpt_credentials(
        config.credential_source
    )
    planting = _load_or_create_canary_planting(config)
    command_runner = _production_command_runner(config)
    _validate_host_child_ledger_audit(
        command_runner.audit_invocation_ledger(),
        campaign_root=config.campaign_root,
        require_quiescent=True,
    )
    auxiliary_backend = ProductionAuxiliaryBackend(
        campaign_root=config.campaign_root,
        command_runner=command_runner,
        configuration=config.auxiliary_backend_configuration,
        launch_configuration=config.auxiliary_launch_configuration,
    )
    docker = Container.DockerContainerBackend(
        command_runner,
        docker_binary=str(config.docker_binary),
    )
    probe = Container.DockerWorkspaceProbeExecutor(
        docker,
        image_reference=config.workspace_probe_image_reference,
    )
    backend = Container.ContiguousDockerAttemptBackend(
        docker,
        result_collector=TrustedCandidateCollector(),
        credentials=credentials,
        probe_executor=probe,
        controller_state_canaries=planting.canaries,
    )
    replay = DockerReplayExecutor(
        docker,
        replay_image_reference=config.replay_image_reference,
        evidence_root=config.replay_evidence_root,
    )
    expected_selective_import = (
        None
        if config.selective_continuation_game is None
        else Runner.selective_frontier_import_from_dict(
            preflight["selective_frontier_import"]
        )
    )
    promotion_gate = ProductionPromotionGate(
        config.promotion_root,
        replay_executor=replay,
        secret_sentinels=credentials.leak_sentinels,
        frontier_import_root=config.frontier_import_root,
        selective_frontier_import=expected_selective_import,
    )
    runner = Runner.ContiguousCampaignRunner(
        config.campaign_root,
        backend=backend,
        promotion_gate=promotion_gate,
        input_builder=Runner.ProductionInputBundleBuilder(),
        backend_configuration=config.backend_configuration,
        cost_window_id=config.cost_window_id,
        max_lanes=config.max_lanes,
        limit=config.limit,
        operator_configuration_sha256=config.config_sha256,
        secret_sentinels=credentials.leak_sentinels,
        controller_state_canaries=planting.canaries,
        auxiliary_backend=auxiliary_backend,
        auxiliary_launch_configuration=(
            config.auxiliary_launch_configuration
        ),
        selective_continuation_game=(
            config.selective_continuation_game
        ),
        selective_frontier_import=expected_selective_import,
        selective_frontier_import_sha256=(
            config.selective_frontier_import_sha256
        ),
    )
    cycles = 0
    while True:
        operator_lease.assert_healthy()
        report = runner.cycle()
        operator_lease.assert_healthy()
        cycles += 1
        state = runner.state()
        journal_events = runner.journal.read()
        blocked_terminal = _quiescent_authenticated_blocked_projection(
            state,
            journal_head_sequence=journal_events[-1]["sequence"],
            journal_head_digest=journal_events[-1]["digest"],
        )
        runner_incident = state.get("operator_incident")
        if isinstance(runner_incident, dict):
            cleanup_pending = _runner_cleanup_pending(state)
            if not cleanup_pending:
                meta_diagnostic: dict[str, Any] | None = None
                if (
                    runner_incident.get("operation")
                    == "substrate_health_reprobe"
                    and runner_incident.get("fault_domain")
                    == "controller_substrate"
                ):
                    try:
                        meta_diagnostic = (
                            _run_latched_substrate_meta_diagnostic(
                                config,
                                runner=runner,
                                state=state,
                                journal_events=journal_events,
                                command_runner=command_runner,
                                operator_lease=operator_lease,
                            )
                        )
                        resumed, recovered_state = (
                            _apply_latched_substrate_meta_recovery(
                                runner=runner,
                                latched_state=state,
                                meta_diagnostic=meta_diagnostic,
                                operator_lease=operator_lease,
                            )
                        )
                        if resumed:
                            # The trusted runner, not the diagnostic text,
                            # emitted the fresh-probe PASS and one-shot resume
                            # authorization.  Re-enter through a normal cycle.
                            continue
                        state = recovered_state
                        runner_incident = state.get(
                            "operator_incident"
                        )
                    except Exception as exc:
                        error_class = type(exc).__name__
                        if (
                            re.fullmatch(
                                r"[A-Za-z_][A-Za-z0-9_]{0,127}",
                                error_class,
                            )
                            is None
                        ):
                            error_class = "MetaDiagnosticError"
                        failed_meta_body = {
                            "schema": 1,
                            "kind":
                                "arc_agi3_contiguous_post_incident_"
                                "meta_failure",
                            "status": "FAILED_CLOSED",
                            "error_class": error_class,
                            "human_intervention_required": True,
                            "runner_remained_paused": True,
                            "scheduler_authority": False,
                            "solver_authority": False,
                            "wip_authority": False,
                            "cost_authority": False,
                            "retry_authority": False,
                            "dispatch_authority": False,
                            "promotion_authority": False,
                        }
                        meta_diagnostic = {
                            **failed_meta_body,
                            "receipt_sha256":
                                _json_sha256(failed_meta_body),
                        }
                incident_value = _operator_incident_value(
                    config,
                    reason_code="failure_circuit_exhausted",
                    error_class=None,
                    runner_incident=runner_incident,
                    solved_levels=state["solved_levels"],
                    total_levels=state["total_levels"],
                    meta_diagnostic=meta_diagnostic,
                )
                return _finalize_operator_terminal(
                    campaign_root=config.campaign_root,
                    planting=planting,
                    terminal_receipt_path=(
                        config.campaign_root
                        / "operator_incident.json"
                    ),
                    terminal_value=incident_value,
                )
        storage_incident = state.get("storage_incident")
        if (
            isinstance(storage_incident, dict)
            and not _runner_cleanup_pending(state)
        ):
            storage_value = _storage_exhausted_terminal_value(
                config,
                state=state,
                journal_events=journal_events,
            )
            return _finalize_operator_terminal(
                campaign_root=config.campaign_root,
                planting=planting,
                terminal_receipt_path=(
                    config.campaign_root
                    / "operator_storage_exhausted.json"
                ),
                terminal_value=storage_value,
            )
        if config.selective_continuation_game is not None:
            reached_terminal = (
                state.get("selective_complete") is True
                and not _runner_cleanup_pending(state)
                and state.get("pending_scheduler_decision") is None
                and state.get("pending_auxiliary_decision") is None
            )
        else:
            reached_terminal = state["complete"] is True
        if reached_terminal:
            terminal = config.terminal_condition
            break
        if blocked_terminal is not None:
            blocked_terminal = {
                **blocked_terminal,
                "cycles": cycles,
                "campaign_root": str(config.campaign_root),
                "operator_config_sha256": config.config_sha256,
                "terminal_cleanup_intent": str(
                    _operator_terminal_cleanup_intent_path(
                        config.campaign_root
                    )
                ),
                "canary_cleanup_required": True,
            }
            blocked_path = (
                config.campaign_root / "operator_terminal_blocked.json"
            )
            return _finalize_operator_terminal(
                campaign_root=config.campaign_root,
                planting=planting,
                terminal_receipt_path=blocked_path,
                terminal_value=blocked_terminal,
            )
        operator_status = {
            "schema": 1,
            "kind": "contiguous_operator_status",
            "cycles": cycles,
            "solved_levels": report["solved_levels"],
            "total_levels": report["total_levels"],
            "active_lanes": report["active_lanes"],
            "draining": report["draining"],
            "operator_incident": report["operator_incident"],
            "recoverable_errors": report["recoverable_errors"],
        }
        if config.selective_continuation_game is not None:
            operator_status.update({
                "campaign_mode": "selective_continuation",
                "selective_continuation_game":
                    config.selective_continuation_game,
                "operator_authorized_selective_frontier_import_sha256":
                    config.selective_frontier_import_sha256,
                "selective_scope_solved_levels": state[
                    "selective_scope_solved_levels"
                ],
                "selective_scope_total_levels": state[
                    "selective_scope_total_levels"
                ],
                "selective_complete": state["selective_complete"],
                "complete": False,
            })
        _replace_json(
            config.campaign_root / "operator_status.json",
            operator_status,
        )
        time.sleep(config.poll_interval_seconds)
    operator_lease.assert_healthy()
    audits = _terminal_campaign_audit(
        config=config,
        credentials=credentials,
        planting=planting,
        command_runner=command_runner,
    )
    operator_lease.assert_healthy()
    result = {
        "schema": 1,
        "kind": "arc_agi3_contiguous_operator_terminal",
        "status": "PASS",
        "terminal_condition": terminal,
        "cycles": cycles,
        "campaign_root": str(config.campaign_root),
        "promotion_root": str(config.promotion_root),
        "operator_config_sha256": config.config_sha256,
        **audits,
        "canary_placement_receipt": str(planting.receipt_path),
        "terminal_cleanup_intent": str(
            _operator_terminal_cleanup_intent_path(
                config.campaign_root
            )
        ),
        "canary_cleanup_required": True,
    }
    result["receipt_sha256"] = _json_sha256(result)
    return _finalize_operator_terminal(
        campaign_root=config.campaign_root,
        planting=planting,
        terminal_receipt_path=(
            config.campaign_root
            / "terminal_audits"
            / "operator.json"
        ),
        terminal_value=result,
    )


def _handle_operator_exception(
    config: OperatorConfiguration,
    exc: Exception,
) -> dict[str, Any]:
    """Durably envelope a post-genesis fatal while its owner lease is held."""

    campaign_root = Path(
        getattr(config, "campaign_root", Path(""))
    )
    genesis = campaign_root / "operator_genesis.json"
    # Configuration/release preflight is intentionally no-mutation.  Its
    # typed failure remains the caller's exception and the CLI converts it
    # to structured stdout without manufacturing campaign state.
    if (
        not campaign_root.is_absolute()
        or genesis.is_symlink()
        or not genesis.is_file()
    ):
        raise exc
    incident_path = campaign_root / "operator_incident.json"
    if incident_path.exists() and not incident_path.is_symlink():
        return _strict_json(
            _read_regular(
                incident_path, maximum=MAX_JSON_BYTES
            ),
            label="operator incident receipt",
        )
    incident = _operator_incident_value(
        config,
        reason_code="uncaught_post_genesis_fatal",
        error_class=type(exc).__name__,
    )
    control_root = _canary_control_root(campaign_root)
    if (
        (control_root / "master_escrow.json").is_file()
        and (control_root / "placement_receipt.json").is_file()
    ):
        planting = _load_or_create_canary_planting(config)
        return _finalize_operator_terminal(
            campaign_root=campaign_root,
            planting=planting,
            terminal_receipt_path=incident_path,
            terminal_value=incident,
        )
    _ensure_receipt(incident_path, incident)
    return incident


def run_operator(config: OperatorConfiguration) -> dict[str, Any]:
    """Run the operator and durably envelope every post-genesis fatal."""

    try:
        return _run_operator_impl(config)
    except Exception as exc:
        # Acquisition/preflight failures have no genesis and re-raise.  A
        # release-time failure after an already durable incident reopens it
        # without mutating campaign state outside the lease.
        return _handle_operator_exception(config, exc)


class _SingleUseConfigAction(argparse.Action):
    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: object,
        option_string: str | None = None,
    ) -> None:
        del option_string
        if getattr(namespace, self.dest, None) is not None:
            parser.error("--config may be supplied exactly once")
        setattr(namespace, self.dest, values)


def _build_operator_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the fail-closed ARC-AGI-3 contiguous production operator"
        ),
        allow_abbrev=False,
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        action=_SingleUseConfigAction,
        default=None,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_operator_parser().parse_args(argv)
    try:
        result = run_operator(
            load_operator_configuration(args.config)
        )
    except Exception as exc:
        error_class = type(exc).__name__
        if (
            re.fullmatch(
                r"[A-Za-z_][A-Za-z0-9_]{0,127}", error_class
            )
            is None
        ):
            error_class = "OperatorPreflightError"
        body = {
            "schema": 1,
            "kind": "arc_agi3_contiguous_operator_preflight",
            "status": "PREFLIGHT_FAILED",
            "reason_code": "operator_preflight_failed",
            "error_class": error_class,
        }
        result = {
            **body,
            "receipt_sha256": _json_sha256(body),
        }
        sys.stdout.buffer.write(_canonical_json(result) + b"\n")
        return 2
    sys.stdout.buffer.write(_canonical_json(result) + b"\n")
    return (
        2
        if result.get("status")
        in {
            "OPERATOR_INCIDENT",
            "JOURNAL_OR_STORAGE_EXHAUSTED",
        }
        else 0
    )


__all__ = [
    "AUXILIARY_DRIVER_PROTOCOL_SHA256",
    "AuxiliaryBackendDriverConfiguration",
    "ContiguousOrchestratorError",
    "DockerReplayExecutor",
    "IsolatedReplayEvidence",
    "OperatorConfiguration",
    "ProductionPromotionGate",
    "ProductionAuxiliaryBackend",
    "SELECTIVE_TERMINAL_CONDITION",
    "SourceReplayExecutor",
    "TrustedCandidateCollector",
    "audit_contiguous_campaign_unified",
    "audit_selective_continuation_unified",
    "issue_selective_frontier_import_read_only",
    "load_operator_configuration",
    "main",
    "run_operator",
    "verify_contiguous_campaign_unified_audit",
    "verify_selective_continuation_unified_audit",
]


if __name__ == "__main__":
    raise SystemExit(main())
