"""Inert source archive and metadata-only retirement receipts.

The source snapshot preserves exact historical Python bytes after an obsolete
pipeline is physically removed.  Archived bytes are data: this module never
imports, compiles, evaluates, or executes them.  A retirement receipt is only
a canonical pre-retirement inventory.  It cannot authorize execution or
deletion and cannot claim that any file has already been removed.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
from types import MappingProxyType
from typing import Any, Mapping, Sequence
import zlib

from bongard.canonical import canonical_digest, canonical_json


SOURCE_SNAPSHOT_SCHEMA = "gkm.bongard-retired-pipeline-source-snapshot.v1"
SOURCE_ARCHIVE_SCHEMA = "gkm.bongard-retired-pipeline-source-archive.v1"
ARTIFACT_BINDING_SCHEMA = (
    "gkm.bongard-retired-pipeline-artifact-binding.v1"
)
RETIREMENT_RECEIPT_SCHEMA = (
    "gkm.bongard-retired-pipeline-retirement-receipt.v1"
)
RETIREMENT_RECEIPT_CLAIM = (
    "metadata-only-pre-retirement-inventory-not-deletion-authorization"
)

DEFAULT_SOURCE_SNAPSHOT = (
    Path(__file__).resolve().parent
    / "data/panel_retired_pipeline_source_snapshot_20260810_v1.json"
)
DEFAULT_SOURCE_SNAPSHOT_RECORD_DIGEST = (
    "sha256:1e454e0d175ad7971608404005dfa59340c6dbeae1d8a52490b7d97fa84825dd"
)

MAX_SNAPSHOT_FILE_BYTES = 4 * 1024 * 1024
MAX_COMPRESSED_ARCHIVE_BYTES = 2 * 1024 * 1024
MAX_UNCOMPRESSED_ARCHIVE_BYTES = 8 * 1024 * 1024
MAX_SOURCE_BYTES = 512 * 1024
MAX_SOURCE_COUNT = 128
MAX_RECEIPT_FILE_BYTES = 16 * 1024 * 1024
MAX_ARTIFACT_BYTES = 16 * 1024 * 1024 * 1024
MAX_ARTIFACT_BINDINGS = 4096
MAX_LEGACY_MODULES = 256
DECOMPRESSION_INPUT_CHUNK_BYTES = 64 * 1024

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_ADDRESS = re.compile(r"sha256:[0-9a-f]{64}\Z")
_GIT_OID = re.compile(r"[0-9a-f]{40}\Z")
_MODULE = re.compile(r"bongard(?:\.[A-Za-z_][A-Za-z0-9_]*)*\Z")
_PIPELINE_ID = re.compile(r"[a-z0-9][a-z0-9._-]{0,255}\Z")
_PROVENANCE_KINDS = frozenset({"git_blob", "working_tree_and_git_blob"})
_DISPOSITIONS = frozenset(
    {"remove_after_successor_acceptance", "retain_irreducible"}
)


class RetiredPipelineArchiveError(RuntimeError):
    """A source preimage, snapshot, or receipt differs."""


def _address(value: object, label: str) -> str:
    if type(value) is not str or _ADDRESS.fullmatch(value) is None:
        raise RetiredPipelineArchiveError(f"{label} is not a SHA-256 address")
    return value


def _raw_sha256(value: object, label: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise RetiredPipelineArchiveError(f"{label} is not a raw SHA-256")
    return value


def _module(value: object, label: str) -> str:
    if type(value) is not str or _MODULE.fullmatch(value) is None:
        raise RetiredPipelineArchiveError(f"{label} is not a module name")
    return value


def _relative_path(value: object, label: str) -> str:
    if type(value) is not str or not value or "\\" in value:
        raise RetiredPipelineArchiveError(f"{label} is not a relative POSIX path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or value != path.as_posix()
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise RetiredPipelineArchiveError(f"{label} is not a canonical relative path")
    return value


def _git_identity(value: object, label: str) -> str:
    if type(value) is not str or _GIT_OID.fullmatch(value) is None:
        raise RetiredPipelineArchiveError(f"{label} is not a Git identity")
    return value


def _git_blob_oid(source: bytes) -> str:
    header = b"blob " + str(len(source)).encode("ascii") + b"\0"
    return hashlib.sha1(header + source).hexdigest()  # noqa: S324


def _stable_regular_bytes(path: str | Path, *, maximum: int, label: str) -> bytes:
    supplied = Path(os.path.abspath(os.fspath(path)))
    try:
        before = supplied.lstat()
    except OSError as exc:
        raise RetiredPipelineArchiveError(f"cannot stat {label}") from exc
    if (
        stat.S_ISLNK(before.st_mode)
        or not stat.S_ISREG(before.st_mode)
        or not 0 < before.st_size <= maximum
    ):
        raise RetiredPipelineArchiveError(f"{label} is not a bounded regular file")
    flags = os.O_RDONLY
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(supplied, flags)
        try:
            opened = os.fstat(descriptor)
            chunks: list[bytes] = []
            total = 0
            while True:
                chunk = os.read(descriptor, min(1024 * 1024, maximum + 1 - total))
                if not chunk:
                    break
                chunks.append(chunk)
                total += len(chunk)
                if total > maximum:
                    raise RetiredPipelineArchiveError(f"{label} exceeds its bound")
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
    except RetiredPipelineArchiveError:
        raise
    except OSError as exc:
        raise RetiredPipelineArchiveError(f"cannot read {label}") from exc
    identity = lambda item: (  # noqa: E731 - compact immutable stat identity
        item.st_dev,
        item.st_ino,
        item.st_mode,
        item.st_size,
        item.st_mtime_ns,
    )
    if identity(before) != identity(opened) or identity(opened) != identity(after):
        raise RetiredPipelineArchiveError(f"{label} changed while read")
    raw = b"".join(chunks)
    if len(raw) != before.st_size:
        raise RetiredPipelineArchiveError(f"{label} read size differs")
    return raw


def _canonical_object(raw: bytes, label: str) -> dict[str, Any]:
    if not raw.endswith(b"\n"):
        raise RetiredPipelineArchiveError(f"{label} lacks its canonical newline")
    try:
        value = json.loads(raw[:-1].decode("utf-8", errors="strict"))
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise RetiredPipelineArchiveError(f"{label} is malformed") from exc
    if type(value) is not dict or raw != canonical_json(value) + b"\n":
        raise RetiredPipelineArchiveError(f"{label} is not canonical JSON")
    return value


def _bounded_gzip_decompress(payload: bytes) -> bytes:
    if not 0 < len(payload) <= MAX_COMPRESSED_ARCHIVE_BYTES:
        raise RetiredPipelineArchiveError("compressed source archive is not bounded")
    decoder = zlib.decompressobj(wbits=16 + zlib.MAX_WBITS)
    output = bytearray()
    cursor = 0
    try:
        while cursor < len(payload):
            pending = payload[cursor : cursor + DECOMPRESSION_INPUT_CHUNK_BYTES]
            cursor += len(pending)
            while pending:
                if decoder.eof:
                    raise RetiredPipelineArchiveError(
                        "source archive gzip has trailing data"
                    )
                remaining = MAX_UNCOMPRESSED_ARCHIVE_BYTES - len(output)
                if remaining <= 0:
                    raise RetiredPipelineArchiveError(
                        "source archive gzip exceeds its output bound"
                    )
                before = len(pending)
                piece = decoder.decompress(pending, remaining)
                output.extend(piece)
                pending = decoder.unconsumed_tail
                if decoder.unused_data:
                    raise RetiredPipelineArchiveError(
                        "source archive gzip has trailing data"
                    )
                if pending and not piece and len(pending) >= before:
                    raise RetiredPipelineArchiveError(
                        "source archive gzip decoder made no progress"
                    )
                if decoder.eof:
                    if pending or cursor != len(payload):
                        raise RetiredPipelineArchiveError(
                            "source archive gzip has trailing data"
                        )
                    break
            if decoder.eof:
                break
    except zlib.error as exc:
        raise RetiredPipelineArchiveError("source archive gzip is malformed") from exc
    if (
        not decoder.eof
        or decoder.unconsumed_tail
        or decoder.unused_data
        or cursor != len(payload)
    ):
        raise RetiredPipelineArchiveError("source archive gzip differs or exceeds bounds")
    return bytes(output)


@dataclass(frozen=True, slots=True)
class RetiredPipelineSourceArchive:
    """Authenticated inert bytes indexed by module and source address."""

    record_digest: str
    snapshot_file_sha256: str
    entries: Mapping[str, Mapping[str, Any]]
    sources: Mapping[str, bytes]

    def source_for(self, module: str, source_sha256: str) -> bytes:
        checked_module = _module(module, "source module")
        checked_sha256 = _raw_sha256(source_sha256, "source SHA-256")
        snapshot_id = f"{checked_module}@sha256:{checked_sha256}"
        try:
            return self.sources[snapshot_id]
        except KeyError as exc:
            raise RetiredPipelineArchiveError(
                f"retired source preimage is absent: {snapshot_id}"
            ) from exc


def load_retired_pipeline_source_archive(
    path: str | Path = DEFAULT_SOURCE_SNAPSHOT,
    *,
    expected_record_digest: str = DEFAULT_SOURCE_SNAPSHOT_RECORD_DIGEST,
) -> RetiredPipelineSourceArchive:
    """Authenticate and decode exact source bytes without executing them."""

    expected = _address(expected_record_digest, "expected source snapshot")
    raw_file = _stable_regular_bytes(
        path, maximum=MAX_SNAPSHOT_FILE_BYTES, label="source snapshot"
    )
    value = _canonical_object(raw_file, "source snapshot")
    if set(value) != {
        "archive",
        "entries",
        "record_digest",
        "record_digest_policy",
        "schema",
    } or value.get("schema") != SOURCE_SNAPSHOT_SCHEMA:
        raise RetiredPipelineArchiveError("source snapshot schema differs")
    body = dict(value)
    record_digest = body.pop("record_digest", None)
    if (
        value.get("record_digest_policy")
        != "sha256(canonical JSON of this object with record_digest omitted)"
        or record_digest != "sha256:" + canonical_digest(body)
        or record_digest != expected
    ):
        raise RetiredPipelineArchiveError("source snapshot digest differs")

    metadata = value.get("archive")
    entries = value.get("entries")
    if type(metadata) is not dict or type(entries) is not list:
        raise RetiredPipelineArchiveError("source snapshot structure differs")
    if not 0 < len(entries) <= MAX_SOURCE_COUNT:
        raise RetiredPipelineArchiveError("source snapshot entry count differs")
    if set(metadata) != {
        "compressed_byte_count",
        "compressed_sha256",
        "compression",
        "content_schema",
        "payload_base64",
        "uncompressed_byte_count",
        "uncompressed_sha256",
    } or metadata.get("compression") != "gzip-mtime-0":
        raise RetiredPipelineArchiveError("source archive metadata differs")
    try:
        compressed = base64.b64decode(metadata.get("payload_base64"), validate=True)
    except (TypeError, ValueError) as exc:
        raise RetiredPipelineArchiveError("source archive base64 differs") from exc
    if (
        len(compressed) != metadata.get("compressed_byte_count")
        or hashlib.sha256(compressed).hexdigest()
        != metadata.get("compressed_sha256")
    ):
        raise RetiredPipelineArchiveError("compressed source archive digest differs")
    raw_archive = _bounded_gzip_decompress(compressed)
    if (
        len(raw_archive) != metadata.get("uncompressed_byte_count")
        or hashlib.sha256(raw_archive).hexdigest()
        != metadata.get("uncompressed_sha256")
    ):
        raise RetiredPipelineArchiveError("source archive payload digest differs")
    try:
        archive = json.loads(raw_archive.decode("utf-8", errors="strict"))
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise RetiredPipelineArchiveError("source archive payload is malformed") from exc
    if (
        type(archive) is not dict
        or raw_archive != canonical_json(archive)
        or set(archive) != {"schema", "sources"}
        or archive.get("schema") != SOURCE_ARCHIVE_SCHEMA
        or metadata.get("content_schema") != SOURCE_ARCHIVE_SCHEMA
        or type(archive.get("sources")) is not list
    ):
        raise RetiredPipelineArchiveError("source archive payload schema differs")

    by_id: dict[str, Mapping[str, Any]] = {}
    for entry in entries:
        if type(entry) is not dict or set(entry) != {
            "artifact_bindings",
            "git_blob_oid",
            "module",
            "provenance_kind",
            "relative_path",
            "snapshot_id",
            "source_byte_count",
            "source_commit",
            "source_sha256",
        }:
            raise RetiredPipelineArchiveError("source snapshot entry differs")
        module = _module(entry.get("module"), "snapshot module")
        source_sha256 = _raw_sha256(
            entry.get("source_sha256"), "snapshot source SHA-256"
        )
        snapshot_id = f"{module}@sha256:{source_sha256}"
        bindings = entry.get("artifact_bindings")
        if (
            entry.get("snapshot_id") != snapshot_id
            or snapshot_id in by_id
            or _relative_path(entry.get("relative_path"), "snapshot source path")
            != entry["relative_path"]
            or _git_identity(entry.get("git_blob_oid"), "snapshot Git blob")
            != entry["git_blob_oid"]
            or _git_identity(entry.get("source_commit"), "snapshot source commit")
            != entry["source_commit"]
            or entry.get("provenance_kind") not in _PROVENANCE_KINDS
            or type(entry.get("source_byte_count")) is not int
            or not 0 < entry["source_byte_count"] <= MAX_SOURCE_BYTES
            or type(bindings) is not list
            or not bindings
            or bindings != sorted(set(bindings))
            or any(type(item) is not str or not item for item in bindings)
        ):
            raise RetiredPipelineArchiveError("source snapshot identity differs")
        frozen_entry = dict(entry)
        frozen_entry["artifact_bindings"] = tuple(bindings)
        by_id[snapshot_id] = MappingProxyType(frozen_entry)
    if list(by_id) != sorted(by_id):
        raise RetiredPipelineArchiveError("source snapshot entries are not ordered")

    source_items = archive["sources"]
    if len(source_items) != len(by_id):
        raise RetiredPipelineArchiveError("source archive inventory differs")
    sources: dict[str, bytes] = {}
    for item in source_items:
        if type(item) is not dict or set(item) != {"snapshot_id", "source_utf8"}:
            raise RetiredPipelineArchiveError("archived source item differs")
        snapshot_id = item.get("snapshot_id")
        source_text = item.get("source_utf8")
        if type(snapshot_id) is not str or type(source_text) is not str:
            raise RetiredPipelineArchiveError("archived source identity differs")
        try:
            entry = by_id[snapshot_id]
        except KeyError as exc:
            raise RetiredPipelineArchiveError("unregistered archived source") from exc
        source = source_text.encode("utf-8", errors="strict")
        if (
            snapshot_id in sources
            or len(source) != entry["source_byte_count"]
            or hashlib.sha256(source).hexdigest() != entry["source_sha256"]
            or _git_blob_oid(source) != entry["git_blob_oid"]
        ):
            raise RetiredPipelineArchiveError("archived source preimage differs")
        sources[snapshot_id] = source
    if list(sources) != sorted(sources) or set(sources) != set(by_id):
        raise RetiredPipelineArchiveError("source archive ordering differs")
    return RetiredPipelineSourceArchive(
        record_digest=record_digest,
        snapshot_file_sha256="sha256:" + hashlib.sha256(raw_file).hexdigest(),
        entries=MappingProxyType(by_id),
        sources=MappingProxyType(sources),
    )


def verify_retired_pipeline_source_binding(
    module: str,
    source_sha256: str,
    *,
    archive: RetiredPipelineSourceArchive | None = None,
) -> bytes:
    """Return exact inert bytes after their address and Git identity verify."""

    loaded = archive or load_retired_pipeline_source_archive()
    source = loaded.source_for(module, source_sha256)
    snapshot_id = f"{module}@sha256:{source_sha256}"
    entry = loaded.entries[snapshot_id]
    if (
        hashlib.sha256(source).hexdigest() != source_sha256
        or _git_blob_oid(source) != entry["git_blob_oid"]
    ):
        raise RetiredPipelineArchiveError("retired source binding differs")
    return source


@dataclass(frozen=True, slots=True)
class RetiredPipelineArtifactBinding:
    """Hash-only inventory row; it grants no filesystem authority."""

    relative_path: str
    raw_sha256: str
    byte_count: int
    canonical_record_digest: str | None
    disposition: str

    def __post_init__(self) -> None:
        _relative_path(self.relative_path, "artifact path")
        _address(self.raw_sha256, "artifact raw SHA-256")
        if (
            type(self.byte_count) is not int
            or not 0 < self.byte_count <= MAX_ARTIFACT_BYTES
        ):
            raise RetiredPipelineArchiveError("artifact byte count differs")
        if self.canonical_record_digest is not None:
            _address(self.canonical_record_digest, "artifact record digest")
        if self.disposition not in _DISPOSITIONS:
            raise RetiredPipelineArchiveError("artifact disposition differs")

    def to_data(self) -> dict[str, Any]:
        return {
            "byte_count": self.byte_count,
            "canonical_record_digest": self.canonical_record_digest,
            "disposition": self.disposition,
            "raw_sha256": self.raw_sha256,
            "relative_path": self.relative_path,
            "schema": ARTIFACT_BINDING_SCHEMA,
        }

    @classmethod
    def from_data(cls, value: object) -> "RetiredPipelineArtifactBinding":
        if type(value) is not dict or set(value) != {
            "byte_count",
            "canonical_record_digest",
            "disposition",
            "raw_sha256",
            "relative_path",
            "schema",
        } or value.get("schema") != ARTIFACT_BINDING_SCHEMA:
            raise RetiredPipelineArchiveError("artifact binding fields differ")
        return cls(
            relative_path=value["relative_path"],
            raw_sha256=value["raw_sha256"],
            byte_count=value["byte_count"],
            canonical_record_digest=value["canonical_record_digest"],
            disposition=value["disposition"],
        )


def _receipt_content(
    *,
    active_successor_pipeline_id: str,
    source_snapshot_record_digest: str,
    source_snapshot_file_sha256: str,
    source_snapshot_entry_count: int,
    legacy_modules: Sequence[str],
    artifact_bindings: Sequence[RetiredPipelineArtifactBinding],
) -> dict[str, Any]:
    return {
        "active_successor_pipeline_id": active_successor_pipeline_id,
        "artifact_bindings": [item.to_data() for item in artifact_bindings],
        "claim": RETIREMENT_RECEIPT_CLAIM,
        "deletion_authorized": False,
        "execution_authorized": False,
        "files_removed": 0,
        "legacy_modules": list(legacy_modules),
        "schema": RETIREMENT_RECEIPT_SCHEMA,
        "source_snapshot_entry_count": source_snapshot_entry_count,
        "source_snapshot_file_sha256": source_snapshot_file_sha256,
        "source_snapshot_record_digest": source_snapshot_record_digest,
    }


@dataclass(frozen=True, slots=True)
class RetiredPipelineRetirementReceipt:
    """Canonical pre-retirement inventory with permanently false authority."""

    active_successor_pipeline_id: str
    source_snapshot_record_digest: str
    source_snapshot_file_sha256: str
    source_snapshot_entry_count: int
    legacy_modules: tuple[str, ...]
    artifact_bindings: tuple[RetiredPipelineArtifactBinding, ...]
    execution_authorized: bool
    deletion_authorized: bool
    files_removed: int
    record_digest: str

    def __post_init__(self) -> None:
        if (
            type(self.active_successor_pipeline_id) is not str
            or _PIPELINE_ID.fullmatch(self.active_successor_pipeline_id) is None
        ):
            raise RetiredPipelineArchiveError("successor pipeline id differs")
        _address(self.source_snapshot_record_digest, "source snapshot record")
        _address(self.source_snapshot_file_sha256, "source snapshot file")
        if (
            type(self.source_snapshot_entry_count) is not int
            or not 0 < self.source_snapshot_entry_count <= MAX_SOURCE_COUNT
        ):
            raise RetiredPipelineArchiveError("source snapshot entry count differs")
        if (
            type(self.legacy_modules) is not tuple
            or not 0 < len(self.legacy_modules) <= MAX_LEGACY_MODULES
            or self.legacy_modules != tuple(sorted(set(self.legacy_modules)))
        ):
            raise RetiredPipelineArchiveError("legacy module inventory differs")
        for item in self.legacy_modules:
            _module(item, "legacy module")
        if (
            type(self.artifact_bindings) is not tuple
            or not 0 < len(self.artifact_bindings) <= MAX_ARTIFACT_BINDINGS
            or any(
                type(item) is not RetiredPipelineArtifactBinding
                for item in self.artifact_bindings
            )
            or tuple(item.relative_path for item in self.artifact_bindings)
            != tuple(sorted({item.relative_path for item in self.artifact_bindings}))
        ):
            raise RetiredPipelineArchiveError("artifact inventory differs")
        if (
            self.execution_authorized is not False
            or self.deletion_authorized is not False
            or type(self.files_removed) is not int
            or self.files_removed != 0
        ):
            raise RetiredPipelineArchiveError("retirement receipt cannot grant authority")
        content = _receipt_content(
            active_successor_pipeline_id=self.active_successor_pipeline_id,
            source_snapshot_record_digest=self.source_snapshot_record_digest,
            source_snapshot_file_sha256=self.source_snapshot_file_sha256,
            source_snapshot_entry_count=self.source_snapshot_entry_count,
            legacy_modules=self.legacy_modules,
            artifact_bindings=self.artifact_bindings,
        )
        if self.record_digest != "sha256:" + canonical_digest(content):
            raise RetiredPipelineArchiveError("retirement receipt digest differs")

    @classmethod
    def create(
        cls,
        *,
        active_successor_pipeline_id: str,
        source_snapshot_record_digest: str,
        source_snapshot_file_sha256: str,
        source_snapshot_entry_count: int,
        legacy_modules: Sequence[str],
        artifact_bindings: Sequence[RetiredPipelineArtifactBinding],
        execution_authorized: bool = False,
        deletion_authorized: bool = False,
        files_removed: int = 0,
    ) -> "RetiredPipelineRetirementReceipt":
        if (
            execution_authorized is not False
            or deletion_authorized is not False
            or type(files_removed) is not int
            or files_removed != 0
        ):
            raise RetiredPipelineArchiveError("retirement receipt cannot grant authority")
        modules = tuple(sorted(legacy_modules))
        bindings = tuple(sorted(artifact_bindings, key=lambda item: item.relative_path))
        content = _receipt_content(
            active_successor_pipeline_id=active_successor_pipeline_id,
            source_snapshot_record_digest=source_snapshot_record_digest,
            source_snapshot_file_sha256=source_snapshot_file_sha256,
            source_snapshot_entry_count=source_snapshot_entry_count,
            legacy_modules=modules,
            artifact_bindings=bindings,
        )
        return cls(
            active_successor_pipeline_id=active_successor_pipeline_id,
            source_snapshot_record_digest=source_snapshot_record_digest,
            source_snapshot_file_sha256=source_snapshot_file_sha256,
            source_snapshot_entry_count=source_snapshot_entry_count,
            legacy_modules=modules,
            artifact_bindings=bindings,
            execution_authorized=False,
            deletion_authorized=False,
            files_removed=0,
            record_digest="sha256:" + canonical_digest(content),
        )

    def to_data(self) -> dict[str, Any]:
        return {
            **_receipt_content(
                active_successor_pipeline_id=self.active_successor_pipeline_id,
                source_snapshot_record_digest=self.source_snapshot_record_digest,
                source_snapshot_file_sha256=self.source_snapshot_file_sha256,
                source_snapshot_entry_count=self.source_snapshot_entry_count,
                legacy_modules=self.legacy_modules,
                artifact_bindings=self.artifact_bindings,
            ),
            "record_digest": self.record_digest,
        }

    @classmethod
    def from_data(cls, value: object) -> "RetiredPipelineRetirementReceipt":
        if type(value) is not dict or set(value) != {
            "active_successor_pipeline_id",
            "artifact_bindings",
            "claim",
            "deletion_authorized",
            "execution_authorized",
            "files_removed",
            "legacy_modules",
            "record_digest",
            "schema",
            "source_snapshot_entry_count",
            "source_snapshot_file_sha256",
            "source_snapshot_record_digest",
        } or value.get("schema") != RETIREMENT_RECEIPT_SCHEMA:
            raise RetiredPipelineArchiveError("retirement receipt fields differ")
        if value.get("claim") != RETIREMENT_RECEIPT_CLAIM:
            raise RetiredPipelineArchiveError("retirement receipt claim differs")
        raw_modules = value.get("legacy_modules")
        raw_bindings = value.get("artifact_bindings")
        if type(raw_modules) is not list or type(raw_bindings) is not list:
            raise RetiredPipelineArchiveError("retirement receipt inventory differs")
        return cls(
            active_successor_pipeline_id=value["active_successor_pipeline_id"],
            source_snapshot_record_digest=value["source_snapshot_record_digest"],
            source_snapshot_file_sha256=value["source_snapshot_file_sha256"],
            source_snapshot_entry_count=value["source_snapshot_entry_count"],
            legacy_modules=tuple(raw_modules),
            artifact_bindings=tuple(
                RetiredPipelineArtifactBinding.from_data(item)
                for item in raw_bindings
            ),
            execution_authorized=value["execution_authorized"],
            deletion_authorized=value["deletion_authorized"],
            files_removed=value["files_removed"],
            record_digest=value["record_digest"],
        )


def load_retirement_receipt(
    path: str | Path, *, expected_record_digest: str | None = None
) -> RetiredPipelineRetirementReceipt:
    """Load a strict canonical receipt without opening any bound artifact."""

    raw = _stable_regular_bytes(
        path, maximum=MAX_RECEIPT_FILE_BYTES, label="retirement receipt"
    )
    receipt = RetiredPipelineRetirementReceipt.from_data(
        _canonical_object(raw, "retirement receipt")
    )
    if (
        expected_record_digest is not None
        and receipt.record_digest
        != _address(expected_record_digest, "expected retirement receipt")
    ):
        raise RetiredPipelineArchiveError("retirement receipt address differs")
    return receipt


__all__ = (
    "ARTIFACT_BINDING_SCHEMA",
    "DEFAULT_SOURCE_SNAPSHOT",
    "DEFAULT_SOURCE_SNAPSHOT_RECORD_DIGEST",
    "RETIREMENT_RECEIPT_CLAIM",
    "RETIREMENT_RECEIPT_SCHEMA",
    "RetiredPipelineArchiveError",
    "RetiredPipelineArtifactBinding",
    "RetiredPipelineRetirementReceipt",
    "RetiredPipelineSourceArchive",
    "SOURCE_ARCHIVE_SCHEMA",
    "SOURCE_SNAPSHOT_SCHEMA",
    "load_retired_pipeline_source_archive",
    "load_retirement_receipt",
    "verify_retired_pipeline_source_binding",
)
