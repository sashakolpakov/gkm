"""Decode authenticated source snapshots for physically retired panel probes.

The archive is data, not an import or execution fallback.  It preserves the
exact source preimages named by historical authorizations after their live
command modules are removed.  This decoder never compiles or executes an
archived byte.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from types import MappingProxyType
from typing import Any, Mapping
import zlib

from bongard.canonical import canonical_digest, canonical_json


SOURCE_SNAPSHOT_SCHEMA = "gkm.bongard-retired-panel-probe-source-snapshot.v1"
SOURCE_ARCHIVE_SCHEMA = "gkm.bongard-retired-panel-probe-source-archive.v1"
DEFAULT_SOURCE_SNAPSHOT = (
    Path(__file__).resolve().parent
    / "data/panel_retired_probe_source_snapshot_20260810_v1.json"
)
DEFAULT_SOURCE_SNAPSHOT_RECORD_DIGEST = (
    "sha256:91663bcc85bc907cd2215473c61910190cbcd53a9a21f7afc69e1dc0ae858d66"
)
MAX_SNAPSHOT_BYTES = 256 * 1024
MAX_COMPRESSED_ARCHIVE_BYTES = 192 * 1024
MAX_UNCOMPRESSED_ARCHIVE_BYTES = 768 * 1024
MAX_SOURCE_BYTES = 128 * 1024
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_GIT_OID = re.compile(r"[0-9a-f]{40}\Z")


class RetiredProbeSourceArchiveError(RuntimeError):
    """The snapshot manifest or one of its source preimages differs."""


@dataclass(frozen=True, slots=True)
class RetiredProbeSourceArchive:
    """Authenticated, inert bytes for historical source-digest checks."""

    record_digest: str
    entries: Mapping[str, Mapping[str, Any]]
    sources: Mapping[str, bytes]

    def source_for(self, module: str, source_sha256: str) -> bytes:
        snapshot_id = f"{module}@sha256:{source_sha256}"
        try:
            return self.sources[snapshot_id]
        except KeyError as exc:
            raise RetiredProbeSourceArchiveError(
                f"retired source preimage is absent: {snapshot_id}"
            ) from exc


def _read_bounded_canonical(path: str | Path) -> tuple[dict[str, Any], bytes]:
    source = Path(os.path.abspath(os.fspath(path)))
    if source.is_symlink():
        raise RetiredProbeSourceArchiveError("source snapshot is a symlink")
    try:
        before = source.stat()
        if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= MAX_SNAPSHOT_BYTES:
            raise RetiredProbeSourceArchiveError("source snapshot is not bounded")
        raw = source.read_bytes()
        after = source.stat()
    except OSError as exc:
        raise RetiredProbeSourceArchiveError("source snapshot is unavailable") from exc
    if (
        len(raw) != before.st_size
        or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        or not raw.endswith(b"\n")
    ):
        raise RetiredProbeSourceArchiveError("source snapshot changed while read")
    try:
        value = json.loads(raw[:-1].decode("utf-8", errors="strict"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise RetiredProbeSourceArchiveError("source snapshot is malformed") from exc
    if type(value) is not dict or raw != canonical_json(value) + b"\n":
        raise RetiredProbeSourceArchiveError("source snapshot is not canonical")
    return value, raw


def _bounded_gzip_decompress(payload: bytes) -> bytes:
    if not 0 < len(payload) <= MAX_COMPRESSED_ARCHIVE_BYTES:
        raise RetiredProbeSourceArchiveError("compressed source archive is not bounded")
    decoder = zlib.decompressobj(wbits=16 + zlib.MAX_WBITS)
    try:
        raw = decoder.decompress(payload, MAX_UNCOMPRESSED_ARCHIVE_BYTES + 1)
    except zlib.error as exc:
        raise RetiredProbeSourceArchiveError("source archive gzip is malformed") from exc
    if (
        len(raw) > MAX_UNCOMPRESSED_ARCHIVE_BYTES
        or not decoder.eof
        or decoder.unconsumed_tail
        or decoder.unused_data
        or decoder.flush()
    ):
        raise RetiredProbeSourceArchiveError("source archive gzip differs or exceeds bounds")
    return raw


def _git_blob_oid(source: bytes) -> str:
    header = b"blob " + str(len(source)).encode("ascii") + b"\0"
    return hashlib.sha1(header + source).hexdigest()  # noqa: S324 - Git object identity


def load_retired_probe_source_archive(
    path: str | Path = DEFAULT_SOURCE_SNAPSHOT,
    *,
    expected_record_digest: str = DEFAULT_SOURCE_SNAPSHOT_RECORD_DIGEST,
) -> RetiredProbeSourceArchive:
    """Authenticate and decode the inert source snapshot archive."""

    value, _ = _read_bounded_canonical(path)
    if set(value) != {
        "archive",
        "entries",
        "record_digest",
        "record_digest_policy",
        "schema",
    } or value.get("schema") != SOURCE_SNAPSHOT_SCHEMA:
        raise RetiredProbeSourceArchiveError("source snapshot schema differs")
    body = dict(value)
    record_digest = body.pop("record_digest", None)
    if (
        value.get("record_digest_policy")
        != "sha256(canonical JSON of this object with record_digest omitted)"
        or record_digest != "sha256:" + canonical_digest(body)
        or record_digest != expected_record_digest
    ):
        raise RetiredProbeSourceArchiveError("source snapshot digest differs")

    archive_metadata = value["archive"]
    entries = value["entries"]
    if type(archive_metadata) is not dict or type(entries) is not list:
        raise RetiredProbeSourceArchiveError("source snapshot structure differs")
    if set(archive_metadata) != {
        "compressed_byte_count",
        "compressed_sha256",
        "compression",
        "content_schema",
        "payload_base64",
        "uncompressed_byte_count",
        "uncompressed_sha256",
    } or archive_metadata.get("compression") != "gzip-mtime-0":
        raise RetiredProbeSourceArchiveError("source archive metadata differs")
    try:
        compressed = base64.b64decode(
            archive_metadata["payload_base64"], validate=True
        )
    except (TypeError, ValueError) as exc:
        raise RetiredProbeSourceArchiveError("source archive base64 differs") from exc
    if (
        len(compressed) != archive_metadata.get("compressed_byte_count")
        or hashlib.sha256(compressed).hexdigest()
        != archive_metadata.get("compressed_sha256")
    ):
        raise RetiredProbeSourceArchiveError("compressed source archive digest differs")
    raw_archive = _bounded_gzip_decompress(compressed)
    if (
        len(raw_archive) != archive_metadata.get("uncompressed_byte_count")
        or hashlib.sha256(raw_archive).hexdigest()
        != archive_metadata.get("uncompressed_sha256")
    ):
        raise RetiredProbeSourceArchiveError("source archive payload digest differs")
    try:
        archive = json.loads(raw_archive.decode("utf-8", errors="strict"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise RetiredProbeSourceArchiveError("source archive payload is malformed") from exc
    if (
        type(archive) is not dict
        or raw_archive != canonical_json(archive)
        or set(archive) != {"schema", "sources"}
        or archive.get("schema") != SOURCE_ARCHIVE_SCHEMA
        or archive_metadata.get("content_schema") != SOURCE_ARCHIVE_SCHEMA
        or type(archive.get("sources")) is not list
    ):
        raise RetiredProbeSourceArchiveError("source archive payload schema differs")

    by_id: dict[str, Mapping[str, Any]] = {}
    for entry in entries:
        if type(entry) is not dict or set(entry) != {
            "artifact_bindings",
            "git_blob_oid",
            "module",
            "relative_path",
            "snapshot_id",
            "source_byte_count",
            "source_commit",
            "source_sha256",
        }:
            raise RetiredProbeSourceArchiveError("source snapshot entry differs")
        snapshot_id = entry.get("snapshot_id")
        module = entry.get("module")
        source_sha256 = entry.get("source_sha256")
        if (
            type(snapshot_id) is not str
            or type(module) is not str
            or type(source_sha256) is not str
            or snapshot_id != f"{module}@sha256:{source_sha256}"
            or _SHA256.fullmatch(source_sha256) is None
            or _GIT_OID.fullmatch(str(entry.get("git_blob_oid"))) is None
            or _GIT_OID.fullmatch(str(entry.get("source_commit"))) is None
            or type(entry.get("source_byte_count")) is not int
            or not 0 < entry["source_byte_count"] <= MAX_SOURCE_BYTES
            or type(entry.get("artifact_bindings")) is not list
            or any(type(item) is not str or not item for item in entry["artifact_bindings"])
            or snapshot_id in by_id
        ):
            raise RetiredProbeSourceArchiveError("source snapshot identity differs")
        by_id[snapshot_id] = MappingProxyType(entry)
    if list(by_id) != sorted(by_id):
        raise RetiredProbeSourceArchiveError("source snapshot entries are not ordered")

    sources: dict[str, bytes] = {}
    for item in archive["sources"]:
        if type(item) is not dict or set(item) != {"snapshot_id", "source_utf8"}:
            raise RetiredProbeSourceArchiveError("archived source item differs")
        snapshot_id = item.get("snapshot_id")
        source_text = item.get("source_utf8")
        if type(snapshot_id) is not str or type(source_text) is not str:
            raise RetiredProbeSourceArchiveError("archived source identity differs")
        try:
            metadata = by_id[snapshot_id]
        except KeyError as exc:
            raise RetiredProbeSourceArchiveError("unregistered archived source") from exc
        source = source_text.encode("utf-8", errors="strict")
        if (
            snapshot_id in sources
            or len(source) != metadata["source_byte_count"]
            or hashlib.sha256(source).hexdigest() != metadata["source_sha256"]
            or _git_blob_oid(source) != metadata["git_blob_oid"]
        ):
            raise RetiredProbeSourceArchiveError("archived source preimage differs")
        sources[snapshot_id] = source
    if set(sources) != set(by_id):
        raise RetiredProbeSourceArchiveError("source archive inventory differs")
    return RetiredProbeSourceArchive(
        record_digest=record_digest,
        entries=MappingProxyType(by_id),
        sources=MappingProxyType(sources),
    )


def verify_retired_source_binding(
    module: str,
    source_sha256: str,
    *,
    archive: RetiredProbeSourceArchive | None = None,
) -> bytes:
    """Return exact inert bytes only after their archived SHA-256 is verified."""

    loaded = archive or load_retired_probe_source_archive()
    source = loaded.source_for(module, source_sha256)
    if hashlib.sha256(source).hexdigest() != source_sha256:
        raise RetiredProbeSourceArchiveError("retired source binding differs")
    return source


def verify_retired_source_bound_record(
    path: str | Path,
    module: str,
    *,
    source_digest_field: str = "command_source_digest",
    archive: RetiredProbeSourceArchive | None = None,
) -> Mapping[str, Any]:
    """Authenticate a historical record and its exact archived source bytes."""

    value, _ = _read_bounded_canonical(path)
    body = dict(value)
    record_digest = body.pop("record_digest", None)
    if record_digest != "sha256:" + canonical_digest(body):
        raise RetiredProbeSourceArchiveError("historical record digest differs")
    source_sha256 = value.get(source_digest_field)
    if type(source_sha256) is not str or _SHA256.fullmatch(source_sha256) is None:
        raise RetiredProbeSourceArchiveError("historical source binding is absent")
    verify_retired_source_binding(module, source_sha256, archive=archive)
    return MappingProxyType(value)


__all__ = (
    "DEFAULT_SOURCE_SNAPSHOT",
    "DEFAULT_SOURCE_SNAPSHOT_RECORD_DIGEST",
    "RetiredProbeSourceArchive",
    "RetiredProbeSourceArchiveError",
    "SOURCE_ARCHIVE_SCHEMA",
    "SOURCE_SNAPSHOT_SCHEMA",
    "load_retired_probe_source_archive",
    "verify_retired_source_binding",
    "verify_retired_source_bound_record",
)
