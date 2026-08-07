"""Exactly-once journals for the two non-shard Bongard Codex turns.

The semantic proposer uses a named-image structured transport while the
survivor ranker uses a zero-image text structured transport.  These wrappers
freeze the complete input and runtime envelopes for those two call shapes.
Each physical turn is admitted by an exclusive, fsynced claim; its complete
``CodexStructuredResult`` is then persisted and fsynced before a separate
terminal marker is created.  A terminal turn is replayed from disk without
calling Codex.  A stranded claim is never retried.

This module is deliberately Python-authoritative.  Lean is neither imported
nor consulted and has no effect on record identity, replay, or acceptance.
"""

from __future__ import annotations

from bongard.runtime_source_snapshot import capture_loaded_source, verify_loaded_source


_LOADED_SOURCE_SHA256 = capture_loaded_source(__name__, __file__)

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import tempfile
from typing import Any, Callable, Mapping, Sequence

from bongard.canonical import canonical_digest, canonical_json
from bongard.codex_no_tools_preflight import CodexNoToolsAttestation
from bongard.python_predicate_authority import PYTHON_PREDICATE_AUTHORITY_ID
from bongard.transport import (
    DEFAULT_CODEX_MODEL,
    DEFAULT_REASONING_EFFORT,
    CloudPolicyCacheSnapshot,
    CodexModelCatalogSnapshot,
    CodexReceipt,
    CodexStructuredResult,
    validate_codex_named_image_receipt,
    validate_codex_receipt,
    validate_codex_strict_output_schema,
    validate_codex_text_receipt,
)


TURN_JOURNAL_MANIFEST_SCHEMA = "gkm.bongard-codex-turn-journal-manifest.v1"
TURN_CLAIM_SCHEMA = "gkm.bongard-codex-turn-claim.v1"
TURN_RESULT_SCHEMA = "gkm.bongard-codex-turn-result.v1"
TURN_OUTCOME_SCHEMA = "gkm.bongard-codex-turn-outcome.v1"
TURN_JOURNAL_SUMMARY_SCHEMA = "gkm.bongard-codex-turn-journal-summary.v1"
TURN_JOURNAL_PROTOCOL_ID = "bongard.codex-turn/exclusive-result-before-terminal-v1"

NAMED_IMAGE_MODALITY = "named_image_structured"
TEXT_MODALITY = "text_structured"

MAX_RECORD_BYTES = 24 * 1024 * 1024
MAX_PROMPT_BYTES = 2 * 1024 * 1024
MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_NAMED_IMAGES = 64

_DIGEST = re.compile(r"(?:sha256:)?[0-9a-f]{64}\Z")
_RAW_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_TASK_ID = re.compile(r"(?:bd|ff|hd)_[A-Za-z0-9_.-]+\Z")
_TURN_KIND = re.compile(r"[a-z][a-z0-9_-]{0,63}\Z")
_IMAGE_NAME = re.compile(r"[a-z][a-z0-9_-]{0,63}\.png\Z")
_EXCEPTION_TYPE = re.compile(r"[A-Za-z_][A-Za-z0-9_.]{0,255}\Z")


class ObjectBongardTurnJournalError(RuntimeError):
    """A turn journal, invocation, or replay invariant failed."""


class ObjectBongardTurnNonterminalClaim(ObjectBongardTurnJournalError):
    """A process claimed a physical turn but never terminalized it."""


class ObjectBongardTurnCallFailed(ObjectBongardTurnJournalError):
    """The admitted physical Codex turn has a durable typed failure."""

    def __init__(self, *, turn_key: str, failure_digest: str) -> None:
        super().__init__("physical Codex turn failed")
        self.turn_key = turn_key
        self.failure_digest = failure_digest


def object_bongard_turn_journal_source_digest() -> str:
    return verify_loaded_source(
        __name__, expected_source_sha256=_LOADED_SOURCE_SHA256
    )


def _authority_data() -> dict[str, object]:
    return {
        "predicate_authority_id": PYTHON_PREDICATE_AUTHORITY_ID,
        "python_is_canonical_authority": True,
        "lean_present": False,
        "lean_required": False,
        "lean_removable": True,
        "lean_affects_identity_or_replay": False,
    }


def _address(value: object) -> str:
    return "sha256:" + canonical_digest(value)


def _require_digest(value: object, label: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise ObjectBongardTurnJournalError(
            f"{label} must be lowercase SHA-256, raw or sha256:-addressed"
        )
    return value


def _require_raw_digest(value: object, label: str) -> str:
    if not isinstance(value, str) or _RAW_DIGEST.fullmatch(value) is None:
        raise ObjectBongardTurnJournalError(
            f"{label} must be 64 lowercase hexadecimal characters"
        )
    return value


def _require_task_id(value: object) -> str:
    if not isinstance(value, str) or _TASK_ID.fullmatch(value) is None:
        raise ObjectBongardTurnJournalError("task ID is outside official grammar")
    return value


def _require_turn_kind(value: object) -> str:
    if not isinstance(value, str) or _TURN_KIND.fullmatch(value) is None:
        raise ObjectBongardTurnJournalError("turn kind is not a bounded identifier")
    return value


def _canonical_mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) for key in value
    ):
        raise ObjectBongardTurnJournalError(f"{label} must be a JSON object")
    try:
        encoded = canonical_json(dict(value))
        decoded = json.loads(encoded.decode("utf-8"))
    except (TypeError, ValueError, UnicodeError, json.JSONDecodeError) as exc:
        raise ObjectBongardTurnJournalError(
            f"{label} is not canonical finite JSON"
        ) from exc
    if not isinstance(decoded, dict):
        raise ObjectBongardTurnJournalError(f"{label} must be a JSON object")
    return decoded


def _freeze_prompt(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ObjectBongardTurnJournalError("expected prompt must be nonempty text")
    try:
        encoded = value.encode("utf-8", errors="strict")
    except UnicodeError as exc:
        raise ObjectBongardTurnJournalError("expected prompt is not UTF-8") from exc
    if len(encoded) > MAX_PROMPT_BYTES:
        raise ObjectBongardTurnJournalError("expected prompt is oversized")
    return value


def _freeze_schema(value: object) -> dict[str, Any]:
    frozen = _canonical_mapping(value, "output schema")
    try:
        validate_codex_strict_output_schema(frozen)
    except Exception as exc:
        raise ObjectBongardTurnJournalError(
            "output schema is not a strict Codex schema"
        ) from exc
    return frozen


def _record(content: Mapping[str, Any]) -> dict[str, Any]:
    body = _canonical_mapping(content, "journal record")
    return {**body, "record_digest": _address(body)}


def _validate_record(
    value: object,
    *,
    schema: str,
    fields: set[str],
    label: str,
) -> dict[str, Any]:
    raw = _canonical_mapping(value, label)
    if set(raw) != fields | {"record_digest"} or raw.get("schema") != schema:
        raise ObjectBongardTurnJournalError(f"{label} fields differ")
    _require_digest(raw.get("record_digest"), f"{label} digest")
    body = {key: item for key, item in raw.items() if key != "record_digest"}
    if raw["record_digest"] != _address(body):
        raise ObjectBongardTurnJournalError(f"{label} digest differs")
    return raw


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_once(path: Path, value: Mapping[str, Any]) -> None:
    payload = canonical_json(dict(value)) + b"\n"
    if not 0 < len(payload) <= MAX_RECORD_BYTES:
        raise ObjectBongardTurnJournalError("journal record is oversized")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise ObjectBongardTurnJournalError("journal write made no progress")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent)


def _read_canonical(path: Path, label: str) -> dict[str, Any]:
    try:
        before = os.lstat(path)
    except OSError as exc:
        raise ObjectBongardTurnJournalError(f"{label} is missing") from exc
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or not 0 < before.st_size <= MAX_RECORD_BYTES
    ):
        raise ObjectBongardTurnJournalError(f"{label} is not a bounded file")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened = os.fstat(descriptor)
        identity = (
            opened.st_dev,
            opened.st_ino,
            opened.st_size,
            opened.st_mtime_ns,
            opened.st_ctime_ns,
        )
        if opened.st_nlink != 1 or identity != (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ):
            raise ObjectBongardTurnJournalError(f"{label} changed while opening")
        chunks: list[bytes] = []
        remaining = opened.st_size
        while remaining:
            block = os.read(descriptor, min(remaining, 1_048_576))
            if not block:
                raise ObjectBongardTurnJournalError(f"{label} was truncated")
            chunks.append(block)
            remaining -= len(block)
        after = os.fstat(descriptor)
        if after.st_nlink != 1 or (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ) != identity:
            raise ObjectBongardTurnJournalError(f"{label} changed while reading")
    finally:
        os.close(descriptor)
    payload = b"".join(chunks)
    if not payload.endswith(b"\n") or payload.endswith(b"\n\n"):
        raise ObjectBongardTurnJournalError(f"{label} encoding differs")
    try:
        decoded = json.loads(payload[:-1].decode("utf-8", errors="strict"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ObjectBongardTurnJournalError(f"{label} is malformed JSON") from exc
    if not isinstance(decoded, dict) or canonical_json(decoded) + b"\n" != payload:
        raise ObjectBongardTurnJournalError(f"{label} is not canonical JSON")
    return decoded


def _path_present(path: Path) -> bool:
    return os.path.lexists(path)


def _read_exact_file(path: object, label: str) -> bytes:
    if not isinstance(path, str) or not os.path.isabs(path):
        raise ObjectBongardTurnJournalError(f"{label} path must be absolute")
    try:
        before = os.lstat(path)
    except OSError as exc:
        raise ObjectBongardTurnJournalError(f"{label} is unavailable") from exc
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or not 0 < before.st_size <= MAX_IMAGE_BYTES
    ):
        raise ObjectBongardTurnJournalError(
            f"{label} must be a bounded singly-linked regular file"
        )
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened = os.fstat(descriptor)
        identity = (
            opened.st_dev,
            opened.st_ino,
            opened.st_size,
            opened.st_mtime_ns,
            opened.st_ctime_ns,
        )
        if opened.st_nlink != 1 or identity != (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ):
            raise ObjectBongardTurnJournalError(f"{label} changed while opening")
        chunks: list[bytes] = []
        remaining = opened.st_size
        while remaining:
            block = os.read(descriptor, min(remaining, 1_048_576))
            if not block:
                raise ObjectBongardTurnJournalError(f"{label} was truncated")
            chunks.append(block)
            remaining -= len(block)
        after = os.fstat(descriptor)
        if after.st_nlink != 1 or (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ) != identity:
            raise ObjectBongardTurnJournalError(f"{label} changed while reading")
    finally:
        os.close(descriptor)
    return b"".join(chunks)


def _receipt_from_data(value: object) -> CodexReceipt:
    expected = set(CodexReceipt.__dataclass_fields__)
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ObjectBongardTurnJournalError("stored Codex receipt fields differ")
    raw = dict(value)
    if not isinstance(raw.get("event_types"), list) or not isinstance(
        raw.get("item_types"), list
    ):
        raise ObjectBongardTurnJournalError(
            "stored Codex receipt event summaries differ"
        )
    try:
        validate_codex_receipt(raw)
        receipt = CodexReceipt(
            **{
                **raw,
                "event_types": tuple(raw["event_types"]),
                "item_types": tuple(raw["item_types"]),
            }
        )
    except Exception as exc:
        raise ObjectBongardTurnJournalError("stored Codex receipt is invalid") from exc
    if receipt.to_dict() != raw:
        raise ObjectBongardTurnJournalError("stored Codex receipt is not canonical")
    return receipt


@dataclass(frozen=True, slots=True)
class ObjectBongardTurnRuntime:
    """Exact runtime/preflight envelope expected by one journaled turn."""

    model: str
    reasoning_effort: str
    minutes: int
    verbose: bool
    executable: str
    cloud_policy_cache_snapshot: CloudPolicyCacheSnapshot | None
    model_catalog_snapshot: CodexModelCatalogSnapshot
    expected_launcher_digest: str
    no_tools_attestation: CodexNoToolsAttestation
    transport_source_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.model, str) or not self.model:
            raise ObjectBongardTurnJournalError("runtime model is invalid")
        if not isinstance(self.reasoning_effort, str) or not self.reasoning_effort:
            raise ObjectBongardTurnJournalError("runtime reasoning effort is invalid")
        if (
            isinstance(self.minutes, bool)
            or not isinstance(self.minutes, int)
            or not 1 <= self.minutes <= 120
        ):
            raise ObjectBongardTurnJournalError("runtime minutes must lie in 1..120")
        if not isinstance(self.verbose, bool):
            raise ObjectBongardTurnJournalError("runtime verbosity is invalid")
        if not isinstance(self.executable, str) or not self.executable:
            raise ObjectBongardTurnJournalError("runtime executable is invalid")
        if self.cloud_policy_cache_snapshot is not None and not isinstance(
            self.cloud_policy_cache_snapshot, CloudPolicyCacheSnapshot
        ):
            raise ObjectBongardTurnJournalError(
                "runtime policy-cache snapshot type differs"
            )
        if not isinstance(self.model_catalog_snapshot, CodexModelCatalogSnapshot):
            raise ObjectBongardTurnJournalError(
                "runtime model catalog snapshot type differs"
            )
        if not isinstance(self.no_tools_attestation, CodexNoToolsAttestation):
            raise ObjectBongardTurnJournalError(
                "runtime no-tools attestation type differs"
            )
        _require_raw_digest(self.expected_launcher_digest, "launcher digest")
        _require_raw_digest(self.transport_source_digest, "transport source digest")
        policy_binding = self.policy_cache_binding
        if (
            self.expected_launcher_digest
            != self.no_tools_attestation.launcher_digest
            or self.model_catalog_snapshot.raw_digest
            != self.no_tools_attestation.model_catalog_digest
            or policy_binding
            != self.no_tools_attestation.cloud_config_bundle_cache_binding
        ):
            raise ObjectBongardTurnJournalError(
                "runtime objects differ from no-tools attestation"
            )

    @property
    def policy_cache_binding(self) -> str:
        snapshot = self.cloud_policy_cache_snapshot
        return "absent" if snapshot is None else snapshot.binding

    @property
    def binding(self) -> dict[str, object]:
        return {
            "model": self.model,
            "reasoning_effort": self.reasoning_effort,
            "minutes": self.minutes,
            "verbose": self.verbose,
            "executable": self.executable,
            "cloud_policy_cache_snapshot_present": (
                self.cloud_policy_cache_snapshot is not None
            ),
            "cloud_policy_cache_binding": self.policy_cache_binding,
            "model_catalog_raw_digest": self.model_catalog_snapshot.raw_digest,
            "model_catalog_canonical_digest": (
                self.model_catalog_snapshot.canonical_digest
            ),
            "expected_launcher_digest": self.expected_launcher_digest,
            "no_tools_attestation_digest": (
                self.no_tools_attestation.attestation_digest
            ),
            "transport_source_digest": self.transport_source_digest,
        }

    def assert_invocation(
        self,
        *,
        model: object,
        reasoning_effort: object,
        minutes: object,
        verbose: object,
        executable: object,
        cloud_policy_cache_snapshot: object,
        model_catalog_snapshot: object,
        tool_surface_attestation: object,
        expected_launcher_digest: object,
        expected_tool_surface_attestation_digest: object,
    ) -> None:
        expected = (
            self.model,
            self.reasoning_effort,
            self.minutes,
            self.verbose,
            self.executable,
            self.cloud_policy_cache_snapshot,
            self.model_catalog_snapshot,
            self.no_tools_attestation,
            self.expected_launcher_digest,
            self.no_tools_attestation.attestation_digest,
        )
        actual = (
            model,
            reasoning_effort,
            minutes,
            verbose,
            executable,
            cloud_policy_cache_snapshot,
            model_catalog_snapshot,
            tool_surface_attestation,
            expected_launcher_digest,
            expected_tool_surface_attestation_digest,
        )
        if any(type(got) is not type(want) or got != want for got, want in zip(
            actual, expected, strict=True
        )):
            raise ObjectBongardTurnJournalError(
                "Codex turn model/runtime invocation differs"
            )

    def validate_receipt(self, receipt: CodexReceipt) -> None:
        if (
            receipt.requested_model != self.model
            or receipt.requested_reasoning_effort != self.reasoning_effort
            or receipt.codex_launcher_digest != self.expected_launcher_digest
            or receipt.cloud_config_bundle_cache_binding
            != self.policy_cache_binding
            or receipt.model_catalog_digest
            != self.model_catalog_snapshot.raw_digest
            or receipt.tool_surface_attestation_digest
            != self.no_tools_attestation.attestation_digest
        ):
            raise ObjectBongardTurnJournalError(
                "stored Codex receipt runtime binding differs"
            )


@dataclass(frozen=True, slots=True)
class ObjectBongardTurnJournalSummary:
    manifest_digest: str
    turn_key: str
    terminal_status: str
    claim_digest: str | None
    result_digest: str | None
    outcome_digest: str | None
    record_digest: str

    def to_data(self) -> dict[str, object]:
        return {
            "schema": TURN_JOURNAL_SUMMARY_SCHEMA,
            "manifest_digest": self.manifest_digest,
            "turn_key": self.turn_key,
            "terminal_status": self.terminal_status,
            "claim_digest": self.claim_digest,
            "result_digest": self.result_digest,
            "outcome_digest": self.outcome_digest,
            "record_digest": self.record_digest,
            **_authority_data(),
        }


class _ObjectBongardTurnJournalBase:
    modality: str

    def __init__(
        self,
        journal_directory: str | os.PathLike[str],
        *,
        authorization_digest: str,
        execution_precommit_digest: str,
        task_id: str,
        turn_kind: str,
        expected_prompt: str,
        expected_output_schema: Mapping[str, Any],
        runtime: ObjectBongardTurnRuntime,
        expected_images: Sequence[tuple[str, bytes]],
        underlying_transport: Callable[..., CodexStructuredResult],
    ) -> None:
        if not callable(underlying_transport):
            raise TypeError("underlying transport must be callable")
        if not isinstance(runtime, ObjectBongardTurnRuntime):
            raise TypeError("runtime must be ObjectBongardTurnRuntime")
        self.authorization_digest = _require_digest(
            authorization_digest, "authorization digest"
        )
        self.execution_precommit_digest = _require_digest(
            execution_precommit_digest, "execution precommit digest"
        )
        self.task_id = _require_task_id(task_id)
        self.turn_kind = _require_turn_kind(turn_kind)
        self.expected_prompt = _freeze_prompt(expected_prompt)
        self.expected_output_schema = _freeze_schema(expected_output_schema)
        self.runtime = runtime
        self._expected_images = self._freeze_images(expected_images)
        self._underlying_transport = underlying_transport

        self.directory = Path(journal_directory)
        self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        if not self.directory.is_dir() or self.directory.is_symlink():
            raise ObjectBongardTurnJournalError(
                "turn journal directory must be a real directory"
            )
        self.manifest_path = self.directory / "manifest.json"
        self.claim_path = self.directory / "claim.json"
        self.result_path = self.directory / "result.json"
        self.outcome_path = self.directory / "outcome.json"
        self._manifest = self._build_manifest()
        self._persist_or_verify_manifest()
        self.attempted_call_count = 0
        self.fresh_call_count = 0
        self.reused_call_count = 0
        self.refused_call_count = 0

    def _freeze_images(
        self, values: Sequence[tuple[str, bytes]]
    ) -> tuple[tuple[str, bytes], ...]:
        if isinstance(values, (str, bytes)):
            raise ObjectBongardTurnJournalError("expected images must be a sequence")
        rows = tuple(values)
        if self.modality == TEXT_MODALITY:
            if rows:
                raise ObjectBongardTurnJournalError(
                    "text turn must commit exactly zero images"
                )
            return ()
        if not 1 <= len(rows) <= MAX_NAMED_IMAGES:
            raise ObjectBongardTurnJournalError(
                "named-image turn image count is outside bounds"
            )
        if any(
            not isinstance(row, tuple)
            or len(row) != 2
            or not isinstance(row[0], str)
            or _IMAGE_NAME.fullmatch(row[0]) is None
            or not isinstance(row[1], bytes)
            or not 0 < len(row[1]) <= MAX_IMAGE_BYTES
            for row in rows
        ):
            raise ObjectBongardTurnJournalError(
                "named-image turn contains an invalid name or byte snapshot"
            )
        names = tuple(row[0] for row in rows)
        if len(names) != len(set(names)):
            raise ObjectBongardTurnJournalError("named-image names must be unique")
        return rows

    def _build_manifest(self) -> dict[str, Any]:
        return _record(
            {
                "schema": TURN_JOURNAL_MANIFEST_SCHEMA,
                "protocol_id": TURN_JOURNAL_PROTOCOL_ID,
                "modality": self.modality,
                "authorization_digest": self.authorization_digest,
                "execution_precommit_digest": self.execution_precommit_digest,
                "task_id": self.task_id,
                "turn_kind": self.turn_kind,
                "prompt": self.expected_prompt,
                "prompt_sha256": hashlib.sha256(
                    self.expected_prompt.encode("utf-8")
                ).hexdigest(),
                "output_schema": self.expected_output_schema,
                "output_schema_digest": _address(self.expected_output_schema),
                "named_images": [
                    {
                        "name": name,
                        "byte_count": len(data),
                        "sha256": hashlib.sha256(data).hexdigest(),
                    }
                    for name, data in self._expected_images
                ],
                "runtime_binding": self.runtime.binding,
                "journal_source_digest": object_bongard_turn_journal_source_digest(),
                "exclusive_claim_fsynced_before_transport": True,
                "complete_result_fsynced_before_terminal": True,
                "terminal_replay_calls_model": False,
                "nonterminal_claim_policy": "refuse-without-transport",
                **_authority_data(),
            }
        )

    def _persist_or_verify_manifest(self) -> None:
        try:
            _write_once(self.manifest_path, self._manifest)
        except FileExistsError:
            persisted = _read_canonical(self.manifest_path, "turn manifest")
            if persisted != self._manifest:
                raise ObjectBongardTurnJournalError(
                    "turn manifest differs from exact input/runtime binding"
                )
        reloaded = _validate_record(
            _read_canonical(self.manifest_path, "turn manifest"),
            schema=TURN_JOURNAL_MANIFEST_SCHEMA,
            fields=set(self._manifest) - {"record_digest"},
            label="turn manifest",
        )
        if reloaded != self._manifest:
            raise ObjectBongardTurnJournalError("turn manifest replay differs")

    @property
    def manifest_digest(self) -> str:
        return self._manifest["record_digest"]

    @property
    def turn_key(self) -> str:
        return _address(
            {
                "schema": "gkm.bongard-codex-turn-key.v1",
                "authorization_digest": self.authorization_digest,
                "execution_precommit_digest": self.execution_precommit_digest,
                "task_id": self.task_id,
                "turn_kind": self.turn_kind,
                "modality": self.modality,
                "manifest_digest": self.manifest_digest,
            }
        )

    def _expected_claim(self) -> dict[str, Any]:
        return _record(
            {
                "schema": TURN_CLAIM_SCHEMA,
                "turn_key": self.turn_key,
                "manifest_digest": self.manifest_digest,
                "authorization_digest": self.authorization_digest,
                "execution_precommit_digest": self.execution_precommit_digest,
                "task_id": self.task_id,
                "turn_kind": self.turn_kind,
                "modality": self.modality,
                "exclusive_create_and_fsync_before_transport": True,
            }
        )

    def _load_claim(self) -> dict[str, Any]:
        expected = self._expected_claim()
        claim = _validate_record(
            _read_canonical(self.claim_path, "turn claim"),
            schema=TURN_CLAIM_SCHEMA,
            fields=set(expected) - {"record_digest"},
            label="turn claim",
        )
        if claim != expected:
            raise ObjectBongardTurnJournalError("turn claim binding differs")
        return claim

    def _load_outcome(
        self, *, claim: Mapping[str, Any]
    ) -> dict[str, Any] | None:
        if not _path_present(self.outcome_path):
            return None
        outcome = _validate_record(
            _read_canonical(self.outcome_path, "turn terminal outcome"),
            schema=TURN_OUTCOME_SCHEMA,
            fields={
                "schema",
                "turn_key",
                "claim_digest",
                "manifest_digest",
                "terminal_status",
                "result_digest",
                "terminal",
                "result_persisted_and_fsynced_before_terminal",
            },
            label="turn terminal outcome",
        )
        if (
            outcome["turn_key"] != self.turn_key
            or outcome["claim_digest"] != claim["record_digest"]
            or outcome["manifest_digest"] != self.manifest_digest
            or outcome["terminal_status"] not in {"success", "failure"}
            or outcome["terminal"] is not True
            or outcome["result_persisted_and_fsynced_before_terminal"] is not True
        ):
            raise ObjectBongardTurnJournalError("turn outcome lineage differs")
        return outcome

    def _load_result(
        self,
        *,
        claim: Mapping[str, Any],
        outcome: Mapping[str, Any],
    ) -> dict[str, Any]:
        result = _validate_record(
            _read_canonical(self.result_path, "turn durable result"),
            schema=TURN_RESULT_SCHEMA,
            fields={
                "schema",
                "turn_key",
                "claim_digest",
                "manifest_digest",
                "status",
                "codex_structured_result",
                "payload_digest",
                "receipt_digest",
                "failure_code",
                "source_exception_type",
            },
            label="turn durable result",
        )
        if (
            result["turn_key"] != self.turn_key
            or result["claim_digest"] != claim["record_digest"]
            or result["manifest_digest"] != self.manifest_digest
            or result["record_digest"] != outcome["result_digest"]
            or result["status"] != outcome["terminal_status"]
        ):
            raise ObjectBongardTurnJournalError("turn result lineage differs")
        if result["status"] == "success":
            structured = _canonical_mapping(
                result["codex_structured_result"], "stored structured result"
            )
            if set(structured) != {"payload", "receipt"}:
                raise ObjectBongardTurnJournalError(
                    "stored structured result fields differ"
                )
            payload = _canonical_mapping(structured["payload"], "stored payload")
            receipt = _receipt_from_data(structured["receipt"])
            self.runtime.validate_receipt(receipt)
            if (
                result["payload_digest"] != _address(payload)
                or result["receipt_digest"] != receipt.receipt_digest
                or receipt.structured_output_digest != canonical_digest(payload)
                or result["failure_code"] is not None
                or result["source_exception_type"] is not None
            ):
                raise ObjectBongardTurnJournalError(
                    "successful turn result bindings differ"
                )
        elif result["status"] == "failure":
            if (
                result["codex_structured_result"] is not None
                or result["payload_digest"] is not None
                or result["receipt_digest"] is not None
                or result["failure_code"] != "physical_codex_turn_failed"
                or not isinstance(result["source_exception_type"], str)
                or _EXCEPTION_TYPE.fullmatch(result["source_exception_type"]) is None
            ):
                raise ObjectBongardTurnJournalError(
                    "failed turn result bindings differ"
                )
        else:
            raise ObjectBongardTurnJournalError("turn result status differs")
        return result

    def _restore_success(self, result: Mapping[str, Any]) -> CodexStructuredResult:
        structured = _canonical_mapping(
            result["codex_structured_result"], "stored structured result"
        )
        payload = _canonical_mapping(structured["payload"], "stored payload")
        receipt = _receipt_from_data(structured["receipt"])
        self.runtime.validate_receipt(receipt)
        return CodexStructuredResult(payload=payload, receipt=receipt)

    def _success_record(
        self,
        *,
        claim: Mapping[str, Any],
        raw_result: CodexStructuredResult,
        validate_envelope: Callable[[CodexStructuredResult], None],
    ) -> dict[str, Any]:
        if not isinstance(raw_result, CodexStructuredResult):
            raise ObjectBongardTurnJournalError(
                "underlying transport returned the wrong result type"
            )
        payload = _canonical_mapping(raw_result.payload, "transport payload")
        if not isinstance(raw_result.receipt, CodexReceipt):
            raise ObjectBongardTurnJournalError(
                "underlying transport returned no full Codex receipt"
            )
        normalized = CodexStructuredResult(payload=payload, receipt=raw_result.receipt)
        self.runtime.validate_receipt(normalized.receipt)
        validate_envelope(normalized)
        return _record(
            {
                "schema": TURN_RESULT_SCHEMA,
                "turn_key": self.turn_key,
                "claim_digest": claim["record_digest"],
                "manifest_digest": self.manifest_digest,
                "status": "success",
                "codex_structured_result": {
                    "payload": payload,
                    "receipt": normalized.receipt.to_dict(),
                },
                "payload_digest": _address(payload),
                "receipt_digest": normalized.receipt.receipt_digest,
                "failure_code": None,
                "source_exception_type": None,
            }
        )

    def _failure_record(
        self, *, claim: Mapping[str, Any], exception: Exception
    ) -> dict[str, Any]:
        source_type = f"{type(exception).__module__}.{type(exception).__qualname__}"
        if _EXCEPTION_TYPE.fullmatch(source_type) is None:
            source_type = "builtins.Exception"
        return _record(
            {
                "schema": TURN_RESULT_SCHEMA,
                "turn_key": self.turn_key,
                "claim_digest": claim["record_digest"],
                "manifest_digest": self.manifest_digest,
                "status": "failure",
                "codex_structured_result": None,
                "payload_digest": None,
                "receipt_digest": None,
                "failure_code": "physical_codex_turn_failed",
                "source_exception_type": source_type,
            }
        )

    def _finish(
        self, *, claim: Mapping[str, Any], result: Mapping[str, Any]
    ) -> dict[str, Any]:
        _write_once(self.result_path, result)
        if _read_canonical(self.result_path, "turn durable result") != dict(result):
            raise ObjectBongardTurnJournalError("durable turn result reload differs")
        outcome = _record(
            {
                "schema": TURN_OUTCOME_SCHEMA,
                "turn_key": self.turn_key,
                "claim_digest": claim["record_digest"],
                "manifest_digest": self.manifest_digest,
                "terminal_status": result["status"],
                "result_digest": result["record_digest"],
                "terminal": True,
                "result_persisted_and_fsynced_before_terminal": True,
            }
        )
        _write_once(self.outcome_path, outcome)
        if _read_canonical(self.outcome_path, "turn terminal outcome") != outcome:
            raise ObjectBongardTurnJournalError("terminal turn reload differs")
        return outcome

    def _execute(
        self,
        *,
        physical_call: Callable[[], CodexStructuredResult],
        validate_envelope: Callable[[CodexStructuredResult], None],
    ) -> CodexStructuredResult:
        allowed = {"manifest.json", "claim.json", "result.json", "outcome.json"}
        actual = {item.name for item in self.directory.iterdir()}
        if not actual <= allowed:
            raise ObjectBongardTurnJournalError(
                "turn journal contains an unexpected record path"
            )
        if not _path_present(self.claim_path) and (
            _path_present(self.result_path) or _path_present(self.outcome_path)
        ):
            raise ObjectBongardTurnJournalError(
                "turn result exists without an admission claim"
            )
        self.attempted_call_count += 1
        expected_claim = self._expected_claim()
        try:
            _write_once(self.claim_path, expected_claim)
        except FileExistsError:
            claim = self._load_claim()
            outcome = self._load_outcome(claim=claim)
            if outcome is None:
                self.refused_call_count += 1
                raise ObjectBongardTurnNonterminalClaim(
                    "physical turn has a preexisting nonterminal claim; "
                    "transport rerun is forbidden"
                )
            result = self._load_result(claim=claim, outcome=outcome)
            self.reused_call_count += 1
            if result["status"] == "failure":
                raise ObjectBongardTurnCallFailed(
                    turn_key=self.turn_key,
                    failure_digest=result["record_digest"],
                )
            restored = self._restore_success(result)
            validate_envelope(restored)
            return restored

        claim = self._load_claim()
        if _path_present(self.result_path) or _path_present(self.outcome_path):
            raise ObjectBongardTurnJournalError(
                "fresh turn claim collided with a preexisting result"
            )
        self.fresh_call_count += 1
        try:
            raw_result = physical_call()
            result = self._success_record(
                claim=claim,
                raw_result=raw_result,
                validate_envelope=validate_envelope,
            )
        except Exception as exc:
            failure = self._failure_record(claim=claim, exception=exc)
            self._finish(claim=claim, result=failure)
            raise ObjectBongardTurnCallFailed(
                turn_key=self.turn_key, failure_digest=failure["record_digest"]
            ) from None

        outcome = self._finish(claim=claim, result=result)
        durable = self._load_result(claim=claim, outcome=outcome)
        restored = self._restore_success(durable)
        validate_envelope(restored)
        return restored

    def _cold_validate_success(self, result: Mapping[str, Any]) -> None:
        raise NotImplementedError

    def verify(self) -> ObjectBongardTurnJournalSummary:
        """Cold-verify the complete one-turn journal without a model call."""

        self._persist_or_verify_manifest()
        allowed = {"manifest.json", "claim.json", "result.json", "outcome.json"}
        actual = {item.name for item in self.directory.iterdir()}
        if not actual <= allowed:
            raise ObjectBongardTurnJournalError(
                "turn journal contains an unexpected record path"
            )
        if not _path_present(self.claim_path):
            if _path_present(self.result_path) or _path_present(self.outcome_path):
                raise ObjectBongardTurnJournalError(
                    "turn result exists without an admission claim"
                )
            return self._summary(
                status="unclaimed", claim=None, result=None, outcome=None
            )
        claim = self._load_claim()
        outcome = self._load_outcome(claim=claim)
        if outcome is None:
            if _path_present(self.result_path):
                raise ObjectBongardTurnJournalError(
                    "nonterminal turn claim has an uncommitted result"
                )
            raise ObjectBongardTurnNonterminalClaim(
                "turn journal contains a stranded nonterminal claim"
            )
        result = self._load_result(claim=claim, outcome=outcome)
        if result["status"] == "success":
            self._cold_validate_success(result)
        return self._summary(
            status=result["status"], claim=claim, result=result, outcome=outcome
        )

    def _summary(
        self,
        *,
        status: str,
        claim: Mapping[str, Any] | None,
        result: Mapping[str, Any] | None,
        outcome: Mapping[str, Any] | None,
    ) -> ObjectBongardTurnJournalSummary:
        content = {
            "schema": TURN_JOURNAL_SUMMARY_SCHEMA,
            "manifest_digest": self.manifest_digest,
            "turn_key": self.turn_key,
            "terminal_status": status,
            "claim_digest": None if claim is None else claim["record_digest"],
            "result_digest": None if result is None else result["record_digest"],
            "outcome_digest": (
                None if outcome is None else outcome["record_digest"]
            ),
            **_authority_data(),
        }
        return ObjectBongardTurnJournalSummary(
            manifest_digest=self.manifest_digest,
            turn_key=self.turn_key,
            terminal_status=status,
            claim_digest=content["claim_digest"],  # type: ignore[arg-type]
            result_digest=content["result_digest"],  # type: ignore[arg-type]
            outcome_digest=content["outcome_digest"],  # type: ignore[arg-type]
            record_digest=_address(content),
        )


class ObjectBongardNamedImageTurnJournalTransport(_ObjectBongardTurnJournalBase):
    """One task/turn-bound exactly-once named-image transport."""

    modality = NAMED_IMAGE_MODALITY

    def __init__(
        self,
        journal_directory: str | os.PathLike[str],
        *,
        authorization_digest: str,
        execution_precommit_digest: str,
        task_id: str,
        turn_kind: str,
        expected_prompt: str,
        expected_images: Sequence[tuple[str, bytes]],
        expected_output_schema: Mapping[str, Any],
        runtime: ObjectBongardTurnRuntime,
        underlying_transport: Callable[..., CodexStructuredResult],
    ) -> None:
        super().__init__(
            journal_directory,
            authorization_digest=authorization_digest,
            execution_precommit_digest=execution_precommit_digest,
            task_id=task_id,
            turn_kind=turn_kind,
            expected_prompt=expected_prompt,
            expected_output_schema=expected_output_schema,
            runtime=runtime,
            expected_images=expected_images,
            underlying_transport=underlying_transport,
        )

    def _validate_inputs(
        self,
        prompt: object,
        image_paths: object,
        image_names: object,
        output_schema: object,
    ) -> tuple[str, ...]:
        if prompt != self.expected_prompt:
            raise ObjectBongardTurnJournalError(
                "named-image turn prompt differs from commitment"
            )
        if _freeze_schema(output_schema) != self.expected_output_schema:
            raise ObjectBongardTurnJournalError(
                "named-image turn output schema differs from commitment"
            )
        if (
            isinstance(image_paths, (str, bytes))
            or not isinstance(image_paths, Sequence)
            or isinstance(image_names, (str, bytes))
            or not isinstance(image_names, Sequence)
        ):
            raise ObjectBongardTurnJournalError(
                "named-image turn inputs must be finite sequences"
            )
        paths = tuple(image_paths)
        names = tuple(image_names)
        expected_names = tuple(name for name, _ in self._expected_images)
        if names != expected_names or len(paths) != len(expected_names):
            raise ObjectBongardTurnJournalError(
                "named-image turn names or order differ from commitment"
            )
        for index, (path, (_, expected)) in enumerate(
            zip(paths, self._expected_images, strict=True)
        ):
            if _read_exact_file(path, f"named image {index}") != expected:
                raise ObjectBongardTurnJournalError(
                    "named-image turn bytes differ from commitment"
                )
        return paths  # type: ignore[return-value]

    def _validator(
        self, paths: Sequence[str]
    ) -> Callable[[CodexStructuredResult], None]:
        names = tuple(name for name, _ in self._expected_images)

        def validate(result: CodexStructuredResult) -> None:
            self.runtime.validate_receipt(result.receipt)
            try:
                validate_codex_named_image_receipt(
                    result.receipt,
                    self.expected_prompt,
                    paths,
                    names,
                    self.expected_output_schema,
                    result.payload,
                )
            except Exception as exc:
                raise ObjectBongardTurnJournalError(
                    "named-image receipt fails exact-input replay"
                ) from exc
            for index, (path, (_, expected)) in enumerate(
                zip(paths, self._expected_images, strict=True)
            ):
                if _read_exact_file(path, f"named image {index}") != expected:
                    raise ObjectBongardTurnJournalError(
                        "named-image bytes changed during turn"
                    )

        return validate

    def __call__(
        self,
        task: str,
        image_png_paths: Sequence[str],
        image_names: Sequence[str],
        output_schema: Mapping[str, Any],
        model: str = DEFAULT_CODEX_MODEL,
        reasoning_effort: str = DEFAULT_REASONING_EFFORT,
        minutes: int = 15,
        verbose: bool = False,
        executable: str = "codex",
        cloud_policy_cache_snapshot: CloudPolicyCacheSnapshot | None = None,
        model_catalog_snapshot: CodexModelCatalogSnapshot | None = None,
        tool_surface_attestation: Any | None = None,
        expected_launcher_digest: str | None = None,
        expected_tool_surface_attestation_digest: str | None = None,
    ) -> CodexStructuredResult:
        paths = self._validate_inputs(
            task, image_png_paths, image_names, output_schema
        )
        self.runtime.assert_invocation(
            model=model,
            reasoning_effort=reasoning_effort,
            minutes=minutes,
            verbose=verbose,
            executable=executable,
            cloud_policy_cache_snapshot=cloud_policy_cache_snapshot,
            model_catalog_snapshot=model_catalog_snapshot,
            tool_surface_attestation=tool_surface_attestation,
            expected_launcher_digest=expected_launcher_digest,
            expected_tool_surface_attestation_digest=(
                expected_tool_surface_attestation_digest
            ),
        )
        validator = self._validator(paths)
        return self._execute(
            physical_call=lambda: self._underlying_transport(
                task,
                image_png_paths,
                image_names,
                output_schema,
                model=model,
                reasoning_effort=reasoning_effort,
                minutes=minutes,
                verbose=verbose,
                executable=executable,
                cloud_policy_cache_snapshot=cloud_policy_cache_snapshot,
                model_catalog_snapshot=model_catalog_snapshot,
                tool_surface_attestation=tool_surface_attestation,
                expected_launcher_digest=expected_launcher_digest,
                expected_tool_surface_attestation_digest=(
                    expected_tool_surface_attestation_digest
                ),
            ),
            validate_envelope=validator,
        )

    def _cold_validate_success(self, result: Mapping[str, Any]) -> None:
        restored = self._restore_success(result)
        with tempfile.TemporaryDirectory(prefix="bongard-turn-cold-") as raw:
            paths: list[str] = []
            for name, data in self._expected_images:
                path = Path(raw) / name
                path.write_bytes(data)
                paths.append(str(path.resolve()))
            self._validator(tuple(paths))(restored)


class ObjectBongardTextTurnJournalTransport(_ObjectBongardTurnJournalBase):
    """One task/turn-bound exactly-once zero-image text transport."""

    modality = TEXT_MODALITY

    def __init__(
        self,
        journal_directory: str | os.PathLike[str],
        *,
        authorization_digest: str,
        execution_precommit_digest: str,
        task_id: str,
        turn_kind: str,
        expected_prompt: str,
        expected_output_schema: Mapping[str, Any],
        runtime: ObjectBongardTurnRuntime,
        underlying_transport: Callable[..., CodexStructuredResult],
    ) -> None:
        super().__init__(
            journal_directory,
            authorization_digest=authorization_digest,
            execution_precommit_digest=execution_precommit_digest,
            task_id=task_id,
            turn_kind=turn_kind,
            expected_prompt=expected_prompt,
            expected_output_schema=expected_output_schema,
            runtime=runtime,
            expected_images=(),
            underlying_transport=underlying_transport,
        )

    def _validator(self) -> Callable[[CodexStructuredResult], None]:
        def validate(result: CodexStructuredResult) -> None:
            self.runtime.validate_receipt(result.receipt)
            try:
                validate_codex_text_receipt(
                    result.receipt.to_dict(),
                    self.expected_prompt,
                    self.expected_output_schema,
                )
            except Exception as exc:
                raise ObjectBongardTurnJournalError(
                    "text receipt fails exact-input replay"
                ) from exc
            if result.receipt.structured_output_digest != canonical_digest(
                result.payload
            ):
                raise ObjectBongardTurnJournalError(
                    "text receipt does not bind the stored payload"
                )

        return validate

    def __call__(
        self,
        prompt: str,
        output_schema: Mapping[str, Any],
        model: str = DEFAULT_CODEX_MODEL,
        reasoning_effort: str = DEFAULT_REASONING_EFFORT,
        minutes: int = 15,
        verbose: bool = False,
        executable: str = "codex",
        cloud_policy_cache_snapshot: CloudPolicyCacheSnapshot | None = None,
        model_catalog_snapshot: CodexModelCatalogSnapshot | None = None,
        tool_surface_attestation: Any | None = None,
        expected_launcher_digest: str | None = None,
        expected_tool_surface_attestation_digest: str | None = None,
    ) -> CodexStructuredResult:
        if prompt != self.expected_prompt:
            raise ObjectBongardTurnJournalError(
                "text turn prompt differs from commitment"
            )
        if _freeze_schema(output_schema) != self.expected_output_schema:
            raise ObjectBongardTurnJournalError(
                "text turn output schema differs from commitment"
            )
        self.runtime.assert_invocation(
            model=model,
            reasoning_effort=reasoning_effort,
            minutes=minutes,
            verbose=verbose,
            executable=executable,
            cloud_policy_cache_snapshot=cloud_policy_cache_snapshot,
            model_catalog_snapshot=model_catalog_snapshot,
            tool_surface_attestation=tool_surface_attestation,
            expected_launcher_digest=expected_launcher_digest,
            expected_tool_surface_attestation_digest=(
                expected_tool_surface_attestation_digest
            ),
        )
        validator = self._validator()
        return self._execute(
            physical_call=lambda: self._underlying_transport(
                prompt,
                output_schema,
                model=model,
                reasoning_effort=reasoning_effort,
                minutes=minutes,
                verbose=verbose,
                executable=executable,
                cloud_policy_cache_snapshot=cloud_policy_cache_snapshot,
                model_catalog_snapshot=model_catalog_snapshot,
                tool_surface_attestation=tool_surface_attestation,
                expected_launcher_digest=expected_launcher_digest,
                expected_tool_surface_attestation_digest=(
                    expected_tool_surface_attestation_digest
                ),
            ),
            validate_envelope=validator,
        )

    def _cold_validate_success(self, result: Mapping[str, Any]) -> None:
        self._validator()(self._restore_success(result))


def verify_object_bongard_turn_journal(
    journal: (
        ObjectBongardNamedImageTurnJournalTransport
        | ObjectBongardTextTurnJournalTransport
    ),
) -> ObjectBongardTurnJournalSummary:
    if not isinstance(
        journal,
        (
            ObjectBongardNamedImageTurnJournalTransport,
            ObjectBongardTextTurnJournalTransport,
        ),
    ):
        raise TypeError("journal must be a typed Bongard turn journal")
    return journal.verify()


__all__ = (
    "NAMED_IMAGE_MODALITY",
    "TEXT_MODALITY",
    "TURN_CLAIM_SCHEMA",
    "TURN_JOURNAL_MANIFEST_SCHEMA",
    "TURN_JOURNAL_PROTOCOL_ID",
    "TURN_JOURNAL_SUMMARY_SCHEMA",
    "TURN_OUTCOME_SCHEMA",
    "TURN_RESULT_SCHEMA",
    "ObjectBongardNamedImageTurnJournalTransport",
    "ObjectBongardTextTurnJournalTransport",
    "ObjectBongardTurnCallFailed",
    "ObjectBongardTurnJournalError",
    "ObjectBongardTurnJournalSummary",
    "ObjectBongardTurnNonterminalClaim",
    "ObjectBongardTurnRuntime",
    "object_bongard_turn_journal_source_digest",
    "verify_object_bongard_turn_journal",
)
