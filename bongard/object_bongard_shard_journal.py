"""Durable one-shot admission for physical object-observer shard calls.

Each instance is bound to one panel, its pixel-derived hypothesis packet, and
the packet's only canonical feature-shard plan.  The callable wrapper accepts
the same arguments as ``NamedImageTransport``.  It checks the complete model
envelope before creating an exclusive claim, persists the payload and full
Codex receipt before its terminal marker, and replays terminal records without
calling the underlying transport.

This is deliberately a Python-only persistence boundary.  It neither imports
Lean nor delegates any identity or replay decision to an external checker.
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
from bongard.prototype_object_hypotheses import (
    ObjectHypothesisPacket,
    verify_object_hypothesis_packet,
)
from bongard.prototype_object_observer_protocol import (
    ObjectFeatureShardPlan,
    ObjectFeatureShardSpec,
    plan_prototype_object_feature_shards,
    prototype_object_feature_output_schema,
    prototype_object_feature_shard_prompt,
    verify_prototype_object_feature_shard_plan,
)
from bongard.prototype_object_scene_observer import _receipt_from_data
from bongard.python_predicate_authority import PYTHON_PREDICATE_AUTHORITY_ID
from bongard.transport import (
    CloudPolicyCacheSnapshot,
    CodexModelCatalogSnapshot,
    CodexReceipt,
    CodexStructuredResult,
    validate_codex_named_image_receipt,
    validate_codex_strict_output_schema,
)


JOURNAL_MANIFEST_SCHEMA = "gkm.bongard-object-shard-journal-manifest.v1"
SHARD_CLAIM_SCHEMA = "gkm.bongard-object-physical-shard-claim.v1"
SHARD_RESULT_SCHEMA = "gkm.bongard-object-physical-shard-result.v1"
SHARD_OUTCOME_SCHEMA = "gkm.bongard-object-physical-shard-outcome.v1"
JOURNAL_PROTOCOL_ID = (
    "bongard.object-observer/physical-shard-exclusive-journal-v1"
)
MAX_RECORD_BYTES = 16 * 1024 * 1024
MAX_IMAGE_BYTES = 8 * 1024 * 1024

_DIGEST = re.compile(r"(?:sha256:)?[0-9a-f]{64}\Z")
_PANEL_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/-]{0,511}\Z")
_EXCEPTION_TYPE = re.compile(r"[A-Za-z_][A-Za-z0-9_.]{0,255}\Z")


class ObjectBongardShardJournalError(RuntimeError):
    """A journal, invocation, or canonical replay invariant failed."""


class ObjectBongardShardNonterminalClaim(ObjectBongardShardJournalError):
    """A prior process claimed this physical call but never terminalized it."""


class ObjectBongardShardCallFailed(ObjectBongardShardJournalError):
    """The admitted physical transport call has a durable typed failure."""

    def __init__(self, *, call_key: str, failure_digest: str) -> None:
        super().__init__("physical object-observer shard transport failed")
        self.call_key = call_key
        self.failure_digest = failure_digest


def object_bongard_shard_journal_source_digest() -> str:
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
        raise ObjectBongardShardJournalError(
            f"{label} must be lowercase SHA-256, raw or sha256:-addressed"
        )
    return value


def _require_panel_id(value: object) -> str:
    if not isinstance(value, str) or _PANEL_ID.fullmatch(value) is None:
        raise ObjectBongardShardJournalError("panel ID is not a bounded identifier")
    return value


def _canonical_mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) for key in value
    ):
        raise ObjectBongardShardJournalError(f"{label} must be a JSON object")
    try:
        encoded = canonical_json(dict(value))
        decoded = json.loads(encoded.decode("utf-8"))
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise ObjectBongardShardJournalError(
            f"{label} is not canonical finite JSON"
        ) from exc
    if not isinstance(decoded, dict):
        raise ObjectBongardShardJournalError(f"{label} must be a JSON object")
    return decoded


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
        raise ObjectBongardShardJournalError(f"{label} fields differ")
    record_digest = raw.get("record_digest")
    _require_digest(record_digest, f"{label} digest")
    body = {key: item for key, item in raw.items() if key != "record_digest"}
    if record_digest != _address(body):
        raise ObjectBongardShardJournalError(f"{label} digest differs")
    return raw


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_once(path: Path, value: Mapping[str, Any]) -> None:
    payload = canonical_json(dict(value)) + b"\n"
    if len(payload) > MAX_RECORD_BYTES:
        raise ObjectBongardShardJournalError("journal record is oversized")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent)


def _read_canonical(path: Path, label: str) -> dict[str, Any]:
    try:
        before = os.lstat(path)
    except OSError as exc:
        raise ObjectBongardShardJournalError(f"{label} is missing") from exc
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or not 0 < before.st_size <= MAX_RECORD_BYTES
    ):
        raise ObjectBongardShardJournalError(f"{label} is not a bounded file")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
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
            raise ObjectBongardShardJournalError(f"{label} changed while opening")
        chunks: list[bytes] = []
        remaining = opened.st_size
        while remaining:
            block = os.read(descriptor, min(remaining, 1_048_576))
            if not block:
                raise ObjectBongardShardJournalError(f"{label} was truncated")
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
            raise ObjectBongardShardJournalError(f"{label} changed while reading")
    finally:
        os.close(descriptor)
    payload = b"".join(chunks)
    if not payload.endswith(b"\n") or payload.endswith(b"\n\n"):
        raise ObjectBongardShardJournalError(f"{label} encoding differs")
    try:
        decoded = json.loads(payload[:-1].decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ObjectBongardShardJournalError(f"{label} is malformed JSON") from exc
    if not isinstance(decoded, dict) or canonical_json(decoded) + b"\n" != payload:
        raise ObjectBongardShardJournalError(f"{label} is not canonical JSON")
    return decoded


def _read_exact_image(path: object) -> bytes:
    if not isinstance(path, str) or not os.path.isabs(path):
        raise ObjectBongardShardJournalError("shard image path must be absolute")
    try:
        before = os.lstat(path)
    except OSError as exc:
        raise ObjectBongardShardJournalError("shard image is unavailable") from exc
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or not 0 < before.st_size <= MAX_IMAGE_BYTES
    ):
        raise ObjectBongardShardJournalError(
            "shard image must be a bounded singly-linked regular file"
        )
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
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
            raise ObjectBongardShardJournalError("shard image changed while opening")
        data = b""
        while len(data) < opened.st_size:
            block = os.read(descriptor, opened.st_size - len(data))
            if not block:
                raise ObjectBongardShardJournalError("shard image was truncated")
            data += block
        after = os.fstat(descriptor)
        if after.st_nlink != 1 or (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ) != identity:
            raise ObjectBongardShardJournalError("shard image changed while reading")
    finally:
        os.close(descriptor)
    return data


def _runtime_binding(kwargs: Mapping[str, object]) -> dict[str, object]:
    expected_keys = {
        "model",
        "reasoning_effort",
        "minutes",
        "verbose",
        "executable",
        "cloud_policy_cache_snapshot",
        "model_catalog_snapshot",
        "expected_launcher_digest",
        "tool_surface_attestation",
        "expected_tool_surface_attestation_digest",
    }
    if set(kwargs) != expected_keys:
        raise ObjectBongardShardJournalError(
            "expected transport keyword set differs from observer transport"
        )
    model = kwargs["model"]
    effort = kwargs["reasoning_effort"]
    minutes = kwargs["minutes"]
    verbose = kwargs["verbose"]
    executable = kwargs["executable"]
    policy = kwargs["cloud_policy_cache_snapshot"]
    catalog = kwargs["model_catalog_snapshot"]
    launcher = kwargs["expected_launcher_digest"]
    attestation = kwargs["tool_surface_attestation"]
    attestation_digest = kwargs["expected_tool_surface_attestation_digest"]
    if not isinstance(model, str) or not model:
        raise ObjectBongardShardJournalError("transport model is invalid")
    if not isinstance(effort, str) or not effort:
        raise ObjectBongardShardJournalError("reasoning effort is invalid")
    if isinstance(minutes, bool) or not isinstance(minutes, int) or minutes <= 0:
        raise ObjectBongardShardJournalError("transport minutes are invalid")
    if not isinstance(verbose, bool):
        raise ObjectBongardShardJournalError("transport verbosity is invalid")
    if not isinstance(executable, str) or not executable:
        raise ObjectBongardShardJournalError("transport executable is invalid")
    if policy is not None and not isinstance(policy, CloudPolicyCacheSnapshot):
        raise ObjectBongardShardJournalError("policy cache snapshot type differs")
    if not isinstance(catalog, CodexModelCatalogSnapshot):
        raise ObjectBongardShardJournalError("model catalog snapshot type differs")
    if not isinstance(attestation, CodexNoToolsAttestation):
        raise ObjectBongardShardJournalError("no-tools attestation type differs")
    _require_digest(launcher, "expected launcher digest")
    _require_digest(attestation_digest, "expected no-tools attestation digest")
    if (
        launcher != attestation.launcher_digest
        or attestation_digest != attestation.attestation_digest
        or catalog.raw_digest != attestation.model_catalog_digest
    ):
        raise ObjectBongardShardJournalError("transport preflight bindings differ")
    policy_binding = "absent" if policy is None else policy.binding
    if policy_binding != attestation.cloud_config_bundle_cache_binding:
        raise ObjectBongardShardJournalError("policy cache binding differs")
    return {
        "model": model,
        "reasoning_effort": effort,
        "minutes": minutes,
        "verbose": verbose,
        "expected_launcher_digest": launcher,
        "cloud_policy_cache_binding": policy_binding,
        "model_catalog_digest": catalog.raw_digest,
        "no_tools_attestation_digest": attestation_digest,
        "executable_policy": (
            "exact-per-process-value;durable-identity-is-expected-launcher-digest"
        ),
    }


def _same_runtime_value(actual: object, expected: object) -> bool:
    if isinstance(expected, (CodexModelCatalogSnapshot, CloudPolicyCacheSnapshot)):
        return type(actual) is type(expected) and actual == expected
    if isinstance(expected, CodexNoToolsAttestation):
        return isinstance(actual, CodexNoToolsAttestation) and actual == expected
    return type(actual) is type(expected) and actual == expected


@dataclass(frozen=True, slots=True)
class ObjectBongardShardJournalSummary:
    manifest_digest: str
    terminal_call_keys: tuple[str, ...]
    nonterminal_call_keys: tuple[str, ...]
    success_count: int
    failure_count: int
    record_digest: str

    def to_data(self) -> dict[str, object]:
        return {
            "schema": "gkm.bongard-object-shard-journal-summary.v1",
            "manifest_digest": self.manifest_digest,
            "terminal_call_keys": list(self.terminal_call_keys),
            "nonterminal_call_keys": list(self.nonterminal_call_keys),
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "record_digest": self.record_digest,
            **_authority_data(),
        }


class ObjectBongardShardJournalTransport:
    """One panel-bound, durable ``NamedImageTransport`` wrapper."""

    def __init__(
        self,
        journal_directory: str | os.PathLike[str],
        *,
        authorization_digest: str,
        precommit_digest: str,
        context_digest: str,
        panel_id: str,
        packet: ObjectHypothesisPacket,
        atlas: Sequence[tuple[str, bytes]],
        expected_transport_kwargs: Mapping[str, object],
        underlying_transport: Callable[..., CodexStructuredResult],
    ) -> None:
        if not callable(underlying_transport):
            raise TypeError("underlying transport must be callable")
        self.authorization_digest = _require_digest(
            authorization_digest, "authorization digest"
        )
        self.precommit_digest = _require_digest(
            precommit_digest, "precommit digest"
        )
        self.context_digest = _require_digest(context_digest, "context digest")
        self.panel_id = _require_panel_id(panel_id)
        if not isinstance(packet, ObjectHypothesisPacket):
            raise TypeError("packet must be ObjectHypothesisPacket")
        verify_object_hypothesis_packet(packet)
        self.packet = packet
        self.plan = plan_prototype_object_feature_shards(packet)
        verify_prototype_object_feature_shard_plan(self.plan, packet)
        self._atlas = self._freeze_atlas(atlas)
        self._expected_transport_kwargs = dict(expected_transport_kwargs)
        self._runtime_binding = _runtime_binding(self._expected_transport_kwargs)
        self._underlying_transport = underlying_transport
        self.directory = Path(journal_directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        if not self.directory.is_dir() or self.directory.is_symlink():
            raise ObjectBongardShardJournalError(
                "journal directory must be a real directory"
            )
        self._claims = self.directory / "claims"
        self._results = self.directory / "results"
        self._outcomes = self.directory / "outcomes"
        for path in (self._claims, self._results, self._outcomes):
            path.mkdir(mode=0o700, exist_ok=True)
            if not path.is_dir() or path.is_symlink():
                raise ObjectBongardShardJournalError(
                    "journal record directory must be a real directory"
                )
        self._manifest_path = self.directory / "manifest.json"
        self._manifest = self._build_manifest()
        self._persist_or_verify_manifest()
        self._next_invocation = 0
        self.attempted_call_count = 0
        self.fresh_call_count = 0
        self.reused_call_count = 0
        self.refused_call_count = 0
        self._ordered_call_keys: list[str] = []

    def _freeze_atlas(
        self, atlas: Sequence[tuple[str, bytes]]
    ) -> dict[str, bytes]:
        if isinstance(atlas, (str, bytes)):
            raise ObjectBongardShardJournalError("atlas must be a named byte sequence")
        rows = tuple(atlas)
        if any(
            not isinstance(row, tuple)
            or len(row) != 2
            or not isinstance(row[0], str)
            or not isinstance(row[1], bytes)
            for row in rows
        ):
            raise ObjectBongardShardJournalError("atlas row type differs")
        names = tuple(row[0] for row in rows)
        expected_names = tuple(sheet.name for sheet in self.packet.atlas_sheets)
        if names != expected_names or len(names) != len(set(names)):
            raise ObjectBongardShardJournalError("atlas sheet order differs")
        result = {name: data for name, data in rows}
        for sheet in self.packet.atlas_sheets:
            data = result[sheet.name]
            if (
                len(data) != sheet.png_byte_count
                or hashlib.sha256(data).hexdigest() != sheet.png_digest
            ):
                raise ObjectBongardShardJournalError(
                    "atlas bytes differ from packet commitment"
                )
        return result

    def _build_manifest(self) -> dict[str, Any]:
        content = {
            "schema": JOURNAL_MANIFEST_SCHEMA,
            "protocol_id": JOURNAL_PROTOCOL_ID,
            "authorization_digest": self.authorization_digest,
            "precommit_digest": self.precommit_digest,
            "context_digest": self.context_digest,
            "panel_id": self.panel_id,
            "packet_digest": self.packet.digest(),
            "shard_plan": self.plan.to_data(),
            "atlas_sheets": [
                {
                    "name": sheet.name,
                    "byte_count": len(self._atlas[sheet.name]),
                    "sha256": hashlib.sha256(self._atlas[sheet.name]).hexdigest(),
                }
                for sheet in self.packet.atlas_sheets
            ],
            "output_schema_digest": canonical_digest(
                prototype_object_feature_output_schema()
            ),
            "runtime_binding": self._runtime_binding,
            "journal_source_digest": object_bongard_shard_journal_source_digest(),
            "one_physical_call_per_shard": True,
            "terminal_replay_calls_model": False,
            "nonterminal_claim_policy": "refuse-without-transport",
            **_authority_data(),
        }
        return _record(content)

    def _persist_or_verify_manifest(self) -> None:
        try:
            _write_once(self._manifest_path, self._manifest)
        except FileExistsError:
            persisted = _read_canonical(self._manifest_path, "journal manifest")
            if persisted != self._manifest:
                raise ObjectBongardShardJournalError(
                    "journal manifest differs from exact panel binding"
                )
        reloaded = _read_canonical(self._manifest_path, "journal manifest")
        _validate_record(
            reloaded,
            schema=JOURNAL_MANIFEST_SCHEMA,
            fields=set(self._manifest) - {"record_digest"},
            label="journal manifest",
        )
        if reloaded != self._manifest:
            raise ObjectBongardShardJournalError("journal manifest replay differs")

    @property
    def manifest_digest(self) -> str:
        return self._manifest["record_digest"]

    @property
    def ordered_call_keys(self) -> tuple[str, ...]:
        return tuple(self._ordered_call_keys)

    def _call_content(self, spec: ObjectFeatureShardSpec) -> dict[str, object]:
        return {
            "schema": "gkm.bongard-object-physical-shard-key.v1",
            "authorization_digest": self.authorization_digest,
            "precommit_digest": self.precommit_digest,
            "context_digest": self.context_digest,
            "panel_id": self.panel_id,
            "packet_digest": self.packet.digest(),
            "plan_digest": self.plan.plan_digest,
            "spec_digest": spec.spec_digest,
            "invocation_index": spec.shard_index,
        }

    def call_key_for_spec(self, spec: ObjectFeatureShardSpec) -> str:
        if (
            not isinstance(spec, ObjectFeatureShardSpec)
            or spec.shard_index >= len(self.plan.shards)
            or self.plan.shards[spec.shard_index] != spec
        ):
            raise ObjectBongardShardJournalError("shard spec is outside bound plan")
        return _address(self._call_content(spec))

    @staticmethod
    def _stem(call_key: str) -> str:
        _require_digest(call_key, "call key")
        return call_key.removeprefix("sha256:")

    def record_paths(self, call_key: str) -> tuple[Path, Path, Path]:
        stem = self._stem(call_key)
        return (
            self._claims / f"{stem}.json",
            self._results / f"{stem}.json",
            self._outcomes / f"{stem}.json",
        )

    def _expected_claim(self, spec: ObjectFeatureShardSpec) -> dict[str, Any]:
        call_key = self.call_key_for_spec(spec)
        binding = {
            key: value
            for key, value in self._call_content(spec).items()
            if key != "schema"
        }
        return _record(
            {
                "schema": SHARD_CLAIM_SCHEMA,
                **binding,
                "call_key": call_key,
                "manifest_digest": self.manifest_digest,
                "exclusive_create_before_transport": True,
            }
        )

    def _resolve_invocation(
        self,
        prompt: object,
        image_paths: object,
        image_names: object,
        output_schema: object,
        kwargs: Mapping[str, object],
    ) -> tuple[ObjectFeatureShardSpec, str]:
        if self._next_invocation >= len(self.plan.shards):
            raise ObjectBongardShardJournalError("shard invocation exceeds frozen plan")
        spec = self.plan.shards[self._next_invocation]
        expected_prompt = prototype_object_feature_shard_prompt(self.packet, spec)
        if prompt != expected_prompt:
            raise ObjectBongardShardJournalError(
                "shard prompt differs from frozen spec"
            )
        if (
            isinstance(image_paths, (str, bytes))
            or not isinstance(image_paths, Sequence)
            or isinstance(image_names, (str, bytes))
            or not isinstance(image_names, Sequence)
            or tuple(image_names) != (spec.sheet_name,)
            or len(image_paths) != 1
        ):
            raise ObjectBongardShardJournalError(
                "shard transport must receive exactly one frozen sheet"
            )
        path = image_paths[0]
        if _read_exact_image(path) != self._atlas[spec.sheet_name]:
            raise ObjectBongardShardJournalError("shard sheet bytes differ")
        expected_schema = prototype_object_feature_output_schema()
        try:
            validate_codex_strict_output_schema(output_schema)  # type: ignore[arg-type]
            actual_schema = canonical_json(output_schema)
        except Exception as exc:
            raise ObjectBongardShardJournalError(
                "shard output schema is invalid"
            ) from exc
        if actual_schema != canonical_json(expected_schema):
            raise ObjectBongardShardJournalError("shard output schema differs")
        if set(kwargs) != set(self._expected_transport_kwargs) or any(
            not _same_runtime_value(kwargs[key], expected)
            for key, expected in self._expected_transport_kwargs.items()
        ):
            raise ObjectBongardShardJournalError("shard model/runtime kwargs differ")
        return spec, path

    def _load_claim(self, path: Path, expected: Mapping[str, Any]) -> dict[str, Any]:
        claim = _validate_record(
            _read_canonical(path, "shard claim"),
            schema=SHARD_CLAIM_SCHEMA,
            fields=set(expected) - {"record_digest"},
            label="shard claim",
        )
        if claim != dict(expected):
            raise ObjectBongardShardJournalError("shard claim binding differs")
        return claim

    def _load_outcome(
        self, path: Path, *, claim: Mapping[str, Any]
    ) -> dict[str, Any] | None:
        if not path.exists():
            return None
        outcome = _validate_record(
            _read_canonical(path, "shard terminal outcome"),
            schema=SHARD_OUTCOME_SCHEMA,
            fields={
                "schema",
                "call_key",
                "claim_digest",
                "terminal_status",
                "result_digest",
                "manifest_digest",
                "terminal",
            },
            label="shard terminal outcome",
        )
        if (
            outcome["call_key"] != claim["call_key"]
            or outcome["claim_digest"] != claim["record_digest"]
            or outcome["manifest_digest"] != self.manifest_digest
            or outcome["terminal"] is not True
            or outcome["terminal_status"] not in {"success", "failure"}
        ):
            raise ObjectBongardShardJournalError("shard outcome lineage differs")
        return outcome

    def _load_result(
        self,
        path: Path,
        *,
        claim: Mapping[str, Any],
        outcome: Mapping[str, Any],
    ) -> dict[str, Any]:
        result = _validate_record(
            _read_canonical(path, "shard durable result"),
            schema=SHARD_RESULT_SCHEMA,
            fields={
                "schema",
                "call_key",
                "claim_digest",
                "status",
                "payload",
                "receipt",
                "payload_digest",
                "receipt_digest",
                "failure_code",
                "source_exception_type",
                "manifest_digest",
            },
            label="shard durable result",
        )
        if (
            result["call_key"] != claim["call_key"]
            or result["claim_digest"] != claim["record_digest"]
            or result["manifest_digest"] != self.manifest_digest
            or result["record_digest"] != outcome["result_digest"]
            or result["status"] != outcome["terminal_status"]
        ):
            raise ObjectBongardShardJournalError("shard result lineage differs")
        status = result["status"]
        if status == "success":
            payload = _canonical_mapping(result["payload"], "stored shard payload")
            receipt = result["receipt"]
            if (
                result["payload_digest"] != _address(payload)
                or not isinstance(receipt, Mapping)
                or result["receipt_digest"] != receipt.get("receipt_digest")
                or result["failure_code"] is not None
                or result["source_exception_type"] is not None
            ):
                raise ObjectBongardShardJournalError("successful shard result differs")
            restored = _receipt_from_data(receipt)
            if not isinstance(restored, CodexReceipt):
                raise ObjectBongardShardJournalError(
                    "successful shard receipt type differs"
                )
            self._validate_receipt_runtime(restored)
        elif status == "failure":
            if (
                result["payload"] is not None
                or result["receipt"] is not None
                or result["payload_digest"] is not None
                or result["receipt_digest"] is not None
                or result["failure_code"] != "physical_shard_transport_failed"
                or not isinstance(result["source_exception_type"], str)
                or _EXCEPTION_TYPE.fullmatch(result["source_exception_type"]) is None
            ):
                raise ObjectBongardShardJournalError("failed shard result differs")
        else:
            raise ObjectBongardShardJournalError("shard result status differs")
        return result

    def _validate_receipt_runtime(self, receipt: CodexReceipt) -> None:
        binding = self._runtime_binding
        if (
            receipt.requested_model != binding["model"]
            or receipt.requested_reasoning_effort != binding["reasoning_effort"]
            or receipt.codex_launcher_digest
            != binding["expected_launcher_digest"]
            or receipt.cloud_config_bundle_cache_binding
            != binding["cloud_policy_cache_binding"]
            or receipt.model_catalog_digest != binding["model_catalog_digest"]
            or receipt.tool_surface_attestation_digest
            != binding["no_tools_attestation_digest"]
        ):
            raise ObjectBongardShardJournalError(
                "shard receipt model/runtime binding differs"
            )

    def _replay_result(
        self,
        result: Mapping[str, Any],
        *,
        prompt: str,
        path: str,
        name: str,
        schema: Mapping[str, Any],
    ) -> CodexStructuredResult:
        if result["status"] == "failure":
            raise ObjectBongardShardCallFailed(
                call_key=result["call_key"],
                failure_digest=result["record_digest"],
            )
        payload = _canonical_mapping(result["payload"], "stored shard payload")
        receipt = _receipt_from_data(result["receipt"])
        if not isinstance(receipt, CodexReceipt):
            raise ObjectBongardShardJournalError("stored shard receipt type differs")
        self._validate_receipt_runtime(receipt)
        try:
            validate_codex_named_image_receipt(
                receipt, prompt, (path,), (name,), schema, payload
            )
        except Exception as exc:
            raise ObjectBongardShardJournalError(
                "stored shard receipt fails cold input replay"
            ) from exc
        return CodexStructuredResult(payload=payload, receipt=receipt)

    def _cold_validate_success_result(
        self,
        result: Mapping[str, Any],
        spec: ObjectFeatureShardSpec,
    ) -> None:
        payload = _canonical_mapping(result["payload"], "stored shard payload")
        receipt = _receipt_from_data(result["receipt"])
        if not isinstance(receipt, CodexReceipt):
            raise ObjectBongardShardJournalError("stored shard receipt type differs")
        self._validate_receipt_runtime(receipt)
        prompt = prototype_object_feature_shard_prompt(self.packet, spec)
        schema = prototype_object_feature_output_schema()
        with tempfile.TemporaryDirectory(prefix="bongard-shard-cold-replay-") as raw:
            path = Path(raw) / spec.sheet_name
            path.write_bytes(self._atlas[spec.sheet_name])
            try:
                validate_codex_named_image_receipt(
                    receipt,
                    prompt,
                    (str(path.resolve()),),
                    (spec.sheet_name,),
                    schema,
                    payload,
                )
            except Exception as exc:
                raise ObjectBongardShardJournalError(
                    "stored shard receipt fails cold exact-input replay"
                ) from exc

    def _success_result(
        self,
        *,
        claim: Mapping[str, Any],
        result: CodexStructuredResult,
        prompt: str,
        path: str,
        spec: ObjectFeatureShardSpec,
        schema: Mapping[str, Any],
    ) -> dict[str, Any]:
        if not isinstance(result, CodexStructuredResult):
            raise ObjectBongardShardJournalError(
                "underlying transport returned the wrong result type"
            )
        payload = _canonical_mapping(result.payload, "transport shard payload")
        if not isinstance(result.receipt, CodexReceipt):
            raise ObjectBongardShardJournalError(
                "underlying transport returned no full Codex receipt"
            )
        self._validate_receipt_runtime(result.receipt)
        validate_codex_named_image_receipt(
            result.receipt,
            prompt,
            (path,),
            (spec.sheet_name,),
            schema,
            payload,
        )
        if _read_exact_image(path) != self._atlas[spec.sheet_name]:
            raise ObjectBongardShardJournalError(
                "shard sheet changed during transport"
            )
        return _record(
            {
                "schema": SHARD_RESULT_SCHEMA,
                "call_key": claim["call_key"],
                "claim_digest": claim["record_digest"],
                "status": "success",
                "payload": payload,
                "receipt": result.receipt.to_dict(),
                "payload_digest": _address(payload),
                "receipt_digest": result.receipt.receipt_digest,
                "failure_code": None,
                "source_exception_type": None,
                "manifest_digest": self.manifest_digest,
            }
        )

    def _failure_result(
        self, *, claim: Mapping[str, Any], exception: Exception
    ) -> dict[str, Any]:
        source_type = f"{type(exception).__module__}.{type(exception).__qualname__}"
        if _EXCEPTION_TYPE.fullmatch(source_type) is None:
            source_type = "builtins.Exception"
        return _record(
            {
                "schema": SHARD_RESULT_SCHEMA,
                "call_key": claim["call_key"],
                "claim_digest": claim["record_digest"],
                "status": "failure",
                "payload": None,
                "receipt": None,
                "payload_digest": None,
                "receipt_digest": None,
                "failure_code": "physical_shard_transport_failed",
                "source_exception_type": source_type,
                "manifest_digest": self.manifest_digest,
            }
        )

    def _finish(
        self,
        *,
        claim: Mapping[str, Any],
        result: Mapping[str, Any],
        result_path: Path,
        outcome_path: Path,
    ) -> None:
        _write_once(result_path, result)
        if _read_canonical(result_path, "shard durable result") != dict(result):
            raise ObjectBongardShardJournalError("durable shard result reload differs")
        outcome = _record(
            {
                "schema": SHARD_OUTCOME_SCHEMA,
                "call_key": claim["call_key"],
                "claim_digest": claim["record_digest"],
                "terminal_status": result["status"],
                "result_digest": result["record_digest"],
                "manifest_digest": self.manifest_digest,
                "terminal": True,
            }
        )
        _write_once(outcome_path, outcome)
        if _read_canonical(outcome_path, "shard terminal outcome") != outcome:
            raise ObjectBongardShardJournalError("terminal shard reload differs")

    def __call__(
        self,
        prompt: str,
        image_paths: Sequence[str],
        image_names: Sequence[str],
        output_schema: Mapping[str, Any],
        **kwargs: object,
    ) -> CodexStructuredResult:
        spec, path = self._resolve_invocation(
            prompt, image_paths, image_names, output_schema, kwargs
        )
        self._next_invocation += 1
        call_key = self.call_key_for_spec(spec)
        self.attempted_call_count += 1
        self._ordered_call_keys.append(call_key)
        claim_path, result_path, outcome_path = self.record_paths(call_key)
        expected_claim = self._expected_claim(spec)
        try:
            _write_once(claim_path, expected_claim)
        except FileExistsError:
            claim = self._load_claim(claim_path, expected_claim)
            outcome = self._load_outcome(outcome_path, claim=claim)
            if outcome is None:
                self.refused_call_count += 1
                raise ObjectBongardShardNonterminalClaim(
                    "physical shard has a preexisting nonterminal claim; "
                    "transport rerun is forbidden"
                )
            result = self._load_result(
                result_path, claim=claim, outcome=outcome
            )
            self.reused_call_count += 1
            return self._replay_result(
                result,
                prompt=prompt,
                path=path,
                name=spec.sheet_name,
                schema=output_schema,
            )
        claim = self._load_claim(claim_path, expected_claim)
        self.fresh_call_count += 1
        try:
            raw_result = self._underlying_transport(
                prompt,
                image_paths,
                image_names,
                output_schema,
                **kwargs,
            )
            result = self._success_result(
                claim=claim,
                result=raw_result,
                prompt=prompt,
                path=path,
                spec=spec,
                schema=output_schema,
            )
        except Exception as exc:
            failure = self._failure_result(claim=claim, exception=exc)
            self._finish(
                claim=claim,
                result=failure,
                result_path=result_path,
                outcome_path=outcome_path,
            )
            raise ObjectBongardShardCallFailed(
                call_key=call_key, failure_digest=failure["record_digest"]
            ) from None
        self._finish(
            claim=claim,
            result=result,
            result_path=result_path,
            outcome_path=outcome_path,
        )
        return raw_result

    def verify(self) -> ObjectBongardShardJournalSummary:
        """Cold-verify every expected record and reject all extra files."""

        self._persist_or_verify_manifest()
        expected_stems = {
            self._stem(self.call_key_for_spec(spec)) for spec in self.plan.shards
        }
        terminal: list[str] = []
        nonterminal: list[str] = []
        success_count = 0
        failure_count = 0
        for directory in (self._claims, self._results, self._outcomes):
            actual_names = {item.name for item in directory.iterdir()}
            if any(
                not name.endswith(".json")
                or name[:-5] not in expected_stems
                for name in actual_names
            ):
                raise ObjectBongardShardJournalError(
                    "journal contains an unexpected record path"
                )
        for spec in self.plan.shards:
            call_key = self.call_key_for_spec(spec)
            claim_path, result_path, outcome_path = self.record_paths(call_key)
            if not claim_path.exists():
                if result_path.exists() or outcome_path.exists():
                    raise ObjectBongardShardJournalError(
                        "journal result exists without a claim"
                    )
                continue
            claim = self._load_claim(claim_path, self._expected_claim(spec))
            outcome = self._load_outcome(outcome_path, claim=claim)
            if outcome is None:
                if result_path.exists():
                    raise ObjectBongardShardJournalError(
                        "nonterminal claim has an uncommitted result"
                    )
                nonterminal.append(call_key)
                continue
            result = self._load_result(
                result_path, claim=claim, outcome=outcome
            )
            terminal.append(call_key)
            if result["status"] == "success":
                self._cold_validate_success_result(result, spec)
                success_count += 1
            else:
                failure_count += 1
        content = {
            "schema": "gkm.bongard-object-shard-journal-summary.v1",
            "manifest_digest": self.manifest_digest,
            "terminal_call_keys": terminal,
            "nonterminal_call_keys": nonterminal,
            "success_count": success_count,
            "failure_count": failure_count,
            **_authority_data(),
        }
        return ObjectBongardShardJournalSummary(
            manifest_digest=self.manifest_digest,
            terminal_call_keys=tuple(terminal),
            nonterminal_call_keys=tuple(nonterminal),
            success_count=success_count,
            failure_count=failure_count,
            record_digest=_address(content),
        )


def verify_object_bongard_shard_journal(
    journal: ObjectBongardShardJournalTransport,
) -> ObjectBongardShardJournalSummary:
    if not isinstance(journal, ObjectBongardShardJournalTransport):
        raise TypeError("journal must be ObjectBongardShardJournalTransport")
    return journal.verify()


__all__ = (
    "JOURNAL_MANIFEST_SCHEMA",
    "JOURNAL_PROTOCOL_ID",
    "ObjectBongardShardCallFailed",
    "ObjectBongardShardJournalError",
    "ObjectBongardShardJournalSummary",
    "ObjectBongardShardJournalTransport",
    "ObjectBongardShardNonterminalClaim",
    "SHARD_CLAIM_SCHEMA",
    "SHARD_OUTCOME_SCHEMA",
    "SHARD_RESULT_SCHEMA",
    "object_bongard_shard_journal_source_digest",
    "verify_object_bongard_shard_journal",
)
