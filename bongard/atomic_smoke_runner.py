"""Causal Python-only runner for the exploratory atomic semantic smoke.

The runner is deliberately narrower than the general Bongard benchmark.  It
performs exactly 29 isolated headless Codex calls on success, freezes a
positive-only formula from the twelve labelled support panels, and does not
open a query source until that formula exists.  Query labels remain behind the
``EpisodePlan`` seal until the runner-owned prediction store has fsynced and
independently decoded the exact commitment.

This is an exploratory transport/synthesis smoke, not a calibrated benchmark.
All four scientific-authorisation flags are therefore permanently false.
Lean and every other execution backend are outside this module's dependency
and artifact identity.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
import re
import stat
import unicodedata
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

from bongard.artifacts import TruthEvidenceRecord, canonical_digest, canonical_json
from bongard.atomic_semantic_synthesis import (
    MAX_ATOMIC_CONJUNCTION_SIZE,
    OPERATIONAL_SELECTION_SCOPE,
    AtomicEvidenceBinding,
    AtomicSelectionArchive,
    AtomicSoftPredicate,
    AtomicSupportCell,
    AtomicSupportMatrix,
    OperationalNonmatchRecord,
    PanelDescriptionBinding,
    cold_decode_and_replay_atomic_selection,
    evaluate_atomic_formula,
    synthesize_atomic_conjunction,
)
from bongard.atomic_smoke_precommit import AtomicSmokePrecommit
from bongard.evidence import Disposition, Evidence, Provenance
from bongard.transport import (
    DEFAULT_CODEX_MODEL,
    DEFAULT_REASONING_EFFORT,
    NAMED_IMAGE_INPUT_DIGEST_SCHEMA,
    TEXT_STRUCTURED_INPUT_DIGEST_SCHEMA,
    CloudPolicyCacheSnapshot,
    CodexReceipt,
    CodexStructuredResult,
    run_codex_named_images_structured,
    run_codex_text_structured,
    validate_codex_named_image_receipt,
    validate_codex_receipt,
    validate_codex_strict_output_schema,
    validate_codex_text_receipt,
)


ATOMIC_SMOKE_RUN_SCHEMA = "gkm.bongard-atomic-smoke-run.v1"
ATOMIC_SMOKE_CALL_SCHEMA = "gkm.bongard-atomic-smoke-call.v1"
ATOMIC_SMOKE_PREDICTION_SCHEMA = "gkm.bongard-atomic-smoke-predictions.v1"
ATOMIC_SMOKE_LABEL_REVEAL_SCHEMA = "gkm.bongard-atomic-smoke-label-reveal.v1"
ATOMIC_SMOKE_PERSISTENCE_SCHEMA = "gkm.bongard-atomic-smoke-persistence.v1"
ATOMIC_SMOKE_SUCCESS_CALL_COUNT = 29
ATOMIC_SMOKE_MAX_ATOMS = 12

_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_ADDRESS = re.compile(r"sha256:[0-9a-f]{64}\Z")
_PANEL_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")


class AtomicSmokeRunError(ValueError):
    """A live run or cold replay violated the atomic smoke protocol."""


def _description_schema() -> dict[str, object]:
    return {
        "type": "object",
        "properties": {"description": {"type": "string"}},
        "required": ["description"],
        "additionalProperties": False,
    }


def _proposal_schema() -> dict[str, object]:
    atom = {
        "type": "object",
        "properties": {
            "phrase": {"type": "string"},
        },
        "required": ["phrase"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {"atoms": {"type": "array", "items": atom}},
        "required": ["atoms"],
        "additionalProperties": False,
    }


def _scorer_schema() -> dict[str, object]:
    result = {
        "type": "object",
        "properties": {
            "atom_id": {"type": "string"},
            "disposition": {
                "type": "string",
                "enum": [
                    "present",
                    "operational_nonmatch",
                    "indeterminate",
                    "error",
                ],
            },
            "explanation": {"type": "string"},
        },
        "required": ["atom_id", "disposition", "explanation"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {"results": {"type": "array", "items": result}},
        "required": ["results"],
        "additionalProperties": False,
    }


_DESCRIPTION_PROMPT = (
    "Inspect the single neutrally named panel. Describe only visible geometry, "
    "objects or object-like gestalt, topology, orientation, and spatial "
    "relations. Use one self-contained ASCII sentence. Do not infer a Bongard "
    "side, compare with unseen panels, mention filenames, or emit code."
)

_SCORER_INSTRUCTIONS = (
    "Judge every supplied affirmative atom independently against this one "
    "neutrally named panel and its frozen vision description. Return present "
    "only when the visible panel supports the atom; operational_nonmatch only "
    "for a clear exploratory observer nonmatch; indeterminate for ambiguity; error "
    "only for an observation failure. This is uncalibrated evidence, not a "
    "semantic truth certificate. Return every atom exactly once."
)


def atomic_smoke_description_protocol_digest() -> str:
    return canonical_digest(
        {
            "schema": "gkm.bongard-atomic-smoke-neutral-description-protocol.v1",
            "prompt": _DESCRIPTION_PROMPT,
            "output_schema": _description_schema(),
            "image_count": 1,
            "neutral_image_name": "panel.png",
            "side_label_visible": False,
        }
    )


def atomic_smoke_scorer_protocol_digest() -> str:
    return canonical_digest(
        {
            "schema": "gkm.bongard-atomic-smoke-scorer-protocol.v1",
            "instructions": _SCORER_INSTRUCTIONS,
            "output_schema": _scorer_schema(),
            "model_output_vocabulary": [
                "present",
                "operational_nonmatch",
                "indeterminate",
                "error",
            ],
            "internal_nonmatch_representation": (
                "operational-nonmatch-record-projects-to-indeterminate/v1"
            ),
            "scope": "operational-uncalibrated-observer-nonmatch/v1",
        }
    )


def atomic_smoke_run_protocol_digest() -> str:
    """Identity of the static Python-only 29-call causal protocol."""

    return canonical_digest(
        {
            "schema": "gkm.bongard-atomic-smoke-run-protocol.v1",
            "run_schema": ATOMIC_SMOKE_RUN_SCHEMA,
            "call_schema": ATOMIC_SMOKE_CALL_SCHEMA,
            "description_prompt": _DESCRIPTION_PROMPT,
            "description_schema": _description_schema(),
            "description_protocol_digest": atomic_smoke_description_protocol_digest(),
            "proposal_schema": _proposal_schema(),
            "scorer_protocol_digest": atomic_smoke_scorer_protocol_digest(),
            "success_call_order": [
                "support-description"
            ]
            * 12
            + ["atom-proposal"]
            + ["support-scoring"] * 12
            + ["query-description"] * 2
            + ["query-scoring"] * 2,
            "positive_formula": {
                "kind": "all",
                "minimum_atoms": 1,
                "maximum_atoms": MAX_ATOMIC_CONJUNCTION_SIZE,
                "not": False,
                "or": False,
                "polarity_flip": False,
            },
            "query_pixels_after_formula_freeze": True,
            "labels_after_verified_prediction_persistence": True,
            "prediction_persistence": {
                "schema": ATOMIC_SMOKE_PERSISTENCE_SCHEMA,
                "protocol": "exclusive-create-or-identical-fsync-reload/v1",
                "core_owned": True,
                "arbitrary_callback": False,
            },
            "cold_replay_external_label_nonce_required": True,
            "cold_replay_external_precommit_required": True,
            "python_predicate_authoritative": True,
            "exploratory_uncalibrated_nonmatch": True,
            "optional_checker_required": False,
            "dependence_design_authorized": False,
            "calibration_authorized": False,
            "benchmark_claim_authorized": False,
            "official_test_authorized": False,
        }
    )


def _mapping(value: object, fields: frozenset[str], label: str) -> Mapping[str, Any]:
    if (
        not isinstance(value, Mapping)
        or any(not isinstance(key, str) for key in value)
        or set(value) != fields
    ):
        raise AtomicSmokeRunError(f"{label} fields differ from the static schema")
    return value


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise AtomicSmokeRunError(f"{label} must be a lowercase SHA-256")
    return value


def _address(value: object, label: str) -> str:
    if not isinstance(value, str) or _ADDRESS.fullmatch(value) is None:
        raise AtomicSmokeRunError(f"{label} must be a sha256: content address")
    return value


def _text(value: object, label: str, *, maximum: int = 4096) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or value != value.strip()
        or unicodedata.normalize("NFKC", value) != value
        or len(value.encode("utf-8")) > maximum
        or "\x00" in value
        or any(unicodedata.category(character) in {"Cc", "Cf"} for character in value)
    ):
        raise AtomicSmokeRunError(f"{label} must be bounded canonical exact text")
    return value


def _prompt_text(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or value != value.strip()
        or unicodedata.normalize("NFKC", value) != value
        or len(value.encode("utf-8")) > 65536
        or "\x00" in value
        or any(
            unicodedata.category(character) in {"Cc", "Cf"}
            and character != "\n"
            for character in value
        )
    ):
        raise AtomicSmokeRunError(f"{label} must be bounded canonical prompt text")
    return value


def _clone_json(value: object, label: str) -> Any:
    try:
        return json.loads(canonical_json(value))
    except (TypeError, ValueError, UnicodeError) as exc:
        raise AtomicSmokeRunError(f"{label} is not finite canonical JSON") from exc


def _freeze_json(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    return value


def _thaw_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _payload_digest(payload: Mapping[str, Any]) -> str:
    return canonical_digest(dict(payload))


def _receipt_data(receipt: CodexReceipt | Mapping[str, Any]) -> dict[str, Any]:
    raw = receipt.to_dict() if isinstance(receipt, CodexReceipt) else dict(receipt)
    validate_codex_receipt(raw)
    return raw


@dataclass(frozen=True, slots=True)
class PredictionPersistenceReceipt:
    """Typed receipt for the runner-owned durable prediction boundary."""

    prediction_commitment_digest: str
    payload_sha256: str
    payload_bytes: int
    filename: str
    persistence_protocol: str
    receipt_digest: str

    def __post_init__(self) -> None:
        _digest(self.prediction_commitment_digest, "persisted prediction digest")
        _digest(self.payload_sha256, "persisted prediction payload digest")
        if type(self.payload_bytes) is not int or self.payload_bytes <= 0:
            raise AtomicSmokeRunError("persisted prediction byte count is invalid")
        expected_filename = self.prediction_commitment_digest + ".predictions.json"
        if self.filename != expected_filename:
            raise AtomicSmokeRunError("prediction persistence filename is not canonical")
        if self.persistence_protocol != "exclusive-create-or-identical-fsync-reload/v1":
            raise AtomicSmokeRunError("prediction persistence protocol differs")
        _digest(self.receipt_digest, "prediction persistence receipt digest")
        if self.receipt_digest != canonical_digest(self.content_data()):
            raise AtomicSmokeRunError("prediction persistence receipt digest differs")

    def content_data(self) -> dict[str, object]:
        return {
            "schema": ATOMIC_SMOKE_PERSISTENCE_SCHEMA,
            "prediction_commitment_digest": self.prediction_commitment_digest,
            "payload_sha256": self.payload_sha256,
            "payload_bytes": self.payload_bytes,
            "filename": self.filename,
            "persistence_protocol": self.persistence_protocol,
        }

    def to_data(self) -> dict[str, object]:
        return {**self.content_data(), "receipt_digest": self.receipt_digest}

    @classmethod
    def create(
        cls, prediction_commitment_digest: str, payload: bytes
    ) -> "PredictionPersistenceReceipt":
        values = {
            "prediction_commitment_digest": prediction_commitment_digest,
            "payload_sha256": hashlib.sha256(payload).hexdigest(),
            "payload_bytes": len(payload),
            "filename": prediction_commitment_digest + ".predictions.json",
            "persistence_protocol": "exclusive-create-or-identical-fsync-reload/v1",
        }
        content = {"schema": ATOMIC_SMOKE_PERSISTENCE_SCHEMA, **values}
        return cls(**values, receipt_digest=canonical_digest(content))

    @classmethod
    def from_data(cls, value: Mapping[str, Any]) -> "PredictionPersistenceReceipt":
        data = _mapping(
            value,
            frozenset(
                {
                    "schema",
                    "prediction_commitment_digest",
                    "payload_sha256",
                    "payload_bytes",
                    "filename",
                    "persistence_protocol",
                    "receipt_digest",
                }
            ),
            "prediction persistence receipt",
        )
        if data["schema"] != ATOMIC_SMOKE_PERSISTENCE_SCHEMA:
            raise AtomicSmokeRunError("unsupported prediction persistence receipt")
        result = cls(
            prediction_commitment_digest=data["prediction_commitment_digest"],
            payload_sha256=data["payload_sha256"],
            payload_bytes=data["payload_bytes"],
            filename=data["filename"],
            persistence_protocol=data["persistence_protocol"],
            receipt_digest=data["receipt_digest"],
        )
        if result.to_data() != _clone_json(value, "prediction persistence receipt"):
            raise AtomicSmokeRunError("prediction persistence receipt is not canonical")
        return result


def _open_prediction_store(prediction_store_dir: str | Path) -> tuple[Path, int]:
    raw = Path(prediction_store_dir)
    absolute = raw.absolute()
    try:
        supplied_info = os.lstat(absolute)
        resolved = raw.resolve(strict=True)
    except OSError as exc:
        raise AtomicSmokeRunError("prediction store directory must already exist") from exc
    if stat.S_ISLNK(supplied_info.st_mode) or not resolved.is_dir():
        raise AtomicSmokeRunError("prediction store directory must be canonical and real")
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    descriptor = -1
    try:
        descriptor = os.open(resolved, flags)
        info = os.fstat(descriptor)
    except OSError as exc:
        if descriptor >= 0:
            os.close(descriptor)
        raise AtomicSmokeRunError("cannot open prediction store directory") from exc
    if not stat.S_ISDIR(info.st_mode):
        os.close(descriptor)
        raise AtomicSmokeRunError("prediction store descriptor is not a directory")
    return resolved, descriptor


def _read_stable_prediction_at(directory_fd: int, filename: str) -> bytes:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    try:
        descriptor = os.open(filename, flags, dir_fd=directory_fd)
    except OSError as exc:
        raise AtomicSmokeRunError("persisted prediction file cannot be opened") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise AtomicSmokeRunError("persisted prediction is not one regular file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 65536)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        identity_after = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if identity_after != identity_before:
            raise AtomicSmokeRunError("persisted prediction changed during reload")
        payload = b"".join(chunks)
        if len(payload) != before.st_size:
            raise AtomicSmokeRunError("persisted prediction reload was incomplete")
        return payload
    finally:
        os.close(descriptor)


def _persist_prediction_commitment(
    prediction: Mapping[str, Any], prediction_store_dir: str | Path
) -> tuple[dict[str, Any], PredictionPersistenceReceipt]:
    exact = _clone_json(prediction, "prediction commitment")
    payload = canonical_json(exact)
    prediction_digest = _digest(
        exact.get("commitment_digest"), "prediction commitment digest"
    )
    receipt = PredictionPersistenceReceipt.create(prediction_digest, payload)
    _directory, directory_fd = _open_prediction_store(prediction_store_dir)
    created = False
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    try:
        try:
            descriptor = os.open(
                receipt.filename, flags, 0o600, dir_fd=directory_fd
            )
        except FileExistsError:
            descriptor = -1
        except OSError as exc:
            raise AtomicSmokeRunError("cannot create persisted prediction file") from exc
        if descriptor >= 0:
            created = True
            try:
                offset = 0
                while offset < len(payload):
                    written = os.write(descriptor, payload[offset:])
                    if written <= 0:
                        raise AtomicSmokeRunError(
                            "prediction persistence write made no progress"
                        )
                    offset += written
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            os.fsync(directory_fd)
        reloaded_payload = _read_stable_prediction_at(directory_fd, receipt.filename)
        if reloaded_payload != payload:
            action = "created" if created else "existing"
            raise AtomicSmokeRunError(
                f"{action} prediction file differs from exact commitment"
            )
        try:
            reloaded = json.loads(reloaded_payload)
        except (json.JSONDecodeError, UnicodeError) as exc:
            raise AtomicSmokeRunError("persisted prediction JSON cannot be decoded") from exc
        if canonical_json(reloaded) != payload:
            raise AtomicSmokeRunError("persisted prediction JSON is not canonical")
        return _clone_json(reloaded, "reloaded prediction"), receipt
    finally:
        os.close(directory_fd)


def _verify_persisted_prediction(
    prediction: Mapping[str, Any],
    receipt: PredictionPersistenceReceipt,
    prediction_store_dir: str | Path,
) -> None:
    payload = canonical_json(_clone_json(prediction, "prediction commitment"))
    expected = PredictionPersistenceReceipt.create(
        prediction["commitment_digest"], payload
    )
    if expected != receipt:
        raise AtomicSmokeRunError("prediction persistence receipt differs")
    _directory, directory_fd = _open_prediction_store(prediction_store_dir)
    try:
        if _read_stable_prediction_at(directory_fd, receipt.filename) != payload:
            raise AtomicSmokeRunError("external persisted prediction bytes differ")
    finally:
        os.close(directory_fd)


def _validate_cold_named_envelope(
    receipt: Mapping[str, Any],
    *,
    prompt: str,
    output_schema: Mapping[str, Any],
    image_name: str,
    image_digest: str,
    image_byte_count: int,
) -> None:
    """Reproduce a named-image receipt without reopening experiment pixels."""

    validate_codex_receipt(receipt)
    validate_codex_strict_output_schema(output_schema)
    if receipt["input_digest_schema"] != NAMED_IMAGE_INPUT_DIGEST_SCHEMA:
        raise AtomicSmokeRunError("named-image call carries another transport domain")
    identities = [
        {
            "name": image_name,
            "byte_count": image_byte_count,
            "content_digest": image_digest,
        }
    ]
    schema_digest = canonical_digest(dict(output_schema))
    prompt_digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    view_digest = canonical_digest(identities)
    set_digest = "sha256:" + canonical_digest(
        {"schema": NAMED_IMAGE_INPUT_DIGEST_SCHEMA, "images": identities}
    )
    envelope = {
        "schema": NAMED_IMAGE_INPUT_DIGEST_SCHEMA,
        "task": prompt,
        "ordered_image_identities": identities,
        "image_view_digest": view_digest,
        "image_set_digest": set_digest,
        "prompt_digest": prompt_digest,
        "output_schema_digest": schema_digest,
    }
    expected = {
        "task_digest": prompt_digest,
        "prompt_digest": prompt_digest,
        "input_digest_schema": NAMED_IMAGE_INPUT_DIGEST_SCHEMA,
        "input_digest": canonical_digest(envelope),
        "output_schema_digest": schema_digest,
        "panel_view_digest": view_digest,
        "panel_set_digest": set_digest,
    }
    if any(receipt[key] != value for key, value in expected.items()):
        raise AtomicSmokeRunError("named-image receipt differs from frozen evidence")


@dataclass(frozen=True, slots=True)
class AtomicSmokeCallRecord:
    """One successful causally bound model call."""

    ordinal: int
    phase: str
    domain: str
    panel_id: str | None
    image_name: str | None
    image_digest: str | None
    image_byte_count: int | None
    atom_ids: tuple[str, ...]
    prompt: str
    output_schema: Mapping[str, Any]
    payload: Mapping[str, Any]
    receipt: Mapping[str, Any]
    call_digest: str

    def __post_init__(self) -> None:
        if type(self.ordinal) is not int or not 1 <= self.ordinal <= 29:
            raise AtomicSmokeRunError("call ordinal must be a literal integer in 1..29")
        if self.phase not in {
            "support-description",
            "atom-proposal",
            "support-scoring",
            "query-description",
            "query-scoring",
        }:
            raise AtomicSmokeRunError("unknown atomic smoke call phase")
        _prompt_text(self.prompt, "call prompt")
        if self.atom_ids != tuple(sorted(self.atom_ids)) or len(self.atom_ids) != len(
            set(self.atom_ids)
        ):
            raise AtomicSmokeRunError("call atom IDs must be unique digest order")
        for atom_id in self.atom_ids:
            _digest(atom_id, "call atom ID")
        schema = _clone_json(_thaw_json(self.output_schema), "call output schema")
        payload = _clone_json(_thaw_json(self.payload), "call payload")
        receipt = _clone_json(_thaw_json(self.receipt), "call receipt")
        validate_codex_strict_output_schema(schema)
        validate_codex_receipt(receipt)
        if receipt["structured_output_digest"] != _payload_digest(payload):
            raise AtomicSmokeRunError("call receipt does not bind its payload")
        if self.domain == "text":
            if any(
                item is not None
                for item in (
                    self.panel_id,
                    self.image_name,
                    self.image_digest,
                    self.image_byte_count,
                )
            ):
                raise AtomicSmokeRunError("text call contains an image binding")
            validate_codex_text_receipt(receipt, self.prompt, schema)
        elif self.domain == "named-image":
            if (
                not isinstance(self.panel_id, str)
                or _PANEL_ID.fullmatch(self.panel_id) is None
                or self.image_name != "panel.png"
                or self.image_digest is None
                or self.image_byte_count is None
            ):
                raise AtomicSmokeRunError("named-image call lacks its neutral image binding")
            _digest(self.image_digest, "call image digest")
            if type(self.image_byte_count) is not int or self.image_byte_count <= 0:
                raise AtomicSmokeRunError("call image byte count must be positive")
            _validate_cold_named_envelope(
                receipt,
                prompt=self.prompt,
                output_schema=schema,
                image_name=self.image_name,
                image_digest=self.image_digest,
                image_byte_count=self.image_byte_count,
            )
        else:
            raise AtomicSmokeRunError("call domain must be text or named-image")
        _digest(self.call_digest, "call digest")
        if self.call_digest != canonical_digest(self.content_data()):
            raise AtomicSmokeRunError("call digest differs from its exact envelope")

    def content_data(self) -> dict[str, object]:
        return {
            "schema": ATOMIC_SMOKE_CALL_SCHEMA,
            "ordinal": self.ordinal,
            "phase": self.phase,
            "domain": self.domain,
            "panel_id": self.panel_id,
            "image_name": self.image_name,
            "image_digest": self.image_digest,
            "image_byte_count": self.image_byte_count,
            "atom_ids": list(self.atom_ids),
            "prompt": self.prompt,
            "output_schema": _thaw_json(self.output_schema),
            "payload": _thaw_json(self.payload),
            "receipt": _thaw_json(self.receipt),
        }

    def to_data(self) -> dict[str, object]:
        return {**self.content_data(), "call_digest": self.call_digest}

    @classmethod
    def capture(
        cls,
        *,
        ordinal: int,
        phase: str,
        panel_id: str | None,
        atom_ids: Sequence[str],
        prompt: str,
        output_schema: Mapping[str, Any],
        result: CodexStructuredResult,
        image_payload: bytes | None = None,
    ) -> "AtomicSmokeCallRecord":
        if not isinstance(result, CodexStructuredResult):
            raise TypeError("transport must return CodexStructuredResult")
        schema = _clone_json(output_schema, "call output schema")
        payload = _clone_json(result.payload, "call payload")
        receipt = _receipt_data(result.receipt)
        domain = "text" if image_payload is None else "named-image"
        values = {
            "ordinal": ordinal,
            "phase": phase,
            "domain": domain,
            "panel_id": panel_id,
            "image_name": None if image_payload is None else "panel.png",
            "image_digest": (
                None if image_payload is None else hashlib.sha256(image_payload).hexdigest()
            ),
            "image_byte_count": None if image_payload is None else len(image_payload),
            "atom_ids": tuple(sorted(atom_ids)),
            "prompt": prompt,
            "output_schema": _freeze_json(schema),
            "payload": _freeze_json(payload),
            "receipt": _freeze_json(receipt),
        }
        content = {
            "schema": ATOMIC_SMOKE_CALL_SCHEMA,
            **{
                key: (
                    list(value)
                    if key == "atom_ids"
                    else _thaw_json(value)
                    if key in {"output_schema", "payload", "receipt"}
                    else value
                )
                for key, value in values.items()
            },
        }
        return cls(**values, call_digest=canonical_digest(content))  # type: ignore[arg-type]

    @classmethod
    def from_data(cls, value: Mapping[str, Any]) -> "AtomicSmokeCallRecord":
        data = _mapping(
            value,
            frozenset(
                {
                    "schema",
                    "ordinal",
                    "phase",
                    "domain",
                    "panel_id",
                    "image_name",
                    "image_digest",
                    "image_byte_count",
                    "atom_ids",
                    "prompt",
                    "output_schema",
                    "payload",
                    "receipt",
                    "call_digest",
                }
            ),
            "atomic smoke call",
        )
        if data["schema"] != ATOMIC_SMOKE_CALL_SCHEMA or not isinstance(
            data["atom_ids"], list
        ):
            raise AtomicSmokeRunError("unsupported or malformed atomic smoke call")
        result = cls(
            ordinal=data["ordinal"],
            phase=data["phase"],
            domain=data["domain"],
            panel_id=data["panel_id"],
            image_name=data["image_name"],
            image_digest=data["image_digest"],
            image_byte_count=data["image_byte_count"],
            atom_ids=tuple(data["atom_ids"]),
            prompt=data["prompt"],
            output_schema=_freeze_json(_clone_json(data["output_schema"], "schema")),  # type: ignore[arg-type]
            payload=_freeze_json(_clone_json(data["payload"], "payload")),  # type: ignore[arg-type]
            receipt=_freeze_json(_clone_json(data["receipt"], "receipt")),  # type: ignore[arg-type]
            call_digest=data["call_digest"],
        )
        if result.to_data() != _clone_json(value, "atomic smoke call"):
            raise AtomicSmokeRunError("atomic smoke call is not canonical")
        return result


_SUCCESS_PHASES = (
    ("support-description",) * 12
    + ("atom-proposal",)
    + ("support-scoring",) * 12
    + ("query-description",) * 2
    + ("query-scoring",) * 2
)


def _validate_call_prefix(calls: Sequence[AtomicSmokeCallRecord]) -> None:
    if len(calls) > ATOMIC_SMOKE_SUCCESS_CALL_COUNT:
        raise AtomicSmokeRunError("atomic smoke has more than 29 successful calls")
    for index, call in enumerate(calls):
        if call.ordinal != index + 1 or call.phase != _SUCCESS_PHASES[index]:
            raise AtomicSmokeRunError("atomic smoke calls are not an exact schedule prefix")
        expected_domain = "text" if index == 12 else "named-image"
        if call.domain != expected_domain:
            raise AtomicSmokeRunError("atomic smoke call uses the wrong transport domain")
    for start, end in ((0, min(12, len(calls))), (13, min(25, len(calls)))):
        for index in range(start, end):
            expected = f"support-panel-{index - start:02d}"
            if calls[index].panel_id != expected:
                raise AtomicSmokeRunError("support call panel order differs")
    for index in range(25, min(27, len(calls))):
        if calls[index].panel_id != f"query-{index - 25}":
            raise AtomicSmokeRunError("query description order differs")
    for index in range(27, min(29, len(calls))):
        if calls[index].panel_id != f"query-{index - 27}":
            raise AtomicSmokeRunError("query scoring order differs")


def _append_call(
    calls: list[AtomicSmokeCallRecord], call: AtomicSmokeCallRecord
) -> None:
    for field, label in (
        ("receipt_digest", "receipt"),
        ("thread_id", "Codex thread"),
        ("event_stream_digest", "Codex event stream"),
    ):
        if any(previous.receipt[field] == call.receipt[field] for previous in calls):
            raise AtomicSmokeRunError(
                f"atomic smoke reused a {label}; rerolls/replays are forbidden"
            )
    _validate_call_prefix((*calls, call))
    calls.append(call)


def _description_binding_from_call(
    call: AtomicSmokeCallRecord,
    *,
    phase: str,
    run_commitment_digest: str,
) -> PanelDescriptionBinding:
    if call.panel_id is None or call.image_digest is None:
        raise AtomicSmokeRunError("description call has no panel identity")
    return PanelDescriptionBinding.create(
        call.panel_id,
        call.image_digest,
        _description(call.payload),
        phase=phase,
        description_protocol_digest=atomic_smoke_description_protocol_digest(),
        validated_receipt_digest=call.receipt["receipt_digest"],
        run_commitment_digest=run_commitment_digest,
        call_ordinal=call.ordinal,
    )


def _validate_description_call_binding(
    call: AtomicSmokeCallRecord,
    binding: PanelDescriptionBinding,
    *,
    phase: str,
    run_commitment_digest: str,
) -> None:
    reproduced = _description_binding_from_call(
        call, phase=phase, run_commitment_digest=run_commitment_digest
    )
    if (
        call.domain != "named-image"
        or call.panel_id != binding.panel_id
        or call.prompt != _DESCRIPTION_PROMPT
        or canonical_json(_thaw_json(call.output_schema))
        != canonical_json(_description_schema())
        or call.image_name != "panel.png"
        or call.image_digest != binding.panel_digest
        or _description(call.payload) != binding.description
        or reproduced.to_data() != binding.to_data()
    ):
        raise AtomicSmokeRunError(
            "panel description binding differs from its neutral description call"
        )


def _authorized_evidence_binding(
    atom: AtomicSoftPredicate,
    binding: PanelDescriptionBinding,
    call: AtomicSmokeCallRecord,
    run_id: str,
) -> AtomicEvidenceBinding:
    return AtomicEvidenceBinding(
        atom_id=atom.atom_id,
        panel_digest=binding.panel_digest,
        panel_description_digest=binding.description_digest,
        scorer_protocol_digest=atomic_smoke_scorer_protocol_digest(),
        run_commitment_digest=binding.run_commitment_digest,
        scorer_producer="headless-codex-atomic-soft-scorer",
        scorer_version="1",
        scorer_method="operational-uncalibrated-four-disposition",
        scorer_run_id=run_id,
        scorer_receipt_digest=call.receipt["receipt_digest"],
        scorer_output_digest=call.receipt["structured_output_digest"],
        scorer_call_digest=call.call_digest,
        scorer_call_ordinal=call.ordinal,
        observation_scope=OPERATIONAL_SELECTION_SCOPE,
        calibration_digest=None,
    )


def _validate_authorized_scorer_record(
    record: TruthEvidenceRecord | OperationalNonmatchRecord,
    *,
    call: AtomicSmokeCallRecord,
    run_id: str,
) -> None:
    provenance = record.provenance
    if (
        provenance.producer != "headless-codex-atomic-soft-scorer"
        or provenance.version != "1"
        or provenance.method != "operational-uncalibrated-four-disposition"
        or provenance.run_id != run_id
        or provenance.artifact_digest != call.receipt["receipt_digest"]
        or len(provenance.details) != 5
        or provenance.details[1][1] != call.call_digest
        or provenance.details[2][1] != str(call.ordinal)
        or provenance.details[3][1]
        != call.receipt["structured_output_digest"]
        or provenance.details[4][1] != OPERATIONAL_SELECTION_SCOPE
    ):
        raise AtomicSmokeRunError("scorer evidence provenance is not authorized")


def _validate_scorer_call_and_records(
    *,
    call: AtomicSmokeCallRecord,
    binding: PanelDescriptionBinding,
    atoms: Sequence[AtomicSoftPredicate],
    records: Mapping[str, TruthEvidenceRecord | OperationalNonmatchRecord],
    run_id: str,
) -> None:
    expected_ids = tuple(atom.atom_id for atom in atoms)
    if (
        call.atom_ids != expected_ids
        or call.prompt != _scorer_prompt(binding, atoms)
        or canonical_json(_thaw_json(call.output_schema))
        != canonical_json(_scorer_schema())
        or call.image_digest != binding.panel_digest
        or call.panel_id != binding.panel_id
        or set(records) != set(expected_ids)
    ):
        raise AtomicSmokeRunError("scorer call differs from its frozen atom/panel inputs")
    raw = _mapping(
        call.payload, frozenset({"results"}), "authorized scorer payload"
    )["results"]
    if not isinstance(raw, (list, tuple)) or len(raw) != len(atoms):
        raise AtomicSmokeRunError("authorized scorer payload is incomplete")
    payload_by_id: dict[str, Mapping[str, Any]] = {}
    for item in raw:
        parsed = _mapping(
            item,
            frozenset({"atom_id", "disposition", "explanation"}),
            "authorized scorer result",
        )
        if parsed["atom_id"] in payload_by_id:
            raise AtomicSmokeRunError("authorized scorer payload repeats an atom")
        payload_by_id[parsed["atom_id"]] = parsed
    if set(payload_by_id) != set(expected_ids):
        raise AtomicSmokeRunError("authorized scorer payload atom IDs differ")
    atoms_by_id = {atom.atom_id: atom for atom in atoms}
    for atom_id, record in records.items():
        _validate_authorized_scorer_record(record, call=call, run_id=run_id)
        authorized = _authorized_evidence_binding(
            atoms_by_id[atom_id], binding, call, run_id
        )
        if (
            record.provenance.input_digests != authorized.input_digests
            or record.provenance.details != authorized.provenance_details
        ):
            raise AtomicSmokeRunError(
                "scorer record differs from exact atom/panel authorization"
            )
        model_value = payload_by_id[atom_id]["disposition"]
        expected_disposition = {
            "present": Disposition.PRESENT,
            "operational_nonmatch": Disposition.INDETERMINATE,
            "indeterminate": Disposition.INDETERMINATE,
            "error": Disposition.ERROR,
        }.get(model_value)
        if expected_disposition is None or record.disposition is not expected_disposition:
            raise AtomicSmokeRunError("scorer evidence differs from model vocabulary")
        explanation = payload_by_id[atom_id]["explanation"]
        if (
            model_value == "operational_nonmatch"
            and (
                not isinstance(record, OperationalNonmatchRecord)
                or record.reason != explanation
            )
        ) or (
            model_value == "indeterminate"
            and (
                not isinstance(record, TruthEvidenceRecord)
                or record.reason != explanation
            )
        ) or (
            model_value == "error"
            and (
                not isinstance(record, TruthEvidenceRecord)
                or record.reason != explanation
                or record.error_type != "ObserverReportedError"
            )
        ):
            raise AtomicSmokeRunError("scorer evidence text differs from exact output")


def _atomic_record_from_data(
    value: Mapping[str, Any],
) -> TruthEvidenceRecord | OperationalNonmatchRecord:
    if value.get("disposition") == "operational_nonmatch":
        return OperationalNonmatchRecord.from_data(value)
    return TruthEvidenceRecord.from_data(value)


def _atomic_record_from_evidence(
    evidence: Evidence[bool],
) -> TruthEvidenceRecord | OperationalNonmatchRecord:
    if evidence.is_operational_nonmatch:
        return OperationalNonmatchRecord.from_evidence(evidence)
    return TruthEvidenceRecord.from_evidence(evidence)


def _validate_prediction_commitment(
    value: Mapping[str, Any],
    archive: AtomicSelectionArchive,
    *,
    calls: Sequence[AtomicSmokeCallRecord],
    run_id: str,
    precommit_digest: str,
) -> dict[str, Any]:
    data = _mapping(
        value,
        frozenset(
            {
                "schema",
                "run_id",
                "precommit_digest",
                "selection_archive_digest",
                "formula",
                "queries",
                "commitment_digest",
            }
        ),
        "atomic smoke prediction commitment",
    )
    if data["schema"] != ATOMIC_SMOKE_PREDICTION_SCHEMA:
        raise AtomicSmokeRunError("unsupported prediction commitment schema")
    _text(data["run_id"], "prediction run ID", maximum=128)
    _address(data["precommit_digest"], "prediction precommit digest")
    if data["run_id"] != run_id or data["precommit_digest"] != precommit_digest:
        raise AtomicSmokeRunError("prediction live/precommit identity differs")
    if data["selection_archive_digest"] != archive.archive_digest:
        raise AtomicSmokeRunError("predictions descend from another formula archive")
    if data["formula"] != archive.formula:
        raise AtomicSmokeRunError("prediction formula differs from frozen archive")
    raw_queries = data["queries"]
    if not isinstance(raw_queries, list) or len(raw_queries) != 2:
        raise AtomicSmokeRunError("prediction commitment requires two queries")
    expected_fields = frozenset(
        {
            "query_id",
            "panel_byte_count",
            "description_binding",
            "description_call_digest",
            "scoring_call_digest",
            "atom_evidence",
            "formula_evidence",
            "predicted_positive",
        }
    )
    query_ids: list[str] = []
    for query_index, raw in enumerate(raw_queries):
        query = _mapping(raw, expected_fields, "query prediction")
        query_id = query["query_id"]
        if not isinstance(query_id, str) or _PANEL_ID.fullmatch(query_id) is None:
            raise AtomicSmokeRunError("prediction query ID is invalid")
        query_ids.append(query_id)
        if type(query["panel_byte_count"]) is not int or query["panel_byte_count"] <= 0:
            raise AtomicSmokeRunError("prediction panel byte count is invalid")
        if not isinstance(query["description_binding"], Mapping):
            raise AtomicSmokeRunError("prediction description binding must be an object")
        binding = PanelDescriptionBinding.from_data(query["description_binding"])
        if binding.panel_id != query_id:
            raise AtomicSmokeRunError("prediction description belongs to another query")
        _digest(query["description_call_digest"], "description call digest")
        _digest(query["scoring_call_digest"], "scoring call digest")
        description_call = calls[25 + query_index]
        scoring_call = calls[27 + query_index]
        if (
            query["description_call_digest"] != description_call.call_digest
            or query["scoring_call_digest"] != scoring_call.call_digest
        ):
            raise AtomicSmokeRunError("query row call digests differ from calls 26..29")
        _validate_description_call_binding(
            description_call,
            binding,
            phase="query",
            run_commitment_digest=precommit_digest.removeprefix("sha256:"),
        )
        raw_evidence = query["atom_evidence"]
        if not isinstance(raw_evidence, list) or len(raw_evidence) != len(
            archive.selected_atom_ids
        ):
            raise AtomicSmokeRunError("query atom evidence is not formula-complete")
        evidence_by_atom: dict[str, Evidence[bool]] = {}
        evidence_records: dict[
            str, TruthEvidenceRecord | OperationalNonmatchRecord
        ] = {}
        bindings: dict[str, AtomicEvidenceBinding] = {}
        for expected_atom, raw_item in zip(
            archive.selected_atom_ids, raw_evidence, strict=True
        ):
            item = _mapping(
                raw_item, frozenset({"atom_id", "evidence"}), "query atom evidence"
            )
            if item["atom_id"] != expected_atom or not isinstance(
                item["evidence"], Mapping
            ):
                raise AtomicSmokeRunError("query atom evidence order differs")
            record = _atomic_record_from_data(item["evidence"])
            evidence_by_atom[expected_atom] = record.to_evidence()
            evidence_records[expected_atom] = record
        selected_atoms = tuple(
            atom for atom in archive.matrix.atoms if atom.atom_id in set(archive.selected_atom_ids)
        )
        atoms_by_id = {atom.atom_id: atom for atom in selected_atoms}
        bindings = {
            atom_id: _authorized_evidence_binding(
                atoms_by_id[atom_id], binding, scoring_call, run_id
            )
            for atom_id in archive.selected_atom_ids
        }
        _validate_scorer_call_and_records(
            call=scoring_call,
            binding=binding,
            atoms=selected_atoms,
            records=evidence_records,
            run_id=run_id,
        )
        combined = evaluate_atomic_formula(
            archive.formula,
            evidence_by_atom,
            provenance_bindings=bindings,
            selection_scope=OPERATIONAL_SELECTION_SCOPE,
        )
        formula_record = _atomic_record_from_evidence(combined)
        if not isinstance(query["formula_evidence"], Mapping) or (
            canonical_json(formula_record.to_data())
            != canonical_json(dict(query["formula_evidence"]))
        ):
            raise AtomicSmokeRunError("query formula evidence does not replay")
        predicted = (
            True
            if combined.disposition is Disposition.PRESENT
            else False
            if combined.is_operational_nonmatch
            else None
        )
        if query["predicted_positive"] is not predicted:
            raise AtomicSmokeRunError("query prediction differs from formula evidence")
    if query_ids != ["query-0", "query-1"]:
        raise AtomicSmokeRunError("prediction query IDs are not canonical")
    content = {key: data[key] for key in data if key != "commitment_digest"}
    if data["commitment_digest"] != canonical_digest(content):
        raise AtomicSmokeRunError("prediction commitment digest differs")
    return _clone_json(value, "prediction commitment")


def _validate_label_reveal(
    value: Mapping[str, Any], prediction_digest: str, run_id: str
) -> dict[str, Any]:
    data = _mapping(
        value,
        frozenset(
            {
                "schema",
                "run_id",
                "prediction_commitment_digest",
                "label_commitment_digest",
                "labels",
                "reveal_digest",
            }
        ),
        "atomic smoke label reveal",
    )
    if (
        data["schema"] != ATOMIC_SMOKE_LABEL_REVEAL_SCHEMA
        or data["run_id"] != run_id
        or data["prediction_commitment_digest"] != prediction_digest
    ):
        raise AtomicSmokeRunError("label reveal parent identity differs")
    _digest(data["label_commitment_digest"], "latent label commitment digest")
    labels = data["labels"]
    if not isinstance(labels, list) or len(labels) != 2:
        raise AtomicSmokeRunError("label reveal requires two labels")
    seen: list[str] = []
    for raw in labels:
        item = _mapping(raw, frozenset({"query_id", "positive"}), "revealed label")
        if type(item["positive"]) is not bool:
            raise AtomicSmokeRunError("revealed label must be a literal Boolean")
        seen.append(item["query_id"])
    if seen != ["query-0", "query-1"]:
        raise AtomicSmokeRunError("revealed label IDs are not canonical")
    content = {key: data[key] for key in data if key != "reveal_digest"}
    if data["reveal_digest"] != canonical_digest(content):
        raise AtomicSmokeRunError("label reveal digest differs")
    return _clone_json(value, "label reveal")


def _score_predictions(
    prediction: Mapping[str, Any], reveal: Mapping[str, Any]
) -> dict[str, object]:
    labels = {item["query_id"]: item["positive"] for item in reveal["labels"]}
    predictions = {
        item["query_id"]: item["predicted_positive"]
        for item in prediction["queries"]
    }
    determinate = sum(value is not None for value in predictions.values())
    correct = sum(
        value is not None and value is labels[query_id]
        for query_id, value in predictions.items()
    )
    return {
        "image_correct": correct,
        "image_total": 2,
        "puzzle_correct": correct == 2,
        "determinate": determinate,
        "abstentions": 2 - determinate,
    }


def _evidence_digest(
    calls: Sequence[AtomicSmokeCallRecord],
    archive_digest: str | None,
    prediction_digest: str | None,
    persistence_receipt_digest: str | None,
) -> str:
    return canonical_digest(
        {
            "schema": "gkm.bongard-atomic-smoke-evidence-set.v1",
            "call_digests": [call.call_digest for call in calls],
            "receipt_digests": [call.receipt["receipt_digest"] for call in calls],
            "selection_archive_digest": archive_digest,
            "prediction_commitment_digest": prediction_digest,
            "prediction_persistence_receipt_digest": persistence_receipt_digest,
        }
    )


@dataclass(frozen=True, slots=True)
class AtomicSmokeRun:
    """Complete or failed terminal artifact for one no-reroll attempt."""

    status: str
    terminal_phase: str
    precommit_digest: str
    precommit_public_data: Mapping[str, Any]
    run_id: str
    source_dependency_digest: str
    protocol_digest: str
    expected_launcher_digest: str
    model: str
    reasoning_effort: str
    calls: tuple[AtomicSmokeCallRecord, ...]
    selection_archive_data: Mapping[str, Any] | None
    selection_archive_digest: str | None
    prediction_commitment_data: Mapping[str, Any] | None
    prediction_commitment_digest: str | None
    prediction_persistence_receipt: PredictionPersistenceReceipt | None
    label_reveal_data: Mapping[str, Any] | None
    score: Mapping[str, Any] | None
    evidence_digest: str
    failure: Mapping[str, Any] | None
    run_digest: str
    dependence_design_authorized: bool = False
    calibration_authorized: bool = False
    benchmark_claim_authorized: bool = False
    official_test_authorized: bool = False
    exploratory_uncalibrated_nonmatch: bool = True

    def __post_init__(self) -> None:
        if self.status not in {"complete", "failed"}:
            raise AtomicSmokeRunError("run status must be complete or failed")
        _text(self.terminal_phase, "terminal phase", maximum=128)
        _address(self.precommit_digest, "run precommit digest")
        public_precommit = AtomicSmokePrecommit.from_data(
            _thaw_json(self.precommit_public_data)  # type: ignore[arg-type]
        )
        if (
            public_precommit.digest != self.precommit_digest
            or public_precommit.source_dependency_digest
            != self.source_dependency_digest
            or public_precommit.episode_public_data["run_id"] != self.run_id
        ):
            raise AtomicSmokeRunError("run differs from exact public precommit")
        _text(self.run_id, "run ID", maximum=128)
        _digest(self.source_dependency_digest, "source dependency digest")
        if self.protocol_digest != atomic_smoke_run_protocol_digest():
            raise AtomicSmokeRunError("run protocol digest differs")
        _digest(self.expected_launcher_digest, "expected launcher digest")
        _text(self.model, "model", maximum=128)
        _text(self.reasoning_effort, "reasoning effort", maximum=32)
        if any(
            getattr(self, name) is not False
            for name in (
                "dependence_design_authorized",
                "calibration_authorized",
                "benchmark_claim_authorized",
                "official_test_authorized",
            )
        ):
            raise AtomicSmokeRunError("scientific authorization flags are immutable false")
        if self.exploratory_uncalibrated_nonmatch is not True:
            raise AtomicSmokeRunError(
                "exploratory uncalibrated nonmatch flag is immutable true"
            )
        _validate_call_prefix(self.calls)
        if any(
            call.receipt["codex_launcher_digest"] != self.expected_launcher_digest
            or call.receipt["requested_model"] != self.model
            or call.receipt["requested_reasoning_effort"] != self.reasoning_effort
            for call in self.calls
        ):
            raise AtomicSmokeRunError("call receipt differs from external transport pins")
        cache_bindings = {
            call.receipt["cloud_config_bundle_cache_binding"] for call in self.calls
        }
        if len(cache_bindings) > 1:
            raise AtomicSmokeRunError("cloud policy cache binding changed between calls")
        for field, label in (
            ("receipt_digest", "receipt"),
            ("thread_id", "Codex thread"),
            ("event_stream_digest", "Codex event stream"),
        ):
            values = [call.receipt[field] for call in self.calls]
            if len(values) != len(set(values)):
                raise AtomicSmokeRunError(
                    f"atomic smoke reused a {label}; rerolls/replays are forbidden"
                )

        archive: AtomicSelectionArchive | None = None
        if self.selection_archive_data is None:
            if self.selection_archive_digest is not None:
                raise AtomicSmokeRunError("selection digest exists without archive")
        else:
            archive = AtomicSelectionArchive.from_data(
                _thaw_json(self.selection_archive_data)  # type: ignore[arg-type]
            )
            if self.selection_archive_digest != archive.archive_digest:
                raise AtomicSmokeRunError("selection archive digest differs")
            if archive.selection_scope != OPERATIONAL_SELECTION_SCOPE:
                raise AtomicSmokeRunError("runner archive is not operational-only")
            if len(self.calls) < 25:
                raise AtomicSmokeRunError("formula archive predates support matrix completion")
            if archive.matrix.source_proposal_digest != self.calls[12].call_digest:
                raise AtomicSmokeRunError("formula archive descends from another proposal")
            all_atom_ids = tuple(atom.atom_id for atom in archive.matrix.atoms)
            support_labels = dict(archive.support_labels)
            canonical_bindings = archive.matrix.atoms[0].panel_descriptions
            for index, binding in enumerate(canonical_bindings):
                _validate_description_call_binding(
                    self.calls[index],
                    binding,
                    phase="support",
                    run_commitment_digest=self.precommit_digest.removeprefix("sha256:"),
                )
            if (
                self.calls[12].prompt
                != _proposal_prompt(canonical_bindings, support_labels)
                or canonical_json(_thaw_json(self.calls[12].output_schema))
                != canonical_json(_proposal_schema())
            ):
                raise AtomicSmokeRunError("atom proposal call differs from frozen descriptions")
            replayed_atoms = _parse_atoms(
                self.calls[12].payload,
                proposal_digest=self.calls[12].call_digest,
                panel_bindings=canonical_bindings,
            )
            if tuple(atom.to_data() for atom in replayed_atoms) != tuple(
                atom.to_data() for atom in archive.matrix.atoms
            ):
                raise AtomicSmokeRunError("archived atoms differ from proposal payload")
            for index, call in enumerate(self.calls[13:25]):
                binding = canonical_bindings[index]
                records = {
                    atom.atom_id: archive.matrix.cell(
                        atom.atom_id, binding.panel_id
                    ).evidence
                    for atom in archive.matrix.atoms
                }
                _validate_scorer_call_and_records(
                    call=call,
                    binding=binding,
                    atoms=archive.matrix.atoms,
                    records=records,
                    run_id=self.run_id,
                )
                for atom in archive.matrix.atoms:
                    cell = archive.matrix.cell(atom.atom_id, binding.panel_id)
                    if cell.evidence_binding.to_data() != (
                        _authorized_evidence_binding(
                            atom, binding, call, self.run_id
                        ).to_data()
                    ):
                        raise AtomicSmokeRunError(
                            "support cell authorization differs from exact scorer call"
                        )
            for call in self.calls[25:]:
                if call.atom_ids != archive.selected_atom_ids:
                    raise AtomicSmokeRunError("query call differs from frozen formula atoms")

        prediction: dict[str, Any] | None = None
        if self.prediction_commitment_data is None:
            if self.prediction_commitment_digest is not None:
                raise AtomicSmokeRunError("prediction digest exists without commitment")
        else:
            if archive is None or len(self.calls) != 29:
                raise AtomicSmokeRunError("predictions exist before all query calls")
            prediction = _validate_prediction_commitment(
                _thaw_json(self.prediction_commitment_data),  # type: ignore[arg-type]
                archive,
                calls=self.calls,
                run_id=self.run_id,
                precommit_digest=self.precommit_digest,
            )
            if self.prediction_commitment_digest != prediction["commitment_digest"]:
                raise AtomicSmokeRunError("prediction commitment digest differs")
        if self.prediction_persistence_receipt is not None:
            if prediction is None or (
                self.prediction_persistence_receipt.prediction_commitment_digest
                != self.prediction_commitment_digest
            ):
                raise AtomicSmokeRunError(
                    "prediction persistence receipt lacks its exact commitment"
                )

        reveal: dict[str, Any] | None = None
        if self.label_reveal_data is not None:
            if prediction is None or self.prediction_commitment_digest is None:
                raise AtomicSmokeRunError("labels exist before predictions")
            reveal = _validate_label_reveal(
                _thaw_json(self.label_reveal_data),  # type: ignore[arg-type]
                self.prediction_commitment_digest,
                self.run_id,
            )
        if self.status == "complete":
            if (
                len(self.calls) != 29
                or archive is None
                or prediction is None
                or reveal is None
                or self.prediction_persistence_receipt is None
                or self.failure is not None
                or self.terminal_phase != "cold-replay-verified"
            ):
                raise AtomicSmokeRunError("complete run lacks its exact terminal chain")
            expected_score = _score_predictions(prediction, reveal)
            if _thaw_json(self.score) != expected_score:
                raise AtomicSmokeRunError("terminal score differs from predictions/labels")
        else:
            if self.failure is None or self.label_reveal_data is not None or self.score is not None:
                raise AtomicSmokeRunError("failed run has invalid terminal fields")
            failure = _mapping(
                _thaw_json(self.failure),
                frozenset({"phase", "error_type", "reason"}),
                "run failure",
            )
            for key in ("phase", "error_type", "reason"):
                _text(failure[key], f"failure {key}", maximum=2048)
        expected_evidence = _evidence_digest(
            self.calls,
            self.selection_archive_digest,
            self.prediction_commitment_digest,
            None
            if self.prediction_persistence_receipt is None
            else self.prediction_persistence_receipt.receipt_digest,
        )
        if self.evidence_digest != expected_evidence:
            raise AtomicSmokeRunError("run evidence digest differs")
        _digest(self.run_digest, "run digest")
        if self.run_digest != canonical_digest(self.content_data()):
            raise AtomicSmokeRunError("run digest differs from terminal artifact")

    @property
    def digest(self) -> str:
        return self.run_digest

    def content_data(self) -> dict[str, object]:
        return {
            "schema": ATOMIC_SMOKE_RUN_SCHEMA,
            "status": self.status,
            "terminal_phase": self.terminal_phase,
            "precommit_digest": self.precommit_digest,
            "precommit_public_data": _thaw_json(self.precommit_public_data),
            "run_id": self.run_id,
            "source_dependency_digest": self.source_dependency_digest,
            "protocol_digest": self.protocol_digest,
            "expected_launcher_digest": self.expected_launcher_digest,
            "model": self.model,
            "reasoning_effort": self.reasoning_effort,
            "calls": [call.to_data() for call in self.calls],
            "selection_archive_data": _thaw_json(self.selection_archive_data),
            "selection_archive_digest": self.selection_archive_digest,
            "prediction_commitment_data": _thaw_json(self.prediction_commitment_data),
            "prediction_commitment_digest": self.prediction_commitment_digest,
            "prediction_persistence_receipt": (
                None
                if self.prediction_persistence_receipt is None
                else self.prediction_persistence_receipt.to_data()
            ),
            "label_reveal_data": _thaw_json(self.label_reveal_data),
            "score": _thaw_json(self.score),
            "evidence_digest": self.evidence_digest,
            "failure": _thaw_json(self.failure),
            "dependence_design_authorized": self.dependence_design_authorized,
            "calibration_authorized": self.calibration_authorized,
            "benchmark_claim_authorized": self.benchmark_claim_authorized,
            "official_test_authorized": self.official_test_authorized,
            "exploratory_uncalibrated_nonmatch": (
                self.exploratory_uncalibrated_nonmatch
            ),
        }

    def to_data(self) -> dict[str, object]:
        return {**self.content_data(), "run_digest": self.run_digest}

    @classmethod
    def from_data(cls, value: Mapping[str, Any]) -> "AtomicSmokeRun":
        public = {
            "status",
            "terminal_phase",
            "precommit_digest",
            "run_id",
            "source_dependency_digest",
            "protocol_digest",
            "expected_launcher_digest",
            "model",
            "reasoning_effort",
            "selection_archive_digest",
            "prediction_commitment_digest",
            "evidence_digest",
            "dependence_design_authorized",
            "calibration_authorized",
            "benchmark_claim_authorized",
            "official_test_authorized",
            "exploratory_uncalibrated_nonmatch",
            "run_digest",
        }
        data = _mapping(
            value,
            frozenset(
                {
                    "schema",
                    "calls",
                    "precommit_public_data",
                    "selection_archive_data",
                    "prediction_commitment_data",
                    "prediction_persistence_receipt",
                    "label_reveal_data",
                    "score",
                    "failure",
                    *public,
                }
            ),
            "atomic smoke run",
        )
        if data["schema"] != ATOMIC_SMOKE_RUN_SCHEMA or not isinstance(
            data["calls"], list
        ):
            raise AtomicSmokeRunError("unsupported or malformed atomic smoke run")
        def frozen_optional(name: str) -> Mapping[str, Any] | None:
            raw = data[name]
            if raw is None:
                return None
            if not isinstance(raw, Mapping):
                raise AtomicSmokeRunError(f"{name} must be an object or null")
            return _freeze_json(_clone_json(raw, name))  # type: ignore[return-value]

        values = {name: data[name] for name in public}
        if not isinstance(data["precommit_public_data"], Mapping):
            raise AtomicSmokeRunError("precommit_public_data must be an object")
        persistence_raw = data["prediction_persistence_receipt"]
        if persistence_raw is not None and not isinstance(persistence_raw, Mapping):
            raise AtomicSmokeRunError(
                "prediction_persistence_receipt must be an object or null"
            )
        result = cls(
            **values,
            calls=tuple(AtomicSmokeCallRecord.from_data(item) for item in data["calls"]),
            precommit_public_data=_freeze_json(
                _clone_json(data["precommit_public_data"], "precommit public data")
            ),  # type: ignore[arg-type]
            selection_archive_data=frozen_optional("selection_archive_data"),
            prediction_commitment_data=frozen_optional("prediction_commitment_data"),
            prediction_persistence_receipt=(
                None
                if persistence_raw is None
                else PredictionPersistenceReceipt.from_data(persistence_raw)
            ),
            label_reveal_data=frozen_optional("label_reveal_data"),
            score=frozen_optional("score"),
            failure=frozen_optional("failure"),
        )  # type: ignore[arg-type]
        if result.to_data() != _clone_json(value, "atomic smoke run"):
            raise AtomicSmokeRunError("atomic smoke run is not canonical")
        return result


def _description(payload: Mapping[str, Any]) -> str:
    data = _mapping(payload, frozenset({"description"}), "panel description payload")
    return _text(data["description"], "panel description", maximum=384)


def _proposal_prompt(
    bindings: Sequence[PanelDescriptionBinding], labels: Mapping[str, bool]
) -> str:
    support = [
        {
            "panel_id": binding.panel_id,
            "label": "positive" if labels[binding.panel_id] else "negative",
            "description": binding.description,
            "description_digest": binding.description_digest,
        }
        for binding in bindings
    ]
    return (
        "Using only the frozen labelled panel descriptions below, propose 1 to "
        "12 reusable affirmative one-cue visual atoms. Each atom must be "
        "single-panel checkable, contain no negation or class-relative wording, "
        "and contain no alternative joined by or/either. Do not combine cues, "
        "write code, flip polarity, or inspect images. phrase is the one exact "
        "affirmative observer question used everywhere downstream.\n"
        + canonical_json({"support": support}).decode("utf-8")
    )


def _parse_atoms(
    payload: Mapping[str, Any],
    *,
    proposal_digest: str,
    panel_bindings: Sequence[PanelDescriptionBinding],
) -> tuple[AtomicSoftPredicate, ...]:
    data = _mapping(payload, frozenset({"atoms"}), "atom proposal payload")
    raw_atoms = data["atoms"]
    if not isinstance(raw_atoms, (list, tuple)) or not 1 <= len(raw_atoms) <= ATOMIC_SMOKE_MAX_ATOMS:
        raise AtomicSmokeRunError("atom proposal must contain 1..12 atoms")
    atoms: list[AtomicSoftPredicate] = []
    for raw in raw_atoms:
        item = _mapping(
            raw,
            frozenset({"phrase"}),
            "proposed atom",
        )
        atoms.append(
            AtomicSoftPredicate.create(
                source_proposal_digest=proposal_digest,
                scorer_protocol_digest=atomic_smoke_scorer_protocol_digest(),
                positive_description=item["phrase"],
                cue_description=item["phrase"],
                panel_descriptions=panel_bindings,
            )
        )
    result = tuple(sorted(atoms, key=lambda atom: atom.atom_id))
    if len({atom.atom_id for atom in result}) != len(result):
        raise AtomicSmokeRunError("atom proposal contains duplicate atoms")
    return result


def _scorer_prompt(
    binding: PanelDescriptionBinding, atoms: Sequence[AtomicSoftPredicate]
) -> str:
    return (
        _SCORER_INSTRUCTIONS
        + "\n"
        + canonical_json(
            {
                "panel": {
                    "panel_id": binding.panel_id,
                    "description": binding.description,
                    "description_digest": binding.description_digest,
                    "description_protocol_digest": (
                        atomic_smoke_description_protocol_digest()
                    ),
                },
                "atoms": [
                    {
                        "atom_id": atom.atom_id,
                        "phrase": atom.positive_description,
                    }
                    for atom in atoms
                ],
            }
        ).decode("utf-8")
    )


def _parse_scorer_evidence(
    payload: Mapping[str, Any],
    *,
    atoms: Sequence[AtomicSoftPredicate],
    binding: PanelDescriptionBinding,
    call: AtomicSmokeCallRecord,
    run_id: str,
) -> dict[str, Evidence[bool]]:
    data = _mapping(payload, frozenset({"results"}), "scorer payload")
    raw_results = data["results"]
    if not isinstance(raw_results, (list, tuple)) or len(raw_results) != len(atoms):
        raise AtomicSmokeRunError("scorer result does not cover every frozen atom")
    by_id: dict[str, Mapping[str, Any]] = {}
    for raw in raw_results:
        item = _mapping(
            raw,
            frozenset({"atom_id", "disposition", "explanation"}),
            "scorer result",
        )
        atom_id = _digest(item["atom_id"], "scored atom ID")
        if atom_id in by_id:
            raise AtomicSmokeRunError("scorer returned an atom twice")
        _text(item["explanation"], "scorer explanation", maximum=2048)
        by_id[atom_id] = item
    expected = {atom.atom_id for atom in atoms}
    if set(by_id) != expected:
        raise AtomicSmokeRunError("scorer atom IDs differ from frozen atoms")
    evidence: dict[str, Evidence[bool]] = {}
    for atom in atoms:
        item = by_id[atom.atom_id]
        authorization = _authorized_evidence_binding(atom, binding, call, run_id)
        provenance = Provenance(
            producer=authorization.scorer_producer,
            version=authorization.scorer_version,
            method=authorization.scorer_method,
            input_digests=authorization.input_digests,
            artifact_digest=authorization.scorer_receipt_digest,
            run_id=authorization.scorer_run_id,
            details=authorization.provenance_details,
        )
        disposition_name = item["disposition"]
        explanation = item["explanation"]
        if disposition_name == "present":
            observed = Evidence.present(True, provenance)
        elif disposition_name == "operational_nonmatch":
            observed = Evidence.operational_nonmatch(provenance, explanation)
        elif disposition_name == "indeterminate":
            observed = Evidence.indeterminate(provenance, explanation)
        elif disposition_name == "error":
            observed = Evidence.error(provenance, "ObserverReportedError", explanation)
        else:
            raise AtomicSmokeRunError("scorer used vocabulary outside its static schema")
        evidence[atom.atom_id] = observed
    return evidence


NamedImageTransport = Callable[..., CodexStructuredResult]
TextTransport = Callable[..., CodexStructuredResult]


def _invoke_named(
    *,
    ordinal: int,
    phase: str,
    panel_id: str,
    atom_ids: Sequence[str],
    prompt: str,
    schema: Mapping[str, Any],
    source: Any,
    transport: NamedImageTransport,
    model: str,
    reasoning_effort: str,
    minutes: int,
    executable: str,
    cloud_policy_cache_snapshot: CloudPolicyCacheSnapshot | None,
    expected_launcher_digest: str,
    verbose: bool,
) -> AtomicSmokeCallRecord:
    before = source.read_verified()
    result = transport(
        prompt,
        (str(source.path),),
        ("panel.png",),
        schema,
        model=model,
        reasoning_effort=reasoning_effort,
        minutes=minutes,
        verbose=verbose,
        executable=executable,
        cloud_policy_cache_snapshot=cloud_policy_cache_snapshot,
        expected_launcher_digest=expected_launcher_digest,
    )
    validate_codex_named_image_receipt(
        result.receipt,
        prompt,
        (str(source.path),),
        ("panel.png",),
        schema,
        payload=result.payload,
    )
    after = source.read_verified()
    if before != after or hashlib.sha256(before).hexdigest() != source.panel.sha256:
        raise AtomicSmokeRunError("panel bytes changed across isolated Codex call")
    return AtomicSmokeCallRecord.capture(
        ordinal=ordinal,
        phase=phase,
        panel_id=panel_id,
        atom_ids=atom_ids,
        prompt=prompt,
        output_schema=schema,
        result=result,
        image_payload=before,
    )


def _invoke_text(
    *,
    ordinal: int,
    prompt: str,
    schema: Mapping[str, Any],
    transport: TextTransport,
    model: str,
    reasoning_effort: str,
    minutes: int,
    executable: str,
    cloud_policy_cache_snapshot: CloudPolicyCacheSnapshot | None,
    expected_launcher_digest: str,
    verbose: bool,
) -> AtomicSmokeCallRecord:
    result = transport(
        prompt,
        schema,
        model=model,
        reasoning_effort=reasoning_effort,
        minutes=minutes,
        verbose=verbose,
        executable=executable,
        cloud_policy_cache_snapshot=cloud_policy_cache_snapshot,
        expected_launcher_digest=expected_launcher_digest,
    )
    validate_codex_text_receipt(result.receipt.to_dict(), prompt, schema)
    return AtomicSmokeCallRecord.capture(
        ordinal=ordinal,
        phase="atom-proposal",
        panel_id=None,
        atom_ids=(),
        prompt=prompt,
        output_schema=schema,
        result=result,
    )


def _prediction_commitment(
    *,
    precommit: AtomicSmokePrecommit,
    archive: AtomicSelectionArchive,
    query_rows: Sequence[Mapping[str, Any]],
    calls: Sequence[AtomicSmokeCallRecord],
) -> dict[str, Any]:
    content = {
        "schema": ATOMIC_SMOKE_PREDICTION_SCHEMA,
        "run_id": precommit.episode_plan.run_id,
        "precommit_digest": precommit.digest,
        "selection_archive_digest": archive.archive_digest,
        "formula": archive.formula,
        "queries": [_clone_json(item, "query prediction") for item in query_rows],
    }
    value = {**content, "commitment_digest": canonical_digest(content)}
    return _validate_prediction_commitment(
        value,
        archive,
        calls=calls,
        run_id=precommit.episode_plan.run_id,
        precommit_digest=precommit.digest,
    )


def _label_reveal(
    precommit: AtomicSmokePrecommit, prediction_digest: str
) -> dict[str, Any]:
    plan = precommit.episode_plan
    labels = [item.to_data() for item in sorted(plan._revealed_labels())]
    content = {
        "schema": ATOMIC_SMOKE_LABEL_REVEAL_SCHEMA,
        "run_id": plan.run_id,
        "prediction_commitment_digest": prediction_digest,
        "label_commitment_digest": plan.label_commitment_digest,
        "labels": labels,
    }
    value = {**content, "reveal_digest": canonical_digest(content)}
    return _validate_label_reveal(value, prediction_digest, plan.run_id)


def _failure_text(value: object) -> str:
    text = str(value) if str(value).strip() else "atomic smoke failed"
    text = "".join(
        " " if unicodedata.category(character) in {"Cc", "Cf"} else character
        for character in text
    )
    text = " ".join(text.split())
    text = unicodedata.normalize("NFKC", text)
    return text[:2048] or "atomic smoke failed"


def _make_terminal(
    *,
    status: str,
    terminal_phase: str,
    precommit: AtomicSmokePrecommit,
    source_dependency_digest: str,
    protocol_digest: str,
    expected_launcher_digest: str,
    model: str,
    reasoning_effort: str,
    calls: Sequence[AtomicSmokeCallRecord],
    archive: AtomicSelectionArchive | None,
    prediction: Mapping[str, Any] | None,
    persistence_receipt: PredictionPersistenceReceipt | None,
    reveal: Mapping[str, Any] | None,
    score: Mapping[str, Any] | None,
    failure: Mapping[str, Any] | None,
) -> AtomicSmokeRun:
    archive_data = None if archive is None else archive.to_data()
    archive_digest = None if archive is None else archive.archive_digest
    prediction_data = None if prediction is None else _clone_json(prediction, "prediction")
    prediction_digest = (
        None if prediction is None else prediction["commitment_digest"]
    )
    values: dict[str, Any] = {
        "status": status,
        "terminal_phase": terminal_phase,
        "precommit_digest": precommit.digest,
        "precommit_public_data": _freeze_json(precommit.to_data()),
        "run_id": precommit.episode_plan.run_id,
        "source_dependency_digest": source_dependency_digest,
        "protocol_digest": protocol_digest,
        "expected_launcher_digest": expected_launcher_digest,
        "model": model,
        "reasoning_effort": reasoning_effort,
        "calls": tuple(calls),
        "selection_archive_data": (
            None if archive_data is None else _freeze_json(archive_data)
        ),
        "selection_archive_digest": archive_digest,
        "prediction_commitment_data": (
            None if prediction_data is None else _freeze_json(prediction_data)
        ),
        "prediction_commitment_digest": prediction_digest,
        "prediction_persistence_receipt": persistence_receipt,
        "label_reveal_data": None if reveal is None else _freeze_json(reveal),
        "score": None if score is None else _freeze_json(score),
        "evidence_digest": _evidence_digest(
            calls,
            archive_digest,
            prediction_digest,
            None if persistence_receipt is None else persistence_receipt.receipt_digest,
        ),
        "failure": None if failure is None else _freeze_json(failure),
        "dependence_design_authorized": False,
        "calibration_authorized": False,
        "benchmark_claim_authorized": False,
        "official_test_authorized": False,
        "exploratory_uncalibrated_nonmatch": True,
    }
    content = {
        "schema": ATOMIC_SMOKE_RUN_SCHEMA,
        **{
            key: (
                [item.to_data() for item in value]
                if key == "calls"
                else value.to_data()
                if key == "prediction_persistence_receipt" and value is not None
                else _thaw_json(value)
                if key
                in {
                    "selection_archive_data",
                    "prediction_commitment_data",
                    "label_reveal_data",
                    "score",
                    "failure",
                    "precommit_public_data",
                }
                else value
            )
            for key, value in values.items()
        }
    }
    return AtomicSmokeRun(**values, run_digest=canonical_digest(content))


def run_atomic_smoke(
    precommit: AtomicSmokePrecommit,
    *,
    source_dependency_digest: str,
    expected_protocol_digest: str,
    expected_launcher_digest: str,
    prediction_store_dir: str | Path,
    model: str = DEFAULT_CODEX_MODEL,
    reasoning_effort: str = DEFAULT_REASONING_EFFORT,
    minutes: int = 15,
    executable: str = "codex",
    cloud_policy_cache_snapshot: CloudPolicyCacheSnapshot | None = None,
    named_image_transport: NamedImageTransport | None = None,
    text_transport: TextTransport | None = None,
    verbose: bool = False,
) -> AtomicSmokeRun:
    """Execute one exact no-reroll atomic smoke attempt.

    The runner exclusively creates (or verifies byte-identical) one
    content-addressed prediction file, fsyncs it and its directory, and reloads
    it through a stable no-follow descriptor before labels can be materialised.
    """

    if not isinstance(precommit, AtomicSmokePrecommit):
        raise TypeError("precommit must be AtomicSmokePrecommit")
    source_digest = _digest(source_dependency_digest, "source dependency digest")
    if precommit.source_dependency_digest != source_digest:
        raise AtomicSmokeRunError("precommit differs from source dependency pin")
    protocol_digest = _digest(expected_protocol_digest, "expected protocol digest")
    if protocol_digest != atomic_smoke_run_protocol_digest():
        raise AtomicSmokeRunError("external run protocol pin differs")
    launcher_digest = _digest(expected_launcher_digest, "expected launcher digest")
    # Resolve and authenticate the store before spending a model call.
    _store_path, store_descriptor = _open_prediction_store(prediction_store_dir)
    os.close(store_descriptor)
    named = named_image_transport or run_codex_named_images_structured
    text_only = text_transport or run_codex_text_structured
    if not callable(named) or not callable(text_only):
        raise TypeError("atomic smoke transports must be callable")

    plan = precommit.episode_plan
    if plan.split == "test":
        raise AtomicSmokeRunError("atomic smoke cannot run on official test")
    calls: list[AtomicSmokeCallRecord] = []
    archive: AtomicSelectionArchive | None = None
    prediction: dict[str, Any] | None = None
    persistence_receipt: PredictionPersistenceReceipt | None = None
    phase = "support-description"
    try:
        # Digest order erases EpisodePlan's side-grouped support order.  The
        # isolated vision turns receive only opaque IDs and one neutral name.
        support_sources = tuple(
            sorted(plan._support_sources, key=lambda item: (item.panel.sha256, item.panel.blob_id))
        )
        bindings: list[PanelDescriptionBinding] = []
        labels: dict[str, bool] = {}
        description_schema = _description_schema()
        for index, source in enumerate(support_sources):
            panel_id = f"support-panel-{index:02d}"
            call = _invoke_named(
                ordinal=len(calls) + 1,
                phase="support-description",
                panel_id=panel_id,
                atom_ids=(),
                prompt=_DESCRIPTION_PROMPT,
                schema=description_schema,
                source=source,
                transport=named,
                model=model,
                reasoning_effort=reasoning_effort,
                minutes=minutes,
                executable=executable,
                cloud_policy_cache_snapshot=cloud_policy_cache_snapshot,
                expected_launcher_digest=launcher_digest,
                verbose=verbose,
            )
            _append_call(calls, call)
            binding = _description_binding_from_call(
                call,
                phase="support",
                run_commitment_digest=precommit.digest.removeprefix("sha256:"),
            )
            if call.image_digest != binding.panel_digest:
                raise AtomicSmokeRunError("support call bytes differ from panel binding")
            bindings.append(binding)
            labels[panel_id] = source.positive

        phase = "atom-proposal"
        ordered_bindings = tuple(sorted(bindings, key=lambda item: item.panel_id))
        proposal_prompt = _proposal_prompt(ordered_bindings, labels)
        proposal_call = _invoke_text(
            ordinal=len(calls) + 1,
            prompt=proposal_prompt,
            schema=_proposal_schema(),
            transport=text_only,
            model=model,
            reasoning_effort=reasoning_effort,
            minutes=minutes,
            executable=executable,
            cloud_policy_cache_snapshot=cloud_policy_cache_snapshot,
            expected_launcher_digest=launcher_digest,
            verbose=verbose,
        )
        _append_call(calls, proposal_call)
        atoms = _parse_atoms(
            proposal_call.payload,
            proposal_digest=proposal_call.call_digest,
            panel_bindings=ordered_bindings,
        )
        all_atom_ids = tuple(atom.atom_id for atom in atoms)

        phase = "support-scoring"
        cells: list[AtomicSupportCell] = []
        bindings_by_id = {item.panel_id: item for item in ordered_bindings}
        source_by_id = {
            f"support-panel-{index:02d}": source
            for index, source in enumerate(support_sources)
        }
        for index in range(12):
            panel_id = f"support-panel-{index:02d}"
            binding = bindings_by_id[panel_id]
            call = _invoke_named(
                ordinal=len(calls) + 1,
                phase="support-scoring",
                panel_id=panel_id,
                atom_ids=all_atom_ids,
                prompt=_scorer_prompt(binding, atoms),
                schema=_scorer_schema(),
                source=source_by_id[panel_id],
                transport=named,
                model=model,
                reasoning_effort=reasoning_effort,
                minutes=minutes,
                executable=executable,
                cloud_policy_cache_snapshot=cloud_policy_cache_snapshot,
                expected_launcher_digest=launcher_digest,
                verbose=verbose,
            )
            _append_call(calls, call)
            observations = _parse_scorer_evidence(
                call.payload,
                atoms=atoms,
                binding=binding,
                call=call,
                run_id=plan.run_id,
            )
            cells.extend(
                AtomicSupportCell.capture(
                    atom,
                    panel_id,
                    observations[atom.atom_id],
                    evidence_binding=_authorized_evidence_binding(
                        atom, binding, call, plan.run_id
                    ),
                )
                for atom in atoms
            )
        matrix = AtomicSupportMatrix.create(atoms, cells)
        archive = synthesize_atomic_conjunction(
            matrix, labels, selection_scope=OPERATIONAL_SELECTION_SCOPE
        )
        selected_atoms = tuple(
            atom for atom in atoms if atom.atom_id in set(archive.selected_atom_ids)
        )

        # This is the first access to a query source/path in this runner.
        phase = "query-description"
        query_bindings: dict[str, PanelDescriptionBinding] = {}
        query_sources: dict[str, Any] = {}
        for query, source in zip(plan.queries, plan._query_sources, strict=True):
            call = _invoke_named(
                ordinal=len(calls) + 1,
                phase="query-description",
                panel_id=query.query_id,
                atom_ids=archive.selected_atom_ids,
                prompt=_DESCRIPTION_PROMPT,
                schema=description_schema,
                source=source,
                transport=named,
                model=model,
                reasoning_effort=reasoning_effort,
                minutes=minutes,
                executable=executable,
                cloud_policy_cache_snapshot=cloud_policy_cache_snapshot,
                expected_launcher_digest=launcher_digest,
                verbose=verbose,
            )
            _append_call(calls, call)
            binding = _description_binding_from_call(
                call,
                phase="query",
                run_commitment_digest=precommit.digest.removeprefix("sha256:"),
            )
            if call.image_digest != binding.panel_digest:
                raise AtomicSmokeRunError("query call bytes differ from panel binding")
            query_bindings[query.query_id] = binding
            query_sources[query.query_id] = source

        phase = "query-scoring"
        query_rows: list[dict[str, Any]] = []
        description_calls = {call.panel_id: call for call in calls[25:27]}
        for query in plan.queries:
            binding = query_bindings[query.query_id]
            call = _invoke_named(
                ordinal=len(calls) + 1,
                phase="query-scoring",
                panel_id=query.query_id,
                atom_ids=archive.selected_atom_ids,
                prompt=_scorer_prompt(binding, selected_atoms),
                schema=_scorer_schema(),
                source=query_sources[query.query_id],
                transport=named,
                model=model,
                reasoning_effort=reasoning_effort,
                minutes=minutes,
                executable=executable,
                cloud_policy_cache_snapshot=cloud_policy_cache_snapshot,
                expected_launcher_digest=launcher_digest,
                verbose=verbose,
            )
            _append_call(calls, call)
            evidence = _parse_scorer_evidence(
                call.payload,
                atoms=selected_atoms,
                binding=binding,
                call=call,
                run_id=plan.run_id,
            )
            provenance_bindings = {
                atom.atom_id: _authorized_evidence_binding(
                    atom, binding, call, plan.run_id
                )
                for atom in selected_atoms
            }
            combined = evaluate_atomic_formula(
                archive.formula,
                evidence,
                provenance_bindings=provenance_bindings,
                selection_scope=OPERATIONAL_SELECTION_SCOPE,
            )
            formula_record = _atomic_record_from_evidence(combined)
            predicted = (
                True
                if combined.disposition is Disposition.PRESENT
                else False
                if combined.is_operational_nonmatch
                else None
            )
            query_rows.append(
                {
                    "query_id": query.query_id,
                    "panel_byte_count": call.image_byte_count,
                    "description_binding": binding.to_data(),
                    "description_call_digest": description_calls[query.query_id].call_digest,
                    "scoring_call_digest": call.call_digest,
                    "atom_evidence": [
                        {
                            "atom_id": atom_id,
                            "evidence": _atomic_record_from_evidence(
                                evidence[atom_id]
                            ).to_data(),
                        }
                        for atom_id in archive.selected_atom_ids
                    ],
                    "formula_evidence": formula_record.to_data(),
                    "predicted_positive": predicted,
                }
            )

        phase = "prediction-persistence"
        prediction = _prediction_commitment(
            precommit=precommit,
            archive=archive,
            query_rows=query_rows,
            calls=calls,
        )
        reloaded, persistence_receipt = _persist_prediction_commitment(
            prediction, prediction_store_dir
        )
        reloaded = _validate_prediction_commitment(
            reloaded,
            archive,
            calls=calls,
            run_id=plan.run_id,
            precommit_digest=precommit.digest,
        )
        if reloaded != prediction:
            raise AtomicSmokeRunError(
                "reloaded prediction commitment differs across persistence boundary"
            )

        # Labels are materialised only after the exact prediction payload has
        # crossed the verified durability boundary above.
        phase = "label-reveal"
        reveal = _label_reveal(precommit, prediction["commitment_digest"])
        score = _score_predictions(prediction, reveal)
        terminal = _make_terminal(
            status="complete",
            terminal_phase="cold-replay-verified",
            precommit=precommit,
            source_dependency_digest=source_digest,
            protocol_digest=protocol_digest,
            expected_launcher_digest=launcher_digest,
            model=model,
            reasoning_effort=reasoning_effort,
            calls=calls,
            archive=archive,
            prediction=prediction,
            persistence_receipt=persistence_receipt,
            reveal=reveal,
            score=score,
            failure=None,
        )
        # Decode into detached JSON and replay before returning a success.
        return cold_decode_and_replay_atomic_smoke_run(
            terminal.to_data(),
            expected_run_digest=terminal.digest,
            expected_source_dependency_digest=source_digest,
            expected_precommit_digest=precommit.digest,
            expected_protocol_digest=protocol_digest,
            expected_launcher_digest=launcher_digest,
            expected_evidence_digest=terminal.evidence_digest,
            precommit_public_data=precommit.to_data(),
            label_seal_nonce=plan._label_nonce,
            prediction_store_dir=prediction_store_dir,
        )
    except Exception as exc:  # noqa: BLE001 - terminalize one no-reroll attempt.
        return _make_terminal(
            status="failed",
            terminal_phase=phase,
            precommit=precommit,
            source_dependency_digest=source_digest,
            protocol_digest=protocol_digest,
            expected_launcher_digest=launcher_digest,
            model=model,
            reasoning_effort=reasoning_effort,
            calls=calls,
            archive=archive,
            prediction=prediction,
            persistence_receipt=persistence_receipt,
            reveal=None,
            score=None,
            failure={
                "phase": phase,
                "error_type": type(exc).__name__,
                "reason": _failure_text(exc),
            },
        )


def cold_decode_and_replay_atomic_smoke_run(
    value: Mapping[str, Any],
    *,
    expected_run_digest: str,
    expected_source_dependency_digest: str,
    expected_precommit_digest: str,
    expected_protocol_digest: str,
    expected_launcher_digest: str,
    expected_evidence_digest: str,
    precommit_public_data: Mapping[str, Any],
    label_seal_nonce: str,
    prediction_store_dir: str | Path,
) -> AtomicSmokeRun:
    """Model-free replay under external precommit, nonce and evidence pins."""

    try:
        external_precommit = AtomicSmokePrecommit.from_data(precommit_public_data)
    except Exception as exc:  # noqa: BLE001 - normalize an untrusted archive boundary.
        raise AtomicSmokeRunError(
            "external precommit cannot be decoded"
        ) from exc
    expected_precommit = _address(
        expected_precommit_digest, "expected precommit digest"
    )
    if external_precommit.digest != expected_precommit:
        raise AtomicSmokeRunError("external precommit differs from exact digest pin")
    nonce = _digest(label_seal_nonce, "external label seal nonce")
    try:
        run = AtomicSmokeRun.from_data(value)
    except AtomicSmokeRunError:
        raise
    except Exception as exc:  # noqa: BLE001 - normalize nested archive failures.
        raise AtomicSmokeRunError("atomic smoke run cannot be decoded") from exc
    if run.digest != _digest(expected_run_digest, "expected run digest"):
        raise AtomicSmokeRunError("run differs from external run digest")
    if run.source_dependency_digest != _digest(
        expected_source_dependency_digest, "expected source dependency digest"
    ):
        raise AtomicSmokeRunError("run differs from external source dependency pin")
    if run.precommit_digest != expected_precommit:
        raise AtomicSmokeRunError("run differs from external precommit pin")
    if canonical_json(_thaw_json(run.precommit_public_data)) != canonical_json(
        external_precommit.to_data()
    ):
        raise AtomicSmokeRunError("run embeds another public precommit")
    if (
        external_precommit.source_dependency_digest
        != run.source_dependency_digest
        or external_precommit.episode_public_data["run_id"] != run.run_id
    ):
        raise AtomicSmokeRunError("external precommit live identity differs from run")
    if run.protocol_digest != _digest(
        expected_protocol_digest, "expected protocol digest"
    ):
        raise AtomicSmokeRunError("run differs from external protocol pin")
    if run.expected_launcher_digest != _digest(
        expected_launcher_digest, "expected launcher digest"
    ):
        raise AtomicSmokeRunError("run differs from external launcher pin")
    if run.evidence_digest != _digest(
        expected_evidence_digest, "expected evidence digest"
    ):
        raise AtomicSmokeRunError("run differs from external evidence pin")
    if run.selection_archive_data is not None:
        assert run.selection_archive_digest is not None
        cold_decode_and_replay_atomic_selection(
            _thaw_json(run.selection_archive_data),  # type: ignore[arg-type]
            expected_archive_digest=run.selection_archive_digest,
        )
    if run.prediction_persistence_receipt is not None:
        if run.prediction_commitment_data is None:
            raise AtomicSmokeRunError("persistence receipt has no prediction payload")
        _verify_persisted_prediction(
            _thaw_json(run.prediction_commitment_data),  # type: ignore[arg-type]
            run.prediction_persistence_receipt,
            prediction_store_dir,
        )
    if run.label_reveal_data is not None:
        reveal = _thaw_json(run.label_reveal_data)
        labels = reveal["labels"]  # type: ignore[index]
        recomputed_label_commitment = canonical_digest(
            {
                "run_id": run.run_id,
                "labels": labels,
                "nonce": nonce,
                "version": "latent-label-seal/v1",
            }
        )
        precommit_label_commitment = external_precommit.episode_public_data[
            "label_commitment_digest"
        ]
        if (
            reveal["label_commitment_digest"]  # type: ignore[index]
            != recomputed_label_commitment
            or recomputed_label_commitment != precommit_label_commitment
        ):
            raise AtomicSmokeRunError(
                "revealed labels do not reproduce external precommit label seal"
            )
    # AtomicSmokeRun construction already replays every query formula and the
    # final score. Re-decode once more to detect mutable/canonical projections.
    if AtomicSmokeRun.from_data(run.to_data()).to_data() != run.to_data():
        raise AtomicSmokeRunError("cold replay projection is not stable")
    return run


__all__ = [
    "ATOMIC_SMOKE_CALL_SCHEMA",
    "ATOMIC_SMOKE_LABEL_REVEAL_SCHEMA",
    "ATOMIC_SMOKE_PREDICTION_SCHEMA",
    "ATOMIC_SMOKE_PERSISTENCE_SCHEMA",
    "ATOMIC_SMOKE_RUN_SCHEMA",
    "ATOMIC_SMOKE_SUCCESS_CALL_COUNT",
    "AtomicSmokeCallRecord",
    "AtomicSmokeRun",
    "AtomicSmokeRunError",
    "PredictionPersistenceReceipt",
    "atomic_smoke_description_protocol_digest",
    "atomic_smoke_run_protocol_digest",
    "atomic_smoke_scorer_protocol_digest",
    "cold_decode_and_replay_atomic_smoke_run",
    "run_atomic_smoke",
]
