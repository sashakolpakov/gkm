"""Support-only transport boundary for typed visual proposals.

The adapter in this module has one deliberately small job.  It copies exactly
six positive and six negative support PNGs into the canonical
``pos_0.png`` .. ``neg_5.png`` presentation, makes one structured Codex call,
and parses the returned JSON against verifier-frozen proposal dependencies.
Original paths and experiment metadata are never sent to the proposer.

An accepted turn binds the canonical proposal, raw structured payload, Codex
receipt, and exact presented bytes.  A semantically invalid payload instead
becomes a content-addressed rejected-attempt record carried by a dedicated
exception; it is never converted into a false or negative proposal.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import tempfile
from typing import Any, Callable, Mapping, Sequence

from bongard.artifacts import canonical_digest, canonical_json
from bongard.canonical_cache import cached_content_data, cached_content_digest
from bongard.proposer import SupportPanelIdentity
from bongard.soft_predicates import SoftScorerProtocol
from bongard.transport import (
    MAX_PANEL_PNG_BYTES,
    STRUCTURED_INPUT_DIGEST_SCHEMA,
    CloudPolicyCacheSnapshot,
    CodexProposerFailure,
    CodexReceipt,
    CodexStructuredResult,
    semantic_panel_set_digest,
    run_codex_structured,
    validate_codex_receipt,
)
from bongard.typed_visual_proposal import (
    RegisteredAtomCatalog,
    TYPED_VISUAL_PROPOSER_GRAMMAR_ID,
    TYPED_VISUAL_PROPOSER_PROMPT_ID,
    TypedVisualProposal,
    TypedVisualProposalError,
    parse_typed_visual_proposal,
    typed_visual_proposal_grammar_digest,
    typed_visual_proposal_prompt,
    typed_visual_proposal_prompt_digest,
    typed_visual_proposal_schema,
)


TYPED_VISUAL_TRANSPORT_RESULT_SCHEMA_VERSION = (
    "gkm.bongard-typed-visual-transport-result.v1"
)
REJECTED_TYPED_VISUAL_PROPOSAL_SCHEMA_VERSION = (
    "gkm.bongard-rejected-typed-visual-proposal.v1"
)

_PANEL_STEMS = tuple(
    [f"pos_{index}" for index in range(6)]
    + [f"neg_{index}" for index in range(6)]
)
_PANEL_NAMES = tuple(f"{stem}.png" for stem in _PANEL_STEMS)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

StructuredTransport = Callable[..., CodexStructuredResult]


class TypedVisualTransportError(ValueError):
    """The typed proposer call or its causal binding is malformed."""


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _exact_sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise TypedVisualTransportError(f"{name} must be a lowercase sha256")
    return value


def _canonical_payload(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) for key in value
    ):
        raise TypedVisualTransportError(f"{name} must be a JSON object")
    try:
        decoded = json.loads(canonical_json(dict(value)).decode("utf-8"))
    except (TypeError, ValueError, UnicodeError, json.JSONDecodeError) as exc:
        raise TypedVisualTransportError(
            f"{name} is not canonical finite JSON: {exc}"
        ) from exc
    if not isinstance(decoded, dict):  # Kept explicit for static and cold checks.
        raise TypedVisualTransportError(f"{name} must decode as a JSON object")
    return decoded


def _strict_record(
    value: object, expected: set[str], name: str
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) for key in value
    ):
        raise TypedVisualTransportError(f"{name} must be a JSON object")
    actual = set(value)
    if actual != expected:
        raise TypedVisualTransportError(
            f"{name} fields differ from schema: "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )
    return value


def _receipt_from_data(value: object) -> CodexReceipt:
    """Reconstruct one exact JSON receipt through the transport validator."""

    expected = set(CodexReceipt.__dataclass_fields__)
    raw = dict(_strict_record(value, expected, "typed proposer receipt"))
    try:
        # Validate the JSON representation before converting its two arrays to
        # the tuples used by the in-memory frozen dataclass.
        validate_codex_receipt(raw)
        event_types = raw["event_types"]
        item_types = raw["item_types"]
        if not isinstance(event_types, list) or not isinstance(item_types, list):
            raise TypedVisualTransportError(
                "typed proposer receipt event/item types must be JSON lists"
            )
        receipt = CodexReceipt(
            **{
                **raw,
                "event_types": tuple(event_types),
                "item_types": tuple(item_types),
            }
        )
    except (CodexProposerFailure, KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, TypedVisualTransportError):
            raise
        raise TypedVisualTransportError(
            f"invalid archived typed proposer receipt: {str(exc) or repr(exc)}"
        ) from exc
    if receipt.to_dict() != raw:
        raise TypedVisualTransportError(
            "typed proposer receipt is not canonically represented"
        )
    return receipt


def _presentation_from_data(
    value: object,
) -> tuple[SupportPanelIdentity, ...]:
    if not isinstance(value, list):
        raise TypedVisualTransportError(
            "support_presentation must be a JSON list"
        )
    items: list[SupportPanelIdentity] = []
    for index, item in enumerate(value):
        raw = dict(
            _strict_record(
                item,
                {"name", "byte_count", "content_digest"},
                f"support_presentation[{index}]",
            )
        )
        try:
            identity = SupportPanelIdentity(
                name=raw["name"],
                byte_count=raw["byte_count"],
                content_digest=raw["content_digest"],
            )
        except (TypeError, ValueError) as exc:
            raise TypedVisualTransportError(
                f"support_presentation[{index}] is invalid: {exc}"
            ) from exc
        if identity.to_dict() != raw:
            raise TypedVisualTransportError(
                f"support_presentation[{index}] is not canonically represented"
            )
        items.append(identity)
    return _validate_presentation(tuple(items))


def _validate_frozen_dependencies(
    catalog: RegisteredAtomCatalog,
    protocol: SoftScorerProtocol,
) -> str:
    if not isinstance(catalog, RegisteredAtomCatalog):
        raise TypeError("catalog must be RegisteredAtomCatalog")
    if not isinstance(protocol, SoftScorerProtocol):
        raise TypeError("protocol must be a SoftScorerProtocol")
    try:
        protocol.assert_untampered()
    except (TypeError, ValueError) as exc:
        raise TypedVisualTransportError(
            f"scorer protocol is invalid: {exc}"
        ) from exc
    if protocol.proposer_grammar_id != TYPED_VISUAL_PROPOSER_GRAMMAR_ID or (
        protocol.proposer_grammar_digest
        != typed_visual_proposal_grammar_digest(catalog)
    ):
        raise TypedVisualTransportError(
            "scorer protocol typed proposer grammar differs from the current catalog"
        )
    if protocol.proposer_prompt_id != TYPED_VISUAL_PROPOSER_PROMPT_ID or (
        protocol.proposer_prompt_digest
        != typed_visual_proposal_prompt_digest(catalog)
    ):
        raise TypedVisualTransportError(
            "scorer protocol typed proposer prompt differs from the current catalog"
        )
    return protocol.digest()


def _validate_archived_call_binding(
    *,
    receipt: CodexReceipt,
    catalog: RegisteredAtomCatalog,
    protocol: SoftScorerProtocol,
    support_presentation: tuple[SupportPanelIdentity, ...],
) -> None:
    """Replay every call identity recoverable without retained support pixels."""

    prompt = typed_visual_proposal_prompt(catalog)
    prompt_digest = _sha256(prompt.encode("utf-8"))
    output_schema_digest = canonical_digest(typed_visual_proposal_schema(catalog))
    panel_view_digest = canonical_digest(
        [item.to_dict() for item in support_presentation]
    )
    expected = {
        "prompt_digest": prompt_digest,
        "task_digest": prompt_digest,
        "output_schema_digest": output_schema_digest,
        "panel_view_digest": panel_view_digest,
        "requested_model": protocol.proposer_model_id,
        "requested_reasoning_effort": protocol.proposer_reasoning_effort,
        "input_digest_schema": STRUCTURED_INPUT_DIGEST_SCHEMA,
    }
    for field, expected_value in expected.items():
        if getattr(receipt, field) != expected_value:
            raise TypedVisualTransportError(
                f"archived typed proposer receipt {field} differs from frozen inputs"
            )
    input_envelope = {
        "schema": STRUCTURED_INPUT_DIGEST_SCHEMA,
        "task": prompt,
        "ordered_panel_identities": [
            item.to_dict() for item in support_presentation
        ],
        "panel_view_digest": panel_view_digest,
        # The semantic pixel digest cannot be recomputed without the support
        # bytes, but its exact receipt-bound value remains part of the envelope.
        "panel_set_digest": receipt.panel_set_digest,
        "prompt_digest": prompt_digest,
        "output_schema_digest": output_schema_digest,
    }
    if receipt.input_digest != canonical_digest(input_envelope):
        raise TypedVisualTransportError(
            "archived typed proposer receipt input_digest does not reproduce"
        )


def _validate_presentation(
    value: tuple[SupportPanelIdentity, ...],
) -> tuple[SupportPanelIdentity, ...]:
    if not isinstance(value, tuple) or not all(
        isinstance(item, SupportPanelIdentity) for item in value
    ):
        raise TypedVisualTransportError(
            "support_presentation must contain SupportPanelIdentity values"
        )
    if tuple(item.name for item in value) != _PANEL_NAMES:
        raise TypedVisualTransportError(
            "support_presentation must use canonical 6+6 order"
        )
    return value


def _validate_receipt(receipt: object) -> CodexReceipt:
    if not isinstance(receipt, CodexReceipt):
        raise TypedVisualTransportError(
            "typed proposer transport did not return a CodexReceipt"
        )
    try:
        validate_codex_receipt(receipt.to_dict())
    except (CodexProposerFailure, TypeError, ValueError) as exc:
        raise TypedVisualTransportError(
            f"typed proposer transport receipt is invalid: {exc}"
        ) from exc
    return receipt


def _common_record_checks(
    *,
    model_payload: Mapping[str, Any],
    receipt: CodexReceipt,
    support_presentation: tuple[SupportPanelIdentity, ...],
    catalog_digest: str,
    scorer_protocol_digest: str,
) -> dict[str, Any]:
    payload = _canonical_payload(model_payload, "model_payload")
    _validate_receipt(receipt)
    _validate_presentation(support_presentation)
    _exact_sha256(catalog_digest, "catalog_digest")
    _exact_sha256(scorer_protocol_digest, "scorer_protocol_digest")
    if receipt.structured_output_digest != canonical_digest(payload):
        raise TypedVisualTransportError(
            "receipt does not bind the retained structured model payload"
        )
    expected_view = canonical_digest(
        [item.to_dict() for item in support_presentation]
    )
    if receipt.panel_view_digest != expected_view:
        raise TypedVisualTransportError(
            "receipt does not bind the retained support presentation bytes"
        )
    return payload


@dataclass(frozen=True)
class TypedVisualTransportResult:
    """One accepted canonical proposal and its complete support-only turn."""

    proposal: TypedVisualProposal
    model_payload: Mapping[str, Any]
    receipt: CodexReceipt
    support_presentation: tuple[SupportPanelIdentity, ...]
    catalog_digest: str
    scorer_protocol_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.proposal, TypedVisualProposal):
            raise TypedVisualTransportError(
                "proposal must be a canonical TypedVisualProposal"
            )
        payload = _common_record_checks(
            model_payload=self.model_payload,
            receipt=self.receipt,
            support_presentation=self.support_presentation,
            catalog_digest=self.catalog_digest,
            scorer_protocol_digest=self.scorer_protocol_digest,
        )
        object.__setattr__(self, "model_payload", payload)
        if self.proposal.catalog_digest != self.catalog_digest:
            raise TypedVisualTransportError(
                "canonical proposal belongs to a different atom catalog"
            )
        if self.proposal.soft_claim is not None and (
            self.proposal.soft_claim.scorer_protocol_digest
            != self.scorer_protocol_digest
        ):
            raise TypedVisualTransportError(
                "canonical proposal belongs to a different scorer protocol"
            )

    def _canonical_anchor(self) -> tuple[object, ...]:
        return (
            self.proposal.digest,
            canonical_json(_canonical_payload(self.model_payload, "model_payload")),
            canonical_json(self.receipt.to_dict()),
            canonical_json(
                [item.to_dict() for item in self.support_presentation]
            ),
            self.catalog_digest,
            self.scorer_protocol_digest,
        )

    @property
    def support_presentation_digest(self) -> str:
        return canonical_digest(
            [item.to_dict() for item in self.support_presentation]
        )

    def _uncached_content_data(self) -> dict[str, object]:
        return {
            "schema": TYPED_VISUAL_TRANSPORT_RESULT_SCHEMA_VERSION,
            "catalog_digest": self.catalog_digest,
            "scorer_protocol_digest": self.scorer_protocol_digest,
            "proposal": self.proposal.to_data(),
            "model_payload": _canonical_payload(
                self.model_payload, "model_payload"
            ),
            "receipt": self.receipt.to_dict(),
            "support_presentation": [
                item.to_dict() for item in self.support_presentation
            ],
            "support_presentation_digest": self.support_presentation_digest,
        }

    def content_data(self) -> dict[str, object]:
        return cached_content_data(
            self,
            self._canonical_anchor(),
            self._uncached_content_data,
        )

    @property
    def digest(self) -> str:
        return cached_content_digest(
            self,
            self._canonical_anchor(),
            self._uncached_content_data,
        )

    def to_data(self) -> dict[str, object]:
        return {**self.content_data(), "result_digest": self.digest}

    @classmethod
    def from_data(
        cls,
        value: Mapping[str, Any],
        *,
        catalog: RegisteredAtomCatalog,
        protocol: SoftScorerProtocol,
        expected_digest: str | None = None,
    ) -> "TypedVisualTransportResult":
        """Cold-decode and replay an accepted support-only proposer turn."""

        fields = {
            "schema",
            "catalog_digest",
            "scorer_protocol_digest",
            "proposal",
            "model_payload",
            "receipt",
            "support_presentation",
            "support_presentation_digest",
            "result_digest",
        }
        data = dict(_strict_record(value, fields, "typed transport result"))
        if data["schema"] != TYPED_VISUAL_TRANSPORT_RESULT_SCHEMA_VERSION:
            raise TypedVisualTransportError(
                "unsupported typed visual transport-result schema"
            )
        protocol_digest = _validate_frozen_dependencies(catalog, protocol)
        catalog_digest = _exact_sha256(
            data["catalog_digest"], "archived catalog_digest"
        )
        if catalog_digest != catalog.digest:
            raise TypedVisualTransportError(
                "typed transport result belongs to another atom catalog"
            )
        archived_protocol_digest = _exact_sha256(
            data["scorer_protocol_digest"],
            "archived scorer_protocol_digest",
        )
        if archived_protocol_digest != protocol_digest:
            raise TypedVisualTransportError(
                "typed transport result belongs to another scorer protocol"
            )

        proposal_raw = data["proposal"]
        if not isinstance(proposal_raw, Mapping):
            raise TypedVisualTransportError("proposal must be a JSON object")
        try:
            proposal = TypedVisualProposal.from_data(
                proposal_raw,
                catalog=catalog,
                expected_scorer_protocol_digest=protocol_digest,
            )
        except (TypeError, ValueError) as exc:
            raise TypedVisualTransportError(
                f"archived canonical proposal is invalid: {exc}"
            ) from exc

        payload = _canonical_payload(data["model_payload"], "model_payload")
        try:
            replayed_proposal = parse_typed_visual_proposal(
                payload,
                catalog=catalog,
                scorer_protocol_digest=protocol_digest,
            )
        except TypedVisualProposalError as exc:
            raise TypedVisualTransportError(
                "accepted model payload no longer parses under the frozen grammar: "
                f"{str(exc) or repr(exc)}"
            ) from exc
        if replayed_proposal.to_data() != proposal.to_data():
            raise TypedVisualTransportError(
                "retained model payload does not reproduce the canonical proposal"
            )

        receipt = _receipt_from_data(data["receipt"])
        support_presentation = _presentation_from_data(
            data["support_presentation"]
        )
        result = cls(
            proposal=proposal,
            model_payload=payload,
            receipt=receipt,
            support_presentation=support_presentation,
            catalog_digest=catalog_digest,
            scorer_protocol_digest=archived_protocol_digest,
        )
        _validate_archived_call_binding(
            receipt=receipt,
            catalog=catalog,
            protocol=protocol,
            support_presentation=support_presentation,
        )
        archived_presentation_digest = _exact_sha256(
            data["support_presentation_digest"],
            "support_presentation_digest",
        )
        if archived_presentation_digest != result.support_presentation_digest:
            raise TypedVisualTransportError(
                "archived support_presentation_digest does not reproduce"
            )
        archived_digest = _exact_sha256(data["result_digest"], "result_digest")
        if archived_digest != result.digest:
            raise TypedVisualTransportError(
                "archived typed transport result digest does not reproduce"
            )
        if expected_digest is not None and result.digest != _exact_sha256(
            expected_digest, "expected typed transport result digest"
        ):
            raise TypedVisualTransportError(
                "typed transport result differs from the expected digest"
            )
        if result.to_data() != data:
            raise TypedVisualTransportError(
                "typed transport result is not canonically represented"
            )
        return result


@dataclass(frozen=True)
class RejectedTypedVisualProposalAttempt:
    """Receipt-bound parser rejection with no admitted proposal semantics."""

    model_payload: Mapping[str, Any]
    receipt: CodexReceipt
    support_presentation: tuple[SupportPanelIdentity, ...]
    catalog_digest: str
    scorer_protocol_digest: str
    parse_error_type: str
    parse_error_reason: str

    def __post_init__(self) -> None:
        payload = _common_record_checks(
            model_payload=self.model_payload,
            receipt=self.receipt,
            support_presentation=self.support_presentation,
            catalog_digest=self.catalog_digest,
            scorer_protocol_digest=self.scorer_protocol_digest,
        )
        object.__setattr__(self, "model_payload", payload)
        for name, value in (
            ("parse_error_type", self.parse_error_type),
            ("parse_error_reason", self.parse_error_reason),
        ):
            if (
                not isinstance(value, str)
                or not value
                or value != value.strip()
                or "\x00" in value
            ):
                raise TypedVisualTransportError(
                    f"{name} must be non-empty exact text"
                )

    @property
    def support_presentation_digest(self) -> str:
        return canonical_digest(
            [item.to_dict() for item in self.support_presentation]
        )

    def content_data(self) -> dict[str, object]:
        # There is intentionally no ``proposal`` or truth-valued field here.
        return {
            "schema": REJECTED_TYPED_VISUAL_PROPOSAL_SCHEMA_VERSION,
            "catalog_digest": self.catalog_digest,
            "scorer_protocol_digest": self.scorer_protocol_digest,
            "model_payload": _canonical_payload(
                self.model_payload, "model_payload"
            ),
            "receipt": self.receipt.to_dict(),
            "support_presentation": [
                item.to_dict() for item in self.support_presentation
            ],
            "support_presentation_digest": self.support_presentation_digest,
            "parse_error": {
                "error_type": self.parse_error_type,
                "reason": self.parse_error_reason,
            },
        }

    @property
    def digest(self) -> str:
        return canonical_digest(self.content_data())

    def to_data(self) -> dict[str, object]:
        return {**self.content_data(), "attempt_digest": self.digest}

    @classmethod
    def from_data(
        cls,
        value: Mapping[str, Any],
        *,
        catalog: RegisteredAtomCatalog,
        protocol: SoftScorerProtocol,
        expected_digest: str | None = None,
    ) -> "RejectedTypedVisualProposalAttempt":
        """Cold-decode and reproduce one parser-rejected proposer turn."""

        fields = {
            "schema",
            "catalog_digest",
            "scorer_protocol_digest",
            "model_payload",
            "receipt",
            "support_presentation",
            "support_presentation_digest",
            "parse_error",
            "attempt_digest",
        }
        data = dict(_strict_record(value, fields, "rejected typed attempt"))
        if data["schema"] != REJECTED_TYPED_VISUAL_PROPOSAL_SCHEMA_VERSION:
            raise TypedVisualTransportError(
                "unsupported rejected typed-proposal schema"
            )
        protocol_digest = _validate_frozen_dependencies(catalog, protocol)
        catalog_digest = _exact_sha256(
            data["catalog_digest"], "archived catalog_digest"
        )
        if catalog_digest != catalog.digest:
            raise TypedVisualTransportError(
                "rejected typed attempt belongs to another atom catalog"
            )
        archived_protocol_digest = _exact_sha256(
            data["scorer_protocol_digest"],
            "archived scorer_protocol_digest",
        )
        if archived_protocol_digest != protocol_digest:
            raise TypedVisualTransportError(
                "rejected typed attempt belongs to another scorer protocol"
            )
        payload = _canonical_payload(data["model_payload"], "model_payload")
        parse_error = _strict_record(
            data["parse_error"], {"error_type", "reason"}, "parse_error"
        )
        try:
            parse_typed_visual_proposal(
                payload,
                catalog=catalog,
                scorer_protocol_digest=protocol_digest,
            )
        except TypedVisualProposalError as exc:
            replayed_error_type = type(exc).__name__
            replayed_reason = str(exc) or "typed proposal validation failed"
        else:
            raise TypedVisualTransportError(
                "rejected model payload is accepted by the frozen grammar"
            )
        if (
            parse_error["error_type"] != replayed_error_type
            or parse_error["reason"] != replayed_reason
        ):
            raise TypedVisualTransportError(
                "archived parser rejection does not reproduce exactly"
            )

        receipt = _receipt_from_data(data["receipt"])
        support_presentation = _presentation_from_data(
            data["support_presentation"]
        )
        result = cls(
            model_payload=payload,
            receipt=receipt,
            support_presentation=support_presentation,
            catalog_digest=catalog_digest,
            scorer_protocol_digest=archived_protocol_digest,
            parse_error_type=parse_error["error_type"],
            parse_error_reason=parse_error["reason"],
        )
        _validate_archived_call_binding(
            receipt=receipt,
            catalog=catalog,
            protocol=protocol,
            support_presentation=support_presentation,
        )
        archived_presentation_digest = _exact_sha256(
            data["support_presentation_digest"],
            "support_presentation_digest",
        )
        if archived_presentation_digest != result.support_presentation_digest:
            raise TypedVisualTransportError(
                "archived support_presentation_digest does not reproduce"
            )
        archived_digest = _exact_sha256(
            data["attempt_digest"], "attempt_digest"
        )
        if archived_digest != result.digest:
            raise TypedVisualTransportError(
                "archived rejected-attempt digest does not reproduce"
            )
        if expected_digest is not None and result.digest != _exact_sha256(
            expected_digest, "expected rejected-attempt digest"
        ):
            raise TypedVisualTransportError(
                "rejected typed attempt differs from the expected digest"
            )
        if result.to_data() != data:
            raise TypedVisualTransportError(
                "rejected typed attempt is not canonically represented"
            )
        return result


class TypedVisualProposalRejected(TypedVisualTransportError):
    """The transport succeeded, but the typed parser rejected its payload."""

    def __init__(self, attempt: RejectedTypedVisualProposalAttempt):
        if not isinstance(attempt, RejectedTypedVisualProposalAttempt):
            raise TypeError("attempt must be RejectedTypedVisualProposalAttempt")
        self.attempt = attempt
        super().__init__(attempt.parse_error_reason)


def _read_exact_support(path_value: str | Path, label: str) -> tuple[Path, bytes, tuple[int, int]]:
    try:
        path = Path(path_value).resolve(strict=True)
        before = path.stat()
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise TypedVisualTransportError(
            f"{label} support PNG does not exist: {path_value!r}"
        ) from exc
    if (
        not stat.S_ISREG(before.st_mode)
        or not 0 < before.st_size <= MAX_PANEL_PNG_BYTES
    ):
        raise TypedVisualTransportError(
            f"{label} support must be a bounded regular PNG"
        )
    try:
        payload = path.read_bytes()
        after = path.stat()
    except OSError as exc:
        raise TypedVisualTransportError(
            f"cannot read {label} support PNG: {path}"
        ) from exc
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if before_identity != after_identity or len(payload) != before.st_size:
        raise TypedVisualTransportError(
            f"{label} support PNG changed while being read"
        )
    if not payload.startswith(_PNG_SIGNATURE):
        raise TypedVisualTransportError(f"{label} support is not a PNG")
    return path, payload, (before.st_dev, before.st_ino)


def _support_snapshot(
    positive_support: Sequence[str | Path],
    negative_support: Sequence[str | Path],
) -> tuple[tuple[str, bytes], ...]:
    if isinstance(positive_support, (str, bytes)) or isinstance(
        negative_support, (str, bytes)
    ):
        raise TypedVisualTransportError("support inputs must be path sequences")
    if len(positive_support) != 6 or len(negative_support) != 6:
        raise TypedVisualTransportError(
            "typed proposer requires exactly 6 positive and 6 negative supports"
        )
    resolved: list[Path] = []
    file_ids: list[tuple[int, int]] = []
    snapshot: list[tuple[str, bytes]] = []
    for name, source in zip(
        _PANEL_NAMES, (*positive_support, *negative_support), strict=True
    ):
        path, payload, file_id = _read_exact_support(source, name)
        resolved.append(path)
        file_ids.append(file_id)
        snapshot.append((name, payload))
    if len(set(resolved)) != 12 or len(set(file_ids)) != 12:
        raise TypedVisualTransportError(
            "the 12 support PNGs must be distinct files"
        )
    return tuple(snapshot)


def _stage_support(
    directory: Path, snapshot: Sequence[tuple[str, bytes]]
) -> tuple[str, ...]:
    paths: list[str] = []
    for expected_name, (name, payload) in zip(_PANEL_NAMES, snapshot, strict=True):
        if name != expected_name:
            raise TypedVisualTransportError("support snapshot order is non-canonical")
        target = directory / name
        try:
            descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
            try:
                offset = 0
                while offset < len(payload):
                    offset += os.write(descriptor, payload[offset:])
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        except OSError as exc:
            raise TypedVisualTransportError(
                f"cannot stage canonical support PNG {name}: {exc}"
            ) from exc
        paths.append(str(target.resolve()))
    return tuple(paths)


def _presentation_from_snapshot(
    snapshot: Sequence[tuple[str, bytes]],
) -> tuple[SupportPanelIdentity, ...]:
    return tuple(
        SupportPanelIdentity(
            name=name,
            byte_count=len(payload),
            content_digest=_sha256(payload),
        )
        for name, payload in snapshot
    )


def _assert_staged_unchanged(
    paths: Sequence[str], snapshot: Sequence[tuple[str, bytes]]
) -> None:
    for path_value, (name, expected) in zip(paths, snapshot, strict=True):
        path = Path(path_value)
        try:
            observed = path.read_bytes()
        except OSError as exc:
            raise TypedVisualTransportError(
                f"canonical support presentation disappeared: {name}"
            ) from exc
        if observed != expected:
            raise TypedVisualTransportError(
                f"canonical support presentation changed during transport: {name}"
            )


def _validate_call_binding(
    *,
    receipt: CodexReceipt,
    payload: Mapping[str, Any],
    prompt: str,
    schema: Mapping[str, Any],
    paths: Sequence[str],
    presentation: tuple[SupportPanelIdentity, ...],
    model: str,
    reasoning_effort: str,
) -> None:
    _validate_receipt(receipt)
    prompt_digest = _sha256(prompt.encode("utf-8"))
    schema_digest = canonical_digest(dict(schema))
    payload_digest = canonical_digest(dict(payload))
    panel_view_digest = canonical_digest(
        [item.to_dict() for item in presentation]
    )
    try:
        panel_set_digest = semantic_panel_set_digest(paths)
    except (CodexProposerFailure, OSError, TypeError, ValueError) as exc:
        raise TypedVisualTransportError(
            f"cannot reproduce transport panel-set identity: {exc}"
        ) from exc
    expected = {
        "prompt_digest": prompt_digest,
        "task_digest": prompt_digest,
        "output_schema_digest": schema_digest,
        "structured_output_digest": payload_digest,
        "panel_view_digest": panel_view_digest,
        "panel_set_digest": panel_set_digest,
        "requested_model": model,
        "requested_reasoning_effort": reasoning_effort,
        "input_digest_schema": STRUCTURED_INPUT_DIGEST_SCHEMA,
    }
    for field, value in expected.items():
        if getattr(receipt, field) != value:
            raise TypedVisualTransportError(
                f"typed proposer receipt {field} differs from the executed call"
            )
    input_envelope = {
        "schema": STRUCTURED_INPUT_DIGEST_SCHEMA,
        "task": prompt,
        "ordered_panel_identities": [
            item.to_dict() for item in presentation
        ],
        "panel_view_digest": panel_view_digest,
        "panel_set_digest": panel_set_digest,
        "prompt_digest": prompt_digest,
        "output_schema_digest": schema_digest,
    }
    if receipt.input_digest != canonical_digest(input_envelope):
        raise TypedVisualTransportError(
            "typed proposer receipt input_digest differs from the executed call"
        )


def propose_typed_visual(
    positive_support: Sequence[str | Path],
    negative_support: Sequence[str | Path],
    *,
    catalog: RegisteredAtomCatalog,
    protocol: SoftScorerProtocol,
    model: str | None = None,
    reasoning_effort: str | None = None,
    minutes: int = 15,
    verbose: bool = False,
    executable: str = "codex",
    cloud_policy_cache_snapshot: CloudPolicyCacheSnapshot | None = None,
    transport: StructuredTransport = run_codex_structured,
) -> TypedVisualTransportResult:
    """Run exactly one typed, support-only structured proposal turn.

    The signature intentionally has no task, corpus, source, query, run, or
    label-metadata parameter.  Only copied support bytes, their canonical
    ``pos_*``/``neg_*`` names, the frozen typed prompt, and the frozen schema
    cross the transport boundary.
    """

    if not isinstance(catalog, RegisteredAtomCatalog):
        raise TypeError("catalog must be RegisteredAtomCatalog")
    if not isinstance(protocol, SoftScorerProtocol):
        raise TypeError("protocol must be a SoftScorerProtocol")
    protocol.assert_untampered()
    scorer_digest = protocol.digest()
    if protocol.proposer_grammar_id != TYPED_VISUAL_PROPOSER_GRAMMAR_ID:
        raise TypedVisualTransportError(
            "scorer protocol uses a different typed proposer grammar"
        )
    if (
        protocol.proposer_grammar_digest
        != typed_visual_proposal_grammar_digest(catalog)
    ):
        raise TypedVisualTransportError(
            "scorer protocol typed proposer grammar digest differs"
        )
    if protocol.proposer_prompt_id != TYPED_VISUAL_PROPOSER_PROMPT_ID:
        raise TypedVisualTransportError(
            "scorer protocol uses a different typed proposer prompt"
        )
    if (
        protocol.proposer_prompt_digest
        != typed_visual_proposal_prompt_digest(catalog)
    ):
        raise TypedVisualTransportError(
            "scorer protocol typed proposer prompt digest differs"
        )
    resolved_model = protocol.proposer_model_id if model is None else model
    if resolved_model != protocol.proposer_model_id:
        raise TypedVisualTransportError(
            "requested proposer model differs from the frozen scorer protocol"
        )
    resolved_effort = (
        protocol.proposer_reasoning_effort
        if reasoning_effort is None
        else reasoning_effort
    )
    if resolved_effort != protocol.proposer_reasoning_effort:
        raise TypedVisualTransportError(
            "requested proposer reasoning effort differs from the frozen scorer protocol"
        )
    if not callable(transport):
        raise TypeError("transport must be callable")
    snapshot = _support_snapshot(positive_support, negative_support)
    presentation = _presentation_from_snapshot(snapshot)
    prompt = typed_visual_proposal_prompt(catalog)
    schema = typed_visual_proposal_schema(catalog)

    with tempfile.TemporaryDirectory(prefix="bongard-typed-support-") as raw_dir:
        directory = Path(raw_dir)
        paths = _stage_support(directory, snapshot)
        result = transport(
            prompt,
            paths,
            schema,
            model=resolved_model,
            reasoning_effort=resolved_effort,
            minutes=minutes,
            verbose=verbose,
            executable=executable,
            cloud_policy_cache_snapshot=cloud_policy_cache_snapshot,
        )
        _assert_staged_unchanged(paths, snapshot)
        try:
            payload_value = result.payload
            receipt_value = result.receipt
        except AttributeError as exc:
            raise TypedVisualTransportError(
                "structured transport result lacks payload or receipt"
            ) from exc
        payload = _canonical_payload(payload_value, "transport payload")
        receipt = _validate_receipt(receipt_value)
        _validate_call_binding(
            receipt=receipt,
            payload=payload,
            prompt=prompt,
            schema=schema,
            paths=paths,
            presentation=presentation,
            model=resolved_model,
            reasoning_effort=resolved_effort,
        )

    try:
        proposal = parse_typed_visual_proposal(
            payload,
            catalog=catalog,
            scorer_protocol_digest=scorer_digest,
        )
    except TypedVisualProposalError as exc:
        attempt = RejectedTypedVisualProposalAttempt(
            model_payload=payload,
            receipt=receipt,
            support_presentation=presentation,
            catalog_digest=catalog.digest,
            scorer_protocol_digest=scorer_digest,
            parse_error_type=type(exc).__name__,
            parse_error_reason=str(exc) or "typed proposal validation failed",
        )
        raise TypedVisualProposalRejected(attempt) from exc

    return TypedVisualTransportResult(
        proposal=proposal,
        model_payload=payload,
        receipt=receipt,
        support_presentation=presentation,
        catalog_digest=catalog.digest,
        scorer_protocol_digest=scorer_digest,
    )


# A more explicit alias for callers that use the older ``propose_rule`` name.
propose_typed_visual_rule = propose_typed_visual


__all__ = [
    "REJECTED_TYPED_VISUAL_PROPOSAL_SCHEMA_VERSION",
    "RejectedTypedVisualProposalAttempt",
    "StructuredTransport",
    "TYPED_VISUAL_TRANSPORT_RESULT_SCHEMA_VERSION",
    "TypedVisualProposalRejected",
    "TypedVisualTransportError",
    "TypedVisualTransportResult",
    "propose_typed_visual",
    "propose_typed_visual_rule",
]
