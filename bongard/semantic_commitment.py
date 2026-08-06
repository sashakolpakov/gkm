"""Pre-observation commitment for the visual-semantic benchmark path.

The typed proposer turn and pure-Python lowering must be frozen before any
support replay or query observation can run.  This module binds their complete
preimages, together with the support commitment and the already calibrated
policy.  A blind score record points to this artifact; the later support gate
then points to those score records.  That direction is acyclic.

No query panel, query label, support-gate result, or optional proof-checker
identity is admitted here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Mapping

from bongard.artifacts import SupportCommitment, canonical_digest
from bongard.ir import formula_digest
from bongard.semantic_synthesis import CompiledVisualSemanticProposal
from bongard.typed_visual_transport import TypedVisualTransportResult


SEMANTIC_PRE_OBSERVATION_SCHEMA = (
    "gkm.bongard-visual-semantic-pre-observation-commitment.v1"
)
REFERENCE_EXECUTION_SEMANTICS = "python-closed-ir-authoritative/v1"

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class SemanticCommitmentError(ValueError):
    """The causal proposer/lowering/support chain is inconsistent."""


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise SemanticCommitmentError(f"{label} must be a lowercase SHA-256")
    return value


def _validate_support_presentation(
    support: SupportCommitment,
    transport: TypedVisualTransportResult,
) -> None:
    by_id = {item.panel.blob_id: item for item in support.support}
    if len(by_id) != 12 or len(transport.support_presentation) != 12:
        raise SemanticCommitmentError(
            "semantic proposer commitment requires exactly twelve support panels"
        )
    for presented in transport.support_presentation:
        stem = presented.name.removesuffix(".png")
        try:
            side, raw_index = stem.split("_", 1)
            index = int(raw_index)
        except (ValueError, TypeError) as exc:
            raise SemanticCommitmentError(
                "typed proposer support presentation name is malformed"
            ) from exc
        if side not in {"pos", "neg"} or not 0 <= index < 6:
            raise SemanticCommitmentError(
                "typed proposer support presentation lies outside canonical 6+6 slots"
            )
        blob_id = f"support-{'positive' if side == 'pos' else 'negative'}-{index}"
        try:
            committed = by_id[blob_id]
        except KeyError as exc:
            raise SemanticCommitmentError(
                f"support commitment lacks proposer slot {blob_id!r}"
            ) from exc
        if committed.positive is not (side == "pos"):
            raise SemanticCommitmentError(
                f"support commitment polarity differs for {blob_id!r}"
            )
        if (
            committed.panel.sha256 != presented.content_digest
            or committed.panel.byte_count != presented.byte_count
            or committed.panel.media_type != "image/png"
        ):
            raise SemanticCommitmentError(
                f"typed proposer bytes differ from support commitment at {blob_id!r}"
            )


@dataclass(frozen=True)
class SemanticPreObservationCommitment:
    """Complete preimage bundle frozen before empirical rule evaluation."""

    support: SupportCommitment
    proposal_transport: TypedVisualTransportResult
    compiled: CompiledVisualSemanticProposal = field(repr=False, compare=False)
    _sealed_digest: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        self._validate()
        object.__setattr__(self, "_sealed_digest", self.digest)

    def _validate(self) -> None:
        if not isinstance(self.support, SupportCommitment):
            raise TypeError("support must be SupportCommitment")
        if not isinstance(self.proposal_transport, TypedVisualTransportResult):
            raise TypeError("proposal_transport must be TypedVisualTransportResult")
        if not isinstance(self.compiled, CompiledVisualSemanticProposal):
            raise TypeError("compiled must be CompiledVisualSemanticProposal")
        compiled = self.compiled
        canonical_policy = compiled.policy.__class__.from_data(
            compiled.policy.to_data()
        )
        if canonical_policy != compiled.policy:
            raise SemanticCommitmentError(
                "compiled visual-semantic policy is not canonical"
            )
        compiled.family.assert_untampered()
        compiled.family.verify_calibration()
        if self.proposal_transport.proposal != compiled.proposal:
            raise SemanticCommitmentError(
                "typed transport proposal differs from compiled proposal"
            )
        if (
            self.proposal_transport.catalog_digest
            != compiled.policy.direct_predicate_catalog_digest
        ):
            raise SemanticCommitmentError(
                "typed transport direct catalog differs from semantic policy"
            )
        if (
            self.proposal_transport.scorer_protocol_digest
            != compiled.policy.soft_scorer_protocol_digest
            or compiled.family.protocol_digest
            != compiled.policy.soft_scorer_protocol_digest
        ):
            raise SemanticCommitmentError(
                "typed transport, family, and policy scorer protocols differ"
            )
        if compiled.family.digest() != compiled.policy.soft_scorer_family_digest:
            raise SemanticCommitmentError(
                "compiled family differs from semantic policy"
            )
        if (
            compiled.family.development_manifest_digest
            != compiled.policy.soft_family_development_manifest_digest
        ):
            raise SemanticCommitmentError(
                "family development manifest differs from semantic policy"
            )
        if self.support.issued_by != compiled.attachment_contract.issued_by:
            raise SemanticCommitmentError(
                "support and compiled attachment issuers differ"
            )
        if formula_digest(compiled.formula) != (
            compiled.lowering_archive.compiled_formula_digest
        ):
            raise SemanticCommitmentError(
                "compiled formula differs from lowering archive"
            )
        if compiled.registry.digest() != compiled.lowering_archive.registry_digest:
            raise SemanticCommitmentError(
                "compiled registry differs from lowering archive"
            )
        if compiled.attachment_contract.digest() != (
            compiled.lowering_archive.attachment_digest
        ):
            raise SemanticCommitmentError(
                "compiled attachment differs from lowering archive"
            )
        _validate_support_presentation(self.support, self.proposal_transport)

    def identity_data(self) -> dict[str, str]:
        """Return the redundant digest join used by downstream score records."""

        compiled = self.compiled
        return {
            "support_commitment_digest": self.support.digest(),
            "proposal_transport_digest": self.proposal_transport.digest,
            "proposer_receipt_digest": (
                self.proposal_transport.receipt.receipt_digest
            ),
            "typed_proposal_digest": compiled.proposal.digest,
            "policy_digest": compiled.policy.digest(),
            "scorer_protocol_digest": compiled.family.protocol_digest,
            "scorer_family_digest": compiled.family.digest(),
            "family_development_manifest_digest": (
                compiled.family.development_manifest_digest
            ),
            "lowering_archive_digest": compiled.lowering_archive.digest,
            "compiled_formula_digest": formula_digest(compiled.formula),
            "registry_digest": compiled.registry.digest(),
            "attachment_contract_digest": (
                compiled.attachment_contract.digest()
            ),
        }

    def content_data(self) -> dict[str, object]:
        """Return every preimage needed for model-free archive verification."""

        compiled = self.compiled
        return {
            "schema": SEMANTIC_PRE_OBSERVATION_SCHEMA,
            "reference_execution_semantics": REFERENCE_EXECUTION_SEMANTICS,
            "optional_checker_may_affect_result": False,
            "identities": self.identity_data(),
            "support_commitment": self.support.to_data(),
            "proposal_transport": self.proposal_transport.to_data(),
            "visual_semantic_policy": compiled.policy.to_data(),
            "soft_scorer_family": compiled.family.to_data(),
            "lowering_archive": compiled.lowering_archive.to_data(),
            "compiled_formula": compiled.formula.to_data(),
            "registry_snapshot": compiled.registry.snapshot().to_data(),
            "attachment_contract": compiled.attachment_contract.to_data(),
        }

    @property
    def digest(self) -> str:
        return canonical_digest(self.content_data())

    def to_data(self) -> dict[str, object]:
        return {**self.content_data(), "commitment_digest": self.digest}

    def assert_untampered(self) -> None:
        self._validate()
        if self.digest != self._sealed_digest:
            raise SemanticCommitmentError(
                "semantic pre-observation commitment changed after sealing"
            )

    @classmethod
    def verify_data(
        cls,
        value: Mapping[str, Any],
        *,
        support: SupportCommitment,
        proposal_transport: TypedVisualTransportResult,
        compiled: CompiledVisualSemanticProposal,
        expected_digest: str | None = None,
    ) -> "SemanticPreObservationCommitment":
        """Rebuild from independently decoded preimages and compare exact bytes."""

        if not isinstance(value, Mapping):
            raise SemanticCommitmentError("pre-observation artifact must be an object")
        result = cls(
            support=support,
            proposal_transport=proposal_transport,
            compiled=compiled,
        )
        expected = result.digest if expected_digest is None else _digest(
            expected_digest, "expected commitment digest"
        )
        if result.digest != expected:
            raise SemanticCommitmentError(
                "reconstructed pre-observation commitment differs from expected digest"
            )
        if result.to_data() != dict(value):
            raise SemanticCommitmentError(
                "pre-observation artifact differs from reconstructed preimages"
            )
        return result


__all__ = [
    "REFERENCE_EXECUTION_SEMANTICS",
    "SEMANTIC_PRE_OBSERVATION_SCHEMA",
    "SemanticCommitmentError",
    "SemanticPreObservationCommitment",
]
