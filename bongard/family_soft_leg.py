"""Registered IR bridge for one family-calibrated task-local soft claim.

The vision scorer emits only ordinal cue judgments.  This leg verifies the
resulting frozen score record, maps its Python-computed score through the
development-frozen scorer-family calibration, and exposes the calibrated
interval as a typed probability measurement.  The positive threshold remains
an explicit ``AT_LEAST`` value in the closed IR.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

from bongard.evidence import Disposition, Evidence, Provenance, Uncertainty
from bongard.ir import Atom, Quantity, Relation, StaticLegCall
from bongard.legs import (
    FROZEN_VISUAL_SCORE,
    SOFT_SEMANTIC,
    AffirmativeRelation,
    InvarianceContract,
    LegContract,
    LegReference,
    LegRegistry,
    LegSemantics,
    Unit,
)
from bongard.soft_predicates import (
    BlindSoftScoreRecord,
    SoftPredicateIntegrityError,
    SoftScorerFamily,
    measure_blind_soft_score,
)


FAMILY_SOFT_LEG_SCHEMA = "bongard.family-calibrated-soft-leg/v1"


class FamilySoftLegError(ValueError):
    """A family-calibrated task-local claim cannot be registered safely."""


def _sha256(data: object) -> str:
    return hashlib.sha256(
        json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _require_digest(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise FamilySoftLegError(f"{label} must be a lowercase SHA-256")
    return value


@dataclass(frozen=True, slots=True)
class RegisteredFamilySoftPredicate:
    """Static handle for the task-local calibrated soft measurement leg."""

    family: SoftScorerFamily
    protocol_digest: str
    claim_digest: str
    claim_description: str
    cue_ids: tuple[str, ...]
    operational_digest: str
    reference: LegReference

    def __post_init__(self) -> None:
        if not isinstance(self.family, SoftScorerFamily):
            raise TypeError("family must be SoftScorerFamily")
        _require_digest(self.protocol_digest, "protocol_digest")
        if self.family.protocol_digest != self.protocol_digest:
            raise SoftPredicateIntegrityError(
                "registered soft claim belongs to another scorer protocol"
            )
        _require_digest(self.claim_digest, "claim_digest")
        _require_digest(self.operational_digest, "operational_digest")
        if not isinstance(self.claim_description, str) or not self.claim_description.strip():
            raise FamilySoftLegError("claim_description must be non-empty")
        if not isinstance(self.cue_ids, tuple) or not self.cue_ids:
            raise FamilySoftLegError("cue_ids must be a non-empty immutable tuple")
        if len(self.cue_ids) != len(set(self.cue_ids)) or any(
            not isinstance(item, str) or not item for item in self.cue_ids
        ):
            raise FamilySoftLegError("cue_ids must be unique non-empty strings")

    def atom(self, *, boundary_name: str = "soft_score") -> Atom:
        if not isinstance(boundary_name, str) or not boundary_name.strip():
            raise FamilySoftLegError("boundary_name must be non-empty")
        return Atom(
            call=StaticLegCall(self.reference, (boundary_name,)),
            relation=Relation.AT_LEAST,
            claim="family-calibrated predictive support for: "
            + self.claim_description,
            lower=Quantity(self.family.affirmative_boundary, Unit.PROBABILITY),
        )


def register_family_soft_predicate(
    registry: LegRegistry,
    *,
    name: str,
    version: str,
    family: SoftScorerFamily,
    expected_protocol_digest: str,
    expected_family_digest: str,
    claim_digest: str,
    claim_description: str,
    cue_ids: tuple[str, ...],
    cost: int = 1,
) -> RegisteredFamilySoftPredicate:
    """Register one task-local claim against a frozen scorer-family policy."""

    if not isinstance(registry, LegRegistry):
        raise TypeError("registry must be a verifier-owned LegRegistry")
    if not isinstance(family, SoftScorerFamily):
        raise TypeError("family must be SoftScorerFamily")
    expected_family_digest = _require_digest(
        expected_family_digest, "expected_family_digest"
    )
    expected_protocol_digest = _require_digest(
        expected_protocol_digest, "expected_protocol_digest"
    )
    claim_digest = _require_digest(claim_digest, "claim_digest")
    if not isinstance(claim_description, str) or not claim_description.strip():
        raise FamilySoftLegError("claim_description must be non-empty")
    if not isinstance(cue_ids, tuple) or not cue_ids:
        raise FamilySoftLegError("cue_ids must be a non-empty immutable tuple")
    if len(cue_ids) != len(set(cue_ids)) or any(
        not isinstance(item, str) or not item for item in cue_ids
    ):
        raise FamilySoftLegError("cue_ids must be unique non-empty strings")
    family.assert_untampered()
    if family.digest() != expected_family_digest:
        raise SoftPredicateIntegrityError(
            "soft scorer family differs from the verifier commitment"
        )
    if family.protocol_digest != expected_protocol_digest:
        raise SoftPredicateIntegrityError(
            "soft scorer family belongs to a different prospective protocol"
        )
    operation = {
        "schema": FAMILY_SOFT_LEG_SCHEMA,
        "protocol_digest": expected_protocol_digest,
        "family_digest": expected_family_digest,
        "claim_digest": claim_digest,
        "claim_description": claim_description,
        "cue_ids": list(cue_ids),
        "affirmative_boundary": family.affirmative_boundary,
        "calibration_algorithm": (
            "family_fixed_bin_cluster_raw_simultaneous_hoeffding"
        ),
    }
    operational_digest = _sha256(operation)
    base_provenance = Provenance(
        producer="bongard.family_soft_leg",
        version="1",
        method="family_calibrated_predictive_support",
        input_digests=(
            expected_protocol_digest,
            expected_family_digest,
            claim_digest,
        ),
        artifact_digest=operational_digest,
        details=(("cue_ids_digest", _sha256(list(cue_ids))),),
    )

    def calibrated_soft_measurement(record: object) -> Evidence[float]:
        try:
            family.assert_untampered()
            if family.digest() != expected_family_digest:
                raise SoftPredicateIntegrityError(
                    "soft scorer family changed after registration"
                )
            if family.protocol_digest != expected_protocol_digest:
                raise SoftPredicateIntegrityError(
                    "soft scorer protocol changed after registration"
                )
            if not isinstance(record, BlindSoftScoreRecord):
                raise TypeError(
                    "family soft leg requires a BlindSoftScoreRecord"
                )
            record.assert_untampered()
            if record.claim_digest != claim_digest:
                raise SoftPredicateIntegrityError(
                    "blind score belongs to another task-local claim"
                )
            if record.scorer_protocol_digest != expected_protocol_digest:
                raise SoftPredicateIntegrityError(
                    "blind score belongs to another scorer protocol"
                )
            if record.declared_cue_ids != cue_ids:
                raise SoftPredicateIntegrityError(
                    "blind score cue inventory differs from the frozen claim"
                )
        except (SoftPredicateIntegrityError, TypeError, ValueError) as exc:
            return Evidence.error(
                base_provenance, type(exc).__name__, str(exc) or repr(exc)
            )

        measurement = measure_blind_soft_score(
            family,
            record,
            expected_family_digest=expected_family_digest,
        )
        if measurement.disposition is not Disposition.PRESENT:
            return Evidence(
                disposition=measurement.disposition,
                provenance=measurement.provenance,
                uncertainty=measurement.uncertainty,
                certificate=measurement.certificate,
                reason=measurement.reason,
                error_type=measurement.error_type,
            )
        score = measurement.unwrap()
        try:
            lower, upper, bin_index = family.calibrated_interval(score)
        except (TypeError, ValueError) as exc:
            return Evidence.error(
                measurement.provenance, type(exc).__name__, str(exc) or repr(exc)
            )
        provenance = Provenance.composed(
            producer="bongard.family_soft_leg",
            version="1",
            method="development_frozen_family_interval",
            parents=(base_provenance, measurement.provenance),
            details=(
                ("record_digest", record.digest()),
                ("score_bin_index", str(bin_index)),
            ),
        )
        return Evidence.present(
            score,
            provenance,
            Uncertainty(
                lower,
                upper,
                confidence_level=family.confidence_level,
                causes=(
                    "family_level_cluster_calibration",
                    "fixed_score_bin",
                ),
            ),
        )

    contract = LegContract(
        name=name,
        version=version,
        domain=(FROZEN_VISUAL_SCORE,),
        codomain=SOFT_SEMANTIC,
        implementation=calibrated_soft_measurement,
        affirmative_relations=frozenset({AffirmativeRelation.AT_LEAST}),
        invariance=InvarianceContract(),
        semantics=LegSemantics.DERIVED,
        cost=cost,
        operational_digest=operational_digest,
    )
    reference = registry.register(contract)
    return RegisteredFamilySoftPredicate(
        family=family,
        protocol_digest=expected_protocol_digest,
        claim_digest=claim_digest,
        claim_description=claim_description,
        cue_ids=cue_ids,
        operational_digest=operational_digest,
        reference=reference,
    )


__all__ = [
    "FAMILY_SOFT_LEG_SCHEMA",
    "FamilySoftLegError",
    "RegisteredFamilySoftPredicate",
    "register_family_soft_predicate",
]
