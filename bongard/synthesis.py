"""Compile visual prose into the closed typed predicate boundary.

PURE proposals attach to an existing verifier-owned registry.  HYBRID
proposals receive a task-local, content-addressed empirical leg.  The latter is
rigorous as an operational measurement (exact claim, observer, and receipts),
not as an assertion that a neural vision judgment is mathematical pixel truth.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Callable

from bongard.admission import TypedAttachmentContract
from bongard.evidence import Disposition, Evidence
from bongard.ir import Atom, Formula, Relation, StaticLegCall, validate_formula
from bongard.legs import (
    BOOLEAN_WITNESS,
    PANEL,
    InvarianceContract,
    LegContract,
    LegRegistry,
    LegSemantics,
)
from bongard.proposer import HybridObservation, RuleProposal, observe_hybrid_panel
from bongard.support_prototypes import (
    SUPPORT_PROTOTYPE_FEATURES,
    FrozenFeatureSpace,
    FrozenSupportPrototypes,
    PositivePrototypeFormula,
    register_support_prototype_leg,
    validate_prototype_formula,
)


class SynthesisError(ValueError):
    """A proposal cannot be attached to the current typed registry."""


HybridObserver = Callable[[RuleProposal, object], HybridObservation]


@dataclass(frozen=True)
class CompiledProposal:
    proposal: RuleProposal
    registry: LegRegistry
    formula: Formula
    attachment_contract: TypedAttachmentContract

    @property
    def proposer_digest(self) -> str:
        return self.proposal.digest.removeprefix("sha256:")


def truth_from_hybrid_observation(
    observation: HybridObservation,
) -> Evidence[bool]:
    """Project one empirical claim observation into atom truth evidence.

    This projection preserves all four dispositions.  In particular, a model
    failure or ambiguous resemblance cannot become a negative prediction.
    """

    evidence = observation.evidence
    if evidence.disposition is Disposition.PRESENT:
        return Evidence.present(True, evidence.provenance, evidence.uncertainty)
    if evidence.disposition is Disposition.CERTIFIED_ABSENT:
        return Evidence.certified_absent(
            evidence.provenance,
            evidence.certificate
            or "archived model nonmatch for the frozen operational claim",
            evidence.uncertainty,
        )
    if evidence.disposition is Disposition.INDETERMINATE:
        return Evidence.indeterminate(
            evidence.provenance,
            evidence.reason or "hybrid observer was indeterminate",
            evidence.uncertainty,
        )
    return Evidence.error(
        evidence.provenance,
        evidence.error_type or "HybridObserverError",
        evidence.reason or "hybrid observer failed",
    )


def compile_hybrid_proposal(
    proposal: RuleProposal,
    *,
    issued_by: str = "canonical-bongard-verifier",
    observer: Callable[..., HybridObservation] = observe_hybrid_panel,
) -> CompiledProposal:
    """Attach one frozen HYBRID claim as a positive, task-local IR atom."""

    if proposal.hybrid_claim is None:
        raise SynthesisError("compile_hybrid_proposal requires a HYBRID proposal")
    if not issued_by.strip():
        raise SynthesisError("verifier issuer must be non-empty")
    proposal_digest = proposal.digest.removeprefix("sha256:")
    version = "hybrid-" + proposal_digest[:16]

    def empirical_claim(panel: object) -> Evidence[bool]:
        observation = observer(proposal, panel)
        if observation.proposal_digest != proposal.digest:
            raise SynthesisError("observer result belongs to a different proposal")
        return truth_from_hybrid_observation(observation)

    # The source digest is always recomputed from the Python closure body.  This
    # separate operational digest binds the exact proposal captured by it.
    operational_digest = hashlib.sha256(
        ("hybrid-observer/v1\x00" + proposal.digest).encode("utf-8")
    ).hexdigest()
    registry = LegRegistry()
    reference = registry.register(
        LegContract(
            name="hybrid_claim",
            version=version,
            domain=(PANEL,),
            codomain=BOOLEAN_WITNESS,
            implementation=empirical_claim,
            invariance=InvarianceContract(),
            semantics=LegSemantics.EMPIRICAL_WITNESS,
            operational_digest=operational_digest,
        )
    )
    registry.freeze()
    formula = Atom(
        call=StaticLegCall(reference, ("panel",)),
        relation=Relation.PRESENT,
        claim=proposal.hybrid_claim.operational_definition,
    )
    attachment = TypedAttachmentContract.issue(
        issued_by=issued_by,
        registry=registry,
        boundary_types={"panel": PANEL},
    )
    validate_formula(formula, registry, {"panel": PANEL})
    return CompiledProposal(proposal, registry, formula, attachment)


def compile_prototype_proposal(
    proposal: RuleProposal,
    feature_space: FrozenFeatureSpace,
    prototypes: FrozenSupportPrototypes,
    predicate: PositivePrototypeFormula,
    *,
    issued_by: str = "canonical-bongard-verifier",
) -> CompiledProposal:
    """Canonically attach a verifier-owned support-prototype surrogate.

    The visual proposal selects from a previously frozen feature catalog, but
    it does not own extraction, fitting, orientation, or the decision margin.
    The IR claim therefore states only the exact operational surrogate; it
    does not pretend that centroid similarity proves the proposal's prose.
    An outer pre-query artifact must additionally verify that these compiler
    inputs are the allowed projection selected by ``proposal``.
    """

    if proposal.is_hybrid:
        raise SynthesisError("prototype compilation requires a PURE proposal")
    if not issued_by.strip():
        raise SynthesisError("verifier issuer must be non-empty")
    validate_prototype_formula(predicate, prototypes, feature_space)
    expected_claim = (
        "fixed positive-support prototype match for visual proposal "
        + proposal.digest
    )
    if predicate.claim != expected_claim:
        raise SynthesisError(
            "prototype formula must use the canonical operational claim"
        )

    registry = LegRegistry()
    reference = register_support_prototype_leg(
        registry,
        predicate,
        prototypes,
        feature_space,
    )
    registry.freeze()
    formula = Atom(
        call=StaticLegCall(reference, ("features",)),
        relation=Relation.PRESENT,
        claim=expected_claim,
    )
    attachment = TypedAttachmentContract.issue(
        issued_by=issued_by,
        registry=registry,
        boundary_types={"features": SUPPORT_PROTOTYPE_FEATURES},
    )
    validate_formula(
        formula,
        registry,
        {"features": SUPPORT_PROTOTYPE_FEATURES},
    )
    return CompiledProposal(proposal, registry, formula, attachment)


__all__ = [
    "CompiledProposal",
    "HybridObserver",
    "SynthesisError",
    "compile_hybrid_proposal",
    "compile_prototype_proposal",
    "truth_from_hybrid_observation",
]
