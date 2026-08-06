"""Pure-Python compiler for one frozen typed visual-semantic proposal.

Direct model selections lower to one joint-scenario Boolean leg.  An optional
soft claim lowers to one family-calibrated scalar leg.  A mixed proposal is
their outer positive conjunction; no proof-assistant or execution-backend
identity participates in the serialized semantics or dependency digests.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Mapping

from bongard.admission import TypedAttachmentContract
from bongard.artifacts import canonical_digest
from bongard.direct_visual_leg import (
    DirectVisualLowering,
    RegisteredDirectVisualPredicate,
    register_direct_visual_predicate,
)
from bongard.family_soft_leg import (
    RegisteredFamilySoftPredicate,
    register_family_soft_predicate,
)
from bongard.ir import AllOf, Formula, formula_digest, validate_formula
from bongard.legs import FROZEN_VISUAL_SCORE, LegReference, LegRegistry, ValueType
from bongard.semantic_policy import VisualSemanticPolicy
from bongard.semantic_protocol import (
    VISUAL_SOFT_WITNESS_INTERFACE_ID,
    visual_semantic_proposal_procedure_digest,
)
from bongard.soft_predicates import SoftScorerFamily, SoftScorerProtocol
from bongard.typed_visual_proposal import (
    MAX_DETERMINISTIC_ATOMS,
    MAX_SOFT_CUES,
    TypedSoftClaim,
    TypedSoftCue,
    TypedVisualProposal,
)
from bongard.visual_predicate_catalog import (
    DIRECT_VISUAL_ATOM_CATALOG,
    direct_visual_catalog_digest,
)
from bongard.visual_witness_bundle import (
    VISUAL_WITNESS_BUNDLE,
    VISUAL_WITNESS_BUNDLE_EXTRACTOR_ID,
    VISUAL_WITNESS_BUNDLE_VERSION,
    visual_witness_bundle_catalog_digest,
    visual_witness_bundle_extractor_digest,
)
from bongard.visual_witness_summaries import (
    visual_joint_soft_witness_interface_digest,
)
from bongard.visual_witnesses import VISUAL_WITNESS_SCENARIO_IDS


SEMANTIC_SYNTHESIS_ARCHIVE_SCHEMA = "gkm.bongard-semantic-synthesis-lowering.v1"
SOFT_CLAIM_LOWERING_SCHEMA = "gkm.bongard-soft-claim-lowering.v1"
DIRECT_BOUNDARY_NAME = "visual_witness_bundle"
SOFT_BOUNDARY_NAME = "soft_score"

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_ATOM_ID = re.compile(r"atom-[0-9]{2}\Z")


class SemanticSynthesisError(ValueError):
    """Proposal, policy, calibration, or lowering identities disagree."""


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise SemanticSynthesisError(f"{label} must be a lowercase SHA-256")
    return value


def _mapping(
    value: object, fields: frozenset[str], label: str
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise SemanticSynthesisError(f"{label} fields differ from the static schema")
    return value


def visual_proposal_protocol_digest(protocol: SoftScorerProtocol) -> str:
    """Return the sole proposal-procedure identity from semantic_protocol."""

    return visual_semantic_proposal_procedure_digest(protocol)


@dataclass(frozen=True, slots=True)
class SoftClaimLowering:
    """Canonical task-local soft claim retained before leg registration."""

    claim: TypedSoftClaim
    claim_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.claim, TypedSoftClaim):
            raise TypeError("claim must be TypedSoftClaim")
        _digest(self.claim_digest, "soft claim_digest")
        if canonical_digest(self.claim.to_data()) != self.claim_digest:
            raise SemanticSynthesisError("soft claim digest differs from claim data")

    @classmethod
    def from_claim(cls, claim: TypedSoftClaim) -> "SoftClaimLowering":
        if not isinstance(claim, TypedSoftClaim):
            raise TypeError("claim must be TypedSoftClaim")
        return cls(claim, canonical_digest(claim.to_data()))

    def to_data(self) -> dict[str, object]:
        return {
            "schema": SOFT_CLAIM_LOWERING_SCHEMA,
            "claim": self.claim.to_data(),
            "claim_digest": self.claim_digest,
        }

    @classmethod
    def from_data(cls, value: Mapping[str, Any]) -> "SoftClaimLowering":
        data = _mapping(
            value,
            frozenset({"schema", "claim", "claim_digest"}),
            "soft claim lowering",
        )
        if data["schema"] != SOFT_CLAIM_LOWERING_SCHEMA:
            raise SemanticSynthesisError("unsupported soft claim lowering schema")
        raw = _mapping(
            data["claim"],
            frozenset(
                {
                    "atom_id",
                    "positive_description",
                    "cues",
                    "aggregation",
                    "scorer_protocol_digest",
                }
            ),
            "lowered soft claim",
        )
        cues = raw["cues"]
        if not isinstance(cues, list):
            raise SemanticSynthesisError("lowered soft cues must be a list")
        decoded_cues: list[TypedSoftCue] = []
        for cue in cues:
            item = _mapping(
                cue,
                frozenset({"cue_id", "positive_description"}),
                "lowered soft cue",
            )
            decoded_cues.append(
                TypedSoftCue(item["cue_id"], item["positive_description"])
            )
        claim = TypedSoftClaim(
            atom_id=raw["atom_id"],
            positive_description=raw["positive_description"],
            cues=tuple(decoded_cues),
            aggregation=raw["aggregation"],
            scorer_protocol_digest=raw["scorer_protocol_digest"],
        )
        result = cls(claim=claim, claim_digest=data["claim_digest"])
        if result.to_data() != dict(value):
            raise SemanticSynthesisError(
                "soft claim lowering is not canonically represented"
            )
        return result


@dataclass(frozen=True, slots=True)
class CompositeAtomMapping:
    """Map original model atom IDs to one verifier-owned registered leg."""

    composite_id: str
    source_atom_ids: tuple[str, ...]
    leg_reference: LegReference

    def __post_init__(self) -> None:
        if self.composite_id not in {"direct-composite", "soft-calibrated"}:
            raise SemanticSynthesisError("unknown semantic composite ID")
        if not isinstance(self.source_atom_ids, tuple) or not self.source_atom_ids:
            raise SemanticSynthesisError("composite source atom IDs must be non-empty")
        if any(
            not isinstance(item, str) or _ATOM_ID.fullmatch(item) is None
            for item in self.source_atom_ids
        ):
            raise SemanticSynthesisError("composite contains an invalid source atom ID")
        if len(self.source_atom_ids) != len(set(self.source_atom_ids)):
            raise SemanticSynthesisError("composite repeats a source atom ID")
        if self.composite_id == "soft-calibrated" and len(self.source_atom_ids) != 1:
            raise SemanticSynthesisError("soft composite must map exactly one atom")
        if not isinstance(self.leg_reference, LegReference):
            raise TypeError("composite leg_reference must be LegReference")

    def to_data(self) -> dict[str, object]:
        return {
            "composite_id": self.composite_id,
            "source_atom_ids": list(self.source_atom_ids),
            "leg_reference": self.leg_reference.to_data(),
        }

    @classmethod
    def from_data(cls, value: Mapping[str, Any]) -> "CompositeAtomMapping":
        data = _mapping(
            value,
            frozenset({"composite_id", "source_atom_ids", "leg_reference"}),
            "composite atom mapping",
        )
        source = data["source_atom_ids"]
        reference = _mapping(
            data["leg_reference"],
            frozenset({"name", "version", "contract_digest"}),
            "composite leg reference",
        )
        if not isinstance(source, list) or any(not isinstance(x, str) for x in source):
            raise SemanticSynthesisError("source_atom_ids must be a string list")
        return cls(
            composite_id=data["composite_id"],
            source_atom_ids=tuple(source),
            leg_reference=LegReference(
                reference["name"], reference["version"], reference["contract_digest"]
            ),
        )


@dataclass(frozen=True, slots=True)
class VisualSemanticLoweringArchive:
    """Content-addressed compiler record independent of an execution backend."""

    proposal_digest: str
    policy_digest: str
    proposal_protocol_digest: str
    direct_catalog_digest: str
    scorer_protocol_digest: str
    scorer_family_digest: str
    family_development_manifest_digest: str
    original_formula_atom_ids: tuple[str, ...]
    direct_lowering: DirectVisualLowering | None
    soft_lowering: SoftClaimLowering | None
    composite_mapping: tuple[CompositeAtomMapping, ...]
    compiled_formula_digest: str
    registry_digest: str
    attachment_digest: str

    def __post_init__(self) -> None:
        for name in (
            "proposal_digest",
            "policy_digest",
            "proposal_protocol_digest",
            "direct_catalog_digest",
            "scorer_protocol_digest",
            "scorer_family_digest",
            "family_development_manifest_digest",
            "compiled_formula_digest",
            "registry_digest",
            "attachment_digest",
        ):
            _digest(getattr(self, name), name)
        if self.direct_catalog_digest != direct_visual_catalog_digest():
            raise SemanticSynthesisError("archive direct catalog digest drift")
        expected_ids = tuple(
            f"atom-{index:02d}" for index in range(len(self.original_formula_atom_ids))
        )
        if not expected_ids or self.original_formula_atom_ids != expected_ids:
            raise SemanticSynthesisError("archive original atom IDs are not canonical")
        if self.direct_lowering is None and self.soft_lowering is None:
            raise SemanticSynthesisError("archive cannot lower an empty proposal")
        expected_mapping: list[tuple[str, tuple[str, ...]]] = []
        if self.direct_lowering is not None:
            if self.direct_lowering.source_proposal_digest != self.proposal_digest:
                raise SemanticSynthesisError("direct lowering belongs to another proposal")
            expected_mapping.append(("direct-composite", self.direct_lowering.atom_ids))
        if self.soft_lowering is not None:
            if self.soft_lowering.claim.scorer_protocol_digest != self.scorer_protocol_digest:
                raise SemanticSynthesisError("soft lowering belongs to another protocol")
            expected_mapping.append(
                ("soft-calibrated", (self.soft_lowering.claim.atom_id,))
            )
        actual_mapping = [
            (item.composite_id, item.source_atom_ids) for item in self.composite_mapping
        ]
        if actual_mapping != expected_mapping:
            raise SemanticSynthesisError("archive composite mapping differs from lowering")
        flattened = tuple(
            atom_id for item in self.composite_mapping for atom_id in item.source_atom_ids
        )
        if flattened != self.original_formula_atom_ids:
            raise SemanticSynthesisError("composite mapping does not cover original formula")
        references = tuple(item.leg_reference for item in self.composite_mapping)
        if len(references) != len(set(references)):
            raise SemanticSynthesisError("composite mappings repeat a registered leg")

    def to_data(self) -> dict[str, object]:
        return {
            "schema": SEMANTIC_SYNTHESIS_ARCHIVE_SCHEMA,
            "proposal_digest": self.proposal_digest,
            "policy_digest": self.policy_digest,
            "proposal_protocol_digest": self.proposal_protocol_digest,
            "direct_catalog_digest": self.direct_catalog_digest,
            "scorer_protocol_digest": self.scorer_protocol_digest,
            "scorer_family_digest": self.scorer_family_digest,
            "family_development_manifest_digest": (
                self.family_development_manifest_digest
            ),
            "original_formula_atom_ids": list(self.original_formula_atom_ids),
            "direct_lowering": (
                None if self.direct_lowering is None else self.direct_lowering.to_data()
            ),
            "soft_lowering": (
                None if self.soft_lowering is None else self.soft_lowering.to_data()
            ),
            "composite_mapping": [item.to_data() for item in self.composite_mapping],
            "compiled_formula_digest": self.compiled_formula_digest,
            "registry_digest": self.registry_digest,
            "attachment_digest": self.attachment_digest,
        }

    @classmethod
    def from_data(cls, value: Mapping[str, Any]) -> "VisualSemanticLoweringArchive":
        fields = frozenset(
            {
                "schema",
                "proposal_digest",
                "policy_digest",
                "proposal_protocol_digest",
                "direct_catalog_digest",
                "scorer_protocol_digest",
                "scorer_family_digest",
                "family_development_manifest_digest",
                "original_formula_atom_ids",
                "direct_lowering",
                "soft_lowering",
                "composite_mapping",
                "compiled_formula_digest",
                "registry_digest",
                "attachment_digest",
            }
        )
        data = _mapping(value, fields, "semantic synthesis archive")
        if data["schema"] != SEMANTIC_SYNTHESIS_ARCHIVE_SCHEMA:
            raise SemanticSynthesisError("unsupported semantic synthesis archive")
        ids = data["original_formula_atom_ids"]
        mappings = data["composite_mapping"]
        if not isinstance(ids, list) or any(not isinstance(x, str) for x in ids):
            raise SemanticSynthesisError("original_formula_atom_ids must be a string list")
        if not isinstance(mappings, list) or any(
            not isinstance(item, Mapping) for item in mappings
        ):
            raise SemanticSynthesisError("composite_mapping must be an object list")
        direct_data = data["direct_lowering"]
        soft_data = data["soft_lowering"]
        if direct_data is not None and not isinstance(direct_data, Mapping):
            raise SemanticSynthesisError("direct_lowering must be an object or null")
        if soft_data is not None and not isinstance(soft_data, Mapping):
            raise SemanticSynthesisError("soft_lowering must be an object or null")
        result = cls(
            proposal_digest=data["proposal_digest"],
            policy_digest=data["policy_digest"],
            proposal_protocol_digest=data["proposal_protocol_digest"],
            direct_catalog_digest=data["direct_catalog_digest"],
            scorer_protocol_digest=data["scorer_protocol_digest"],
            scorer_family_digest=data["scorer_family_digest"],
            family_development_manifest_digest=data[
                "family_development_manifest_digest"
            ],
            original_formula_atom_ids=tuple(ids),
            direct_lowering=(
                None
                if direct_data is None
                else DirectVisualLowering.from_data(direct_data)
            ),
            soft_lowering=(
                None if soft_data is None else SoftClaimLowering.from_data(soft_data)
            ),
            composite_mapping=tuple(
                CompositeAtomMapping.from_data(item) for item in mappings
            ),
            compiled_formula_digest=data["compiled_formula_digest"],
            registry_digest=data["registry_digest"],
            attachment_digest=data["attachment_digest"],
        )
        if result.to_data() != dict(value):
            raise SemanticSynthesisError("semantic synthesis archive is not canonical")
        return result

    @property
    def digest(self) -> str:
        return canonical_digest(self.to_data())


@dataclass(frozen=True)
class CompiledVisualSemanticProposal:
    proposal: TypedVisualProposal
    policy: VisualSemanticPolicy
    family: SoftScorerFamily
    registry: LegRegistry
    formula: Formula
    attachment_contract: TypedAttachmentContract
    lowering_archive: VisualSemanticLoweringArchive

    def __post_init__(self) -> None:
        if not self.registry.frozen:
            raise SemanticSynthesisError("compiled registry must already be frozen")
        if self.proposal.digest != self.lowering_archive.proposal_digest:
            raise SemanticSynthesisError("compiled proposal differs from lowering archive")
        if self.policy.digest() != self.lowering_archive.policy_digest:
            raise SemanticSynthesisError("compiled policy differs from lowering archive")
        if self.family.digest() != self.lowering_archive.scorer_family_digest:
            raise SemanticSynthesisError("compiled family differs from lowering archive")
        if formula_digest(self.formula) != self.lowering_archive.compiled_formula_digest:
            raise SemanticSynthesisError("compiled formula differs from lowering archive")
        if self.registry.digest() != self.lowering_archive.registry_digest:
            raise SemanticSynthesisError("compiled registry differs from lowering archive")
        if self.attachment_contract.digest() != self.lowering_archive.attachment_digest:
            raise SemanticSynthesisError("attachment differs from lowering archive")
        self.attachment_contract.validate(self.formula, self.registry)

    @property
    def boundary_types(self) -> dict[str, ValueType]:
        return dict(self.attachment_contract.boundary_types)


def _validate_dependencies(
    *,
    policy: VisualSemanticPolicy,
    expected_policy_digest: str,
    family: SoftScorerFamily,
) -> str:
    if not isinstance(policy, VisualSemanticPolicy):
        raise TypeError("policy must be VisualSemanticPolicy")
    if VisualSemanticPolicy.from_data(policy.to_data()) != policy:
        raise SemanticSynthesisError("visual semantic policy is not canonical")
    expected = _digest(expected_policy_digest, "expected_policy_digest")
    if policy.digest() != expected:
        raise SemanticSynthesisError("visual semantic policy differs from commitment")
    if not isinstance(family, SoftScorerFamily):
        raise TypeError("family must be SoftScorerFamily")
    family.assert_untampered()
    family.verify_calibration()
    actual_witness_digest = visual_witness_bundle_extractor_digest()
    checks = (
        (
            policy.witness_extractor_id,
            VISUAL_WITNESS_BUNDLE_EXTRACTOR_ID,
            "witness extractor ID",
        ),
        (
            policy.witness_extractor_version,
            VISUAL_WITNESS_BUNDLE_VERSION,
            "witness extractor version",
        ),
        (policy.witness_extractor_digest, actual_witness_digest, "witness extractor digest"),
        (
            policy.witness_catalog_digest,
            visual_witness_bundle_catalog_digest(),
            "witness catalog digest",
        ),
        (policy.scenario_ids, VISUAL_WITNESS_SCENARIO_IDS, "witness scenario IDs"),
        (
            policy.direct_predicate_catalog_digest,
            direct_visual_catalog_digest(),
            "direct predicate catalog digest",
        ),
        (policy.soft_scorer_protocol_digest, family.protocol_digest, "scorer protocol digest"),
        (policy.soft_scorer_family_digest, family.digest(), "scorer family digest"),
        (
            policy.soft_family_development_manifest_digest,
            family.development_manifest_digest,
            "family development manifest digest",
        ),
        (
            family.protocol.witness_extractor_id,
            VISUAL_SOFT_WITNESS_INTERFACE_ID,
            "family soft-witness interface ID",
        ),
        (
            family.protocol.witness_extractor_digest,
            visual_joint_soft_witness_interface_digest(),
            "family soft-witness interface digest",
        ),
        (policy.max_direct_atoms, MAX_DETERMINISTIC_ATOMS, "direct atom limit"),
        (policy.max_soft_cues, MAX_SOFT_CUES, "soft cue limit"),
        (policy.max_soft_claims, 1, "soft claim limit"),
    )
    for actual, wanted, label in checks:
        if actual != wanted:
            raise SemanticSynthesisError(f"{label} differs from executable dependency")
    proposal_protocol = visual_proposal_protocol_digest(family.protocol)
    if policy.proposal_protocol_digest != proposal_protocol:
        raise SemanticSynthesisError("proposal protocol digest differs from policy")
    return proposal_protocol


def compile_visual_semantic_proposal(
    proposal: TypedVisualProposal,
    *,
    policy: VisualSemanticPolicy,
    expected_policy_digest: str,
    family: SoftScorerFamily,
    issued_by: str = "canonical-bongard-verifier",
) -> CompiledVisualSemanticProposal:
    """Compile a typed proposal into a frozen registry, formula, and attachment."""

    if not isinstance(proposal, TypedVisualProposal):
        raise TypeError("proposal must be TypedVisualProposal")
    if not isinstance(issued_by, str) or not issued_by.strip():
        raise SemanticSynthesisError("issued_by must be non-empty")
    proposal_protocol = _validate_dependencies(
        policy=policy,
        expected_policy_digest=expected_policy_digest,
        family=family,
    )
    if proposal.catalog_digest != policy.direct_predicate_catalog_digest:
        raise SemanticSynthesisError("proposal belongs to a different direct catalog")
    # Cold-decode through the strict boundary to reject hand-built or mutated
    # proposal objects before any registered closure captures them.
    try:
        canonical_proposal = TypedVisualProposal.from_data(
            proposal.to_data(),
            catalog=DIRECT_VISUAL_ATOM_CATALOG,
            expected_scorer_protocol_digest=policy.soft_scorer_protocol_digest,
        )
    except (TypeError, ValueError) as exc:
        raise SemanticSynthesisError(str(exc) or repr(exc)) from exc
    if canonical_proposal != proposal:
        raise SemanticSynthesisError("proposal is not canonically represented")
    if not proposal.deterministic_atoms and proposal.soft_claim is None:
        raise SemanticSynthesisError("proposal contains no executable atoms")
    if proposal.soft_claim is not None and (
        proposal.soft_claim.scorer_protocol_digest
        != policy.soft_scorer_protocol_digest
    ):
        raise SemanticSynthesisError("soft claim belongs to another scorer protocol")

    registry = LegRegistry()
    version = "proposal-" + proposal.digest[:16]
    direct_handle: RegisteredDirectVisualPredicate | None = None
    soft_handle: RegisteredFamilySoftPredicate | None = None
    direct_lowering: DirectVisualLowering | None = None
    soft_lowering: SoftClaimLowering | None = None
    terms: list[Formula] = []
    mappings: list[CompositeAtomMapping] = []
    boundary_types: dict[str, ValueType] = {}

    if proposal.deterministic_atoms:
        direct_handle = register_direct_visual_predicate(
            registry,
            name="visual_direct_composite",
            version=version,
            proposal=proposal,
            expected_catalog_digest=policy.direct_predicate_catalog_digest,
        )
        direct_lowering = direct_handle.lowering
        terms.append(direct_handle.atom(boundary_name=DIRECT_BOUNDARY_NAME))
        mappings.append(
            CompositeAtomMapping(
                "direct-composite",
                direct_lowering.atom_ids,
                direct_handle.reference,
            )
        )
        boundary_types[DIRECT_BOUNDARY_NAME] = VISUAL_WITNESS_BUNDLE

    if proposal.soft_claim is not None:
        soft_lowering = SoftClaimLowering.from_claim(proposal.soft_claim)
        soft_handle = register_family_soft_predicate(
            registry,
            name="visual_soft_claim",
            version=version,
            family=family,
            expected_protocol_digest=policy.soft_scorer_protocol_digest,
            expected_family_digest=policy.soft_scorer_family_digest,
            claim_digest=soft_lowering.claim_digest,
            claim_description=proposal.soft_claim.positive_description,
            cue_ids=tuple(cue.cue_id for cue in proposal.soft_claim.cues),
        )
        terms.append(soft_handle.atom(boundary_name=SOFT_BOUNDARY_NAME))
        mappings.append(
            CompositeAtomMapping(
                "soft-calibrated",
                (proposal.soft_claim.atom_id,),
                soft_handle.reference,
            )
        )
        boundary_types[SOFT_BOUNDARY_NAME] = FROZEN_VISUAL_SCORE

    if len(terms) == 1:
        formula = terms[0]
    elif len(terms) == 2:
        formula = AllOf(
            tuple(terms),
            "typed visual proposal requires its direct composite and calibrated soft claim",
        )
    else:  # Defensive even though the typed proposal already enforces this.
        raise SemanticSynthesisError("unsupported semantic formula shape")

    registry.freeze()
    validate_formula(formula, registry, boundary_types)
    attachment = TypedAttachmentContract.issue(
        issued_by=issued_by,
        registry=registry,
        boundary_types=boundary_types,
    )
    attachment.validate(formula, registry)
    archive = VisualSemanticLoweringArchive(
        proposal_digest=proposal.digest,
        policy_digest=policy.digest(),
        proposal_protocol_digest=proposal_protocol,
        direct_catalog_digest=direct_visual_catalog_digest(),
        scorer_protocol_digest=family.protocol_digest,
        scorer_family_digest=family.digest(),
        family_development_manifest_digest=family.development_manifest_digest,
        original_formula_atom_ids=proposal.formula.atom_ids,
        direct_lowering=direct_lowering,
        soft_lowering=soft_lowering,
        composite_mapping=tuple(mappings),
        compiled_formula_digest=formula_digest(formula),
        registry_digest=registry.digest(),
        attachment_digest=attachment.digest(),
    )
    return CompiledVisualSemanticProposal(
        proposal=proposal,
        policy=policy,
        family=family,
        registry=registry,
        formula=formula,
        attachment_contract=attachment,
        lowering_archive=archive,
    )


__all__ = [
    "DIRECT_BOUNDARY_NAME",
    "SEMANTIC_SYNTHESIS_ARCHIVE_SCHEMA",
    "SOFT_BOUNDARY_NAME",
    "SOFT_CLAIM_LOWERING_SCHEMA",
    "CompiledVisualSemanticProposal",
    "CompositeAtomMapping",
    "SemanticSynthesisError",
    "SoftClaimLowering",
    "VisualSemanticLoweringArchive",
    "compile_visual_semantic_proposal",
    "visual_proposal_protocol_digest",
]
