"""Registered IR bridge for one proposal-local direct visual conjunction.

The proposer selects one to three options from the verifier-owned direct
catalog.  This module lowers those original selections into one Boolean leg:
all selected atoms in the joint witness bundle are evaluated inside each retained preprocessing scenario,
then the complete scenario outcomes are compared.  The lowering archive keeps
the original atom IDs and option arguments; it never rewrites them into prose.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from bongard.artifacts import canonical_digest, canonical_json
from bongard.evidence import Disposition, Evidence, Provenance
from bongard.ir import Atom, Relation, StaticLegCall
from bongard.legs import (
    BOOLEAN_WITNESS,
    AffirmativeRelation,
    InvarianceContract,
    LegContract,
    LegReference,
    LegRegistry,
    LegSemantics,
)
from bongard.scenario_semantics import (
    JOINT_SCENARIO_SEMANTICS_VERSION,
    ScenarioConjunctionResult,
    evaluate_joint_scenario_conjunction,
)
from bongard.typed_visual_proposal import (
    MAX_DETERMINISTIC_ATOMS,
    TypedDeterministicAtom,
    TypedVisualProposal,
    TypedVisualProposalError,
)
from bongard.visual_predicate_catalog import (
    DIRECT_VISUAL_ATOM_CATALOG,
    DIRECT_VISUAL_CATALOG_VERSION,
    direct_visual_catalog_digest,
    evaluate_direct_atom_by_scenario,
)
from bongard.visual_witness_bundle import (
    VISUAL_WITNESS_BUNDLE,
    VisualWitnessBundle,
    verify_visual_witness_bundle,
    visual_witness_bundle_catalog_digest,
    visual_witness_bundle_extractor_digest,
)
from bongard.visual_witnesses import VISUAL_WITNESS_SCENARIO_IDS


DIRECT_VISUAL_LOWERING_SCHEMA = "gkm.bongard-direct-visual-lowering.v1"
DIRECT_VISUAL_LEG_SCHEMA = "gkm.bongard-direct-visual-leg.v1"


class DirectVisualLegError(ValueError):
    """A direct proposal cannot be lowered or registered without ambiguity."""


def _require_digest(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise DirectVisualLegError(f"{label} must be a lowercase SHA-256")
    return value


def _strict_mapping(
    value: object, expected: frozenset[str], label: str
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise DirectVisualLegError(f"{label} fields differ from the static schema")
    if any(not isinstance(key, str) for key in value):
        raise DirectVisualLegError(f"{label} keys must be strings")
    return value


def _validate_atom(atom: TypedDeterministicAtom) -> TypedDeterministicAtom:
    if not isinstance(atom, TypedDeterministicAtom):
        raise TypeError("direct lowering atoms must be TypedDeterministicAtom values")
    spec = DIRECT_VISUAL_ATOM_CATALOG.get(atom.catalog_key)
    comparison, arguments = spec.canonical_selection(
        atom.comparison, dict(atom.arguments), atom.atom_id
    )
    if comparison != atom.comparison or arguments != atom.arguments:
        raise DirectVisualLegError("direct atom is not canonically represented")
    return atom


@dataclass(frozen=True, slots=True)
class DirectVisualLowering:
    """Archive-safe lowering of the direct portion of one typed proposal."""

    source_proposal_digest: str
    positive_description: str
    catalog_digest: str
    atoms: tuple[TypedDeterministicAtom, ...]

    def __post_init__(self) -> None:
        _require_digest(self.source_proposal_digest, "source_proposal_digest")
        if (
            not isinstance(self.positive_description, str)
            or not self.positive_description
            or self.positive_description != self.positive_description.strip()
        ):
            raise DirectVisualLegError(
                "positive_description must be non-empty exact text"
            )
        _require_digest(self.catalog_digest, "catalog_digest")
        if self.catalog_digest != direct_visual_catalog_digest():
            raise DirectVisualLegError(
                "direct lowering belongs to a different registered catalog"
            )
        if not isinstance(self.atoms, tuple) or not 1 <= len(self.atoms) <= (
            MAX_DETERMINISTIC_ATOMS
        ):
            raise DirectVisualLegError(
                f"direct lowering must contain 1..{MAX_DETERMINISTIC_ATOMS} atoms"
            )
        for atom in self.atoms:
            _validate_atom(atom)
        expected_ids = tuple(f"atom-{index:02d}" for index in range(len(self.atoms)))
        if tuple(atom.atom_id for atom in self.atoms) != expected_ids:
            raise DirectVisualLegError(
                "direct lowering must preserve canonical proposal atom IDs"
            )
        selections = tuple(
            canonical_json(
                {
                    "catalog_key": atom.catalog_key,
                    "comparison": atom.comparison,
                    "arguments": dict(atom.arguments),
                }
            )
            for atom in self.atoms
        )
        if len(selections) != len(set(selections)):
            raise DirectVisualLegError("direct lowering repeats a catalog option")

    @property
    def atom_ids(self) -> tuple[str, ...]:
        return tuple(atom.atom_id for atom in self.atoms)

    def to_data(self) -> dict[str, object]:
        return {
            "schema": DIRECT_VISUAL_LOWERING_SCHEMA,
            "source_proposal_digest": self.source_proposal_digest,
            "positive_description": self.positive_description,
            "catalog_version": DIRECT_VISUAL_CATALOG_VERSION,
            "catalog_digest": self.catalog_digest,
            "joint_semantics_version": JOINT_SCENARIO_SEMANTICS_VERSION,
            "selected_atoms": [atom.to_data() for atom in self.atoms],
            "formula": {"kind": "all", "atom_ids": list(self.atom_ids)},
        }

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "DirectVisualLowering":
        raw = _strict_mapping(
            data,
            frozenset(
                {
                    "schema",
                    "source_proposal_digest",
                    "positive_description",
                    "catalog_version",
                    "catalog_digest",
                    "joint_semantics_version",
                    "selected_atoms",
                    "formula",
                }
            ),
            "direct visual lowering",
        )
        if raw["schema"] != DIRECT_VISUAL_LOWERING_SCHEMA:
            raise DirectVisualLegError("unsupported direct visual lowering schema")
        if raw["catalog_version"] != DIRECT_VISUAL_CATALOG_VERSION:
            raise DirectVisualLegError("direct visual catalog version drift")
        if raw["joint_semantics_version"] != JOINT_SCENARIO_SEMANTICS_VERSION:
            raise DirectVisualLegError("joint scenario semantics version drift")
        selected = raw["selected_atoms"]
        if not isinstance(selected, list) or not 1 <= len(selected) <= (
            MAX_DETERMINISTIC_ATOMS
        ):
            raise DirectVisualLegError(
                f"selected_atoms must be a list of 1..{MAX_DETERMINISTIC_ATOMS} objects"
            )
        atoms: list[TypedDeterministicAtom] = []
        for index, value in enumerate(selected):
            item = _strict_mapping(
                value,
                frozenset({"atom_id", "catalog_key", "comparison", "arguments"}),
                f"selected_atoms[{index}]",
            )
            expected_id = f"atom-{index:02d}"
            if item["atom_id"] != expected_id:
                raise DirectVisualLegError(
                    f"selected_atoms[{index}].atom_id must remain {expected_id!r}"
                )
            key = item["catalog_key"]
            if not isinstance(key, str):
                raise DirectVisualLegError("selected atom catalog_key must be text")
            arguments = item["arguments"]
            if not isinstance(arguments, Mapping):
                raise DirectVisualLegError("selected atom arguments must be an object")
            try:
                spec = DIRECT_VISUAL_ATOM_CATALOG.get(key)
                comparison, canonical_arguments = spec.canonical_selection(
                    item["comparison"], arguments, expected_id
                )
            except TypedVisualProposalError as exc:
                raise DirectVisualLegError(str(exc)) from exc
            atoms.append(
                TypedDeterministicAtom(
                    atom_id=expected_id,
                    catalog_key=key,
                    comparison=comparison,
                    arguments=canonical_arguments,
                )
            )
        formula = _strict_mapping(
            raw["formula"], frozenset({"kind", "atom_ids"}), "lowered formula"
        )
        expected_ids = [atom.atom_id for atom in atoms]
        if formula["kind"] != "all" or formula["atom_ids"] != expected_ids:
            raise DirectVisualLegError(
                "lowered formula must preserve every original direct atom ID in order"
            )
        result = cls(
            source_proposal_digest=raw["source_proposal_digest"],
            positive_description=raw["positive_description"],
            catalog_digest=raw["catalog_digest"],
            atoms=tuple(atoms),
        )
        if result.to_data() != dict(data):
            raise DirectVisualLegError(
                "direct visual lowering is not the exact canonical representation"
            )
        return result

    @property
    def digest(self) -> str:
        return canonical_digest(self.to_data())

    def assert_untampered(self) -> None:
        if DirectVisualLowering.from_data(self.to_data()) != self:
            raise DirectVisualLegError("direct visual lowering changed after freeze")


def lower_direct_visual_proposal(
    proposal: TypedVisualProposal,
) -> DirectVisualLowering:
    """Freeze the deterministic selections of one canonical typed proposal."""

    if not isinstance(proposal, TypedVisualProposal):
        raise TypeError("proposal must be a TypedVisualProposal")
    if proposal.catalog_digest != direct_visual_catalog_digest():
        raise DirectVisualLegError(
            "typed proposal belongs to a different direct visual catalog"
        )
    if not proposal.deterministic_atoms:
        raise DirectVisualLegError("typed proposal contains no direct visual atoms")
    return DirectVisualLowering(
        source_proposal_digest=proposal.digest,
        positive_description=proposal.positive_description,
        catalog_digest=proposal.catalog_digest,
        atoms=proposal.deterministic_atoms,
    )


def _operational_digest(lowering: DirectVisualLowering) -> str:
    return canonical_digest(
        {
            "schema": DIRECT_VISUAL_LEG_SCHEMA,
            "direct_catalog_version": DIRECT_VISUAL_CATALOG_VERSION,
            "direct_catalog_digest": direct_visual_catalog_digest(),
            "visual_witness_bundle_catalog_digest": (
                visual_witness_bundle_catalog_digest()
            ),
            "visual_witness_bundle_extractor_digest": (
                visual_witness_bundle_extractor_digest()
            ),
            "selected_atoms": [atom.to_data() for atom in lowering.atoms],
            "joint_semantics_version": JOINT_SCENARIO_SEMANTICS_VERSION,
            "joint_semantics": (
                "complete conjunction inside each correlated scenario, then "
                "four-disposition scenario consensus"
            ),
            "lowering_digest": lowering.digest,
        }
    )


def _transfer_joint_result(
    result: ScenarioConjunctionResult,
    *,
    base_provenance: Provenance,
    operational_digest: str,
    bundle_digest: str,
) -> Evidence[bool]:
    provenance = Provenance(
        producer="bongard.direct_visual_leg",
        version="1",
        method="proposal_local_complete_joint_scenario_conjunction",
        input_digests=(
            base_provenance.digest(),
            bundle_digest,
            result.evidence.provenance.digest(),
        ),
        artifact_digest=operational_digest,
        details=(
            ("joint_result_digest", result.digest),
            ("scenario_count", str(len(result.scenario_dispositions))),
        ),
    )
    if result.evidence.disposition is Disposition.PRESENT:
        return Evidence.present(True, provenance)
    if result.evidence.disposition is Disposition.CERTIFIED_ABSENT:
        return Evidence.certified_absent(
            provenance,
            result.evidence.certificate
            or "complete direct conjunction is absent in every scenario",
        )
    if result.evidence.disposition is Disposition.INDETERMINATE:
        return Evidence.indeterminate(
            provenance,
            result.evidence.reason or "direct preprocessing scenarios disagree",
        )
    return Evidence.error(
        provenance,
        result.evidence.error_type or "DirectScenarioError",
        result.evidence.reason or "direct scenario evaluation failed",
    )


@dataclass(frozen=True, slots=True)
class RegisteredDirectVisualPredicate:
    """Static handle for one registered proposal-local direct conjunction."""

    lowering: DirectVisualLowering
    operational_digest: str
    reference: LegReference

    def __post_init__(self) -> None:
        if not isinstance(self.lowering, DirectVisualLowering):
            raise TypeError("lowering must be DirectVisualLowering")
        _require_digest(self.operational_digest, "operational_digest")
        if self.operational_digest != _operational_digest(self.lowering):
            raise DirectVisualLegError(
                "registered operational digest differs from the lowering"
            )
        if not isinstance(self.reference, LegReference):
            raise TypeError("reference must be LegReference")

    def atom(self, *, boundary_name: str = "visual_witness_bundle") -> Atom:
        if (
            not isinstance(boundary_name, str)
            or not boundary_name
            or boundary_name != boundary_name.strip()
        ):
            raise DirectVisualLegError("boundary_name must be non-empty exact text")
        return Atom(
            call=StaticLegCall(self.reference, (boundary_name,)),
            relation=Relation.PRESENT,
            claim=(
                "joint-scenario registered direct conjunction for: "
                + self.lowering.positive_description
            ),
        )


def register_direct_visual_predicate(
    registry: LegRegistry,
    *,
    name: str,
    version: str,
    proposal: TypedVisualProposal,
    expected_catalog_digest: str,
    cost: int = 1,
) -> RegisteredDirectVisualPredicate:
    """Register the complete deterministic conjunction of one typed proposal."""

    if not isinstance(registry, LegRegistry):
        raise TypeError("registry must be a verifier-owned LegRegistry")
    expected = _require_digest(expected_catalog_digest, "expected_catalog_digest")
    if expected != direct_visual_catalog_digest():
        raise DirectVisualLegError(
            "direct catalog differs from the verifier commitment"
        )
    lowering = lower_direct_visual_proposal(proposal)
    lowering.assert_untampered()
    operational_digest = _operational_digest(lowering)
    base_provenance = Provenance(
        producer="bongard.direct_visual_leg",
        version="1",
        method="frozen_proposal_local_direct_conjunction",
        input_digests=(lowering.source_proposal_digest, lowering.digest),
        artifact_digest=operational_digest,
        details=(
            ("catalog_digest", expected),
            ("joint_semantics_version", JOINT_SCENARIO_SEMANTICS_VERSION),
            ("selected_atom_count", str(len(lowering.atoms))),
        ),
    )

    def direct_visual_conjunction(bundle: object) -> Evidence[bool]:
        try:
            lowering.assert_untampered()
            if direct_visual_catalog_digest() != expected:
                raise DirectVisualLegError(
                    "direct visual catalog changed after registration"
                )
            if _operational_digest(lowering) != operational_digest:
                raise DirectVisualLegError(
                    "direct visual operation changed after registration"
                )
            if not isinstance(bundle, VisualWitnessBundle):
                raise TypeError(
                    "direct visual leg requires a VisualWitnessBundle"
                )
            verify_visual_witness_bundle(bundle)
            evidence_by_scenario: dict[str, dict[str, Evidence[bool]]] = {
                scenario_id: {} for scenario_id in VISUAL_WITNESS_SCENARIO_IDS
            }
            for selected_atom in lowering.atoms:
                atom_evidence = evaluate_direct_atom_by_scenario(
                    bundle, selected_atom
                )
                if tuple(atom_evidence) != VISUAL_WITNESS_SCENARIO_IDS:
                    raise DirectVisualLegError(
                        "direct atom evaluation dropped or reordered scenarios"
                    )
                for scenario_id in VISUAL_WITNESS_SCENARIO_IDS:
                    evidence_by_scenario[scenario_id][selected_atom.atom_id] = (
                        atom_evidence[scenario_id]
                    )
            joint = evaluate_joint_scenario_conjunction(evidence_by_scenario)
        except (TypeError, ValueError) as exc:
            return Evidence.error(
                base_provenance, type(exc).__name__, str(exc) or repr(exc)
            )
        return _transfer_joint_result(
            joint,
            base_provenance=base_provenance,
            operational_digest=operational_digest,
            bundle_digest=bundle.digest(),
        )

    reference = registry.register(
        LegContract(
            name=name,
            version=version,
            domain=(VISUAL_WITNESS_BUNDLE,),
            codomain=BOOLEAN_WITNESS,
            implementation=direct_visual_conjunction,
            affirmative_relations=frozenset({AffirmativeRelation.PRESENT}),
            invariance=InvarianceContract(),
            semantics=LegSemantics.DERIVED,
            cost=cost,
            operational_digest=operational_digest,
        )
    )
    return RegisteredDirectVisualPredicate(
        lowering=lowering,
        operational_digest=operational_digest,
        reference=reference,
    )


__all__ = [
    "DIRECT_VISUAL_LEG_SCHEMA",
    "DIRECT_VISUAL_LOWERING_SCHEMA",
    "DirectVisualLegError",
    "DirectVisualLowering",
    "RegisteredDirectVisualPredicate",
    "lower_direct_visual_proposal",
    "register_direct_visual_predicate",
]
