"""Finite direct-predicate catalog over candidate-independent visual bundles.

Every advertised option has a deterministic extractor and exact-byte cold
replay.  Count targets are positive only: zero would smuggle a negated
existence claim into the support-only language.
"""

from __future__ import annotations

import hashlib
import json

from bongard.contour_witnesses import (
    CONTOUR_WITNESS_CAPABILITY_IDS,
    evaluate_contour_count_by_scenario,
)
from bongard.evidence import Disposition, Evidence, Provenance, Uncertainty
from bongard.typed_visual_proposal import (
    ArgumentKind,
    AtomArgument,
    RegisteredAtomCatalog,
    RegisteredAtomOption,
    RegisteredAtomSpec,
    TypedDeterministicAtom,
)
from bongard.visual_witnesses import (
    VISUAL_WITNESS_CAPABILITY_IDS,
    component_count_by_scenario,
    owned_hole_count_by_scenario,
)
from bongard.visual_witness_bundle import (
    VisualWitnessBundle,
    verify_visual_witness_bundle,
)


DIRECT_VISUAL_CATALOG_VERSION = "direct-visual-catalog/v2"
# Zero is an absence predicate disguised as equality (for example,
# ``hole.owner_count == 0`` means "has no owned hole").  The support-only
# language admits constructive positive witnesses only, so every count option
# starts at one.  Certified absence remains an *evaluation disposition* for a
# positive claim; it is never a proposer-selectable complement.
_COUNT_OPTIONS = tuple(range(1, 9))


class DirectVisualPredicateError(ValueError):
    """A direct selection does not belong to the executable visual catalog."""


def _digest(data: object) -> str:
    return hashlib.sha256(
        json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _count_options() -> tuple[RegisteredAtomOption, ...]:
    return tuple(
        RegisteredAtomOption.from_mapping("equal", {"target_count": count})
        for count in _COUNT_OPTIONS
    )


DIRECT_VISUAL_ATOM_CATALOG = RegisteredAtomCatalog(
    (
        RegisteredAtomSpec(
            catalog_key="component.count",
            affirmative_description=(
                "the panel has a registered exact number of separated ink components"
            ),
            arguments=(AtomArgument("target_count", ArgumentKind.INTEGER),),
            allowed_options=_count_options(),
        ),
        RegisteredAtomSpec(
            catalog_key="hole.owner_count",
            affirmative_description=(
                "the panel has a registered exact number of enclosed regions "
                "with one ink-component owner"
            ),
            arguments=(AtomArgument("target_count", ArgumentKind.INTEGER),),
            allowed_options=_count_options(),
        ),
        RegisteredAtomSpec(
            catalog_key="topology.endpoint_count",
            affirmative_description=(
                "the panel has a registered exact positive number of open "
                "stroke endpoints"
            ),
            arguments=(AtomArgument("target_count", ArgumentKind.INTEGER),),
            allowed_options=_count_options(),
        ),
        RegisteredAtomSpec(
            catalog_key="topology.branchpoint_count",
            affirmative_description=(
                "the panel has a registered exact positive number of stroke "
                "branchpoints"
            ),
            arguments=(AtomArgument("target_count", ArgumentKind.INTEGER),),
            allowed_options=_count_options(),
        ),
        RegisteredAtomSpec(
            catalog_key="topology.cycle_count",
            affirmative_description=(
                "the panel has a registered exact positive number of stroke cycles"
            ),
            arguments=(AtomArgument("target_count", ArgumentKind.INTEGER),),
            allowed_options=_count_options(),
        ),
        RegisteredAtomSpec(
            catalog_key="topology.crossing_count",
            affirmative_description=(
                "the panel has a registered exact positive number of certified "
                "four-arm X junctions"
            ),
            arguments=(AtomArgument("target_count", ArgumentKind.INTEGER),),
            allowed_options=_count_options(),
        ),
        RegisteredAtomSpec(
            catalog_key="curvature.reversal_count",
            affirmative_description=(
                "the panel has a registered exact positive number of persistent "
                "signed-curvature reversals on simple open strokes"
            ),
            arguments=(AtomArgument("target_count", ArgumentKind.INTEGER),),
            allowed_options=_count_options(),
        ),
        RegisteredAtomSpec(
            catalog_key="curvature.run_count",
            affirmative_description=(
                "the panel has a registered exact positive number of persistent "
                "signed-curvature runs on simple open strokes"
            ),
            arguments=(AtomArgument("target_count", ArgumentKind.INTEGER),),
            allowed_options=_count_options(),
        ),
        RegisteredAtomSpec(
            catalog_key="curvature.s_like_count",
            affirmative_description=(
                "the panel has a registered exact positive number of simple open "
                "strokes with certified reversing curvature"
            ),
            arguments=(AtomArgument("target_count", ArgumentKind.INTEGER),),
            allowed_options=_count_options(),
        ),
        RegisteredAtomSpec(
            catalog_key="curvature.u_like_count",
            affirmative_description=(
                "the panel has a registered exact positive number of simple open "
                "strokes with certified substantive one-direction curvature"
            ),
            arguments=(AtomArgument("target_count", ArgumentKind.INTEGER),),
            allowed_options=_count_options(),
        ),
    )
)

if tuple(
    sorted(VISUAL_WITNESS_CAPABILITY_IDS + CONTOUR_WITNESS_CAPABILITY_IDS)
) != tuple(
    atom.catalog_key for atom in sorted(
        DIRECT_VISUAL_ATOM_CATALOG.atoms, key=lambda item: item.catalog_key
    )
):
    raise RuntimeError(
        "direct visual catalog advertises a capability absent from the extractor"
    )


def direct_visual_catalog_digest() -> str:
    return DIRECT_VISUAL_ATOM_CATALOG.digest


def evaluate_direct_atom_by_scenario(
    bundle: VisualWitnessBundle,
    atom: TypedDeterministicAtom,
) -> dict[str, Evidence[bool]]:
    """Return one constructive truth value per retained packet scenario."""

    if not isinstance(bundle, VisualWitnessBundle):
        raise TypeError("bundle must be a VisualWitnessBundle")
    if not isinstance(atom, TypedDeterministicAtom):
        raise TypeError("atom must be a TypedDeterministicAtom")
    verify_visual_witness_bundle(bundle)
    spec = DIRECT_VISUAL_ATOM_CATALOG.get(atom.catalog_key)
    comparison, arguments = spec.canonical_selection(
        atom.comparison,
        dict(atom.arguments),
        atom.atom_id,
    )
    if comparison != "equal":  # pragma: no cover - closed grid above.
        raise DirectVisualPredicateError("v2 direct count atoms require equality")
    target = dict(arguments)["target_count"]
    if isinstance(target, bool) or not isinstance(target, int):
        raise DirectVisualPredicateError("target_count must be an integer")
    if atom.catalog_key == "component.count":
        base_result = component_count_by_scenario(bundle.base_packet, target)
    elif atom.catalog_key == "hole.owner_count":
        base_result = owned_hole_count_by_scenario(bundle.base_packet, target)
    elif atom.catalog_key in CONTOUR_WITNESS_CAPABILITY_IDS:
        contour_result = evaluate_contour_count_by_scenario(
            bundle.contour_packet, atom.catalog_key, target
        )
        base_result = None
    else:  # pragma: no cover - catalog lookup already closes the key set.
        raise DirectVisualPredicateError(
            f"unsupported direct visual capability {atom.catalog_key!r}"
        )

    selection_digest = _digest(
        {
            "version": DIRECT_VISUAL_CATALOG_VERSION,
            "atom": atom.to_data(),
            "bundle_digest": bundle.digest(),
        }
    )
    evidence: dict[str, Evidence[bool]] = {}
    if base_result is not None:
        observations = tuple(
            (
                item.scenario_id,
                item.observed_count,
                item.observed_count,
                Disposition.PRESENT
                if item.matches
                else Disposition.CERTIFIED_ABSENT,
            )
            for item in base_result.observations
        )
    else:
        observations = tuple(
            (
                item.scenario_id,
                item.observed.lower,
                item.observed.upper,
                Disposition(item.disposition),
            )
            for item in contour_result.observations
        )
    for scenario_id, lower, upper, disposition in observations:
        interval_text = str(lower) if lower == upper else f"[{lower},{upper}]"
        provenance = Provenance(
            producer="bongard.direct_visual_catalog",
            version="2",
            method=atom.catalog_key,
            input_digests=(bundle.digest(), selection_digest),
            artifact_digest=selection_digest,
            details=(
                ("observed_count_interval", interval_text),
                ("scenario_id", scenario_id),
                ("target_count", str(target)),
            ),
        )
        uncertainty = Uncertainty(
            float(lower),
            float(upper),
            causes=()
            if lower == upper
            else ("retained visual preprocessing or topology ambiguity",),
        )
        if disposition is Disposition.PRESENT:
            evidence[scenario_id] = Evidence.present(
                True, provenance, uncertainty
            )
        elif disposition is Disposition.CERTIFIED_ABSENT:
            evidence[scenario_id] = Evidence.certified_absent(
                provenance,
                f"scenario {scenario_id} observed count interval "
                f"{interval_text}, excluding expected count {target}",
                uncertainty,
            )
        else:
            evidence[scenario_id] = Evidence.indeterminate(
                provenance,
                f"scenario {scenario_id} observed count interval "
                f"{interval_text}, which contains but does not establish {target}",
                uncertainty,
            )
    return evidence


__all__ = [
    "DIRECT_VISUAL_ATOM_CATALOG",
    "DIRECT_VISUAL_CATALOG_VERSION",
    "DirectVisualPredicateError",
    "direct_visual_catalog_digest",
    "evaluate_direct_atom_by_scenario",
]
