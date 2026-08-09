from __future__ import annotations

from copy import deepcopy
import inspect

import pytest

import bongard.panel_feature_closed_catalog_inventory as closed_catalog_module
import bongard.panel_feature_predicate as predicate_module
from bongard.canonical import canonical_digest
from bongard.panel_batched_typed_codex_observer import (
    complete_whole_panel_feature_axes,
)
from bongard.panel_feature_closed_catalog_inventory import (
    ClosedCatalogSupportGapKind,
    ClosedCatalogSupportInventory,
    ClosedCatalogSupportInventoryError,
    ClosedCatalogSupportInventoryStatus,
    cold_replay_closed_catalog_support_inventory,
    complete_whole_panel_feature_vocabulary,
)
from bongard.panel_feature_observation import (
    BindingFeatureObservation,
    BindingResolution,
    ObservationIssue,
    PanelAxisObservation,
    PanelFeatureObservationSet,
    PanelOnlyObservationContext,
    eligible_axis_bindings,
)
from bongard.panel_feature_observer_protocol import all_axis_variants
from bongard.panel_feature_predicate import (
    AllOf,
    EngineeringDisposition,
    enumerate_all_of,
    panel_feature_predicate_algorithm_digest,
)
from bongard.panel_feature_proposer import PanelFeatureProposerResult
from bongard.panel_soft_ontology import (
    FeatureFamily,
    GestaltKind,
    NativeOrientation,
    PanelFeatureSpec,
    QuantizedPoint,
    SymmetryKind,
)


_CONTRACT = "a" * 64
_PROTOCOL = "b" * 64


def _empty_proposer() -> PanelFeatureProposerResult:
    return PanelFeatureProposerResult(
        payload_digest="c" * 64,
        receipt_digest="d" * 64,
        nominations=(),
        language_gaps=(),
        nomination_gaps=(),
        observer_vocabulary=None,
    )


def _spec(family: FeatureFamily, kind: object) -> PanelFeatureSpec:
    matches = tuple(
        spec
        for axis in complete_whole_panel_feature_axes()
        if axis.family is family
        for spec in all_axis_variants(axis)
        if getattr(spec.parameters, "kind", None) is kind
    )
    assert len(matches) == 1
    return matches[0]


def _observation(
    ordinal: int,
    *,
    gestalt: GestaltKind | None = None,
    symmetry: SymmetryKind | None = None,
) -> PanelFeatureObservationSet:
    panel_digest = canonical_digest({"support-panel": ordinal})
    context = PanelOnlyObservationContext.create(
        panel_digest=panel_digest,
        observer_contract_digest=_CONTRACT,
        panel_context_receipt_digest=canonical_digest(
            {"panel-context": ordinal}
        ),
    )
    resolved = {
        FeatureFamily.GESTALT_RESEMBLANCE: (
            None
            if gestalt is None
            else _spec(FeatureFamily.GESTALT_RESEMBLANCE, gestalt)
        ),
        FeatureFamily.SYMMETRY: (
            None
            if symmetry is None
            else _spec(FeatureFamily.SYMMETRY, symmetry)
        ),
    }
    rows = []
    for axis in complete_whole_panel_feature_axes():
        bindings = eligible_axis_bindings(axis, context)
        assert len(bindings) == 1
        selected = resolved.get(axis.family)
        if selected is not None:
            row = BindingFeatureObservation(
                axis.axis_digest,
                bindings[0],
                BindingResolution.COMPLETE,
                (selected,),
                (QuantizedPoint(8, 8),),
                None,
                canonical_digest(
                    {"panel": panel_digest, "axis": axis.axis_digest}
                ),
            )
        else:
            row = BindingFeatureObservation(
                axis.axis_digest,
                bindings[0],
                BindingResolution.UNCLEAR,
                (),
                (),
                ObservationIssue.AMBIGUOUS_GEOMETRY,
                canonical_digest(
                    {"panel": panel_digest, "axis": axis.axis_digest}
                ),
            )
        rows.append(
            PanelAxisObservation(
                context,
                axis,
                _CONTRACT,
                _PROTOCOL,
                (row,),
            )
        )
    return PanelFeatureObservationSet(
        panel_digest,
        _CONTRACT,
        _PROTOCOL,
        tuple(rows),
    )


def _all_unclear_support() -> tuple[PanelFeatureObservationSet, ...]:
    return tuple(_observation(index) for index in range(12))


def _heterogeneous_negative_support() -> tuple[PanelFeatureObservationSet, ...]:
    # Positive side: A and B.  Each negative violates exactly one atom.
    positive = tuple(
        _observation(
            index,
            gestalt=GestaltKind.BIRD_LIKE,
            symmetry=SymmetryKind.REFLECTIONAL,
        )
        for index in range(6)
    )
    negative_a_only = tuple(
        _observation(
            index,
            gestalt=GestaltKind.BIRD_LIKE,
            symmetry=SymmetryKind.HALF_TURN,
        )
        for index in range(6, 9)
    )
    negative_b_only = tuple(
        _observation(
            index,
            gestalt=GestaltKind.ANIMAL_LIKE,
            symmetry=SymmetryKind.REFLECTIONAL,
        )
        for index in range(9, 12)
    )
    return positive + negative_a_only + negative_b_only


def test_complete_catalog_emits_typed_primary_gap_and_cold_replays() -> None:
    artifact = ClosedCatalogSupportInventory.create(
        _empty_proposer(),
        _all_unclear_support(),
        primary_orientation=NativeOrientation.SIDE0_POSITIVE,
    )

    assert len(artifact.vocabulary.specs) == 78
    assert artifact.vocabulary == complete_whole_panel_feature_vocabulary()
    assert len(artifact.primary_version_space.formulas) == 3_081
    assert artifact.proposer_snapshot.nominated_spec_digests == ()
    assert artifact.status is ClosedCatalogSupportInventoryStatus.PRIMARY_SUPPORT_GAP
    assert artifact.support_gap is not None
    assert (
        artifact.support_gap.kind
        is ClosedCatalogSupportGapKind.NO_PRIMARY_SUPPORT_CONSISTENT_FORMULA
    )
    assert artifact.support_gap.primary_formula_count == 3_081
    assert artifact.support_gap.primary_survivor_count == 0
    assert artifact.to_data()["query_pixels_included"] is False
    assert artifact.to_data()["cold_replay_model_calls"] == 0
    assert artifact.to_data()["candidate_catalog_selected_by_proposer"] is False
    assert artifact.to_data()["lean_required"] is False

    replayed = cold_replay_closed_catalog_support_inventory(
        artifact, expected_artifact_address=artifact.artifact_address
    )
    assert replayed == artifact
    assert ClosedCatalogSupportInventory.from_data(artifact.to_data()) == artifact


def test_every_optimized_formula_round_trips_through_public_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exhaust the finite catalog, not a sampled formula witness.

    The public implementation deliberately verifies its source on every
    constructor/serializer call.  Pinning that already-verified no-argument
    result inside this regression preserves the exact public semantics while
    avoiding thousands of redundant reads of the same sealed source bytes.
    """

    algorithm_digest = panel_feature_predicate_algorithm_digest()
    monkeypatch.setattr(
        predicate_module,
        "panel_feature_predicate_algorithm_digest",
        lambda: algorithm_digest,
    )
    vocabulary = complete_whole_panel_feature_vocabulary()
    for orientation in NativeOrientation:
        optimized = closed_catalog_module._complete_formula_inventory(orientation)
        authoritative = enumerate_all_of(vocabulary, orientation)

        assert len(optimized) == 3_081
        assert optimized == authoritative
        assert all(type(item) is AllOf for item in optimized)
        assert tuple(AllOf.from_data(item.to_data()) for item in optimized) == (
            optimized
        )


def test_composite_is_admitted_even_when_each_atom_fails_contrast_rule() -> None:
    artifact = ClosedCatalogSupportInventory.create(
        _empty_proposer(),
        _heterogeneous_negative_support(),
        primary_orientation=NativeOrientation.SIDE0_POSITIVE,
    )
    space = artifact.primary_version_space
    bird = _spec(FeatureFamily.GESTALT_RESEMBLANCE, GestaltKind.BIRD_LIKE)
    reflectional = _spec(FeatureFamily.SYMMETRY, SymmetryKind.REFLECTIONAL)
    by_atoms = {formula.spec_digests: formula for formula in space.formulas}
    bird_atom = by_atoms[(bird.spec_digest,)]
    symmetry_atom = by_atoms[(reflectional.spec_digest,)]
    conjunction_key = tuple(sorted((bird.spec_digest, reflectional.spec_digest)))
    conjunction = by_atoms[conjunction_key]

    assert bird_atom.formula_digest not in space.survivor_formula_digests
    assert symmetry_atom.formula_digest not in space.survivor_formula_digests
    assert conjunction.formula_digest in space.survivor_formula_digests
    assert space.survivor_formula_digests == (conjunction.formula_digest,)
    assert artifact.status is (
        ClosedCatalogSupportInventoryStatus.PRIMARY_VERSION_SPACE_NONEMPTY
    )
    assert artifact.support_gap is None
    assert artifact.opposite_diagnostic_version_space.survivor_formula_digests == ()

    formula_index = space.formulas.index(conjunction)
    row = space.rows[formula_index]
    assert row[:6] == (EngineeringDisposition.MATCH,) * 6
    assert row[6:] == (EngineeringDisposition.NONMATCH,) * 6


def test_proposer_is_retained_but_cannot_change_catalog_or_survivors() -> None:
    observations = _heterogeneous_negative_support()
    first = ClosedCatalogSupportInventory.create(
        _empty_proposer(),
        observations,
        primary_orientation=NativeOrientation.SIDE0_POSITIVE,
    )
    another_empty_proposer = PanelFeatureProposerResult(
        payload_digest="e" * 64,
        receipt_digest="f" * 64,
        nominations=(),
        language_gaps=(),
        nomination_gaps=(),
        observer_vocabulary=None,
    )
    second = ClosedCatalogSupportInventory.create(
        another_empty_proposer,
        observations,
        primary_orientation=NativeOrientation.SIDE0_POSITIVE,
    )

    assert first.proposer_snapshot != second.proposer_snapshot
    assert first.vocabulary == second.vocabulary
    assert first.support_table == second.support_table
    assert first.side0_version_space == second.side0_version_space
    assert first.side1_version_space == second.side1_version_space
    assert first.record_digest != second.record_digest


def test_tamper_and_callback_shaped_replay_inputs_fail_closed() -> None:
    artifact = ClosedCatalogSupportInventory.create(
        _empty_proposer(),
        _all_unclear_support(),
        primary_orientation=NativeOrientation.SIDE0_POSITIVE,
    )
    tampered = deepcopy(artifact.to_data())
    tampered["atom_level_contrast_prefilter"] = True
    with pytest.raises(ClosedCatalogSupportInventoryError):
        ClosedCatalogSupportInventory.from_data(tampered)

    tampered = deepcopy(artifact.to_data())
    tampered["proposer_snapshot"]["canonical_result_json"] += " "
    with pytest.raises(ClosedCatalogSupportInventoryError):
        ClosedCatalogSupportInventory.from_data(tampered)

    signature = inspect.signature(cold_replay_closed_catalog_support_inventory)
    assert tuple(signature.parameters) == (
        "archived",
        "expected_artifact_address",
    )
    with pytest.raises(TypeError):
        cold_replay_closed_catalog_support_inventory(  # type: ignore[call-arg]
            artifact, model_callback=lambda: None
        )
