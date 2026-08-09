from __future__ import annotations

from copy import deepcopy
import json

import pytest

import bongard.panel_feature_observation as f
import bongard.panel_soft_ontology as o


def _d(char: str) -> str:
    return char * 64


def _region(x0: int, y0: int, x1: int, y1: int) -> o.QuantizedRegion:
    return o.QuantizedRegion(o.QuantizedPoint(x0, y0), o.QuantizedPoint(x1, y1))


def _inventory(*, complete: bool = True, three_figures: bool = False) -> o.OwnerInventory:
    owners = [
        o.PanelLocalOwner(
            o.OwnerId("owner_0001"), o.OwnerKind.FIGURE, _region(0, 0, 6, 6)
        ),
        o.PanelLocalOwner(
            o.OwnerId("owner_0002"), o.OwnerKind.FIGURE, _region(8, 0, 15, 7)
        ),
    ]
    if three_figures:
        owners.append(
            o.PanelLocalOwner(
                o.OwnerId("owner_0003"), o.OwnerKind.FIGURE, _region(4, 9, 10, 15)
            )
        )
    return o.OwnerInventory(
        _d("a"),
        _d("b"),
        o.EnumerationResolution.GRID16_FULL_PANEL,
        _d("c"),
        complete,
        tuple(owners),
    )


def _gestalt(kind: o.GestaltKind) -> o.PanelFeatureSpec:
    return o.PanelFeatureSpec(
        o.FeatureFamily.GESTALT_RESEMBLANCE,
        o.SubjectScope.ONE_COHERENT_FIGURE,
        o.ReferenceFrame.NONE,
        o.GestaltResemblanceParameters(kind),
    )


def _point(kind: o.PointContactKind) -> o.PanelFeatureSpec:
    return o.PanelFeatureSpec(
        o.FeatureFamily.POINT_CONTACT,
        o.SubjectScope.FIGURE_PAIR,
        o.ReferenceFrame.NONE,
        o.PointContactParameters(kind),
    )


def _row(
    axis: f.FeatureAxis,
    binding: o.SubjectBinding,
    *,
    resolution: f.BindingResolution = f.BindingResolution.COMPLETE,
    observed: tuple[o.PanelFeatureSpec, ...] = (),
    point: o.QuantizedPoint | None = None,
    issue: f.ObservationIssue | None = None,
    receipt: str = _d("d"),
) -> f.BindingFeatureObservation:
    ordered = tuple(sorted(observed, key=lambda item: item.spec_digest))
    return f.BindingFeatureObservation(
        axis.axis_digest,
        binding,
        resolution,
        ordered,
        () if point is None else (point,),
        issue,
        receipt,
    )


def _gestalt_axis_observation(
    *,
    inventory: o.OwnerInventory | None = None,
    second_resolution: f.BindingResolution = f.BindingResolution.COMPLETE,
    second_issue: f.ObservationIssue | None = None,
) -> tuple[f.PanelAxisObservation, o.PanelFeatureSpec, o.PanelFeatureSpec]:
    inventory = _inventory() if inventory is None else inventory
    bird = _gestalt(o.GestaltKind.BIRD_LIKE)
    tool = _gestalt(o.GestaltKind.TOOL_LIKE)
    axis = f.FeatureAxis.for_spec(bird)
    bindings = f.eligible_axis_bindings(axis, inventory)
    owner_regions = {item.owner_id: item.region for item in inventory.owners}
    rows = []
    for index, binding in enumerate(bindings):
        region = owner_regions[binding.owner_ids[0]]
        if index == 0:
            rows.append(
                _row(
                    axis,
                    binding,
                    observed=(bird,),
                    point=region.minimum,
                    receipt=_d("d"),
                )
            )
        elif second_resolution is f.BindingResolution.COMPLETE:
            rows.append(
                _row(
                    axis,
                    binding,
                    observed=(bird,),
                    point=region.minimum,
                    receipt=_d("e"),
                )
            )
        else:
            rows.append(
                _row(
                    axis,
                    binding,
                    resolution=second_resolution,
                    issue=second_issue,
                    receipt=_d("e"),
                )
            )
    return (
        f.PanelAxisObservation(inventory, axis, _d("f"), _d("1"), tuple(rows)),
        bird,
        tool,
    )


def test_feature_axis_is_candidate_parameter_independent() -> None:
    bird = _gestalt(o.GestaltKind.BIRD_LIKE)
    tool = _gestalt(o.GestaltKind.TOOL_LIKE)
    assert bird.spec_digest != tool.spec_digest
    assert f.FeatureAxis.for_spec(bird) == f.FeatureAxis.for_spec(tool)
    assert "parameters" not in f.FeatureAxis.for_spec(bird).to_data()


def test_one_complete_axis_observation_evaluates_sibling_specs_afterward() -> None:
    observation, bird, tool = _gestalt_axis_observation()
    assert observation.evaluate(bird) is f.EngineeringFeatureDisposition.MATCH
    assert observation.evaluate(tool) is f.EngineeringFeatureDisposition.NONMATCH


def test_complete_empty_rows_do_not_manufacture_operational_nonmatch() -> None:
    inventory = _inventory()
    bird = _gestalt(o.GestaltKind.BIRD_LIKE)
    axis = f.FeatureAxis.for_spec(bird)
    rows = tuple(
        _row(axis, binding) for binding in f.eligible_axis_bindings(axis, inventory)
    )
    observation = f.PanelAxisObservation(
        inventory, axis, _d("a"), _d("b"), rows
    )
    assert (
        observation.evaluate(bird)
        is f.EngineeringFeatureDisposition.INDETERMINATE
    )


def test_unresolved_binding_prevents_operational_nonmatch() -> None:
    observation, _, tool = _gestalt_axis_observation(
        second_resolution=f.BindingResolution.UNCLEAR,
        second_issue=f.ObservationIssue.AMBIGUOUS_GEOMETRY,
    )
    assert observation.evaluate(tool) is f.EngineeringFeatureDisposition.INDETERMINATE


def test_incomplete_owner_inventory_prevents_operational_nonmatch() -> None:
    observation, bird, tool = _gestalt_axis_observation(
        inventory=_inventory(complete=False)
    )
    assert observation.evaluate(bird) is f.EngineeringFeatureDisposition.MATCH
    assert observation.evaluate(tool) is f.EngineeringFeatureDisposition.INDETERMINATE


def test_empty_eligible_domain_is_a_typed_gap_not_vacuous_nonmatch() -> None:
    inventory = o.OwnerInventory(
        _d("a"),
        _d("b"),
        o.EnumerationResolution.GRID16_FULL_PANEL,
        _d("c"),
        True,
        (
            o.PanelLocalOwner(
                o.OwnerId("owner_0001"),
                o.OwnerKind.TRACE,
                _region(0, 0, 15, 15),
            ),
        ),
    )
    bird = _gestalt(o.GestaltKind.BIRD_LIKE)
    axis = f.FeatureAxis.for_spec(bird)
    assert f.eligible_axis_bindings(axis, inventory) == ()
    with pytest.raises(f.PanelFeatureObservationError, match="typed unresolved gap"):
        f.PanelAxisObservation(inventory, axis, _d("d"), _d("e"), ())
    gap = f.EligibleDomainGap.unverified_empty(inventory, axis)
    observation = f.PanelAxisObservation(
        inventory, axis, _d("d"), _d("e"), (), gap
    )
    assert (
        observation.evaluate(bird)
        is f.EngineeringFeatureDisposition.INDETERMINATE
    )
    assert f.PanelAxisObservation.from_data(observation.to_data()) == observation
    tampered = deepcopy(observation.to_data())
    tampered["domain_gap"]["eligible_binding_count"] = 1
    with pytest.raises(f.PanelFeatureObservationError):
        f.PanelAxisObservation.from_data(tampered)


def test_positive_witness_wins_but_uncovered_error_is_not_negative() -> None:
    observation, bird, tool = _gestalt_axis_observation(
        second_resolution=f.BindingResolution.ERROR,
        second_issue=f.ObservationIssue.TRANSPORT_FAILURE,
    )
    assert observation.evaluate(bird) is f.EngineeringFeatureDisposition.MATCH
    assert observation.evaluate(tool) is f.EngineeringFeatureDisposition.ERROR


def test_panel_axis_requires_every_eligible_binding_once() -> None:
    inventory = _inventory(three_figures=True)
    spec = _point(o.PointContactKind.TANGENTIAL)
    axis = f.FeatureAxis.for_spec(spec)
    bindings = f.eligible_axis_bindings(axis, inventory)
    assert len(bindings) == 3
    rows = tuple(_row(axis, binding, receipt=_d(str(index + 1))) for index, binding in enumerate(bindings))
    f.PanelAxisObservation(inventory, axis, _d("a"), _d("b"), rows)
    with pytest.raises(f.PanelFeatureObservationError, match="eligible binding"):
        f.PanelAxisObservation(inventory, axis, _d("a"), _d("b"), rows[:-1])


def test_evidence_must_be_inside_derived_binding_region() -> None:
    inventory = _inventory()
    bird = _gestalt(o.GestaltKind.BIRD_LIKE)
    axis = f.FeatureAxis.for_spec(bird)
    bindings = f.eligible_axis_bindings(axis, inventory)
    rows = []
    for binding in bindings:
        rows.append(
            _row(
                axis,
                binding,
                observed=(bird,),
                point=o.QuantizedPoint(15, 15),
            )
        )
    with pytest.raises(f.PanelFeatureObservationError, match="outside"):
        f.PanelAxisObservation(inventory, axis, _d("a"), _d("b"), tuple(rows))


def test_observation_set_missing_axis_is_indeterminate() -> None:
    observation, bird, _ = _gestalt_axis_observation()
    observation_set = f.PanelFeatureObservationSet(
        observation.inventory,
        observation.observer_contract_digest,
        observation.measurement_protocol_digest,
        (observation,),
    )
    point = _point(o.PointContactKind.TANGENTIAL)
    assert observation_set.evaluate(bird) is f.EngineeringFeatureDisposition.MATCH
    assert (
        observation_set.evaluate(point)
        is f.EngineeringFeatureDisposition.INDETERMINATE
    )


def test_roundtrip_is_canonical_and_engineering_boundary_is_explicit() -> None:
    observation, bird, _ = _gestalt_axis_observation()
    observation_set = f.PanelFeatureObservationSet(
        observation.inventory,
        observation.observer_contract_digest,
        observation.measurement_protocol_digest,
        (observation,),
    )
    restored = f.PanelFeatureObservationSet.from_data(observation_set.to_data())
    assert restored == observation_set
    cell = f.EngineeringFeatureCell.evaluate(restored, bird)
    assert f.EngineeringFeatureCell.from_data(cell.to_data()) == cell
    rendered = json.dumps(observation_set.to_data(), sort_keys=True).lower()
    assert "lean" not in rendered
    assert '"engineering_only": true' in rendered
    assert '"scientific_calibration_supplied": false' in rendered


def test_tampering_with_coverage_or_policy_fails_closed() -> None:
    observation, _, _ = _gestalt_axis_observation()
    data = observation.to_data()
    tampered = deepcopy(data)
    tampered["candidate_parameter_visible_during_measurement"] = True
    with pytest.raises(f.PanelFeatureObservationError):
        f.PanelAxisObservation.from_data(tampered)
    tampered = deepcopy(data)
    tampered["binding_observations"].pop()
    with pytest.raises(f.PanelFeatureObservationError):
        f.PanelAxisObservation.from_data(tampered)


def test_resolution_issue_matrix_rejects_false_negative_shortcuts() -> None:
    inventory = _inventory()
    spec = _gestalt(o.GestaltKind.BIRD_LIKE)
    axis = f.FeatureAxis.for_spec(spec)
    binding = f.eligible_axis_bindings(axis, inventory)[0]
    with pytest.raises(f.PanelFeatureObservationError):
        _row(
            axis,
            binding,
            resolution=f.BindingResolution.UNCLEAR,
            issue=f.ObservationIssue.TRANSPORT_FAILURE,
        )
    with pytest.raises(f.PanelFeatureObservationError):
        _row(
            axis,
            binding,
            resolution=f.BindingResolution.COMPLETE,
            issue=f.ObservationIssue.AMBIGUOUS_GEOMETRY,
        )


def test_complete_inventory_derives_count_before_candidate_comparison() -> None:
    inventory = _inventory()
    two = o.PanelFeatureSpec(
        o.FeatureFamily.COMPONENT_COUNT,
        o.SubjectScope.WHOLE_PANEL,
        o.ReferenceFrame.NONE,
        o.ComponentCountParameters(o.ClosedCount.TWO),
    )
    three = o.PanelFeatureSpec(
        o.FeatureFamily.COMPONENT_COUNT,
        o.SubjectScope.WHOLE_PANEL,
        o.ReferenceFrame.NONE,
        o.ComponentCountParameters(o.ClosedCount.THREE),
    )
    observation = f.derive_inventory_count_observation(
        inventory,
        f.FeatureAxis.for_spec(two),
        observer_contract_digest=_d("a"),
        measurement_protocol_digest=_d("b"),
    )
    assert observation.evaluate(two) is f.EngineeringFeatureDisposition.MATCH
    assert observation.evaluate(three) is f.EngineeringFeatureDisposition.NONMATCH
    rendered = json.dumps(observation.to_data(), sort_keys=True)
    assert '"count": "two"' in rendered
    assert '"count": "three"' not in rendered


def test_count_derivation_uses_coherent_roots_and_descendant_segments() -> None:
    root_trace = o.PanelLocalOwner(
        o.OwnerId("owner_0001"), o.OwnerKind.TRACE, _region(0, 0, 6, 6)
    )
    root_loop = o.PanelLocalOwner(
        o.OwnerId("owner_0002"), o.OwnerKind.LOOP, _region(8, 0, 15, 7)
    )
    components = o.OwnerInventory(
        _d("a"),
        _d("b"),
        o.EnumerationResolution.GRID16_FULL_PANEL,
        _d("c"),
        True,
        (root_trace, root_loop),
    )
    two = o.PanelFeatureSpec(
        o.FeatureFamily.COMPONENT_COUNT,
        o.SubjectScope.WHOLE_PANEL,
        o.ReferenceFrame.NONE,
        o.ComponentCountParameters(o.ClosedCount.TWO),
    )
    component_observation = f.derive_inventory_count_observation(
        components,
        f.FeatureAxis.for_spec(two),
        observer_contract_digest=_d("d"),
        measurement_protocol_digest=_d("e"),
    )
    assert (
        component_observation.evaluate(two)
        is f.EngineeringFeatureDisposition.MATCH
    )

    figure = o.PanelLocalOwner(
        o.OwnerId("owner_0001"), o.OwnerKind.FIGURE, _region(0, 0, 15, 15)
    )
    trace = o.PanelLocalOwner(
        o.OwnerId("owner_0002"),
        o.OwnerKind.TRACE,
        _region(1, 1, 14, 14),
        (figure.owner_id,),
    )
    segment = o.PanelLocalOwner(
        o.OwnerId("owner_0003"),
        o.OwnerKind.SEGMENT,
        _region(2, 2, 10, 2),
        (trace.owner_id,),
    )
    nested = o.OwnerInventory(
        _d("a"),
        _d("b"),
        o.EnumerationResolution.GRID16_FULL_PANEL,
        _d("c"),
        True,
        (figure, trace, segment),
    )
    one_segment = o.PanelFeatureSpec(
        o.FeatureFamily.EXACT_SEGMENT_COUNT,
        o.SubjectScope.ONE_COHERENT_FIGURE,
        o.ReferenceFrame.NONE,
        o.ExactSegmentCountParameters(o.ClosedCount.ONE),
    )
    segment_observation = f.derive_inventory_count_observation(
        nested,
        f.FeatureAxis.for_spec(one_segment),
        observer_contract_digest=_d("d"),
        measurement_protocol_digest=_d("e"),
    )
    assert (
        segment_observation.evaluate(one_segment)
        is f.EngineeringFeatureDisposition.MATCH
    )
