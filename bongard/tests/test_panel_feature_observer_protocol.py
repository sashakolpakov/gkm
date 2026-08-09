from __future__ import annotations

from copy import deepcopy
import json

import pytest

import bongard.panel_feature_observation as f
import bongard.panel_feature_observer_protocol as p
import bongard.panel_soft_ontology as o
from bongard.transport import validate_codex_strict_output_schema


def _d(char: str) -> str:
    return char * 64


def _region(index: int) -> o.QuantizedRegion:
    x = (index % 4) * 4
    y = (index // 4) * 4
    return o.QuantizedRegion(
        o.QuantizedPoint(x, y),
        o.QuantizedPoint(min(x + 3, 15), min(y + 3, 15)),
    )


def _inventory(count: int = 2, *, complete: bool = True) -> o.OwnerInventory:
    return o.OwnerInventory(
        _d("a"),
        _d("b"),
        o.EnumerationResolution.GRID16_FULL_PANEL,
        _d("c"),
        complete,
        tuple(
            o.PanelLocalOwner(
                o.OwnerId(f"owner_{index + 1:04d}"),
                o.OwnerKind.FIGURE,
                _region(index),
            )
            for index in range(count)
        ),
    )


def _gestalt(kind: o.GestaltKind) -> o.PanelFeatureSpec:
    return o.PanelFeatureSpec(
        o.FeatureFamily.GESTALT_RESEMBLANCE,
        o.SubjectScope.ONE_COHERENT_FIGURE,
        o.ReferenceFrame.NONE,
        o.GestaltResemblanceParameters(kind),
    )


def _contact() -> o.PanelFeatureSpec:
    return o.PanelFeatureSpec(
        o.FeatureFamily.POINT_CONTACT,
        o.SubjectScope.FIGURE_PAIR,
        o.ReferenceFrame.NONE,
        o.PointContactParameters(o.PointContactKind.TANGENTIAL),
    )


def _straight_count(count: o.ClosedCount) -> o.PanelFeatureSpec:
    return o.PanelFeatureSpec(
        o.FeatureFamily.STRAIGHT_SEGMENT_COUNT,
        o.SubjectScope.WHOLE_PANEL,
        o.ReferenceFrame.NONE,
        o.StraightSegmentCountParameters(count),
    )


def _convexity(kind: o.ConvexityKind) -> o.PanelFeatureSpec:
    return o.PanelFeatureSpec(
        o.FeatureFamily.CONVEXITY,
        o.SubjectScope.WHOLE_PANEL,
        o.ReferenceFrame.NONE,
        o.ConvexityParameters(kind),
    )


def _empty_payload(view: p.FeatureAxisObservationView) -> dict[str, object]:
    return {
        item.alias: {
            "resolution": "complete",
            "variant_evidence": [],
            "issue": "none",
        }
        for item in view.bindings
    }


def test_axis_variant_catalog_is_full_not_candidate_selected() -> None:
    bird = _gestalt(o.GestaltKind.BIRD_LIKE)
    axis = f.FeatureAxis.for_spec(bird)
    variants = p.all_axis_variants(axis)
    assert len(variants) == len(o.GestaltKind) == 6
    assert {item.parameters.kind for item in variants} == set(o.GestaltKind)

    marker = o.PanelFeatureSpec(
        o.FeatureFamily.MARKER_PATTERN,
        o.SubjectScope.ONE_COHERENT_FIGURE,
        o.ReferenceFrame.NONE,
        o.MarkerPatternParameters(
            o.MarkerPrimitive.DOT,
            o.ClosedCount.ONE,
            o.MarkerArrangement.LINEAR,
        ),
    )
    assert len(p.all_axis_variants(f.FeatureAxis.for_spec(marker))) == 5 * 12 * 4


def test_model_view_is_role_blind_and_omits_target_designation() -> None:
    view = p.FeatureAxisObservationView.build(
        _inventory(), f.FeatureAxis.for_spec(_gestalt(o.GestaltKind.BIRD_LIKE))
    )
    rendered = json.dumps(view.model_data(), sort_keys=True).lower()
    assert "owner_" not in rendered
    assert "spec_digest" not in rendered
    assert "candidate" not in rendered
    assert "orientation" not in rendered
    assert "side0" not in rendered and "side1" not in rendered
    assert "lean" not in rendered
    assert {row["variant_alias"] for row in view.model_data()["registered_variants"]} == {
        item.alias for item in view.variants
    }


def test_strict_payload_resolves_variants_then_python_compares_specs() -> None:
    inventory = _inventory()
    bird = _gestalt(o.GestaltKind.BIRD_LIKE)
    tool = _gestalt(o.GestaltKind.TOOL_LIKE)
    view = p.FeatureAxisObservationView.build(inventory, f.FeatureAxis.for_spec(bird))
    payload = _empty_payload(view)
    bird_alias = next(item.alias for item in view.variants if item.spec == bird)
    for binding in view.bindings:
        payload[binding.alias] = {
            "resolution": "complete",
            "variant_evidence": [{
                "variant_alias": bird_alias,
                "evidence_x": binding.search_region.minimum.x,
                "evidence_y": binding.search_region.minimum.y,
            }],
            "issue": "none",
        }
    observation = p.parse_feature_axis_observer_payload(
        view,
        payload,
        observer_contract_digest=_d("d"),
        measurement_protocol_digest=_d("e"),
        observation_receipt_digest=_d("f"),
    )
    assert observation.evaluate(bird) is f.EngineeringFeatureDisposition.MATCH
    assert observation.evaluate(tool) is f.EngineeringFeatureDisposition.NONMATCH
    # The target choices are not present in the prompt/schema as preferred values.
    validate_codex_strict_output_schema(p.feature_axis_observer_output_schema(view))
    prompt = p.feature_axis_observer_prompt(view)
    assert "preferred answer" in prompt
    assert "group 0" not in prompt.lower() and "group 1" not in prompt.lower()


def test_straight_count_is_derived_only_from_explicit_line_records() -> None:
    context = f.panel_only_observation_inventory(
        panel_digest=_d("a"),
        observer_contract_digest=_d("d"),
        panel_context_receipt_digest=_d("c"),
    )
    two = _straight_count(o.ClosedCount.TWO)
    one = _straight_count(o.ClosedCount.ONE)
    view = p.FeatureAxisObservationView.build(context, f.FeatureAxis.for_spec(two))
    schema = p.feature_axis_observer_output_schema(view)
    validate_codex_strict_output_schema(schema)
    row_schema = schema["properties"][view.bindings[0].alias]
    assert "straight_segment_evidence" in row_schema["properties"]
    assert "variant_evidence" not in row_schema["properties"]
    prompt = p.feature_axis_observer_prompt(view)
    assert "structural contour or boundary segments" in prompt
    assert "not generic segment owners" in prompt
    assert "Do not select a count alias" in prompt

    payload = {
        view.bindings[0].alias: {
            "resolution": "complete",
            # Endpoint direction and record order are deliberately noncanonical;
            # Python canonicalizes geometry before deriving the count.
            "straight_segment_evidence": [
                {"start_x": 12, "start_y": 12, "end_x": 8, "end_y": 8},
                {"start_x": 6, "start_y": 6, "end_x": 2, "end_y": 2},
            ],
            "issue": "none",
        }
    }
    observation = p.parse_feature_axis_observer_payload(
        view,
        payload,
        observer_contract_digest=_d("d"),
        measurement_protocol_digest=_d("e"),
        observation_receipt_digest=_d("f"),
    )
    row = observation.binding_observations[0]
    assert row.observed_specs == (two,)
    assert len(row.straight_segment_evidence) == 2
    assert row.evidence_points == ()
    assert observation.evaluate(two) is f.EngineeringFeatureDisposition.MATCH
    assert observation.evaluate(one) is f.EngineeringFeatureDisposition.NONMATCH


def test_missing_or_out_of_catalog_straightness_stays_indeterminate() -> None:
    context = f.panel_only_observation_inventory(
        panel_digest=_d("a"),
        observer_contract_digest=_d("d"),
        panel_context_receipt_digest=_d("c"),
    )
    one = _straight_count(o.ClosedCount.ONE)
    view = p.FeatureAxisObservationView.build(context, f.FeatureAxis.for_spec(one))
    binding_alias = view.bindings[0].alias

    unclear = p.parse_feature_axis_observer_payload(
        view,
        {
            binding_alias: {
                "resolution": "unclear",
                "straight_segment_evidence": [],
                "issue": "missing_straightness_evidence",
            }
        },
        observer_contract_digest=_d("d"),
        measurement_protocol_digest=_d("e"),
        observation_receipt_digest=_d("f"),
    )
    assert (
        unclear.binding_observations[0].issue
        is f.ObservationIssue.MISSING_STRAIGHTNESS_EVIDENCE
    )
    assert unclear.evaluate(one) is f.EngineeringFeatureDisposition.INDETERMINATE

    zero = p.parse_feature_axis_observer_payload(
        view,
        {
            binding_alias: {
                "resolution": "complete",
                "straight_segment_evidence": [],
                "issue": "none",
            }
        },
        observer_contract_digest=_d("d"),
        measurement_protocol_digest=_d("e"),
        observation_receipt_digest=_d("f"),
    )
    assert zero.binding_observations[0].resolution is f.BindingResolution.UNCLEAR
    assert (
        zero.binding_observations[0].issue
        is f.ObservationIssue.OUTSIDE_CLOSED_CATALOG
    )
    assert zero.evaluate(one) is f.EngineeringFeatureDisposition.INDETERMINATE


def test_convexity_parser_derives_variant_and_canonicalizes_boundary_walk() -> None:
    context = f.panel_only_observation_inventory(
        panel_digest=_d("a"),
        observer_contract_digest=_d("d"),
        panel_context_receipt_digest=_d("c"),
    )
    convex = _convexity(o.ConvexityKind.CONVEX_CLOSED_BOUNDARY)
    concave = _convexity(o.ConvexityKind.CONCAVE_CLOSED_BOUNDARY)
    view = p.FeatureAxisObservationView.build(
        context, f.FeatureAxis.for_spec(convex)
    )
    schema = p.feature_axis_observer_output_schema(view)
    validate_codex_strict_output_schema(schema)
    row_schema = schema["properties"][view.bindings[0].alias]
    assert "outer_boundary_vertices" in row_schema["properties"]
    assert "variant_evidence" not in row_schema["properties"]
    prompt = p.feature_axis_observer_prompt(view)
    assert "Do not select a variant alias" in prompt
    assert "bare convex=true/false" in prompt
    assert "exact integer cross products" in prompt

    walks = (
        ((1, 1), (1, 12), (12, 12), (12, 1), (1, 1)),
        ((12, 12), (1, 12), (1, 1), (12, 1), (12, 12)),
    )
    observations = []
    for walk in walks:
        observations.append(
            p.parse_feature_axis_observer_payload(
                view,
                {
                    view.bindings[0].alias: {
                        "resolution": "complete",
                        "outer_boundary_vertices": [
                            {"x": x, "y": y} for x, y in walk
                        ],
                        "issue": "none",
                    }
                },
                observer_contract_digest=_d("d"),
                measurement_protocol_digest=_d("e"),
                observation_receipt_digest=_d("f"),
            )
        )
    assert all(
        item.evaluate(convex) is f.EngineeringFeatureDisposition.MATCH
        for item in observations
    )
    assert all(
        item.evaluate(concave) is f.EngineeringFeatureDisposition.NONMATCH
        for item in observations
    )
    assert (
        observations[0].binding_observations[0].outer_boundary_evidence
        == observations[1].binding_observations[0].outer_boundary_evidence
    )

    concave_observation = p.parse_feature_axis_observer_payload(
        view,
        {
            view.bindings[0].alias: {
                "resolution": "complete",
                "outer_boundary_vertices": [
                    {"x": x, "y": y}
                    for x, y in (
                        (1, 1),
                        (12, 1),
                        (6, 6),
                        (12, 12),
                        (1, 12),
                        (1, 1),
                    )
                ],
                "issue": "none",
            }
        },
        observer_contract_digest=_d("d"),
        measurement_protocol_digest=_d("e"),
        observation_receipt_digest=_d("f"),
    )
    assert (
        concave_observation.evaluate(concave)
        is f.EngineeringFeatureDisposition.MATCH
    )


@pytest.mark.parametrize(
    ("walk", "expected_issue"),
    [
        (
            ((1, 1), (12, 1), (12, 12), (1, 12)),
            f.ObservationIssue.OPEN_BOUNDARY,
        ),
        (
            ((1, 1), (12, 12), (1, 12), (12, 1), (1, 1)),
            f.ObservationIssue.SELF_INTERSECTING_BOUNDARY,
        ),
        (
            ((1, 1), (6, 1), (12, 1), (1, 1)),
            f.ObservationIssue.DEGENERATE_BOUNDARY,
        ),
    ],
)
def test_invalid_complete_boundary_walks_become_typed_indeterminate(
    walk: tuple[tuple[int, int], ...],
    expected_issue: f.ObservationIssue,
) -> None:
    context = f.panel_only_observation_inventory(
        panel_digest=_d("a"),
        observer_contract_digest=_d("d"),
        panel_context_receipt_digest=_d("c"),
    )
    convex = _convexity(o.ConvexityKind.CONVEX_CLOSED_BOUNDARY)
    view = p.FeatureAxisObservationView.build(
        context, f.FeatureAxis.for_spec(convex)
    )
    observation = p.parse_feature_axis_observer_payload(
        view,
        {
            view.bindings[0].alias: {
                "resolution": "complete",
                "outer_boundary_vertices": [
                    {"x": x, "y": y} for x, y in walk
                ],
                "issue": "none",
            }
        },
        observer_contract_digest=_d("d"),
        measurement_protocol_digest=_d("e"),
        observation_receipt_digest=_d("f"),
    )
    row = observation.binding_observations[0]
    assert row.resolution is f.BindingResolution.UNCLEAR
    assert row.issue is expected_issue
    assert row.outer_boundary_evidence is None
    assert (
        observation.evaluate(convex)
        is f.EngineeringFeatureDisposition.INDETERMINATE
    )


def test_uncertain_convexity_cannot_claim_partial_boundary_evidence() -> None:
    context = f.panel_only_observation_inventory(
        panel_digest=_d("a"),
        observer_contract_digest=_d("d"),
        panel_context_receipt_digest=_d("c"),
    )
    convex = _convexity(o.ConvexityKind.CONVEX_CLOSED_BOUNDARY)
    view = p.FeatureAxisObservationView.build(
        context, f.FeatureAxis.for_spec(convex)
    )
    payload = {
        view.bindings[0].alias: {
            "resolution": "unclear",
            "outer_boundary_vertices": [],
            "issue": "missing_boundary_evidence",
        }
    }
    observation = p.parse_feature_axis_observer_payload(
        view,
        payload,
        observer_contract_digest=_d("d"),
        measurement_protocol_digest=_d("e"),
        observation_receipt_digest=_d("f"),
    )
    assert (
        observation.evaluate(convex)
        is f.EngineeringFeatureDisposition.INDETERMINATE
    )
    payload[view.bindings[0].alias]["outer_boundary_vertices"] = [
        {"x": 1, "y": 1}
    ]
    with pytest.raises(p.PanelFeatureObserverProtocolError, match="resolved evidence"):
        p.parse_feature_axis_observer_payload(
            view,
            payload,
            observer_contract_digest=_d("d"),
            measurement_protocol_digest=_d("e"),
            observation_receipt_digest=_d("f"),
        )


def test_complete_empty_payload_stays_indeterminate() -> None:
    bird = _gestalt(o.GestaltKind.BIRD_LIKE)
    view = p.FeatureAxisObservationView.build(
        _inventory(), f.FeatureAxis.for_spec(bird)
    )
    observation = p.parse_feature_axis_observer_payload(
        view,
        _empty_payload(view),
        observer_contract_digest=_d("d"),
        measurement_protocol_digest=_d("e"),
        observation_receipt_digest=_d("f"),
    )
    assert observation.evaluate(bird) is f.EngineeringFeatureDisposition.INDETERMINATE


def test_missing_binding_and_false_empty_shortcuts_are_rejected() -> None:
    view = p.FeatureAxisObservationView.build(
        _inventory(), f.FeatureAxis.for_spec(_gestalt(o.GestaltKind.BIRD_LIKE))
    )
    payload = _empty_payload(view)
    payload.pop(view.bindings[-1].alias)
    with pytest.raises(p.PanelFeatureObserverProtocolError, match="fields"):
        p.parse_feature_axis_observer_payload(
            view,
            payload,
            observer_contract_digest=_d("d"),
            measurement_protocol_digest=_d("e"),
            observation_receipt_digest=_d("f"),
        )

    payload = _empty_payload(view)
    payload[view.bindings[0].alias]["issue"] = "ambiguous_geometry"
    with pytest.raises(p.PanelFeatureObserverProtocolError, match="inconsistent"):
        p.parse_feature_axis_observer_payload(
            view,
            payload,
            observer_contract_digest=_d("d"),
            measurement_protocol_digest=_d("e"),
            observation_receipt_digest=_d("f"),
        )


def test_unclear_row_stays_indeterminate_and_cannot_claim_variants() -> None:
    bird = _gestalt(o.GestaltKind.BIRD_LIKE)
    view = p.FeatureAxisObservationView.build(_inventory(), f.FeatureAxis.for_spec(bird))
    payload = _empty_payload(view)
    payload[view.bindings[0].alias] = {
        "resolution": "unclear",
        "variant_evidence": [],
        "issue": "ambiguous_geometry",
    }
    observation = p.parse_feature_axis_observer_payload(
        view,
        payload,
        observer_contract_digest=_d("d"),
        measurement_protocol_digest=_d("e"),
        observation_receipt_digest=_d("f"),
    )
    tool = _gestalt(o.GestaltKind.TOOL_LIKE)
    assert observation.evaluate(tool) is f.EngineeringFeatureDisposition.INDETERMINATE

    bird_alias = next(item.alias for item in view.variants if item.spec == bird)
    payload[view.bindings[0].alias]["variant_evidence"] = [{
        "variant_alias": bird_alias,
        "evidence_x": 1,
        "evidence_y": 1,
    }]
    with pytest.raises(p.PanelFeatureObserverProtocolError, match="resolved evidence"):
        p.parse_feature_axis_observer_payload(
            view,
            payload,
            observer_contract_digest=_d("d"),
            measurement_protocol_digest=_d("e"),
            observation_receipt_digest=_d("f"),
        )


def test_out_of_binding_evidence_is_rejected_by_typed_observation() -> None:
    bird = _gestalt(o.GestaltKind.BIRD_LIKE)
    view = p.FeatureAxisObservationView.build(_inventory(), f.FeatureAxis.for_spec(bird))
    payload = _empty_payload(view)
    bird_alias = next(item.alias for item in view.variants if item.spec == bird)
    first = view.bindings[0]
    outside = o.QuantizedPoint(15, 15)
    assert not (
        first.search_region.minimum.x <= outside.x <= first.search_region.maximum.x
        and first.search_region.minimum.y <= outside.y <= first.search_region.maximum.y
    )
    payload[first.alias] = {
        "resolution": "complete",
        "variant_evidence": [{
            "variant_alias": bird_alias,
            "evidence_x": outside.x,
            "evidence_y": outside.y,
        }],
        "issue": "none",
    }
    with pytest.raises(p.PanelFeatureObserverProtocolError, match="typed observation"):
        p.parse_feature_axis_observer_payload(
            view,
            payload,
            observer_contract_digest=_d("d"),
            measurement_protocol_digest=_d("e"),
            observation_receipt_digest=_d("f"),
        )


def test_capacity_gap_yields_unclear_rows_not_absence() -> None:
    inventory = _inventory(10)
    spec = _contact()
    axis = f.FeatureAxis.for_spec(spec)
    assert len(f.eligible_axis_bindings(axis, inventory)) == 45
    with pytest.raises(p.PanelFeatureObserverProtocolError, match="capacity"):
        p.FeatureAxisObservationView.build(inventory, axis)
    observation = p.unresolved_axis_observation(
        inventory,
        axis,
        observer_contract_digest=_d("d"),
        measurement_protocol_digest=_d("e"),
        observation_receipt_digest=_d("f"),
        issue=f.ObservationIssue.CAPACITY_LIMIT,
    )
    assert observation.evaluate(spec) is f.EngineeringFeatureDisposition.INDETERMINATE


def test_zero_binding_payload_creates_unverified_domain_gap() -> None:
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
                _region(0),
            ),
        ),
    )
    bird = _gestalt(o.GestaltKind.BIRD_LIKE)
    view = p.FeatureAxisObservationView.build(inventory, f.FeatureAxis.for_spec(bird))
    assert view.bindings == ()
    observation = p.parse_feature_axis_observer_payload(
        view,
        {},
        observer_contract_digest=_d("d"),
        measurement_protocol_digest=_d("e"),
        observation_receipt_digest=_d("f"),
    )
    assert observation.domain_gap == f.EligibleDomainGap.unverified_empty(
        inventory, view.axis
    )
    assert observation.evaluate(bird) is f.EngineeringFeatureDisposition.INDETERMINATE


def test_view_roundtrip_rejects_catalog_and_policy_tampering() -> None:
    view = p.FeatureAxisObservationView.build(
        _inventory(), f.FeatureAxis.for_spec(_gestalt(o.GestaltKind.BIRD_LIKE))
    )
    assert p.FeatureAxisObservationView.from_data(view.to_data()) == view
    tampered = deepcopy(view.to_data())
    tampered["variants"].pop()
    with pytest.raises(p.PanelFeatureObserverProtocolError, match="full variant"):
        p.FeatureAxisObservationView.from_data(tampered)
    tampered = deepcopy(view.to_data())
    tampered["candidate_parameter_in_view"] = True
    with pytest.raises(p.PanelFeatureObserverProtocolError, match="policy"):
        p.FeatureAxisObservationView.from_data(tampered)
