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


def _empty_payload(view: p.FeatureAxisObservationView) -> dict[str, object]:
    return {
        item.alias: {
            "resolution": "complete",
            "variant_aliases": [],
            "evidence_x": -1,
            "evidence_y": -1,
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
    first = view.bindings[0]
    payload[first.alias] = {
        "resolution": "complete",
        "variant_aliases": [bird_alias],
        "evidence_x": first.search_region.minimum.x,
        "evidence_y": first.search_region.minimum.y,
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
        "variant_aliases": [],
        "evidence_x": -1,
        "evidence_y": -1,
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
    payload[view.bindings[0].alias]["variant_aliases"] = [bird_alias]
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
        "variant_aliases": [bird_alias],
        "evidence_x": outside.x,
        "evidence_y": outside.y,
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

