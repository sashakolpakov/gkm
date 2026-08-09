from __future__ import annotations

import bongard.panel_feature_observation as observation
import bongard.panel_feature_observer_protocol as observer_protocol
import bongard.panel_owner_inventory as owners
import bongard.panel_soft_ontology as ontology


def _d(char: str) -> str:
    return char * 64


def _owner_slot(
    *, kind: str, parent: str, x0: int, y0: int, x1: int, y1: int
) -> dict[str, object]:
    return {
        "slot_state": "owner",
        "owner_kind": kind,
        "parent_slot": parent,
        "x_min": x0,
        "y_min": y0,
        "x_max": x1,
        "y_max": y1,
    }


def _unused_slot() -> dict[str, object]:
    return {
        "slot_state": "unused",
        "owner_kind": "not_applicable",
        "parent_slot": "not_applicable",
        "x_min": -1,
        "y_min": -1,
        "x_max": -1,
        "y_max": -1,
    }


def test_receipted_pixels_to_neutral_owners_to_python_count_predicate() -> None:
    panel_png = b"\x89PNG\r\n\x1a\nsynthetic-fixture"
    slots = {name: _unused_slot() for name in owners.PANEL_OWNER_SLOT_NAMES}
    # Deliberately put the rightmost figure in the first temporary slot.  Slot
    # order is transport scaffolding, not semantic owner identity.
    slots["slot_00"] = _owner_slot(
        kind="figure", parent="root", x0=9, y0=2, x1=14, y1=9
    )
    slots["slot_01"] = _owner_slot(
        kind="figure", parent="root", x0=1, y0=3, x1=6, y1=10
    )
    payload = {"inventory_status": "complete", "slots": slots}
    receipt = owners.bind_panel_owner_inventory_receipt(
        panel_png=panel_png,
        observer_contract_digest=_d("a"),
        payload=payload,
        transport_kind=owners.InventoryTransportKind.INJECTED_RECEIPTED,
        model_id="fixture-model-v1",
        transport_receipt_digest=_d("b"),
    )
    artifact = owners.build_panel_owner_inventory_artifact(
        panel_png=panel_png,
        observer_contract_digest=_d("a"),
        payload=payload,
        receipt=receipt,
    )
    inventory = artifact.to_owner_inventory()
    assert [item.owner_id.value for item in inventory.owners] == [
        "owner_0001",
        "owner_0002",
    ]
    assert inventory.owners[0].region.minimum.x == 1

    two = ontology.PanelFeatureSpec(
        ontology.FeatureFamily.COMPONENT_COUNT,
        ontology.SubjectScope.WHOLE_PANEL,
        ontology.ReferenceFrame.NONE,
        ontology.ComponentCountParameters(ontology.ClosedCount.TWO),
    )
    three = ontology.PanelFeatureSpec(
        ontology.FeatureFamily.COMPONENT_COUNT,
        ontology.SubjectScope.WHOLE_PANEL,
        ontology.ReferenceFrame.NONE,
        ontology.ComponentCountParameters(ontology.ClosedCount.THREE),
    )
    count_observation = observation.derive_inventory_count_observation(
        inventory,
        observation.FeatureAxis.for_spec(two),
        observer_contract_digest=_d("c"),
        measurement_protocol_digest=_d("d"),
    )
    assert (
        count_observation.evaluate(two)
        is observation.EngineeringFeatureDisposition.MATCH
    )
    assert (
        count_observation.evaluate(three)
        is observation.EngineeringFeatureDisposition.NONMATCH
    )

    # A subsequent vision call would receive the complete 1..12 count axis,
    # never a designation that "two" is the candidate under test.
    view = observer_protocol.FeatureAxisObservationView.build(
        inventory, observation.FeatureAxis.for_spec(two)
    )
    assert len(view.variants) == 12
    assert "candidate" not in str(view.model_data()).lower()

