from __future__ import annotations

from copy import deepcopy
import json

import pytest

import bongard.panel_owner_inventory as p


def _d(char: str) -> str:
    return char * 64


PNG = b"\x89PNG\r\n\x1a\nowner-inventory-fixture"
OBSERVER = _d("a")


def _sentinel(state: p.RawSlotState) -> dict[str, object]:
    return {
        "slot_state": state.value,
        "owner_kind": "not_applicable",
        "parent_slot": "not_applicable",
        "x_min": -1,
        "y_min": -1,
        "x_max": -1,
        "y_max": -1,
    }


def _owner(
    kind: str,
    bbox: tuple[int, int, int, int],
    parent: str = "root",
) -> dict[str, object]:
    return {
        "slot_state": "owner",
        "owner_kind": kind,
        "parent_slot": parent,
        "x_min": bbox[0],
        "y_min": bbox[1],
        "x_max": bbox[2],
        "y_max": bbox[3],
    }


def _payload(
    rows: dict[str, dict[str, object]] | None = None,
    *,
    status: p.InventoryStatus = p.InventoryStatus.COMPLETE,
) -> dict[str, object]:
    rows = {} if rows is None else rows
    if status is p.InventoryStatus.COMPLETE:
        default = p.RawSlotState.UNUSED
    elif status is p.InventoryStatus.UNRESOLVED:
        default = p.RawSlotState.UNRESOLVED
    elif status is p.InventoryStatus.CAPACITY_EXCEEDED:
        default = p.RawSlotState.CAPACITY_EXCEEDED
    else:
        default = p.RawSlotState.ERROR
    return {
        "inventory_status": status.value,
        "slots": {
            name: deepcopy(rows.get(name, _sentinel(default)))
            for name in p.PANEL_OWNER_SLOT_NAMES
        },
    }


def _receipt(payload: dict[str, object], *, panel_png: bytes = PNG):
    return p.bind_panel_owner_inventory_receipt(
        panel_png=panel_png,
        observer_contract_digest=OBSERVER,
        payload=payload,
        transport_kind=p.InventoryTransportKind.INJECTED_RECEIPTED,
        model_id="gpt-5.6-sol",
        transport_receipt_digest=_d("b"),
    )


def _artifact(payload: dict[str, object], *, panel_png: bytes = PNG):
    return p.build_panel_owner_inventory_artifact(
        panel_png=panel_png,
        observer_contract_digest=OBSERVER,
        payload=payload,
        receipt=_receipt(payload, panel_png=panel_png),
    )


def _base_rows() -> dict[str, dict[str, object]]:
    return {
        "slot_00": _owner("figure", (0, 0, 7, 7)),
        "slot_01": _owner("segment", (1, 1, 4, 1), "slot_00"),
        "slot_02": _owner("figure", (8, 0, 15, 7)),
    }


def test_model_view_is_fixed_capacity_and_role_blind() -> None:
    view = p.panel_owner_inventory_model_view()
    schema = view["output_schema"]
    slots = schema["properties"]["slots"]
    assert view["image_name"] == "panel.png"
    assert set(slots["properties"]) == set(p.PANEL_OWNER_SLOT_NAMES)
    assert slots["required"] == list(p.PANEL_OWNER_SLOT_NAMES)
    assert p.PANEL_OWNER_SLOT_CAPACITY == 12
    rendered = json.dumps(view, sort_keys=True).casefold()
    for forbidden in (
        "task",
        "group",
        "orientation",
        "query",
        "candidate",
        "support",
        "lean",
    ):
        assert forbidden not in rendered


def test_parser_rejects_bool_as_int_extra_slots_and_mixed_sentinels() -> None:
    payload = _payload(_base_rows())
    bad_bool = deepcopy(payload)
    bad_bool["slots"]["slot_00"]["x_min"] = True
    with pytest.raises(p.PanelOwnerInventoryError):
        p.parse_panel_owner_inventory_payload(bad_bool)
    extra = deepcopy(payload)
    extra["slots"]["slot_12"] = _sentinel(p.RawSlotState.UNUSED)
    with pytest.raises(p.PanelOwnerInventoryError):
        p.parse_panel_owner_inventory_payload(extra)
    mixed = _payload(status=p.InventoryStatus.UNRESOLVED)
    mixed["slots"]["slot_05"] = _sentinel(p.RawSlotState.UNUSED)
    with pytest.raises(p.PanelOwnerInventoryError):
        p.parse_panel_owner_inventory_payload(mixed)


def test_permutation_invariance_and_parent_remapping() -> None:
    first = _payload(_base_rows())
    permuted = _payload(
        {
            "slot_09": _owner("figure", (0, 0, 7, 7)),
            "slot_03": _owner("segment", (1, 1, 4, 1), "slot_09"),
            "slot_00": _owner("figure", (8, 0, 15, 7)),
        }
    )
    artifact_a = _artifact(first)
    artifact_b = _artifact(permuted)
    assert artifact_a.status is p.InventoryStatus.COMPLETE
    assert artifact_a.owners == artifact_b.owners
    assert artifact_a.semantic_inventory_digest == artifact_b.semantic_inventory_digest
    assert artifact_a.artifact_digest != artifact_b.artifact_digest
    assert [item.owner_id.value for item in artifact_a.owners] == [
        "owner_0001",
        "owner_0002",
        "owner_0003",
    ]
    assert artifact_a.owners[1].parent_owner_ids == (
        artifact_a.owners[0].owner_id,
    )
    assert artifact_a.owners[0].parent_owner_ids == ()
    assert artifact_a.owners[2].parent_owner_ids == ()


def test_semantic_collision_is_a_permutation_invariant_typed_gap() -> None:
    first = _payload(
        {
            "slot_00": _owner("figure", (0, 0, 7, 7)),
            "slot_11": _owner("figure", (0, 0, 7, 7)),
        }
    )
    second = _payload(
        {
            "slot_02": _owner("figure", (0, 0, 7, 7)),
            "slot_07": _owner("figure", (0, 0, 7, 7)),
        }
    )
    a = _artifact(first)
    b = _artifact(second)
    assert a.status is p.InventoryStatus.UNRESOLVED
    assert a.gap.kind is p.InventoryGapKind.SEMANTIC_COLLISION
    assert a.owners == ()
    assert a.gap == b.gap
    assert a.semantic_inventory_digest == b.semantic_inventory_digest


def test_cycle_and_invalid_parent_are_typed_gaps() -> None:
    cycle = _payload(
        {
            "slot_00": _owner("figure", (0, 0, 7, 7), "slot_01"),
            "slot_01": _owner("trace", (1, 1, 6, 6), "slot_00"),
        }
    )
    cycle_artifact = _artifact(cycle)
    assert cycle_artifact.status is p.InventoryStatus.UNRESOLVED
    assert cycle_artifact.gap.kind is p.InventoryGapKind.PARENT_CYCLE
    invalid = _payload(
        {"slot_00": _owner("segment", (1, 1, 4, 1), "slot_10")}
    )
    invalid_artifact = _artifact(invalid)
    assert invalid_artifact.status is p.InventoryStatus.UNRESOLVED
    assert invalid_artifact.gap.kind is p.InventoryGapKind.INVALID_PARENT


def test_capacity_unresolved_and_error_sentinels_are_closed_typed_gaps() -> None:
    capacity = _artifact(_payload(status=p.InventoryStatus.CAPACITY_EXCEEDED))
    assert capacity.status is p.InventoryStatus.CAPACITY_EXCEEDED
    assert capacity.gap.kind is p.InventoryGapKind.CAPACITY_EXCEEDED
    unresolved = _artifact(_payload(status=p.InventoryStatus.UNRESOLVED))
    assert unresolved.status is p.InventoryStatus.UNRESOLVED
    assert unresolved.gap.kind is p.InventoryGapKind.UNRESOLVED
    error = _artifact(_payload(status=p.InventoryStatus.ERROR))
    assert error.status is p.InventoryStatus.ERROR
    assert error.gap.kind is p.InventoryGapKind.TRANSPORT_ERROR


def test_artifact_roundtrip_tamper_custody_and_ontology_adapter() -> None:
    payload = _payload(_base_rows())
    artifact = _artifact(payload)
    assert p.PanelOwnerInventoryArtifact.from_data(artifact.to_data()) == artifact
    generic = artifact.to_owner_inventory()
    assert generic.panel_digest == artifact.panel_png_digest
    assert generic.owners == artifact.owners
    assert generic.enumeration_complete is True
    assert generic.enumeration_receipt_digest == artifact.receipt.receipt_digest
    assert generic.enumeration_receipt_digest != (
        artifact.receipt.transport_receipt_digest
    )

    for mutation in ("prompt_digest", "output_schema_digest"):
        data = deepcopy(artifact.to_data())
        data[mutation] = _d("0")
        with pytest.raises(p.PanelOwnerInventoryError):
            p.PanelOwnerInventoryArtifact.from_data(data)
    data = deepcopy(artifact.to_data())
    data["panel_png_byte_count"] = True
    with pytest.raises(p.PanelOwnerInventoryError):
        p.PanelOwnerInventoryArtifact.from_data(data)
    data = deepcopy(artifact.to_data())
    data["owners"][0]["owner_id"] = "owner_0099"
    with pytest.raises((p.PanelOwnerInventoryError, ValueError)):
        p.PanelOwnerInventoryArtifact.from_data(data)
    with pytest.raises(p.PanelOwnerInventoryError):
        p.build_panel_owner_inventory_artifact(
            panel_png=PNG + b"changed",
            observer_contract_digest=OBSERVER,
            payload=payload,
            receipt=_receipt(payload),
        )
    changed = deepcopy(payload)
    changed["slots"]["slot_00"]["x_max"] = 6
    with pytest.raises(p.PanelOwnerInventoryError):
        p.build_panel_owner_inventory_artifact(
            panel_png=PNG,
            observer_contract_digest=OBSERVER,
            payload=changed,
            receipt=_receipt(payload),
        )


def test_candidate_independence_and_no_lean_in_vnext_canonical_data() -> None:
    artifact = _artifact(_payload(_base_rows()))
    assert "feature_catalog_digest" not in artifact.to_data()
    assert artifact.to_data()["owner_semantics_digest"] == p.panel_owner_semantics_digest()
    rendered = json.dumps(artifact.to_data(), sort_keys=True).casefold()
    assert "lean" not in rendered
    for forbidden in ("task_id", "group", "orientation", "query", "candidate"):
        assert forbidden not in rendered
    assert set(p.panel_owner_inventory_request_data(
        transport_kind=p.InventoryTransportKind.INJECTED_RECEIPTED,
        model_id="gpt-5.6-sol",
        panel_png_digest=artifact.panel_png_digest,
        panel_png_byte_count=artifact.panel_png_byte_count,
        observer_contract_digest=OBSERVER,
    )) == {
        "schema",
        "transport_kind",
        "model_id",
        "image_name",
        "panel_png_digest",
        "panel_png_byte_count",
        "observer_contract_digest",
        "prompt_digest",
        "output_schema_digest",
        "inventory_contract_digest",
    }


def test_injectable_transport_sees_only_neutral_fixed_view() -> None:
    payload = _payload(_base_rows())
    calls: list[dict[str, object]] = []

    def transport(**kwargs):
        calls.append(kwargs)
        assert set(kwargs) == {"prompt", "panel_png", "image_name", "output_schema"}
        assert kwargs["image_name"] == "panel.png"
        assert kwargs["panel_png"] == PNG
        return p.PanelOwnerInventoryTransportResult(payload, _receipt(payload))

    artifact = p.observe_panel_owner_inventory(
        panel_png=PNG,
        observer_contract_digest=OBSERVER,
        transport=transport,
    )
    assert artifact.status is p.InventoryStatus.COMPLETE
    assert len(calls) == 1


def test_receipt_roundtrip_and_exact_request_binding() -> None:
    payload = _payload(_base_rows())
    receipt = _receipt(payload)
    assert p.PanelOwnerInventoryCallReceipt.from_data(receipt.to_data()) == receipt
    data = deepcopy(receipt.to_data())
    data["observer_contract_digest"] = _d("f")
    with pytest.raises(p.PanelOwnerInventoryError):
        p.PanelOwnerInventoryCallReceipt.from_data(data)
    data = deepcopy(receipt.to_data())
    data["panel_png_byte_count"] = False
    with pytest.raises(p.PanelOwnerInventoryError):
        p.PanelOwnerInventoryCallReceipt.from_data(data)
    with pytest.raises(p.PanelOwnerInventoryError):
        p.bind_panel_owner_inventory_receipt(
            panel_png=PNG,
            observer_contract_digest=OBSERVER,
            payload=payload,
            transport_kind=p.InventoryTransportKind.INJECTED_RECEIPTED,
            model_id="lean-backend",
            transport_receipt_digest=_d("b"),
        )
