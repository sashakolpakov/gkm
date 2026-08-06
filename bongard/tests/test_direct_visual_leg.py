from __future__ import annotations

from io import BytesIO
import math

from PIL import Image, ImageDraw
import pytest

from bongard.direct_visual_leg import (
    DirectVisualLegError,
    DirectVisualLowering,
    lower_direct_visual_proposal,
    register_direct_visual_predicate,
)
from bongard.evidence import Disposition, Evidence, Provenance
from bongard.ir import evaluate_formula, validate_formula
from bongard.legs import LegRegistry, TypedValue
from bongard.typed_visual_proposal import (
    PANEL_DESCRIPTION_KEYS,
    parse_typed_visual_proposal,
)
from bongard.visual_predicate_catalog import (
    DIRECT_VISUAL_ATOM_CATALOG,
    direct_visual_catalog_digest,
)
from bongard.visual_witness_bundle import (
    VISUAL_WITNESS_BUNDLE,
    extract_visual_witness_bundle,
)
from bongard.visual_witnesses import VISUAL_WITNESS_SCENARIO_IDS


def _panel(*, second_component: bool = True) -> bytes:
    image = Image.new("RGB", (64, 64), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((6, 8, 30, 44), fill="black")
    draw.rectangle((12, 14, 24, 38), fill="white")
    if second_component:
        draw.rectangle((44, 20, 54, 34), fill="black")
    output = BytesIO()
    image.save(output, format="PNG", optimize=False)
    return output.getvalue()


def _junction_panel(*, five_arms: bool = False) -> bytes:
    image = Image.new("RGB", (112, 112), "white")
    draw = ImageDraw.Draw(image)
    if five_arms:
        for degrees in (0, 72, 144, 216, 288):
            angle = math.radians(degrees)
            draw.line(
                [
                    (56, 56),
                    (56 + 38 * math.cos(angle), 56 + 38 * math.sin(angle)),
                ],
                fill="black",
                width=2,
            )
    else:
        draw.line([(22, 22), (90, 90)], fill="black", width=4)
        draw.line([(90, 22), (22, 90)], fill="black", width=4)
    output = BytesIO()
    image.save(output, format="PNG", optimize=False)
    return output.getvalue()


def _proposal(*selections: tuple[str, int]):
    return parse_typed_visual_proposal(
        {
            "positive_description": "a registered arrangement of ink components",
            "panel_descriptions": {
                name: f"literal drawing {index}"
                for index, name in enumerate(PANEL_DESCRIPTION_KEYS)
            },
            "view": "literal_ink",
            "deterministic_atoms": [
                {
                    "catalog_key": catalog_key,
                    "comparison": "equal",
                    "arguments": {"target_count": count},
                }
                for catalog_key, count in selections
            ],
            "soft_claim": None,
            "formula": {
                "kind": "all",
                "atom_indices": list(range(len(selections))),
            },
        },
        catalog=DIRECT_VISUAL_ATOM_CATALOG,
    )


def _registered(proposal):
    registry = LegRegistry()
    handle = register_direct_visual_predicate(
        registry,
        name="direct_visual_claim",
        version="1",
        proposal=proposal,
        expected_catalog_digest=direct_visual_catalog_digest(),
    )
    registry.freeze()
    return registry, handle


def test_lowering_round_trip_preserves_original_atom_ids_and_options() -> None:
    proposal = _proposal(("component.count", 2), ("hole.owner_count", 1))
    lowering = lower_direct_visual_proposal(proposal)

    assert lowering.atom_ids == ("atom-00", "atom-01")
    assert [atom.to_data() for atom in lowering.atoms] == [
        {
            "atom_id": "atom-00",
            "catalog_key": "component.count",
            "comparison": "equal",
            "arguments": {"target_count": 2},
        },
        {
            "atom_id": "atom-01",
            "catalog_key": "hole.owner_count",
            "comparison": "equal",
            "arguments": {"target_count": 1},
        },
    ]
    assert lowering.to_data()["formula"] == {
        "kind": "all",
        "atom_ids": ["atom-00", "atom-01"],
    }
    assert DirectVisualLowering.from_data(lowering.to_data()) == lowering


def test_registered_leg_runs_complete_direct_conjunction_in_closed_ir() -> None:
    proposal = _proposal(("component.count", 2), ("hole.owner_count", 1))
    registry, handle = _registered(proposal)
    formula = handle.atom()
    validate_formula(
        formula, registry, {"visual_witness_bundle": VISUAL_WITNESS_BUNDLE}
    )

    present = evaluate_formula(
        formula,
        registry,
        {
            "visual_witness_bundle": TypedValue(
                VISUAL_WITNESS_BUNDLE, extract_visual_witness_bundle(_panel())
            )
        },
    )
    absent = evaluate_formula(
        formula,
        registry,
        {
            "visual_witness_bundle": TypedValue(
                VISUAL_WITNESS_BUNDLE,
                extract_visual_witness_bundle(_panel(second_component=False)),
            )
        },
    )

    assert present.disposition is Disposition.PRESENT
    assert absent.disposition is Disposition.CERTIFIED_ABSENT
    assert (
        registry.snapshot().contracts[0].operational_digest
        == handle.operational_digest
    )


def test_complete_conjunction_precedes_scenario_consensus(monkeypatch) -> None:
    proposal = _proposal(("component.count", 1), ("component.count", 2))
    registry, handle = _registered(proposal)

    def complementary_scenario_evidence(packet, atom):
        del packet
        evidence = {}
        for index, scenario_id in enumerate(VISUAL_WITNESS_SCENARIO_IDS):
            provenance = Provenance(
                "fixture", "1", atom.atom_id, details=(("scenario_id", scenario_id),)
            )
            first_matches = index == 0
            matches = first_matches if atom.atom_id == "atom-00" else not first_matches
            evidence[scenario_id] = (
                Evidence.present(True, provenance)
                if matches
                else Evidence.certified_absent(provenance, "complementary near miss")
            )
        return evidence

    monkeypatch.setattr(
        "bongard.direct_visual_leg.evaluate_direct_atom_by_scenario",
        complementary_scenario_evidence,
    )
    result = evaluate_formula(
        handle.atom(),
        registry,
        {
            "visual_witness_bundle": TypedValue(
                VISUAL_WITNESS_BUNDLE, extract_visual_witness_bundle(_panel())
            )
        },
    )

    # Neither atom is absent in every scenario, but every complete scenario
    # conjunction has a constructive nonmatch.
    assert result.disposition is Disposition.CERTIFIED_ABSENT


def test_registered_leg_executes_contour_atoms_inside_joint_scenarios() -> None:
    registry, handle = _registered(
        _proposal(
            ("topology.endpoint_count", 4),
            ("topology.branchpoint_count", 1),
            ("topology.crossing_count", 1),
        )
    )
    result = evaluate_formula(
        handle.atom(),
        registry,
        {
            "visual_witness_bundle": TypedValue(
                VISUAL_WITNESS_BUNDLE,
                extract_visual_witness_bundle(_junction_panel()),
            )
        },
    )

    assert result.disposition is Disposition.PRESENT


def test_interval_valued_contour_atom_remains_indeterminate_through_leg() -> None:
    registry, handle = _registered(_proposal(("topology.crossing_count", 1)))
    result = evaluate_formula(
        handle.atom(),
        registry,
        {
            "visual_witness_bundle": TypedValue(
                VISUAL_WITNESS_BUNDLE,
                extract_visual_witness_bundle(_junction_panel(five_arms=True)),
            )
        },
    )

    assert result.disposition is Disposition.INDETERMINATE


def test_operational_digest_binds_selected_options() -> None:
    _, first = _registered(_proposal(("component.count", 1)))
    _, second = _registered(_proposal(("component.count", 2)))

    assert first.operational_digest != second.operational_digest
    assert first.reference.contract_digest != second.reference.contract_digest


def test_malformed_packet_is_error_not_certified_absence() -> None:
    registry, handle = _registered(_proposal(("component.count", 2)))
    packet = extract_visual_witness_bundle(_panel())
    object.__setattr__(packet.base_packet, "extractor_source_digest", "0" * 64)

    result = evaluate_formula(
        handle.atom(),
        registry,
        {"visual_witness_bundle": TypedValue(VISUAL_WITNESS_BUNDLE, packet)},
    )

    assert result.disposition is Disposition.ERROR
    assert result.error_type == "ValueError"


def test_lowering_rejects_option_tamper_and_missing_direct_atoms() -> None:
    lowering = lower_direct_visual_proposal(_proposal(("component.count", 2)))
    tampered = lowering.to_data()
    tampered["selected_atoms"][0]["arguments"]["target_count"] = 99
    with pytest.raises(DirectVisualLegError, match="outside the verifier grid"):
        DirectVisualLowering.from_data(tampered)

    with pytest.raises(DirectVisualLegError, match="1..3"):
        DirectVisualLowering(
            source_proposal_digest=lowering.source_proposal_digest,
            positive_description=lowering.positive_description,
            catalog_digest=lowering.catalog_digest,
            atoms=(),
        )


def test_registration_rejects_wrong_catalog_commitment() -> None:
    with pytest.raises(DirectVisualLegError, match="catalog differs"):
        register_direct_visual_predicate(
            LegRegistry(),
            name="direct_visual_claim",
            version="1",
            proposal=_proposal(("component.count", 2)),
            expected_catalog_digest="0" * 64,
        )
