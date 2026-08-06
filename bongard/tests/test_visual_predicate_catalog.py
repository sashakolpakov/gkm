from __future__ import annotations

from io import BytesIO
import math

from PIL import Image, ImageDraw
import pytest

from bongard.evidence import Disposition
from bongard.typed_visual_proposal import (
    TypedDeterministicAtom,
    TypedVisualProposalError,
)
from bongard.visual_predicate_catalog import (
    DIRECT_VISUAL_ATOM_CATALOG,
    evaluate_direct_atom_by_scenario,
)
from bongard.contour_witnesses import CONTOUR_WITNESS_CAPABILITY_IDS
from bongard.visual_witness_bundle import extract_visual_witness_bundle
from bongard.visual_witnesses import (
    VISUAL_WITNESS_CAPABILITY_IDS,
    VISUAL_WITNESS_SCENARIO_IDS,
)


def _panel() -> bytes:
    image = Image.new("RGB", (64, 64), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((6, 8, 30, 44), fill="black")
    draw.rectangle((12, 14, 24, 38), fill="white")
    draw.rectangle((44, 20, 54, 34), fill="black")
    output = BytesIO()
    image.save(output, format="PNG", optimize=False)
    return output.getvalue()


def _atom(atom_id: str, catalog_key: str, count: int) -> TypedDeterministicAtom:
    return TypedDeterministicAtom(
        atom_id=atom_id,
        catalog_key=catalog_key,
        comparison="equal",
        arguments=(("target_count", count),),
    )


def _stroke_panel(kind: str) -> bytes:
    image = Image.new("RGB", (112, 112), "white")
    draw = ImageDraw.Draw(image)
    if kind == "crossing":
        draw.line([(22, 22), (90, 90)], fill="black", width=4)
        draw.line([(90, 22), (22, 90)], fill="black", width=4)
    elif kind == "loop":
        draw.ellipse((25, 25, 87, 87), outline="black", width=4)
    elif kind in {"s", "u"}:
        if kind == "s":
            points = [
                (
                    56 + 25 * math.sin(math.pi * (-1 + 2 * index / 120) / 2),
                    56 + 32 * (-1 + 2 * index / 120),
                )
                for index in range(121)
            ]
        else:
            points = [
                (
                    56 + 28 * math.cos(math.pi + index * math.pi / 120),
                    48 + 28 * math.sin(math.pi + index * math.pi / 120),
                )
                for index in range(121)
            ]
        draw.line(points, fill="black", width=4, joint="curve")
    else:
        raise AssertionError(kind)
    output = BytesIO()
    image.save(output, format="PNG", optimize=False)
    return output.getvalue()


def test_catalog_and_extractor_advertise_exactly_the_same_capabilities() -> None:
    catalog_keys = tuple(
        sorted(atom.catalog_key for atom in DIRECT_VISUAL_ATOM_CATALOG.atoms)
    )
    assert catalog_keys == tuple(
        sorted(VISUAL_WITNESS_CAPABILITY_IDS + CONTOUR_WITNESS_CAPABILITY_IDS)
    )


def test_exact_count_evidence_remains_separate_for_every_scenario() -> None:
    packet = extract_visual_witness_bundle(_panel())

    components = evaluate_direct_atom_by_scenario(
        packet, _atom("atom-00", "component.count", 2)
    )
    holes = evaluate_direct_atom_by_scenario(
        packet, _atom("atom-01", "hole.owner_count", 1)
    )

    assert tuple(components) == VISUAL_WITNESS_SCENARIO_IDS
    assert tuple(holes) == VISUAL_WITNESS_SCENARIO_IDS
    assert all(
        item.disposition is Disposition.PRESENT for item in components.values()
    )
    assert all(item.disposition is Disposition.PRESENT for item in holes.values())
    assert {
        dict(item.provenance.details)["observed_count_interval"]
        for item in components.values()
    } == {"2"}

    mismatches = evaluate_direct_atom_by_scenario(
        packet, _atom("atom-00", "component.count", 1)
    )
    assert all(
        item.disposition is Disposition.CERTIFIED_ABSENT
        for item in mismatches.values()
    )


def test_contour_angle_is_not_advertised_or_executable() -> None:
    assert all(
        "angle" not in atom.catalog_key
        for atom in DIRECT_VISUAL_ATOM_CATALOG.atoms
    )
    assert all(
        "angle" not in item
        for item in VISUAL_WITNESS_CAPABILITY_IDS
        + CONTOUR_WITNESS_CAPABILITY_IDS
    )
    with pytest.raises(TypedVisualProposalError, match="unknown registered atom"):
        DIRECT_VISUAL_ATOM_CATALOG.get("contour.angle")
    with pytest.raises(TypedVisualProposalError, match="unknown registered atom"):
        evaluate_direct_atom_by_scenario(
            extract_visual_witness_bundle(_panel()),
            _atom("atom-00", "contour.angle", 1),
        )


@pytest.mark.parametrize(
    ("kind", "capability", "target"),
    (
        ("crossing", "topology.endpoint_count", 4),
        ("crossing", "topology.branchpoint_count", 1),
        ("crossing", "topology.crossing_count", 1),
        ("loop", "topology.cycle_count", 1),
        ("s", "curvature.reversal_count", 1),
        ("s", "curvature.run_count", 2),
        ("s", "curvature.s_like_count", 1),
        ("u", "curvature.u_like_count", 1),
    ),
)
def test_registered_contour_counts_are_executable(
    kind: str, capability: str, target: int
) -> None:
    result = evaluate_direct_atom_by_scenario(
        extract_visual_witness_bundle(_stroke_panel(kind)),
        _atom("atom-00", capability, target),
    )

    assert tuple(result) == VISUAL_WITNESS_SCENARIO_IDS
    assert all(item.disposition is Disposition.PRESENT for item in result.values())


def test_interval_containing_target_is_indeterminate_not_absent() -> None:
    center = (56.0, 56.0)
    paths = []
    for degrees in (0, 72, 144, 216, 288):
        angle = math.radians(degrees)
        paths.append(
            [center, (56 + 38 * math.cos(angle), 56 + 38 * math.sin(angle))]
        )
    image = Image.new("RGB", (112, 112), "white")
    draw = ImageDraw.Draw(image)
    for path in paths:
        draw.line(path, fill="black", width=2)
    output = BytesIO()
    image.save(output, format="PNG", optimize=False)

    result = evaluate_direct_atom_by_scenario(
        extract_visual_witness_bundle(output.getvalue()),
        _atom("atom-00", "topology.crossing_count", 1),
    )

    assert all(
        item.disposition is Disposition.INDETERMINATE for item in result.values()
    )
    assert all(item.uncertainty is not None for item in result.values())
    assert all(item.uncertainty.lower == 0 for item in result.values())
    assert all(item.uncertainty.upper == 1 for item in result.values())


def test_malformed_or_tampered_packet_raises_instead_of_becoming_absence() -> None:
    packet = extract_visual_witness_bundle(_panel())
    object.__setattr__(packet.base_packet, "extractor_source_digest", "0" * 64)

    with pytest.raises(ValueError, match="artifact digest does not bind"):
        evaluate_direct_atom_by_scenario(
            packet, _atom("atom-00", "component.count", 2)
        )


@pytest.mark.parametrize("count", (0, 99))
def test_absence_disguised_as_zero_or_option_outside_grid_is_rejected(
    count: int,
) -> None:
    with pytest.raises(TypedVisualProposalError, match="outside the verifier grid"):
        evaluate_direct_atom_by_scenario(
            extract_visual_witness_bundle(_panel()),
            _atom("atom-00", "component.count", count),
        )


def test_every_contour_capability_has_only_positive_one_through_eight_options() -> None:
    for capability in CONTOUR_WITNESS_CAPABILITY_IDS:
        spec = DIRECT_VISUAL_ATOM_CATALOG.get(capability)
        assert tuple(
            dict(option.arguments)["target_count"] for option in spec.allowed_options
        ) == tuple(range(1, 9))
        with pytest.raises(
            TypedVisualProposalError, match="outside the verifier grid"
        ):
            evaluate_direct_atom_by_scenario(
                extract_visual_witness_bundle(_panel()),
                _atom("atom-00", capability, 0),
            )
