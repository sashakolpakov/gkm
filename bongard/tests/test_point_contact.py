from __future__ import annotations

from dataclasses import replace
from io import BytesIO

import numpy as np
from PIL import Image, ImageDraw
import pytest

from bongard.contour_witnesses import Q16Point
from bongard.evidence import Disposition
from bongard.loop_geometry import IntInterval
from bongard.loop_scene_witnesses import extract_loop_scene_witnesses
from bongard.point_contact import ContactKind, IncidentRayWitness, PairContactObservation
from bongard import point_contact as _contact


def _vertex_pair(*, stroke_width: int = 4, separation: int = 0) -> Image.Image:
    image = Image.new("RGB", (144, 128), "white")
    draw = ImageDraw.Draw(image)
    first = [(14, 24), (14, 104), (64, 64)]
    second = [
        (64 + separation, 64),
        (103 + separation, 24),
        (128 + separation, 64),
        (103 + separation, 104),
    ]
    draw.line(first + [first[0]], fill="black", width=stroke_width, joint="curve")
    draw.line(second + [second[0]], fill="black", width=stroke_width, joint="curve")
    return image


def _png(image: Image.Image) -> bytes:
    output = BytesIO()
    image.save(output, format="PNG", optimize=False)
    return output.getvalue()


def test_complete_signature_retains_four_owned_rays_and_two_gaps() -> None:
    packet = extract_loop_scene_witnesses(_png(_vertex_pair(stroke_width=2)))

    for scenario in packet.scenarios:
        assert len(scenario.contacts) == 1
        observation = scenario.contacts[0]
        assert observation.disposition is Disposition.PRESENT
        assert observation.owner_component_ids[0] == observation.owner_component_ids[1]
        signature = observation.signature
        assert signature is not None
        assert len(signature.rays) == 4
        assert {
            loop_id: sum(ray.owner_loop_id == loop_id for ray in signature.rays)
            for loop_id in signature.loop_ids
        } == {loop_id: 2 for loop_id in signature.loop_ids}
        assert len(signature.exterior_gaps) == 2
        assert PairContactObservation.from_data(observation.to_data()) == observation


def test_distinct_exact_foreground_owners_certify_separation() -> None:
    packet = extract_loop_scene_witnesses(
        _png(_vertex_pair(stroke_width=4, separation=4))
    )

    for scenario in packet.scenarios:
        observation = scenario.contacts[0]
        assert observation.disposition is Disposition.CERTIFIED_ABSENT
        assert observation.reason_code == "distinct_source_foreground_components"
        assert observation.owner_component_ids[0] != observation.owner_component_ids[1]
        assert observation.certificate is not None
        assert observation.signature is None


def test_thick_point_contact_becomes_indeterminate_never_false_absence() -> None:
    for width in (1, 2, 3, 4, 6, 8):
        image = _vertex_pair(stroke_width=width)
        dispositions_by_transform = []
        for transformed in (
            image,
            image.transpose(Image.Transpose.ROTATE_180),
            image.transpose(Image.Transpose.FLIP_LEFT_RIGHT),
        ):
            packet = extract_loop_scene_witnesses(_png(transformed))
            dispositions = tuple(
                scenario.contacts[0].disposition for scenario in packet.scenarios
            )
            assert Disposition.CERTIFIED_ABSENT not in dispositions
            dispositions_by_transform.append(dispositions)
        assert len(set(dispositions_by_transform)) == 1


def test_split_dilated_region_is_indeterminate_and_transform_stable() -> None:
    image = Image.new("RGB", (128, 128), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((8, 8, 119, 119), fill="black")
    draw.rectangle((18, 18, 50, 48), fill="white")
    draw.rectangle((18, 70, 50, 102), fill="white")
    draw.rectangle((34, 48, 34, 70), fill="white")
    draw.rectangle((52, 50, 79, 77), fill="white")

    for transformed in (
        image,
        image.transpose(Image.Transpose.ROTATE_180),
        image.transpose(Image.Transpose.FLIP_LEFT_RIGHT),
    ):
        packet = extract_loop_scene_witnesses(_png(transformed))
        raw = {
            scenario.scenario_id: scenario for scenario in packet.scenarios
        }["threshold032.raw"]
        assert len(raw.contacts) == 1
        assert raw.contacts[0].disposition is Disposition.INDETERMINATE
        assert raw.contacts[0].reason_code == "dilated_loop_mapping_unresolved"


def test_signature_and_observation_reject_forged_telemetry_or_owners() -> None:
    observation = extract_loop_scene_witnesses(
        _png(_vertex_pair(stroke_width=2))
    ).scenarios[0].contacts[0]
    signature = observation.signature
    assert signature is not None

    with pytest.raises(ValueError, match="telemetry differs"):
        replace(
            observation,
            normalized_gap_ppm_upper=observation.normalized_gap_ppm_upper + 1,  # type: ignore[operator]
        )
    original_gap = signature.exterior_gaps[0]
    forged_gap = replace(
        original_gap,
        owner_a=original_gap.owner_b,
        owner_b=original_gap.owner_a,
    )
    with pytest.raises(ValueError, match="owners differ"):
        replace(signature, exterior_gaps=(forged_gap, signature.exterior_gaps[1]))
    uncertain_rays = tuple(
        replace(ray, uncertainty_millidegrees=40_000)
        if ray.owner_loop_id == signature.loop_ids[0]
        else ray
        for ray in signature.rays
    )
    uncertain_by_id = {item.ray_id: item for item in uncertain_rays}
    uncertain_gaps = tuple(
        replace(
            gap,
            interval_millidegrees=IntInterval(
                gap.nominal_millidegrees
                - uncertain_by_id[gap.ray_a_id].uncertainty_millidegrees
                - uncertain_by_id[gap.ray_b_id].uncertainty_millidegrees,
                gap.nominal_millidegrees
                + uncertain_by_id[gap.ray_a_id].uncertainty_millidegrees
                + uncertain_by_id[gap.ray_b_id].uncertainty_millidegrees,
            ),
        )
        for gap in signature.exterior_gaps
    )
    with pytest.raises(ValueError, match="not uncertainty-certified"):
        replace(signature, rays=uncertain_rays, exterior_gaps=uncertain_gaps)
    with pytest.raises(ValueError, match="interleaving certificate fields"):
        PairContactObservation(
            loop_ids=signature.loop_ids,
            owner_component_ids=(
                "component-00000000",
                "component-00000000",
            ),
            disposition=Disposition.CERTIFIED_ABSENT,
            contact_kind=ContactKind.INTERLEAVING,
            normalized_gap_ppm_upper=999_999_999,
            interface_spread_ppm_upper=0,
            signature=None,
            reason_code="cyclic_owners_interleave",
            certificate=_contact._INTERLEAVING_CERTIFICATE,
        )


def test_uncertain_cyclic_ray_order_is_not_negative(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _contact._DilatedRegion(
        "loop-00000000", 100, np.asarray([[0.0, 0.0]]), "0" * 64
    )
    second = _contact._DilatedRegion(
        "loop-00000001", 100, np.asarray([[1.0, 0.0]]), "1" * 64
    )
    directions = {
        ("loop-00000000", -1): 0,
        ("loop-00000001", -1): 5_000,
        ("loop-00000000", 1): 10_000,
        ("loop-00000001", 1): 15_000,
    }

    def fake_fit(
        region: _contact._DilatedRegion,
        contact_index: int,
        *,
        step: int,
        width: int,
        height: int,
    ) -> IncidentRayWitness:
        del contact_index, width, height
        endpoint = "start" if step > 0 else "end"
        return IncidentRayWitness(
            ray_id=f"{region.loop_id}:{endpoint}:boundary-ray",
            owner_loop_id=region.loop_id,
            endpoint_name=endpoint,
            direction_millidegrees=directions[(region.loop_id, step)],
            uncertainty_millidegrees=5_000,
            residual_ppm_upper=87_000,
            endpoint_q16=Q16Point(1, 1),
            source_boundary_digest=region.boundary_digest,
        )

    monkeypatch.setattr(_contact, "_fit_ray", fake_fit)
    observation = _contact._pair_observation(
        first,
        second,
        owner_component_ids=("component-00000000", "component-00000000"),
        width=128,
        height=128,
    )

    assert observation.disposition is Disposition.INDETERMINATE
    assert observation.reason_code == "ray_cyclic_order_uncertain"
