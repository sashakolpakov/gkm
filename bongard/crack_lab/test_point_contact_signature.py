"""Focused contracts for complete two-loop point-contact geometry."""
from __future__ import annotations

import dataclasses
import math
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import semantic_legs as L
from dataset import sample_problem_programs
from visual_witnesses import (
    ContactWitness,
    ContourWitness,
    ExteriorGapWitness,
    IncidentRayWitness,
    IntersectionWitness,
    PartGraphWitness,
    PartWitness,
    PointContactSignature,
    PointWitness,
)


SMOKE = Path(__file__).with_name("semantic_soft_runs") \
    / "smoke_20260805" / "workspace" / "problem_00"
DATASET = Path(__file__).parents[2] / "downloads" / "Bongard-LOGO"


def _polar(angle_degrees: float, radius: float) -> tuple[float, float]:
    angle = math.radians(angle_degrees)
    return radius * math.cos(angle), radius * math.sin(angle)


def _loop_part(
        part_id: str, first_angle: float, second_angle: float,
        *, zigzag: bool = False,
        ) -> PartWitness:
    first = [_polar(first_angle, radius)
             for radius in np.linspace(1.0, 30.0, 30)]
    if zigzag:
        first = [
            (x, y + (5.0 if index % 2 else -5.0))
            for index, (x, y) in enumerate(first)
        ]
    outer = [
        _polar(angle, 30.0)
        for angle in np.linspace(first_angle, second_angle, 61)[1:-1]
    ]
    second = [_polar(second_angle, radius)
              for radius in np.linspace(30.0, 1.0, 30)]
    contour = ContourWitness(
        source_component_id=part_id,
        points=tuple(first + outer + second),
        is_closed=False,
        confidence=0.98,
        provenance=("synthetic-two-ray-loop",),
    )
    return PartWitness(
        part_id=part_id,
        role="stroke-loop",
        source_component_id=part_id,
        contour=contour,
        confidence=0.98,
        provenance=("synthetic-loop-part",),
    )


def _graph(
        angles: tuple[tuple[float, float], tuple[float, float]],
        *, first_zigzag: bool = False,
        ) -> PartGraphWitness:
    parts = (
        _loop_part("part-a", *angles[0], zigzag=first_zigzag),
        _loop_part("part-b", *angles[1]),
    )
    vertex = PointWitness(x=0.0, y=0.0, source_id="junction")
    contact = IntersectionWitness(
        source_a="part-a",
        source_b="part-b",
        points=(vertex,),
        relation="intersection",
        confidence=0.97,
        provenance=("synthetic-contact",),
    )
    return PartGraphWitness(
        parts=parts,
        contacts=(contact,),
        adjacency=(("part-a", "part-b"),),
        confidence=0.96,
        provenance=("synthetic-graph",),
    )


def test_signature_retains_four_owned_rays_and_both_exterior_gaps() -> None:
    signature = L._extract_graph_point_contact_signature(
        _graph(((0.0, 100.0), (130.0, 220.0))))

    assert isinstance(signature, PointContactSignature)
    assert signature.contact_count == 1
    assert signature.part_ids == ("part-a", "part-b")
    assert signature.loop_incidence == (
        ("part-a", True, True), ("part-b", True, True))
    assert len(signature.rays) == 4
    assert all(isinstance(ray, IncidentRayWitness) for ray in signature.rays)
    assert Counter(ray.owner_id for ray in signature.rays) \
        == {"part-a": 2, "part-b": 2}
    assert tuple(ray.direction_degrees for ray in signature.rays) \
        == tuple(sorted(ray.direction_degrees for ray in signature.rays))
    assert sum(
        signature.cyclic_owners[index]
        != signature.cyclic_owners[(index + 1) % 4]
        for index in range(4)
    ) == 2
    assert all(isinstance(gap, ExteriorGapWitness)
               for gap in signature.exterior_gaps)
    assert [gap.degrees for gap in signature.exterior_gaps] \
        == pytest.approx([30.0, 140.0], abs=1e-9)
    assert all(gap.owner_a != gap.owner_b
               for gap in signature.exterior_gaps)
    assert all(gap.uncertainty_degrees >= 0.0
               for gap in signature.exterior_gaps)
    assert "assemble_point_contact_signature" in signature.provenance


def test_gap_measurements_do_not_collapse_the_contact_to_a_minimum() -> None:
    acute = L._extract_graph_point_contact_signature(
        _graph(((0.0, 100.0), (130.0, 220.0))))
    balanced = L._extract_graph_point_contact_signature(
        _graph(((60.0, 150.0), (0.0, 260.0))))

    assert L.point_contact_small_exterior_gap_degrees(acute) \
        == pytest.approx(30.0, abs=1e-9)
    assert L.point_contact_large_exterior_gap_degrees(acute) \
        == pytest.approx(140.0, abs=1e-9)
    assert L.point_contact_exterior_gap_ratio(acute) \
        == pytest.approx(140.0 / 30.0, abs=1e-9)
    assert L.point_contact_small_exterior_gap_degrees(balanced) \
        == pytest.approx(60.0, abs=1e-9)
    assert L.point_contact_large_exterior_gap_degrees(balanced) \
        == pytest.approx(110.0, abs=1e-9)
    assert L.point_contact_exterior_gap_ratio(acute) \
        > 2.5 * L.point_contact_exterior_gap_ratio(balanced)
    assert L.point_contact_gap_ratio_lower_bound(acute) \
        <= L.point_contact_exterior_gap_ratio(acute)


def test_interleaving_ownership_is_semantic_non_membership() -> None:
    crossing = _graph(((0.0, 180.0), (60.0, 240.0)))
    with pytest.raises(L.WitnessAbsent) as raised:
        L._extract_graph_point_contact_signature(crossing)
    assert raised.value.failure_mode == "no_point_contact_signature"


def test_fit_failure_is_indeterminate_not_negative_evidence() -> None:
    missing = _graph(((0.0, 100.0), (130.0, 220.0)))
    missing = dataclasses.replace(
        missing,
        parts=(dataclasses.replace(missing.parts[0], contour=None),
               missing.parts[1]),
    )
    with pytest.raises(L.WitnessIndeterminate) as absent_support:
        L._extract_graph_point_contact_signature(missing)
    assert absent_support.value.failure_mode \
        == "point_contact_fit_indeterminate"

    with pytest.raises(L.WitnessIndeterminate) as poor_fit:
            L._extract_graph_point_contact_signature(
                _graph(((0.0, 100.0), (130.0, 220.0)), first_zigzag=True))
    assert poor_fit.value.failure_mode == "point_contact_fit_indeterminate"


def test_signature_rejects_forged_gap_measurement() -> None:
    signature = L._extract_graph_point_contact_signature(
        _graph(((0.0, 100.0), (130.0, 220.0))))
    forged = dataclasses.replace(
        signature.exterior_gaps[0],
        degrees=signature.exterior_gaps[0].degrees + 5.0,
    )
    with pytest.raises(ValueError, match="degrees do not match"):
        dataclasses.replace(
            signature,
            exterior_gaps=(forged, signature.exterior_gaps[1]),
        )


def test_registry_exposes_one_canonical_panel_extractor_and_honest_bounds() -> None:
    registry = L.default_registry()
    extractor = registry.get("extract_point_contact_signature")
    assert extractor.domain == ("Panel",)
    assert extractor.codomain == "PointContactSignature"
    assert extractor.failure_modes == ("no_point_contact_signature",)
    assert extractor.indeterminate_modes \
        == ("point_contact_fit_indeterminate",)
    assert "minimum angle" not in extractor.proxy_for

    ratio = registry.get("point_contact_gap_ratio_lower_bound")
    assert ratio.domain == ("PointContactSignature",)
    assert ratio.codomain == "Measurement"
    assert ratio.invariances == frozenset({
        "translation", "uniform_scale", "rotation", "reflection"})
    assert dict(ratio.proxy_directions) \
        == {term: "high" for term in ratio.proxy_for}


@pytest.mark.skipif(not SMOKE.is_dir(), reason="immutable smoke fixture absent")
def test_panel_extractor_separates_support_by_complete_contact_signature() -> None:
    positives = []
    for index in range(6):
        panel = np.load(SMOKE / f"pos_{index}.npy", allow_pickle=False)
        signature = L.extract_point_contact_signature(panel)
        positives.append((
            L.point_contact_small_exterior_gap_degrees(signature),
            L.point_contact_large_exterior_gap_degrees(signature),
            L.point_contact_gap_ratio_lower_bound(signature),
        ))
    hard_negative = L.extract_point_contact_signature(
        np.load(SMOKE / "neg_4.npy", allow_pickle=False))

    assert [item[0] for item in positives] == pytest.approx(
        [29.62, 29.41, 29.62, 26.65, 33.46, 28.62], abs=0.08)
    assert [item[1] for item in positives] == pytest.approx(
        [137.96, 132.73, 139.43, 138.66, 133.50, 135.15], abs=0.08)
    assert min(item[2] for item in positives) > 3.7
    assert L.point_contact_small_exterior_gap_degrees(hard_negative) \
        == pytest.approx(59.02, abs=0.08)
    assert L.point_contact_large_exterior_gap_degrees(hard_negative) \
        == pytest.approx(110.51, abs=0.08)
    assert L.point_contact_gap_ratio_lower_bound(hard_negative) < 1.9

    for index in (0, 1, 2, 3, 5):
        panel = np.load(SMOKE / f"neg_{index}.npy", allow_pickle=False)
        with pytest.raises(L.WitnessAbsent) as absent:
            L.extract_point_contact_signature(panel)
        assert absent.value.failure_mode == "no_point_contact_signature"


@pytest.mark.skipif(
    not (DATASET / "data" / "human_designed_shapes.tsv").is_file(),
    reason="Bongard-LOGO latent programs unavailable",
)
def test_hidden_rerenders_have_no_positive_indeterminacy() -> None:
    latent = sample_problem_programs(
        str(DATASET), limit=1, seed=20260805, source="basic")[0]
    assert latent.concept == "mismatch_sector_rec2"

    for seed in (20260806, 20260807, 20260905):
        problem = latent.render(seed)
        positive_ratios = [
            L.point_contact_gap_ratio_lower_bound(
                L.extract_point_contact_signature(panel))
            for panel in problem.pos
        ]
        assert min(positive_ratios) > 3.3

        hard_negative = L.extract_point_contact_signature(problem.neg[4])
        assert L.point_contact_gap_ratio_lower_bound(hard_negative) < 1.9
        for index in (0, 1, 2, 3, 5):
            with pytest.raises(L.WitnessAbsent) as absent:
                L.extract_point_contact_signature(problem.neg[index])
            assert absent.value.failure_mode == "no_point_contact_signature"


@pytest.mark.skipif(not SMOKE.is_dir(), reason="immutable smoke fixture absent")
@pytest.mark.parametrize("transform", (np.rot90, np.fliplr))
def test_panel_measurements_are_rotation_and_reflection_invariant(transform) -> None:
    panel = np.load(SMOKE / "pos_0.npy", allow_pickle=False)
    before = L.extract_point_contact_signature(panel)
    after = L.extract_point_contact_signature(transform(panel))
    assert L.point_contact_small_exterior_gap_degrees(after) \
        == pytest.approx(L.point_contact_small_exterior_gap_degrees(before))
    assert L.point_contact_large_exterior_gap_degrees(after) \
        == pytest.approx(L.point_contact_large_exterior_gap_degrees(before))
    assert L.point_contact_gap_ratio_lower_bound(after) \
        == pytest.approx(L.point_contact_gap_ratio_lower_bound(before))
