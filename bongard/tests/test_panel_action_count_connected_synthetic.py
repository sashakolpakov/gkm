from __future__ import annotations

from collections import Counter
from dataclasses import replace
import hashlib

import numpy as np
import pytest

from bongard import panel_action_count_connected_synthetic as subject


EXPECTED_SOURCE_SHA256 = (
    "011d4d763bcd98283d520d01bae5049a1df2339ed94014780daaa55af14fc30c"
)


@pytest.fixture(scope="module")
def corpus() -> tuple[subject.ConnectedSyntheticSample, ...]:
    return subject.build_connected_corpus()


def _component_count(pixels: set[int]) -> int:
    remaining = {(pixel // 64, pixel % 64) for pixel in pixels}
    count = 0
    while remaining:
        count += 1
        frontier = [remaining.pop()]
        while frontier:
            y, x = frontier.pop()
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    point = (y + dy, x + dx)
                    if point in remaining:
                        remaining.remove(point)
                        frontier.append(point)
    return count


def test_corpus_is_the_exact_connected_54_plus_52_grid(
    corpus: tuple[subject.ConnectedSyntheticSample, ...],
) -> None:
    assert len(corpus) == 1_060
    assert len({sample.sample_id for sample in corpus}) == 1_060
    assert len({sample.raster_digest for sample in corpus}) == 1_060
    assert Counter(sample.panel_program.carrier_family for sample in corpus) == Counter(
        {family: 212 for family in subject.connected_carrier_families()}
    )
    assert Counter(sample.layout_truth for sample in corpus) == Counter(
        {"single_shape": 540, "two_shape": 520}
    )
    for family in subject.connected_carrier_families():
        for nuisance in subject.connected_nuisances():
            rows = [
                sample for sample in corpus
                if sample.panel_program.carrier_family == family
                and sample.nuisance.identity == nuisance.identity
            ]
            assert Counter(sample.layout_truth for sample in rows) == Counter(
                {"single_shape": 54, "two_shape": 52}
            )
            assert {
                sample.declared_pair.as_tuple()
                for sample in rows if sample.layout_truth == "single_shape"
            } == {
                (straight, arc)
                for straight in range(10)
                for arc in range(10)
                if 1 <= straight + arc <= 9
            }


def test_raster_component_and_boundary_truth_are_derived_and_exact(
    corpus: tuple[subject.ConnectedSyntheticSample, ...],
) -> None:
    catalog = {row.primitive_id: row for row in subject.primitive_catalog()}
    observed_kinds: set[str] = set()
    for sample in corpus:
        ids = tuple(
            primitive_id
            for shape in sample.panel_program.shapes
            for primitive_id in shape.primitive_ids
        )
        union = set().union(*(set(catalog[item].ink_pixels) for item in ids))
        assert _component_count(union) == len(sample.panel_program.shapes)
        assert subject.render_catalog_program(ids) == sample.png_bytes
        assert sample.raster_digest == "sha256:" + hashlib.sha256(
            sample.png_bytes
        ).hexdigest()
        for shape in sample.panel_program.shapes:
            masks = [set(catalog[item].ink_pixels) for item in shape.primitive_ids]
            for index in range(len(masks) - 1):
                assert any(
                    max(
                        abs(left // 64 - right // 64),
                        abs(left % 64 - right % 64),
                    ) <= 1
                    for left in masks[index]
                    for right in masks[index + 1]
                )
            for left in range(len(masks)):
                for right in range(left + 2, len(masks)):
                    assert not (masks[left] & masks[right])
        observed_kinds.update(row.kind for row in sample.boundary_truth)
    assert observed_kinds == {"AA", "AL", "LA", "LL"}


def test_exact_cover_target_is_png_only_integral_and_matches_declared_visible_pair(
    corpus: tuple[subject.ConnectedSyntheticSample, ...],
) -> None:
    representatives = corpus[::53]
    assert len(representatives) >= 20
    for sample in representatives:
        target = subject.exact_cover_target(sample.png_bytes)
        assert target.png_digest == sample.raster_digest
        assert sample.declared_pair in target.count_pairs
        assert target.minimum_primitive_count == sum(
            sample.declared_pair.as_tuple()
        )
        assert all(
            hypothesis.covered_pixels
            and len(hypothesis.primitive_ids) == target.minimum_primitive_count
            for hypothesis in target.hypotheses
        )
        clone = bytes(bytearray(sample.png_bytes))
        assert subject.exact_cover_target(clone) == target


def test_fixed_catalog_retains_materially_distinct_minimum_ambiguity() -> None:
    first = ("stress.ambiguity.a.line", "stress.ambiguity.b.line")
    second = ("stress.ambiguity.c.line", "stress.ambiguity.d.line")
    first_png = subject.render_catalog_program(first)
    assert subject.render_catalog_program(second) == first_png

    target = subject.exact_cover_target(first_png)

    assert target.minimum_primitive_count == 2
    assert target.count_pairs == (subject.CountPair(2, 0),)
    inventories = {hypothesis.primitive_ids for hypothesis in target.hypotheses}
    assert first in inventories
    assert second in inventories
    assert len(inventories) >= 2
    assert all(
        subject.render_catalog_program(hypothesis.primitive_ids) == first_png
        for hypothesis in target.hypotheses
    )


def test_carrier_roles_are_exact_png_and_full_d4_orbit_disjoint(
    corpus: tuple[subject.ConnectedSyntheticSample, ...],
) -> None:
    train = {
        subject.d4_raster_orbit_digest(sample.png_bytes)
        for sample in corpus
        if sample.panel_program.carrier_family not in {"radial", "staggered"}
    }
    held = {
        subject.d4_raster_orbit_digest(sample.png_bytes)
        for sample in corpus
        if sample.panel_program.carrier_family in {"radial", "staggered"}
    }
    assert not (train & held)


def test_unissued_bytes_and_mutated_exact_records_fail_closed(
    corpus: tuple[subject.ConnectedSyntheticSample, ...],
) -> None:
    sample = corpus[0]
    raw = bytearray(sample.png_bytes)
    raw[-1] ^= 1
    with pytest.raises(subject.ConnectedSyntheticError, match="not issued"):
        subject.require_issued_connected_png(bytes(raw))
    with pytest.raises(subject.ConnectedSyntheticError, match="not issued"):
        subject.exact_cover_target(bytes(raw))

    class BytesSubclass(bytes):
        pass

    with pytest.raises(subject.ConnectedSyntheticError, match="exact bytes"):
        subject.require_issued_connected_png(BytesSubclass(sample.png_bytes))

    if sample.declared_pair != subject.CountPair(1, 0):
        with pytest.raises(ValueError):
            replace(sample, declared_pair=subject.CountPair(1, 0))

    other_nuisance = next(
        nuisance
        for nuisance in subject.connected_nuisances()
        if nuisance != sample.nuisance
    )
    with pytest.raises(ValueError):
        replace(sample, nuisance=other_nuisance)
    with pytest.raises(ValueError):
        replace(
            sample,
            panel_program=replace(sample.panel_program, carrier_family="radial"),
        )
    with pytest.raises(ValueError):
        replace(
            sample,
            panel_program=replace(
                sample.panel_program,
                shapes=(replace(sample.panel_program.shapes[0], shape_id="renamed"),),
            ),
        )

    pair = subject.CountPair(1, 0)
    object.__setattr__(pair, "straight", True)
    with pytest.raises((TypeError, ValueError)):
        subject.CountPair.__post_init__(pair)
    with pytest.raises(ValueError):
        subject.CountPair(-1, 2)
    with pytest.raises(ValueError):
        subject.ExactCoverHypothesis(
            subject.CountPair(0, 1),
            ("lattice.0.single_shape.s0.p0.line",),
            (0,),
        )
    with pytest.raises(ValueError):
        subject.render_catalog_program(
            tuple(row.primitive_id for row in subject.primitive_catalog()[:10])
        )


def test_inventories_are_fresh_and_source_sealed() -> None:
    assert subject.source_sha256() == EXPECTED_SOURCE_SHA256
    first = subject.primitive_catalog()
    second = subject.primitive_catalog()
    assert first == second and first is not second and first[0] is not second[0]
    assert len(first) == 384
    nuisances = subject.connected_nuisances()
    assert nuisances == (
        subject.ConnectedNuisance("identity", 2, 1000),
        subject.ConnectedNuisance("r90", 3, 1000),
    )
