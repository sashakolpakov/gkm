"""Focused policy tests for shared-witness support and slate selection."""

from __future__ import annotations

import ast
from pathlib import Path

from bongard.evidence import Disposition
from bongard.object_bongard_shared_witness import (
    ObjectBongardSharedWitnessContrast,
    ObjectBongardSharedWitnessRubricSpec,
)
from bongard.object_bongard_shared_witness_support import (
    ObjectBongardSharedWitnessCandidate,
    ObjectBongardSharedWitnessSupportGap,
    SharedWitnessSupportGapKind,
    SharedWitnessSupportSide,
    _bounded_admissible,
    _make_gap,
)


def _spec(rank: int = 0) -> ObjectBongardSharedWitnessRubricSpec:
    contrast = ObjectBongardSharedWitnessContrast.create(
        rank,
        shared_anchor="patterned contour network",
        visual_axis="contour termination",
        group_0_endpoint="closed circuit",
        group_1_endpoint="free ended",
    )
    return ObjectBongardSharedWitnessRubricSpec.from_contrast("e" * 64, contrast)


def test_fixed_symmetric_five_of_six_rule_has_no_error_or_contradiction_rescue() -> None:
    bounded_target = (Disposition.PRESENT,) * 5 + (Disposition.INDETERMINATE,)
    bounded_foil = (Disposition.CERTIFIED_ABSENT,) * 5 + (
        Disposition.INDETERMINATE,
    )
    assert _bounded_admissible(bounded_target, bounded_foil)

    assert not _bounded_admissible(
        (Disposition.PRESENT,) * 5 + (Disposition.CERTIFIED_ABSENT,),
        (Disposition.CERTIFIED_ABSENT,) * 6,
    )
    assert not _bounded_admissible(
        (Disposition.PRESENT,) * 5 + (Disposition.ERROR,),
        (Disposition.CERTIFIED_ABSENT,) * 6,
    )
    assert not _bounded_admissible(
        (Disposition.PRESENT,) * 4 + (Disposition.INDETERMINATE,) * 2,
        (Disposition.CERTIFIED_ABSENT,) * 6,
    )
    assert not _bounded_admissible(
        (Disposition.PRESENT,) * 6,
        (Disposition.CERTIFIED_ABSENT,) * 5 + (Disposition.PRESENT,),
    )


def test_gap_kind_is_typed_and_error_does_not_become_absence() -> None:
    candidate = ObjectBongardSharedWitnessCandidate.create(_spec())
    ids = tuple(f"panel/{index}" for index in range(12))
    sides = (SharedWitnessSupportSide.TARGET,) * 6 + (
        SharedWitnessSupportSide.FOIL,
    ) * 6

    language = _make_gap(
        candidate,
        ids,
        sides,
        (Disposition.PRESENT,) * 5
        + (Disposition.CERTIFIED_ABSENT,)
        + (Disposition.CERTIFIED_ABSENT,) * 6,
    )
    assert language.kind is SharedWitnessSupportGapKind.LANGUAGE_GAP

    witness = _make_gap(
        candidate,
        ids,
        sides,
        (Disposition.PRESENT,) * 4
        + (Disposition.INDETERMINATE,) * 2
        + (Disposition.CERTIFIED_ABSENT,) * 6,
    )
    assert witness.kind is SharedWitnessSupportGapKind.WITNESS_GAP

    error = _make_gap(
        candidate,
        ids,
        sides,
        (Disposition.PRESENT,) * 5
        + (Disposition.ERROR,)
        + (Disposition.CERTIFIED_ABSENT,) * 6,
    )
    assert error.kind is SharedWitnessSupportGapKind.ERROR_GAP
    assert error.error_panel_ids == ("panel/5",)
    assert ObjectBongardSharedWitnessSupportGap.from_data(error.to_data()) == error


def test_candidate_is_singleton_orientation_preserving_and_spec_bound() -> None:
    first = ObjectBongardSharedWitnessCandidate.create(_spec(0))
    second = ObjectBongardSharedWitnessCandidate.create(_spec(1))
    assert first.candidate_rank == 0
    assert second.candidate_rank == 1
    assert first.candidate_digest != second.candidate_digest
    assert first.to_data()["candidate_id"] == "shared-witness:group-0-target"
    assert first.to_data()["polarity_flip_allowed"] is False
    assert first.to_data()["threshold_tuning_allowed"] is False


def test_support_and_slate_have_no_lean_ranker_or_atlas_import() -> None:
    package = Path(__file__).parents[1]
    for name in (
        "object_bongard_shared_witness_support.py",
        "object_bongard_shared_witness_slate.py",
    ):
        tree = ast.parse((package / name).read_text(encoding="utf-8"))
        imports: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                imports.append(node.module)
        lowered = tuple(item.lower() for item in imports)
        assert not any(
            "lean" in item or "ranker" in item or "atlas" in item
            for item in lowered
        )
