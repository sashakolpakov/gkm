"""Focused tests for the two-rank pure-Python rubric slate."""

from __future__ import annotations

from copy import deepcopy
import hashlib

import pytest

from bongard.canonical import canonical_digest
from bongard.evidence import Disposition
from bongard.object_bongard_rubric_observer import (
    ObjectBongardRubricSpec,
    RubricScope,
)
from bongard.object_bongard_rubric_slate import (
    ObjectBongardRubricSlateError,
    ObjectBongardRubricSlateSelection,
    cold_verify_object_bongard_rubric_slate,
    enumerate_object_bongard_rubric_slate,
    select_object_bongard_rubric_slate,
)
from bongard.object_bongard_soft_cues import ObjectBongardSoftCue
from bongard.object_bongard_rubric_version_space import (
    ObjectBongardRubricSupportVersionSpace,
    RUBRIC_SUPPORT_PANELS_PER_SIDE,
    RubricSupportSide,
    enumerate_object_bongard_rubric_candidates,
    object_bongard_rubric_version_space_algorithm_digest,
)
import bongard.object_bongard_rubric_version_space as version_module


def _raw(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _specs() -> tuple[ObjectBongardRubricSpec, ObjectBongardRubricSpec]:
    semantic_digest = _raw("semantic-artifact")
    sector = ObjectBongardSoftCue.create(
        "Unequal sector-like subshapes joined at a common apex."
    )
    leaf = ObjectBongardSoftCue.create(
        "Rounded contour tapering toward a pointed junction."
    )
    triangle = ObjectBongardSoftCue.create(
        "Three line-like spans forming a triangular arrangement."
    )
    return (
        ObjectBongardRubricSpec.from_soft_cues(
            semantic_digest, sector, leaf, 0
        ),
        ObjectBongardRubricSpec.from_soft_cues(
            semantic_digest, sector, triangle, 1
        ),
    )


def _space(
    spec: ObjectBongardRubricSpec,
    rows: tuple[tuple[Disposition, ...], tuple[Disposition, ...]],
    *,
    panel_prefix: str = "pilot",
) -> ObjectBongardRubricSupportVersionSpace:
    side_size = RUBRIC_SUPPORT_PANELS_PER_SIDE
    panel_ids = tuple(
        f"{panel_prefix}/positive/{index}" for index in range(side_size)
    ) + tuple(f"{panel_prefix}/negative/{index}" for index in range(side_size))
    candidates = enumerate_object_bongard_rubric_candidates(spec)
    sides = (RubricSupportSide.POSITIVE,) * side_size + (
        RubricSupportSide.NEGATIVE,
    ) * side_size
    survivors = tuple(
        candidate.candidate_digest
        for candidate, row in zip(candidates, rows, strict=True)
        if version_module._is_survivor(row, sides)
    )
    gap = (
        None
        if survivors
        else version_module._make_support_gap(candidates, panel_ids, sides, rows)
    )
    values = {
        "algorithm_digest": object_bongard_rubric_version_space_algorithm_digest(),
        "rubric_spec_digest": spec.spec_digest,
        "observer_catalog_digest": _raw("observer-catalog"),
        "observer_runtime_identity_digest": _raw("observer-runtime"),
        "candidates": candidates,
        "support_panel_ids": panel_ids,
        "support_artifact_digests": tuple(
            _raw(f"rank-{spec.candidate_rank}-artifact-{index}")
            for index in range(side_size * 2)
        ),
        "support_sides": sides,
        "rows": rows,
        "survivor_candidate_digests": survivors,
        "gap": gap,
    }
    provisional = object.__new__(ObjectBongardRubricSupportVersionSpace)
    for name, item in values.items():
        object.__setattr__(provisional, name, item)
    return ObjectBongardRubricSupportVersionSpace(
        **values,
        version_space_digest=canonical_digest(
            version_module._version_content(provisional)
        ),
    )


def _exact_row() -> tuple[Disposition, ...]:
    return (Disposition.PRESENT,) * 6 + (Disposition.CERTIFIED_ABSENT,) * 6


def test_four_candidate_order_and_first_exact_survivor_are_frozen() -> None:
    specs = _specs()
    spaces = (
        _space(specs[0], (_exact_row(), _exact_row())),
        _space(specs[1], (_exact_row(), _exact_row())),
    )
    selection = select_object_bongard_rubric_slate(specs, spaces)

    assert tuple(item.scope for item in selection.ordered_candidates) == (
        RubricScope.OBJECT,
        RubricScope.SCENE,
        RubricScope.OBJECT,
        RubricScope.SCENE,
    )
    assert selection.ordered_candidates == enumerate_object_bongard_rubric_slate(
        specs
    )
    assert selection.survivor_candidate_digests == tuple(
        item.candidate_digest for item in selection.ordered_candidates
    )
    assert selection.selected_candidate == selection.ordered_candidates[0]
    assert ObjectBongardRubricSlateSelection.from_data(selection.to_data()) == selection
    assert cold_verify_object_bongard_rubric_slate(
        selection, specs, spaces
    ) == selection


def test_indeterminate_and_error_rows_cannot_become_negative_absence() -> None:
    specs = _specs()
    unsafe = (
        (Disposition.PRESENT,) * 6
        + (Disposition.CERTIFIED_ABSENT,) * 4
        + (Disposition.INDETERMINATE, Disposition.ERROR)
    )
    spaces = (
        _space(specs[0], (_exact_row(), unsafe)),
        _space(specs[1], (unsafe, _exact_row())),
    )
    selection = select_object_bongard_rubric_slate(specs, spaces)
    assert selection.survivor_candidate_digests == (
        selection.ordered_candidates[0].candidate_digest,
        selection.ordered_candidates[3].candidate_digest,
    )
    data = selection.to_data()
    assert data["failed_indeterminate_or_error_is_absence"] is False
    assert data["negation_allowed"] is False
    assert data["polarity_flip_allowed"] is False
    assert data["lean_required"] is False
    assert data["query_or_broad_panels_included"] is False

    no_survivor_spaces = (
        _space(specs[0], (unsafe, unsafe)),
        _space(specs[1], (unsafe, unsafe)),
    )
    rejected = select_object_bongard_rubric_slate(specs, no_survivor_spaces)
    assert rejected.selected_candidate is None
    assert rejected.survivor_candidate_digests == ()
    assert rejected.to_data()["status"] == "no_exact_survivor"


def test_rank_support_and_selection_tampering_fail_closed() -> None:
    specs = _specs()
    spaces = (
        _space(specs[0], (_exact_row(), _exact_row())),
        _space(specs[1], (_exact_row(), _exact_row())),
    )
    with pytest.raises(ObjectBongardRubricSlateError, match="ranks zero and one"):
        select_object_bongard_rubric_slate(tuple(reversed(specs)), spaces)
    mismatched = _space(
        specs[1], (_exact_row(), _exact_row()), panel_prefix="different"
    )
    with pytest.raises(ObjectBongardRubricSlateError, match="shared canonical"):
        select_object_bongard_rubric_slate(specs, (spaces[0], mismatched))

    selection = select_object_bongard_rubric_slate(specs, spaces)
    changed = deepcopy(selection.to_data())
    changed["selected_candidate_digest"] = selection.ordered_candidates[1].candidate_digest
    with pytest.raises(ObjectBongardRubricSlateError, match="selection differs"):
        ObjectBongardRubricSlateSelection.from_data(changed)
