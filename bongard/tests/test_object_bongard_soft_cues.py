from __future__ import annotations

from copy import deepcopy

import pytest

from bongard.object_bongard_soft_cues import (
    ObjectBongardSoftCue,
    ObjectBongardSoftCueError,
    ObjectBongardSoftCuePair,
)


def test_positive_atomic_cue_and_pair_are_content_addressed() -> None:
    sector = ObjectBongardSoftCue.create(
        "Unequal sector-like subshapes joined at a common apex."
    )
    triangle = ObjectBongardSoftCue.create(
        "Three line-like spans forming a triangular arrangement."
    )
    pair = ObjectBongardSoftCuePair.create(0, sector, triangle)
    assert ObjectBongardSoftCue.from_data(sector.to_data()) == sector
    assert ObjectBongardSoftCuePair.from_data(pair.to_data()) == pair
    assert pair.to_data()["python_is_canonical_authority"] is True
    assert pair.to_data()["lean_required"] is False
    assert pair.to_data()["lean_required_for_replay"] is False


@pytest.mark.parametrize(
    "text",
    (
        "No curved spans.",
        "A circle and a triangle.",
        "More oblique than the other group.",
        "Target score >= 3.",
        "Positive class shape.",
        "A form with 3 spans.",
    ),
)
def test_negation_boolean_comparison_roles_digits_and_operators_fail_closed(
    text: str,
) -> None:
    with pytest.raises(ObjectBongardSoftCueError, match="positive atomic"):
        ObjectBongardSoftCue.create(text)


def test_spelled_out_visible_counts_are_allowed() -> None:
    cue = ObjectBongardSoftCue.create(
        "Three line-like spans forming a triangular arrangement."
    )
    assert cue.text.startswith("Three")


def test_pair_may_reuse_one_group_cue_across_ranks() -> None:
    sector = ObjectBongardSoftCue.create(
        "Unequal sector-like subshapes joined at a common apex."
    )
    leaf = ObjectBongardSoftCue.create(
        "Rounded contour tapering toward a pointed junction."
    )
    triangle = ObjectBongardSoftCue.create(
        "Three line-like spans forming a triangular arrangement."
    )
    first = ObjectBongardSoftCuePair.create(0, sector, leaf)
    second = ObjectBongardSoftCuePair.create(1, sector, triangle)
    assert first.group_0_cue == second.group_0_cue
    assert first.pair_digest != second.pair_digest


def test_same_group_cues_and_digest_tamper_are_rejected() -> None:
    cue = ObjectBongardSoftCue.create("One enclosed triangular arrangement.")
    with pytest.raises(ObjectBongardSoftCueError, match="different positive cues"):
        ObjectBongardSoftCuePair.create(0, cue, cue)
    changed = deepcopy(cue.to_data())
    changed["text"] = "One enclosed rounded arrangement."
    with pytest.raises(ObjectBongardSoftCueError, match="digest differs"):
        ObjectBongardSoftCue.from_data(changed)
