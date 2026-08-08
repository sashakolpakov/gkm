"""Focused tests for the observer-neutral prose rubric language."""

from __future__ import annotations

import hashlib

import pytest

from bongard.evidence import Disposition
from bongard.object_bongard_rubric_language import (
    ObjectBongardRubricLanguageError,
    ObjectBongardRubricSpec,
    OrdinalLevelInterval,
    RUBRIC_ABSENCE_UPPER_BOUND,
    RUBRIC_PRESENT_LOWER_BOUND,
    classify_object_bongard_rubric_interval,
    object_bongard_rubric_ordinal_scale_digest,
)
from bongard.object_bongard_soft_cues import ObjectBongardSoftCue


def _spec() -> ObjectBongardRubricSpec:
    return ObjectBongardRubricSpec.from_soft_cues(
        hashlib.sha256(b"semantic-artifact").hexdigest(),
        ObjectBongardSoftCue.create(
            "Two closed wedge-shaped lobes share one pointed junction."
        ),
        ObjectBongardSoftCue.create(
            "One closed triangular form carries three open spokes."
        ),
        0,
    )


def test_spec_identity_is_ordered_prose_but_observer_and_scale_neutral() -> None:
    spec = _spec()
    data = spec.to_data()

    assert ObjectBongardRubricSpec.from_data(data) == spec
    assert data["ordered_cue_roles"] == ["target", "foil"]
    assert data["observation_scope_bound_in_spec"] is False
    assert data["ordinal_scale_bound_in_spec"] is False
    assert "ordinal_scale_digest" not in data
    assert "observer_source_digest" not in data
    assert "atlas" not in spec.rubric.lower()
    assert spec.target_cue.text in spec.rubric
    assert spec.foil_cue.text in spec.rubric


@pytest.mark.parametrize(
    ("interval", "expected"),
    (
        (OrdinalLevelInterval(3, 3), Disposition.PRESENT),
        (OrdinalLevelInterval(3, 4), Disposition.PRESENT),
        (OrdinalLevelInterval(0, 0), Disposition.CERTIFIED_ABSENT),
        (OrdinalLevelInterval(0, 1), Disposition.CERTIFIED_ABSENT),
        (OrdinalLevelInterval(2, 2), Disposition.INDETERMINATE),
        (OrdinalLevelInterval(1, 3), Disposition.INDETERMINATE),
        (OrdinalLevelInterval(0, 4), Disposition.INDETERMINATE),
    ),
)
def test_panel_scale_has_one_fixed_python_projection(
    interval: OrdinalLevelInterval, expected: Disposition
) -> None:
    assert RUBRIC_PRESENT_LOWER_BOUND == 3
    assert RUBRIC_ABSENCE_UPPER_BOUND == 1
    assert classify_object_bongard_rubric_interval(interval) is expected
    assert len(object_bongard_rubric_ordinal_scale_digest()) == 64


def test_interval_rejects_boolean_out_of_range_and_reverse_bounds() -> None:
    for lower, upper in ((False, 1), (0, True), (-1, 1), (0, 5), (3, 2)):
        with pytest.raises(ObjectBongardRubricLanguageError, match="0..4"):
            OrdinalLevelInterval(lower, upper)  # type: ignore[arg-type]
