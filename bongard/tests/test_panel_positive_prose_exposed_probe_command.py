from __future__ import annotations

import inspect

import pytest

from bongard.evidence import Disposition
from bongard.object_bongard_batch import ObjectBongardTaskPlan
from bongard.panel_positive_prose_exposed_probe_command import (
    PositiveProseExposedProbeError,
    _authorization,
    _cue,
    _interval,
    run_positive_prose_exposed_probe,
)


def test_positive_cue_and_fixed_interval_projection() -> None:
    estimates = {
        **{f"group_a_{index:02d}_estimate": "supports" for index in range(6)},
        **{
            f"group_b_{index:02d}_estimate": "does_not_support"
            for index in range(6)
        },
    }
    cue = _cue(
        {
            "cue_text": (
                "A convex closed carrier with exactly four structural straight sides"
            ),
            "component_1": "A convex closed structural carrier",
            "component_2": "Exactly four structural straight carrier sides",
            **estimates,
        }
    )
    assert cue["component_1"].startswith("A convex")
    assert _interval({"lower": 3, "upper": 4})[2] is Disposition.PRESENT
    assert _interval({"lower": 0, "upper": 1})[2] is Disposition.CERTIFIED_ABSENT
    assert _interval({"lower": 1, "upper": 3})[2] is Disposition.INDETERMINATE

    with pytest.raises(PositiveProseExposedProbeError):
        _cue(
            {
                "cue_text": "The positive group has four sides",
                "component_1": "A convex carrier",
                "component_2": "Four straight carrier sides",
                **estimates,
            }
        )
    with pytest.raises(PositiveProseExposedProbeError):
        _interval({"lower": 4, "upper": 3})


def test_probe_precommit_has_no_query_or_negative_predicate_surface() -> None:
    task = ObjectBongardTaskPlan.create(
        "hd_convex-has_four_straight_lines_0001",
        seed_digest="sha256:" + "34" * 32,
    )
    ids = task.side_0_support_panel_ids + task.side_1_support_panel_ids
    panels = tuple(b"support" + bytes([index]) for index in range(12))
    authorization, precommit = _authorization(task, ids, panels, "ab" * 32)

    assert authorization["one_positive_conjunction_only"] is True
    assert authorization["negative_description_or_formula_required"] is False
    assert authorization["query_pixels_available"] is False
    assert precommit["physical_call_plan"] == {
        "positive_proposer": 1,
        "support_observers": 12,
        "query": 0,
    }
    assert precommit["present_when_lower_at_least"] == 3
    assert precommit["certified_absent_when_upper_at_most"] == 1
    assert precommit["negation_or_polarity_flip_allowed"] is False
    assert all(
        "query" not in name
        for name in inspect.signature(run_positive_prose_exposed_probe).parameters
    )
