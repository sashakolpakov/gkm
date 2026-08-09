from __future__ import annotations

import inspect

from bongard.object_bongard_batch import ObjectBongardTaskPlan
from bongard.panel_hierarchical_exposed_support_smoke_command import (
    HIERARCHICAL_AUTHORIZATION_SCHEMA,
    HIERARCHICAL_PRECOMMIT_SCHEMA,
    _authorization,
    run_hierarchical_exposed_support_smoke,
)
from bongard.panel_soft_ontology import NativeOrientation


def _task() -> ObjectBongardTaskPlan:
    return ObjectBongardTaskPlan.create(
        "hd_convex-has_four_straight_lines_0001",
        seed_digest="sha256:" + "12" * 32,
    )


def test_hierarchical_smoke_precommit_is_query_free_and_one_positive() -> None:
    task = _task()
    panel_ids = task.side_0_support_panel_ids + task.side_1_support_panel_ids
    panels = tuple(b"synthetic-support-" + bytes([index]) for index in range(12))

    authorization, precommit = _authorization(
        task, panel_ids, panels, "ab" * 32
    )

    assert authorization["schema"] == HIERARCHICAL_AUTHORIZATION_SCHEMA
    assert precommit["schema"] == HIERARCHICAL_PRECOMMIT_SCHEMA
    assert (
        authorization["primary_orientation"]
        == NativeOrientation.SIDE0_POSITIVE.value
    )
    assert authorization["candidate_independent_observation"] is True
    assert authorization["composites_enumerated_before_contrast_consistency"] is True
    assert authorization["opposite_orientation_is_diagnostic_only"] is True
    assert authorization["query_release_or_observation_authorized"] is False
    assert precommit["physical_call_plan"] == {
        "proposer": 1,
        "support_hierarchical_observers": 12,
        "support_positive_formula_ranker_maximum": 1,
        "query": 0,
    }
    assert precommit["negative_formula_required"] is False
    assert precommit["negation_or_polarity_flip_allowed"] is False
    assert precommit["query_pixels_available_to_command"] is False


def test_hierarchical_smoke_has_no_query_input_surface() -> None:
    parameters = inspect.signature(
        run_hierarchical_exposed_support_smoke
    ).parameters
    assert all("query" not in name for name in parameters)
    assert "source_archive" in parameters
    assert "output_root" in parameters

