from __future__ import annotations

import inspect
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from bongard.evidence import Disposition
from bongard.object_bongard_batch import ObjectBongardTaskPlan
from bongard.panel_positive_prose_exposed_probe_command import (
    PositiveProseExposedProbeError,
    _authorization,
    _component_conjunction_disposition,
    _cue,
    _deterministic_ink_zoom,
    _interval,
    _load_componentwise_semantic_cue,
    _load_frozen_semantic_cue,
    _load_ink_zoom_policy,
    _strict_component_observer_schema,
    run_positive_prose_exposed_probe,
)
from bongard.transport import validate_codex_strict_output_schema


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


def test_preregistered_known_cue_skips_proposer_and_cannot_authorize_target() -> None:
    cue_path = (
        Path(__file__).parents[1]
        / "data"
        / "panel_positive_prose_known_semantic_cue_20260809_v1.json"
    )
    cue, cue_file_sha256 = _load_frozen_semantic_cue(cue_path)
    task = ObjectBongardTaskPlan.create(
        "hd_convex-has_four_straight_lines_0001",
        seed_digest="sha256:" + "34" * 32,
    )
    ids = task.side_0_support_panel_ids + task.side_1_support_panel_ids
    panels = tuple(b"support" + bytes([index]) for index in range(12))
    authorization, precommit = _authorization(
        task,
        ids,
        panels,
        "ab" * 32,
        frozen_cue_record=cue,
        frozen_cue_file_sha256=cue_file_sha256,
    )

    assert authorization["cue_origin"] == "preregistered_known_semantic_reuse"
    assert authorization["headless_model_generated"] is False
    assert authorization["query_pixels_available"] is False
    assert authorization["frozen_cue_record_digest"] == cue["record_digest"]
    assert precommit["physical_call_plan"] == {
        "positive_proposer": 0,
        "support_observers": 12,
        "query": 0,
    }
    assert precommit["known_semantic_cue_cannot_authorize_target"] is True


def test_componentwise_cue_scores_atoms_before_python_conjunction() -> None:
    validate_codex_strict_output_schema(_strict_component_observer_schema())
    cue_path = (
        Path(__file__).parents[1]
        / "data"
        / "panel_positive_prose_componentwise_known_cue_20260809_v1.json"
    )
    cue, cue_file_sha256 = _load_componentwise_semantic_cue(cue_path)
    task = ObjectBongardTaskPlan.create(
        "hd_convex-has_four_straight_lines_0001",
        seed_digest="sha256:" + "34" * 32,
    )
    ids = task.side_0_support_panel_ids + task.side_1_support_panel_ids
    panels = tuple(b"support" + bytes([index]) for index in range(12))
    authorization, precommit = _authorization(
        task,
        ids,
        panels,
        "ab" * 32,
        componentwise_cue_record=cue,
        componentwise_cue_file_sha256=cue_file_sha256,
    )

    assert authorization["componentwise_python_conjunction"] is True
    assert precommit["physical_call_plan"] == {
        "positive_proposer": 0,
        "support_component_observers": 12,
        "query": 0,
    }
    assert (
        _component_conjunction_disposition(
            Disposition.PRESENT, Disposition.PRESENT
        )
        is Disposition.PRESENT
    )
    assert (
        _component_conjunction_disposition(
            Disposition.PRESENT, Disposition.CERTIFIED_ABSENT
        )
        is Disposition.CERTIFIED_ABSENT
    )
    assert (
        _component_conjunction_disposition(
            Disposition.PRESENT, Disposition.INDETERMINATE
        )
        is Disposition.INDETERMINATE
    )
    assert (
        _component_conjunction_disposition(
            Disposition.ERROR, Disposition.CERTIFIED_ABSENT
        )
        is Disposition.ERROR
    )


def test_candidate_independent_ink_zoom_is_exact_and_binds_both_images() -> None:
    policy_path = (
        Path(__file__).parents[1]
        / "data"
        / "panel_positive_prose_ink_zoom_20260809_v1.json"
    )
    policy, policy_file_sha256 = _load_ink_zoom_policy(policy_path)
    source = Image.new("RGB", (512, 512), "white")
    ImageDraw.Draw(source).rectangle((230, 240, 270, 260), fill="black")
    output = BytesIO()
    source.save(output, format="PNG")
    panel = output.getvalue()
    view_1, result_1 = _deterministic_ink_zoom(panel, policy)
    view_2, result_2 = _deterministic_ink_zoom(panel, policy)

    assert view_1 == view_2
    assert result_1 == result_2
    assert result_1["ink_bbox"] == [230, 240, 271, 261]
    assert result_1["source_png_sha256"] != result_1["observer_view_png_sha256"]
    with Image.open(BytesIO(view_1)) as decoded:
        assert decoded.size == (512, 512)
        assert decoded.mode == "RGB"

    cue_path = (
        Path(__file__).parents[1]
        / "data"
        / "panel_positive_prose_componentwise_known_cue_20260809_v1.json"
    )
    cue, cue_file_sha256 = _load_componentwise_semantic_cue(cue_path)
    task = ObjectBongardTaskPlan.create(
        "hd_convex-has_four_straight_lines_0001",
        seed_digest="sha256:" + "34" * 32,
    )
    ids = task.side_0_support_panel_ids + task.side_1_support_panel_ids
    panels = (panel,) * 12
    views = (view_1,) * 12
    records = (result_1,) * 12
    authorization, precommit = _authorization(
        task,
        ids,
        panels,
        "ab" * 32,
        componentwise_cue_record=cue,
        componentwise_cue_file_sha256=cue_file_sha256,
        observer_panels=views,
        ink_zoom_policy_record=policy,
        ink_zoom_policy_file_sha256=policy_file_sha256,
        ink_zoom_records=records,
    )
    assert authorization["candidate_independent_ink_zoom"] is True
    assert authorization["support_png_sha256"][0] == result_1["source_png_sha256"]
    assert (
        authorization["observer_view_png_sha256"][0]
        == result_1["observer_view_png_sha256"]
    )
    assert precommit["ink_zoom_policy_record_digest"] == policy["record_digest"]


def test_corrected_straight_action_cue_explicitly_allows_curved_sections() -> None:
    cue_path = (
        Path(__file__).parents[1]
        / "data"
        / "panel_positive_prose_componentwise_known_cue_20260809_v2.json"
    )
    cue, _digest = _load_componentwise_semantic_cue(cue_path)
    assert cue["exposed_positive_support_straight_action_counts"] == [4] * 6
    assert cue["exposed_positive_support_arc_action_counts"] == [0, 0, 0, 4, 0, 0]
    assert "additional curved carrier sections may be present" in cue["component_2"]
