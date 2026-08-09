"""Offline boundary tests for the exposed-support smoke command."""

from __future__ import annotations

import json

import pytest

from bongard.panel_feature_exposed_support_smoke_command import (
    DEFAULT_SOURCE_ARCHIVE,
    PanelFeatureExposedSupportSmokeError,
    _authorization,
    _metadata_only,
    _read_source,
)


def test_real_source_is_exactly_twelve_supports_and_zero_queries() -> None:
    result = _metadata_only(DEFAULT_SOURCE_ARCHIVE)
    assert result["task_id"] == "hd_convex-has_four_straight_lines_0001"
    assert result["support_panel_count"] == 12
    assert result["query_pixel_count"] == 0
    assert result["observer_axis_count"] == 9
    assert "straight_segment_count" in result["observer_axis_families"]
    assert "convexity" in result["observer_axis_families"]


def test_source_with_any_query_or_freeze_material_fails_closed(tmp_path) -> None:
    raw = json.loads(DEFAULT_SOURCE_ARCHIVE.read_text(encoding="utf-8"))
    for field, value in (
        ("query_png_base64_by_side", {"side_0": "AA=="}),
        ("query_source_calls_made", 1),
        ("freeze", {"forged": True}),
        ("rank_artifact", {"forged": True}),
    ):
        changed = dict(raw)
        changed[field] = value
        path = tmp_path / f"{field}.json"
        path.write_text(json.dumps(changed), encoding="utf-8")
        with pytest.raises(PanelFeatureExposedSupportSmokeError):
            _read_source(path)


def test_precommit_authorizes_no_query_or_freeze() -> None:
    task, panel_ids, panels, source_digest = _read_source(DEFAULT_SOURCE_ARCHIVE)
    authorization, precommit = _authorization(
        task, panel_ids, panels, source_digest
    )
    assert authorization["query_release_or_observation_authorized"] is False
    assert precommit["physical_call_plan"]["query"] == 0
    assert precommit["query_pixels_available_to_command"] is False
    assert precommit["frozen_predicate_created"] is False

