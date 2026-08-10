from __future__ import annotations

from collections import Counter
from dataclasses import replace
import json
from pathlib import Path

import pytest

import bongard.panel_action_local_supervision_authority as authority_module
from bongard.panel_action_local_supervision_authority import (
    AUTHORITY_SCHEMA,
    Disposition,
    EXPECTED_ACTION_COUNT_HISTOGRAM,
    LocalSupervisionError,
    SUPERVISION_SCHEMA,
    compile_pose_free_panel,
    load_development_authority,
    verify_development_authority,
)


ZERO_ADDRESS = "sha256:" + "0" * 64
ROOT = Path(__file__).resolve().parents[2]
LIVE_ACTION_SOURCE = (
    ROOT
    / "downloads/ShapeBongard_V2_full/ShapeBongard_V2/hd/hd_action_programs.json"
)


def _compile(image_program: object):
    return authority_module._compile_image_program(
        panel_id="synthetic/development/0",
        cohort="train",
        image_program=image_program,
        authority_record_digest=ZERO_ADDRESS,
    )


def test_exact_token_centers_have_conservative_rounding_intervals_and_no_pixels():
    result = _compile(
        [[
            "line_normal_0.250-0.500",
            "arc_zigzag_0.500_0.750-0.250",
        ]]
    )
    assert result.disposition is Disposition.CERTIFIED
    data = result.to_data()
    assert data["schema"] == SUPERVISION_SCHEMA
    assert data["carrier_instance_count"] == {
        "disposition": "CERTIFIED",
        "value": 2,
    }
    assert data["shape_instance_count"] == {
        "disposition": "CERTIFIED",
        "value": 1,
    }
    shape = data["shape_multiset"][0]
    assert shape["action_count"] == 2
    line = next(row for row in shape["action_multiset"] if row["primitive"] == "line")
    arc = next(row for row in shape["action_multiset"] if row["primitive"] == "arc")
    assert line == {
        "length_normalized_micro_interval": {
            "lower": 249_500,
            "upper": 250_500,
            "unit": "normalized_micro",
        },
        "length_source_normalized_milli": 250,
        "multiplicity": 1,
        "primitive": "line",
    }
    assert arc == {
        "multiplicity": 1,
        "primitive": "arc",
        "radius_normalized_micro_interval": {
            "lower": 499_500,
            "upper": 500_500,
            "unit": "normalized_micro",
        },
        "radius_source_normalized_milli": 500,
        "sweep_magnitude_degrees_milli_interval": {
            "lower": 179_640,
            "upper": 180_360,
            "unit": "degree_milli",
        },
        "sweep_magnitude_source_degrees_milli": 180_000,
    }
    assert shape["internal_junction_multiset"][0][
        "turn_magnitude_degrees_milli_interval"
    ] == {"lower": 89_820, "upper": 90_180, "unit": "degree_milli"}
    assert shape["internal_junction_multiset"][0][
        "turn_magnitude_source_degrees_milli"
    ] == 90_000
    assert data["pixel_registration"]["disposition"] == "GAP"
    assert data["pixel_instance_assignment"]["disposition"] == "GAP"
    assert data["sequence_endpoint_localization"]["disposition"] == "GAP"
    serialized = json.dumps(data, sort_keys=True)
    assert "source_signed_turn" not in serialized
    assert "source_signed_sweep" not in serialized
    assert "zigzag" not in serialized


def test_shapes_are_an_unordered_multiset_but_carrier_instances_are_actions():
    line_shape = ["line_circle_0.400-0.500"]
    arc_shape = ["arc_square_0.500_0.625-0.500"]
    first = _compile([line_shape, arc_shape]).to_data()
    reversed_shapes = _compile([arc_shape, line_shape]).to_data()
    assert first["shape_multiset"] == reversed_shapes["shape_multiset"]
    assert first["carrier_instance_count"]["value"] == 2
    assert first["shape_instance_count"]["value"] == 2

    duplicates = _compile([line_shape, line_shape]).to_data()
    assert duplicates["shape_instance_count"]["value"] == 2
    assert duplicates["carrier_instance_count"]["value"] == 2
    assert len(duplicates["shape_multiset"]) == 1
    assert duplicates["shape_multiset"][0]["multiplicity"] == 2


@pytest.mark.parametrize(
    ("program", "code"),
    [
        ([["bezier_normal_0.500-0.500"]], "unsupported_action_syntax"),
        ([["line_dotted_0.500-0.500"]], "unsupported_style"),
        ([["line_normal_0.000-0.500"]], "degenerate_line"),
        ([[]], "unsupported_action_capacity"),
        ([], "unsupported_shape_capacity"),
    ],
)
def test_unsupported_or_degenerate_programs_are_typed_gaps(program, code):
    result = _compile(program)
    assert result.disposition is Disposition.GAP
    data = result.to_data()
    assert data["gap"] == {
        "code": code,
        "detail": data["gap"]["detail"],
        "disposition": "GAP",
    }
    assert "carrier_instance_count" not in data
    assert "shape_multiset" not in data


def test_top_level_selector_materializes_only_allowlisted_values():
    payload = (
        b'{"sealed_target":{"deep":[{"latent":"must-not-return"}]},'
        b'"allowed":[[[["line_normal_0.500-0.500"]]]],'
        b'"sealed_calibration":[1,2,3]}'
    )
    selected = authority_module._select_top_level_values(payload, {"allowed"})
    assert set(selected) == {"allowed"}
    assert selected["allowed"] == [[[['line_normal_0.500-0.500']]]]


@pytest.mark.skipif(not LIVE_ACTION_SOURCE.exists(), reason="official local release absent")
def test_live_development_authority_sweeps_all_panels_and_rejects_forgery():
    authority = load_development_authority(repository_root=ROOT)
    verify_development_authority(authority)
    record = authority.to_record()
    assert record["schema"] == AUTHORITY_SCHEMA
    assert record["custody"] == {
        "action_program_raw_bytes_scanned_for_digest_and_key_selection": True,
        "authorized_cohorts": ["train", "validation"],
        "calibration_or_evaluation_identifiers_opened": 0,
        "label_manifests_opened": 0,
        "nonselected_action_program_values_materialized": 0,
        "png_files_opened": 0,
        "query_or_target_pixels_opened": 0,
        "target_family_prefix_forbidden": "hd_convex-has_four_straight_lines_",
    }
    assert len(authority.selected_programs) == 900

    for cohort, panel_ids in authority.cohort_panel_ids:
        histogram: Counter[int] = Counter()
        gaps: Counter[str] = Counter()
        for panel_id in panel_ids:
            result = compile_pose_free_panel(authority, panel_id)
            if result.disposition is Disposition.GAP:
                assert result.gap is not None
                gaps[result.gap.code] += 1
            else:
                assert result.carrier_instance_count is not None
                histogram[result.carrier_instance_count] += 1
                assert result.shape_instance_count == 1
        assert not gaps
        assert dict(sorted(histogram.items())) == EXPECTED_ACTION_COUNT_HISTOGRAM[cohort]

    target_panel = "hd/hd_convex-has_four_straight_lines_0000/1/0.png"
    with pytest.raises(LocalSupervisionError, match="outside sealed development"):
        compile_pose_free_panel(authority, target_panel)

    forged_rows = replace(
        authority,
        cohort_panel_ids=(
            ("train", authority.cohort_panel_ids[0][1][:-1] + (target_panel,)),
            authority.cohort_panel_ids[1],
        ),
    )
    with pytest.raises(LocalSupervisionError, match="builder seal"):
        verify_development_authority(forged_rows)
    with pytest.raises(LocalSupervisionError, match="outside sealed development"):
        compile_pose_free_panel(forged_rows, target_panel)

    naked = replace(authority, _seal=None)
    with pytest.raises(LocalSupervisionError, match="frozen loader"):
        compile_pose_free_panel(naked, authority.cohort_panel_ids[0][1][0])
