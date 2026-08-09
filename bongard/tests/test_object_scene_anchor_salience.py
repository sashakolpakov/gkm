"""Tests for deterministic whole-graph object-anchor salience."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

import bongard.object_scene_anchor_salience as salience
from bongard.canonical import canonical_digest
from bongard.object_scene_anchor_graph import AnchorExtractionLimits
from bongard.object_scene_anchor_salience import (
    ANCHOR_SALIENCE_Q_DEFINITION,
    ANCHOR_SALIENCE_RADIUS_RULE,
    AnchorSalienceLimits,
    ObjectSceneAnchorSalience,
    extract_object_scene_anchor_salience,
    object_scene_anchor_salience_extractor_digest,
    verify_object_scene_anchor_salience,
)


def _line() -> np.ndarray:
    mask = np.zeros((23, 29), dtype=bool)
    mask[11, 5:24] = True
    return mask


def _plus() -> np.ndarray:
    mask = np.zeros((31, 31), dtype=bool)
    mask[15, 4:27] = True
    mask[4:27, 15] = True
    return mask


def _loop() -> np.ndarray:
    mask = np.zeros((41, 41), dtype=bool)
    mask[10, 10:31] = True
    mask[30, 10:31] = True
    mask[10:31, 10] = True
    mask[10:31, 30] = True
    return mask


def _reseal_salience(data: dict[str, object]) -> dict[str, object]:
    data["artifact_digest"] = canonical_digest(
        {key: value for key, value in data.items() if key != "artifact_digest"}
    )
    return data


def test_singleton_uses_infinite_false_exterior_and_fixed_rounding_schedule() -> None:
    mask = np.zeros((9, 11), dtype=bool)
    mask[0, 10] = True
    result = extract_object_scene_anchor_salience(mask, "singleton")

    assert result.status.state == "clean"
    assert result.q_pixels == 1
    assert result.padding_pixels == 6
    assert result.radius_schedule_pixels == (2, 3, 3, 4, 4)
    assert result.audit_radius_pixels == 5
    assert (result.crop_y0, result.crop_x0, result.crop_y1, result.crop_x1) == (
        0,
        10,
        1,
        11,
    )
    assert result.selected_attempt_index == 0
    assert result.selected_graph is not None
    assert result.selected_graph.mask_height_pixels == 13
    assert result.selected_graph.mask_width_pixels == 13
    assert len(result.selected_graph.parts) == 0
    assert len(result.selected_graph.compact_components) == 1
    assert [(item.anchor_id, item.raw_skeleton_pixel_count) for item in result.selected_support_counts] == [
        ("compact-00000000", 1)
    ]
    assert len(result.ownership) == 1
    point = result.ownership[0]
    assert (point.source_y, point.source_x) == (0, 10)
    assert (point.padded_y, point.padded_x) == (6, 6)


def test_q_is_nearest_rank_p90_with_explicit_false_boundary() -> None:
    # A solid tight 5x5 crop thins to its center.  Its exact chessboard distance
    # to the false exterior is three, which would be ambiguous without padding.
    result = extract_object_scene_anchor_salience(
        np.ones((5, 5), dtype=bool), "solid-square"
    )
    assert result.q_pixels == 3
    assert result.padding_pixels == 16
    assert result.to_data()["q_definition"] == ANCHOR_SALIENCE_Q_DEFINITION
    assert result.to_data()["radius_rule"] == ANCHOR_SALIENCE_RADIUS_RULE


def test_line_ownership_is_complete_exact_and_coordinate_bound() -> None:
    mask = _line()
    result = extract_object_scene_anchor_salience(mask, "line")
    graph = result.selected_graph
    assert result.status.state == "clean"
    assert graph is not None
    assert len(graph.parts) == 1
    assert len(graph.cyclic_frames) == 0
    assert result.raw_graph is not None
    assert len(result.ownership) == result.raw_graph.skeleton_pixel_count == 19
    assert sum(
        item.raw_skeleton_pixel_count for item in result.selected_support_counts
    ) == len(result.ownership)
    assert {item.selected_anchor_id for item in result.ownership} == {
        "part-00000000"
    }
    assert all(item.raw_owner_anchor_ids == ("part-00000000",) for item in result.ownership)
    assert all(
        item.padded_y
        == item.source_y - result.crop_y0 + result.padding_pixels
        and item.padded_x
        == item.source_x - result.crop_x0 + result.padding_pixels
        for item in result.ownership
    )
    assert result.raw_part_span_digest == canonical_digest(
        [
            {
                "raw_point_id": item.raw_point_id,
                "raw_owner_anchor_ids": list(item.raw_owner_anchor_ids),
            }
            for item in result.ownership
        ]
    )


def test_branch_graph_keeps_complete_frame_and_every_incident_part() -> None:
    result = extract_object_scene_anchor_salience(_plus(), "plus")
    graph = result.selected_graph
    assert result.status.state == "clean"
    assert graph is not None
    assert len(graph.cyclic_frames) == 1
    assert len(graph.joins) == 1
    assert len(graph.parts) == 4
    frame = graph.cyclic_frames[0]
    assert sorted(frame.clockwise_incident_part_ids) == [
        f"part-{index:08d}" for index in range(4)
    ]
    assert {item.anchor_id for item in result.selected_support_counts} == {
        f"part-{index:08d}" for index in range(4)
    }
    assert all(
        item.raw_skeleton_pixel_count >= result.q_pixels
        for item in result.selected_support_counts
    )


def test_loop_remains_one_complete_closed_part_and_holes_are_not_filled() -> None:
    result = extract_object_scene_anchor_salience(_loop(), "loop")
    graph = result.selected_graph
    assert result.status.state == "clean"
    assert graph is not None
    assert len(graph.parts) == 1
    assert graph.parts[0].closed is True
    assert result.to_data()["holes_filled"] is False
    # Dilation preserves a substantial central hole at the selected scale.
    assert graph.foreground_pixel_count < graph.mask_height_pixels * graph.mask_width_pixels


def test_whole_graph_caps_return_indeterminate_and_never_truncate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_ownership(**_: object) -> object:
        raise AssertionError("over-cap graph reached ownership allocation")

    monkeypatch.setattr(salience, "_assign_raw_points", forbidden_ownership)
    result = extract_object_scene_anchor_salience(
        _plus(), "capped-plus", AnchorSalienceLimits(max_frames=0)
    )
    assert result.status.state == "indeterminate"
    assert result.status.reason == "salience_cap_exceeded"
    assert len(result.attempts) == 5
    assert all(item.reason == "frame_cap_exceeded" for item in result.attempts)
    assert all(len(item.graph.cyclic_frames) == 1 for item in result.attempts)
    assert result.selected_graph is None
    assert result.ownership == ()
    assert result.selected_support_counts == ()
    assert result.audit_graph is not None
    assert result.to_data()["whole_graph_selected_never_top_k"] is True
    assert result.to_data()["omitted_anchor_means_absence"] is False
    assert result.to_data()["audit_sentinel_affects_selection"] is False


def test_caller_cannot_relax_the_fixed_complete_graph_cap() -> None:
    with pytest.raises(ValueError, match="cannot relax"):
        AnchorSalienceLimits(max_parts=9)
    with pytest.raises(ValueError, match="fixed resource cap"):
        AnchorSalienceLimits(max_radius_pixels=4097)
    with pytest.raises(ValueError, match="fixed resource cap"):
        AnchorSalienceLimits(max_padded_pixels=16_777_217)
    with pytest.raises(ValueError, match="max_skeleton_pixels"):
        AnchorSalienceLimits(
            anchor_graph_limits=AnchorExtractionLimits(
                max_skeleton_pixels=131_073
            )
        )


def test_morphology_work_bound_is_pure_conservative_and_monotone() -> None:
    # q=2 has padding 11, padded shape 25x27, scheduled radii 4,5,6,7,8,
    # and audit radius 10.  The conservative square footprints sum to 1326.
    assert salience._morphology_work_upper_bound((3, 5), 2) == 675 * 1_326
    assert salience._morphology_work_upper_bound(
        (4, 5), 2
    ) > salience._morphology_work_upper_bound((3, 5), 2)
    assert salience._morphology_work_upper_bound(
        (3, 5), 3
    ) > salience._morphology_work_upper_bound((3, 5), 2)


def test_morphology_work_cap_is_bound_into_extractor_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert salience.ANCHOR_SALIENCE_HARD_MAX_MORPHOLOGY_WORK == 536_870_912
    original = object_scene_anchor_salience_extractor_digest()
    monkeypatch.setattr(
        salience,
        "ANCHOR_SALIENCE_HARD_MAX_MORPHOLOGY_WORK",
        salience.ANCHOR_SALIENCE_HARD_MAX_MORPHOLOGY_WORK - 1,
    )
    assert object_scene_anchor_salience_extractor_digest() != original


def test_morphology_work_cap_boundary_is_inclusive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    crop_shape = (3, 5)
    q = 2
    work = salience._morphology_work_upper_bound(crop_shape, q)
    limits = AnchorSalienceLimits()

    monkeypatch.setattr(
        salience, "ANCHOR_SALIENCE_HARD_MAX_MORPHOLOGY_WORK", work
    )
    assert salience._salience_resource_exceeded(crop_shape, q, limits) is False
    monkeypatch.setattr(
        salience, "ANCHOR_SALIENCE_HARD_MAX_MORPHOLOGY_WORK", work - 1
    )
    assert salience._salience_resource_exceeded(crop_shape, q, limits) is True


def test_compact_macro_anchor_is_not_subject_to_path_support_threshold() -> None:
    mask = np.ones((3, 3), dtype=bool)
    result = extract_object_scene_anchor_salience(mask, "compact-only")
    assert result.status.state == "clean"
    assert result.selected_graph is not None
    assert len(result.selected_graph.parts) == 0
    assert len(result.selected_graph.compact_components) == 1
    assert result.selected_support_counts[0].raw_skeleton_pixel_count == 1
    assert result.q_pixels == 2


def test_many_compact_components_cannot_bypass_the_macro_cap() -> None:
    mask = np.zeros((31, 31), dtype=bool)
    for y in (5, 15, 25):
        for x in (5, 15, 25):
            mask[y, x] = True
    result = extract_object_scene_anchor_salience(mask, "nine-dots")
    assert result.status.reason == "salience_cap_exceeded"
    assert len(result.attempts) == 5
    assert all(item.reason == "compact_cap_exceeded" for item in result.attempts)
    assert all(len(item.graph.compact_components) == 9 for item in result.attempts)
    assert result.selected_graph is None


def test_native_resource_and_raw_graph_caps_are_typed_not_negative(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resource = extract_object_scene_anchor_salience(
        _line(),
        "resource-gap",
        AnchorSalienceLimits(max_padded_pixels=64),
    )
    assert resource.status.state == "indeterminate"
    assert resource.status.reason == "salience_resource_cap_exceeded"
    assert resource.attempts == ()
    assert resource.selected_graph is None
    assert resource.audit_graph is None
    assert resource.audit_disk_footprint_digest is None
    assert resource.audit_envelope_mask_digest is None

    # Force a q whose radius and padded extent each fit their independent caps,
    # but whose cumulative padded-pixel/footprint product exceeds the fixed
    # morphology-work cap.  No native morphology allocation may be reached.
    monkeypatch.setattr(salience, "_q_from_raw_skeleton", lambda *_: 50)
    assert 5 * 50 < salience.ANCHOR_SALIENCE_HARD_MAX_RADIUS_PIXELS
    assert 503 * 521 < salience.ANCHOR_SALIENCE_HARD_MAX_PADDED_PIXELS
    assert 501 * 501 < salience.ANCHOR_SALIENCE_HARD_MAX_PADDED_PIXELS
    assert (
        salience._morphology_work_upper_bound((1, 19), 50)
        > salience.ANCHOR_SALIENCE_HARD_MAX_MORPHOLOGY_WORK
    )

    def forbidden_native(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("product-capped salience reached native morphology")

    original_pad = salience.np.pad

    def forbid_morphology_pad(
        array: object, pad_width: object, *args: object, **kwargs: object
    ) -> object:
        # Raw skeletonization legitimately adds a one-pixel boundary.  The
        # q-derived morphology pad is the allocation guarded by this cap.
        if pad_width != 1:
            return forbidden_native(array, pad_width, *args, **kwargs)
        return original_pad(array, pad_width, *args, **kwargs)

    monkeypatch.setattr(salience.np, "pad", forbid_morphology_pad)
    monkeypatch.setattr(salience, "_disk", forbidden_native)
    monkeypatch.setattr(salience.ndimage, "binary_dilation", forbidden_native)
    product_resource = extract_object_scene_anchor_salience(
        _line(), "product-resource-gap"
    )
    assert product_resource.q_pixels == 50
    assert product_resource.status.state == "indeterminate"
    assert product_resource.status.reason == "salience_resource_cap_exceeded"
    assert product_resource.attempts == ()
    assert product_resource.audit_graph is None
    assert ObjectSceneAnchorSalience.from_data(
        product_resource.to_data()
    ) == product_resource

    forged_product_resource = deepcopy(product_resource.to_data())
    forged_product_resource["audit_graph"] = deepcopy(
        forged_product_resource["raw_graph"]
    )
    forged_product_resource["audit_disk_footprint_digest"] = "0" * 64
    forged_product_resource["audit_envelope_mask_digest"] = "0" * 64
    with pytest.raises(ValueError, match="resource-gap salience exposes"):
        ObjectSceneAnchorSalience.from_data(
            _reseal_salience(forged_product_resource)
        )

    raw = extract_object_scene_anchor_salience(
        _line(),
        "raw-gap",
        AnchorSalienceLimits(
            anchor_graph_limits=AnchorExtractionLimits(max_skeleton_pixels=1)
        ),
    )
    assert raw.status.state == "indeterminate"
    assert raw.status.reason == "raw_anchor_indeterminate"
    assert raw.raw_graph is not None
    assert raw.raw_graph.status.reason == "skeleton_pixel_cap_exceeded"
    assert raw.selected_graph is None


def test_empty_and_invalid_masks_do_not_manufacture_absence() -> None:
    empty = extract_object_scene_anchor_salience(
        np.zeros((4, 7), dtype=bool), "empty"
    )
    assert empty.status.state == "error"
    assert empty.status.reason == "empty_foreground"
    assert empty.raw_graph is None
    assert empty.selected_graph is None
    assert empty.q_pixels == 0

    thinned_empty = np.zeros((4, 4), dtype=bool)
    thinned_empty[:2, :2] = True
    compact_gap = extract_object_scene_anchor_salience(
        thinned_empty, "nonempty-thinned-empty"
    )
    assert compact_gap.source_foreground_pixel_count == 4
    assert compact_gap.raw_graph is not None
    assert len(compact_gap.raw_graph.compact_components) == 1
    assert compact_gap.status.state == "error"
    assert compact_gap.status.reason == "empty_raw_skeleton"
    assert compact_gap.selected_graph is None

    with pytest.raises(TypeError, match="exact bool"):
        extract_object_scene_anchor_salience(
            np.zeros((4, 7), dtype=np.uint8), "wrong-dtype"
        )
    with pytest.raises(ValueError, match="two-dimensional"):
        extract_object_scene_anchor_salience(
            np.zeros((2, 3, 4), dtype=bool), "wrong-rank"
        )
    with pytest.raises(ValueError, match="bounded nonempty string"):
        extract_object_scene_anchor_salience(
            np.zeros((4, 7), dtype=bool), "bad\nobject-id"
        )


def test_round_trip_replay_determinism_and_tamper_detection() -> None:
    mask = _plus()
    first = extract_object_scene_anchor_salience(mask, "replay-plus")
    second = extract_object_scene_anchor_salience(mask.copy(), "replay-plus")
    assert second == first
    assert ObjectSceneAnchorSalience.from_data(first.to_data()) == first
    assert verify_object_scene_anchor_salience(
        first, expected_mask=mask, expected_object_id="replay-plus"
    ) == first
    assert len(object_scene_anchor_salience_extractor_digest()) == 64

    changed = mask.copy()
    changed[5, 5] = True
    with pytest.raises(ValueError, match="exact-mask replay"):
        verify_object_scene_anchor_salience(first, expected_mask=changed)

    tampered = deepcopy(first.to_data())
    tampered["ownership"][0]["selected_distance_pixels"] += 1
    with pytest.raises(ValueError, match="artifact digest"):
        ObjectSceneAnchorSalience.from_data(tampered)

    forged = deepcopy(first.to_data())
    forged["attempts"][0]["graph"]["foreground_pixel_count"] += 1
    forged["artifact_digest"] = canonical_digest(
        {key: value for key, value in forged.items() if key != "artifact_digest"}
    )
    with pytest.raises(ValueError, match="anchor graph artifact digest"):
        ObjectSceneAnchorSalience.from_data(forged)


def test_resealed_derived_fields_must_still_satisfy_structural_replay() -> None:
    source = extract_object_scene_anchor_salience(_plus(), "resealed-plus")

    bad_disk = deepcopy(source.to_data())
    bad_disk["attempts"][0]["disk_footprint_digest"] = "0" * 64
    with pytest.raises(ValueError, match="schedule or dimensions"):
        ObjectSceneAnchorSalience.from_data(_reseal_salience(bad_disk))

    duplicate_point = deepcopy(source.to_data())
    duplicate_point["ownership"][1]["source_y"] = duplicate_point["ownership"][0][
        "source_y"
    ]
    duplicate_point["ownership"][1]["source_x"] = duplicate_point["ownership"][0][
        "source_x"
    ]
    duplicate_point["ownership"][1]["padded_y"] = duplicate_point["ownership"][0][
        "padded_y"
    ]
    duplicate_point["ownership"][1]["padded_x"] = duplicate_point["ownership"][0][
        "padded_x"
    ]
    with pytest.raises(ValueError, match="coordinates are not canonical"):
        ObjectSceneAnchorSalience.from_data(_reseal_salience(duplicate_point))

    missing_raw_owner = deepcopy(source.to_data())
    missing_raw_owner["ownership"][0]["raw_owner_anchor_ids"] = []
    missing_raw_owner["raw_part_span_digest"] = canonical_digest(
        [
            {
                "raw_point_id": item["raw_point_id"],
                "raw_owner_anchor_ids": item["raw_owner_anchor_ids"],
            }
            for item in missing_raw_owner["ownership"]
        ]
    )
    with pytest.raises(ValueError, match="unknown raw anchor"):
        ObjectSceneAnchorSalience.from_data(_reseal_salience(missing_raw_owner))

    cap_gap = extract_object_scene_anchor_salience(
        _plus(), "resealed-cap", AnchorSalienceLimits(max_frames=0)
    ).to_data()
    cap_gap["status"] = {
        "state": "error",
        "reason": "candidate_anchor_error",
    }
    with pytest.raises(ValueError, match="candidate failure status binding"):
        ObjectSceneAnchorSalience.from_data(_reseal_salience(cap_gap))

    missing_audit = extract_object_scene_anchor_salience(
        _plus(), "resealed-audit", AnchorSalienceLimits(max_frames=0)
    ).to_data()
    missing_audit["audit_disk_footprint_digest"] = None
    missing_audit["audit_envelope_mask_digest"] = None
    missing_audit["audit_graph"] = None
    with pytest.raises(ValueError, match="lacks attempts or audit"):
        ObjectSceneAnchorSalience.from_data(_reseal_salience(missing_audit))


def test_audit_graph_error_is_recorded_but_decision_inert() -> None:
    data = extract_object_scene_anchor_salience(_plus(), "audit-inert").to_data()
    audit = data["audit_graph"]
    assert isinstance(audit, dict)
    audit["status"] = {"state": "error", "reason": "unsupported_pixel_graph"}
    for key in ("terminals", "joins", "compact_components", "parts", "cyclic_frames"):
        audit[key] = []
    audit["artifact_digest"] = canonical_digest(
        {key: value for key, value in audit.items() if key != "artifact_digest"}
    )
    restored = ObjectSceneAnchorSalience.from_data(_reseal_salience(data))
    assert restored.status.state == "clean"
    assert restored.audit_graph is not None
    assert restored.audit_graph.status.state == "error"
    assert restored.to_data()["audit_sentinel_affects_selection"] is False


def test_serialized_policy_is_python_canonical() -> None:
    data = extract_object_scene_anchor_salience(_line(), "authority").to_data()
    assert data["python_is_canonical_authority"] is True
    assert not any("lean" in key.casefold() for key in data)
    assert data["predicate_authority_id"].endswith("/python-v1")
