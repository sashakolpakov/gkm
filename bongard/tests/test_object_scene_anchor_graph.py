from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from bongard.canonical import canonical_digest
from bongard.object_scene_anchor_graph import (
    ANCHOR_GRAPH_ALGORITHM_ID,
    AnchorExtractionLimits,
    ObjectSceneAnchorGraph,
    extract_object_scene_anchor_graph,
    object_scene_anchor_graph_extractor_digest,
    object_scene_anchor_graph_source_digest,
    verify_object_scene_anchor_graph,
)


def _bar() -> np.ndarray:
    mask = np.zeros((15, 15), dtype=bool)
    mask[7, 2:13] = True
    return mask


def _tee() -> np.ndarray:
    mask = np.zeros((15, 15), dtype=bool)
    mask[3, 3:12] = True
    mask[3:13, 7] = True
    return mask


def _why() -> np.ndarray:
    mask = np.zeros((15, 15), dtype=bool)
    for offset in range(5):
        mask[7 - offset, 7 - offset] = True
        mask[7 - offset, 7 + offset] = True
    mask[7:13, 7] = True
    return mask


def _cycle() -> np.ndarray:
    mask = np.zeros((15, 15), dtype=bool)
    mask[3, 3:12] = True
    mask[11, 3:12] = True
    mask[3:12, 3] = True
    mask[3:12, 11] = True
    return mask


@pytest.mark.parametrize("factory", [_tee, _why])
def test_three_arm_join_has_maximal_parts_and_complete_cyclic_frame(factory) -> None:
    mask = factory()
    graph = extract_object_scene_anchor_graph(mask, "object-0")

    assert graph.status.state == "clean"
    assert graph.status.reason == "complete"
    assert len(graph.terminals) == 3
    assert len(graph.joins) == 1
    assert len(graph.parts) == 3
    assert len(graph.cyclic_frames) == 1
    assert all(not part.closed and len(part.endpoint_node_ids) == 2 for part in graph.parts)

    join = graph.joins[0]
    frame = graph.cyclic_frames[0]
    assert join.join_id == "join-00000000"
    assert join.cyclic_frame_id == frame.frame_id
    assert tuple(sorted(frame.clockwise_incident_part_ids)) == join.incident_part_ids
    assert len(frame.clockwise_tangent_points_q16) == len(join.incident_part_ids) == 3
    assert {terminal.incident_part_id for terminal in graph.terminals} == {
        part.part_id for part in graph.parts
    }
    verified = verify_object_scene_anchor_graph(graph, expected_mask=mask)
    assert verified == graph
    assert verified is not graph


def test_bar_and_cycle_are_complete_without_synthetic_joins() -> None:
    bar = extract_object_scene_anchor_graph(_bar(), "bar")
    cycle = extract_object_scene_anchor_graph(_cycle(), "cycle")

    assert (len(bar.terminals), len(bar.joins), len(bar.parts)) == (2, 0, 1)
    assert bar.parts[0].endpoint_node_ids == (
        "terminal-00000000",
        "terminal-00000001",
    )
    assert not bar.parts[0].closed

    assert (len(cycle.terminals), len(cycle.joins), len(cycle.parts)) == (0, 0, 1)
    assert cycle.parts[0].endpoint_node_ids == ()
    assert cycle.parts[0].closed
    assert cycle.cyclic_frames == ()


def test_exact_mask_shape_digest_roundtrip_and_determinism() -> None:
    mask = _why()
    first = extract_object_scene_anchor_graph(mask, "stable-object")
    second = extract_object_scene_anchor_graph(np.asfortranarray(mask), "stable-object")

    assert first == second
    assert first.artifact_digest == second.artifact_digest
    assert first.extractor_artifact_digest == object_scene_anchor_graph_extractor_digest()
    assert len(object_scene_anchor_graph_source_digest()) == 64
    assert first.to_data()["algorithm_id"] == ANCHOR_GRAPH_ALGORITHM_ID
    assert ObjectSceneAnchorGraph.from_data(first.to_data()) == first

    reshaped = mask.reshape(9, 25)
    other = extract_object_scene_anchor_graph(reshaped, "stable-object")
    assert other.mask_digest != first.mask_digest
    assert (first.mask_height_pixels, first.mask_width_pixels) == mask.shape


def test_resource_cap_is_indeterminate_and_exposes_no_partial_graph() -> None:
    limits = AnchorExtractionLimits(max_skeleton_pixels=3)
    graph = extract_object_scene_anchor_graph(_bar(), "capped", limits)

    assert graph.status.state == "indeterminate"
    assert graph.status.reason == "skeleton_pixel_cap_exceeded"
    assert graph.skeleton_pixel_count > limits.max_skeleton_pixels
    assert (
        graph.terminals
        == graph.joins
        == graph.compact_components
        == graph.parts
        == graph.cyclic_frames
        == ()
    )
    assert verify_object_scene_anchor_graph(graph, expected_mask=_bar()) == graph


def test_blank_is_clean_and_isolated_pixel_is_a_local_compact_anchor() -> None:
    blank = np.zeros((7, 7), dtype=bool)
    isolated = blank.copy()
    isolated[3, 3] = True

    empty_graph = extract_object_scene_anchor_graph(blank, "blank")
    isolated_graph = extract_object_scene_anchor_graph(isolated, "isolated")

    assert empty_graph.status.state == "clean"
    assert empty_graph.parts == ()
    assert isolated_graph.status.state == "clean"
    assert isolated_graph.status.reason == "complete"
    assert len(isolated_graph.compact_components) == 1
    compact = isolated_graph.compact_components[0]
    assert compact.reason == "isolated_skeleton_component"
    assert compact.foreground_pixel_count == compact.skeleton_pixel_count == 1
    assert isolated_graph.parts == ()


@pytest.mark.parametrize("diagonal", [False, True])
def test_two_pixel_component_has_two_terminals_and_one_part(diagonal: bool) -> None:
    mask = np.zeros((4, 4), dtype=bool)
    mask[1, 1] = True
    mask[2 if diagonal else 1, 2] = True

    graph = extract_object_scene_anchor_graph(mask, f"two-pixel-{diagonal}")

    assert graph.status.state == "clean"
    assert len(graph.terminals) == 2
    assert len(graph.parts) == 1
    assert graph.parts[0].endpoint_node_ids == (
        "terminal-00000000",
        "terminal-00000001",
    )
    assert graph.compact_components == ()


def test_nonempty_component_thinned_empty_becomes_a_typed_compact_anchor() -> None:
    mask = np.zeros((6, 6), dtype=bool)
    mask[2:4, 2:4] = True

    graph = extract_object_scene_anchor_graph(mask, "two-by-two")

    assert graph.status.state == "clean"
    assert graph.skeleton_pixel_count == 0
    assert graph.terminals == graph.joins == graph.parts == ()
    assert len(graph.compact_components) == 1
    compact = graph.compact_components[0]
    assert compact.reason == "source_component_thinned_empty"
    assert compact.foreground_pixel_count == 4
    assert compact.skeleton_pixel_count == 0
    assert len(compact.source_component_digest) == 64


def test_compact_component_does_not_poison_a_valid_bar_component() -> None:
    mask = _bar()
    mask[1, 1] = True

    graph = extract_object_scene_anchor_graph(mask, "bar-plus-dot")

    assert graph.status.state == "clean"
    assert len(graph.compact_components) == 1
    assert len(graph.terminals) == 2
    assert len(graph.parts) == 1


def test_compact_component_cap_is_explicit_indeterminate() -> None:
    mask = np.zeros((9, 9), dtype=bool)
    mask[1, 1] = True
    mask[7, 7] = True
    limits = AnchorExtractionLimits(max_compact_components=1)

    graph = extract_object_scene_anchor_graph(mask, "compact-cap", limits)

    assert graph.status.state == "indeterminate"
    assert graph.status.reason == "compact_component_cap_exceeded"
    assert graph.compact_components == ()


def test_join_reentry_pixel_is_absorbed_without_whole_object_error() -> None:
    mask = np.asarray(
        [
            [False, False, False, True, False],
            [False, False, False, True, False],
            [False, False, True, False, True],
            [True, True, False, True, False],
            [False, True, True, True, False],
        ],
        dtype=bool,
    )

    graph = extract_object_scene_anchor_graph(mask, "join-reentry")

    assert graph.status.state == "clean"
    assert len(graph.joins) == 1
    assert len(graph.terminals) == 1
    assert len(graph.parts) == 2
    assert any(part.closed for part in graph.parts)


def test_composite_dtos_reject_lists_and_wrong_member_types() -> None:
    graph = extract_object_scene_anchor_graph(_tee(), "strict-tuples")
    part = graph.parts[0]
    join = graph.joins[0]

    with pytest.raises(TypeError, match="exact tuple"):
        replace(part, path_q16=list(part.path_q16))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="exact tuple"):
        replace(join, incident_part_ids=list(join.incident_part_ids))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="exact tuple"):
        replace(graph, parts=list(graph.parts))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="exact tuple"):
        replace(
            join,
            incident_part_ids=(join.incident_part_ids[0], 7, *join.incident_part_ids[2:]),  # type: ignore[arg-type]
        )


def test_verify_rejects_nonexact_graph_subclasses() -> None:
    graph = extract_object_scene_anchor_graph(_bar(), "canonical-type")

    class DerivedGraph(ObjectSceneAnchorGraph):
        pass

    derived = DerivedGraph(**graph.__dict__)
    with pytest.raises(TypeError, match="exact ObjectSceneAnchorGraph"):
        verify_object_scene_anchor_graph(derived)


def test_digest_and_exact_replay_reject_tampering() -> None:
    mask = _tee()
    graph = extract_object_scene_anchor_graph(mask, "tamper-target")

    stale_digest = deepcopy(graph.to_data())
    stale_digest["mask_digest"] = "0" * 64
    with pytest.raises(ValueError, match="artifact digest"):
        ObjectSceneAnchorGraph.from_data(stale_digest)

    resigned = deepcopy(graph.to_data())
    resigned["mask_digest"] = "0" * 64
    unsigned = {key: value for key, value in resigned.items() if key != "artifact_digest"}
    resigned["artifact_digest"] = canonical_digest(unsigned)
    forged = ObjectSceneAnchorGraph.from_data(resigned)
    with pytest.raises(ValueError, match="exact mask replay"):
        verify_object_scene_anchor_graph(forged, expected_mask=mask)


@pytest.mark.parametrize(
    "bad",
    [np.zeros((4, 4), dtype=np.uint8), np.zeros((4,), dtype=bool)],
)
def test_input_requires_an_exact_two_dimensional_bool_mask(bad: np.ndarray) -> None:
    with pytest.raises((TypeError, ValueError), match="bool|two-dimensional"):
        extract_object_scene_anchor_graph(bad, "bad")
