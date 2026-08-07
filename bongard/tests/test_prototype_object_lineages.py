from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from io import BytesIO

from PIL import Image, ImageDraw
import pytest

from bongard.evidence import Disposition
from bongard.prototype_object_hypotheses import extract_object_hypothesis_packet
from bongard.prototype_object_lineages import (
    LineageOwnershipState,
    ObjectLineageError,
    ObjectLineageObservationAggregation,
    ObjectLineagePacket,
    aggregate_lineage_observations,
    extract_object_lineage_packet,
    object_scene_evidence_from_lineage_aggregation,
    verify_object_lineage_observation_aggregation,
    verify_object_lineage_packet,
)
from bongard.prototype_object_profiles import (
    OBJECT_FEATURE_IDS,
    IntegerInterval,
    ObjectFeatureCell,
    ObjectFeatureCellState,
    ObjectHypothesisBinding,
    ObjectLocalObservationPacket,
)
from bongard.prototype_object_version_space import ObjectSceneEvidence
from bongard.visual_witnesses import VISUAL_WITNESS_SCENARIO_IDS


def _png(kind: str) -> bytes:
    image = Image.new("RGB", (96, 64), "white")
    draw = ImageDraw.Draw(image)
    if kind == "two_stable_objects":
        draw.rectangle((8, 12, 26, 34), fill="black")
        draw.ellipse((60, 16, 78, 34), fill="black")
    elif kind == "stable_plus_threshold_only":
        draw.rectangle((8, 12, 26, 34), fill="black")
        # Foreground at strengths 32 and 64, but absent at strength 96.
        draw.ellipse((60, 16, 78, 34), fill=(180, 180, 180))
    elif kind == "unstable_union_only":
        draw.rectangle((12, 18, 29, 40), fill="black")
        draw.rectangle((66, 18, 83, 40), fill="black")
        # Joins the blocks in the lower-threshold scenarios only.
        draw.rectangle((30, 28, 65, 30), fill=(180, 180, 180))
    elif kind != "blank":
        raise AssertionError(f"unknown test image kind: {kind}")
    output = BytesIO()
    image.save(output, format="PNG", optimize=False)
    return output.getvalue()


def _binding(hypothesis, *, catalog_digest: str) -> ObjectHypothesisBinding:
    return ObjectHypothesisBinding(
        scenario_id=hypothesis.scenario_id,
        hypothesis_id=hypothesis.hypothesis_id,
        source_component_ids=hypothesis.source_component_ids,
        source_component_mask_digests=hypothesis.source_component_mask_digests,
        union_mask_digest=hypothesis.union_mask_digest,
        union_bbox=hypothesis.bbox_pixels,
        union_crop_digest=hypothesis.masked_crop_pixel_digest,
        hypothesis_catalog_digest=catalog_digest,
    )


def _scored(
    hypothesis_id: str, feature_id: str, lower: int, upper: int | None = None
) -> ObjectFeatureCell:
    return ObjectFeatureCell(
        hypothesis_id,
        feature_id,
        ObjectFeatureCellState.SCORED,
        IntegerInterval(lower, lower if upper is None else upper),
    )


def _local_packets(png_bytes: bytes):
    hypothesis_packet = extract_object_hypothesis_packet(png_bytes)
    lineage_packet = extract_object_lineage_packet(png_bytes, hypothesis_packet)
    assert len(lineage_packet.lineages) == 3
    assert sum(item.eligible_for_aggregation for item in lineage_packet.lineages) == 2

    eligible = tuple(
        item for item in lineage_packet.lineages if item.eligible_for_aggregation
    )
    eligible_by_member = {
        (member.scenario_id, member.hypothesis_id): lineage_index
        for lineage_index, lineage in enumerate(eligible)
        for member in lineage.members
    }
    scenario_index = {
        scenario_id: index
        for index, scenario_id in enumerate(VISUAL_WITNESS_SCENARIO_IDS)
    }
    bird_intervals = (
        ((100_000, 110_000), (200_000, 210_000), (150_000, 160_000)),
        ((800_000, 810_000), (850_000, 860_000), (900_000, 910_000)),
    )
    catalog_digest = hypothesis_packet.digest()
    packets: list[ObjectLocalObservationPacket] = []
    for scenario in hypothesis_packet.scenarios:
        cells: list[ObjectFeatureCell] = []
        for hypothesis in scenario.hypotheses:
            lineage_index = eligible_by_member.get(
                (scenario.scenario_id, hypothesis.hypothesis_id)
            )
            for feature_id in OBJECT_FEATURE_IDS:
                if lineage_index is not None and feature_id == "bird_like_support_ppm":
                    lower, upper = bird_intervals[lineage_index][
                        scenario_index[scenario.scenario_id]
                    ]
                    cell = _scored(hypothesis.hypothesis_id, feature_id, lower, upper)
                elif (
                    lineage_index == 0
                    and feature_id == "straight_span_count"
                    and scenario_index[scenario.scenario_id] == 1
                ):
                    cell = ObjectFeatureCell(
                        hypothesis.hypothesis_id,
                        feature_id,
                        ObjectFeatureCellState.INDETERMINATE,
                        None,
                        reason="visually_ambiguous",
                    )
                elif (
                    lineage_index == 0
                    and feature_id == "straight_span_count"
                    and scenario_index[scenario.scenario_id] == 2
                ):
                    cell = ObjectFeatureCell(
                        hypothesis.hypothesis_id,
                        feature_id,
                        ObjectFeatureCellState.ERROR,
                        None,
                        reason="observer_failed",
                        error_type="SyntheticVisionError",
                    )
                elif (
                    lineage_index == 0
                    and feature_id == "inward_arc_count"
                    and scenario_index[scenario.scenario_id] == 1
                ):
                    cell = ObjectFeatureCell(
                        hypothesis.hypothesis_id,
                        feature_id,
                        ObjectFeatureCellState.INDETERMINATE,
                        None,
                        reason="visually_ambiguous",
                    )
                elif lineage_index is None and feature_id == "bird_like_support_ppm":
                    # The excluded whole-scene union must never overwrite either
                    # stable lineage, even though its score is the largest.
                    cell = _scored(hypothesis.hypothesis_id, feature_id, 999_999)
                else:
                    cell = _scored(hypothesis.hypothesis_id, feature_id, 0)
                cells.append(cell)
        packets.append(
            ObjectLocalObservationPacket.create(
                scenario.scenario_id,
                tuple(
                    _binding(item, catalog_digest=catalog_digest)
                    for item in scenario.hypotheses
                ),
                cells,
                panel_digest=hypothesis_packet.panel_digest,
                visual_witness_packet_digest=(
                    hypothesis_packet.visual_witness_packet_digest
                ),
                hypothesis_catalog_digest=catalog_digest,
                feature_protocol_digest="1" * 64,
                feature_model_id="synthetic-test-model",
                feature_receipt_digest="2" * 64,
                feature_payload_digest="3" * 64,
            )
        )
    return lineage_packet, tuple(packets)


def _feature(observation, feature_id: str):
    return next(item for item in observation.features if item.feature_id == feature_id)


def _scene_feature(lineage, feature_id: str):
    return next(
        item for item in lineage.feature_values if item.feature_id == feature_id
    )


def test_lineage_extraction_is_deterministic_and_cold_replays_exact_pixels() -> None:
    png_bytes = _png("two_stable_objects")
    first = extract_object_lineage_packet(png_bytes)
    second = extract_object_lineage_packet(png_bytes)

    assert first == second
    assert first.digest() == second.digest()
    assert ObjectLineagePacket.from_data(first.to_data()) == first
    assert verify_object_lineage_packet(first, png_bytes) == first

    tampered = deepcopy(first.to_data())
    tampered["width_pixels"] += 1
    altered = ObjectLineagePacket.from_data(tampered)
    with pytest.raises(ObjectLineageError, match="exact PNG replay"):
        verify_object_lineage_packet(altered, png_bytes)


def test_reciprocal_stable_lineages_preserve_ownership_and_exclude_scene_union() -> None:
    packet = extract_object_lineage_packet(_png("two_stable_objects"))

    assert len(packet.lineages) == 3
    assert packet.unlinked_hypothesis_count == 0
    assert packet.ambiguous_member_target_count == 0
    assert packet.has_unresolved_lineages is False
    first, second, whole_scene = packet.lineages
    for lineage in (first, second):
        assert lineage.ownership_state is LineageOwnershipState.SAFE_SINGLETON
        assert lineage.eligible_for_aggregation is True
        assert tuple(item.scenario_id for item in lineage.members) == (
            VISUAL_WITNESS_SCENARIO_IDS
        )
        assert len(lineage.links) == 3
        assert all(item.to_data()["reciprocal_unique_best"] for item in lineage.links)
    assert whole_scene.ownership_state is LineageOwnershipState.UNRESOLVED_UNION
    assert whole_scene.eligible_for_aggregation is False
    assert all(item.is_whole_scene_union for item in whole_scene.members)


def test_unmatched_hypotheses_are_audited_without_poisoning_a_stable_lineage() -> None:
    unmatched = extract_object_lineage_packet(_png("stable_plus_threshold_only"))
    assert unmatched.unlinked_hypothesis_count > 0
    assert any(item.eligible_for_aggregation for item in unmatched.lineages)
    assert unmatched.ambiguous_member_target_count == 0
    assert unmatched.has_unresolved_lineages is False

    ambiguous = replace(
        unmatched,
        ambiguous_member_target_count=1,
        has_unresolved_lineages=True,
    )
    assert ambiguous.has_unresolved_lineages is True
    with pytest.raises(ObjectLineageError, match="unresolved-lineage flag"):
        replace(ambiguous, has_unresolved_lineages=False)

    no_eligible = extract_object_lineage_packet(_png("unstable_union_only"))
    assert no_eligible.lineages
    assert not any(item.eligible_for_aggregation for item in no_eligible.lineages)
    assert no_eligible.has_unresolved_lineages is True

    blank = extract_object_lineage_packet(_png("blank"))
    assert blank.lineages == ()
    assert blank.has_unresolved_lineages is True


def test_aggregation_envelopes_each_lineage_without_cross_lineage_max() -> None:
    lineage_packet, local_packets = _local_packets(_png("two_stable_objects"))
    aggregation = aggregate_lineage_observations(lineage_packet, local_packets)

    assert verify_object_lineage_observation_aggregation(
        aggregation, lineage_packet, local_packets
    ) == aggregation
    assert ObjectLineageObservationAggregation.from_data(
        aggregation.to_data()
    ) == aggregation
    assert tuple(item.lineage_id for item in aggregation.lineages) == (
        "lineage-00000000",
        "lineage-00000001",
    )
    assert aggregation.excluded_lineage_ids == ("lineage-00000002",)
    assert aggregation.unresolved_lineage_possible is False

    first_bird = _feature(aggregation.lineages[0], "bird_like_support_ppm")
    second_bird = _feature(aggregation.lineages[1], "bird_like_support_ppm")
    assert first_bird.state is ObjectFeatureCellState.SCORED
    assert first_bird.interval == IntegerInterval(100_000, 210_000)
    assert second_bird.interval == IntegerInterval(800_000, 910_000)
    assert first_bird.interval != second_bird.interval
    assert all(item.interval != IntegerInterval(100_000, 999_999) for item in (
        first_bird,
        second_bird,
    ))


def test_aggregation_failure_precedence_and_version_space_conversion() -> None:
    lineage_packet, local_packets = _local_packets(_png("two_stable_objects"))
    aggregation = aggregate_lineage_observations(lineage_packet, local_packets)
    first = aggregation.lineages[0]

    errored = _feature(first, "straight_span_count")
    assert errored.state is ObjectFeatureCellState.ERROR
    assert errored.interval is None
    assert errored.reason == "lineage_member_error"
    assert errored.error_type == "ObjectFeatureCellError"

    indeterminate = _feature(first, "inward_arc_count")
    assert indeterminate.state is ObjectFeatureCellState.INDETERMINATE
    assert indeterminate.interval is None
    assert indeterminate.reason == "lineage_member_indeterminate"

    evidence = object_scene_evidence_from_lineage_aggregation(
        "bd/train/synthetic-task/1/0.png", aggregation
    )
    assert isinstance(evidence, ObjectSceneEvidence)
    assert evidence.lineage_catalog_digest == aggregation.lineage_packet_digest
    assert evidence.unresolved_lineage_possible is False
    assert tuple(item.lineage_id for item in evidence.lineages) == (
        "lineage-00000000",
        "lineage-00000001",
    )
    converted_error = _scene_feature(evidence.lineages[0], "straight_span_count")
    converted_indeterminate = _scene_feature(
        evidence.lineages[0], "inward_arc_count"
    )
    converted_bird = _scene_feature(
        evidence.lineages[0], "bird_like_support_ppm"
    )
    assert converted_error.disposition is Disposition.ERROR
    assert converted_indeterminate.disposition is Disposition.INDETERMINATE
    assert converted_bird.disposition is Disposition.PRESENT
    assert converted_bird.interval == IntegerInterval(100_000, 210_000)


def test_aggregation_rejects_serialized_and_input_tampering() -> None:
    lineage_packet, local_packets = _local_packets(_png("two_stable_objects"))
    aggregation = aggregate_lineage_observations(lineage_packet, local_packets)

    serialized = deepcopy(aggregation.to_data())
    serialized["lineages"][0]["features"][0]["member_cell_digests"][0] = "f" * 64
    with pytest.raises(ObjectLineageError, match="digest differs"):
        ObjectLineageObservationAggregation.from_data(serialized)

    packet = local_packets[0]
    changed_cells = list(packet.cells)
    old = changed_cells[0]
    assert old.state is ObjectFeatureCellState.SCORED
    changed_cells[0] = _scored(old.hypothesis_id, old.feature_id, 7)
    changed_packet = ObjectLocalObservationPacket.create(
        packet.scenario_id,
        packet.hypotheses,
        changed_cells,
        panel_digest=packet.panel_digest,
        visual_witness_packet_digest=packet.visual_witness_packet_digest,
        hypothesis_catalog_digest=packet.hypothesis_catalog_digest,
        feature_protocol_digest=packet.feature_protocol_digest,
        feature_model_id=packet.feature_model_id,
        feature_receipt_digest=packet.feature_receipt_digest,
        feature_payload_digest=packet.feature_payload_digest,
    )
    with pytest.raises(ObjectLineageError, match="aggregation differs"):
        verify_object_lineage_observation_aggregation(
            aggregation,
            lineage_packet,
            (changed_packet, *local_packets[1:]),
        )
