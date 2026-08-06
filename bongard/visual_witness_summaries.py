"""Neutral one-line summaries of verified joint visual witness bundles.

The blind soft scorer accepts only a sorted inventory of verifier-owned
``(witness_id, description)`` pairs.  This module is the deterministic bridge
from the richer :class:`~bongard.visual_witness_bundle.VisualWitnessBundle` to that
inventory.  It has no task, side, role, label, path, or candidate input.

Summaries retain exact Q16 bounding boxes and pixel areas.  Scenario-local
contour and hole identifiers are prefixed by their scenario identifier, so
every emitted witness identifier is globally unique within the panel bundle.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from bongard.artifacts import canonical_digest
from bongard.blind_soft_transport import canonical_witness_summaries
from bongard.contour_witnesses import CountInterval
from bongard.visual_witness_bundle import (
    VisualWitnessBundle,
    verify_visual_witness_bundle,
    visual_witness_bundle_catalog_digest,
    visual_witness_bundle_extractor_digest,
)
from bongard.visual_witnesses import Q16BBox


VISUAL_WITNESS_SUMMARY_ALGORITHM_ID = "bongard.visual-witness-summaries/v2"
VISUAL_WITNESS_SUMMARY_SCHEMA = "gkm.bongard-visual-witness-summaries.v2"
VISUAL_SOFT_WITNESS_INTERFACE_SCHEMA = (
    "gkm.bongard-visual-soft-witness-interface.v2"
)

_MAX_WITNESS_SUMMARIES = 512

VisualWitnessSummary = tuple[str, str]
VisualWitnessSummaries = tuple[VisualWitnessSummary, ...]


def _source_digest() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _summary_artifact_digest(source_digest: str) -> str:
    return canonical_digest(
        {
            "schema": "gkm.bongard-visual-witness-summary-artifact.v2",
            "algorithm_id": VISUAL_WITNESS_SUMMARY_ALGORITHM_ID,
            "source_digest": source_digest,
            "input": "verified_visual_witness_bundle_only",
            "output_schema": VISUAL_WITNESS_SUMMARY_SCHEMA,
            "ordering": "lexicographic_witness_id",
            "geometry": "exact_half_open_unsigned_q16_bbox_and_pixel_area",
            "topology": (
                "per-scenario aggregate and per-contour interval-valued endpoint, "
                "branchpoint, cycle, X-crossing, and signed-curvature summaries"
            ),
            "ownership": (
                "scenario_qualified_hole_to_component_and_component_owned_hole_count"
            ),
            "forbidden_inputs": [
                "task",
                "side",
                "role",
                "label",
                "source_path",
                "candidate",
            ],
        }
    )


def visual_witness_summary_artifact_digest() -> str:
    """Return the source-bound identity of this deterministic summarizer."""

    return _summary_artifact_digest(_source_digest())


def visual_joint_soft_witness_interface_digest() -> str:
    """Bind joint extraction, both witness vocabularies, and summaries."""

    return canonical_digest(
        {
            "schema": VISUAL_SOFT_WITNESS_INTERFACE_SCHEMA,
            "bundle_extractor_artifact_digest": (
                visual_witness_bundle_extractor_digest()
            ),
            "bundle_catalog_digest": visual_witness_bundle_catalog_digest(),
            "witness_summary_artifact_digest": (
                visual_witness_summary_artifact_digest()
            ),
        }
    )


def visual_soft_witness_interface_digest() -> str:
    """Backward-compatible name for the joint soft-witness interface digest."""

    return visual_joint_soft_witness_interface_digest()


def _scenario_prefix(scenario_id: str) -> str:
    return f"scenario:{scenario_id}"


def _hole_witness_id(scenario_id: str, hole_id: str) -> str:
    return f"{_scenario_prefix(scenario_id)}:hole:{hole_id}"


def _contour_witness_id(scenario_id: str, contour_id: str) -> str:
    return f"{_scenario_prefix(scenario_id)}:contour:{contour_id}"


def _bbox_description(bbox: Q16BBox) -> str:
    return (
        f"bbox_q16 x=[{bbox.x0},{bbox.x1}), y=[{bbox.y0},{bbox.y1}) "
        "on normalized 0..65535 axes"
    )


def _interval_text(value: CountInterval) -> str:
    return str(value.lower) if value.exact else f"[{value.lower},{value.upper}]"


def _sum_intervals(values: tuple[CountInterval, ...]) -> CountInterval:
    return CountInterval(
        sum(value.lower for value in values),
        sum(value.upper for value in values),
    )


def visual_witness_summaries(
    bundle: VisualWitnessBundle,
    *,
    expected_png_bytes: bytes | None = None,
) -> VisualWitnessSummaries:
    """Summarize one verified bundle for the blind one-panel soft scorer.

    Passing ``expected_png_bytes`` additionally replays extraction from the
    exact panel bytes.  The argument is deliberately byte-only: experiment
    metadata has no place in the summary interface.
    """

    verified = verify_visual_witness_bundle(
        bundle,
        expected_png_bytes=expected_png_bytes,
    )
    base_packet = verified.base_packet
    contour_packet = verified.contour_packet
    summaries: list[VisualWitnessSummary] = [
        (
            "panel:geometry",
            "Panel raster geometry: "
            f"width_pixels={base_packet.width_pixels}; "
            f"height_pixels={base_packet.height_pixels}; "
            f"area_pixels={base_packet.width_pixels * base_packet.height_pixels}; "
            "bbox_q16 values use half-open normalized 0..65535 axes.",
        )
    ]

    detail_summaries: list[VisualWitnessSummary] = []
    for scenario, contour_scenario in zip(
        base_packet.scenarios, contour_packet.scenarios, strict=True
    ):
        prefix = _scenario_prefix(scenario.scenario_id)
        owned_holes = tuple(
            hole for hole in scenario.holes if hole.owner_component_id is not None
        )
        summaries.append(
            (
                f"{prefix}:counts",
                f"Scenario {scenario.scenario_id}: "
                f"foreground_strength_threshold={scenario.foreground_strength_threshold}; "
                f"morphology={scenario.morphology}; "
                f"component_count={len(scenario.components)}; "
                f"hole_count={len(scenario.holes)}; "
                f"owned_hole_count={len(owned_holes)}.",
            )
        )

        endpoints = _sum_intervals(
            tuple(item.endpoint_count for item in contour_scenario.contours)
        )
        branchpoints = _sum_intervals(
            tuple(item.branchpoint_count for item in contour_scenario.contours)
        )
        cycles = _sum_intervals(
            tuple(item.cycle_count for item in contour_scenario.contours)
        )
        crossings = _sum_intervals(
            tuple(item.crossing_count for item in contour_scenario.contours)
        )
        reversals = _sum_intervals(
            tuple(
                item.curvature.reversal_count for item in contour_scenario.contours
            )
        )
        runs = _sum_intervals(
            tuple(item.curvature.run_count for item in contour_scenario.contours)
        )
        definite_s = sum(
            item.curvature.curve_class == "s-like"
            for item in contour_scenario.contours
        )
        definite_u = sum(
            item.curvature.curve_class == "u-like"
            for item in contour_scenario.contours
        )
        uncertain_curves = sum(
            item.curvature.curve_class == "indeterminate"
            for item in contour_scenario.contours
        )
        summaries.append(
            (
                f"{prefix}:topology-counts",
                f"Scenario {scenario.scenario_id} contour aggregates: "
                f"contour_count={len(contour_scenario.contours)}; "
                f"endpoint_count={_interval_text(endpoints)}; "
                f"branchpoint_count={_interval_text(branchpoints)}; "
                f"cycle_count={_interval_text(cycles)}; "
                f"x_crossing_count={_interval_text(crossings)}; "
                f"signed_curvature_reversal_count={_interval_text(reversals)}; "
                f"signed_curvature_run_count={_interval_text(runs)}; "
                f"s_like_count={_interval_text(CountInterval(definite_s, definite_s + uncertain_curves))}; "
                f"u_like_count={_interval_text(CountInterval(definite_u, definite_u + uncertain_curves))}.",
            )
        )

        owned_counts = {component.component_id: 0 for component in scenario.components}
        for hole in owned_holes:
            assert hole.owner_component_id is not None
            owned_counts[hole.owner_component_id] += 1

        for component, contour in zip(
            scenario.components, contour_scenario.contours, strict=True
        ):
            witness_id = _contour_witness_id(
                scenario.scenario_id, contour.contour_id
            )
            detail_summaries.append(
                (
                    witness_id,
                    "Contour topology and curvature: "
                    f"owner={component.component_id}; "
                    f"bbox_q16=[{component.bbox_q16.x0},{component.bbox_q16.x1})x"
                    f"[{component.bbox_q16.y0},{component.bbox_q16.y1}); "
                    f"area_px={component.area_pixels}; "
                    f"owned_holes={owned_counts[component.component_id]}; "
                    f"skeleton_px={contour.skeleton_pixel_count}; "
                    f"endpoints={_interval_text(contour.endpoint_count)}; "
                    f"branchpoints={_interval_text(contour.branchpoint_count)}; "
                    f"cycles={_interval_text(contour.cycle_count)}; "
                    f"x_crossings={_interval_text(contour.crossing_count)}; "
                    f"topology={contour.topology_disposition}/{contour.topology_reason}; "
                    f"curvature_reversals={_interval_text(contour.curvature.reversal_count)}; "
                    f"curvature_runs={_interval_text(contour.curvature.run_count)}; "
                    f"abs_turn_mrad={_interval_text(contour.curvature.absolute_turn_milliradians)}; "
                    f"net_turn_mrad={_interval_text(contour.curvature.net_turn_milliradians)}; "
                    f"curve_class={contour.curvature.curve_class}; "
                    f"curvature={contour.curvature.reason}.",
                )
            )

        for hole in scenario.holes:
            witness_id = _hole_witness_id(scenario.scenario_id, hole.hole_id)
            if hole.owner_component_id is None:
                ownership = "owner_component_id=none"
            else:
                ownership = "owner_component_id=" + hole.owner_component_id
            detail_summaries.append(
                (
                    witness_id,
                    f"Scenario {scenario.scenario_id} hole {hole.hole_id}: "
                    f"{_bbox_description(hole.bbox_q16)}; "
                    f"area_pixels={hole.area_pixels}; {ownership}.",
                )
            )

    fixed_count = len(summaries)
    detail_summaries.sort()
    if fixed_count + len(detail_summaries) > _MAX_WITNESS_SUMMARIES:
        detail_budget = _MAX_WITNESS_SUMMARIES - fixed_count - 1
        omitted = len(detail_summaries) - detail_budget
        summaries.append(
            (
                "panel:inventory-bounds",
                "Deterministic detail inventory bound applied: "
                f"total_detail_count={len(detail_summaries)}; "
                f"emitted_detail_count={detail_budget}; omitted_detail_count={omitted}; "
                "aggregate scenario counts above still cover every contour and hole.",
            )
        )
        detail_summaries = detail_summaries[:detail_budget]
    summaries.extend(detail_summaries)
    result = tuple(sorted(summaries))
    # Keep transport constraints executable at this boundary.  This rejects
    # overlarge inventories, illegal IDs, multiline text, metadata leaks, and
    # accidental duplicate IDs before a model-visible prompt can be formed.
    canonical = canonical_witness_summaries(result)
    normalized = tuple((item.witness_id, item.description) for item in canonical)
    if normalized != result:  # Defensive: canonical sequence input must not repair.
        raise ValueError("blind witness summary canonicalization changed the inventory")
    return result


__all__ = [
    "VISUAL_SOFT_WITNESS_INTERFACE_SCHEMA",
    "VISUAL_WITNESS_SUMMARY_ALGORITHM_ID",
    "VISUAL_WITNESS_SUMMARY_SCHEMA",
    "VisualWitnessSummaries",
    "VisualWitnessSummary",
    "visual_joint_soft_witness_interface_digest",
    "visual_soft_witness_interface_digest",
    "visual_witness_summaries",
    "visual_witness_summary_artifact_digest",
]
