"""Cold-replayable contour and stroke-topology witnesses for exact PNG bytes.

This module is deliberately separate from the direct visual catalog.  It is a
bounded P0 extractor, not a claim that raster skeletons solve general visual
topology.  In particular, ``topology.crossing_count`` means a geometrically
certified four-arm X-junction in the thinned foreground graph.  A four-way
attachment and an over/under crossing have the same raster topology; this
extractor cannot distinguish them and does not claim to.

All scalar predicates retain the three frozen visual preprocessing scenarios.
Unstable quantities are closed integer intervals.  A target inside a non-point
interval is indeterminate, never a negative observation.
"""

from __future__ import annotations

from bongard.runtime_source_snapshot import capture_loaded_source


_LOADED_SOURCE_SHA256 = capture_loaded_source(__name__, __file__)

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import re
from typing import Any, Mapping

import numpy as np
from scipy import ndimage

from bongard.canonical import canonical_digest, canonical_json
from bongard.legs.contracts import ValueType
from bongard import visual_witnesses as _base
from bongard.visual_witnesses import Q16BBox


CONTOUR_WITNESS_CAPABILITY_IDS = (
    "curvature.reversal_count",
    "curvature.run_count",
    "curvature.s_like_count",
    "curvature.u_like_count",
    "topology.branchpoint_count",
    "topology.crossing_count",
    "topology.cycle_count",
    "topology.endpoint_count",
)
CONTOUR_WITNESS_SCENARIO_IDS = _base.VISUAL_WITNESS_SCENARIO_IDS
CONTOUR_WITNESS_PACKET = ValueType("contour_witness_packet")
CONTOUR_WITNESS_EXTRACTOR_ID = "bongard.contour_witnesses"
CONTOUR_WITNESS_EXTRACTOR_VERSION = "1"

ALGORITHM_ID = "bongard.contour-witness-extractor/v1"
PACKET_SCHEMA = "gkm.bongard-contour-witness-packet.v1"
COUNT_RESULT_SCHEMA = "gkm.bongard-contour-count-result.v1"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_COMPONENT_ID = re.compile(r"component-[0-9]{8}\Z")
_CONTOUR_ID = re.compile(r"contour-[0-9]{8}\Z")
_ENDPOINT_ID = re.compile(r"endpoint-[0-9]{8}\Z")
_BRANCHPOINT_ID = re.compile(r"branchpoint-[0-9]{8}\Z")
_DISPOSITIONS = frozenset(("present", "certified_absent", "indeterminate"))
_CURVE_CLASSES = frozenset(("s-like", "u-like", "other", "indeterminate"))
_CURVATURE_REASONS = frozenset(
    ("simple_open_curve", "variant_disagreement", "not_simple_open_curve", "too_short")
)
_TOPOLOGY_REASONS = frozenset(
    ("stable_pixel_graph", "disconnected_skeleton", "graph_raster_cycle_disagreement")
)
_FOREGROUND_STRUCTURE = np.ones((3, 3), dtype=bool)
_BACKGROUND_STRUCTURE = np.asarray(
    ((False, True, False), (True, True, True), (False, True, False)), dtype=bool
)
_NEIGHBOR_OFFSETS = tuple(
    (dy, dx)
    for dy in (-1, 0, 1)
    for dx in (-1, 0, 1)
    if (dy, dx) != (0, 0)
)


def _exact_fields(
    data: Mapping[str, Any], expected: frozenset[str], label: str
) -> None:
    if not isinstance(data, Mapping) or set(data) != expected:
        raise ValueError(f"{label} fields differ from the static schema")


def _integer(value: object, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    if value < minimum:
        raise ValueError(f"{label} must be at least {minimum}")
    return value


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase sha256")
    return value


def _source_digest() -> str:
    current = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if current != _LOADED_SOURCE_SHA256:
        raise RuntimeError("contour witness source changed after import")
    return _LOADED_SOURCE_SHA256


def _artifact_digest(source_digest: str, base_extractor_digest: str) -> str:
    return canonical_digest(
        {
            "algorithm_id": ALGORITHM_ID,
            "source_digest": source_digest,
            "base_visual_extractor_digest": base_extractor_digest,
            "preprocessing_scenarios": list(CONTOUR_WITNESS_SCENARIO_IDS),
            "thinning": "zhang-suen/vectorized/until-fixed-point",
            "graph": {
                "foreground_connectivity": 8,
                "suppress_redundant_corner_diagonals": True,
                "branchpoints": "connected clusters of graph-degree >= 3",
                "cycles": "closed-background count cross-checked with graph Betti-1",
                "crossings": "four arms with two opposed pairs; otherwise interval",
            },
            "curvature": {
                "domain": "simple open skeleton paths only",
                "arc_length_samples": 128,
                "variant_grid": [
                    {"sigma": 3.25, "step": 2, "deadband": 0.012, "persistence": 3},
                    {"sigma": 4.0, "step": 2, "deadband": 0.018, "persistence": 3},
                    {"sigma": 4.75, "step": 3, "deadband": 0.024, "persistence": 4},
                ],
                "minimum_signed_run_mass_radians": 0.54,
                "u_like_min_absolute_turn_milliradians": 1200,
                "u_like_min_net_to_absolute_turn_ratio": 0.75,
                "certified_reversal_min_cancellation_milliradians": 300,
                "orientation": "counts invariant to traversal reversal; no canvas axis",
            },
        }
    )


def contour_witness_extractor_digest() -> str:
    """Return the source- and dependency-bound extractor identity."""

    return _artifact_digest(_source_digest(), _base.visual_witness_extractor_digest())


def contour_witness_catalog_digest() -> str:
    """Return the finite, non-prose capability inventory identity."""

    return canonical_digest(
        {
            "schema": "gkm.bongard-contour-witness-catalog.v1",
            "extractor_id": CONTOUR_WITNESS_EXTRACTOR_ID,
            "extractor_version": CONTOUR_WITNESS_EXTRACTOR_VERSION,
            "packet_type": CONTOUR_WITNESS_PACKET.to_data(),
            "capability_ids": list(CONTOUR_WITNESS_CAPABILITY_IDS),
            "scenario_ids": list(CONTOUR_WITNESS_SCENARIO_IDS),
        }
    )


@dataclass(frozen=True, order=True)
class CountInterval:
    """Closed integer interval; a non-point interval is explicit uncertainty."""

    lower: int
    upper: int

    def __post_init__(self) -> None:
        _integer(self.lower, "count lower")
        _integer(self.upper, "count upper")
        if self.lower > self.upper:
            raise ValueError("count interval lower exceeds upper")

    @property
    def exact(self) -> bool:
        return self.lower == self.upper

    @classmethod
    def point(cls, value: int) -> "CountInterval":
        checked = _integer(value, "count")
        return cls(checked, checked)

    def to_data(self) -> dict[str, int]:
        return {"lower": self.lower, "upper": self.upper}

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "CountInterval":
        _exact_fields(data, frozenset(("lower", "upper")), "count interval")
        return cls(data["lower"], data["upper"])


@dataclass(frozen=True, order=True)
class Q16Point:
    x: int
    y: int

    def __post_init__(self) -> None:
        for name, value in (("x", self.x), ("y", self.y)):
            _integer(value, name)
            if value > 65535:
                raise ValueError(f"{name} exceeds unsigned Q16 range")

    def to_data(self) -> dict[str, int]:
        return {"x": self.x, "y": self.y}

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "Q16Point":
        _exact_fields(data, frozenset(("x", "y")), "Q16 point")
        return cls(data["x"], data["y"])


@dataclass(frozen=True)
class TopologyNodeWitness:
    node_id: str
    location_q16: Q16Point
    incident_arm_count: CountInterval

    def __post_init__(self) -> None:
        if not isinstance(self.node_id, str) or (
            _ENDPOINT_ID.fullmatch(self.node_id) is None
            and _BRANCHPOINT_ID.fullmatch(self.node_id) is None
        ):
            raise ValueError("topology node_id is not canonical")
        if not isinstance(self.location_q16, Q16Point):
            raise TypeError("topology node location_q16 must be a Q16Point")
        if not isinstance(self.incident_arm_count, CountInterval):
            raise TypeError("incident_arm_count must be a CountInterval")
        if _ENDPOINT_ID.fullmatch(self.node_id) is not None and (
            self.incident_arm_count != CountInterval.point(1)
        ):
            raise ValueError("endpoint must have exactly one incident arm")
        if _BRANCHPOINT_ID.fullmatch(self.node_id) is not None and (
            self.incident_arm_count.lower < 3
        ):
            raise ValueError("branchpoint must have at least three incident arms")

    def to_data(self) -> dict[str, object]:
        return {
            "node_id": self.node_id,
            "location_q16": self.location_q16.to_data(),
            "incident_arm_count": self.incident_arm_count.to_data(),
        }

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "TopologyNodeWitness":
        _exact_fields(
            data,
            frozenset(("node_id", "location_q16", "incident_arm_count")),
            "topology node",
        )
        location = data["location_q16"]
        arms = data["incident_arm_count"]
        if not isinstance(location, Mapping) or not isinstance(arms, Mapping):
            raise TypeError("topology node location and arm count must be objects")
        return cls(
            data["node_id"], Q16Point.from_data(location), CountInterval.from_data(arms)
        )


@dataclass(frozen=True)
class CurvatureWitness:
    reversal_count: CountInterval
    run_count: CountInterval
    absolute_turn_milliradians: CountInterval
    net_turn_milliradians: CountInterval
    curve_class: str
    sample_count: int
    reason: str

    def __post_init__(self) -> None:
        for name, value in (
            ("reversal_count", self.reversal_count),
            ("run_count", self.run_count),
            ("absolute_turn_milliradians", self.absolute_turn_milliradians),
            ("net_turn_milliradians", self.net_turn_milliradians),
        ):
            if not isinstance(value, CountInterval):
                raise TypeError(f"{name} must be a CountInterval")
        if self.curve_class not in _CURVE_CLASSES:
            raise ValueError("unknown curve_class")
        _integer(self.sample_count, "sample_count")
        if self.reason not in _CURVATURE_REASONS:
            raise ValueError("unknown curvature reason")
        if self.reason in {"not_simple_open_curve", "too_short"} and (
            self.curve_class != "indeterminate" or self.sample_count != 0
        ):
            raise ValueError("unavailable curvature must be indeterminate and unsampled")
        if self.curve_class == "s-like" and self.reversal_count.lower < 1:
            raise ValueError("s-like witness requires a certified reversal")
        if self.curve_class == "u-like" and (
            self.reversal_count.upper != 0
            or self.absolute_turn_milliradians.lower < 1200
            or 4 * self.net_turn_milliradians.lower
            < 3 * self.absolute_turn_milliradians.upper
        ):
            raise ValueError("u-like witness requires monotone substantive turning")

    def to_data(self) -> dict[str, object]:
        return {
            "reversal_count": self.reversal_count.to_data(),
            "run_count": self.run_count.to_data(),
            "absolute_turn_milliradians": self.absolute_turn_milliradians.to_data(),
            "net_turn_milliradians": self.net_turn_milliradians.to_data(),
            "curve_class": self.curve_class,
            "sample_count": self.sample_count,
            "reason": self.reason,
        }

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "CurvatureWitness":
        _exact_fields(
            data,
            frozenset(
                (
                    "reversal_count",
                    "run_count",
                    "absolute_turn_milliradians",
                    "net_turn_milliradians",
                    "curve_class",
                    "sample_count",
                    "reason",
                )
            ),
            "curvature witness",
        )
        intervals = (
            data["reversal_count"],
            data["run_count"],
            data["absolute_turn_milliradians"],
            data["net_turn_milliradians"],
        )
        if any(not isinstance(item, Mapping) for item in intervals):
            raise TypeError("curvature intervals must be objects")
        return cls(
            reversal_count=CountInterval.from_data(intervals[0]),
            run_count=CountInterval.from_data(intervals[1]),
            absolute_turn_milliradians=CountInterval.from_data(intervals[2]),
            net_turn_milliradians=CountInterval.from_data(intervals[3]),
            curve_class=data["curve_class"],
            sample_count=data["sample_count"],
            reason=data["reason"],
        )


@dataclass(frozen=True)
class ContourStrokeWitness:
    contour_id: str
    owner_component_id: str
    bbox_q16: Q16BBox
    source_mask_digest: str
    skeleton_digest: str
    skeleton_pixel_count: int
    endpoints: tuple[TopologyNodeWitness, ...]
    branchpoints: tuple[TopologyNodeWitness, ...]
    crossing_branchpoint_ids: tuple[str, ...]
    endpoint_count: CountInterval
    branchpoint_count: CountInterval
    cycle_count: CountInterval
    crossing_count: CountInterval
    topology_disposition: str
    topology_reason: str
    curvature: CurvatureWitness

    def __post_init__(self) -> None:
        if not isinstance(self.contour_id, str) or _CONTOUR_ID.fullmatch(
            self.contour_id
        ) is None:
            raise ValueError("contour_id is not canonical")
        if not isinstance(self.owner_component_id, str) or _COMPONENT_ID.fullmatch(
            self.owner_component_id
        ) is None:
            raise ValueError("owner_component_id is not canonical")
        if not isinstance(self.bbox_q16, Q16BBox):
            raise TypeError("bbox_q16 must be a Q16BBox")
        _digest(self.source_mask_digest, "source_mask_digest")
        _digest(self.skeleton_digest, "skeleton_digest")
        _integer(self.skeleton_pixel_count, "skeleton_pixel_count", minimum=1)
        if not isinstance(self.endpoints, tuple) or any(
            not isinstance(item, TopologyNodeWitness) for item in self.endpoints
        ):
            raise TypeError("endpoints must be a typed tuple")
        if not isinstance(self.branchpoints, tuple) or any(
            not isinstance(item, TopologyNodeWitness) for item in self.branchpoints
        ):
            raise TypeError("branchpoints must be a typed tuple")
        if tuple(item.node_id for item in self.endpoints) != tuple(
            f"endpoint-{index:08d}" for index in range(len(self.endpoints))
        ):
            raise ValueError("endpoint IDs must be consecutive and ordered")
        branch_ids = tuple(item.node_id for item in self.branchpoints)
        if branch_ids != tuple(
            f"branchpoint-{index:08d}" for index in range(len(self.branchpoints))
        ):
            raise ValueError("branchpoint IDs must be consecutive and ordered")
        if (
            not isinstance(self.crossing_branchpoint_ids, tuple)
            or tuple(sorted(self.crossing_branchpoint_ids))
            != self.crossing_branchpoint_ids
            or len(set(self.crossing_branchpoint_ids))
            != len(self.crossing_branchpoint_ids)
            or any(item not in set(branch_ids) for item in self.crossing_branchpoint_ids)
        ):
            raise ValueError("crossing branchpoint IDs must be sorted branch references")
        for name, value in (
            ("endpoint_count", self.endpoint_count),
            ("branchpoint_count", self.branchpoint_count),
            ("cycle_count", self.cycle_count),
            ("crossing_count", self.crossing_count),
        ):
            if not isinstance(value, CountInterval):
                raise TypeError(f"{name} must be a CountInterval")
        if self.endpoint_count != CountInterval.point(len(self.endpoints)):
            raise ValueError("endpoint_count disagrees with endpoint witnesses")
        if self.branchpoint_count != CountInterval.point(len(self.branchpoints)):
            raise ValueError("branchpoint_count disagrees with branchpoint witnesses")
        if self.crossing_count.lower != len(self.crossing_branchpoint_ids):
            raise ValueError("crossing lower bound disagrees with certified crossings")
        if self.crossing_count.upper > len(self.branchpoints):
            raise ValueError("crossing upper bound exceeds branchpoint count")
        if self.topology_disposition not in {"determinate", "indeterminate"}:
            raise ValueError("unknown topology disposition")
        if self.topology_reason not in _TOPOLOGY_REASONS:
            raise ValueError("unknown topology reason")
        if self.topology_disposition == "determinate" and (
            not self.cycle_count.exact or not self.crossing_count.exact
        ):
            raise ValueError("determinate topology requires point intervals")
        if not isinstance(self.curvature, CurvatureWitness):
            raise TypeError("curvature must be a CurvatureWitness")

    def to_data(self) -> dict[str, object]:
        return {
            "contour_id": self.contour_id,
            "owner_component_id": self.owner_component_id,
            "bbox_q16": self.bbox_q16.to_data(),
            "source_mask_digest": self.source_mask_digest,
            "skeleton_digest": self.skeleton_digest,
            "skeleton_pixel_count": self.skeleton_pixel_count,
            "endpoints": [item.to_data() for item in self.endpoints],
            "branchpoints": [item.to_data() for item in self.branchpoints],
            "crossing_branchpoint_ids": list(self.crossing_branchpoint_ids),
            "endpoint_count": self.endpoint_count.to_data(),
            "branchpoint_count": self.branchpoint_count.to_data(),
            "cycle_count": self.cycle_count.to_data(),
            "crossing_count": self.crossing_count.to_data(),
            "topology_disposition": self.topology_disposition,
            "topology_reason": self.topology_reason,
            "curvature": self.curvature.to_data(),
        }

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "ContourStrokeWitness":
        fields = frozenset(
            (
                "contour_id",
                "owner_component_id",
                "bbox_q16",
                "source_mask_digest",
                "skeleton_digest",
                "skeleton_pixel_count",
                "endpoints",
                "branchpoints",
                "crossing_branchpoint_ids",
                "endpoint_count",
                "branchpoint_count",
                "cycle_count",
                "crossing_count",
                "topology_disposition",
                "topology_reason",
                "curvature",
            )
        )
        _exact_fields(data, fields, "contour stroke witness")
        endpoints = data["endpoints"]
        branchpoints = data["branchpoints"]
        crossing_ids = data["crossing_branchpoint_ids"]
        if not isinstance(endpoints, list) or not isinstance(branchpoints, list):
            raise TypeError("topology node collections must be JSON lists")
        if any(not isinstance(item, Mapping) for item in endpoints + branchpoints):
            raise TypeError("topology nodes must be JSON objects")
        if not isinstance(crossing_ids, list) or any(
            not isinstance(item, str) for item in crossing_ids
        ):
            raise TypeError("crossing branchpoint IDs must be a string list")
        bbox = data["bbox_q16"]
        curvature = data["curvature"]
        interval_names = (
            "endpoint_count",
            "branchpoint_count",
            "cycle_count",
            "crossing_count",
        )
        if (
            not isinstance(bbox, Mapping)
            or not isinstance(curvature, Mapping)
            or any(not isinstance(data[name], Mapping) for name in interval_names)
        ):
            raise TypeError("contour nested DTO values must be objects")
        return cls(
            contour_id=data["contour_id"],
            owner_component_id=data["owner_component_id"],
            bbox_q16=Q16BBox.from_data(bbox),
            source_mask_digest=data["source_mask_digest"],
            skeleton_digest=data["skeleton_digest"],
            skeleton_pixel_count=data["skeleton_pixel_count"],
            endpoints=tuple(TopologyNodeWitness.from_data(item) for item in endpoints),
            branchpoints=tuple(
                TopologyNodeWitness.from_data(item) for item in branchpoints
            ),
            crossing_branchpoint_ids=tuple(crossing_ids),
            endpoint_count=CountInterval.from_data(data["endpoint_count"]),
            branchpoint_count=CountInterval.from_data(data["branchpoint_count"]),
            cycle_count=CountInterval.from_data(data["cycle_count"]),
            crossing_count=CountInterval.from_data(data["crossing_count"]),
            topology_disposition=data["topology_disposition"],
            topology_reason=data["topology_reason"],
            curvature=CurvatureWitness.from_data(curvature),
        )


@dataclass(frozen=True)
class ContourScenarioWitness:
    scenario_id: str
    foreground_strength_threshold: int
    morphology: str
    contours: tuple[ContourStrokeWitness, ...]

    def __post_init__(self) -> None:
        expected = {item[0]: item[1:] for item in _base._SCENARIOS}
        if self.scenario_id not in expected:
            raise ValueError("unknown contour scenario_id")
        if (self.foreground_strength_threshold, self.morphology) != expected[
            self.scenario_id
        ]:
            raise ValueError("scenario parameters do not match the frozen scenario ID")
        if not isinstance(self.contours, tuple) or any(
            not isinstance(item, ContourStrokeWitness) for item in self.contours
        ):
            raise TypeError("contours must be a typed tuple")
        if tuple(item.contour_id for item in self.contours) != tuple(
            f"contour-{index:08d}" for index in range(len(self.contours))
        ):
            raise ValueError("contour IDs must be consecutive and ordered")
        if tuple(item.owner_component_id for item in self.contours) != tuple(
            f"component-{index:08d}" for index in range(len(self.contours))
        ):
            raise ValueError("contour owners must be consecutive and ordered")

    def to_data(self) -> dict[str, object]:
        return {
            "scenario_id": self.scenario_id,
            "foreground_strength_threshold": self.foreground_strength_threshold,
            "morphology": self.morphology,
            "contours": [item.to_data() for item in self.contours],
        }

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "ContourScenarioWitness":
        _exact_fields(
            data,
            frozenset(
                (
                    "scenario_id",
                    "foreground_strength_threshold",
                    "morphology",
                    "contours",
                )
            ),
            "contour scenario witness",
        )
        contours = data["contours"]
        if not isinstance(contours, list) or any(
            not isinstance(item, Mapping) for item in contours
        ):
            raise TypeError("scenario contours must be a JSON object list")
        return cls(
            scenario_id=data["scenario_id"],
            foreground_strength_threshold=data["foreground_strength_threshold"],
            morphology=data["morphology"],
            contours=tuple(ContourStrokeWitness.from_data(item) for item in contours),
        )


@dataclass(frozen=True)
class ContourWitnessPacket:
    panel_digest: str
    width_pixels: int
    height_pixels: int
    extractor_source_digest: str
    base_visual_extractor_digest: str
    extractor_artifact_digest: str
    scenarios: tuple[ContourScenarioWitness, ...]

    def __post_init__(self) -> None:
        _digest(self.panel_digest, "panel_digest")
        _integer(self.width_pixels, "width_pixels", minimum=2)
        _integer(self.height_pixels, "height_pixels", minimum=2)
        source = _digest(self.extractor_source_digest, "extractor_source_digest")
        dependency = _digest(
            self.base_visual_extractor_digest, "base_visual_extractor_digest"
        )
        _digest(self.extractor_artifact_digest, "extractor_artifact_digest")
        if self.extractor_artifact_digest != _artifact_digest(source, dependency):
            raise ValueError("extractor artifact digest does not bind source/dependency")
        if not isinstance(self.scenarios, tuple) or any(
            not isinstance(item, ContourScenarioWitness) for item in self.scenarios
        ):
            raise TypeError("packet scenarios must be a typed tuple")
        if tuple(item.scenario_id for item in self.scenarios) != (
            CONTOUR_WITNESS_SCENARIO_IDS
        ):
            raise ValueError("packet must retain all scenarios in canonical order")

    def to_data(self) -> dict[str, object]:
        return {
            "schema": PACKET_SCHEMA,
            "algorithm_id": ALGORITHM_ID,
            "panel_digest": self.panel_digest,
            "width_pixels": self.width_pixels,
            "height_pixels": self.height_pixels,
            "extractor_source_digest": self.extractor_source_digest,
            "base_visual_extractor_digest": self.base_visual_extractor_digest,
            "extractor_artifact_digest": self.extractor_artifact_digest,
            "scenarios": [item.to_data() for item in self.scenarios],
        }

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "ContourWitnessPacket":
        _exact_fields(
            data,
            frozenset(
                (
                    "schema",
                    "algorithm_id",
                    "panel_digest",
                    "width_pixels",
                    "height_pixels",
                    "extractor_source_digest",
                    "base_visual_extractor_digest",
                    "extractor_artifact_digest",
                    "scenarios",
                )
            ),
            "contour witness packet",
        )
        if data["schema"] != PACKET_SCHEMA or data["algorithm_id"] != ALGORITHM_ID:
            raise ValueError("unsupported contour witness packet")
        scenarios = data["scenarios"]
        if not isinstance(scenarios, list) or any(
            not isinstance(item, Mapping) for item in scenarios
        ):
            raise TypeError("packet scenarios must be a JSON object list")
        return cls(
            panel_digest=data["panel_digest"],
            width_pixels=data["width_pixels"],
            height_pixels=data["height_pixels"],
            extractor_source_digest=data["extractor_source_digest"],
            base_visual_extractor_digest=data["base_visual_extractor_digest"],
            extractor_artifact_digest=data["extractor_artifact_digest"],
            scenarios=tuple(ContourScenarioWitness.from_data(item) for item in scenarios),
        )

    def digest(self) -> str:
        return canonical_digest(self.to_data())


@dataclass(frozen=True)
class ScenarioContourCountObservation:
    scenario_id: str
    observed: CountInterval
    disposition: str

    def __post_init__(self) -> None:
        if self.scenario_id not in CONTOUR_WITNESS_SCENARIO_IDS:
            raise ValueError("unknown count-observation scenario_id")
        if not isinstance(self.observed, CountInterval):
            raise TypeError("observed must be a CountInterval")
        if self.disposition not in _DISPOSITIONS:
            raise ValueError("unknown count-observation disposition")

    def to_data(self) -> dict[str, object]:
        return {
            "scenario_id": self.scenario_id,
            "observed": self.observed.to_data(),
            "disposition": self.disposition,
        }

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "ScenarioContourCountObservation":
        _exact_fields(
            data,
            frozenset(("scenario_id", "observed", "disposition")),
            "scenario contour count observation",
        )
        observed = data["observed"]
        if not isinstance(observed, Mapping):
            raise TypeError("observed count must be an object")
        return cls(
            data["scenario_id"],
            CountInterval.from_data(observed),
            data["disposition"],
        )


@dataclass(frozen=True)
class ContourCountResult:
    capability_id: str
    expected_count: int
    packet_digest: str
    observations: tuple[ScenarioContourCountObservation, ...]

    def __post_init__(self) -> None:
        if self.capability_id not in CONTOUR_WITNESS_CAPABILITY_IDS:
            raise ValueError("unknown contour capability_id")
        _integer(self.expected_count, "expected_count")
        _digest(self.packet_digest, "packet_digest")
        if not isinstance(self.observations, tuple) or any(
            not isinstance(item, ScenarioContourCountObservation)
            for item in self.observations
        ):
            raise TypeError("observations must be a typed tuple")
        if tuple(item.scenario_id for item in self.observations) != (
            CONTOUR_WITNESS_SCENARIO_IDS
        ):
            raise ValueError("observations must retain canonical scenarios")
        for item in self.observations:
            expected_disposition = (
                "present"
                if item.observed.exact and item.observed.lower == self.expected_count
                else "certified_absent"
                if self.expected_count < item.observed.lower
                or self.expected_count > item.observed.upper
                else "indeterminate"
            )
            if item.disposition != expected_disposition:
                raise ValueError("count disposition disagrees with its interval")

    def to_data(self) -> dict[str, object]:
        return {
            "schema": COUNT_RESULT_SCHEMA,
            "capability_id": self.capability_id,
            "expected_count": self.expected_count,
            "packet_digest": self.packet_digest,
            "observations": [item.to_data() for item in self.observations],
        }

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "ContourCountResult":
        _exact_fields(
            data,
            frozenset(
                ("schema", "capability_id", "expected_count", "packet_digest", "observations")
            ),
            "contour count result",
        )
        if data["schema"] != COUNT_RESULT_SCHEMA:
            raise ValueError("unsupported contour count result")
        observations = data["observations"]
        if not isinstance(observations, list) or any(
            not isinstance(item, Mapping) for item in observations
        ):
            raise TypeError("count observations must be a JSON object list")
        return cls(
            capability_id=data["capability_id"],
            expected_count=data["expected_count"],
            packet_digest=data["packet_digest"],
            observations=tuple(
                ScenarioContourCountObservation.from_data(item) for item in observations
            ),
        )

    def digest(self) -> str:
        return canonical_digest(self.to_data())


def _mask_digest(mask: np.ndarray, kind: str) -> str:
    height, width = mask.shape
    prefix = canonical_json(
        {
            "schema": "gkm.bongard-contour-binary-mask.v1",
            "kind": kind,
            "height_pixels": height,
            "width_pixels": width,
            "packing": "numpy.packbits-axis-none-bitorder-big",
        }
    )
    packed = np.packbits(mask.reshape(-1), bitorder="big").tobytes()
    return hashlib.sha256(prefix + b"\x00" + packed).hexdigest()


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _q16(value: int, extent: int) -> int:
    return (value * 65535 + extent // 2) // extent


def _q16_bbox(
    pixel_bbox: tuple[int, int, int, int], width: int, height: int
) -> Q16BBox:
    x0, y0, x1, y1 = pixel_bbox
    return Q16BBox(_q16(x0, width), _q16(y0, height), _q16(x1, width), _q16(y1, height))


def _q16_cluster_point(
    cluster: set[tuple[int, int]], width: int, height: int
) -> Q16Point:
    count = len(cluster)
    x_numerator = sum(2 * x + 1 for y, x in cluster) * 65535
    y_numerator = sum(2 * y + 1 for y, x in cluster) * 65535
    return Q16Point(
        (x_numerator + count * width) // (2 * count * width),
        (y_numerator + count * height) // (2 * count * height),
    )


def _zhang_suen(mask: np.ndarray) -> np.ndarray:
    """Topology-preserving thinning without an optional image dependency."""

    current = np.ascontiguousarray(mask, dtype=bool)
    if not current.any():
        return current
    # Cropping makes the worst-case iteration bound about stroke thickness,
    # while the one-pixel zero frame makes border rules explicit.
    ys, xs = np.nonzero(current)
    cropped = current[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    work = np.pad(cropped, 1, constant_values=False)
    iteration_limit = max(work.shape)
    for _ in range(iteration_limit):
        changed = False
        for phase in (0, 1):
            p = np.pad(work, 1, constant_values=False)
            p2 = p[:-2, 1:-1]
            p3 = p[:-2, 2:]
            p4 = p[1:-1, 2:]
            p5 = p[2:, 2:]
            p6 = p[2:, 1:-1]
            p7 = p[2:, :-2]
            p8 = p[1:-1, :-2]
            p9 = p[:-2, :-2]
            neighbours = (p2, p3, p4, p5, p6, p7, p8, p9)
            count = sum(item.astype(np.uint8) for item in neighbours)
            transitions = sum(
                ((~neighbours[index]) & neighbours[(index + 1) % 8]).astype(np.uint8)
                for index in range(8)
            )
            if phase == 0:
                gate_a = ~(p2 & p4 & p6)
                gate_b = ~(p4 & p6 & p8)
            else:
                gate_a = ~(p2 & p4 & p8)
                gate_b = ~(p2 & p6 & p8)
            delete = work & (count >= 2) & (count <= 6) & (transitions == 1) & gate_a & gate_b
            if delete.any():
                work = work & ~delete
                changed = True
        if not changed:
            break
    else:  # pragma: no cover - a convergence guard, not a semantic fallback.
        raise RuntimeError("Zhang-Suen thinning exceeded its fixed dimension bound")
    result = np.zeros_like(current)
    result[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1] = work[1:-1, 1:-1]
    return np.ascontiguousarray(result, dtype=bool)


def _pixel_graph(mask: np.ndarray) -> dict[tuple[int, int], tuple[tuple[int, int], ...]]:
    coords = {(int(y), int(x)) for y, x in np.argwhere(mask)}
    graph: dict[tuple[int, int], tuple[tuple[int, int], ...]] = {}
    for y, x in coords:
        neighbours: list[tuple[int, int]] = []
        for dy, dx in _NEIGHBOR_OFFSETS:
            other = (y + dy, x + dx)
            if other not in coords:
                continue
            # In a 2x2 corner, the diagonal duplicates an orthogonal path and
            # creates a fake triangle/branch.  A lone diagonal remains an edge.
            if dy and dx and (
                (y, x + dx) in coords or (y + dy, x) in coords
            ):
                continue
            neighbours.append(other)
        graph[(y, x)] = tuple(sorted(neighbours))
    return graph


def _clusters(
    points: set[tuple[int, int]], *, chebyshev_radius: int = 1
) -> list[set[tuple[int, int]]]:
    remaining = set(points)
    clusters: list[set[tuple[int, int]]] = []
    while remaining:
        seed = min(remaining)
        remaining.remove(seed)
        cluster = {seed}
        stack = [seed]
        while stack:
            y, x = stack.pop()
            neighbours = {
                (y + dy, x + dx)
                for dy in range(-chebyshev_radius, chebyshev_radius + 1)
                for dx in range(-chebyshev_radius, chebyshev_radius + 1)
                if (dy, dx) != (0, 0)
            }
            found = remaining & neighbours
            remaining.difference_update(found)
            cluster.update(found)
            stack.extend(sorted(found))
        clusters.append(cluster)
    return clusters


def _graph_components(
    graph: Mapping[tuple[int, int], tuple[tuple[int, int], ...]]
) -> int:
    remaining = set(graph)
    count = 0
    while remaining:
        count += 1
        seed = min(remaining)
        remaining.remove(seed)
        stack = [seed]
        while stack:
            point = stack.pop()
            found = set(graph[point]) & remaining
            remaining.difference_update(found)
            stack.extend(found)
    return count


def _contracted_graph_cycle_count(
    graph: Mapping[tuple[int, int], tuple[tuple[int, int], ...]],
    branch_clusters: list[set[tuple[int, int]]],
) -> int:
    """First Betti number after contracting each raster junction cluster.

    A thick X can thin to a 2x2-ish junction with an internal microscopic
    cycle.  Contracting the semantic branch vertex removes that raster-only
    cycle without changing a macroscopic loop elsewhere in the graph.
    """

    owner: dict[tuple[int, int], tuple[str, int | tuple[int, int]]] = {}
    for index, cluster in enumerate(branch_clusters):
        for point in cluster:
            owner[point] = ("branch", index)
    for point in graph:
        owner.setdefault(point, ("pixel", point))
    vertices = set(owner.values())
    edges: set[
        tuple[tuple[str, int | tuple[int, int]], tuple[str, int | tuple[int, int]]]
    ] = set()
    for point, neighbours in graph.items():
        first = owner[point]
        for neighbour in neighbours:
            second = owner[neighbour]
            if first == second:
                continue
            edges.add(tuple(sorted((first, second), key=repr)))
    adjacency = {vertex: set() for vertex in vertices}
    for first, second in edges:
        adjacency[first].add(second)
        adjacency[second].add(first)
    components = 0
    remaining = set(vertices)
    while remaining:
        components += 1
        seed = min(remaining, key=repr)
        remaining.remove(seed)
        stack = [seed]
        while stack:
            found = adjacency[stack.pop()] & remaining
            remaining.difference_update(found)
            stack.extend(found)
    return max(0, len(edges) - len(vertices) + components)


def _background_cycle_count(mask: np.ndarray) -> int:
    padded = np.pad(mask, 1, constant_values=False)
    labels, count = ndimage.label(~padded, structure=_BACKGROUND_STRUCTURE)
    exterior = int(labels[0, 0])
    return sum(label != exterior for label in range(1, count + 1))


def _incident_arm_roots(
    graph: Mapping[tuple[int, int], tuple[tuple[int, int], ...]],
    cluster: set[tuple[int, int]],
) -> list[set[tuple[int, int]]]:
    roots = {
        neighbour
        for point in cluster
        for neighbour in graph[point]
        if neighbour not in cluster
    }
    # Use the same corner-suppressed graph, not raw Chebyshev adjacency.
    # Raw adjacency merges two distinct acute arms at a raster corner.
    remaining = set(roots)
    groups: list[set[tuple[int, int]]] = []
    while remaining:
        seed = min(remaining)
        remaining.remove(seed)
        group = {seed}
        stack = [seed]
        while stack:
            found = set(graph[stack.pop()]) & remaining
            remaining.difference_update(found)
            group.update(found)
            stack.extend(found)
        groups.append(group)
    return groups


def _arm_vector(
    graph: Mapping[tuple[int, int], tuple[tuple[int, int], ...]],
    junction: set[tuple[int, int]],
    roots: set[tuple[int, int]],
) -> np.ndarray | None:
    center = np.mean(np.asarray([(x, y) for y, x in junction], dtype=float), axis=0)
    visited = set(junction)
    frontier = sorted(roots)
    visited.update(frontier)
    distances = {point: 1 for point in frontier}
    farthest = frontier[0] if frontier else None
    while frontier:
        point = frontier.pop(0)
        if distances[point] >= 10:
            continue
        for neighbour in graph[point]:
            if neighbour in visited:
                continue
            visited.add(neighbour)
            distances[neighbour] = distances[point] + 1
            frontier.append(neighbour)
            if farthest is None or distances[neighbour] > distances[farthest] or (
                distances[neighbour] == distances[farthest] and neighbour < farthest
            ):
                farthest = neighbour
    if farthest is None or distances[farthest] < 4:
        return None
    vector = np.asarray((farthest[1], farthest[0]), dtype=float) - center
    length = float(np.linalg.norm(vector))
    return None if length <= 0 else vector / length


def _certified_x_junction(
    graph: Mapping[tuple[int, int], tuple[tuple[int, int], ...]],
    cluster: set[tuple[int, int]],
) -> tuple[bool, bool, int]:
    """Return (certified, ambiguous, arm_count)."""

    arms = _incident_arm_roots(graph, cluster)
    arm_count = len(arms)
    if arm_count != 4:
        # Three or fewer graph arms cannot contain an X crossing.  Five or
        # more may contain one plus an attachment, which this P0 witness
        # cannot decompose safely.
        return False, arm_count > 4, arm_count
    vectors = [_arm_vector(graph, cluster, arm) for arm in arms]
    if any(vector is None for vector in vectors):
        return False, True, arm_count
    directions = [vector for vector in vectors if vector is not None]
    angles = sorted(math.atan2(float(item[1]), float(item[0])) for item in directions)
    circular_gaps = [
        (angles[(index + 1) % 4] - angles[index]) % (2 * math.pi)
        for index in range(4)
    ]
    if min(circular_gaps) < math.radians(28):
        return False, True, arm_count
    pairings = (((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2)))
    best = min(
        max(float(directions[a] @ directions[b]), float(directions[c] @ directions[d]))
        for (a, b), (c, d) in pairings
    )
    return (True, False, arm_count) if best <= -math.cos(math.radians(32)) else (
        False,
        True,
        arm_count,
    )


def _ordered_simple_path(
    graph: Mapping[tuple[int, int], tuple[tuple[int, int], ...]],
    endpoints: list[set[tuple[int, int]]],
) -> np.ndarray | None:
    if len(endpoints) != 2 or any(len(cluster) != 1 for cluster in endpoints):
        return None
    start = min(next(iter(cluster)) for cluster in endpoints)
    path = [start]
    previous: tuple[int, int] | None = None
    current = start
    while True:
        options = [item for item in graph[current] if item != previous]
        if not options:
            break
        if len(options) != 1:
            return None
        nxt = options[0]
        if nxt in path:
            return None
        path.append(nxt)
        previous, current = current, nxt
    if len(path) != len(graph):
        return None
    return np.asarray([(x, y) for y, x in path], dtype=float)


def _resample_path(points: np.ndarray, count: int = 128) -> np.ndarray:
    segments = np.linalg.norm(np.diff(points, axis=0), axis=1)
    keep = np.concatenate(([True], segments > np.finfo(float).tiny))
    points = points[keep]
    if len(points) < 2:
        return np.empty((0, 2), dtype=float)
    segments = np.linalg.norm(np.diff(points, axis=0), axis=1)
    arc = np.concatenate(([0.0], np.cumsum(segments)))
    if arc[-1] <= np.finfo(float).tiny:
        return np.empty((0, 2), dtype=float)
    positions = np.linspace(0.0, arc[-1], count)
    return np.stack(
        [np.interp(positions, arc, points[:, axis]) for axis in range(2)], axis=1
    )


def _turn_profile(points: np.ndarray, step: int) -> np.ndarray:
    result = np.zeros(len(points), dtype=float)
    for index in range(step, len(points) - step):
        first = points[index] - points[index - step]
        second = points[index + step] - points[index]
        n1 = float(np.linalg.norm(first))
        n2 = float(np.linalg.norm(second))
        if n1 <= np.finfo(float).tiny or n2 <= np.finfo(float).tiny:
            continue
        cross = first[0] * second[1] - first[1] * second[0]
        result[index] = math.atan2(float(cross), float(first @ second))
    return result


def _curvature_variant(
    samples: np.ndarray, sigma: float, step: int, deadband: float, persistence: int
) -> tuple[int, int, int, int]:
    smooth = np.stack(
        [ndimage.gaussian_filter1d(samples[:, axis], sigma, mode="nearest") for axis in range(2)],
        axis=1,
    )
    turning = ndimage.gaussian_filter1d(
        _turn_profile(smooth, step), sigma, mode="nearest"
    )
    signs = np.where(turning > deadband, 1, np.where(turning < -deadband, -1, 0))
    raw_runs: list[list[float]] = []
    current = 0
    start = 0
    for index, value in enumerate(signs):
        value = int(value)
        if value == current:
            continue
        if current:
            raw_runs.append(
                [
                    float(current),
                    float(index - start),
                    float(np.sum(np.abs(turning[start:index]))),
                ]
            )
        current = value
        start = index
    if current:
        raw_runs.append(
            [
                float(current),
                float(len(signs) - start),
                float(np.sum(np.abs(turning[start:]))),
            ]
        )
    persistent = [item for item in raw_runs if item[1] >= persistence]
    merged: list[list[float]] = []
    for sign, length, mass in persistent:
        if merged and merged[-1][0] == sign:
            merged[-1][1] += length
            merged[-1][2] += mass
        else:
            merged.append([sign, length, mass])
    # Raster staircases can sustain a short opposite-sign run.  Length alone
    # therefore does not certify a curvature regime: require integrated
    # signed-turn mass, then merge equal signs again across discarded noise.
    massive = [item for item in merged if item[2] >= 0.54]
    merged = []
    for sign, length, mass in massive:
        if merged and merged[-1][0] == sign:
            merged[-1][1] += length
            merged[-1][2] += mass
        else:
            merged.append([sign, length, mass])
    runs = len(merged)
    reversals = max(0, runs - 1)

    tangents = np.diff(smooth, axis=0)
    angles = np.unwrap(np.arctan2(tangents[:, 1], tangents[:, 0]))
    angular_steps = np.diff(angles)
    absolute_turn = int(round(1000 * float(np.sum(np.abs(angular_steps)))))
    net_turn = int(round(1000 * abs(float(np.sum(angular_steps)))))
    return reversals, runs, absolute_turn, net_turn


def _unavailable_curvature(skeleton_pixels: int, reason: str) -> CurvatureWitness:
    upper = max(0, skeleton_pixels - 2)
    return CurvatureWitness(
        reversal_count=CountInterval(0, upper),
        run_count=CountInterval(0, max(1, upper + 1)),
        absolute_turn_milliradians=CountInterval(0, max(0, 3142 * upper)),
        net_turn_milliradians=CountInterval(0, max(0, 3142 * upper)),
        curve_class="indeterminate",
        sample_count=0,
        reason=reason,
    )


def _curvature_witness(path: np.ndarray | None, skeleton_pixels: int) -> CurvatureWitness:
    if path is None:
        return _unavailable_curvature(skeleton_pixels, "not_simple_open_curve")
    if len(path) < 8:
        return _unavailable_curvature(skeleton_pixels, "too_short")
    samples = _resample_path(path)
    if len(samples) != 128:
        return _unavailable_curvature(skeleton_pixels, "too_short")
    variants = (
        _curvature_variant(samples, 3.25, 2, 0.012, 3),
        _curvature_variant(samples, 4.0, 2, 0.018, 3),
        _curvature_variant(samples, 4.75, 3, 0.024, 4),
    )
    reversals = CountInterval(min(item[0] for item in variants), max(item[0] for item in variants))
    runs = CountInterval(min(item[1] for item in variants), max(item[1] for item in variants))
    absolute_turn = CountInterval(
        min(item[2] for item in variants), max(item[2] for item in variants)
    )
    net_turn = CountInterval(
        min(item[3] for item in variants), max(item[3] for item in variants)
    )
    # The triangle inequality makes this a traversal-invariant constructive
    # reversal witness: substantial |turn| - |net turn| requires both signs.
    if (
        absolute_turn.lower - net_turn.upper >= 300
        and 4 * net_turn.upper <= 3 * absolute_turn.lower
        and reversals.lower == 0
    ):
        reversals = CountInterval(1, max(1, reversals.upper))
        runs = CountInterval(max(2, runs.lower), max(2, runs.upper))
    if reversals.lower >= 1:
        curve_class = "s-like"
    elif (
        reversals.upper == 0
        and absolute_turn.lower >= 1200
        and 4 * net_turn.lower >= 3 * absolute_turn.upper
    ):
        curve_class = "u-like"
    elif reversals.exact and absolute_turn.upper < 700:
        curve_class = "other"
    else:
        curve_class = "indeterminate"
    reason = (
        "simple_open_curve"
        if reversals.exact and runs.exact
        else "variant_disagreement"
    )
    return CurvatureWitness(
        reversal_count=reversals,
        run_count=runs,
        absolute_turn_milliradians=absolute_turn,
        net_turn_milliradians=net_turn,
        curve_class=curve_class,
        sample_count=128,
        reason=reason,
    )


def _extract_contour(
    component_mask: np.ndarray,
    contour_index: int,
    width: int,
    height: int,
) -> ContourStrokeWitness:
    skeleton = _zhang_suen(component_mask)
    if not skeleton.any():  # Zhang-Suen preserves nonempty components.
        raise RuntimeError("thinning erased a nonempty connected component")
    graph = _pixel_graph(skeleton)
    endpoint_clusters = _clusters({point for point, edges in graph.items() if len(edges) == 1})
    branch_clusters = _clusters({point for point, edges in graph.items() if len(edges) >= 3})
    endpoint_clusters.sort(key=lambda item: tuple(sorted(item)))
    branch_clusters.sort(key=lambda item: tuple(sorted(item)))
    endpoints = tuple(
        TopologyNodeWitness(
            node_id=f"endpoint-{index:08d}",
            location_q16=_q16_cluster_point(cluster, width, height),
            incident_arm_count=CountInterval.point(1),
        )
        for index, cluster in enumerate(endpoint_clusters)
    )

    branch_records: list[tuple[TopologyNodeWitness, bool, bool]] = []
    for index, cluster in enumerate(branch_clusters):
        certified, ambiguous, arm_count = _certified_x_junction(graph, cluster)
        # Degree >= 3 establishes at least three arms even if a tight raster
        # cluster makes the root counter underestimate them.
        arm_count = max(3, arm_count)
        branch_records.append(
            (
                TopologyNodeWitness(
                    node_id=f"branchpoint-{index:08d}",
                    location_q16=_q16_cluster_point(cluster, width, height),
                    incident_arm_count=CountInterval.point(arm_count),
                ),
                certified,
                ambiguous,
            )
        )
    branchpoints = tuple(item[0] for item in branch_records)
    certified_ids = tuple(item[0].node_id for item in branch_records if item[1])
    ambiguous_crossings = sum(item[2] for item in branch_records)
    crossing_count = CountInterval(
        len(certified_ids), len(certified_ids) + ambiguous_crossings
    )

    graph_components = _graph_components(graph)
    graph_cycles = _contracted_graph_cycle_count(graph, branch_clusters)
    raster_cycles = _background_cycle_count(skeleton)
    if graph_components != 1:
        cycle_count = CountInterval(min(graph_cycles, raster_cycles), max(graph_cycles, raster_cycles))
        topology_disposition = "indeterminate"
        topology_reason = "disconnected_skeleton"
    elif graph_cycles != raster_cycles:
        cycle_count = CountInterval(min(graph_cycles, raster_cycles), max(graph_cycles, raster_cycles))
        topology_disposition = "indeterminate"
        topology_reason = "graph_raster_cycle_disagreement"
    else:
        cycle_count = CountInterval.point(graph_cycles)
        topology_disposition = "indeterminate" if not crossing_count.exact else "determinate"
        topology_reason = "stable_pixel_graph"

    path = None
    if (
        graph_components == 1
        and not branchpoints
        and len(endpoints) == 2
        and cycle_count == CountInterval.point(0)
    ):
        path = _ordered_simple_path(graph, endpoint_clusters)

    return ContourStrokeWitness(
        contour_id=f"contour-{contour_index:08d}",
        owner_component_id=f"component-{contour_index:08d}",
        bbox_q16=_q16_bbox(_bbox(component_mask), width, height),
        source_mask_digest=_mask_digest(component_mask, "scenario-component"),
        skeleton_digest=_mask_digest(skeleton, "zhang-suen-skeleton"),
        skeleton_pixel_count=len(graph),
        endpoints=endpoints,
        branchpoints=branchpoints,
        crossing_branchpoint_ids=certified_ids,
        endpoint_count=CountInterval.point(len(endpoints)),
        branchpoint_count=CountInterval.point(len(branchpoints)),
        cycle_count=cycle_count,
        crossing_count=crossing_count,
        topology_disposition=topology_disposition,
        topology_reason=topology_reason,
        curvature=_curvature_witness(path, len(graph)),
    )


def _extract_scenario(
    strength: np.ndarray, scenario_id: str, threshold: int, morphology: str
) -> ContourScenarioWitness:
    mask = _base._scenario_mask(strength, threshold, morphology)
    labels, count = ndimage.label(mask, structure=_FOREGROUND_STRUCTURE)
    raw: list[tuple[tuple[object, ...], np.ndarray]] = []
    for label in range(1, count + 1):
        component = labels == label
        bbox = _bbox(component)
        area = int(np.count_nonzero(component))
        digest = _mask_digest(component, "scenario-component")
        raw.append(((bbox[0], bbox[1], bbox[2], bbox[3], area, digest), component))
    raw.sort(key=lambda item: item[0])
    height, width = mask.shape
    return ContourScenarioWitness(
        scenario_id=scenario_id,
        foreground_strength_threshold=threshold,
        morphology=morphology,
        contours=tuple(
            _extract_contour(component, index, width, height)
            for index, (_key, component) in enumerate(raw)
        ),
    )


def extract_contour_witnesses(png_bytes: bytes) -> ContourWitnessPacket:
    """Extract topology/curvature alternatives from exact PNG bytes only."""

    strength = _base._decode_png(png_bytes)
    height, width = strength.shape
    source = _source_digest()
    dependency = _base.visual_witness_extractor_digest()
    return ContourWitnessPacket(
        panel_digest=hashlib.sha256(png_bytes).hexdigest(),
        width_pixels=width,
        height_pixels=height,
        extractor_source_digest=source,
        base_visual_extractor_digest=dependency,
        extractor_artifact_digest=_artifact_digest(source, dependency),
        scenarios=tuple(
            _extract_scenario(strength, scenario_id, threshold, morphology)
            for scenario_id, threshold, morphology in _base._SCENARIOS
        ),
    )


def verify_contour_witness_packet(
    packet: ContourWitnessPacket, expected_png_bytes: bytes | None = None
) -> ContourWitnessPacket:
    """Validate canonical DTOs and optionally cold-replay exact panel bytes."""

    if not isinstance(packet, ContourWitnessPacket):
        raise TypeError("packet must be a ContourWitnessPacket")
    if ContourWitnessPacket.from_data(packet.to_data()) != packet:
        raise ValueError("contour witness packet is not canonically represented")
    current_source = _source_digest()
    current_dependency = _base.visual_witness_extractor_digest()
    if (
        packet.extractor_source_digest != current_source
        or packet.base_visual_extractor_digest != current_dependency
        or packet.extractor_artifact_digest
        != _artifact_digest(current_source, current_dependency)
    ):
        raise ValueError("contour witness extractor source or dependency has drifted")
    if expected_png_bytes is not None:
        if not isinstance(expected_png_bytes, bytes):
            raise TypeError("expected_png_bytes must be exact bytes or null")
        replayed = extract_contour_witnesses(expected_png_bytes)
        if replayed != packet:
            raise ValueError("contour witness packet differs from exact PNG replay")
    return packet


def _sum_intervals(values: tuple[CountInterval, ...]) -> CountInterval:
    return CountInterval(sum(item.lower for item in values), sum(item.upper for item in values))


def _scenario_capability_interval(
    scenario: ContourScenarioWitness, capability_id: str
) -> CountInterval:
    contours = scenario.contours
    if capability_id == "topology.endpoint_count":
        return _sum_intervals(tuple(item.endpoint_count for item in contours))
    if capability_id == "topology.branchpoint_count":
        return _sum_intervals(tuple(item.branchpoint_count for item in contours))
    if capability_id == "topology.cycle_count":
        return _sum_intervals(tuple(item.cycle_count for item in contours))
    if capability_id == "topology.crossing_count":
        return _sum_intervals(tuple(item.crossing_count for item in contours))
    if capability_id == "curvature.reversal_count":
        return _sum_intervals(tuple(item.curvature.reversal_count for item in contours))
    if capability_id == "curvature.run_count":
        return _sum_intervals(tuple(item.curvature.run_count for item in contours))
    if capability_id in {"curvature.s_like_count", "curvature.u_like_count"}:
        target_class = "s-like" if capability_id.endswith("s_like_count") else "u-like"
        lower = sum(item.curvature.curve_class == target_class for item in contours)
        uncertain = sum(item.curvature.curve_class == "indeterminate" for item in contours)
        return CountInterval(lower, lower + uncertain)
    raise ValueError(f"unknown contour capability_id {capability_id!r}")


def evaluate_contour_count_by_scenario(
    packet: ContourWitnessPacket, capability_id: str, expected_count: int
) -> ContourCountResult:
    """Evaluate one exact-count claim without collapsing scenario uncertainty."""

    verify_contour_witness_packet(packet)
    if capability_id not in CONTOUR_WITNESS_CAPABILITY_IDS:
        raise ValueError(f"unknown contour capability_id {capability_id!r}")
    expected = _integer(expected_count, "expected_count")
    observations: list[ScenarioContourCountObservation] = []
    for scenario in packet.scenarios:
        interval = _scenario_capability_interval(scenario, capability_id)
        disposition = (
            "present"
            if interval.exact and interval.lower == expected
            else "certified_absent"
            if expected < interval.lower or expected > interval.upper
            else "indeterminate"
        )
        observations.append(
            ScenarioContourCountObservation(scenario.scenario_id, interval, disposition)
        )
    return ContourCountResult(
        capability_id=capability_id,
        expected_count=expected,
        packet_digest=packet.digest(),
        observations=tuple(observations),
    )


__all__ = [
    "CONTOUR_WITNESS_CAPABILITY_IDS",
    "CONTOUR_WITNESS_EXTRACTOR_ID",
    "CONTOUR_WITNESS_EXTRACTOR_VERSION",
    "CONTOUR_WITNESS_PACKET",
    "CONTOUR_WITNESS_SCENARIO_IDS",
    "ContourCountResult",
    "ContourScenarioWitness",
    "ContourStrokeWitness",
    "ContourWitnessPacket",
    "CountInterval",
    "CurvatureWitness",
    "Q16Point",
    "ScenarioContourCountObservation",
    "TopologyNodeWitness",
    "contour_witness_catalog_digest",
    "contour_witness_extractor_digest",
    "evaluate_contour_count_by_scenario",
    "extract_contour_witnesses",
    "verify_contour_witness_packet",
]
