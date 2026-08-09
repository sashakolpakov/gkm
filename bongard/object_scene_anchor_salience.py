"""Deterministic complete macro-anchor graphs for exact object masks.

The raw anchor graph deliberately preserves every raster-scale detail.  This
module derives a second, bounded *whole graph* by dilating the exact ink mask at
a mask-derived schedule of scales.  It never ranks or truncates nodes inside a
graph.  If no complete graph satisfies the declared cap and raw-support rule,
the result is typed indeterminate rather than a negative visual fact.

Python owns extraction, identity, verification, and replay.
"""

from __future__ import annotations

from bongard.runtime_source_snapshot import capture_loaded_source


_LOADED_SOURCE_SHA256 = capture_loaded_source(__name__, __file__)

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import ndimage

from bongard.canonical import canonical_digest, canonical_json
from bongard.object_scene_anchor_graph import (
    AnchorExtractionLimits,
    ObjectSceneAnchorGraph,
    _absorb_join_reentry_pixels,
    _clusters,
    _exact_bool_mask,
    _mask_digest,
    _pixel_graph,
    _q16_cluster_center,
    _q16_pixel_center,
    _trace_parts,
    _zhang_suen,
    extract_object_scene_anchor_graph,
    object_scene_anchor_graph_extractor_digest,
)
from bongard.prototype_visual_runtime import visual_runtime_dependency_digest
from bongard.python_predicate_authority import PYTHON_PREDICATE_AUTHORITY_ID


ANCHOR_SALIENCE_SCHEMA = "gkm.object-scene-anchor-salience.v1"
ANCHOR_SALIENCE_ALGORITHM_ID = (
    "bongard.object-scene-anchor-salience/adaptive-unfilled-ink-envelope-v2"
)
ANCHOR_SALIENCE_Q_DEFINITION = (
    "nearest-rank-p90-of-exact-integer-chessboard-distance-to-infinite-false-"
    "exterior-on-raw-zhang-suen-skeleton"
)
ANCHOR_SALIENCE_RADIUS_RULE = (
    "2q,ceil(5q/2),3q,ceil(7q/2),4q;5q-audit-sentinel"
)
ANCHOR_SALIENCE_HARD_COMPLETE_CAP = 8
ANCHOR_SALIENCE_HARD_MAX_RADIUS_PIXELS = 4096
ANCHOR_SALIENCE_HARD_MAX_PADDED_PIXELS = 16_777_216
# A morphology call may satisfy both independent extent caps while its
# padded-pixel/footprint product is still impractically large.  This fixed
# cumulative ceiling covers every scheduled attempt plus the mandatory audit.
# It is the smallest tested power-of-two above every work bound in the frozen
# 700-proposal support-only TRAIN telemetry cohort; larger cases remain a typed
# resource abstention, and salience remains optional to the panel-primary lane.
ANCHOR_SALIENCE_HARD_MAX_MORPHOLOGY_WORK = 536_870_912
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_ANCHOR_ID = re.compile(r"(?:part|compact)-[0-9]{8}\Z")
_RAW_POINT_ID = re.compile(r"raw-point-[0-9]{8}\Z")
_STATUS_REASONS = frozenset(
    (
        "complete",
        "empty_foreground",
        "empty_raw_skeleton",
        "raw_anchor_indeterminate",
        "raw_anchor_error",
        "candidate_anchor_indeterminate",
        "candidate_anchor_error",
        "salience_resource_cap_exceeded",
        "salience_cap_exceeded",
        "ownership_error",
    )
)
_ATTEMPT_REASONS = frozenset(
    (
        "accepted",
        "frame_cap_exceeded",
        "part_cap_exceeded",
        "compact_cap_exceeded",
        "macro_anchor_cap_exceeded",
        "support_below_q",
        "graph_indeterminate",
        "graph_error",
        "ownership_error",
    )
)
_WHOLE_GRAPH_CAP_REASONS = frozenset(
    (
        "frame_cap_exceeded",
        "part_cap_exceeded",
        "compact_cap_exceeded",
        "macro_anchor_cap_exceeded",
    )
)


def object_scene_anchor_salience_source_digest() -> str:
    """Return the loaded source identity, rejecting post-import mutation."""

    current = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if current != _LOADED_SOURCE_SHA256:
        raise RuntimeError("object scene anchor salience source changed after import")
    return _LOADED_SOURCE_SHA256


def object_scene_anchor_salience_extractor_digest() -> str:
    """Return the source-, graph-, and native-runtime-bound extractor identity."""

    return canonical_digest(
        {
            "algorithm_id": ANCHOR_SALIENCE_ALGORITHM_ID,
            "source_digest": object_scene_anchor_salience_source_digest(),
            "anchor_graph_extractor_digest": object_scene_anchor_graph_extractor_digest(),
            "visual_runtime_dependency_digest": visual_runtime_dependency_digest(),
            "q_definition": ANCHOR_SALIENCE_Q_DEFINITION,
            "radius_rule": ANCHOR_SALIENCE_RADIUS_RULE,
            "dilation_footprint": "exact-integer-euclidean-disk",
            "morphology_work_upper_bound": (
                "padded-pixels-times-sum-of-square-footprint-cells-for-all-"
                "scheduled-attempts-plus-audit"
            ),
            "hard_max_morphology_work": (
                ANCHOR_SALIENCE_HARD_MAX_MORPHOLOGY_WORK
            ),
            "holes_filled": False,
            "selection": "first-acceptable-complete-whole-graph-never-top-k",
            "ownership_distance": "integer-chebyshev",
            "ownership_tie_break": "canonical-macro-anchor-id",
        }
    )


def _exact_fields(value: Mapping[str, Any], expected: frozenset[str], label: str) -> None:
    if not isinstance(value, Mapping) or set(value) != expected:
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


def _list(value: object, label: str) -> list[object]:
    if not isinstance(value, list):
        raise TypeError(f"{label} must be a JSON list")
    return value


@dataclass(frozen=True)
class AnchorSalienceLimits:
    """Bound whole-graph exposure and native resource use without truncation."""

    max_frames: int = 8
    max_parts: int = 8
    max_compact_components: int = 8
    max_macro_anchors: int = 8
    max_radius_pixels: int = ANCHOR_SALIENCE_HARD_MAX_RADIUS_PIXELS
    max_padded_pixels: int = ANCHOR_SALIENCE_HARD_MAX_PADDED_PIXELS
    anchor_graph_limits: AnchorExtractionLimits = AnchorExtractionLimits()

    def __post_init__(self) -> None:
        for label, value in (
            ("max_frames", self.max_frames),
            ("max_parts", self.max_parts),
            ("max_compact_components", self.max_compact_components),
            ("max_macro_anchors", self.max_macro_anchors),
        ):
            _integer(value, label)
            if value > ANCHOR_SALIENCE_HARD_COMPLETE_CAP:
                raise ValueError(
                    f"{label} cannot relax the fixed complete-graph cap of "
                    f"{ANCHOR_SALIENCE_HARD_COMPLETE_CAP}"
                )
        _integer(self.max_radius_pixels, "max_radius_pixels", minimum=1)
        _integer(self.max_padded_pixels, "max_padded_pixels", minimum=1)
        if self.max_radius_pixels > ANCHOR_SALIENCE_HARD_MAX_RADIUS_PIXELS:
            raise ValueError("max_radius_pixels cannot relax the fixed resource cap")
        if self.max_padded_pixels > ANCHOR_SALIENCE_HARD_MAX_PADDED_PIXELS:
            raise ValueError("max_padded_pixels cannot relax the fixed resource cap")
        if type(self.anchor_graph_limits) is not AnchorExtractionLimits:
            raise TypeError("anchor_graph_limits must be exact AnchorExtractionLimits")
        default_graph_limits = AnchorExtractionLimits()
        for field_name in (
            "max_skeleton_pixels",
            "max_terminals",
            "max_joins",
            "max_compact_components",
            "max_parts",
            "max_points_per_part",
        ):
            if getattr(self.anchor_graph_limits, field_name) > getattr(
                default_graph_limits, field_name
            ):
                raise ValueError(
                    f"anchor_graph_limits.{field_name} cannot relax the fixed "
                    "resource cap"
                )

    def to_data(self) -> dict[str, object]:
        return {
            "max_frames": self.max_frames,
            "max_parts": self.max_parts,
            "max_compact_components": self.max_compact_components,
            "max_macro_anchors": self.max_macro_anchors,
            "max_radius_pixels": self.max_radius_pixels,
            "max_padded_pixels": self.max_padded_pixels,
            "anchor_graph_limits": self.anchor_graph_limits.to_data(),
        }

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "AnchorSalienceLimits":
        _exact_fields(
            data,
            frozenset(
                (
                    "max_frames",
                    "max_parts",
                    "max_compact_components",
                    "max_macro_anchors",
                    "max_radius_pixels",
                    "max_padded_pixels",
                    "anchor_graph_limits",
                )
            ),
            "anchor salience limits",
        )
        raw = data["anchor_graph_limits"]
        if not isinstance(raw, Mapping):
            raise TypeError("anchor_graph_limits must be an object")
        return cls(
            max_frames=data["max_frames"],
            max_parts=data["max_parts"],
            max_compact_components=data["max_compact_components"],
            max_macro_anchors=data["max_macro_anchors"],
            max_radius_pixels=data["max_radius_pixels"],
            max_padded_pixels=data["max_padded_pixels"],
            anchor_graph_limits=AnchorExtractionLimits.from_data(raw),
        )


@dataclass(frozen=True)
class AnchorSalienceStatus:
    state: str
    reason: str

    def __post_init__(self) -> None:
        if self.state not in ("clean", "indeterminate", "error"):
            raise ValueError("anchor salience state differs")
        if self.reason not in _STATUS_REASONS:
            raise ValueError("anchor salience reason differs")
        if (self.state == "clean") != (self.reason == "complete"):
            raise ValueError("only complete anchor salience may be clean")
        if self.state == "indeterminate" and self.reason not in (
            "raw_anchor_indeterminate",
            "candidate_anchor_indeterminate",
            "salience_resource_cap_exceeded",
            "salience_cap_exceeded",
        ):
            raise ValueError("indeterminate salience reason differs")
        if self.state == "error" and self.reason not in (
            "empty_foreground",
            "empty_raw_skeleton",
            "raw_anchor_error",
            "candidate_anchor_error",
            "ownership_error",
        ):
            raise ValueError("error salience reason differs")

    def to_data(self) -> dict[str, str]:
        return {"state": self.state, "reason": self.reason}

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "AnchorSalienceStatus":
        _exact_fields(data, frozenset(("state", "reason")), "anchor salience status")
        return cls(data["state"], data["reason"])


@dataclass(frozen=True, order=True)
class AnchorSupportCount:
    anchor_id: str
    raw_skeleton_pixel_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.anchor_id, str) or _ANCHOR_ID.fullmatch(self.anchor_id) is None:
            raise ValueError("support anchor_id differs")
        _integer(self.raw_skeleton_pixel_count, "raw_skeleton_pixel_count")

    def to_data(self) -> dict[str, object]:
        return {
            "anchor_id": self.anchor_id,
            "raw_skeleton_pixel_count": self.raw_skeleton_pixel_count,
        }

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "AnchorSupportCount":
        _exact_fields(
            data,
            frozenset(("anchor_id", "raw_skeleton_pixel_count")),
            "anchor support count",
        )
        return cls(data["anchor_id"], data["raw_skeleton_pixel_count"])


@dataclass(frozen=True)
class RawSkeletonOwnership:
    raw_point_id: str
    source_y: int
    source_x: int
    padded_y: int
    padded_x: int
    raw_owner_anchor_ids: tuple[str, ...]
    selected_anchor_id: str
    selected_distance_pixels: int

    def __post_init__(self) -> None:
        if not isinstance(self.raw_point_id, str) or _RAW_POINT_ID.fullmatch(self.raw_point_id) is None:
            raise ValueError("raw_point_id differs")
        for label, value in (
            ("source_y", self.source_y),
            ("source_x", self.source_x),
            ("padded_y", self.padded_y),
            ("padded_x", self.padded_x),
            ("selected_distance_pixels", self.selected_distance_pixels),
        ):
            _integer(value, label)
        if (
            type(self.raw_owner_anchor_ids) is not tuple
            or tuple(sorted(set(self.raw_owner_anchor_ids))) != self.raw_owner_anchor_ids
            or any(_ANCHOR_ID.fullmatch(item) is None for item in self.raw_owner_anchor_ids)
        ):
            raise ValueError("raw owner anchor IDs differ")
        if not isinstance(self.selected_anchor_id, str) or _ANCHOR_ID.fullmatch(self.selected_anchor_id) is None:
            raise ValueError("selected anchor ID differs")

    def to_data(self) -> dict[str, object]:
        return {
            "raw_point_id": self.raw_point_id,
            "source_y": self.source_y,
            "source_x": self.source_x,
            "padded_y": self.padded_y,
            "padded_x": self.padded_x,
            "raw_owner_anchor_ids": list(self.raw_owner_anchor_ids),
            "selected_anchor_id": self.selected_anchor_id,
            "selected_distance_pixels": self.selected_distance_pixels,
        }

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "RawSkeletonOwnership":
        _exact_fields(
            data,
            frozenset(
                (
                    "raw_point_id",
                    "source_y",
                    "source_x",
                    "padded_y",
                    "padded_x",
                    "raw_owner_anchor_ids",
                    "selected_anchor_id",
                    "selected_distance_pixels",
                )
            ),
            "raw skeleton ownership",
        )
        owners = _list(data["raw_owner_anchor_ids"], "raw owner anchor IDs")
        if any(not isinstance(item, str) for item in owners):
            raise TypeError("raw owner anchor IDs must be strings")
        return cls(
            data["raw_point_id"],
            data["source_y"],
            data["source_x"],
            data["padded_y"],
            data["padded_x"],
            tuple(owners),
            data["selected_anchor_id"],
            data["selected_distance_pixels"],
        )


@dataclass(frozen=True)
class AnchorSalienceAttempt:
    schedule_index: int
    radius_pixels: int
    disk_footprint_digest: str
    envelope_mask_digest: str
    graph: ObjectSceneAnchorGraph
    support_counts: tuple[AnchorSupportCount, ...]
    acceptable: bool
    reason: str

    def __post_init__(self) -> None:
        _integer(self.schedule_index, "schedule_index")
        _integer(self.radius_pixels, "radius_pixels", minimum=1)
        _digest(self.disk_footprint_digest, "disk footprint digest")
        _digest(self.envelope_mask_digest, "envelope mask digest")
        if type(self.graph) is not ObjectSceneAnchorGraph:
            raise TypeError("attempt graph must be exact ObjectSceneAnchorGraph")
        if (
            type(self.support_counts) is not tuple
            or any(type(item) is not AnchorSupportCount for item in self.support_counts)
            or tuple(sorted(self.support_counts)) != self.support_counts
        ):
            raise ValueError("attempt support counts differ")
        if type(self.acceptable) is not bool or self.reason not in _ATTEMPT_REASONS:
            raise ValueError("attempt decision differs")
        if self.acceptable != (self.reason == "accepted"):
            raise ValueError("attempt acceptance and reason disagree")

    def to_data(self) -> dict[str, object]:
        return {
            "schedule_index": self.schedule_index,
            "radius_pixels": self.radius_pixels,
            "disk_footprint_digest": self.disk_footprint_digest,
            "envelope_mask_digest": self.envelope_mask_digest,
            "graph": self.graph.to_data(),
            "support_counts": [item.to_data() for item in self.support_counts],
            "acceptable": self.acceptable,
            "reason": self.reason,
        }

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "AnchorSalienceAttempt":
        _exact_fields(
            data,
            frozenset(
                (
                    "schedule_index",
                    "radius_pixels",
                    "disk_footprint_digest",
                    "envelope_mask_digest",
                    "graph",
                    "support_counts",
                    "acceptable",
                    "reason",
                )
            ),
            "anchor salience attempt",
        )
        graph = data["graph"]
        counts = _list(data["support_counts"], "attempt support counts")
        if not isinstance(graph, Mapping) or any(not isinstance(item, Mapping) for item in counts):
            raise TypeError("attempt graph or support count fields differ")
        return cls(
            data["schedule_index"],
            data["radius_pixels"],
            data["disk_footprint_digest"],
            data["envelope_mask_digest"],
            ObjectSceneAnchorGraph.from_data(graph),
            tuple(AnchorSupportCount.from_data(item) for item in counts),
            data["acceptable"],
            data["reason"],
        )


def _authority_data() -> dict[str, object]:
    return {
        "predicate_authority_id": PYTHON_PREDICATE_AUTHORITY_ID,
        "python_is_canonical_authority": True,
        "whole_graph_selected_never_top_k": True,
        "holes_filled": False,
        "omitted_anchor_means_absence": False,
        "audit_sentinel_affects_selection": False,
    }


@dataclass(frozen=True)
class ObjectSceneAnchorSalience:
    object_id: str
    extractor_artifact_digest: str
    anchor_graph_extractor_digest: str
    runtime_dependency_digest: str
    source_height_pixels: int
    source_width_pixels: int
    source_foreground_pixel_count: int
    source_mask_digest: str
    crop_y0: int
    crop_x0: int
    crop_y1: int
    crop_x1: int
    q_pixels: int
    padding_pixels: int
    radius_schedule_pixels: tuple[int, ...]
    limits: AnchorSalienceLimits
    status: AnchorSalienceStatus
    raw_graph: ObjectSceneAnchorGraph | None
    attempts: tuple[AnchorSalienceAttempt, ...]
    selected_attempt_index: int | None
    audit_radius_pixels: int
    audit_disk_footprint_digest: str | None
    audit_envelope_mask_digest: str | None
    audit_graph: ObjectSceneAnchorGraph | None
    ownership: tuple[RawSkeletonOwnership, ...]
    selected_support_counts: tuple[AnchorSupportCount, ...]
    raw_part_span_digest: str | None
    artifact_digest: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.object_id, str)
            or not self.object_id
            or len(self.object_id) > 256
            or any(ord(character) < 32 for character in self.object_id)
        ):
            raise ValueError("salience object_id differs")
        _digest(self.extractor_artifact_digest, "salience extractor digest")
        if self.extractor_artifact_digest != object_scene_anchor_salience_extractor_digest():
            raise ValueError("salience extractor digest is stale")
        _digest(self.anchor_graph_extractor_digest, "anchor graph extractor digest")
        if self.anchor_graph_extractor_digest != object_scene_anchor_graph_extractor_digest():
            raise ValueError("anchor graph extractor digest is stale")
        _digest(self.runtime_dependency_digest, "runtime dependency digest")
        if self.runtime_dependency_digest != visual_runtime_dependency_digest():
            raise ValueError("runtime dependency digest is stale")
        for label, value, minimum in (
            ("source_height_pixels", self.source_height_pixels, 1),
            ("source_width_pixels", self.source_width_pixels, 1),
            ("source_foreground_pixel_count", self.source_foreground_pixel_count, 0),
            ("crop_y0", self.crop_y0, 0),
            ("crop_x0", self.crop_x0, 0),
            ("crop_y1", self.crop_y1, 0),
            ("crop_x1", self.crop_x1, 0),
            ("q_pixels", self.q_pixels, 0),
            ("padding_pixels", self.padding_pixels, 0),
            ("audit_radius_pixels", self.audit_radius_pixels, 0),
        ):
            _integer(value, label, minimum=minimum)
        _digest(self.source_mask_digest, "source mask digest")
        if not (
            0 <= self.crop_y0 <= self.crop_y1 <= self.source_height_pixels
            and 0 <= self.crop_x0 <= self.crop_x1 <= self.source_width_pixels
        ):
            raise ValueError("salience crop bounds differ")
        if type(self.radius_schedule_pixels) is not tuple or any(
            type(item) is not int or item < 1 for item in self.radius_schedule_pixels
        ):
            raise TypeError("radius schedule differs")
        if type(self.limits) is not AnchorSalienceLimits or type(self.status) is not AnchorSalienceStatus:
            raise TypeError("salience limits/status differ")
        if self.raw_graph is not None and type(self.raw_graph) is not ObjectSceneAnchorGraph:
            raise TypeError("raw graph differs")
        if (
            type(self.attempts) is not tuple
            or any(type(item) is not AnchorSalienceAttempt for item in self.attempts)
            or tuple(item.schedule_index for item in self.attempts)
            != tuple(range(len(self.attempts)))
        ):
            raise ValueError("salience attempts differ")
        if self.selected_attempt_index is not None:
            _integer(self.selected_attempt_index, "selected_attempt_index")
        if self.audit_graph is not None and type(self.audit_graph) is not ObjectSceneAnchorGraph:
            raise TypeError("audit graph differs")
        for label, value in (
            ("audit_disk_footprint_digest", self.audit_disk_footprint_digest),
            ("audit_envelope_mask_digest", self.audit_envelope_mask_digest),
            ("raw_part_span_digest", self.raw_part_span_digest),
        ):
            if value is not None:
                _digest(value, label)
        if (
            type(self.ownership) is not tuple
            or any(type(item) is not RawSkeletonOwnership for item in self.ownership)
            or tuple(item.raw_point_id for item in self.ownership)
            != tuple(f"raw-point-{index:08d}" for index in range(len(self.ownership)))
        ):
            raise ValueError("salience ownership differs")
        if (
            type(self.selected_support_counts) is not tuple
            or any(type(item) is not AnchorSupportCount for item in self.selected_support_counts)
            or tuple(sorted(self.selected_support_counts)) != self.selected_support_counts
        ):
            raise ValueError("selected support counts differ")
        self._validate_policy()
        _digest(self.artifact_digest, "salience artifact digest")
        if self.artifact_digest != canonical_digest(self._unsigned_data()):
            raise ValueError("salience artifact digest differs")

    def _validate_policy(self) -> None:
        expected_schedule = _radius_schedule(self.q_pixels) if self.q_pixels else ()
        if self.radius_schedule_pixels != expected_schedule:
            raise ValueError("salience radius schedule is not canonical")
        if self.q_pixels:
            if self.padding_pixels != 5 * self.q_pixels + 1 or self.audit_radius_pixels != 5 * self.q_pixels:
                raise ValueError("salience padding or audit radius differs")
        elif self.padding_pixels or self.audit_radius_pixels:
            raise ValueError("zero-q salience must have zero padding and audit radius")
        if self.raw_graph is not None and self.raw_graph.object_id != self.object_id:
            raise ValueError("raw graph object binding differs")
        if any(item.graph.object_id != self.object_id for item in self.attempts):
            raise ValueError("attempt graph object binding differs")
        if self.audit_graph is not None and self.audit_graph.object_id != self.object_id:
            raise ValueError("audit graph object binding differs")
        graphs = tuple(
            item
            for item in (
                self.raw_graph,
                *(attempt.graph for attempt in self.attempts),
                self.audit_graph,
            )
            if item is not None
        )
        if any(item.limits != self.limits.anchor_graph_limits for item in graphs):
            raise ValueError("salience graph extraction limits differ")
        crop_height = self.crop_y1 - self.crop_y0
        crop_width = self.crop_x1 - self.crop_x0
        if self.source_foreground_pixel_count > (
            self.source_height_pixels * self.source_width_pixels
        ):
            raise ValueError("source foreground count exceeds source extent")
        if self.source_foreground_pixel_count == 0:
            if (
                (self.crop_y0, self.crop_x0, self.crop_y1, self.crop_x1)
                != (0, 0, 0, 0)
                or self.raw_graph is not None
            ):
                raise ValueError("empty salience source exposes a crop or graph")
        elif (
            crop_height < 1
            or crop_width < 1
            or self.source_foreground_pixel_count > crop_height * crop_width
            or self.raw_graph is None
        ):
            raise ValueError("nonempty salience source lacks its raw crop graph")
        if self.raw_graph is not None and (
            self.raw_graph.mask_height_pixels != crop_height
            or self.raw_graph.mask_width_pixels != crop_width
            or self.raw_graph.foreground_pixel_count
            != self.source_foreground_pixel_count
        ):
            raise ValueError("raw graph crop binding differs")
        padded_shape = (
            crop_height + 2 * self.padding_pixels,
            crop_width + 2 * self.padding_pixels,
        )
        resource_exceeded = self.q_pixels > 0 and _salience_resource_exceeded(
            (crop_height, crop_width), self.q_pixels, self.limits
        )
        if resource_exceeded and (self.attempts or self.audit_graph is not None):
            raise ValueError("resource-gap salience exposes post-cap artifacts")
        disk_digests: dict[int, str] = {}

        def expected_disk_digest(radius: int) -> str:
            if radius not in disk_digests:
                disk_digests[radius] = _disk_footprint_digest(radius)
            return disk_digests[radius]

        for index, attempt in enumerate(self.attempts):
            if (
                index >= len(self.radius_schedule_pixels)
                or attempt.radius_pixels != self.radius_schedule_pixels[index]
                or attempt.disk_footprint_digest
                != expected_disk_digest(attempt.radius_pixels)
                or (
                    attempt.graph.mask_height_pixels,
                    attempt.graph.mask_width_pixels,
                )
                != padded_shape
            ):
                raise ValueError("salience attempt schedule or dimensions differ")
            macro_ids = tuple(
                sorted(
                    [item.part_id for item in attempt.graph.parts]
                    + [
                        item.compact_id
                        for item in attempt.graph.compact_components
                    ]
                )
            )
            if (
                attempt.graph.status.state == "clean"
                and attempt.reason in _WHOLE_GRAPH_CAP_REASONS
            ):
                if (
                    attempt.support_counts
                    or attempt.reason
                    != _attempt_reason(
                        attempt.graph,
                        (),
                        self.q_pixels,
                        self.limits,
                    )
                ):
                    raise ValueError("over-cap salience attempt differs")
            elif attempt.graph.status.state == "clean" and attempt.reason != "ownership_error":
                if (
                    tuple(item.anchor_id for item in attempt.support_counts)
                    != macro_ids
                    or self.raw_graph is None
                    or sum(
                        item.raw_skeleton_pixel_count
                        for item in attempt.support_counts
                    )
                    != self.raw_graph.skeleton_pixel_count
                    or attempt.reason
                    != _attempt_reason(
                        attempt.graph,
                        attempt.support_counts,
                        self.q_pixels,
                        self.limits,
                    )
                ):
                    raise ValueError("salience attempt decision witness differs")
            elif attempt.graph.status.state != "clean" and (
                attempt.support_counts
                or attempt.reason
                != _attempt_reason(
                    attempt.graph,
                    (),
                    self.q_pixels,
                    self.limits,
                )
            ):
                raise ValueError("non-clean salience attempt differs")
        if self.audit_graph is not None and (
            (self.audit_graph.mask_height_pixels, self.audit_graph.mask_width_pixels)
            != padded_shape
            or self.audit_disk_footprint_digest is None
            or self.audit_envelope_mask_digest is None
            or self.audit_disk_footprint_digest
            != expected_disk_digest(self.audit_radius_pixels)
        ):
            raise ValueError("audit sentinel binding differs")
        if self.audit_graph is None and (
            self.audit_disk_footprint_digest is not None
            or self.audit_envelope_mask_digest is not None
        ):
            raise ValueError("partial audit sentinel differs")
        if self.status.state == "clean":
            if (
                self.selected_attempt_index is None
                or self.selected_attempt_index >= len(self.attempts)
                or not self.attempts[self.selected_attempt_index].acceptable
                or any(item.acceptable for item in self.attempts[: self.selected_attempt_index])
                or len(self.attempts) != self.selected_attempt_index + 1
                or self.audit_graph is None
                or self.raw_graph is None
                or self.raw_graph.status.state != "clean"
            ):
                raise ValueError("clean salience selection differs")
            selected = self.attempts[self.selected_attempt_index]
            if self.selected_support_counts != selected.support_counts:
                raise ValueError("selected support counts disagree with selected attempt")
            selected_ids = tuple(
                sorted(
                    [item.part_id for item in selected.graph.parts]
                    + [item.compact_id for item in selected.graph.compact_components]
                )
            )
            if tuple(item.anchor_id for item in self.selected_support_counts) != selected_ids:
                raise ValueError("selected support anchor inventory differs")
            part_ids = {item.part_id for item in selected.graph.parts}
            if any(
                item.anchor_id in part_ids
                and item.raw_skeleton_pixel_count < self.q_pixels
                for item in self.selected_support_counts
            ):
                raise ValueError("clean selected anchor has insufficient raw support")
            if len(self.ownership) != self.raw_graph.skeleton_pixel_count:
                raise ValueError("raw skeleton ownership is incomplete")
            if sum(item.raw_skeleton_pixel_count for item in self.selected_support_counts) != len(self.ownership):
                raise ValueError("selected support counts do not partition raw skeleton")
            if self.raw_part_span_digest != canonical_digest(
                [
                    {
                        "raw_point_id": item.raw_point_id,
                        "raw_owner_anchor_ids": list(item.raw_owner_anchor_ids),
                    }
                    for item in self.ownership
                ]
            ):
                raise ValueError("raw part span digest differs")
            support = {item.anchor_id: item.raw_skeleton_pixel_count for item in self.selected_support_counts}
            observed = {key: 0 for key in support}
            raw_anchor_ids = {
                *(item.part_id for item in self.raw_graph.parts),
                *(item.compact_id for item in self.raw_graph.compact_components),
            }
            reconstructed_raw_skeleton = np.zeros(
                (crop_height, crop_width), dtype=bool
            )
            previous_source_point: tuple[int, int] | None = None
            for item in self.ownership:
                if item.selected_anchor_id not in observed:
                    raise ValueError("ownership names an unknown selected anchor")
                source_point = (item.source_y, item.source_x)
                if (
                    previous_source_point is not None
                    and source_point <= previous_source_point
                ):
                    raise ValueError("ownership source coordinates are not canonical")
                previous_source_point = source_point
                local_y = item.source_y - self.crop_y0
                local_x = item.source_x - self.crop_x0
                if not (0 <= local_y < crop_height and 0 <= local_x < crop_width):
                    raise ValueError("ownership source coordinate is outside the crop")
                if reconstructed_raw_skeleton[local_y, local_x]:
                    raise ValueError("ownership duplicates a raw skeleton coordinate")
                reconstructed_raw_skeleton[local_y, local_x] = True
                if (
                    not item.raw_owner_anchor_ids
                    or not set(item.raw_owner_anchor_ids) <= raw_anchor_ids
                ):
                    raise ValueError("ownership names an unknown raw anchor")
                if item.padded_y != local_y + self.padding_pixels or item.padded_x != local_x + self.padding_pixels:
                    raise ValueError("ownership coordinate frames disagree")
                if item.selected_distance_pixels > max(padded_shape) - 1:
                    raise ValueError("ownership distance exceeds the padded extent")
                observed[item.selected_anchor_id] += 1
            if observed != support:
                raise ValueError("ownership and support counts disagree")
            if _mask_digest(
                reconstructed_raw_skeleton, "zhang-suen-skeleton"
            ) != self.raw_graph.skeleton_digest:
                raise ValueError("ownership coordinates differ from raw skeleton")
        elif (
            self.selected_attempt_index is not None
            or self.ownership
            or self.selected_support_counts
            or self.raw_part_span_digest is not None
            or any(item.acceptable for item in self.attempts)
        ):
            raise ValueError("non-clean salience exposes a selected partial graph")
        if self.status.reason == "salience_cap_exceeded" and len(self.attempts) != len(
            self.radius_schedule_pixels
        ):
            raise ValueError("salience cap gap did not exhaust the schedule")
        if self.status.reason in (
            "candidate_anchor_indeterminate",
            "candidate_anchor_error",
            "ownership_error",
        ) and not self.attempts:
            raise ValueError("candidate failure lacks its exact attempt")
        self._validate_status_binding(resource_exceeded)

    def _validate_status_binding(self, resource_exceeded: bool) -> None:
        reason = self.status.reason
        if self.source_foreground_pixel_count == 0:
            if (
                reason != "empty_foreground"
                or self.q_pixels
                or self.attempts
                or self.audit_graph is not None
            ):
                raise ValueError("empty-foreground status binding differs")
            return
        if self.raw_graph is None:
            raise ValueError("nonempty salience lacks a raw graph")
        if self.raw_graph.status.state != "clean":
            expected = (
                "raw_anchor_indeterminate"
                if self.raw_graph.status.state == "indeterminate"
                else "raw_anchor_error"
            )
            if (
                reason != expected
                or self.q_pixels
                or self.attempts
                or self.audit_graph is not None
            ):
                raise ValueError("raw-anchor failure status binding differs")
            return
        if self.raw_graph.skeleton_pixel_count == 0:
            if (
                reason != "empty_raw_skeleton"
                or self.q_pixels
                or self.attempts
                or self.audit_graph is not None
            ):
                raise ValueError("empty-skeleton status binding differs")
            return
        if self.q_pixels < 1:
            raise ValueError("nonempty raw skeleton lacks positive q")
        if resource_exceeded:
            if (
                reason != "salience_resource_cap_exceeded"
                or self.attempts
                or self.audit_graph is not None
            ):
                raise ValueError("resource-gap status binding differs")
            return
        if reason == "salience_resource_cap_exceeded":
            raise ValueError("spurious salience resource gap")
        if not self.attempts or self.audit_graph is None:
            raise ValueError("post-q salience lacks attempts or audit sentinel")
        ordinary_rejections = _WHOLE_GRAPH_CAP_REASONS | {"support_below_q"}
        terminal_map = {
            "candidate_anchor_indeterminate": "graph_indeterminate",
            "candidate_anchor_error": "graph_error",
            "ownership_error": "ownership_error",
        }
        if reason in terminal_map:
            if (
                self.attempts[-1].reason != terminal_map[reason]
                or any(
                    item.reason not in ordinary_rejections
                    for item in self.attempts[:-1]
                )
            ):
                raise ValueError("candidate failure status binding differs")
            return
        if reason == "salience_cap_exceeded":
            if any(item.reason not in ordinary_rejections for item in self.attempts):
                raise ValueError("salience cap gap contains a terminal attempt")
            return
        if reason == "complete":
            if self.attempts[-1].reason != "accepted" or any(
                item.reason not in ordinary_rejections
                for item in self.attempts[:-1]
            ):
                raise ValueError("clean selection status binding differs")
            return
        raise ValueError("salience status lacks a canonical extraction path")

    def _unsigned_data(self) -> dict[str, object]:
        return {
            "schema": ANCHOR_SALIENCE_SCHEMA,
            "algorithm_id": ANCHOR_SALIENCE_ALGORITHM_ID,
            "object_id": self.object_id,
            "extractor_artifact_digest": self.extractor_artifact_digest,
            "anchor_graph_extractor_digest": self.anchor_graph_extractor_digest,
            "runtime_dependency_digest": self.runtime_dependency_digest,
            "source_height_pixels": self.source_height_pixels,
            "source_width_pixels": self.source_width_pixels,
            "source_foreground_pixel_count": self.source_foreground_pixel_count,
            "source_mask_digest": self.source_mask_digest,
            "crop_y0": self.crop_y0,
            "crop_x0": self.crop_x0,
            "crop_y1": self.crop_y1,
            "crop_x1": self.crop_x1,
            "q_pixels": self.q_pixels,
            "q_definition": ANCHOR_SALIENCE_Q_DEFINITION,
            "distance_metric": "integer-chessboard",
            "padding_pixels": self.padding_pixels,
            "radius_schedule_pixels": list(self.radius_schedule_pixels),
            "radius_rule": ANCHOR_SALIENCE_RADIUS_RULE,
            "limits": self.limits.to_data(),
            "status": self.status.to_data(),
            "raw_graph": None if self.raw_graph is None else self.raw_graph.to_data(),
            "attempts": [item.to_data() for item in self.attempts],
            "selected_attempt_index": self.selected_attempt_index,
            "audit_radius_pixels": self.audit_radius_pixels,
            "audit_disk_footprint_digest": self.audit_disk_footprint_digest,
            "audit_envelope_mask_digest": self.audit_envelope_mask_digest,
            "audit_graph": None if self.audit_graph is None else self.audit_graph.to_data(),
            "ownership": [item.to_data() for item in self.ownership],
            "selected_support_counts": [item.to_data() for item in self.selected_support_counts],
            "raw_part_span_digest": self.raw_part_span_digest,
            **_authority_data(),
        }

    def to_data(self) -> dict[str, object]:
        return {**self._unsigned_data(), "artifact_digest": self.artifact_digest}

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "ObjectSceneAnchorSalience":
        expected = frozenset(
            (
                "schema",
                "algorithm_id",
                "object_id",
                "extractor_artifact_digest",
                "anchor_graph_extractor_digest",
                "runtime_dependency_digest",
                "source_height_pixels",
                "source_width_pixels",
                "source_foreground_pixel_count",
                "source_mask_digest",
                "crop_y0",
                "crop_x0",
                "crop_y1",
                "crop_x1",
                "q_pixels",
                "q_definition",
                "distance_metric",
                "padding_pixels",
                "radius_schedule_pixels",
                "radius_rule",
                "limits",
                "status",
                "raw_graph",
                "attempts",
                "selected_attempt_index",
                "audit_radius_pixels",
                "audit_disk_footprint_digest",
                "audit_envelope_mask_digest",
                "audit_graph",
                "ownership",
                "selected_support_counts",
                "raw_part_span_digest",
                *_authority_data(),
                "artifact_digest",
            )
        )
        _exact_fields(data, expected, "object scene anchor salience")
        if (
            data["schema"] != ANCHOR_SALIENCE_SCHEMA
            or data["algorithm_id"] != ANCHOR_SALIENCE_ALGORITHM_ID
            or data["q_definition"] != ANCHOR_SALIENCE_Q_DEFINITION
            or data["distance_metric"] != "integer-chessboard"
            or data["radius_rule"] != ANCHOR_SALIENCE_RADIUS_RULE
            or any(data[key] != value for key, value in _authority_data().items())
        ):
            raise ValueError("anchor salience policy differs")
        limits, status = data["limits"], data["status"]
        raw_graph, audit_graph = data["raw_graph"], data["audit_graph"]
        attempts = _list(data["attempts"], "salience attempts")
        ownership = _list(data["ownership"], "salience ownership")
        counts = _list(data["selected_support_counts"], "selected support counts")
        schedule = _list(data["radius_schedule_pixels"], "radius schedule")
        if (
            not isinstance(limits, Mapping)
            or not isinstance(status, Mapping)
            or (raw_graph is not None and not isinstance(raw_graph, Mapping))
            or (audit_graph is not None and not isinstance(audit_graph, Mapping))
            or any(not isinstance(item, Mapping) for item in (*attempts, *ownership, *counts))
        ):
            raise TypeError("anchor salience nested fields differ")
        return cls(
            object_id=data["object_id"],
            extractor_artifact_digest=data["extractor_artifact_digest"],
            anchor_graph_extractor_digest=data["anchor_graph_extractor_digest"],
            runtime_dependency_digest=data["runtime_dependency_digest"],
            source_height_pixels=data["source_height_pixels"],
            source_width_pixels=data["source_width_pixels"],
            source_foreground_pixel_count=data["source_foreground_pixel_count"],
            source_mask_digest=data["source_mask_digest"],
            crop_y0=data["crop_y0"],
            crop_x0=data["crop_x0"],
            crop_y1=data["crop_y1"],
            crop_x1=data["crop_x1"],
            q_pixels=data["q_pixels"],
            padding_pixels=data["padding_pixels"],
            radius_schedule_pixels=tuple(schedule),
            limits=AnchorSalienceLimits.from_data(limits),
            status=AnchorSalienceStatus.from_data(status),
            raw_graph=None if raw_graph is None else ObjectSceneAnchorGraph.from_data(raw_graph),
            attempts=tuple(AnchorSalienceAttempt.from_data(item) for item in attempts),
            selected_attempt_index=data["selected_attempt_index"],
            audit_radius_pixels=data["audit_radius_pixels"],
            audit_disk_footprint_digest=data["audit_disk_footprint_digest"],
            audit_envelope_mask_digest=data["audit_envelope_mask_digest"],
            audit_graph=None if audit_graph is None else ObjectSceneAnchorGraph.from_data(audit_graph),
            ownership=tuple(RawSkeletonOwnership.from_data(item) for item in ownership),
            selected_support_counts=tuple(AnchorSupportCount.from_data(item) for item in counts),
            raw_part_span_digest=data["raw_part_span_digest"],
            artifact_digest=data["artifact_digest"],
        )

    @property
    def selected_graph(self) -> ObjectSceneAnchorGraph | None:
        if self.selected_attempt_index is None:
            return None
        return self.attempts[self.selected_attempt_index].graph


def _radius_schedule(q: int) -> tuple[int, ...]:
    if q < 1:
        return ()
    return (2 * q, (5 * q + 1) // 2, 3 * q, (7 * q + 1) // 2, 4 * q)


def _disk(radius: int) -> np.ndarray:
    axis = np.arange(-radius, radius + 1, dtype=np.int64)
    squared = axis * axis
    result = np.empty((len(axis), len(axis)), dtype=bool)
    radius_squared = radius * radius
    for index, y_squared in enumerate(squared):
        result[index] = squared + y_squared <= radius_squared
    return result


def _disk_footprint_digest(radius: int) -> str:
    return _mask_digest(_disk(radius), "euclidean-disk-footprint")


def _morphology_work_upper_bound(crop_shape: tuple[int, int], q: int) -> int:
    """Bound all scheduled dilation work without allocating native arrays."""

    crop_height, crop_width = crop_shape
    padding = 5 * q + 1
    padded_pixels = (crop_height + 2 * padding) * (crop_width + 2 * padding)
    radii = (*_radius_schedule(q), 5 * q)
    # The exact Euclidean disk is a subset of its square bounding footprint.
    # Count repeated schedule radii because the current extractor invokes each
    # attempt independently, then always invokes the audit sentinel.
    footprint_cells = sum((2 * radius + 1) ** 2 for radius in radii)
    return padded_pixels * footprint_cells


def _salience_resource_exceeded(
    crop_shape: tuple[int, int],
    q: int,
    limits: AnchorSalienceLimits,
) -> bool:
    padding = 5 * q + 1
    padded_height = crop_shape[0] + 2 * padding
    padded_width = crop_shape[1] + 2 * padding
    return (
        5 * q > limits.max_radius_pixels
        or padded_height * padded_width > limits.max_padded_pixels
        or (10 * q + 1) * (10 * q + 1) > limits.max_padded_pixels
        or _morphology_work_upper_bound(crop_shape, q)
        > ANCHOR_SALIENCE_HARD_MAX_MORPHOLOGY_WORK
    )


def _tight_crop(mask: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]] | None:
    ys, xs = np.nonzero(mask)
    if not len(ys):
        return None
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    return np.ascontiguousarray(mask[y0:y1, x0:x1]), (y0, x0, y1, x1)


def _q_from_raw_skeleton(mask: np.ndarray, skeleton: np.ndarray) -> int:
    # The tight crop's outside is false.  An explicit false ring makes that
    # infinite-exterior convention independent of SciPy boundary behaviour.
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    distances = ndimage.distance_transform_cdt(padded, metric="chessboard")[1:-1, 1:-1]
    values = np.sort(np.asarray(distances[skeleton], dtype=np.int64))
    if not len(values):
        return 0
    rank = (9 * len(values) + 9) // 10
    return int(values[rank - 1])


def _part_pixel_sets(
    mask: np.ndarray, graph_artifact: ObjectSceneAnchorGraph
) -> tuple[np.ndarray, dict[str, set[tuple[int, int]]]]:
    """Rebuild the graph's exact pixel spans and prove canonical part order."""

    skeleton = _zhang_suen(mask)
    pixel_graph = _pixel_graph(skeleton)
    terminals = [{point} for point, neighbours in pixel_graph.items() if len(neighbours) == 1]
    joins = _clusters({point for point, neighbours in pixel_graph.items() if len(neighbours) >= 3})
    terminals.sort(key=lambda item: tuple(sorted(item)))
    joins.sort(key=lambda item: tuple(sorted(item)))
    _absorb_join_reentry_pixels(pixel_graph, joins)
    joins.sort(key=lambda item: tuple(sorted(item)))
    raw_parts, _ = _trace_parts(pixel_graph, terminals, joins)
    if len(raw_parts) != len(graph_artifact.parts):
        raise RuntimeError("ownership_error")
    height, width = mask.shape
    terminal_centers = tuple(_q16_cluster_center(item, width, height) for item in terminals)
    join_centers = tuple(_q16_cluster_center(item, width, height) for item in joins)

    def q16(vertex: tuple[object, ...]):
        if vertex[0] == "terminal":
            return terminal_centers[int(vertex[1])]
        if vertex[0] == "join":
            return join_centers[int(vertex[1])]
        return _q16_pixel_center((int(vertex[1]), int(vertex[2])), width, height)

    result: dict[str, set[tuple[int, int]]] = {}
    for index, ((endpoint_ids, path, closed), frozen) in enumerate(
        zip(raw_parts, graph_artifact.parts, strict=True)
    ):
        if (
            frozen.part_id != f"part-{index:08d}"
            or frozen.endpoint_node_ids != endpoint_ids
            or frozen.closed != closed
            or frozen.path_q16 != tuple(q16(vertex) for vertex in path)
        ):
            raise RuntimeError("ownership_error")
        points: set[tuple[int, int]] = set()
        for vertex in path:
            if vertex[0] == "terminal":
                points.update(terminals[int(vertex[1])])
            elif vertex[0] == "join":
                points.update(joins[int(vertex[1])])
            else:
                points.add((int(vertex[1]), int(vertex[2])))
        result[frozen.part_id] = points
    return skeleton, result


def _compact_pixel_sets(
    mask: np.ndarray,
    skeleton: np.ndarray,
    graph_artifact: ObjectSceneAnchorGraph,
) -> dict[str, set[tuple[int, int]]]:
    labels, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=bool))
    components: dict[str, tuple[set[tuple[int, int]], set[tuple[int, int]]]] = {}
    for label in range(1, count + 1):
        component = labels == label
        points = {(int(y), int(x)) for y, x in np.argwhere(component)}
        digest = _mask_digest(component, "source-connected-component")
        components[digest] = (points, {point for point in points if skeleton[point]})
    result: dict[str, set[tuple[int, int]]] = {}
    used_isolated: dict[str, set[tuple[int, int]]] = {}
    height, width = mask.shape
    for item in graph_artifact.compact_components:
        if item.source_component_digest not in components:
            raise RuntimeError("ownership_error")
        source_points, skeleton_points = components[item.source_component_digest]
        if item.reason == "source_component_thinned_empty":
            points = source_points
        else:
            available = sorted(skeleton_points - used_isolated.setdefault(item.source_component_digest, set()))
            matching = [
                point
                for point in available
                if _q16_pixel_center(point, width, height) == item.location_q16
            ]
            if len(matching) != 1:
                raise RuntimeError("ownership_error")
            points = {matching[0]}
            used_isolated[item.source_component_digest].add(matching[0])
        result[item.compact_id] = set(points)
    return result


def _macro_pixel_sets(
    mask: np.ndarray, graph_artifact: ObjectSceneAnchorGraph
) -> tuple[np.ndarray, dict[str, set[tuple[int, int]]]]:
    if graph_artifact.status.state != "clean":
        raise RuntimeError("ownership_error")
    skeleton, parts = _part_pixel_sets(mask, graph_artifact)
    compacts = _compact_pixel_sets(mask, skeleton, graph_artifact)
    result = {**parts, **compacts}
    expected = {
        *(item.part_id for item in graph_artifact.parts),
        *(item.compact_id for item in graph_artifact.compact_components),
    }
    if set(result) != expected or any(not points for points in result.values()):
        raise RuntimeError("ownership_error")
    return skeleton, result


def _assign_raw_points(
    *,
    raw_skeleton: np.ndarray,
    raw_macro_sets: Mapping[str, set[tuple[int, int]]],
    selected_mask: np.ndarray,
    selected_graph: ObjectSceneAnchorGraph,
    crop_bounds: tuple[int, int, int, int],
    padding: int,
) -> tuple[tuple[RawSkeletonOwnership, ...], tuple[AnchorSupportCount, ...]]:
    _, selected_sets = _macro_pixel_sets(selected_mask, selected_graph)
    if not selected_sets:
        raise RuntimeError("ownership_error")
    raw_owner: dict[tuple[int, int], list[str]] = {
        (int(y), int(x)): [] for y, x in np.argwhere(raw_skeleton)
    }
    for anchor_id, points in raw_macro_sets.items():
        for point in points:
            if point in raw_owner:
                raw_owner[point].append(anchor_id)
    if any(not owners for owners in raw_owner.values()):
        raise RuntimeError("ownership_error")
    raw_points = sorted(raw_owner)
    selected_ids = tuple(sorted(selected_sets))
    padded_y = np.asarray(
        [point[0] + padding for point in raw_points], dtype=np.int64
    )
    padded_x = np.asarray(
        [point[1] + padding for point in raw_points], dtype=np.int64
    )
    best_distances = np.full(len(raw_points), np.iinfo(np.int64).max, dtype=np.int64)
    best_anchor_indices = np.zeros(len(raw_points), dtype=np.uint8)
    for anchor_index, anchor_id in enumerate(selected_ids):
        seeds = np.zeros(selected_mask.shape, dtype=bool)
        for point in selected_sets[anchor_id]:
            seeds[point] = True
        distance_field = ndimage.distance_transform_cdt(
            ~seeds, metric="chessboard"
        )
        candidate_distances = np.asarray(
            distance_field[padded_y, padded_x], dtype=np.int64
        )
        closer = candidate_distances < best_distances
        best_distances[closer] = candidate_distances[closer]
        best_anchor_indices[closer] = anchor_index
    y0, x0, _, _ = crop_bounds
    counts = {item: 0 for item in selected_ids}
    rows: list[RawSkeletonOwnership] = []
    for index, (y, x) in enumerate(raw_points):
        py, px = y + padding, x + padding
        distance = int(best_distances[index])
        selected_id = selected_ids[int(best_anchor_indices[index])]
        counts[selected_id] += 1
        rows.append(
            RawSkeletonOwnership(
                raw_point_id=f"raw-point-{index:08d}",
                source_y=y0 + y,
                source_x=x0 + x,
                padded_y=py,
                padded_x=px,
                raw_owner_anchor_ids=tuple(sorted(raw_owner[(y, x)])),
                selected_anchor_id=selected_id,
                selected_distance_pixels=distance,
            )
        )
    return tuple(rows), tuple(
        AnchorSupportCount(anchor_id, counts[anchor_id]) for anchor_id in selected_ids
    )


def _attempt_reason(
    graph_artifact: ObjectSceneAnchorGraph,
    counts: tuple[AnchorSupportCount, ...],
    q: int,
    limits: AnchorSalienceLimits,
) -> str:
    if graph_artifact.status.state == "indeterminate":
        return "graph_indeterminate"
    if graph_artifact.status.state == "error":
        return "graph_error"
    if len(graph_artifact.cyclic_frames) > limits.max_frames:
        return "frame_cap_exceeded"
    if len(graph_artifact.parts) > limits.max_parts:
        return "part_cap_exceeded"
    if len(graph_artifact.compact_components) > limits.max_compact_components:
        return "compact_cap_exceeded"
    if len(graph_artifact.parts) + len(graph_artifact.compact_components) > limits.max_macro_anchors:
        return "macro_anchor_cap_exceeded"
    part_ids = {item.part_id for item in graph_artifact.parts}
    if not counts or any(
        item.anchor_id in part_ids and item.raw_skeleton_pixel_count < q
        for item in counts
    ):
        return "support_below_q"
    return "accepted"


def _construct(
    values: Mapping[str, object],
) -> ObjectSceneAnchorSalience:
    provisional = object.__new__(ObjectSceneAnchorSalience)
    for key, value in values.items():
        object.__setattr__(provisional, key, value)
    return ObjectSceneAnchorSalience(
        **values,
        artifact_digest=canonical_digest(provisional._unsigned_data()),
    )


def _base_values(
    mask: np.ndarray,
    object_id: str,
    limits: AnchorSalienceLimits,
) -> dict[str, object]:
    return {
        "object_id": object_id,
        "extractor_artifact_digest": object_scene_anchor_salience_extractor_digest(),
        "anchor_graph_extractor_digest": object_scene_anchor_graph_extractor_digest(),
        "runtime_dependency_digest": visual_runtime_dependency_digest(),
        "source_height_pixels": int(mask.shape[0]),
        "source_width_pixels": int(mask.shape[1]),
        "source_foreground_pixel_count": int(np.count_nonzero(mask)),
        "source_mask_digest": _mask_digest(mask, "source-mask"),
        "crop_y0": 0,
        "crop_x0": 0,
        "crop_y1": 0,
        "crop_x1": 0,
        "q_pixels": 0,
        "padding_pixels": 0,
        "radius_schedule_pixels": (),
        "limits": limits,
        "status": AnchorSalienceStatus("error", "empty_foreground"),
        "raw_graph": None,
        "attempts": (),
        "selected_attempt_index": None,
        "audit_radius_pixels": 0,
        "audit_disk_footprint_digest": None,
        "audit_envelope_mask_digest": None,
        "audit_graph": None,
        "ownership": (),
        "selected_support_counts": (),
        "raw_part_span_digest": None,
    }


def extract_object_scene_anchor_salience(
    mask: np.ndarray,
    object_id: str,
    limits: AnchorSalienceLimits | None = None,
) -> ObjectSceneAnchorSalience:
    """Extract the first acceptable complete envelope graph or a typed gap."""

    exact = _exact_bool_mask(mask)
    active_limits = limits if limits is not None else AnchorSalienceLimits()
    if type(active_limits) is not AnchorSalienceLimits:
        raise TypeError("limits must be exact AnchorSalienceLimits")
    if (
        not isinstance(object_id, str)
        or not object_id
        or len(object_id) > 256
        or any(ord(character) < 32 for character in object_id)
    ):
        raise ValueError("object_id must be a bounded nonempty string")
    values = _base_values(exact, object_id, active_limits)
    cropped = _tight_crop(exact)
    if cropped is None:
        return _construct(values)
    tight, crop_bounds = cropped
    y0, x0, y1, x1 = crop_bounds
    values.update(crop_y0=y0, crop_x0=x0, crop_y1=y1, crop_x1=x1)
    raw_graph = extract_object_scene_anchor_graph(
        tight, object_id, active_limits.anchor_graph_limits
    )
    values["raw_graph"] = raw_graph
    if raw_graph.status.state != "clean":
        values["status"] = AnchorSalienceStatus(
            raw_graph.status.state,
            "raw_anchor_indeterminate"
            if raw_graph.status.state == "indeterminate"
            else "raw_anchor_error",
        )
        return _construct(values)
    raw_skeleton, raw_macro_sets = _macro_pixel_sets(tight, raw_graph)
    q = _q_from_raw_skeleton(tight, raw_skeleton)
    if q < 1:
        values["status"] = AnchorSalienceStatus("error", "empty_raw_skeleton")
        return _construct(values)
    padding = 5 * q + 1
    schedule = _radius_schedule(q)
    values.update(
        q_pixels=q,
        padding_pixels=padding,
        radius_schedule_pixels=schedule,
        audit_radius_pixels=5 * q,
    )
    if _salience_resource_exceeded(tight.shape, q, active_limits):
        values["status"] = AnchorSalienceStatus(
            "indeterminate", "salience_resource_cap_exceeded"
        )
        return _construct(values)
    padded = np.pad(tight, padding, mode="constant", constant_values=False)
    attempts: list[AnchorSalienceAttempt] = []
    selected_index: int | None = None
    selected_ownership: tuple[RawSkeletonOwnership, ...] = ()
    selected_counts: tuple[AnchorSupportCount, ...] = ()
    terminal_status: AnchorSalienceStatus | None = None
    for index, radius in enumerate(schedule):
        disk = _disk(radius)
        envelope = np.ascontiguousarray(
            ndimage.binary_dilation(padded, structure=disk), dtype=bool
        )
        graph_artifact = extract_object_scene_anchor_graph(
            envelope, object_id, active_limits.anchor_graph_limits
        )
        ownership: tuple[RawSkeletonOwnership, ...] = ()
        counts: tuple[AnchorSupportCount, ...] = ()
        if graph_artifact.status.state == "clean":
            # Reject an over-cap whole graph before constructing one full
            # distance field per anchor.  The caps are both semantic (no
            # truncation) and resource boundaries.
            reason = _attempt_reason(graph_artifact, (), q, active_limits)
        else:
            reason = _attempt_reason(graph_artifact, counts, q, active_limits)
        if graph_artifact.status.state == "clean" and reason == "support_below_q":
            try:
                ownership, counts = _assign_raw_points(
                    raw_skeleton=raw_skeleton,
                    raw_macro_sets=raw_macro_sets,
                    selected_mask=envelope,
                    selected_graph=graph_artifact,
                    crop_bounds=crop_bounds,
                    padding=padding,
                )
                reason = _attempt_reason(graph_artifact, counts, q, active_limits)
            except RuntimeError:
                reason = "ownership_error"
        attempt = AnchorSalienceAttempt(
            schedule_index=index,
            radius_pixels=radius,
            disk_footprint_digest=_mask_digest(disk, "euclidean-disk-footprint"),
            envelope_mask_digest=_mask_digest(envelope, "unfilled-envelope-mask"),
            graph=graph_artifact,
            support_counts=counts,
            acceptable=reason == "accepted",
            reason=reason,
        )
        attempts.append(attempt)
        if reason == "accepted":
            selected_index = index
            selected_ownership = ownership
            selected_counts = counts
            break
        if reason in ("graph_indeterminate", "graph_error", "ownership_error"):
            terminal_status = AnchorSalienceStatus(
                "indeterminate" if reason == "graph_indeterminate" else "error",
                {
                    "graph_indeterminate": "candidate_anchor_indeterminate",
                    "graph_error": "candidate_anchor_error",
                    "ownership_error": "ownership_error",
                }[reason],
            )
            break
    values["attempts"] = tuple(attempts)
    audit_radius = 5 * q
    audit_disk = _disk(audit_radius)
    audit_envelope = np.ascontiguousarray(
        ndimage.binary_dilation(padded, structure=audit_disk), dtype=bool
    )
    audit_graph = extract_object_scene_anchor_graph(
        audit_envelope, object_id, active_limits.anchor_graph_limits
    )
    values.update(
        audit_disk_footprint_digest=_mask_digest(
            audit_disk, "euclidean-disk-footprint"
        ),
        audit_envelope_mask_digest=_mask_digest(
            audit_envelope, "unfilled-envelope-mask"
        ),
        audit_graph=audit_graph,
    )
    if terminal_status is not None:
        values["status"] = terminal_status
        return _construct(values)
    if selected_index is None:
        values["status"] = AnchorSalienceStatus("indeterminate", "salience_cap_exceeded")
        return _construct(values)
    values.update(
        status=AnchorSalienceStatus("clean", "complete"),
        selected_attempt_index=selected_index,
        ownership=selected_ownership,
        selected_support_counts=selected_counts,
        raw_part_span_digest=canonical_digest(
            [
                {
                    "raw_point_id": item.raw_point_id,
                    "raw_owner_anchor_ids": list(item.raw_owner_anchor_ids),
                }
                for item in selected_ownership
            ]
        ),
    )
    return _construct(values)


def verify_object_scene_anchor_salience(
    artifact: ObjectSceneAnchorSalience,
    *,
    expected_mask: np.ndarray | None = None,
    expected_object_id: str | None = None,
) -> ObjectSceneAnchorSalience:
    """Verify canonical structure and optionally replay from the exact mask."""

    if type(artifact) is not ObjectSceneAnchorSalience:
        raise TypeError("artifact must be exact ObjectSceneAnchorSalience")
    restored = ObjectSceneAnchorSalience.from_data(artifact.to_data())
    if restored != artifact:
        raise ValueError("anchor salience is not canonical")
    if expected_object_id is not None and artifact.object_id != expected_object_id:
        raise ValueError("anchor salience object_id differs")
    if expected_mask is not None:
        replay = extract_object_scene_anchor_salience(
            expected_mask, artifact.object_id, artifact.limits
        )
        if replay != artifact:
            raise ValueError("anchor salience differs from exact-mask replay")
    return restored


__all__ = (
    "ANCHOR_SALIENCE_ALGORITHM_ID",
    "ANCHOR_SALIENCE_HARD_COMPLETE_CAP",
    "ANCHOR_SALIENCE_Q_DEFINITION",
    "ANCHOR_SALIENCE_RADIUS_RULE",
    "ANCHOR_SALIENCE_SCHEMA",
    "AnchorSalienceAttempt",
    "AnchorSalienceLimits",
    "AnchorSalienceStatus",
    "AnchorSupportCount",
    "ObjectSceneAnchorSalience",
    "RawSkeletonOwnership",
    "extract_object_scene_anchor_salience",
    "object_scene_anchor_salience_extractor_digest",
    "object_scene_anchor_salience_source_digest",
    "verify_object_scene_anchor_salience",
)
