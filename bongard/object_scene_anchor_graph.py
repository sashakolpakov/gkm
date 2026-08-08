"""Deterministic, bounded anchor graphs extracted from exact binary masks.

The graph is deliberately a pixel witness, not a semantic interpretation.  It
contracts individual endpoint pixels and junction pixel clusters in an
eight-neighbour thinned foreground graph, records every maximal intervening
path, retains compact source components that have no traceable skeleton edge,
and freezes the clockwise incident frame at every junction.  A resource cap
produces an explicit indeterminate artifact; it never masquerades as certified
absence.
"""

from __future__ import annotations

from bongard.runtime_source_snapshot import capture_loaded_source


_LOADED_SOURCE_SHA256 = capture_loaded_source(__name__, __file__)

from dataclasses import dataclass
from functools import cmp_to_key
import hashlib
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy import ndimage

from bongard.canonical import canonical_digest, canonical_json
from bongard.contour_witnesses import (
    _clusters,
    _pixel_graph,
    _zhang_suen,
    contour_witness_extractor_digest,
)
from bongard.prototype_visual_runtime import visual_runtime_dependency_digest


ANCHOR_GRAPH_SCHEMA = "gkm.object-scene-anchor-graph.v1"
ANCHOR_GRAPH_ALGORITHM_ID = "bongard.object-scene-anchor-graph/v1"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_TERMINAL_ID = re.compile(r"terminal-[0-9]{8}\Z")
_JOIN_ID = re.compile(r"join-[0-9]{8}\Z")
_PART_ID = re.compile(r"part-[0-9]{8}\Z")
_FRAME_ID = re.compile(r"frame-[0-9]{8}\Z")
_COMPACT_ID = re.compile(r"compact-[0-9]{8}\Z")
_NODE_ID = re.compile(r"(?:terminal|join)-[0-9]{8}\Z")
_STATUS_STATES = frozenset(("clean", "indeterminate", "error"))
_STATUS_REASONS = frozenset(
    (
        "complete",
        "skeleton_pixel_cap_exceeded",
        "terminal_cap_exceeded",
        "join_cap_exceeded",
        "compact_component_cap_exceeded",
        "part_cap_exceeded",
        "part_point_cap_exceeded",
        "unsupported_pixel_graph",
        "thinning_error",
    )
)
_COMPACT_REASONS = frozenset(
    ("isolated_skeleton_component", "source_component_thinned_empty")
)


def object_scene_anchor_graph_source_digest() -> str:
    """Return the loaded source identity, rejecting post-import mutation."""

    current = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if current != _LOADED_SOURCE_SHA256:
        raise RuntimeError("object scene anchor graph source changed after import")
    return _LOADED_SOURCE_SHA256


def object_scene_anchor_graph_extractor_digest() -> str:
    """Return the source- and thinning-dependency-bound extractor identity."""

    return canonical_digest(
        {
            "algorithm_id": ANCHOR_GRAPH_ALGORITHM_ID,
            "source_digest": object_scene_anchor_graph_source_digest(),
            "contour_witness_extractor_digest": contour_witness_extractor_digest(),
            "visual_runtime_dependency_digest": visual_runtime_dependency_digest(),
            "source_connectivity": 8,
            "skeleton": "zhang-suen/until-fixed-point",
            "pixel_graph": "8-neighbour/corner-diagonal-suppression",
            "semantic_nodes": {
                "terminal": "individual graph-degree-1 pixel",
                "join": (
                    "connected clusters of graph-degree-at-least-3 pixels;"
                    "then fixed-point lexicographic batches absorb every unowned "
                    "degree-2 pixel whose two neighbours have the same join owner"
                ),
                "compact": "source component with empty or degree-0 skeleton",
            },
            "paths": "all maximal paths after semantic-node contraction",
            "join_frames": "complete east-first clockwise incident half-edge order",
        }
    )


def _exact_fields(
    value: Mapping[str, Any], expected: frozenset[str], label: str
) -> None:
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


def _sequence(value: object, label: str) -> Sequence[object]:
    if not isinstance(value, list):
        raise TypeError(f"{label} must be a JSON list")
    return value


def _mask_digest(mask: np.ndarray, kind: str) -> str:
    height, width = mask.shape
    header = canonical_json(
        {
            "schema": "gkm.object-scene-anchor-binary-mask.v1",
            "kind": kind,
            "height_pixels": int(height),
            "width_pixels": int(width),
            "packing": "numpy.packbits-axis-none-bitorder-big",
        }
    )
    packed = np.packbits(mask.reshape(-1), bitorder="big").tobytes()
    return hashlib.sha256(header + b"\x00" + packed).hexdigest()


def _exact_bool_mask(mask: object) -> np.ndarray:
    if not isinstance(mask, np.ndarray):
        raise TypeError("mask must be an exact numpy bool array")
    if mask.dtype != np.dtype(bool):
        raise TypeError("mask must have exact bool dtype")
    if mask.ndim != 2:
        raise ValueError("mask must be two-dimensional")
    if mask.shape[0] < 1 or mask.shape[1] < 1:
        raise ValueError("mask dimensions must be nonzero")
    return np.ascontiguousarray(mask, dtype=bool)


def _q16_pixel_center(point: tuple[int, int], width: int, height: int) -> "Q16Point":
    y, x = point
    return Q16Point(
        ((2 * x + 1) * 65535 + width) // (2 * width),
        ((2 * y + 1) * 65535 + height) // (2 * height),
    )


def _q16_cluster_center(
    cluster: Iterable[tuple[int, int]], width: int, height: int
) -> "Q16Point":
    points = tuple(cluster)
    count = len(points)
    return Q16Point(
        (sum(2 * x + 1 for y, x in points) * 65535 + count * width)
        // (2 * count * width),
        (sum(2 * y + 1 for y, x in points) * 65535 + count * height)
        // (2 * count * height),
    )


@dataclass(frozen=True, order=True)
class Q16Point:
    """Unsigned Q16 location in the full source-mask coordinate frame."""

    x: int
    y: int

    def __post_init__(self) -> None:
        for label, value in (("x", self.x), ("y", self.y)):
            _integer(value, label)
            if value > 65535:
                raise ValueError(f"{label} exceeds unsigned Q16 range")

    def to_data(self) -> dict[str, int]:
        return {"x": self.x, "y": self.y}

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "Q16Point":
        _exact_fields(data, frozenset(("x", "y")), "Q16 point")
        return cls(data["x"], data["y"])


@dataclass(frozen=True)
class AnchorExtractionLimits:
    max_skeleton_pixels: int = 131_072
    max_terminals: int = 4_096
    max_joins: int = 4_096
    max_compact_components: int = 4_096
    max_parts: int = 8_192
    max_points_per_part: int = 131_072

    def __post_init__(self) -> None:
        for label, value in (
            ("max_skeleton_pixels", self.max_skeleton_pixels),
            ("max_terminals", self.max_terminals),
            ("max_joins", self.max_joins),
            ("max_compact_components", self.max_compact_components),
            ("max_parts", self.max_parts),
            ("max_points_per_part", self.max_points_per_part),
        ):
            _integer(value, label, minimum=1)

    def to_data(self) -> dict[str, int]:
        return {
            "max_skeleton_pixels": self.max_skeleton_pixels,
            "max_terminals": self.max_terminals,
            "max_joins": self.max_joins,
            "max_compact_components": self.max_compact_components,
            "max_parts": self.max_parts,
            "max_points_per_part": self.max_points_per_part,
        }

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "AnchorExtractionLimits":
        _exact_fields(
            data,
            frozenset(
                (
                    "max_skeleton_pixels",
                    "max_terminals",
                    "max_joins",
                    "max_compact_components",
                    "max_parts",
                    "max_points_per_part",
                )
            ),
            "anchor extraction limits",
        )
        return cls(**data)


@dataclass(frozen=True)
class AnchorExtractionStatus:
    state: str
    reason: str

    def __post_init__(self) -> None:
        if type(self.state) is not str or type(self.reason) is not str:
            raise TypeError("anchor extraction status fields must be exact strings")
        if self.state not in _STATUS_STATES:
            raise ValueError("unknown anchor extraction state")
        if self.reason not in _STATUS_REASONS:
            raise ValueError("unknown anchor extraction reason")
        if (self.state == "clean") != (self.reason == "complete"):
            raise ValueError("only a complete extraction may be clean")
        if self.state == "indeterminate" and not self.reason.endswith("cap_exceeded"):
            raise ValueError("indeterminate extraction must identify a resource cap")
        if self.state == "error" and self.reason.endswith("cap_exceeded"):
            raise ValueError("resource caps are indeterminate, not errors")

    def to_data(self) -> dict[str, str]:
        return {"state": self.state, "reason": self.reason}

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "AnchorExtractionStatus":
        _exact_fields(data, frozenset(("state", "reason")), "anchor status")
        return cls(data["state"], data["reason"])


@dataclass(frozen=True)
class AnchorTerminal:
    terminal_id: str
    location_q16: Q16Point
    incident_part_id: str

    def __post_init__(self) -> None:
        if type(self.terminal_id) is not str or _TERMINAL_ID.fullmatch(self.terminal_id) is None:
            raise ValueError("terminal_id is not canonical")
        if type(self.location_q16) is not Q16Point:
            raise TypeError("terminal location must be an exact Q16Point")
        if type(self.incident_part_id) is not str or _PART_ID.fullmatch(self.incident_part_id) is None:
            raise ValueError("terminal incident_part_id is not canonical")

    def to_data(self) -> dict[str, object]:
        return {
            "terminal_id": self.terminal_id,
            "location_q16": self.location_q16.to_data(),
            "incident_part_id": self.incident_part_id,
        }

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "AnchorTerminal":
        _exact_fields(
            data,
            frozenset(("terminal_id", "location_q16", "incident_part_id")),
            "anchor terminal",
        )
        location = data["location_q16"]
        if not isinstance(location, Mapping):
            raise TypeError("terminal location_q16 must be an object")
        return cls(data["terminal_id"], Q16Point.from_data(location), data["incident_part_id"])

    def digest(self) -> str:
        return canonical_digest(self.to_data())


@dataclass(frozen=True)
class AnchorCompactComponent:
    """A nonempty source component without a traceable skeleton edge."""

    compact_id: str
    location_q16: Q16Point
    bbox_min_q16: Q16Point
    bbox_max_q16: Q16Point
    foreground_pixel_count: int
    skeleton_pixel_count: int
    source_component_digest: str
    reason: str

    def __post_init__(self) -> None:
        if type(self.compact_id) is not str or _COMPACT_ID.fullmatch(self.compact_id) is None:
            raise ValueError("compact_id is not canonical")
        for label, value in (
            ("location_q16", self.location_q16),
            ("bbox_min_q16", self.bbox_min_q16),
            ("bbox_max_q16", self.bbox_max_q16),
        ):
            if type(value) is not Q16Point:
                raise TypeError(f"compact {label} must be an exact Q16Point")
        if self.bbox_min_q16.x > self.bbox_max_q16.x or self.bbox_min_q16.y > self.bbox_max_q16.y:
            raise ValueError("compact component bbox is inverted")
        _integer(self.foreground_pixel_count, "compact foreground_pixel_count", minimum=1)
        _integer(self.skeleton_pixel_count, "compact skeleton_pixel_count")
        _digest(self.source_component_digest, "source_component_digest")
        if self.reason not in _COMPACT_REASONS:
            raise ValueError("unknown compact component reason")
        if (self.reason == "source_component_thinned_empty") != (
            self.skeleton_pixel_count == 0
        ):
            raise ValueError("compact component reason disagrees with skeleton count")

    def to_data(self) -> dict[str, object]:
        return {
            "compact_id": self.compact_id,
            "location_q16": self.location_q16.to_data(),
            "bbox_min_q16": self.bbox_min_q16.to_data(),
            "bbox_max_q16": self.bbox_max_q16.to_data(),
            "foreground_pixel_count": self.foreground_pixel_count,
            "skeleton_pixel_count": self.skeleton_pixel_count,
            "source_component_digest": self.source_component_digest,
            "reason": self.reason,
        }

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "AnchorCompactComponent":
        _exact_fields(
            data,
            frozenset(
                (
                    "compact_id",
                    "location_q16",
                    "bbox_min_q16",
                    "bbox_max_q16",
                    "foreground_pixel_count",
                    "skeleton_pixel_count",
                    "source_component_digest",
                    "reason",
                )
            ),
            "anchor compact component",
        )
        points = (data["location_q16"], data["bbox_min_q16"], data["bbox_max_q16"])
        if any(not isinstance(item, Mapping) for item in points):
            raise TypeError("compact component points must be objects")
        return cls(
            compact_id=data["compact_id"],
            location_q16=Q16Point.from_data(points[0]),
            bbox_min_q16=Q16Point.from_data(points[1]),
            bbox_max_q16=Q16Point.from_data(points[2]),
            foreground_pixel_count=data["foreground_pixel_count"],
            skeleton_pixel_count=data["skeleton_pixel_count"],
            source_component_digest=data["source_component_digest"],
            reason=data["reason"],
        )

    def digest(self) -> str:
        return canonical_digest(self.to_data())


@dataclass(frozen=True)
class AnchorJoin:
    join_id: str
    location_q16: Q16Point
    incident_part_ids: tuple[str, ...]
    cyclic_frame_id: str

    def __post_init__(self) -> None:
        if type(self.join_id) is not str or _JOIN_ID.fullmatch(self.join_id) is None:
            raise ValueError("join_id is not canonical")
        if type(self.location_q16) is not Q16Point:
            raise TypeError("join location must be an exact Q16Point")
        if type(self.incident_part_ids) is not tuple or any(
            type(item) is not str for item in self.incident_part_ids
        ):
            raise TypeError("join incident_part_ids must be an exact tuple of strings")
        if len(self.incident_part_ids) < 3 or tuple(sorted(self.incident_part_ids)) != self.incident_part_ids:
            raise ValueError("join incident parts must be a sorted complete multiset of at least three")
        if any(_PART_ID.fullmatch(item) is None for item in self.incident_part_ids):
            raise ValueError("join incident_part_ids are not canonical")
        if type(self.cyclic_frame_id) is not str or _FRAME_ID.fullmatch(self.cyclic_frame_id) is None:
            raise ValueError("join cyclic_frame_id is not canonical")

    def to_data(self) -> dict[str, object]:
        return {
            "join_id": self.join_id,
            "location_q16": self.location_q16.to_data(),
            "incident_part_ids": list(self.incident_part_ids),
            "cyclic_frame_id": self.cyclic_frame_id,
        }

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "AnchorJoin":
        _exact_fields(
            data,
            frozenset(("join_id", "location_q16", "incident_part_ids", "cyclic_frame_id")),
            "anchor join",
        )
        location = data["location_q16"]
        parts = _sequence(data["incident_part_ids"], "join incident_part_ids")
        if not isinstance(location, Mapping) or any(not isinstance(item, str) for item in parts):
            raise TypeError("join location or incident part list has the wrong type")
        return cls(data["join_id"], Q16Point.from_data(location), tuple(parts), data["cyclic_frame_id"])

    def digest(self) -> str:
        return canonical_digest(self.to_data())


@dataclass(frozen=True)
class AnchorPart:
    part_id: str
    endpoint_node_ids: tuple[str, ...]
    path_q16: tuple[Q16Point, ...]
    closed: bool

    def __post_init__(self) -> None:
        if type(self.part_id) is not str or _PART_ID.fullmatch(self.part_id) is None:
            raise ValueError("part_id is not canonical")
        if type(self.endpoint_node_ids) is not tuple or any(
            type(item) is not str for item in self.endpoint_node_ids
        ):
            raise TypeError("part endpoint_node_ids must be an exact tuple of strings")
        if type(self.path_q16) is not tuple or any(
            type(item) is not Q16Point for item in self.path_q16
        ):
            raise TypeError("part path_q16 must be an exact tuple of Q16Point values")
        if len(self.endpoint_node_ids) not in (0, 2):
            raise ValueError("part must have zero or two endpoint node references")
        if any(_NODE_ID.fullmatch(item) is None for item in self.endpoint_node_ids):
            raise ValueError("part endpoint_node_ids are not canonical")
        if type(self.closed) is not bool:
            raise TypeError("part closed flag must be bool")
        if self.closed != (
            len(self.endpoint_node_ids) == 0
            or (
                len(self.endpoint_node_ids) == 2
                and self.endpoint_node_ids[0] == self.endpoint_node_ids[1]
            )
        ):
            raise ValueError("part closed flag disagrees with its endpoint nodes")
        if len(self.path_q16) < 2:
            raise ValueError("part path_q16 must contain at least two Q16 points")
        if self.endpoint_node_ids and self.endpoint_node_ids[0] != self.endpoint_node_ids[1]:
            if self.endpoint_node_ids[0] > self.endpoint_node_ids[1]:
                raise ValueError("open part endpoint node IDs must be canonicalized")

    def to_data(self) -> dict[str, object]:
        return {
            "part_id": self.part_id,
            "endpoint_node_ids": list(self.endpoint_node_ids),
            "path_q16": [item.to_data() for item in self.path_q16],
            "closed": self.closed,
        }

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "AnchorPart":
        _exact_fields(
            data,
            frozenset(("part_id", "endpoint_node_ids", "path_q16", "closed")),
            "anchor part",
        )
        endpoint_ids = _sequence(data["endpoint_node_ids"], "part endpoint_node_ids")
        path = _sequence(data["path_q16"], "part path_q16")
        if any(not isinstance(item, str) for item in endpoint_ids) or any(
            not isinstance(item, Mapping) for item in path
        ):
            raise TypeError("part endpoints or path have the wrong type")
        return cls(
            data["part_id"],
            tuple(endpoint_ids),
            tuple(Q16Point.from_data(item) for item in path),
            data["closed"],
        )

    def digest(self) -> str:
        return canonical_digest(self.to_data())


@dataclass(frozen=True)
class AnchorCyclicFrame:
    frame_id: str
    join_id: str
    clockwise_incident_part_ids: tuple[str, ...]
    clockwise_tangent_points_q16: tuple[Q16Point, ...]

    def __post_init__(self) -> None:
        if type(self.frame_id) is not str or _FRAME_ID.fullmatch(self.frame_id) is None:
            raise ValueError("frame_id is not canonical")
        if type(self.join_id) is not str or _JOIN_ID.fullmatch(self.join_id) is None:
            raise ValueError("frame join_id is not canonical")
        if type(self.clockwise_incident_part_ids) is not tuple or any(
            type(item) is not str for item in self.clockwise_incident_part_ids
        ):
            raise TypeError("frame incident parts must be an exact tuple of strings")
        if type(self.clockwise_tangent_points_q16) is not tuple or any(
            type(item) is not Q16Point for item in self.clockwise_tangent_points_q16
        ):
            raise TypeError("frame tangents must be an exact tuple of Q16Point values")
        if len(self.clockwise_incident_part_ids) < 3 or len(
            self.clockwise_incident_part_ids
        ) != len(self.clockwise_tangent_points_q16):
            raise ValueError("cyclic frame must contain every incident part and tangent")
        if any(_PART_ID.fullmatch(item) is None for item in self.clockwise_incident_part_ids):
            raise ValueError("frame incident part IDs are not canonical")

    def to_data(self) -> dict[str, object]:
        return {
            "frame_id": self.frame_id,
            "join_id": self.join_id,
            "clockwise_incident_part_ids": list(self.clockwise_incident_part_ids),
            "clockwise_tangent_points_q16": [
                item.to_data() for item in self.clockwise_tangent_points_q16
            ],
        }

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "AnchorCyclicFrame":
        _exact_fields(
            data,
            frozenset(
                (
                    "frame_id",
                    "join_id",
                    "clockwise_incident_part_ids",
                    "clockwise_tangent_points_q16",
                )
            ),
            "anchor cyclic frame",
        )
        parts = _sequence(data["clockwise_incident_part_ids"], "frame part IDs")
        tangents = _sequence(data["clockwise_tangent_points_q16"], "frame tangents")
        if any(not isinstance(item, str) for item in parts) or any(
            not isinstance(item, Mapping) for item in tangents
        ):
            raise TypeError("frame part IDs or tangents have the wrong type")
        return cls(
            data["frame_id"],
            data["join_id"],
            tuple(parts),
            tuple(Q16Point.from_data(item) for item in tangents),
        )

    def digest(self) -> str:
        return canonical_digest(self.to_data())


@dataclass(frozen=True)
class ObjectSceneAnchorGraph:
    object_id: str
    extractor_artifact_digest: str
    mask_height_pixels: int
    mask_width_pixels: int
    foreground_pixel_count: int
    mask_digest: str
    skeleton_digest: str
    skeleton_pixel_count: int
    limits: AnchorExtractionLimits
    status: AnchorExtractionStatus
    terminals: tuple[AnchorTerminal, ...]
    joins: tuple[AnchorJoin, ...]
    compact_components: tuple[AnchorCompactComponent, ...]
    parts: tuple[AnchorPart, ...]
    cyclic_frames: tuple[AnchorCyclicFrame, ...]
    artifact_digest: str

    def __post_init__(self) -> None:
        if type(self.object_id) is not str or not self.object_id or len(self.object_id) > 256 or any(
            ord(char) < 32 for char in self.object_id
        ):
            raise ValueError("object_id must be a nonempty bounded printable string")
        _digest(self.extractor_artifact_digest, "extractor_artifact_digest")
        if self.extractor_artifact_digest != object_scene_anchor_graph_extractor_digest():
            raise ValueError("anchor graph extractor artifact digest is stale")
        _integer(self.mask_height_pixels, "mask_height_pixels", minimum=1)
        _integer(self.mask_width_pixels, "mask_width_pixels", minimum=1)
        _integer(self.foreground_pixel_count, "foreground_pixel_count")
        _integer(self.skeleton_pixel_count, "skeleton_pixel_count")
        _digest(self.mask_digest, "mask_digest")
        _digest(self.skeleton_digest, "skeleton_digest")
        _digest(self.artifact_digest, "artifact_digest")
        if type(self.limits) is not AnchorExtractionLimits or type(
            self.status
        ) is not AnchorExtractionStatus:
            raise TypeError("anchor graph limits and status have the wrong type")
        collection_specs = (
            ("terminals", self.terminals, AnchorTerminal),
            ("joins", self.joins, AnchorJoin),
            ("compact_components", self.compact_components, AnchorCompactComponent),
            ("parts", self.parts, AnchorPart),
            ("cyclic_frames", self.cyclic_frames, AnchorCyclicFrame),
        )
        for label, collection, member_type in collection_specs:
            if type(collection) is not tuple or any(
                type(item) is not member_type for item in collection
            ):
                raise TypeError(f"graph {label} must be an exact tuple of exact members")
        expected_ids = {
            "terminal": tuple(f"terminal-{index:08d}" for index in range(len(self.terminals))),
            "join": tuple(f"join-{index:08d}" for index in range(len(self.joins))),
            "compact": tuple(
                f"compact-{index:08d}" for index in range(len(self.compact_components))
            ),
            "part": tuple(f"part-{index:08d}" for index in range(len(self.parts))),
            "frame": tuple(f"frame-{index:08d}" for index in range(len(self.cyclic_frames))),
        }
        if tuple(item.terminal_id for item in self.terminals) != expected_ids["terminal"]:
            raise ValueError("terminal order or IDs are not canonical")
        if tuple(item.join_id for item in self.joins) != expected_ids["join"]:
            raise ValueError("join order or IDs are not canonical")
        if tuple(item.compact_id for item in self.compact_components) != expected_ids["compact"]:
            raise ValueError("compact component order or IDs are not canonical")
        if tuple(item.part_id for item in self.parts) != expected_ids["part"]:
            raise ValueError("part order or IDs are not canonical")
        if tuple(item.frame_id for item in self.cyclic_frames) != expected_ids["frame"]:
            raise ValueError("frame order or IDs are not canonical")
        if self.status.state != "clean":
            if (
                self.terminals
                or self.joins
                or self.compact_components
                or self.parts
                or self.cyclic_frames
            ):
                raise ValueError("non-clean extraction must not expose a partial anchor graph")
        else:
            self._validate_clean_graph()
        if self.artifact_digest != canonical_digest(self._unsigned_data()):
            raise ValueError("anchor graph artifact digest does not match its content")

    def _validate_clean_graph(self) -> None:
        parts = {item.part_id: item for item in self.parts}
        terminals = {item.terminal_id: item for item in self.terminals}
        joins = {item.join_id: item for item in self.joins}
        frames = {item.frame_id: item for item in self.cyclic_frames}
        if len(frames) != len(joins):
            raise ValueError("clean graph requires exactly one cyclic frame per join")
        incidence: dict[str, list[str]] = {**{key: [] for key in terminals}, **{key: [] for key in joins}}
        for part in self.parts:
            for node_id in part.endpoint_node_ids:
                if node_id not in incidence:
                    raise ValueError("part references an unknown endpoint node")
                incidence[node_id].append(part.part_id)
        for terminal in self.terminals:
            if incidence[terminal.terminal_id] != [terminal.incident_part_id] or terminal.incident_part_id not in parts:
                raise ValueError("terminal incidence is incomplete")
        for join in self.joins:
            observed = tuple(sorted(incidence[join.join_id]))
            if observed != join.incident_part_ids:
                raise ValueError("join incidence is incomplete")
            frame = frames.get(join.cyclic_frame_id)
            if frame is None or frame.join_id != join.join_id or tuple(
                sorted(frame.clockwise_incident_part_ids)
            ) != join.incident_part_ids:
                raise ValueError("join cyclic frame is incomplete")

    def _unsigned_data(self) -> dict[str, object]:
        return {
            "schema": ANCHOR_GRAPH_SCHEMA,
            "algorithm_id": ANCHOR_GRAPH_ALGORITHM_ID,
            "object_id": self.object_id,
            "extractor_artifact_digest": self.extractor_artifact_digest,
            "mask_height_pixels": self.mask_height_pixels,
            "mask_width_pixels": self.mask_width_pixels,
            "foreground_pixel_count": self.foreground_pixel_count,
            "mask_digest": self.mask_digest,
            "skeleton_digest": self.skeleton_digest,
            "skeleton_pixel_count": self.skeleton_pixel_count,
            "limits": self.limits.to_data(),
            "status": self.status.to_data(),
            "terminals": [item.to_data() for item in self.terminals],
            "joins": [item.to_data() for item in self.joins],
            "compact_components": [
                item.to_data() for item in self.compact_components
            ],
            "parts": [item.to_data() for item in self.parts],
            "cyclic_frames": [item.to_data() for item in self.cyclic_frames],
        }

    def to_data(self) -> dict[str, object]:
        return {**self._unsigned_data(), "artifact_digest": self.artifact_digest}

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "ObjectSceneAnchorGraph":
        expected = frozenset(
            (
                "schema",
                "algorithm_id",
                "object_id",
                "extractor_artifact_digest",
                "mask_height_pixels",
                "mask_width_pixels",
                "foreground_pixel_count",
                "mask_digest",
                "skeleton_digest",
                "skeleton_pixel_count",
                "limits",
                "status",
                "terminals",
                "joins",
                "compact_components",
                "parts",
                "cyclic_frames",
                "artifact_digest",
            )
        )
        _exact_fields(data, expected, "object scene anchor graph")
        if data["schema"] != ANCHOR_GRAPH_SCHEMA or data["algorithm_id"] != ANCHOR_GRAPH_ALGORITHM_ID:
            raise ValueError("unsupported object scene anchor graph schema or algorithm")
        limits, status = data["limits"], data["status"]
        if not isinstance(limits, Mapping) or not isinstance(status, Mapping):
            raise TypeError("anchor limits and status must be objects")

        def records(name: str, record_type: type[Any]) -> tuple[Any, ...]:
            raw = _sequence(data[name], name)
            if any(not isinstance(item, Mapping) for item in raw):
                raise TypeError(f"{name} must contain objects")
            return tuple(record_type.from_data(item) for item in raw)

        return cls(
            object_id=data["object_id"],
            extractor_artifact_digest=data["extractor_artifact_digest"],
            mask_height_pixels=data["mask_height_pixels"],
            mask_width_pixels=data["mask_width_pixels"],
            foreground_pixel_count=data["foreground_pixel_count"],
            mask_digest=data["mask_digest"],
            skeleton_digest=data["skeleton_digest"],
            skeleton_pixel_count=data["skeleton_pixel_count"],
            limits=AnchorExtractionLimits.from_data(limits),
            status=AnchorExtractionStatus.from_data(status),
            terminals=records("terminals", AnchorTerminal),
            joins=records("joins", AnchorJoin),
            compact_components=records(
                "compact_components", AnchorCompactComponent
            ),
            parts=records("parts", AnchorPart),
            cyclic_frames=records("cyclic_frames", AnchorCyclicFrame),
            artifact_digest=data["artifact_digest"],
        )

    def digest(self) -> str:
        return self.artifact_digest


def _vertex_key(value: tuple[object, ...]) -> tuple[object, ...]:
    order = {"terminal": 0, "join": 1, "pixel": 2}
    return (order[str(value[0])],) + value[1:]


def _edge(first: tuple[object, ...], second: tuple[object, ...]) -> tuple[tuple[object, ...], tuple[object, ...]]:
    return (first, second) if _vertex_key(first) <= _vertex_key(second) else (second, first)


def _node_id(vertex: tuple[object, ...]) -> str:
    return f"{vertex[0]}-{int(vertex[1]):08d}"


def _absorb_join_reentry_pixels(
    graph: Mapping[tuple[int, int], tuple[tuple[int, int], ...]],
    join_clusters: list[set[tuple[int, int]]],
) -> None:
    """Absorb degree-two pixels whose two half-edges re-enter one join.

    Contracted adjacency is intentionally simple rather than a multigraph.  A
    one-pixel raster pocket can otherwise place two boundary half-edges between
    the same join and pixel, which a set would collapse.  Absorbing precisely
    those degree-two re-entry pixels removes only the microscopic pocket and
    preserves every path that actually exits the join.
    """

    owner = {
        point: index for index, cluster in enumerate(join_clusters) for point in cluster
    }
    while True:
        additions: list[tuple[tuple[int, int], int]] = []
        for point in sorted(graph):
            if point in owner or len(graph[point]) != 2:
                continue
            neighbour_owners = [owner.get(neighbour) for neighbour in graph[point]]
            if neighbour_owners[0] is not None and neighbour_owners[0] == neighbour_owners[1]:
                additions.append((point, neighbour_owners[0]))
        if not additions:
            return
        for point, index in additions:
            join_clusters[index].add(point)
            owner[point] = index


def _compact_component_records(
    mask: np.ndarray,
    skeleton: np.ndarray,
    graph: Mapping[tuple[int, int], tuple[tuple[int, int], ...]],
) -> tuple[AnchorCompactComponent, ...]:
    height, width = mask.shape
    labels, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=bool))
    raw: list[
        tuple[
            tuple[object, ...],
            Q16Point,
            Q16Point,
            Q16Point,
            int,
            int,
            str,
            str,
        ]
    ] = []
    for label in range(1, count + 1):
        component = labels == label
        source_points = [(int(y), int(x)) for y, x in np.argwhere(component)]
        skeleton_points = [point for point in source_points if skeleton[point]]
        isolated_points = [point for point in skeleton_points if len(graph[point]) == 0]
        if skeleton_points and not isolated_points:
            continue
        ys, xs = np.nonzero(component)
        x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
        bbox_min = Q16Point(
            (x0 * 65535 + width // 2) // width,
            (y0 * 65535 + height // 2) // height,
        )
        bbox_max = Q16Point(
            (x1 * 65535 + width // 2) // width,
            (y1 * 65535 + height // 2) // height,
        )
        source_digest = _mask_digest(component, "source-connected-component")
        anchors: list[tuple[Q16Point, int, str]]
        if not skeleton_points:
            anchors = [
                (
                    _q16_cluster_center(source_points, width, height),
                    0,
                    "source_component_thinned_empty",
                )
            ]
        else:
            anchors = [
                (
                    _q16_pixel_center(point, width, height),
                    1,
                    "isolated_skeleton_component",
                )
                for point in sorted(isolated_points)
            ]
        for location, skeleton_count, reason in anchors:
            key = (
                y0,
                x0,
                y1,
                x1,
                location.y,
                location.x,
                source_digest,
                reason,
            )
            raw.append(
                (
                    key,
                    location,
                    bbox_min,
                    bbox_max,
                    len(source_points),
                    skeleton_count,
                    source_digest,
                    reason,
                )
            )
    raw.sort(key=lambda item: item[0])
    return tuple(
        AnchorCompactComponent(
            compact_id=f"compact-{index:08d}",
            location_q16=item[1],
            bbox_min_q16=item[2],
            bbox_max_q16=item[3],
            foreground_pixel_count=item[4],
            skeleton_pixel_count=item[5],
            source_component_digest=item[6],
            reason=item[7],
        )
        for index, item in enumerate(raw)
    )


def _canonical_cycle(
    adjacency: Mapping[tuple[object, ...], set[tuple[object, ...]]],
    seed: tuple[object, ...],
) -> tuple[tuple[object, ...], ...]:
    candidates: list[tuple[tuple[object, ...], ...]] = []
    for first in sorted(adjacency[seed], key=_vertex_key):
        path = [seed, first]
        previous, current = seed, first
        while current != seed:
            following = [item for item in adjacency[current] if item != previous]
            if len(following) != 1:
                raise RuntimeError("unsupported_pixel_graph")
            previous, current = current, following[0]
            if current != seed:
                path.append(current)
            if len(path) > len(adjacency):
                raise RuntimeError("unsupported_pixel_graph")
        candidates.append(tuple(path))
    return min(candidates, key=lambda path: tuple(_vertex_key(item) for item in path))


def _trace_parts(
    graph: Mapping[tuple[int, int], tuple[tuple[int, int], ...]],
    terminal_clusters: list[set[tuple[int, int]]],
    join_clusters: list[set[tuple[int, int]]],
) -> tuple[
    list[tuple[tuple[str, ...], tuple[tuple[object, ...], ...], bool]],
    dict[tuple[object, ...], set[tuple[object, ...]]],
]:
    owner: dict[tuple[int, int], tuple[object, ...]] = {}
    for index, cluster in enumerate(terminal_clusters):
        for point in cluster:
            owner[point] = ("terminal", index)
    for index, cluster in enumerate(join_clusters):
        for point in cluster:
            owner[point] = ("join", index)
    for point in graph:
        owner.setdefault(point, ("pixel", point[0], point[1]))

    vertices = set(owner.values())
    adjacency: dict[tuple[object, ...], set[tuple[object, ...]]] = {
        vertex: set() for vertex in vertices
    }
    for point, neighbours in graph.items():
        first = owner[point]
        for neighbour in neighbours:
            second = owner[neighbour]
            if first != second:
                adjacency[first].add(second)
                adjacency[second].add(first)

    semantic = {item for item in vertices if item[0] != "pixel"}
    used: set[tuple[tuple[object, ...], tuple[object, ...]]] = set()
    raw: list[tuple[tuple[str, ...], tuple[tuple[object, ...], ...], bool]] = []
    for start in sorted(semantic, key=_vertex_key):
        for neighbour in sorted(adjacency[start], key=_vertex_key):
            if _edge(start, neighbour) in used:
                continue
            path = [start, neighbour]
            used.add(_edge(start, neighbour))
            previous, current = start, neighbour
            while current not in semantic:
                following = [item for item in adjacency[current] if item != previous]
                if len(following) != 1:
                    raise RuntimeError("unsupported_pixel_graph")
                next_vertex = following[0]
                used.add(_edge(current, next_vertex))
                path.append(next_vertex)
                previous, current = current, next_vertex
                if len(path) > len(vertices) + 1:
                    raise RuntimeError("unsupported_pixel_graph")
            first_id, last_id = _node_id(path[0]), _node_id(path[-1])
            if first_id > last_id:
                path.reverse()
                first_id, last_id = last_id, first_id
            raw.append(((first_id, last_id), tuple(path), first_id == last_id))

    remaining_edges = {
        _edge(vertex, neighbour)
        for vertex, neighbours in adjacency.items()
        for neighbour in neighbours
    } - used
    while remaining_edges:
        seed = min((point for edge in remaining_edges for point in edge), key=_vertex_key)
        if seed in semantic:
            raise RuntimeError("unsupported_pixel_graph")
        cycle = _canonical_cycle(adjacency, seed)
        cycle_edges = {
            _edge(cycle[index], cycle[(index + 1) % len(cycle)])
            for index in range(len(cycle))
        }
        if not cycle_edges <= remaining_edges:
            raise RuntimeError("unsupported_pixel_graph")
        remaining_edges.difference_update(cycle_edges)
        raw.append(((), cycle, True))

    raw.sort(
        key=lambda item: (
            item[0],
            tuple(_vertex_key(vertex) for vertex in item[1]),
            item[2],
        )
    )
    return raw, adjacency


def _clockwise_compare(
    first: tuple[int, int, str, int, Q16Point],
    second: tuple[int, int, str, int, Q16Point],
) -> int:
    ax, ay, apart, aend, _ = first
    bx, by, bpart, bend, _ = second
    # Canvas y grows down, so increasing polar angle is clockwise.  Divide at
    # the negative-x ray to obtain a deterministic east-first cyclic frame.
    ahalf = 0 if ay > 0 or (ay == 0 and ax >= 0) else 1
    bhalf = 0 if by > 0 or (by == 0 and bx >= 0) else 1
    if ahalf != bhalf:
        return -1 if ahalf < bhalf else 1
    cross = ax * by - ay * bx
    if cross:
        return -1 if cross > 0 else 1
    atie, btie = (apart, aend, ax, ay), (bpart, bend, bx, by)
    return -1 if atie < btie else (1 if atie > btie else 0)


def _graph_with_status(
    *,
    object_id: str,
    mask: np.ndarray,
    skeleton: np.ndarray,
    limits: AnchorExtractionLimits,
    status: AnchorExtractionStatus,
    terminals: tuple[AnchorTerminal, ...] = (),
    joins: tuple[AnchorJoin, ...] = (),
    compact_components: tuple[AnchorCompactComponent, ...] = (),
    parts: tuple[AnchorPart, ...] = (),
    frames: tuple[AnchorCyclicFrame, ...] = (),
) -> ObjectSceneAnchorGraph:
    unsigned = {
        "schema": ANCHOR_GRAPH_SCHEMA,
        "algorithm_id": ANCHOR_GRAPH_ALGORITHM_ID,
        "object_id": object_id,
        "extractor_artifact_digest": object_scene_anchor_graph_extractor_digest(),
        "mask_height_pixels": int(mask.shape[0]),
        "mask_width_pixels": int(mask.shape[1]),
        "foreground_pixel_count": int(np.count_nonzero(mask)),
        "mask_digest": _mask_digest(mask, "source-mask"),
        "skeleton_digest": _mask_digest(skeleton, "zhang-suen-skeleton"),
        "skeleton_pixel_count": int(np.count_nonzero(skeleton)),
        "limits": limits.to_data(),
        "status": status.to_data(),
        "terminals": [item.to_data() for item in terminals],
        "joins": [item.to_data() for item in joins],
        "compact_components": [item.to_data() for item in compact_components],
        "parts": [item.to_data() for item in parts],
        "cyclic_frames": [item.to_data() for item in frames],
    }
    return ObjectSceneAnchorGraph(
        object_id=object_id,
        extractor_artifact_digest=unsigned["extractor_artifact_digest"],
        mask_height_pixels=int(mask.shape[0]),
        mask_width_pixels=int(mask.shape[1]),
        foreground_pixel_count=int(np.count_nonzero(mask)),
        mask_digest=unsigned["mask_digest"],
        skeleton_digest=unsigned["skeleton_digest"],
        skeleton_pixel_count=int(np.count_nonzero(skeleton)),
        limits=limits,
        status=status,
        terminals=terminals,
        joins=joins,
        compact_components=compact_components,
        parts=parts,
        cyclic_frames=frames,
        artifact_digest=canonical_digest(unsigned),
    )


def extract_object_scene_anchor_graph(
    mask: np.ndarray,
    object_id: str,
    limits: AnchorExtractionLimits | None = None,
) -> ObjectSceneAnchorGraph:
    """Extract a complete anchor graph or an explicit non-clean artifact."""

    exact = _exact_bool_mask(mask)
    active_limits = limits if limits is not None else AnchorExtractionLimits()
    if type(active_limits) is not AnchorExtractionLimits:
        raise TypeError("limits must be AnchorExtractionLimits")
    # Validate object_id before entering the extraction failure boundary.
    if type(object_id) is not str or not object_id or len(object_id) > 256 or any(
        ord(char) < 32 for char in object_id
    ):
        raise ValueError("object_id must be a nonempty bounded printable string")
    try:
        skeleton = _zhang_suen(exact)
    except RuntimeError:
        skeleton = np.zeros_like(exact)
        return _graph_with_status(
            object_id=object_id,
            mask=exact,
            skeleton=skeleton,
            limits=active_limits,
            status=AnchorExtractionStatus("error", "thinning_error"),
        )
    skeleton_count = int(np.count_nonzero(skeleton))
    if skeleton_count > active_limits.max_skeleton_pixels:
        return _graph_with_status(
            object_id=object_id,
            mask=exact,
            skeleton=skeleton,
            limits=active_limits,
            status=AnchorExtractionStatus("indeterminate", "skeleton_pixel_cap_exceeded"),
        )
    graph = _pixel_graph(skeleton)
    compact_components = _compact_component_records(exact, skeleton, graph)
    terminal_clusters = [
        {point} for point, neighbours in graph.items() if len(neighbours) == 1
    ]
    join_clusters = _clusters(
        {point for point, neighbours in graph.items() if len(neighbours) >= 3}
    )
    terminal_clusters.sort(key=lambda item: tuple(sorted(item)))
    join_clusters.sort(key=lambda item: tuple(sorted(item)))
    _absorb_join_reentry_pixels(graph, join_clusters)
    join_clusters.sort(key=lambda item: tuple(sorted(item)))
    if len(terminal_clusters) > active_limits.max_terminals:
        reason = "terminal_cap_exceeded"
    elif len(join_clusters) > active_limits.max_joins:
        reason = "join_cap_exceeded"
    elif len(compact_components) > active_limits.max_compact_components:
        reason = "compact_component_cap_exceeded"
    else:
        reason = ""
    if reason:
        return _graph_with_status(
            object_id=object_id,
            mask=exact,
            skeleton=skeleton,
            limits=active_limits,
            status=AnchorExtractionStatus("indeterminate", reason),
        )
    try:
        raw_parts, _ = _trace_parts(graph, terminal_clusters, join_clusters)
    except RuntimeError:
        return _graph_with_status(
            object_id=object_id,
            mask=exact,
            skeleton=skeleton,
            limits=active_limits,
            status=AnchorExtractionStatus("error", "unsupported_pixel_graph"),
        )
    if len(raw_parts) > active_limits.max_parts:
        return _graph_with_status(
            object_id=object_id,
            mask=exact,
            skeleton=skeleton,
            limits=active_limits,
            status=AnchorExtractionStatus("indeterminate", "part_cap_exceeded"),
        )
    if any(len(item[1]) > active_limits.max_points_per_part for item in raw_parts):
        return _graph_with_status(
            object_id=object_id,
            mask=exact,
            skeleton=skeleton,
            limits=active_limits,
            status=AnchorExtractionStatus("indeterminate", "part_point_cap_exceeded"),
        )

    height, width = exact.shape
    terminal_locations = tuple(
        _q16_cluster_center(cluster, width, height) for cluster in terminal_clusters
    )
    join_locations = tuple(
        _q16_cluster_center(cluster, width, height) for cluster in join_clusters
    )

    def location(vertex: tuple[object, ...]) -> Q16Point:
        if vertex[0] == "terminal":
            return terminal_locations[int(vertex[1])]
        if vertex[0] == "join":
            return join_locations[int(vertex[1])]
        return _q16_pixel_center((int(vertex[1]), int(vertex[2])), width, height)

    parts = tuple(
        AnchorPart(
            part_id=f"part-{index:08d}",
            endpoint_node_ids=endpoint_ids,
            path_q16=tuple(location(vertex) for vertex in path),
            closed=closed,
        )
        for index, (endpoint_ids, path, closed) in enumerate(raw_parts)
    )
    incidence: dict[str, list[tuple[str, int, Q16Point]]] = {
        **{f"terminal-{index:08d}": [] for index in range(len(terminal_clusters))},
        **{f"join-{index:08d}": [] for index in range(len(join_clusters))},
    }
    for part in parts:
        if not part.endpoint_node_ids:
            continue
        incidence[part.endpoint_node_ids[0]].append((part.part_id, 0, part.path_q16[1]))
        incidence[part.endpoint_node_ids[1]].append((part.part_id, 1, part.path_q16[-2]))
    if any(len(incidence[f"terminal-{index:08d}"]) != 1 for index in range(len(terminal_clusters))) or any(
        len(incidence[f"join-{index:08d}"]) < 3 for index in range(len(join_clusters))
    ):
        return _graph_with_status(
            object_id=object_id,
            mask=exact,
            skeleton=skeleton,
            limits=active_limits,
            status=AnchorExtractionStatus("error", "unsupported_pixel_graph"),
        )
    terminals = tuple(
        AnchorTerminal(
            terminal_id=f"terminal-{index:08d}",
            location_q16=terminal_locations[index],
            incident_part_id=incidence[f"terminal-{index:08d}"][0][0],
        )
        for index in range(len(terminal_clusters))
    )
    frames_list: list[AnchorCyclicFrame] = []
    joins_list: list[AnchorJoin] = []
    for index, center in enumerate(join_locations):
        join_id = f"join-{index:08d}"
        rays = [
            (
                tangent.x - center.x,
                tangent.y - center.y,
                part_id,
                end_index,
                tangent,
            )
            for part_id, end_index, tangent in incidence[join_id]
        ]
        rays.sort(key=cmp_to_key(_clockwise_compare))
        frame_id = f"frame-{index:08d}"
        frames_list.append(
            AnchorCyclicFrame(
                frame_id=frame_id,
                join_id=join_id,
                clockwise_incident_part_ids=tuple(item[2] for item in rays),
                clockwise_tangent_points_q16=tuple(item[4] for item in rays),
            )
        )
        joins_list.append(
            AnchorJoin(
                join_id=join_id,
                location_q16=center,
                incident_part_ids=tuple(sorted(item[0] for item in incidence[join_id])),
                cyclic_frame_id=frame_id,
            )
        )
    return _graph_with_status(
        object_id=object_id,
        mask=exact,
        skeleton=skeleton,
        limits=active_limits,
        status=AnchorExtractionStatus("clean", "complete"),
        terminals=terminals,
        joins=tuple(joins_list),
        compact_components=compact_components,
        parts=parts,
        frames=tuple(frames_list),
    )


def verify_object_scene_anchor_graph(
    graph: ObjectSceneAnchorGraph,
    *,
    expected_mask: np.ndarray | None = None,
    expected_object_id: str | None = None,
) -> ObjectSceneAnchorGraph:
    """Verify canonical structure and, when supplied, exact-mask replay."""

    if type(graph) is not ObjectSceneAnchorGraph:
        raise TypeError("graph must be an exact ObjectSceneAnchorGraph")
    canonical = ObjectSceneAnchorGraph.from_data(graph.to_data())
    if canonical != graph:
        raise ValueError("anchor graph is not canonical")
    if expected_object_id is not None and canonical.object_id != expected_object_id:
        raise ValueError("anchor graph object_id differs from the expected object")
    if expected_mask is not None:
        replay = extract_object_scene_anchor_graph(
            expected_mask, canonical.object_id, canonical.limits
        )
        if replay != canonical:
            raise ValueError("anchor graph differs from exact mask replay")
    return canonical


__all__ = (
    "ANCHOR_GRAPH_ALGORITHM_ID",
    "ANCHOR_GRAPH_SCHEMA",
    "AnchorCompactComponent",
    "AnchorCyclicFrame",
    "AnchorExtractionLimits",
    "AnchorExtractionStatus",
    "AnchorJoin",
    "AnchorPart",
    "AnchorTerminal",
    "ObjectSceneAnchorGraph",
    "Q16Point",
    "extract_object_scene_anchor_graph",
    "object_scene_anchor_graph_extractor_digest",
    "object_scene_anchor_graph_source_digest",
    "verify_object_scene_anchor_graph",
)
