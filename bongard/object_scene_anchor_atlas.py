"""Deterministic grayscale atlas for one selected object-anchor graph.

The renderer accepts only a frozen :class:`ObjectSceneAnchorDecisionManifest`.
It never reads panel pixels, full salience, raw graphs, audit graphs, or
raw-skeleton ownership.
Clean decisions render an exhaustive fixed-grid atlas; non-clean decisions
produce a typed gap and no PNG bytes.
"""

from __future__ import annotations

from bongard.runtime_source_snapshot import capture_loaded_source, verify_loaded_source


_LOADED_SOURCE_SHA256 = capture_loaded_source(__name__, __file__)

from dataclasses import dataclass
import binascii
import hashlib
import re
from typing import Any, Mapping

import numpy as np

from bongard.canonical import canonical_digest
from bongard.object_scene_anchor_catalog import (
    OBJECT_SCENE_ANCHOR_DECISION_MANIFEST_SCHEMA,
    ObjectSceneAnchorDecisionManifest,
)
from bongard.object_scene_anchor_graph import (
    AnchorCompactComponent,
    AnchorCyclicFrame,
    AnchorJoin,
    AnchorPart,
    ObjectSceneAnchorGraph,
    Q16Point,
)
from bongard.object_scene_anchor_salience import (
    ANCHOR_SALIENCE_HARD_COMPLETE_CAP,
    AnchorSalienceStatus,
)
from bongard.python_predicate_authority import PYTHON_PREDICATE_AUTHORITY_ID


OBJECT_SCENE_ANCHOR_ATLAS_SCHEMA = "gkm.object-scene-anchor-atlas.v1"
OBJECT_SCENE_ANCHOR_ATLAS_SLOT_SCHEMA = "gkm.object-scene-anchor-atlas-slot.v1"
OBJECT_SCENE_ANCHOR_ATLAS_STATUS_SCHEMA = (
    "gkm.object-scene-anchor-atlas-status.v1"
)
OBJECT_SCENE_ANCHOR_ATLAS_ALGORITHM_ID = (
    "bongard.object-scene-anchor-atlas/selected-graph-q16-grayscale-v1"
)

OBJECT_SCENE_ANCHOR_ATLAS_MAX_SLOTS = (
    1 + 2 * ANCHOR_SALIENCE_HARD_COMPLETE_CAP
)
OBJECT_SCENE_ANCHOR_ATLAS_COLUMNS = 5
OBJECT_SCENE_ANCHOR_ATLAS_ROWS = 4
OBJECT_SCENE_ANCHOR_ATLAS_TILE_SIZE_PIXELS = 128
OBJECT_SCENE_ANCHOR_ATLAS_WIDTH_PIXELS = (
    OBJECT_SCENE_ANCHOR_ATLAS_COLUMNS * OBJECT_SCENE_ANCHOR_ATLAS_TILE_SIZE_PIXELS
)
OBJECT_SCENE_ANCHOR_ATLAS_HEIGHT_PIXELS = (
    OBJECT_SCENE_ANCHOR_ATLAS_ROWS * OBJECT_SCENE_ANCHOR_ATLAS_TILE_SIZE_PIXELS
)
OBJECT_SCENE_ANCHOR_ATLAS_MODE = "L"

_TILE_MARGIN_PIXELS = 10
_TILE_BORDER_PIXELS = 2
_BACKGROUND_VALUE = 255
_UNUSED_BORDER_VALUE = 232
_USED_BORDER_VALUE = 144
_BASE_GRAPH_VALUE = 188
_HIGHLIGHT_VALUE = 16
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_PNG_CHUNK_MAX_PAYLOAD_BYTE_COUNT = 0xFFFFFFFF

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_OBJECT_ID = re.compile(r"object_[0-9]{4}\Z")
_PART_ID = re.compile(r"part-[0-9]{8}\Z")
_COMPACT_ID = re.compile(r"compact-[0-9]{8}\Z")
_JOIN_ID = re.compile(r"join-[0-9]{8}\Z")
_FRAME_ID = re.compile(r"frame-[0-9]{8}\Z")
_SLOT_ID = re.compile(r"slot-[0-9]{4}\Z")

if OBJECT_SCENE_ANCHOR_ATLAS_MAX_SLOTS != 17:
    raise RuntimeError("anchor atlas hard-cap derivation drifted")
if (
    OBJECT_SCENE_ANCHOR_ATLAS_COLUMNS * OBJECT_SCENE_ANCHOR_ATLAS_ROWS
    < OBJECT_SCENE_ANCHOR_ATLAS_MAX_SLOTS
):
    raise RuntimeError("anchor atlas grid cannot hold its exhaustive hard cap")


class ObjectSceneAnchorAtlasError(ValueError):
    """The selected-anchor atlas or its exact replay binding is invalid."""


def _exact_fields(
    value: object, expected: set[str] | frozenset[str], label: str
) -> Mapping[str, Any]:
    if (
        not isinstance(value, Mapping)
        or any(not isinstance(key, str) for key in value)
        or set(value) != set(expected)
    ):
        raise ObjectSceneAnchorAtlasError(f"{label} fields differ from schema")
    return value


def _integer(value: object, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ObjectSceneAnchorAtlasError(
            f"{label} must be an integer >= {minimum}"
        )
    return value


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ObjectSceneAnchorAtlasError(f"{label} must be a lowercase SHA-256")
    return value


def _assert_python_only_keys(value: object) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str) or "lean" in key.lower():
                raise ObjectSceneAnchorAtlasError(
                    "atlas contains a direct non-Python backend key"
                )
            _assert_python_only_keys(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _assert_python_only_keys(item)


def _authority_data() -> dict[str, object]:
    return {
        "predicate_authority_id": PYTHON_PREDICATE_AUTHORITY_ID,
        "python_is_canonical_authority": True,
        "selected_decision_manifest_is_only_geometry_input": True,
        "raw_graph_consumed": False,
        "audit_graph_consumed": False,
        "fresh_or_query_pixels_consumed": False,
        "top_k_or_truncation_applied": False,
    }


def object_scene_anchor_atlas_source_digest() -> str:
    return verify_loaded_source(__name__, expected_source_sha256=_LOADED_SOURCE_SHA256)


def object_scene_anchor_atlas_renderer_digest() -> str:
    """Bind only this deterministic presentation algorithm and its input schema."""

    return canonical_digest(
        {
            "algorithm_id": OBJECT_SCENE_ANCHOR_ATLAS_ALGORITHM_ID,
            "source_digest": object_scene_anchor_atlas_source_digest(),
            "input_schema": OBJECT_SCENE_ANCHOR_DECISION_MANIFEST_SCHEMA,
            "grid": {
                "columns": OBJECT_SCENE_ANCHOR_ATLAS_COLUMNS,
                "rows": OBJECT_SCENE_ANCHOR_ATLAS_ROWS,
                "maximum_used_slots": OBJECT_SCENE_ANCHOR_ATLAS_MAX_SLOTS,
                "tile_size_pixels": OBJECT_SCENE_ANCHOR_ATLAS_TILE_SIZE_PIXELS,
            },
            "slot_order": (
                "whole-entity,all-selected-anchor-ids,all-selected-frame-ids"
            ),
            "coordinates": "selected-graph-unsigned-q16",
            "png": "grayscale8-filter0-zlib-stored-deflate-v1",
            "authority": _authority_data(),
        }
    )


@dataclass(frozen=True, slots=True)
class ObjectSceneAnchorAtlasStatus:
    state: str
    reason: str

    def __post_init__(self) -> None:
        # Reuse the closed salience disposition vocabulary exactly.  The atlas
        # cannot convert an indeterminate/error source into a visual negative.
        AnchorSalienceStatus(self.state, self.reason)

    def to_data(self) -> dict[str, str]:
        return {
            "schema": OBJECT_SCENE_ANCHOR_ATLAS_STATUS_SCHEMA,
            "state": self.state,
            "reason": self.reason,
        }

    @classmethod
    def from_data(cls, value: object) -> "ObjectSceneAnchorAtlasStatus":
        raw = _exact_fields(
            value, {"schema", "state", "reason"}, "anchor atlas status"
        )
        if raw["schema"] != OBJECT_SCENE_ANCHOR_ATLAS_STATUS_SCHEMA:
            raise ObjectSceneAnchorAtlasError("anchor atlas status schema differs")
        result = cls(raw["state"], raw["reason"])
        if result.to_data() != dict(raw):
            raise ObjectSceneAnchorAtlasError("anchor atlas status is not canonical")
        return result


def _slot_content(value: "ObjectSceneAnchorAtlasSlot") -> dict[str, object]:
    return {
        "schema": OBJECT_SCENE_ANCHOR_ATLAS_SLOT_SCHEMA,
        "slot_index": value.slot_index,
        "row_index": value.row_index,
        "column_index": value.column_index,
        "slot_id": value.slot_id,
        "slot_kind": value.slot_kind,
        "subject_id": value.subject_id,
        "subject_digest": value.subject_digest,
        "highlight_part_ids": list(value.highlight_part_ids),
        "highlight_compact_ids": list(value.highlight_compact_ids),
        "highlight_join_ids": list(value.highlight_join_ids),
        "highlight_tangent_points_q16": [
            item.to_data() for item in value.highlight_tangent_points_q16
        ],
    }


@dataclass(frozen=True, slots=True)
class ObjectSceneAnchorAtlasSlot:
    slot_index: int
    row_index: int
    column_index: int
    slot_id: str
    slot_kind: str
    subject_id: str
    subject_digest: str
    highlight_part_ids: tuple[str, ...]
    highlight_compact_ids: tuple[str, ...]
    highlight_join_ids: tuple[str, ...]
    highlight_tangent_points_q16: tuple[Q16Point, ...]
    slot_digest: str

    def __post_init__(self) -> None:
        index = _integer(self.slot_index, "atlas slot index")
        if index >= OBJECT_SCENE_ANCHOR_ATLAS_MAX_SLOTS:
            raise ObjectSceneAnchorAtlasError("atlas slot exceeds the hard cap")
        if (
            self.row_index != index // OBJECT_SCENE_ANCHOR_ATLAS_COLUMNS
            or self.column_index != index % OBJECT_SCENE_ANCHOR_ATLAS_COLUMNS
            or self.slot_id != f"slot-{index:04d}"
            or _SLOT_ID.fullmatch(self.slot_id) is None
        ):
            raise ObjectSceneAnchorAtlasError("atlas slot position differs")
        if self.slot_kind not in (
            "whole_entity",
            "part_anchor",
            "compact_anchor",
            "cyclic_frame",
        ):
            raise ObjectSceneAnchorAtlasError("atlas slot kind differs")
        _digest(self.subject_digest, "atlas slot subject digest")
        for label, items, pattern in (
            ("highlight part IDs", self.highlight_part_ids, _PART_ID),
            ("highlight compact IDs", self.highlight_compact_ids, _COMPACT_ID),
            ("highlight join IDs", self.highlight_join_ids, _JOIN_ID),
        ):
            if type(items) is not tuple or any(
                not isinstance(item, str) or pattern.fullmatch(item) is None
                for item in items
            ):
                raise ObjectSceneAnchorAtlasError(f"{label} differ")
        if type(self.highlight_tangent_points_q16) is not tuple or any(
            type(item) is not Q16Point for item in self.highlight_tangent_points_q16
        ):
            raise TypeError("highlight tangent points must be exact Q16 points")
        if self.slot_kind == "whole_entity":
            if index != 0 or _OBJECT_ID.fullmatch(self.subject_id) is None:
                raise ObjectSceneAnchorAtlasError("whole-entity slot differs")
        elif self.slot_kind == "part_anchor":
            if (
                _PART_ID.fullmatch(self.subject_id) is None
                or self.highlight_part_ids != (self.subject_id,)
                or self.highlight_compact_ids
                or self.highlight_join_ids
                or self.highlight_tangent_points_q16
            ):
                raise ObjectSceneAnchorAtlasError("part-anchor slot differs")
        elif self.slot_kind == "compact_anchor":
            if (
                _COMPACT_ID.fullmatch(self.subject_id) is None
                or self.highlight_compact_ids != (self.subject_id,)
                or self.highlight_part_ids
                or self.highlight_join_ids
                or self.highlight_tangent_points_q16
            ):
                raise ObjectSceneAnchorAtlasError("compact-anchor slot differs")
        elif (
            _FRAME_ID.fullmatch(self.subject_id) is None
            or len(self.highlight_join_ids) != 1
            or len(self.highlight_part_ids) < 3
            or len(self.highlight_part_ids)
            != len(self.highlight_tangent_points_q16)
            or self.highlight_compact_ids
        ):
            raise ObjectSceneAnchorAtlasError("cyclic-frame slot differs")
        _digest(self.slot_digest, "atlas slot digest")
        if self.slot_digest != canonical_digest(_slot_content(self)):
            raise ObjectSceneAnchorAtlasError("atlas slot digest differs")

    def to_data(self) -> dict[str, object]:
        return {**_slot_content(self), "slot_digest": self.slot_digest}

    @classmethod
    def from_data(cls, value: object) -> "ObjectSceneAnchorAtlasSlot":
        raw = _exact_fields(
            value,
            {
                "schema",
                "slot_index",
                "row_index",
                "column_index",
                "slot_id",
                "slot_kind",
                "subject_id",
                "subject_digest",
                "highlight_part_ids",
                "highlight_compact_ids",
                "highlight_join_ids",
                "highlight_tangent_points_q16",
                "slot_digest",
            },
            "anchor atlas slot",
        )
        list_fields = (
            "highlight_part_ids",
            "highlight_compact_ids",
            "highlight_join_ids",
            "highlight_tangent_points_q16",
        )
        if (
            raw["schema"] != OBJECT_SCENE_ANCHOR_ATLAS_SLOT_SCHEMA
            or any(not isinstance(raw[name], list) for name in list_fields)
            or any(
                not isinstance(item, Mapping)
                for item in raw["highlight_tangent_points_q16"]
            )
        ):
            raise ObjectSceneAnchorAtlasError("anchor atlas slot policy differs")
        result = cls(
            slot_index=raw["slot_index"],
            row_index=raw["row_index"],
            column_index=raw["column_index"],
            slot_id=raw["slot_id"],
            slot_kind=raw["slot_kind"],
            subject_id=raw["subject_id"],
            subject_digest=raw["subject_digest"],
            highlight_part_ids=tuple(raw["highlight_part_ids"]),
            highlight_compact_ids=tuple(raw["highlight_compact_ids"]),
            highlight_join_ids=tuple(raw["highlight_join_ids"]),
            highlight_tangent_points_q16=tuple(
                Q16Point.from_data(item)
                for item in raw["highlight_tangent_points_q16"]
            ),
            slot_digest=raw["slot_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectSceneAnchorAtlasError("anchor atlas slot is not canonical")
        return result


def _slot_map_digest(slots: tuple[ObjectSceneAnchorAtlasSlot, ...]) -> str:
    return canonical_digest(
        {
            "schema": "gkm.object-scene-anchor-atlas-slot-map.v1",
            "order": "row-major-exhaustive-prefix",
            "slots": [item.to_data() for item in slots],
        }
    )


def _atlas_content(value: "ObjectSceneAnchorAtlas") -> dict[str, object]:
    return {
        "schema": OBJECT_SCENE_ANCHOR_ATLAS_SCHEMA,
        "algorithm_id": OBJECT_SCENE_ANCHOR_ATLAS_ALGORITHM_ID,
        "object_id": value.object_id,
        "decision_manifest_digest": value.decision_manifest_digest,
        "decision_manifest": value.decision_manifest.to_data(),
        "selected_graph_artifact_digest": value.selected_graph_artifact_digest,
        "renderer_artifact_digest": value.renderer_artifact_digest,
        "status": value.status.to_data(),
        "png_mode": OBJECT_SCENE_ANCHOR_ATLAS_MODE,
        "grid_columns": OBJECT_SCENE_ANCHOR_ATLAS_COLUMNS,
        "grid_rows": OBJECT_SCENE_ANCHOR_ATLAS_ROWS,
        "maximum_used_slots": OBJECT_SCENE_ANCHOR_ATLAS_MAX_SLOTS,
        "tile_size_pixels": OBJECT_SCENE_ANCHOR_ATLAS_TILE_SIZE_PIXELS,
        "image_width_pixels": value.image_width_pixels,
        "image_height_pixels": value.image_height_pixels,
        "slot_count": value.slot_count,
        "slots": [item.to_data() for item in value.slots],
        "slot_map_digest": value.slot_map_digest,
        "png_byte_count": value.png_byte_count,
        "png_digest": value.png_digest,
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class ObjectSceneAnchorAtlas:
    object_id: str
    decision_manifest_digest: str
    decision_manifest: ObjectSceneAnchorDecisionManifest
    selected_graph_artifact_digest: str | None
    renderer_artifact_digest: str
    status: ObjectSceneAnchorAtlasStatus
    image_width_pixels: int
    image_height_pixels: int
    slot_count: int
    slots: tuple[ObjectSceneAnchorAtlasSlot, ...]
    slot_map_digest: str
    png_byte_count: int
    png_digest: str | None
    artifact_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.object_id, str) or _OBJECT_ID.fullmatch(self.object_id) is None:
            raise ObjectSceneAnchorAtlasError("atlas object ID differs")
        for label, item in (
            ("decision manifest digest", self.decision_manifest_digest),
            ("renderer artifact digest", self.renderer_artifact_digest),
            ("slot map digest", self.slot_map_digest),
        ):
            _digest(item, label)
        if type(self.decision_manifest) is not ObjectSceneAnchorDecisionManifest:
            raise TypeError("atlas decision manifest has the wrong type")
        if type(self.status) is not ObjectSceneAnchorAtlasStatus:
            raise TypeError("atlas status has the wrong type")
        manifest = self.decision_manifest
        if (
            manifest.object_id != self.object_id
            or manifest.manifest_digest != self.decision_manifest_digest
            or manifest.selected_graph_artifact_digest
            != self.selected_graph_artifact_digest
            or manifest.salience_state != self.status.state
            or manifest.salience_reason != self.status.reason
        ):
            raise ObjectSceneAnchorAtlasError("atlas decision manifest binding differs")
        if self.renderer_artifact_digest != object_scene_anchor_atlas_renderer_digest():
            raise ObjectSceneAnchorAtlasError("atlas renderer binding is stale")
        for label, item in (
            ("image width", self.image_width_pixels),
            ("image height", self.image_height_pixels),
            ("slot count", self.slot_count),
            ("PNG byte count", self.png_byte_count),
        ):
            _integer(item, label)
        if type(self.slots) is not tuple or any(
            type(item) is not ObjectSceneAnchorAtlasSlot for item in self.slots
        ):
            raise TypeError("atlas slots must be an exact typed tuple")
        if (
            self.slot_count != len(self.slots)
            or self.slot_count > OBJECT_SCENE_ANCHOR_ATLAS_MAX_SLOTS
            or tuple(item.slot_index for item in self.slots)
            != tuple(range(self.slot_count))
            or self.slot_map_digest != _slot_map_digest(self.slots)
        ):
            raise ObjectSceneAnchorAtlasError(
                "atlas slot map is not an exhaustive ordered prefix"
            )
        if self.status.state == "clean":
            if (
                self.selected_graph_artifact_digest is None
                or self.png_digest is None
                or self.image_width_pixels
                != OBJECT_SCENE_ANCHOR_ATLAS_WIDTH_PIXELS
                or self.image_height_pixels
                != OBJECT_SCENE_ANCHOR_ATLAS_HEIGHT_PIXELS
                or self.png_byte_count < 1
                or not self.slots
                or self.slots[0].slot_kind != "whole_entity"
                or self.slots[0].subject_id != self.object_id
            ):
                raise ObjectSceneAnchorAtlasError(
                    "clean atlas lacks its complete image or whole-entity slot"
                )
            _digest(self.selected_graph_artifact_digest, "selected graph digest")
            _digest(self.png_digest, "atlas PNG digest")
            anchor_slots = tuple(
                item
                for item in self.slots[1:]
                if item.slot_kind in ("part_anchor", "compact_anchor")
            )
            frame_slots = tuple(
                item for item in self.slots[1:] if item.slot_kind == "cyclic_frame"
            )
            if (
                len(anchor_slots) + len(frame_slots) != len(self.slots) - 1
                or tuple(item.subject_id for item in anchor_slots)
                != tuple(sorted(item.subject_id for item in anchor_slots))
                or tuple(item.subject_id for item in frame_slots)
                != tuple(f"frame-{index:08d}" for index in range(len(frame_slots)))
                or self.slots[1 : 1 + len(anchor_slots)] != anchor_slots
            ):
                raise ObjectSceneAnchorAtlasError("atlas slot kind/order differs")
        elif (
            self.selected_graph_artifact_digest is not None
            or self.image_width_pixels
            or self.image_height_pixels
            or self.slot_count
            or self.slots
            or self.png_byte_count
            or self.png_digest is not None
        ):
            raise ObjectSceneAnchorAtlasError(
                "non-clean atlas exposes a partial graph, slot map, or PNG"
            )
        if self.slots != _slots_for_manifest(manifest):
            raise ObjectSceneAnchorAtlasError(
                "atlas slots differ from the embedded selected decision graph"
            )
        unsigned = _atlas_content(self)
        _assert_python_only_keys(unsigned)
        _digest(self.artifact_digest, "anchor atlas artifact digest")
        if self.artifact_digest != canonical_digest(unsigned):
            raise ObjectSceneAnchorAtlasError("anchor atlas artifact digest differs")

    def to_data(self) -> dict[str, object]:
        return {**_atlas_content(self), "artifact_digest": self.artifact_digest}

    @classmethod
    def from_data(cls, value: object) -> "ObjectSceneAnchorAtlas":
        raw = _exact_fields(
            value,
            {
                "schema",
                "algorithm_id",
                "object_id",
                "decision_manifest_digest",
                "decision_manifest",
                "selected_graph_artifact_digest",
                "renderer_artifact_digest",
                "status",
                "png_mode",
                "grid_columns",
                "grid_rows",
                "maximum_used_slots",
                "tile_size_pixels",
                "image_width_pixels",
                "image_height_pixels",
                "slot_count",
                "slots",
                "slot_map_digest",
                "png_byte_count",
                "png_digest",
                *_authority_data(),
                "artifact_digest",
            },
            "object scene anchor atlas",
        )
        if (
            raw["schema"] != OBJECT_SCENE_ANCHOR_ATLAS_SCHEMA
            or raw["algorithm_id"] != OBJECT_SCENE_ANCHOR_ATLAS_ALGORITHM_ID
            or raw["png_mode"] != OBJECT_SCENE_ANCHOR_ATLAS_MODE
            or raw["grid_columns"] != OBJECT_SCENE_ANCHOR_ATLAS_COLUMNS
            or raw["grid_rows"] != OBJECT_SCENE_ANCHOR_ATLAS_ROWS
            or raw["maximum_used_slots"] != OBJECT_SCENE_ANCHOR_ATLAS_MAX_SLOTS
            or raw["tile_size_pixels"]
            != OBJECT_SCENE_ANCHOR_ATLAS_TILE_SIZE_PIXELS
            or any(raw[key] != item for key, item in _authority_data().items())
            or not isinstance(raw["status"], Mapping)
            or not isinstance(raw["decision_manifest"], Mapping)
            or not isinstance(raw["slots"], list)
            or any(not isinstance(item, Mapping) for item in raw["slots"])
        ):
            raise ObjectSceneAnchorAtlasError("anchor atlas policy differs")
        result = cls(
            object_id=raw["object_id"],
            decision_manifest_digest=raw["decision_manifest_digest"],
            decision_manifest=ObjectSceneAnchorDecisionManifest.from_data(
                raw["decision_manifest"]
            ),
            selected_graph_artifact_digest=raw[
                "selected_graph_artifact_digest"
            ],
            renderer_artifact_digest=raw["renderer_artifact_digest"],
            status=ObjectSceneAnchorAtlasStatus.from_data(raw["status"]),
            image_width_pixels=raw["image_width_pixels"],
            image_height_pixels=raw["image_height_pixels"],
            slot_count=raw["slot_count"],
            slots=tuple(ObjectSceneAnchorAtlasSlot.from_data(item) for item in raw["slots"]),
            slot_map_digest=raw["slot_map_digest"],
            png_byte_count=raw["png_byte_count"],
            png_digest=raw["png_digest"],
            artifact_digest=raw["artifact_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectSceneAnchorAtlasError("anchor atlas is not canonical")
        return result


def _make_slot(
    *,
    index: int,
    kind: str,
    subject_id: str,
    subject_digest: str,
    part_ids: tuple[str, ...] = (),
    compact_ids: tuple[str, ...] = (),
    join_ids: tuple[str, ...] = (),
    tangent_points: tuple[Q16Point, ...] = (),
) -> ObjectSceneAnchorAtlasSlot:
    values = {
        "slot_index": index,
        "row_index": index // OBJECT_SCENE_ANCHOR_ATLAS_COLUMNS,
        "column_index": index % OBJECT_SCENE_ANCHOR_ATLAS_COLUMNS,
        "slot_id": f"slot-{index:04d}",
        "slot_kind": kind,
        "subject_id": subject_id,
        "subject_digest": subject_digest,
        "highlight_part_ids": part_ids,
        "highlight_compact_ids": compact_ids,
        "highlight_join_ids": join_ids,
        "highlight_tangent_points_q16": tangent_points,
    }
    provisional = object.__new__(ObjectSceneAnchorAtlasSlot)
    for name, item in values.items():
        object.__setattr__(provisional, name, item)
    return ObjectSceneAnchorAtlasSlot(
        **values,
        slot_digest=canonical_digest(_slot_content(provisional)),
    )


def _slots_for_manifest(
    manifest: ObjectSceneAnchorDecisionManifest,
) -> tuple[ObjectSceneAnchorAtlasSlot, ...]:
    graph = manifest.selected_graph
    if graph is None:
        return ()
    parts = {item.part_id: item for item in graph.parts}
    compacts = {item.compact_id: item for item in graph.compact_components}
    joins = {item.join_id: item for item in graph.joins}
    frames = {item.frame_id: item for item in graph.cyclic_frames}
    slots = [
        _make_slot(
            index=0,
            kind="whole_entity",
            subject_id=manifest.object_id,
            subject_digest=graph.artifact_digest,
            part_ids=tuple(item.part_id for item in graph.parts),
            compact_ids=tuple(item.compact_id for item in graph.compact_components),
            join_ids=tuple(item.join_id for item in graph.joins),
            tangent_points=tuple(
                point
                for frame in graph.cyclic_frames
                for point in frame.clockwise_tangent_points_q16
            ),
        )
    ]
    for anchor_id in manifest.selected_anchor_ids:
        if anchor_id in parts:
            part = parts[anchor_id]
            slots.append(
                _make_slot(
                    index=len(slots),
                    kind="part_anchor",
                    subject_id=anchor_id,
                    subject_digest=part.digest(),
                    part_ids=(anchor_id,),
                )
            )
        elif anchor_id in compacts:
            compact = compacts[anchor_id]
            slots.append(
                _make_slot(
                    index=len(slots),
                    kind="compact_anchor",
                    subject_id=anchor_id,
                    subject_digest=compact.digest(),
                    compact_ids=(anchor_id,),
                )
            )
        else:  # pragma: no cover - decision manifest validation closes this.
            raise ObjectSceneAnchorAtlasError("selected anchor is absent from graph")
    for frame_id in manifest.selected_frame_ids:
        frame = frames.get(frame_id)
        if frame is None or frame.join_id not in joins:
            raise ObjectSceneAnchorAtlasError("selected frame is absent from graph")
        slots.append(
            _make_slot(
                index=len(slots),
                kind="cyclic_frame",
                subject_id=frame.frame_id,
                subject_digest=frame.digest(),
                part_ids=frame.clockwise_incident_part_ids,
                join_ids=(frame.join_id,),
                tangent_points=frame.clockwise_tangent_points_q16,
            )
        )
    frozen = tuple(slots)
    expected_count = 1 + len(graph.parts) + len(graph.compact_components) + len(
        graph.cyclic_frames
    )
    if len(frozen) != expected_count or len(frozen) > OBJECT_SCENE_ANCHOR_ATLAS_MAX_SLOTS:
        raise ObjectSceneAnchorAtlasError(
            "selected graph exceeds the exhaustive fixed atlas cap"
        )
    return frozen


def _point_in_tile(point: Q16Point, slot: ObjectSceneAnchorAtlasSlot) -> tuple[int, int]:
    tile = OBJECT_SCENE_ANCHOR_ATLAS_TILE_SIZE_PIXELS
    viewport = tile - 2 * _TILE_MARGIN_PIXELS
    x0 = slot.column_index * tile + _TILE_MARGIN_PIXELS
    y0 = slot.row_index * tile + _TILE_MARGIN_PIXELS
    x = x0 + (point.x * (viewport - 1) + 32767) // 65535
    y = y0 + (point.y * (viewport - 1) + 32767) // 65535
    return x, y


def _draw_disk(
    pixels: np.ndarray, x: int, y: int, radius: int, value: int
) -> None:
    height, width = pixels.shape
    for py in range(max(0, y - radius), min(height, y + radius + 1)):
        for px in range(max(0, x - radius), min(width, x + radius + 1)):
            if (px - x) * (px - x) + (py - y) * (py - y) <= radius * radius:
                pixels[py, px] = min(int(pixels[py, px]), value)


def _draw_line(
    pixels: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
    value: int,
    radius: int,
) -> None:
    x0, y0 = start
    x1, y1 = end
    dx = abs(x1 - x0)
    sx = 1 if x0 < x1 else -1
    dy = -abs(y1 - y0)
    sy = 1 if y0 < y1 else -1
    error = dx + dy
    while True:
        _draw_disk(pixels, x0, y0, radius, value)
        if x0 == x1 and y0 == y1:
            break
        doubled = 2 * error
        if doubled >= dy:
            error += dy
            x0 += sx
        if doubled <= dx:
            error += dx
            y0 += sy


def _draw_part(
    pixels: np.ndarray,
    slot: ObjectSceneAnchorAtlasSlot,
    part: AnchorPart,
    value: int,
    radius: int,
) -> None:
    points: list[tuple[int, int]] = []
    for point in part.path_q16:
        mapped = _point_in_tile(point, slot)
        if not points or mapped != points[-1]:
            points.append(mapped)
    for left, right in zip(points, points[1:]):
        _draw_line(pixels, left, right, value, radius)
    if len(points) == 1:
        _draw_disk(pixels, points[0][0], points[0][1], radius, value)
    elif part.closed and points[-1] != points[0]:
        _draw_line(pixels, points[-1], points[0], value, radius)


def _draw_compact(
    pixels: np.ndarray,
    slot: ObjectSceneAnchorAtlasSlot,
    compact: AnchorCompactComponent,
    value: int,
    radius: int,
) -> None:
    x0, y0 = _point_in_tile(compact.bbox_min_q16, slot)
    x1, y1 = _point_in_tile(compact.bbox_max_q16, slot)
    _draw_line(pixels, (x0, y0), (x1, y0), value, radius)
    _draw_line(pixels, (x1, y0), (x1, y1), value, radius)
    _draw_line(pixels, (x1, y1), (x0, y1), value, radius)
    _draw_line(pixels, (x0, y1), (x0, y0), value, radius)
    center_x, center_y = _point_in_tile(compact.location_q16, slot)
    _draw_disk(pixels, center_x, center_y, radius + 1, value)


def _draw_join(
    pixels: np.ndarray,
    slot: ObjectSceneAnchorAtlasSlot,
    join: AnchorJoin,
    value: int,
    radius: int,
) -> None:
    x, y = _point_in_tile(join.location_q16, slot)
    _draw_disk(pixels, x, y, radius, value)


def _draw_complete_graph(
    pixels: np.ndarray,
    slot: ObjectSceneAnchorAtlasSlot,
    graph: ObjectSceneAnchorGraph,
    value: int,
    bold: bool,
) -> None:
    for part in graph.parts:
        _draw_part(pixels, slot, part, value, 1 if bold else 0)
    for compact in graph.compact_components:
        _draw_compact(pixels, slot, compact, value, 1 if bold else 0)
    for terminal in graph.terminals:
        x, y = _point_in_tile(terminal.location_q16, slot)
        _draw_disk(pixels, x, y, 2 if bold else 1, value)
    for join in graph.joins:
        _draw_join(pixels, slot, join, value, 3 if bold else 2)
    for frame in graph.cyclic_frames:
        for point in frame.clockwise_tangent_points_q16:
            x, y = _point_in_tile(point, slot)
            _draw_disk(pixels, x, y, 2 if bold else 1, value)


def _draw_slot(
    pixels: np.ndarray,
    slot: ObjectSceneAnchorAtlasSlot,
    graph: ObjectSceneAnchorGraph,
) -> None:
    _draw_complete_graph(pixels, slot, graph, _BASE_GRAPH_VALUE, False)
    parts = {item.part_id: item for item in graph.parts}
    compacts = {item.compact_id: item for item in graph.compact_components}
    joins = {item.join_id: item for item in graph.joins}
    if slot.slot_kind == "whole_entity":
        _draw_complete_graph(pixels, slot, graph, _HIGHLIGHT_VALUE, True)
        return
    for part_id in slot.highlight_part_ids:
        _draw_part(pixels, slot, parts[part_id], _HIGHLIGHT_VALUE, 1)
    for compact_id in slot.highlight_compact_ids:
        _draw_compact(
            pixels, slot, compacts[compact_id], _HIGHLIGHT_VALUE, 1
        )
    for join_id in slot.highlight_join_ids:
        _draw_join(pixels, slot, joins[join_id], _HIGHLIGHT_VALUE, 4)
    for point in slot.highlight_tangent_points_q16:
        x, y = _point_in_tile(point, slot)
        _draw_disk(pixels, x, y, 2, _HIGHLIGHT_VALUE)


def _adler32(payload: bytes) -> int:
    first = 1
    second = 0
    modulus = 65521
    for cursor in range(0, len(payload), 5552):
        for value in payload[cursor : cursor + 5552]:
            first += value
            second += first
        first %= modulus
        second %= modulus
    return (second << 16) | first


def _stored_zlib(payload: bytes) -> bytes:
    result = bytearray(b"\x78\x01")
    if not payload:
        result.extend(b"\x01\x00\x00\xff\xff")
    else:
        cursor = 0
        while cursor < len(payload):
            block = payload[cursor : cursor + 65535]
            cursor += len(block)
            result.append(1 if cursor == len(payload) else 0)
            length = len(block)
            result.extend(length.to_bytes(2, "little"))
            result.extend((length ^ 0xFFFF).to_bytes(2, "little"))
            result.extend(block)
    result.extend(_adler32(payload).to_bytes(4, "big"))
    return bytes(result)


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    crc = binascii.crc32(kind + payload) & 0xFFFFFFFF
    return (
        len(payload).to_bytes(4, "big")
        + kind
        + payload
        + crc.to_bytes(4, "big")
    )


def object_scene_anchor_grayscale_png_byte_count(
    width_pixels: int,
    height_pixels: int,
) -> int:
    """Return the exact byte count emitted by the grayscale PNG encoder.

    Every scanline has one filter byte, and ``_stored_zlib`` emits stored
    DEFLATE blocks of at most 65,535 bytes.  The remaining 63 bytes are the
    PNG signature/chunk framing plus the zlib header and Adler-32 trailer.
    This is a size identity, not a compression estimate.  Its domain is the
    domain of ``_encode_grayscale_png``: each dimension must fit IHDR and the
    complete stored-zlib stream must fit the encoder's single IDAT chunk.
    """

    width = _integer(width_pixels, "grayscale PNG width", minimum=1)
    height = _integer(height_pixels, "grayscale PNG height", minimum=1)
    if width > 0xFFFFFFFF or height > 0xFFFFFFFF:
        raise ObjectSceneAnchorAtlasError(
            "grayscale PNG dimensions exceed the four-byte IHDR fields"
        )
    scanline_byte_count = height * (width + 1)
    stored_block_count = (scanline_byte_count + 65534) // 65535
    stored_zlib_byte_count = scanline_byte_count + 6 + 5 * stored_block_count
    if stored_zlib_byte_count > _PNG_CHUNK_MAX_PAYLOAD_BYTE_COUNT:
        raise ObjectSceneAnchorAtlasError(
            "grayscale PNG payload exceeds the single IDAT chunk field"
        )
    return scanline_byte_count + 63 + 5 * stored_block_count


def _encode_grayscale_png(pixels: np.ndarray) -> bytes:
    if pixels.dtype != np.uint8 or pixels.ndim != 2:
        raise TypeError("atlas pixels must be a two-dimensional uint8 array")
    height, width = pixels.shape
    ihdr = (
        width.to_bytes(4, "big")
        + height.to_bytes(4, "big")
        + bytes((8, 0, 0, 0, 0))
    )
    scanlines = b"".join(
        b"\x00" + np.ascontiguousarray(row).tobytes() for row in pixels
    )
    result = (
        _PNG_SIGNATURE
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"IDAT", _stored_zlib(scanlines))
        + _png_chunk(b"IEND", b"")
    )
    if len(result) != object_scene_anchor_grayscale_png_byte_count(width, height):
        raise ObjectSceneAnchorAtlasError(
            "grayscale PNG encoder differs from its exact byte-count identity"
        )
    return result


def _render_pixels(
    slots: tuple[ObjectSceneAnchorAtlasSlot, ...], graph: ObjectSceneAnchorGraph
) -> np.ndarray:
    pixels = np.full(
        (
            OBJECT_SCENE_ANCHOR_ATLAS_HEIGHT_PIXELS,
            OBJECT_SCENE_ANCHOR_ATLAS_WIDTH_PIXELS,
        ),
        _BACKGROUND_VALUE,
        dtype=np.uint8,
    )
    tile = OBJECT_SCENE_ANCHOR_ATLAS_TILE_SIZE_PIXELS
    for index in range(OBJECT_SCENE_ANCHOR_ATLAS_MAX_SLOTS):
        row, column = divmod(index, OBJECT_SCENE_ANCHOR_ATLAS_COLUMNS)
        y0, x0 = row * tile, column * tile
        border_value = (
            _USED_BORDER_VALUE if index < len(slots) else _UNUSED_BORDER_VALUE
        )
        pixels[y0 : y0 + _TILE_BORDER_PIXELS, x0 : x0 + tile] = border_value
        pixels[y0 + tile - _TILE_BORDER_PIXELS : y0 + tile, x0 : x0 + tile] = border_value
        pixels[y0 : y0 + tile, x0 : x0 + _TILE_BORDER_PIXELS] = border_value
        pixels[y0 : y0 + tile, x0 + tile - _TILE_BORDER_PIXELS : x0 + tile] = border_value
    for slot in slots:
        _draw_slot(pixels, slot, graph)
    return pixels


def _make_atlas(
    *,
    manifest: ObjectSceneAnchorDecisionManifest,
    slots: tuple[ObjectSceneAnchorAtlasSlot, ...],
    png_bytes: bytes | None,
) -> ObjectSceneAnchorAtlas:
    clean = manifest.salience_state == "clean"
    values = {
        "object_id": manifest.object_id,
        "decision_manifest_digest": manifest.manifest_digest,
        "decision_manifest": manifest,
        "selected_graph_artifact_digest": manifest.selected_graph_artifact_digest,
        "renderer_artifact_digest": object_scene_anchor_atlas_renderer_digest(),
        "status": ObjectSceneAnchorAtlasStatus(
            manifest.salience_state, manifest.salience_reason
        ),
        "image_width_pixels": OBJECT_SCENE_ANCHOR_ATLAS_WIDTH_PIXELS if clean else 0,
        "image_height_pixels": OBJECT_SCENE_ANCHOR_ATLAS_HEIGHT_PIXELS if clean else 0,
        "slot_count": len(slots),
        "slots": slots,
        "slot_map_digest": _slot_map_digest(slots),
        "png_byte_count": 0 if png_bytes is None else len(png_bytes),
        "png_digest": (
            None if png_bytes is None else hashlib.sha256(png_bytes).hexdigest()
        ),
    }
    provisional = object.__new__(ObjectSceneAnchorAtlas)
    for name, item in values.items():
        object.__setattr__(provisional, name, item)
    return ObjectSceneAnchorAtlas(
        **values,
        artifact_digest=canonical_digest(_atlas_content(provisional)),
    )


def render_object_scene_anchor_atlas(
    decision_manifest: ObjectSceneAnchorDecisionManifest,
) -> tuple[ObjectSceneAnchorAtlas, bytes | None]:
    """Render every selected anchor/frame, or return an image-free typed gap."""

    if type(decision_manifest) is not ObjectSceneAnchorDecisionManifest:
        raise TypeError(
            "decision_manifest must be exact ObjectSceneAnchorDecisionManifest"
        )
    # Canonicalize the complete decision view before rendering.  No catalog
    # entry or full salience artifact is accepted by this boundary.
    manifest = ObjectSceneAnchorDecisionManifest.from_data(
        decision_manifest.to_data()
    )
    if manifest.salience_state != "clean":
        return _make_atlas(
            manifest=manifest,
            slots=(),
            png_bytes=None,
        ), None
    graph = manifest.selected_graph
    if graph is None:  # pragma: no cover - manifest validation closes this.
        raise ObjectSceneAnchorAtlasError("clean decision lacks selected graph")
    slots = _slots_for_manifest(manifest)
    pixels = _render_pixels(slots, graph)
    png_bytes = _encode_grayscale_png(pixels)
    return _make_atlas(
        manifest=manifest,
        slots=slots,
        png_bytes=png_bytes,
    ), png_bytes


def verify_object_scene_anchor_atlas(
    artifact: ObjectSceneAnchorAtlas,
    png_bytes: bytes | None,
    decision_manifest: ObjectSceneAnchorDecisionManifest,
    *,
    expected_artifact_digest: str | None = None,
) -> ObjectSceneAnchorAtlas:
    """Verify canonical metadata and replay bytes from the selected manifest."""

    if type(artifact) is not ObjectSceneAnchorAtlas:
        raise TypeError("artifact must be exact ObjectSceneAnchorAtlas")
    if png_bytes is not None and not isinstance(png_bytes, bytes):
        raise TypeError("png_bytes must be exact bytes or null")
    restored = ObjectSceneAnchorAtlas.from_data(artifact.to_data())
    if expected_artifact_digest is not None and restored.artifact_digest != _digest(
        expected_artifact_digest, "expected atlas artifact digest"
    ):
        raise ObjectSceneAnchorAtlasError("anchor atlas differs from commitment")
    replay, expected_png = render_object_scene_anchor_atlas(decision_manifest)
    if replay != restored or expected_png != png_bytes:
        raise ObjectSceneAnchorAtlasError(
            "anchor atlas differs from exact selected-manifest replay"
        )
    return restored


extract_object_scene_anchor_atlas = render_object_scene_anchor_atlas


__all__ = (
    "OBJECT_SCENE_ANCHOR_ATLAS_ALGORITHM_ID",
    "OBJECT_SCENE_ANCHOR_ATLAS_COLUMNS",
    "OBJECT_SCENE_ANCHOR_ATLAS_HEIGHT_PIXELS",
    "OBJECT_SCENE_ANCHOR_ATLAS_MAX_SLOTS",
    "OBJECT_SCENE_ANCHOR_ATLAS_MODE",
    "OBJECT_SCENE_ANCHOR_ATLAS_ROWS",
    "OBJECT_SCENE_ANCHOR_ATLAS_SCHEMA",
    "OBJECT_SCENE_ANCHOR_ATLAS_SLOT_SCHEMA",
    "OBJECT_SCENE_ANCHOR_ATLAS_STATUS_SCHEMA",
    "OBJECT_SCENE_ANCHOR_ATLAS_TILE_SIZE_PIXELS",
    "OBJECT_SCENE_ANCHOR_ATLAS_WIDTH_PIXELS",
    "ObjectSceneAnchorAtlas",
    "ObjectSceneAnchorAtlasError",
    "ObjectSceneAnchorAtlasSlot",
    "ObjectSceneAnchorAtlasStatus",
    "extract_object_scene_anchor_atlas",
    "object_scene_anchor_grayscale_png_byte_count",
    "object_scene_anchor_atlas_renderer_digest",
    "object_scene_anchor_atlas_source_digest",
    "render_object_scene_anchor_atlas",
    "verify_object_scene_anchor_atlas",
)
