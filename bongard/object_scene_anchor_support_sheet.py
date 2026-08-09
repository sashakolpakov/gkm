"""Frozen per-panel presentation for the anchor-card proposer.

One deterministic grayscale PNG presents the complete original panel first,
then one row per inventoried object.  Every object row contains its exact
full-style crop followed by its exhaustive selected-anchor atlas.  Component
pixels are copied at native resolution; no interpolation or resampling is
permitted.

The persisted receipt contains only decision-facing identities and exact
presentation placements.  The rich anchor catalog is accepted transiently by
the builder and cold verifier so object crops can be replayed from the source
panel, but catalog entries and salience/raw/audit provenance are never stored.
"""

from __future__ import annotations

from bongard.runtime_source_snapshot import capture_loaded_source, verify_loaded_source


_LOADED_SOURCE_SHA256 = capture_loaded_source(__name__, __file__)

from dataclasses import dataclass
import hashlib
from io import BytesIO
import re
from typing import Any, Mapping

import numpy as np
from PIL import Image

from bongard.canonical import canonical_digest
from bongard.object_scene_anchor_atlas import (
    OBJECT_SCENE_ANCHOR_ATLAS_COLUMNS,
    OBJECT_SCENE_ANCHOR_ATLAS_TILE_SIZE_PIXELS,
    ObjectSceneAnchorAtlas,
    ObjectSceneAnchorAtlasSlot,
    _encode_grayscale_png,
    object_scene_anchor_atlas_renderer_digest,
    render_object_scene_anchor_atlas,
)
from bongard.object_scene_anchor_catalog import ObjectSceneAnchorCatalog
from bongard.object_scene_anchor_crop import (
    OBJECT_SCENE_ANCHOR_CROP_RENDERER_ID,
    render_object_scene_anchor_object_crop,
)
from bongard.object_scene_anchor_panel_manifest import (
    OBJECT_SCENE_ANCHOR_PANEL_DECISION_MANIFEST_SCHEMA,
    ObjectSceneAnchorPanelDecisionManifest,
    verify_object_scene_anchor_panel_decision_manifest,
)
from bongard.object_scene_visual_frontend import ObjectSceneProposalInventory
from bongard.python_predicate_authority import PYTHON_PREDICATE_AUTHORITY_ID


OBJECT_SCENE_ANCHOR_SUPPORT_SHEET_SCHEMA = (
    "gkm.object-scene-anchor-support-sheet.v1"
)
OBJECT_SCENE_ANCHOR_SUPPORT_SHEET_OBJECT_SCHEMA = (
    "gkm.object-scene-anchor-support-sheet-object.v1"
)
OBJECT_SCENE_ANCHOR_SUPPORT_SHEET_SLOT_SCHEMA = (
    "gkm.object-scene-anchor-support-sheet-slot.v1"
)
OBJECT_SCENE_ANCHOR_SUPPORT_SHEET_ALGORITHM_ID = (
    "bongard.object-scene-anchor-support-sheet/panel-crop-atlas-native-l-v1"
)
OBJECT_SCENE_ANCHOR_SUPPORT_SHEET_MODE = "L"

OBJECT_SCENE_ANCHOR_SUPPORT_SHEET_PADDING_PIXELS = 12
OBJECT_SCENE_ANCHOR_SUPPORT_SHEET_BORDER_PIXELS = 2
OBJECT_SCENE_ANCHOR_SUPPORT_SHEET_COMPONENT_GAP_PIXELS = 10
OBJECT_SCENE_ANCHOR_SUPPORT_SHEET_SECTION_GAP_PIXELS = 14

_BACKGROUND_VALUE = 255
_PANEL_BORDER_VALUE = 24
_CROP_BORDER_VALUE = 96
_ATLAS_BORDER_VALUE = 168
_MAX_SHEET_PIXELS = 64 * 1024 * 1024
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_OBJECT_ID = re.compile(r"object_[0-9]{4}\Z")
_SLOT_ID = re.compile(r"slot-[0-9]{4}\Z")
_BINDING_ALIAS = re.compile(r"binding_[0-9]{3}\Z")


class ObjectSceneAnchorSupportSheetError(ValueError):
    """A support sheet is incomplete, malformed, or fails exact replay."""


def _exact_fields(
    value: object, expected: set[str] | frozenset[str], label: str
) -> Mapping[str, Any]:
    if (
        not isinstance(value, Mapping)
        or any(not isinstance(key, str) for key in value)
        or set(value) != set(expected)
    ):
        raise ObjectSceneAnchorSupportSheetError(f"{label} fields differ")
    return value


def _integer(value: object, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ObjectSceneAnchorSupportSheetError(
            f"{label} must be an integer at least {minimum}"
        )
    return value


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise ObjectSceneAnchorSupportSheetError(
            f"{label} must be a lowercase SHA-256"
        )
    return value


def _assert_decision_only_keys(value: object) -> None:
    forbidden = (
        "lean",
        "entry_digest",
        "salience_artifact",
        "raw_graph",
        "audit_graph",
        "catalog_digest",
    )
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str) or any(
                token in key.casefold() for token in forbidden
            ):
                raise ObjectSceneAnchorSupportSheetError(
                    "support sheet contains non-decision provenance"
                )
            _assert_decision_only_keys(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _assert_decision_only_keys(item)


def _authority_data() -> dict[str, object]:
    return {
        "predicate_authority_id": PYTHON_PREDICATE_AUTHORITY_ID,
        "python_is_canonical_authority": True,
        "complete_panel_manifest_required": True,
        "complete_object_inventory_required": True,
        "object_omission_allowed": False,
        "component_resampling_allowed": False,
        "binding_alias_scope": (
            "complete-entity-part-full-frame-kind-catalogs"
        ),
        "original_panel_presented_first": True,
        "catalog_is_transient_replay_input": True,
        "fresh_or_query_pixels_consumed": False,
    }


def object_scene_anchor_support_sheet_source_digest() -> str:
    return verify_loaded_source(__name__, expected_source_sha256=_LOADED_SOURCE_SHA256)


def object_scene_anchor_support_sheet_renderer_digest() -> str:
    """Bind this native-resolution compositor and its decision-only inputs."""

    return canonical_digest(
        {
            "algorithm_id": OBJECT_SCENE_ANCHOR_SUPPORT_SHEET_ALGORITHM_ID,
            "source_digest": object_scene_anchor_support_sheet_source_digest(),
            "panel_manifest_schema": (
                OBJECT_SCENE_ANCHOR_PANEL_DECISION_MANIFEST_SCHEMA
            ),
            "crop_renderer_id": OBJECT_SCENE_ANCHOR_CROP_RENDERER_ID,
            "atlas_renderer_digest": object_scene_anchor_atlas_renderer_digest(),
            "layout": {
                "order": "original-panel-then-complete-object-inventory",
                "object_row": "full-style-crop-then-exhaustive-anchor-atlas",
                "padding_pixels": (
                    OBJECT_SCENE_ANCHOR_SUPPORT_SHEET_PADDING_PIXELS
                ),
                "border_pixels": (
                    OBJECT_SCENE_ANCHOR_SUPPORT_SHEET_BORDER_PIXELS
                ),
                "component_gap_pixels": (
                    OBJECT_SCENE_ANCHOR_SUPPORT_SHEET_COMPONENT_GAP_PIXELS
                ),
                "section_gap_pixels": (
                    OBJECT_SCENE_ANCHOR_SUPPORT_SHEET_SECTION_GAP_PIXELS
                ),
            },
            "pixels": "native-grayscale-integer-copy-no-resampling",
            "png": "grayscale8-filter0-zlib-stored-deflate-v1",
            "authority": _authority_data(),
        }
    )


def _slot_content(
    value: "ObjectSceneAnchorSupportSheetSlotPlacement",
) -> dict[str, object]:
    return {
        "schema": OBJECT_SCENE_ANCHOR_SUPPORT_SHEET_SLOT_SCHEMA,
        "slot_index": value.slot_index,
        "slot_id": value.slot_id,
        "atlas_row_index": value.atlas_row_index,
        "atlas_column_index": value.atlas_column_index,
        "sheet_x_pixels": value.sheet_x_pixels,
        "sheet_y_pixels": value.sheet_y_pixels,
        "width_pixels": value.width_pixels,
        "height_pixels": value.height_pixels,
        "slot_kind": value.slot_kind,
        "anchor_kind": value.anchor_kind,
        "anchor_id": value.anchor_id,
        "binding_alias": value.binding_alias,
        "subject_digest": value.subject_digest,
        "atlas_slot_digest": value.atlas_slot_digest,
    }


@dataclass(frozen=True, slots=True)
class ObjectSceneAnchorSupportSheetSlotPlacement:
    """One used atlas tile and its complete-kind binding alias."""

    slot_index: int
    slot_id: str
    atlas_row_index: int
    atlas_column_index: int
    sheet_x_pixels: int
    sheet_y_pixels: int
    width_pixels: int
    height_pixels: int
    slot_kind: str
    anchor_kind: str
    anchor_id: str
    binding_alias: str
    subject_digest: str
    atlas_slot_digest: str
    placement_digest: str

    def __post_init__(self) -> None:
        index = _integer(self.slot_index, "support-sheet slot index")
        if (
            self.slot_id != f"slot-{index:04d}"
            or _SLOT_ID.fullmatch(self.slot_id) is None
            or self.atlas_row_index
            != index // OBJECT_SCENE_ANCHOR_ATLAS_COLUMNS
            or self.atlas_column_index
            != index % OBJECT_SCENE_ANCHOR_ATLAS_COLUMNS
            or self.width_pixels != OBJECT_SCENE_ANCHOR_ATLAS_TILE_SIZE_PIXELS
            or self.height_pixels != OBJECT_SCENE_ANCHOR_ATLAS_TILE_SIZE_PIXELS
        ):
            raise ObjectSceneAnchorSupportSheetError(
                "support-sheet slot identity or extent differs"
            )
        for label, item in (
            ("atlas slot row", self.atlas_row_index),
            ("atlas slot column", self.atlas_column_index),
            ("support-sheet slot x", self.sheet_x_pixels),
            ("support-sheet slot y", self.sheet_y_pixels),
        ):
            _integer(item, label)
        if self.slot_kind not in (
            "whole_entity",
            "part_anchor",
            "compact_anchor",
            "cyclic_frame",
        ):
            raise ObjectSceneAnchorSupportSheetError(
                "support-sheet slot kind differs"
            )
        if self.anchor_kind not in ("entity", "part", "frame"):
            raise ObjectSceneAnchorSupportSheetError(
                "support-sheet slot anchor kind differs"
            )
        if (
            not isinstance(self.anchor_id, str)
            or not self.anchor_id
            or _BINDING_ALIAS.fullmatch(self.binding_alias) is None
        ):
            raise ObjectSceneAnchorSupportSheetError(
                "support-sheet slot binding identity differs"
            )
        _digest(self.subject_digest, "support-sheet slot subject digest")
        _digest(self.atlas_slot_digest, "support-sheet atlas slot digest")
        _digest(self.placement_digest, "support-sheet slot placement digest")
        if self.placement_digest != canonical_digest(_slot_content(self)):
            raise ObjectSceneAnchorSupportSheetError(
                "support-sheet slot placement digest differs"
            )

    def to_data(self) -> dict[str, object]:
        return {
            **_slot_content(self),
            "placement_digest": self.placement_digest,
        }

    @classmethod
    def from_data(
        cls, value: object
    ) -> "ObjectSceneAnchorSupportSheetSlotPlacement":
        raw = _exact_fields(
            value,
            {
                "schema",
                "slot_index",
                "slot_id",
                "atlas_row_index",
                "atlas_column_index",
                "sheet_x_pixels",
                "sheet_y_pixels",
                "width_pixels",
                "height_pixels",
                "slot_kind",
                "anchor_kind",
                "anchor_id",
                "binding_alias",
                "subject_digest",
                "atlas_slot_digest",
                "placement_digest",
            },
            "support-sheet slot placement",
        )
        if raw["schema"] != OBJECT_SCENE_ANCHOR_SUPPORT_SHEET_SLOT_SCHEMA:
            raise ObjectSceneAnchorSupportSheetError(
                "support-sheet slot placement schema differs"
            )
        result = cls(
            slot_index=raw["slot_index"],
            slot_id=raw["slot_id"],
            atlas_row_index=raw["atlas_row_index"],
            atlas_column_index=raw["atlas_column_index"],
            sheet_x_pixels=raw["sheet_x_pixels"],
            sheet_y_pixels=raw["sheet_y_pixels"],
            width_pixels=raw["width_pixels"],
            height_pixels=raw["height_pixels"],
            slot_kind=raw["slot_kind"],
            anchor_kind=raw["anchor_kind"],
            anchor_id=raw["anchor_id"],
            binding_alias=raw["binding_alias"],
            subject_digest=raw["subject_digest"],
            atlas_slot_digest=raw["atlas_slot_digest"],
            placement_digest=raw["placement_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectSceneAnchorSupportSheetError(
                "support-sheet slot placement is not canonical"
            )
        return result


def _object_content(
    value: "ObjectSceneAnchorSupportSheetObject",
) -> dict[str, object]:
    return {
        "schema": OBJECT_SCENE_ANCHOR_SUPPORT_SHEET_OBJECT_SCHEMA,
        "inventory_index": value.inventory_index,
        "object_id": value.object_id,
        "decision_manifest_digest": value.decision_manifest_digest,
        "crop_x_pixels": value.crop_x_pixels,
        "crop_y_pixels": value.crop_y_pixels,
        "crop_width_pixels": value.crop_width_pixels,
        "crop_height_pixels": value.crop_height_pixels,
        "crop_png_byte_count": value.crop_png_byte_count,
        "crop_png_digest": value.crop_png_digest,
        "atlas_x_pixels": value.atlas_x_pixels,
        "atlas_y_pixels": value.atlas_y_pixels,
        "atlas_width_pixels": value.atlas_width_pixels,
        "atlas_height_pixels": value.atlas_height_pixels,
        "atlas_slot_count": value.atlas_slot_count,
        "atlas_slots": [item.to_data() for item in value.atlas_slots],
        "atlas_slot_map_digest": value.atlas_slot_map_digest,
        "atlas_png_byte_count": value.atlas_png_byte_count,
        "atlas_png_digest": value.atlas_png_digest,
        "atlas_artifact_digest": value.atlas_artifact_digest,
    }


@dataclass(frozen=True, slots=True)
class ObjectSceneAnchorSupportSheetObject:
    """One complete native-resolution crop/atlas section."""

    inventory_index: int
    object_id: str
    decision_manifest_digest: str
    crop_x_pixels: int
    crop_y_pixels: int
    crop_width_pixels: int
    crop_height_pixels: int
    crop_png_byte_count: int
    crop_png_digest: str
    atlas_x_pixels: int
    atlas_y_pixels: int
    atlas_width_pixels: int
    atlas_height_pixels: int
    atlas_slot_count: int
    atlas_slots: tuple[ObjectSceneAnchorSupportSheetSlotPlacement, ...]
    atlas_slot_map_digest: str
    atlas_png_byte_count: int
    atlas_png_digest: str
    atlas_artifact_digest: str
    presentation_digest: str

    def __post_init__(self) -> None:
        index = _integer(self.inventory_index, "support-sheet inventory index")
        if (
            self.object_id != f"object_{index:04d}"
            or _OBJECT_ID.fullmatch(self.object_id) is None
        ):
            raise ObjectSceneAnchorSupportSheetError(
                "support-sheet object order differs"
            )
        for label, item in (
            ("decision manifest digest", self.decision_manifest_digest),
            ("crop PNG digest", self.crop_png_digest),
            ("atlas slot-map digest", self.atlas_slot_map_digest),
            ("atlas PNG digest", self.atlas_png_digest),
            ("atlas artifact digest", self.atlas_artifact_digest),
        ):
            _digest(item, label)
        for label, item, minimum in (
            ("crop x", self.crop_x_pixels, 0),
            ("crop y", self.crop_y_pixels, 0),
            ("crop width", self.crop_width_pixels, 1),
            ("crop height", self.crop_height_pixels, 1),
            ("crop PNG byte count", self.crop_png_byte_count, 1),
            ("atlas x", self.atlas_x_pixels, 0),
            ("atlas y", self.atlas_y_pixels, 0),
            ("atlas width", self.atlas_width_pixels, 1),
            ("atlas height", self.atlas_height_pixels, 1),
            ("atlas slot count", self.atlas_slot_count, 1),
            ("atlas PNG byte count", self.atlas_png_byte_count, 1),
        ):
            _integer(item, label, minimum=minimum)
        if type(self.atlas_slots) is not tuple or any(
            type(item) is not ObjectSceneAnchorSupportSheetSlotPlacement
            for item in self.atlas_slots
        ):
            raise TypeError("support-sheet atlas slots must be exact tuples")
        if (
            self.atlas_slot_count != len(self.atlas_slots)
            or tuple(item.slot_index for item in self.atlas_slots)
            != tuple(range(self.atlas_slot_count))
            or tuple(item.slot_id for item in self.atlas_slots)
            != tuple(f"slot-{index:04d}" for index in range(self.atlas_slot_count))
        ):
            raise ObjectSceneAnchorSupportSheetError(
                "support-sheet atlas slot inventory differs"
            )
        kind_counts = {"entity": 0, "part": 0, "frame": 0}
        for slot in self.atlas_slots:
            expected_kind = {
                "whole_entity": "entity",
                "part_anchor": "part",
                "compact_anchor": "part",
                "cyclic_frame": "frame",
            }[slot.slot_kind]
            expected_anchor_id = (
                "entity" if slot.slot_kind == "whole_entity" else slot.anchor_id
            )
            expected_alias = f"binding_{kind_counts[expected_kind]:03d}"
            if (
                slot.anchor_kind != expected_kind
                or slot.anchor_id != expected_anchor_id
                or slot.binding_alias != expected_alias
                or slot.sheet_x_pixels
                != self.atlas_x_pixels
                + slot.atlas_column_index
                * OBJECT_SCENE_ANCHOR_ATLAS_TILE_SIZE_PIXELS
                or slot.sheet_y_pixels
                != self.atlas_y_pixels
                + slot.atlas_row_index
                * OBJECT_SCENE_ANCHOR_ATLAS_TILE_SIZE_PIXELS
            ):
                raise ObjectSceneAnchorSupportSheetError(
                    "support-sheet atlas slot binding or placement differs"
                )
            kind_counts[expected_kind] += 1
        _digest(self.presentation_digest, "support-sheet object presentation digest")
        if self.presentation_digest != canonical_digest(_object_content(self)):
            raise ObjectSceneAnchorSupportSheetError(
                "support-sheet object presentation digest differs"
            )

    def to_data(self) -> dict[str, object]:
        return {
            **_object_content(self),
            "presentation_digest": self.presentation_digest,
        }

    @classmethod
    def from_data(cls, value: object) -> "ObjectSceneAnchorSupportSheetObject":
        raw = _exact_fields(
            value,
            {
                "schema",
                "inventory_index",
                "object_id",
                "decision_manifest_digest",
                "crop_x_pixels",
                "crop_y_pixels",
                "crop_width_pixels",
                "crop_height_pixels",
                "crop_png_byte_count",
                "crop_png_digest",
                "atlas_x_pixels",
                "atlas_y_pixels",
                "atlas_width_pixels",
                "atlas_height_pixels",
                "atlas_slot_count",
                "atlas_slots",
                "atlas_slot_map_digest",
                "atlas_png_byte_count",
                "atlas_png_digest",
                "atlas_artifact_digest",
                "presentation_digest",
            },
            "support-sheet object presentation",
        )
        if (
            raw["schema"] != OBJECT_SCENE_ANCHOR_SUPPORT_SHEET_OBJECT_SCHEMA
            or not isinstance(raw["atlas_slots"], list)
        ):
            raise ObjectSceneAnchorSupportSheetError(
                "support-sheet object presentation policy differs"
            )
        result = cls(
            inventory_index=raw["inventory_index"],
            object_id=raw["object_id"],
            decision_manifest_digest=raw["decision_manifest_digest"],
            crop_x_pixels=raw["crop_x_pixels"],
            crop_y_pixels=raw["crop_y_pixels"],
            crop_width_pixels=raw["crop_width_pixels"],
            crop_height_pixels=raw["crop_height_pixels"],
            crop_png_byte_count=raw["crop_png_byte_count"],
            crop_png_digest=raw["crop_png_digest"],
            atlas_x_pixels=raw["atlas_x_pixels"],
            atlas_y_pixels=raw["atlas_y_pixels"],
            atlas_width_pixels=raw["atlas_width_pixels"],
            atlas_height_pixels=raw["atlas_height_pixels"],
            atlas_slot_count=raw["atlas_slot_count"],
            atlas_slots=tuple(
                ObjectSceneAnchorSupportSheetSlotPlacement.from_data(item)
                for item in raw["atlas_slots"]
            ),
            atlas_slot_map_digest=raw["atlas_slot_map_digest"],
            atlas_png_byte_count=raw["atlas_png_byte_count"],
            atlas_png_digest=raw["atlas_png_digest"],
            atlas_artifact_digest=raw["atlas_artifact_digest"],
            presentation_digest=raw["presentation_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectSceneAnchorSupportSheetError(
                "support-sheet object presentation is not canonical"
            )
        return result


def _sheet_content(value: "ObjectSceneAnchorSupportSheet") -> dict[str, object]:
    return {
        "schema": OBJECT_SCENE_ANCHOR_SUPPORT_SHEET_SCHEMA,
        "algorithm_id": OBJECT_SCENE_ANCHOR_SUPPORT_SHEET_ALGORITHM_ID,
        "renderer_digest": value.renderer_digest,
        "panel_manifest_digest": value.panel_manifest_digest,
        "panel_digest": value.panel_digest,
        "inventory_digest": value.inventory_digest,
        "proposal_count": value.proposal_count,
        "object_ids": list(value.object_ids),
        "panel_x_pixels": value.panel_x_pixels,
        "panel_y_pixels": value.panel_y_pixels,
        "panel_width_pixels": value.panel_width_pixels,
        "panel_height_pixels": value.panel_height_pixels,
        "original_panel_png_byte_count": value.original_panel_png_byte_count,
        "original_panel_png_digest": value.original_panel_png_digest,
        "objects": [item.to_data() for item in value.objects],
        "sheet_width_pixels": value.sheet_width_pixels,
        "sheet_height_pixels": value.sheet_height_pixels,
        "sheet_png_byte_count": value.sheet_png_byte_count,
        "sheet_png_digest": value.sheet_png_digest,
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class ObjectSceneAnchorSupportSheet:
    """Decision-only receipt for one complete proposer support image."""

    renderer_digest: str
    panel_manifest_digest: str
    panel_digest: str
    inventory_digest: str
    proposal_count: int
    object_ids: tuple[str, ...]
    panel_x_pixels: int
    panel_y_pixels: int
    panel_width_pixels: int
    panel_height_pixels: int
    original_panel_png_byte_count: int
    original_panel_png_digest: str
    objects: tuple[ObjectSceneAnchorSupportSheetObject, ...]
    sheet_width_pixels: int
    sheet_height_pixels: int
    sheet_png_byte_count: int
    sheet_png_digest: str
    artifact_digest: str

    def __post_init__(self) -> None:
        for label, item in (
            ("support-sheet renderer digest", self.renderer_digest),
            ("panel manifest digest", self.panel_manifest_digest),
            ("panel digest", self.panel_digest),
            ("inventory digest", self.inventory_digest),
            ("original panel PNG digest", self.original_panel_png_digest),
            ("support-sheet PNG digest", self.sheet_png_digest),
        ):
            _digest(item, label)
        if self.renderer_digest != object_scene_anchor_support_sheet_renderer_digest():
            raise ObjectSceneAnchorSupportSheetError(
                "support-sheet renderer binding is stale"
            )
        count = _integer(self.proposal_count, "support-sheet proposal count")
        if type(self.object_ids) is not tuple or type(self.objects) is not tuple:
            raise TypeError("support-sheet object inventories must use exact tuples")
        expected_ids = tuple(f"object_{index:04d}" for index in range(count))
        if (
            self.object_ids != expected_ids
            or len(self.objects) != count
            or tuple(item.inventory_index for item in self.objects)
            != tuple(range(count))
            or tuple(item.object_id for item in self.objects) != self.object_ids
        ):
            raise ObjectSceneAnchorSupportSheetError(
                "support sheet does not exhaust panel objects in order"
            )
        for label, item, minimum in (
            ("panel x", self.panel_x_pixels, 0),
            ("panel y", self.panel_y_pixels, 0),
            ("panel width", self.panel_width_pixels, 2),
            ("panel height", self.panel_height_pixels, 2),
            ("original panel PNG byte count", self.original_panel_png_byte_count, 1),
            ("sheet width", self.sheet_width_pixels, 1),
            ("sheet height", self.sheet_height_pixels, 1),
            ("sheet PNG byte count", self.sheet_png_byte_count, 1),
        ):
            _integer(item, label, minimum=minimum)
        expected_layout = _layout(
            self.panel_width_pixels,
            self.panel_height_pixels,
            tuple(
                (
                    item.crop_width_pixels,
                    item.crop_height_pixels,
                    item.atlas_width_pixels,
                    item.atlas_height_pixels,
                )
                for item in self.objects
            ),
        )
        if (
            (self.panel_x_pixels, self.panel_y_pixels)
            != expected_layout[0]
            or tuple(
                (
                    item.crop_x_pixels,
                    item.crop_y_pixels,
                    item.atlas_x_pixels,
                    item.atlas_y_pixels,
                )
                for item in self.objects
            )
            != expected_layout[1]
            or (self.sheet_width_pixels, self.sheet_height_pixels)
            != expected_layout[2]
            or self.sheet_width_pixels * self.sheet_height_pixels
            > _MAX_SHEET_PIXELS
        ):
            raise ObjectSceneAnchorSupportSheetError(
                "support-sheet native layout differs"
            )
        unsigned = _sheet_content(self)
        _assert_decision_only_keys(unsigned)
        _digest(self.artifact_digest, "support-sheet artifact digest")
        if self.artifact_digest != canonical_digest(unsigned):
            raise ObjectSceneAnchorSupportSheetError(
                "support-sheet artifact digest differs"
            )

    @property
    def by_object_id(self) -> dict[str, ObjectSceneAnchorSupportSheetObject]:
        return dict(zip(self.object_ids, self.objects, strict=True))

    def to_data(self) -> dict[str, object]:
        return {**_sheet_content(self), "artifact_digest": self.artifact_digest}

    @classmethod
    def from_data(cls, value: object) -> "ObjectSceneAnchorSupportSheet":
        raw = _exact_fields(
            value,
            {
                "schema",
                "algorithm_id",
                "renderer_digest",
                "panel_manifest_digest",
                "panel_digest",
                "inventory_digest",
                "proposal_count",
                "object_ids",
                "panel_x_pixels",
                "panel_y_pixels",
                "panel_width_pixels",
                "panel_height_pixels",
                "original_panel_png_byte_count",
                "original_panel_png_digest",
                "objects",
                "sheet_width_pixels",
                "sheet_height_pixels",
                "sheet_png_byte_count",
                "sheet_png_digest",
                *_authority_data(),
                "artifact_digest",
            },
            "object scene anchor support sheet",
        )
        if (
            raw["schema"] != OBJECT_SCENE_ANCHOR_SUPPORT_SHEET_SCHEMA
            or raw["algorithm_id"]
            != OBJECT_SCENE_ANCHOR_SUPPORT_SHEET_ALGORITHM_ID
            or any(raw[key] != item for key, item in _authority_data().items())
            or not isinstance(raw["object_ids"], list)
            or not isinstance(raw["objects"], list)
        ):
            raise ObjectSceneAnchorSupportSheetError(
                "support-sheet policy differs"
            )
        result = cls(
            renderer_digest=raw["renderer_digest"],
            panel_manifest_digest=raw["panel_manifest_digest"],
            panel_digest=raw["panel_digest"],
            inventory_digest=raw["inventory_digest"],
            proposal_count=raw["proposal_count"],
            object_ids=tuple(raw["object_ids"]),
            panel_x_pixels=raw["panel_x_pixels"],
            panel_y_pixels=raw["panel_y_pixels"],
            panel_width_pixels=raw["panel_width_pixels"],
            panel_height_pixels=raw["panel_height_pixels"],
            original_panel_png_byte_count=raw["original_panel_png_byte_count"],
            original_panel_png_digest=raw["original_panel_png_digest"],
            objects=tuple(
                ObjectSceneAnchorSupportSheetObject.from_data(item)
                for item in raw["objects"]
            ),
            sheet_width_pixels=raw["sheet_width_pixels"],
            sheet_height_pixels=raw["sheet_height_pixels"],
            sheet_png_byte_count=raw["sheet_png_byte_count"],
            sheet_png_digest=raw["sheet_png_digest"],
            artifact_digest=raw["artifact_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectSceneAnchorSupportSheetError(
                "support sheet is not canonical"
            )
        return result


def _layout(
    panel_width: int,
    panel_height: int,
    dimensions: tuple[tuple[int, int, int, int], ...],
) -> tuple[
    tuple[int, int],
    tuple[tuple[int, int, int, int], ...],
    tuple[int, int],
]:
    padding = OBJECT_SCENE_ANCHOR_SUPPORT_SHEET_PADDING_PIXELS
    border = OBJECT_SCENE_ANCHOR_SUPPORT_SHEET_BORDER_PIXELS
    component_gap = OBJECT_SCENE_ANCHOR_SUPPORT_SHEET_COMPONENT_GAP_PIXELS
    section_gap = OBJECT_SCENE_ANCHOR_SUPPORT_SHEET_SECTION_GAP_PIXELS
    panel_x = padding + border
    panel_y = padding + border
    content_width = panel_width + 2 * border
    rows: list[tuple[int, int, int, int]] = []
    y = padding + panel_height + 2 * border
    if dimensions:
        y += section_gap
    for crop_width, crop_height, atlas_width, atlas_height in dimensions:
        crop_x = padding + border
        crop_y = y + border
        atlas_x = padding + 2 * border + crop_width + component_gap + border
        atlas_y = y + border
        rows.append((crop_x, crop_y, atlas_x, atlas_y))
        content_width = max(
            content_width,
            crop_width + 2 * border + component_gap + atlas_width + 2 * border,
        )
        y += max(crop_height, atlas_height) + 2 * border + section_gap
    if dimensions:
        y -= section_gap
    sheet_width = 2 * padding + content_width
    sheet_height = y + padding
    return (panel_x, panel_y), tuple(rows), (sheet_width, sheet_height)


def _decode_luminance(png_bytes: bytes, label: str) -> np.ndarray:
    try:
        if not isinstance(png_bytes, bytes) or not png_bytes.startswith(
            b"\x89PNG\r\n\x1a\n"
        ):
            raise ValueError("input lacks the PNG signature")
        with Image.open(BytesIO(png_bytes)) as encoded:
            if encoded.format != "PNG" or getattr(encoded, "n_frames", 1) != 1:
                raise ValueError("input is not one PNG frame")
            width, height = encoded.size
            if width < 1 or height < 1 or width * height > _MAX_SHEET_PIXELS:
                raise ValueError("PNG dimensions exceed the component guard")
            rgba = np.asarray(encoded.convert("RGBA"), dtype=np.uint8)
        values = rgba.astype(np.uint32, copy=False)
        alpha = values[..., 3:4]
        rgb = (values[..., :3] * alpha + 255 * (255 - alpha) + 127) // 255
        luminance = np.min(rgb, axis=2).astype(np.uint8)
    except Exception as exc:
        raise ObjectSceneAnchorSupportSheetError(
            f"{label} does not decode as one exact PNG"
        ) from exc
    return np.ascontiguousarray(luminance, dtype=np.uint8)


def _draw_border(
    pixels: np.ndarray,
    *,
    x: int,
    y: int,
    width: int,
    height: int,
    value: int,
) -> None:
    border = OBJECT_SCENE_ANCHOR_SUPPORT_SHEET_BORDER_PIXELS
    pixels[y - border : y, x - border : x + width + border] = value
    pixels[y + height : y + height + border, x - border : x + width + border] = value
    pixels[y - border : y + height + border, x - border : x] = value
    pixels[y - border : y + height + border, x + width : x + width + border] = value


def _slot_binding_identity(
    slot: ObjectSceneAnchorAtlasSlot,
    counts: dict[str, int],
) -> tuple[str, str, str]:
    if slot.slot_kind == "whole_entity":
        anchor_kind, anchor_id = "entity", "entity"
    elif slot.slot_kind in ("part_anchor", "compact_anchor"):
        anchor_kind, anchor_id = "part", slot.subject_id
    else:
        anchor_kind, anchor_id = "frame", slot.subject_id
    binding_alias = f"binding_{counts[anchor_kind]:03d}"
    counts[anchor_kind] += 1
    return anchor_kind, anchor_id, binding_alias


def _make_slot_placement(
    slot: ObjectSceneAnchorAtlasSlot,
    *,
    atlas_x: int,
    atlas_y: int,
    counts: dict[str, int],
) -> ObjectSceneAnchorSupportSheetSlotPlacement:
    anchor_kind, anchor_id, binding_alias = _slot_binding_identity(slot, counts)
    tile = OBJECT_SCENE_ANCHOR_ATLAS_TILE_SIZE_PIXELS
    values = {
        "slot_index": slot.slot_index,
        "slot_id": slot.slot_id,
        "atlas_row_index": slot.row_index,
        "atlas_column_index": slot.column_index,
        "sheet_x_pixels": atlas_x + slot.column_index * tile,
        "sheet_y_pixels": atlas_y + slot.row_index * tile,
        "width_pixels": tile,
        "height_pixels": tile,
        "slot_kind": slot.slot_kind,
        "anchor_kind": anchor_kind,
        "anchor_id": anchor_id,
        "binding_alias": binding_alias,
        "subject_digest": slot.subject_digest,
        "atlas_slot_digest": slot.slot_digest,
    }
    provisional = object.__new__(ObjectSceneAnchorSupportSheetSlotPlacement)
    for name, item in values.items():
        object.__setattr__(provisional, name, item)
    return ObjectSceneAnchorSupportSheetSlotPlacement(
        **values,
        placement_digest=canonical_digest(_slot_content(provisional)),
    )


def _make_object_presentation(
    *,
    inventory_index: int,
    object_id: str,
    decision_manifest_digest: str,
    crop_x: int,
    crop_y: int,
    crop_pixels: np.ndarray,
    crop_png: bytes,
    atlas_x: int,
    atlas_y: int,
    atlas: ObjectSceneAnchorAtlas,
    atlas_pixels: np.ndarray,
    atlas_png: bytes,
) -> ObjectSceneAnchorSupportSheetObject:
    counts = {"entity": 0, "part": 0, "frame": 0}
    slots = tuple(
        _make_slot_placement(
            slot,
            atlas_x=atlas_x,
            atlas_y=atlas_y,
            counts=counts,
        )
        for slot in atlas.slots
    )
    values = {
        "inventory_index": inventory_index,
        "object_id": object_id,
        "decision_manifest_digest": decision_manifest_digest,
        "crop_x_pixels": crop_x,
        "crop_y_pixels": crop_y,
        "crop_width_pixels": int(crop_pixels.shape[1]),
        "crop_height_pixels": int(crop_pixels.shape[0]),
        "crop_png_byte_count": len(crop_png),
        "crop_png_digest": hashlib.sha256(crop_png).hexdigest(),
        "atlas_x_pixels": atlas_x,
        "atlas_y_pixels": atlas_y,
        "atlas_width_pixels": int(atlas_pixels.shape[1]),
        "atlas_height_pixels": int(atlas_pixels.shape[0]),
        "atlas_slot_count": atlas.slot_count,
        "atlas_slots": slots,
        "atlas_slot_map_digest": atlas.slot_map_digest,
        "atlas_png_byte_count": len(atlas_png),
        "atlas_png_digest": hashlib.sha256(atlas_png).hexdigest(),
        "atlas_artifact_digest": atlas.artifact_digest,
    }
    provisional = object.__new__(ObjectSceneAnchorSupportSheetObject)
    for name, item in values.items():
        object.__setattr__(provisional, name, item)
    return ObjectSceneAnchorSupportSheetObject(
        **values,
        presentation_digest=canonical_digest(_object_content(provisional)),
    )


def _make_sheet(
    *,
    panel_manifest: ObjectSceneAnchorPanelDecisionManifest,
    original_png: bytes,
    panel_pixels: np.ndarray,
    components: tuple[
        tuple[bytes, np.ndarray, ObjectSceneAnchorAtlas, bytes, np.ndarray], ...
    ],
) -> tuple[ObjectSceneAnchorSupportSheet, bytes]:
    dimensions = tuple(
        (
            int(crop_pixels.shape[1]),
            int(crop_pixels.shape[0]),
            int(atlas_pixels.shape[1]),
            int(atlas_pixels.shape[0]),
        )
        for _, crop_pixels, _, _, atlas_pixels in components
    )
    panel_placement, object_placements, sheet_size = _layout(
        panel_manifest.width_pixels,
        panel_manifest.height_pixels,
        dimensions,
    )
    sheet_width, sheet_height = sheet_size
    if sheet_width * sheet_height > _MAX_SHEET_PIXELS:
        raise ObjectSceneAnchorSupportSheetError(
            "support sheet exceeds the fixed pixel guard"
        )
    pixels = np.full(
        (sheet_height, sheet_width), _BACKGROUND_VALUE, dtype=np.uint8
    )
    panel_x, panel_y = panel_placement
    pixels[
        panel_y : panel_y + panel_pixels.shape[0],
        panel_x : panel_x + panel_pixels.shape[1],
    ] = panel_pixels
    _draw_border(
        pixels,
        x=panel_x,
        y=panel_y,
        width=int(panel_pixels.shape[1]),
        height=int(panel_pixels.shape[0]),
        value=_PANEL_BORDER_VALUE,
    )
    presentations: list[ObjectSceneAnchorSupportSheetObject] = []
    for index, (component, placement) in enumerate(
        zip(components, object_placements, strict=True)
    ):
        crop_png, crop_pixels, atlas, atlas_png, atlas_pixels = component
        crop_x, crop_y, atlas_x, atlas_y = placement
        pixels[
            crop_y : crop_y + crop_pixels.shape[0],
            crop_x : crop_x + crop_pixels.shape[1],
        ] = crop_pixels
        pixels[
            atlas_y : atlas_y + atlas_pixels.shape[0],
            atlas_x : atlas_x + atlas_pixels.shape[1],
        ] = atlas_pixels
        _draw_border(
            pixels,
            x=crop_x,
            y=crop_y,
            width=int(crop_pixels.shape[1]),
            height=int(crop_pixels.shape[0]),
            value=_CROP_BORDER_VALUE,
        )
        _draw_border(
            pixels,
            x=atlas_x,
            y=atlas_y,
            width=int(atlas_pixels.shape[1]),
            height=int(atlas_pixels.shape[0]),
            value=_ATLAS_BORDER_VALUE,
        )
        decision = panel_manifest.object_decisions[index]
        presentations.append(
            _make_object_presentation(
                inventory_index=index,
                object_id=panel_manifest.object_ids[index],
                decision_manifest_digest=decision.manifest_digest,
                crop_x=crop_x,
                crop_y=crop_y,
                crop_pixels=crop_pixels,
                crop_png=crop_png,
                atlas_x=atlas_x,
                atlas_y=atlas_y,
                atlas=atlas,
                atlas_pixels=atlas_pixels,
                atlas_png=atlas_png,
            )
        )
    sheet_png = _encode_grayscale_png(pixels)
    values = {
        "renderer_digest": object_scene_anchor_support_sheet_renderer_digest(),
        "panel_manifest_digest": panel_manifest.manifest_digest,
        "panel_digest": panel_manifest.panel_digest,
        "inventory_digest": panel_manifest.inventory_digest,
        "proposal_count": panel_manifest.proposal_count,
        "object_ids": panel_manifest.object_ids,
        "panel_x_pixels": panel_x,
        "panel_y_pixels": panel_y,
        "panel_width_pixels": panel_manifest.width_pixels,
        "panel_height_pixels": panel_manifest.height_pixels,
        "original_panel_png_byte_count": len(original_png),
        "original_panel_png_digest": hashlib.sha256(original_png).hexdigest(),
        "objects": tuple(presentations),
        "sheet_width_pixels": sheet_width,
        "sheet_height_pixels": sheet_height,
        "sheet_png_byte_count": len(sheet_png),
        "sheet_png_digest": hashlib.sha256(sheet_png).hexdigest(),
    }
    provisional = object.__new__(ObjectSceneAnchorSupportSheet)
    for name, item in values.items():
        object.__setattr__(provisional, name, item)
    return ObjectSceneAnchorSupportSheet(
        **values,
        artifact_digest=canonical_digest(_sheet_content(provisional)),
    ), sheet_png


def build_object_scene_anchor_support_sheet(
    png_bytes: bytes,
    inventory: ObjectSceneProposalInventory,
    catalog: ObjectSceneAnchorCatalog,
    panel_manifest: ObjectSceneAnchorPanelDecisionManifest,
) -> tuple[ObjectSceneAnchorSupportSheet, bytes]:
    """Cold-verify all panel inputs, then render the complete support sheet."""

    if not isinstance(png_bytes, bytes):
        raise TypeError("support-sheet panel input must be exact PNG bytes")
    if type(inventory) is not ObjectSceneProposalInventory:
        raise TypeError("inventory must be exact ObjectSceneProposalInventory")
    if type(catalog) is not ObjectSceneAnchorCatalog:
        raise TypeError("catalog must be exact ObjectSceneAnchorCatalog")
    if type(panel_manifest) is not ObjectSceneAnchorPanelDecisionManifest:
        raise TypeError(
            "panel_manifest must be exact ObjectSceneAnchorPanelDecisionManifest"
        )
    manifest = verify_object_scene_anchor_panel_decision_manifest(
        panel_manifest,
        catalog,
        png_bytes,
        inventory,
        expected_manifest_digest=panel_manifest.manifest_digest,
    )
    panel_pixels = _decode_luminance(png_bytes, "support-sheet original panel")
    if panel_pixels.shape != (manifest.height_pixels, manifest.width_pixels):
        raise ObjectSceneAnchorSupportSheetError(
            "support-sheet panel dimensions differ from the manifest"
        )
    components: list[
        tuple[bytes, np.ndarray, ObjectSceneAnchorAtlas, bytes, np.ndarray]
    ] = []
    for index, (object_id, decision, entry) in enumerate(
        zip(
            manifest.object_ids,
            manifest.object_decisions,
            catalog.entries,
            strict=True,
        )
    ):
        if (
            entry.inventory_index != index
            or entry.object_id != object_id
            or entry.decision_manifest != decision
        ):
            raise ObjectSceneAnchorSupportSheetError(
                "support-sheet object input order differs"
            )
        crop_png = render_object_scene_anchor_object_crop(
            png_bytes, inventory, entry
        )
        crop_pixels = _decode_luminance(crop_png, f"{object_id} crop")
        atlas, atlas_png_or_none = render_object_scene_anchor_atlas(decision)
        if atlas_png_or_none is None or atlas.status.state != "clean":
            raise ObjectSceneAnchorSupportSheetError(
                f"{object_id} lacks a clean exhaustive anchor atlas"
            )
        atlas_png = atlas_png_or_none
        atlas_pixels = _decode_luminance(atlas_png, f"{object_id} atlas")
        if (
            crop_pixels.shape
            != (entry.crop_height_pixels, entry.crop_width_pixels)
            or atlas_pixels.shape
            != (atlas.image_height_pixels, atlas.image_width_pixels)
        ):
            raise ObjectSceneAnchorSupportSheetError(
                "support-sheet component dimensions differ"
            )
        components.append(
            (crop_png, crop_pixels, atlas, atlas_png, atlas_pixels)
        )
    return _make_sheet(
        panel_manifest=manifest,
        original_png=png_bytes,
        panel_pixels=panel_pixels,
        components=tuple(components),
    )


def verify_object_scene_anchor_support_sheet(
    artifact: ObjectSceneAnchorSupportSheet,
    sheet_png_bytes: bytes,
    png_bytes: bytes,
    inventory: ObjectSceneProposalInventory,
    catalog: ObjectSceneAnchorCatalog,
    panel_manifest: ObjectSceneAnchorPanelDecisionManifest,
    *,
    expected_artifact_digest: str | None = None,
) -> ObjectSceneAnchorSupportSheet:
    """Cold-replay the manifest, every component, and the exact sheet bytes."""

    if type(artifact) is not ObjectSceneAnchorSupportSheet:
        raise TypeError("artifact must be exact ObjectSceneAnchorSupportSheet")
    if not isinstance(sheet_png_bytes, bytes):
        raise TypeError("support-sheet PNG must be exact bytes")
    restored = ObjectSceneAnchorSupportSheet.from_data(artifact.to_data())
    if expected_artifact_digest is not None and restored.artifact_digest != _digest(
        expected_artifact_digest, "expected support-sheet artifact digest"
    ):
        raise ObjectSceneAnchorSupportSheetError(
            "support sheet differs from commitment"
        )
    replayed, replayed_png = build_object_scene_anchor_support_sheet(
        png_bytes,
        inventory,
        catalog,
        panel_manifest,
    )
    if replayed != restored or replayed_png != sheet_png_bytes:
        raise ObjectSceneAnchorSupportSheetError(
            "support sheet differs from exact panel/component replay"
        )
    return restored


render_object_scene_anchor_support_sheet = build_object_scene_anchor_support_sheet


__all__ = (
    "OBJECT_SCENE_ANCHOR_SUPPORT_SHEET_ALGORITHM_ID",
    "OBJECT_SCENE_ANCHOR_SUPPORT_SHEET_BORDER_PIXELS",
    "OBJECT_SCENE_ANCHOR_SUPPORT_SHEET_COMPONENT_GAP_PIXELS",
    "OBJECT_SCENE_ANCHOR_SUPPORT_SHEET_MODE",
    "OBJECT_SCENE_ANCHOR_SUPPORT_SHEET_OBJECT_SCHEMA",
    "OBJECT_SCENE_ANCHOR_SUPPORT_SHEET_PADDING_PIXELS",
    "OBJECT_SCENE_ANCHOR_SUPPORT_SHEET_SCHEMA",
    "OBJECT_SCENE_ANCHOR_SUPPORT_SHEET_SECTION_GAP_PIXELS",
    "OBJECT_SCENE_ANCHOR_SUPPORT_SHEET_SLOT_SCHEMA",
    "ObjectSceneAnchorSupportSheet",
    "ObjectSceneAnchorSupportSheetError",
    "ObjectSceneAnchorSupportSheetObject",
    "ObjectSceneAnchorSupportSheetSlotPlacement",
    "build_object_scene_anchor_support_sheet",
    "object_scene_anchor_support_sheet_renderer_digest",
    "object_scene_anchor_support_sheet_source_digest",
    "render_object_scene_anchor_support_sheet",
    "verify_object_scene_anchor_support_sheet",
)
