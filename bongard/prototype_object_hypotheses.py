"""Candidate-independent multicomponent hypotheses and a canonical atlas.

Connected foreground components are low-level raster regions, not semantic
objects.  This module therefore enumerates deterministic *candidate* unions
before any prototype profile or rubric exists.  It makes no claim that the
catalog contains the semantically correct object.

For each frozen visual-witness scenario, the catalog contains every singleton
and every connected cluster that exists at any unique exact integer chessboard
mask-gap threshold under single linkage.  Atlas PNGs are rendered by a small
font-free, uncompressed encoder implemented here; no Lean value is imported or
consulted.  Exact input pixels remain the sole replay authority.
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
from scipy import ndimage

from bongard.canonical import canonical_digest, canonical_json
from bongard import visual_witnesses as _visual
from bongard.prototype_visual_runtime import visual_runtime_dependency_digest
from bongard.python_predicate_authority import PYTHON_PREDICATE_AUTHORITY_ID
from bongard.visual_witnesses import Q16BBox, VISUAL_WITNESS_SCENARIO_IDS


OBJECT_HYPOTHESIS_PACKET_SCHEMA = "gkm.bongard-object-hypothesis-packet.v1"
OBJECT_HYPOTHESIS_ALGORITHM_ID = (
    "bongard.prototype-object-hypotheses/chessboard-single-linkage-v1"
)
OBJECT_HYPOTHESIS_EXTRACTOR_ID = "bongard.prototype_object_hypotheses"
OBJECT_HYPOTHESIS_EXTRACTOR_VERSION = "1"

ATLAS_COLUMNS = 4
ATLAS_ROWS = 4
ATLAS_SLOT_SIZE_PIXELS = 128
ATLAS_SLOT_CAPACITY = ATLAS_COLUMNS * ATLAS_ROWS
ATLAS_MAX_SHEETS = 32
ATLAS_WIDTH_PIXELS = ATLAS_COLUMNS * ATLAS_SLOT_SIZE_PIXELS
ATLAS_HEIGHT_PIXELS = ATLAS_ROWS * ATLAS_SLOT_SIZE_PIXELS

_SLOT_BORDER_PIXELS = 2
_SLOT_MARGIN_PIXELS = 8
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_HYPOTHESIS_ID = re.compile(r"hypothesis-[0-9]{8}\Z")
_SLOT_ID = re.compile(r"slot-[0-9]{8}\Z")
_SHEET_NAME = re.compile(r"sheet_[0-9]{3}\.png\Z")


class ObjectHypothesisError(ValueError):
    """The candidate catalog, atlas, or replay commitment is invalid."""


def _exact_fields(
    value: object, expected: set[str] | frozenset[str], label: str
) -> Mapping[str, Any]:
    if (
        not isinstance(value, Mapping)
        or any(not isinstance(key, str) for key in value)
        or set(value) != set(expected)
    ):
        raise ObjectHypothesisError(f"{label} fields differ from schema")
    return value


def _integer(value: object, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ObjectHypothesisError(f"{label} must be an integer >= {minimum}")
    return value


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ObjectHypothesisError(f"{label} must be a lowercase SHA-256")
    return value


def _authority_data() -> dict[str, object]:
    return {
        "predicate_authority_id": PYTHON_PREDICATE_AUTHORITY_ID,
        "python_is_canonical_authority": True,
        "lean_present": False,
        "lean_required": False,
        "lean_removable": True,
        "lean_affects_decision": False,
    }


def _candidate_policy_data() -> dict[str, object]:
    return {
        "connected_components_are_low_level_regions": True,
        "hypotheses_are_candidate_groupings": True,
        "candidate_independent_of_profile_and_rubric": True,
        "semantic_object_completeness_claimed": False,
        "omission_on_atlas_overflow": False,
    }


def object_hypothesis_extractor_source_digest() -> str:
    return verify_loaded_source(
        __name__, expected_source_sha256=_LOADED_SOURCE_SHA256
    )


def object_hypothesis_extractor_artifact_digest() -> str:
    """Bind this source and the exact upstream segmentation implementation."""

    return canonical_digest(
        {
            "algorithm_id": OBJECT_HYPOTHESIS_ALGORITHM_ID,
            "source_digest": object_hypothesis_extractor_source_digest(),
            "visual_witness_extractor_artifact_digest": (
                _visual.visual_witness_extractor_digest()
            ),
            "visual_runtime_dependency_digest": visual_runtime_dependency_digest(),
            "gap_metric": "max(abs(dx),abs(dy))-1 between foreground pixels",
            "cluster_rule": (
                "graph components at every unique exact pair gap threshold"
            ),
            "atlas": {
                "columns": ATLAS_COLUMNS,
                "rows": ATLAS_ROWS,
                "slot_size_pixels": ATLAS_SLOT_SIZE_PIXELS,
                "max_sheets": ATLAS_MAX_SHEETS,
                "font": None,
                "png": "grayscale8-filter0-zlib-stored-deflate-v1",
            },
            "runtime_authority": _authority_data(),
            "candidate_policy": _candidate_policy_data(),
        }
    )


def _crop_pixel_digest(crop_strength: np.ndarray) -> str:
    if crop_strength.dtype != np.uint8 or crop_strength.ndim != 2:
        raise TypeError("masked crop must be a two-dimensional uint8 array")
    height, width = crop_strength.shape
    prefix = canonical_json(
        {
            "schema": "gkm.bongard-masked-strength-crop.v1",
            "width_pixels": width,
            "height_pixels": height,
            "background_strength": 0,
            "layout": "row-major-uint8",
        }
    )
    return hashlib.sha256(prefix + b"\x00" + crop_strength.tobytes()).hexdigest()


@dataclass(frozen=True, order=True, slots=True)
class ObjectHypothesis:
    scenario_id: str
    hypothesis_id: str
    source_component_ids: tuple[str, ...]
    source_component_mask_digests: tuple[str, ...]
    union_mask_digest: str
    union_area_pixels: int
    bbox_pixels: tuple[int, int, int, int]
    bbox_q16: Q16BBox
    emergence_gap_pixels: int
    crop_width_pixels: int
    crop_height_pixels: int
    masked_crop_pixel_digest: str

    def __post_init__(self) -> None:
        if self.scenario_id not in VISUAL_WITNESS_SCENARIO_IDS:
            raise ObjectHypothesisError("unknown hypothesis scenario")
        if not isinstance(self.hypothesis_id, str) or _HYPOTHESIS_ID.fullmatch(
            self.hypothesis_id
        ) is None:
            raise ObjectHypothesisError("hypothesis_id is not canonical")
        if (
            not isinstance(self.source_component_ids, tuple)
            or not self.source_component_ids
            or tuple(sorted(set(self.source_component_ids)))
            != self.source_component_ids
        ):
            raise ObjectHypothesisError(
                "source component IDs must be a nonempty ordered unique tuple"
            )
        if any(
            re.fullmatch(r"component-[0-9]{8}", item) is None
            for item in self.source_component_ids
        ):
            raise ObjectHypothesisError("source component ID is not canonical")
        if (
            not isinstance(self.source_component_mask_digests, tuple)
            or len(self.source_component_mask_digests)
            != len(self.source_component_ids)
        ):
            raise ObjectHypothesisError("source mask digest tuple differs")
        for item in self.source_component_mask_digests:
            _digest(item, "source component mask digest")
        _digest(self.union_mask_digest, "union mask digest")
        _integer(self.union_area_pixels, "union area", minimum=1)
        if (
            not isinstance(self.bbox_pixels, tuple)
            or len(self.bbox_pixels) != 4
            or any(isinstance(item, bool) or not isinstance(item, int) for item in self.bbox_pixels)
        ):
            raise ObjectHypothesisError("pixel bbox must be an integer tuple")
        x0, y0, x1, y1 = self.bbox_pixels
        if min(x0, y0) < 0 or x1 <= x0 or y1 <= y0:
            raise ObjectHypothesisError("pixel bbox must have positive extent")
        if not isinstance(self.bbox_q16, Q16BBox):
            raise TypeError("hypothesis bbox_q16 must be Q16BBox")
        gap = _integer(self.emergence_gap_pixels, "emergence gap")
        if (len(self.source_component_ids) == 1) != (gap == 0):
            raise ObjectHypothesisError(
                "only singleton hypotheses emerge at zero mask gap"
            )
        _integer(self.crop_width_pixels, "crop width", minimum=1)
        _integer(self.crop_height_pixels, "crop height", minimum=1)
        if (
            self.crop_width_pixels != x1 - x0
            or self.crop_height_pixels != y1 - y0
        ):
            raise ObjectHypothesisError("crop dimensions differ from pixel bbox")
        _digest(self.masked_crop_pixel_digest, "masked crop pixel digest")
        if len(self.source_component_ids) == 1 and (
            self.union_mask_digest != self.source_component_mask_digests[0]
        ):
            raise ObjectHypothesisError(
                "singleton union mask must equal its component mask"
            )

    def to_data(self) -> dict[str, object]:
        return {
            "scenario_id": self.scenario_id,
            "hypothesis_id": self.hypothesis_id,
            "source_component_ids": list(self.source_component_ids),
            "source_component_mask_digests": list(
                self.source_component_mask_digests
            ),
            "union_mask_digest": self.union_mask_digest,
            "union_area_pixels": self.union_area_pixels,
            "bbox_pixels": list(self.bbox_pixels),
            "bbox_q16": self.bbox_q16.to_data(),
            "emergence_gap_pixels": self.emergence_gap_pixels,
            "crop_width_pixels": self.crop_width_pixels,
            "crop_height_pixels": self.crop_height_pixels,
            "masked_crop_pixel_digest": self.masked_crop_pixel_digest,
        }

    @classmethod
    def from_data(cls, value: Mapping[str, Any]) -> "ObjectHypothesis":
        raw = _exact_fields(
            value,
            {
                "scenario_id",
                "hypothesis_id",
                "source_component_ids",
                "source_component_mask_digests",
                "union_mask_digest",
                "union_area_pixels",
                "bbox_pixels",
                "bbox_q16",
                "emergence_gap_pixels",
                "crop_width_pixels",
                "crop_height_pixels",
                "masked_crop_pixel_digest",
            },
            "object hypothesis",
        )
        ids = raw["source_component_ids"]
        masks = raw["source_component_mask_digests"]
        pixel_bbox = raw["bbox_pixels"]
        bbox = raw["bbox_q16"]
        if not isinstance(ids, list) or not isinstance(masks, list):
            raise ObjectHypothesisError("hypothesis sources must be JSON lists")
        if not isinstance(bbox, Mapping):
            raise ObjectHypothesisError("hypothesis bbox must be an object")
        if not isinstance(pixel_bbox, list):
            raise ObjectHypothesisError("hypothesis pixel bbox must be a JSON list")
        result = cls(
            scenario_id=raw["scenario_id"],
            hypothesis_id=raw["hypothesis_id"],
            source_component_ids=tuple(ids),
            source_component_mask_digests=tuple(masks),
            union_mask_digest=raw["union_mask_digest"],
            union_area_pixels=raw["union_area_pixels"],
            bbox_pixels=tuple(pixel_bbox),
            bbox_q16=Q16BBox.from_data(bbox),
            emergence_gap_pixels=raw["emergence_gap_pixels"],
            crop_width_pixels=raw["crop_width_pixels"],
            crop_height_pixels=raw["crop_height_pixels"],
            masked_crop_pixel_digest=raw["masked_crop_pixel_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectHypothesisError("object hypothesis is not canonical")
        return result

    def digest(self) -> str:
        return canonical_digest(self.to_data())


@dataclass(frozen=True, slots=True)
class ObjectHypothesisScenario:
    scenario_id: str
    hypotheses: tuple[ObjectHypothesis, ...]

    def __post_init__(self) -> None:
        if self.scenario_id not in VISUAL_WITNESS_SCENARIO_IDS:
            raise ObjectHypothesisError("unknown hypothesis-catalog scenario")
        if not isinstance(self.hypotheses, tuple) or any(
            not isinstance(item, ObjectHypothesis) for item in self.hypotheses
        ):
            raise TypeError("hypotheses must be a typed tuple")
        if any(item.scenario_id != self.scenario_id for item in self.hypotheses):
            raise ObjectHypothesisError("hypothesis scenario differs from owner")
        expected_ids = tuple(
            f"hypothesis-{index:08d}" for index in range(len(self.hypotheses))
        )
        if tuple(item.hypothesis_id for item in self.hypotheses) != expected_ids:
            raise ObjectHypothesisError("hypothesis IDs must be consecutive")
        keys = tuple(
            (item.emergence_gap_pixels, item.source_component_ids)
            for item in self.hypotheses
        )
        if keys != tuple(sorted(keys)) or len(set(keys)) != len(keys):
            raise ObjectHypothesisError("hypothesis catalog order is not canonical")
        singleton = {
            item.source_component_ids[0]: item.source_component_mask_digests[0]
            for item in self.hypotheses
            if len(item.source_component_ids) == 1
        }
        if len(singleton) != sum(
            len(item.source_component_ids) == 1 for item in self.hypotheses
        ):
            raise ObjectHypothesisError("singleton source component is duplicated")
        for item in self.hypotheses:
            try:
                expected = tuple(singleton[x] for x in item.source_component_ids)
            except KeyError as exc:
                raise ObjectHypothesisError(
                    "multicomponent hypothesis lacks a singleton source"
                ) from exc
            if item.source_component_mask_digests != expected:
                raise ObjectHypothesisError(
                    "source mask binding differs from singleton catalog"
                )

    def to_data(self) -> dict[str, object]:
        return {
            "scenario_id": self.scenario_id,
            "hypotheses": [item.to_data() for item in self.hypotheses],
        }

    @classmethod
    def from_data(cls, value: Mapping[str, Any]) -> "ObjectHypothesisScenario":
        raw = _exact_fields(value, {"scenario_id", "hypotheses"}, "scenario catalog")
        rows = raw["hypotheses"]
        if not isinstance(rows, list) or any(
            not isinstance(item, Mapping) for item in rows
        ):
            raise ObjectHypothesisError("scenario hypotheses must be object list")
        result = cls(
            scenario_id=raw["scenario_id"],
            hypotheses=tuple(ObjectHypothesis.from_data(item) for item in rows),
        )
        if result.to_data() != dict(raw):
            raise ObjectHypothesisError("scenario catalog is not canonical")
        return result


def _component_masks(
    strength: np.ndarray, scenario: _visual.ScenarioWitness
) -> tuple[np.ndarray, ...]:
    """Replay the exact base segmentation and bind every low-level mask."""

    scenario_by_id = {row[0]: row[1:] for row in _visual._SCENARIOS}
    threshold, morphology = scenario_by_id[scenario.scenario_id]
    foreground = _visual._scenario_mask(strength, threshold, morphology)
    labels, count = ndimage.label(foreground, structure=_visual._FOREGROUND_STRUCTURE)
    rows: list[tuple[tuple[object, ...], np.ndarray]] = []
    for label in range(1, count + 1):
        mask = np.ascontiguousarray(labels == label, dtype=bool)
        x0, y0, x1, y1 = _visual._bbox(mask)
        area = int(np.count_nonzero(mask))
        digest = _visual._mask_digest(mask)
        rows.append(((x0, y0, x1, y1, area, digest), mask))
    rows.sort(key=lambda item: item[0])
    masks = tuple(item[1] for item in rows)
    if len(masks) != len(scenario.components):
        raise ObjectHypothesisError("base component replay count differs")
    height, width = strength.shape
    for component, (key, mask) in zip(scenario.components, rows, strict=True):
        x0, y0, x1, y1, area, digest = key
        if (
            component.area_pixels != area
            or component.mask_digest != digest
            or component.bbox_q16
            != _visual._q16_bbox((x0, y0, x1, y1), width, height)
        ):
            raise ObjectHypothesisError("base component replay binding differs")
    return masks


def _pair_mask_gap_from_distance(
    distance_from_first: np.ndarray, second: np.ndarray
) -> int:
    """Exact count of empty chessboard steps separating disjoint masks."""

    pixel_distance = int(np.min(distance_from_first[second]))
    if pixel_distance < 2:
        raise ObjectHypothesisError("base components overlap or are 8-connected")
    return pixel_distance - 1


def _cluster_emergence(masks: tuple[np.ndarray, ...]) -> dict[tuple[int, ...], int]:
    count = len(masks)
    emergence: dict[tuple[int, ...], int] = {(index,): 0 for index in range(count)}
    if count < 2:
        return emergence
    edges: list[tuple[int, int, int]] = []
    for first in range(count):
        distance = ndimage.distance_transform_cdt(
            ~masks[first], metric="chessboard"
        )
        for second in range(first + 1, count):
            edges.append(
                (
                    _pair_mask_gap_from_distance(distance, masks[second]),
                    first,
                    second,
                )
            )
    edges.sort()
    parent = list(range(count))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(first: int, second: int) -> None:
        left, right = find(first), find(second)
        if left != right:
            if left > right:
                left, right = right, left
            parent[right] = left

    cursor = 0
    while cursor < len(edges):
        threshold = edges[cursor][0]
        while cursor < len(edges) and edges[cursor][0] == threshold:
            _, first, second = edges[cursor]
            union(first, second)
            cursor += 1
        groups: dict[int, list[int]] = {}
        for index in range(count):
            groups.setdefault(find(index), []).append(index)
        for members in groups.values():
            if len(members) > 1:
                emergence.setdefault(tuple(members), threshold)
    return emergence


def _scenario_hypotheses(
    strength: np.ndarray,
    scenario: _visual.ScenarioWitness,
) -> tuple[ObjectHypothesis, ...]:
    masks = _component_masks(strength, scenario)
    emergence = _cluster_emergence(masks)
    ordered = sorted(emergence.items(), key=lambda item: (item[1], item[0]))
    height, width = strength.shape
    result: list[ObjectHypothesis] = []
    for hypothesis_index, (indices, gap) in enumerate(ordered):
        union = np.zeros_like(strength, dtype=bool)
        for index in indices:
            union |= masks[index]
        x0, y0, x1, y1 = _visual._bbox(union)
        crop = np.where(
            union[y0:y1, x0:x1], strength[y0:y1, x0:x1], 0
        ).astype(np.uint8, copy=False)
        components = tuple(scenario.components[index] for index in indices)
        result.append(
            ObjectHypothesis(
                scenario_id=scenario.scenario_id,
                hypothesis_id=f"hypothesis-{hypothesis_index:08d}",
                source_component_ids=tuple(x.component_id for x in components),
                source_component_mask_digests=tuple(x.mask_digest for x in components),
                union_mask_digest=_visual._mask_digest(union),
                union_area_pixels=int(np.count_nonzero(union)),
                bbox_pixels=(x0, y0, x1, y1),
                bbox_q16=_visual._q16_bbox((x0, y0, x1, y1), width, height),
                emergence_gap_pixels=gap,
                crop_width_pixels=x1 - x0,
                crop_height_pixels=y1 - y0,
                masked_crop_pixel_digest=_crop_pixel_digest(crop),
            )
        )
    return tuple(result)


@dataclass(frozen=True, order=True, slots=True)
class ObjectHypothesisAtlasSlot:
    global_slot_index: int
    sheet_index: int
    row_index: int
    column_index: int
    slot_id: str
    scenario_id: str
    hypothesis_id: str
    hypothesis_digest: str

    def __post_init__(self) -> None:
        global_index = _integer(self.global_slot_index, "global slot index")
        sheet = _integer(self.sheet_index, "slot sheet index")
        row = _integer(self.row_index, "slot row index")
        column = _integer(self.column_index, "slot column index")
        if sheet >= ATLAS_MAX_SHEETS or row >= ATLAS_ROWS or column >= ATLAS_COLUMNS:
            raise ObjectHypothesisError("atlas slot position exceeds fixed grid")
        local = row * ATLAS_COLUMNS + column
        if global_index != sheet * ATLAS_SLOT_CAPACITY + local:
            raise ObjectHypothesisError("atlas slot is not canonical row-major")
        if self.slot_id != f"slot-{global_index:08d}" or _SLOT_ID.fullmatch(
            self.slot_id
        ) is None:
            raise ObjectHypothesisError("atlas slot ID differs from position")
        if self.scenario_id not in VISUAL_WITNESS_SCENARIO_IDS:
            raise ObjectHypothesisError("atlas slot scenario is unknown")
        if not isinstance(self.hypothesis_id, str) or _HYPOTHESIS_ID.fullmatch(
            self.hypothesis_id
        ) is None:
            raise ObjectHypothesisError("atlas hypothesis ID is invalid")
        _digest(self.hypothesis_digest, "atlas hypothesis digest")

    def to_data(self) -> dict[str, object]:
        return {
            "global_slot_index": self.global_slot_index,
            "sheet_index": self.sheet_index,
            "row_index": self.row_index,
            "column_index": self.column_index,
            "slot_id": self.slot_id,
            "scenario_id": self.scenario_id,
            "hypothesis_id": self.hypothesis_id,
            "hypothesis_digest": self.hypothesis_digest,
        }

    @classmethod
    def from_data(cls, value: Mapping[str, Any]) -> "ObjectHypothesisAtlasSlot":
        raw = _exact_fields(
            value,
            {
                "global_slot_index",
                "sheet_index",
                "row_index",
                "column_index",
                "slot_id",
                "scenario_id",
                "hypothesis_id",
                "hypothesis_digest",
            },
            "atlas slot",
        )
        result = cls(**dict(raw))
        if result.to_data() != dict(raw):
            raise ObjectHypothesisError("atlas slot is not canonical")
        return result


@dataclass(frozen=True, slots=True)
class ObjectHypothesisAtlasSheet:
    sheet_index: int
    name: str
    width_pixels: int
    height_pixels: int
    slots: tuple[ObjectHypothesisAtlasSlot, ...]
    png_byte_count: int
    png_digest: str

    def __post_init__(self) -> None:
        index = _integer(self.sheet_index, "atlas sheet index")
        if index >= ATLAS_MAX_SHEETS:
            raise ObjectHypothesisError("atlas sheet exceeds fixed maximum")
        if self.name != f"sheet_{index:03d}.png" or _SHEET_NAME.fullmatch(
            self.name
        ) is None:
            raise ObjectHypothesisError("atlas sheet name is not opaque/canonical")
        if (
            self.width_pixels != ATLAS_WIDTH_PIXELS
            or self.height_pixels != ATLAS_HEIGHT_PIXELS
        ):
            raise ObjectHypothesisError("atlas sheet dimensions differ")
        if not isinstance(self.slots, tuple) or any(
            not isinstance(item, ObjectHypothesisAtlasSlot) for item in self.slots
        ):
            raise TypeError("atlas slots must be a typed tuple")
        if len(self.slots) > ATLAS_SLOT_CAPACITY:
            raise ObjectHypothesisError("atlas sheet slot capacity exceeded")
        expected_globals = tuple(
            index * ATLAS_SLOT_CAPACITY + offset
            for offset in range(len(self.slots))
        )
        if tuple(item.global_slot_index for item in self.slots) != expected_globals:
            raise ObjectHypothesisError("atlas sheet slots are not exhaustive prefix")
        if any(item.sheet_index != index for item in self.slots):
            raise ObjectHypothesisError("atlas slot belongs to another sheet")
        _integer(self.png_byte_count, "atlas PNG byte count", minimum=1)
        _digest(self.png_digest, "atlas PNG digest")

    def to_data(self) -> dict[str, object]:
        return {
            "sheet_index": self.sheet_index,
            "name": self.name,
            "width_pixels": self.width_pixels,
            "height_pixels": self.height_pixels,
            "slots": [item.to_data() for item in self.slots],
            "png_byte_count": self.png_byte_count,
            "png_digest": self.png_digest,
        }

    @classmethod
    def from_data(cls, value: Mapping[str, Any]) -> "ObjectHypothesisAtlasSheet":
        raw = _exact_fields(
            value,
            {
                "sheet_index",
                "name",
                "width_pixels",
                "height_pixels",
                "slots",
                "png_byte_count",
                "png_digest",
            },
            "atlas sheet",
        )
        slots = raw["slots"]
        if not isinstance(slots, list) or any(
            not isinstance(item, Mapping) for item in slots
        ):
            raise ObjectHypothesisError("atlas sheet slots must be object list")
        result = cls(
            sheet_index=raw["sheet_index"],
            name=raw["name"],
            width_pixels=raw["width_pixels"],
            height_pixels=raw["height_pixels"],
            slots=tuple(ObjectHypothesisAtlasSlot.from_data(item) for item in slots),
            png_byte_count=raw["png_byte_count"],
            png_digest=raw["png_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectHypothesisError("atlas sheet is not canonical")
        return result


@dataclass(frozen=True, slots=True)
class ObjectHypothesisPacket:
    panel_digest: str
    width_pixels: int
    height_pixels: int
    visual_witness_packet_digest: str
    visual_witness_extractor_artifact_digest: str
    source_digest: str
    extractor_artifact_digest: str
    scenarios: tuple[ObjectHypothesisScenario, ...]
    atlas_sheets: tuple[ObjectHypothesisAtlasSheet, ...]

    def __post_init__(self) -> None:
        _digest(self.panel_digest, "hypothesis panel digest")
        _integer(self.width_pixels, "panel width", minimum=2)
        _integer(self.height_pixels, "panel height", minimum=2)
        _digest(self.visual_witness_packet_digest, "visual witness packet digest")
        _digest(
            self.visual_witness_extractor_artifact_digest,
            "visual witness extractor artifact digest",
        )
        if self.visual_witness_extractor_artifact_digest != (
            _visual.visual_witness_extractor_digest()
        ):
            raise ObjectHypothesisError("visual witness extractor identity drifted")
        if self.source_digest != object_hypothesis_extractor_source_digest():
            raise ObjectHypothesisError("object hypothesis source identity drifted")
        if self.extractor_artifact_digest != (
            object_hypothesis_extractor_artifact_digest()
        ):
            raise ObjectHypothesisError("object hypothesis artifact identity drifted")
        if not isinstance(self.scenarios, tuple) or any(
            not isinstance(item, ObjectHypothesisScenario) for item in self.scenarios
        ):
            raise TypeError("packet scenarios must be a typed tuple")
        if tuple(item.scenario_id for item in self.scenarios) != (
            VISUAL_WITNESS_SCENARIO_IDS
        ):
            raise ObjectHypothesisError("packet must retain all three scenarios")
        hypotheses = tuple(
            item for scenario in self.scenarios for item in scenario.hypotheses
        )
        if len(hypotheses) > ATLAS_MAX_SHEETS * ATLAS_SLOT_CAPACITY:
            raise ObjectHypothesisError(
                "hypothesis atlas exceeds 32 sheets; omission is forbidden"
            )
        if not isinstance(self.atlas_sheets, tuple) or any(
            not isinstance(item, ObjectHypothesisAtlasSheet)
            for item in self.atlas_sheets
        ):
            raise TypeError("atlas sheets must be a typed tuple")
        expected_sheet_count = max(
            1, (len(hypotheses) + ATLAS_SLOT_CAPACITY - 1) // ATLAS_SLOT_CAPACITY
        )
        if len(self.atlas_sheets) != expected_sheet_count:
            raise ObjectHypothesisError("atlas sheet count omits or adds candidates")
        if tuple(item.sheet_index for item in self.atlas_sheets) != tuple(
            range(expected_sheet_count)
        ):
            raise ObjectHypothesisError("atlas sheets are not consecutive")
        slots = tuple(item for sheet in self.atlas_sheets for item in sheet.slots)
        if len(slots) != len(hypotheses):
            raise ObjectHypothesisError("atlas does not present every hypothesis")
        for index, (slot, hypothesis) in enumerate(zip(slots, hypotheses, strict=True)):
            if (
                slot.global_slot_index != index
                or slot.scenario_id != hypothesis.scenario_id
                or slot.hypothesis_id != hypothesis.hypothesis_id
                or slot.hypothesis_digest != hypothesis.digest()
            ):
                raise ObjectHypothesisError("atlas slot binding differs from catalog")
        if not hypotheses and self.atlas_sheets[0].slots:
            raise ObjectHypothesisError("canonical empty sheet must have no slots")

    def to_data(self) -> dict[str, object]:
        return {
            "schema": OBJECT_HYPOTHESIS_PACKET_SCHEMA,
            "algorithm_id": OBJECT_HYPOTHESIS_ALGORITHM_ID,
            "panel_digest": self.panel_digest,
            "width_pixels": self.width_pixels,
            "height_pixels": self.height_pixels,
            "visual_witness_packet_digest": self.visual_witness_packet_digest,
            "visual_witness_extractor_artifact_digest": (
                self.visual_witness_extractor_artifact_digest
            ),
            "source_digest": self.source_digest,
            "extractor_artifact_digest": self.extractor_artifact_digest,
            "scenarios": [item.to_data() for item in self.scenarios],
            "atlas_sheets": [item.to_data() for item in self.atlas_sheets],
            "runtime_authority": _authority_data(),
            "candidate_policy": _candidate_policy_data(),
        }

    @classmethod
    def from_data(cls, value: Mapping[str, Any]) -> "ObjectHypothesisPacket":
        raw = _exact_fields(
            value,
            {
                "schema",
                "algorithm_id",
                "panel_digest",
                "width_pixels",
                "height_pixels",
                "visual_witness_packet_digest",
                "visual_witness_extractor_artifact_digest",
                "source_digest",
                "extractor_artifact_digest",
                "scenarios",
                "atlas_sheets",
                "runtime_authority",
                "candidate_policy",
            },
            "object hypothesis packet",
        )
        if (
            raw["schema"] != OBJECT_HYPOTHESIS_PACKET_SCHEMA
            or raw["algorithm_id"] != OBJECT_HYPOTHESIS_ALGORITHM_ID
            or raw["runtime_authority"] != _authority_data()
            or raw["candidate_policy"] != _candidate_policy_data()
        ):
            raise ObjectHypothesisError("unsupported hypothesis packet policy")
        scenarios = raw["scenarios"]
        sheets = raw["atlas_sheets"]
        if not isinstance(scenarios, list) or any(
            not isinstance(item, Mapping) for item in scenarios
        ):
            raise ObjectHypothesisError("packet scenarios must be object list")
        if not isinstance(sheets, list) or any(
            not isinstance(item, Mapping) for item in sheets
        ):
            raise ObjectHypothesisError("packet atlas sheets must be object list")
        result = cls(
            panel_digest=raw["panel_digest"],
            width_pixels=raw["width_pixels"],
            height_pixels=raw["height_pixels"],
            visual_witness_packet_digest=raw["visual_witness_packet_digest"],
            visual_witness_extractor_artifact_digest=raw[
                "visual_witness_extractor_artifact_digest"
            ],
            source_digest=raw["source_digest"],
            extractor_artifact_digest=raw["extractor_artifact_digest"],
            scenarios=tuple(ObjectHypothesisScenario.from_data(item) for item in scenarios),
            atlas_sheets=tuple(ObjectHypothesisAtlasSheet.from_data(item) for item in sheets),
        )
        if result.to_data() != dict(raw):
            raise ObjectHypothesisError("object hypothesis packet is not canonical")
        return result

    def digest(self) -> str:
        return canonical_digest(self.to_data())


def _require_atlas_capacity(hypothesis_count: int) -> None:
    count = _integer(hypothesis_count, "hypothesis count")
    if count > ATLAS_MAX_SHEETS * ATLAS_SLOT_CAPACITY:
        raise ObjectHypothesisError(
            "hypothesis atlas exceeds 32 sheets; omission is forbidden"
        )


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
    """A deterministic zlib stream containing only stored DEFLATE blocks."""

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
    if len(kind) != 4:
        raise AssertionError("PNG chunk type must have four bytes")
    crc = binascii.crc32(kind + payload) & 0xFFFFFFFF
    return (
        len(payload).to_bytes(4, "big")
        + kind
        + payload
        + crc.to_bytes(4, "big")
    )


def _encode_grayscale_png(pixels: np.ndarray) -> bytes:
    """Encode exact grayscale8 PNG bytes without Pillow or zlib heuristics."""

    if pixels.dtype != np.uint8 or pixels.ndim != 2:
        raise TypeError("atlas pixels must be a two-dimensional uint8 array")
    height, width = pixels.shape
    if width < 1 or height < 1:
        raise ObjectHypothesisError("atlas raster must be nonempty")
    ihdr = (
        width.to_bytes(4, "big")
        + height.to_bytes(4, "big")
        + bytes((8, 0, 0, 0, 0))
    )
    scanlines = b"".join(
        b"\x00" + np.ascontiguousarray(row).tobytes() for row in pixels
    )
    return (
        _PNG_SIGNATURE
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"IDAT", _stored_zlib(scanlines))
        + _png_chunk(b"IEND", b"")
    )


def _fit_strength_crop(crop: np.ndarray, width: int, height: int) -> np.ndarray:
    """Aspect-fit with deterministic max pooling so thin ink is not dropped."""

    source_height, source_width = crop.shape
    if source_width * height >= source_height * width:
        target_width = width
        target_height = max(1, (source_height * width) // source_width)
    else:
        target_height = height
        target_width = max(1, (source_width * height) // source_height)
    fitted = np.zeros((target_height, target_width), dtype=np.uint8)
    for target_y in range(target_height):
        source_y0 = (target_y * source_height) // target_height
        source_y1 = max(
            source_y0 + 1,
            ((target_y + 1) * source_height + target_height - 1)
            // target_height,
        )
        source_y1 = min(source_y1, source_height)
        for target_x in range(target_width):
            source_x0 = (target_x * source_width) // target_width
            source_x1 = max(
                source_x0 + 1,
                ((target_x + 1) * source_width + target_width - 1)
                // target_width,
            )
            source_x1 = min(source_x1, source_width)
            fitted[target_y, target_x] = np.max(
                crop[source_y0:source_y1, source_x0:source_x1]
            )
    return fitted


def _render_atlas_sheet(crops: tuple[np.ndarray, ...]) -> bytes:
    if len(crops) > ATLAS_SLOT_CAPACITY:
        raise ObjectHypothesisError("renderer received too many atlas crops")
    pixels = np.full(
        (ATLAS_HEIGHT_PIXELS, ATLAS_WIDTH_PIXELS), 255, dtype=np.uint8
    )
    for local_index in range(ATLAS_SLOT_CAPACITY):
        row, column = divmod(local_index, ATLAS_COLUMNS)
        y0 = row * ATLAS_SLOT_SIZE_PIXELS
        x0 = column * ATLAS_SLOT_SIZE_PIXELS
        border_value = 160 if local_index < len(crops) else 224
        border = _SLOT_BORDER_PIXELS
        pixels[y0 : y0 + border, x0 : x0 + ATLAS_SLOT_SIZE_PIXELS] = border_value
        pixels[
            y0 + ATLAS_SLOT_SIZE_PIXELS - border : y0 + ATLAS_SLOT_SIZE_PIXELS,
            x0 : x0 + ATLAS_SLOT_SIZE_PIXELS,
        ] = border_value
        pixels[y0 : y0 + ATLAS_SLOT_SIZE_PIXELS, x0 : x0 + border] = border_value
        pixels[
            y0 : y0 + ATLAS_SLOT_SIZE_PIXELS,
            x0 + ATLAS_SLOT_SIZE_PIXELS - border : x0 + ATLAS_SLOT_SIZE_PIXELS,
        ] = border_value
        if local_index >= len(crops):
            continue
        viewport = ATLAS_SLOT_SIZE_PIXELS - 2 * _SLOT_MARGIN_PIXELS
        fitted = _fit_strength_crop(crops[local_index], viewport, viewport)
        image = 255 - fitted
        paste_y = y0 + (ATLAS_SLOT_SIZE_PIXELS - fitted.shape[0]) // 2
        paste_x = x0 + (ATLAS_SLOT_SIZE_PIXELS - fitted.shape[1]) // 2
        pixels[
            paste_y : paste_y + fitted.shape[0],
            paste_x : paste_x + fitted.shape[1],
        ] = image
    return _encode_grayscale_png(pixels)


def _crops_for_catalog(
    strength: np.ndarray,
    visual_packet: _visual.VisualWitnessPacket,
    scenarios: tuple[ObjectHypothesisScenario, ...],
) -> tuple[np.ndarray, ...]:
    result: list[np.ndarray] = []
    for visual_scenario, catalog_scenario in zip(
        visual_packet.scenarios, scenarios, strict=True
    ):
        masks = _component_masks(strength, visual_scenario)
        by_id = {
            component.component_id: mask
            for component, mask in zip(visual_scenario.components, masks, strict=True)
        }
        for hypothesis in catalog_scenario.hypotheses:
            union = np.zeros_like(strength, dtype=bool)
            for component_id in hypothesis.source_component_ids:
                union |= by_id[component_id]
            x0, y0, x1, y1 = _visual._bbox(union)
            crop = np.where(
                union[y0:y1, x0:x1], strength[y0:y1, x0:x1], 0
            ).astype(np.uint8, copy=False)
            if (
                crop.shape
                != (hypothesis.crop_height_pixels, hypothesis.crop_width_pixels)
                or _crop_pixel_digest(crop)
                != hypothesis.masked_crop_pixel_digest
                or _visual._mask_digest(union) != hypothesis.union_mask_digest
            ):
                raise ObjectHypothesisError("atlas crop replay differs from catalog")
            result.append(np.ascontiguousarray(crop))
    return tuple(result)


def _build_packet_and_atlas(
    png_bytes: bytes,
) -> tuple[ObjectHypothesisPacket, tuple[tuple[str, bytes], ...]]:
    if not isinstance(png_bytes, bytes):
        raise TypeError("object hypothesis input must be exact PNG bytes")
    visual_packet = _visual.extract_visual_witnesses(png_bytes)
    _visual.verify_visual_witness_packet(
        visual_packet, expected_png_bytes=png_bytes
    )
    # Every base component necessarily contributes a singleton hypothesis.
    # Reject impossible presentations before allocating the linkage matrices.
    _require_atlas_capacity(
        sum(len(scenario.components) for scenario in visual_packet.scenarios)
    )
    strength = _visual._decode_png(png_bytes)
    scenarios = tuple(
        ObjectHypothesisScenario(
            scenario_id=scenario.scenario_id,
            hypotheses=_scenario_hypotheses(strength, scenario),
        )
        for scenario in visual_packet.scenarios
    )
    hypotheses = tuple(
        item for scenario in scenarios for item in scenario.hypotheses
    )
    _require_atlas_capacity(len(hypotheses))
    crops = _crops_for_catalog(strength, visual_packet, scenarios)
    sheet_count = max(
        1, (len(hypotheses) + ATLAS_SLOT_CAPACITY - 1) // ATLAS_SLOT_CAPACITY
    )
    rendered: list[tuple[str, bytes]] = []
    sheets: list[ObjectHypothesisAtlasSheet] = []
    for sheet_index in range(sheet_count):
        first = sheet_index * ATLAS_SLOT_CAPACITY
        sheet_hypotheses = hypotheses[first : first + ATLAS_SLOT_CAPACITY]
        sheet_crops = crops[first : first + ATLAS_SLOT_CAPACITY]
        slots = tuple(
            ObjectHypothesisAtlasSlot(
                global_slot_index=first + local_index,
                sheet_index=sheet_index,
                row_index=local_index // ATLAS_COLUMNS,
                column_index=local_index % ATLAS_COLUMNS,
                slot_id=f"slot-{first + local_index:08d}",
                scenario_id=hypothesis.scenario_id,
                hypothesis_id=hypothesis.hypothesis_id,
                hypothesis_digest=hypothesis.digest(),
            )
            for local_index, hypothesis in enumerate(sheet_hypotheses)
        )
        png = _render_atlas_sheet(sheet_crops)
        name = f"sheet_{sheet_index:03d}.png"
        rendered.append((name, png))
        sheets.append(
            ObjectHypothesisAtlasSheet(
                sheet_index=sheet_index,
                name=name,
                width_pixels=ATLAS_WIDTH_PIXELS,
                height_pixels=ATLAS_HEIGHT_PIXELS,
                slots=slots,
                png_byte_count=len(png),
                png_digest=hashlib.sha256(png).hexdigest(),
            )
        )
    packet = ObjectHypothesisPacket(
        panel_digest=hashlib.sha256(png_bytes).hexdigest(),
        width_pixels=visual_packet.width_pixels,
        height_pixels=visual_packet.height_pixels,
        visual_witness_packet_digest=visual_packet.digest(),
        visual_witness_extractor_artifact_digest=(
            _visual.visual_witness_extractor_digest()
        ),
        source_digest=object_hypothesis_extractor_source_digest(),
        extractor_artifact_digest=object_hypothesis_extractor_artifact_digest(),
        scenarios=scenarios,
        atlas_sheets=tuple(sheets),
    )
    return packet, tuple(rendered)


def extract_object_hypothesis_packet(png_bytes: bytes) -> ObjectHypothesisPacket:
    """Freeze the exhaustive candidate catalog from exact pixels only."""

    return _build_packet_and_atlas(png_bytes)[0]


def extract_object_hypotheses(png_bytes: bytes) -> ObjectHypothesisPacket:
    """Compatibility spelling for :func:`extract_object_hypothesis_packet`."""

    return extract_object_hypothesis_packet(png_bytes)


def render_object_hypothesis_atlas(
    packet: ObjectHypothesisPacket, expected_png_bytes: bytes
) -> tuple[tuple[str, bytes], ...]:
    """Cold-render all opaque sheets after exact packet replay."""

    if not isinstance(packet, ObjectHypothesisPacket):
        raise TypeError("packet must be ObjectHypothesisPacket")
    rebuilt, rendered = _build_packet_and_atlas(expected_png_bytes)
    if rebuilt != packet:
        raise ObjectHypothesisError("hypothesis packet differs from PNG replay")
    return rendered


def verify_object_hypothesis_packet(
    packet: ObjectHypothesisPacket,
    expected_png_bytes: bytes | None = None,
    *,
    expected_atlas_png_by_name: Mapping[str, bytes] | None = None,
) -> ObjectHypothesisPacket:
    """Verify canonical data and optionally replay catalog and atlas bytes."""

    if not isinstance(packet, ObjectHypothesisPacket):
        raise TypeError("packet must be ObjectHypothesisPacket")
    if ObjectHypothesisPacket.from_data(packet.to_data()) != packet:
        raise ObjectHypothesisError("hypothesis packet is not canonical")
    if expected_png_bytes is None:
        if expected_atlas_png_by_name is not None:
            raise ObjectHypothesisError("atlas bytes require source PNG replay")
        return packet
    rebuilt, rendered = _build_packet_and_atlas(expected_png_bytes)
    if rebuilt != packet:
        raise ObjectHypothesisError("hypothesis packet differs from PNG replay")
    if expected_atlas_png_by_name is not None:
        if (
            not isinstance(expected_atlas_png_by_name, Mapping)
            or any(not isinstance(key, str) for key in expected_atlas_png_by_name)
            or any(not isinstance(value, bytes) for value in expected_atlas_png_by_name.values())
        ):
            raise TypeError("expected atlas must map names to exact bytes")
        rendered_map = dict(rendered)
        if dict(expected_atlas_png_by_name) != rendered_map:
            raise ObjectHypothesisError("atlas bytes differ from canonical replay")
    return packet


__all__ = (
    "ATLAS_COLUMNS",
    "ATLAS_HEIGHT_PIXELS",
    "ATLAS_MAX_SHEETS",
    "ATLAS_ROWS",
    "ATLAS_SLOT_CAPACITY",
    "ATLAS_SLOT_SIZE_PIXELS",
    "ATLAS_WIDTH_PIXELS",
    "OBJECT_HYPOTHESIS_ALGORITHM_ID",
    "OBJECT_HYPOTHESIS_PACKET_SCHEMA",
    "ObjectHypothesis",
    "ObjectHypothesisAtlasSheet",
    "ObjectHypothesisAtlasSlot",
    "ObjectHypothesisError",
    "ObjectHypothesisPacket",
    "ObjectHypothesisScenario",
    "extract_object_hypotheses",
    "extract_object_hypothesis_packet",
    "object_hypothesis_extractor_artifact_digest",
    "object_hypothesis_extractor_source_digest",
    "render_object_hypothesis_atlas",
    "verify_object_hypothesis_packet",
)
