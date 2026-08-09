"""Exact per-proposal anchor salience over a verified object-scene inventory.

The catalog is a bridge, not a second object detector.  It cold-replays the
existing hypothesis crops and component masks from the exact panel PNG,
tight-crops each canonical component union, and runs the Python
anchor-salience extractor on every inventory object in inventory order.  The
masked-strength crop is verified separately and is never used as the mask.

Full salience artifacts are retained for audit.  The separate decision
manifest contains only the selected salience graph (or a typed non-clean
status); raw and audit graphs are never decision inputs.
"""

from __future__ import annotations

from bongard.runtime_source_snapshot import capture_loaded_source, verify_loaded_source


_LOADED_SOURCE_SHA256 = capture_loaded_source(__name__, __file__)

from dataclasses import dataclass
import hashlib
import re
from typing import Any, Mapping

import numpy as np

from bongard.canonical import canonical_digest, canonical_json
from bongard.object_scene_anchor_graph import ObjectSceneAnchorGraph
from bongard.object_scene_anchor_salience import (
    AnchorSalienceLimits,
    ObjectSceneAnchorSalience,
    extract_object_scene_anchor_salience,
    object_scene_anchor_salience_extractor_digest,
    verify_object_scene_anchor_salience,
)
from bongard import object_scene_visual_frontend as _frontend
from bongard import prototype_object_hypotheses as _hypotheses
from bongard import visual_witnesses as _visual
from bongard.object_scene_visual_frontend import (
    OBJECT_SCENE_CANONICAL_SCENARIO_ID,
    ObjectSceneProposalInventory,
    object_scene_inventory_protocol_digest,
    object_scene_visual_frontend_source_digest,
    verify_object_scene_proposal_inventory,
)
from bongard.prototype_object_hypotheses import (
    extract_object_hypothesis_packet,
    object_hypothesis_extractor_artifact_digest,
    verify_object_hypothesis_packet,
)
from bongard.prototype_object_lineages import (
    object_lineage_artifact_digest,
    verify_object_lineage_packet,
)
from bongard.python_predicate_authority import PYTHON_PREDICATE_AUTHORITY_ID


OBJECT_SCENE_ANCHOR_CATALOG_SCHEMA = "gkm.object-scene-anchor-catalog.v1"
OBJECT_SCENE_ANCHOR_CATALOG_ENTRY_SCHEMA = (
    "gkm.object-scene-anchor-catalog-entry.v1"
)
OBJECT_SCENE_ANCHOR_DECISION_MANIFEST_SCHEMA = (
    "gkm.object-scene-anchor-decision-manifest.v1"
)
OBJECT_SCENE_PROPOSAL_BOOL_MASK_SCHEMA = (
    "gkm.object-scene-anchor-binary-mask.v1"
)
OBJECT_SCENE_ANCHOR_CATALOG_ALGORITHM_ID = (
    "bongard.object-scene-anchor-catalog/exact-inventory-union-mask-v1"
)
OBJECT_SCENE_ANCHOR_CROP_REPLAY_HELPER_ID = (
    "bongard.object_scene_visual_frontend._hypothesis_crop_map"
)
OBJECT_SCENE_ANCHOR_MASK_DERIVATION = (
    "exact-canonical-scenario-component-union-tight-crop"
)

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_OBJECT_ID = re.compile(r"object_[0-9]{4}\Z")
_LINEAGE_ID = re.compile(r"lineage-[0-9]{8}\Z")
_HYPOTHESIS_ID = re.compile(r"hypothesis-[0-9]{8}\Z")


class ObjectSceneAnchorCatalogError(ValueError):
    """The exact proposal-to-salience bridge is invalid or stale."""


def _exact_fields(
    value: object, expected: set[str] | frozenset[str], label: str
) -> Mapping[str, Any]:
    if (
        not isinstance(value, Mapping)
        or any(not isinstance(key, str) for key in value)
        or set(value) != set(expected)
    ):
        raise ObjectSceneAnchorCatalogError(f"{label} fields differ from schema")
    return value


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ObjectSceneAnchorCatalogError(
            f"{label} must be a lowercase SHA-256"
        )
    return value


def _integer(value: object, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ObjectSceneAnchorCatalogError(
            f"{label} must be an integer >= {minimum}"
        )
    return value


def _assert_python_only_keys(value: object) -> None:
    """Reject direct backend-specific fields anywhere in the frozen catalog."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str) or "lean" in key.lower():
                raise ObjectSceneAnchorCatalogError(
                    "catalog contains a direct non-Python backend key"
                )
            _assert_python_only_keys(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _assert_python_only_keys(item)


def _authority_data() -> dict[str, object]:
    return {
        "predicate_authority_id": PYTHON_PREDICATE_AUTHORITY_ID,
        "python_is_canonical_authority": True,
        "exact_png_and_inventory_replay_required": True,
        "all_inventory_objects_exhausted_in_order": True,
        "raw_graph_decision_bearing": False,
        "audit_graph_decision_bearing": False,
        "selected_graph_is_only_graph_decision_input": True,
    }


def object_scene_anchor_catalog_source_digest() -> str:
    """Return the loaded source identity, rejecting post-import mutation."""

    return verify_loaded_source(__name__, expected_source_sha256=_LOADED_SOURCE_SHA256)


def object_scene_anchor_catalog_extractor_digest() -> str:
    """Bind the bridge to every extractor used by its exact crop replay."""

    return canonical_digest(
        {
            "algorithm_id": OBJECT_SCENE_ANCHOR_CATALOG_ALGORITHM_ID,
            "source_digest": object_scene_anchor_catalog_source_digest(),
            "frontend_source_digest": object_scene_visual_frontend_source_digest(),
            "inventory_protocol_digest": object_scene_inventory_protocol_digest(),
            "hypothesis_extractor_artifact_digest": (
                object_hypothesis_extractor_artifact_digest()
            ),
            "lineage_extractor_artifact_digest": object_lineage_artifact_digest(),
            "salience_extractor_artifact_digest": (
                object_scene_anchor_salience_extractor_digest()
            ),
            "crop_replay_helper_id": OBJECT_SCENE_ANCHOR_CROP_REPLAY_HELPER_ID,
            "mask_derivation": OBJECT_SCENE_ANCHOR_MASK_DERIVATION,
            "authority": _authority_data(),
        }
    )


def object_scene_proposal_bool_mask_digest(mask: np.ndarray) -> str:
    """Return the salience source-mask commitment for an exact Boolean mask."""

    if type(mask) is not np.ndarray or mask.ndim != 2 or mask.dtype != np.bool_:
        raise TypeError("proposal mask must be an exact two-dimensional bool array")
    if min(mask.shape) < 1:
        raise ObjectSceneAnchorCatalogError("proposal mask must have positive extent")
    exact = np.ascontiguousarray(mask, dtype=np.bool_)
    height, width = exact.shape
    header = canonical_json(
        {
            "schema": OBJECT_SCENE_PROPOSAL_BOOL_MASK_SCHEMA,
            "kind": "source-mask",
            "height_pixels": height,
            "width_pixels": width,
            "packing": "numpy.packbits-axis-none-bitorder-big",
        }
    )
    packed = np.packbits(exact.reshape(-1), bitorder="big").tobytes()
    return hashlib.sha256(header + b"\x00" + packed).hexdigest()


def _selected_anchor_ids(graph: ObjectSceneAnchorGraph) -> tuple[str, ...]:
    return tuple(
        sorted(
            [item.part_id for item in graph.parts]
            + [item.compact_id for item in graph.compact_components]
        )
    )


def _decision_content(value: "ObjectSceneAnchorDecisionManifest") -> dict[str, object]:
    return {
        "schema": OBJECT_SCENE_ANCHOR_DECISION_MANIFEST_SCHEMA,
        "object_id": value.object_id,
        "salience_state": value.salience_state,
        "salience_reason": value.salience_reason,
        "decision_kind": value.decision_kind,
        "selected_graph_artifact_digest": value.selected_graph_artifact_digest,
        "selected_graph": (
            None if value.selected_graph is None else value.selected_graph.to_data()
        ),
        "selected_anchor_ids": list(value.selected_anchor_ids),
        "selected_frame_ids": list(value.selected_frame_ids),
        "raw_graph_decision_bearing": False,
        "audit_graph_decision_bearing": False,
        "selected_graph_is_only_graph_decision_input": True,
    }


@dataclass(frozen=True, slots=True)
class ObjectSceneAnchorDecisionManifest:
    """The exact, minimal graph-bearing view exposed to predicate evaluation."""

    object_id: str
    salience_state: str
    salience_reason: str
    decision_kind: str
    selected_graph_artifact_digest: str | None
    selected_graph: ObjectSceneAnchorGraph | None
    selected_anchor_ids: tuple[str, ...]
    selected_frame_ids: tuple[str, ...]
    manifest_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.object_id, str) or _OBJECT_ID.fullmatch(self.object_id) is None:
            raise ObjectSceneAnchorCatalogError("decision object ID differs")
        if self.salience_state not in ("clean", "indeterminate", "error"):
            raise ObjectSceneAnchorCatalogError("decision salience state differs")
        if not isinstance(self.salience_reason, str) or not self.salience_reason:
            raise ObjectSceneAnchorCatalogError("decision salience reason differs")
        if self.decision_kind not in ("selected_graph", "typed_salience_gap"):
            raise ObjectSceneAnchorCatalogError("decision kind differs")
        if (
            type(self.selected_anchor_ids) is not tuple
            or type(self.selected_frame_ids) is not tuple
        ):
            raise TypeError("decision selected inventories have the wrong type")
        clean = self.salience_state == "clean"
        selected_scalars = (
            self.selected_graph_artifact_digest,
            self.selected_graph,
        )
        if clean:
            if (
                self.salience_reason != "complete"
                or self.decision_kind != "selected_graph"
                or any(item is None for item in selected_scalars)
                or type(self.selected_graph) is not ObjectSceneAnchorGraph
            ):
                raise ObjectSceneAnchorCatalogError(
                    "clean decision lacks its selected complete graph"
                )
            _digest(self.selected_graph_artifact_digest, "selected graph digest")
            assert self.selected_graph is not None
            if (
                self.selected_graph.object_id != self.object_id
                or self.selected_graph.status.state != "clean"
                or self.selected_graph.artifact_digest
                != self.selected_graph_artifact_digest
                or self.selected_anchor_ids != _selected_anchor_ids(self.selected_graph)
                or self.selected_frame_ids
                != tuple(item.frame_id for item in self.selected_graph.cyclic_frames)
            ):
                raise ObjectSceneAnchorCatalogError(
                    "selected decision graph inventory differs"
                )
        elif (
            self.salience_reason == "complete"
            or self.decision_kind != "typed_salience_gap"
            or any(item is not None for item in selected_scalars)
            or self.selected_anchor_ids
            or self.selected_frame_ids
        ):
            raise ObjectSceneAnchorCatalogError(
                "non-clean decision exposes a partial selected graph"
            )
        _digest(self.manifest_digest, "decision manifest digest")
        if self.manifest_digest != canonical_digest(_decision_content(self)):
            raise ObjectSceneAnchorCatalogError("decision manifest digest differs")

    def to_data(self) -> dict[str, object]:
        return {**_decision_content(self), "manifest_digest": self.manifest_digest}

    @classmethod
    def from_salience(
        cls, artifact: ObjectSceneAnchorSalience
    ) -> "ObjectSceneAnchorDecisionManifest":
        verify_object_scene_anchor_salience(artifact)
        graph = artifact.selected_graph
        if graph is None:
            values: dict[str, object] = {
                "object_id": artifact.object_id,
                "salience_state": artifact.status.state,
                "salience_reason": artifact.status.reason,
                "decision_kind": "typed_salience_gap",
                "selected_graph_artifact_digest": None,
                "selected_graph": None,
                "selected_anchor_ids": (),
                "selected_frame_ids": (),
            }
        else:
            values = {
                "object_id": artifact.object_id,
                "salience_state": artifact.status.state,
                "salience_reason": artifact.status.reason,
                "decision_kind": "selected_graph",
                "selected_graph_artifact_digest": graph.artifact_digest,
                "selected_graph": graph,
                "selected_anchor_ids": _selected_anchor_ids(graph),
                "selected_frame_ids": tuple(
                    item.frame_id for item in graph.cyclic_frames
                ),
            }
        provisional = object.__new__(cls)
        for name, item in values.items():
            object.__setattr__(provisional, name, item)
        return cls(
            **values,  # type: ignore[arg-type]
            manifest_digest=canonical_digest(_decision_content(provisional)),
        )

    @classmethod
    def from_data(cls, value: object) -> "ObjectSceneAnchorDecisionManifest":
        raw = _exact_fields(
            value,
            {
                "schema",
                "object_id",
                "salience_state",
                "salience_reason",
                "decision_kind",
                "selected_graph_artifact_digest",
                "selected_graph",
                "selected_anchor_ids",
                "selected_frame_ids",
                "raw_graph_decision_bearing",
                "audit_graph_decision_bearing",
                "selected_graph_is_only_graph_decision_input",
                "manifest_digest",
            },
            "anchor decision manifest",
        )
        if (
            raw["schema"] != OBJECT_SCENE_ANCHOR_DECISION_MANIFEST_SCHEMA
            or raw["raw_graph_decision_bearing"] is not False
            or raw["audit_graph_decision_bearing"] is not False
            or raw["selected_graph_is_only_graph_decision_input"] is not True
            or not isinstance(raw["selected_anchor_ids"], list)
            or not isinstance(raw["selected_frame_ids"], list)
        ):
            raise ObjectSceneAnchorCatalogError("decision manifest policy differs")
        graph = raw["selected_graph"]
        if graph is not None and not isinstance(graph, Mapping):
            raise TypeError("selected graph must be an object or null")
        result = cls(
            object_id=raw["object_id"],
            salience_state=raw["salience_state"],
            salience_reason=raw["salience_reason"],
            decision_kind=raw["decision_kind"],
            selected_graph_artifact_digest=raw["selected_graph_artifact_digest"],
            selected_graph=(
                None if graph is None else ObjectSceneAnchorGraph.from_data(graph)
            ),
            selected_anchor_ids=tuple(raw["selected_anchor_ids"]),
            selected_frame_ids=tuple(raw["selected_frame_ids"]),
            manifest_digest=raw["manifest_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectSceneAnchorCatalogError(
                "anchor decision manifest is not canonical"
            )
        return result


def _entry_content(value: "ObjectSceneAnchorCatalogEntry") -> dict[str, object]:
    return {
        "schema": OBJECT_SCENE_ANCHOR_CATALOG_ENTRY_SCHEMA,
        "object_id": value.object_id,
        "inventory_index": value.inventory_index,
        "crop_receipt_digest": value.crop_receipt_digest,
        "lineage_id": value.lineage_id,
        "lineage_digest": value.lineage_digest,
        "scenario_id": value.scenario_id,
        "hypothesis_id": value.hypothesis_id,
        "hypothesis_digest": value.hypothesis_digest,
        "masked_crop_pixel_digest": value.masked_crop_pixel_digest,
        "crop_width_pixels": value.crop_width_pixels,
        "crop_height_pixels": value.crop_height_pixels,
        "mask_derivation": OBJECT_SCENE_ANCHOR_MASK_DERIVATION,
        "bool_mask_digest": value.bool_mask_digest,
        "foreground_pixel_count": value.foreground_pixel_count,
        "salience_artifact_digest": value.salience_artifact_digest,
        "salience": value.salience.to_data(),
        "decision_manifest": value.decision_manifest.to_data(),
    }


@dataclass(frozen=True, slots=True)
class ObjectSceneAnchorCatalogEntry:
    object_id: str
    inventory_index: int
    crop_receipt_digest: str
    lineage_id: str
    lineage_digest: str
    scenario_id: str
    hypothesis_id: str
    hypothesis_digest: str
    masked_crop_pixel_digest: str
    crop_width_pixels: int
    crop_height_pixels: int
    bool_mask_digest: str
    foreground_pixel_count: int
    salience_artifact_digest: str
    salience: ObjectSceneAnchorSalience
    decision_manifest: ObjectSceneAnchorDecisionManifest
    entry_digest: str

    def __post_init__(self) -> None:
        _integer(self.inventory_index, "inventory index")
        if self.object_id != f"object_{self.inventory_index:04d}":
            raise ObjectSceneAnchorCatalogError("entry object order differs")
        if _LINEAGE_ID.fullmatch(self.lineage_id) is None:
            raise ObjectSceneAnchorCatalogError("entry lineage ID differs")
        if self.scenario_id != OBJECT_SCENE_CANONICAL_SCENARIO_ID:
            raise ObjectSceneAnchorCatalogError("entry scenario differs")
        if _HYPOTHESIS_ID.fullmatch(self.hypothesis_id) is None:
            raise ObjectSceneAnchorCatalogError("entry hypothesis ID differs")
        for label, item in (
            ("crop receipt digest", self.crop_receipt_digest),
            ("lineage digest", self.lineage_digest),
            ("hypothesis digest", self.hypothesis_digest),
            ("masked crop pixel digest", self.masked_crop_pixel_digest),
            ("Boolean mask digest", self.bool_mask_digest),
            ("salience artifact digest", self.salience_artifact_digest),
        ):
            _digest(item, label)
        _integer(self.crop_width_pixels, "crop width", minimum=1)
        _integer(self.crop_height_pixels, "crop height", minimum=1)
        _integer(self.foreground_pixel_count, "foreground pixel count", minimum=1)
        if type(self.salience) is not ObjectSceneAnchorSalience or type(
            self.decision_manifest
        ) is not ObjectSceneAnchorDecisionManifest:
            raise TypeError("entry artifacts have the wrong type")
        verify_object_scene_anchor_salience(
            self.salience, expected_object_id=self.object_id
        )
        if (
            self.salience.artifact_digest != self.salience_artifact_digest
            or self.salience.source_width_pixels != self.crop_width_pixels
            or self.salience.source_height_pixels != self.crop_height_pixels
            or self.salience.source_foreground_pixel_count
            != self.foreground_pixel_count
            or self.bool_mask_digest != self.salience.source_mask_digest
            or self.decision_manifest
            != ObjectSceneAnchorDecisionManifest.from_salience(self.salience)
        ):
            raise ObjectSceneAnchorCatalogError("entry salience binding differs")
        _digest(self.entry_digest, "catalog entry digest")
        if self.entry_digest != canonical_digest(_entry_content(self)):
            raise ObjectSceneAnchorCatalogError("catalog entry digest differs")

    def to_data(self) -> dict[str, object]:
        return {**_entry_content(self), "entry_digest": self.entry_digest}

    @classmethod
    def from_data(cls, value: object) -> "ObjectSceneAnchorCatalogEntry":
        raw = _exact_fields(
            value,
            {
                "schema",
                "object_id",
                "inventory_index",
                "crop_receipt_digest",
                "lineage_id",
                "lineage_digest",
                "scenario_id",
                "hypothesis_id",
                "hypothesis_digest",
                "masked_crop_pixel_digest",
                "crop_width_pixels",
                "crop_height_pixels",
                "mask_derivation",
                "bool_mask_digest",
                "foreground_pixel_count",
                "salience_artifact_digest",
                "salience",
                "decision_manifest",
                "entry_digest",
            },
            "anchor catalog entry",
        )
        if (
            raw["schema"] != OBJECT_SCENE_ANCHOR_CATALOG_ENTRY_SCHEMA
            or raw["mask_derivation"] != OBJECT_SCENE_ANCHOR_MASK_DERIVATION
            or not isinstance(raw["salience"], Mapping)
            or not isinstance(raw["decision_manifest"], Mapping)
        ):
            raise ObjectSceneAnchorCatalogError("anchor catalog entry policy differs")
        result = cls(
            object_id=raw["object_id"],
            inventory_index=raw["inventory_index"],
            crop_receipt_digest=raw["crop_receipt_digest"],
            lineage_id=raw["lineage_id"],
            lineage_digest=raw["lineage_digest"],
            scenario_id=raw["scenario_id"],
            hypothesis_id=raw["hypothesis_id"],
            hypothesis_digest=raw["hypothesis_digest"],
            masked_crop_pixel_digest=raw["masked_crop_pixel_digest"],
            crop_width_pixels=raw["crop_width_pixels"],
            crop_height_pixels=raw["crop_height_pixels"],
            bool_mask_digest=raw["bool_mask_digest"],
            foreground_pixel_count=raw["foreground_pixel_count"],
            salience_artifact_digest=raw["salience_artifact_digest"],
            salience=ObjectSceneAnchorSalience.from_data(raw["salience"]),
            decision_manifest=ObjectSceneAnchorDecisionManifest.from_data(
                raw["decision_manifest"]
            ),
            entry_digest=raw["entry_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectSceneAnchorCatalogError("anchor catalog entry is not canonical")
        return result


def _catalog_content(value: "ObjectSceneAnchorCatalog") -> dict[str, object]:
    return {
        "schema": OBJECT_SCENE_ANCHOR_CATALOG_SCHEMA,
        "algorithm_id": OBJECT_SCENE_ANCHOR_CATALOG_ALGORITHM_ID,
        "panel_digest": value.panel_digest,
        "panel_png_byte_count": value.panel_png_byte_count,
        "width_pixels": value.width_pixels,
        "height_pixels": value.height_pixels,
        "inventory_digest": value.inventory_digest,
        "inventory_protocol_digest": value.inventory_protocol_digest,
        "frontend_source_digest": value.frontend_source_digest,
        "visual_witness_packet_digest": value.visual_witness_packet_digest,
        "hypothesis_packet_digest": value.hypothesis_packet_digest,
        "hypothesis_extractor_artifact_digest": (
            value.hypothesis_extractor_artifact_digest
        ),
        "lineage_packet_digest": value.lineage_packet_digest,
        "lineage_extractor_artifact_digest": value.lineage_extractor_artifact_digest,
        "salience_extractor_artifact_digest": value.salience_extractor_artifact_digest,
        "extractor_artifact_digest": value.extractor_artifact_digest,
        "crop_replay_helper_id": OBJECT_SCENE_ANCHOR_CROP_REPLAY_HELPER_ID,
        "mask_derivation": OBJECT_SCENE_ANCHOR_MASK_DERIVATION,
        "salience_limits": value.salience_limits.to_data(),
        "proposal_count": value.proposal_count,
        "object_ids": list(value.object_ids),
        "objects": {item.object_id: item.to_data() for item in value.entries},
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class ObjectSceneAnchorCatalog:
    panel_digest: str
    panel_png_byte_count: int
    width_pixels: int
    height_pixels: int
    inventory_digest: str
    inventory_protocol_digest: str
    frontend_source_digest: str
    visual_witness_packet_digest: str
    hypothesis_packet_digest: str
    hypothesis_extractor_artifact_digest: str
    lineage_packet_digest: str
    lineage_extractor_artifact_digest: str
    salience_extractor_artifact_digest: str
    extractor_artifact_digest: str
    salience_limits: AnchorSalienceLimits
    proposal_count: int
    object_ids: tuple[str, ...]
    entries: tuple[ObjectSceneAnchorCatalogEntry, ...]
    catalog_digest: str

    def __post_init__(self) -> None:
        for label, item in (
            ("panel digest", self.panel_digest),
            ("inventory digest", self.inventory_digest),
            ("inventory protocol digest", self.inventory_protocol_digest),
            ("frontend source digest", self.frontend_source_digest),
            ("visual witness packet digest", self.visual_witness_packet_digest),
            ("hypothesis packet digest", self.hypothesis_packet_digest),
            ("hypothesis extractor digest", self.hypothesis_extractor_artifact_digest),
            ("lineage packet digest", self.lineage_packet_digest),
            ("lineage extractor digest", self.lineage_extractor_artifact_digest),
            ("salience extractor digest", self.salience_extractor_artifact_digest),
            ("catalog extractor digest", self.extractor_artifact_digest),
        ):
            _digest(item, label)
        _integer(self.panel_png_byte_count, "panel PNG byte count", minimum=1)
        _integer(self.width_pixels, "panel width", minimum=2)
        _integer(self.height_pixels, "panel height", minimum=2)
        _integer(self.proposal_count, "proposal count")
        if type(self.salience_limits) is not AnchorSalienceLimits:
            raise TypeError("salience_limits must be exact AnchorSalienceLimits")
        if type(self.object_ids) is not tuple or type(self.entries) is not tuple:
            raise TypeError("catalog objects must be exact tuples")
        expected_ids = tuple(
            f"object_{index:04d}" for index in range(self.proposal_count)
        )
        if (
            self.object_ids != expected_ids
            or len(self.entries) != self.proposal_count
            or tuple(item.object_id for item in self.entries) != self.object_ids
            or tuple(item.inventory_index for item in self.entries)
            != tuple(range(self.proposal_count))
            or any(item.salience.limits != self.salience_limits for item in self.entries)
        ):
            raise ObjectSceneAnchorCatalogError(
                "catalog does not exhaust inventory objects in order"
            )
        if (
            self.inventory_protocol_digest != object_scene_inventory_protocol_digest()
            or self.frontend_source_digest != object_scene_visual_frontend_source_digest()
            or self.hypothesis_extractor_artifact_digest
            != object_hypothesis_extractor_artifact_digest()
            or self.lineage_extractor_artifact_digest
            != object_lineage_artifact_digest()
            or self.salience_extractor_artifact_digest
            != object_scene_anchor_salience_extractor_digest()
            or self.extractor_artifact_digest
            != object_scene_anchor_catalog_extractor_digest()
        ):
            raise ObjectSceneAnchorCatalogError("catalog extractor binding is stale")
        unsigned = _catalog_content(self)
        _assert_python_only_keys(unsigned)
        _digest(self.catalog_digest, "catalog digest")
        if self.catalog_digest != canonical_digest(unsigned):
            raise ObjectSceneAnchorCatalogError("catalog digest differs")

    @property
    def by_object_id(self) -> dict[str, ObjectSceneAnchorCatalogEntry]:
        return {item.object_id: item for item in self.entries}

    def to_data(self) -> dict[str, object]:
        return {**_catalog_content(self), "catalog_digest": self.catalog_digest}

    @classmethod
    def from_data(cls, value: object) -> "ObjectSceneAnchorCatalog":
        raw = _exact_fields(
            value,
            {
                "schema",
                "algorithm_id",
                "panel_digest",
                "panel_png_byte_count",
                "width_pixels",
                "height_pixels",
                "inventory_digest",
                "inventory_protocol_digest",
                "frontend_source_digest",
                "visual_witness_packet_digest",
                "hypothesis_packet_digest",
                "hypothesis_extractor_artifact_digest",
                "lineage_packet_digest",
                "lineage_extractor_artifact_digest",
                "salience_extractor_artifact_digest",
                "extractor_artifact_digest",
                "crop_replay_helper_id",
                "mask_derivation",
                "salience_limits",
                "proposal_count",
                "object_ids",
                "objects",
                *_authority_data(),
                "catalog_digest",
            },
            "object scene anchor catalog",
        )
        if (
            raw["schema"] != OBJECT_SCENE_ANCHOR_CATALOG_SCHEMA
            or raw["algorithm_id"] != OBJECT_SCENE_ANCHOR_CATALOG_ALGORITHM_ID
            or raw["crop_replay_helper_id"]
            != OBJECT_SCENE_ANCHOR_CROP_REPLAY_HELPER_ID
            or raw["mask_derivation"] != OBJECT_SCENE_ANCHOR_MASK_DERIVATION
            or any(raw[key] != item for key, item in _authority_data().items())
            or not isinstance(raw["salience_limits"], Mapping)
            or not isinstance(raw["object_ids"], list)
            or not isinstance(raw["objects"], Mapping)
            or any(not isinstance(key, str) for key in raw["objects"])
        ):
            raise ObjectSceneAnchorCatalogError("anchor catalog policy differs")
        object_ids = tuple(raw["object_ids"])
        if set(raw["objects"]) != set(object_ids) or len(raw["objects"]) != len(
            object_ids
        ):
            raise ObjectSceneAnchorCatalogError(
                "catalog object map differs from ordered inventory"
            )
        entries = tuple(
            ObjectSceneAnchorCatalogEntry.from_data(raw["objects"][object_id])
            for object_id in object_ids
        )
        result = cls(
            panel_digest=raw["panel_digest"],
            panel_png_byte_count=raw["panel_png_byte_count"],
            width_pixels=raw["width_pixels"],
            height_pixels=raw["height_pixels"],
            inventory_digest=raw["inventory_digest"],
            inventory_protocol_digest=raw["inventory_protocol_digest"],
            frontend_source_digest=raw["frontend_source_digest"],
            visual_witness_packet_digest=raw["visual_witness_packet_digest"],
            hypothesis_packet_digest=raw["hypothesis_packet_digest"],
            hypothesis_extractor_artifact_digest=raw[
                "hypothesis_extractor_artifact_digest"
            ],
            lineage_packet_digest=raw["lineage_packet_digest"],
            lineage_extractor_artifact_digest=raw[
                "lineage_extractor_artifact_digest"
            ],
            salience_extractor_artifact_digest=raw[
                "salience_extractor_artifact_digest"
            ],
            extractor_artifact_digest=raw["extractor_artifact_digest"],
            salience_limits=AnchorSalienceLimits.from_data(raw["salience_limits"]),
            proposal_count=raw["proposal_count"],
            object_ids=object_ids,
            entries=entries,
            catalog_digest=raw["catalog_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectSceneAnchorCatalogError("anchor catalog is not canonical")
        return result


def _make_entry(
    *,
    inventory_index: int,
    receipt: Any,
    mask: np.ndarray,
    salience: ObjectSceneAnchorSalience,
) -> ObjectSceneAnchorCatalogEntry:
    height, width = mask.shape
    decision = ObjectSceneAnchorDecisionManifest.from_salience(salience)
    values = {
        "object_id": receipt.object_id,
        "inventory_index": inventory_index,
        "crop_receipt_digest": receipt.receipt_digest,
        "lineage_id": receipt.lineage_id,
        "lineage_digest": receipt.lineage_digest,
        "scenario_id": receipt.scenario_id,
        "hypothesis_id": receipt.hypothesis_id,
        "hypothesis_digest": receipt.hypothesis_digest,
        "masked_crop_pixel_digest": receipt.masked_crop_pixel_digest,
        "crop_width_pixels": width,
        "crop_height_pixels": height,
        "bool_mask_digest": object_scene_proposal_bool_mask_digest(mask),
        "foreground_pixel_count": int(np.count_nonzero(mask)),
        "salience_artifact_digest": salience.artifact_digest,
        "salience": salience,
        "decision_manifest": decision,
    }
    provisional = object.__new__(ObjectSceneAnchorCatalogEntry)
    for name, item in values.items():
        object.__setattr__(provisional, name, item)
    return ObjectSceneAnchorCatalogEntry(
        **values,
        entry_digest=canonical_digest(_entry_content(provisional)),
    )


def extract_object_scene_anchor_catalog(
    png_bytes: bytes,
    inventory: ObjectSceneProposalInventory,
    limits: AnchorSalienceLimits | None = None,
) -> ObjectSceneAnchorCatalog:
    """Cold-replay and freeze anchor salience for every inventory proposal."""

    if not isinstance(png_bytes, bytes):
        raise TypeError("catalog input must be exact PNG bytes")
    if type(inventory) is not ObjectSceneProposalInventory:
        raise TypeError("inventory must be exact ObjectSceneProposalInventory")
    active_limits = limits if limits is not None else AnchorSalienceLimits()
    if type(active_limits) is not AnchorSalienceLimits:
        raise TypeError("limits must be exact AnchorSalienceLimits")

    verified_inventory = verify_object_scene_proposal_inventory(
        inventory,
        png_bytes,
        expected_inventory_digest=inventory.inventory_digest,
    )
    hypotheses = extract_object_hypothesis_packet(png_bytes)
    verify_object_hypothesis_packet(hypotheses, png_bytes)
    lineages = verify_object_lineage_packet(
        verified_inventory.lineage_packet, png_bytes
    )
    if (
        hypotheses.digest() != verified_inventory.hypothesis_packet_digest
        or lineages.digest() != verified_inventory.lineage_packet_digest
        or hypotheses.panel_digest != verified_inventory.panel_digest
        or hypotheses.width_pixels != verified_inventory.width_pixels
        or hypotheses.height_pixels != verified_inventory.height_pixels
    ):
        raise ObjectSceneAnchorCatalogError(
            "inventory hypothesis or lineage replay binding differs"
        )

    hypothesis_by_key = {
        (item.scenario_id, item.hypothesis_id): item
        for scenario in hypotheses.scenarios
        for item in scenario.hypotheses
    }
    # The display-crop replay is bound by frontend_source_digest.  Exact union
    # masks below use the hypothesis extractors' own component-mask replay,
    # bound by hypothesis_extractor_artifact_digest.
    crop_by_key = _frontend._hypothesis_crop_map(png_bytes, hypotheses)
    visual = _visual.extract_visual_witnesses(png_bytes)
    if visual.digest() != hypotheses.visual_witness_packet_digest:
        raise ObjectSceneAnchorCatalogError("visual witness replay binding differs")
    strength = _visual._decode_png(png_bytes)
    visual_scenario = next(
        item
        for item in visual.scenarios
        if item.scenario_id == OBJECT_SCENE_CANONICAL_SCENARIO_ID
    )
    component_masks = _hypotheses._component_masks(strength, visual_scenario)
    component_mask_by_id = {
        component.component_id: mask
        for component, mask in zip(
            visual_scenario.components, component_masks, strict=True
        )
    }
    lineage_by_id = {item.lineage_id: item for item in lineages.lineages}
    eligible_lineages = tuple(
        item for item in lineages.lineages if item.eligible_for_aggregation
    )
    if tuple(item.lineage_id for item in eligible_lineages) != tuple(
        item.lineage_id for item in verified_inventory.objects
    ):
        raise ObjectSceneAnchorCatalogError(
            "inventory does not exhaust eligible lineages in order"
        )

    entries: list[ObjectSceneAnchorCatalogEntry] = []
    consumed_keys: list[tuple[str, str]] = []
    for index, receipt in enumerate(verified_inventory.objects):
        if receipt.object_id != f"object_{index:04d}":
            raise ObjectSceneAnchorCatalogError("inventory object order differs")
        lineage = lineage_by_id.get(receipt.lineage_id)
        key = (receipt.scenario_id, receipt.hypothesis_id)
        hypothesis = hypothesis_by_key.get(key)
        crop = crop_by_key.get(key)
        if lineage is None or hypothesis is None or crop is None:
            raise ObjectSceneAnchorCatalogError("proposal replay identity is missing")
        member = next(
            (
                item
                for item in lineage.members
                if item.scenario_id == OBJECT_SCENE_CANONICAL_SCENARIO_ID
            ),
            None,
        )
        if (
            not lineage.eligible_for_aggregation
            or canonical_digest(lineage.to_data()) != receipt.lineage_digest
            or member is None
            or member.hypothesis_id != receipt.hypothesis_id
            or member.hypothesis_digest != receipt.hypothesis_digest
            or hypothesis.digest() != receipt.hypothesis_digest
            or hypothesis.source_component_ids != receipt.source_component_ids
            or hypothesis.bbox_pixels != receipt.bbox_pixels
            or hypothesis.bbox_q16 != receipt.bbox_q16
            or hypothesis.union_area_pixels != receipt.union_area_pixels
            or hypothesis.emergence_gap_pixels != receipt.emergence_gap_pixels
            or hypothesis.masked_crop_pixel_digest
            != receipt.masked_crop_pixel_digest
            or crop.dtype != np.uint8
            or crop.ndim != 2
            or crop.shape
            != (hypothesis.crop_height_pixels, hypothesis.crop_width_pixels)
            or _hypotheses._crop_pixel_digest(crop)
            != receipt.masked_crop_pixel_digest
        ):
            raise ObjectSceneAnchorCatalogError(
                "proposal receipt differs from exact hypothesis/crop replay"
            )
        union = np.zeros_like(strength, dtype=np.bool_)
        try:
            for component_id in hypothesis.source_component_ids:
                union |= component_mask_by_id[component_id]
        except KeyError as exc:
            raise ObjectSceneAnchorCatalogError(
                "hypothesis names an unknown canonical component"
            ) from exc
        union_bbox = _visual._bbox(union)
        x0, y0, x1, y1 = union_bbox
        mask = np.ascontiguousarray(union[y0:y1, x0:x1], dtype=np.bool_)
        foreground_count = int(np.count_nonzero(mask))
        if (
            union_bbox != receipt.bbox_pixels
            or _visual._mask_digest(union) != hypothesis.union_mask_digest
            or foreground_count != receipt.union_area_pixels
            or mask.shape != crop.shape
        ):
            raise ObjectSceneAnchorCatalogError(
                "proposal Boolean union mask differs from receipt"
            )
        salience = extract_object_scene_anchor_salience(
            mask, receipt.object_id, active_limits
        )
        verify_object_scene_anchor_salience(
            salience,
            expected_mask=mask,
            expected_object_id=receipt.object_id,
        )
        entries.append(
            _make_entry(
                inventory_index=index,
                receipt=receipt,
                mask=mask,
                salience=salience,
            )
        )
        consumed_keys.append(key)

    expected_keys = tuple(
        (
            OBJECT_SCENE_CANONICAL_SCENARIO_ID,
            next(
                item.hypothesis_id
                for item in lineage.members
                if item.scenario_id == OBJECT_SCENE_CANONICAL_SCENARIO_ID
            ),
        )
        for lineage in eligible_lineages
    )
    if tuple(consumed_keys) != expected_keys or len(set(consumed_keys)) != len(
        consumed_keys
    ):
        raise ObjectSceneAnchorCatalogError(
            "catalog did not consume every canonical proposal exactly once"
        )

    frozen_entries = tuple(entries)
    object_ids = tuple(item.object_id for item in frozen_entries)
    values = {
        "panel_digest": verified_inventory.panel_digest,
        "panel_png_byte_count": len(png_bytes),
        "width_pixels": verified_inventory.width_pixels,
        "height_pixels": verified_inventory.height_pixels,
        "inventory_digest": verified_inventory.inventory_digest,
        "inventory_protocol_digest": verified_inventory.protocol_digest,
        "frontend_source_digest": verified_inventory.source_digest,
        "visual_witness_packet_digest": (
            verified_inventory.visual_witness_packet_digest
        ),
        "hypothesis_packet_digest": hypotheses.digest(),
        "hypothesis_extractor_artifact_digest": (
            hypotheses.extractor_artifact_digest
        ),
        "lineage_packet_digest": lineages.digest(),
        "lineage_extractor_artifact_digest": lineages.extractor_artifact_digest,
        "salience_extractor_artifact_digest": (
            object_scene_anchor_salience_extractor_digest()
        ),
        "extractor_artifact_digest": object_scene_anchor_catalog_extractor_digest(),
        "salience_limits": active_limits,
        "proposal_count": len(frozen_entries),
        "object_ids": object_ids,
        "entries": frozen_entries,
    }
    provisional = object.__new__(ObjectSceneAnchorCatalog)
    for name, item in values.items():
        object.__setattr__(provisional, name, item)
    return ObjectSceneAnchorCatalog(
        **values,
        catalog_digest=canonical_digest(_catalog_content(provisional)),
    )


def verify_object_scene_anchor_catalog(
    catalog: ObjectSceneAnchorCatalog,
    png_bytes: bytes,
    inventory: ObjectSceneProposalInventory,
    *,
    expected_catalog_digest: str | None = None,
) -> ObjectSceneAnchorCatalog:
    """Verify canonical data and cold-replay the exact PNG/inventory bridge."""

    if type(catalog) is not ObjectSceneAnchorCatalog:
        raise TypeError("catalog must be exact ObjectSceneAnchorCatalog")
    restored = ObjectSceneAnchorCatalog.from_data(catalog.to_data())
    if expected_catalog_digest is not None and restored.catalog_digest != _digest(
        expected_catalog_digest, "expected catalog digest"
    ):
        raise ObjectSceneAnchorCatalogError("catalog differs from commitment")
    replay = extract_object_scene_anchor_catalog(
        png_bytes, inventory, restored.salience_limits
    )
    if replay != restored:
        raise ObjectSceneAnchorCatalogError(
            "catalog differs from exact PNG/inventory replay"
        )
    return restored


# Intent-revealing spelling for callers that treat the result as a frozen input.
freeze_object_scene_anchor_catalog = extract_object_scene_anchor_catalog


__all__ = (
    "OBJECT_SCENE_ANCHOR_CATALOG_ALGORITHM_ID",
    "OBJECT_SCENE_ANCHOR_CATALOG_ENTRY_SCHEMA",
    "OBJECT_SCENE_ANCHOR_CATALOG_SCHEMA",
    "OBJECT_SCENE_ANCHOR_CROP_REPLAY_HELPER_ID",
    "OBJECT_SCENE_ANCHOR_DECISION_MANIFEST_SCHEMA",
    "OBJECT_SCENE_ANCHOR_MASK_DERIVATION",
    "OBJECT_SCENE_PROPOSAL_BOOL_MASK_SCHEMA",
    "ObjectSceneAnchorCatalog",
    "ObjectSceneAnchorCatalogEntry",
    "ObjectSceneAnchorCatalogError",
    "ObjectSceneAnchorDecisionManifest",
    "extract_object_scene_anchor_catalog",
    "freeze_object_scene_anchor_catalog",
    "object_scene_anchor_catalog_extractor_digest",
    "object_scene_anchor_catalog_source_digest",
    "object_scene_proposal_bool_mask_digest",
    "verify_object_scene_anchor_catalog",
)
