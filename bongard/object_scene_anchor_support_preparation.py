"""Exact, model-free preparation of the exposed anchor support corpus.

This module is the boundary between source panel pixels and any headless
proposer.  Geometry extraction sees only an exact PNG.  The source's two
support buckets are attached afterwards as opaque partition metadata; no
positive/negative role, predicate, query, or model output is represented here.

Persistent freezes contain canonical typed artifacts and byte commitments, but
never image bytes.  Runtime bundles carry the exact original and rendered PNGs
needed for transport and cold replay.
"""

from __future__ import annotations

from bongard.runtime_source_snapshot import capture_loaded_source, verify_loaded_source


_LOADED_SOURCE_SHA256 = capture_loaded_source(__name__, __file__)

from dataclasses import dataclass
import hashlib
import re
from typing import Any, Mapping

from bongard.canonical import canonical_digest
from bongard.object_bongard_panel_rubric_calibration import (
    ObjectBongardPanelRubricCalibrationPanel,
    ObjectBongardPanelRubricCalibrationSource,
)
from bongard.object_scene_anchor_catalog import (
    ObjectSceneAnchorCatalog,
    extract_object_scene_anchor_catalog,
)
from bongard.object_scene_anchor_panel_manifest import (
    ObjectSceneAnchorPanelDecisionManifest,
    build_object_scene_anchor_panel_decision_manifest,
)
from bongard.object_scene_anchor_support_sheet import (
    ObjectSceneAnchorSupportSheet,
    build_object_scene_anchor_support_sheet,
)
from bongard.object_scene_visual_frontend import (
    ObjectSceneProposalInventory,
    extract_object_scene_proposal_inventory,
)
from bongard.python_predicate_authority import PYTHON_PREDICATE_AUTHORITY_ID
from bongard.transport import MAX_PANEL_PNG_BYTES


OBJECT_SCENE_ANCHOR_SUPPORT_PANEL_FREEZE_SCHEMA = (
    "gkm.object-scene-anchor-support-panel-freeze.v1"
)
OBJECT_SCENE_ANCHOR_SUPPORT_CORPUS_FREEZE_SCHEMA = (
    "gkm.object-scene-anchor-support-corpus-freeze.v1"
)
OBJECT_SCENE_ANCHOR_SUPPORT_PREPARATION_ID = (
    "bongard.object-scene-anchor-support-preparation/exact-pixels-v1"
)
OBJECT_SCENE_ANCHOR_SUPPORT_PANEL_COUNT = 12
OBJECT_SCENE_ANCHOR_SUPPORT_BUCKET_SIZE = 6

_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_PANEL_ALIAS = re.compile(r"panel_[0-9]{3}\Z")
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


class ObjectSceneAnchorSupportPreparationError(ValueError):
    """A support input, freeze, runtime payload, or replay differs."""


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise ObjectSceneAnchorSupportPreparationError(
            f"{label} must be a lowercase SHA-256"
        )
    return value


def _integer(value: object, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ObjectSceneAnchorSupportPreparationError(
            f"{label} must be an integer at least {minimum}"
        )
    return value


def _fields(
    value: object, expected: set[str] | frozenset[str], label: str
) -> Mapping[str, Any]:
    if (
        not isinstance(value, Mapping)
        or any(not isinstance(key, str) for key in value)
        or set(value) != set(expected)
    ):
        raise ObjectSceneAnchorSupportPreparationError(f"{label} fields differ")
    return value


def _png(value: object, label: str) -> bytes:
    if not isinstance(value, bytes) or not value.startswith(_PNG_SIGNATURE):
        raise ObjectSceneAnchorSupportPreparationError(
            f"{label} must be exact PNG bytes"
        )
    return value


def _panel_alias(value: object) -> str:
    if not isinstance(value, str) or _PANEL_ALIAS.fullmatch(value) is None:
        raise ObjectSceneAnchorSupportPreparationError(
            "panel alias must be neutral panel_NNN"
        )
    return value


def _bucket(value: object) -> int:
    if type(value) is not int or value not in (0, 1):
        raise ObjectSceneAnchorSupportPreparationError(
            "support bucket index must be exactly 0 or 1"
        )
    return value


def _assert_persistent_payload(value: object) -> None:
    """Reject image bytes and any positive Lean authority declaration.

    The canonical proposal inventory still carries explicit historical
    ``lean_*`` booleans saying that Lean is absent, unnecessary, and removable.
    Those nested fields must survive an exact canonical embedding until that
    upstream schema is migrated; they may never grant Lean authority here.
    """

    if isinstance(value, bytes):
        raise ObjectSceneAnchorSupportPreparationError(
            "persistent support freeze contains raw bytes"
        )
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ObjectSceneAnchorSupportPreparationError(
                    "persistent support freeze has a non-string key"
                )
            lowered = key.casefold()
            if "lean" in lowered:
                safe_value = True if "removable" in lowered else False
                if item is not safe_value:
                    raise ObjectSceneAnchorSupportPreparationError(
                        "Lean cannot enter support preparation authority"
                    )
            _assert_persistent_payload(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _assert_persistent_payload(item)


def _authority_data() -> dict[str, object]:
    return {
        "predicate_authority_id": PYTHON_PREDICATE_AUTHORITY_ID,
        "python_is_canonical_authority": True,
        "model_calls_permitted": False,
        "predicate_content_permitted": False,
        "query_pixels_consumed": False,
        "raw_png_bytes_persisted": False,
        "support_bucket_semantics": "opaque-downstream-proposer-partition-only",
        "geometry_extraction_receives_bucket": False,
    }


def object_scene_anchor_support_preparation_source_digest() -> str:
    return verify_loaded_source(__name__, expected_source_sha256=_LOADED_SOURCE_SHA256)


@dataclass(frozen=True, slots=True)
class ObjectSceneAnchorSupportPanelInput:
    """The sole runtime input for preparing one neutral support panel."""

    panel_alias: str
    support_bucket_index: int
    source_digest: str
    source_panel_binding_digest: str
    source_ordinal: int
    task_id: str
    panel_id: str
    original_panel_png_digest: str
    exact_original_png_bytes: bytes

    def __post_init__(self) -> None:
        _panel_alias(self.panel_alias)
        _bucket(self.support_bucket_index)
        _digest(self.source_digest, "source digest")
        _digest(self.source_panel_binding_digest, "source panel binding digest")
        _integer(self.source_ordinal, "source ordinal")
        _digest(self.original_panel_png_digest, "original panel PNG digest")
        payload = _png(self.exact_original_png_bytes, "original panel")
        if (
            not isinstance(self.task_id, str)
            or not self.task_id
            or not isinstance(self.panel_id, str)
            or not self.panel_id
            or hashlib.sha256(payload).hexdigest()
            != self.original_panel_png_digest
        ):
            raise ObjectSceneAnchorSupportPreparationError(
                "support panel identity or exact bytes differ"
            )

    @classmethod
    def from_source_panel(
        cls,
        source: ObjectBongardPanelRubricCalibrationSource,
        panel: ObjectBongardPanelRubricCalibrationPanel,
        *,
        panel_alias: str,
    ) -> "ObjectSceneAnchorSupportPanelInput":
        if type(source) is not ObjectBongardPanelRubricCalibrationSource:
            raise TypeError("source must be exact panel calibration source")
        if type(panel) is not ObjectBongardPanelRubricCalibrationPanel:
            raise TypeError("panel must be exact panel calibration panel")
        source_panel = source.panel_by_id(panel.panel_id)
        if source_panel != panel:
            raise ObjectSceneAnchorSupportPreparationError(
                "panel is not the exact source member"
            )
        return cls(
            panel_alias=panel_alias,
            support_bucket_index=panel.group_index,
            source_digest=source.source_digest,
            source_panel_binding_digest=panel.panel_binding_digest,
            source_ordinal=panel.ordinal,
            task_id=panel.task_id,
            panel_id=panel.panel_id,
            original_panel_png_digest=panel.png_sha256,
            exact_original_png_bytes=panel.exact_png_bytes,
        )


def _panel_content(value: "ObjectSceneAnchorSupportPanelFreeze") -> dict[str, object]:
    return {
        "schema": OBJECT_SCENE_ANCHOR_SUPPORT_PANEL_FREEZE_SCHEMA,
        "preparation_id": OBJECT_SCENE_ANCHOR_SUPPORT_PREPARATION_ID,
        "preparation_source_digest": (
            object_scene_anchor_support_preparation_source_digest()
        ),
        "panel_alias": value.panel_alias,
        "support_bucket_index": value.support_bucket_index,
        "source_digest": value.source_digest,
        "source_panel_binding_digest": value.source_panel_binding_digest,
        "source_ordinal": value.source_ordinal,
        "task_id": value.task_id,
        "panel_id": value.panel_id,
        "original_panel_png_byte_count": value.original_panel_png_byte_count,
        "original_panel_png_digest": value.original_panel_png_digest,
        "inventory": value.inventory.to_data(),
        "catalog": value.catalog.to_data(),
        "panel_manifest": value.panel_manifest.to_data(),
        "support_sheet": value.support_sheet.to_data(),
        "support_sheet_png_byte_count": value.support_sheet_png_byte_count,
        "support_sheet_png_digest": value.support_sheet_png_digest,
        "proposal_count": value.proposal_count,
        "object_ids": list(value.object_ids),
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class ObjectSceneAnchorSupportPanelFreeze:
    """Persistent exact panel stack; deliberately contains no PNG bytes."""

    panel_alias: str
    support_bucket_index: int
    source_digest: str
    source_panel_binding_digest: str
    source_ordinal: int
    task_id: str
    panel_id: str
    original_panel_png_byte_count: int
    original_panel_png_digest: str
    inventory: ObjectSceneProposalInventory
    catalog: ObjectSceneAnchorCatalog
    panel_manifest: ObjectSceneAnchorPanelDecisionManifest
    support_sheet: ObjectSceneAnchorSupportSheet
    support_sheet_png_byte_count: int
    support_sheet_png_digest: str
    proposal_count: int
    object_ids: tuple[str, ...]
    freeze_digest: str

    def __post_init__(self) -> None:
        _panel_alias(self.panel_alias)
        _bucket(self.support_bucket_index)
        _digest(self.source_digest, "source digest")
        _digest(self.source_panel_binding_digest, "source panel binding digest")
        _integer(self.source_ordinal, "source ordinal")
        _digest(self.original_panel_png_digest, "original panel PNG digest")
        _digest(self.support_sheet_png_digest, "support-sheet PNG digest")
        _integer(
            self.original_panel_png_byte_count,
            "original panel PNG byte count",
            minimum=1,
        )
        _integer(
            self.support_sheet_png_byte_count,
            "support-sheet PNG byte count",
            minimum=1,
        )
        _integer(self.proposal_count, "proposal count")
        if (
            not isinstance(self.task_id, str)
            or not self.task_id
            or not isinstance(self.panel_id, str)
            or not self.panel_id
        ):
            raise ObjectSceneAnchorSupportPreparationError(
                "support panel source identity differs"
            )
        if type(self.inventory) is not ObjectSceneProposalInventory:
            raise TypeError("inventory must be exact ObjectSceneProposalInventory")
        if type(self.catalog) is not ObjectSceneAnchorCatalog:
            raise TypeError("catalog must be exact ObjectSceneAnchorCatalog")
        if type(self.panel_manifest) is not ObjectSceneAnchorPanelDecisionManifest:
            raise TypeError(
                "panel_manifest must be exact ObjectSceneAnchorPanelDecisionManifest"
            )
        if type(self.support_sheet) is not ObjectSceneAnchorSupportSheet:
            raise TypeError("support_sheet must be exact ObjectSceneAnchorSupportSheet")
        if type(self.object_ids) is not tuple:
            raise TypeError("object_ids must be an exact tuple")
        expected_panel_digest = self.original_panel_png_digest
        if (
            self.inventory.panel_digest != expected_panel_digest
            or self.catalog.panel_digest != expected_panel_digest
            or self.catalog.panel_png_byte_count
            != self.original_panel_png_byte_count
            or self.catalog.inventory_digest != self.inventory.inventory_digest
            or self.panel_manifest.panel_digest != expected_panel_digest
            or self.panel_manifest.inventory_digest
            != self.inventory.inventory_digest
            or self.panel_manifest.manifest_digest
            != self.support_sheet.panel_manifest_digest
            or self.support_sheet.panel_digest != expected_panel_digest
            or self.support_sheet.inventory_digest
            != self.inventory.inventory_digest
            or self.support_sheet.original_panel_png_byte_count
            != self.original_panel_png_byte_count
            or self.support_sheet.original_panel_png_digest
            != self.original_panel_png_digest
            or self.support_sheet.sheet_png_byte_count
            != self.support_sheet_png_byte_count
            or self.support_sheet.sheet_png_digest
            != self.support_sheet_png_digest
            or self.proposal_count != len(self.inventory.objects)
            or self.proposal_count != self.catalog.proposal_count
            or self.proposal_count != self.panel_manifest.proposal_count
            or self.proposal_count != self.support_sheet.proposal_count
            or self.object_ids
            != tuple(item.object_id for item in self.inventory.objects)
            or self.object_ids != self.catalog.object_ids
            or self.object_ids != self.panel_manifest.object_ids
            or self.object_ids != self.support_sheet.object_ids
            or self.support_sheet_png_byte_count > MAX_PANEL_PNG_BYTES
        ):
            raise ObjectSceneAnchorSupportPreparationError(
                "support panel nested artifact or byte commitment differs"
            )
        unsigned = _panel_content(self)
        _assert_persistent_payload(unsigned)
        _digest(self.freeze_digest, "support panel freeze digest")
        if self.freeze_digest != canonical_digest(unsigned):
            raise ObjectSceneAnchorSupportPreparationError(
                "support panel freeze digest differs"
            )

    def to_data(self) -> dict[str, object]:
        return {**_panel_content(self), "freeze_digest": self.freeze_digest}

    @classmethod
    def from_data(cls, value: object) -> "ObjectSceneAnchorSupportPanelFreeze":
        expected = {
            "schema",
            "preparation_id",
            "preparation_source_digest",
            "panel_alias",
            "support_bucket_index",
            "source_digest",
            "source_panel_binding_digest",
            "source_ordinal",
            "task_id",
            "panel_id",
            "original_panel_png_byte_count",
            "original_panel_png_digest",
            "inventory",
            "catalog",
            "panel_manifest",
            "support_sheet",
            "support_sheet_png_byte_count",
            "support_sheet_png_digest",
            "proposal_count",
            "object_ids",
            *_authority_data(),
            "freeze_digest",
        }
        raw = _fields(value, expected, "support panel freeze")
        if (
            raw["schema"] != OBJECT_SCENE_ANCHOR_SUPPORT_PANEL_FREEZE_SCHEMA
            or raw["preparation_id"] != OBJECT_SCENE_ANCHOR_SUPPORT_PREPARATION_ID
            or raw["preparation_source_digest"]
            != object_scene_anchor_support_preparation_source_digest()
            or any(raw[key] != item for key, item in _authority_data().items())
            or not isinstance(raw["object_ids"], list)
        ):
            raise ObjectSceneAnchorSupportPreparationError(
                "support panel freeze policy differs"
            )
        result = cls(
            panel_alias=raw["panel_alias"],
            support_bucket_index=raw["support_bucket_index"],
            source_digest=raw["source_digest"],
            source_panel_binding_digest=raw["source_panel_binding_digest"],
            source_ordinal=raw["source_ordinal"],
            task_id=raw["task_id"],
            panel_id=raw["panel_id"],
            original_panel_png_byte_count=raw["original_panel_png_byte_count"],
            original_panel_png_digest=raw["original_panel_png_digest"],
            inventory=ObjectSceneProposalInventory.from_data(raw["inventory"]),
            catalog=ObjectSceneAnchorCatalog.from_data(raw["catalog"]),
            panel_manifest=ObjectSceneAnchorPanelDecisionManifest.from_data(
                raw["panel_manifest"]
            ),
            support_sheet=ObjectSceneAnchorSupportSheet.from_data(
                raw["support_sheet"]
            ),
            support_sheet_png_byte_count=raw["support_sheet_png_byte_count"],
            support_sheet_png_digest=raw["support_sheet_png_digest"],
            proposal_count=raw["proposal_count"],
            object_ids=tuple(raw["object_ids"]),
            freeze_digest=raw["freeze_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectSceneAnchorSupportPreparationError(
                "support panel freeze is not canonical"
            )
        return result


@dataclass(frozen=True, slots=True)
class ObjectSceneAnchorSupportPanelRuntimeBundle:
    """A persistent panel freeze plus its two exact runtime PNG payloads."""

    freeze: ObjectSceneAnchorSupportPanelFreeze
    exact_original_png_bytes: bytes
    exact_support_sheet_png_bytes: bytes

    def __post_init__(self) -> None:
        if type(self.freeze) is not ObjectSceneAnchorSupportPanelFreeze:
            raise TypeError("freeze must be exact support panel freeze")
        original = _png(self.exact_original_png_bytes, "runtime original panel")
        sheet = _png(
            self.exact_support_sheet_png_bytes, "runtime support sheet"
        )
        if (
            len(original) != self.freeze.original_panel_png_byte_count
            or hashlib.sha256(original).hexdigest()
            != self.freeze.original_panel_png_digest
            or len(sheet) != self.freeze.support_sheet_png_byte_count
            or hashlib.sha256(sheet).hexdigest()
            != self.freeze.support_sheet_png_digest
            or len(sheet) > MAX_PANEL_PNG_BYTES
        ):
            raise ObjectSceneAnchorSupportPreparationError(
                "runtime support bytes differ from the persistent freeze"
            )

    @property
    def panel_alias(self) -> str:
        return self.freeze.panel_alias

    @property
    def support_bucket_index(self) -> int:
        return self.freeze.support_bucket_index


def _make_panel_freeze(
    panel_input: ObjectSceneAnchorSupportPanelInput,
    inventory: ObjectSceneProposalInventory,
    catalog: ObjectSceneAnchorCatalog,
    manifest: ObjectSceneAnchorPanelDecisionManifest,
    sheet: ObjectSceneAnchorSupportSheet,
    sheet_png: bytes,
) -> ObjectSceneAnchorSupportPanelFreeze:
    values = {
        "panel_alias": panel_input.panel_alias,
        "support_bucket_index": panel_input.support_bucket_index,
        "source_digest": panel_input.source_digest,
        "source_panel_binding_digest": panel_input.source_panel_binding_digest,
        "source_ordinal": panel_input.source_ordinal,
        "task_id": panel_input.task_id,
        "panel_id": panel_input.panel_id,
        "original_panel_png_byte_count": len(
            panel_input.exact_original_png_bytes
        ),
        "original_panel_png_digest": panel_input.original_panel_png_digest,
        "inventory": inventory,
        "catalog": catalog,
        "panel_manifest": manifest,
        "support_sheet": sheet,
        "support_sheet_png_byte_count": len(sheet_png),
        "support_sheet_png_digest": hashlib.sha256(sheet_png).hexdigest(),
        "proposal_count": len(inventory.objects),
        "object_ids": tuple(item.object_id for item in inventory.objects),
    }
    provisional = object.__new__(ObjectSceneAnchorSupportPanelFreeze)
    for name, item in values.items():
        object.__setattr__(provisional, name, item)
    return ObjectSceneAnchorSupportPanelFreeze(
        **values,
        freeze_digest=canonical_digest(_panel_content(provisional)),
    )


def build_object_scene_anchor_support_panel(
    panel_input: ObjectSceneAnchorSupportPanelInput,
) -> ObjectSceneAnchorSupportPanelRuntimeBundle:
    """Build the complete typed stack from pixels before attaching metadata."""

    if type(panel_input) is not ObjectSceneAnchorSupportPanelInput:
        raise TypeError("panel_input must be exact support panel input")
    # Deliberately pass only pixels into every geometry extraction stage.
    png_bytes = panel_input.exact_original_png_bytes
    inventory = extract_object_scene_proposal_inventory(png_bytes)
    catalog = extract_object_scene_anchor_catalog(png_bytes, inventory)
    manifest = build_object_scene_anchor_panel_decision_manifest(
        catalog, png_bytes, inventory
    )
    sheet, sheet_png = build_object_scene_anchor_support_sheet(
        png_bytes, inventory, catalog, manifest
    )
    if len(sheet_png) > MAX_PANEL_PNG_BYTES:
        raise ObjectSceneAnchorSupportPreparationError(
            "support sheet exceeds transport MAX_PANEL_PNG_BYTES"
        )
    freeze = _make_panel_freeze(
        panel_input, inventory, catalog, manifest, sheet, sheet_png
    )
    return ObjectSceneAnchorSupportPanelRuntimeBundle(
        freeze=freeze,
        exact_original_png_bytes=png_bytes,
        exact_support_sheet_png_bytes=sheet_png,
    )


def verify_object_scene_anchor_support_panel_runtime(
    bundle: ObjectSceneAnchorSupportPanelRuntimeBundle,
    panel_input: ObjectSceneAnchorSupportPanelInput,
    *,
    expected_freeze_digest: str | None = None,
) -> ObjectSceneAnchorSupportPanelRuntimeBundle:
    """Cold-replay one frozen stack from the exact source panel bytes."""

    if type(bundle) is not ObjectSceneAnchorSupportPanelRuntimeBundle:
        raise TypeError("bundle must be exact support panel runtime bundle")
    if type(panel_input) is not ObjectSceneAnchorSupportPanelInput:
        raise TypeError("panel_input must be exact support panel input")
    restored = ObjectSceneAnchorSupportPanelFreeze.from_data(
        bundle.freeze.to_data()
    )
    if expected_freeze_digest is not None and restored.freeze_digest != _digest(
        expected_freeze_digest, "expected support panel freeze digest"
    ):
        raise ObjectSceneAnchorSupportPreparationError(
            "support panel freeze differs from commitment"
        )
    expected_metadata = (
        panel_input.panel_alias,
        panel_input.support_bucket_index,
        panel_input.source_digest,
        panel_input.source_panel_binding_digest,
        panel_input.source_ordinal,
        panel_input.task_id,
        panel_input.panel_id,
        panel_input.original_panel_png_digest,
    )
    frozen_metadata = (
        restored.panel_alias,
        restored.support_bucket_index,
        restored.source_digest,
        restored.source_panel_binding_digest,
        restored.source_ordinal,
        restored.task_id,
        restored.panel_id,
        restored.original_panel_png_digest,
    )
    if expected_metadata != frozen_metadata:
        raise ObjectSceneAnchorSupportPreparationError(
            "support panel source input differs from freeze"
        )
    if panel_input.exact_original_png_bytes != bundle.exact_original_png_bytes:
        raise ObjectSceneAnchorSupportPreparationError(
            "runtime original bytes differ from exact source panel bytes"
        )
    replayed = build_object_scene_anchor_support_panel(panel_input)
    if replayed != bundle:
        raise ObjectSceneAnchorSupportPreparationError(
            "support panel differs from cold pixel replay"
        )
    return ObjectSceneAnchorSupportPanelRuntimeBundle(
        freeze=restored,
        exact_original_png_bytes=bundle.exact_original_png_bytes,
        exact_support_sheet_png_bytes=bundle.exact_support_sheet_png_bytes,
    )


def _corpus_content(
    value: "ObjectSceneAnchorSupportCorpusFreeze",
) -> dict[str, object]:
    return {
        "schema": OBJECT_SCENE_ANCHOR_SUPPORT_CORPUS_FREEZE_SCHEMA,
        "preparation_id": OBJECT_SCENE_ANCHOR_SUPPORT_PREPARATION_ID,
        "preparation_source_digest": (
            object_scene_anchor_support_preparation_source_digest()
        ),
        "source_digest": value.source_digest,
        "panel_count": value.panel_count,
        "bucket_0_count": value.bucket_0_count,
        "bucket_1_count": value.bucket_1_count,
        "panel_aliases": list(value.panel_aliases),
        "source_panel_binding_digests": list(
            value.source_panel_binding_digests
        ),
        "original_panel_png_digests": list(value.original_panel_png_digests),
        "complete_object_count": value.complete_object_count,
        "panels": [item.to_data() for item in value.panels],
        "panel_order": "panel_000-through-panel_011",
        "bucket_order": "six-bucket-0-then-six-bucket-1",
        "exact_panel_reuse_allowed": False,
        "complete_object_count_is_sum": True,
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class ObjectSceneAnchorSupportCorpusFreeze:
    """The exact twelve-panel, 6/6 exposed support corpus."""

    source_digest: str
    panel_count: int
    bucket_0_count: int
    bucket_1_count: int
    panel_aliases: tuple[str, ...]
    source_panel_binding_digests: tuple[str, ...]
    original_panel_png_digests: tuple[str, ...]
    complete_object_count: int
    panels: tuple[ObjectSceneAnchorSupportPanelFreeze, ...]
    freeze_digest: str

    def __post_init__(self) -> None:
        _digest(self.source_digest, "corpus source digest")
        for item in self.source_panel_binding_digests:
            _digest(item, "source panel binding digest")
        for item in self.original_panel_png_digests:
            _digest(item, "original panel PNG digest")
        for label, item in (
            ("panel count", self.panel_count),
            ("bucket 0 count", self.bucket_0_count),
            ("bucket 1 count", self.bucket_1_count),
            ("complete object count", self.complete_object_count),
        ):
            _integer(item, label)
        expected_aliases = tuple(
            f"panel_{index:03d}"
            for index in range(OBJECT_SCENE_ANCHOR_SUPPORT_PANEL_COUNT)
        )
        if (
            type(self.panels) is not tuple
            or any(
                type(item) is not ObjectSceneAnchorSupportPanelFreeze
                for item in self.panels
            )
            or self.panel_count != OBJECT_SCENE_ANCHOR_SUPPORT_PANEL_COUNT
            or len(self.panels) != self.panel_count
            or self.bucket_0_count != OBJECT_SCENE_ANCHOR_SUPPORT_BUCKET_SIZE
            or self.bucket_1_count != OBJECT_SCENE_ANCHOR_SUPPORT_BUCKET_SIZE
            or tuple(item.support_bucket_index for item in self.panels)
            != (0,) * OBJECT_SCENE_ANCHOR_SUPPORT_BUCKET_SIZE
            + (1,) * OBJECT_SCENE_ANCHOR_SUPPORT_BUCKET_SIZE
            or self.panel_aliases != expected_aliases
            or self.panel_aliases
            != tuple(item.panel_alias for item in self.panels)
            or self.source_panel_binding_digests
            != tuple(item.source_panel_binding_digest for item in self.panels)
            or self.original_panel_png_digests
            != tuple(item.original_panel_png_digest for item in self.panels)
            or any(item.source_digest != self.source_digest for item in self.panels)
            or len({item.panel_id for item in self.panels}) != self.panel_count
            or len(set(self.source_panel_binding_digests)) != self.panel_count
            or len(set(self.original_panel_png_digests)) != self.panel_count
            or len({item.inventory.panel_digest for item in self.panels})
            != self.panel_count
            or self.complete_object_count
            != sum(item.proposal_count for item in self.panels)
        ):
            raise ObjectSceneAnchorSupportPreparationError(
                "support corpus is not the exact unique 6/6 panel inventory"
            )
        unsigned = _corpus_content(self)
        _assert_persistent_payload(unsigned)
        _digest(self.freeze_digest, "support corpus freeze digest")
        if self.freeze_digest != canonical_digest(unsigned):
            raise ObjectSceneAnchorSupportPreparationError(
                "support corpus freeze digest differs"
            )

    @property
    def by_panel_alias(self) -> dict[str, ObjectSceneAnchorSupportPanelFreeze]:
        return dict(zip(self.panel_aliases, self.panels, strict=True))

    def to_data(self) -> dict[str, object]:
        return {**_corpus_content(self), "freeze_digest": self.freeze_digest}

    @classmethod
    def from_data(cls, value: object) -> "ObjectSceneAnchorSupportCorpusFreeze":
        expected = {
            "schema",
            "preparation_id",
            "preparation_source_digest",
            "source_digest",
            "panel_count",
            "bucket_0_count",
            "bucket_1_count",
            "panel_aliases",
            "source_panel_binding_digests",
            "original_panel_png_digests",
            "complete_object_count",
            "panels",
            "panel_order",
            "bucket_order",
            "exact_panel_reuse_allowed",
            "complete_object_count_is_sum",
            *_authority_data(),
            "freeze_digest",
        }
        raw = _fields(value, expected, "support corpus freeze")
        if (
            raw["schema"] != OBJECT_SCENE_ANCHOR_SUPPORT_CORPUS_FREEZE_SCHEMA
            or raw["preparation_id"] != OBJECT_SCENE_ANCHOR_SUPPORT_PREPARATION_ID
            or raw["preparation_source_digest"]
            != object_scene_anchor_support_preparation_source_digest()
            or raw["panel_order"] != "panel_000-through-panel_011"
            or raw["bucket_order"] != "six-bucket-0-then-six-bucket-1"
            or raw["exact_panel_reuse_allowed"] is not False
            or raw["complete_object_count_is_sum"] is not True
            or any(raw[key] != item for key, item in _authority_data().items())
            or not isinstance(raw["panel_aliases"], list)
            or not isinstance(raw["source_panel_binding_digests"], list)
            or not isinstance(raw["original_panel_png_digests"], list)
            or not isinstance(raw["panels"], list)
        ):
            raise ObjectSceneAnchorSupportPreparationError(
                "support corpus freeze policy differs"
            )
        result = cls(
            source_digest=raw["source_digest"],
            panel_count=raw["panel_count"],
            bucket_0_count=raw["bucket_0_count"],
            bucket_1_count=raw["bucket_1_count"],
            panel_aliases=tuple(raw["panel_aliases"]),
            source_panel_binding_digests=tuple(
                raw["source_panel_binding_digests"]
            ),
            original_panel_png_digests=tuple(raw["original_panel_png_digests"]),
            complete_object_count=raw["complete_object_count"],
            panels=tuple(
                ObjectSceneAnchorSupportPanelFreeze.from_data(item)
                for item in raw["panels"]
            ),
            freeze_digest=raw["freeze_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectSceneAnchorSupportPreparationError(
                "support corpus freeze is not canonical"
            )
        return result


def freeze_object_scene_anchor_support_corpus(
    source_digest: str,
    panels: tuple[ObjectSceneAnchorSupportPanelFreeze, ...],
) -> ObjectSceneAnchorSupportCorpusFreeze:
    """Freeze an already prepared exact twelve-panel support inventory."""

    _digest(source_digest, "corpus source digest")
    if type(panels) is not tuple:
        raise TypeError("panels must be an exact tuple")
    values = {
        "source_digest": source_digest,
        "panel_count": len(panels),
        "bucket_0_count": sum(item.support_bucket_index == 0 for item in panels),
        "bucket_1_count": sum(item.support_bucket_index == 1 for item in panels),
        "panel_aliases": tuple(item.panel_alias for item in panels),
        "source_panel_binding_digests": tuple(
            item.source_panel_binding_digest for item in panels
        ),
        "original_panel_png_digests": tuple(
            item.original_panel_png_digest for item in panels
        ),
        "complete_object_count": sum(item.proposal_count for item in panels),
        "panels": panels,
    }
    provisional = object.__new__(ObjectSceneAnchorSupportCorpusFreeze)
    for name, item in values.items():
        object.__setattr__(provisional, name, item)
    return ObjectSceneAnchorSupportCorpusFreeze(
        **values,
        freeze_digest=canonical_digest(_corpus_content(provisional)),
    )


@dataclass(frozen=True, slots=True)
class ObjectSceneAnchorSupportCorpusRuntimeBundle:
    """The persistent corpus freeze plus every panel's runtime bytes."""

    freeze: ObjectSceneAnchorSupportCorpusFreeze
    panels: tuple[ObjectSceneAnchorSupportPanelRuntimeBundle, ...]

    def __post_init__(self) -> None:
        if type(self.freeze) is not ObjectSceneAnchorSupportCorpusFreeze:
            raise TypeError("freeze must be exact support corpus freeze")
        if (
            type(self.panels) is not tuple
            or any(
                type(item) is not ObjectSceneAnchorSupportPanelRuntimeBundle
                for item in self.panels
            )
            or tuple(item.freeze for item in self.panels) != self.freeze.panels
        ):
            raise ObjectSceneAnchorSupportPreparationError(
                "corpus runtime panels differ from persistent freeze"
            )

    @property
    def by_panel_alias(
        self,
    ) -> dict[str, ObjectSceneAnchorSupportPanelRuntimeBundle]:
        return {item.panel_alias: item for item in self.panels}


def build_object_scene_anchor_support_corpus(
    source: ObjectBongardPanelRubricCalibrationSource,
) -> ObjectSceneAnchorSupportCorpusRuntimeBundle:
    """Prepare all twelve exact exposed panels from their source bytes."""

    if type(source) is not ObjectBongardPanelRubricCalibrationSource:
        raise TypeError("source must be exact panel calibration source")
    runtime_panels = tuple(
        build_object_scene_anchor_support_panel(
            ObjectSceneAnchorSupportPanelInput.from_source_panel(
                source, panel, panel_alias=f"panel_{index:03d}"
            )
        )
        for index, panel in enumerate(source.panels)
    )
    freeze = freeze_object_scene_anchor_support_corpus(
        source.source_digest,
        tuple(item.freeze for item in runtime_panels),
    )
    return ObjectSceneAnchorSupportCorpusRuntimeBundle(
        freeze=freeze, panels=runtime_panels
    )


def verify_object_scene_anchor_support_corpus_runtime(
    bundle: ObjectSceneAnchorSupportCorpusRuntimeBundle,
    source: ObjectBongardPanelRubricCalibrationSource,
    *,
    expected_freeze_digest: str | None = None,
) -> ObjectSceneAnchorSupportCorpusRuntimeBundle:
    """Cold-replay every nested artifact and sheet from exact source pixels."""

    if type(bundle) is not ObjectSceneAnchorSupportCorpusRuntimeBundle:
        raise TypeError("bundle must be exact support corpus runtime bundle")
    if type(source) is not ObjectBongardPanelRubricCalibrationSource:
        raise TypeError("source must be exact panel calibration source")
    restored = ObjectSceneAnchorSupportCorpusFreeze.from_data(
        bundle.freeze.to_data()
    )
    if expected_freeze_digest is not None and restored.freeze_digest != _digest(
        expected_freeze_digest, "expected support corpus freeze digest"
    ):
        raise ObjectSceneAnchorSupportPreparationError(
            "support corpus freeze differs from commitment"
        )
    if restored.source_digest != source.source_digest:
        raise ObjectSceneAnchorSupportPreparationError(
            "support corpus source commitment differs"
        )
    if len(source.panels) != len(bundle.panels):
        raise ObjectSceneAnchorSupportPreparationError(
            "support corpus source panel count differs"
        )
    verified_panels = tuple(
        verify_object_scene_anchor_support_panel_runtime(
            runtime,
            ObjectSceneAnchorSupportPanelInput.from_source_panel(
                source, source_panel, panel_alias=f"panel_{index:03d}"
            ),
            expected_freeze_digest=runtime.freeze.freeze_digest,
        )
        for index, (source_panel, runtime) in enumerate(
            zip(source.panels, bundle.panels, strict=True)
        )
    )
    replayed_freeze = freeze_object_scene_anchor_support_corpus(
        source.source_digest,
        tuple(item.freeze for item in verified_panels),
    )
    if replayed_freeze != restored:
        raise ObjectSceneAnchorSupportPreparationError(
            "support corpus differs from exact source replay"
        )
    return ObjectSceneAnchorSupportCorpusRuntimeBundle(
        freeze=restored, panels=verified_panels
    )


__all__ = (
    "OBJECT_SCENE_ANCHOR_SUPPORT_BUCKET_SIZE",
    "OBJECT_SCENE_ANCHOR_SUPPORT_CORPUS_FREEZE_SCHEMA",
    "OBJECT_SCENE_ANCHOR_SUPPORT_PANEL_COUNT",
    "OBJECT_SCENE_ANCHOR_SUPPORT_PANEL_FREEZE_SCHEMA",
    "OBJECT_SCENE_ANCHOR_SUPPORT_PREPARATION_ID",
    "ObjectSceneAnchorSupportCorpusFreeze",
    "ObjectSceneAnchorSupportCorpusRuntimeBundle",
    "ObjectSceneAnchorSupportPanelFreeze",
    "ObjectSceneAnchorSupportPanelInput",
    "ObjectSceneAnchorSupportPanelRuntimeBundle",
    "ObjectSceneAnchorSupportPreparationError",
    "build_object_scene_anchor_support_corpus",
    "build_object_scene_anchor_support_panel",
    "freeze_object_scene_anchor_support_corpus",
    "object_scene_anchor_support_preparation_source_digest",
    "verify_object_scene_anchor_support_corpus_runtime",
    "verify_object_scene_anchor_support_panel_runtime",
)
