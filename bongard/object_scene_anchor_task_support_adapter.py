"""Exact clean-TRAIN task adapter for anchor support preparation.

One frozen :class:`ObjectBongardTaskPlan`, one durably prepared release, and
the task's twelve exact released support panels are projected into the neutral
support-preparation pipeline.  The adapter does not implement geometry.  It
authenticates task/release custody, derives opaque 6/6 partition metadata, and
then delegates exact pixels to :mod:`object_scene_anchor_support_preparation`.

Persistent artifacts contain only canonical metadata and the existing
byte-free support corpus freeze.  Runtime bundles retain the released records
and the original/support-sheet bytes needed for cold replay.
"""

from __future__ import annotations

from bongard.runtime_source_snapshot import capture_loaded_source, verify_loaded_source


_LOADED_SOURCE_SHA256 = capture_loaded_source(__name__, __file__)

from dataclasses import dataclass, field
import hashlib
import re
from typing import Any, Mapping

from bongard.canonical import canonical_digest
from bongard.object_bongard_batch import ObjectBongardTaskPlan
from bongard.object_bongard_release_gate import (
    PreparedObjectBongardRelease,
    verify_prepared_object_bongard_release,
)
from bongard.object_scene_anchor_support_preparation import (
    ObjectSceneAnchorSupportCorpusFreeze,
    ObjectSceneAnchorSupportCorpusRuntimeBundle,
    ObjectSceneAnchorSupportPanelInput,
    ObjectSceneAnchorSupportPanelRuntimeBundle,
    build_object_scene_anchor_support_panel,
    freeze_object_scene_anchor_support_corpus,
    verify_object_scene_anchor_support_panel_runtime,
)
from bongard.official_panel_archive import ReleasedOfficialPanel
from bongard.python_predicate_authority import PYTHON_PREDICATE_AUTHORITY_ID


OBJECT_SCENE_ANCHOR_TASK_SUPPORT_PANEL_BINDING_SCHEMA = (
    "gkm.object-scene-anchor-task-support-panel-binding.v1"
)
OBJECT_SCENE_ANCHOR_TASK_SUPPORT_ADAPTER_SCHEMA = (
    "gkm.object-scene-anchor-task-support-adapter.v1"
)
OBJECT_SCENE_ANCHOR_TASK_SUPPORT_ADAPTER_ID = (
    "bongard.object-scene-anchor-task-support-adapter/exact-train-task-v1"
)
OBJECT_SCENE_ANCHOR_TASK_SUPPORT_PANEL_COUNT = 12
OBJECT_SCENE_ANCHOR_TASK_SUPPORT_BUCKET_SIZE = 6

_RAW_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_ADDRESS = re.compile(r"sha256:[0-9a-f]{64}\Z")
_PANEL_ALIAS = re.compile(r"panel_[0-9]{3}\Z")
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


class ObjectSceneAnchorTaskSupportAdapterError(ValueError):
    """The task, prepared release, panel inventory, or replay differs."""


def _raw_digest(value: object, label: str) -> str:
    if not isinstance(value, str) or _RAW_DIGEST.fullmatch(value) is None:
        raise ObjectSceneAnchorTaskSupportAdapterError(
            f"{label} must be a raw lowercase SHA-256"
        )
    return value


def _address(value: object, label: str) -> str:
    if not isinstance(value, str) or _ADDRESS.fullmatch(value) is None:
        raise ObjectSceneAnchorTaskSupportAdapterError(
            f"{label} must be a sha256: address"
        )
    return value


def _integer(value: object, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ObjectSceneAnchorTaskSupportAdapterError(
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
        raise ObjectSceneAnchorTaskSupportAdapterError(f"{label} fields differ")
    return value


def _authority_data() -> dict[str, object]:
    return {
        "predicate_authority_id": PYTHON_PREDICATE_AUTHORITY_ID,
        "python_is_canonical_authority": True,
        "model_calls_permitted": False,
        "official_test_pixels_consumed": False,
        "clean_train_task_required": True,
        "query_panels_permitted": False,
        "caller_supplied_labels_permitted": False,
        "geometry_input_policy": "neutral-panel-alias-and-exact-PNG-only",
    }


def object_scene_anchor_task_support_adapter_source_digest() -> str:
    return verify_loaded_source(__name__, expected_source_sha256=_LOADED_SOURCE_SHA256)


def _panel_binding_content(
    value: "ObjectSceneAnchorTaskSupportPanelBinding",
) -> dict[str, object]:
    return {
        "schema": OBJECT_SCENE_ANCHOR_TASK_SUPPORT_PANEL_BINDING_SCHEMA,
        "adapter_id": OBJECT_SCENE_ANCHOR_TASK_SUPPORT_ADAPTER_ID,
        "panel_alias": value.panel_alias,
        "source_ordinal": value.source_ordinal,
        "opaque_support_bucket_index": value.opaque_support_bucket_index,
        "task_id": value.task_id,
        "task_plan_digest": value.task_plan_digest,
        "panel_id": value.panel_id,
        "released_panel_record_digest": value.released_panel_record_digest,
        "release_receipt_digest": value.release_receipt_digest,
        "release_descriptor_digest": value.release_descriptor_digest,
        "execution_precommit_digest": value.execution_precommit_digest,
        "exposure_successor_digest": value.exposure_successor_digest,
        "png_byte_count": value.png_byte_count,
        "png_sha256": value.png_sha256,
        "source_panel_binding_digest": value.source_panel_binding_digest,
        "support_partition_is_not_geometry_input": True,
    }


@dataclass(frozen=True, slots=True)
class ObjectSceneAnchorTaskSupportPanelBinding:
    panel_alias: str
    source_ordinal: int
    opaque_support_bucket_index: int
    task_id: str
    task_plan_digest: str
    panel_id: str
    released_panel_record_digest: str
    release_receipt_digest: str
    release_descriptor_digest: str
    execution_precommit_digest: str
    exposure_successor_digest: str
    png_byte_count: int
    png_sha256: str
    source_panel_binding_digest: str
    binding_digest: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.panel_alias, str)
            or _PANEL_ALIAS.fullmatch(self.panel_alias) is None
            or type(self.opaque_support_bucket_index) is not int
            or self.opaque_support_bucket_index not in (0, 1)
            or not isinstance(self.task_id, str)
            or not self.task_id
            or not isinstance(self.panel_id, str)
            or not self.panel_id
        ):
            raise ObjectSceneAnchorTaskSupportAdapterError(
                "task support panel identity differs"
            )
        _integer(self.source_ordinal, "support panel source ordinal")
        for label, item in (
            ("task plan digest", self.task_plan_digest),
            ("released panel record digest", self.released_panel_record_digest),
            ("release receipt digest", self.release_receipt_digest),
            ("release descriptor digest", self.release_descriptor_digest),
            ("execution precommit digest", self.execution_precommit_digest),
            ("exposure successor digest", self.exposure_successor_digest),
        ):
            _address(item, label)
        _integer(self.png_byte_count, "support panel PNG byte count", minimum=1)
        _raw_digest(self.png_sha256, "support panel PNG digest")
        _raw_digest(self.source_panel_binding_digest, "source panel binding digest")
        _raw_digest(self.binding_digest, "support panel binding digest")
        if self.binding_digest != canonical_digest(_panel_binding_content(self)):
            raise ObjectSceneAnchorTaskSupportAdapterError(
                "task support panel binding digest differs"
            )

    def to_data(self) -> dict[str, object]:
        return {**_panel_binding_content(self), "binding_digest": self.binding_digest}

    @classmethod
    def from_data(
        cls, value: object
    ) -> "ObjectSceneAnchorTaskSupportPanelBinding":
        raw = _fields(
            value,
            {
                "schema",
                "adapter_id",
                "panel_alias",
                "source_ordinal",
                "opaque_support_bucket_index",
                "task_id",
                "task_plan_digest",
                "panel_id",
                "released_panel_record_digest",
                "release_receipt_digest",
                "release_descriptor_digest",
                "execution_precommit_digest",
                "exposure_successor_digest",
                "png_byte_count",
                "png_sha256",
                "source_panel_binding_digest",
                "support_partition_is_not_geometry_input",
                "binding_digest",
            },
            "task support panel binding",
        )
        if (
            raw["schema"] != OBJECT_SCENE_ANCHOR_TASK_SUPPORT_PANEL_BINDING_SCHEMA
            or raw["adapter_id"] != OBJECT_SCENE_ANCHOR_TASK_SUPPORT_ADAPTER_ID
            or raw["support_partition_is_not_geometry_input"] is not True
        ):
            raise ObjectSceneAnchorTaskSupportAdapterError(
                "task support panel binding policy differs"
            )
        result = cls(
            raw["panel_alias"],
            raw["source_ordinal"],
            raw["opaque_support_bucket_index"],
            raw["task_id"],
            raw["task_plan_digest"],
            raw["panel_id"],
            raw["released_panel_record_digest"],
            raw["release_receipt_digest"],
            raw["release_descriptor_digest"],
            raw["execution_precommit_digest"],
            raw["exposure_successor_digest"],
            raw["png_byte_count"],
            raw["png_sha256"],
            raw["source_panel_binding_digest"],
            raw["binding_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectSceneAnchorTaskSupportAdapterError(
                "task support panel binding is not canonical"
            )
        return result


def _adapter_content(value: "ObjectSceneAnchorTaskSupportAdapter") -> dict[str, object]:
    return {
        "schema": OBJECT_SCENE_ANCHOR_TASK_SUPPORT_ADAPTER_SCHEMA,
        "adapter_id": OBJECT_SCENE_ANCHOR_TASK_SUPPORT_ADAPTER_ID,
        "adapter_source_digest": (
            object_scene_anchor_task_support_adapter_source_digest()
        ),
        "source_digest": value.source_digest,
        "task_id": value.task_id,
        "task_plan_digest": value.task_plan_digest,
        "prepared_batch_plan_digest": value.prepared_batch_plan_digest,
        "execution_precommit_digest": value.execution_precommit_digest,
        "release_authorization_digest": value.release_authorization_digest,
        "exposure_successor_digest": value.exposure_successor_digest,
        "plan_store_receipt_digest": value.plan_store_receipt_digest,
        "precommit_store_receipt_digest": value.precommit_store_receipt_digest,
        "exposure_store_receipt_digest": value.exposure_store_receipt_digest,
        "authorization_store_receipt_digest": (
            value.authorization_store_receipt_digest
        ),
        "expected_support_panel_ids": list(value.expected_support_panel_ids),
        "panel_bindings": [item.to_data() for item in value.panel_bindings],
        "support_corpus_freeze": value.support_corpus_freeze.to_data(),
        "complete_object_count": value.complete_object_count,
        "raw_bytes_persisted": False,
        "support_order": "side-0-six-then-side-1-six",
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class ObjectSceneAnchorTaskSupportAdapter:
    source_digest: str
    task_id: str
    task_plan_digest: str
    prepared_batch_plan_digest: str
    execution_precommit_digest: str
    release_authorization_digest: str
    exposure_successor_digest: str
    plan_store_receipt_digest: str
    precommit_store_receipt_digest: str
    exposure_store_receipt_digest: str
    authorization_store_receipt_digest: str
    expected_support_panel_ids: tuple[str, ...]
    panel_bindings: tuple[ObjectSceneAnchorTaskSupportPanelBinding, ...]
    support_corpus_freeze: ObjectSceneAnchorSupportCorpusFreeze
    complete_object_count: int
    adapter_digest: str

    def __post_init__(self) -> None:
        _raw_digest(self.source_digest, "task support source digest")
        if not isinstance(self.task_id, str) or not self.task_id:
            raise ObjectSceneAnchorTaskSupportAdapterError(
                "task support adapter task ID differs"
            )
        for label, item in (
            ("task plan digest", self.task_plan_digest),
            ("prepared batch plan digest", self.prepared_batch_plan_digest),
            ("execution precommit digest", self.execution_precommit_digest),
            ("release authorization digest", self.release_authorization_digest),
            ("exposure successor digest", self.exposure_successor_digest),
            ("plan store receipt digest", self.plan_store_receipt_digest),
            ("precommit store receipt digest", self.precommit_store_receipt_digest),
            ("exposure store receipt digest", self.exposure_store_receipt_digest),
            (
                "authorization store receipt digest",
                self.authorization_store_receipt_digest,
            ),
        ):
            _address(item, label)
        _integer(self.complete_object_count, "complete object count")
        if (
            type(self.support_corpus_freeze)
            is not ObjectSceneAnchorSupportCorpusFreeze
        ):
            raise TypeError("support_corpus_freeze must be exact support corpus freeze")
        expected_aliases = tuple(
            f"panel_{index:03d}"
            for index in range(OBJECT_SCENE_ANCHOR_TASK_SUPPORT_PANEL_COUNT)
        )
        if (
            type(self.expected_support_panel_ids) is not tuple
            or type(self.panel_bindings) is not tuple
            or len(self.expected_support_panel_ids)
            != OBJECT_SCENE_ANCHOR_TASK_SUPPORT_PANEL_COUNT
            or len(set(self.expected_support_panel_ids))
            != OBJECT_SCENE_ANCHOR_TASK_SUPPORT_PANEL_COUNT
            or len(self.panel_bindings) != OBJECT_SCENE_ANCHOR_TASK_SUPPORT_PANEL_COUNT
            or any(
                type(item) is not ObjectSceneAnchorTaskSupportPanelBinding
                for item in self.panel_bindings
            )
            or tuple(item.panel_alias for item in self.panel_bindings)
            != expected_aliases
            or tuple(item.source_ordinal for item in self.panel_bindings)
            != tuple(range(OBJECT_SCENE_ANCHOR_TASK_SUPPORT_PANEL_COUNT))
            or tuple(item.opaque_support_bucket_index for item in self.panel_bindings)
            != (0,) * OBJECT_SCENE_ANCHOR_TASK_SUPPORT_BUCKET_SIZE
            + (1,) * OBJECT_SCENE_ANCHOR_TASK_SUPPORT_BUCKET_SIZE
            or tuple(item.panel_id for item in self.panel_bindings)
            != self.expected_support_panel_ids
            or any(
                item.task_id != self.task_id
                or item.task_plan_digest != self.task_plan_digest
                or item.execution_precommit_digest
                != self.execution_precommit_digest
                or item.exposure_successor_digest != self.exposure_successor_digest
                for item in self.panel_bindings
            )
            or self.support_corpus_freeze.source_digest != self.source_digest
            or self.support_corpus_freeze.panel_aliases != expected_aliases
            or self.support_corpus_freeze.complete_object_count
            != self.complete_object_count
            or tuple(item.panel_id for item in self.support_corpus_freeze.panels)
            != self.expected_support_panel_ids
            or tuple(
                item.source_panel_binding_digest
                for item in self.support_corpus_freeze.panels
            )
            != tuple(item.source_panel_binding_digest for item in self.panel_bindings)
            or any(
                binding.panel_alias != panel.panel_alias
                or binding.source_ordinal != panel.source_ordinal
                or binding.opaque_support_bucket_index
                != panel.support_bucket_index
                or binding.task_id != panel.task_id
                or binding.panel_id != panel.panel_id
                or binding.png_byte_count
                != panel.original_panel_png_byte_count
                or binding.png_sha256 != panel.original_panel_png_digest
                for binding, panel in zip(
                    self.panel_bindings,
                    self.support_corpus_freeze.panels,
                    strict=True,
                )
            )
        ):
            raise ObjectSceneAnchorTaskSupportAdapterError(
                "task support adapter inventory differs"
            )
        _raw_digest(self.adapter_digest, "task support adapter digest")
        if self.adapter_digest != canonical_digest(_adapter_content(self)):
            raise ObjectSceneAnchorTaskSupportAdapterError(
                "task support adapter digest differs"
            )

    def to_data(self) -> dict[str, object]:
        return {**_adapter_content(self), "adapter_digest": self.adapter_digest}

    @classmethod
    def from_data(cls, value: object) -> "ObjectSceneAnchorTaskSupportAdapter":
        raw = _fields(
            value,
            {
                "schema",
                "adapter_id",
                "adapter_source_digest",
                "source_digest",
                "task_id",
                "task_plan_digest",
                "prepared_batch_plan_digest",
                "execution_precommit_digest",
                "release_authorization_digest",
                "exposure_successor_digest",
                "plan_store_receipt_digest",
                "precommit_store_receipt_digest",
                "exposure_store_receipt_digest",
                "authorization_store_receipt_digest",
                "expected_support_panel_ids",
                "panel_bindings",
                "support_corpus_freeze",
                "complete_object_count",
                "raw_bytes_persisted",
                "support_order",
                *_authority_data(),
                "adapter_digest",
            },
            "task support adapter",
        )
        if (
            raw["schema"] != OBJECT_SCENE_ANCHOR_TASK_SUPPORT_ADAPTER_SCHEMA
            or raw["adapter_id"] != OBJECT_SCENE_ANCHOR_TASK_SUPPORT_ADAPTER_ID
            or raw["adapter_source_digest"]
            != object_scene_anchor_task_support_adapter_source_digest()
            or raw["raw_bytes_persisted"] is not False
            or raw["support_order"] != "side-0-six-then-side-1-six"
            or any(raw[key] != item for key, item in _authority_data().items())
            or not isinstance(raw["expected_support_panel_ids"], list)
            or not isinstance(raw["panel_bindings"], list)
            or not isinstance(raw["support_corpus_freeze"], Mapping)
        ):
            raise ObjectSceneAnchorTaskSupportAdapterError(
                "task support adapter policy differs"
            )
        result = cls(
            raw["source_digest"],
            raw["task_id"],
            raw["task_plan_digest"],
            raw["prepared_batch_plan_digest"],
            raw["execution_precommit_digest"],
            raw["release_authorization_digest"],
            raw["exposure_successor_digest"],
            raw["plan_store_receipt_digest"],
            raw["precommit_store_receipt_digest"],
            raw["exposure_store_receipt_digest"],
            raw["authorization_store_receipt_digest"],
            tuple(raw["expected_support_panel_ids"]),
            tuple(
                ObjectSceneAnchorTaskSupportPanelBinding.from_data(item)
                for item in raw["panel_bindings"]
            ),
            ObjectSceneAnchorSupportCorpusFreeze.from_data(
                raw["support_corpus_freeze"]
            ),
            raw["complete_object_count"],
            raw["adapter_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectSceneAnchorTaskSupportAdapterError(
                "task support adapter is not canonical"
            )
        return result


@dataclass(frozen=True, slots=True)
class ObjectSceneAnchorTaskGeometryPanelInput:
    """Label-free view passed to geometry/observer preparation."""

    panel_alias: str
    exact_png_bytes: bytes = field(repr=False)
    png_sha256: str
    source_panel_binding_digest: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.panel_alias, str)
            or _PANEL_ALIAS.fullmatch(self.panel_alias) is None
        ):
            raise ObjectSceneAnchorTaskSupportAdapterError(
                "geometry panel alias differs"
            )
        _raw_digest(self.png_sha256, "geometry PNG digest")
        _raw_digest(self.source_panel_binding_digest, "source panel binding digest")
        if (
            not isinstance(self.exact_png_bytes, bytes)
            or not self.exact_png_bytes.startswith(_PNG_SIGNATURE)
            or hashlib.sha256(self.exact_png_bytes).hexdigest() != self.png_sha256
        ):
            raise ObjectSceneAnchorTaskSupportAdapterError(
                "geometry panel bytes differ"
            )


@dataclass(frozen=True, slots=True)
class ObjectSceneAnchorTaskSupportRuntimeBundle:
    adapter: ObjectSceneAnchorTaskSupportAdapter
    support_corpus: ObjectSceneAnchorSupportCorpusRuntimeBundle
    released_panels: tuple[ReleasedOfficialPanel, ...] = field(repr=False)

    def __post_init__(self) -> None:
        if type(self.adapter) is not ObjectSceneAnchorTaskSupportAdapter:
            raise TypeError("adapter must be exact task support adapter")
        if (
            type(self.support_corpus)
            is not ObjectSceneAnchorSupportCorpusRuntimeBundle
        ):
            raise TypeError(
                "support_corpus must be exact support corpus runtime bundle"
            )
        if (
            type(self.released_panels) is not tuple
            or len(self.released_panels)
            != OBJECT_SCENE_ANCHOR_TASK_SUPPORT_PANEL_COUNT
            or any(
                type(item) is not ReleasedOfficialPanel
                for item in self.released_panels
            )
            or self.support_corpus.freeze != self.adapter.support_corpus_freeze
        ):
            raise ObjectSceneAnchorTaskSupportAdapterError(
                "task support runtime inventory differs"
            )
        for binding, prepared, released in zip(
            self.adapter.panel_bindings,
            self.support_corpus.panels,
            self.released_panels,
            strict=True,
        ):
            if (
                released.panel_id != binding.panel_id
                or released.record_digest != binding.released_panel_record_digest
                or released.release_receipt.record_digest
                != binding.release_receipt_digest
                or released.release_receipt.release_descriptor_digest
                != binding.release_descriptor_digest
                or released.execution_precommit_digest
                != binding.execution_precommit_digest
                or released.exposure_successor_digest
                != binding.exposure_successor_digest
                or len(released.exact_png_bytes) != binding.png_byte_count
                or released.exact_png_digest
                != f"sha256:{binding.png_sha256}"
                or released.exact_png_bytes != prepared.exact_original_png_bytes
                or prepared.freeze.source_panel_binding_digest
                != binding.source_panel_binding_digest
            ):
                raise ObjectSceneAnchorTaskSupportAdapterError(
                    "task support runtime panel differs from adapter"
                )

    @property
    def geometry_panel_inputs(
        self,
    ) -> tuple[ObjectSceneAnchorTaskGeometryPanelInput, ...]:
        return tuple(
            ObjectSceneAnchorTaskGeometryPanelInput(
                panel_alias=binding.panel_alias,
                exact_png_bytes=released.exact_png_bytes,
                png_sha256=binding.png_sha256,
                source_panel_binding_digest=binding.source_panel_binding_digest,
            )
            for binding, released in zip(
                self.adapter.panel_bindings,
                self.released_panels,
                strict=True,
            )
        )
def _expected_task_and_panels(
    task: ObjectBongardTaskPlan,
    prepared: PreparedObjectBongardRelease,
    released_panels: tuple[ReleasedOfficialPanel, ...],
) -> tuple[ObjectBongardTaskPlan, tuple[str, ...], tuple[ReleasedOfficialPanel, ...]]:
    if type(task) is not ObjectBongardTaskPlan:
        raise TypeError("task must be exact ObjectBongardTaskPlan")
    if type(prepared) is not PreparedObjectBongardRelease:
        raise TypeError("prepared must be exact PreparedObjectBongardRelease")
    if type(released_panels) is not tuple or any(
        type(item) is not ReleasedOfficialPanel for item in released_panels
    ):
        raise TypeError("released_panels must be an exact typed tuple")
    frozen_task = ObjectBongardTaskPlan.from_data(task.to_data())
    if frozen_task.split != "train":
        raise ObjectSceneAnchorTaskSupportAdapterError(
            "task support adapter accepts TRAIN only"
        )
    verify_prepared_object_bongard_release(prepared)
    matches = tuple(
        item for item in prepared.plan.tasks if item.task_id == frozen_task.task_id
    )
    expected_ids = (
        *frozen_task.side_0_support_panel_ids,
        *frozen_task.side_1_support_panel_ids,
    )
    query_ids = {
        frozen_task.side_0_query_panel_id,
        frozen_task.side_1_query_panel_id,
    }
    if (
        len(matches) != 1
        or matches[0] != frozen_task
        or prepared.precommit.batch_plan_digest != prepared.plan.record_digest
        or prepared.authorization.batch_plan_digest != prepared.plan.record_digest
        or prepared.authorization.execution_precommit_digest
        != prepared.precommit.record_digest
        or prepared.authorization.exposure_successor_digest
        != prepared.successor.digest
        or len(released_panels) != OBJECT_SCENE_ANCHOR_TASK_SUPPORT_PANEL_COUNT
        or tuple(item.panel_id for item in released_panels) != expected_ids
        or len({item.panel_id for item in released_panels})
        != OBJECT_SCENE_ANCHOR_TASK_SUPPORT_PANEL_COUNT
        or len({item.record_digest for item in released_panels})
        != OBJECT_SCENE_ANCHOR_TASK_SUPPORT_PANEL_COUNT
        or len({item.exact_png_digest for item in released_panels})
        != OBJECT_SCENE_ANCHOR_TASK_SUPPORT_PANEL_COUNT
        or any(item.panel_id in query_ids for item in released_panels)
        or any(
            item.panel_id
            not in prepared.authorization.authorized_support_panel_ids
            for item in released_panels
        )
    ):
        raise ObjectSceneAnchorTaskSupportAdapterError(
            "task support release inventory/order differs"
        )
    frozen_releases = tuple(
        ReleasedOfficialPanel.from_data(item.to_data()) for item in released_panels
    )
    for item in frozen_releases:
        if (
            item.execution_precommit_digest != prepared.precommit.record_digest
            or item.exposure_successor_digest != prepared.successor.digest
            or item.release_receipt.release_descriptor_digest
            != prepared.precommit.release_descriptor_digest
            or item.release_receipt.archive_digest
            != prepared.precommit.archive_digest
            or item.release_receipt.central_directory_digest
            != prepared.precommit.archive_central_directory_digest
        ):
            raise ObjectSceneAnchorTaskSupportAdapterError(
                "released support panel custody differs from prepared release"
            )
    return frozen_task, expected_ids, frozen_releases


def _source_content(
    task: ObjectBongardTaskPlan,
    prepared: PreparedObjectBongardRelease,
    released_panels: tuple[ReleasedOfficialPanel, ...],
) -> dict[str, object]:
    return {
        "schema": "gkm.object-scene-anchor-task-support-source.v1",
        "adapter_id": OBJECT_SCENE_ANCHOR_TASK_SUPPORT_ADAPTER_ID,
        "task_id": task.task_id,
        "task_plan_digest": task.record_digest,
        "prepared_batch_plan_digest": prepared.plan.record_digest,
        "execution_precommit_digest": prepared.precommit.record_digest,
        "release_authorization_digest": prepared.authorization.record_digest,
        "exposure_successor_digest": prepared.successor.digest,
        "ordered_panels": [
            {
                "panel_id": item.panel_id,
                "released_panel_record_digest": item.record_digest,
                "release_receipt_digest": item.release_receipt.record_digest,
                "exact_png_digest": item.exact_png_digest,
            }
            for item in released_panels
        ],
        "order": "side-0-six-then-side-1-six",
        "query_panels_included": False,
    }


def _make_panel_binding(
    *,
    index: int,
    task: ObjectBongardTaskPlan,
    prepared: PreparedObjectBongardRelease,
    released: ReleasedOfficialPanel,
    source_digest: str,
) -> ObjectSceneAnchorTaskSupportPanelBinding:
    source_panel_binding_digest = canonical_digest(
        {
            "schema": "gkm.object-scene-anchor-task-support-source-panel.v1",
            "adapter_id": OBJECT_SCENE_ANCHOR_TASK_SUPPORT_ADAPTER_ID,
            "source_digest": source_digest,
            "task_id": task.task_id,
            "task_plan_digest": task.record_digest,
            "panel_alias": f"panel_{index:03d}",
            "source_ordinal": index,
            "opaque_support_bucket_index": (
                0 if index < OBJECT_SCENE_ANCHOR_TASK_SUPPORT_BUCKET_SIZE else 1
            ),
            "panel_id": released.panel_id,
            "released_panel_record_digest": released.record_digest,
            "release_receipt_digest": released.release_receipt.record_digest,
            "execution_precommit_digest": prepared.precommit.record_digest,
            "exposure_successor_digest": prepared.successor.digest,
            "png_sha256": released.exact_png_digest.removeprefix("sha256:"),
        }
    )
    values = {
        "panel_alias": f"panel_{index:03d}",
        "source_ordinal": index,
        "opaque_support_bucket_index": (
            0 if index < OBJECT_SCENE_ANCHOR_TASK_SUPPORT_BUCKET_SIZE else 1
        ),
        "task_id": task.task_id,
        "task_plan_digest": task.record_digest,
        "panel_id": released.panel_id,
        "released_panel_record_digest": released.record_digest,
        "release_receipt_digest": released.release_receipt.record_digest,
        "release_descriptor_digest": released.release_receipt.release_descriptor_digest,
        "execution_precommit_digest": prepared.precommit.record_digest,
        "exposure_successor_digest": prepared.successor.digest,
        "png_byte_count": len(released.exact_png_bytes),
        "png_sha256": released.exact_png_digest.removeprefix("sha256:"),
        "source_panel_binding_digest": source_panel_binding_digest,
    }
    provisional = object.__new__(ObjectSceneAnchorTaskSupportPanelBinding)
    for name, item in values.items():
        object.__setattr__(provisional, name, item)
    return ObjectSceneAnchorTaskSupportPanelBinding(
        **values,
        binding_digest=canonical_digest(_panel_binding_content(provisional)),
    )


def _make_adapter(
    *,
    task: ObjectBongardTaskPlan,
    prepared: PreparedObjectBongardRelease,
    expected_ids: tuple[str, ...],
    bindings: tuple[ObjectSceneAnchorTaskSupportPanelBinding, ...],
    corpus: ObjectSceneAnchorSupportCorpusRuntimeBundle,
    source_digest: str,
) -> ObjectSceneAnchorTaskSupportAdapter:
    values = {
        "source_digest": source_digest,
        "task_id": task.task_id,
        "task_plan_digest": task.record_digest,
        "prepared_batch_plan_digest": prepared.plan.record_digest,
        "execution_precommit_digest": prepared.precommit.record_digest,
        "release_authorization_digest": prepared.authorization.record_digest,
        "exposure_successor_digest": prepared.successor.digest,
        "plan_store_receipt_digest": prepared.plan_receipt.record_digest,
        "precommit_store_receipt_digest": prepared.precommit_receipt.record_digest,
        "exposure_store_receipt_digest": prepared.exposure_receipt.record_digest,
        "authorization_store_receipt_digest": (
            prepared.authorization_receipt.record_digest
        ),
        "expected_support_panel_ids": expected_ids,
        "panel_bindings": bindings,
        "support_corpus_freeze": corpus.freeze,
        "complete_object_count": corpus.freeze.complete_object_count,
    }
    provisional = object.__new__(ObjectSceneAnchorTaskSupportAdapter)
    for name, item in values.items():
        object.__setattr__(provisional, name, item)
    return ObjectSceneAnchorTaskSupportAdapter(
        **values,
        adapter_digest=canonical_digest(_adapter_content(provisional)),
    )


def build_object_scene_anchor_task_support_corpus(
    *,
    task: ObjectBongardTaskPlan,
    prepared: PreparedObjectBongardRelease,
    released_panels: tuple[ReleasedOfficialPanel, ...],
) -> ObjectSceneAnchorTaskSupportRuntimeBundle:
    """Authenticate one exact TRAIN support inventory, then prepare pixels."""

    frozen_task, expected_ids, frozen_releases = _expected_task_and_panels(
        task, prepared, released_panels
    )
    source_digest = canonical_digest(
        _source_content(frozen_task, prepared, frozen_releases)
    )
    bindings = tuple(
        _make_panel_binding(
            index=index,
            task=frozen_task,
            prepared=prepared,
            released=released,
            source_digest=source_digest,
        )
        for index, released in enumerate(frozen_releases)
    )
    # Partition metadata is derived only after custody checks.  The delegated
    # support builder itself passes only ``exact_original_png_bytes`` into all
    # geometry extraction stages.
    prepared_panels = tuple(
        build_object_scene_anchor_support_panel(
            ObjectSceneAnchorSupportPanelInput(
                panel_alias=binding.panel_alias,
                support_bucket_index=binding.opaque_support_bucket_index,
                source_digest=source_digest,
                source_panel_binding_digest=binding.source_panel_binding_digest,
                source_ordinal=binding.source_ordinal,
                task_id=frozen_task.task_id,
                panel_id=binding.panel_id,
                original_panel_png_digest=binding.png_sha256,
                exact_original_png_bytes=released.exact_png_bytes,
            )
        )
        for binding, released in zip(bindings, frozen_releases, strict=True)
    )
    corpus_freeze = freeze_object_scene_anchor_support_corpus(
        source_digest,
        tuple(item.freeze for item in prepared_panels),
    )
    corpus = ObjectSceneAnchorSupportCorpusRuntimeBundle(
        freeze=corpus_freeze,
        panels=prepared_panels,
    )
    adapter = _make_adapter(
        task=frozen_task,
        prepared=prepared,
        expected_ids=expected_ids,
        bindings=bindings,
        corpus=corpus,
        source_digest=source_digest,
    )
    return ObjectSceneAnchorTaskSupportRuntimeBundle(
        adapter=adapter,
        support_corpus=corpus,
        released_panels=frozen_releases,
    )


def verify_object_scene_anchor_task_support_corpus(
    bundle: ObjectSceneAnchorTaskSupportRuntimeBundle,
    *,
    task: ObjectBongardTaskPlan,
    prepared: PreparedObjectBongardRelease,
    expected_adapter_digest: str | None = None,
) -> ObjectSceneAnchorTaskSupportRuntimeBundle:
    """Cold-replay custody and every panel's deterministic geometry stack."""

    if type(bundle) is not ObjectSceneAnchorTaskSupportRuntimeBundle:
        raise TypeError("bundle must be exact task support runtime bundle")
    restored = ObjectSceneAnchorTaskSupportAdapter.from_data(
        bundle.adapter.to_data()
    )
    if expected_adapter_digest is not None and restored.adapter_digest != _raw_digest(
        expected_adapter_digest, "expected task support adapter digest"
    ):
        raise ObjectSceneAnchorTaskSupportAdapterError(
            "task support adapter differs from commitment"
        )
    frozen_task, expected_ids, frozen_releases = _expected_task_and_panels(
        task, prepared, bundle.released_panels
    )
    source_digest = canonical_digest(
        _source_content(frozen_task, prepared, frozen_releases)
    )
    bindings = tuple(
        _make_panel_binding(
            index=index,
            task=frozen_task,
            prepared=prepared,
            released=released,
            source_digest=source_digest,
        )
        for index, released in enumerate(frozen_releases)
    )
    verified_panels: list[ObjectSceneAnchorSupportPanelRuntimeBundle] = []
    for binding, released, panel_runtime in zip(
        bindings,
        frozen_releases,
        bundle.support_corpus.panels,
        strict=True,
    ):
        panel_input = ObjectSceneAnchorSupportPanelInput(
            panel_alias=binding.panel_alias,
            support_bucket_index=binding.opaque_support_bucket_index,
            source_digest=source_digest,
            source_panel_binding_digest=binding.source_panel_binding_digest,
            source_ordinal=binding.source_ordinal,
            task_id=frozen_task.task_id,
            panel_id=binding.panel_id,
            original_panel_png_digest=binding.png_sha256,
            exact_original_png_bytes=released.exact_png_bytes,
        )
        verified_panels.append(
            verify_object_scene_anchor_support_panel_runtime(
                panel_runtime,
                panel_input,
                expected_freeze_digest=panel_runtime.freeze.freeze_digest,
            )
        )
    corpus_freeze = freeze_object_scene_anchor_support_corpus(
        source_digest,
        tuple(item.freeze for item in verified_panels),
    )
    corpus = ObjectSceneAnchorSupportCorpusRuntimeBundle(
        freeze=corpus_freeze,
        panels=tuple(verified_panels),
    )
    adapter = _make_adapter(
        task=frozen_task,
        prepared=prepared,
        expected_ids=expected_ids,
        bindings=bindings,
        corpus=corpus,
        source_digest=source_digest,
    )
    replayed = ObjectSceneAnchorTaskSupportRuntimeBundle(
        adapter=adapter,
        support_corpus=corpus,
        released_panels=frozen_releases,
    )
    if replayed != bundle or adapter != restored:
        raise ObjectSceneAnchorTaskSupportAdapterError(
            "task support adapter differs from exact cold replay"
        )
    return replayed


__all__ = (
    "OBJECT_SCENE_ANCHOR_TASK_SUPPORT_ADAPTER_ID",
    "OBJECT_SCENE_ANCHOR_TASK_SUPPORT_ADAPTER_SCHEMA",
    "OBJECT_SCENE_ANCHOR_TASK_SUPPORT_BUCKET_SIZE",
    "OBJECT_SCENE_ANCHOR_TASK_SUPPORT_PANEL_BINDING_SCHEMA",
    "OBJECT_SCENE_ANCHOR_TASK_SUPPORT_PANEL_COUNT",
    "ObjectSceneAnchorTaskGeometryPanelInput",
    "ObjectSceneAnchorTaskSupportAdapter",
    "ObjectSceneAnchorTaskSupportAdapterError",
    "ObjectSceneAnchorTaskSupportPanelBinding",
    "ObjectSceneAnchorTaskSupportRuntimeBundle",
    "build_object_scene_anchor_task_support_corpus",
    "object_scene_anchor_task_support_adapter_source_digest",
    "verify_object_scene_anchor_task_support_corpus",
)
