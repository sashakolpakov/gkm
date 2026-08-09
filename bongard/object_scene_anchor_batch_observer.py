"""Batched role-blind visual observation over exact anchor preparations.

The committed single-catalog observer defines the semantic cell: one exact
resolved binding crossed with one affirmative global witness, with P/A/I model
states, E for observer failures, and a same-key two-pass merge.  This module
changes only the transport packing.  Preparations sharing an exact object crop
and anchor atlas reuse those two images.  A deterministic greedy partition
respects the sixteen-view, thirty-two-image, and fixed-cell caps; if one view's
catalogs cross a cell boundary, its exact two images are presented again in the
following batch while every indivisible preparation remains intact.

No comparison role, predicate, polarity, or downstream Boolean expression is
visible at the model boundary.  Every batch is one exhaustive
subject/catalog/binding/witness rectangle and is attempted exactly twice.
"""

from __future__ import annotations

from bongard.runtime_source_snapshot import capture_loaded_source, verify_loaded_source


_LOADED_SOURCE_SHA256 = capture_loaded_source(__name__, __file__)

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping, Sequence

from bongard.canonical import canonical_digest, canonical_json
from bongard.evidence import Disposition
from bongard.object_scene_anchor_observer import (
    ObjectSceneAnchorObserverCell,
    OBJECT_SCENE_ANCHOR_OBSERVER_MAX_CELLS,
    ObjectSceneAnchorObserverPreparation,
    ObjectSceneAnchorObserverVocabulary,
    ObjectSceneAnchorObserverVocabularyEntry,
    _canonical_payload,
    _exception_type,
    _merge_cell_dispositions,
    _model_digest,
    _receipt_data,
    _receipt_from_data,
)
from bongard.prototype_scene_observer import (
    CloudPolicyCacheSnapshot,
    CodexModelCatalogSnapshot,
    CodexNoToolsAttestation,
    PrototypeImageIdentity,
)
from bongard import prototype_scene_observer as _scene_runtime
from bongard.python_predicate_authority import PYTHON_PREDICATE_AUTHORITY_ID
from bongard.transport import (
    CODEX_RECEIPT_SCHEMA,
    DEFAULT_CODEX_MODEL,
    DEFAULT_REASONING_EFFORT,
    NAMED_IMAGE_INPUT_DIGEST_SCHEMA,
    CodexReceipt,
    run_codex_named_images_structured,
    validate_codex_named_image_receipt,
    validate_codex_strict_output_schema,
)


OBJECT_SCENE_ANCHOR_BATCH_CATALOG_SCHEMA = (
    "gkm.object-scene-anchor-batch-observer-catalog.v1"
)
OBJECT_SCENE_ANCHOR_BATCH_SUBJECT_SCHEMA = (
    "gkm.object-scene-anchor-batch-observer-subject.v1"
)
OBJECT_SCENE_ANCHOR_BATCH_PLAN_ITEM_SCHEMA = (
    "gkm.object-scene-anchor-batch-observer-plan-item.v2"
)
OBJECT_SCENE_ANCHOR_BATCH_PLAN_SCHEMA = (
    "gkm.object-scene-anchor-batch-observer-plan.v2"
)
OBJECT_SCENE_ANCHOR_BATCH_PASS_SCHEMA = (
    "gkm.object-scene-anchor-batch-observer-pass.v1"
)
OBJECT_SCENE_ANCHOR_BATCH_RESULT_SCHEMA = (
    "gkm.object-scene-anchor-batch-observer-result.v1"
)
OBJECT_SCENE_ANCHOR_BATCH_ARTIFACT_SCHEMA = (
    "gkm.object-scene-anchor-batch-observer-artifact.v1"
)
OBJECT_SCENE_ANCHOR_BATCH_PROTOCOL_ID = (
    "bongard.object-scene-anchor-batch-observer/role-blind-cell-aware-two-pass-v2"
)
OBJECT_SCENE_ANCHOR_BATCH_MAX_VIEWS = 16
OBJECT_SCENE_ANCHOR_BATCH_MAX_IMAGES = 32
# One maximum legal preparation is 17 bindings x 32 global witnesses.
OBJECT_SCENE_ANCHOR_BATCH_MAX_CELLS = OBJECT_SCENE_ANCHOR_OBSERVER_MAX_CELLS

_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_ADDRESS = re.compile(r"sha256:[0-9a-f]{64}\Z")
_SUBJECT_ALIAS = re.compile(r"subject_[0-9]{2}\Z")
_CATALOG_ALIAS = re.compile(r"catalog_[0-9]{2}\Z")
_BATCH_ID = re.compile(r"batch_[0-9]{3}\Z")
_PASS_ID = re.compile(r"pass_0[01]\Z")
_ROLE_WORD = re.compile(
    r"\b(?:target|foil|support|contrast|candidate|formula|"
    r"query|answer|class|label|group)\b",
    re.IGNORECASE,
)
_MODEL_STATES = frozenset(("P", "A", "I"))
_REASONS = {
    "P": frozenset(("visible_match",)),
    "A": frozenset(("visible_mismatch",)),
    "I": frozenset(
        (
            "anchor_unresolved",
            "conflicting_evidence",
            "image_quality",
            "unclear_geometry",
            "unclear_marking",
        )
    ),
}
_PARTITION_POLICY = "view-catalog-digest-ascending-greedy-cell-aware-v2"


class ObjectSceneAnchorBatchObserverError(ValueError):
    """A batch plan, model payload, artifact, or replay is not canonical."""


class ObjectSceneAnchorBatchObserverPayloadError(
    ObjectSceneAnchorBatchObserverError
):
    """A receipted batch payload violates the exhaustive finite grammar."""


class ObjectSceneAnchorBatchCapacityGap(ObjectSceneAnchorBatchObserverError):
    """One indivisible observer preparation cannot fit the fixed cell cap."""

    def __init__(
        self,
        preparation_digest: str,
        cell_count: int,
        maximum_cell_count: int,
    ) -> None:
        self.preparation_digest = preparation_digest
        self.cell_count = cell_count
        self.maximum_cell_count = maximum_cell_count
        super().__init__(
            "one observer preparation exceeds the batch cell capacity; "
            "nothing was split, dropped, or pruned"
        )


def _authority_data() -> dict[str, object]:
    return {
        "predicate_authority_id": PYTHON_PREDICATE_AUTHORITY_ID,
        "python_is_canonical_authority": True,
        "role_blind_model_boundary": True,
        "comparison_roles_model_visible": False,
        "logical_expression_model_visible": False,
        "polarity_reversal_allowed": False,
        "uncertain_or_failed_vision_counts_as_absence": False,
        "two_independent_passes_per_batch": True,
        "merge_occurs_before_downstream_logic": True,
        "query_identity_model_visible": False,
    }


def _exact_fields(
    value: object, expected: set[str] | frozenset[str], label: str
) -> Mapping[str, Any]:
    if (
        not isinstance(value, Mapping)
        or any(not isinstance(key, str) for key in value)
        or set(value) != set(expected)
    ):
        raise ObjectSceneAnchorBatchObserverError(f"{label} fields differ")
    return value


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise ObjectSceneAnchorBatchObserverError(
            f"{label} must be a lowercase SHA-256"
        )
    return value


def _address(value: object, label: str) -> str:
    if not isinstance(value, str) or _ADDRESS.fullmatch(value) is None:
        raise ObjectSceneAnchorBatchObserverError(f"{label} must be a sha256: address")
    return value


def _integer(value: object, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ObjectSceneAnchorBatchObserverError(
            f"{label} must be an integer at least {minimum}"
        )
    return value


def object_scene_anchor_batch_observer_source_digest() -> str:
    return verify_loaded_source(__name__, expected_source_sha256=_LOADED_SOURCE_SHA256)


def _view_content(preparation: ObjectSceneAnchorObserverPreparation) -> dict[str, object]:
    return {
        "schema": "gkm.object-scene-anchor-batch-observer-view.v1",
        "panel_manifest_digest": preparation.panel_manifest_digest,
        "object_index": preparation.object_index,
        "object_id": preparation.object_id,
        "decision_manifest_digest": preparation.decision_manifest_digest,
        "crop_png_byte_count": preparation.crop_png_byte_count,
        "crop_png_digest": preparation.crop_png_digest,
        "crop_pixel_digest": preparation.crop_pixel_digest,
        "crop_width_pixels": preparation.crop_width_pixels,
        "crop_height_pixels": preparation.crop_height_pixels,
        "atlas_artifact_digest": preparation.atlas_artifact_digest,
        "atlas_png_byte_count": preparation.atlas_png_byte_count,
        "atlas_png_digest": preparation.atlas_png_digest,
    }


def _view_digest(preparation: ObjectSceneAnchorObserverPreparation) -> str:
    return canonical_digest(_view_content(preparation))


@dataclass(frozen=True, slots=True)
class ObjectSceneAnchorBatchObserverInput:
    """Runtime-only exact preparation and its already-verified two PNGs."""

    preparation: ObjectSceneAnchorObserverPreparation
    crop_png_bytes: bytes
    atlas_png_bytes: bytes

    def __post_init__(self) -> None:
        if type(self.preparation) is not ObjectSceneAnchorObserverPreparation:
            raise TypeError("batch input requires an exact observer preparation")
        restored = ObjectSceneAnchorObserverPreparation.from_data(
            self.preparation.to_data()
        )
        if restored != self.preparation:
            raise ObjectSceneAnchorBatchObserverError(
                "batch input preparation is not canonical"
            )
        crop = _scene_runtime._validate_exact_png(
            self.crop_png_bytes, "batch object crop"
        )
        atlas = _scene_runtime._validate_exact_png(
            self.atlas_png_bytes, "batch anchor atlas"
        )
        if (
            len(crop) != restored.crop_png_byte_count
            or hashlib.sha256(crop).hexdigest() != restored.crop_png_digest
            or len(atlas) != restored.atlas_png_byte_count
            or hashlib.sha256(atlas).hexdigest() != restored.atlas_png_digest
        ):
            raise ObjectSceneAnchorBatchObserverError(
                "batch PNG bytes differ from preparation commitments"
            )

    @property
    def view_digest(self) -> str:
        return _view_digest(self.preparation)


def _catalog_plan_content(
    value: "ObjectSceneAnchorBatchCatalogPlan",
) -> dict[str, object]:
    return {
        "schema": OBJECT_SCENE_ANCHOR_BATCH_CATALOG_SCHEMA,
        "catalog_alias": value.catalog_alias,
        "preparation": value.preparation.to_data(),
        "preparation_digest": value.preparation_digest,
        "catalog_digest": value.catalog_digest,
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class ObjectSceneAnchorBatchCatalogPlan:
    catalog_alias: str
    preparation: ObjectSceneAnchorObserverPreparation
    preparation_digest: str
    catalog_digest: str
    plan_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.catalog_alias, str) or _CATALOG_ALIAS.fullmatch(
            self.catalog_alias
        ) is None:
            raise ObjectSceneAnchorBatchObserverError("catalog alias differs")
        if type(self.preparation) is not ObjectSceneAnchorObserverPreparation:
            raise TypeError("catalog plan preparation has the wrong type")
        for label, item in (
            ("preparation digest", self.preparation_digest),
            ("catalog digest", self.catalog_digest),
            ("catalog plan digest", self.plan_digest),
        ):
            _digest(item, label)
        if (
            self.preparation.preparation_digest != self.preparation_digest
            or self.preparation.catalog_digest != self.catalog_digest
        ):
            raise ObjectSceneAnchorBatchObserverError(
                "catalog plan differs from exact preparation"
            )
        if self.plan_digest != canonical_digest(_catalog_plan_content(self)):
            raise ObjectSceneAnchorBatchObserverError("catalog plan digest differs")

    @classmethod
    def create(
        cls, alias: str, preparation: ObjectSceneAnchorObserverPreparation
    ) -> "ObjectSceneAnchorBatchCatalogPlan":
        values = {
            "catalog_alias": alias,
            "preparation": preparation,
            "preparation_digest": preparation.preparation_digest,
            "catalog_digest": preparation.catalog_digest,
        }
        provisional = object.__new__(cls)
        for name, item in values.items():
            object.__setattr__(provisional, name, item)
        return cls(**values, plan_digest=canonical_digest(_catalog_plan_content(provisional)))

    def to_data(self) -> dict[str, object]:
        return {**_catalog_plan_content(self), "plan_digest": self.plan_digest}

    @classmethod
    def from_data(cls, value: object) -> "ObjectSceneAnchorBatchCatalogPlan":
        raw = _exact_fields(
            value,
            {
                "schema",
                "catalog_alias",
                "preparation",
                "preparation_digest",
                "catalog_digest",
                *_authority_data(),
                "plan_digest",
            },
            "batch catalog plan",
        )
        if (
            raw["schema"] != OBJECT_SCENE_ANCHOR_BATCH_CATALOG_SCHEMA
            or any(raw[key] != item for key, item in _authority_data().items())
            or not isinstance(raw["preparation"], Mapping)
        ):
            raise ObjectSceneAnchorBatchObserverError("catalog plan policy differs")
        result = cls(
            raw["catalog_alias"],
            ObjectSceneAnchorObserverPreparation.from_data(raw["preparation"]),
            raw["preparation_digest"],
            raw["catalog_digest"],
            raw["plan_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectSceneAnchorBatchObserverError("catalog plan is not canonical")
        return result


def _subject_content(value: "ObjectSceneAnchorBatchSubjectPlan") -> dict[str, object]:
    return {
        "schema": OBJECT_SCENE_ANCHOR_BATCH_SUBJECT_SCHEMA,
        "subject_alias": value.subject_alias,
        "view_digest": value.view_digest,
        "object_image_name": value.object_image_name,
        "anchor_image_name": value.anchor_image_name,
        "catalogs": [item.to_data() for item in value.catalogs],
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class ObjectSceneAnchorBatchSubjectPlan:
    subject_alias: str
    view_digest: str
    object_image_name: str
    anchor_image_name: str
    catalogs: tuple[ObjectSceneAnchorBatchCatalogPlan, ...]
    subject_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.subject_alias, str) or _SUBJECT_ALIAS.fullmatch(
            self.subject_alias
        ) is None:
            raise ObjectSceneAnchorBatchObserverError("subject alias differs")
        _digest(self.view_digest, "subject view digest")
        expected_names = (
            f"{self.subject_alias}_object.png",
            f"{self.subject_alias}_anchors.png",
        )
        if (self.object_image_name, self.anchor_image_name) != expected_names:
            raise ObjectSceneAnchorBatchObserverError("subject image aliases differ")
        if (
            type(self.catalogs) is not tuple
            or not self.catalogs
            or any(type(item) is not ObjectSceneAnchorBatchCatalogPlan for item in self.catalogs)
            or tuple(item.catalog_alias for item in self.catalogs)
            != tuple(f"catalog_{index:02d}" for index in range(len(self.catalogs)))
            or tuple(item.catalog_digest for item in self.catalogs)
            != tuple(sorted(item.catalog_digest for item in self.catalogs))
            or len({item.preparation_digest for item in self.catalogs})
            != len(self.catalogs)
            or any(_view_digest(item.preparation) != self.view_digest for item in self.catalogs)
        ):
            raise ObjectSceneAnchorBatchObserverError(
                "subject catalogs are not one canonical shared view"
            )
        vocabularies = {item.preparation.vocabulary_digest for item in self.catalogs}
        if len(vocabularies) != 1:
            raise ObjectSceneAnchorBatchObserverError(
                "shared-view catalogs use different global vocabularies"
            )
        _digest(self.subject_digest, "subject digest")
        if self.subject_digest != canonical_digest(_subject_content(self)):
            raise ObjectSceneAnchorBatchObserverError("subject digest differs")

    @classmethod
    def create(
        cls,
        alias: str,
        preparations: Sequence[ObjectSceneAnchorObserverPreparation],
    ) -> "ObjectSceneAnchorBatchSubjectPlan":
        ordered = tuple(sorted(preparations, key=lambda item: item.catalog_digest))
        if not ordered:
            raise ObjectSceneAnchorBatchObserverError("subject has no catalogs")
        catalogs = tuple(
            ObjectSceneAnchorBatchCatalogPlan.create(f"catalog_{index:02d}", item)
            for index, item in enumerate(ordered)
        )
        values = {
            "subject_alias": alias,
            "view_digest": _view_digest(ordered[0]),
            "object_image_name": f"{alias}_object.png",
            "anchor_image_name": f"{alias}_anchors.png",
            "catalogs": catalogs,
        }
        provisional = object.__new__(cls)
        for name, item in values.items():
            object.__setattr__(provisional, name, item)
        return cls(**values, subject_digest=canonical_digest(_subject_content(provisional)))

    def to_data(self) -> dict[str, object]:
        return {**_subject_content(self), "subject_digest": self.subject_digest}

    @classmethod
    def from_data(cls, value: object) -> "ObjectSceneAnchorBatchSubjectPlan":
        raw = _exact_fields(
            value,
            {
                "schema",
                "subject_alias",
                "view_digest",
                "object_image_name",
                "anchor_image_name",
                "catalogs",
                *_authority_data(),
                "subject_digest",
            },
            "batch subject plan",
        )
        if (
            raw["schema"] != OBJECT_SCENE_ANCHOR_BATCH_SUBJECT_SCHEMA
            or any(raw[key] != item for key, item in _authority_data().items())
            or not isinstance(raw["catalogs"], list)
        ):
            raise ObjectSceneAnchorBatchObserverError("subject plan policy differs")
        result = cls(
            raw["subject_alias"],
            raw["view_digest"],
            raw["object_image_name"],
            raw["anchor_image_name"],
            tuple(ObjectSceneAnchorBatchCatalogPlan.from_data(item) for item in raw["catalogs"]),
            raw["subject_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectSceneAnchorBatchObserverError("subject plan is not canonical")
        return result


def _batch_plan_content(value: "ObjectSceneAnchorBatchPlanItem") -> dict[str, object]:
    return {
        "schema": OBJECT_SCENE_ANCHOR_BATCH_PLAN_ITEM_SCHEMA,
        "batch_id": value.batch_id,
        "batch_index": value.batch_index,
        "subjects": [item.to_data() for item in value.subjects],
        "view_count": value.view_count,
        "image_count": value.image_count,
        "catalog_count": value.catalog_count,
        "cell_count": value.cell_count,
        "maximum_cell_count": OBJECT_SCENE_ANCHOR_BATCH_MAX_CELLS,
        "partition_policy": _PARTITION_POLICY,
        "preparation_split_allowed": False,
        "silent_pruning_allowed": False,
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class ObjectSceneAnchorBatchPlanItem:
    batch_id: str
    batch_index: int
    subjects: tuple[ObjectSceneAnchorBatchSubjectPlan, ...]
    view_count: int
    image_count: int
    catalog_count: int
    cell_count: int
    batch_digest: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.batch_id, str)
            or _BATCH_ID.fullmatch(self.batch_id) is None
            or self.batch_id != f"batch_{self.batch_index:03d}"
        ):
            raise ObjectSceneAnchorBatchObserverError("batch position differs")
        _integer(self.batch_index, "batch index")
        if (
            type(self.subjects) is not tuple
            or not 1 <= len(self.subjects) <= OBJECT_SCENE_ANCHOR_BATCH_MAX_VIEWS
            or any(type(item) is not ObjectSceneAnchorBatchSubjectPlan for item in self.subjects)
            or tuple(item.subject_alias for item in self.subjects)
            != tuple(f"subject_{index:02d}" for index in range(len(self.subjects)))
            or tuple(item.view_digest for item in self.subjects)
            != tuple(sorted(item.view_digest for item in self.subjects))
            or len({item.view_digest for item in self.subjects}) != len(self.subjects)
        ):
            raise ObjectSceneAnchorBatchObserverError("batch subjects differ")
        expected_catalogs = sum(len(item.catalogs) for item in self.subjects)
        expected_cells = sum(
            catalog.preparation.cell_count
            for subject in self.subjects
            for catalog in subject.catalogs
        )
        if (
            self.view_count != len(self.subjects)
            or self.image_count != 2 * len(self.subjects)
            or self.image_count > OBJECT_SCENE_ANCHOR_BATCH_MAX_IMAGES
            or self.catalog_count != expected_catalogs
            or self.cell_count != expected_cells
            or not 1 <= self.cell_count <= OBJECT_SCENE_ANCHOR_BATCH_MAX_CELLS
        ):
            raise ObjectSceneAnchorBatchObserverError("batch counts differ")
        _digest(self.batch_digest, "batch digest")
        if self.batch_digest != canonical_digest(_batch_plan_content(self)):
            raise ObjectSceneAnchorBatchObserverError("batch digest differs")

    @classmethod
    def create(
        cls,
        index: int,
        grouped_preparations: Sequence[Sequence[ObjectSceneAnchorObserverPreparation]],
    ) -> "ObjectSceneAnchorBatchPlanItem":
        subjects = tuple(
            ObjectSceneAnchorBatchSubjectPlan.create(f"subject_{subject_index:02d}", group)
            for subject_index, group in enumerate(grouped_preparations)
        )
        values = {
            "batch_id": f"batch_{index:03d}",
            "batch_index": index,
            "subjects": subjects,
            "view_count": len(subjects),
            "image_count": 2 * len(subjects),
            "catalog_count": sum(len(item.catalogs) for item in subjects),
            "cell_count": sum(
                catalog.preparation.cell_count
                for subject in subjects
                for catalog in subject.catalogs
            ),
        }
        provisional = object.__new__(cls)
        for name, item in values.items():
            object.__setattr__(provisional, name, item)
        return cls(**values, batch_digest=canonical_digest(_batch_plan_content(provisional)))

    def to_data(self) -> dict[str, object]:
        return {**_batch_plan_content(self), "batch_digest": self.batch_digest}

    @classmethod
    def from_data(cls, value: object) -> "ObjectSceneAnchorBatchPlanItem":
        raw = _exact_fields(
            value,
            {
                "schema",
                "batch_id",
                "batch_index",
                "subjects",
                "view_count",
                "image_count",
                "catalog_count",
                "cell_count",
                "maximum_cell_count",
                "partition_policy",
                "preparation_split_allowed",
                "silent_pruning_allowed",
                *_authority_data(),
                "batch_digest",
            },
            "batch plan item",
        )
        if (
            raw["schema"] != OBJECT_SCENE_ANCHOR_BATCH_PLAN_ITEM_SCHEMA
            or raw["maximum_cell_count"] != OBJECT_SCENE_ANCHOR_BATCH_MAX_CELLS
            or raw["partition_policy"] != _PARTITION_POLICY
            or raw["preparation_split_allowed"] is not False
            or raw["silent_pruning_allowed"] is not False
            or any(raw[key] != item for key, item in _authority_data().items())
            or not isinstance(raw["subjects"], list)
        ):
            raise ObjectSceneAnchorBatchObserverError("batch plan policy differs")
        result = cls(
            raw["batch_id"],
            raw["batch_index"],
            tuple(ObjectSceneAnchorBatchSubjectPlan.from_data(item) for item in raw["subjects"]),
            raw["view_count"],
            raw["image_count"],
            raw["catalog_count"],
            raw["cell_count"],
            raw["batch_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectSceneAnchorBatchObserverError("batch plan item is not canonical")
        return result


def _plan_content(value: "ObjectSceneAnchorBatchObserverPlan") -> dict[str, object]:
    return {
        "schema": OBJECT_SCENE_ANCHOR_BATCH_PLAN_SCHEMA,
        "vocabulary": value.vocabulary.to_data(),
        "vocabulary_digest": value.vocabulary_digest,
        "batches": [item.to_data() for item in value.batches],
        "view_count": value.view_count,
        "view_presentation_count": value.view_presentation_count,
        "repeated_view_presentation_count": (
            value.view_presentation_count - value.view_count
        ),
        "catalog_count": value.catalog_count,
        "cell_count": value.cell_count,
        "physical_call_count": value.physical_call_count,
        "maximum_views_per_batch": OBJECT_SCENE_ANCHOR_BATCH_MAX_VIEWS,
        "maximum_named_images_per_batch": OBJECT_SCENE_ANCHOR_BATCH_MAX_IMAGES,
        "maximum_cells_per_batch": OBJECT_SCENE_ANCHOR_BATCH_MAX_CELLS,
        "partition_policy": _PARTITION_POLICY,
        "repeated_view_presentations_allowed": True,
        "preparation_split_allowed": False,
        "silent_pruning_allowed": False,
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class ObjectSceneAnchorBatchObserverPlan:
    vocabulary: ObjectSceneAnchorObserverVocabulary
    vocabulary_digest: str
    batches: tuple[ObjectSceneAnchorBatchPlanItem, ...]
    view_count: int
    view_presentation_count: int
    catalog_count: int
    cell_count: int
    physical_call_count: int
    plan_digest: str

    def __post_init__(self) -> None:
        if type(self.vocabulary) is not ObjectSceneAnchorObserverVocabulary:
            raise TypeError("batch plan vocabulary has the wrong type")
        _digest(self.vocabulary_digest, "batch vocabulary digest")
        if self.vocabulary.vocabulary_digest != self.vocabulary_digest:
            raise ObjectSceneAnchorBatchObserverError("batch vocabulary differs")
        for label, item in (
            ("plan view count", self.view_count),
            ("plan view presentation count", self.view_presentation_count),
            ("plan catalog count", self.catalog_count),
            ("plan cell count", self.cell_count),
            ("plan physical call count", self.physical_call_count),
        ):
            _integer(item, label, minimum=1)
        if (
            type(self.batches) is not tuple
            or not self.batches
            or any(type(item) is not ObjectSceneAnchorBatchPlanItem for item in self.batches)
            or tuple(item.batch_index for item in self.batches)
            != tuple(range(len(self.batches)))
        ):
            raise ObjectSceneAnchorBatchObserverError("batch inventory differs")
        all_subjects = tuple(subject for batch in self.batches for subject in batch.subjects)
        all_catalogs = tuple(
            catalog for subject in all_subjects for catalog in subject.catalogs
        )
        view_digests = tuple(item.view_digest for item in all_subjects)
        catalog_keys = tuple(
            (subject.view_digest, catalog.catalog_digest)
            for batch in self.batches
            for subject in batch.subjects
            for catalog in subject.catalogs
        )
        if (
            view_digests != tuple(sorted(view_digests))
            or catalog_keys != tuple(sorted(catalog_keys))
            or any(
                catalog.preparation.vocabulary != self.vocabulary
                for catalog in all_catalogs
            )
            or len({item.preparation_digest for item in all_catalogs})
            != len(all_catalogs)
            or self.view_count != len(set(view_digests))
            or self.view_presentation_count != len(all_subjects)
            or self.view_presentation_count < self.view_count
            or self.catalog_count != len(all_catalogs)
            or self.cell_count != sum(item.cell_count for item in self.batches)
            or self.cell_count
            != sum(item.preparation.cell_count for item in all_catalogs)
            or self.physical_call_count != 2 * len(self.batches)
        ):
            raise ObjectSceneAnchorBatchObserverError(
                "batch plan is not a complete deterministic partition"
            )
        for batch_index, (batch, following) in enumerate(
            zip(self.batches, self.batches[1:], strict=False)
        ):
            next_subject = following.subjects[0]
            next_preparation = next_subject.catalogs[0].preparation
            adds_view = next_subject.view_digest != batch.subjects[-1].view_digest
            view_limit_blocks = (
                adds_view
                and batch.view_count == OBJECT_SCENE_ANCHOR_BATCH_MAX_VIEWS
            )
            cell_limit_blocks = (
                batch.cell_count + next_preparation.cell_count
                > OBJECT_SCENE_ANCHOR_BATCH_MAX_CELLS
            )
            if not (view_limit_blocks or cell_limit_blocks):
                raise ObjectSceneAnchorBatchObserverError(
                    f"nonterminal batch {batch_index} is not a maximal greedy prefix"
                )
        _digest(self.plan_digest, "batch plan digest")
        if self.plan_digest != canonical_digest(_plan_content(self)):
            raise ObjectSceneAnchorBatchObserverError("batch plan digest differs")

    @property
    def preparations(self) -> tuple[ObjectSceneAnchorObserverPreparation, ...]:
        return tuple(
            catalog.preparation
            for batch in self.batches
            for subject in batch.subjects
            for catalog in subject.catalogs
        )

    def to_data(self) -> dict[str, object]:
        return {**_plan_content(self), "plan_digest": self.plan_digest}

    @classmethod
    def from_data(cls, value: object) -> "ObjectSceneAnchorBatchObserverPlan":
        raw = _exact_fields(
            value,
            {
                "schema",
                "vocabulary",
                "vocabulary_digest",
                "batches",
                "view_count",
                "view_presentation_count",
                "repeated_view_presentation_count",
                "catalog_count",
                "cell_count",
                "physical_call_count",
                "maximum_views_per_batch",
                "maximum_named_images_per_batch",
                "maximum_cells_per_batch",
                "partition_policy",
                "repeated_view_presentations_allowed",
                "preparation_split_allowed",
                "silent_pruning_allowed",
                *_authority_data(),
                "plan_digest",
            },
            "batch observer plan",
        )
        if (
            raw["schema"] != OBJECT_SCENE_ANCHOR_BATCH_PLAN_SCHEMA
            or raw["maximum_views_per_batch"] != OBJECT_SCENE_ANCHOR_BATCH_MAX_VIEWS
            or raw["maximum_named_images_per_batch"] != OBJECT_SCENE_ANCHOR_BATCH_MAX_IMAGES
            or raw["maximum_cells_per_batch"] != OBJECT_SCENE_ANCHOR_BATCH_MAX_CELLS
            or raw["partition_policy"] != _PARTITION_POLICY
            or raw["repeated_view_presentations_allowed"] is not True
            or raw["preparation_split_allowed"] is not False
            or raw["silent_pruning_allowed"] is not False
            or type(raw["view_count"]) is not int
            or type(raw["view_presentation_count"]) is not int
            or raw["repeated_view_presentation_count"]
            != raw["view_presentation_count"] - raw["view_count"]
            or type(raw["catalog_count"]) is not int
            or type(raw["cell_count"]) is not int
            or any(raw[key] != item for key, item in _authority_data().items())
            or not isinstance(raw["vocabulary"], Mapping)
            or not isinstance(raw["batches"], list)
        ):
            raise ObjectSceneAnchorBatchObserverError("batch observer plan policy differs")
        result = cls(
            ObjectSceneAnchorObserverVocabulary.from_data(raw["vocabulary"]),
            raw["vocabulary_digest"],
            tuple(ObjectSceneAnchorBatchPlanItem.from_data(item) for item in raw["batches"]),
            raw["view_count"],
            raw["view_presentation_count"],
            raw["catalog_count"],
            raw["cell_count"],
            raw["physical_call_count"],
            raw["plan_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectSceneAnchorBatchObserverError("batch plan is not canonical")
        return result


def freeze_object_scene_anchor_batch_observer_plan(
    inputs: Sequence[ObjectSceneAnchorBatchObserverInput],
) -> ObjectSceneAnchorBatchObserverPlan:
    """Freeze an order-independent, exact, finite transport partition."""

    if isinstance(inputs, (str, bytes)) or not isinstance(inputs, Sequence):
        raise TypeError("batch inputs must be a sequence")
    frozen = tuple(inputs)
    if not frozen or any(type(item) is not ObjectSceneAnchorBatchObserverInput for item in frozen):
        raise ObjectSceneAnchorBatchObserverError("batch inputs differ")
    if len({item.preparation.preparation_digest for item in frozen}) != len(frozen):
        raise ObjectSceneAnchorBatchObserverError("batch inputs repeat a preparation")
    vocabulary = frozen[0].preparation.vocabulary
    if any(item.preparation.vocabulary != vocabulary for item in frozen):
        raise ObjectSceneAnchorBatchObserverError(
            "batch inputs do not share one exact global vocabulary"
        )
    by_view: dict[str, list[ObjectSceneAnchorObserverPreparation]] = {}
    bytes_by_view: dict[str, tuple[bytes, bytes]] = {}
    for item in frozen:
        view = item.view_digest
        previous = bytes_by_view.get(view)
        current = (item.crop_png_bytes, item.atlas_png_bytes)
        if previous is not None and previous != current:
            raise ObjectSceneAnchorBatchObserverError(
                "one view digest is paired with different PNG bytes"
            )
        bytes_by_view[view] = current
        by_view.setdefault(view, []).append(item.preparation)
    ordered_preparations = tuple(
        preparation
        for view_digest in sorted(by_view)
        for preparation in sorted(
            by_view[view_digest], key=lambda item: item.catalog_digest
        )
    )
    for preparation in ordered_preparations:
        if preparation.cell_count > OBJECT_SCENE_ANCHOR_BATCH_MAX_CELLS:
            raise ObjectSceneAnchorBatchCapacityGap(
                preparation.preparation_digest,
                preparation.cell_count,
                OBJECT_SCENE_ANCHOR_BATCH_MAX_CELLS,
            )
    packed: list[tuple[tuple[ObjectSceneAnchorObserverPreparation, ...], ...]] = []
    current: list[list[ObjectSceneAnchorObserverPreparation]] = []
    current_cells = 0
    for preparation in ordered_preparations:
        view_digest = _view_digest(preparation)
        same_view = bool(current) and _view_digest(current[-1][0]) == view_digest
        cell_limit_reached = (
            bool(current)
            and current_cells + preparation.cell_count
            > OBJECT_SCENE_ANCHOR_BATCH_MAX_CELLS
        )
        view_limit_reached = (
            bool(current)
            and not same_view
            and len(current) == OBJECT_SCENE_ANCHOR_BATCH_MAX_VIEWS
        )
        if cell_limit_reached or view_limit_reached:
            packed.append(tuple(tuple(group) for group in current))
            current = []
            current_cells = 0
            same_view = False
        if same_view:
            current[-1].append(preparation)
        else:
            current.append([preparation])
        current_cells += preparation.cell_count
    if current:
        packed.append(tuple(tuple(group) for group in current))
    batches = tuple(
        ObjectSceneAnchorBatchPlanItem.create(
            index, grouped_preparations,
        )
        for index, grouped_preparations in enumerate(packed)
    )
    view_presentation_count = sum(
        item.view_count for item in batches
    )
    cell_count = sum(
        preparation.cell_count for preparation in ordered_preparations
    )
    values = {
        "vocabulary": vocabulary,
        "vocabulary_digest": vocabulary.vocabulary_digest,
        "batches": batches,
        "view_count": len(by_view),
        "view_presentation_count": view_presentation_count,
        "catalog_count": len(frozen),
        "cell_count": cell_count,
        "physical_call_count": 2 * len(batches),
    }
    provisional = object.__new__(ObjectSceneAnchorBatchObserverPlan)
    for name, item in values.items():
        object.__setattr__(provisional, name, item)
    return ObjectSceneAnchorBatchObserverPlan(
        **values, plan_digest=canonical_digest(_plan_content(provisional))
    )


def _expected_records(
    batch: ObjectSceneAnchorBatchPlanItem,
    vocabulary: ObjectSceneAnchorObserverVocabulary,
) -> tuple[tuple[
    ObjectSceneAnchorBatchSubjectPlan,
    ObjectSceneAnchorBatchCatalogPlan,
    object,
    object,
    ObjectSceneAnchorObserverVocabularyEntry,
], ...]:
    """Return subject/catalog/binding/locator/witness in canonical order."""

    return tuple(
        (subject, catalog, binding, locator, witness)
        for subject in batch.subjects
        for catalog in subject.catalogs
        for binding, locator in zip(
            catalog.preparation.catalog.bindings,
            catalog.preparation.locators,
            strict=True,
        )
        for witness in vocabulary.entries
    )


def _record_key(record: tuple[object, ...]) -> tuple[str, str, str, str, str]:
    subject, catalog, _binding, locator, witness = record
    assert isinstance(subject, ObjectSceneAnchorBatchSubjectPlan)
    assert isinstance(catalog, ObjectSceneAnchorBatchCatalogPlan)
    return (
        subject.subject_alias,
        catalog.catalog_alias,
        locator.binding_digest,
        witness.witness_digest,
        catalog.preparation_digest,
    )


def _cell_key(cell: ObjectSceneAnchorObserverCell) -> tuple[str, str, str]:
    return (
        cell.locator.catalog_digest,
        cell.locator.binding_digest,
        cell.witness.witness_digest,
    )


def _rectangle_digest(
    batch: ObjectSceneAnchorBatchPlanItem,
    vocabulary: ObjectSceneAnchorObserverVocabulary,
) -> str:
    return canonical_digest(
        {
            "schema": "gkm.object-scene-anchor-batch-observer-rectangle.v1",
            "batch_digest": batch.batch_digest,
            "vocabulary_digest": vocabulary.vocabulary_digest,
            "keys": [list(_record_key(item)) for item in _expected_records(batch, vocabulary)],
        }
    )


def object_scene_anchor_batch_observer_prompt(
    batch: ObjectSceneAnchorBatchPlanItem,
    vocabulary: ObjectSceneAnchorObserverVocabulary,
) -> str:
    if type(batch) is not ObjectSceneAnchorBatchPlanItem:
        raise TypeError("batch must be an exact batch plan item")
    if type(vocabulary) is not ObjectSceneAnchorObserverVocabulary:
        raise TypeError("vocabulary must be an exact observer vocabulary")
    image_rows = "\n".join(
        f"- {subject.subject_alias}: drawing={subject.object_image_name}; "
        f"anchor_map={subject.anchor_image_name}"
        for subject in batch.subjects
    )
    catalog_rows: list[str] = []
    for subject in batch.subjects:
        for catalog in subject.catalogs:
            catalog_rows.append(
                f"- {subject.subject_alias}/{catalog.catalog_alias}:"
            )
            catalog_rows.extend(
                (
                    f"  - {locator.binding_id}: kind={locator.anchor_kind}; "
                    f"anchor={locator.anchor_id}; atlas_tile={locator.atlas_slot_id}; "
                    f"zero_based_row={locator.atlas_row_index}; "
                    f"zero_based_column={locator.atlas_column_index}"
                )
                for locator in catalog.preparation.locators
            )
    witness_rows = "\n".join(
        f"- {item.witness_id} [{item.kind}]: {item.statement}"
        for item in vocabulary.entries
    )
    prompt = (
        "Act as a literal visual observer. Each neutral subject has one exact "
        "full-style isolated drawing and one grayscale anchor map. Anchor-map "
        "tiles form a five-column grid read from the top row downward. A "
        "subject may have several neutral catalogs; these reuse the same two "
        "images and declare different exact anchors. Judge every declared "
        "subject, catalog, binding, and affirmative visible statement in the "
        "listed subject-major, catalog-major, binding-major, witness-major "
        "order. Judge only the exact highlighted anchor inside its own drawing; "
        "never combine subjects, catalogs, or anchors. Return P only when the "
        "statement clearly holds, A only when the anchor is clearly resolved "
        "and visible evidence clearly conflicts with the statement, and I "
        "whenever localization, geometry, markings, or image quality leave the "
        "judgment unresolved. A failed fit, unreadable view, or uncertainty "
        "must never become A. Use exactly one finite reason code allowed for "
        "the state. Return every cell exactly once with no omissions, additions, "
        "or reordering. All identifiers are neutral.\n\nSubjects and images:\n"
        f"This batch contains exactly {batch.cell_count} cells; the fixed "
        f"batch limit is {OBJECT_SCENE_ANCHOR_BATCH_MAX_CELLS}.\n\n"
        f"{image_rows}\n\nDeclared catalogs and bindings:\n"
        f"{'\n'.join(catalog_rows)}\n\nAffirmative visible statements:\n"
        f"{witness_rows}"
    )
    if len(prompt.encode("utf-8")) > 131_072:
        raise ObjectSceneAnchorBatchObserverError("batch prompt exceeds fixed bound")
    names = tuple(
        name
        for subject in batch.subjects
        for name in (subject.object_image_name, subject.anchor_image_name)
    )
    hidden = tuple(
        value
        for subject in batch.subjects
        for catalog in subject.catalogs
        for value in (
            catalog.preparation.panel_manifest_digest,
            catalog.preparation.decision_manifest_digest,
            catalog.preparation.catalog_digest,
            catalog.preparation.vocabulary_digest,
            catalog.preparation.preparation_digest,
        )
    )
    _scene_runtime._assert_model_visible_boundary(
        prompt,
        {},
        names,
        hidden_values=hidden,
        allowed_visual_words=("side", "path"),
    )
    if _ROLE_WORD.search(prompt) is not None:
        raise ObjectSceneAnchorBatchObserverError(
            "batch prompt discloses an experimental role"
        )
    return prompt


def object_scene_anchor_batch_observer_output_schema(
    batch: ObjectSceneAnchorBatchPlanItem,
    vocabulary: ObjectSceneAnchorObserverVocabulary,
) -> dict[str, object]:
    if type(batch) is not ObjectSceneAnchorBatchPlanItem:
        raise TypeError("batch must be an exact batch plan item")
    if type(vocabulary) is not ObjectSceneAnchorObserverVocabulary:
        raise TypeError("vocabulary must be an exact observer vocabulary")
    cell_properties: dict[str, object] = {
        "subject_id": {
            "type": "string",
            "enum": [item.subject_alias for item in batch.subjects],
        },
        "catalog_id": {
            "type": "string",
            "enum": sorted(
                {catalog.catalog_alias for subject in batch.subjects for catalog in subject.catalogs}
            ),
        },
        "binding_id": {
            "type": "string",
            "enum": sorted(
                {
                    locator.binding_id
                    for subject in batch.subjects
                    for catalog in subject.catalogs
                    for locator in catalog.preparation.locators
                }
            ),
        },
        "witness_id": {
            "type": "string",
            "enum": [item.witness_id for item in vocabulary.entries],
        },
        "state": {"type": "string", "enum": ["P", "A", "I"]},
        "reason_code": {
            "type": "string",
            "enum": sorted(set().union(*_REASONS.values())),
        },
    }
    schema: dict[str, object] = {
        "type": "object",
        "properties": {
            "cells": {
                "type": "array",
                "description": (
                    f"Exactly {batch.cell_count} cells in the listed order; "
                    f"the fixed batch limit is "
                    f"{OBJECT_SCENE_ANCHOR_BATCH_MAX_CELLS}."
                ),
                "items": {
                    "type": "object",
                    "properties": cell_properties,
                    "required": list(cell_properties),
                    "additionalProperties": False,
                },
            }
        },
        "required": ["cells"],
        "additionalProperties": False,
    }
    validate_codex_strict_output_schema(schema)
    if len(canonical_json(schema)) > 65_536:
        raise ObjectSceneAnchorBatchObserverError("batch schema exceeds fixed bound")
    return schema


def object_scene_anchor_batch_observer_protocol_digest() -> str:
    return canonical_digest(
        {
            "schema": "gkm.object-scene-anchor-batch-observer-protocol.v2",
            "protocol_id": OBJECT_SCENE_ANCHOR_BATCH_PROTOCOL_ID,
            "source_digest": object_scene_anchor_batch_observer_source_digest(),
            "transport_source_digest": _scene_runtime.prototype_scene_transport_source_digest(),
            "receipt_schema": CODEX_RECEIPT_SCHEMA,
            "input_digest_schema": NAMED_IMAGE_INPUT_DIGEST_SCHEMA,
            "maximum_views_per_batch": OBJECT_SCENE_ANCHOR_BATCH_MAX_VIEWS,
            "maximum_images_per_batch": OBJECT_SCENE_ANCHOR_BATCH_MAX_IMAGES,
            "maximum_cells_per_batch": OBJECT_SCENE_ANCHOR_BATCH_MAX_CELLS,
            "partition_policy": _PARTITION_POLICY,
            "repeated_view_presentations_allowed": True,
            "preparation_split_allowed": False,
            "silent_pruning_allowed": False,
            "pass_count_per_batch": 2,
            "pass_merge": "P+P=P;A+A=A;any-E=E;all-other-pairs=I",
            "failure_semantics": "failed-or-uncertain-vision-never-A",
            **_authority_data(),
        }
    )


def _runtime_identity_digest(
    *,
    model: str,
    reasoning_effort: str,
    expected_launcher_digest: str,
    cloud_policy_cache_binding: str,
    model_catalog_digest: str,
    no_tools_attestation_digest: str,
) -> str:
    _digest(expected_launcher_digest, "runtime launcher digest")
    _digest(model_catalog_digest, "runtime model catalog digest")
    _digest(no_tools_attestation_digest, "runtime no-tools digest")
    if cloud_policy_cache_binding != "absent":
        _address(cloud_policy_cache_binding, "runtime cloud policy cache binding")
    return canonical_digest(
        {
            "schema": "gkm.object-scene-anchor-batch-observer-runtime.v1",
            "model_digest": _model_digest(model, reasoning_effort),
            "expected_launcher_digest": expected_launcher_digest,
            "cloud_policy_cache_binding": cloud_policy_cache_binding,
            "model_catalog_digest": model_catalog_digest,
            "no_tools_attestation_digest": no_tools_attestation_digest,
            "protocol_digest": object_scene_anchor_batch_observer_protocol_digest(),
            **_authority_data(),
        }
    )


def _payload_cells(
    payload: Mapping[str, Any],
    batch: ObjectSceneAnchorBatchPlanItem,
    vocabulary: ObjectSceneAnchorObserverVocabulary,
) -> tuple[ObjectSceneAnchorObserverCell, ...]:
    raw = _exact_fields(payload, {"cells"}, "batch observer payload")
    rows = raw["cells"]
    expected = _expected_records(batch, vocabulary)
    if not isinstance(rows, list) or len(rows) != len(expected):
        raise ObjectSceneAnchorBatchObserverPayloadError(
            "payload does not exhaust the subject/catalog/binding/witness rectangle"
        )
    cells: list[ObjectSceneAnchorObserverCell] = []
    for index, (item, record) in enumerate(zip(rows, expected, strict=True)):
        subject, catalog, binding, locator, witness = record
        cell = _exact_fields(
            item,
            {"subject_id", "catalog_id", "binding_id", "witness_id", "state", "reason_code"},
            f"batch payload cell {index}",
        )
        if (
            cell["subject_id"] != subject.subject_alias
            or cell["catalog_id"] != catalog.catalog_alias
            or cell["binding_id"] != binding.binding_id
            or cell["witness_id"] != witness.witness_id
            or cell["state"] not in _MODEL_STATES
            or cell["reason_code"] not in _REASONS.get(cell["state"], frozenset())
        ):
            raise ObjectSceneAnchorBatchObserverPayloadError(
                "payload cell order, state, or finite reason differs"
            )
        disposition = {
            "P": Disposition.PRESENT,
            "A": Disposition.CERTIFIED_ABSENT,
            "I": Disposition.INDETERMINATE,
        }[cell["state"]]
        cells.append(
            ObjectSceneAnchorObserverCell.create(
                cell_phase="pass",
                binding=binding,
                locator=locator,
                witness=witness,
                disposition=disposition,
                reason_code=cell["reason_code"],
            )
        )
    return tuple(cells)


def _error_cells(
    batch: ObjectSceneAnchorBatchPlanItem,
    vocabulary: ObjectSceneAnchorObserverVocabulary,
    reason_code: str,
) -> tuple[ObjectSceneAnchorObserverCell, ...]:
    if reason_code not in ("payload_rejected", "transport_failed"):
        raise ObjectSceneAnchorBatchObserverError("batch pass error reason differs")
    return tuple(
        ObjectSceneAnchorObserverCell.create(
            cell_phase="pass",
            binding=binding,
            locator=locator,
            witness=witness,
            disposition=Disposition.ERROR,
            reason_code=reason_code,
        )
        for _subject, _catalog, binding, locator, witness in _expected_records(batch, vocabulary)
    )


def _pass_content(value: "ObjectSceneAnchorBatchObserverPassArtifact") -> dict[str, object]:
    return {
        "schema": OBJECT_SCENE_ANCHOR_BATCH_PASS_SCHEMA,
        "batch_digest": value.batch_digest,
        "rectangle_digest": value.rectangle_digest,
        "expected_cell_count": value.expected_cell_count,
        "pass_id": value.pass_id,
        "pass_index": value.pass_index,
        "prompt_digest": value.prompt_digest,
        "output_schema_digest": value.output_schema_digest,
        "protocol_digest": value.protocol_digest,
        "source_digest": value.source_digest,
        "transport_source_digest": value.transport_source_digest,
        "model": value.model,
        "reasoning_effort": value.reasoning_effort,
        "model_digest": value.model_digest,
        "expected_launcher_digest": value.expected_launcher_digest,
        "cloud_policy_cache_binding": value.cloud_policy_cache_binding,
        "model_catalog_digest": value.model_catalog_digest,
        "no_tools_attestation_digest": value.no_tools_attestation_digest,
        "runtime_identity_digest": value.runtime_identity_digest,
        "presentation": [item.to_data() for item in value.presentation],
        "physical_call_count": value.physical_call_count,
        "status": value.status,
        "model_payload": value.model_payload,
        "receipt": _receipt_data(value.receipt),
        "failure_code": value.failure_code,
        "failure_type": value.failure_type,
        "cells": [item.to_data() for item in value.cells],
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class ObjectSceneAnchorBatchObserverPassArtifact:
    batch_digest: str
    rectangle_digest: str
    expected_cell_count: int
    pass_id: str
    pass_index: int
    prompt_digest: str
    output_schema_digest: str
    protocol_digest: str
    source_digest: str
    transport_source_digest: str
    model: str
    reasoning_effort: str
    model_digest: str
    expected_launcher_digest: str
    cloud_policy_cache_binding: str
    model_catalog_digest: str
    no_tools_attestation_digest: str
    runtime_identity_digest: str
    presentation: tuple[PrototypeImageIdentity, ...]
    physical_call_count: int
    status: str
    model_payload: Mapping[str, Any] | None
    receipt: CodexReceipt | None
    failure_code: str | None
    failure_type: str | None
    cells: tuple[ObjectSceneAnchorObserverCell, ...]
    pass_digest: str

    def __post_init__(self) -> None:
        for label, item in (
            ("batch digest", self.batch_digest),
            ("rectangle digest", self.rectangle_digest),
            ("prompt digest", self.prompt_digest),
            ("output schema digest", self.output_schema_digest),
            ("protocol digest", self.protocol_digest),
            ("source digest", self.source_digest),
            ("transport source digest", self.transport_source_digest),
            ("model digest", self.model_digest),
            ("launcher digest", self.expected_launcher_digest),
            ("model catalog digest", self.model_catalog_digest),
            ("no-tools digest", self.no_tools_attestation_digest),
            ("runtime identity digest", self.runtime_identity_digest),
            ("pass digest", self.pass_digest),
        ):
            _digest(item, label)
        _integer(self.expected_cell_count, "expected cell count", minimum=1)
        if (
            not isinstance(self.pass_id, str)
            or _PASS_ID.fullmatch(self.pass_id) is None
            or self.pass_index not in (0, 1)
            or self.pass_id != f"pass_{self.pass_index:02d}"
        ):
            raise ObjectSceneAnchorBatchObserverError("batch pass position differs")
        if self.cloud_policy_cache_binding != "absent":
            _address(self.cloud_policy_cache_binding, "cloud policy cache binding")
        if (
            self.protocol_digest != object_scene_anchor_batch_observer_protocol_digest()
            or self.source_digest != object_scene_anchor_batch_observer_source_digest()
            or self.transport_source_digest != _scene_runtime.prototype_scene_transport_source_digest()
            or self.model_digest != _model_digest(self.model, self.reasoning_effort)
            or self.runtime_identity_digest
            != _runtime_identity_digest(
                model=self.model,
                reasoning_effort=self.reasoning_effort,
                expected_launcher_digest=self.expected_launcher_digest,
                cloud_policy_cache_binding=self.cloud_policy_cache_binding,
                model_catalog_digest=self.model_catalog_digest,
                no_tools_attestation_digest=self.no_tools_attestation_digest,
            )
        ):
            raise ObjectSceneAnchorBatchObserverError("batch pass protocol differs")
        if (
            type(self.presentation) is not tuple
            or not 2 <= len(self.presentation) <= OBJECT_SCENE_ANCHOR_BATCH_MAX_IMAGES
            or len(self.presentation) % 2
            or any(type(item) is not PrototypeImageIdentity for item in self.presentation)
            or len({item.name for item in self.presentation}) != len(self.presentation)
            or self.physical_call_count != 1
            or self.status not in ("success", "parser_error", "transport_error")
            or type(self.cells) is not tuple
            or len(self.cells) != self.expected_cell_count
            or any(type(item) is not ObjectSceneAnchorObserverCell for item in self.cells)
            or any(item.cell_phase != "pass" for item in self.cells)
        ):
            raise ObjectSceneAnchorBatchObserverError("batch pass structure differs")
        if self.model_payload is not None:
            object.__setattr__(self, "model_payload", _canonical_payload(self.model_payload))
        if self.status == "success":
            if (
                self.model_payload is None
                or not isinstance(self.receipt, CodexReceipt)
                or self.failure_code is not None
                or self.failure_type is not None
                or any(item.disposition is Disposition.ERROR for item in self.cells)
            ):
                raise ObjectSceneAnchorBatchObserverError("successful batch pass differs")
        elif self.status == "parser_error":
            if (
                self.model_payload is None
                or not isinstance(self.receipt, CodexReceipt)
                or self.failure_code != "payload_rejected"
                or not isinstance(self.failure_type, str)
                or any(
                    item.disposition is not Disposition.ERROR
                    or item.reason_code != "payload_rejected"
                    for item in self.cells
                )
            ):
                raise ObjectSceneAnchorBatchObserverError("parser-error batch pass differs")
        else:
            if (
                self.model_payload is not None
                or self.receipt is not None
                or self.failure_code != "transport_failed"
                or not isinstance(self.failure_type, str)
                or any(
                    item.disposition is not Disposition.ERROR
                    or item.reason_code != "transport_failed"
                    for item in self.cells
                )
            ):
                raise ObjectSceneAnchorBatchObserverError("transport-error batch pass differs")
        if self.receipt is not None:
            view = [item.to_data() for item in self.presentation]
            expected_set = "sha256:" + canonical_digest(
                {"schema": NAMED_IMAGE_INPUT_DIGEST_SCHEMA, "images": view}
            )
            if (
                self.receipt.prompt_digest != self.prompt_digest
                or self.receipt.output_schema_digest != self.output_schema_digest
                or self.receipt.structured_output_digest
                != canonical_digest(dict(self.model_payload or {}))
                or self.receipt.panel_view_digest != canonical_digest(view)
                or self.receipt.panel_set_digest != expected_set
                or self.receipt.requested_model != self.model
                or self.receipt.requested_reasoning_effort != self.reasoning_effort
                or self.receipt.codex_launcher_digest != self.expected_launcher_digest
                or self.receipt.cloud_config_bundle_cache_binding != self.cloud_policy_cache_binding
                or self.receipt.model_catalog_digest != self.model_catalog_digest
                or self.receipt.tool_surface_attestation_digest != self.no_tools_attestation_digest
            ):
                raise ObjectSceneAnchorBatchObserverError("batch pass receipt binding differs")
        if self.pass_digest != canonical_digest(_pass_content(self)):
            raise ObjectSceneAnchorBatchObserverError("batch pass digest differs")

    def to_data(self) -> dict[str, object]:
        return {**_pass_content(self), "pass_digest": self.pass_digest}

    @classmethod
    def from_data(cls, value: object) -> "ObjectSceneAnchorBatchObserverPassArtifact":
        raw = _exact_fields(
            value,
            {
                "schema", "batch_digest", "rectangle_digest", "expected_cell_count",
                "pass_id", "pass_index", "prompt_digest", "output_schema_digest",
                "protocol_digest", "source_digest", "transport_source_digest",
                "model", "reasoning_effort", "model_digest", "expected_launcher_digest",
                "cloud_policy_cache_binding", "model_catalog_digest",
                "no_tools_attestation_digest", "runtime_identity_digest", "presentation",
                "physical_call_count", "status", "model_payload", "receipt",
                "failure_code", "failure_type", "cells", *_authority_data(), "pass_digest",
            },
            "batch observer pass",
        )
        if (
            raw["schema"] != OBJECT_SCENE_ANCHOR_BATCH_PASS_SCHEMA
            or any(raw[key] != item for key, item in _authority_data().items())
            or not isinstance(raw["presentation"], list)
            or not isinstance(raw["cells"], list)
        ):
            raise ObjectSceneAnchorBatchObserverError("batch pass policy differs")
        result = cls(
            batch_digest=raw["batch_digest"],
            rectangle_digest=raw["rectangle_digest"],
            expected_cell_count=raw["expected_cell_count"],
            pass_id=raw["pass_id"],
            pass_index=raw["pass_index"],
            prompt_digest=raw["prompt_digest"],
            output_schema_digest=raw["output_schema_digest"],
            protocol_digest=raw["protocol_digest"],
            source_digest=raw["source_digest"],
            transport_source_digest=raw["transport_source_digest"],
            model=raw["model"],
            reasoning_effort=raw["reasoning_effort"],
            model_digest=raw["model_digest"],
            expected_launcher_digest=raw["expected_launcher_digest"],
            cloud_policy_cache_binding=raw["cloud_policy_cache_binding"],
            model_catalog_digest=raw["model_catalog_digest"],
            no_tools_attestation_digest=raw["no_tools_attestation_digest"],
            runtime_identity_digest=raw["runtime_identity_digest"],
            presentation=tuple(PrototypeImageIdentity.from_data(item) for item in raw["presentation"]),
            physical_call_count=raw["physical_call_count"],
            status=raw["status"],
            model_payload=None if raw["model_payload"] is None else _canonical_payload(raw["model_payload"]),
            receipt=_receipt_from_data(raw["receipt"]),
            failure_code=raw["failure_code"],
            failure_type=raw["failure_type"],
            cells=tuple(ObjectSceneAnchorObserverCell.from_data(item) for item in raw["cells"]),
            pass_digest=raw["pass_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectSceneAnchorBatchObserverError("batch pass is not canonical")
        return result


def _seal_pass(
    *,
    pass_index: int,
    batch: ObjectSceneAnchorBatchPlanItem,
    vocabulary: ObjectSceneAnchorObserverVocabulary,
    presentation: tuple[PrototypeImageIdentity, ...],
    model: str,
    reasoning_effort: str,
    expected_launcher_digest: str,
    cloud_policy_cache_binding: str,
    model_catalog_digest: str,
    no_tools_attestation_digest: str,
    status: str,
    payload: Mapping[str, Any] | None,
    receipt: CodexReceipt | None,
    failure_code: str | None,
    failure_type: str | None,
    cells: tuple[ObjectSceneAnchorObserverCell, ...],
) -> ObjectSceneAnchorBatchObserverPassArtifact:
    prompt = object_scene_anchor_batch_observer_prompt(batch, vocabulary)
    schema = object_scene_anchor_batch_observer_output_schema(batch, vocabulary)
    runtime = _runtime_identity_digest(
        model=model,
        reasoning_effort=reasoning_effort,
        expected_launcher_digest=expected_launcher_digest,
        cloud_policy_cache_binding=cloud_policy_cache_binding,
        model_catalog_digest=model_catalog_digest,
        no_tools_attestation_digest=no_tools_attestation_digest,
    )
    values = {
        "batch_digest": batch.batch_digest,
        "rectangle_digest": _rectangle_digest(batch, vocabulary),
        "expected_cell_count": batch.cell_count,
        "pass_id": f"pass_{pass_index:02d}",
        "pass_index": pass_index,
        "prompt_digest": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "output_schema_digest": canonical_digest(schema),
        "protocol_digest": object_scene_anchor_batch_observer_protocol_digest(),
        "source_digest": object_scene_anchor_batch_observer_source_digest(),
        "transport_source_digest": _scene_runtime.prototype_scene_transport_source_digest(),
        "model": model,
        "reasoning_effort": reasoning_effort,
        "model_digest": _model_digest(model, reasoning_effort),
        "expected_launcher_digest": expected_launcher_digest,
        "cloud_policy_cache_binding": cloud_policy_cache_binding,
        "model_catalog_digest": model_catalog_digest,
        "no_tools_attestation_digest": no_tools_attestation_digest,
        "runtime_identity_digest": runtime,
        "presentation": presentation,
        "physical_call_count": 1,
        "status": status,
        "model_payload": None if payload is None else _canonical_payload(payload),
        "receipt": receipt,
        "failure_code": failure_code,
        "failure_type": failure_type,
        "cells": cells,
    }
    provisional = object.__new__(ObjectSceneAnchorBatchObserverPassArtifact)
    for name, item in values.items():
        object.__setattr__(provisional, name, item)
    return ObjectSceneAnchorBatchObserverPassArtifact(
        **values, pass_digest=canonical_digest(_pass_content(provisional))
    )


def _cells_match_rectangle(
    cells: tuple[ObjectSceneAnchorObserverCell, ...],
    batch: ObjectSceneAnchorBatchPlanItem,
    vocabulary: ObjectSceneAnchorObserverVocabulary,
    *,
    phase: str,
) -> bool:
    expected = _expected_records(batch, vocabulary)
    return (
        len(cells) == len(expected)
        and all(item.cell_phase == phase for item in cells)
        and all(
            cell.locator == locator and cell.witness == witness
            for cell, (_subject, _catalog, _binding, locator, witness) in zip(
                cells, expected, strict=True
            )
        )
        and tuple(_cell_key(item) for item in cells)
        == tuple(
            (
                catalog.catalog_digest,
                locator.binding_digest,
                witness.witness_digest,
            )
            for _subject, catalog, _binding, locator, witness in expected
        )
    )


def _merge_pass_cells(
    batch: ObjectSceneAnchorBatchPlanItem,
    vocabulary: ObjectSceneAnchorObserverVocabulary,
    first: ObjectSceneAnchorBatchObserverPassArtifact,
    second: ObjectSceneAnchorBatchObserverPassArtifact,
) -> tuple[ObjectSceneAnchorObserverCell, ...]:
    if not _cells_match_rectangle(first.cells, batch, vocabulary, phase="pass") or not (
        _cells_match_rectangle(second.cells, batch, vocabulary, phase="pass")
    ):
        raise ObjectSceneAnchorBatchObserverError(
            "batch pass does not contain the exact rectangle"
        )
    merged: list[ObjectSceneAnchorObserverCell] = []
    for left, right, record in zip(
        first.cells, second.cells, _expected_records(batch, vocabulary), strict=True
    ):
        if _cell_key(left) != _cell_key(right):
            raise ObjectSceneAnchorBatchObserverError(
                "batch passes do not share exact cell keys"
            )
        _subject, _catalog, binding, locator, witness = record
        disposition, reason = _merge_cell_dispositions(
            left.disposition, right.disposition
        )
        merged.append(
            ObjectSceneAnchorObserverCell.create(
                cell_phase="merged",
                binding=binding,
                locator=locator,
                witness=witness,
                disposition=disposition,
                reason_code=reason,
            )
        )
    return tuple(merged)


def _result_content(value: "ObjectSceneAnchorBatchObserverResult") -> dict[str, object]:
    return {
        "schema": OBJECT_SCENE_ANCHOR_BATCH_RESULT_SCHEMA,
        "batch": value.batch.to_data(),
        "batch_digest": value.batch_digest,
        "vocabulary": value.vocabulary.to_data(),
        "vocabulary_digest": value.vocabulary_digest,
        "physical_call_count": value.physical_call_count,
        "passes": [item.to_data() for item in value.passes],
        "merged_cells": [item.to_data() for item in value.merged_cells],
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class ObjectSceneAnchorBatchObserverResult:
    batch: ObjectSceneAnchorBatchPlanItem
    batch_digest: str
    vocabulary: ObjectSceneAnchorObserverVocabulary
    vocabulary_digest: str
    physical_call_count: int
    passes: tuple[
        ObjectSceneAnchorBatchObserverPassArtifact,
        ObjectSceneAnchorBatchObserverPassArtifact,
    ]
    merged_cells: tuple[ObjectSceneAnchorObserverCell, ...]
    result_digest: str

    def __post_init__(self) -> None:
        if type(self.batch) is not ObjectSceneAnchorBatchPlanItem:
            raise TypeError("batch result plan has the wrong type")
        if type(self.vocabulary) is not ObjectSceneAnchorObserverVocabulary:
            raise TypeError("batch result vocabulary has the wrong type")
        if (
            self.batch.batch_digest != self.batch_digest
            or self.vocabulary.vocabulary_digest != self.vocabulary_digest
            or self.physical_call_count != 2
            or type(self.passes) is not tuple
            or len(self.passes) != 2
            or any(type(item) is not ObjectSceneAnchorBatchObserverPassArtifact for item in self.passes)
            or tuple(item.pass_index for item in self.passes) != (0, 1)
        ):
            raise ObjectSceneAnchorBatchObserverError("batch result structure differs")
        prompt = object_scene_anchor_batch_observer_prompt(self.batch, self.vocabulary)
        schema = object_scene_anchor_batch_observer_output_schema(self.batch, self.vocabulary)
        prompt_digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        schema_digest = canonical_digest(schema)
        rectangle = _rectangle_digest(self.batch, self.vocabulary)
        expected_names = tuple(
            name
            for subject in self.batch.subjects
            for name in (subject.object_image_name, subject.anchor_image_name)
        )
        for item in self.passes:
            if (
                item.batch_digest != self.batch_digest
                or item.rectangle_digest != rectangle
                or item.expected_cell_count != self.batch.cell_count
                or item.prompt_digest != prompt_digest
                or item.output_schema_digest != schema_digest
                or tuple(image.name for image in item.presentation) != expected_names
                or not _cells_match_rectangle(item.cells, self.batch, self.vocabulary, phase="pass")
            ):
                raise ObjectSceneAnchorBatchObserverError(
                    "batch pass differs from frozen plan rectangle"
                )
            if item.status == "success":
                assert item.model_payload is not None
                try:
                    replayed = _payload_cells(item.model_payload, self.batch, self.vocabulary)
                except Exception as exc:
                    raise ObjectSceneAnchorBatchObserverError(
                        "successful batch payload does not replay"
                    ) from exc
                if replayed != item.cells:
                    raise ObjectSceneAnchorBatchObserverError(
                        "batch cells differ from frozen payload"
                    )
            elif item.status == "parser_error":
                assert item.model_payload is not None
                try:
                    _payload_cells(item.model_payload, self.batch, self.vocabulary)
                except Exception as exc:
                    if item.failure_type != _exception_type(exc):
                        raise ObjectSceneAnchorBatchObserverError(
                            "batch parser failure type differs on replay"
                        ) from exc
                else:
                    raise ObjectSceneAnchorBatchObserverError(
                        "parser-error batch payload now parses"
                    )
        if self.passes[0].presentation != self.passes[1].presentation:
            raise ObjectSceneAnchorBatchObserverError(
                "batch pass presentations differ"
            )
        receipts = tuple(item.receipt for item in self.passes)
        if receipts[0] is not None and receipts[1] is not None and (
            receipts[0].receipt_digest == receipts[1].receipt_digest
            or receipts[0].thread_id == receipts[1].thread_id
        ):
            raise ObjectSceneAnchorBatchObserverError(
                "batch passes do not have independent receipts"
            )
        expected_merged = _merge_pass_cells(
            self.batch, self.vocabulary, self.passes[0], self.passes[1]
        )
        if (
            type(self.merged_cells) is not tuple
            or self.merged_cells != expected_merged
            or not _cells_match_rectangle(
                self.merged_cells, self.batch, self.vocabulary, phase="merged"
            )
        ):
            raise ObjectSceneAnchorBatchObserverError(
                "batch merged cells differ from exact same-key merge"
            )
        _digest(self.result_digest, "batch result digest")
        if self.result_digest != canonical_digest(_result_content(self)):
            raise ObjectSceneAnchorBatchObserverError("batch result digest differs")

    def to_data(self) -> dict[str, object]:
        return {**_result_content(self), "result_digest": self.result_digest}

    @classmethod
    def from_data(cls, value: object) -> "ObjectSceneAnchorBatchObserverResult":
        raw = _exact_fields(
            value,
            {
                "schema", "batch", "batch_digest", "vocabulary",
                "vocabulary_digest", "physical_call_count", "passes",
                "merged_cells", *_authority_data(), "result_digest",
            },
            "batch observer result",
        )
        if (
            raw["schema"] != OBJECT_SCENE_ANCHOR_BATCH_RESULT_SCHEMA
            or any(raw[key] != item for key, item in _authority_data().items())
            or not isinstance(raw["batch"], Mapping)
            or not isinstance(raw["vocabulary"], Mapping)
            or not isinstance(raw["passes"], list)
            or not isinstance(raw["merged_cells"], list)
        ):
            raise ObjectSceneAnchorBatchObserverError("batch result policy differs")
        passes = tuple(ObjectSceneAnchorBatchObserverPassArtifact.from_data(item) for item in raw["passes"])
        if len(passes) != 2:
            raise ObjectSceneAnchorBatchObserverError("batch result pass count differs")
        result = cls(
            batch=ObjectSceneAnchorBatchPlanItem.from_data(raw["batch"]),
            batch_digest=raw["batch_digest"],
            vocabulary=ObjectSceneAnchorObserverVocabulary.from_data(raw["vocabulary"]),
            vocabulary_digest=raw["vocabulary_digest"],
            physical_call_count=raw["physical_call_count"],
            passes=(passes[0], passes[1]),
            merged_cells=tuple(ObjectSceneAnchorObserverCell.from_data(item) for item in raw["merged_cells"]),
            result_digest=raw["result_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectSceneAnchorBatchObserverError("batch result is not canonical")
        return result


def _seal_result(
    batch: ObjectSceneAnchorBatchPlanItem,
    vocabulary: ObjectSceneAnchorObserverVocabulary,
    passes: tuple[
        ObjectSceneAnchorBatchObserverPassArtifact,
        ObjectSceneAnchorBatchObserverPassArtifact,
    ],
) -> ObjectSceneAnchorBatchObserverResult:
    values = {
        "batch": batch,
        "batch_digest": batch.batch_digest,
        "vocabulary": vocabulary,
        "vocabulary_digest": vocabulary.vocabulary_digest,
        "physical_call_count": 2,
        "passes": passes,
        "merged_cells": _merge_pass_cells(batch, vocabulary, passes[0], passes[1]),
    }
    provisional = object.__new__(ObjectSceneAnchorBatchObserverResult)
    for name, item in values.items():
        object.__setattr__(provisional, name, item)
    return ObjectSceneAnchorBatchObserverResult(
        **values, result_digest=canonical_digest(_result_content(provisional))
    )


def _artifact_content(value: "ObjectSceneAnchorBatchObserverArtifact") -> dict[str, object]:
    return {
        "schema": OBJECT_SCENE_ANCHOR_BATCH_ARTIFACT_SCHEMA,
        "observation_plan_digest": value.observation_plan_digest,
        "plan": value.plan.to_data(),
        "plan_digest": value.plan_digest,
        "protocol_digest": value.protocol_digest,
        "source_digest": value.source_digest,
        "transport_source_digest": value.transport_source_digest,
        "model": value.model,
        "reasoning_effort": value.reasoning_effort,
        "model_digest": value.model_digest,
        "expected_launcher_digest": value.expected_launcher_digest,
        "cloud_policy_cache_binding": value.cloud_policy_cache_binding,
        "model_catalog_digest": value.model_catalog_digest,
        "no_tools_attestation_digest": value.no_tools_attestation_digest,
        "runtime_identity_digest": value.runtime_identity_digest,
        "physical_call_count": value.physical_call_count,
        "results": [item.to_data() for item in value.results],
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class ObjectSceneAnchorBatchObserverArtifact:
    observation_plan_digest: str
    plan: ObjectSceneAnchorBatchObserverPlan
    plan_digest: str
    protocol_digest: str
    source_digest: str
    transport_source_digest: str
    model: str
    reasoning_effort: str
    model_digest: str
    expected_launcher_digest: str
    cloud_policy_cache_binding: str
    model_catalog_digest: str
    no_tools_attestation_digest: str
    runtime_identity_digest: str
    physical_call_count: int
    results: tuple[ObjectSceneAnchorBatchObserverResult, ...]
    artifact_digest: str

    def __post_init__(self) -> None:
        _address(self.observation_plan_digest, "observation plan digest")
        if type(self.plan) is not ObjectSceneAnchorBatchObserverPlan:
            raise TypeError("batch artifact plan has the wrong type")
        for label, item in (
            ("plan digest", self.plan_digest), ("protocol digest", self.protocol_digest),
            ("source digest", self.source_digest), ("transport source digest", self.transport_source_digest),
            ("model digest", self.model_digest), ("launcher digest", self.expected_launcher_digest),
            ("model catalog digest", self.model_catalog_digest),
            ("no-tools digest", self.no_tools_attestation_digest),
            ("runtime identity", self.runtime_identity_digest), ("artifact digest", self.artifact_digest),
        ):
            _digest(item, label)
        if self.cloud_policy_cache_binding != "absent":
            _address(self.cloud_policy_cache_binding, "cloud policy cache binding")
        if (
            self.plan.plan_digest != self.plan_digest
            or self.protocol_digest != object_scene_anchor_batch_observer_protocol_digest()
            or self.source_digest != object_scene_anchor_batch_observer_source_digest()
            or self.transport_source_digest != _scene_runtime.prototype_scene_transport_source_digest()
            or self.model_digest != _model_digest(self.model, self.reasoning_effort)
            or self.runtime_identity_digest
            != _runtime_identity_digest(
                model=self.model, reasoning_effort=self.reasoning_effort,
                expected_launcher_digest=self.expected_launcher_digest,
                cloud_policy_cache_binding=self.cloud_policy_cache_binding,
                model_catalog_digest=self.model_catalog_digest,
                no_tools_attestation_digest=self.no_tools_attestation_digest,
            )
            or self.physical_call_count != self.plan.physical_call_count
            or type(self.results) is not tuple
            or tuple(item.batch for item in self.results) != self.plan.batches
            or any(item.vocabulary != self.plan.vocabulary for item in self.results)
        ):
            raise ObjectSceneAnchorBatchObserverError("batch artifact structure differs")
        for result in self.results:
            for item in result.passes:
                if (
                    item.model != self.model
                    or item.reasoning_effort != self.reasoning_effort
                    or item.expected_launcher_digest != self.expected_launcher_digest
                    or item.cloud_policy_cache_binding != self.cloud_policy_cache_binding
                    or item.model_catalog_digest != self.model_catalog_digest
                    or item.no_tools_attestation_digest != self.no_tools_attestation_digest
                    or item.runtime_identity_digest != self.runtime_identity_digest
                ):
                    raise ObjectSceneAnchorBatchObserverError("batch pass runtime differs")
        if self.artifact_digest != canonical_digest(_artifact_content(self)):
            raise ObjectSceneAnchorBatchObserverError("batch artifact digest differs")

    def to_data(self) -> dict[str, object]:
        return {**_artifact_content(self), "artifact_digest": self.artifact_digest}

    @classmethod
    def from_data(cls, value: object) -> "ObjectSceneAnchorBatchObserverArtifact":
        raw = _exact_fields(
            value,
            {
                "schema", "observation_plan_digest", "plan", "plan_digest",
                "protocol_digest", "source_digest", "transport_source_digest", "model",
                "reasoning_effort", "model_digest", "expected_launcher_digest",
                "cloud_policy_cache_binding", "model_catalog_digest",
                "no_tools_attestation_digest", "runtime_identity_digest",
                "physical_call_count", "results", *_authority_data(), "artifact_digest",
            },
            "batch observer artifact",
        )
        if (
            raw["schema"] != OBJECT_SCENE_ANCHOR_BATCH_ARTIFACT_SCHEMA
            or any(raw[key] != item for key, item in _authority_data().items())
            or not isinstance(raw["plan"], Mapping)
            or not isinstance(raw["results"], list)
        ):
            raise ObjectSceneAnchorBatchObserverError("batch artifact policy differs")
        result = cls(
            observation_plan_digest=raw["observation_plan_digest"],
            plan=ObjectSceneAnchorBatchObserverPlan.from_data(raw["plan"]),
            plan_digest=raw["plan_digest"], protocol_digest=raw["protocol_digest"],
            source_digest=raw["source_digest"], transport_source_digest=raw["transport_source_digest"],
            model=raw["model"], reasoning_effort=raw["reasoning_effort"],
            model_digest=raw["model_digest"], expected_launcher_digest=raw["expected_launcher_digest"],
            cloud_policy_cache_binding=raw["cloud_policy_cache_binding"],
            model_catalog_digest=raw["model_catalog_digest"],
            no_tools_attestation_digest=raw["no_tools_attestation_digest"],
            runtime_identity_digest=raw["runtime_identity_digest"],
            physical_call_count=raw["physical_call_count"],
            results=tuple(ObjectSceneAnchorBatchObserverResult.from_data(item) for item in raw["results"]),
            artifact_digest=raw["artifact_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectSceneAnchorBatchObserverError("batch artifact is not canonical")
        return result


def _runtime_inputs_by_view(
    inputs: Sequence[ObjectSceneAnchorBatchObserverInput],
    plan: ObjectSceneAnchorBatchObserverPlan,
) -> dict[str, ObjectSceneAnchorBatchObserverInput]:
    rebuilt = freeze_object_scene_anchor_batch_observer_plan(inputs)
    if rebuilt != plan:
        raise ObjectSceneAnchorBatchObserverError(
            "runtime inputs differ from frozen batch plan"
        )
    result: dict[str, ObjectSceneAnchorBatchObserverInput] = {}
    for item in inputs:
        result.setdefault(item.view_digest, item)
    if set(result) != {
        subject.view_digest for batch in plan.batches for subject in batch.subjects
    }:
        raise ObjectSceneAnchorBatchObserverError("runtime view inventory differs")
    return result


def _presentation_bytes(
    batch: ObjectSceneAnchorBatchPlanItem,
    by_view: Mapping[str, ObjectSceneAnchorBatchObserverInput],
) -> tuple[tuple[str, bytes], ...]:
    values: list[tuple[str, bytes]] = []
    for subject in batch.subjects:
        source = by_view.get(subject.view_digest)
        if source is None:
            raise ObjectSceneAnchorBatchObserverError("batch view bytes are missing")
        values.extend(
            (
                (subject.object_image_name, source.crop_png_bytes),
                (subject.anchor_image_name, source.atlas_png_bytes),
            )
        )
    return tuple(values)


def observe_object_scene_anchor_batches_twice(
    inputs: Sequence[ObjectSceneAnchorBatchObserverInput],
    *,
    plan: ObjectSceneAnchorBatchObserverPlan,
    expected_plan_digest: str,
    observation_plan_digest: str,
    model: str = DEFAULT_CODEX_MODEL,
    reasoning_effort: str = DEFAULT_REASONING_EFFORT,
    minutes: int = 15,
    verbose: bool = False,
    executable: str = "codex",
    cloud_policy_cache_snapshot: CloudPolicyCacheSnapshot | None = None,
    expected_launcher_digest: str,
    model_catalog_snapshot: CodexModelCatalogSnapshot,
    no_tools_attestation: CodexNoToolsAttestation,
    transport=run_codex_named_images_structured,
) -> ObjectSceneAnchorBatchObserverArtifact:
    """Attempt exactly two independent no-tools calls for every frozen batch."""

    if type(plan) is not ObjectSceneAnchorBatchObserverPlan:
        raise TypeError("plan must be an exact batch observer plan")
    frozen_plan = ObjectSceneAnchorBatchObserverPlan.from_data(plan.to_data())
    if frozen_plan.plan_digest != _digest(expected_plan_digest, "expected batch plan digest"):
        raise ObjectSceneAnchorBatchObserverError("batch plan differs from commitment")
    context = _address(observation_plan_digest, "observation plan digest")
    by_view = _runtime_inputs_by_view(inputs, frozen_plan)
    if not callable(transport):
        raise TypeError("transport must be callable")
    launcher = _digest(expected_launcher_digest, "expected launcher digest")
    policy_binding = _scene_runtime._policy_cache_binding(cloud_policy_cache_snapshot)
    model_catalog_digest, no_tools_digest = _scene_runtime._validate_no_tools_runtime(
        model_catalog_snapshot=model_catalog_snapshot,
        no_tools_attestation=no_tools_attestation,
        expected_launcher_digest=launcher,
        cloud_policy_cache_binding=policy_binding,
    )
    results: list[ObjectSceneAnchorBatchObserverResult] = []
    for batch in frozen_plan.batches:
        presentation_bytes = _presentation_bytes(batch, by_view)
        presentation = _scene_runtime._image_identities(presentation_bytes)
        prompt = object_scene_anchor_batch_observer_prompt(batch, frozen_plan.vocabulary)
        schema = object_scene_anchor_batch_observer_output_schema(batch, frozen_plan.vocabulary)
        passes: list[ObjectSceneAnchorBatchObserverPassArtifact] = []
        for pass_index in range(2):
            try:
                payload, receipt = _scene_runtime._stage_and_call(
                    presentation_bytes,
                    prompt=prompt,
                    schema=schema,
                    model=model,
                    reasoning_effort=reasoning_effort,
                    minutes=minutes,
                    verbose=verbose,
                    executable=executable,
                    cloud_policy_cache_snapshot=cloud_policy_cache_snapshot,
                    expected_launcher_digest=launcher,
                    model_catalog_snapshot=model_catalog_snapshot,
                    no_tools_attestation=no_tools_attestation,
                    transport=transport,
                )
            except Exception as exc:
                passes.append(
                    _seal_pass(
                        pass_index=pass_index, batch=batch, vocabulary=frozen_plan.vocabulary,
                        presentation=presentation, model=model, reasoning_effort=reasoning_effort,
                        expected_launcher_digest=launcher, cloud_policy_cache_binding=policy_binding,
                        model_catalog_digest=model_catalog_digest,
                        no_tools_attestation_digest=no_tools_digest, status="transport_error",
                        payload=None, receipt=None, failure_code="transport_failed",
                        failure_type=_exception_type(exc),
                        cells=_error_cells(batch, frozen_plan.vocabulary, "transport_failed"),
                    )
                )
                continue
            try:
                cells = _payload_cells(payload, batch, frozen_plan.vocabulary)
            except Exception as exc:
                passes.append(
                    _seal_pass(
                        pass_index=pass_index, batch=batch, vocabulary=frozen_plan.vocabulary,
                        presentation=presentation, model=model, reasoning_effort=reasoning_effort,
                        expected_launcher_digest=launcher, cloud_policy_cache_binding=policy_binding,
                        model_catalog_digest=model_catalog_digest,
                        no_tools_attestation_digest=no_tools_digest, status="parser_error",
                        payload=payload, receipt=receipt, failure_code="payload_rejected",
                        failure_type=_exception_type(exc),
                        cells=_error_cells(batch, frozen_plan.vocabulary, "payload_rejected"),
                    )
                )
                continue
            passes.append(
                _seal_pass(
                    pass_index=pass_index, batch=batch, vocabulary=frozen_plan.vocabulary,
                    presentation=presentation, model=model, reasoning_effort=reasoning_effort,
                    expected_launcher_digest=launcher, cloud_policy_cache_binding=policy_binding,
                    model_catalog_digest=model_catalog_digest,
                    no_tools_attestation_digest=no_tools_digest, status="success",
                    payload=payload, receipt=receipt, failure_code=None, failure_type=None,
                    cells=cells,
                )
            )
        if len(passes) != 2:  # pragma: no cover - fixed loop above.
            raise ObjectSceneAnchorBatchObserverError("batch did not attempt two passes")
        results.append(
            _seal_result(batch, frozen_plan.vocabulary, (passes[0], passes[1]))
        )
    runtime = _runtime_identity_digest(
        model=model, reasoning_effort=reasoning_effort,
        expected_launcher_digest=launcher, cloud_policy_cache_binding=policy_binding,
        model_catalog_digest=model_catalog_digest,
        no_tools_attestation_digest=no_tools_digest,
    )
    values = {
        "observation_plan_digest": context,
        "plan": frozen_plan,
        "plan_digest": frozen_plan.plan_digest,
        "protocol_digest": object_scene_anchor_batch_observer_protocol_digest(),
        "source_digest": object_scene_anchor_batch_observer_source_digest(),
        "transport_source_digest": _scene_runtime.prototype_scene_transport_source_digest(),
        "model": model,
        "reasoning_effort": reasoning_effort,
        "model_digest": _model_digest(model, reasoning_effort),
        "expected_launcher_digest": launcher,
        "cloud_policy_cache_binding": policy_binding,
        "model_catalog_digest": model_catalog_digest,
        "no_tools_attestation_digest": no_tools_digest,
        "runtime_identity_digest": runtime,
        "physical_call_count": 2 * len(results),
        "results": tuple(results),
    }
    provisional = object.__new__(ObjectSceneAnchorBatchObserverArtifact)
    for name, item in values.items():
        object.__setattr__(provisional, name, item)
    return ObjectSceneAnchorBatchObserverArtifact(
        **values, artifact_digest=canonical_digest(_artifact_content(provisional))
    )


def verify_object_scene_anchor_batch_observer_artifact(
    artifact: ObjectSceneAnchorBatchObserverArtifact,
    inputs: Sequence[ObjectSceneAnchorBatchObserverInput],
    *,
    expected_artifact_digest: str,
    expected_plan_digest: str,
    expected_observation_plan_digest: str,
    expected_runtime_identity_digest: str | None = None,
) -> ObjectSceneAnchorBatchObserverArtifact:
    """Cold replay exact PNG identities, payloads, receipts, rectangles, and merges."""

    if type(artifact) is not ObjectSceneAnchorBatchObserverArtifact:
        raise TypeError("artifact must be an exact batch observer artifact")
    restored = ObjectSceneAnchorBatchObserverArtifact.from_data(artifact.to_data())
    if restored.artifact_digest != _digest(expected_artifact_digest, "expected artifact digest"):
        raise ObjectSceneAnchorBatchObserverError("batch artifact differs from commitment")
    if restored.plan_digest != _digest(expected_plan_digest, "expected plan digest"):
        raise ObjectSceneAnchorBatchObserverError("batch plan differs from commitment")
    if restored.observation_plan_digest != _address(
        expected_observation_plan_digest, "expected observation plan digest"
    ):
        raise ObjectSceneAnchorBatchObserverError("observation plan differs from commitment")
    if expected_runtime_identity_digest is not None and restored.runtime_identity_digest != _digest(
        expected_runtime_identity_digest, "expected runtime identity digest"
    ):
        raise ObjectSceneAnchorBatchObserverError("batch runtime differs from commitment")
    by_view = _runtime_inputs_by_view(inputs, restored.plan)
    for batch, result in zip(restored.plan.batches, restored.results, strict=True):
        presentation_bytes = _presentation_bytes(batch, by_view)
        identities = _scene_runtime._image_identities(presentation_bytes)
        if identities != result.passes[0].presentation or identities != result.passes[1].presentation:
            raise ObjectSceneAnchorBatchObserverError(
                "batch presentation differs from exact cold replay"
            )
        prompt = object_scene_anchor_batch_observer_prompt(batch, restored.plan.vocabulary)
        schema = object_scene_anchor_batch_observer_output_schema(batch, restored.plan.vocabulary)
        with tempfile.TemporaryDirectory(prefix="bongard-anchor-batch-replay-") as raw:
            directory = Path(raw)
            paths: list[str] = []
            for name, data in presentation_bytes:
                target = directory / name
                target.write_bytes(data)
                paths.append(str(target.resolve()))
            names = tuple(name for name, _data in presentation_bytes)
            for item in result.passes:
                if item.receipt is None:
                    continue
                assert item.model_payload is not None
                validate_codex_named_image_receipt(
                    item.receipt, prompt, tuple(paths), names, schema, dict(item.model_payload)
                )
            for path, (_name, expected) in zip(paths, presentation_bytes, strict=True):
                if Path(path).read_bytes() != expected:
                    raise ObjectSceneAnchorBatchObserverError(
                        "batch replay presentation changed"
                    )
    return restored


def object_scene_anchor_object_matrices_from_batch_artifact(
    artifact: ObjectSceneAnchorBatchObserverArtifact,
    language: object,
) -> tuple[object, ...]:
    """Project every present catalog to the version-space matrix type.

    Hard A/I/E catalogs never create observer preparations and therefore never
    enter this adapter; the version builder constructs their zero-row matrices
    directly without a model call.
    """

    if type(artifact) is not ObjectSceneAnchorBatchObserverArtifact:
        raise TypeError("artifact must be an exact batch observer artifact")
    restored = ObjectSceneAnchorBatchObserverArtifact.from_data(artifact.to_data())
    from bongard.object_scene_anchor_version_space import (
        ObjectSceneAnchorObjectWitnessMatrix,
        ObjectSceneAnchorPredicateLanguage,
    )
    if type(language) is not ObjectSceneAnchorPredicateLanguage:
        raise TypeError("language must be an exact anchor predicate language")
    if language.vocabulary != restored.plan.vocabulary:
        raise ObjectSceneAnchorBatchObserverError(
            "batch vocabulary differs from frozen predicate language"
        )
    matrices: list[object] = []
    for result in restored.results:
        offset = 0
        for subject in result.batch.subjects:
            for catalog in subject.catalogs:
                count = catalog.preparation.cell_count
                cells = result.merged_cells[offset : offset + count]
                offset += count
                matrices.append(
                    ObjectSceneAnchorObjectWitnessMatrix.create(
                        catalog=catalog.preparation.catalog,
                        vocabulary=restored.plan.vocabulary,
                        cells=tuple(item.binding_cell for item in cells),
                    )
                )
        if offset != len(result.merged_cells):
            raise ObjectSceneAnchorBatchObserverError("batch matrix projection differs")
    return tuple(matrices)


__all__ = (
    "OBJECT_SCENE_ANCHOR_BATCH_ARTIFACT_SCHEMA",
    "OBJECT_SCENE_ANCHOR_BATCH_MAX_CELLS",
    "OBJECT_SCENE_ANCHOR_BATCH_MAX_IMAGES",
    "OBJECT_SCENE_ANCHOR_BATCH_MAX_VIEWS",
    "OBJECT_SCENE_ANCHOR_BATCH_PASS_SCHEMA",
    "OBJECT_SCENE_ANCHOR_BATCH_PLAN_SCHEMA",
    "OBJECT_SCENE_ANCHOR_BATCH_PROTOCOL_ID",
    "OBJECT_SCENE_ANCHOR_BATCH_RESULT_SCHEMA",
    "ObjectSceneAnchorBatchCatalogPlan",
    "ObjectSceneAnchorBatchCapacityGap",
    "ObjectSceneAnchorBatchObserverArtifact",
    "ObjectSceneAnchorBatchObserverError",
    "ObjectSceneAnchorBatchObserverInput",
    "ObjectSceneAnchorBatchObserverPassArtifact",
    "ObjectSceneAnchorBatchObserverPayloadError",
    "ObjectSceneAnchorBatchObserverPlan",
    "ObjectSceneAnchorBatchObserverResult",
    "ObjectSceneAnchorBatchPlanItem",
    "ObjectSceneAnchorBatchSubjectPlan",
    "freeze_object_scene_anchor_batch_observer_plan",
    "object_scene_anchor_batch_observer_output_schema",
    "object_scene_anchor_batch_observer_prompt",
    "object_scene_anchor_batch_observer_protocol_digest",
    "object_scene_anchor_batch_observer_source_digest",
    "object_scene_anchor_object_matrices_from_batch_artifact",
    "observe_object_scene_anchor_batches_twice",
    "verify_object_scene_anchor_batch_observer_artifact",
)
