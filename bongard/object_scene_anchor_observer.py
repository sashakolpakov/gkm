"""Role-blind two-pass visual observations on exact selected anchors.

One call boundary contains one exact full-style object crop and its deterministic
selected-anchor atlas.  The model sees only neutral anchor locators and a frozen
union of affirmative visible statements.  It never sees Bongard roles, a
proposed classifier, or downstream Boolean structure.

Python verifies the crop against the full catalog entry, then immediately drops
the entry-only provenance.  Persisted identities contain only the decoded crop,
the decision-only panel/object manifests, the exhaustive binding catalog, the
atlas presentation, and the neutral vocabulary.  Two independent no-tools
turns are attempted.  Their cells are merged on the exact
``binding_digest x witness_digest`` key before any downstream expression is
constructed.
"""

from __future__ import annotations

from bongard.runtime_source_snapshot import capture_loaded_source, verify_loaded_source


_LOADED_SOURCE_SHA256 = capture_loaded_source(__name__, __file__)

from dataclasses import dataclass
from io import BytesIO
import hashlib
import json
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

from bongard import prototype_object_hypotheses as _hypotheses
from bongard import prototype_scene_observer as _scene_runtime
from bongard.canonical import canonical_digest, canonical_json
from bongard.evidence import Disposition
from bongard.object_scene_anchor_atlas import (
    ObjectSceneAnchorAtlas,
    ObjectSceneAnchorAtlasSlot,
    verify_object_scene_anchor_atlas,
)
from bongard.object_scene_anchor_bindings import (
    ObjectSceneAnchorBindingCatalog,
    ObjectSceneAnchorWitnessCell,
    ObjectSceneAnchorWitnessSpec,
    ObjectSceneResolvedAnchorBinding,
)
from bongard.object_scene_anchor_catalog import ObjectSceneAnchorCatalogEntry
from bongard.object_scene_anchor_crop import (
    OBJECT_SCENE_ANCHOR_CROP_RENDERER_ID,
    verify_object_scene_anchor_object_crop,
)
from bongard.object_scene_anchor_panel_manifest import (
    ObjectSceneAnchorPanelDecisionManifest,
)
from bongard.object_scene_visual_frontend import ObjectSceneProposalInventory
from bongard.prototype_scene_observer import (
    CloudPolicyCacheSnapshot,
    CodexModelCatalogSnapshot,
    CodexNoToolsAttestation,
    PrototypeImageIdentity,
)
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


OBJECT_SCENE_ANCHOR_OBSERVER_VOCABULARY_ENTRY_SCHEMA = (
    "gkm.object-scene-anchor-observer-vocabulary-entry.v1"
)
OBJECT_SCENE_ANCHOR_OBSERVER_VOCABULARY_SCHEMA = (
    "gkm.object-scene-anchor-observer-vocabulary.v1"
)
OBJECT_SCENE_ANCHOR_OBSERVER_LOCATOR_SCHEMA = (
    "gkm.object-scene-anchor-observer-binding-locator.v1"
)
OBJECT_SCENE_ANCHOR_OBSERVER_CELL_SCHEMA = (
    "gkm.object-scene-anchor-observer-cell.v1"
)
OBJECT_SCENE_ANCHOR_OBSERVER_PREPARATION_SCHEMA = (
    "gkm.object-scene-anchor-observer-preparation.v1"
)
OBJECT_SCENE_ANCHOR_OBSERVER_PASS_SCHEMA = (
    "gkm.object-scene-anchor-observer-pass.v1"
)
OBJECT_SCENE_ANCHOR_OBSERVER_ARTIFACT_SCHEMA = (
    "gkm.object-scene-anchor-observer-artifact.v1"
)
OBJECT_SCENE_ANCHOR_OBSERVER_PROTOCOL_ID = (
    "bongard.object-scene-anchor-observer/role-blind-two-pass-rectangle-v1"
)

OBJECT_SCENE_ANCHOR_OBSERVER_MAX_WITNESSES = 32
OBJECT_SCENE_ANCHOR_OBSERVER_MAX_CELLS = 17 * 32
OBJECT_SCENE_ANCHOR_OBSERVER_IMAGE_NAMES = ("object.png", "anchors.png")

_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_ADDRESS = re.compile(r"sha256:[0-9a-f]{64}\Z")
_OBJECT_ID = re.compile(r"object_[0-9]{4}\Z")
_BINDING_ID = re.compile(r"binding_[0-9]{3}\Z")
_WITNESS_ID = re.compile(r"witness_[0-9]{2}\Z")
_PASS_ID = re.compile(r"pass_0[01]\Z")
_PRINTABLE = re.compile(r"[ -~]+\Z")
_ROLE_WORD = re.compile(
    r"\b(?:target|foil|candidate|formula|query|answer|class|label|group)\b",
    re.IGNORECASE,
)
_NEGATIVE_WORD = re.compile(
    r"\b(?:not|no|without|absent|absence|lack|lacks|neither|nor)\b",
    re.IGNORECASE,
)

_WITNESS_KINDS = frozenset(
    ("shape_appearance", "marking_pattern", "spatial_relation", "part_topology")
)
_MODEL_STATES = frozenset(("P", "A", "I"))
_MODEL_REASON_BY_STATE = {
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
_PASS_ERROR_REASONS = frozenset(("payload_rejected", "transport_failed"))
_MERGED_REASON_BY_DISPOSITION = {
    Disposition.PRESENT: frozenset(("two_pass_visible_match",)),
    Disposition.CERTIFIED_ABSENT: frozenset(("two_pass_visible_mismatch",)),
    Disposition.INDETERMINATE: frozenset(
        ("two_pass_disagreement", "two_pass_indeterminate")
    ),
    Disposition.ERROR: frozenset(("one_or_both_pass_error",)),
}


class ObjectSceneAnchorObserverError(ValueError):
    """An observer input, payload, artifact, or replay is not canonical."""


class ObjectSceneAnchorObserverPayloadError(ObjectSceneAnchorObserverError):
    """A receipted model payload violates the finite observer grammar."""


def _authority_data() -> dict[str, object]:
    return {
        "predicate_authority_id": PYTHON_PREDICATE_AUTHORITY_ID,
        "python_is_canonical_authority": True,
        "role_blind_model_boundary": True,
        "comparison_roles_model_visible": False,
        "logical_expression_model_visible": False,
        "polarity_reversal_allowed": False,
        "uncertain_or_failed_vision_counts_as_absence": False,
        "two_independent_passes_required": True,
        "merge_occurs_before_downstream_logic": True,
        "extraction_provenance_persisted": False,
        "decision_projection_only": True,
    }


def _exact_fields(
    value: object, expected: set[str] | frozenset[str], label: str
) -> Mapping[str, Any]:
    if (
        not isinstance(value, Mapping)
        or any(not isinstance(key, str) for key in value)
        or set(value) != set(expected)
    ):
        raise ObjectSceneAnchorObserverError(f"{label} fields differ")
    return value


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise ObjectSceneAnchorObserverError(
            f"{label} must be a lowercase SHA-256"
        )
    return value


def _address(value: object, label: str) -> str:
    if not isinstance(value, str) or _ADDRESS.fullmatch(value) is None:
        raise ObjectSceneAnchorObserverError(
            f"{label} must be a sha256: address"
        )
    return value


def _integer(value: object, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ObjectSceneAnchorObserverError(
            f"{label} must be an integer at least {minimum}"
        )
    return value


def _bounded_statement(value: object) -> str:
    if (
        not isinstance(value, str)
        or not 3 <= len(value) <= 240
        or value != value.strip()
        or "  " in value
        or _PRINTABLE.fullmatch(value) is None
        or _ROLE_WORD.search(value) is not None
        or _NEGATIVE_WORD.search(value) is not None
    ):
        raise ObjectSceneAnchorObserverError(
            "witness statement violates the bounded visible-text grammar"
        )
    return value


def _canonical_payload(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) for key in value
    ):
        raise ObjectSceneAnchorObserverError("observer payload must be an object")
    try:
        decoded = json.loads(canonical_json(dict(value)).decode("utf-8"))
    except (TypeError, ValueError, UnicodeError, json.JSONDecodeError) as exc:
        raise ObjectSceneAnchorObserverError(
            "observer payload is not canonical finite JSON"
        ) from exc
    if not isinstance(decoded, dict):
        raise ObjectSceneAnchorObserverError("observer payload must be an object")
    return decoded


def _receipt_data(value: CodexReceipt | None) -> object:
    return None if value is None else value.to_dict()


def _receipt_from_data(value: object) -> CodexReceipt | None:
    if value is None:
        return None
    try:
        receipt = _scene_runtime._receipt_from_data(value)
    except Exception as exc:
        raise ObjectSceneAnchorObserverError("observer receipt is invalid") from exc
    if not isinstance(receipt, CodexReceipt):
        raise ObjectSceneAnchorObserverError("observer receipt has the wrong type")
    return receipt


def _disposition_from_value(value: object) -> Disposition:
    try:
        return Disposition(value)
    except (TypeError, ValueError) as exc:
        raise ObjectSceneAnchorObserverError("observer disposition differs") from exc


def object_scene_anchor_observer_source_digest() -> str:
    return verify_loaded_source(__name__, expected_source_sha256=_LOADED_SOURCE_SHA256)


def _witness_semantic_content(kind: str, statement: str) -> dict[str, object]:
    """Mirror the card witness's ID-free semantic identity.

    The cards module validates this exact digest at the freeze boundary.  The
    local projection remains independently replayable after card-local aliases
    have been discarded.
    """

    return {
        "schema": "gkm.object-scene-anchor-card-witness.v1",
        "kind": kind,
        "statement": statement,
    }


def _vocabulary_entry_content(
    value: "ObjectSceneAnchorObserverVocabularyEntry",
) -> dict[str, object]:
    return {
        "schema": OBJECT_SCENE_ANCHOR_OBSERVER_VOCABULARY_ENTRY_SCHEMA,
        "witness_id": value.witness_id,
        "kind": value.kind,
        "statement": value.statement,
        "witness_digest": value.witness_digest,
        **_authority_data(),
    }


@dataclass(frozen=True, order=True, slots=True)
class ObjectSceneAnchorObserverVocabularyEntry:
    """One globally re-aliased, affirmative, visible witness."""

    witness_id: str
    kind: str
    statement: str
    witness_digest: str
    vocabulary_record_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.witness_id, str) or _WITNESS_ID.fullmatch(
            self.witness_id
        ) is None:
            raise ObjectSceneAnchorObserverError("observer witness ID differs")
        if self.kind not in _WITNESS_KINDS:
            raise ObjectSceneAnchorObserverError("observer witness kind differs")
        _bounded_statement(self.statement)
        _digest(self.witness_digest, "observer witness digest")
        if self.witness_digest != canonical_digest(
            _witness_semantic_content(self.kind, self.statement)
        ):
            raise ObjectSceneAnchorObserverError(
                "observer witness semantic digest differs"
            )
        _digest(
            self.vocabulary_record_digest, "observer vocabulary record digest"
        )
        if self.vocabulary_record_digest != canonical_digest(
            _vocabulary_entry_content(self)
        ):
            raise ObjectSceneAnchorObserverError(
                "observer vocabulary entry digest differs"
            )

    @classmethod
    def create(
        cls,
        witness_id: str,
        kind: str,
        statement: str,
        witness_digest: str | None = None,
    ) -> "ObjectSceneAnchorObserverVocabularyEntry":
        semantic_digest = witness_digest or canonical_digest(
            _witness_semantic_content(kind, statement)
        )
        values = {
            "witness_id": witness_id,
            "kind": kind,
            "statement": statement,
            "witness_digest": semantic_digest,
        }
        provisional = object.__new__(cls)
        for name, item in values.items():
            object.__setattr__(provisional, name, item)
        return cls(
            **values,
            vocabulary_record_digest=canonical_digest(
                _vocabulary_entry_content(provisional)
            ),
        )

    @property
    def binding_witness_spec(self) -> ObjectSceneAnchorWitnessSpec:
        return ObjectSceneAnchorWitnessSpec(self.witness_id, self.witness_digest)

    def to_data(self) -> dict[str, object]:
        return {
            **_vocabulary_entry_content(self),
            "vocabulary_record_digest": self.vocabulary_record_digest,
        }

    @classmethod
    def from_data(
        cls, value: object
    ) -> "ObjectSceneAnchorObserverVocabularyEntry":
        raw = _exact_fields(
            value,
            {
                "schema",
                "witness_id",
                "kind",
                "statement",
                "witness_digest",
                *_authority_data(),
                "vocabulary_record_digest",
            },
            "observer vocabulary entry",
        )
        if (
            raw["schema"]
            != OBJECT_SCENE_ANCHOR_OBSERVER_VOCABULARY_ENTRY_SCHEMA
            or any(raw[key] != item for key, item in _authority_data().items())
        ):
            raise ObjectSceneAnchorObserverError(
                "observer vocabulary entry policy differs"
            )
        result = cls(
            raw["witness_id"],
            raw["kind"],
            raw["statement"],
            raw["witness_digest"],
            raw["vocabulary_record_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectSceneAnchorObserverError(
                "observer vocabulary entry is not canonical"
            )
        return result


def _vocabulary_content(
    value: "ObjectSceneAnchorObserverVocabulary",
) -> dict[str, object]:
    return {
        "schema": OBJECT_SCENE_ANCHOR_OBSERVER_VOCABULARY_SCHEMA,
        "ordering": "semantic-digest-ascending",
        "entries": [item.to_data() for item in value.entries],
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class ObjectSceneAnchorObserverVocabulary:
    entries: tuple[ObjectSceneAnchorObserverVocabularyEntry, ...]
    vocabulary_digest: str

    def __post_init__(self) -> None:
        if (
            type(self.entries) is not tuple
            or not self.entries
            or len(self.entries) > OBJECT_SCENE_ANCHOR_OBSERVER_MAX_WITNESSES
            or any(
                type(item) is not ObjectSceneAnchorObserverVocabularyEntry
                for item in self.entries
            )
            or tuple(item.witness_id for item in self.entries)
            != tuple(f"witness_{index:02d}" for index in range(len(self.entries)))
            or tuple(item.witness_digest for item in self.entries)
            != tuple(sorted(item.witness_digest for item in self.entries))
            or len({item.witness_digest for item in self.entries})
            != len(self.entries)
        ):
            raise ObjectSceneAnchorObserverError(
                "observer vocabulary is not a complete digest-sorted union"
            )
        _digest(self.vocabulary_digest, "observer vocabulary digest")
        if self.vocabulary_digest != canonical_digest(_vocabulary_content(self)):
            raise ObjectSceneAnchorObserverError("observer vocabulary digest differs")

    @property
    def binding_witness_specs(self) -> tuple[ObjectSceneAnchorWitnessSpec, ...]:
        return tuple(item.binding_witness_spec for item in self.entries)

    def to_data(self) -> dict[str, object]:
        return {
            **_vocabulary_content(self),
            "vocabulary_digest": self.vocabulary_digest,
        }

    @classmethod
    def from_data(cls, value: object) -> "ObjectSceneAnchorObserverVocabulary":
        raw = _exact_fields(
            value,
            {
                "schema",
                "ordering",
                "entries",
                *_authority_data(),
                "vocabulary_digest",
            },
            "observer vocabulary",
        )
        if (
            raw["schema"] != OBJECT_SCENE_ANCHOR_OBSERVER_VOCABULARY_SCHEMA
            or raw["ordering"] != "semantic-digest-ascending"
            or any(raw[key] != item for key, item in _authority_data().items())
            or not isinstance(raw["entries"], list)
        ):
            raise ObjectSceneAnchorObserverError("observer vocabulary policy differs")
        result = cls(
            tuple(
                ObjectSceneAnchorObserverVocabularyEntry.from_data(item)
                for item in raw["entries"]
            ),
            raw["vocabulary_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectSceneAnchorObserverError(
                "observer vocabulary is not canonical"
            )
        return result


def freeze_object_scene_anchor_observer_vocabulary(
    witnesses: Sequence[object],
) -> ObjectSceneAnchorObserverVocabulary:
    """Freeze a content-deduplicated union of exact card witnesses.

    The import is local so the observer's persisted vocabulary remains
    independently readable.  The freeze boundary itself nevertheless requires
    the exact card witness class and its own cold round trip.
    """

    if isinstance(witnesses, (str, bytes)) or not isinstance(witnesses, Sequence):
        raise TypeError("card witnesses must be a sequence")
    try:
        from bongard.object_scene_anchor_cards import ObjectSceneAnchorCardWitness
    except ImportError as exc:  # pragma: no cover - dependency deployment error.
        raise ObjectSceneAnchorObserverError(
            "anchor card witness type is unavailable"
        ) from exc
    by_digest: dict[str, object] = {}
    for item in witnesses:
        if type(item) is not ObjectSceneAnchorCardWitness:
            raise TypeError("every vocabulary source must be an exact card witness")
        restored = ObjectSceneAnchorCardWitness.from_data(item.to_data())
        if restored != item:
            raise ObjectSceneAnchorObserverError("card witness is not canonical")
        previous = by_digest.get(restored.witness_digest)
        if previous is not None and (
            previous.kind != restored.kind
            or previous.statement != restored.statement
        ):
            raise ObjectSceneAnchorObserverError("card witness digest collision")
        by_digest[restored.witness_digest] = restored
    if not by_digest or len(by_digest) > OBJECT_SCENE_ANCHOR_OBSERVER_MAX_WITNESSES:
        raise ObjectSceneAnchorObserverError(
            "observer vocabulary union must contain one to thirty-two witnesses"
        )
    ordered = sorted(by_digest.values(), key=lambda item: item.witness_digest)
    entries = tuple(
        ObjectSceneAnchorObserverVocabularyEntry.create(
            f"witness_{index:02d}",
            item.kind,
            item.statement,
            item.witness_digest,
        )
        for index, item in enumerate(ordered)
    )
    provisional = object.__new__(ObjectSceneAnchorObserverVocabulary)
    object.__setattr__(provisional, "entries", entries)
    return ObjectSceneAnchorObserverVocabulary(
        entries,
        canonical_digest(_vocabulary_content(provisional)),
    )


def _locator_content(
    value: "ObjectSceneAnchorObserverBindingLocator",
) -> dict[str, object]:
    return {
        "schema": OBJECT_SCENE_ANCHOR_OBSERVER_LOCATOR_SCHEMA,
        "object_id": value.object_id,
        "decision_manifest_digest": value.decision_manifest_digest,
        "catalog_digest": value.catalog_digest,
        "spec_digest": value.spec_digest,
        "binding_id": value.binding_id,
        "binding_digest": value.binding_digest,
        "anchor_kind": value.anchor_kind,
        "anchor_id": value.anchor_id,
        "anchor_digest": value.anchor_digest,
        "selected_graph_digest": value.selected_graph_digest,
        "atlas_slot_id": value.atlas_slot_id,
        "atlas_row_index": value.atlas_row_index,
        "atlas_column_index": value.atlas_column_index,
        "atlas_slot_digest": value.atlas_slot_digest,
        **_authority_data(),
    }


@dataclass(frozen=True, order=True, slots=True)
class ObjectSceneAnchorObserverBindingLocator:
    """Exact bridge from one exhaustive binding to one rendered atlas slot."""

    object_id: str
    decision_manifest_digest: str
    catalog_digest: str
    spec_digest: str
    binding_id: str
    binding_digest: str
    anchor_kind: str
    anchor_id: str
    anchor_digest: str
    selected_graph_digest: str
    atlas_slot_id: str
    atlas_row_index: int
    atlas_column_index: int
    atlas_slot_digest: str
    locator_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.object_id, str) or _OBJECT_ID.fullmatch(
            self.object_id
        ) is None:
            raise ObjectSceneAnchorObserverError("locator object ID differs")
        if not isinstance(self.binding_id, str) or _BINDING_ID.fullmatch(
            self.binding_id
        ) is None:
            raise ObjectSceneAnchorObserverError("locator binding ID differs")
        if self.anchor_kind not in ("entity", "part", "frame"):
            raise ObjectSceneAnchorObserverError("locator anchor kind differs")
        if not isinstance(self.anchor_id, str) or not self.anchor_id:
            raise ObjectSceneAnchorObserverError("locator anchor ID differs")
        if not isinstance(self.atlas_slot_id, str) or not re.fullmatch(
            r"slot-[0-9]{4}", self.atlas_slot_id
        ):
            raise ObjectSceneAnchorObserverError("locator atlas slot ID differs")
        _integer(self.atlas_row_index, "locator atlas row")
        _integer(self.atlas_column_index, "locator atlas column")
        slot_index = int(self.atlas_slot_id.removeprefix("slot-"))
        if (
            self.atlas_row_index != slot_index // 5
            or self.atlas_column_index != slot_index % 5
            or self.atlas_row_index >= 4
            or self.atlas_column_index >= 5
        ):
            raise ObjectSceneAnchorObserverError("locator atlas grid position differs")
        for label, item in (
            ("decision manifest digest", self.decision_manifest_digest),
            ("catalog digest", self.catalog_digest),
            ("spec digest", self.spec_digest),
            ("binding digest", self.binding_digest),
            ("anchor digest", self.anchor_digest),
            ("selected graph digest", self.selected_graph_digest),
            ("atlas slot digest", self.atlas_slot_digest),
            ("locator digest", self.locator_digest),
        ):
            _digest(item, label)
        if self.locator_digest != canonical_digest(_locator_content(self)):
            raise ObjectSceneAnchorObserverError("binding locator digest differs")

    @classmethod
    def create(
        cls,
        *,
        binding: ObjectSceneResolvedAnchorBinding,
        catalog: ObjectSceneAnchorBindingCatalog,
        slot: ObjectSceneAnchorAtlasSlot,
    ) -> "ObjectSceneAnchorObserverBindingLocator":
        if type(binding) is not ObjectSceneResolvedAnchorBinding:
            raise TypeError("binding must be an exact resolved anchor binding")
        if type(catalog) is not ObjectSceneAnchorBindingCatalog:
            raise TypeError("catalog must be an exact anchor binding catalog")
        if type(slot) is not ObjectSceneAnchorAtlasSlot:
            raise TypeError("slot must be an exact anchor atlas slot")
        values = {
            "object_id": binding.object_id,
            "decision_manifest_digest": binding.decision_manifest_digest,
            "catalog_digest": catalog.catalog_digest,
            "spec_digest": binding.spec_digest,
            "binding_id": binding.binding_id,
            "binding_digest": binding.binding_digest,
            "anchor_kind": binding.anchor_kind,
            "anchor_id": binding.anchor_id,
            "anchor_digest": binding.anchor_digest,
            "selected_graph_digest": binding.selected_graph_digest,
            "atlas_slot_id": slot.slot_id,
            "atlas_row_index": slot.row_index,
            "atlas_column_index": slot.column_index,
            "atlas_slot_digest": slot.slot_digest,
        }
        provisional = object.__new__(cls)
        for name, item in values.items():
            object.__setattr__(provisional, name, item)
        return cls(
            **values,
            locator_digest=canonical_digest(_locator_content(provisional)),
        )

    def to_data(self) -> dict[str, object]:
        return {**_locator_content(self), "locator_digest": self.locator_digest}

    @classmethod
    def from_data(
        cls, value: object
    ) -> "ObjectSceneAnchorObserverBindingLocator":
        raw = _exact_fields(
            value,
            {
                "schema",
                "object_id",
                "decision_manifest_digest",
                "catalog_digest",
                "spec_digest",
                "binding_id",
                "binding_digest",
                "anchor_kind",
                "anchor_id",
                "anchor_digest",
                "selected_graph_digest",
                "atlas_slot_id",
                "atlas_row_index",
                "atlas_column_index",
                "atlas_slot_digest",
                *_authority_data(),
                "locator_digest",
            },
            "observer binding locator",
        )
        if (
            raw["schema"] != OBJECT_SCENE_ANCHOR_OBSERVER_LOCATOR_SCHEMA
            or any(raw[key] != item for key, item in _authority_data().items())
        ):
            raise ObjectSceneAnchorObserverError("binding locator policy differs")
        result = cls(
            raw["object_id"],
            raw["decision_manifest_digest"],
            raw["catalog_digest"],
            raw["spec_digest"],
            raw["binding_id"],
            raw["binding_digest"],
            raw["anchor_kind"],
            raw["anchor_id"],
            raw["anchor_digest"],
            raw["selected_graph_digest"],
            raw["atlas_slot_id"],
            raw["atlas_row_index"],
            raw["atlas_column_index"],
            raw["atlas_slot_digest"],
            raw["locator_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectSceneAnchorObserverError(
                "binding locator is not canonical"
            )
        return result


def _atlas_slot_for_binding(
    atlas: ObjectSceneAnchorAtlas,
    binding: ObjectSceneResolvedAnchorBinding,
) -> ObjectSceneAnchorAtlasSlot:
    expected_subject = (
        atlas.object_id if binding.anchor_kind == "entity" else binding.anchor_id
    )
    expected_kind = {
        "entity": "whole_entity",
        "part": None,
        "frame": "cyclic_frame",
    }[binding.anchor_kind]
    matches = tuple(
        slot
        for slot in atlas.slots
        if slot.subject_id == expected_subject
        and (
            expected_kind is None
            and slot.slot_kind in ("part_anchor", "compact_anchor")
            or slot.slot_kind == expected_kind
        )
    )
    if len(matches) != 1:
        raise ObjectSceneAnchorObserverError(
            "binding does not resolve to exactly one atlas slot"
        )
    return matches[0]


def _locators_for(
    catalog: ObjectSceneAnchorBindingCatalog,
    atlas: ObjectSceneAnchorAtlas,
) -> tuple[ObjectSceneAnchorObserverBindingLocator, ...]:
    locators = tuple(
        ObjectSceneAnchorObserverBindingLocator.create(
            binding=binding,
            catalog=catalog,
            slot=_atlas_slot_for_binding(atlas, binding),
        )
        for binding in catalog.bindings
    )
    if tuple(item.binding_id for item in locators) != tuple(
        item.binding_id for item in catalog.bindings
    ):
        raise ObjectSceneAnchorObserverError("binding locator order differs")
    return locators


def object_scene_anchor_observer_prompt(
    locators: Sequence[ObjectSceneAnchorObserverBindingLocator],
    vocabulary: ObjectSceneAnchorObserverVocabulary,
) -> str:
    frozen_locators = tuple(locators)
    if (
        not frozen_locators
        or len(frozen_locators) > 17
        or any(
            type(item) is not ObjectSceneAnchorObserverBindingLocator
            for item in frozen_locators
        )
    ):
        raise ObjectSceneAnchorObserverError("prompt locator inventory differs")
    if type(vocabulary) is not ObjectSceneAnchorObserverVocabulary:
        raise TypeError("vocabulary must be exact observer vocabulary")
    rendered_locators = "\n".join(
        (
            f"- {item.binding_id}: kind={item.anchor_kind}; "
            f"anchor={item.anchor_id}; atlas_tile={item.atlas_slot_id}; "
            f"zero_based_row={item.atlas_row_index}; "
            f"zero_based_column={item.atlas_column_index}"
        )
        for item in frozen_locators
    )
    rendered_witnesses = "\n".join(
        f"- {item.witness_id} [{item.kind}]: {item.statement}"
        for item in vocabulary.entries
    )
    prompt = (
        "Act as a literal visual observer. Inspect object.png, the exact "
        "full-style isolated drawing, together with anchors.png, a grayscale "
        "map of anchors in that same drawing. Its tiles form a five-column grid "
        "read from the top row downward. Each declared binding names one exact "
        "zero-based grid position. Judge every declared binding with every affirmative visible "
        "statement below, in binding-major then witness-major order. Judge only "
        "the exact highlighted anchor inside this one drawing; never combine "
        "different anchors. Return P only when the statement clearly holds, A "
        "only when the anchor is clearly resolved and the visible evidence "
        "clearly conflicts with the statement, and I whenever localization, "
        "geometry, markings, or image quality leave the judgment unresolved. "
        "A failed fit, unreadable view, or uncertainty must never become A. "
        "Use exactly the finite reason code allowed for each state. Return every "
        "cell exactly once with no omissions, additions, or reordering. These "
        "identifiers and statements are neutral; do not infer any comparison "
        "role or downstream answer.\n\nDeclared bindings:\n"
        f"{rendered_locators}\n\nAffirmative visible statements:\n"
        f"{rendered_witnesses}"
    )
    if len(prompt.encode("utf-8")) > 32_768:
        raise ObjectSceneAnchorObserverError("observer prompt exceeds fixed bound")
    return prompt


def object_scene_anchor_observer_output_schema(
    locators: Sequence[ObjectSceneAnchorObserverBindingLocator],
    vocabulary: ObjectSceneAnchorObserverVocabulary,
) -> dict[str, object]:
    binding_ids = [item.binding_id for item in tuple(locators)]
    witness_ids = [item.witness_id for item in vocabulary.entries]
    if not binding_ids or len(binding_ids) > 17 or not witness_ids:
        raise ObjectSceneAnchorObserverError("observer schema inventory differs")
    cell_properties: dict[str, object] = {
        "binding_id": {"type": "string", "enum": binding_ids},
        "witness_id": {"type": "string", "enum": witness_ids},
        "state": {"type": "string", "enum": ["P", "A", "I"]},
        "reason_code": {
            "type": "string",
            "enum": sorted(
                set().union(*_MODEL_REASON_BY_STATE.values())
            ),
        },
    }
    schema: dict[str, object] = {
        "type": "object",
        "properties": {
            "cells": {
                "type": "array",
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
        raise ObjectSceneAnchorObserverError("observer schema exceeds fixed bound")
    return schema


def object_scene_anchor_observer_protocol_digest() -> str:
    return canonical_digest(
        {
            "schema": "gkm.object-scene-anchor-observer-protocol.v1",
            "protocol_id": OBJECT_SCENE_ANCHOR_OBSERVER_PROTOCOL_ID,
            "source_digest": object_scene_anchor_observer_source_digest(),
            "transport_source_digest": (
                _scene_runtime.prototype_scene_transport_source_digest()
            ),
            "receipt_schema": CODEX_RECEIPT_SCHEMA,
            "image_names": list(OBJECT_SCENE_ANCHOR_OBSERVER_IMAGE_NAMES),
            "maximum_witnesses": OBJECT_SCENE_ANCHOR_OBSERVER_MAX_WITNESSES,
            "maximum_cells": OBJECT_SCENE_ANCHOR_OBSERVER_MAX_CELLS,
            "pass_count": 2,
            "pass_merge": (
                "P+P=P;A+A=A;any-E=E;all-other-pairs=I"
            ),
            "failure_semantics": "failed-or-uncertain-vision-never-A",
            **_authority_data(),
        }
    )


def _decode_bound_crop(
    crop_png_bytes: bytes,
    entry: ObjectSceneAnchorCatalogEntry,
) -> tuple[bytes, np.ndarray]:
    exact = _scene_runtime._validate_exact_png(crop_png_bytes, "object crop")
    try:
        with Image.open(BytesIO(exact)) as image:
            if (
                image.format != "PNG"
                or getattr(image, "n_frames", 1) != 1
                or image.mode != "L"
            ):
                raise ObjectSceneAnchorObserverError(
                    "object crop must be one exact grayscale PNG"
                )
            luminance = np.ascontiguousarray(np.asarray(image, dtype=np.uint8))
    except ObjectSceneAnchorObserverError:
        raise
    except Exception as exc:
        raise ObjectSceneAnchorObserverError(
            "object crop grayscale decode failed"
        ) from exc
    if luminance.shape != (entry.crop_height_pixels, entry.crop_width_pixels):
        raise ObjectSceneAnchorObserverError(
            "object crop dimensions differ from catalog entry"
        )
    strength = np.ascontiguousarray(255 - luminance, dtype=np.uint8)
    if _hypotheses._crop_pixel_digest(strength) != entry.masked_crop_pixel_digest:
        raise ObjectSceneAnchorObserverError(
            "object crop pixels differ from catalog entry"
        )
    return exact, strength


def _preparation_content(
    value: "ObjectSceneAnchorObserverPreparation",
) -> dict[str, object]:
    return {
        "schema": OBJECT_SCENE_ANCHOR_OBSERVER_PREPARATION_SCHEMA,
        "panel_manifest": value.panel_manifest.to_data(),
        "panel_manifest_digest": value.panel_manifest_digest,
        "object_index": value.object_index,
        "object_id": value.object_id,
        "decision_manifest_digest": value.decision_manifest_digest,
        "crop_renderer_id": OBJECT_SCENE_ANCHOR_CROP_RENDERER_ID,
        "crop_png_byte_count": value.crop_png_byte_count,
        "crop_png_digest": value.crop_png_digest,
        "crop_pixel_digest": value.crop_pixel_digest,
        "crop_width_pixels": value.crop_width_pixels,
        "crop_height_pixels": value.crop_height_pixels,
        "atlas_artifact_digest": value.atlas_artifact_digest,
        "atlas_png_byte_count": value.atlas_png_byte_count,
        "atlas_png_digest": value.atlas_png_digest,
        "catalog": value.catalog.to_data(),
        "catalog_digest": value.catalog_digest,
        "vocabulary": value.vocabulary.to_data(),
        "vocabulary_digest": value.vocabulary_digest,
        "locators": [item.to_data() for item in value.locators],
        "prompt_digest": value.prompt_digest,
        "output_schema_digest": value.output_schema_digest,
        "model_image_names": list(OBJECT_SCENE_ANCHOR_OBSERVER_IMAGE_NAMES),
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class ObjectSceneAnchorObserverPreparation:
    """Decision-only frozen preimage of one per-catalog observer rectangle."""

    panel_manifest: ObjectSceneAnchorPanelDecisionManifest
    panel_manifest_digest: str
    object_index: int
    object_id: str
    decision_manifest_digest: str
    crop_png_byte_count: int
    crop_png_digest: str
    crop_pixel_digest: str
    crop_width_pixels: int
    crop_height_pixels: int
    atlas_artifact_digest: str
    atlas_png_byte_count: int
    atlas_png_digest: str
    catalog: ObjectSceneAnchorBindingCatalog
    catalog_digest: str
    vocabulary: ObjectSceneAnchorObserverVocabulary
    vocabulary_digest: str
    locators: tuple[ObjectSceneAnchorObserverBindingLocator, ...]
    prompt_digest: str
    output_schema_digest: str
    preparation_digest: str

    def __post_init__(self) -> None:
        if type(self.panel_manifest) is not ObjectSceneAnchorPanelDecisionManifest:
            raise TypeError("preparation panel manifest has the wrong type")
        if type(self.catalog) is not ObjectSceneAnchorBindingCatalog:
            raise TypeError("preparation catalog has the wrong type")
        if type(self.vocabulary) is not ObjectSceneAnchorObserverVocabulary:
            raise TypeError("preparation vocabulary has the wrong type")
        _integer(self.object_index, "preparation object index")
        _integer(self.crop_png_byte_count, "crop PNG byte count", minimum=1)
        _integer(self.crop_width_pixels, "crop width", minimum=1)
        _integer(self.crop_height_pixels, "crop height", minimum=1)
        _integer(self.atlas_png_byte_count, "atlas PNG byte count", minimum=1)
        if not isinstance(self.object_id, str) or _OBJECT_ID.fullmatch(
            self.object_id
        ) is None:
            raise ObjectSceneAnchorObserverError("preparation object ID differs")
        for label, item in (
            ("panel manifest digest", self.panel_manifest_digest),
            ("decision manifest digest", self.decision_manifest_digest),
            ("crop PNG digest", self.crop_png_digest),
            ("crop pixel digest", self.crop_pixel_digest),
            ("atlas artifact digest", self.atlas_artifact_digest),
            ("atlas PNG digest", self.atlas_png_digest),
            ("catalog digest", self.catalog_digest),
            ("vocabulary digest", self.vocabulary_digest),
            ("prompt digest", self.prompt_digest),
            ("output schema digest", self.output_schema_digest),
            ("preparation digest", self.preparation_digest),
        ):
            _digest(item, label)
        panel = self.panel_manifest
        if (
            panel.manifest_digest != self.panel_manifest_digest
            or self.object_index >= panel.proposal_count
            or panel.object_ids[self.object_index] != self.object_id
            or panel.object_decisions[self.object_index].manifest_digest
            != self.decision_manifest_digest
        ):
            raise ObjectSceneAnchorObserverError(
                "preparation panel object binding differs"
            )
        catalog = self.catalog
        if (
            catalog.catalog_digest != self.catalog_digest
            or catalog.object_id != self.object_id
            or catalog.decision_manifest_digest != self.decision_manifest_digest
            or catalog.hard_disposition is not Disposition.PRESENT
            or not catalog.catalog_complete_under_spec
            or not catalog.bindings
        ):
            raise ObjectSceneAnchorObserverError(
                "preparation requires one complete nonempty binding catalog"
            )
        if self.vocabulary.vocabulary_digest != self.vocabulary_digest:
            raise ObjectSceneAnchorObserverError(
                "preparation vocabulary binding differs"
            )
        if (
            type(self.locators) is not tuple
            or len(self.locators) != len(catalog.bindings)
            or any(
                type(item) is not ObjectSceneAnchorObserverBindingLocator
                for item in self.locators
            )
        ):
            raise ObjectSceneAnchorObserverError(
                "preparation locator inventory differs"
            )
        for binding, locator in zip(catalog.bindings, self.locators, strict=True):
            if (
                locator.object_id != self.object_id
                or locator.decision_manifest_digest != self.decision_manifest_digest
                or locator.catalog_digest != self.catalog_digest
                or locator.spec_digest != catalog.binding_spec.spec_digest
                or locator.binding_id != binding.binding_id
                or locator.binding_digest != binding.binding_digest
                or locator.anchor_kind != binding.anchor_kind
                or locator.anchor_id != binding.anchor_id
                or locator.anchor_digest != binding.anchor_digest
                or locator.selected_graph_digest != binding.selected_graph_digest
            ):
                raise ObjectSceneAnchorObserverError(
                    "preparation locator differs from exhaustive catalog"
                )
        prompt = object_scene_anchor_observer_prompt(self.locators, self.vocabulary)
        schema = object_scene_anchor_observer_output_schema(
            self.locators, self.vocabulary
        )
        if (
            self.prompt_digest
            != hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            or self.output_schema_digest != canonical_digest(schema)
        ):
            raise ObjectSceneAnchorObserverError(
                "preparation prompt or schema binding differs"
            )
        _digest(self.preparation_digest, "preparation digest")
        if self.preparation_digest != canonical_digest(_preparation_content(self)):
            raise ObjectSceneAnchorObserverError("preparation digest differs")

    @property
    def cell_count(self) -> int:
        return len(self.locators) * len(self.vocabulary.entries)

    def to_data(self) -> dict[str, object]:
        return {
            **_preparation_content(self),
            "preparation_digest": self.preparation_digest,
        }

    @classmethod
    def from_data(cls, value: object) -> "ObjectSceneAnchorObserverPreparation":
        raw = _exact_fields(
            value,
            {
                "schema",
                "panel_manifest",
                "panel_manifest_digest",
                "object_index",
                "object_id",
                "decision_manifest_digest",
                "crop_renderer_id",
                "crop_png_byte_count",
                "crop_png_digest",
                "crop_pixel_digest",
                "crop_width_pixels",
                "crop_height_pixels",
                "atlas_artifact_digest",
                "atlas_png_byte_count",
                "atlas_png_digest",
                "catalog",
                "catalog_digest",
                "vocabulary",
                "vocabulary_digest",
                "locators",
                "prompt_digest",
                "output_schema_digest",
                "model_image_names",
                *_authority_data(),
                "preparation_digest",
            },
            "observer preparation",
        )
        if (
            raw["schema"] != OBJECT_SCENE_ANCHOR_OBSERVER_PREPARATION_SCHEMA
            or raw["crop_renderer_id"] != OBJECT_SCENE_ANCHOR_CROP_RENDERER_ID
            or raw["model_image_names"]
            != list(OBJECT_SCENE_ANCHOR_OBSERVER_IMAGE_NAMES)
            or any(raw[key] != item for key, item in _authority_data().items())
            or not isinstance(raw["panel_manifest"], Mapping)
            or not isinstance(raw["catalog"], Mapping)
            or not isinstance(raw["vocabulary"], Mapping)
            or not isinstance(raw["locators"], list)
        ):
            raise ObjectSceneAnchorObserverError("observer preparation policy differs")
        result = cls(
            panel_manifest=ObjectSceneAnchorPanelDecisionManifest.from_data(
                raw["panel_manifest"]
            ),
            panel_manifest_digest=raw["panel_manifest_digest"],
            object_index=raw["object_index"],
            object_id=raw["object_id"],
            decision_manifest_digest=raw["decision_manifest_digest"],
            crop_png_byte_count=raw["crop_png_byte_count"],
            crop_png_digest=raw["crop_png_digest"],
            crop_pixel_digest=raw["crop_pixel_digest"],
            crop_width_pixels=raw["crop_width_pixels"],
            crop_height_pixels=raw["crop_height_pixels"],
            atlas_artifact_digest=raw["atlas_artifact_digest"],
            atlas_png_byte_count=raw["atlas_png_byte_count"],
            atlas_png_digest=raw["atlas_png_digest"],
            catalog=ObjectSceneAnchorBindingCatalog.from_data(raw["catalog"]),
            catalog_digest=raw["catalog_digest"],
            vocabulary=ObjectSceneAnchorObserverVocabulary.from_data(
                raw["vocabulary"]
            ),
            vocabulary_digest=raw["vocabulary_digest"],
            locators=tuple(
                ObjectSceneAnchorObserverBindingLocator.from_data(item)
                for item in raw["locators"]
            ),
            prompt_digest=raw["prompt_digest"],
            output_schema_digest=raw["output_schema_digest"],
            preparation_digest=raw["preparation_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectSceneAnchorObserverError(
                "observer preparation is not canonical"
            )
        return result


def prepare_object_scene_anchor_observer_inputs(
    crop_png_bytes: bytes,
    *,
    catalog_entry: ObjectSceneAnchorCatalogEntry,
    panel_manifest: ObjectSceneAnchorPanelDecisionManifest,
    atlas: ObjectSceneAnchorAtlas,
    atlas_png_bytes: bytes,
    catalog: ObjectSceneAnchorBindingCatalog,
    vocabulary: ObjectSceneAnchorObserverVocabulary,
) -> ObjectSceneAnchorObserverPreparation:
    """Verify full provenance, then freeze only the decision-facing projection."""

    if type(catalog_entry) is not ObjectSceneAnchorCatalogEntry:
        raise TypeError("catalog_entry must be exact ObjectSceneAnchorCatalogEntry")
    if type(panel_manifest) is not ObjectSceneAnchorPanelDecisionManifest:
        raise TypeError(
            "panel_manifest must be exact ObjectSceneAnchorPanelDecisionManifest"
        )
    if type(atlas) is not ObjectSceneAnchorAtlas:
        raise TypeError("atlas must be exact ObjectSceneAnchorAtlas")
    if type(catalog) is not ObjectSceneAnchorBindingCatalog:
        raise TypeError("catalog must be exact ObjectSceneAnchorBindingCatalog")
    if type(vocabulary) is not ObjectSceneAnchorObserverVocabulary:
        raise TypeError("vocabulary must be exact observer vocabulary")
    entry = ObjectSceneAnchorCatalogEntry.from_data(catalog_entry.to_data())
    panel = ObjectSceneAnchorPanelDecisionManifest.from_data(panel_manifest.to_data())
    frozen_catalog = ObjectSceneAnchorBindingCatalog.from_data(catalog.to_data())
    frozen_vocabulary = ObjectSceneAnchorObserverVocabulary.from_data(
        vocabulary.to_data()
    )
    if entry.inventory_index >= panel.proposal_count or (
        panel.object_ids[entry.inventory_index] != entry.object_id
        or panel.object_decisions[entry.inventory_index] != entry.decision_manifest
        or frozen_catalog.object_id != entry.object_id
        or frozen_catalog.decision_manifest_digest
        != entry.decision_manifest.manifest_digest
    ):
        raise ObjectSceneAnchorObserverError(
            "entry, panel manifest, and binding catalog describe different objects"
        )
    crop, _strength = _decode_bound_crop(crop_png_bytes, entry)
    if not isinstance(atlas_png_bytes, bytes):
        raise TypeError("atlas_png_bytes must be exact bytes")
    verified_atlas = verify_object_scene_anchor_atlas(
        atlas,
        atlas_png_bytes,
        entry.decision_manifest,
        expected_artifact_digest=atlas.artifact_digest,
    )
    if (
        verified_atlas.status.state != "clean"
        or verified_atlas.object_id != entry.object_id
        or verified_atlas.decision_manifest_digest
        != entry.decision_manifest.manifest_digest
        or verified_atlas.selected_graph_artifact_digest
        != frozen_catalog.selected_graph_digest
        or verified_atlas.png_digest
        != hashlib.sha256(atlas_png_bytes).hexdigest()
    ):
        raise ObjectSceneAnchorObserverError(
            "atlas differs from exact object decision or binding catalog"
        )
    locators = _locators_for(frozen_catalog, verified_atlas)
    prompt = object_scene_anchor_observer_prompt(locators, frozen_vocabulary)
    schema = object_scene_anchor_observer_output_schema(locators, frozen_vocabulary)
    _scene_runtime._assert_model_visible_boundary(
        prompt,
        schema,
        OBJECT_SCENE_ANCHOR_OBSERVER_IMAGE_NAMES,
        hidden_values=(
            panel.panel_digest,
            panel.inventory_digest,
            panel.manifest_digest,
            frozen_catalog.catalog_digest,
            frozen_vocabulary.vocabulary_digest,
            verified_atlas.artifact_digest,
            entry.masked_crop_pixel_digest,
        ),
        allowed_visual_words=("side", "path"),
    )
    values = {
        "panel_manifest": panel,
        "panel_manifest_digest": panel.manifest_digest,
        "object_index": entry.inventory_index,
        "object_id": entry.object_id,
        "decision_manifest_digest": entry.decision_manifest.manifest_digest,
        "crop_png_byte_count": len(crop),
        "crop_png_digest": hashlib.sha256(crop).hexdigest(),
        "crop_pixel_digest": entry.masked_crop_pixel_digest,
        "crop_width_pixels": entry.crop_width_pixels,
        "crop_height_pixels": entry.crop_height_pixels,
        "atlas_artifact_digest": verified_atlas.artifact_digest,
        "atlas_png_byte_count": len(atlas_png_bytes),
        "atlas_png_digest": hashlib.sha256(atlas_png_bytes).hexdigest(),
        "catalog": frozen_catalog,
        "catalog_digest": frozen_catalog.catalog_digest,
        "vocabulary": frozen_vocabulary,
        "vocabulary_digest": frozen_vocabulary.vocabulary_digest,
        "locators": locators,
        "prompt_digest": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "output_schema_digest": canonical_digest(schema),
    }
    provisional = object.__new__(ObjectSceneAnchorObserverPreparation)
    for name, item in values.items():
        object.__setattr__(provisional, name, item)
    return ObjectSceneAnchorObserverPreparation(
        **values,
        preparation_digest=canonical_digest(_preparation_content(provisional)),
    )


def _cell_content(value: "ObjectSceneAnchorObserverCell") -> dict[str, object]:
    return {
        "schema": OBJECT_SCENE_ANCHOR_OBSERVER_CELL_SCHEMA,
        "cell_phase": value.cell_phase,
        "locator": value.locator.to_data(),
        "witness": value.witness.to_data(),
        "binding_cell": value.binding_cell.to_data(),
        "reason_code": value.reason_code,
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class ObjectSceneAnchorObserverCell:
    """One reasoned cell wrapping the binding layer's canonical disposition."""

    cell_phase: str
    locator: ObjectSceneAnchorObserverBindingLocator
    witness: ObjectSceneAnchorObserverVocabularyEntry
    binding_cell: ObjectSceneAnchorWitnessCell
    reason_code: str
    cell_digest: str

    def __post_init__(self) -> None:
        if self.cell_phase not in ("pass", "merged"):
            raise ObjectSceneAnchorObserverError("observer cell phase differs")
        if type(self.locator) is not ObjectSceneAnchorObserverBindingLocator:
            raise TypeError("observer cell locator has the wrong type")
        if type(self.witness) is not ObjectSceneAnchorObserverVocabularyEntry:
            raise TypeError("observer cell witness has the wrong type")
        if type(self.binding_cell) is not ObjectSceneAnchorWitnessCell:
            raise TypeError("observer binding cell has the wrong type")
        if (
            self.binding_cell.binding_digest != self.locator.binding_digest
            or self.binding_cell.witness_id != self.witness.witness_id
            or self.binding_cell.witness_digest != self.witness.witness_digest
        ):
            raise ObjectSceneAnchorObserverError(
                "observer cell locator/witness binding differs"
            )
        if not isinstance(self.reason_code, str):
            raise ObjectSceneAnchorObserverError("observer cell reason differs")
        disposition = self.binding_cell.disposition
        if self.cell_phase == "pass":
            allowed = (
                _PASS_ERROR_REASONS
                if disposition is Disposition.ERROR
                else _MODEL_REASON_BY_STATE[
                    {
                        Disposition.PRESENT: "P",
                        Disposition.CERTIFIED_ABSENT: "A",
                        Disposition.INDETERMINATE: "I",
                    }[disposition]
                ]
            )
        else:
            allowed = _MERGED_REASON_BY_DISPOSITION[disposition]
        if self.reason_code not in allowed:
            raise ObjectSceneAnchorObserverError(
                "observer cell reason/disposition pair differs"
            )
        _digest(self.cell_digest, "observer cell digest")
        if self.cell_digest != canonical_digest(_cell_content(self)):
            raise ObjectSceneAnchorObserverError("observer cell digest differs")

    @classmethod
    def create(
        cls,
        *,
        cell_phase: str,
        binding: ObjectSceneResolvedAnchorBinding,
        locator: ObjectSceneAnchorObserverBindingLocator,
        witness: ObjectSceneAnchorObserverVocabularyEntry,
        disposition: Disposition,
        reason_code: str,
    ) -> "ObjectSceneAnchorObserverCell":
        base = ObjectSceneAnchorWitnessCell.create(
            binding, witness.binding_witness_spec, disposition
        )
        values = {
            "cell_phase": cell_phase,
            "locator": locator,
            "witness": witness,
            "binding_cell": base,
            "reason_code": reason_code,
        }
        provisional = object.__new__(cls)
        for name, item in values.items():
            object.__setattr__(provisional, name, item)
        return cls(
            **values,
            cell_digest=canonical_digest(_cell_content(provisional)),
        )

    @property
    def disposition(self) -> Disposition:
        return self.binding_cell.disposition

    def to_data(self) -> dict[str, object]:
        return {**_cell_content(self), "cell_digest": self.cell_digest}

    @classmethod
    def from_data(cls, value: object) -> "ObjectSceneAnchorObserverCell":
        raw = _exact_fields(
            value,
            {
                "schema",
                "cell_phase",
                "locator",
                "witness",
                "binding_cell",
                "reason_code",
                *_authority_data(),
                "cell_digest",
            },
            "observer cell",
        )
        if (
            raw["schema"] != OBJECT_SCENE_ANCHOR_OBSERVER_CELL_SCHEMA
            or any(raw[key] != item for key, item in _authority_data().items())
            or not isinstance(raw["locator"], Mapping)
            or not isinstance(raw["witness"], Mapping)
            or not isinstance(raw["binding_cell"], Mapping)
        ):
            raise ObjectSceneAnchorObserverError("observer cell policy differs")
        result = cls(
            raw["cell_phase"],
            ObjectSceneAnchorObserverBindingLocator.from_data(raw["locator"]),
            ObjectSceneAnchorObserverVocabularyEntry.from_data(raw["witness"]),
            ObjectSceneAnchorWitnessCell.from_data(raw["binding_cell"]),
            raw["reason_code"],
            raw["cell_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectSceneAnchorObserverError("observer cell is not canonical")
        return result


def _expected_cell_keys(
    preparation: ObjectSceneAnchorObserverPreparation,
) -> tuple[tuple[str, str], ...]:
    return tuple(
        (locator.binding_digest, witness.witness_digest)
        for locator in preparation.locators
        for witness in preparation.vocabulary.entries
    )


def _cells_have_exact_rectangle(
    cells: tuple[ObjectSceneAnchorObserverCell, ...],
    preparation: ObjectSceneAnchorObserverPreparation,
    *,
    phase: str,
) -> bool:
    return (
        len(cells) == preparation.cell_count
        and tuple(
            (item.locator.binding_digest, item.witness.witness_digest)
            for item in cells
        )
        == _expected_cell_keys(preparation)
        and all(item.cell_phase == phase for item in cells)
        and all(
            item.locator == locator and item.witness == witness
            for item, (locator, witness) in zip(
                cells,
                (
                    (locator, witness)
                    for locator in preparation.locators
                    for witness in preparation.vocabulary.entries
                ),
                strict=True,
            )
        )
    )


def _payload_cells(
    payload: Mapping[str, Any],
    preparation: ObjectSceneAnchorObserverPreparation,
) -> tuple[ObjectSceneAnchorObserverCell, ...]:
    raw = _exact_fields(payload, {"cells"}, "observer payload")
    rows = raw["cells"]
    if not isinstance(rows, list) or len(rows) != preparation.cell_count:
        raise ObjectSceneAnchorObserverPayloadError(
            "payload does not exhaust the binding/witness rectangle"
        )
    expected = tuple(
        (binding, locator, witness)
        for binding, locator in zip(
            preparation.catalog.bindings, preparation.locators, strict=True
        )
        for witness in preparation.vocabulary.entries
    )
    result: list[ObjectSceneAnchorObserverCell] = []
    for index, (item, (binding, locator, witness)) in enumerate(
        zip(rows, expected, strict=True)
    ):
        cell = _exact_fields(
            item,
            {"binding_id", "witness_id", "state", "reason_code"},
            f"observer payload cell {index}",
        )
        if (
            cell["binding_id"] != binding.binding_id
            or cell["witness_id"] != witness.witness_id
            or cell["state"] not in _MODEL_STATES
            or cell["reason_code"]
            not in _MODEL_REASON_BY_STATE.get(cell["state"], frozenset())
        ):
            raise ObjectSceneAnchorObserverPayloadError(
                "payload cell order, state, or finite reason differs"
            )
        disposition = {
            "P": Disposition.PRESENT,
            "A": Disposition.CERTIFIED_ABSENT,
            "I": Disposition.INDETERMINATE,
        }[cell["state"]]
        result.append(
            ObjectSceneAnchorObserverCell.create(
                cell_phase="pass",
                binding=binding,
                locator=locator,
                witness=witness,
                disposition=disposition,
                reason_code=cell["reason_code"],
            )
        )
    return tuple(result)


def _error_cells(
    preparation: ObjectSceneAnchorObserverPreparation,
    reason_code: str,
) -> tuple[ObjectSceneAnchorObserverCell, ...]:
    if reason_code not in _PASS_ERROR_REASONS:
        raise ObjectSceneAnchorObserverError("pass error reason differs")
    return tuple(
        ObjectSceneAnchorObserverCell.create(
            cell_phase="pass",
            binding=binding,
            locator=locator,
            witness=witness,
            disposition=Disposition.ERROR,
            reason_code=reason_code,
        )
        for binding, locator in zip(
            preparation.catalog.bindings, preparation.locators, strict=True
        )
        for witness in preparation.vocabulary.entries
    )


def _model_digest(model: str, reasoning_effort: str) -> str:
    if not isinstance(model, str) or not model:
        raise ObjectSceneAnchorObserverError("observer model differs")
    if not isinstance(reasoning_effort, str) or not reasoning_effort:
        raise ObjectSceneAnchorObserverError("observer reasoning effort differs")
    return canonical_digest(
        {
            "schema": "gkm.object-scene-anchor-observer-model.v1",
            "model": model,
            "reasoning_effort": reasoning_effort,
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
            "schema": "gkm.object-scene-anchor-observer-runtime.v1",
            "model_digest": _model_digest(model, reasoning_effort),
            "expected_launcher_digest": expected_launcher_digest,
            "cloud_policy_cache_binding": cloud_policy_cache_binding,
            "model_catalog_digest": model_catalog_digest,
            "no_tools_attestation_digest": no_tools_attestation_digest,
            "protocol_digest": object_scene_anchor_observer_protocol_digest(),
            **_authority_data(),
        }
    )


def _exception_type(exception: BaseException) -> str:
    value = type(exception).__name__
    return value if re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,127}", value) else (
        "UnclassifiedObserverError"
    )


def _pass_content(value: "ObjectSceneAnchorObserverPassArtifact") -> dict[str, object]:
    return {
        "schema": OBJECT_SCENE_ANCHOR_OBSERVER_PASS_SCHEMA,
        "pass_id": value.pass_id,
        "pass_index": value.pass_index,
        "preparation_digest": value.preparation_digest,
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
class ObjectSceneAnchorObserverPassArtifact:
    pass_id: str
    pass_index: int
    preparation_digest: str
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
        if (
            not isinstance(self.pass_id, str)
            or _PASS_ID.fullmatch(self.pass_id) is None
            or self.pass_index not in (0, 1)
            or self.pass_id != f"pass_{self.pass_index:02d}"
        ):
            raise ObjectSceneAnchorObserverError("observer pass position differs")
        for label, item in (
            ("preparation digest", self.preparation_digest),
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
        if self.cloud_policy_cache_binding != "absent":
            _address(self.cloud_policy_cache_binding, "cloud policy cache binding")
        if (
            self.protocol_digest != object_scene_anchor_observer_protocol_digest()
            or self.source_digest != object_scene_anchor_observer_source_digest()
            or self.transport_source_digest
            != _scene_runtime.prototype_scene_transport_source_digest()
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
            raise ObjectSceneAnchorObserverError("observer pass protocol differs")
        if (
            type(self.presentation) is not tuple
            or tuple(item.name for item in self.presentation)
            != OBJECT_SCENE_ANCHOR_OBSERVER_IMAGE_NAMES
            or any(type(item) is not PrototypeImageIdentity for item in self.presentation)
            or self.physical_call_count != 1
            or self.status not in ("success", "parser_error", "transport_error")
            or type(self.cells) is not tuple
            or not self.cells
            or any(type(item) is not ObjectSceneAnchorObserverCell for item in self.cells)
            or any(item.cell_phase != "pass" for item in self.cells)
        ):
            raise ObjectSceneAnchorObserverError("observer pass structure differs")
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
                raise ObjectSceneAnchorObserverError("successful pass differs")
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
                raise ObjectSceneAnchorObserverError("parser-error pass differs")
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
                raise ObjectSceneAnchorObserverError("transport-error pass differs")
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
                or self.receipt.codex_launcher_digest
                != self.expected_launcher_digest
                or self.receipt.cloud_config_bundle_cache_binding
                != self.cloud_policy_cache_binding
                or self.receipt.model_catalog_digest != self.model_catalog_digest
                or self.receipt.tool_surface_attestation_digest
                != self.no_tools_attestation_digest
            ):
                raise ObjectSceneAnchorObserverError(
                    "observer pass receipt binding differs"
                )
        _digest(self.pass_digest, "observer pass digest")
        if self.pass_digest != canonical_digest(_pass_content(self)):
            raise ObjectSceneAnchorObserverError("observer pass digest differs")

    def to_data(self) -> dict[str, object]:
        return {**_pass_content(self), "pass_digest": self.pass_digest}

    @classmethod
    def from_data(cls, value: object) -> "ObjectSceneAnchorObserverPassArtifact":
        raw = _exact_fields(
            value,
            {
                "schema",
                "pass_id",
                "pass_index",
                "preparation_digest",
                "prompt_digest",
                "output_schema_digest",
                "protocol_digest",
                "source_digest",
                "transport_source_digest",
                "model",
                "reasoning_effort",
                "model_digest",
                "expected_launcher_digest",
                "cloud_policy_cache_binding",
                "model_catalog_digest",
                "no_tools_attestation_digest",
                "runtime_identity_digest",
                "presentation",
                "physical_call_count",
                "status",
                "model_payload",
                "receipt",
                "failure_code",
                "failure_type",
                "cells",
                *_authority_data(),
                "pass_digest",
            },
            "observer pass",
        )
        if (
            raw["schema"] != OBJECT_SCENE_ANCHOR_OBSERVER_PASS_SCHEMA
            or any(raw[key] != item for key, item in _authority_data().items())
            or not isinstance(raw["presentation"], list)
            or not isinstance(raw["cells"], list)
        ):
            raise ObjectSceneAnchorObserverError("observer pass policy differs")
        result = cls(
            pass_id=raw["pass_id"],
            pass_index=raw["pass_index"],
            preparation_digest=raw["preparation_digest"],
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
            presentation=tuple(
                PrototypeImageIdentity.from_data(item)
                for item in raw["presentation"]
            ),
            physical_call_count=raw["physical_call_count"],
            status=raw["status"],
            model_payload=(
                None
                if raw["model_payload"] is None
                else _canonical_payload(raw["model_payload"])
            ),
            receipt=_receipt_from_data(raw["receipt"]),
            failure_code=raw["failure_code"],
            failure_type=raw["failure_type"],
            cells=tuple(
                ObjectSceneAnchorObserverCell.from_data(item)
                for item in raw["cells"]
            ),
            pass_digest=raw["pass_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectSceneAnchorObserverError("observer pass is not canonical")
        return result


def _seal_pass(
    *,
    pass_index: int,
    preparation: ObjectSceneAnchorObserverPreparation,
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
) -> ObjectSceneAnchorObserverPassArtifact:
    runtime = _runtime_identity_digest(
        model=model,
        reasoning_effort=reasoning_effort,
        expected_launcher_digest=expected_launcher_digest,
        cloud_policy_cache_binding=cloud_policy_cache_binding,
        model_catalog_digest=model_catalog_digest,
        no_tools_attestation_digest=no_tools_attestation_digest,
    )
    values = {
        "pass_id": f"pass_{pass_index:02d}",
        "pass_index": pass_index,
        "preparation_digest": preparation.preparation_digest,
        "prompt_digest": preparation.prompt_digest,
        "output_schema_digest": preparation.output_schema_digest,
        "protocol_digest": object_scene_anchor_observer_protocol_digest(),
        "source_digest": object_scene_anchor_observer_source_digest(),
        "transport_source_digest": (
            _scene_runtime.prototype_scene_transport_source_digest()
        ),
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
    provisional = object.__new__(ObjectSceneAnchorObserverPassArtifact)
    for name, item in values.items():
        object.__setattr__(provisional, name, item)
    return ObjectSceneAnchorObserverPassArtifact(
        **values,
        pass_digest=canonical_digest(_pass_content(provisional)),
    )


def _merge_cell_dispositions(
    first: Disposition,
    second: Disposition,
) -> tuple[Disposition, str]:
    if Disposition.ERROR in (first, second):
        return Disposition.ERROR, "one_or_both_pass_error"
    if first is second is Disposition.PRESENT:
        return Disposition.PRESENT, "two_pass_visible_match"
    if first is second is Disposition.CERTIFIED_ABSENT:
        return Disposition.CERTIFIED_ABSENT, "two_pass_visible_mismatch"
    if Disposition.INDETERMINATE in (first, second):
        return Disposition.INDETERMINATE, "two_pass_indeterminate"
    return Disposition.INDETERMINATE, "two_pass_disagreement"


def _merge_pass_cells(
    preparation: ObjectSceneAnchorObserverPreparation,
    first: ObjectSceneAnchorObserverPassArtifact,
    second: ObjectSceneAnchorObserverPassArtifact,
) -> tuple[ObjectSceneAnchorObserverCell, ...]:
    if (
        not _cells_have_exact_rectangle(first.cells, preparation, phase="pass")
        or not _cells_have_exact_rectangle(second.cells, preparation, phase="pass")
    ):
        raise ObjectSceneAnchorObserverError(
            "observer pass omits or reorders the exact cell rectangle"
        )
    result: list[ObjectSceneAnchorObserverCell] = []
    expected = tuple(
        (binding, locator, witness)
        for binding, locator in zip(
            preparation.catalog.bindings, preparation.locators, strict=True
        )
        for witness in preparation.vocabulary.entries
    )
    for first_cell, second_cell, (binding, locator, witness) in zip(
        first.cells,
        second.cells,
        expected,
        strict=True,
    ):
        disposition, reason = _merge_cell_dispositions(
            first_cell.disposition, second_cell.disposition
        )
        result.append(
            ObjectSceneAnchorObserverCell.create(
                cell_phase="merged",
                binding=binding,
                locator=locator,
                witness=witness,
                disposition=disposition,
                reason_code=reason,
            )
        )
    return tuple(result)


def _artifact_content(
    value: "ObjectSceneAnchorObserverArtifact",
) -> dict[str, object]:
    return {
        "schema": OBJECT_SCENE_ANCHOR_OBSERVER_ARTIFACT_SCHEMA,
        "observation_plan_digest": value.observation_plan_digest,
        "preparation": value.preparation.to_data(),
        "preparation_digest": value.preparation_digest,
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
        "passes": [item.to_data() for item in value.passes],
        "merged_cells": [item.to_data() for item in value.merged_cells],
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class ObjectSceneAnchorObserverArtifact:
    """Exactly two per-catalog passes and their pre-expression cell merge."""

    observation_plan_digest: str
    preparation: ObjectSceneAnchorObserverPreparation
    preparation_digest: str
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
    passes: tuple[
        ObjectSceneAnchorObserverPassArtifact,
        ObjectSceneAnchorObserverPassArtifact,
    ]
    merged_cells: tuple[ObjectSceneAnchorObserverCell, ...]
    artifact_digest: str

    def __post_init__(self) -> None:
        _address(self.observation_plan_digest, "observation plan digest")
        if type(self.preparation) is not ObjectSceneAnchorObserverPreparation:
            raise TypeError("observer preparation has the wrong type")
        for label, item in (
            ("preparation digest", self.preparation_digest),
            ("protocol digest", self.protocol_digest),
            ("source digest", self.source_digest),
            ("transport source digest", self.transport_source_digest),
            ("model digest", self.model_digest),
            ("launcher digest", self.expected_launcher_digest),
            ("model catalog digest", self.model_catalog_digest),
            ("no-tools digest", self.no_tools_attestation_digest),
            ("runtime identity digest", self.runtime_identity_digest),
            ("artifact digest", self.artifact_digest),
        ):
            _digest(item, label)
        if self.cloud_policy_cache_binding != "absent":
            _address(self.cloud_policy_cache_binding, "cloud policy cache binding")
        expected_runtime = _runtime_identity_digest(
            model=self.model,
            reasoning_effort=self.reasoning_effort,
            expected_launcher_digest=self.expected_launcher_digest,
            cloud_policy_cache_binding=self.cloud_policy_cache_binding,
            model_catalog_digest=self.model_catalog_digest,
            no_tools_attestation_digest=self.no_tools_attestation_digest,
        )
        if (
            self.preparation.preparation_digest != self.preparation_digest
            or self.protocol_digest != object_scene_anchor_observer_protocol_digest()
            or self.source_digest != object_scene_anchor_observer_source_digest()
            or self.transport_source_digest
            != _scene_runtime.prototype_scene_transport_source_digest()
            or self.model_digest != _model_digest(self.model, self.reasoning_effort)
            or self.runtime_identity_digest != expected_runtime
            or self.physical_call_count != 2
            or type(self.passes) is not tuple
            or len(self.passes) != 2
            or any(
                type(item) is not ObjectSceneAnchorObserverPassArtifact
                for item in self.passes
            )
            or tuple(item.pass_id for item in self.passes)
            != ("pass_00", "pass_01")
        ):
            raise ObjectSceneAnchorObserverError("observer artifact protocol differs")
        for item in self.passes:
            if (
                item.preparation_digest != self.preparation_digest
                or item.prompt_digest != self.preparation.prompt_digest
                or item.output_schema_digest != self.preparation.output_schema_digest
                or item.protocol_digest != self.protocol_digest
                or item.source_digest != self.source_digest
                or item.transport_source_digest != self.transport_source_digest
                or item.model != self.model
                or item.reasoning_effort != self.reasoning_effort
                or item.model_digest != self.model_digest
                or item.expected_launcher_digest != self.expected_launcher_digest
                or item.cloud_policy_cache_binding
                != self.cloud_policy_cache_binding
                or item.model_catalog_digest != self.model_catalog_digest
                or item.no_tools_attestation_digest
                != self.no_tools_attestation_digest
                or item.runtime_identity_digest != self.runtime_identity_digest
                or not _cells_have_exact_rectangle(
                    item.cells, self.preparation, phase="pass"
                )
            ):
                raise ObjectSceneAnchorObserverError(
                    "observer pass differs from frozen preparation, runtime, or exact rectangle"
                )
            if item.status == "success":
                assert item.model_payload is not None
                try:
                    replayed_cells = _payload_cells(
                        item.model_payload, self.preparation
                    )
                except Exception as exc:
                    raise ObjectSceneAnchorObserverError(
                        "successful pass payload does not replay"
                    ) from exc
                if replayed_cells != item.cells:
                    raise ObjectSceneAnchorObserverError(
                        "successful pass cells differ from frozen payload"
                    )
            elif item.status == "parser_error":
                assert item.model_payload is not None
                try:
                    _payload_cells(item.model_payload, self.preparation)
                except Exception as exc:
                    if item.failure_type != _exception_type(exc):
                        raise ObjectSceneAnchorObserverError(
                            "parser failure type differs from deterministic replay"
                        ) from exc
                else:
                    raise ObjectSceneAnchorObserverError(
                        "parser-error pass payload now parses successfully"
                    )
        receipts = tuple(item.receipt for item in self.passes)
        if receipts[0] is not None and receipts[1] is not None and (
            receipts[0].receipt_digest == receipts[1].receipt_digest
            or receipts[0].thread_id == receipts[1].thread_id
        ):
            raise ObjectSceneAnchorObserverError(
                "observer passes do not have independent receipts"
            )
        expected_merged = _merge_pass_cells(
            self.preparation, self.passes[0], self.passes[1]
        )
        if (
            type(self.merged_cells) is not tuple
            or self.merged_cells != expected_merged
            or not _cells_have_exact_rectangle(
                self.merged_cells, self.preparation, phase="merged"
            )
        ):
            raise ObjectSceneAnchorObserverError(
                "observer merged cells differ from exact same-key pass merge"
            )
        _digest(self.artifact_digest, "observer artifact digest")
        if self.artifact_digest != canonical_digest(_artifact_content(self)):
            raise ObjectSceneAnchorObserverError("observer artifact digest differs")

    @property
    def catalog(self) -> ObjectSceneAnchorBindingCatalog:
        return self.preparation.catalog

    @property
    def vocabulary(self) -> ObjectSceneAnchorObserverVocabulary:
        return self.preparation.vocabulary

    def to_data(self) -> dict[str, object]:
        return {**_artifact_content(self), "artifact_digest": self.artifact_digest}

    @classmethod
    def from_data(cls, value: object) -> "ObjectSceneAnchorObserverArtifact":
        raw = _exact_fields(
            value,
            {
                "schema",
                "observation_plan_digest",
                "preparation",
                "preparation_digest",
                "protocol_digest",
                "source_digest",
                "transport_source_digest",
                "model",
                "reasoning_effort",
                "model_digest",
                "expected_launcher_digest",
                "cloud_policy_cache_binding",
                "model_catalog_digest",
                "no_tools_attestation_digest",
                "runtime_identity_digest",
                "physical_call_count",
                "passes",
                "merged_cells",
                *_authority_data(),
                "artifact_digest",
            },
            "observer artifact",
        )
        if (
            raw["schema"] != OBJECT_SCENE_ANCHOR_OBSERVER_ARTIFACT_SCHEMA
            or any(raw[key] != item for key, item in _authority_data().items())
            or not isinstance(raw["preparation"], Mapping)
            or not isinstance(raw["passes"], list)
            or not isinstance(raw["merged_cells"], list)
        ):
            raise ObjectSceneAnchorObserverError("observer artifact policy differs")
        passes = tuple(
            ObjectSceneAnchorObserverPassArtifact.from_data(item)
            for item in raw["passes"]
        )
        if len(passes) != 2:
            raise ObjectSceneAnchorObserverError("observer artifact pass count differs")
        result = cls(
            observation_plan_digest=raw["observation_plan_digest"],
            preparation=ObjectSceneAnchorObserverPreparation.from_data(
                raw["preparation"]
            ),
            preparation_digest=raw["preparation_digest"],
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
            physical_call_count=raw["physical_call_count"],
            passes=passes,  # type: ignore[arg-type]
            merged_cells=tuple(
                ObjectSceneAnchorObserverCell.from_data(item)
                for item in raw["merged_cells"]
            ),
            artifact_digest=raw["artifact_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectSceneAnchorObserverError(
                "observer artifact is not canonical"
            )
        return result


def _seal_artifact(
    *,
    observation_plan_digest: str,
    preparation: ObjectSceneAnchorObserverPreparation,
    model: str,
    reasoning_effort: str,
    expected_launcher_digest: str,
    cloud_policy_cache_binding: str,
    model_catalog_digest: str,
    no_tools_attestation_digest: str,
    passes: tuple[
        ObjectSceneAnchorObserverPassArtifact,
        ObjectSceneAnchorObserverPassArtifact,
    ],
) -> ObjectSceneAnchorObserverArtifact:
    values = {
        "observation_plan_digest": observation_plan_digest,
        "preparation": preparation,
        "preparation_digest": preparation.preparation_digest,
        "protocol_digest": object_scene_anchor_observer_protocol_digest(),
        "source_digest": object_scene_anchor_observer_source_digest(),
        "transport_source_digest": (
            _scene_runtime.prototype_scene_transport_source_digest()
        ),
        "model": model,
        "reasoning_effort": reasoning_effort,
        "model_digest": _model_digest(model, reasoning_effort),
        "expected_launcher_digest": expected_launcher_digest,
        "cloud_policy_cache_binding": cloud_policy_cache_binding,
        "model_catalog_digest": model_catalog_digest,
        "no_tools_attestation_digest": no_tools_attestation_digest,
        "runtime_identity_digest": _runtime_identity_digest(
            model=model,
            reasoning_effort=reasoning_effort,
            expected_launcher_digest=expected_launcher_digest,
            cloud_policy_cache_binding=cloud_policy_cache_binding,
            model_catalog_digest=model_catalog_digest,
            no_tools_attestation_digest=no_tools_attestation_digest,
        ),
        "physical_call_count": 2,
        "passes": passes,
        "merged_cells": _merge_pass_cells(preparation, passes[0], passes[1]),
    }
    provisional = object.__new__(ObjectSceneAnchorObserverArtifact)
    for name, item in values.items():
        object.__setattr__(provisional, name, item)
    return ObjectSceneAnchorObserverArtifact(
        **values,
        artifact_digest=canonical_digest(_artifact_content(provisional)),
    )


def observe_object_scene_anchor_catalog_twice(
    crop_png_bytes: bytes,
    *,
    catalog_entry: ObjectSceneAnchorCatalogEntry,
    panel_manifest: ObjectSceneAnchorPanelDecisionManifest,
    atlas: ObjectSceneAnchorAtlas,
    atlas_png_bytes: bytes,
    catalog: ObjectSceneAnchorBindingCatalog,
    vocabulary: ObjectSceneAnchorObserverVocabulary,
    expected_panel_manifest_digest: str,
    expected_crop_png_digest: str,
    expected_crop_pixel_digest: str,
    expected_atlas_artifact_digest: str,
    expected_atlas_png_digest: str,
    expected_catalog_digest: str,
    expected_vocabulary_digest: str,
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
) -> ObjectSceneAnchorObserverArtifact:
    """Attempt exactly two isolated no-tools observations of one rectangle."""

    context = _address(observation_plan_digest, "observation plan digest")
    preparation = prepare_object_scene_anchor_observer_inputs(
        crop_png_bytes,
        catalog_entry=catalog_entry,
        panel_manifest=panel_manifest,
        atlas=atlas,
        atlas_png_bytes=atlas_png_bytes,
        catalog=catalog,
        vocabulary=vocabulary,
    )
    commitments = (
        (
            preparation.panel_manifest_digest,
            _digest(expected_panel_manifest_digest, "expected panel manifest digest"),
            "panel manifest",
        ),
        (
            preparation.crop_png_digest,
            _digest(expected_crop_png_digest, "expected crop PNG digest"),
            "crop PNG",
        ),
        (
            preparation.crop_pixel_digest,
            _digest(expected_crop_pixel_digest, "expected crop pixel digest"),
            "crop pixels",
        ),
        (
            preparation.atlas_artifact_digest,
            _digest(expected_atlas_artifact_digest, "expected atlas artifact digest"),
            "atlas artifact",
        ),
        (
            preparation.atlas_png_digest,
            _digest(expected_atlas_png_digest, "expected atlas PNG digest"),
            "atlas PNG",
        ),
        (
            preparation.catalog_digest,
            _digest(expected_catalog_digest, "expected binding catalog digest"),
            "binding catalog",
        ),
        (
            preparation.vocabulary_digest,
            _digest(expected_vocabulary_digest, "expected vocabulary digest"),
            "vocabulary",
        ),
    )
    for observed, expected, label in commitments:
        if observed != expected:
            raise ObjectSceneAnchorObserverError(
                f"{label} differs from external commitment"
            )
    if not callable(transport):
        raise TypeError("transport must be callable")
    launcher = _digest(expected_launcher_digest, "expected launcher digest")
    policy_binding = _scene_runtime._policy_cache_binding(
        cloud_policy_cache_snapshot
    )
    model_catalog_digest, no_tools_digest = (
        _scene_runtime._validate_no_tools_runtime(
            model_catalog_snapshot=model_catalog_snapshot,
            no_tools_attestation=no_tools_attestation,
            expected_launcher_digest=launcher,
            cloud_policy_cache_binding=policy_binding,
        )
    )
    crop = _scene_runtime._validate_exact_png(crop_png_bytes, "object crop")
    atlas_png = _scene_runtime._validate_exact_png(
        atlas_png_bytes, "anchor atlas"
    )
    presentation_bytes = (
        ("object.png", crop),
        ("anchors.png", atlas_png),
    )
    presentation = _scene_runtime._image_identities(presentation_bytes)
    prompt = object_scene_anchor_observer_prompt(
        preparation.locators, preparation.vocabulary
    )
    schema = object_scene_anchor_observer_output_schema(
        preparation.locators, preparation.vocabulary
    )
    passes: list[ObjectSceneAnchorObserverPassArtifact] = []
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
                    pass_index=pass_index,
                    preparation=preparation,
                    presentation=presentation,
                    model=model,
                    reasoning_effort=reasoning_effort,
                    expected_launcher_digest=launcher,
                    cloud_policy_cache_binding=policy_binding,
                    model_catalog_digest=model_catalog_digest,
                    no_tools_attestation_digest=no_tools_digest,
                    status="transport_error",
                    payload=None,
                    receipt=None,
                    failure_code="transport_failed",
                    failure_type=_exception_type(exc),
                    cells=_error_cells(preparation, "transport_failed"),
                )
            )
            continue
        try:
            cells = _payload_cells(payload, preparation)
        except Exception as exc:
            passes.append(
                _seal_pass(
                    pass_index=pass_index,
                    preparation=preparation,
                    presentation=presentation,
                    model=model,
                    reasoning_effort=reasoning_effort,
                    expected_launcher_digest=launcher,
                    cloud_policy_cache_binding=policy_binding,
                    model_catalog_digest=model_catalog_digest,
                    no_tools_attestation_digest=no_tools_digest,
                    status="parser_error",
                    payload=payload,
                    receipt=receipt,
                    failure_code="payload_rejected",
                    failure_type=_exception_type(exc),
                    cells=_error_cells(preparation, "payload_rejected"),
                )
            )
            continue
        passes.append(
            _seal_pass(
                pass_index=pass_index,
                preparation=preparation,
                presentation=presentation,
                model=model,
                reasoning_effort=reasoning_effort,
                expected_launcher_digest=launcher,
                cloud_policy_cache_binding=policy_binding,
                model_catalog_digest=model_catalog_digest,
                no_tools_attestation_digest=no_tools_digest,
                status="success",
                payload=payload,
                receipt=receipt,
                failure_code=None,
                failure_type=None,
                cells=cells,
            )
        )
    if len(passes) != 2:  # pragma: no cover - closed loop above.
        raise ObjectSceneAnchorObserverError("observer did not attempt two passes")
    return _seal_artifact(
        observation_plan_digest=context,
        preparation=preparation,
        model=model,
        reasoning_effort=reasoning_effort,
        expected_launcher_digest=launcher,
        cloud_policy_cache_binding=policy_binding,
        model_catalog_digest=model_catalog_digest,
        no_tools_attestation_digest=no_tools_digest,
        passes=(passes[0], passes[1]),
    )


def verify_object_scene_anchor_observer_artifact(
    artifact: ObjectSceneAnchorObserverArtifact,
    crop_png_bytes: bytes,
    *,
    catalog_entry: ObjectSceneAnchorCatalogEntry,
    panel_manifest: ObjectSceneAnchorPanelDecisionManifest,
    atlas: ObjectSceneAnchorAtlas,
    atlas_png_bytes: bytes,
    catalog: ObjectSceneAnchorBindingCatalog,
    vocabulary: ObjectSceneAnchorObserverVocabulary,
    expected_artifact_digest: str,
    expected_observation_plan_digest: str,
    expected_runtime_identity_digest: str | None = None,
    panel_png_bytes: bytes | None = None,
    inventory: ObjectSceneProposalInventory | None = None,
) -> ObjectSceneAnchorObserverArtifact:
    """Cold replay exact inputs, finite payload parsing, receipts, and merge.

    Supplying ``panel_png_bytes`` and ``inventory`` additionally re-renders the
    exact deterministic object crop through
    :func:`verify_object_scene_anchor_object_crop`.  The two arguments are an
    all-or-nothing pair so a caller cannot claim panel-level replay from only
    one half of its provenance.
    """

    if type(artifact) is not ObjectSceneAnchorObserverArtifact:
        raise TypeError("artifact must be exact ObjectSceneAnchorObserverArtifact")
    restored = ObjectSceneAnchorObserverArtifact.from_data(artifact.to_data())
    if restored.artifact_digest != _digest(
        expected_artifact_digest, "expected observer artifact digest"
    ):
        raise ObjectSceneAnchorObserverError(
            "observer artifact differs from commitment"
        )
    if restored.observation_plan_digest != _address(
        expected_observation_plan_digest, "expected observation plan digest"
    ):
        raise ObjectSceneAnchorObserverError(
            "observer plan differs from commitment"
        )
    if expected_runtime_identity_digest is not None and (
        restored.runtime_identity_digest
        != _digest(expected_runtime_identity_digest, "expected runtime identity")
    ):
        raise ObjectSceneAnchorObserverError(
            "observer runtime differs from commitment"
        )
    if (panel_png_bytes is None) != (inventory is None):
        raise ObjectSceneAnchorObserverError(
            "panel PNG and inventory must be supplied together"
        )
    if panel_png_bytes is not None:
        assert inventory is not None
        verify_object_scene_anchor_object_crop(
            crop_png_bytes,
            panel_png_bytes,
            inventory,
            catalog_entry,
        )
    replayed_preparation = prepare_object_scene_anchor_observer_inputs(
        crop_png_bytes,
        catalog_entry=catalog_entry,
        panel_manifest=panel_manifest,
        atlas=atlas,
        atlas_png_bytes=atlas_png_bytes,
        catalog=catalog,
        vocabulary=vocabulary,
    )
    if replayed_preparation != restored.preparation:
        raise ObjectSceneAnchorObserverError(
            "observer preparation differs from exact cold replay"
        )
    presentation_bytes = (
        ("object.png", _scene_runtime._validate_exact_png(crop_png_bytes, "object crop")),
        (
            "anchors.png",
            _scene_runtime._validate_exact_png(atlas_png_bytes, "anchor atlas"),
        ),
    )
    if _scene_runtime._image_identities(presentation_bytes) != (
        restored.passes[0].presentation
    ) or restored.passes[0].presentation != restored.passes[1].presentation:
        raise ObjectSceneAnchorObserverError(
            "observer presentation differs from exact cold replay"
        )
    prompt = object_scene_anchor_observer_prompt(
        replayed_preparation.locators, replayed_preparation.vocabulary
    )
    schema = object_scene_anchor_observer_output_schema(
        replayed_preparation.locators, replayed_preparation.vocabulary
    )
    with tempfile.TemporaryDirectory(prefix="bongard-anchor-observer-replay-") as raw:
        directory = Path(raw)
        paths: list[str] = []
        for name, data in presentation_bytes:
            target = directory / name
            target.write_bytes(data)
            paths.append(str(target.resolve()))
        for item in restored.passes:
            if item.receipt is None:
                continue
            assert item.model_payload is not None
            validate_codex_named_image_receipt(
                item.receipt,
                prompt,
                tuple(paths),
                OBJECT_SCENE_ANCHOR_OBSERVER_IMAGE_NAMES,
                schema,
                dict(item.model_payload),
            )
        for path, (_name, expected) in zip(paths, presentation_bytes, strict=True):
            if Path(path).read_bytes() != expected:
                raise ObjectSceneAnchorObserverError(
                    "observer replay presentation changed"
                )
    return restored


__all__ = (
    "OBJECT_SCENE_ANCHOR_OBSERVER_ARTIFACT_SCHEMA",
    "OBJECT_SCENE_ANCHOR_OBSERVER_CELL_SCHEMA",
    "OBJECT_SCENE_ANCHOR_OBSERVER_IMAGE_NAMES",
    "OBJECT_SCENE_ANCHOR_OBSERVER_LOCATOR_SCHEMA",
    "OBJECT_SCENE_ANCHOR_OBSERVER_MAX_CELLS",
    "OBJECT_SCENE_ANCHOR_OBSERVER_MAX_WITNESSES",
    "OBJECT_SCENE_ANCHOR_OBSERVER_PASS_SCHEMA",
    "OBJECT_SCENE_ANCHOR_OBSERVER_PREPARATION_SCHEMA",
    "OBJECT_SCENE_ANCHOR_OBSERVER_PROTOCOL_ID",
    "OBJECT_SCENE_ANCHOR_OBSERVER_VOCABULARY_ENTRY_SCHEMA",
    "OBJECT_SCENE_ANCHOR_OBSERVER_VOCABULARY_SCHEMA",
    "ObjectSceneAnchorObserverArtifact",
    "ObjectSceneAnchorObserverBindingLocator",
    "ObjectSceneAnchorObserverCell",
    "ObjectSceneAnchorObserverError",
    "ObjectSceneAnchorObserverPassArtifact",
    "ObjectSceneAnchorObserverPayloadError",
    "ObjectSceneAnchorObserverPreparation",
    "ObjectSceneAnchorObserverVocabulary",
    "ObjectSceneAnchorObserverVocabularyEntry",
    "freeze_object_scene_anchor_observer_vocabulary",
    "object_scene_anchor_observer_output_schema",
    "object_scene_anchor_observer_prompt",
    "object_scene_anchor_observer_protocol_digest",
    "object_scene_anchor_observer_source_digest",
    "observe_object_scene_anchor_catalog_twice",
    "prepare_object_scene_anchor_observer_inputs",
    "verify_object_scene_anchor_observer_artifact",
)
