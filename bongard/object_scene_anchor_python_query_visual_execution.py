"""Exact visual execution of one released query under a frozen Python predicate.

Panel geometry is extracted from released pixels before the predicate is
inspected.  Only hard-present binding catalogs are sent to the existing
role-blind, cell-aware, two-pass observer.  Hard A/I/E catalogs make no model
call.  Persistent plans and results contain commitments and typed artifacts,
never image bytes; only the runtime bundle carries presentation PNGs.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
from typing import Any, Mapping, Sequence

from bongard.canonical import canonical_digest
from bongard.evidence import Disposition
from bongard.object_scene_anchor_atlas import render_object_scene_anchor_atlas
from bongard.object_scene_anchor_batch_observer import (
    ObjectSceneAnchorBatchObserverArtifact,
    ObjectSceneAnchorBatchObserverInput,
    ObjectSceneAnchorBatchObserverPlan,
    freeze_object_scene_anchor_batch_observer_plan,
)
from bongard.object_scene_anchor_bindings import (
    ObjectSceneAnchorBindingCatalog,
    ObjectSceneAnchorWitnessCell,
    build_object_scene_anchor_binding_catalog,
)
from bongard.object_scene_anchor_catalog import (
    ObjectSceneAnchorCatalog,
    extract_object_scene_anchor_catalog,
)
from bongard.object_scene_anchor_crop import (
    render_object_scene_anchor_object_crop,
)
from bongard.object_scene_anchor_observer import (
    ObjectSceneAnchorObserverVocabulary,
    ObjectSceneAnchorObserverVocabularyEntry,
    _vocabulary_content,
    prepare_object_scene_anchor_observer_inputs,
)
from bongard.object_scene_anchor_panel_manifest import (
    ObjectSceneAnchorPanelDecisionManifest,
    build_object_scene_anchor_panel_decision_manifest,
)
from bongard.object_scene_anchor_python_bridge import (
    ObjectSceneAnchorPythonPrediction,
    object_scene_anchor_python_prediction_algorithm_digest,
    project_object_scene_anchor_python_prediction,
)
from bongard.object_scene_anchor_python_predicate import (
    ObjectSceneAnchorPythonPredicate,
)
from bongard.object_scene_anchor_python_query_observation import (
    ObjectSceneAnchorPythonQueryEvaluation,
    ObjectSceneAnchorPythonQueryObservation,
    ObjectSceneAnchorPythonQueryVocabulary,
    build_object_scene_anchor_python_query_observation,
    evaluate_object_scene_anchor_python_query_observation,
    freeze_object_scene_anchor_python_query_vocabulary,
    object_scene_anchor_python_query_algorithm_digest,
)
from bongard.object_scene_visual_frontend import (
    ObjectSceneProposalInventory,
    extract_object_scene_proposal_inventory,
)
from bongard.official_panel_archive import ReleasedOfficialPanel
from bongard.python_predicate_authority import PYTHON_PREDICATE_AUTHORITY_ID
from bongard.runtime_source_snapshot import capture_loaded_source, verify_loaded_source


_LOADED_SOURCE_SHA256 = capture_loaded_source(__name__, __file__)

OBJECT_SCENE_ANCHOR_PYTHON_QUERY_VISUAL_PLAN_SCHEMA = (
    "gkm.object-scene-anchor-python-query-visual-plan.v1"
)
OBJECT_SCENE_ANCHOR_PYTHON_QUERY_VISUAL_RESULT_SCHEMA = (
    "gkm.object-scene-anchor-python-query-visual-result.v1"
)
OBJECT_SCENE_ANCHOR_PYTHON_QUERY_VISUAL_EXECUTION_ID = (
    "bongard.object-scene-anchor-python-query-visual-execution/selected-v1"
)

_RAW_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_ADDRESS = re.compile(r"sha256:[0-9a-f]{64}\Z")
_PANEL_ALIAS = re.compile(r"panel_[0-9]{3}\Z")


class ObjectSceneAnchorPythonQueryVisualExecutionError(ValueError):
    """A released panel, visual plan, observation, or replay differs."""


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or _RAW_DIGEST.fullmatch(value) is None:
        raise ObjectSceneAnchorPythonQueryVisualExecutionError(
            f"{label} must be a raw lowercase SHA-256"
        )
    return value


def _address(value: object, label: str) -> str:
    if not isinstance(value, str) or _ADDRESS.fullmatch(value) is None:
        raise ObjectSceneAnchorPythonQueryVisualExecutionError(
            f"{label} must be a sha256: address"
        )
    return value


def _fields(value: object, expected: set[str], label: str) -> Mapping[str, Any]:
    if (
        not isinstance(value, Mapping)
        or any(not isinstance(key, str) for key in value)
        or set(value) != expected
    ):
        raise ObjectSceneAnchorPythonQueryVisualExecutionError(
            f"{label} fields differ"
        )
    return value


def _assert_no_bytes(value: object) -> None:
    if isinstance(value, bytes):
        raise ObjectSceneAnchorPythonQueryVisualExecutionError(
            "persistent query visual artifact contains image bytes"
        )
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ObjectSceneAnchorPythonQueryVisualExecutionError(
                    "persistent query visual artifact has a non-string key"
                )
            _assert_no_bytes(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _assert_no_bytes(item)


def _authority_data() -> dict[str, object]:
    return {
        "predicate_authority_id": PYTHON_PREDICATE_AUTHORITY_ID,
        "python_is_canonical_authority": True,
        "candidate_independent_geometry": True,
        "selected_predicate_vocabulary_only": True,
        "support_language_payload_present": False,
        "panel_label_present": False,
        "comparison_role_present": False,
        "query_answer_hint_present": False,
        "query_identity_model_visible": False,
        "hard_nonpresent_catalogs_call_model": False,
        "hard_present_catalogs_observed_twice": True,
        "persistent_image_bytes_present": False,
    }


def object_scene_anchor_python_query_visual_execution_source_digest() -> str:
    """Return the import-time source SHA after verifying no runtime drift."""

    return verify_loaded_source(
        __name__, expected_source_sha256=_LOADED_SOURCE_SHA256
    )


def object_scene_anchor_python_query_visual_execution_algorithm_digest() -> str:
    return canonical_digest(
        {
            "schema": (
                "gkm.object-scene-anchor-python-query-visual-algorithm.v1"
            ),
            "execution_id": OBJECT_SCENE_ANCHOR_PYTHON_QUERY_VISUAL_EXECUTION_ID,
            "source_digest": (
                object_scene_anchor_python_query_visual_execution_source_digest()
            ),
            "query_algorithm_digest": (
                object_scene_anchor_python_query_algorithm_digest()
            ),
            "prediction_algorithm_digest": (
                object_scene_anchor_python_prediction_algorithm_digest()
            ),
            "visual_call_rule": "P-catalogs-only-two-passes;A-I-E-zero-calls",
            "vocabulary_translation": "local-to-selected-by-witness-digest",
            **_authority_data(),
        }
    )


@dataclass(frozen=True, slots=True)
class ObjectSceneAnchorPythonQueryPanelInput:
    """Runtime-only released PNG plus neutral custody metadata."""

    released_panel: ReleasedOfficialPanel
    panel_alias: str
    source_binding_digest: str

    def __post_init__(self) -> None:
        if type(self.released_panel) is not ReleasedOfficialPanel:
            raise TypeError("released_panel must be exact ReleasedOfficialPanel")
        restored = ReleasedOfficialPanel.from_data(self.released_panel.to_data())
        if restored != self.released_panel:
            raise ObjectSceneAnchorPythonQueryVisualExecutionError(
                "released panel is not canonical"
            )
        if (
            not isinstance(self.panel_alias, str)
            or _PANEL_ALIAS.fullmatch(self.panel_alias) is None
        ):
            raise ObjectSceneAnchorPythonQueryVisualExecutionError(
                "query execution alias must be neutral panel_NNN"
            )
        _address(self.source_binding_digest, "released source binding digest")


def _local_vocabulary(
    selected: ObjectSceneAnchorPythonQueryVocabulary,
) -> ObjectSceneAnchorObserverVocabulary:
    entries = tuple(
        ObjectSceneAnchorObserverVocabularyEntry.create(
            f"witness_{index:02d}",
            entry.kind,
            entry.statement,
            entry.witness_digest,
        )
        for index, entry in enumerate(selected.entries)
    )
    provisional = object.__new__(ObjectSceneAnchorObserverVocabulary)
    object.__setattr__(provisional, "entries", entries)
    return ObjectSceneAnchorObserverVocabulary(
        entries, canonical_digest(_vocabulary_content(provisional))
    )


def _plan_content(
    value: "ObjectSceneAnchorPythonQueryVisualPlan",
) -> dict[str, object]:
    return {
        "schema": OBJECT_SCENE_ANCHOR_PYTHON_QUERY_VISUAL_PLAN_SCHEMA,
        "execution_id": OBJECT_SCENE_ANCHOR_PYTHON_QUERY_VISUAL_EXECUTION_ID,
        "source_digest": value.source_digest,
        "algorithm_digest": value.algorithm_digest,
        "panel_alias": value.panel_alias,
        "source_binding_digest": value.source_binding_digest,
        "released_panel_record_digest": value.released_panel_record_digest,
        "released_panel_id_digest": value.released_panel_id_digest,
        "released_panel_png_digest": value.released_panel_png_digest,
        "released_panel_png_byte_count": value.released_panel_png_byte_count,
        "release_receipt_digest": value.release_receipt_digest,
        "execution_precommit_digest": value.execution_precommit_digest,
        "exposure_successor_digest": value.exposure_successor_digest,
        "predicate": value.predicate.to_data(),
        "predicate_digest": value.predicate_digest,
        "query_vocabulary": value.query_vocabulary.to_data(),
        "query_vocabulary_digest": value.query_vocabulary_digest,
        "local_observer_vocabulary": value.local_observer_vocabulary.to_data(),
        "local_observer_vocabulary_digest": (
            value.local_observer_vocabulary_digest
        ),
        "inventory": value.inventory.to_data(),
        "inventory_digest": value.inventory_digest,
        "geometry_catalog": value.geometry_catalog.to_data(),
        "geometry_catalog_digest": value.geometry_catalog_digest,
        "panel_manifest": value.panel_manifest.to_data(),
        "panel_manifest_digest": value.panel_manifest_digest,
        "binding_catalogs": [item.to_data() for item in value.binding_catalogs],
        "preparation_digests": list(value.preparation_digests),
        "batch_plan": None if value.batch_plan is None else value.batch_plan.to_data(),
        "batch_plan_digest": value.batch_plan_digest,
        "observation_context_digest": value.observation_context_digest,
        "object_count": value.object_count,
        "present_catalog_count": value.present_catalog_count,
        "nonpresent_catalog_count": value.nonpresent_catalog_count,
        "physical_call_count": value.physical_call_count,
        "zero_present_catalogs_implies_zero_calls": True,
        "geometry_built_before_predicate_projection": True,
        "local_vocabulary_semantically_identical_by_witness_digest": True,
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class ObjectSceneAnchorPythonQueryVisualPlan:
    source_digest: str
    algorithm_digest: str
    panel_alias: str
    source_binding_digest: str
    released_panel_record_digest: str
    released_panel_id_digest: str
    released_panel_png_digest: str
    released_panel_png_byte_count: int
    release_receipt_digest: str
    execution_precommit_digest: str
    exposure_successor_digest: str
    predicate: ObjectSceneAnchorPythonPredicate
    predicate_digest: str
    query_vocabulary: ObjectSceneAnchorPythonQueryVocabulary
    query_vocabulary_digest: str
    local_observer_vocabulary: ObjectSceneAnchorObserverVocabulary
    local_observer_vocabulary_digest: str
    inventory: ObjectSceneProposalInventory
    inventory_digest: str
    geometry_catalog: ObjectSceneAnchorCatalog
    geometry_catalog_digest: str
    panel_manifest: ObjectSceneAnchorPanelDecisionManifest
    panel_manifest_digest: str
    binding_catalogs: tuple[ObjectSceneAnchorBindingCatalog, ...]
    preparation_digests: tuple[str | None, ...]
    batch_plan: ObjectSceneAnchorBatchObserverPlan | None
    batch_plan_digest: str | None
    observation_context_digest: str
    object_count: int
    present_catalog_count: int
    nonpresent_catalog_count: int
    physical_call_count: int
    plan_digest: str

    def __post_init__(self) -> None:
        for item, label in (
            (self.source_digest, "query visual source digest"),
            (self.algorithm_digest, "query visual algorithm digest"),
            (self.released_panel_id_digest, "released panel ID digest"),
            (self.predicate_digest, "predicate digest"),
            (self.query_vocabulary_digest, "query vocabulary digest"),
            (self.local_observer_vocabulary_digest, "local vocabulary digest"),
            (self.inventory_digest, "inventory digest"),
            (self.geometry_catalog_digest, "geometry catalog digest"),
            (self.panel_manifest_digest, "panel manifest digest"),
            (self.plan_digest, "query visual plan digest"),
        ):
            _digest(item, label)
        for item, label in (
            (self.source_binding_digest, "source binding digest"),
            (self.released_panel_record_digest, "released panel record digest"),
            (self.released_panel_png_digest, "released panel PNG digest"),
            (self.release_receipt_digest, "release receipt digest"),
            (self.execution_precommit_digest, "execution precommit digest"),
            (self.exposure_successor_digest, "exposure successor digest"),
            (self.observation_context_digest, "observation context digest"),
        ):
            _address(item, label)
        if _PANEL_ALIAS.fullmatch(self.panel_alias) is None:
            raise ObjectSceneAnchorPythonQueryVisualExecutionError(
                "query visual plan alias differs"
            )
        for item, label, minimum in (
            (self.released_panel_png_byte_count, "panel byte count", 1),
            (self.object_count, "object count", 0),
            (self.present_catalog_count, "present catalog count", 0),
            (self.nonpresent_catalog_count, "nonpresent catalog count", 0),
            (self.physical_call_count, "physical call count", 0),
        ):
            if type(item) is not int or item < minimum:
                raise ObjectSceneAnchorPythonQueryVisualExecutionError(
                    f"{label} differs"
                )
        if type(self.predicate) is not ObjectSceneAnchorPythonPredicate:
            raise TypeError("query visual predicate has the wrong type")
        if type(self.query_vocabulary) is not ObjectSceneAnchorPythonQueryVocabulary:
            raise TypeError("query vocabulary has the wrong type")
        if type(self.local_observer_vocabulary) is not ObjectSceneAnchorObserverVocabulary:
            raise TypeError("local observer vocabulary has the wrong type")
        if type(self.inventory) is not ObjectSceneProposalInventory:
            raise TypeError("query inventory has the wrong type")
        if type(self.geometry_catalog) is not ObjectSceneAnchorCatalog:
            raise TypeError("query geometry catalog has the wrong type")
        if type(self.panel_manifest) is not ObjectSceneAnchorPanelDecisionManifest:
            raise TypeError("query panel manifest has the wrong type")
        predicate = ObjectSceneAnchorPythonPredicate.from_data(
            self.predicate.to_data()
        )
        selected = ObjectSceneAnchorPythonQueryVocabulary.from_data(
            self.query_vocabulary.to_data()
        )
        local = ObjectSceneAnchorObserverVocabulary.from_data(
            self.local_observer_vocabulary.to_data()
        )
        inventory = ObjectSceneProposalInventory.from_data(self.inventory.to_data())
        geometry = ObjectSceneAnchorCatalog.from_data(self.geometry_catalog.to_data())
        manifest = ObjectSceneAnchorPanelDecisionManifest.from_data(
            self.panel_manifest.to_data()
        )
        if (
            self.source_digest
            != object_scene_anchor_python_query_visual_execution_source_digest()
            or self.algorithm_digest
            != object_scene_anchor_python_query_visual_execution_algorithm_digest()
            or predicate != self.predicate
            or predicate.predicate_digest != self.predicate_digest
            or selected != freeze_object_scene_anchor_python_query_vocabulary(predicate)
            or selected.vocabulary_digest != self.query_vocabulary_digest
            or local != _local_vocabulary(selected)
            or local.vocabulary_digest != self.local_observer_vocabulary_digest
            or tuple(
                (item.kind, item.statement, item.witness_digest)
                for item in local.entries
            )
            != tuple(
                (item.kind, item.statement, item.witness_digest)
                for item in selected.entries
            )
            or inventory != self.inventory
            or inventory.inventory_digest != self.inventory_digest
            or geometry != self.geometry_catalog
            or geometry.inventory_digest != inventory.inventory_digest
            or geometry.catalog_digest != self.geometry_catalog_digest
            or manifest != self.panel_manifest
            or manifest.inventory_digest != inventory.inventory_digest
            or manifest.manifest_digest != self.panel_manifest_digest
            or inventory.panel_digest != self.released_panel_png_digest.removeprefix("sha256:")
            or geometry.panel_digest != inventory.panel_digest
            or manifest.panel_digest != inventory.panel_digest
        ):
            raise ObjectSceneAnchorPythonQueryVisualExecutionError(
                "query visual authority, vocabulary, or geometry binding differs"
            )
        expected_catalogs = tuple(
            build_object_scene_anchor_binding_catalog(
                decision,
                predicate.binding_spec,
                expected_object_id=object_id,
            )
            for object_id, decision in zip(
                manifest.object_ids, manifest.object_decisions, strict=True
            )
        )
        if (
            type(self.binding_catalogs) is not tuple
            or self.binding_catalogs != expected_catalogs
            or type(self.preparation_digests) is not tuple
            or len(self.preparation_digests) != len(expected_catalogs)
        ):
            raise ObjectSceneAnchorPythonQueryVisualExecutionError(
                "query binding catalog inventory differs"
            )
        present = tuple(
            index
            for index, item in enumerate(expected_catalogs)
            if item.hard_disposition is Disposition.PRESENT
        )
        if any(
            (self.preparation_digests[index] is not None) != (index in present)
            for index in range(len(expected_catalogs))
        ):
            raise ObjectSceneAnchorPythonQueryVisualExecutionError(
                "query preparation scope is not exactly the P catalogs"
            )
        for item in self.preparation_digests:
            if item is not None:
                _digest(item, "query preparation digest")
        if self.batch_plan is None:
            if (
                present
                or self.batch_plan_digest is not None
                or self.physical_call_count != 0
            ):
                raise ObjectSceneAnchorPythonQueryVisualExecutionError(
                    "zero-P query plan does not have zero calls"
                )
        else:
            if type(self.batch_plan) is not ObjectSceneAnchorBatchObserverPlan:
                raise TypeError("query batch plan has the wrong type")
            batch = ObjectSceneAnchorBatchObserverPlan.from_data(
                self.batch_plan.to_data()
            )
            if (
                not present
                or batch != self.batch_plan
                or batch.plan_digest != self.batch_plan_digest
                or batch.vocabulary != local
                or {item.preparation_digest for item in batch.preparations}
                != {item for item in self.preparation_digests if item is not None}
                or len(batch.preparations) != len(present)
                or self.physical_call_count != batch.physical_call_count
            ):
                raise ObjectSceneAnchorPythonQueryVisualExecutionError(
                    "query batch plan differs from exact P catalog scope"
                )
        if (
            self.object_count != len(manifest.object_ids)
            or self.present_catalog_count != len(present)
            or self.nonpresent_catalog_count != len(expected_catalogs) - len(present)
            or self.object_count
            != self.present_catalog_count + self.nonpresent_catalog_count
        ):
            raise ObjectSceneAnchorPythonQueryVisualExecutionError(
                "query visual plan counts differ"
            )
        unsigned = _plan_content(self)
        _assert_no_bytes(unsigned)
        if self.plan_digest != canonical_digest(unsigned):
            raise ObjectSceneAnchorPythonQueryVisualExecutionError(
                "query visual plan digest differs"
            )

    def to_data(self) -> dict[str, object]:
        return {**_plan_content(self), "plan_digest": self.plan_digest}

    @classmethod
    def from_data(cls, value: object) -> "ObjectSceneAnchorPythonQueryVisualPlan":
        raw = _fields(
            value,
            {
                "schema", "execution_id", "source_digest", "algorithm_digest",
                "panel_alias", "source_binding_digest",
                "released_panel_record_digest", "released_panel_id_digest",
                "released_panel_png_digest", "released_panel_png_byte_count",
                "release_receipt_digest", "execution_precommit_digest",
                "exposure_successor_digest", "predicate", "predicate_digest",
                "query_vocabulary", "query_vocabulary_digest",
                "local_observer_vocabulary", "local_observer_vocabulary_digest",
                "inventory", "inventory_digest", "geometry_catalog",
                "geometry_catalog_digest", "panel_manifest",
                "panel_manifest_digest", "binding_catalogs",
                "preparation_digests", "batch_plan", "batch_plan_digest",
                "observation_context_digest", "object_count",
                "present_catalog_count", "nonpresent_catalog_count",
                "physical_call_count", "zero_present_catalogs_implies_zero_calls",
                "geometry_built_before_predicate_projection",
                "local_vocabulary_semantically_identical_by_witness_digest",
                *_authority_data(), "plan_digest",
            },
            "query visual plan",
        )
        if (
            raw["schema"] != OBJECT_SCENE_ANCHOR_PYTHON_QUERY_VISUAL_PLAN_SCHEMA
            or raw["execution_id"]
            != OBJECT_SCENE_ANCHOR_PYTHON_QUERY_VISUAL_EXECUTION_ID
            or raw["zero_present_catalogs_implies_zero_calls"] is not True
            or raw["geometry_built_before_predicate_projection"] is not True
            or raw[
                "local_vocabulary_semantically_identical_by_witness_digest"
            ] is not True
            or any(raw[key] != item for key, item in _authority_data().items())
            or not isinstance(raw["predicate"], Mapping)
            or not isinstance(raw["query_vocabulary"], Mapping)
            or not isinstance(raw["local_observer_vocabulary"], Mapping)
            or not isinstance(raw["inventory"], Mapping)
            or not isinstance(raw["geometry_catalog"], Mapping)
            or not isinstance(raw["panel_manifest"], Mapping)
            or not isinstance(raw["binding_catalogs"], list)
            or not isinstance(raw["preparation_digests"], list)
            or (
                raw["batch_plan"] is not None
                and not isinstance(raw["batch_plan"], Mapping)
            )
        ):
            raise ObjectSceneAnchorPythonQueryVisualExecutionError(
                "query visual plan policy differs"
            )
        result = cls(
            raw["source_digest"], raw["algorithm_digest"], raw["panel_alias"],
            raw["source_binding_digest"], raw["released_panel_record_digest"],
            raw["released_panel_id_digest"], raw["released_panel_png_digest"],
            raw["released_panel_png_byte_count"], raw["release_receipt_digest"],
            raw["execution_precommit_digest"], raw["exposure_successor_digest"],
            ObjectSceneAnchorPythonPredicate.from_data(raw["predicate"]),
            raw["predicate_digest"],
            ObjectSceneAnchorPythonQueryVocabulary.from_data(
                raw["query_vocabulary"]
            ),
            raw["query_vocabulary_digest"],
            ObjectSceneAnchorObserverVocabulary.from_data(
                raw["local_observer_vocabulary"]
            ),
            raw["local_observer_vocabulary_digest"],
            ObjectSceneProposalInventory.from_data(raw["inventory"]),
            raw["inventory_digest"],
            ObjectSceneAnchorCatalog.from_data(raw["geometry_catalog"]),
            raw["geometry_catalog_digest"],
            ObjectSceneAnchorPanelDecisionManifest.from_data(
                raw["panel_manifest"]
            ),
            raw["panel_manifest_digest"],
            tuple(
                ObjectSceneAnchorBindingCatalog.from_data(item)
                for item in raw["binding_catalogs"]
            ),
            tuple(raw["preparation_digests"]),
            None
            if raw["batch_plan"] is None
            else ObjectSceneAnchorBatchObserverPlan.from_data(raw["batch_plan"]),
            raw["batch_plan_digest"], raw["observation_context_digest"],
            raw["object_count"], raw["present_catalog_count"],
            raw["nonpresent_catalog_count"], raw["physical_call_count"],
            raw["plan_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectSceneAnchorPythonQueryVisualExecutionError(
                "query visual plan is not canonical"
            )
        return result


@dataclass(frozen=True, slots=True)
class ObjectSceneAnchorPythonQueryVisualRuntimeBundle:
    """Persistent plan plus the exact released/crop/atlas runtime PNGs."""

    plan: ObjectSceneAnchorPythonQueryVisualPlan
    panel_input: ObjectSceneAnchorPythonQueryPanelInput
    batch_inputs: tuple[ObjectSceneAnchorBatchObserverInput, ...]

    def __post_init__(self) -> None:
        if type(self.plan) is not ObjectSceneAnchorPythonQueryVisualPlan:
            raise TypeError("query visual runtime plan has the wrong type")
        if type(self.panel_input) is not ObjectSceneAnchorPythonQueryPanelInput:
            raise TypeError("query visual panel input has the wrong type")
        if (
            type(self.batch_inputs) is not tuple
            or any(
                type(item) is not ObjectSceneAnchorBatchObserverInput
                for item in self.batch_inputs
            )
        ):
            raise ObjectSceneAnchorPythonQueryVisualExecutionError(
                "query visual runtime batch inputs differ"
            )
        panel = self.panel_input.released_panel
        if (
            self.plan.panel_alias != self.panel_input.panel_alias
            or self.plan.source_binding_digest
            != self.panel_input.source_binding_digest
            or self.plan.released_panel_record_digest != panel.record_digest
            or self.plan.released_panel_png_digest != panel.exact_png_digest
            or self.plan.released_panel_png_byte_count
            != len(panel.exact_png_bytes)
        ):
            raise ObjectSceneAnchorPythonQueryVisualExecutionError(
                "query visual runtime released panel differs from plan"
            )
        expected = (
            ()
            if self.plan.batch_plan is None
            else self.plan.batch_plan.preparations
        )
        if tuple(item.preparation for item in self.batch_inputs) != expected:
            raise ObjectSceneAnchorPythonQueryVisualExecutionError(
                "query visual runtime presentations differ from plan"
            )


def _make_plan(
    *,
    panel_input: ObjectSceneAnchorPythonQueryPanelInput,
    predicate: ObjectSceneAnchorPythonPredicate,
    inventory: ObjectSceneProposalInventory,
    geometry: ObjectSceneAnchorCatalog,
    manifest: ObjectSceneAnchorPanelDecisionManifest,
    selected: ObjectSceneAnchorPythonQueryVocabulary,
    local: ObjectSceneAnchorObserverVocabulary,
    catalogs: tuple[ObjectSceneAnchorBindingCatalog, ...],
    preparation_digests: tuple[str | None, ...],
    batch_plan: ObjectSceneAnchorBatchObserverPlan | None,
) -> ObjectSceneAnchorPythonQueryVisualPlan:
    released = panel_input.released_panel
    batch_digest = None if batch_plan is None else batch_plan.plan_digest
    context = "sha256:" + canonical_digest(
        {
            "schema": "gkm.object-scene-anchor-query-visual-context.v1",
            "released_panel_record_digest": released.record_digest,
            "source_binding_digest": panel_input.source_binding_digest,
            "panel_alias": panel_input.panel_alias,
            "predicate_digest": predicate.predicate_digest,
            "panel_manifest_digest": manifest.manifest_digest,
            "batch_plan_digest": batch_digest,
        }
    )
    present_count = sum(
        item.hard_disposition is Disposition.PRESENT for item in catalogs
    )
    values = {
        "source_digest": (
            object_scene_anchor_python_query_visual_execution_source_digest()
        ),
        "algorithm_digest": (
            object_scene_anchor_python_query_visual_execution_algorithm_digest()
        ),
        "panel_alias": panel_input.panel_alias,
        "source_binding_digest": panel_input.source_binding_digest,
        "released_panel_record_digest": released.record_digest,
        "released_panel_id_digest": canonical_digest(
            {"schema": "gkm.official-panel-id-binding.v1", "panel_id": released.panel_id}
        ),
        "released_panel_png_digest": released.exact_png_digest,
        "released_panel_png_byte_count": len(released.exact_png_bytes),
        "release_receipt_digest": released.release_receipt.record_digest,
        "execution_precommit_digest": released.execution_precommit_digest,
        "exposure_successor_digest": released.exposure_successor_digest,
        "predicate": predicate,
        "predicate_digest": predicate.predicate_digest,
        "query_vocabulary": selected,
        "query_vocabulary_digest": selected.vocabulary_digest,
        "local_observer_vocabulary": local,
        "local_observer_vocabulary_digest": local.vocabulary_digest,
        "inventory": inventory,
        "inventory_digest": inventory.inventory_digest,
        "geometry_catalog": geometry,
        "geometry_catalog_digest": geometry.catalog_digest,
        "panel_manifest": manifest,
        "panel_manifest_digest": manifest.manifest_digest,
        "binding_catalogs": catalogs,
        "preparation_digests": preparation_digests,
        "batch_plan": batch_plan,
        "batch_plan_digest": batch_digest,
        "observation_context_digest": context,
        "object_count": len(manifest.object_ids),
        "present_catalog_count": present_count,
        "nonpresent_catalog_count": len(catalogs) - present_count,
        "physical_call_count": (
            0 if batch_plan is None else batch_plan.physical_call_count
        ),
    }
    provisional = object.__new__(ObjectSceneAnchorPythonQueryVisualPlan)
    for name, item in values.items():
        object.__setattr__(provisional, name, item)
    return ObjectSceneAnchorPythonQueryVisualPlan(
        **values, plan_digest=canonical_digest(_plan_content(provisional))
    )


def build_object_scene_anchor_python_query_visual_plan(
    panel_input: ObjectSceneAnchorPythonQueryPanelInput,
    predicate: ObjectSceneAnchorPythonPredicate,
) -> ObjectSceneAnchorPythonQueryVisualRuntimeBundle:
    """Extract neutral geometry, then project only the frozen selected predicate."""

    if type(panel_input) is not ObjectSceneAnchorPythonQueryPanelInput:
        raise TypeError("panel_input must be exact query panel input")
    if type(predicate) is not ObjectSceneAnchorPythonPredicate:
        raise TypeError("predicate must be exact ObjectSceneAnchorPythonPredicate")
    released = ReleasedOfficialPanel.from_data(panel_input.released_panel.to_data())
    png = released.exact_png_bytes

    # This complete geometry stack is intentionally built before predicate use.
    inventory = extract_object_scene_proposal_inventory(png)
    geometry = extract_object_scene_anchor_catalog(png, inventory)
    manifest = build_object_scene_anchor_panel_decision_manifest(
        geometry, png, inventory
    )

    frozen_predicate = ObjectSceneAnchorPythonPredicate.from_data(
        predicate.to_data()
    )
    selected = freeze_object_scene_anchor_python_query_vocabulary(
        frozen_predicate
    )
    local = _local_vocabulary(selected)
    catalogs = tuple(
        build_object_scene_anchor_binding_catalog(
            decision,
            frozen_predicate.binding_spec,
            expected_object_id=object_id,
        )
        for object_id, decision in zip(
            manifest.object_ids, manifest.object_decisions, strict=True
        )
    )
    entry_by_object = geometry.by_object_id
    inputs: list[ObjectSceneAnchorBatchObserverInput] = []
    preparation_digests: list[str | None] = []
    for object_id, decision, catalog in zip(
        manifest.object_ids, manifest.object_decisions, catalogs, strict=True
    ):
        if catalog.hard_disposition is not Disposition.PRESENT:
            preparation_digests.append(None)
            continue
        entry = entry_by_object[object_id]
        crop_png = render_object_scene_anchor_object_crop(png, inventory, entry)
        atlas, atlas_png = render_object_scene_anchor_atlas(decision)
        if atlas_png is None:
            raise ObjectSceneAnchorPythonQueryVisualExecutionError(
                "hard-present query catalog has no clean anchor atlas"
            )
        preparation = prepare_object_scene_anchor_observer_inputs(
            crop_png,
            catalog_entry=entry,
            panel_manifest=manifest,
            atlas=atlas,
            atlas_png_bytes=atlas_png,
            catalog=catalog,
            vocabulary=local,
        )
        inputs.append(
            ObjectSceneAnchorBatchObserverInput(
                preparation, crop_png, atlas_png
            )
        )
        preparation_digests.append(preparation.preparation_digest)
    batch_plan = (
        None
        if not inputs
        else freeze_object_scene_anchor_batch_observer_plan(tuple(inputs))
    )
    plan = _make_plan(
        panel_input=panel_input,
        predicate=frozen_predicate,
        inventory=inventory,
        geometry=geometry,
        manifest=manifest,
        selected=selected,
        local=local,
        catalogs=catalogs,
        preparation_digests=tuple(preparation_digests),
        batch_plan=batch_plan,
    )
    by_digest = {
        item.preparation.preparation_digest: item for item in inputs
    }
    ordered = (
        ()
        if batch_plan is None
        else tuple(
            by_digest[item.preparation_digest]
            for item in batch_plan.preparations
        )
    )
    return ObjectSceneAnchorPythonQueryVisualRuntimeBundle(
        plan, panel_input, ordered
    )


def verify_object_scene_anchor_python_query_visual_runtime(
    bundle: ObjectSceneAnchorPythonQueryVisualRuntimeBundle,
    *,
    panel_input: ObjectSceneAnchorPythonQueryPanelInput,
    predicate: ObjectSceneAnchorPythonPredicate,
    expected_plan_digest: str,
) -> ObjectSceneAnchorPythonQueryVisualRuntimeBundle:
    """Cold replay geometry and every runtime presentation from released bytes."""

    if type(bundle) is not ObjectSceneAnchorPythonQueryVisualRuntimeBundle:
        raise TypeError("bundle must be exact query visual runtime bundle")
    restored = ObjectSceneAnchorPythonQueryVisualPlan.from_data(
        bundle.plan.to_data()
    )
    if restored.plan_digest != _digest(
        expected_plan_digest, "expected query visual plan digest"
    ):
        raise ObjectSceneAnchorPythonQueryVisualExecutionError(
            "query visual plan differs from external commitment"
        )
    replayed = build_object_scene_anchor_python_query_visual_plan(
        panel_input, predicate
    )
    if replayed != bundle:
        raise ObjectSceneAnchorPythonQueryVisualExecutionError(
            "query visual runtime differs from cold pixel replay"
        )
    return replayed


def _result_content(
    value: "ObjectSceneAnchorPythonQueryVisualResult",
) -> dict[str, object]:
    return {
        "schema": OBJECT_SCENE_ANCHOR_PYTHON_QUERY_VISUAL_RESULT_SCHEMA,
        "execution_id": OBJECT_SCENE_ANCHOR_PYTHON_QUERY_VISUAL_EXECUTION_ID,
        "source_digest": value.source_digest,
        "algorithm_digest": value.algorithm_digest,
        "plan_digest": value.plan_digest,
        "panel_alias": value.panel_alias,
        "source_binding_digest": value.source_binding_digest,
        "released_panel_record_digest": value.released_panel_record_digest,
        "predicate_digest": value.predicate_digest,
        "batch_artifact_digest": value.batch_artifact_digest,
        "physical_call_count": value.physical_call_count,
        "query_observation": value.query_observation.to_data(),
        "query_observation_digest": value.query_observation_digest,
        "query_evaluation": value.query_evaluation.to_data(),
        "query_evaluation_digest": value.query_evaluation_digest,
        "prediction": value.prediction.to_data(),
        "prediction_digest": value.prediction_digest,
        "local_cells_translated_by_witness_digest": True,
        "zero_present_catalogs_implies_zero_calls": True,
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class ObjectSceneAnchorPythonQueryVisualResult:
    source_digest: str
    algorithm_digest: str
    plan_digest: str
    panel_alias: str
    source_binding_digest: str
    released_panel_record_digest: str
    predicate_digest: str
    batch_artifact_digest: str | None
    physical_call_count: int
    query_observation: ObjectSceneAnchorPythonQueryObservation
    query_observation_digest: str
    query_evaluation: ObjectSceneAnchorPythonQueryEvaluation
    query_evaluation_digest: str
    prediction: ObjectSceneAnchorPythonPrediction
    prediction_digest: str
    result_digest: str

    def __post_init__(self) -> None:
        for item, label in (
            (self.source_digest, "result source digest"),
            (self.algorithm_digest, "result algorithm digest"),
            (self.plan_digest, "result plan digest"),
            (self.predicate_digest, "result predicate digest"),
            (self.query_observation_digest, "query observation digest"),
            (self.query_evaluation_digest, "query evaluation digest"),
            (self.prediction_digest, "prediction digest"),
            (self.result_digest, "query visual result digest"),
        ):
            _digest(item, label)
        _address(self.source_binding_digest, "result source binding digest")
        _address(
            self.released_panel_record_digest, "result released panel digest"
        )
        if self.batch_artifact_digest is not None:
            _digest(self.batch_artifact_digest, "result batch artifact digest")
        if type(self.physical_call_count) is not int or self.physical_call_count < 0:
            raise ObjectSceneAnchorPythonQueryVisualExecutionError(
                "result physical call count differs"
            )
        if _PANEL_ALIAS.fullmatch(self.panel_alias) is None:
            raise ObjectSceneAnchorPythonQueryVisualExecutionError(
                "result panel alias differs"
            )
        if type(self.query_observation) is not ObjectSceneAnchorPythonQueryObservation:
            raise TypeError("result query observation has the wrong type")
        if type(self.query_evaluation) is not ObjectSceneAnchorPythonQueryEvaluation:
            raise TypeError("result query evaluation has the wrong type")
        if type(self.prediction) is not ObjectSceneAnchorPythonPrediction:
            raise TypeError("result prediction has the wrong type")
        observation = ObjectSceneAnchorPythonQueryObservation.from_data(
            self.query_observation.to_data()
        )
        evaluation = ObjectSceneAnchorPythonQueryEvaluation.from_data(
            self.query_evaluation.to_data()
        )
        prediction = ObjectSceneAnchorPythonPrediction.from_data(
            self.prediction.to_data()
        )
        if (
            self.source_digest
            != object_scene_anchor_python_query_visual_execution_source_digest()
            or self.algorithm_digest
            != object_scene_anchor_python_query_visual_execution_algorithm_digest()
            or observation.predicate_digest != self.predicate_digest
            or observation.panel_id != self.panel_alias
            or observation.observation_digest != self.query_observation_digest
            or evaluation.predicate_digest != self.predicate_digest
            or evaluation.panel_id != self.panel_alias
            or evaluation.observation_digest != observation.observation_digest
            or evaluation.evaluation_digest != self.query_evaluation_digest
            or prediction.predicate_digest != self.predicate_digest
            or prediction.query_record_digest != evaluation.evaluation_digest
            or prediction.prediction_digest != self.prediction_digest
        ):
            raise ObjectSceneAnchorPythonQueryVisualExecutionError(
                "query visual result chain differs"
            )
        unsigned = _result_content(self)
        _assert_no_bytes(unsigned)
        if self.result_digest != canonical_digest(unsigned):
            raise ObjectSceneAnchorPythonQueryVisualExecutionError(
                "query visual result digest differs"
            )

    def to_data(self) -> dict[str, object]:
        return {**_result_content(self), "result_digest": self.result_digest}

    @classmethod
    def from_data(cls, value: object) -> "ObjectSceneAnchorPythonQueryVisualResult":
        raw = _fields(
            value,
            {
                "schema", "execution_id", "source_digest", "algorithm_digest",
                "plan_digest", "panel_alias", "source_binding_digest",
                "released_panel_record_digest", "predicate_digest",
                "batch_artifact_digest", "physical_call_count",
                "query_observation", "query_observation_digest",
                "query_evaluation", "query_evaluation_digest", "prediction",
                "prediction_digest", "local_cells_translated_by_witness_digest",
                "zero_present_catalogs_implies_zero_calls", *_authority_data(),
                "result_digest",
            },
            "query visual result",
        )
        if (
            raw["schema"] != OBJECT_SCENE_ANCHOR_PYTHON_QUERY_VISUAL_RESULT_SCHEMA
            or raw["execution_id"]
            != OBJECT_SCENE_ANCHOR_PYTHON_QUERY_VISUAL_EXECUTION_ID
            or raw["local_cells_translated_by_witness_digest"] is not True
            or raw["zero_present_catalogs_implies_zero_calls"] is not True
            or any(raw[key] != item for key, item in _authority_data().items())
            or not isinstance(raw["query_observation"], Mapping)
            or not isinstance(raw["query_evaluation"], Mapping)
            or not isinstance(raw["prediction"], Mapping)
        ):
            raise ObjectSceneAnchorPythonQueryVisualExecutionError(
                "query visual result policy differs"
            )
        result = cls(
            raw["source_digest"], raw["algorithm_digest"], raw["plan_digest"],
            raw["panel_alias"], raw["source_binding_digest"],
            raw["released_panel_record_digest"], raw["predicate_digest"],
            raw["batch_artifact_digest"], raw["physical_call_count"],
            ObjectSceneAnchorPythonQueryObservation.from_data(
                raw["query_observation"]
            ),
            raw["query_observation_digest"],
            ObjectSceneAnchorPythonQueryEvaluation.from_data(
                raw["query_evaluation"]
            ),
            raw["query_evaluation_digest"],
            ObjectSceneAnchorPythonPrediction.from_data(raw["prediction"]),
            raw["prediction_digest"], raw["result_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectSceneAnchorPythonQueryVisualExecutionError(
                "query visual result is not canonical"
            )
        return result


def _translated_cells(
    plan: ObjectSceneAnchorPythonQueryVisualPlan,
    artifact: ObjectSceneAnchorBatchObserverArtifact | None,
) -> tuple[ObjectSceneAnchorWitnessCell, ...]:
    if plan.batch_plan is None:
        if artifact is not None:
            raise ObjectSceneAnchorPythonQueryVisualExecutionError(
                "zero-P query supplied a model artifact"
            )
        return ()
    if type(artifact) is not ObjectSceneAnchorBatchObserverArtifact:
        raise TypeError("P-catalog query requires an exact batch artifact")
    restored = ObjectSceneAnchorBatchObserverArtifact.from_data(
        artifact.to_data()
    )
    if (
        restored.plan != plan.batch_plan
        or restored.plan_digest != plan.batch_plan_digest
        or restored.observation_plan_digest != plan.observation_context_digest
        or restored.physical_call_count != plan.physical_call_count
    ):
        raise ObjectSceneAnchorPythonQueryVisualExecutionError(
            "query observer artifact differs from exact visual plan"
        )
    by_key = {}
    for batch_result in restored.results:
        for cell in batch_result.merged_cells:
            key = (
                cell.locator.catalog_digest,
                cell.locator.binding_digest,
                cell.witness.witness_digest,
            )
            if key in by_key:
                raise ObjectSceneAnchorPythonQueryVisualExecutionError(
                    "query observer artifact repeats a semantic cell"
                )
            by_key[key] = cell
    translated = []
    expected_keys = set()
    for catalog in plan.binding_catalogs:
        if catalog.hard_disposition is not Disposition.PRESENT:
            continue
        for binding in catalog.bindings:
            for entry in plan.query_vocabulary.entries:
                key = (
                    catalog.catalog_digest,
                    binding.binding_digest,
                    entry.witness_digest,
                )
                expected_keys.add(key)
                try:
                    local_cell = by_key[key]
                except KeyError as exc:
                    raise ObjectSceneAnchorPythonQueryVisualExecutionError(
                        "query observer artifact omits a selected semantic cell"
                    ) from exc
                translated.append(
                    ObjectSceneAnchorWitnessCell.create(
                        binding,
                        entry.binding_witness_spec,
                        local_cell.binding_cell.disposition,
                    )
                )
    if set(by_key) != expected_keys:
        raise ObjectSceneAnchorPythonQueryVisualExecutionError(
            "query observer artifact contains an extra semantic cell"
        )
    return tuple(translated)


def finalize_object_scene_anchor_python_query_visual_execution(
    plan: ObjectSceneAnchorPythonQueryVisualPlan,
    artifact: ObjectSceneAnchorBatchObserverArtifact | None,
) -> ObjectSceneAnchorPythonQueryVisualResult:
    """Join P-only observer cells with typed A/I/E catalogs and predict."""

    if type(plan) is not ObjectSceneAnchorPythonQueryVisualPlan:
        raise TypeError("plan must be exact query visual plan")
    frozen = ObjectSceneAnchorPythonQueryVisualPlan.from_data(plan.to_data())
    cells = _translated_cells(frozen, artifact)
    observation = build_object_scene_anchor_python_query_observation(
        predicate=frozen.predicate,
        panel_id=frozen.panel_alias,
        panel_manifest=frozen.panel_manifest,
        cells=cells,
    )
    evaluation = evaluate_object_scene_anchor_python_query_observation(
        frozen.predicate, observation
    )
    prediction = project_object_scene_anchor_python_prediction(
        frozen.predicate, evaluation
    )
    artifact_digest = None if artifact is None else artifact.artifact_digest
    values = {
        "source_digest": (
            object_scene_anchor_python_query_visual_execution_source_digest()
        ),
        "algorithm_digest": (
            object_scene_anchor_python_query_visual_execution_algorithm_digest()
        ),
        "plan_digest": frozen.plan_digest,
        "panel_alias": frozen.panel_alias,
        "source_binding_digest": frozen.source_binding_digest,
        "released_panel_record_digest": frozen.released_panel_record_digest,
        "predicate_digest": frozen.predicate_digest,
        "batch_artifact_digest": artifact_digest,
        "physical_call_count": frozen.physical_call_count,
        "query_observation": observation,
        "query_observation_digest": observation.observation_digest,
        "query_evaluation": evaluation,
        "query_evaluation_digest": evaluation.evaluation_digest,
        "prediction": prediction,
        "prediction_digest": prediction.prediction_digest,
    }
    provisional = object.__new__(ObjectSceneAnchorPythonQueryVisualResult)
    for name, item in values.items():
        object.__setattr__(provisional, name, item)
    return ObjectSceneAnchorPythonQueryVisualResult(
        **values, result_digest=canonical_digest(_result_content(provisional))
    )


def cold_verify_object_scene_anchor_python_query_visual_result(
    result: ObjectSceneAnchorPythonQueryVisualResult,
    *,
    plan: ObjectSceneAnchorPythonQueryVisualPlan,
    artifact: ObjectSceneAnchorBatchObserverArtifact | None,
    expected_result_digest: str,
) -> ObjectSceneAnchorPythonQueryVisualResult:
    """Rebuild observation, evaluation, and prediction from exact parents."""

    if type(result) is not ObjectSceneAnchorPythonQueryVisualResult:
        raise TypeError("result must be exact query visual result")
    restored = ObjectSceneAnchorPythonQueryVisualResult.from_data(
        result.to_data()
    )
    if restored.result_digest != _digest(
        expected_result_digest, "expected query visual result digest"
    ):
        raise ObjectSceneAnchorPythonQueryVisualExecutionError(
            "query visual result differs from external commitment"
        )
    expected = finalize_object_scene_anchor_python_query_visual_execution(
        plan, artifact
    )
    if restored != expected:
        raise ObjectSceneAnchorPythonQueryVisualExecutionError(
            "query visual result differs from cold replay"
        )
    return restored


# Explicit noun aliases keep runner call sites readable without changing semantics.
finalize_object_scene_anchor_python_query_visual_result = (
    finalize_object_scene_anchor_python_query_visual_execution
)
cold_verify_object_scene_anchor_python_query_visual_execution = (
    cold_verify_object_scene_anchor_python_query_visual_result
)


__all__ = (
    "OBJECT_SCENE_ANCHOR_PYTHON_QUERY_VISUAL_EXECUTION_ID",
    "OBJECT_SCENE_ANCHOR_PYTHON_QUERY_VISUAL_PLAN_SCHEMA",
    "OBJECT_SCENE_ANCHOR_PYTHON_QUERY_VISUAL_RESULT_SCHEMA",
    "ObjectSceneAnchorPythonQueryPanelInput",
    "ObjectSceneAnchorPythonQueryVisualExecutionError",
    "ObjectSceneAnchorPythonQueryVisualPlan",
    "ObjectSceneAnchorPythonQueryVisualResult",
    "ObjectSceneAnchorPythonQueryVisualRuntimeBundle",
    "build_object_scene_anchor_python_query_visual_plan",
    "cold_verify_object_scene_anchor_python_query_visual_execution",
    "cold_verify_object_scene_anchor_python_query_visual_result",
    "finalize_object_scene_anchor_python_query_visual_execution",
    "finalize_object_scene_anchor_python_query_visual_result",
    "object_scene_anchor_python_query_visual_execution_algorithm_digest",
    "object_scene_anchor_python_query_visual_execution_source_digest",
    "verify_object_scene_anchor_python_query_visual_runtime",
)
