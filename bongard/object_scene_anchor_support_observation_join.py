"""Deterministically join support preparations, observations, and version spaces.

The persistent plan contains the exact support corpus freeze, projected
predicate language, every panel/spec/object binding catalog, and the role-blind
batch plan.  Runtime crop and atlas PNGs live only in a companion bundle.
Finalization consumes the persistent plan and an exact batch artifact, fills
hard A/I/E catalogs with zero-row matrices, and builds both explicit support
orientations without another visual call.
"""

from __future__ import annotations

from bongard.runtime_source_snapshot import capture_loaded_source, verify_loaded_source


_LOADED_SOURCE_SHA256 = capture_loaded_source(__name__, __file__)

from dataclasses import dataclass
import re
from typing import Any, Mapping, Sequence

from bongard.canonical import canonical_digest
from bongard.evidence import Disposition
from bongard.object_scene_anchor_atlas import (
    object_scene_anchor_atlas_renderer_digest,
    render_object_scene_anchor_atlas,
)
from bongard.object_scene_anchor_batch_observer import (
    OBJECT_SCENE_ANCHOR_BATCH_MAX_CELLS,
    ObjectSceneAnchorBatchObserverArtifact,
    ObjectSceneAnchorBatchObserverInput,
    ObjectSceneAnchorBatchObserverPlan,
    freeze_object_scene_anchor_batch_observer_plan,
    object_scene_anchor_batch_observer_source_digest,
    object_scene_anchor_object_matrices_from_batch_artifact,
)
from bongard.object_scene_anchor_bindings import (
    ObjectSceneAnchorBindingCatalog,
    ObjectSceneAnchorBindingSpec,
    build_object_scene_anchor_binding_catalog,
)
from bongard.object_scene_anchor_crop import (
    OBJECT_SCENE_ANCHOR_CROP_RENDERER_ID,
    render_object_scene_anchor_object_crop,
)
from bongard.object_scene_anchor_observer import (
    prepare_object_scene_anchor_observer_inputs,
)
from bongard.object_scene_anchor_support_preparation import (
    ObjectSceneAnchorSupportCorpusFreeze,
    ObjectSceneAnchorSupportCorpusRuntimeBundle,
    object_scene_anchor_support_preparation_source_digest,
)
from bongard.object_scene_anchor_version_space import (
    ObjectSceneAnchorObjectWitnessMatrix,
    ObjectSceneAnchorOrientation,
    ObjectSceneAnchorPanelWitnessEvaluation,
    ObjectSceneAnchorPredicateLanguage,
    ObjectSceneAnchorSupportVersionSpace,
    build_object_scene_anchor_panel_witness_evaluation,
    build_object_scene_anchor_support_version_space,
    cold_verify_object_scene_anchor_support_version_space,
    object_scene_anchor_version_space_algorithm_digest,
)
from bongard.python_predicate_authority import PYTHON_PREDICATE_AUTHORITY_ID


OBJECT_SCENE_ANCHOR_SUPPORT_CATALOG_RECORD_SCHEMA = (
    "gkm.object-scene-anchor-support-catalog-record.v1"
)
OBJECT_SCENE_ANCHOR_SUPPORT_OBSERVATION_PLAN_SCHEMA = (
    "gkm.object-scene-anchor-support-observation-plan.v1"
)
OBJECT_SCENE_ANCHOR_SUPPORT_OBSERVATION_RESULT_SCHEMA = (
    "gkm.object-scene-anchor-support-observation-result.v1"
)
OBJECT_SCENE_ANCHOR_SUPPORT_OBSERVATION_JOIN_ALGORITHM_ID = (
    "bongard.object-scene-anchor-support-observation-join/python-v1"
)

_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_ADDRESS = re.compile(r"sha256:[0-9a-f]{64}\Z")
_PANEL_ALIAS = re.compile(r"panel_[0-9]{3}\Z")
_OBJECT_ID = re.compile(r"object_[0-9]{4}\Z")
_FORBIDDEN_PERSISTENT_KEY_FRAGMENT = "l" + "ean"


class ObjectSceneAnchorSupportObservationJoinError(ValueError):
    """A support catalog plan, batch join, or replay is not canonical."""


def _authority_data() -> dict[str, object]:
    return {
        "predicate_authority_id": PYTHON_PREDICATE_AUTHORITY_ID,
        "python_is_canonical_authority": True,
        "complete_twelve_panel_inventory_required": True,
        "catalog_order_is_panel_spec_object": True,
        "model_inputs_include_present_catalogs_only": True,
        "hard_nonpresent_catalogs_have_zero_rows": True,
        "bucket_metadata_model_visible": False,
        "comparison_labels_model_visible": False,
        "positive_witnesses_only": True,
        "negation_available": False,
        "polarity_flip_available": False,
    }


def _exact_fields(
    value: object, expected: set[str] | frozenset[str], label: str
) -> Mapping[str, Any]:
    if (
        not isinstance(value, Mapping)
        or any(not isinstance(key, str) for key in value)
        or set(value) != set(expected)
    ):
        raise ObjectSceneAnchorSupportObservationJoinError(
            f"{label} fields differ"
        )
    return value


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise ObjectSceneAnchorSupportObservationJoinError(
            f"{label} must be a lowercase SHA-256"
        )
    return value


def _address(value: object, label: str) -> str:
    if not isinstance(value, str) or _ADDRESS.fullmatch(value) is None:
        raise ObjectSceneAnchorSupportObservationJoinError(
            f"{label} must be a sha256: address"
        )
    return value


def _integer(value: object, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ObjectSceneAnchorSupportObservationJoinError(
            f"{label} must be an integer at least {minimum}"
        )
    return value


def _assert_persistent_payload(value: object) -> None:
    if isinstance(value, bytes):
        raise ObjectSceneAnchorSupportObservationJoinError(
            "persistent support observation data cannot contain image bytes"
        )
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ObjectSceneAnchorSupportObservationJoinError(
                    "persistent support observation key must be text"
                )
            lowered = key.casefold()
            if _FORBIDDEN_PERSISTENT_KEY_FRAGMENT in lowered:
                safe_value = True if "removable" in lowered else False
                if item is not safe_value:
                    raise ObjectSceneAnchorSupportObservationJoinError(
                        "historical checker metadata cannot grant predicate "
                        "authority"
                    )
            _assert_persistent_payload(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _assert_persistent_payload(item)


def _binding_specs(
    language: ObjectSceneAnchorPredicateLanguage,
) -> tuple[ObjectSceneAnchorBindingSpec, ...]:
    by_digest: dict[str, ObjectSceneAnchorBindingSpec] = {}
    for atom in language.atoms:
        previous = by_digest.get(atom.binding_spec.spec_digest)
        if previous is not None and previous != atom.binding_spec:
            raise ObjectSceneAnchorSupportObservationJoinError(
                "language binding spec digest collision"
            )
        by_digest[atom.binding_spec.spec_digest] = atom.binding_spec
    return tuple(by_digest[key] for key in sorted(by_digest))


def object_scene_anchor_support_observation_join_source_digest() -> str:
    """Return the authenticated source bytes loaded for this join."""

    return verify_loaded_source(__name__, expected_source_sha256=_LOADED_SOURCE_SHA256)


def object_scene_anchor_support_observation_join_algorithm_digest() -> str:
    """Bind catalog construction, batch projection, and version-space replay."""

    return canonical_digest(
        {
            "schema": "gkm.object-scene-anchor-support-observation-join-algorithm.v1",
            "algorithm_id": (
                OBJECT_SCENE_ANCHOR_SUPPORT_OBSERVATION_JOIN_ALGORITHM_ID
            ),
            "source_digest": (
                object_scene_anchor_support_observation_join_source_digest()
            ),
            "support_preparation_source_digest": (
                object_scene_anchor_support_preparation_source_digest()
            ),
            "batch_observer_source_digest": (
                object_scene_anchor_batch_observer_source_digest()
            ),
            "version_space_algorithm_digest": (
                object_scene_anchor_version_space_algorithm_digest()
            ),
            "atlas_renderer_digest": object_scene_anchor_atlas_renderer_digest(),
            "crop_renderer_id": OBJECT_SCENE_ANCHOR_CROP_RENDERER_ID,
            "catalog_order": "panel-major-spec-major-object-major",
            "nonpresent_matrix_rule": "same-catalog-zero-witness-rows",
            "orientation_0": "bucket-0-target-bucket-1-contrast",
            "orientation_1": "bucket-1-target-bucket-0-contrast",
            "batch_cell_count_summary_is_exact": True,
            "maximum_cells_per_batch": OBJECT_SCENE_ANCHOR_BATCH_MAX_CELLS,
            **_authority_data(),
        }
    )


def _batch_cell_counts(
    batch_plan: ObjectSceneAnchorBatchObserverPlan,
) -> tuple[int, ...]:
    return tuple(item.cell_count for item in batch_plan.batches)


def _record_content(
    value: "ObjectSceneAnchorSupportCatalogRecord",
) -> dict[str, object]:
    return {
        "schema": OBJECT_SCENE_ANCHOR_SUPPORT_CATALOG_RECORD_SCHEMA,
        "panel_alias": value.panel_alias,
        "panel_index": value.panel_index,
        "support_bucket_index": value.support_bucket_index,
        "spec_index": value.spec_index,
        "binding_spec": value.binding_spec.to_data(),
        "object_index": value.object_index,
        "object_id": value.object_id,
        "catalog": value.catalog.to_data(),
        "preparation_digest": value.preparation_digest,
        "persistent_image_bytes_present": False,
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class ObjectSceneAnchorSupportCatalogRecord:
    """One exact panel/spec/object catalog and optional visual preparation."""

    panel_alias: str
    panel_index: int
    support_bucket_index: int
    spec_index: int
    binding_spec: ObjectSceneAnchorBindingSpec
    object_index: int
    object_id: str
    catalog: ObjectSceneAnchorBindingCatalog
    preparation_digest: str | None
    record_digest: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.panel_alias, str)
            or _PANEL_ALIAS.fullmatch(self.panel_alias) is None
            or self.panel_alias != f"panel_{self.panel_index:03d}"
        ):
            raise ObjectSceneAnchorSupportObservationJoinError(
                "support catalog panel position differs"
            )
        for label, item in (
            ("panel index", self.panel_index),
            ("support bucket index", self.support_bucket_index),
            ("spec index", self.spec_index),
            ("object index", self.object_index),
        ):
            _integer(item, label)
        if self.support_bucket_index not in (0, 1):
            raise ObjectSceneAnchorSupportObservationJoinError(
                "support catalog bucket differs"
            )
        if (
            not isinstance(self.object_id, str)
            or _OBJECT_ID.fullmatch(self.object_id) is None
            or self.object_id != f"object_{self.object_index:04d}"
        ):
            raise ObjectSceneAnchorSupportObservationJoinError(
                "support catalog object position differs"
            )
        if type(self.binding_spec) is not ObjectSceneAnchorBindingSpec:
            raise TypeError("support catalog binding spec has the wrong type")
        if type(self.catalog) is not ObjectSceneAnchorBindingCatalog:
            raise TypeError("support binding catalog has the wrong type")
        spec = ObjectSceneAnchorBindingSpec.from_data(self.binding_spec.to_data())
        catalog = ObjectSceneAnchorBindingCatalog.from_data(self.catalog.to_data())
        if (
            spec != self.binding_spec
            or catalog != self.catalog
            or catalog.binding_spec != spec
            or catalog.object_id != self.object_id
            or (catalog.hard_disposition is Disposition.PRESENT)
            != (self.preparation_digest is not None)
        ):
            raise ObjectSceneAnchorSupportObservationJoinError(
                "support catalog record projection differs"
            )
        if self.preparation_digest is not None:
            _digest(self.preparation_digest, "observer preparation digest")
        _digest(self.record_digest, "support catalog record digest")
        unsigned = _record_content(self)
        _assert_persistent_payload(unsigned)
        if self.record_digest != canonical_digest(unsigned):
            raise ObjectSceneAnchorSupportObservationJoinError(
                "support catalog record digest differs"
            )

    @classmethod
    def create(
        cls,
        *,
        panel_alias: str,
        panel_index: int,
        support_bucket_index: int,
        spec_index: int,
        binding_spec: ObjectSceneAnchorBindingSpec,
        object_index: int,
        object_id: str,
        catalog: ObjectSceneAnchorBindingCatalog,
        preparation_digest: str | None,
    ) -> "ObjectSceneAnchorSupportCatalogRecord":
        values = {
            "panel_alias": panel_alias,
            "panel_index": panel_index,
            "support_bucket_index": support_bucket_index,
            "spec_index": spec_index,
            "binding_spec": binding_spec,
            "object_index": object_index,
            "object_id": object_id,
            "catalog": catalog,
            "preparation_digest": preparation_digest,
        }
        provisional = object.__new__(cls)
        for name, item in values.items():
            object.__setattr__(provisional, name, item)
        return cls(
            **values,
            record_digest=canonical_digest(_record_content(provisional)),
        )

    def to_data(self) -> dict[str, object]:
        return {**_record_content(self), "record_digest": self.record_digest}

    @classmethod
    def from_data(cls, value: object) -> "ObjectSceneAnchorSupportCatalogRecord":
        raw = _exact_fields(
            value,
            {
                "schema",
                "panel_alias",
                "panel_index",
                "support_bucket_index",
                "spec_index",
                "binding_spec",
                "object_index",
                "object_id",
                "catalog",
                "preparation_digest",
                "persistent_image_bytes_present",
                *_authority_data(),
                "record_digest",
            },
            "support catalog record",
        )
        if (
            raw["schema"] != OBJECT_SCENE_ANCHOR_SUPPORT_CATALOG_RECORD_SCHEMA
            or raw["persistent_image_bytes_present"] is not False
            or any(raw[key] != item for key, item in _authority_data().items())
            or not isinstance(raw["binding_spec"], Mapping)
            or not isinstance(raw["catalog"], Mapping)
        ):
            raise ObjectSceneAnchorSupportObservationJoinError(
                "support catalog record policy differs"
            )
        result = cls(
            raw["panel_alias"],
            raw["panel_index"],
            raw["support_bucket_index"],
            raw["spec_index"],
            ObjectSceneAnchorBindingSpec.from_data(raw["binding_spec"]),
            raw["object_index"],
            raw["object_id"],
            ObjectSceneAnchorBindingCatalog.from_data(raw["catalog"]),
            raw["preparation_digest"],
            raw["record_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectSceneAnchorSupportObservationJoinError(
                "support catalog record is not canonical"
            )
        return result


def _observation_context_digest(
    *,
    corpus_freeze_digest: str,
    language_digest: str,
    catalog_records: Sequence[ObjectSceneAnchorSupportCatalogRecord],
    batch_plan_digest: str,
) -> str:
    return "sha256:" + canonical_digest(
        {
            "schema": "gkm.object-scene-anchor-support-observation-context.v1",
            "corpus_freeze_digest": corpus_freeze_digest,
            "language_digest": language_digest,
            "catalog_record_digests": [item.record_digest for item in catalog_records],
            "batch_plan_digest": batch_plan_digest,
            "bucket_metadata_excluded_from_model_input": True,
        }
    )


def _plan_content(
    value: "ObjectSceneAnchorSupportObservationPlan",
) -> dict[str, object]:
    return {
        "schema": OBJECT_SCENE_ANCHOR_SUPPORT_OBSERVATION_PLAN_SCHEMA,
        "algorithm_id": OBJECT_SCENE_ANCHOR_SUPPORT_OBSERVATION_JOIN_ALGORITHM_ID,
        "source_digest": value.source_digest,
        "algorithm_digest": value.algorithm_digest,
        "corpus": value.corpus.to_data(),
        "corpus_freeze_digest": value.corpus_freeze_digest,
        "language": value.language.to_data(),
        "language_digest": value.language_digest,
        "binding_specs": [item.to_data() for item in value.binding_specs],
        "catalog_records": [item.to_data() for item in value.catalog_records],
        "batch_plan": value.batch_plan.to_data(),
        "batch_plan_digest": value.batch_plan_digest,
        "observation_context_digest": value.observation_context_digest,
        "panel_count": value.panel_count,
        "binding_spec_count": value.binding_spec_count,
        "catalog_count": value.catalog_count,
        "present_catalog_count": value.present_catalog_count,
        "nonpresent_catalog_count": value.nonpresent_catalog_count,
        "rendered_view_count": value.rendered_view_count,
        "batch_cell_counts": list(value.batch_cell_counts),
        "total_present_cell_count": value.total_present_cell_count,
        "maximum_batch_cell_count": value.maximum_batch_cell_count,
        "maximum_cells_per_batch": value.maximum_cells_per_batch,
        "catalog_order": "panel-major-spec-major-object-major",
        "persistent_image_bytes_present": False,
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class ObjectSceneAnchorSupportObservationPlan:
    """Byte-free exact catalog inventory and role-blind batch partition."""

    source_digest: str
    algorithm_digest: str
    corpus: ObjectSceneAnchorSupportCorpusFreeze
    corpus_freeze_digest: str
    language: ObjectSceneAnchorPredicateLanguage
    language_digest: str
    binding_specs: tuple[ObjectSceneAnchorBindingSpec, ...]
    catalog_records: tuple[ObjectSceneAnchorSupportCatalogRecord, ...]
    batch_plan: ObjectSceneAnchorBatchObserverPlan
    batch_plan_digest: str
    observation_context_digest: str
    panel_count: int
    binding_spec_count: int
    catalog_count: int
    present_catalog_count: int
    nonpresent_catalog_count: int
    rendered_view_count: int
    batch_cell_counts: tuple[int, ...]
    total_present_cell_count: int
    maximum_batch_cell_count: int
    maximum_cells_per_batch: int
    plan_digest: str

    def __post_init__(self) -> None:
        for label, item in (
            ("support join source digest", self.source_digest),
            ("support join algorithm digest", self.algorithm_digest),
            ("corpus freeze digest", self.corpus_freeze_digest),
            ("language digest", self.language_digest),
            ("batch plan digest", self.batch_plan_digest),
            ("support observation plan digest", self.plan_digest),
        ):
            _digest(item, label)
        _address(self.observation_context_digest, "observation context digest")
        for label, item in (
            ("panel count", self.panel_count),
            ("binding spec count", self.binding_spec_count),
            ("catalog count", self.catalog_count),
            ("present catalog count", self.present_catalog_count),
            ("nonpresent catalog count", self.nonpresent_catalog_count),
            ("rendered view count", self.rendered_view_count),
            ("total present cell count", self.total_present_cell_count),
            ("maximum batch cell count", self.maximum_batch_cell_count),
            ("maximum cells per batch", self.maximum_cells_per_batch),
        ):
            _integer(item, label)
        if type(self.corpus) is not ObjectSceneAnchorSupportCorpusFreeze:
            raise TypeError("support observation corpus has the wrong type")
        if type(self.language) is not ObjectSceneAnchorPredicateLanguage:
            raise TypeError("support observation language has the wrong type")
        if type(self.batch_plan) is not ObjectSceneAnchorBatchObserverPlan:
            raise TypeError("support observation batch plan has the wrong type")
        corpus = ObjectSceneAnchorSupportCorpusFreeze.from_data(
            self.corpus.to_data()
        )
        language = ObjectSceneAnchorPredicateLanguage.from_data(
            self.language.to_data()
        )
        batch_plan = ObjectSceneAnchorBatchObserverPlan.from_data(
            self.batch_plan.to_data()
        )
        batch_cell_counts = _batch_cell_counts(batch_plan)
        specs = _binding_specs(language)
        if (
            self.source_digest
            != object_scene_anchor_support_observation_join_source_digest()
            or self.algorithm_digest
            != object_scene_anchor_support_observation_join_algorithm_digest()
            or corpus != self.corpus
            or corpus.freeze_digest != self.corpus_freeze_digest
            or language != self.language
            or language.language_digest != self.language_digest
            or type(self.binding_specs) is not tuple
            or self.binding_specs != specs
            or batch_plan != self.batch_plan
            or batch_plan.plan_digest != self.batch_plan_digest
            or batch_plan.vocabulary != language.vocabulary
            or type(self.catalog_records) is not tuple
            or any(
                type(item) is not ObjectSceneAnchorSupportCatalogRecord
                for item in self.catalog_records
            )
        ):
            raise ObjectSceneAnchorSupportObservationJoinError(
                "support observation plan authority or nested artifact differs"
            )

        expected_positions = []
        expected_catalogs = []
        for panel_index, panel in enumerate(corpus.panels):
            for spec_index, spec in enumerate(specs):
                for object_index, (object_id, decision) in enumerate(
                    zip(
                        panel.panel_manifest.object_ids,
                        panel.panel_manifest.object_decisions,
                        strict=True,
                    )
                ):
                    expected_positions.append(
                        (
                            panel.panel_alias,
                            panel_index,
                            panel.support_bucket_index,
                            spec_index,
                            spec,
                            object_index,
                            object_id,
                        )
                    )
                    expected_catalogs.append(
                        build_object_scene_anchor_binding_catalog(
                            decision, spec, expected_object_id=object_id
                        )
                    )
        actual_positions = tuple(
            (
                item.panel_alias,
                item.panel_index,
                item.support_bucket_index,
                item.spec_index,
                item.binding_spec,
                item.object_index,
                item.object_id,
            )
            for item in self.catalog_records
        )
        if (
            actual_positions != tuple(expected_positions)
            or tuple(item.catalog for item in self.catalog_records)
            != tuple(expected_catalogs)
        ):
            raise ObjectSceneAnchorSupportObservationJoinError(
                "support catalog records are not complete panel/spec/object order"
            )

        preparations = batch_plan.preparations
        preparation_by_digest = {
            item.preparation_digest: item for item in preparations
        }
        present = tuple(
            item
            for item in self.catalog_records
            if item.catalog.hard_disposition is Disposition.PRESENT
        )
        nonpresent = tuple(
            item
            for item in self.catalog_records
            if item.catalog.hard_disposition is not Disposition.PRESENT
        )
        if (
            any(item.preparation_digest is None for item in present)
            or any(item.preparation_digest is not None for item in nonpresent)
            or {item.preparation_digest for item in present}
            != set(preparation_by_digest)
            or len(present) != len(preparations)
        ):
            raise ObjectSceneAnchorSupportObservationJoinError(
                "batch plan does not contain exactly the hard-present catalogs"
            )
        panel_by_alias = corpus.by_panel_alias
        for record in present:
            assert record.preparation_digest is not None
            preparation = preparation_by_digest[record.preparation_digest]
            panel = panel_by_alias[record.panel_alias]
            if (
                preparation.catalog != record.catalog
                or preparation.vocabulary != language.vocabulary
                or preparation.panel_manifest != panel.panel_manifest
                or preparation.object_index != record.object_index
                or preparation.object_id != record.object_id
                or preparation.decision_manifest_digest
                != panel.panel_manifest.object_decisions[
                    record.object_index
                ].manifest_digest
            ):
                raise ObjectSceneAnchorSupportObservationJoinError(
                    "observer preparation differs from catalog record"
                )

        if (
            self.panel_count != len(corpus.panels)
            or self.binding_spec_count != len(specs)
            or self.catalog_count != len(self.catalog_records)
            or self.present_catalog_count != len(present)
            or self.nonpresent_catalog_count != len(nonpresent)
            or self.catalog_count
            != self.present_catalog_count + self.nonpresent_catalog_count
            or self.rendered_view_count != batch_plan.view_count
            or type(self.batch_cell_counts) is not tuple
            or self.batch_cell_counts != batch_cell_counts
            or self.total_present_cell_count != sum(batch_cell_counts)
            or self.total_present_cell_count != batch_plan.cell_count
            or self.maximum_batch_cell_count != max(batch_cell_counts)
            or self.maximum_cells_per_batch
            != OBJECT_SCENE_ANCHOR_BATCH_MAX_CELLS
            or self.maximum_batch_cell_count > self.maximum_cells_per_batch
            or self.observation_context_digest
            != _observation_context_digest(
                corpus_freeze_digest=corpus.freeze_digest,
                language_digest=language.language_digest,
                catalog_records=self.catalog_records,
                batch_plan_digest=batch_plan.plan_digest,
            )
        ):
            raise ObjectSceneAnchorSupportObservationJoinError(
                "support observation plan counts or commitments differ"
            )
        unsigned = _plan_content(self)
        _assert_persistent_payload(unsigned)
        if self.plan_digest != canonical_digest(unsigned):
            raise ObjectSceneAnchorSupportObservationJoinError(
                "support observation plan digest differs"
            )

    def to_data(self) -> dict[str, object]:
        return {**_plan_content(self), "plan_digest": self.plan_digest}

    @classmethod
    def from_data(cls, value: object) -> "ObjectSceneAnchorSupportObservationPlan":
        raw = _exact_fields(
            value,
            {
                "schema",
                "algorithm_id",
                "source_digest",
                "algorithm_digest",
                "corpus",
                "corpus_freeze_digest",
                "language",
                "language_digest",
                "binding_specs",
                "catalog_records",
                "batch_plan",
                "batch_plan_digest",
                "observation_context_digest",
                "panel_count",
                "binding_spec_count",
                "catalog_count",
                "present_catalog_count",
                "nonpresent_catalog_count",
                "rendered_view_count",
                "batch_cell_counts",
                "total_present_cell_count",
                "maximum_batch_cell_count",
                "maximum_cells_per_batch",
                "catalog_order",
                "persistent_image_bytes_present",
                *_authority_data(),
                "plan_digest",
            },
            "support observation plan",
        )
        if (
            raw["schema"] != OBJECT_SCENE_ANCHOR_SUPPORT_OBSERVATION_PLAN_SCHEMA
            or raw["algorithm_id"]
            != OBJECT_SCENE_ANCHOR_SUPPORT_OBSERVATION_JOIN_ALGORITHM_ID
            or raw["catalog_order"] != "panel-major-spec-major-object-major"
            or raw["persistent_image_bytes_present"] is not False
            or any(raw[key] != item for key, item in _authority_data().items())
            or not isinstance(raw["corpus"], Mapping)
            or not isinstance(raw["language"], Mapping)
            or not isinstance(raw["binding_specs"], list)
            or not isinstance(raw["catalog_records"], list)
            or not isinstance(raw["batch_plan"], Mapping)
            or not isinstance(raw["batch_cell_counts"], list)
        ):
            raise ObjectSceneAnchorSupportObservationJoinError(
                "support observation plan policy differs"
            )
        result = cls(
            raw["source_digest"],
            raw["algorithm_digest"],
            ObjectSceneAnchorSupportCorpusFreeze.from_data(raw["corpus"]),
            raw["corpus_freeze_digest"],
            ObjectSceneAnchorPredicateLanguage.from_data(raw["language"]),
            raw["language_digest"],
            tuple(
                ObjectSceneAnchorBindingSpec.from_data(item)
                for item in raw["binding_specs"]
            ),
            tuple(
                ObjectSceneAnchorSupportCatalogRecord.from_data(item)
                for item in raw["catalog_records"]
            ),
            ObjectSceneAnchorBatchObserverPlan.from_data(raw["batch_plan"]),
            raw["batch_plan_digest"],
            raw["observation_context_digest"],
            raw["panel_count"],
            raw["binding_spec_count"],
            raw["catalog_count"],
            raw["present_catalog_count"],
            raw["nonpresent_catalog_count"],
            raw["rendered_view_count"],
            tuple(raw["batch_cell_counts"]),
            raw["total_present_cell_count"],
            raw["maximum_batch_cell_count"],
            raw["maximum_cells_per_batch"],
            raw["plan_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectSceneAnchorSupportObservationJoinError(
                "support observation plan is not canonical"
            )
        return result


@dataclass(frozen=True, slots=True)
class ObjectSceneAnchorSupportObservationRuntimeBundle:
    """Persistent plan plus exact runtime-only batch presentation bytes."""

    plan: ObjectSceneAnchorSupportObservationPlan
    batch_inputs: tuple[ObjectSceneAnchorBatchObserverInput, ...]

    def __post_init__(self) -> None:
        if type(self.plan) is not ObjectSceneAnchorSupportObservationPlan:
            raise TypeError("support observation runtime plan has the wrong type")
        if (
            type(self.batch_inputs) is not tuple
            or any(
                type(item) is not ObjectSceneAnchorBatchObserverInput
                for item in self.batch_inputs
            )
            or tuple(item.preparation for item in self.batch_inputs)
            != self.plan.batch_plan.preparations
        ):
            raise ObjectSceneAnchorSupportObservationJoinError(
                "runtime batch inputs differ from persistent plan"
            )
        if len(self.batch_inputs) != self.plan.present_catalog_count:
            raise ObjectSceneAnchorSupportObservationJoinError(
                "runtime batch input count differs"
            )


def _make_plan(
    *,
    corpus: ObjectSceneAnchorSupportCorpusFreeze,
    language: ObjectSceneAnchorPredicateLanguage,
    specs: tuple[ObjectSceneAnchorBindingSpec, ...],
    records: tuple[ObjectSceneAnchorSupportCatalogRecord, ...],
    batch_plan: ObjectSceneAnchorBatchObserverPlan,
) -> ObjectSceneAnchorSupportObservationPlan:
    batch_cell_counts = _batch_cell_counts(batch_plan)
    present_count = sum(
        item.catalog.hard_disposition is Disposition.PRESENT for item in records
    )
    values = {
        "source_digest": (
            object_scene_anchor_support_observation_join_source_digest()
        ),
        "algorithm_digest": (
            object_scene_anchor_support_observation_join_algorithm_digest()
        ),
        "corpus": corpus,
        "corpus_freeze_digest": corpus.freeze_digest,
        "language": language,
        "language_digest": language.language_digest,
        "binding_specs": specs,
        "catalog_records": records,
        "batch_plan": batch_plan,
        "batch_plan_digest": batch_plan.plan_digest,
        "observation_context_digest": _observation_context_digest(
            corpus_freeze_digest=corpus.freeze_digest,
            language_digest=language.language_digest,
            catalog_records=records,
            batch_plan_digest=batch_plan.plan_digest,
        ),
        "panel_count": len(corpus.panels),
        "binding_spec_count": len(specs),
        "catalog_count": len(records),
        "present_catalog_count": present_count,
        "nonpresent_catalog_count": len(records) - present_count,
        "rendered_view_count": batch_plan.view_count,
        "batch_cell_counts": batch_cell_counts,
        "total_present_cell_count": sum(batch_cell_counts),
        "maximum_batch_cell_count": max(batch_cell_counts),
        "maximum_cells_per_batch": OBJECT_SCENE_ANCHOR_BATCH_MAX_CELLS,
    }
    provisional = object.__new__(ObjectSceneAnchorSupportObservationPlan)
    for name, item in values.items():
        object.__setattr__(provisional, name, item)
    return ObjectSceneAnchorSupportObservationPlan(
        **values,
        plan_digest=canonical_digest(_plan_content(provisional)),
    )


def build_object_scene_anchor_support_observation_plan(
    corpus_runtime: ObjectSceneAnchorSupportCorpusRuntimeBundle,
    language: ObjectSceneAnchorPredicateLanguage,
) -> ObjectSceneAnchorSupportObservationRuntimeBundle:
    """Build every catalog and render each present object's view exactly once."""

    if type(corpus_runtime) is not ObjectSceneAnchorSupportCorpusRuntimeBundle:
        raise TypeError(
            "corpus_runtime must be exact ObjectSceneAnchorSupportCorpusRuntimeBundle"
        )
    if type(language) is not ObjectSceneAnchorPredicateLanguage:
        raise TypeError("language must be exact ObjectSceneAnchorPredicateLanguage")
    corpus = ObjectSceneAnchorSupportCorpusFreeze.from_data(
        corpus_runtime.freeze.to_data()
    )
    frozen_language = ObjectSceneAnchorPredicateLanguage.from_data(
        language.to_data()
    )
    if (
        tuple(item.freeze for item in corpus_runtime.panels) != corpus.panels
        or len(corpus_runtime.panels) != len(corpus.panels)
    ):
        raise ObjectSceneAnchorSupportObservationJoinError(
            "support runtime panels differ from corpus freeze"
        )
    specs = _binding_specs(frozen_language)
    runtime_by_alias = corpus_runtime.by_panel_alias
    records = []
    inputs = []
    for panel_index, panel in enumerate(corpus.panels):
        runtime_panel = runtime_by_alias[panel.panel_alias]
        view_cache: dict[str, tuple[bytes, object, bytes]] = {}
        entry_by_object = panel.catalog.by_object_id
        for spec_index, spec in enumerate(specs):
            for object_index, (object_id, decision) in enumerate(
                zip(
                    panel.panel_manifest.object_ids,
                    panel.panel_manifest.object_decisions,
                    strict=True,
                )
            ):
                catalog = build_object_scene_anchor_binding_catalog(
                    decision, spec, expected_object_id=object_id
                )
                preparation_digest = None
                if catalog.hard_disposition is Disposition.PRESENT:
                    cached = view_cache.get(object_id)
                    if cached is None:
                        entry = entry_by_object[object_id]
                        crop_png = render_object_scene_anchor_object_crop(
                            runtime_panel.exact_original_png_bytes,
                            panel.inventory,
                            entry,
                        )
                        atlas, atlas_png = render_object_scene_anchor_atlas(
                            decision
                        )
                        if atlas_png is None:
                            raise ObjectSceneAnchorSupportObservationJoinError(
                                "present binding catalog has no clean anchor atlas"
                            )
                        cached = (crop_png, atlas, atlas_png)
                        view_cache[object_id] = cached
                    crop_png, atlas, atlas_png = cached
                    entry = entry_by_object[object_id]
                    preparation = prepare_object_scene_anchor_observer_inputs(
                        crop_png,
                        catalog_entry=entry,
                        panel_manifest=panel.panel_manifest,
                        atlas=atlas,
                        atlas_png_bytes=atlas_png,
                        catalog=catalog,
                        vocabulary=frozen_language.vocabulary,
                    )
                    inputs.append(
                        ObjectSceneAnchorBatchObserverInput(
                            preparation, crop_png, atlas_png
                        )
                    )
                    preparation_digest = preparation.preparation_digest
                records.append(
                    ObjectSceneAnchorSupportCatalogRecord.create(
                        panel_alias=panel.panel_alias,
                        panel_index=panel_index,
                        support_bucket_index=panel.support_bucket_index,
                        spec_index=spec_index,
                        binding_spec=spec,
                        object_index=object_index,
                        object_id=object_id,
                        catalog=catalog,
                        preparation_digest=preparation_digest,
                    )
                )
    if not inputs:
        raise ObjectSceneAnchorSupportObservationJoinError(
            "support language produced no hard-present observer catalog"
        )
    batch_plan = freeze_object_scene_anchor_batch_observer_plan(tuple(inputs))
    persistent = _make_plan(
        corpus=corpus,
        language=frozen_language,
        specs=specs,
        records=tuple(records),
        batch_plan=batch_plan,
    )
    input_by_digest = {
        item.preparation.preparation_digest: item for item in inputs
    }
    ordered_inputs = tuple(
        input_by_digest[item.preparation_digest]
        for item in batch_plan.preparations
    )
    return ObjectSceneAnchorSupportObservationRuntimeBundle(
        persistent, ordered_inputs
    )


def verify_object_scene_anchor_support_observation_runtime(
    bundle: ObjectSceneAnchorSupportObservationRuntimeBundle,
    *,
    corpus_runtime: ObjectSceneAnchorSupportCorpusRuntimeBundle,
    language: ObjectSceneAnchorPredicateLanguage,
    expected_plan_digest: str | None = None,
) -> ObjectSceneAnchorSupportObservationRuntimeBundle:
    """Rerender all runtime views and replay the persistent plan without calls."""

    if type(bundle) is not ObjectSceneAnchorSupportObservationRuntimeBundle:
        raise TypeError(
            "bundle must be exact ObjectSceneAnchorSupportObservationRuntimeBundle"
        )
    restored_plan = ObjectSceneAnchorSupportObservationPlan.from_data(
        bundle.plan.to_data()
    )
    if expected_plan_digest is not None and restored_plan.plan_digest != _digest(
        expected_plan_digest, "expected support observation plan digest"
    ):
        raise ObjectSceneAnchorSupportObservationJoinError(
            "support observation plan differs from commitment"
        )
    replayed = build_object_scene_anchor_support_observation_plan(
        corpus_runtime, language
    )
    if replayed != bundle or replayed.plan != restored_plan:
        raise ObjectSceneAnchorSupportObservationJoinError(
            "support observation runtime differs from cold replay"
        )
    return replayed


def _result_content(
    value: "ObjectSceneAnchorSupportObservationResult",
) -> dict[str, object]:
    return {
        "schema": OBJECT_SCENE_ANCHOR_SUPPORT_OBSERVATION_RESULT_SCHEMA,
        "algorithm_id": OBJECT_SCENE_ANCHOR_SUPPORT_OBSERVATION_JOIN_ALGORITHM_ID,
        "source_digest": value.source_digest,
        "algorithm_digest": value.algorithm_digest,
        "plan_digest": value.plan_digest,
        "corpus_freeze_digest": value.corpus_freeze_digest,
        "language_digest": value.language_digest,
        "batch_artifact_digest": value.batch_artifact_digest,
        "panel_evaluations": [item.to_data() for item in value.panel_evaluations],
        "bucket0_positive_version_space": (
            value.bucket0_positive_version_space.to_data()
        ),
        "bucket1_positive_version_space": (
            value.bucket1_positive_version_space.to_data()
        ),
        "bucket0_positive_mapping": "bucket-0-target-bucket-1-contrast",
        "bucket1_positive_mapping": "bucket-1-target-bucket-0-contrast",
        "panel_evaluations_are_role_neutral": True,
        "persistent_image_bytes_present": False,
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class ObjectSceneAnchorSupportObservationResult:
    """Twelve neutral panels and the two explicit orientation version spaces."""

    source_digest: str
    algorithm_digest: str
    plan_digest: str
    corpus_freeze_digest: str
    language_digest: str
    batch_artifact_digest: str
    panel_evaluations: tuple[ObjectSceneAnchorPanelWitnessEvaluation, ...]
    bucket0_positive_version_space: ObjectSceneAnchorSupportVersionSpace
    bucket1_positive_version_space: ObjectSceneAnchorSupportVersionSpace
    result_digest: str

    def __post_init__(self) -> None:
        for label, item in (
            ("support join source digest", self.source_digest),
            ("support join algorithm digest", self.algorithm_digest),
            ("support observation plan digest", self.plan_digest),
            ("support corpus freeze digest", self.corpus_freeze_digest),
            ("support language digest", self.language_digest),
            ("batch artifact digest", self.batch_artifact_digest),
            ("support observation result digest", self.result_digest),
        ):
            _digest(item, label)
        if (
            type(self.panel_evaluations) is not tuple
            or len(self.panel_evaluations) != 12
            or any(
                type(item) is not ObjectSceneAnchorPanelWitnessEvaluation
                for item in self.panel_evaluations
            )
            or tuple(item.panel_id for item in self.panel_evaluations)
            != tuple(f"panel_{index:03d}" for index in range(12))
            or any(
                item.language_digest != self.language_digest
                for item in self.panel_evaluations
            )
        ):
            raise ObjectSceneAnchorSupportObservationJoinError(
                "support result panel evaluation inventory differs"
            )
        if type(
            self.bucket0_positive_version_space
        ) is not ObjectSceneAnchorSupportVersionSpace or type(
            self.bucket1_positive_version_space
        ) is not ObjectSceneAnchorSupportVersionSpace:
            raise TypeError("support result version space has the wrong type")
        forward = ObjectSceneAnchorSupportVersionSpace.from_data(
            self.bucket0_positive_version_space.to_data()
        )
        inverse = ObjectSceneAnchorSupportVersionSpace.from_data(
            self.bucket1_positive_version_space.to_data()
        )
        by_id = {item.panel_id: item for item in self.panel_evaluations}
        forward_ids = tuple(f"panel_{index:03d}" for index in range(12))
        inverse_ids = tuple(f"panel_{index:03d}" for index in range(6, 12)) + tuple(
            f"panel_{index:03d}" for index in range(6)
        )
        if (
            self.source_digest
            != object_scene_anchor_support_observation_join_source_digest()
            or self.algorithm_digest
            != object_scene_anchor_support_observation_join_algorithm_digest()
            or forward != self.bucket0_positive_version_space
            or inverse != self.bucket1_positive_version_space
            or forward.orientation is not ObjectSceneAnchorOrientation.SIDE0_POSITIVE
            or inverse.orientation is not ObjectSceneAnchorOrientation.SIDE1_POSITIVE
            or forward.language != inverse.language
            or forward.language.language_digest != self.language_digest
            or forward.support_panel_ids != forward_ids
            or inverse.support_panel_ids != inverse_ids
            or forward.support_evaluation_digests
            != tuple(by_id[item].evaluation_digest for item in forward_ids)
            or inverse.support_evaluation_digests
            != tuple(by_id[item].evaluation_digest for item in inverse_ids)
        ):
            raise ObjectSceneAnchorSupportObservationJoinError(
                "support result orientation or version-space binding differs"
            )
        unsigned = _result_content(self)
        _assert_persistent_payload(unsigned)
        if self.result_digest != canonical_digest(unsigned):
            raise ObjectSceneAnchorSupportObservationJoinError(
                "support observation result digest differs"
            )

    def to_data(self) -> dict[str, object]:
        return {**_result_content(self), "result_digest": self.result_digest}

    @classmethod
    def from_data(cls, value: object) -> "ObjectSceneAnchorSupportObservationResult":
        raw = _exact_fields(
            value,
            {
                "schema",
                "algorithm_id",
                "source_digest",
                "algorithm_digest",
                "plan_digest",
                "corpus_freeze_digest",
                "language_digest",
                "batch_artifact_digest",
                "panel_evaluations",
                "bucket0_positive_version_space",
                "bucket1_positive_version_space",
                "bucket0_positive_mapping",
                "bucket1_positive_mapping",
                "panel_evaluations_are_role_neutral",
                "persistent_image_bytes_present",
                *_authority_data(),
                "result_digest",
            },
            "support observation result",
        )
        if (
            raw["schema"] != OBJECT_SCENE_ANCHOR_SUPPORT_OBSERVATION_RESULT_SCHEMA
            or raw["algorithm_id"]
            != OBJECT_SCENE_ANCHOR_SUPPORT_OBSERVATION_JOIN_ALGORITHM_ID
            or raw["bucket0_positive_mapping"]
            != "bucket-0-target-bucket-1-contrast"
            or raw["bucket1_positive_mapping"]
            != "bucket-1-target-bucket-0-contrast"
            or raw["panel_evaluations_are_role_neutral"] is not True
            or raw["persistent_image_bytes_present"] is not False
            or any(raw[key] != item for key, item in _authority_data().items())
            or not isinstance(raw["panel_evaluations"], list)
            or not isinstance(raw["bucket0_positive_version_space"], Mapping)
            or not isinstance(raw["bucket1_positive_version_space"], Mapping)
        ):
            raise ObjectSceneAnchorSupportObservationJoinError(
                "support observation result policy differs"
            )
        result = cls(
            raw["source_digest"],
            raw["algorithm_digest"],
            raw["plan_digest"],
            raw["corpus_freeze_digest"],
            raw["language_digest"],
            raw["batch_artifact_digest"],
            tuple(
                ObjectSceneAnchorPanelWitnessEvaluation.from_data(item)
                for item in raw["panel_evaluations"]
            ),
            ObjectSceneAnchorSupportVersionSpace.from_data(
                raw["bucket0_positive_version_space"]
            ),
            ObjectSceneAnchorSupportVersionSpace.from_data(
                raw["bucket1_positive_version_space"]
            ),
            raw["result_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectSceneAnchorSupportObservationJoinError(
                "support observation result is not canonical"
            )
        return result


def _finalize(
    plan: ObjectSceneAnchorSupportObservationPlan,
    artifact: ObjectSceneAnchorBatchObserverArtifact,
) -> ObjectSceneAnchorSupportObservationResult:
    if artifact.plan != plan.batch_plan or artifact.plan_digest != plan.batch_plan_digest:
        raise ObjectSceneAnchorSupportObservationJoinError(
            "batch artifact plan differs from persistent support plan"
        )
    if artifact.observation_plan_digest != plan.observation_context_digest:
        raise ObjectSceneAnchorSupportObservationJoinError(
            "batch artifact observation context differs"
        )
    projected = object_scene_anchor_object_matrices_from_batch_artifact(
        artifact, plan.language
    )
    preparations = artifact.plan.preparations
    if len(projected) != len(preparations):
        raise ObjectSceneAnchorSupportObservationJoinError(
            "observed present matrix inventory differs from batch preparations"
        )
    observed_by_preparation = dict(
        zip(
            (item.preparation_digest for item in preparations),
            projected,
            strict=True,
        )
    )
    matrices = []
    for record in plan.catalog_records:
        if record.preparation_digest is None:
            matrix = ObjectSceneAnchorObjectWitnessMatrix.create(
                catalog=record.catalog,
                vocabulary=plan.language.vocabulary,
                cells=(),
            )
        else:
            matrix = observed_by_preparation.get(record.preparation_digest)
            if matrix is None or matrix.catalog != record.catalog:
                raise ObjectSceneAnchorSupportObservationJoinError(
                    "observed matrix differs from persistent catalog record"
                )
        matrices.append(matrix)

    evaluations = []
    offset = 0
    for panel in plan.corpus.panels:
        count = len(plan.binding_specs) * len(panel.object_ids)
        panel_matrices = tuple(matrices[offset : offset + count])
        offset += count
        evaluations.append(
            build_object_scene_anchor_panel_witness_evaluation(
                panel_id=panel.panel_alias,
                panel_manifest=panel.panel_manifest,
                language=plan.language,
                object_matrices=panel_matrices,
            )
        )
    if offset != len(matrices):
        raise ObjectSceneAnchorSupportObservationJoinError(
            "panel evaluation join did not consume every catalog matrix"
        )
    frozen_evaluations = tuple(evaluations)
    bucket0 = tuple(
        item
        for panel, item in zip(plan.corpus.panels, frozen_evaluations, strict=True)
        if panel.support_bucket_index == 0
    )
    bucket1 = tuple(
        item
        for panel, item in zip(plan.corpus.panels, frozen_evaluations, strict=True)
        if panel.support_bucket_index == 1
    )
    forward = build_object_scene_anchor_support_version_space(
        language=plan.language,
        orientation=ObjectSceneAnchorOrientation.SIDE0_POSITIVE,
        targets=bucket0,
        contrasts=bucket1,
    )
    inverse = build_object_scene_anchor_support_version_space(
        language=plan.language,
        orientation=ObjectSceneAnchorOrientation.SIDE1_POSITIVE,
        targets=bucket1,
        contrasts=bucket0,
    )
    cold_verify_object_scene_anchor_support_version_space(
        forward,
        language=plan.language,
        orientation=ObjectSceneAnchorOrientation.SIDE0_POSITIVE,
        targets=bucket0,
        contrasts=bucket1,
    )
    cold_verify_object_scene_anchor_support_version_space(
        inverse,
        language=plan.language,
        orientation=ObjectSceneAnchorOrientation.SIDE1_POSITIVE,
        targets=bucket1,
        contrasts=bucket0,
    )
    values = {
        "source_digest": (
            object_scene_anchor_support_observation_join_source_digest()
        ),
        "algorithm_digest": (
            object_scene_anchor_support_observation_join_algorithm_digest()
        ),
        "plan_digest": plan.plan_digest,
        "corpus_freeze_digest": plan.corpus.freeze_digest,
        "language_digest": plan.language.language_digest,
        "batch_artifact_digest": artifact.artifact_digest,
        "panel_evaluations": frozen_evaluations,
        "bucket0_positive_version_space": forward,
        "bucket1_positive_version_space": inverse,
    }
    provisional = object.__new__(ObjectSceneAnchorSupportObservationResult)
    for name, item in values.items():
        object.__setattr__(provisional, name, item)
    return ObjectSceneAnchorSupportObservationResult(
        **values,
        result_digest=canonical_digest(_result_content(provisional)),
    )


def finalize_object_scene_anchor_support_observations(
    plan: ObjectSceneAnchorSupportObservationPlan,
    artifact: ObjectSceneAnchorBatchObserverArtifact,
) -> ObjectSceneAnchorSupportObservationResult:
    """Join observed P catalogs with hard zero-row catalogs and build both spaces."""

    if type(plan) is not ObjectSceneAnchorSupportObservationPlan:
        raise TypeError("plan must be exact ObjectSceneAnchorSupportObservationPlan")
    if type(artifact) is not ObjectSceneAnchorBatchObserverArtifact:
        raise TypeError("artifact must be exact ObjectSceneAnchorBatchObserverArtifact")
    frozen_plan = ObjectSceneAnchorSupportObservationPlan.from_data(plan.to_data())
    frozen_artifact = ObjectSceneAnchorBatchObserverArtifact.from_data(
        artifact.to_data()
    )
    return _finalize(frozen_plan, frozen_artifact)


def cold_verify_object_scene_anchor_support_observation_result(
    result: ObjectSceneAnchorSupportObservationResult,
    *,
    plan: ObjectSceneAnchorSupportObservationPlan,
    artifact: ObjectSceneAnchorBatchObserverArtifact,
) -> ObjectSceneAnchorSupportObservationResult:
    """Replay all matrices, panels, and orientations with no visual call."""

    if type(result) is not ObjectSceneAnchorSupportObservationResult:
        raise TypeError(
            "result must be exact ObjectSceneAnchorSupportObservationResult"
        )
    restored = ObjectSceneAnchorSupportObservationResult.from_data(
        result.to_data()
    )
    expected = finalize_object_scene_anchor_support_observations(plan, artifact)
    if restored != expected:
        raise ObjectSceneAnchorSupportObservationJoinError(
            "support observation result differs from cold replay"
        )
    return restored


__all__ = (
    "OBJECT_SCENE_ANCHOR_SUPPORT_CATALOG_RECORD_SCHEMA",
    "OBJECT_SCENE_ANCHOR_SUPPORT_OBSERVATION_JOIN_ALGORITHM_ID",
    "OBJECT_SCENE_ANCHOR_SUPPORT_OBSERVATION_PLAN_SCHEMA",
    "OBJECT_SCENE_ANCHOR_SUPPORT_OBSERVATION_RESULT_SCHEMA",
    "ObjectSceneAnchorSupportCatalogRecord",
    "ObjectSceneAnchorSupportObservationJoinError",
    "ObjectSceneAnchorSupportObservationPlan",
    "ObjectSceneAnchorSupportObservationResult",
    "ObjectSceneAnchorSupportObservationRuntimeBundle",
    "build_object_scene_anchor_support_observation_plan",
    "cold_verify_object_scene_anchor_support_observation_result",
    "finalize_object_scene_anchor_support_observations",
    "object_scene_anchor_support_observation_join_algorithm_digest",
    "object_scene_anchor_support_observation_join_source_digest",
    "verify_object_scene_anchor_support_observation_runtime",
)
