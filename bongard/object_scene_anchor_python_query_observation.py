"""Predicate-scoped query observations and evaluation in canonical Python.

The frozen predicate already contains the only binding spec and affirmative
witnesses that can affect its decision.  This module projects that selected
vocabulary, builds an exhaustive binding-by-witness observation over every
object in one neutral panel manifest, and evaluates the closed existential.
No support-language payload is required at construction or replay time.
"""

from __future__ import annotations

from bongard.runtime_source_snapshot import capture_loaded_source, verify_loaded_source


_LOADED_SOURCE_SHA256 = capture_loaded_source(__name__, __file__)

from dataclasses import dataclass
import re
from typing import Any, Mapping, Sequence

from bongard.canonical import canonical_digest
from bongard.evidence import Disposition
from bongard.object_scene_anchor_bindings import (
    ObjectSceneAnchorBindingCatalog,
    ObjectSceneAnchorBindingSpec,
    ObjectSceneAnchorWitnessCell,
    build_object_scene_anchor_binding_catalog,
)
from bongard.object_scene_anchor_observer import (
    ObjectSceneAnchorObserverVocabularyEntry,
)
from bongard.object_scene_anchor_panel_manifest import (
    ObjectSceneAnchorPanelDecisionManifest,
)
from bongard.object_scene_anchor_python_predicate import (
    ObjectSceneAnchorPythonPredicate,
    object_scene_anchor_python_predicate_algorithm_digest,
)
from bongard.object_scene_anchor_version_space import ANCHOR_MAX_CONJUNCTS
from bongard.python_predicate_authority import PYTHON_PREDICATE_AUTHORITY_ID


OBJECT_SCENE_ANCHOR_PYTHON_QUERY_VOCABULARY_SCHEMA = (
    "gkm.object-scene-anchor-python-query-vocabulary.v1"
)
OBJECT_SCENE_ANCHOR_PYTHON_QUERY_ROW_SCHEMA = (
    "gkm.object-scene-anchor-python-query-binding-row.v1"
)
OBJECT_SCENE_ANCHOR_PYTHON_QUERY_OBJECT_SCHEMA = (
    "gkm.object-scene-anchor-python-query-object-observation.v1"
)
OBJECT_SCENE_ANCHOR_PYTHON_QUERY_OBSERVATION_SCHEMA = (
    "gkm.object-scene-anchor-python-query-observation.v1"
)
OBJECT_SCENE_ANCHOR_PYTHON_QUERY_EVALUATION_SCHEMA = (
    "gkm.object-scene-anchor-python-query-evaluation.v1"
)
OBJECT_SCENE_ANCHOR_PYTHON_QUERY_ALGORITHM_ID = (
    "bongard.object-scene-anchor-python-query/selected-vocabulary-v1"
)

_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_PANEL_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/-]{0,511}\Z")
_OBJECT_ID = re.compile(r"object_[0-9]{4}\Z")


class ObjectSceneAnchorPythonQueryError(ValueError):
    """A selected vocabulary, observation, or evaluation is not canonical."""


def _authority_data() -> dict[str, object]:
    return {
        "predicate_authority_id": PYTHON_PREDICATE_AUTHORITY_ID,
        "python_is_canonical_authority": True,
        "selected_vocabulary_only": True,
        "complete_object_inventory_required": True,
        "complete_eligible_binding_inventory_required": True,
        "same_binding_required": True,
        "error_dominant": True,
        "failed_observation_is_negative": False,
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
        raise ObjectSceneAnchorPythonQueryError(f"{label} fields differ")
    return value


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise ObjectSceneAnchorPythonQueryError(
            f"{label} must be a lowercase SHA-256"
        )
    return value


def _panel_id(value: object) -> str:
    if not isinstance(value, str) or _PANEL_ID.fullmatch(value) is None:
        raise ObjectSceneAnchorPythonQueryError("panel ID differs")
    return value


def _disposition(value: object) -> Disposition:
    try:
        return Disposition(value)
    except (TypeError, ValueError) as exc:
        raise ObjectSceneAnchorPythonQueryError(
            "query evaluation disposition differs"
        ) from exc


def object_scene_anchor_python_query_source_digest() -> str:
    """Return the authenticated source bytes loaded for this module."""

    return verify_loaded_source(__name__, expected_source_sha256=_LOADED_SOURCE_SHA256)


def object_scene_anchor_python_query_algorithm_digest() -> str:
    """Bind selected-vocabulary construction and evaluation semantics."""

    return canonical_digest(
        {
            "schema": "gkm.object-scene-anchor-python-query-algorithm.v1",
            "algorithm_id": OBJECT_SCENE_ANCHOR_PYTHON_QUERY_ALGORITHM_ID,
            "source_digest": object_scene_anchor_python_query_source_digest(),
            "predicate_algorithm_digest": (
                object_scene_anchor_python_predicate_algorithm_digest()
            ),
            "cell_domain": "selected_witnesses_for_every_eligible_binding",
            "row_conjunction": "E_then_A_then_all_P_else_I",
            "binding_existential": "E_then_P_then_all_A_else_I",
            "support_language_payload_required": False,
            **_authority_data(),
        }
    )


def _vocabulary_content(
    value: "ObjectSceneAnchorPythonQueryVocabulary",
) -> dict[str, object]:
    return {
        "schema": OBJECT_SCENE_ANCHOR_PYTHON_QUERY_VOCABULARY_SCHEMA,
        "predicate_digest": value.predicate_digest,
        "predicate_algorithm_digest": value.predicate_algorithm_digest,
        "candidate_digest": value.candidate_digest,
        "formula_digest": value.formula_digest,
        "binding_spec": value.binding_spec.to_data(),
        "entries": [item.to_data() for item in value.entries],
        "ordering": "selected-candidate-witness-digest-ascending",
        "complete_selected_vocabulary": True,
        "support_language_payload_present": False,
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class ObjectSceneAnchorPythonQueryVocabulary:
    """The exact binding spec and witness prose selected by one predicate."""

    predicate_digest: str
    predicate_algorithm_digest: str
    candidate_digest: str
    formula_digest: str
    binding_spec: ObjectSceneAnchorBindingSpec
    entries: tuple[ObjectSceneAnchorObserverVocabularyEntry, ...]
    vocabulary_digest: str

    def __post_init__(self) -> None:
        for label, item in (
            ("predicate digest", self.predicate_digest),
            ("predicate algorithm digest", self.predicate_algorithm_digest),
            ("candidate digest", self.candidate_digest),
            ("formula digest", self.formula_digest),
            ("selected vocabulary digest", self.vocabulary_digest),
        ):
            _digest(item, label)
        if type(self.binding_spec) is not ObjectSceneAnchorBindingSpec:
            raise TypeError("query vocabulary binding spec has the wrong type")
        if (
            ObjectSceneAnchorBindingSpec.from_data(self.binding_spec.to_data())
            != self.binding_spec
        ):
            raise ObjectSceneAnchorPythonQueryError(
                "query vocabulary binding spec differs"
            )
        if (
            type(self.entries) is not tuple
            or not 1 <= len(self.entries) <= ANCHOR_MAX_CONJUNCTS
            or any(
                type(item) is not ObjectSceneAnchorObserverVocabularyEntry
                for item in self.entries
            )
            or tuple(item.witness_digest for item in self.entries)
            != tuple(sorted({item.witness_digest for item in self.entries}))
            or len({item.witness_id for item in self.entries}) != len(self.entries)
        ):
            raise ObjectSceneAnchorPythonQueryError(
                "query vocabulary entries are not the exact selected order"
            )
        restored_entries = tuple(
            ObjectSceneAnchorObserverVocabularyEntry.from_data(item.to_data())
            for item in self.entries
        )
        if restored_entries != self.entries:
            raise ObjectSceneAnchorPythonQueryError(
                "query vocabulary entry differs"
            )
        if self.predicate_algorithm_digest != (
            object_scene_anchor_python_predicate_algorithm_digest()
        ):
            raise ObjectSceneAnchorPythonQueryError(
                "query vocabulary predicate algorithm differs"
            )
        if self.vocabulary_digest != canonical_digest(_vocabulary_content(self)):
            raise ObjectSceneAnchorPythonQueryError(
                "query vocabulary digest differs"
            )

    @classmethod
    def create(
        cls, predicate: ObjectSceneAnchorPythonPredicate
    ) -> "ObjectSceneAnchorPythonQueryVocabulary":
        if type(predicate) is not ObjectSceneAnchorPythonPredicate:
            raise TypeError("predicate must be exact ObjectSceneAnchorPythonPredicate")
        frozen = ObjectSceneAnchorPythonPredicate.from_data(predicate.to_data())
        values = {
            "predicate_digest": frozen.predicate_digest,
            "predicate_algorithm_digest": frozen.algorithm_digest,
            "candidate_digest": frozen.candidate.candidate_digest,
            "formula_digest": frozen.formula.formula_digest,
            "binding_spec": frozen.binding_spec,
            "entries": frozen.affirmative_witness_entries,
        }
        provisional = object.__new__(cls)
        for name, item in values.items():
            object.__setattr__(provisional, name, item)
        return cls(
            **values,
            vocabulary_digest=canonical_digest(_vocabulary_content(provisional)),
        )

    def assert_matches_predicate(
        self, predicate: ObjectSceneAnchorPythonPredicate
    ) -> ObjectSceneAnchorPythonPredicate:
        if type(predicate) is not ObjectSceneAnchorPythonPredicate:
            raise TypeError("predicate must be exact ObjectSceneAnchorPythonPredicate")
        frozen = ObjectSceneAnchorPythonPredicate.from_data(predicate.to_data())
        if self != ObjectSceneAnchorPythonQueryVocabulary.create(frozen):
            raise ObjectSceneAnchorPythonQueryError(
                "query vocabulary differs from the frozen predicate"
            )
        return frozen

    @property
    def witness_inventory(self) -> tuple[tuple[str, str], ...]:
        return tuple((item.witness_id, item.witness_digest) for item in self.entries)

    def to_data(self) -> dict[str, object]:
        return {
            **_vocabulary_content(self),
            "vocabulary_digest": self.vocabulary_digest,
        }

    @classmethod
    def from_data(cls, value: object) -> "ObjectSceneAnchorPythonQueryVocabulary":
        raw = _exact_fields(
            value,
            {
                "schema",
                "predicate_digest",
                "predicate_algorithm_digest",
                "candidate_digest",
                "formula_digest",
                "binding_spec",
                "entries",
                "ordering",
                "complete_selected_vocabulary",
                "support_language_payload_present",
                *_authority_data(),
                "vocabulary_digest",
            },
            "query vocabulary",
        )
        if (
            raw["schema"] != OBJECT_SCENE_ANCHOR_PYTHON_QUERY_VOCABULARY_SCHEMA
            or raw["ordering"]
            != "selected-candidate-witness-digest-ascending"
            or raw["complete_selected_vocabulary"] is not True
            or raw["support_language_payload_present"] is not False
            or any(raw[key] != item for key, item in _authority_data().items())
            or not isinstance(raw["binding_spec"], Mapping)
            or not isinstance(raw["entries"], list)
        ):
            raise ObjectSceneAnchorPythonQueryError(
                "query vocabulary policy differs"
            )
        result = cls(
            raw["predicate_digest"],
            raw["predicate_algorithm_digest"],
            raw["candidate_digest"],
            raw["formula_digest"],
            ObjectSceneAnchorBindingSpec.from_data(raw["binding_spec"]),
            tuple(
                ObjectSceneAnchorObserverVocabularyEntry.from_data(item)
                for item in raw["entries"]
            ),
            raw["vocabulary_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectSceneAnchorPythonQueryError(
                "query vocabulary is not canonical"
            )
        return result


def freeze_object_scene_anchor_python_query_vocabulary(
    predicate: ObjectSceneAnchorPythonPredicate,
) -> ObjectSceneAnchorPythonQueryVocabulary:
    """Public freeze boundary for the selected predicate vocabulary."""

    return ObjectSceneAnchorPythonQueryVocabulary.create(predicate)


def _row_content(
    value: "ObjectSceneAnchorPythonQueryBindingRow",
) -> dict[str, object]:
    return {
        "schema": OBJECT_SCENE_ANCHOR_PYTHON_QUERY_ROW_SCHEMA,
        "binding_digest": value.binding_digest,
        "cells": [item.to_data() for item in value.cells],
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class ObjectSceneAnchorPythonQueryBindingRow:
    """All selected witness dispositions for one exact eligible binding."""

    binding_digest: str
    cells: tuple[ObjectSceneAnchorWitnessCell, ...]
    row_digest: str

    def __post_init__(self) -> None:
        _digest(self.binding_digest, "query row binding digest")
        if (
            type(self.cells) is not tuple
            or not self.cells
            or any(
                type(item) is not ObjectSceneAnchorWitnessCell
                for item in self.cells
            )
            or any(item.binding_digest != self.binding_digest for item in self.cells)
            or len({(item.witness_id, item.witness_digest) for item in self.cells})
            != len(self.cells)
        ):
            raise ObjectSceneAnchorPythonQueryError("query binding row differs")
        restored = tuple(
            ObjectSceneAnchorWitnessCell.from_data(item.to_data())
            for item in self.cells
        )
        if restored != self.cells:
            raise ObjectSceneAnchorPythonQueryError("query row cell differs")
        _digest(self.row_digest, "query binding row digest")
        if self.row_digest != canonical_digest(_row_content(self)):
            raise ObjectSceneAnchorPythonQueryError(
                "query binding row digest differs"
            )

    @classmethod
    def create(
        cls, binding_digest: str, cells: Sequence[ObjectSceneAnchorWitnessCell]
    ) -> "ObjectSceneAnchorPythonQueryBindingRow":
        values = {"binding_digest": binding_digest, "cells": tuple(cells)}
        provisional = object.__new__(cls)
        for name, item in values.items():
            object.__setattr__(provisional, name, item)
        return cls(
            **values,
            row_digest=canonical_digest(_row_content(provisional)),
        )

    def to_data(self) -> dict[str, object]:
        return {**_row_content(self), "row_digest": self.row_digest}

    @classmethod
    def from_data(cls, value: object) -> "ObjectSceneAnchorPythonQueryBindingRow":
        raw = _exact_fields(
            value,
            {"schema", "binding_digest", "cells", *_authority_data(), "row_digest"},
            "query binding row",
        )
        if (
            raw["schema"] != OBJECT_SCENE_ANCHOR_PYTHON_QUERY_ROW_SCHEMA
            or any(raw[key] != item for key, item in _authority_data().items())
            or not isinstance(raw["cells"], list)
        ):
            raise ObjectSceneAnchorPythonQueryError("query row policy differs")
        result = cls(
            raw["binding_digest"],
            tuple(
                ObjectSceneAnchorWitnessCell.from_data(item)
                for item in raw["cells"]
            ),
            raw["row_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectSceneAnchorPythonQueryError("query row is not canonical")
        return result


def _object_content(
    value: "ObjectSceneAnchorPythonQueryObjectObservation",
) -> dict[str, object]:
    return {
        "schema": OBJECT_SCENE_ANCHOR_PYTHON_QUERY_OBJECT_SCHEMA,
        "catalog": value.catalog.to_data(),
        "rows": [item.to_data() for item in value.rows],
        "one_row_per_eligible_binding": True,
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class ObjectSceneAnchorPythonQueryObjectObservation:
    """The complete selected-witness rectangle for one inventoried object."""

    catalog: ObjectSceneAnchorBindingCatalog
    rows: tuple[ObjectSceneAnchorPythonQueryBindingRow, ...]
    object_observation_digest: str

    def __post_init__(self) -> None:
        if type(self.catalog) is not ObjectSceneAnchorBindingCatalog:
            raise TypeError("query object catalog has the wrong type")
        if (
            ObjectSceneAnchorBindingCatalog.from_data(self.catalog.to_data())
            != self.catalog
        ):
            raise ObjectSceneAnchorPythonQueryError("query object catalog differs")
        if (
            type(self.rows) is not tuple
            or any(
                type(item) is not ObjectSceneAnchorPythonQueryBindingRow
                for item in self.rows
            )
        ):
            raise ObjectSceneAnchorPythonQueryError("query object rows differ")
        if self.catalog.hard_disposition is Disposition.PRESENT:
            if tuple(item.binding_digest for item in self.rows) != tuple(
                item.binding_digest for item in self.catalog.bindings
            ):
                raise ObjectSceneAnchorPythonQueryError(
                    "query object does not exhaust eligible bindings in order"
                )
        elif self.rows:
            raise ObjectSceneAnchorPythonQueryError(
                "query object with no hard bindings cannot contain rows"
            )
        _digest(self.object_observation_digest, "query object observation digest")
        if self.object_observation_digest != canonical_digest(_object_content(self)):
            raise ObjectSceneAnchorPythonQueryError(
                "query object observation digest differs"
            )

    @property
    def object_id(self) -> str:
        return self.catalog.object_id

    @classmethod
    def create(
        cls,
        catalog: ObjectSceneAnchorBindingCatalog,
        rows: Sequence[ObjectSceneAnchorPythonQueryBindingRow],
    ) -> "ObjectSceneAnchorPythonQueryObjectObservation":
        values = {"catalog": catalog, "rows": tuple(rows)}
        provisional = object.__new__(cls)
        for name, item in values.items():
            object.__setattr__(provisional, name, item)
        return cls(
            **values,
            object_observation_digest=canonical_digest(_object_content(provisional)),
        )

    def to_data(self) -> dict[str, object]:
        return {
            **_object_content(self),
            "object_observation_digest": self.object_observation_digest,
        }

    @classmethod
    def from_data(
        cls, value: object
    ) -> "ObjectSceneAnchorPythonQueryObjectObservation":
        raw = _exact_fields(
            value,
            {
                "schema",
                "catalog",
                "rows",
                "one_row_per_eligible_binding",
                *_authority_data(),
                "object_observation_digest",
            },
            "query object observation",
        )
        if (
            raw["schema"] != OBJECT_SCENE_ANCHOR_PYTHON_QUERY_OBJECT_SCHEMA
            or raw["one_row_per_eligible_binding"] is not True
            or any(raw[key] != item for key, item in _authority_data().items())
            or not isinstance(raw["catalog"], Mapping)
            or not isinstance(raw["rows"], list)
        ):
            raise ObjectSceneAnchorPythonQueryError(
                "query object observation policy differs"
            )
        result = cls(
            ObjectSceneAnchorBindingCatalog.from_data(raw["catalog"]),
            tuple(
                ObjectSceneAnchorPythonQueryBindingRow.from_data(item)
                for item in raw["rows"]
            ),
            raw["object_observation_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectSceneAnchorPythonQueryError(
                "query object observation is not canonical"
            )
        return result


def _observation_content(
    value: "ObjectSceneAnchorPythonQueryObservation",
) -> dict[str, object]:
    return {
        "schema": OBJECT_SCENE_ANCHOR_PYTHON_QUERY_OBSERVATION_SCHEMA,
        "algorithm_id": OBJECT_SCENE_ANCHOR_PYTHON_QUERY_ALGORITHM_ID,
        "source_digest": value.source_digest,
        "algorithm_digest": value.algorithm_digest,
        "predicate_digest": value.predicate_digest,
        "vocabulary": value.vocabulary.to_data(),
        "panel_id": value.panel_id,
        "panel_digest": value.panel_digest,
        "panel_manifest_digest": value.panel_manifest_digest,
        "inventory_digest": value.inventory_digest,
        "object_ids": list(value.object_ids),
        "object_decision_manifest_digests": list(
            value.object_decision_manifest_digests
        ),
        "objects": [item.to_data() for item in value.objects],
        "panel_role_present": False,
        "panel_label_present": False,
        "support_language_payload_present": False,
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class ObjectSceneAnchorPythonQueryObservation:
    """A complete neutral panel observation for one frozen predicate."""

    source_digest: str
    algorithm_digest: str
    predicate_digest: str
    vocabulary: ObjectSceneAnchorPythonQueryVocabulary
    panel_id: str
    panel_digest: str
    panel_manifest_digest: str
    inventory_digest: str
    object_ids: tuple[str, ...]
    object_decision_manifest_digests: tuple[str, ...]
    objects: tuple[ObjectSceneAnchorPythonQueryObjectObservation, ...]
    observation_digest: str

    def __post_init__(self) -> None:
        for label, item in (
            ("query source digest", self.source_digest),
            ("query algorithm digest", self.algorithm_digest),
            ("query predicate digest", self.predicate_digest),
            ("panel digest", self.panel_digest),
            ("panel manifest digest", self.panel_manifest_digest),
            ("panel inventory digest", self.inventory_digest),
            ("query observation digest", self.observation_digest),
        ):
            _digest(item, label)
        _panel_id(self.panel_id)
        if type(self.vocabulary) is not ObjectSceneAnchorPythonQueryVocabulary:
            raise TypeError("query observation vocabulary has the wrong type")
        vocabulary = ObjectSceneAnchorPythonQueryVocabulary.from_data(
            self.vocabulary.to_data()
        )
        if (
            self.source_digest != object_scene_anchor_python_query_source_digest()
            or self.algorithm_digest
            != object_scene_anchor_python_query_algorithm_digest()
            or self.predicate_digest != vocabulary.predicate_digest
            or vocabulary != self.vocabulary
        ):
            raise ObjectSceneAnchorPythonQueryError(
                "query observation authority or predicate binding differs"
            )
        if (
            type(self.object_ids) is not tuple
            or self.object_ids
            != tuple(f"object_{index:04d}" for index in range(len(self.object_ids)))
            or any(_OBJECT_ID.fullmatch(item) is None for item in self.object_ids)
            or type(self.object_decision_manifest_digests) is not tuple
            or len(self.object_decision_manifest_digests) != len(self.object_ids)
            or type(self.objects) is not tuple
            or len(self.objects) != len(self.object_ids)
            or any(
                type(item) is not ObjectSceneAnchorPythonQueryObjectObservation
                for item in self.objects
            )
            or tuple(item.object_id for item in self.objects) != self.object_ids
        ):
            raise ObjectSceneAnchorPythonQueryError(
                "query observation does not exhaust the object inventory"
            )
        for item in self.object_decision_manifest_digests:
            _digest(item, "object decision manifest digest")
        expected_decisions = dict(
            zip(
                self.object_ids,
                self.object_decision_manifest_digests,
                strict=True,
            )
        )
        witness_inventory = vocabulary.witness_inventory
        for item in self.objects:
            if (
                item.catalog.binding_spec != vocabulary.binding_spec
                or item.catalog.decision_manifest_digest
                != expected_decisions[item.object_id]
            ):
                raise ObjectSceneAnchorPythonQueryError(
                    "query object differs from its predicate spec or decision manifest"
                )
            if any(
                tuple((cell.witness_id, cell.witness_digest) for cell in row.cells)
                != witness_inventory
                for row in item.rows
            ):
                raise ObjectSceneAnchorPythonQueryError(
                    "query row does not exhaust the selected vocabulary in order"
                )
        if self.observation_digest != canonical_digest(_observation_content(self)):
            raise ObjectSceneAnchorPythonQueryError(
                "query observation digest differs"
            )

    def to_data(self) -> dict[str, object]:
        return {
            **_observation_content(self),
            "observation_digest": self.observation_digest,
        }

    @classmethod
    def from_data(cls, value: object) -> "ObjectSceneAnchorPythonQueryObservation":
        raw = _exact_fields(
            value,
            {
                "schema",
                "algorithm_id",
                "source_digest",
                "algorithm_digest",
                "predicate_digest",
                "vocabulary",
                "panel_id",
                "panel_digest",
                "panel_manifest_digest",
                "inventory_digest",
                "object_ids",
                "object_decision_manifest_digests",
                "objects",
                "panel_role_present",
                "panel_label_present",
                "support_language_payload_present",
                *_authority_data(),
                "observation_digest",
            },
            "query observation",
        )
        if (
            raw["schema"] != OBJECT_SCENE_ANCHOR_PYTHON_QUERY_OBSERVATION_SCHEMA
            or raw["algorithm_id"] != OBJECT_SCENE_ANCHOR_PYTHON_QUERY_ALGORITHM_ID
            or raw["panel_role_present"] is not False
            or raw["panel_label_present"] is not False
            or raw["support_language_payload_present"] is not False
            or any(raw[key] != item for key, item in _authority_data().items())
            or not isinstance(raw["vocabulary"], Mapping)
            or not isinstance(raw["object_ids"], list)
            or not isinstance(raw["object_decision_manifest_digests"], list)
            or not isinstance(raw["objects"], list)
        ):
            raise ObjectSceneAnchorPythonQueryError(
                "query observation policy differs"
            )
        result = cls(
            raw["source_digest"],
            raw["algorithm_digest"],
            raw["predicate_digest"],
            ObjectSceneAnchorPythonQueryVocabulary.from_data(raw["vocabulary"]),
            raw["panel_id"],
            raw["panel_digest"],
            raw["panel_manifest_digest"],
            raw["inventory_digest"],
            tuple(raw["object_ids"]),
            tuple(raw["object_decision_manifest_digests"]),
            tuple(
                ObjectSceneAnchorPythonQueryObjectObservation.from_data(item)
                for item in raw["objects"]
            ),
            raw["observation_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectSceneAnchorPythonQueryError(
                "query observation is not canonical"
            )
        return result


def build_object_scene_anchor_python_query_observation(
    *,
    predicate: ObjectSceneAnchorPythonPredicate,
    panel_id: str,
    panel_manifest: ObjectSceneAnchorPanelDecisionManifest,
    cells: Sequence[ObjectSceneAnchorWitnessCell],
) -> ObjectSceneAnchorPythonQueryObservation:
    """Build the exact object/binding/selected-witness rectangle."""

    if type(predicate) is not ObjectSceneAnchorPythonPredicate:
        raise TypeError("predicate must be exact ObjectSceneAnchorPythonPredicate")
    if type(panel_manifest) is not ObjectSceneAnchorPanelDecisionManifest:
        raise TypeError(
            "panel_manifest must be exact ObjectSceneAnchorPanelDecisionManifest"
        )
    frozen = ObjectSceneAnchorPythonPredicate.from_data(predicate.to_data())
    manifest = ObjectSceneAnchorPanelDecisionManifest.from_data(
        panel_manifest.to_data()
    )
    identifier = _panel_id(panel_id)
    vocabulary = ObjectSceneAnchorPythonQueryVocabulary.create(frozen)
    if isinstance(cells, (str, bytes)) or not isinstance(cells, Sequence):
        raise TypeError("query witness cells must be a finite sequence")
    supplied = tuple(cells)
    if any(type(item) is not ObjectSceneAnchorWitnessCell for item in supplied):
        raise TypeError("every query witness cell must be an exact witness cell")
    supplied = tuple(
        ObjectSceneAnchorWitnessCell.from_data(item.to_data()) for item in supplied
    )

    catalogs = tuple(
        build_object_scene_anchor_binding_catalog(
            decision,
            vocabulary.binding_spec,
            expected_object_id=object_id,
        )
        for object_id, decision in zip(
            manifest.object_ids, manifest.object_decisions, strict=True
        )
    )
    expected_keys = tuple(
        (binding.binding_digest, entry.witness_digest)
        for catalog in catalogs
        if catalog.hard_disposition is Disposition.PRESENT
        for binding in catalog.bindings
        for entry in vocabulary.entries
    )
    witness_ids = {
        item.witness_digest: item.witness_id for item in vocabulary.entries
    }
    supplied_by_key: dict[tuple[str, str], ObjectSceneAnchorWitnessCell] = {}
    for item in supplied:
        key = (item.binding_digest, item.witness_digest)
        if key in supplied_by_key:
            raise ObjectSceneAnchorPythonQueryError(
                "query witness cells contain a duplicate binding-witness key"
            )
        if witness_ids.get(item.witness_digest) != item.witness_id:
            raise ObjectSceneAnchorPythonQueryError(
                "query witness cell is outside the selected vocabulary"
            )
        supplied_by_key[key] = item
    if (
        set(supplied_by_key) != set(expected_keys)
        or len(supplied) != len(expected_keys)
    ):
        raise ObjectSceneAnchorPythonQueryError(
            "query witness cells do not exactly cover every eligible binding"
        )

    objects = []
    for catalog in catalogs:
        rows = tuple(
            ObjectSceneAnchorPythonQueryBindingRow.create(
                binding.binding_digest,
                tuple(
                    supplied_by_key[(binding.binding_digest, entry.witness_digest)]
                    for entry in vocabulary.entries
                ),
            )
            for binding in catalog.bindings
        )
        objects.append(
            ObjectSceneAnchorPythonQueryObjectObservation.create(catalog, rows)
        )
    values = {
        "source_digest": object_scene_anchor_python_query_source_digest(),
        "algorithm_digest": object_scene_anchor_python_query_algorithm_digest(),
        "predicate_digest": frozen.predicate_digest,
        "vocabulary": vocabulary,
        "panel_id": identifier,
        "panel_digest": manifest.panel_digest,
        "panel_manifest_digest": manifest.manifest_digest,
        "inventory_digest": manifest.inventory_digest,
        "object_ids": manifest.object_ids,
        "object_decision_manifest_digests": tuple(
            item.manifest_digest for item in manifest.object_decisions
        ),
        "objects": tuple(objects),
    }
    provisional = object.__new__(ObjectSceneAnchorPythonQueryObservation)
    for name, item in values.items():
        object.__setattr__(provisional, name, item)
    return ObjectSceneAnchorPythonQueryObservation(
        **values,
        observation_digest=canonical_digest(_observation_content(provisional)),
    )


def cold_verify_object_scene_anchor_python_query_observation(
    observation: ObjectSceneAnchorPythonQueryObservation,
    *,
    predicate: ObjectSceneAnchorPythonPredicate,
    panel_manifest: ObjectSceneAnchorPanelDecisionManifest,
) -> ObjectSceneAnchorPythonQueryObservation:
    """Rebuild one observation from its exact cells and frozen input artifacts."""

    if type(observation) is not ObjectSceneAnchorPythonQueryObservation:
        raise TypeError(
            "observation must be exact ObjectSceneAnchorPythonQueryObservation"
        )
    restored = ObjectSceneAnchorPythonQueryObservation.from_data(
        observation.to_data()
    )
    cells = tuple(
        cell
        for item in restored.objects
        for row in item.rows
        for cell in row.cells
    )
    expected = build_object_scene_anchor_python_query_observation(
        predicate=predicate,
        panel_id=restored.panel_id,
        panel_manifest=panel_manifest,
        cells=cells,
    )
    if restored != expected:
        raise ObjectSceneAnchorPythonQueryError(
            "query observation differs from cold replay"
        )
    return restored


def _scene_and(values: Sequence[Disposition]) -> Disposition:
    row = tuple(values)
    if not row:
        return Disposition.PRESENT
    if Disposition.ERROR in row:
        return Disposition.ERROR
    if Disposition.CERTIFIED_ABSENT in row:
        return Disposition.CERTIFIED_ABSENT
    if all(item is Disposition.PRESENT for item in row):
        return Disposition.PRESENT
    return Disposition.INDETERMINATE


def _scene_or(values: Sequence[Disposition]) -> Disposition:
    row = tuple(values)
    if not row:
        return Disposition.CERTIFIED_ABSENT
    if Disposition.ERROR in row:
        return Disposition.ERROR
    if Disposition.PRESENT in row:
        return Disposition.PRESENT
    if all(item is Disposition.CERTIFIED_ABSENT for item in row):
        return Disposition.CERTIFIED_ABSENT
    return Disposition.INDETERMINATE


def _evaluate_observation(
    observation: ObjectSceneAnchorPythonQueryObservation,
) -> Disposition:
    states: list[Disposition] = []
    for item in observation.objects:
        hard = item.catalog.hard_disposition
        if hard is Disposition.ERROR:
            states.append(Disposition.ERROR)
        elif hard is Disposition.INDETERMINATE:
            states.append(Disposition.INDETERMINATE)
        elif hard is Disposition.PRESENT:
            states.extend(
                _scene_and(tuple(cell.disposition for cell in row.cells))
                for row in item.rows
            )
        # A certified-complete empty catalog contributes no existential binding.
    return _scene_or(states)


def _evaluation_content(
    value: "ObjectSceneAnchorPythonQueryEvaluation",
) -> dict[str, object]:
    return {
        "schema": OBJECT_SCENE_ANCHOR_PYTHON_QUERY_EVALUATION_SCHEMA,
        "algorithm_id": OBJECT_SCENE_ANCHOR_PYTHON_QUERY_ALGORITHM_ID,
        "algorithm_digest": value.algorithm_digest,
        "predicate_digest": value.predicate_digest,
        "observation_digest": value.observation_digest,
        "panel_id": value.panel_id,
        "disposition": value.disposition.value,
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class ObjectSceneAnchorPythonQueryEvaluation:
    """Content-addressed P/A/I/E result for one predicate-scoped observation."""

    algorithm_digest: str
    predicate_digest: str
    observation_digest: str
    panel_id: str
    disposition: Disposition
    evaluation_digest: str

    def __post_init__(self) -> None:
        for label, item in (
            ("query evaluation algorithm digest", self.algorithm_digest),
            ("query evaluation predicate digest", self.predicate_digest),
            ("query observation digest", self.observation_digest),
            ("query evaluation digest", self.evaluation_digest),
        ):
            _digest(item, label)
        _panel_id(self.panel_id)
        if not isinstance(self.disposition, Disposition):
            raise TypeError("query evaluation disposition has the wrong type")
        if self.algorithm_digest != object_scene_anchor_python_query_algorithm_digest():
            raise ObjectSceneAnchorPythonQueryError(
                "query evaluation algorithm differs"
            )
        if self.evaluation_digest != canonical_digest(_evaluation_content(self)):
            raise ObjectSceneAnchorPythonQueryError(
                "query evaluation digest differs"
            )

    def to_data(self) -> dict[str, object]:
        return {
            **_evaluation_content(self),
            "evaluation_digest": self.evaluation_digest,
        }

    @classmethod
    def from_data(cls, value: object) -> "ObjectSceneAnchorPythonQueryEvaluation":
        raw = _exact_fields(
            value,
            {
                "schema",
                "algorithm_id",
                "algorithm_digest",
                "predicate_digest",
                "observation_digest",
                "panel_id",
                "disposition",
                *_authority_data(),
                "evaluation_digest",
            },
            "query evaluation",
        )
        if (
            raw["schema"] != OBJECT_SCENE_ANCHOR_PYTHON_QUERY_EVALUATION_SCHEMA
            or raw["algorithm_id"] != OBJECT_SCENE_ANCHOR_PYTHON_QUERY_ALGORITHM_ID
            or any(raw[key] != item for key, item in _authority_data().items())
        ):
            raise ObjectSceneAnchorPythonQueryError(
                "query evaluation policy differs"
            )
        result = cls(
            raw["algorithm_digest"],
            raw["predicate_digest"],
            raw["observation_digest"],
            raw["panel_id"],
            _disposition(raw["disposition"]),
            raw["evaluation_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectSceneAnchorPythonQueryError(
                "query evaluation is not canonical"
            )
        return result


def evaluate_object_scene_anchor_python_query_observation(
    predicate: ObjectSceneAnchorPythonPredicate,
    observation: ObjectSceneAnchorPythonQueryObservation,
) -> ObjectSceneAnchorPythonQueryEvaluation:
    """Evaluate one selected-vocabulary observation without a support language."""

    if type(predicate) is not ObjectSceneAnchorPythonPredicate:
        raise TypeError("predicate must be exact ObjectSceneAnchorPythonPredicate")
    if type(observation) is not ObjectSceneAnchorPythonQueryObservation:
        raise TypeError(
            "observation must be exact ObjectSceneAnchorPythonQueryObservation"
        )
    frozen = ObjectSceneAnchorPythonPredicate.from_data(predicate.to_data())
    restored = ObjectSceneAnchorPythonQueryObservation.from_data(
        observation.to_data()
    )
    restored.vocabulary.assert_matches_predicate(frozen)
    if restored.predicate_digest != frozen.predicate_digest:
        raise ObjectSceneAnchorPythonQueryError(
            "query observation belongs to another frozen predicate"
        )
    values = {
        "algorithm_digest": restored.algorithm_digest,
        "predicate_digest": frozen.predicate_digest,
        "observation_digest": restored.observation_digest,
        "panel_id": restored.panel_id,
        "disposition": _evaluate_observation(restored),
    }
    provisional = object.__new__(ObjectSceneAnchorPythonQueryEvaluation)
    for name, item in values.items():
        object.__setattr__(provisional, name, item)
    return ObjectSceneAnchorPythonQueryEvaluation(
        **values,
        evaluation_digest=canonical_digest(_evaluation_content(provisional)),
    )


def cold_verify_object_scene_anchor_python_query_evaluation(
    evaluation: ObjectSceneAnchorPythonQueryEvaluation,
    *,
    predicate: ObjectSceneAnchorPythonPredicate,
    observation: ObjectSceneAnchorPythonQueryObservation,
) -> ObjectSceneAnchorPythonQueryEvaluation:
    """Recompute one query decision from frozen Python artifacts only."""

    if type(evaluation) is not ObjectSceneAnchorPythonQueryEvaluation:
        raise TypeError(
            "evaluation must be exact ObjectSceneAnchorPythonQueryEvaluation"
        )
    restored = ObjectSceneAnchorPythonQueryEvaluation.from_data(
        evaluation.to_data()
    )
    expected = evaluate_object_scene_anchor_python_query_observation(
        predicate, observation
    )
    if restored != expected:
        raise ObjectSceneAnchorPythonQueryError(
            "query evaluation differs from cold replay"
        )
    return restored


__all__ = (
    "OBJECT_SCENE_ANCHOR_PYTHON_QUERY_ALGORITHM_ID",
    "OBJECT_SCENE_ANCHOR_PYTHON_QUERY_EVALUATION_SCHEMA",
    "OBJECT_SCENE_ANCHOR_PYTHON_QUERY_OBJECT_SCHEMA",
    "OBJECT_SCENE_ANCHOR_PYTHON_QUERY_OBSERVATION_SCHEMA",
    "OBJECT_SCENE_ANCHOR_PYTHON_QUERY_ROW_SCHEMA",
    "OBJECT_SCENE_ANCHOR_PYTHON_QUERY_VOCABULARY_SCHEMA",
    "ObjectSceneAnchorPythonQueryBindingRow",
    "ObjectSceneAnchorPythonQueryError",
    "ObjectSceneAnchorPythonQueryEvaluation",
    "ObjectSceneAnchorPythonQueryObjectObservation",
    "ObjectSceneAnchorPythonQueryObservation",
    "ObjectSceneAnchorPythonQueryVocabulary",
    "build_object_scene_anchor_python_query_observation",
    "cold_verify_object_scene_anchor_python_query_evaluation",
    "cold_verify_object_scene_anchor_python_query_observation",
    "evaluate_object_scene_anchor_python_query_observation",
    "freeze_object_scene_anchor_python_query_vocabulary",
    "object_scene_anchor_python_query_algorithm_digest",
    "object_scene_anchor_python_query_source_digest",
)
