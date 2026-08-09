"""Freeze and execute one positive anchor predicate in canonical Python.

The support version space is the admissibility authority.  A selector may name
exactly one member of its nonempty survivor set, but may not edit the candidate
or its affirmative witnesses.  The resulting closed formula ranges over every
eligible binding of every inventoried object in a neutral panel observation.
"""

from __future__ import annotations

from bongard.runtime_source_snapshot import capture_loaded_source, verify_loaded_source


_LOADED_SOURCE_SHA256 = capture_loaded_source(__name__, __file__)

from dataclasses import dataclass
import importlib
import re
from typing import Any, Mapping, Sequence

from bongard.canonical import canonical_digest
from bongard.evidence import Disposition
from bongard.object_scene_anchor_bindings import ObjectSceneAnchorBindingSpec
from bongard.object_scene_anchor_observer import (
    ObjectSceneAnchorObserverVocabularyEntry,
)
from bongard.object_scene_anchor_version_space import (
    ANCHOR_MAX_CONJUNCTS,
    ObjectSceneAnchorOrientation,
    ObjectSceneAnchorPanelWitnessEvaluation,
    ObjectSceneAnchorPredicateCandidate,
    ObjectSceneAnchorSupportVersionSpace,
    object_scene_anchor_version_space_algorithm_digest,
)
from bongard.python_predicate_authority import PYTHON_PREDICATE_AUTHORITY_ID


OBJECT_SCENE_ANCHOR_SELECTION_COMMITMENT_SCHEMA = (
    "gkm.object-scene-anchor-selection-commitment.v1"
)
OBJECT_SCENE_ANCHOR_PYTHON_FORMULA_SCHEMA = (
    "gkm.object-scene-anchor-python-closed-formula.v1"
)
OBJECT_SCENE_ANCHOR_PYTHON_PREDICATE_SCHEMA = (
    "gkm.object-scene-anchor-python-predicate.v1"
)
OBJECT_SCENE_ANCHOR_PYTHON_EVALUATION_SCHEMA = (
    "gkm.object-scene-anchor-python-predicate-evaluation.v1"
)
OBJECT_SCENE_ANCHOR_PYTHON_PREDICATE_ALGORITHM_ID = (
    "bongard.object-scene-anchor-python-predicate/exhaustive-same-binding-v1"
)
OBJECT_SCENE_ANCHOR_CLOSED_FORMULA = (
    "exists one eligible binding across all inventoried objects satisfying "
    "the selected positive witness conjunction"
)

_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_SELECTION_KINDS = frozenset(("external_exact_selection", "exact_rank_response"))


class ObjectSceneAnchorPythonPredicateError(ValueError):
    """A selection, frozen predicate, or evaluation is not canonical."""


def _authority_data() -> dict[str, object]:
    return {
        "predicate_authority_id": PYTHON_PREDICATE_AUTHORITY_ID,
        "python_is_canonical_authority": True,
        "pure_python_evaluation": True,
        "positive_witnesses_only": True,
        "negation_available": False,
        "polarity_flip_available": False,
    }


def _formula_policy_data() -> dict[str, object]:
    return {
        "complete_object_inventory_required": True,
        "same_binding_required": True,
        "error_dominant": True,
        "failed_observation_is_negative": False,
    }


def _exact_fields(
    value: object, expected: set[str] | frozenset[str], label: str
) -> Mapping[str, Any]:
    if (
        not isinstance(value, Mapping)
        or any(not isinstance(key, str) for key in value)
        or set(value) != set(expected)
    ):
        raise ObjectSceneAnchorPythonPredicateError(f"{label} fields differ")
    return value


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise ObjectSceneAnchorPythonPredicateError(
            f"{label} must be a lowercase SHA-256"
        )
    return value


def _orientation(value: object) -> ObjectSceneAnchorOrientation:
    try:
        return ObjectSceneAnchorOrientation(value)
    except (TypeError, ValueError) as exc:
        raise ObjectSceneAnchorPythonPredicateError(
            "selection orientation differs"
        ) from exc


def _disposition(value: object) -> Disposition:
    try:
        return Disposition(value)
    except (TypeError, ValueError) as exc:
        raise ObjectSceneAnchorPythonPredicateError(
            "evaluation disposition differs"
        ) from exc


def object_scene_anchor_python_predicate_source_digest() -> str:
    """Return the authenticated bytes loaded for this evaluator."""

    return verify_loaded_source(__name__, expected_source_sha256=_LOADED_SOURCE_SHA256)


def object_scene_anchor_python_predicate_algorithm_digest() -> str:
    """Bind the executable formula semantics to both Python source layers."""

    return canonical_digest(
        {
            "schema": "gkm.object-scene-anchor-python-predicate-algorithm.v1",
            "algorithm_id": OBJECT_SCENE_ANCHOR_PYTHON_PREDICATE_ALGORITHM_ID,
            "source_digest": object_scene_anchor_python_predicate_source_digest(),
            "version_space_algorithm_digest": (
                object_scene_anchor_version_space_algorithm_digest()
            ),
            "closed_formula": OBJECT_SCENE_ANCHOR_CLOSED_FORMULA,
            "row_conjunction": "E_then_A_then_all_P_else_I",
            "binding_existential": "E_then_P_then_all_A_else_I",
            **_formula_policy_data(),
            **_authority_data(),
        }
    )


def _survivor_set_digest(
    version_space_digest: str, survivor_candidate_digests: Sequence[str]
) -> str:
    return canonical_digest(
        {
            "schema": "gkm.object-scene-anchor-survivor-set-commitment.v1",
            "version_space_digest": version_space_digest,
            "survivor_candidate_digests": list(survivor_candidate_digests),
            "complete_exact_set": True,
        }
    )


def _selection_content(
    value: "ObjectSceneAnchorSelectionCommitment",
) -> dict[str, object]:
    return {
        "schema": OBJECT_SCENE_ANCHOR_SELECTION_COMMITMENT_SCHEMA,
        "version_space_digest": value.version_space_digest,
        "version_space_algorithm_digest": value.version_space_algorithm_digest,
        "language_digest": value.language_digest,
        "orientation": value.orientation.value,
        "survivor_candidate_digests": list(value.survivor_candidate_digests),
        "survivor_set_digest": value.survivor_set_digest,
        "selected_candidate_digest": value.selected_candidate_digest,
        "selection_kind": value.selection_kind,
        "selector_record_digest": value.selector_record_digest,
        "complete_exact_survivor_set_committed": True,
        "selector_may_edit_candidate": False,
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class ObjectSceneAnchorSelectionCommitment:
    """One selected digest bound to the complete verified survivor set."""

    version_space_digest: str
    version_space_algorithm_digest: str
    language_digest: str
    orientation: ObjectSceneAnchorOrientation
    survivor_candidate_digests: tuple[str, ...]
    survivor_set_digest: str
    selected_candidate_digest: str
    selection_kind: str
    selector_record_digest: str
    selection_commitment_digest: str

    def __post_init__(self) -> None:
        for label, item in (
            ("version-space digest", self.version_space_digest),
            ("version-space algorithm digest", self.version_space_algorithm_digest),
            ("language digest", self.language_digest),
            ("survivor-set digest", self.survivor_set_digest),
            ("selected candidate digest", self.selected_candidate_digest),
            ("selector record digest", self.selector_record_digest),
            ("selection commitment digest", self.selection_commitment_digest),
        ):
            _digest(item, label)
        if not isinstance(self.orientation, ObjectSceneAnchorOrientation):
            raise TypeError("selection orientation must be exact orientation enum")
        if (
            type(self.survivor_candidate_digests) is not tuple
            or not self.survivor_candidate_digests
            or len(set(self.survivor_candidate_digests))
            != len(self.survivor_candidate_digests)
            or self.selected_candidate_digest not in self.survivor_candidate_digests
        ):
            raise ObjectSceneAnchorPythonPredicateError(
                "selection requires one member of a nonempty exact survivor set"
            )
        for item in self.survivor_candidate_digests:
            _digest(item, "survivor candidate digest")
        if self.selection_kind not in _SELECTION_KINDS:
            raise ObjectSceneAnchorPythonPredicateError("selection kind differs")
        if self.version_space_algorithm_digest != (
            object_scene_anchor_version_space_algorithm_digest()
        ):
            raise ObjectSceneAnchorPythonPredicateError(
                "selection version-space algorithm differs"
            )
        if self.survivor_set_digest != _survivor_set_digest(
            self.version_space_digest, self.survivor_candidate_digests
        ):
            raise ObjectSceneAnchorPythonPredicateError(
                "selection survivor-set commitment differs"
            )
        if self.selection_commitment_digest != canonical_digest(
            _selection_content(self)
        ):
            raise ObjectSceneAnchorPythonPredicateError(
                "selection commitment digest differs"
            )

    @classmethod
    def create(
        cls,
        version_space: ObjectSceneAnchorSupportVersionSpace,
        *,
        selected_candidate_digest: str,
        selection_kind: str,
        selector_record_digest: str,
    ) -> "ObjectSceneAnchorSelectionCommitment":
        if type(version_space) is not ObjectSceneAnchorSupportVersionSpace:
            raise TypeError(
                "version_space must be exact ObjectSceneAnchorSupportVersionSpace"
            )
        version = ObjectSceneAnchorSupportVersionSpace.from_data(
            version_space.to_data()
        )
        selected = _digest(selected_candidate_digest, "selected candidate digest")
        source = _digest(selector_record_digest, "selector record digest")
        if not version.survivor_candidate_digests:
            raise ObjectSceneAnchorPythonPredicateError(
                "selection requires a nonempty exact survivor set"
            )
        if selected not in version.survivor_candidate_digests:
            raise ObjectSceneAnchorPythonPredicateError(
                "selected candidate is not an exact verified survivor"
            )
        values = {
            "version_space_digest": version.version_space_digest,
            "version_space_algorithm_digest": version.algorithm_digest,
            "language_digest": version.language.language_digest,
            "orientation": version.orientation,
            "survivor_candidate_digests": version.survivor_candidate_digests,
            "survivor_set_digest": _survivor_set_digest(
                version.version_space_digest,
                version.survivor_candidate_digests,
            ),
            "selected_candidate_digest": selected,
            "selection_kind": selection_kind,
            "selector_record_digest": source,
        }
        provisional = object.__new__(cls)
        for name, item in values.items():
            object.__setattr__(provisional, name, item)
        return cls(
            **values,
            selection_commitment_digest=canonical_digest(
                _selection_content(provisional)
            ),
        )

    def to_data(self) -> dict[str, object]:
        return {
            **_selection_content(self),
            "selection_commitment_digest": self.selection_commitment_digest,
        }

    @classmethod
    def from_data(cls, value: object) -> "ObjectSceneAnchorSelectionCommitment":
        raw = _exact_fields(
            value,
            {
                "schema",
                "version_space_digest",
                "version_space_algorithm_digest",
                "language_digest",
                "orientation",
                "survivor_candidate_digests",
                "survivor_set_digest",
                "selected_candidate_digest",
                "selection_kind",
                "selector_record_digest",
                "complete_exact_survivor_set_committed",
                "selector_may_edit_candidate",
                *_authority_data(),
                "selection_commitment_digest",
            },
            "selection commitment",
        )
        if (
            raw["schema"] != OBJECT_SCENE_ANCHOR_SELECTION_COMMITMENT_SCHEMA
            or raw["complete_exact_survivor_set_committed"] is not True
            or raw["selector_may_edit_candidate"] is not False
            or any(raw[key] != item for key, item in _authority_data().items())
            or not isinstance(raw["survivor_candidate_digests"], list)
        ):
            raise ObjectSceneAnchorPythonPredicateError(
                "selection commitment policy differs"
            )
        result = cls(
            raw["version_space_digest"],
            raw["version_space_algorithm_digest"],
            raw["language_digest"],
            _orientation(raw["orientation"]),
            tuple(raw["survivor_candidate_digests"]),
            raw["survivor_set_digest"],
            raw["selected_candidate_digest"],
            raw["selection_kind"],
            raw["selector_record_digest"],
            raw["selection_commitment_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectSceneAnchorPythonPredicateError(
                "selection commitment is not canonical"
            )
        return result


def object_scene_anchor_selection_commitment_from_rank_response(
    response: object,
    version_space: ObjectSceneAnchorSupportVersionSpace,
) -> ObjectSceneAnchorSelectionCommitment:
    """Adapt the exact rank response without making the ranker an import dependency."""

    if type(version_space) is not ObjectSceneAnchorSupportVersionSpace:
        raise TypeError(
            "version_space must be exact ObjectSceneAnchorSupportVersionSpace"
        )
    version = ObjectSceneAnchorSupportVersionSpace.from_data(version_space.to_data())
    try:
        module = importlib.import_module(
            "bongard.object_scene_anchor_candidate_ranker"
        )
        response_type = getattr(module, "ObjectSceneAnchorRankResponse")
    except (ImportError, AttributeError) as exc:
        raise ObjectSceneAnchorPythonPredicateError(
            "exact rank-response adapter is unavailable"
        ) from exc
    if not isinstance(response_type, type) or type(response) is not response_type:
        raise TypeError("response must be the exact anchor rank-response type")
    restored = response_type.from_data(response.to_data())
    rank_input = restored.rank_input
    if (
        restored != response
        or restored.version_space_digest != version.version_space_digest
        or rank_input.version_space_digest != version.version_space_digest
        or rank_input.version_space_algorithm_digest != version.algorithm_digest
        or rank_input.language_digest != version.language.language_digest
        or rank_input.survivor_candidate_digests
        != version.survivor_candidate_digests
    ):
        raise ObjectSceneAnchorPythonPredicateError(
            "rank response differs from the exact current survivor set"
        )
    return ObjectSceneAnchorSelectionCommitment.create(
        version,
        selected_candidate_digest=restored.selected_candidate_digest,
        selection_kind="exact_rank_response",
        selector_record_digest=restored.response_digest,
    )


def _formula_content(
    value: "ObjectSceneAnchorPythonClosedFormula",
) -> dict[str, object]:
    return {
        "schema": OBJECT_SCENE_ANCHOR_PYTHON_FORMULA_SCHEMA,
        "closed_formula": OBJECT_SCENE_ANCHOR_CLOSED_FORMULA,
        "quantifier": "exists",
        "domain": "eligible_bindings_across_all_inventoried_objects",
        "binding_scope": "one_same_binding",
        "conjunction": "selected_positive_witnesses",
        "witness_digests": list(value.witness_digests),
        "dispositions": [item.value for item in Disposition],
        **_formula_policy_data(),
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class ObjectSceneAnchorPythonClosedFormula:
    """Structured closed formula for one immutable positive conjunction."""

    witness_digests: tuple[str, ...]
    formula_digest: str

    def __post_init__(self) -> None:
        if (
            type(self.witness_digests) is not tuple
            or not 1 <= len(self.witness_digests) <= ANCHOR_MAX_CONJUNCTS
            or self.witness_digests != tuple(sorted(set(self.witness_digests)))
        ):
            raise ObjectSceneAnchorPythonPredicateError(
                "closed formula witness inventory differs"
            )
        for item in self.witness_digests:
            _digest(item, "formula witness digest")
        _digest(self.formula_digest, "closed formula digest")
        if self.formula_digest != canonical_digest(_formula_content(self)):
            raise ObjectSceneAnchorPythonPredicateError(
                "closed formula digest differs"
            )

    @classmethod
    def create(
        cls, witness_digests: Sequence[str]
    ) -> "ObjectSceneAnchorPythonClosedFormula":
        values = {"witness_digests": tuple(witness_digests)}
        provisional = object.__new__(cls)
        object.__setattr__(
            provisional, "witness_digests", values["witness_digests"]
        )
        return cls(
            **values,
            formula_digest=canonical_digest(_formula_content(provisional)),
        )

    def to_data(self) -> dict[str, object]:
        return {**_formula_content(self), "formula_digest": self.formula_digest}

    @classmethod
    def from_data(cls, value: object) -> "ObjectSceneAnchorPythonClosedFormula":
        raw = _exact_fields(
            value,
            {
                "schema",
                "closed_formula",
                "quantifier",
                "domain",
                "binding_scope",
                "conjunction",
                "witness_digests",
                "dispositions",
                *_formula_policy_data(),
                *_authority_data(),
                "formula_digest",
            },
            "closed formula",
        )
        if (
            raw["schema"] != OBJECT_SCENE_ANCHOR_PYTHON_FORMULA_SCHEMA
            or raw["closed_formula"] != OBJECT_SCENE_ANCHOR_CLOSED_FORMULA
            or raw["quantifier"] != "exists"
            or raw["domain"]
            != "eligible_bindings_across_all_inventoried_objects"
            or raw["binding_scope"] != "one_same_binding"
            or raw["conjunction"] != "selected_positive_witnesses"
            or raw["dispositions"] != [item.value for item in Disposition]
            or any(raw[key] != item for key, item in _formula_policy_data().items())
            or any(raw[key] != item for key, item in _authority_data().items())
            or not isinstance(raw["witness_digests"], list)
        ):
            raise ObjectSceneAnchorPythonPredicateError(
                "closed formula policy differs"
            )
        result = cls(tuple(raw["witness_digests"]), raw["formula_digest"])
        if result.to_data() != dict(raw):
            raise ObjectSceneAnchorPythonPredicateError(
                "closed formula is not canonical"
            )
        return result


def _predicate_content(value: "ObjectSceneAnchorPythonPredicate") -> dict[str, object]:
    return {
        "schema": OBJECT_SCENE_ANCHOR_PYTHON_PREDICATE_SCHEMA,
        "algorithm_id": OBJECT_SCENE_ANCHOR_PYTHON_PREDICATE_ALGORITHM_ID,
        "source_digest": value.source_digest,
        "algorithm_digest": value.algorithm_digest,
        "version_space_algorithm_digest": value.version_space_algorithm_digest,
        "version_space_digest": value.version_space_digest,
        "language_digest": value.language_digest,
        "selection_commitment": value.selection_commitment.to_data(),
        "candidate": value.candidate.to_data(),
        "binding_spec": value.binding_spec.to_data(),
        "affirmative_witness_entries": [
            item.to_data() for item in value.affirmative_witness_entries
        ],
        "formula": value.formula.to_data(),
        "candidate_frozen_before_evaluation": True,
        **_formula_policy_data(),
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class ObjectSceneAnchorPythonPredicate:
    """Self-contained positive predicate selected from an exact version space."""

    source_digest: str
    algorithm_digest: str
    version_space_algorithm_digest: str
    version_space_digest: str
    language_digest: str
    selection_commitment: ObjectSceneAnchorSelectionCommitment
    candidate: ObjectSceneAnchorPredicateCandidate
    binding_spec: ObjectSceneAnchorBindingSpec
    affirmative_witness_entries: tuple[
        ObjectSceneAnchorObserverVocabularyEntry, ...
    ]
    formula: ObjectSceneAnchorPythonClosedFormula
    predicate_digest: str

    def __post_init__(self) -> None:
        for label, item in (
            ("predicate source digest", self.source_digest),
            ("predicate algorithm digest", self.algorithm_digest),
            ("version-space algorithm digest", self.version_space_algorithm_digest),
            ("version-space digest", self.version_space_digest),
            ("language digest", self.language_digest),
            ("predicate digest", self.predicate_digest),
        ):
            _digest(item, label)
        if type(self.selection_commitment) is not ObjectSceneAnchorSelectionCommitment:
            raise TypeError("predicate selection commitment has the wrong type")
        if type(self.candidate) is not ObjectSceneAnchorPredicateCandidate:
            raise TypeError("predicate candidate has the wrong type")
        if type(self.binding_spec) is not ObjectSceneAnchorBindingSpec:
            raise TypeError("predicate binding spec has the wrong type")
        if type(self.formula) is not ObjectSceneAnchorPythonClosedFormula:
            raise TypeError("predicate formula has the wrong type")
        if (
            type(self.affirmative_witness_entries) is not tuple
            or not self.affirmative_witness_entries
            or any(
                type(item) is not ObjectSceneAnchorObserverVocabularyEntry
                for item in self.affirmative_witness_entries
            )
        ):
            raise ObjectSceneAnchorPythonPredicateError(
                "predicate witness entries differ"
            )
        selection = ObjectSceneAnchorSelectionCommitment.from_data(
            self.selection_commitment.to_data()
        )
        candidate = ObjectSceneAnchorPredicateCandidate.from_data(
            self.candidate.to_data()
        )
        spec = ObjectSceneAnchorBindingSpec.from_data(self.binding_spec.to_data())
        entries = tuple(
            ObjectSceneAnchorObserverVocabularyEntry.from_data(item.to_data())
            for item in self.affirmative_witness_entries
        )
        formula = ObjectSceneAnchorPythonClosedFormula.from_data(
            self.formula.to_data()
        )
        if (
            self.source_digest
            != object_scene_anchor_python_predicate_source_digest()
            or self.algorithm_digest
            != object_scene_anchor_python_predicate_algorithm_digest()
            or self.version_space_algorithm_digest
            != object_scene_anchor_version_space_algorithm_digest()
            or selection.version_space_algorithm_digest
            != self.version_space_algorithm_digest
            or selection.version_space_digest != self.version_space_digest
            or selection.language_digest != self.language_digest
            or selection.selected_candidate_digest != candidate.candidate_digest
            or candidate.candidate_digest
            not in selection.survivor_candidate_digests
            or candidate.algorithm_digest != self.version_space_algorithm_digest
            or candidate.language_digest != self.language_digest
            or candidate.orientation is not selection.orientation
            or candidate.binding_spec_digest != spec.spec_digest
            or tuple(item.witness_digest for item in entries)
            != candidate.witness_digests
            or formula.witness_digests != candidate.witness_digests
            or selection != self.selection_commitment
            or candidate != self.candidate
            or spec != self.binding_spec
            or entries != self.affirmative_witness_entries
            or formula != self.formula
        ):
            raise ObjectSceneAnchorPythonPredicateError(
                "predicate projection or authority binding differs"
            )
        if self.predicate_digest != canonical_digest(_predicate_content(self)):
            raise ObjectSceneAnchorPythonPredicateError("predicate digest differs")

    def to_data(self) -> dict[str, object]:
        return {**_predicate_content(self), "predicate_digest": self.predicate_digest}

    @classmethod
    def from_data(cls, value: object) -> "ObjectSceneAnchorPythonPredicate":
        raw = _exact_fields(
            value,
            {
                "schema",
                "algorithm_id",
                "source_digest",
                "algorithm_digest",
                "version_space_algorithm_digest",
                "version_space_digest",
                "language_digest",
                "selection_commitment",
                "candidate",
                "binding_spec",
                "affirmative_witness_entries",
                "formula",
                "candidate_frozen_before_evaluation",
                *_formula_policy_data(),
                *_authority_data(),
                "predicate_digest",
            },
            "Python predicate",
        )
        if (
            raw["schema"] != OBJECT_SCENE_ANCHOR_PYTHON_PREDICATE_SCHEMA
            or raw["algorithm_id"]
            != OBJECT_SCENE_ANCHOR_PYTHON_PREDICATE_ALGORITHM_ID
            or raw["candidate_frozen_before_evaluation"] is not True
            or any(raw[key] != item for key, item in _formula_policy_data().items())
            or any(raw[key] != item for key, item in _authority_data().items())
            or not isinstance(raw["selection_commitment"], Mapping)
            or not isinstance(raw["candidate"], Mapping)
            or not isinstance(raw["binding_spec"], Mapping)
            or not isinstance(raw["affirmative_witness_entries"], list)
            or not isinstance(raw["formula"], Mapping)
        ):
            raise ObjectSceneAnchorPythonPredicateError(
                "Python predicate policy differs"
            )
        result = cls(
            raw["source_digest"],
            raw["algorithm_digest"],
            raw["version_space_algorithm_digest"],
            raw["version_space_digest"],
            raw["language_digest"],
            ObjectSceneAnchorSelectionCommitment.from_data(
                raw["selection_commitment"]
            ),
            ObjectSceneAnchorPredicateCandidate.from_data(raw["candidate"]),
            ObjectSceneAnchorBindingSpec.from_data(raw["binding_spec"]),
            tuple(
                ObjectSceneAnchorObserverVocabularyEntry.from_data(item)
                for item in raw["affirmative_witness_entries"]
            ),
            ObjectSceneAnchorPythonClosedFormula.from_data(raw["formula"]),
            raw["predicate_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectSceneAnchorPythonPredicateError(
                "Python predicate is not canonical"
            )
        return result


def freeze_object_scene_anchor_python_predicate(
    version_space: ObjectSceneAnchorSupportVersionSpace,
    selection_commitment: ObjectSceneAnchorSelectionCommitment,
) -> ObjectSceneAnchorPythonPredicate:
    """Freeze one exact survivor without copying support citations into runtime."""

    if type(version_space) is not ObjectSceneAnchorSupportVersionSpace:
        raise TypeError(
            "version_space must be exact ObjectSceneAnchorSupportVersionSpace"
        )
    if type(selection_commitment) is not ObjectSceneAnchorSelectionCommitment:
        raise TypeError(
            "selection_commitment must be exact ObjectSceneAnchorSelectionCommitment"
        )
    version = ObjectSceneAnchorSupportVersionSpace.from_data(version_space.to_data())
    selection = ObjectSceneAnchorSelectionCommitment.from_data(
        selection_commitment.to_data()
    )
    if not version.survivor_candidate_digests:
        raise ObjectSceneAnchorPythonPredicateError(
            "cannot freeze a predicate from an empty survivor set"
        )
    if (
        selection.version_space_digest != version.version_space_digest
        or selection.version_space_algorithm_digest != version.algorithm_digest
        or selection.language_digest != version.language.language_digest
        or selection.orientation is not version.orientation
        or selection.survivor_candidate_digests
        != version.survivor_candidate_digests
        or selection.survivor_set_digest
        != _survivor_set_digest(
            version.version_space_digest, version.survivor_candidate_digests
        )
        or selection.selected_candidate_digest
        not in version.survivor_candidate_digests
    ):
        raise ObjectSceneAnchorPythonPredicateError(
            "selection commitment differs from the exact current version space"
        )
    candidates = tuple(
        item
        for item in version.candidates
        if item.candidate_digest == selection.selected_candidate_digest
    )
    if len(candidates) != 1:
        raise ObjectSceneAnchorPythonPredicateError(
            "selected survivor is absent from the complete candidate inventory"
        )
    candidate = candidates[0]
    atom_by_digest = {item.atom_digest: item for item in version.language.atoms}
    try:
        atoms = tuple(atom_by_digest[item] for item in candidate.atom_digests)
    except KeyError as exc:
        raise ObjectSceneAnchorPythonPredicateError(
            "selected candidate atom is outside the frozen language"
        ) from exc
    specs = {
        item.binding_spec.spec_digest: item.binding_spec for item in atoms
    }
    if set(specs) != {candidate.binding_spec_digest}:
        raise ObjectSceneAnchorPythonPredicateError(
            "selected candidate does not have one exact binding spec"
        )
    witness_by_digest = {
        item.witness_digest: item for item in version.language.vocabulary.entries
    }
    try:
        entries = tuple(witness_by_digest[item] for item in candidate.witness_digests)
    except KeyError as exc:
        raise ObjectSceneAnchorPythonPredicateError(
            "selected candidate witness is outside the frozen vocabulary"
        ) from exc
    formula = ObjectSceneAnchorPythonClosedFormula.create(
        candidate.witness_digests
    )
    values = {
        "source_digest": object_scene_anchor_python_predicate_source_digest(),
        "algorithm_digest": object_scene_anchor_python_predicate_algorithm_digest(),
        "version_space_algorithm_digest": version.algorithm_digest,
        "version_space_digest": version.version_space_digest,
        "language_digest": version.language.language_digest,
        "selection_commitment": selection,
        "candidate": candidate,
        "binding_spec": specs[candidate.binding_spec_digest],
        "affirmative_witness_entries": entries,
        "formula": formula,
    }
    provisional = object.__new__(ObjectSceneAnchorPythonPredicate)
    for name, item in values.items():
        object.__setattr__(provisional, name, item)
    return ObjectSceneAnchorPythonPredicate(
        **values,
        predicate_digest=canonical_digest(_predicate_content(provisional)),
    )


def cold_verify_object_scene_anchor_python_predicate(
    predicate: ObjectSceneAnchorPythonPredicate,
    *,
    version_space: ObjectSceneAnchorSupportVersionSpace,
    selection_commitment: ObjectSceneAnchorSelectionCommitment,
) -> ObjectSceneAnchorPythonPredicate:
    """Rebuild the artifact from exact commitments without observations or calls."""

    if type(predicate) is not ObjectSceneAnchorPythonPredicate:
        raise TypeError("predicate must be exact ObjectSceneAnchorPythonPredicate")
    restored = ObjectSceneAnchorPythonPredicate.from_data(predicate.to_data())
    expected = freeze_object_scene_anchor_python_predicate(
        version_space, selection_commitment
    )
    if restored != expected:
        raise ObjectSceneAnchorPythonPredicateError(
            "frozen predicate differs from cold replay"
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


def _evaluate_formula(
    predicate: ObjectSceneAnchorPythonPredicate,
    panel: ObjectSceneAnchorPanelWitnessEvaluation,
) -> Disposition:
    if panel.language_digest != predicate.language_digest:
        return Disposition.ERROR
    matches = tuple(
        item
        for item in panel.spec_matrices
        if item.binding_spec.spec_digest == predicate.binding_spec.spec_digest
    )
    if len(matches) != 1 or matches[0].binding_spec != predicate.binding_spec:
        return Disposition.ERROR
    states: list[Disposition] = []
    for object_matrix in matches[0].objects:
        hard = object_matrix.catalog.hard_disposition
        if hard is Disposition.ERROR:
            states.append(Disposition.ERROR)
        elif hard is Disposition.INDETERMINATE:
            states.append(Disposition.INDETERMINATE)
        elif hard is Disposition.PRESENT:
            for row in object_matrix.rows:
                by_digest = {
                    item.witness_digest: item.disposition for item in row.cells
                }
                if any(
                    item not in by_digest
                    for item in predicate.formula.witness_digests
                ):
                    states.append(Disposition.ERROR)
                else:
                    states.append(
                        _scene_and(
                            tuple(
                                by_digest[item]
                                for item in predicate.formula.witness_digests
                            )
                        )
                    )
        # A certified-complete empty catalog contributes no existential binding.
    return _scene_or(states)


def _evaluation_content(
    value: "ObjectSceneAnchorPythonPredicateEvaluation",
) -> dict[str, object]:
    return {
        "schema": OBJECT_SCENE_ANCHOR_PYTHON_EVALUATION_SCHEMA,
        "algorithm_id": OBJECT_SCENE_ANCHOR_PYTHON_PREDICATE_ALGORITHM_ID,
        "algorithm_digest": value.algorithm_digest,
        "predicate_digest": value.predicate_digest,
        "panel_id": value.panel_id,
        "panel_evaluation_digest": value.panel_evaluation_digest,
        "disposition": value.disposition.value,
        **_formula_policy_data(),
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class ObjectSceneAnchorPythonPredicateEvaluation:
    """Content-addressed P/A/I/E result for one neutral panel matrix."""

    algorithm_digest: str
    predicate_digest: str
    panel_id: str
    panel_evaluation_digest: str
    disposition: Disposition
    evaluation_digest: str

    def __post_init__(self) -> None:
        for label, item in (
            ("evaluation algorithm digest", self.algorithm_digest),
            ("predicate digest", self.predicate_digest),
            ("panel evaluation digest", self.panel_evaluation_digest),
            ("Python evaluation digest", self.evaluation_digest),
        ):
            _digest(item, label)
        if (
            not isinstance(self.panel_id, str)
            or not self.panel_id
            or len(self.panel_id) > 256
        ):
            raise ObjectSceneAnchorPythonPredicateError(
                "evaluation panel ID differs"
            )
        if not isinstance(self.disposition, Disposition):
            raise TypeError("evaluation disposition has the wrong type")
        if self.algorithm_digest != (
            object_scene_anchor_python_predicate_algorithm_digest()
        ):
            raise ObjectSceneAnchorPythonPredicateError(
                "evaluation algorithm binding differs"
            )
        if self.evaluation_digest != canonical_digest(_evaluation_content(self)):
            raise ObjectSceneAnchorPythonPredicateError(
                "Python evaluation digest differs"
            )

    def to_data(self) -> dict[str, object]:
        return {
            **_evaluation_content(self),
            "evaluation_digest": self.evaluation_digest,
        }

    @classmethod
    def from_data(
        cls, value: object
    ) -> "ObjectSceneAnchorPythonPredicateEvaluation":
        raw = _exact_fields(
            value,
            {
                "schema",
                "algorithm_id",
                "algorithm_digest",
                "predicate_digest",
                "panel_id",
                "panel_evaluation_digest",
                "disposition",
                *_formula_policy_data(),
                *_authority_data(),
                "evaluation_digest",
            },
            "Python predicate evaluation",
        )
        if (
            raw["schema"] != OBJECT_SCENE_ANCHOR_PYTHON_EVALUATION_SCHEMA
            or raw["algorithm_id"]
            != OBJECT_SCENE_ANCHOR_PYTHON_PREDICATE_ALGORITHM_ID
            or any(raw[key] != item for key, item in _formula_policy_data().items())
            or any(raw[key] != item for key, item in _authority_data().items())
        ):
            raise ObjectSceneAnchorPythonPredicateError(
                "Python predicate evaluation policy differs"
            )
        result = cls(
            raw["algorithm_digest"],
            raw["predicate_digest"],
            raw["panel_id"],
            raw["panel_evaluation_digest"],
            _disposition(raw["disposition"]),
            raw["evaluation_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectSceneAnchorPythonPredicateError(
                "Python predicate evaluation is not canonical"
            )
        return result


def evaluate_object_scene_anchor_python_predicate(
    predicate: ObjectSceneAnchorPythonPredicate,
    panel: ObjectSceneAnchorPanelWitnessEvaluation,
) -> ObjectSceneAnchorPythonPredicateEvaluation:
    """Evaluate the frozen existential over every neutral object/binding row."""

    if type(predicate) is not ObjectSceneAnchorPythonPredicate:
        raise TypeError("predicate must be exact ObjectSceneAnchorPythonPredicate")
    if type(panel) is not ObjectSceneAnchorPanelWitnessEvaluation:
        raise TypeError(
            "panel must be exact ObjectSceneAnchorPanelWitnessEvaluation"
        )
    frozen = ObjectSceneAnchorPythonPredicate.from_data(predicate.to_data())
    neutral = ObjectSceneAnchorPanelWitnessEvaluation.from_data(panel.to_data())
    values = {
        "algorithm_digest": frozen.algorithm_digest,
        "predicate_digest": frozen.predicate_digest,
        "panel_id": neutral.panel_id,
        "panel_evaluation_digest": neutral.evaluation_digest,
        "disposition": _evaluate_formula(frozen, neutral),
    }
    provisional = object.__new__(ObjectSceneAnchorPythonPredicateEvaluation)
    for name, item in values.items():
        object.__setattr__(provisional, name, item)
    return ObjectSceneAnchorPythonPredicateEvaluation(
        **values,
        evaluation_digest=canonical_digest(_evaluation_content(provisional)),
    )


def cold_verify_object_scene_anchor_python_predicate_evaluation(
    evaluation: ObjectSceneAnchorPythonPredicateEvaluation,
    *,
    predicate: ObjectSceneAnchorPythonPredicate,
    panel: ObjectSceneAnchorPanelWitnessEvaluation,
) -> ObjectSceneAnchorPythonPredicateEvaluation:
    """Recompute one result solely from frozen Python artifacts."""

    if type(evaluation) is not ObjectSceneAnchorPythonPredicateEvaluation:
        raise TypeError(
            "evaluation must be exact ObjectSceneAnchorPythonPredicateEvaluation"
        )
    restored = ObjectSceneAnchorPythonPredicateEvaluation.from_data(
        evaluation.to_data()
    )
    expected = evaluate_object_scene_anchor_python_predicate(predicate, panel)
    if restored != expected:
        raise ObjectSceneAnchorPythonPredicateError(
            "Python predicate evaluation differs from cold replay"
        )
    return restored


__all__ = (
    "OBJECT_SCENE_ANCHOR_CLOSED_FORMULA",
    "OBJECT_SCENE_ANCHOR_PYTHON_EVALUATION_SCHEMA",
    "OBJECT_SCENE_ANCHOR_PYTHON_FORMULA_SCHEMA",
    "OBJECT_SCENE_ANCHOR_PYTHON_PREDICATE_ALGORITHM_ID",
    "OBJECT_SCENE_ANCHOR_PYTHON_PREDICATE_SCHEMA",
    "OBJECT_SCENE_ANCHOR_SELECTION_COMMITMENT_SCHEMA",
    "ObjectSceneAnchorPythonClosedFormula",
    "ObjectSceneAnchorPythonPredicate",
    "ObjectSceneAnchorPythonPredicateError",
    "ObjectSceneAnchorPythonPredicateEvaluation",
    "ObjectSceneAnchorSelectionCommitment",
    "cold_verify_object_scene_anchor_python_predicate",
    "cold_verify_object_scene_anchor_python_predicate_evaluation",
    "evaluate_object_scene_anchor_python_predicate",
    "freeze_object_scene_anchor_python_predicate",
    "object_scene_anchor_python_predicate_algorithm_digest",
    "object_scene_anchor_python_predicate_source_digest",
    "object_scene_anchor_selection_commitment_from_rank_response",
)
