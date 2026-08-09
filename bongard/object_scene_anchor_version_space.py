"""Deterministic positive version spaces over exact object-scene anchors.

Vision produces one side-label-free rectangular witness matrix: every frozen
witness is judged on every eligible binding of every inventoried object.  This
module alone introduces support roles.  It enumerates all positive atoms and
all same-spec positive conjunctions through size four, requires an exact
cited binding on target panels, and uses an exhaustive all-object existential
on contrast panels.

The serialized predicate language contains only Python schemas, decision-only
anchor identities, affirmative witness identities, and typed dispositions.
Transport and extraction audit provenance remain outside predicate identity.
"""

from __future__ import annotations

from bongard.runtime_source_snapshot import capture_loaded_source, verify_loaded_source


_LOADED_SOURCE_SHA256 = capture_loaded_source(__name__, __file__)

from dataclasses import dataclass
from enum import Enum
from itertools import combinations
import re
from typing import Mapping, Sequence

from bongard.canonical import canonical_digest
from bongard.evidence import Disposition
from bongard.object_scene_anchor_bindings import (
    ObjectSceneAnchorBindingCatalog,
    ObjectSceneAnchorBindingSpec,
    ObjectSceneAnchorWitnessCell,
    ObjectSceneAnchorWitnessSpec,
    ObjectSceneResolvedAnchorBinding,
)
from bongard.object_scene_anchor_observer import (
    OBJECT_SCENE_ANCHOR_OBSERVER_MAX_WITNESSES,
    ObjectSceneAnchorObserverArtifact,
    ObjectSceneAnchorObserverVocabulary,
    freeze_object_scene_anchor_observer_vocabulary,
)
from bongard.object_scene_anchor_panel_manifest import (
    ObjectSceneAnchorPanelDecisionManifest,
)
from bongard.python_predicate_authority import PYTHON_PREDICATE_AUTHORITY_ID


ANCHOR_ATOM_CITATION_SCHEMA = "gkm.object-scene-anchor-atom-citation.v1"
ANCHOR_ATOM_SCHEMA = "gkm.object-scene-anchor-positive-atom.v1"
ANCHOR_LANGUAGE_SCHEMA = "gkm.object-scene-anchor-positive-language.v1"
ANCHOR_BINDING_WITNESS_ROW_SCHEMA = (
    "gkm.object-scene-anchor-binding-witness-row.v1"
)
ANCHOR_OBJECT_WITNESS_MATRIX_SCHEMA = (
    "gkm.object-scene-anchor-object-witness-matrix.v1"
)
ANCHOR_SPEC_WITNESS_MATRIX_SCHEMA = (
    "gkm.object-scene-anchor-spec-witness-matrix.v1"
)
ANCHOR_PANEL_WITNESS_EVALUATION_SCHEMA = (
    "gkm.object-scene-anchor-panel-witness-evaluation.v1"
)
ANCHOR_CANDIDATE_SCHEMA = "gkm.object-scene-anchor-positive-candidate.v1"
ANCHOR_SUPPORT_DIAGNOSTIC_SCHEMA = (
    "gkm.object-scene-anchor-support-diagnostic.v1"
)
ANCHOR_SUPPORT_GAP_SCHEMA = "gkm.object-scene-anchor-support-gap.v1"
ANCHOR_SUPPORT_VERSION_SPACE_SCHEMA = (
    "gkm.object-scene-anchor-support-version-space.v1"
)
ANCHOR_VERSION_SPACE_ALGORITHM_ID = (
    "bongard.object-scene-anchor-version-space/positive-same-binding-v1"
)
ANCHOR_MAX_CONJUNCTS = 4
ANCHOR_SUPPORT_PANELS_PER_SIDE = 6

_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_PANEL_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/-]{0,511}\Z")
_OBJECT_ID = re.compile(r"object_[0-9]{4}\Z")


class ObjectSceneAnchorVersionSpaceError(ValueError):
    """A language, matrix, candidate, or version space is malformed."""


class ObjectSceneAnchorOrientation(str, Enum):
    SIDE0_POSITIVE = "side0_positive"
    SIDE1_POSITIVE = "side1_positive"


class ObjectSceneAnchorSupportSide(str, Enum):
    TARGET = "target"
    CONTRAST = "contrast"


class ObjectSceneAnchorSupportGapKind(str, Enum):
    LANGUAGE_GAP = "language_gap"
    WITNESS_GAP = "witness_gap"


def _authority_data() -> dict[str, object]:
    return {
        "predicate_authority_id": PYTHON_PREDICATE_AUTHORITY_ID,
        "python_is_canonical_authority": True,
    }


def _policy_data() -> dict[str, object]:
    return {
        "positive_atoms_only": True,
        "negation_allowed": False,
        "polarity_flip_allowed": False,
        "maximum_conjunct_count": ANCHOR_MAX_CONJUNCTS,
        "same_binding_spec_required": True,
        "same_binding_required": True,
        "target_requires_exact_cited_binding": True,
        "contrast_quantifier": "exists-over-every-binding-of-every-inventoried-object",
        "error_dominates": True,
        "failed_fit_counts_as_absence": False,
        "target_accept": Disposition.PRESENT.value,
        "contrast_accept": Disposition.CERTIFIED_ABSENT.value,
    }


def _exact_fields(
    value: object, expected: frozenset[str], label: str
) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ObjectSceneAnchorVersionSpaceError(f"{label} fields differ")
    return value


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise ObjectSceneAnchorVersionSpaceError(
            f"{label} must be a lowercase SHA-256"
        )
    return value


def _panel_id(value: object) -> str:
    if not isinstance(value, str) or _PANEL_ID.fullmatch(value) is None:
        raise ObjectSceneAnchorVersionSpaceError(
            "panel_id must be a bounded neutral identifier"
        )
    return value


def _orientation(value: object) -> ObjectSceneAnchorOrientation:
    try:
        return ObjectSceneAnchorOrientation(value)
    except (TypeError, ValueError) as exc:
        raise ObjectSceneAnchorVersionSpaceError("orientation differs") from exc


def _disposition(value: object) -> Disposition:
    try:
        return value if isinstance(value, Disposition) else Disposition(value)
    except (TypeError, ValueError) as exc:
        raise ObjectSceneAnchorVersionSpaceError("disposition differs") from exc


def _scene_and(values: Sequence[Disposition]) -> Disposition:
    states = tuple(values)
    if not states:
        return Disposition.PRESENT
    if Disposition.ERROR in states:
        return Disposition.ERROR
    if Disposition.CERTIFIED_ABSENT in states:
        return Disposition.CERTIFIED_ABSENT
    if all(item is Disposition.PRESENT for item in states):
        return Disposition.PRESENT
    return Disposition.INDETERMINATE


def _scene_or(values: Sequence[Disposition]) -> Disposition:
    states = tuple(values)
    if not states:
        return Disposition.CERTIFIED_ABSENT
    if Disposition.ERROR in states:
        return Disposition.ERROR
    if Disposition.PRESENT in states:
        return Disposition.PRESENT
    if all(item is Disposition.CERTIFIED_ABSENT for item in states):
        return Disposition.CERTIFIED_ABSENT
    return Disposition.INDETERMINATE


def object_scene_anchor_version_space_source_digest() -> str:
    return verify_loaded_source(
        __name__, expected_source_sha256=_LOADED_SOURCE_SHA256
    )


def object_scene_anchor_version_space_algorithm_digest() -> str:
    return canonical_digest(
        {
            "schema": "gkm.object-scene-anchor-version-space-algorithm.v1",
            "algorithm_id": ANCHOR_VERSION_SPACE_ALGORITHM_ID,
            "implementation_source_sha256": (
                object_scene_anchor_version_space_source_digest()
            ),
            "support_panels_per_side": ANCHOR_SUPPORT_PANELS_PER_SIDE,
            **_policy_data(),
            **_authority_data(),
        }
    )


def _citation_content(value: "ObjectSceneAnchorAtomCitation") -> dict[str, object]:
    return {
        "schema": ANCHOR_ATOM_CITATION_SCHEMA,
        "panel_id": value.panel_id,
        "panel_manifest_digest": value.panel_manifest_digest,
        "binding_catalogs_digest": value.binding_catalogs_digest,
        "binding": value.binding.to_data(),
        **_authority_data(),
    }


@dataclass(frozen=True, order=True, slots=True)
class ObjectSceneAnchorAtomCitation:
    """One target-panel commitment to one exact resolved binding."""

    panel_id: str
    panel_manifest_digest: str
    binding_catalogs_digest: str
    binding: ObjectSceneResolvedAnchorBinding
    citation_digest: str

    def __post_init__(self) -> None:
        _panel_id(self.panel_id)
        _digest(self.panel_manifest_digest, "citation panel manifest digest")
        _digest(self.binding_catalogs_digest, "citation binding catalogs digest")
        if type(self.binding) is not ObjectSceneResolvedAnchorBinding:
            raise TypeError("citation binding must be exact resolved binding")
        if (
            ObjectSceneResolvedAnchorBinding.from_data(self.binding.to_data())
            != self.binding
        ):
            raise ObjectSceneAnchorVersionSpaceError(
                "citation binding is not canonical"
            )
        _digest(self.citation_digest, "citation digest")
        if self.citation_digest != canonical_digest(_citation_content(self)):
            raise ObjectSceneAnchorVersionSpaceError("citation digest differs")

    @classmethod
    def create(
        cls,
        panel_id: str,
        panel_manifest_digest: str,
        binding_catalogs_digest: str,
        binding: ObjectSceneResolvedAnchorBinding,
    ) -> "ObjectSceneAnchorAtomCitation":
        provisional = object.__new__(cls)
        object.__setattr__(provisional, "panel_id", panel_id)
        object.__setattr__(provisional, "panel_manifest_digest", panel_manifest_digest)
        object.__setattr__(provisional, "binding_catalogs_digest", binding_catalogs_digest)
        object.__setattr__(provisional, "binding", binding)
        return cls(
            panel_id,
            panel_manifest_digest,
            binding_catalogs_digest,
            binding,
            canonical_digest(_citation_content(provisional)),
        )

    def to_data(self) -> dict[str, object]:
        return {**_citation_content(self), "citation_digest": self.citation_digest}

    @classmethod
    def from_data(cls, value: object) -> "ObjectSceneAnchorAtomCitation":
        raw = _exact_fields(
            value,
            frozenset(
                (
                    "schema",
                    "panel_id",
                    "panel_manifest_digest",
                    "binding_catalogs_digest",
                    "binding",
                    *tuple(_authority_data()),
                    "citation_digest",
                )
            ),
            "atom citation",
        )
        if (
            raw["schema"] != ANCHOR_ATOM_CITATION_SCHEMA
            or any(raw[key] != item for key, item in _authority_data().items())
            or not isinstance(raw["binding"], Mapping)
        ):
            raise ObjectSceneAnchorVersionSpaceError("atom citation policy differs")
        result = cls(
            raw["panel_id"],
            raw["panel_manifest_digest"],
            raw["binding_catalogs_digest"],
            ObjectSceneResolvedAnchorBinding.from_data(raw["binding"]),
            raw["citation_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectSceneAnchorVersionSpaceError(
                "atom citation is not canonical"
            )
        return result


def _atom_content(value: "ObjectSceneAnchorPredicateAtom") -> dict[str, object]:
    return {
        "schema": ANCHOR_ATOM_SCHEMA,
        "source_card_digest": value.source_card_digest,
        "orientation": value.orientation.value,
        "binding_spec": value.binding_spec.to_data(),
        "witness_digests": list(value.witness_digests),
        "positive_support_citations": [
            item.to_data() for item in value.positive_support_citations
        ],
        "operator": "affirmative_witness_conjunction",
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class ObjectSceneAnchorPredicateAtom:
    """Decision-only projection of one validated affirmative prose card."""

    source_card_digest: str
    orientation: ObjectSceneAnchorOrientation
    binding_spec: ObjectSceneAnchorBindingSpec
    witness_digests: tuple[str, ...]
    positive_support_citations: tuple[ObjectSceneAnchorAtomCitation, ...]
    atom_digest: str

    def __post_init__(self) -> None:
        _digest(self.source_card_digest, "source card digest")
        if not isinstance(self.orientation, ObjectSceneAnchorOrientation):
            raise TypeError("atom orientation must be exact orientation enum")
        if type(self.binding_spec) is not ObjectSceneAnchorBindingSpec:
            raise TypeError("atom binding spec must be exact binding spec")
        if (
            ObjectSceneAnchorBindingSpec.from_data(self.binding_spec.to_data())
            != self.binding_spec
        ):
            raise ObjectSceneAnchorVersionSpaceError("atom binding spec differs")
        if (
            type(self.witness_digests) is not tuple
            or len(self.witness_digests) != 1
            or self.witness_digests != tuple(sorted(set(self.witness_digests)))
        ):
            raise ObjectSceneAnchorVersionSpaceError(
                "one language atom must contain exactly one witness digest"
            )
        for item in self.witness_digests:
            _digest(item, "atom witness digest")
        citations = self.positive_support_citations
        if (
            type(citations) is not tuple
            or len(citations) != ANCHOR_SUPPORT_PANELS_PER_SIDE
            or any(type(item) is not ObjectSceneAnchorAtomCitation for item in citations)
            or tuple(item.panel_id for item in citations)
            != tuple(sorted(item.panel_id for item in citations))
            or len({item.panel_id for item in citations}) != len(citations)
            or any(
                item.binding.spec_digest != self.binding_spec.spec_digest
                for item in citations
            )
        ):
            raise ObjectSceneAnchorVersionSpaceError(
                "atom citations must cover six sorted panels under one spec"
            )
        _digest(self.atom_digest, "atom digest")
        if self.atom_digest != canonical_digest(_atom_content(self)):
            raise ObjectSceneAnchorVersionSpaceError("atom digest differs")

    @property
    def target_panel_ids(self) -> tuple[str, ...]:
        return tuple(item.panel_id for item in self.positive_support_citations)

    @classmethod
    def create(
        cls,
        *,
        source_card_digest: str,
        orientation: ObjectSceneAnchorOrientation,
        binding_spec: ObjectSceneAnchorBindingSpec,
        witness_digests: Sequence[str],
        positive_support_citations: Sequence[ObjectSceneAnchorAtomCitation],
    ) -> "ObjectSceneAnchorPredicateAtom":
        values = {
            "source_card_digest": source_card_digest,
            "orientation": orientation,
            "binding_spec": binding_spec,
            "witness_digests": tuple(sorted(witness_digests)),
            "positive_support_citations": tuple(positive_support_citations),
        }
        provisional = object.__new__(cls)
        for name, item in values.items():
            object.__setattr__(provisional, name, item)
        return cls(
            **values,
            atom_digest=canonical_digest(_atom_content(provisional)),
        )

    def to_data(self) -> dict[str, object]:
        return {**_atom_content(self), "atom_digest": self.atom_digest}

    @classmethod
    def from_data(cls, value: object) -> "ObjectSceneAnchorPredicateAtom":
        raw = _exact_fields(
            value,
            frozenset(
                (
                    "schema",
                    "source_card_digest",
                    "orientation",
                    "binding_spec",
                    "witness_digests",
                    "positive_support_citations",
                    "operator",
                    *tuple(_authority_data()),
                    "atom_digest",
                )
            ),
            "predicate atom",
        )
        if (
            raw["schema"] != ANCHOR_ATOM_SCHEMA
            or raw["operator"] != "affirmative_witness_conjunction"
            or any(raw[key] != item for key, item in _authority_data().items())
            or not isinstance(raw["binding_spec"], Mapping)
            or not isinstance(raw["witness_digests"], list)
            or not isinstance(raw["positive_support_citations"], list)
        ):
            raise ObjectSceneAnchorVersionSpaceError("predicate atom policy differs")
        result = cls(
            raw["source_card_digest"],
            _orientation(raw["orientation"]),
            ObjectSceneAnchorBindingSpec.from_data(raw["binding_spec"]),
            tuple(raw["witness_digests"]),
            tuple(
                ObjectSceneAnchorAtomCitation.from_data(item)
                for item in raw["positive_support_citations"]
            ),
            raw["atom_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectSceneAnchorVersionSpaceError("predicate atom is not canonical")
        return result


def _language_content(value: "ObjectSceneAnchorPredicateLanguage") -> dict[str, object]:
    return {
        "schema": ANCHOR_LANGUAGE_SCHEMA,
        "source_proposal_digest": value.source_proposal_digest,
        "vocabulary": value.vocabulary.to_data(),
        "atoms": [item.to_data() for item in value.atoms],
        "atom_order": "orientation-spec-digest-ascending",
        "vocabulary_is_complete_union": True,
        **_policy_data(),
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class ObjectSceneAnchorPredicateLanguage:
    """The complete accepted card projection and its frozen union vocabulary."""

    source_proposal_digest: str
    vocabulary: ObjectSceneAnchorObserverVocabulary
    atoms: tuple[ObjectSceneAnchorPredicateAtom, ...]
    language_digest: str

    def __post_init__(self) -> None:
        _digest(self.source_proposal_digest, "source proposal digest")
        if type(self.vocabulary) is not ObjectSceneAnchorObserverVocabulary:
            raise TypeError("language vocabulary must be exact observer vocabulary")
        if (
            ObjectSceneAnchorObserverVocabulary.from_data(self.vocabulary.to_data())
            != self.vocabulary
        ):
            raise ObjectSceneAnchorVersionSpaceError("language vocabulary differs")
        if (
            type(self.atoms) is not tuple
            or not self.atoms
            or any(type(item) is not ObjectSceneAnchorPredicateAtom for item in self.atoms)
            or self.atoms
            != tuple(
                sorted(
                    self.atoms,
                    key=lambda item: (
                        item.orientation.value,
                        item.binding_spec.spec_digest,
                        item.atom_digest,
                    ),
                )
            )
            or len({item.atom_digest for item in self.atoms}) != len(self.atoms)
        ):
            raise ObjectSceneAnchorVersionSpaceError(
                "language atoms are not a complete canonical inventory"
            )
        union = {digest for atom in self.atoms for digest in atom.witness_digests}
        vocabulary_digests = {
            item.witness_digest for item in self.vocabulary.entries
        }
        if union != vocabulary_digests:
            raise ObjectSceneAnchorVersionSpaceError(
                "language vocabulary differs from exact atom witness union"
            )
        for orientation in ObjectSceneAnchorOrientation:
            oriented = tuple(item for item in self.atoms if item.orientation is orientation)
            if oriented and len({item.target_panel_ids for item in oriented}) != 1:
                raise ObjectSceneAnchorVersionSpaceError(
                    "one orientation must share one exact six-panel target inventory"
                )
        _digest(self.language_digest, "language digest")
        if self.language_digest != canonical_digest(_language_content(self)):
            raise ObjectSceneAnchorVersionSpaceError("language digest differs")

    @classmethod
    def create(
        cls,
        *,
        source_proposal_digest: str,
        vocabulary: ObjectSceneAnchorObserverVocabulary,
        atoms: Sequence[ObjectSceneAnchorPredicateAtom],
    ) -> "ObjectSceneAnchorPredicateLanguage":
        ordered = tuple(
            sorted(
                atoms,
                key=lambda item: (
                    item.orientation.value,
                    item.binding_spec.spec_digest,
                    item.atom_digest,
                ),
            )
        )
        values = {
            "source_proposal_digest": source_proposal_digest,
            "vocabulary": vocabulary,
            "atoms": ordered,
        }
        provisional = object.__new__(cls)
        for name, item in values.items():
            object.__setattr__(provisional, name, item)
        return cls(
            **values,
            language_digest=canonical_digest(_language_content(provisional)),
        )

    def to_data(self) -> dict[str, object]:
        return {**_language_content(self), "language_digest": self.language_digest}

    @classmethod
    def from_data(cls, value: object) -> "ObjectSceneAnchorPredicateLanguage":
        raw = _exact_fields(
            value,
            frozenset(
                (
                    "schema",
                    "source_proposal_digest",
                    "vocabulary",
                    "atoms",
                    "atom_order",
                    "vocabulary_is_complete_union",
                    *tuple(_policy_data()),
                    *tuple(_authority_data()),
                    "language_digest",
                )
            ),
            "predicate language",
        )
        if (
            raw["schema"] != ANCHOR_LANGUAGE_SCHEMA
            or raw["atom_order"] != "orientation-spec-digest-ascending"
            or raw["vocabulary_is_complete_union"] is not True
            or any(raw[key] != item for key, item in _policy_data().items())
            or any(raw[key] != item for key, item in _authority_data().items())
            or not isinstance(raw["vocabulary"], Mapping)
            or not isinstance(raw["atoms"], list)
        ):
            raise ObjectSceneAnchorVersionSpaceError("predicate language policy differs")
        result = cls(
            raw["source_proposal_digest"],
            ObjectSceneAnchorObserverVocabulary.from_data(raw["vocabulary"]),
            tuple(ObjectSceneAnchorPredicateAtom.from_data(item) for item in raw["atoms"]),
            raw["language_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectSceneAnchorVersionSpaceError(
                "predicate language is not canonical"
            )
        return result


def _row_content(value: "ObjectSceneAnchorBindingWitnessRow") -> dict[str, object]:
    return {
        "schema": ANCHOR_BINDING_WITNESS_ROW_SCHEMA,
        "binding_digest": value.binding_digest,
        "cells": [item.to_data() for item in value.cells],
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class ObjectSceneAnchorBindingWitnessRow:
    binding_digest: str
    cells: tuple[ObjectSceneAnchorWitnessCell, ...]
    row_digest: str

    def __post_init__(self) -> None:
        _digest(self.binding_digest, "row binding digest")
        if (
            type(self.cells) is not tuple
            or not self.cells
            or any(type(item) is not ObjectSceneAnchorWitnessCell for item in self.cells)
            or any(item.binding_digest != self.binding_digest for item in self.cells)
        ):
            raise ObjectSceneAnchorVersionSpaceError("binding witness row differs")
        _digest(self.row_digest, "binding witness row digest")
        if self.row_digest != canonical_digest(_row_content(self)):
            raise ObjectSceneAnchorVersionSpaceError("binding witness row digest differs")

    @classmethod
    def create(
        cls,
        binding_digest: str,
        cells: Sequence[ObjectSceneAnchorWitnessCell],
    ) -> "ObjectSceneAnchorBindingWitnessRow":
        values = {"binding_digest": binding_digest, "cells": tuple(cells)}
        provisional = object.__new__(cls)
        for name, item in values.items():
            object.__setattr__(provisional, name, item)
        return cls(
            **values, row_digest=canonical_digest(_row_content(provisional))
        )

    def to_data(self) -> dict[str, object]:
        return {**_row_content(self), "row_digest": self.row_digest}

    @classmethod
    def from_data(cls, value: object) -> "ObjectSceneAnchorBindingWitnessRow":
        raw = _exact_fields(
            value,
            frozenset(
                (
                    "schema",
                    "binding_digest",
                    "cells",
                    *tuple(_authority_data()),
                    "row_digest",
                )
            ),
            "binding witness row",
        )
        if (
            raw["schema"] != ANCHOR_BINDING_WITNESS_ROW_SCHEMA
            or any(raw[key] != item for key, item in _authority_data().items())
            or not isinstance(raw["cells"], list)
        ):
            raise ObjectSceneAnchorVersionSpaceError("binding witness row policy differs")
        result = cls(
            raw["binding_digest"],
            tuple(ObjectSceneAnchorWitnessCell.from_data(item) for item in raw["cells"]),
            raw["row_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectSceneAnchorVersionSpaceError(
                "binding witness row is not canonical"
            )
        return result


def _matrix_content(value: "ObjectSceneAnchorObjectWitnessMatrix") -> dict[str, object]:
    return {
        "schema": ANCHOR_OBJECT_WITNESS_MATRIX_SCHEMA,
        "catalog": value.catalog.to_data(),
        "vocabulary_digest": value.vocabulary_digest,
        "witness_specs": [item.to_data() for item in value.witness_specs],
        "rows": [item.to_data() for item in value.rows],
        "binding_major_witness_minor": True,
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class ObjectSceneAnchorObjectWitnessMatrix:
    """One object's complete binding-by-union-witness rectangle."""

    catalog: ObjectSceneAnchorBindingCatalog
    vocabulary_digest: str
    witness_specs: tuple[ObjectSceneAnchorWitnessSpec, ...]
    rows: tuple[ObjectSceneAnchorBindingWitnessRow, ...]
    matrix_digest: str

    def __post_init__(self) -> None:
        if type(self.catalog) is not ObjectSceneAnchorBindingCatalog:
            raise TypeError("matrix catalog must be exact binding catalog")
        if ObjectSceneAnchorBindingCatalog.from_data(self.catalog.to_data()) != self.catalog:
            raise ObjectSceneAnchorVersionSpaceError("matrix catalog differs")
        _digest(self.vocabulary_digest, "matrix vocabulary digest")
        specs = self.witness_specs
        if (
            type(specs) is not tuple
            or not 1 <= len(specs) <= OBJECT_SCENE_ANCHOR_OBSERVER_MAX_WITNESSES
            or any(type(item) is not ObjectSceneAnchorWitnessSpec for item in specs)
            or tuple(item.witness_id for item in specs)
            != tuple(f"witness_{index:02d}" for index in range(len(specs)))
            or tuple(item.witness_digest for item in specs)
            != tuple(sorted(item.witness_digest for item in specs))
            or len({item.witness_digest for item in specs}) != len(specs)
        ):
            raise ObjectSceneAnchorVersionSpaceError("matrix witness vocabulary differs")
        if type(self.rows) is not tuple or any(
            type(item) is not ObjectSceneAnchorBindingWitnessRow for item in self.rows
        ):
            raise ObjectSceneAnchorVersionSpaceError("matrix rows differ")
        if self.catalog.hard_disposition is Disposition.PRESENT:
            if tuple(item.binding_digest for item in self.rows) != tuple(
                item.binding_digest for item in self.catalog.bindings
            ):
                raise ObjectSceneAnchorVersionSpaceError(
                    "matrix omits or reorders an eligible binding"
                )
            expected_witnesses = tuple(
                (item.witness_id, item.witness_digest) for item in specs
            )
            for row in self.rows:
                if tuple(
                    (item.witness_id, item.witness_digest) for item in row.cells
                ) != expected_witnesses:
                    raise ObjectSceneAnchorVersionSpaceError(
                        "matrix omits or reorders a witness cell"
                    )
        elif self.rows:
            raise ObjectSceneAnchorVersionSpaceError(
                "non-present hard catalogs must have zero witness rows"
            )
        _digest(self.matrix_digest, "object witness matrix digest")
        if self.matrix_digest != canonical_digest(_matrix_content(self)):
            raise ObjectSceneAnchorVersionSpaceError("object witness matrix digest differs")

    @property
    def object_id(self) -> str:
        return self.catalog.object_id

    @property
    def binding_spec(self) -> ObjectSceneAnchorBindingSpec:
        return self.catalog.binding_spec

    @classmethod
    def create(
        cls,
        *,
        catalog: ObjectSceneAnchorBindingCatalog,
        vocabulary: ObjectSceneAnchorObserverVocabulary,
        cells: Sequence[ObjectSceneAnchorWitnessCell] = (),
    ) -> "ObjectSceneAnchorObjectWitnessMatrix":
        if type(vocabulary) is not ObjectSceneAnchorObserverVocabulary:
            raise TypeError("vocabulary must be exact observer vocabulary")
        specs = vocabulary.binding_witness_specs
        flat = tuple(cells)
        rows: tuple[ObjectSceneAnchorBindingWitnessRow, ...]
        if catalog.hard_disposition is Disposition.PRESENT:
            width = len(specs)
            if len(flat) != len(catalog.bindings) * width:
                raise ObjectSceneAnchorVersionSpaceError(
                    "present catalog requires the complete binding-witness rectangle"
                )
            rows = tuple(
                ObjectSceneAnchorBindingWitnessRow.create(
                    binding.binding_digest,
                    flat[index * width : (index + 1) * width],
                )
                for index, binding in enumerate(catalog.bindings)
            )
        else:
            if flat:
                raise ObjectSceneAnchorVersionSpaceError(
                    "non-present catalog cannot accept witness cells"
                )
            rows = ()
        values = {
            "catalog": catalog,
            "vocabulary_digest": vocabulary.vocabulary_digest,
            "witness_specs": specs,
            "rows": rows,
        }
        provisional = object.__new__(cls)
        for name, item in values.items():
            object.__setattr__(provisional, name, item)
        return cls(
            **values,
            matrix_digest=canonical_digest(_matrix_content(provisional)),
        )

    def to_data(self) -> dict[str, object]:
        return {**_matrix_content(self), "matrix_digest": self.matrix_digest}

    @classmethod
    def from_data(cls, value: object) -> "ObjectSceneAnchorObjectWitnessMatrix":
        raw = _exact_fields(
            value,
            frozenset(
                (
                    "schema",
                    "catalog",
                    "vocabulary_digest",
                    "witness_specs",
                    "rows",
                    "binding_major_witness_minor",
                    *tuple(_authority_data()),
                    "matrix_digest",
                )
            ),
            "object witness matrix",
        )
        if (
            raw["schema"] != ANCHOR_OBJECT_WITNESS_MATRIX_SCHEMA
            or raw["binding_major_witness_minor"] is not True
            or any(raw[key] != item for key, item in _authority_data().items())
            or not isinstance(raw["catalog"], Mapping)
            or not isinstance(raw["witness_specs"], list)
            or not isinstance(raw["rows"], list)
        ):
            raise ObjectSceneAnchorVersionSpaceError("object matrix policy differs")
        result = cls(
            ObjectSceneAnchorBindingCatalog.from_data(raw["catalog"]),
            raw["vocabulary_digest"],
            tuple(ObjectSceneAnchorWitnessSpec.from_data(item) for item in raw["witness_specs"]),
            tuple(ObjectSceneAnchorBindingWitnessRow.from_data(item) for item in raw["rows"]),
            raw["matrix_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectSceneAnchorVersionSpaceError("object matrix is not canonical")
        return result


def object_scene_anchor_object_matrix_from_observer_artifact(
    artifact: ObjectSceneAnchorObserverArtifact,
    language: ObjectSceneAnchorPredicateLanguage,
) -> ObjectSceneAnchorObjectWitnessMatrix:
    """Project a verified two-pass observer result into decision-only cells."""

    if type(artifact) is not ObjectSceneAnchorObserverArtifact:
        raise TypeError("artifact must be exact anchor observer artifact")
    restored = ObjectSceneAnchorObserverArtifact.from_data(artifact.to_data())
    if restored != artifact:
        raise ObjectSceneAnchorVersionSpaceError("observer artifact is not canonical")
    if artifact.vocabulary != language.vocabulary:
        raise ObjectSceneAnchorVersionSpaceError(
            "observer vocabulary differs from frozen language union"
        )
    return ObjectSceneAnchorObjectWitnessMatrix.create(
        catalog=artifact.catalog,
        vocabulary=language.vocabulary,
        cells=tuple(item.binding_cell for item in artifact.merged_cells),
    )


def _spec_matrix_content(value: "ObjectSceneAnchorSpecWitnessMatrix") -> dict[str, object]:
    return {
        "schema": ANCHOR_SPEC_WITNESS_MATRIX_SCHEMA,
        "binding_spec": value.binding_spec.to_data(),
        "vocabulary_digest": value.vocabulary_digest,
        "objects": [item.to_data() for item in value.objects],
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class ObjectSceneAnchorSpecWitnessMatrix:
    binding_spec: ObjectSceneAnchorBindingSpec
    vocabulary_digest: str
    objects: tuple[ObjectSceneAnchorObjectWitnessMatrix, ...]
    matrix_digest: str

    def __post_init__(self) -> None:
        if type(self.binding_spec) is not ObjectSceneAnchorBindingSpec:
            raise TypeError("spec matrix binding spec differs")
        _digest(self.vocabulary_digest, "spec matrix vocabulary digest")
        if (
            type(self.objects) is not tuple
            or any(type(item) is not ObjectSceneAnchorObjectWitnessMatrix for item in self.objects)
            or len({item.object_id for item in self.objects}) != len(self.objects)
            or any(
                item.binding_spec != self.binding_spec
                or item.vocabulary_digest != self.vocabulary_digest
                for item in self.objects
            )
        ):
            raise ObjectSceneAnchorVersionSpaceError("spec object matrices differ")
        _digest(self.matrix_digest, "spec witness matrix digest")
        if self.matrix_digest != canonical_digest(_spec_matrix_content(self)):
            raise ObjectSceneAnchorVersionSpaceError("spec witness matrix digest differs")

    @classmethod
    def create(
        cls,
        binding_spec: ObjectSceneAnchorBindingSpec,
        vocabulary_digest: str,
        objects: Sequence[ObjectSceneAnchorObjectWitnessMatrix],
    ) -> "ObjectSceneAnchorSpecWitnessMatrix":
        values = {
            "binding_spec": binding_spec,
            "vocabulary_digest": vocabulary_digest,
            "objects": tuple(objects),
        }
        provisional = object.__new__(cls)
        for name, item in values.items():
            object.__setattr__(provisional, name, item)
        return cls(
            **values,
            matrix_digest=canonical_digest(_spec_matrix_content(provisional)),
        )

    def to_data(self) -> dict[str, object]:
        return {**_spec_matrix_content(self), "matrix_digest": self.matrix_digest}

    @classmethod
    def from_data(cls, value: object) -> "ObjectSceneAnchorSpecWitnessMatrix":
        raw = _exact_fields(
            value,
            frozenset(
                (
                    "schema",
                    "binding_spec",
                    "vocabulary_digest",
                    "objects",
                    *tuple(_authority_data()),
                    "matrix_digest",
                )
            ),
            "spec witness matrix",
        )
        if (
            raw["schema"] != ANCHOR_SPEC_WITNESS_MATRIX_SCHEMA
            or any(raw[key] != item for key, item in _authority_data().items())
            or not isinstance(raw["binding_spec"], Mapping)
            or not isinstance(raw["objects"], list)
        ):
            raise ObjectSceneAnchorVersionSpaceError("spec matrix policy differs")
        result = cls(
            ObjectSceneAnchorBindingSpec.from_data(raw["binding_spec"]),
            raw["vocabulary_digest"],
            tuple(ObjectSceneAnchorObjectWitnessMatrix.from_data(item) for item in raw["objects"]),
            raw["matrix_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectSceneAnchorVersionSpaceError("spec matrix is not canonical")
        return result


def _panel_evaluation_content(
    value: "ObjectSceneAnchorPanelWitnessEvaluation",
) -> dict[str, object]:
    return {
        "schema": ANCHOR_PANEL_WITNESS_EVALUATION_SCHEMA,
        "panel_id": value.panel_id,
        "panel_manifest_digest": value.panel_manifest_digest,
        "inventory_digest": value.inventory_digest,
        "object_ids": list(value.object_ids),
        "object_decision_manifest_digests": list(
            value.object_decision_manifest_digests
        ),
        "language_digest": value.language_digest,
        "vocabulary_digest": value.vocabulary_digest,
        "spec_matrices": [item.to_data() for item in value.spec_matrices],
        "support_side_label_present": False,
        "complete_object_inventory_required": True,
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class ObjectSceneAnchorPanelWitnessEvaluation:
    """One neutral panel's complete object/spec/binding/witness observation."""

    panel_id: str
    panel_manifest_digest: str
    inventory_digest: str
    object_ids: tuple[str, ...]
    object_decision_manifest_digests: tuple[str, ...]
    language_digest: str
    vocabulary_digest: str
    spec_matrices: tuple[ObjectSceneAnchorSpecWitnessMatrix, ...]
    evaluation_digest: str

    def __post_init__(self) -> None:
        _panel_id(self.panel_id)
        for label, item in (
            ("panel manifest digest", self.panel_manifest_digest),
            ("inventory digest", self.inventory_digest),
            ("panel language digest", self.language_digest),
            ("panel vocabulary digest", self.vocabulary_digest),
            ("panel evaluation digest", self.evaluation_digest),
        ):
            _digest(item, label)
        if (
            type(self.object_ids) is not tuple
            or self.object_ids
            != tuple(f"object_{index:04d}" for index in range(len(self.object_ids)))
            or any(_OBJECT_ID.fullmatch(item) is None for item in self.object_ids)
            or type(self.object_decision_manifest_digests) is not tuple
            or len(self.object_decision_manifest_digests) != len(self.object_ids)
        ):
            raise ObjectSceneAnchorVersionSpaceError(
                "panel object decision inventory differs"
            )
        for item in self.object_decision_manifest_digests:
            _digest(item, "object decision manifest digest")
        if (
            type(self.spec_matrices) is not tuple
            or not self.spec_matrices
            or any(
                type(item) is not ObjectSceneAnchorSpecWitnessMatrix
                for item in self.spec_matrices
            )
            or tuple(item.binding_spec.spec_digest for item in self.spec_matrices)
            != tuple(sorted(item.binding_spec.spec_digest for item in self.spec_matrices))
            or len({item.binding_spec.spec_digest for item in self.spec_matrices})
            != len(self.spec_matrices)
        ):
            raise ObjectSceneAnchorVersionSpaceError(
                "panel spec matrix inventory differs"
            )
        decision_by_object = dict(
            zip(
                self.object_ids,
                self.object_decision_manifest_digests,
                strict=True,
            )
        )
        for spec_matrix in self.spec_matrices:
            if (
                spec_matrix.vocabulary_digest != self.vocabulary_digest
                or tuple(item.object_id for item in spec_matrix.objects)
                != self.object_ids
            ):
                raise ObjectSceneAnchorVersionSpaceError(
                    "panel matrix omits or reorders an inventoried object"
                )
            for item in spec_matrix.objects:
                if (
                    item.catalog.decision_manifest_digest
                    != decision_by_object[item.object_id]
                ):
                    raise ObjectSceneAnchorVersionSpaceError(
                        "panel object matrix belongs to another decision manifest"
                    )
        witness_inventories = {
            tuple(
                (item.witness_id, item.witness_digest)
                for item in object_matrix.witness_specs
            )
            for spec_matrix in self.spec_matrices
            for object_matrix in spec_matrix.objects
        }
        if len(witness_inventories) > 1:
            raise ObjectSceneAnchorVersionSpaceError(
                "panel spec matrices do not share one union vocabulary"
            )
        if self.evaluation_digest != canonical_digest(
            _panel_evaluation_content(self)
        ):
            raise ObjectSceneAnchorVersionSpaceError(
                "panel witness evaluation digest differs"
            )

    def to_data(self) -> dict[str, object]:
        return {
            **_panel_evaluation_content(self),
            "evaluation_digest": self.evaluation_digest,
        }

    @classmethod
    def from_data(
        cls, value: object
    ) -> "ObjectSceneAnchorPanelWitnessEvaluation":
        raw = _exact_fields(
            value,
            frozenset(
                (
                    "schema",
                    "panel_id",
                    "panel_manifest_digest",
                    "inventory_digest",
                    "object_ids",
                    "object_decision_manifest_digests",
                    "language_digest",
                    "vocabulary_digest",
                    "spec_matrices",
                    "support_side_label_present",
                    "complete_object_inventory_required",
                    *tuple(_authority_data()),
                    "evaluation_digest",
                )
            ),
            "panel witness evaluation",
        )
        if (
            raw["schema"] != ANCHOR_PANEL_WITNESS_EVALUATION_SCHEMA
            or raw["support_side_label_present"] is not False
            or raw["complete_object_inventory_required"] is not True
            or any(raw[key] != item for key, item in _authority_data().items())
            or not isinstance(raw["object_ids"], list)
            or not isinstance(raw["object_decision_manifest_digests"], list)
            or not isinstance(raw["spec_matrices"], list)
        ):
            raise ObjectSceneAnchorVersionSpaceError(
                "panel witness evaluation policy differs"
            )
        result = cls(
            raw["panel_id"],
            raw["panel_manifest_digest"],
            raw["inventory_digest"],
            tuple(raw["object_ids"]),
            tuple(raw["object_decision_manifest_digests"]),
            raw["language_digest"],
            raw["vocabulary_digest"],
            tuple(
                ObjectSceneAnchorSpecWitnessMatrix.from_data(item)
                for item in raw["spec_matrices"]
            ),
            raw["evaluation_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectSceneAnchorVersionSpaceError(
                "panel witness evaluation is not canonical"
            )
        return result


def _language_specs(
    language: ObjectSceneAnchorPredicateLanguage,
) -> tuple[ObjectSceneAnchorBindingSpec, ...]:
    by_digest: dict[str, ObjectSceneAnchorBindingSpec] = {}
    for atom in language.atoms:
        previous = by_digest.get(atom.binding_spec.spec_digest)
        if previous is not None and previous != atom.binding_spec:
            raise ObjectSceneAnchorVersionSpaceError("binding spec digest collision")
        by_digest[atom.binding_spec.spec_digest] = atom.binding_spec
    return tuple(by_digest[key] for key in sorted(by_digest))


def build_object_scene_anchor_panel_witness_evaluation(
    *,
    panel_id: str,
    panel_manifest: ObjectSceneAnchorPanelDecisionManifest,
    language: ObjectSceneAnchorPredicateLanguage,
    object_matrices: Sequence[ObjectSceneAnchorObjectWitnessMatrix],
) -> ObjectSceneAnchorPanelWitnessEvaluation:
    """Freeze a spec-major/object-major complete matrix with no support label."""

    if type(panel_manifest) is not ObjectSceneAnchorPanelDecisionManifest:
        raise TypeError("panel_manifest must be exact panel decision manifest")
    manifest = ObjectSceneAnchorPanelDecisionManifest.from_data(
        panel_manifest.to_data()
    )
    if manifest != panel_manifest:
        raise ObjectSceneAnchorVersionSpaceError("panel manifest is not canonical")
    language = ObjectSceneAnchorPredicateLanguage.from_data(language.to_data())
    matrices = tuple(object_matrices)
    if any(type(item) is not ObjectSceneAnchorObjectWitnessMatrix for item in matrices):
        raise TypeError("every object matrix must be exact object witness matrix")
    specs = _language_specs(language)
    expected_keys = tuple(
        (spec.spec_digest, object_id)
        for spec in specs
        for object_id in manifest.object_ids
    )
    actual_keys = tuple(
        (item.binding_spec.spec_digest, item.object_id) for item in matrices
    )
    if actual_keys != expected_keys:
        raise ObjectSceneAnchorVersionSpaceError(
            "panel matrices omit, reorder, or add a spec/object pair"
        )
    expected_specs = language.vocabulary.binding_witness_specs
    for matrix in matrices:
        if (
            matrix.vocabulary_digest != language.vocabulary.vocabulary_digest
            or matrix.witness_specs != expected_specs
        ):
            raise ObjectSceneAnchorVersionSpaceError(
                "panel matrix vocabulary differs from complete language union"
            )
    spec_matrices = tuple(
        ObjectSceneAnchorSpecWitnessMatrix.create(
            spec,
            language.vocabulary.vocabulary_digest,
            matrices[index * len(manifest.object_ids) : (index + 1)
            * len(manifest.object_ids)],
        )
        for index, spec in enumerate(specs)
    )
    values = {
        "panel_id": _panel_id(panel_id),
        "panel_manifest_digest": manifest.manifest_digest,
        "inventory_digest": manifest.inventory_digest,
        "object_ids": manifest.object_ids,
        "object_decision_manifest_digests": tuple(
            item.manifest_digest for item in manifest.object_decisions
        ),
        "language_digest": language.language_digest,
        "vocabulary_digest": language.vocabulary.vocabulary_digest,
        "spec_matrices": spec_matrices,
    }
    provisional = object.__new__(ObjectSceneAnchorPanelWitnessEvaluation)
    for name, item in values.items():
        object.__setattr__(provisional, name, item)
    return ObjectSceneAnchorPanelWitnessEvaluation(
        **values,
        evaluation_digest=canonical_digest(_panel_evaluation_content(provisional)),
    )


def _candidate_content(value: "ObjectSceneAnchorPredicateCandidate") -> dict[str, object]:
    return {
        "schema": ANCHOR_CANDIDATE_SCHEMA,
        "algorithm_id": ANCHOR_VERSION_SPACE_ALGORITHM_ID,
        "algorithm_digest": value.algorithm_digest,
        "language_digest": value.language_digest,
        "orientation": value.orientation.value,
        "binding_spec_digest": value.binding_spec_digest,
        "atom_digests": list(value.atom_digests),
        "witness_digests": list(value.witness_digests),
        "operator": "positive_same_binding_conjunction",
        **_policy_data(),
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class ObjectSceneAnchorPredicateCandidate:
    """One positive atom or sorted same-spec conjunction of up to four atoms."""

    algorithm_digest: str
    language_digest: str
    orientation: ObjectSceneAnchorOrientation
    binding_spec_digest: str
    atom_digests: tuple[str, ...]
    witness_digests: tuple[str, ...]
    candidate_digest: str

    def __post_init__(self) -> None:
        if self.algorithm_digest != object_scene_anchor_version_space_algorithm_digest():
            raise ObjectSceneAnchorVersionSpaceError(
                "candidate algorithm binding differs"
            )
        _digest(self.language_digest, "candidate language digest")
        _digest(self.binding_spec_digest, "candidate binding spec digest")
        if not isinstance(self.orientation, ObjectSceneAnchorOrientation):
            raise TypeError("candidate orientation differs")
        if (
            type(self.atom_digests) is not tuple
            or not 1 <= len(self.atom_digests) <= ANCHOR_MAX_CONJUNCTS
            or self.atom_digests != tuple(sorted(set(self.atom_digests)))
        ):
            raise ObjectSceneAnchorVersionSpaceError("candidate atom set differs")
        if (
            type(self.witness_digests) is not tuple
            or not 1 <= len(self.witness_digests) <= ANCHOR_MAX_CONJUNCTS
            or self.witness_digests
            != tuple(sorted(set(self.witness_digests)))
        ):
            raise ObjectSceneAnchorVersionSpaceError("candidate witness union differs")
        for item in (*self.atom_digests, *self.witness_digests):
            _digest(item, "candidate member digest")
        _digest(self.candidate_digest, "candidate digest")
        if self.candidate_digest != canonical_digest(_candidate_content(self)):
            raise ObjectSceneAnchorVersionSpaceError("candidate digest differs")

    @classmethod
    def create(
        cls,
        language: ObjectSceneAnchorPredicateLanguage,
        atoms: Sequence[ObjectSceneAnchorPredicateAtom],
    ) -> "ObjectSceneAnchorPredicateCandidate":
        selected = tuple(sorted(atoms, key=lambda item: item.atom_digest))
        if not selected:
            raise ObjectSceneAnchorVersionSpaceError("candidate must contain an atom")
        orientation = selected[0].orientation
        spec_digest = selected[0].binding_spec.spec_digest
        if any(
            item.orientation is not orientation
            or item.binding_spec.spec_digest != spec_digest
            for item in selected
        ):
            raise ObjectSceneAnchorVersionSpaceError(
                "candidate cannot mix orientation or binding spec"
            )
        values = {
            "algorithm_digest": object_scene_anchor_version_space_algorithm_digest(),
            "language_digest": language.language_digest,
            "orientation": orientation,
            "binding_spec_digest": spec_digest,
            "atom_digests": tuple(item.atom_digest for item in selected),
            "witness_digests": tuple(
                sorted({digest for item in selected for digest in item.witness_digests})
            ),
        }
        provisional = object.__new__(cls)
        for name, item in values.items():
            object.__setattr__(provisional, name, item)
        return cls(
            **values,
            candidate_digest=canonical_digest(_candidate_content(provisional)),
        )

    def to_data(self) -> dict[str, object]:
        return {**_candidate_content(self), "candidate_digest": self.candidate_digest}

    @classmethod
    def from_data(cls, value: object) -> "ObjectSceneAnchorPredicateCandidate":
        raw = _exact_fields(
            value,
            frozenset(
                (
                    "schema",
                    "algorithm_id",
                    "algorithm_digest",
                    "language_digest",
                    "orientation",
                    "binding_spec_digest",
                    "atom_digests",
                    "witness_digests",
                    "operator",
                    *tuple(_policy_data()),
                    *tuple(_authority_data()),
                    "candidate_digest",
                )
            ),
            "predicate candidate",
        )
        if (
            raw["schema"] != ANCHOR_CANDIDATE_SCHEMA
            or raw["algorithm_id"] != ANCHOR_VERSION_SPACE_ALGORITHM_ID
            or raw["operator"] != "positive_same_binding_conjunction"
            or any(raw[key] != item for key, item in _policy_data().items())
            or any(raw[key] != item for key, item in _authority_data().items())
            or not isinstance(raw["atom_digests"], list)
            or not isinstance(raw["witness_digests"], list)
        ):
            raise ObjectSceneAnchorVersionSpaceError("predicate candidate policy differs")
        result = cls(
            raw["algorithm_digest"],
            raw["language_digest"],
            _orientation(raw["orientation"]),
            raw["binding_spec_digest"],
            tuple(raw["atom_digests"]),
            tuple(raw["witness_digests"]),
            raw["candidate_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectSceneAnchorVersionSpaceError(
                "predicate candidate is not canonical"
            )
        return result


def enumerate_object_scene_anchor_candidates(
    language: ObjectSceneAnchorPredicateLanguage,
    orientation: ObjectSceneAnchorOrientation,
) -> tuple[ObjectSceneAnchorPredicateCandidate, ...]:
    """Enumerate every positive atom and same-spec conjunction through size four.

    Four is not an independent search heuristic: it is the exact maximum
    number of affirmative witnesses admitted by one frozen predicate card.
    Keeping the bounds equal prevents an accepted four-witness concept from
    being silently omitted from the deterministic version space.
    """

    language = ObjectSceneAnchorPredicateLanguage.from_data(language.to_data())
    if not isinstance(orientation, ObjectSceneAnchorOrientation):
        raise TypeError("orientation must be exact orientation enum")
    oriented = tuple(item for item in language.atoms if item.orientation is orientation)
    if not oriented:
        raise ObjectSceneAnchorVersionSpaceError(
            "selected orientation has no accepted positive atoms"
        )
    by_spec: dict[str, list[ObjectSceneAnchorPredicateAtom]] = {}
    for atom in oriented:
        by_spec.setdefault(atom.binding_spec.spec_digest, []).append(atom)
    candidates: list[ObjectSceneAnchorPredicateCandidate] = []
    for size in range(1, ANCHOR_MAX_CONJUNCTS + 1):
        for spec_digest in sorted(by_spec):
            atoms = sorted(by_spec[spec_digest], key=lambda item: item.atom_digest)
            candidates.extend(
                ObjectSceneAnchorPredicateCandidate.create(language, subset)
                for subset in combinations(atoms, size)
            )
    return tuple(candidates)


def _candidate_atoms(
    candidate: ObjectSceneAnchorPredicateCandidate,
    language: ObjectSceneAnchorPredicateLanguage,
) -> tuple[ObjectSceneAnchorPredicateAtom, ...]:
    by_digest = {item.atom_digest: item for item in language.atoms}
    try:
        result = tuple(by_digest[item] for item in candidate.atom_digests)
    except KeyError as exc:
        raise ObjectSceneAnchorVersionSpaceError(
            "candidate atom is outside the frozen language"
        ) from exc
    if (
        candidate.language_digest != language.language_digest
        or any(item.orientation is not candidate.orientation for item in result)
        or any(
            item.binding_spec.spec_digest != candidate.binding_spec_digest
            for item in result
        )
        or tuple(sorted({digest for item in result for digest in item.witness_digests}))
        != candidate.witness_digests
    ):
        raise ObjectSceneAnchorVersionSpaceError(
            "candidate differs from its frozen atom projection"
        )
    return result


def _candidate_spec_matrix(
    candidate: ObjectSceneAnchorPredicateCandidate,
    panel: ObjectSceneAnchorPanelWitnessEvaluation,
) -> ObjectSceneAnchorSpecWitnessMatrix:
    if candidate.language_digest != panel.language_digest:
        raise ObjectSceneAnchorVersionSpaceError(
            "candidate and panel language digests differ"
        )
    matches = tuple(
        item
        for item in panel.spec_matrices
        if item.binding_spec.spec_digest == candidate.binding_spec_digest
    )
    if len(matches) != 1:
        raise ObjectSceneAnchorVersionSpaceError(
            "candidate binding spec is absent from panel matrix"
        )
    return matches[0]


def _row_candidate_state(
    row: ObjectSceneAnchorBindingWitnessRow,
    witness_digests: tuple[str, ...],
) -> Disposition:
    by_digest = {item.witness_digest: item.disposition for item in row.cells}
    if any(item not in by_digest for item in witness_digests):
        return Disposition.ERROR
    return _scene_and(tuple(by_digest[item] for item in witness_digests))


def evaluate_object_scene_anchor_candidate_on_target(
    candidate: ObjectSceneAnchorPredicateCandidate,
    language: ObjectSceneAnchorPredicateLanguage,
    panel: ObjectSceneAnchorPanelWitnessEvaluation,
) -> Disposition:
    """Evaluate only the exact common binding cited for this target panel."""

    candidate = ObjectSceneAnchorPredicateCandidate.from_data(candidate.to_data())
    language = ObjectSceneAnchorPredicateLanguage.from_data(language.to_data())
    panel = ObjectSceneAnchorPanelWitnessEvaluation.from_data(panel.to_data())
    atoms = _candidate_atoms(candidate, language)
    matrix = _candidate_spec_matrix(candidate, panel)
    citations = []
    for atom in atoms:
        matching = tuple(
            item for item in atom.positive_support_citations if item.panel_id == panel.panel_id
        )
        if len(matching) != 1:
            return Disposition.ERROR
        citations.append(matching[0])
    current_catalogs_digest = canonical_digest(
        {
            "schema": "gkm.object-scene-anchor-card-panel-binding-catalogs.v1",
            "panel_manifest_digest": panel.panel_manifest_digest,
            "binding_spec_digest": candidate.binding_spec_digest,
            "object_ids": list(panel.object_ids),
            "catalogs": [item.catalog.to_data() for item in matrix.objects],
            "complete_object_inventory_required": True,
        }
    )
    if any(
        item.panel_manifest_digest != panel.panel_manifest_digest
        or item.binding_catalogs_digest != current_catalogs_digest
        for item in citations
    ):
        return Disposition.ERROR
    cited_keys = {
        (item.binding.object_id, item.binding.binding_digest) for item in citations
    }
    if len(cited_keys) != 1:
        return Disposition.CERTIFIED_ABSENT
    cited_object_id, cited_digest = next(iter(cited_keys))
    object_matches = tuple(
        item for item in matrix.objects if item.object_id == cited_object_id
    )
    if len(object_matches) != 1:
        return Disposition.ERROR
    object_matrix = object_matches[0]
    if object_matrix.catalog.hard_disposition is Disposition.ERROR:
        return Disposition.ERROR
    if object_matrix.catalog.hard_disposition is Disposition.INDETERMINATE:
        return Disposition.ERROR
    if object_matrix.catalog.hard_disposition is Disposition.CERTIFIED_ABSENT:
        return Disposition.ERROR
    exact_bindings = tuple(
        binding
        for binding in object_matrix.catalog.bindings
        if binding.binding_digest == cited_digest
    )
    rows = tuple(
        row for row in object_matrix.rows if row.binding_digest == cited_digest
    )
    if len(exact_bindings) != 1 or len(rows) != 1:
        return Disposition.ERROR
    if any(item.binding != exact_bindings[0] for item in citations):
        return Disposition.ERROR
    return _row_candidate_state(rows[0], candidate.witness_digests)


def evaluate_object_scene_anchor_candidate_on_contrast(
    candidate: ObjectSceneAnchorPredicateCandidate,
    language: ObjectSceneAnchorPredicateLanguage,
    panel: ObjectSceneAnchorPanelWitnessEvaluation,
) -> Disposition:
    """Existentially evaluate every eligible binding of every panel object."""

    candidate = ObjectSceneAnchorPredicateCandidate.from_data(candidate.to_data())
    language = ObjectSceneAnchorPredicateLanguage.from_data(language.to_data())
    panel = ObjectSceneAnchorPanelWitnessEvaluation.from_data(panel.to_data())
    _candidate_atoms(candidate, language)
    matrix = _candidate_spec_matrix(candidate, panel)
    states: list[Disposition] = []
    for item in matrix.objects:
        hard = item.catalog.hard_disposition
        if hard is Disposition.ERROR:
            states.append(Disposition.ERROR)
        elif hard is Disposition.INDETERMINATE:
            states.append(Disposition.INDETERMINATE)
        elif hard is Disposition.PRESENT:
            states.extend(
                _row_candidate_state(row, candidate.witness_digests)
                for row in item.rows
            )
        # A complete empty catalog contributes no existential binding.
    return _scene_or(states)


def _diagnostic_content(
    value: "ObjectSceneAnchorSupportDiagnostic",
) -> dict[str, object]:
    return {
        "schema": ANCHOR_SUPPORT_DIAGNOSTIC_SCHEMA,
        "candidate_digest": value.candidate_digest,
        "definite_counterexample_panel_ids": list(
            value.definite_counterexample_panel_ids
        ),
        "indeterminate_panel_ids": list(value.indeterminate_panel_ids),
        "error_panel_ids": list(value.error_panel_ids),
    }


@dataclass(frozen=True, slots=True)
class ObjectSceneAnchorSupportDiagnostic:
    candidate_digest: str
    definite_counterexample_panel_ids: tuple[str, ...]
    indeterminate_panel_ids: tuple[str, ...]
    error_panel_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _digest(self.candidate_digest, "diagnostic candidate digest")
        inventories = []
        for name in (
            "definite_counterexample_panel_ids",
            "indeterminate_panel_ids",
            "error_panel_ids",
        ):
            values = getattr(self, name)
            if (
                type(values) is not tuple
                or values != tuple(sorted(set(values)))
                or any(_PANEL_ID.fullmatch(item) is None for item in values)
            ):
                raise ObjectSceneAnchorVersionSpaceError(
                    f"{name} is not a canonical panel inventory"
                )
            inventories.append(set(values))
        if any(
            inventories[left] & inventories[right]
            for left in range(len(inventories))
            for right in range(left + 1, len(inventories))
        ):
            raise ObjectSceneAnchorVersionSpaceError(
                "diagnostic panel inventories overlap"
            )

    def to_data(self) -> dict[str, object]:
        return _diagnostic_content(self)

    @classmethod
    def from_data(cls, value: object) -> "ObjectSceneAnchorSupportDiagnostic":
        raw = _exact_fields(
            value,
            frozenset(
                (
                    "schema",
                    "candidate_digest",
                    "definite_counterexample_panel_ids",
                    "indeterminate_panel_ids",
                    "error_panel_ids",
                )
            ),
            "support diagnostic",
        )
        for name in (
            "definite_counterexample_panel_ids",
            "indeterminate_panel_ids",
            "error_panel_ids",
        ):
            if not isinstance(raw[name], list):
                raise ObjectSceneAnchorVersionSpaceError(
                    f"{name} must be a JSON list"
                )
        result = cls(
            raw["candidate_digest"],
            tuple(raw["definite_counterexample_panel_ids"]),
            tuple(raw["indeterminate_panel_ids"]),
            tuple(raw["error_panel_ids"]),
        )
        if result.to_data() != dict(raw):
            raise ObjectSceneAnchorVersionSpaceError(
                "support diagnostic is not canonical"
            )
        return result


def _gap_content(value: "ObjectSceneAnchorSupportGap") -> dict[str, object]:
    return {
        "schema": ANCHOR_SUPPORT_GAP_SCHEMA,
        "kind": value.kind.value,
        "diagnostics": [item.to_data() for item in value.diagnostics],
        "language_gap_iff_every_candidate_has_definite_counterexample": True,
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class ObjectSceneAnchorSupportGap:
    kind: ObjectSceneAnchorSupportGapKind
    diagnostics: tuple[ObjectSceneAnchorSupportDiagnostic, ...]
    gap_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.kind, ObjectSceneAnchorSupportGapKind):
            raise TypeError("gap kind differs")
        if (
            type(self.diagnostics) is not tuple
            or not self.diagnostics
            or any(
                type(item) is not ObjectSceneAnchorSupportDiagnostic
                for item in self.diagnostics
            )
            or len({item.candidate_digest for item in self.diagnostics})
            != len(self.diagnostics)
            or tuple(item.candidate_digest for item in self.diagnostics)
            != tuple(sorted(item.candidate_digest for item in self.diagnostics))
        ):
            raise ObjectSceneAnchorVersionSpaceError("gap diagnostics differ")
        expected_kind = (
            ObjectSceneAnchorSupportGapKind.LANGUAGE_GAP
            if all(item.definite_counterexample_panel_ids for item in self.diagnostics)
            else ObjectSceneAnchorSupportGapKind.WITNESS_GAP
        )
        if self.kind is not expected_kind:
            raise ObjectSceneAnchorVersionSpaceError("typed support gap differs")
        _digest(self.gap_digest, "support gap digest")
        if self.gap_digest != canonical_digest(_gap_content(self)):
            raise ObjectSceneAnchorVersionSpaceError("support gap digest differs")

    def to_data(self) -> dict[str, object]:
        return {**_gap_content(self), "gap_digest": self.gap_digest}

    @classmethod
    def from_data(cls, value: object) -> "ObjectSceneAnchorSupportGap":
        raw = _exact_fields(
            value,
            frozenset(
                (
                    "schema",
                    "kind",
                    "diagnostics",
                    "language_gap_iff_every_candidate_has_definite_counterexample",
                    *tuple(_authority_data()),
                    "gap_digest",
                )
            ),
            "support gap",
        )
        if (
            raw["schema"] != ANCHOR_SUPPORT_GAP_SCHEMA
            or raw[
                "language_gap_iff_every_candidate_has_definite_counterexample"
            ]
            is not True
            or any(raw[key] != item for key, item in _authority_data().items())
            or not isinstance(raw["diagnostics"], list)
        ):
            raise ObjectSceneAnchorVersionSpaceError("support gap policy differs")
        try:
            kind = ObjectSceneAnchorSupportGapKind(raw["kind"])
        except (TypeError, ValueError) as exc:
            raise ObjectSceneAnchorVersionSpaceError("support gap kind differs") from exc
        result = cls(
            kind,
            tuple(
                ObjectSceneAnchorSupportDiagnostic.from_data(item)
                for item in raw["diagnostics"]
            ),
            raw["gap_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectSceneAnchorVersionSpaceError("support gap is not canonical")
        return result


def _is_survivor(
    row: tuple[Disposition, ...],
    sides: tuple[ObjectSceneAnchorSupportSide, ...],
) -> bool:
    return all(
        state
        is (
            Disposition.PRESENT
            if side is ObjectSceneAnchorSupportSide.TARGET
            else Disposition.CERTIFIED_ABSENT
        )
        for state, side in zip(row, sides, strict=True)
    )


def _make_support_gap(
    candidates: tuple[ObjectSceneAnchorPredicateCandidate, ...],
    panel_ids: tuple[str, ...],
    sides: tuple[ObjectSceneAnchorSupportSide, ...],
    rows: tuple[tuple[Disposition, ...], ...],
) -> ObjectSceneAnchorSupportGap:
    diagnostics = []
    for candidate, row in zip(candidates, rows, strict=True):
        definite = tuple(
            sorted(
                panel_id
                for panel_id, side, state in zip(panel_ids, sides, row, strict=True)
                if (
                    side is ObjectSceneAnchorSupportSide.TARGET
                    and state is Disposition.CERTIFIED_ABSENT
                )
                or (
                    side is ObjectSceneAnchorSupportSide.CONTRAST
                    and state is Disposition.PRESENT
                )
            )
        )
        indeterminate = tuple(
            sorted(
                panel_id
                for panel_id, state in zip(panel_ids, row, strict=True)
                if state is Disposition.INDETERMINATE
            )
        )
        errors = tuple(
            sorted(
                panel_id
                for panel_id, state in zip(panel_ids, row, strict=True)
                if state is Disposition.ERROR
            )
        )
        diagnostics.append(
            ObjectSceneAnchorSupportDiagnostic(
                candidate.candidate_digest,
                definite,
                indeterminate,
                errors,
            )
        )
    diagnostic_tuple = tuple(
        sorted(diagnostics, key=lambda item: item.candidate_digest)
    )
    kind = (
        ObjectSceneAnchorSupportGapKind.LANGUAGE_GAP
        if all(item.definite_counterexample_panel_ids for item in diagnostic_tuple)
        else ObjectSceneAnchorSupportGapKind.WITNESS_GAP
    )
    provisional = object.__new__(ObjectSceneAnchorSupportGap)
    object.__setattr__(provisional, "kind", kind)
    object.__setattr__(provisional, "diagnostics", diagnostic_tuple)
    return ObjectSceneAnchorSupportGap(
        kind,
        diagnostic_tuple,
        canonical_digest(_gap_content(provisional)),
    )


def _version_content(
    value: "ObjectSceneAnchorSupportVersionSpace",
) -> dict[str, object]:
    return {
        "schema": ANCHOR_SUPPORT_VERSION_SPACE_SCHEMA,
        "algorithm_id": ANCHOR_VERSION_SPACE_ALGORITHM_ID,
        "algorithm_digest": value.algorithm_digest,
        "language": value.language.to_data(),
        "orientation": value.orientation.value,
        "candidates": [item.to_data() for item in value.candidates],
        "support_panel_ids": list(value.support_panel_ids),
        "support_evaluation_digests": list(value.support_evaluation_digests),
        "support_sides": [item.value for item in value.support_sides],
        "rows": [[item.value for item in row] for row in value.rows],
        "survivor_candidate_digests": list(value.survivor_candidate_digests),
        "gap": None if value.gap is None else value.gap.to_data(),
        "support_panels_per_side": ANCHOR_SUPPORT_PANELS_PER_SIDE,
        "complete_finite_inventory": True,
        "codex_may_rank_survivors_only": True,
        **_policy_data(),
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class ObjectSceneAnchorSupportVersionSpace:
    """Complete candidate rows and their exact support-consistent subset."""

    algorithm_digest: str
    language: ObjectSceneAnchorPredicateLanguage
    orientation: ObjectSceneAnchorOrientation
    candidates: tuple[ObjectSceneAnchorPredicateCandidate, ...]
    support_panel_ids: tuple[str, ...]
    support_evaluation_digests: tuple[str, ...]
    support_sides: tuple[ObjectSceneAnchorSupportSide, ...]
    rows: tuple[tuple[Disposition, ...], ...]
    survivor_candidate_digests: tuple[str, ...]
    gap: ObjectSceneAnchorSupportGap | None
    version_space_digest: str

    def __post_init__(self) -> None:
        if self.algorithm_digest != object_scene_anchor_version_space_algorithm_digest():
            raise ObjectSceneAnchorVersionSpaceError(
                "version-space algorithm binding differs"
            )
        if type(self.language) is not ObjectSceneAnchorPredicateLanguage:
            raise TypeError("version-space language differs")
        if not isinstance(self.orientation, ObjectSceneAnchorOrientation):
            raise TypeError("version-space orientation differs")
        expected_candidates = enumerate_object_scene_anchor_candidates(
            self.language, self.orientation
        )
        if self.candidates != expected_candidates:
            raise ObjectSceneAnchorVersionSpaceError(
                "candidate inventory is not complete and canonical"
            )
        side_size = ANCHOR_SUPPORT_PANELS_PER_SIDE
        expected_sides = (ObjectSceneAnchorSupportSide.TARGET,) * side_size + (
            ObjectSceneAnchorSupportSide.CONTRAST,
        ) * side_size
        if (
            type(self.support_panel_ids) is not tuple
            or len(self.support_panel_ids) != side_size * 2
            or len(set(self.support_panel_ids)) != len(self.support_panel_ids)
            or any(_PANEL_ID.fullmatch(item) is None for item in self.support_panel_ids)
            or self.support_panel_ids[:side_size]
            != tuple(sorted(self.support_panel_ids[:side_size]))
            or self.support_panel_ids[side_size:]
            != tuple(sorted(self.support_panel_ids[side_size:]))
            or self.support_sides != expected_sides
        ):
            raise ObjectSceneAnchorVersionSpaceError(
                "support must be exactly six sorted target and six sorted contrast panels"
            )
        if (
            type(self.support_evaluation_digests) is not tuple
            or len(self.support_evaluation_digests) != side_size * 2
        ):
            raise ObjectSceneAnchorVersionSpaceError(
                "support evaluation digest inventory differs"
            )
        for item in self.support_evaluation_digests:
            _digest(item, "support evaluation digest")
        if (
            type(self.rows) is not tuple
            or len(self.rows) != len(self.candidates)
            or any(type(row) is not tuple or len(row) != side_size * 2 for row in self.rows)
            or any(not isinstance(item, Disposition) for row in self.rows for item in row)
        ):
            raise ObjectSceneAnchorVersionSpaceError("version-space rows differ")
        expected_survivors = tuple(
            candidate.candidate_digest
            for candidate, row in zip(self.candidates, self.rows, strict=True)
            if _is_survivor(row, self.support_sides)
        )
        if self.survivor_candidate_digests != expected_survivors:
            raise ObjectSceneAnchorVersionSpaceError(
                "survivor set differs from exact disposition rows"
            )
        expected_gap = (
            None
            if expected_survivors
            else _make_support_gap(
                self.candidates,
                self.support_panel_ids,
                self.support_sides,
                self.rows,
            )
        )
        if self.gap != expected_gap:
            raise ObjectSceneAnchorVersionSpaceError(
                "typed gap differs from exact disposition rows"
            )
        _digest(self.version_space_digest, "version-space digest")
        if self.version_space_digest != canonical_digest(_version_content(self)):
            raise ObjectSceneAnchorVersionSpaceError("version-space digest differs")

    def to_data(self) -> dict[str, object]:
        return {**_version_content(self), "version_space_digest": self.version_space_digest}

    @classmethod
    def from_data(cls, value: object) -> "ObjectSceneAnchorSupportVersionSpace":
        raw = _exact_fields(
            value,
            frozenset(
                (
                    "schema",
                    "algorithm_id",
                    "algorithm_digest",
                    "language",
                    "orientation",
                    "candidates",
                    "support_panel_ids",
                    "support_evaluation_digests",
                    "support_sides",
                    "rows",
                    "survivor_candidate_digests",
                    "gap",
                    "support_panels_per_side",
                    "complete_finite_inventory",
                    "codex_may_rank_survivors_only",
                    *tuple(_policy_data()),
                    *tuple(_authority_data()),
                    "version_space_digest",
                )
            ),
            "support version space",
        )
        for name in (
            "candidates",
            "support_panel_ids",
            "support_evaluation_digests",
            "support_sides",
            "rows",
            "survivor_candidate_digests",
        ):
            if not isinstance(raw[name], list):
                raise ObjectSceneAnchorVersionSpaceError(f"{name} must be a JSON list")
        if (
            raw["schema"] != ANCHOR_SUPPORT_VERSION_SPACE_SCHEMA
            or raw["algorithm_id"] != ANCHOR_VERSION_SPACE_ALGORITHM_ID
            or raw["support_panels_per_side"] != ANCHOR_SUPPORT_PANELS_PER_SIDE
            or raw["complete_finite_inventory"] is not True
            or raw["codex_may_rank_survivors_only"] is not True
            or any(raw[key] != item for key, item in _policy_data().items())
            or any(raw[key] != item for key, item in _authority_data().items())
            or not isinstance(raw["language"], Mapping)
        ):
            raise ObjectSceneAnchorVersionSpaceError("support version-space policy differs")
        try:
            sides = tuple(ObjectSceneAnchorSupportSide(item) for item in raw["support_sides"])
            rows = tuple(tuple(Disposition(item) for item in row) for row in raw["rows"])
        except (TypeError, ValueError) as exc:
            raise ObjectSceneAnchorVersionSpaceError(
                "support side or disposition differs"
            ) from exc
        result = cls(
            raw["algorithm_digest"],
            ObjectSceneAnchorPredicateLanguage.from_data(raw["language"]),
            _orientation(raw["orientation"]),
            tuple(ObjectSceneAnchorPredicateCandidate.from_data(item) for item in raw["candidates"]),
            tuple(raw["support_panel_ids"]),
            tuple(raw["support_evaluation_digests"]),
            sides,
            rows,
            tuple(raw["survivor_candidate_digests"]),
            None if raw["gap"] is None else ObjectSceneAnchorSupportGap.from_data(raw["gap"]),
            raw["version_space_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectSceneAnchorVersionSpaceError(
                "support version space is not canonical"
            )
        return result


def _canonical_support_panels(
    values: Sequence[ObjectSceneAnchorPanelWitnessEvaluation],
    *,
    label: str,
) -> tuple[ObjectSceneAnchorPanelWitnessEvaluation, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{label} panels must be a sequence")
    restored_rows = []
    for item in values:
        if type(item) is not ObjectSceneAnchorPanelWitnessEvaluation:
            raise TypeError("panel evaluation has the wrong type")
        restored_rows.append(
            ObjectSceneAnchorPanelWitnessEvaluation.from_data(item.to_data())
        )
    restored = tuple(restored_rows)
    ordered = tuple(sorted(restored, key=lambda item: item.panel_id))
    if len(ordered) != ANCHOR_SUPPORT_PANELS_PER_SIDE:
        raise ObjectSceneAnchorVersionSpaceError(
            f"{label} support must contain exactly six panels"
        )
    return ordered


def build_object_scene_anchor_support_version_space(
    *,
    language: ObjectSceneAnchorPredicateLanguage,
    orientation: ObjectSceneAnchorOrientation,
    targets: Sequence[ObjectSceneAnchorPanelWitnessEvaluation],
    contrasts: Sequence[ObjectSceneAnchorPanelWitnessEvaluation],
) -> ObjectSceneAnchorSupportVersionSpace:
    """Build the complete exact support version space for one frozen orientation."""

    language = ObjectSceneAnchorPredicateLanguage.from_data(language.to_data())
    if not isinstance(orientation, ObjectSceneAnchorOrientation):
        raise TypeError("orientation must be exact orientation enum")
    target_panels = _canonical_support_panels(targets, label="target")
    contrast_panels = _canonical_support_panels(contrasts, label="contrast")
    panels = target_panels + contrast_panels
    if len({item.panel_id for item in panels}) != len(panels):
        raise ObjectSceneAnchorVersionSpaceError("support panel IDs must be globally unique")
    if any(item.language_digest != language.language_digest for item in panels):
        raise ObjectSceneAnchorVersionSpaceError(
            "support panels do not share the frozen language"
        )
    oriented_atoms = tuple(item for item in language.atoms if item.orientation is orientation)
    if not oriented_atoms:
        raise ObjectSceneAnchorVersionSpaceError("selected orientation has no atoms")
    expected_target_ids = oriented_atoms[0].target_panel_ids
    if tuple(item.panel_id for item in target_panels) != expected_target_ids:
        raise ObjectSceneAnchorVersionSpaceError(
            "target panels differ from exact card citation inventory"
        )
    candidates = enumerate_object_scene_anchor_candidates(language, orientation)
    rows = tuple(
        tuple(
            evaluate_object_scene_anchor_candidate_on_target(candidate, language, panel)
            for panel in target_panels
        )
        + tuple(
            evaluate_object_scene_anchor_candidate_on_contrast(candidate, language, panel)
            for panel in contrast_panels
        )
        for candidate in candidates
    )
    sides = (ObjectSceneAnchorSupportSide.TARGET,) * ANCHOR_SUPPORT_PANELS_PER_SIDE + (
        ObjectSceneAnchorSupportSide.CONTRAST,
    ) * ANCHOR_SUPPORT_PANELS_PER_SIDE
    survivors = tuple(
        candidate.candidate_digest
        for candidate, row in zip(candidates, rows, strict=True)
        if _is_survivor(row, sides)
    )
    panel_ids = tuple(item.panel_id for item in panels)
    gap = None if survivors else _make_support_gap(candidates, panel_ids, sides, rows)
    values = {
        "algorithm_digest": object_scene_anchor_version_space_algorithm_digest(),
        "language": language,
        "orientation": orientation,
        "candidates": candidates,
        "support_panel_ids": panel_ids,
        "support_evaluation_digests": tuple(item.evaluation_digest for item in panels),
        "support_sides": sides,
        "rows": rows,
        "survivor_candidate_digests": survivors,
        "gap": gap,
    }
    provisional = object.__new__(ObjectSceneAnchorSupportVersionSpace)
    for name, item in values.items():
        object.__setattr__(provisional, name, item)
    return ObjectSceneAnchorSupportVersionSpace(
        **values,
        version_space_digest=canonical_digest(_version_content(provisional)),
    )


def cold_verify_object_scene_anchor_support_version_space(
    version_space: ObjectSceneAnchorSupportVersionSpace,
    *,
    language: ObjectSceneAnchorPredicateLanguage,
    orientation: ObjectSceneAnchorOrientation,
    targets: Sequence[ObjectSceneAnchorPanelWitnessEvaluation],
    contrasts: Sequence[ObjectSceneAnchorPanelWitnessEvaluation],
) -> ObjectSceneAnchorSupportVersionSpace:
    """Replay enumeration and all support decisions without pixels or model calls."""

    if type(version_space) is not ObjectSceneAnchorSupportVersionSpace:
        raise TypeError("version_space must be exact anchor support version space")
    restored = ObjectSceneAnchorSupportVersionSpace.from_data(version_space.to_data())
    replayed = build_object_scene_anchor_support_version_space(
        language=language,
        orientation=orientation,
        targets=targets,
        contrasts=contrasts,
    )
    if restored != replayed:
        raise ObjectSceneAnchorVersionSpaceError(
            "cold anchor support version-space replay differs"
        )
    return restored


def project_object_scene_anchor_card_proposal(
    proposal: object,
) -> ObjectSceneAnchorPredicateLanguage:
    """Project the exact validated card proposal into one-witness atoms."""

    try:
        from bongard.object_scene_anchor_cards import (
            ObjectSceneAnchorCardProposal,
        )
    except ImportError as exc:  # pragma: no cover - deployment error.
        raise ObjectSceneAnchorVersionSpaceError(
            "anchor card proposal type is unavailable"
        ) from exc
    if type(proposal) is not ObjectSceneAnchorCardProposal:
        raise TypeError("proposal must be exact anchor card proposal")
    restored = ObjectSceneAnchorCardProposal.from_data(proposal.to_data())
    if restored != proposal:
        raise ObjectSceneAnchorVersionSpaceError("card proposal is not canonical")
    cards = tuple(proposal.side0_positive) + tuple(proposal.side1_positive)
    witnesses = tuple(
        witness for card in cards for witness in card.required_witnesses
    )
    vocabulary = freeze_object_scene_anchor_observer_vocabulary(witnesses)
    atoms = []
    for card in cards:
        orientation = _orientation(card.orientation)
        citations = tuple(
            sorted(
                (
                    ObjectSceneAnchorAtomCitation.create(
                        item.panel_alias,
                        item.panel_manifest_digest,
                        item.binding_catalogs_digest,
                        item.resolved_binding,
                    )
                    for item in card.positive_support_citations
                ),
                key=lambda item: item.panel_id,
            )
        )
        for witness in card.required_witnesses:
            atoms.append(
                ObjectSceneAnchorPredicateAtom.create(
                    source_card_digest=card.card_digest,
                    orientation=orientation,
                    binding_spec=card.binding_spec,
                    witness_digests=(witness.witness_digest,),
                    positive_support_citations=citations,
                )
            )
    return ObjectSceneAnchorPredicateLanguage.create(
        source_proposal_digest=proposal.proposal_digest,
        vocabulary=vocabulary,
        atoms=atoms,
    )


__all__ = (
    "ANCHOR_MAX_CONJUNCTS",
    "ANCHOR_SUPPORT_PANELS_PER_SIDE",
    "ANCHOR_VERSION_SPACE_ALGORITHM_ID",
    "ObjectSceneAnchorAtomCitation",
    "ObjectSceneAnchorBindingWitnessRow",
    "ObjectSceneAnchorObjectWitnessMatrix",
    "ObjectSceneAnchorOrientation",
    "ObjectSceneAnchorPanelWitnessEvaluation",
    "ObjectSceneAnchorPredicateAtom",
    "ObjectSceneAnchorPredicateCandidate",
    "ObjectSceneAnchorPredicateLanguage",
    "ObjectSceneAnchorSpecWitnessMatrix",
    "ObjectSceneAnchorSupportDiagnostic",
    "ObjectSceneAnchorSupportGap",
    "ObjectSceneAnchorSupportGapKind",
    "ObjectSceneAnchorSupportSide",
    "ObjectSceneAnchorSupportVersionSpace",
    "ObjectSceneAnchorVersionSpaceError",
    "build_object_scene_anchor_panel_witness_evaluation",
    "build_object_scene_anchor_support_version_space",
    "cold_verify_object_scene_anchor_support_version_space",
    "enumerate_object_scene_anchor_candidates",
    "evaluate_object_scene_anchor_candidate_on_contrast",
    "evaluate_object_scene_anchor_candidate_on_target",
    "object_scene_anchor_object_matrix_from_observer_artifact",
    "object_scene_anchor_version_space_algorithm_digest",
    "object_scene_anchor_version_space_source_digest",
    "project_object_scene_anchor_card_proposal",
)
